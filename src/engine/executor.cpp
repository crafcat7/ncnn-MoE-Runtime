#include "executor.h"

#include "kernels/attention.h"
#include "kernels/activation.h"
#include "kernels/bfloat16.h"
#include "kernels/gateddeltanet.h"
#include "kernels/gatedresidual.h"
#include "kernels/hyperconnection.h"
#include "kernels/latentattention.h"
#include "kernels/ops.h"
#include "kernels/ple.h"
#include "kernels/qnk.h"
#include "sessionstate.h"
#include "cpu.h"
#include "metrics.h"
#include "expertbackend.h"
#include "expert.h"
#include "storage/expertcache.h"
#include "backends/ncnn/attention.h"
#include "backends/ncnn/linear.h"

#include "graph/router.h"
#include "graph/compiledmodel.h"
#include "ncnn/moe/runtime.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

struct RouterPredictionOutcome
{
    std::vector<uint32_t> predicted_expert_ids;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t predictor_time_microseconds = 0;
};

struct PendingRouterPrediction
{
    ~PendingRouterPrediction()
    {
        if (!result.valid())
            return;
        try
        {
            result.wait();
        }
        catch (...)
        {
        }
    }

    uint32_t target_layer_id = std::numeric_limits<uint32_t>::max();
    std::future<Result<RouterPredictionOutcome>> result;
};

static uint64_t elapsed_microseconds(std::chrono::steady_clock::time_point start)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

CpuTaskWorker::CpuTaskWorker(size_t maximum_outstanding_tasks)
    : task_limit(std::max<size_t>(1, maximum_outstanding_tasks))
{
    worker = std::thread(&CpuTaskWorker::worker_loop, this);
}

CpuTaskWorker::~CpuTaskWorker()
{
    {
        const std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    task_ready.notify_all();
    if (worker.joinable())
        worker.join();
}

bool CpuTaskWorker::try_submit(std::function<void()> task)
{
    if (!task)
        return false;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        if (stop
            || outstanding_tasks >= task_limit)
        {
            return false;
        }
        tasks.push_back(std::move(task));
        ++outstanding_tasks;
    }
    task_ready.notify_one();
    return true;
}

void CpuTaskWorker::worker_loop()
{
    for (;;)
    {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            task_ready.wait(lock, [this] {
                return stop || !tasks.empty();
            });
            if (stop && tasks.empty())
                return;
            task = std::move(tasks.front());
            tasks.pop_front();
        }
        try
        {
            task();
        }
        catch (...)
        {
        }
        {
            const std::lock_guard<std::mutex> lock(mutex);
            --outstanding_tasks;
        }
    }
}

static void resolve_router_predictions(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    const ExpertDispatchPlan& plan,
    CpuSessionState& state,
    SessionStatistics& statistics,
    bool resolve_unused_predictions);

[[nodiscard]] static Result<void> complete_router_prediction(
    const CompiledModel& model,
    uint32_t layer_id,
    PendingRouterPrediction& pending,
    CpuSessionState& state,
    SessionStatistics& statistics,
    size_t token_count);

[[nodiscard]] static Result<void> predict_next_router_routes(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    const CpuBatch& router_input,
    CpuSessionState& state,
    SessionStatistics& statistics,
    PendingRouterPrediction& pending);

static void capture_speculative_hidden(
    const CpuBatch& hidden,
    uint32_t hidden_size,
    uint32_t multiplier,
    size_t target_index,
    CpuBatch& destination)
{
    const float inverse_multiplier = 1.0f / static_cast<float>(multiplier);
    for (size_t row = 0; row < hidden.rows(); ++row)
    {
        float* output = destination.row(row) + target_index * hidden_size;
        const float* input = hidden.row(row);
        for (uint32_t copy = 0; copy < multiplier; ++copy)
        {
            const float* source = input + static_cast<size_t>(copy) * hidden_size;
            for (uint32_t column = 0; column < hidden_size; ++column)
                output[column] += source[column];
        }
        for (uint32_t column = 0; column < hidden_size; ++column)
            output[column] *= inverse_multiplier;
    }
}

static bool has_unknown_vulkan_attention_state(
    const CpuSessionState& state) noexcept
{
    return std::any_of(
        state.layers.begin(),
        state.layers.end(),
        [](const CpuLayerCache& cache) {
            return cache.vulkan_attention_state_unknown;
        });
}

static void prepare_execution_state(
    const CompiledModel& model,
    SessionStatistics& statistics,
    CpuSessionState& state)
{
    if (statistics.expert_token_counts.size() < model.descriptor.expert_count)
        statistics.expert_token_counts.resize(model.descriptor.expert_count, 0);
    if (state.layers.size() != model.graph.layer_plans.size())
        state.layers.resize(model.graph.layer_plans.size());
    if (state.execution_layers.size() != model.graph.layer_plans.size())
        state.execution_layers.resize(model.graph.layer_plans.size());
    for (LayerGraphState& layer_state : state.execution_layers)
        layer_state.reset();
}

Result<std::vector<std::vector<float>>> forward_model(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    SessionStatistics& statistics,
    CpuSessionState& state,
    uint64_t position_offset)
{
    const ScopedExpertBackendForeground expert_backend_foreground(
        model.expert_backend);
    if (has_unknown_vulkan_attention_state(state))
    {
        return Error{
            ErrorCode::InternalError,
            "Session state is unavailable after a failed Vulkan Attention update"};
    }
    const NcnnVulkanExecutionSnapshot initial_vulkan_execution = get_vulkan_execution_snapshot(model.vulkan_context_instance);
    Bfloat16BatchedLinearExecutionCounter cpu_bfloat16_execution;
    const ScopedBfloat16BatchedLinearExecutionCounter cpu_bfloat16_scope(
        &cpu_bfloat16_execution);
    for (int32_t token_id : input_ids)
    {
        if (token_id < 0 || static_cast<uint32_t>(token_id) >= model.descriptor.vocabulary_size)
            return Error{ErrorCode::InvalidArgument, "token id is outside the model vocabulary"};
    }

    prepare_execution_state(model, statistics, state);

    ExpertCacheStatistics execution_cache_before;
    if (model.expert_cache)
        execution_cache_before = model.expert_cache->statistics();
    ExpertBackendStatistics expert_backend_before;
    if (model.expert_backend)
    {
        expert_backend_before = model.expert_backend->statistics();
    }

    CpuBatch& hidden = state.hidden;
    hidden.clear();
    if (model.speculative.enabled()
        && state.use_speculative_context)
    {
        const uint32_t speculative_hidden_columns = model.speculative.kind == SpeculativeModelKind::Mtp
                                                        ? model.descriptor.hidden_size
                                                        : model.descriptor.hidden_size
                                                              * static_cast<uint32_t>(
                                                                  model.speculative.target_layer_ids.size());
        state.speculative_main_hidden.reset(
            input_ids.size(),
            speculative_hidden_columns,
            true);
        state.speculative_main_hidden_position = position_offset;
        if (state.speculative_layers.size() != model.speculative.graph.layer_plans.size())
            state.speculative_layers.resize(model.speculative.graph.layer_plans.size());
        if (model.speculative.kind == SpeculativeModelKind::Mtp)
        {
            state.speculative_input_ids.assign(
                input_ids.begin(),
                input_ids.end());
            state.speculative_direct_alignment_ids.clear();
        }
    }
    else
    {
        state.speculative_main_hidden.clear();
    }
    std::vector<std::vector<float>> logits;
    std::vector<LayerGraphState>& layer_states = state.execution_layers;
    PendingRouterPrediction pending_router_prediction;
    bool deferred_final_norm = false;
    for (const ExecutionBackendRun& backend_run : model.schedule.backend_runs)
    {
        for (uint32_t run_offset = 0;
             run_offset < backend_run.node_count;
             ++run_offset)
        {
            if (backend_run.first_node + run_offset >= model.schedule.node_order.size())
                return Error{ErrorCode::InternalError, "execution backend run exceeds the execution reservation"};
            const ExecutionNodeId node_id = model.schedule.node_order[backend_run.first_node + run_offset];
            if (node_id >= model.graph.nodes.size())
                return Error{ErrorCode::InternalError, "execution schedule references an invalid node"};
            const ExecutionNode* node = &model.graph.nodes[node_id];
            if (node->type == ExecutionNodeType::TokenEmbedding)
            {
                if (node->weight_inputs.size() != 1)
                    return Error{ErrorCode::InternalError, "token embedding node has an invalid weight binding"};
                const auto embedding_start = std::chrono::steady_clock::now();
                embedding_batch_into(
                    model.weights.at(node->weight_inputs[0]),
                    input_ids,
                    hidden);
                hyper_connection_expand(hidden, model.descriptor.hyper_connection_multiplier, state.expert_scratch.staged_output);
                statistics.embedding_time_microseconds += elapsed_microseconds(embedding_start);
                continue;
            }
            if (node->type == ExecutionNodeType::FinalNorm)
            {
                const size_t expected_weights = model.descriptor.final_norm == NormType::None ? 1 : 2;
                if (node->weight_inputs.size() != expected_weights)
                    return Error{ErrorCode::InternalError, "final norm node has an invalid weight binding"};
                const TensorHandle lm_head_handle = node->weight_inputs.back();
                const CompiledOperator& lm_head_operator = model.operators.at_weight(lm_head_handle);
                if (model.descriptor.final_norm == NormType::RmsNorm
                    && model.descriptor.hyper_connection_multiplier == 1
                    && !state.use_speculative_context
                    && lm_head_operator.bfloat16
                    && lm_head_operator.bfloat16->has_rms_norm_chain())
                {
                    deferred_final_norm = true;
                    continue;
                }
                const auto final_norm_start = std::chrono::steady_clock::now();
                if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
                {
                    auto head = hyper_connection_head(
                        hidden,
                        model.weights.at(model.hyper_head_function),
                        model.weights.at(model.hyper_head_scale),
                        model.weights.at(model.hyper_head_base),
                        model.descriptor.hyper_connection_multiplier,
                        model.descriptor.norm_epsilon,
                        model.descriptor.hyper_connection_epsilon,
                        model.opt.optimization_flags);
                    if (!head)
                        return head.error();
                    hidden = std::move(head).value();
                }
                else if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
                {
                    auto head = gated_residual_head(
                        hidden,
                        model.weights.at(model.gated_residual_head.norm_weight),
                        model.weights.at(model.gated_residual_head.mix_down_weight),
                        model.weights.at(model.gated_residual_head.mix_up_weight),
                        model.descriptor.hyper_connection_multiplier,
                        model.descriptor.hidden_size,
                        model.descriptor.norm_epsilon,
                        model.descriptor.norm_weight_offset,
                        model.opt.optimization_flags);
                    if (!head)
                        return head.error();
                    hidden = std::move(head).value();
                }
                if (model.descriptor.final_norm == NormType::RmsNorm)
                {
                    rms_norm_batch_into(
                        hidden,
                        model.weights.at(node->weight_inputs[0]),
                        model.descriptor.norm_epsilon,
                        state.expert_scratch.staged_output,
                        model.descriptor.norm_weight_offset,
                        model.opt.optimization_flags);
                    hidden.swap(state.expert_scratch.staged_output);
                }
                if (model.speculative.kind == SpeculativeModelKind::Mtp
                    && state.use_speculative_context)
                {
                    for (size_t row = 0; row < hidden.rows(); ++row)
                    {
                        std::copy_n(
                            hidden.row(row),
                            model.descriptor.hidden_size,
                            state.speculative_main_hidden.row(row));
                    }
                }
                statistics.final_norm_time_microseconds += elapsed_microseconds(final_norm_start);
                continue;
            }
            if (node->type == ExecutionNodeType::LmHead)
            {
                const size_t expected_weights = model.descriptor.final_norm == NormType::None ? 1 : 2;
                if (node->weight_inputs.size() != expected_weights)
                    return Error{ErrorCode::InternalError, "LM head node has an invalid weight binding"};
                const auto lm_head_start = std::chrono::steady_clock::now();
                const auto& lm_head = model.weights.at(node->weight_inputs[0]);
                const CompiledOperator& lm_head_operator = model.operators.at_weight(node->weight_inputs[0]);
                if (deferred_final_norm
                    && try_fused_rms_norm_linear(
                        lm_head_operator,
                        hidden,
                        state.expert_scratch.staged_output))
                {
                    logits = batch_to_vectors(state.expert_scratch.staged_output);
                    deferred_final_norm = false;
                    statistics.lm_head_time_microseconds += elapsed_microseconds(lm_head_start);
                    continue;
                }
                if (deferred_final_norm)
                {
                    CpuBatch normalized;
                    rms_norm_batch_into(
                        hidden,
                        model.weights.at(node->weight_inputs[1]),
                        model.descriptor.norm_epsilon,
                        normalized,
                        model.descriptor.norm_weight_offset,
                        model.opt.optimization_flags);
                    hidden.swap(normalized);
                    deferred_final_norm = false;
                }
                logits = batch_to_vectors(linear_batch(
                    lm_head,
                    hidden,
                    model.opt.optimization_flags,
                    model.operators.find_weight(node->weight_inputs[0]),
                    node->backend));
                statistics.lm_head_time_microseconds += elapsed_microseconds(lm_head_start);
                continue;
            }
            if (node->layer_plan_index >= model.graph.layer_plans.size())
                return Error{ErrorCode::InternalError, "execution node layer is out of range"};

            const CompiledLayerPlan& layer = model.graph.layer_plans[node->layer_plan_index];
            if (layer.layer_id >= layer_states.size())
                return Error{ErrorCode::InternalError, "execution layer id is out of range"};
            LayerGraphState& layer_state = layer_states[layer.layer_id];
            const MoeBlockPlan& moe = layer.moe;
            if (node->type == ExecutionNodeType::Attention)
            {
                const auto attention_start = std::chrono::steady_clock::now();
                if (layer.ple.enabled())
                {
                    auto ple = execute_ple_into(
                        model.weights, layer.ple,
                        model.descriptor.hyper_connection_multiplier,
                        model.descriptor.hidden_size,
                        model.descriptor.norm_epsilon,
                        model.descriptor.norm_weight_offset,
                        input_ids, state.layers[layer.layer_id], hidden,
                        model.opt.optimization_flags);
                    if (!ple)
                        return ple.error();
                }
                CpuHyperConnectionMix gated_attention_mix;
                const CpuBatch* gated_attention_input = &hidden;
                if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
                {
                    auto mixed = gated_residual_pre(
                        hidden,
                        model.weights.at(layer.attention_gated_residual.norm_weight),
                        model.weights.at(layer.attention_gated_residual.mix_down_weight),
                        model.weights.at(layer.attention_gated_residual.mix_up_weight),
                        model.weights.at(layer.attention_gated_residual.inject_weight),
                        model.descriptor.hyper_connection_multiplier,
                        model.descriptor.hidden_size,
                        model.descriptor.norm_epsilon,
                        model.descriptor.norm_weight_offset,
                        model.opt.optimization_flags);
                    if (!mixed)
                        return mixed.error();
                    gated_attention_mix = std::move(mixed).value();
                    gated_attention_input = &gated_attention_mix.reduced;
                }
                if (layer.attention.kind == AttentionKind::GatedDeltaNet)
                {
                    auto gated_delta = execute_gated_delta_net_into(
                        model.weights,
                        model.operators,
                        layer.attention,
                        node->backend,
                        model.descriptor.norm_epsilon,
                        state.layers[layer.layer_id],
                        state.gated_delta_scratch,
                        *gated_attention_input,
                        state.gated_delta_scratch.output,
                        model.opt.optimization_flags);
                    if (!gated_delta)
                        return gated_delta.error();
                    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
                    {
                        auto connected = gated_residual_post(
                            state.gated_delta_scratch.output, hidden,
                            gated_attention_mix,
                            model.descriptor.hyper_connection_multiplier);
                        if (!connected)
                            return connected.error();
                        hidden = std::move(connected).value();
                    }
                    else
                    {
                        hidden.swap(state.gated_delta_scratch.output);
                    }
                }
                else if (layer.attention.kind == AttentionKind::MultiHeadLatent)
                {
                    CpuHyperConnectionMix hyper_mix;
                    const CpuBatch* attention_input = &hidden;
                    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
                    {
                        auto mixed = hyper_connection_pre(
                            hidden,
                            model.weights.at(layer.hyper_connection.attention_function),
                            model.weights.at(layer.hyper_connection.attention_scale),
                            model.weights.at(layer.hyper_connection.attention_base),
                            model.descriptor.hyper_connection_multiplier,
                            model.descriptor.hyper_connection_iterations,
                            model.descriptor.norm_epsilon,
                            model.descriptor.hyper_connection_epsilon,
                            model.opt.optimization_flags);
                        if (!mixed)
                            return mixed.error();
                        hyper_mix = std::move(mixed).value();
                        attention_input = &hyper_mix.reduced;
                    }
                    auto output = execute_latent_attention(
                        model.weights,
                        model.operators,
                        layer.attention,
                        node->backend,
                        model.descriptor.norm_epsilon,
                        position_offset,
                        state.layers[layer.layer_id],
                        *attention_input,
                        model.opt.optimization_flags);
                    if (!output)
                        return output.error();
                    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
                    {
                        auto connected = hyper_connection_post(output.value(), hidden, hyper_mix, model.descriptor.hyper_connection_multiplier);
                        if (!connected)
                            return connected.error();
                        hidden = std::move(connected).value();
                    }
                    else
                    {
                        add_batch_inplace(hidden, output.value());
                    }
                }
                else
                {
                    auto attention = execute_attention_block_into(
                        model.weights,
                        model.operators,
                        layer.attention,
                        node->backend,
                        model.descriptor.norm_epsilon,
                        model.descriptor.kv_cache_dtype,
                        position_offset,
                        state.layers[layer.layer_id],
                        state.attention_scratch,
                        *gated_attention_input,
                        state.attention_scratch.output,
                        model.opt.optimization_flags);
                    if (!attention)
                        return attention.error();
                    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
                    {
                        auto connected = gated_residual_post(
                            state.attention_scratch.output, hidden,
                            gated_attention_mix,
                            model.descriptor.hyper_connection_multiplier);
                        if (!connected)
                            return connected.error();
                        hidden = std::move(connected).value();
                    }
                    else
                    {
                        hidden.swap(state.attention_scratch.output);
                    }
                }
                statistics.attention_time_microseconds += elapsed_microseconds(attention_start);
                continue;
            }
            if (node->type == ExecutionNodeType::Router)
            {
                auto completed_prediction = complete_router_prediction(
                    model,
                    layer.layer_id,
                    pending_router_prediction,
                    state,
                    statistics,
                    hidden.rows());
                if (!completed_prediction)
                    return completed_prediction.error();
                layer_state.router_start = std::chrono::steady_clock::now();
                if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
                {
                    auto mixed = gated_residual_pre(
                        hidden,
                        model.weights.at(layer.ffn_gated_residual.norm_weight),
                        model.weights.at(layer.ffn_gated_residual.mix_down_weight),
                        model.weights.at(layer.ffn_gated_residual.mix_up_weight),
                        model.weights.at(layer.ffn_gated_residual.inject_weight),
                        model.descriptor.hyper_connection_multiplier,
                        model.descriptor.hidden_size,
                        model.descriptor.norm_epsilon,
                        model.descriptor.norm_weight_offset,
                        model.opt.optimization_flags);
                    if (!mixed)
                        return mixed.error();
                    layer_state.ffn_hyper_mix = std::move(mixed).value();
                    layer_state.normalized = layer_state.ffn_hyper_mix.reduced;
                }
                else if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
                {
                    auto mixed = hyper_connection_pre(
                        hidden,
                        model.weights.at(layer.hyper_connection.ffn_function),
                        model.weights.at(layer.hyper_connection.ffn_scale),
                        model.weights.at(layer.hyper_connection.ffn_base),
                        model.descriptor.hyper_connection_multiplier,
                        model.descriptor.hyper_connection_iterations,
                        model.descriptor.norm_epsilon,
                        model.descriptor.hyper_connection_epsilon,
                        model.opt.optimization_flags);
                    if (!mixed)
                        return mixed.error();
                    layer_state.ffn_hyper_mix = std::move(mixed).value();
                    rms_norm_batch_into(layer_state.ffn_hyper_mix.reduced, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset, model.opt.optimization_flags);
                }
                else
                {
                    rms_norm_batch_into(hidden, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset, model.opt.optimization_flags);
                }
                auto predicted = predict_next_router_routes(
                    model,
                    layer,
                    layer_state.normalized,
                    state,
                    statistics,
                    pending_router_prediction);
                if (!predicted)
                    return predicted.error();
                linear_batch_into(
                    model.weights.at(moe.router_weight),
                    layer_state.normalized,
                    layer_state.router_logits,
                    model.opt.optimization_flags,
                    model.operators.find_weight(moe.router_weight));
                if (moe.router_bias != invalid_tensor_handle)
                {
                    add_bias_inplace(layer_state.router_logits, model.weights.at(moe.router_bias));
                }
                continue;
            }
            if (node->type == ExecutionNodeType::ExpertDispatch)
            {
                ExpertDispatchOptions options;
                options.expert_count = static_cast<uint32_t>(moe.experts.size());
                options.top_k = moe.top_k;
                options.score_function = moe.score_function;
                options.normalization = moe.normalization;
                options.routed_scaling_factor = moe.routed_scaling_factor;
                if (moe.router_selection_bias != invalid_tensor_handle)
                    options.selection_bias = model.weights.at(moe.router_selection_bias).float32_values();
                std::vector<uint32_t> explicit_expert_ids;
                if (moe.token_experts != invalid_tensor_handle)
                {
                    const std::span<const int64_t> table = model.weights.at(moe.token_experts).int64_values();
                    explicit_expert_ids.resize(input_ids.size() * moe.top_k);
                    for (size_t token_index = 0; token_index < input_ids.size(); ++token_index)
                    {
                        for (uint32_t route = 0; route < moe.top_k; ++route)
                            explicit_expert_ids[token_index * moe.top_k + route] = static_cast<uint32_t>(table[static_cast<size_t>(input_ids[token_index]) * moe.top_k + route]);
                    }
                    options.explicit_expert_ids = explicit_expert_ids;
                }
                auto dispatched = dispatch_experts_into(layer_state.router_logits.values(), static_cast<uint32_t>(layer_state.router_logits.rows()), options, layer_state.dispatch_plan);
                if (!dispatched)
                    return dispatched.error();

                const ExpertDispatchPlan& plan = layer_state.dispatch_plan;
                statistics.expert_assignments += static_cast<uint64_t>(plan.assignment_count);
                layer_state.active_experts.resize(plan.batches.size());
                if (hidden.rows() == 1 && layer.layer_id < state.layers.size())
                    resolve_router_predictions(model, layer, plan, state, statistics, true);
                for (size_t batch_index = 0; batch_index < plan.batches.size(); ++batch_index)
                {
                    const ExpertBatch& batch = plan.batches[batch_index];
                    statistics.expert_token_counts[batch.expert_id] += static_cast<uint64_t>(batch.routes.size());
                    const ExpertPlan& expert = moe.experts[batch.expert_id];
                    record_expert_weight_demand(expert, batch.routes.size(), statistics);
                    ActiveExpertExecution& active = layer_state.active_experts[batch_index];
                    active.prepare(batch);
                }
                layer_state.router_logits.clear();
                statistics.router_time_microseconds += elapsed_microseconds(layer_state.router_start);
                layer_state.expert_start = std::chrono::steady_clock::now();

                continue;
            }
            if (node->type == ExecutionNodeType::Expert || node->type == ExecutionNodeType::ExpertGroup)
            {
                if (layer_state.experts_executed)
                    continue;
                const auto expert_engine_start = std::chrono::steady_clock::now();
                auto executed = forward_moe(
                    model,
                    moe,
                    layer_state,
                    statistics,
                    state.expert_scratch,
                    layer.layer_id,
                    node->backend,
                    has_flag(node->flags, ExecutionNodeCpuPrefetch));
                statistics.expert_engine_time_microseconds += elapsed_microseconds(expert_engine_start);
                if (!executed)
                    return executed.error();
                continue;
            }
            if (node->type == ExecutionNodeType::SharedExpertGroup)
            {
                if (!layer_state.experts_executed || !moe.has_shared_expert)
                    return Error{ErrorCode::InternalError, "Shared Expert executed before routed Expert group"};
                const auto shared_start = std::chrono::steady_clock::now();
                ExpertExecutionMetrics shared_metrics;
                layer_state.shared_expert_output = forward_shared_expert(
                    model,
                    moe,
                    layer_state.normalized,
                    shared_metrics,
                    model.opt.optimization_flags);
                statistics.expert_compute_time_microseconds += elapsed_microseconds(shared_start);
                continue;
            }
            if (node->type == ExecutionNodeType::Combine)
            {
                if (!layer_state.experts_executed)
                {
                    return Error{ErrorCode::InternalError, "Combine executed before its Expert wave"};
                }
                const auto combine_start = std::chrono::steady_clock::now();
                if (moe.has_shared_expert && layer_state.shared_expert_output.rows() == 0)
                    return Error{ErrorCode::InternalError, "Combine executed before Shared Expert group"};
                CpuBatch& moe_output = layer_state.normalized;
                const bool has_backend_aggregation = initialize_backend_aggregated_output(
                    state.expert_scratch,
                    hidden.rows(),
                    model.descriptor.hidden_size,
                    moe_output);
                for (size_t active_index = 0; active_index < layer_state.active_experts.size(); ++active_index)
                {
                    const ActiveExpertExecution& active = layer_state.active_experts[active_index];
                    if (has_backend_aggregation
                        && active_index < state.expert_scratch.backend_aggregated.size()
                        && state.expert_scratch.backend_aggregated[active_index] != 0)
                    {
                        continue;
                    }
                    for (size_t batch_index = 0; batch_index < active.batch.routes.size(); ++batch_index)
                    {
                        const ExpertRoute& route = active.batch.routes[batch_index];
                        float* destination = moe_output.row(route.token_index);
                        const float* source = active.output.row(batch_index);
                        for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
                        {
                            destination[column] += route.weight * source[column];
                        }
                    }
                }
                if (moe.has_shared_expert)
                {
                    add_batch_inplace(moe_output, layer_state.shared_expert_output);
                }
                if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
                {
                    auto connected = gated_residual_post(
                        moe_output, hidden, layer_state.ffn_hyper_mix,
                        model.descriptor.hyper_connection_multiplier);
                    if (!connected)
                        return connected.error();
                    hidden = std::move(connected).value();
                }
                else if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
                {
                    auto connected = hyper_connection_post(moe_output, hidden, layer_state.ffn_hyper_mix, model.descriptor.hyper_connection_multiplier);
                    if (!connected)
                        return connected.error();
                    hidden = std::move(connected).value();
                }
                else
                {
                    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
                    {
                        float* hidden_row = hidden.row(token_index);
                        const float* output_row = moe_output.row(token_index);
                        for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
                            hidden_row[column] += output_row[column];
                    }
                }
                statistics.expert_combine_time_microseconds += elapsed_microseconds(combine_start);

                statistics.expert_time_microseconds += elapsed_microseconds(layer_state.expert_start);
                const auto target = std::find(
                    model.speculative.target_layer_ids.begin(),
                    model.speculative.target_layer_ids.end(),
                    layer.layer_id);
                if (target != model.speculative.target_layer_ids.end())
                {
                    capture_speculative_hidden(
                        hidden,
                        model.descriptor.hidden_size,
                        model.descriptor.hyper_connection_multiplier,
                        static_cast<size_t>(std::distance(model.speculative.target_layer_ids.begin(), target)),
                        state.speculative_main_hidden);
                }
                layer_state.reset();
                continue;
            }
            return Error{ErrorCode::InternalError, "unsupported execution graph node"};
        }
    }

    if (pending_router_prediction.result.valid())
    {
        return Error{
            ErrorCode::InternalError,
            "execution graph ended before a Router prediction target"};
    }

    record_model_resource_delta(model, statistics, execution_cache_before, expert_backend_before);

    if (logits.empty())
        return Error{ErrorCode::InternalError, "execution graph did not produce logits"};
    record_vulkan_execution_delta(
        statistics,
        initial_vulkan_execution,
        model.vulkan_context_instance);
    statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
    return logits;
}

Result<std::vector<std::vector<float>>> forward_decode_batch(const CompiledModel& model, std::span<const CpuDecodeBatchEntry> entries)
{
    const ScopedExpertBackendForeground expert_backend_foreground(
        model.expert_backend);
    if (entries.empty())
        return Error{ErrorCode::InvalidArgument, "decode batch cannot be empty"};

    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
    {
        std::vector<std::vector<float>> results;
        results.reserve(entries.size());
        for (const CpuDecodeBatchEntry& entry : entries)
        {
            if (!entry.statistics || !entry.state)
                return Error{ErrorCode::InvalidArgument, "decode batch entry is incomplete"};
            const std::array<int32_t, 1> input = {entry.input_id};
            auto executed = forward_model(
                model, input, *entry.statistics, *entry.state,
                entry.position_offset);
            if (!executed)
                return executed.error();
            if (executed.value().size() != 1)
                return Error{ErrorCode::InternalError, "gated-residual decode produced an invalid row count"};
            results.push_back(std::move(executed).value().front());
        }
        return results;
    }

    const size_t session_count = entries.size();
    Bfloat16BatchedLinearExecutionCounter cpu_bfloat16_execution;
    const ScopedBfloat16BatchedLinearExecutionCounter cpu_bfloat16_scope(
        &cpu_bfloat16_execution);
    for (const CpuDecodeBatchEntry& entry : entries)
    {
        if (!entry.state || has_unknown_vulkan_attention_state(*entry.state))
        {
            return Error{
                ErrorCode::InternalError,
                "staged Session state is unavailable after a failed Vulkan Attention update"};
        }
    }
    const uint32_t hyper_multiplier = model.descriptor.hyper_connection_multiplier;
    const uint32_t hyper_iterations = model.descriptor.hyper_connection_iterations;
    const float hyper_epsilon = model.descriptor.hyper_connection_epsilon;
    std::vector<CpuBatch> hidden(session_count);
    std::vector<std::vector<float>> logits(session_count);
    CpuExpertExecutionScratch& batch_scratch = entries.front().state->expert_scratch;
    for (const CpuDecodeBatchEntry& entry : entries)
    {
        if (!entry.statistics || !entry.state)
        {
            return Error{ErrorCode::InvalidArgument, "decode batch entry is incomplete"};
        }
        if (entry.input_id < 0 || static_cast<uint32_t>(entry.input_id) >= model.descriptor.vocabulary_size)
        {
            return Error{ErrorCode::InvalidArgument, "token id is outside the model vocabulary"};
        }
        prepare_execution_state(model, *entry.statistics, *entry.state);
    }
    ExpertCacheStatistics cache_before;
    if (model.expert_cache)
        cache_before = model.expert_cache->statistics();
    ExpertBackendStatistics backend_before;
    if (model.expert_backend)
    {
        backend_before = model.expert_backend->statistics();
    }
    if (model.speculative.enabled())
    {
        const uint32_t speculative_hidden_columns = model.speculative.kind == SpeculativeModelKind::Mtp
                                                        ? model.descriptor.hidden_size
                                                        : model.descriptor.hidden_size
                                                              * static_cast<uint32_t>(
                                                                  model.speculative.target_layer_ids.size());
        for (const CpuDecodeBatchEntry& entry : entries)
        {
            if (!entry.state->use_speculative_context)
                continue;
            entry.state->speculative_main_hidden.reset(1, speculative_hidden_columns, true);
            entry.state->speculative_main_hidden_position = entry.position_offset;
            if (entry.state->speculative_layers.size() != model.speculative.graph.layer_plans.size())
            {
                entry.state->speculative_layers.resize(model.speculative.graph.layer_plans.size());
            }
            if (model.speculative.kind == SpeculativeModelKind::Mtp)
            {
                entry.state->speculative_input_ids.assign(
                    1,
                    entry.input_id);
                entry.state->speculative_direct_alignment_ids.clear();
            }
        }
    }

    auto merge_rows_into = [session_count](const std::vector<CpuBatch>& batches, CpuBatch& merged) {
        if (batches.empty() || batches.front().rows() != 1)
            return false;
        merged.reset(session_count, batches.front().columns(), false);
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            if (batches[session_index].rows() != 1 || batches[session_index].columns() != merged.columns())
            {
                return false;
            }
            std::copy_n(batches[session_index].row(0), merged.columns(), merged.row(session_index));
        }
        return true;
    };
    auto split_rows = [session_count](const CpuBatch& merged, std::vector<CpuBatch>& batches) {
        batches.resize(session_count);
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            batches[session_index].reset(1, merged.columns(), false);
            std::copy_n(merged.row(session_index), merged.columns(), batches[session_index].row(0));
        }
    };
    auto split_hyper_mix = [session_count, hyper_multiplier](const CpuHyperConnectionMix& merged, std::vector<CpuHyperConnectionMix>& mixes) {
        mixes.resize(session_count);
        const size_t post_stride = hyper_multiplier;
        const size_t combine_stride = static_cast<size_t>(hyper_multiplier) * hyper_multiplier;
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            mixes[session_index].reduced.reset(1, merged.reduced.columns(), false);
            std::copy_n(merged.reduced.row(session_index), merged.reduced.columns(), mixes[session_index].reduced.row(0));
            const auto post_begin = merged.post.begin() + session_index * post_stride;
            mixes[session_index].post.assign(post_begin, post_begin + post_stride);
            const auto combine_begin = merged.combine.begin() + session_index * combine_stride;
            mixes[session_index].combine.assign(combine_begin, combine_begin + combine_stride);
        }
    };
    auto merge_hyper_mixes = [session_count, hyper_multiplier](const std::vector<CpuHyperConnectionMix>& mixes, CpuHyperConnectionMix& merged) {
        const size_t post_stride = hyper_multiplier;
        const size_t combine_stride = static_cast<size_t>(hyper_multiplier) * hyper_multiplier;
        merged.post.resize(session_count * post_stride);
        merged.combine.resize(session_count * combine_stride);
        for (size_t session_index = 0; session_index < session_count; ++session_index)
        {
            std::copy_n(mixes[session_index].post.data(), post_stride, merged.post.data() + session_index * post_stride);
            std::copy_n(mixes[session_index].combine.data(), combine_stride, merged.combine.data() + session_index * combine_stride);
        }
    };

    const bool use_speculative_context = std::any_of(
        entries.begin(), entries.end(), [](const CpuDecodeBatchEntry& entry) {
            return entry.state->use_speculative_context;
        });
    bool deferred_final_norm = false;
    for (const ExecutionBackendRun& backend_run : model.schedule.backend_runs)
    {
        for (uint32_t run_offset = 0;
             run_offset < backend_run.node_count;
             ++run_offset)
        {
            if (backend_run.first_node + run_offset >= model.schedule.node_order.size())
            {
                return Error{ErrorCode::InternalError, "execution backend run exceeds the execution reservation"};
            }
            const ExecutionNodeId node_id = model.schedule.node_order[backend_run.first_node + run_offset];
            if (node_id >= model.graph.nodes.size())
            {
                return Error{ErrorCode::InternalError, "execution schedule references an invalid node"};
            }
            const ExecutionNode* node = &model.graph.nodes[node_id];
            if (node->type == ExecutionNodeType::TokenEmbedding)
            {
                if (node->weight_inputs.size() != 1)
                    return Error{ErrorCode::InternalError, "token embedding node has an invalid weight binding"};
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                std::vector<int32_t>& input_ids = scratch.staged_input_ids;
                input_ids.resize(session_count);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    input_ids[session_index] = entries[session_index].input_id;
                }
                embedding_batch_into(
                    model.weights.at(node->weight_inputs[0]),
                    input_ids,
                    scratch.staged_output);
                if (hyper_multiplier > 1)
                {
                    hyper_connection_expand(scratch.staged_output, hyper_multiplier, scratch.staged_merged);
                }
                split_rows(scratch.staged_output, hidden);
                const uint64_t elapsed = elapsed_microseconds(start);
                for (const CpuDecodeBatchEntry& entry : entries)
                    entry.statistics->embedding_time_microseconds += elapsed;
                continue;
            }
            if (node->type == ExecutionNodeType::FinalNorm)
            {
                if (node->weight_inputs.size() != 2)
                    return Error{ErrorCode::InternalError, "final norm node has an invalid weight binding"};
                const CompiledOperator& lm_head_operator = model.operators.at_weight(
                    node->weight_inputs[1]);
                if (hyper_multiplier == 1
                    && !use_speculative_context
                    && lm_head_operator.bfloat16
                    && lm_head_operator.bfloat16->has_rms_norm_chain())
                {
                    deferred_final_norm = true;
                    continue;
                }
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                if (hyper_multiplier > 1)
                {
                    CpuBatch& merged_hyper = scratch.staged_merged;
                    if (!merge_rows_into(hidden, merged_hyper))
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "cannot merge staged hyper head rows"};
                    }
                    auto head = hyper_connection_head(merged_hyper, model.weights.at(model.hyper_head_function), model.weights.at(model.hyper_head_scale), model.weights.at(model.hyper_head_base),
                                                      hyper_multiplier, model.descriptor.norm_epsilon, hyper_epsilon,
                                                      model.opt.optimization_flags);
                    if (!head)
                        return head.error();
                    split_rows(head.value(), hidden);
                }
                CpuBatch& merged = scratch.staged_merged;
                if (!merge_rows_into(hidden, merged))
                {
                    return Error{ErrorCode::InternalError, "cannot merge staged hidden rows"};
                }
                rms_norm_batch_into(merged, model.weights.at(node->weight_inputs[0]), model.descriptor.norm_epsilon, scratch.staged_output,
                                    model.descriptor.norm_weight_offset, model.opt.optimization_flags);
                split_rows(scratch.staged_output, hidden);
                if (model.speculative.kind == SpeculativeModelKind::Mtp)
                {
                    for (size_t session_index = 0;
                         session_index < session_count;
                         ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        if (!state.use_speculative_context)
                            continue;
                        std::copy_n(
                            hidden[session_index].row(0),
                            model.descriptor.hidden_size,
                            state.speculative_main_hidden.row(0));
                    }
                }
                const uint64_t elapsed = elapsed_microseconds(start);
                for (const CpuDecodeBatchEntry& entry : entries)
                    entry.statistics->final_norm_time_microseconds += elapsed;
                continue;
            }
            if (node->type == ExecutionNodeType::LmHead)
            {
                if (node->weight_inputs.size() != 2)
                    return Error{ErrorCode::InternalError, "LM head node has an invalid weight binding"};
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                CpuBatch& merged = scratch.staged_merged;
                if (!merge_rows_into(hidden, merged))
                {
                    return Error{ErrorCode::InternalError, "cannot merge staged LM head rows"};
                }
                const NcnnVulkanExecutionSnapshot vulkan_before = get_vulkan_execution_snapshot(model.vulkan_context_instance);
                const auto& lm_head = model.weights.at(node->weight_inputs[0]);
                const CompiledOperator& lm_head_operator = model.operators.at_weight(node->weight_inputs[0]);
                if (deferred_final_norm
                    && try_fused_rms_norm_linear(
                        lm_head_operator,
                        merged,
                        scratch.staged_output))
                {
                    deferred_final_norm = false;
                }
                else
                {
                    if (deferred_final_norm)
                    {
                        CpuBatch normalized;
                        rms_norm_batch_into(
                            merged,
                            model.weights.at(node->weight_inputs[1]),
                            model.descriptor.norm_epsilon,
                            normalized,
                            model.descriptor.norm_weight_offset,
                            model.opt.optimization_flags);
                        linear_batch_into(
                            lm_head,
                            normalized,
                            scratch.staged_output,
                            model.opt.optimization_flags,
                            model.operators.find_weight(node->weight_inputs[0]),
                            node->backend);
                        deferred_final_norm = false;
                    }
                    else
                    {
                        linear_batch_into(
                            lm_head,
                            merged,
                            scratch.staged_output,
                            model.opt.optimization_flags,
                            model.operators.find_weight(node->weight_inputs[0]),
                            node->backend);
                    }
                }
                const std::vector<std::vector<float>> merged_logits = batch_to_vectors(scratch.staged_output);
                const uint64_t elapsed = elapsed_microseconds(start);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    logits[session_index] = merged_logits[session_index];
                    entries[session_index].statistics->lm_head_time_microseconds += elapsed;
                }
                for (const CpuDecodeBatchEntry& entry : entries)
                    record_vulkan_execution_delta(*entry.statistics, vulkan_before, model.vulkan_context_instance);
                continue;
            }
            if (node->layer_plan_index >= model.graph.layer_plans.size())
            {
                return Error{ErrorCode::InternalError, "execution node layer is out of range"};
            }

            const CompiledLayerPlan& layer = model.graph.layer_plans[node->layer_plan_index];
            const MoeBlockPlan& moe = layer.moe;
            if (node->type == ExecutionNodeType::Attention)
            {
                if (layer.attention.kind == AttentionKind::GatedDeltaNet)
                {
                    std::vector<CpuGatedDeltaBatchEntry>& gated_delta_entries = batch_scratch.gated_delta_entries;
                    gated_delta_entries.resize(session_count);
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        gated_delta_entries[session_index] = {
                            &hidden[session_index],
                            &state.gated_delta_scratch,
                            &state.layers[layer.layer_id],
                            &state.gated_delta_scratch.output};
                    }
                    const auto start = std::chrono::steady_clock::now();
                    const NcnnVulkanExecutionSnapshot vulkan_before = get_vulkan_execution_snapshot(model.vulkan_context_instance);
                    if (!execute_gated_delta_net_batch_into(
                            model.weights,
                            model.operators,
                            layer.attention,
                            node->backend,
                            model.descriptor.norm_epsilon,
                            gated_delta_entries,
                            batch_scratch.gated_delta_device_entries,
                            model.opt.optimization_flags))
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "gated delta batch execution failed"};
                    }
                    const uint64_t elapsed = elapsed_microseconds(start);
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        hidden[session_index].swap(state.gated_delta_scratch.output);
                        SessionStatistics& statistics = *entries[session_index].statistics;
                        statistics.attention_time_microseconds += elapsed;
                        record_vulkan_execution_delta(statistics, vulkan_before, model.vulkan_context_instance);
                    }
                }
                else if (layer.attention.kind == AttentionKind::MultiHeadLatent)
                {
                    const auto start = std::chrono::steady_clock::now();
                    const NcnnVulkanExecutionSnapshot vulkan_before = get_vulkan_execution_snapshot(model.vulkan_context_instance);
                    std::vector<uint64_t>& positions = batch_scratch.staged_attention_positions;
                    std::vector<CpuLayerCache*>& caches = batch_scratch.staged_attention_caches;
                    positions.resize(session_count);
                    caches.resize(session_count);
                    CpuExpertExecutionScratch& scratch = batch_scratch;
                    CpuBatch& merged_hidden = scratch.staged_merged;
                    if (!merge_rows_into(hidden, merged_hidden))
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "cannot merge staged attention rows"};
                    }
                    CpuHyperConnectionMix merged_mix;
                    const CpuBatch* attention_input = &merged_hidden;
                    if (hyper_multiplier > 1)
                    {
                        auto mixed = hyper_connection_pre(merged_hidden, model.weights.at(layer.hyper_connection.attention_function), model.weights.at(layer.hyper_connection.attention_scale),
                                                          model.weights.at(layer.hyper_connection.attention_base), hyper_multiplier, hyper_iterations,
                                                          model.descriptor.norm_epsilon, hyper_epsilon, model.opt.optimization_flags);
                        if (!mixed)
                            return mixed.error();
                        merged_mix = std::move(mixed).value();
                        attention_input = &merged_mix.reduced;
                    }
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        positions[session_index] = entries[session_index].position_offset;
                        caches[session_index] = &state.layers[layer.layer_id];
                    }
                    auto merged_output = execute_latent_attention_batch(model.weights, model.operators, layer.attention, node->backend, model.descriptor.norm_epsilon, positions, caches,
                                                                        *attention_input, model.opt.optimization_flags);
                    if (!merged_output)
                        return merged_output.error();
                    if (hyper_multiplier > 1)
                    {
                        auto connected = hyper_connection_post(merged_output.value(), merged_hidden, merged_mix, hyper_multiplier);
                        if (!connected)
                            return connected.error();
                        split_rows(connected.value(), hidden);
                    }
                    else
                    {
                        std::vector<CpuBatch>& attention_outputs = batch_scratch.staged_batches;
                        split_rows(merged_output.value(), attention_outputs);
                        for (size_t session_index = 0; session_index < session_count; ++session_index)
                        {
                            add_batch_inplace(hidden[session_index], attention_outputs[session_index]);
                        }
                    }
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        SessionStatistics& statistics = *entries[session_index].statistics;
                        statistics.attention_time_microseconds += elapsed_microseconds(start);
                        record_vulkan_execution_delta(*entries[session_index].statistics, vulkan_before, model.vulkan_context_instance);
                    }
                }
                else
                {
                    std::vector<CpuAttentionBatchEntry>& attention_entries = batch_scratch.attention_batch_entries;
                    attention_entries.resize(session_count);
                    for (size_t session_index = 0;
                         session_index < session_count;
                         ++session_index)
                    {
                        CpuSessionState& state = *entries[session_index].state;
                        attention_entries[session_index] = {
                            entries[session_index].position_offset,
                            &state.layers[layer.layer_id],
                            &state.attention_scratch,
                            &hidden[session_index],
                            &state.attention_scratch.output};
                    }
                    const auto batch_start = std::chrono::steady_clock::now();
                    const NcnnVulkanExecutionSnapshot batch_vulkan_before = get_vulkan_execution_snapshot(model.vulkan_context_instance);
                    auto batched = execute_attention_block_batch_into(
                        model.operators,
                        layer.attention,
                        node->backend,
                        attention_entries,
                        model.opt.optimization_flags);
                    if (!batched)
                        return batched.error();
                    if (batched.value())
                    {
                        const uint64_t elapsed = elapsed_microseconds(batch_start);
                        for (size_t session_index = 0;
                             session_index < session_count;
                             ++session_index)
                        {
                            CpuSessionState& state = *entries[session_index].state;
                            hidden[session_index].swap(
                                state.attention_scratch.output);
                            SessionStatistics& statistics = *entries[session_index].statistics;
                            statistics.attention_time_microseconds += elapsed;
                            record_vulkan_execution_delta(
                                statistics,
                                batch_vulkan_before, model.vulkan_context_instance);
                        }
                    }
                    else
                    {
                        for (size_t session_index = 0;
                             session_index < session_count;
                             ++session_index)
                        {
                            CpuSessionState& state = *entries[session_index].state;
                            const auto start = std::chrono::steady_clock::now();
                            const NcnnVulkanExecutionSnapshot vulkan_before = get_vulkan_execution_snapshot(model.vulkan_context_instance);
                            auto attention = execute_attention_block_into(
                                model.weights,
                                model.operators,
                                layer.attention,
                                node->backend,
                                model.descriptor.norm_epsilon,
                                model.descriptor.kv_cache_dtype,
                                entries[session_index].position_offset,
                                state.layers[layer.layer_id],
                                state.attention_scratch,
                                hidden[session_index],
                                state.attention_scratch.output,
                                model.opt.optimization_flags);
                            if (!attention)
                            {
                                for (size_t affected_index = 0;
                                     affected_index < session_count;
                                     ++affected_index)
                                {
                                    CpuLayerCache& affected_cache = entries[affected_index].state->layers[layer.layer_id];
                                    affected_cache.vulkan_attention_state_unknown = true;
                                }
                                return attention.error();
                            }
                            hidden[session_index].swap(
                                state.attention_scratch.output);
                            SessionStatistics& statistics = *entries[session_index].statistics;
                            statistics.attention_time_microseconds += elapsed_microseconds(start);
                            record_vulkan_execution_delta(
                                statistics,
                                vulkan_before, model.vulkan_context_instance);
                        }
                    }
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Router)
            {
                const auto start = std::chrono::steady_clock::now();
                CpuExpertExecutionScratch& scratch = entries.front().state->expert_scratch;
                CpuBatch& merged_hidden = scratch.staged_merged;
                CpuBatch merged_hyper;
                if (!merge_rows_into(hidden, merged_hyper))
                {
                    return Error{
                        ErrorCode::InternalError,
                        "cannot merge staged FFN hyper rows"};
                }
                if (hyper_multiplier > 1)
                {
                    auto mixed = hyper_connection_pre(merged_hyper, model.weights.at(layer.hyper_connection.ffn_function), model.weights.at(layer.hyper_connection.ffn_scale),
                                                      model.weights.at(layer.hyper_connection.ffn_base), hyper_multiplier, hyper_iterations,
                                                      model.descriptor.norm_epsilon, hyper_epsilon, model.opt.optimization_flags);
                    if (!mixed)
                        return mixed.error();
                    CpuHyperConnectionMix merged_mix = std::move(mixed).value();
                    rms_norm_batch_into(merged_mix.reduced, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, merged_hidden,
                                        model.descriptor.norm_weight_offset, model.opt.optimization_flags);
                    std::vector<CpuHyperConnectionMix>& mixes = batch_scratch.staged_hyper_mixes;
                    split_hyper_mix(merged_mix, mixes);
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                        layer_state.ffn_hyper_mix = std::move(mixes[session_index]);
                    }
                }
                else
                {
                    rms_norm_batch_into(merged_hyper, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, merged_hidden,
                                        model.descriptor.norm_weight_offset, model.opt.optimization_flags);
                }
                std::vector<CpuBatch>& normalized = batch_scratch.staged_batches;
                split_rows(merged_hidden, normalized);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    layer_state.normalized = std::move(normalized[session_index]);
                }
                CpuBatch& merged_logits = scratch.staged_router_logits;
                linear_batch_into(
                    model.weights.at(moe.router_weight),
                    merged_hidden,
                    merged_logits,
                    model.opt.optimization_flags,
                    model.operators.find_weight(moe.router_weight));
                if (moe.router_bias != invalid_tensor_handle)
                {
                    add_bias_inplace(merged_logits, model.weights.at(moe.router_bias));
                }
                const auto router_start = std::chrono::steady_clock::now();
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    layer_state.router_logits.reset(1, merged_logits.columns(), false);
                    std::copy_n(merged_logits.row(session_index), merged_logits.columns(), layer_state.router_logits.row(0));
                    layer_state.router_start = router_start;
                }
                const uint64_t elapsed = elapsed_microseconds(start);
                for (const CpuDecodeBatchEntry& entry : entries)
                    entry.statistics->router_time_microseconds += elapsed;
                continue;
            }
            if (node->type == ExecutionNodeType::ExpertDispatch)
            {
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    CpuSessionState& state = *entries[session_index].state;
                    LayerGraphState& layer_state = state.execution_layers[layer.layer_id];
                    SessionStatistics& statistics = *entries[session_index].statistics;
                    ExpertDispatchOptions options;
                    options.expert_count = static_cast<uint32_t>(moe.experts.size());
                    options.top_k = moe.top_k;
                    options.score_function = moe.score_function;
                    options.normalization = moe.normalization;
                    options.routed_scaling_factor = moe.routed_scaling_factor;
                    if (moe.router_selection_bias != invalid_tensor_handle)
                    {
                        const TensorData& selection_bias = model.weights.at(moe.router_selection_bias);
                        options.selection_bias = selection_bias.float32_values();
                    }
                    std::vector<uint32_t>& explicit_expert_ids = batch_scratch.staged_expert_ids;
                    explicit_expert_ids.clear();
                    if (moe.token_experts != invalid_tensor_handle)
                    {
                        const std::span<const int64_t> table = model.weights.at(moe.token_experts).int64_values();
                        explicit_expert_ids.resize(moe.top_k);
                        for (uint32_t route = 0; route < moe.top_k; ++route)
                        {
                            explicit_expert_ids[route] = static_cast<uint32_t>(table[static_cast<size_t>(entries[session_index].input_id) * moe.top_k + route]);
                        }
                        options.explicit_expert_ids = explicit_expert_ids;
                    }
                    auto dispatched = dispatch_experts_into(layer_state.router_logits.values(), 1, options, layer_state.dispatch_plan);
                    if (!dispatched)
                        return dispatched.error();
                    const ExpertDispatchPlan& plan = layer_state.dispatch_plan;
                    statistics.expert_assignments += plan.assignment_count;
                    layer_state.active_experts.resize(plan.batches.size());
                    resolve_router_predictions(model, layer, plan, state, statistics, false);
                    for (size_t batch_index = 0; batch_index < plan.batches.size(); ++batch_index)
                    {
                        const ExpertBatch& batch = plan.batches[batch_index];
                        statistics.expert_token_counts[batch.expert_id] += batch.routes.size();
                        const ExpertPlan& expert = moe.experts[batch.expert_id];
                        record_expert_weight_demand(expert, batch.routes.size(), statistics);
                        ActiveExpertExecution& active = layer_state.active_experts[batch_index];
                        active.prepare(batch);
                    }
                    layer_state.router_logits.clear();
                    layer_state.expert_start = std::chrono::steady_clock::now();
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Expert || node->type == ExecutionNodeType::ExpertGroup)
            {
                LayerGraphState combined;
                combined.normalized.reset(session_count, model.descriptor.hidden_size, false);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    const LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    std::copy_n(layer_state.normalized.row(0), combined.normalized.columns(), combined.normalized.row(session_index));
                }
                const size_t missing = std::numeric_limits<size_t>::max();
                std::vector<size_t>& combined_by_expert = batch_scratch.combined_by_expert;
                combined_by_expert.assign(moe.experts.size(), missing);
                std::vector<std::vector<CpuDecodeRouteOrigin>>& origins = batch_scratch.staged_route_origins;
                origins.clear();
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    for (size_t active_index = 0; active_index < layer_state.active_experts.size(); ++active_index)
                    {
                        ActiveExpertExecution& source = layer_state.active_experts[active_index];
                        const uint32_t expert_id = source.batch.expert_id;
                        size_t combined_index = combined_by_expert[expert_id];
                        if (combined_index == missing)
                        {
                            combined_index = combined.active_experts.size();
                            combined_by_expert[expert_id] = combined_index;
                            ExpertBatch batch;
                            batch.expert_id = expert_id;
                            combined.active_experts.emplace_back();
                            combined.active_experts.back().prepare(std::move(batch));
                            origins.emplace_back();
                        }
                        ActiveExpertExecution& destination = combined.active_experts[combined_index];
                        for (size_t route_index = 0; route_index < source.batch.routes.size(); ++route_index)
                        {
                            ExpertRoute route = source.batch.routes[route_index];
                            route.token_index = static_cast<uint32_t>(session_index);
                            destination.batch.routes.push_back(route);
                            origins[combined_index].push_back(
                                {session_index, active_index, route_index});
                        }
                    }
                }

                SessionStatistics aggregate_statistics;
                aggregate_statistics.expert_token_counts.resize(model.descriptor.expert_count, 0);
                const auto engine_start = std::chrono::steady_clock::now();
                auto executed = forward_moe(
                    model,
                    moe,
                    combined,
                    aggregate_statistics,
                    entries.front().state->expert_scratch,
                    layer.layer_id,
                    node->backend,
                    has_flag(node->flags, ExecutionNodeCpuPrefetch));
                const uint64_t engine_elapsed = elapsed_microseconds(engine_start);
                if (!executed)
                    return executed.error();

                // Snapshot aggregated output before reusing session zero's scratch.
                const CpuExpertExecutionScratch& combined_scratch = entries.front().state->expert_scratch;
                std::vector<uint8_t>& combined_backend_aggregated = batch_scratch.combined_backend_aggregated;
                combined_backend_aggregated = combined_scratch.backend_aggregated;
                batch_scratch.combined_backend_aggregated_output_valid = combined_scratch.backend_aggregated_output_valid;
                const bool has_combined_backend_aggregation = batch_scratch.combined_backend_aggregated_output_valid
                                                              && combined_scratch.backend_aggregated_output.rows()
                                                                     == session_count
                                                              && combined_scratch.backend_aggregated_output.columns()
                                                                     == model.descriptor.hidden_size;
                CpuBatch& combined_backend_aggregated_output = batch_scratch.combined_backend_aggregated_output;
                combined_backend_aggregated_output.clear();
                if (has_combined_backend_aggregation)
                {
                    combined_backend_aggregated_output = combined_scratch.backend_aggregated_output;
                }
                for (size_t session_index = 0;
                     session_index < session_count;
                     ++session_index)
                {
                    CpuExpertExecutionScratch& session_scratch = entries[session_index].state->expert_scratch;
                    const LayerGraphState& session_layer = entries[session_index].state->execution_layers[layer.layer_id];
                    session_scratch.backend_aggregated.assign(
                        session_layer.active_experts.size(),
                        0);
                    session_scratch.backend_aggregated_output_valid = has_combined_backend_aggregation;
                    if (has_combined_backend_aggregation)
                    {
                        session_scratch.backend_aggregated_output.reset(
                            1,
                            model.descriptor.hidden_size,
                            false);
                        std::copy_n(
                            combined_backend_aggregated_output.row(
                                session_index),
                            model.descriptor.hidden_size,
                            session_scratch.backend_aggregated_output.row(0));
                    }
                    else
                    {
                        session_scratch.backend_aggregated_output.clear();
                    }
                }
                for (size_t combined_index = 0; combined_index < combined.active_experts.size(); ++combined_index)
                {
                    const ActiveExpertExecution& active = combined.active_experts[combined_index];
                    for (size_t route_index = 0; route_index < origins[combined_index].size(); ++route_index)
                    {
                        const ExpertRoute& route = active.batch.routes[route_index];
                        if (route.rank >= maximum_expert_route_ranks)
                            continue;
                        SessionStatistics& route_statistics = *entries[origins[combined_index][route_index].session_index].statistics;
                        ++route_statistics.expert_route_rank_demands[route.rank];
                        route_statistics.expert_route_rank_demand_queue_time_microseconds[route.rank] += active.metrics.cache_wait_time_microseconds;
                    }
                }

                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    SessionStatistics& statistics = *entries[session_index].statistics;
                    statistics.expert_engine_time_microseconds += engine_elapsed;
                    statistics.expert_compute_time_microseconds += aggregate_statistics.expert_compute_time_microseconds;
                    statistics.expert_cache_wait_time_microseconds += aggregate_statistics.expert_cache_wait_time_microseconds;
                    statistics.expert_cache_management_time_microseconds += aggregate_statistics.expert_cache_management_time_microseconds;
                    statistics.expert_regroup_time_microseconds += aggregate_statistics.expert_regroup_time_microseconds;
                    if (session_index == 0)
                    {
                        statistics.mxfp4_reused_input_rows += aggregate_statistics.mxfp4_reused_input_rows;
                    }
                    statistics.expert_batches += layer_state.active_experts.size();
                    for (ActiveExpertExecution& active : layer_state.active_experts)
                    {
                        active.output.reset(active.batch.routes.size(), model.descriptor.hidden_size, false);
                        const ExpertPlan& expert = moe.experts[active.batch.expert_id];
                        ExpertExecutionMetrics logical_metrics;
                        if (expert.gate_up_weight != invalid_tensor_handle)
                        {
                            const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                            const TensorData& down = model.weights.at(expert.down_weight);
                            record_mxfp4(gate_up, active.batch.routes.size(), logical_metrics);
                            record_mxfp4(down, active.batch.routes.size(), logical_metrics);
                            if (gate_up.dtype == DType::MxFp4)
                            {
                                logical_metrics.mxfp4_fused_gate_up_rows += static_cast<uint64_t>(active.batch.routes.size()) * gate_up.shape[0] / 2;
                            }
                        }
                        statistics.mxfp4_decode_gemv_rows += logical_metrics.mxfp4_decode_gemv_rows;
                        statistics.mxfp4_prefill_gemm_rows += logical_metrics.mxfp4_prefill_gemm_rows;
                        statistics.mxfp4_paired_rows += logical_metrics.mxfp4_paired_rows;
                        statistics.mxfp4_fused_gate_up_rows += logical_metrics.mxfp4_fused_gate_up_rows;
                    }
                    layer_state.experts_executed = true;
                }
                for (size_t combined_index = 0; combined_index < combined.active_experts.size(); ++combined_index)
                {
                    const ActiveExpertExecution& source = combined.active_experts[combined_index];
                    const bool backend_aggregated = has_combined_backend_aggregation
                                                    && combined_index
                                                           < combined_backend_aggregated.size()
                                                    && combined_backend_aggregated[combined_index] != 0;
                    for (size_t route_index = 0; route_index < origins[combined_index].size(); ++route_index)
                    {
                        const CpuDecodeRouteOrigin& origin = origins[combined_index][route_index];
                        ActiveExpertExecution& destination = entries[origin.session_index].state->execution_layers[layer.layer_id].active_experts[origin.active_index];
                        if (backend_aggregated)
                        {
                            CpuExpertExecutionScratch& origin_scratch = entries[origin.session_index].state->expert_scratch;
                            origin_scratch.backend_aggregated[origin.active_index] = 1;
                            continue;
                        }
                        std::copy_n(
                            source.output.row(route_index),
                            model.descriptor.hidden_size,
                            destination.output.row(origin.route_index));
                    }
                }
                continue;
            }
            if (node->type == ExecutionNodeType::SharedExpertGroup)
            {
                if (!moe.has_shared_expert)
                    return Error{ErrorCode::InternalError, "Shared Expert graph node has no shared Expert plan"};
                CpuBatch shared_input;
                shared_input.reset(session_count, model.descriptor.hidden_size, false);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    const CpuBatch& normalized = entries[session_index].state->execution_layers[layer.layer_id].normalized;
                    if (normalized.rows() != 1 || normalized.columns() != model.descriptor.hidden_size)
                        return Error{ErrorCode::InternalError, "Shared Expert input has an invalid shape"};
                    std::copy_n(normalized.row(0), normalized.columns(), shared_input.row(session_index));
                }
                const auto shared_start = std::chrono::steady_clock::now();
                ExpertExecutionMetrics shared_metrics;
                const CpuBatch shared_output = forward_shared_expert(
                    model,
                    moe,
                    shared_input,
                    shared_metrics,
                    model.opt.optimization_flags);
                const uint64_t shared_elapsed = elapsed_microseconds(shared_start);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    layer_state.shared_expert_output.reset(1, shared_output.columns(), false);
                    std::copy_n(shared_output.row(session_index), shared_output.columns(), layer_state.shared_expert_output.row(0));
                    entries[session_index].statistics->expert_compute_time_microseconds += shared_elapsed;
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Combine)
            {
                const auto combine_start = std::chrono::steady_clock::now();
                std::vector<CpuBatch>& moe_outputs = batch_scratch.staged_batches;
                moe_outputs.resize(session_count);
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    if (!layer_state.experts_executed)
                    {
                        return Error{ErrorCode::InternalError, "Combine executed before its Expert wave"};
                    }
                    if (moe.has_shared_expert && layer_state.shared_expert_output.rows() == 0)
                        return Error{ErrorCode::InternalError, "Combine executed before Shared Expert group"};
                    CpuBatch& moe_output = layer_state.normalized;
                    const CpuExpertExecutionScratch& expert_scratch = entries[session_index].state->expert_scratch;
                    const bool has_backend_aggregation = initialize_backend_aggregated_output(
                        expert_scratch,
                        1,
                        model.descriptor.hidden_size,
                        moe_output);
                    for (size_t active_index = 0; active_index < layer_state.active_experts.size(); ++active_index)
                    {
                        const ActiveExpertExecution& active = layer_state.active_experts[active_index];
                        if (has_backend_aggregation
                            && active_index < expert_scratch.backend_aggregated.size()
                            && expert_scratch.backend_aggregated[active_index] != 0)
                        {
                            continue;
                        }
                        for (size_t batch_index = 0; batch_index < active.batch.routes.size(); ++batch_index)
                        {
                            const ExpertRoute& route = active.batch.routes[batch_index];
                            float* destination = moe_output.row(route.token_index);
                            const float* source = active.output.row(batch_index);
                            for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
                            {
                                destination[column] += route.weight * source[column];
                            }
                        }
                    }
                    if (moe.has_shared_expert)
                    {
                        add_batch_inplace(moe_output, layer_state.shared_expert_output);
                    }
                    moe_outputs[session_index] = std::move(moe_output);
                }
                if (hyper_multiplier > 1)
                {
                    CpuBatch merged_branch;
                    CpuBatch merged_residual;
                    CpuHyperConnectionMix merged_mix;
                    std::vector<CpuHyperConnectionMix>& mixes = batch_scratch.staged_hyper_mixes;
                    mixes.resize(session_count);
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                        mixes[session_index] = std::move(layer_state.ffn_hyper_mix);
                    }
                    merge_hyper_mixes(mixes, merged_mix);
                    if (!merge_rows_into(moe_outputs, merged_branch) || !merge_rows_into(hidden, merged_residual))
                    {
                        return Error{
                            ErrorCode::InternalError,
                            "cannot merge staged FFN post rows"};
                    }
                    auto connected = hyper_connection_post(merged_branch, merged_residual, merged_mix, hyper_multiplier);
                    if (!connected)
                        return connected.error();
                    split_rows(connected.value(), hidden);
                }
                else
                {
                    for (size_t session_index = 0; session_index < session_count; ++session_index)
                    {
                        add_batch_inplace(hidden[session_index], moe_outputs[session_index]);
                    }
                }
                for (size_t session_index = 0; session_index < session_count; ++session_index)
                {
                    LayerGraphState& layer_state = entries[session_index].state->execution_layers[layer.layer_id];
                    const auto target = std::find(model.speculative.target_layer_ids.begin(), model.speculative.target_layer_ids.end(), layer.layer_id);
                    if (target != model.speculative.target_layer_ids.end())
                    {
                        capture_speculative_hidden(hidden[session_index], model.descriptor.hidden_size, hyper_multiplier,
                                                   static_cast<size_t>(std::distance(model.speculative.target_layer_ids.begin(), target)),
                                                   entries[session_index].state->speculative_main_hidden);
                    }
                    SessionStatistics& statistics = *entries[session_index].statistics;
                    statistics.expert_combine_time_microseconds += elapsed_microseconds(combine_start);
                    statistics.expert_time_microseconds += elapsed_microseconds(layer_state.expert_start);
                    layer_state.reset();
                }
                continue;
            }
            return Error{ErrorCode::InternalError, "unsupported execution graph node"};
        }
    }

    record_batch_resource_delta(model, entries, cache_before, backend_before);
    for (size_t session_index = 0; session_index < session_count; ++session_index)
    {
        if (logits[session_index].empty())
        {
            return Error{ErrorCode::InternalError, "staged execution graph did not produce logits"};
        }
    }
    const uint64_t cpu_bfloat16_dispatches = cpu_bfloat16_execution.dispatch_count();
    for (const CpuDecodeBatchEntry& entry : entries)
    {
        entry.statistics->cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_dispatches;
    }
    return logits;
}

static void collect_ranked_experts(
    const ExpertDispatchPlan& plan,
    std::span<uint32_t> ranked)
{
    const uint32_t invalid_expert = std::numeric_limits<uint32_t>::max();
    std::fill(ranked.begin(), ranked.end(), invalid_expert);
    for (const ExpertBatch& batch : plan.batches)
    {
        for (const ExpertRoute& route : batch.routes)
        {
            if (route.token_index == 0 && route.rank < ranked.size())
                ranked[route.rank] = batch.expert_id;
        }
    }
}

static void configure_router_prefetch(
    CpuLayerCache& cache,
    uint32_t target_top_k)
{
    if (cache.router_target_top_k == target_top_k)
        return;
    cache.router_target_top_k = target_top_k;
    cache.router_prefetch_width = std::min(2u, target_top_k);
    cache.router_decisions = 0;
    cache.router_last_adjustment = 0;
}

static void adapt_router_prefetch_width(
    CpuLayerCache& cache,
    const SessionStatistics& statistics,
    bool adaptive)
{
    if (!adaptive)
    {
        cache.router_prefetch_width = cache.router_target_top_k;
        return;
    }
    if (cache.router_prefetch_width == 0)
        cache.router_prefetch_width = std::min(2u, cache.router_target_top_k);
    ++cache.router_decisions;
    if (cache.router_decisions % 16 != 0)
        return;

    if (cache.router_prefetch_width > 1)
    {
        const uint32_t marginal_rank = cache.router_prefetch_width - 1;
        const uint64_t predictions = statistics.expert_route_rank_predictions[marginal_rank];
        const uint64_t matches = statistics.expert_route_rank_matches[marginal_rank];
        if (predictions >= 8 && matches * 3 < predictions)
        {
            --cache.router_prefetch_width;
            cache.router_last_adjustment = cache.router_decisions;
            return;
        }
    }
    if (cache.router_decisions - cache.router_last_adjustment < 64)
        return;
    if (cache.router_prefetch_width < cache.router_target_top_k)
    {
        const uint32_t widest_prefetched_rank = cache.router_prefetch_width - 1;
        const uint64_t predictions = statistics.expert_route_rank_predictions[widest_prefetched_rank];
        const uint64_t matches = statistics.expert_route_rank_matches[widest_prefetched_rank];
        const uint32_t next_rank = cache.router_prefetch_width;
        const uint64_t demands = statistics.expert_route_rank_demands[next_rank];
        const uint64_t queued_microseconds = statistics.expert_route_rank_demand_queue_time_microseconds[next_rank];
        if (predictions >= 8
            && matches * 2 >= predictions
            && demands >= 8
            && queued_microseconds / demands >= 2000)
        {
            ++cache.router_prefetch_width;
            cache.router_last_adjustment = cache.router_decisions;
        }
    }
}

static void resolve_router_predictions(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    const ExpertDispatchPlan& plan,
    CpuSessionState& state,
    SessionStatistics& statistics,
    bool resolve_unused_predictions)
{
    if (!model.expert_cache
        || layer.layer_id >= state.layers.size()
        || !has_flag(model.opt.flags, OptionRouterPrediction))
    {
        return;
    }

    const uint32_t invalid_expert = std::numeric_limits<uint32_t>::max();
    CpuLayerCache& cache = state.layers[layer.layer_id];
    std::array<uint32_t, maximum_expert_route_ranks> actual_storage;
    std::vector<uint32_t> actual_fallback;
    std::span<const uint32_t> actual_experts;
    if (layer.moe.top_k <= actual_storage.size())
    {
        const std::span<uint32_t> ranked = std::span<uint32_t>(actual_storage).first(layer.moe.top_k);
        collect_ranked_experts(plan, ranked);
        actual_experts = ranked;
    }
    else
    {
        actual_fallback.resize(layer.moe.top_k);
        collect_ranked_experts(plan, actual_fallback);
        actual_experts = actual_fallback;
    }

    for (uint32_t rank = 0; rank < actual_experts.size() && rank < cache.predicted_expert_ids.size(); ++rank)
    {
        const uint32_t predicted = cache.predicted_expert_ids[rank];
        if (predicted == invalid_expert)
            continue;
        if (std::find(actual_experts.begin(), actual_experts.end(), predicted)
            != actual_experts.end())
        {
            ++statistics.expert_route_prediction_matches;
            ++statistics.expert_route_rank_matches[rank];
        }
    }

    std::array<std::string_view, maximum_expert_route_ranks> demanded_keys;
    size_t demanded_key_count = 0;
    for (uint32_t expert_id : actual_experts)
    {
        if (expert_id == invalid_expert
            || expert_id >= layer.moe.experts.size()
            || demanded_key_count == demanded_keys.size())
        {
            continue;
        }
        demanded_keys[demanded_key_count++] = layer.moe.experts[expert_id].cache_key;
    }
    if (resolve_unused_predictions && model.opt.num_concurrent_sessions == 1)
    {
        model.expert_cache->resolve_predictions(
            layer.layer_id,
            std::span<const std::string_view>(demanded_keys.data(), demanded_key_count));
    }

    cache.predicted_expert_ids.clear();
}

static const CompiledLayerPlan* prepare_next_router_prediction(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    const CpuBatch& router_input,
    CpuSessionState& state,
    SessionStatistics& statistics)
{
    if (!model.expert_cache
        || model.opt.num_concurrent_sessions != 1
        || router_input.rows() != 1
        || layer.layer_id >= state.layers.size()
        || layer.moe.router_weight == invalid_tensor_handle
        || layer.moe.token_experts != invalid_tensor_handle
        || !has_flag(model.opt.flags, OptionRouterPrediction))
    {
        return nullptr;
    }

    const CompiledLayerPlan* next_layer = nullptr;
    for (size_t layer_index = static_cast<size_t>(layer.layer_id) + 1;
         layer_index < model.graph.layer_plans.size();
         ++layer_index)
    {
        const CompiledLayerPlan& candidate = model.graph.layer_plans[layer_index];
        if (candidate.moe.router_weight != invalid_tensor_handle
            && candidate.moe.token_experts == invalid_tensor_handle
            && candidate.moe.top_k != 0
            && !candidate.moe.experts.empty())
        {
            next_layer = &candidate;
            break;
        }
    }
    if (!next_layer || next_layer->layer_id >= state.layers.size())
        return nullptr;

    CpuLayerCache& source_cache = state.layers[layer.layer_id];
    configure_router_prefetch(
        source_cache,
        next_layer->moe.top_k);
    adapt_router_prefetch_width(
        source_cache,
        statistics,
        has_flag(model.opt.flags, OptionRankAdaptivePrefetch));
    return next_layer;
}

static Result<RouterPredictionOutcome> run_router_prediction(
    const CompiledModel& model,
    const CompiledLayerPlan& next_layer,
    const CpuBatch& router_input,
    uint32_t prefetch_width)
{
    const auto started = std::chrono::steady_clock::now();
    CpuBatch predicted_logits;
    linear_batch_into(
        model.weights.at(next_layer.moe.router_weight),
        router_input,
        predicted_logits,
        model.opt.optimization_flags,
        model.operators.find_weight(next_layer.moe.router_weight));
    if (next_layer.moe.router_bias != invalid_tensor_handle)
    {
        add_bias_inplace(
            predicted_logits,
            model.weights.at(next_layer.moe.router_bias));
    }

    ExpertDispatchOptions options;
    options.expert_count = static_cast<uint32_t>(next_layer.moe.experts.size());
    options.top_k = next_layer.moe.top_k;
    options.score_function = next_layer.moe.score_function;
    options.normalization = next_layer.moe.normalization;
    options.routed_scaling_factor = next_layer.moe.routed_scaling_factor;
    if (next_layer.moe.router_selection_bias != invalid_tensor_handle)
    {
        options.selection_bias = model.weights.at(next_layer.moe.router_selection_bias).float32_values();
    }
    ExpertDispatchPlan predicted_plan;
    auto dispatched = dispatch_experts_into(
        predicted_logits.values(),
        1,
        options,
        predicted_plan);
    if (!dispatched)
        return dispatched.error();

    const uint32_t invalid_expert = std::numeric_limits<uint32_t>::max();
    std::array<uint32_t, maximum_expert_route_ranks> predicted_storage;
    std::vector<uint32_t> predicted_fallback;
    std::span<const uint32_t> predicted_experts;
    if (next_layer.moe.top_k <= predicted_storage.size())
    {
        const std::span<uint32_t> ranked = std::span<uint32_t>(predicted_storage).first(next_layer.moe.top_k);
        collect_ranked_experts(predicted_plan, ranked);
        predicted_experts = ranked;
    }
    else
    {
        predicted_fallback.resize(next_layer.moe.top_k);
        collect_ranked_experts(predicted_plan, predicted_fallback);
        predicted_experts = predicted_fallback;
    }
    RouterPredictionOutcome outcome;
    outcome.predicted_expert_ids.assign(
        next_layer.moe.top_k,
        invalid_expert);
    const uint32_t width = std::min(prefetch_width, next_layer.moe.top_k);
    for (uint32_t rank = 0; rank < width; ++rank)
    {
        const uint32_t expert_id = predicted_experts[rank];
        if (expert_id == invalid_expert
            || expert_id >= next_layer.moe.experts.size())
            continue;
        const ExpertPlan& predicted = next_layer.moe.experts[expert_id];
        if (predicted.gate_up_weight == invalid_tensor_handle)
            continue;
        outcome.predicted_expert_ids[rank] = expert_id;
        const TensorData& gate_up = model.weights.at(predicted.gate_up_weight);
        const TensorData& down = model.weights.at(predicted.down_weight);
        if (predicted.cache_key.empty())
            continue;
        const auto prediction = model.expert_cache->prefetch_pair(
            gate_up,
            down,
            next_layer.layer_id,
            predicted.cache_key);
        if (prediction && prediction.value())
            ++outcome.cache_hits;
        else
            ++outcome.cache_misses;
    }
    outcome.predictor_time_microseconds = elapsed_microseconds(started);
    return outcome;
}

static Result<void> apply_router_prediction(
    uint32_t target_layer_id,
    RouterPredictionOutcome outcome,
    CpuSessionState& state,
    SessionStatistics& statistics)
{
    if (target_layer_id >= state.layers.size())
    {
        return Error{
            ErrorCode::InternalError,
            "Router prediction target layer is outside Session state"};
    }
    const uint32_t invalid_expert = std::numeric_limits<uint32_t>::max();
    for (uint32_t rank = 0;
         rank < outcome.predicted_expert_ids.size()
         && rank < maximum_expert_route_ranks;
         ++rank)
    {
        if (outcome.predicted_expert_ids[rank] == invalid_expert)
            continue;
        ++statistics.expert_route_predictions;
        ++statistics.expert_route_rank_predictions[rank];
    }
    state.layers[target_layer_id].predicted_expert_ids = std::move(outcome.predicted_expert_ids);
    statistics.expert_route_prediction_cache_hits += outcome.cache_hits;
    statistics.expert_route_prediction_cache_misses += outcome.cache_misses;
    statistics.expert_route_prediction_time_microseconds += outcome.predictor_time_microseconds;
    return {};
}

static void admit_ready_router_prediction(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    CpuSessionState& state,
    size_t token_count)
{
    if (!model.expert_cache
        || !model.expert_backend
        || layer.layer_id >= state.layers.size())
    {
        return;
    }

    CpuLayerCache& cache = state.layers[layer.layer_id];
    if (cache.predicted_expert_ids.empty())
        return;

    std::array<uint32_t, maximum_expert_route_ranks> admitted_ids{};
    size_t admitted_count = 0;
    for (const uint32_t expert_id : cache.predicted_expert_ids)
    {
        if (expert_id == std::numeric_limits<uint32_t>::max()
            || expert_id >= layer.moe.experts.size()
            || admitted_count == admitted_ids.size()
            || std::find(
                   admitted_ids.begin(),
                   admitted_ids.begin() + admitted_count,
                   expert_id)
                   != admitted_ids.begin() + admitted_count)
        {
            continue;
        }

        const ExpertPlan& expert = layer.moe.experts[expert_id];
        if (expert.gate_up_weight == invalid_tensor_handle
            || expert.down_weight == invalid_tensor_handle)
        {
            continue;
        }
        const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
        const TensorData& down = model.weights.at(expert.down_weight);
        if (!gate_up.mxfp4_file_storage
            || !down.mxfp4_file_storage
            || !can_run_vulkan_expert(expert, gate_up, down, model.opt.optimization_flags))
        {
            continue;
        }

        const ExpertCachePairRequest request{
            &gate_up,
            &down,
            layer.layer_id,
            expert.cache_key,
            victim_metadata(model, expert, 1)};
        ExpertCacheLease lease;
        auto ready = model.expert_cache->try_acquire_ready_pairs(
            std::span<const ExpertCachePairRequest>(&request, 1),
            std::span<ExpertCacheLease>(&lease, 1));
        if (!ready || !ready.value())
            continue;

        if (token_count >= vulkan_expert_gpu_admission_min_rows
            || model.opt.hybrid_mode == HybridMode::HybridExperts)
        {
            model.expert_backend->admit(
                expert.cache_key,
                std::move(lease.gate_up),
                expert.gate_up_bias == invalid_tensor_handle
                    ? nullptr
                    : &model.weights.at(expert.gate_up_bias),
                std::move(lease.down),
                expert.down_bias == invalid_tensor_handle
                    ? nullptr
                    : &model.weights.at(expert.down_bias),
                layer.layer_id,
                expert.activation_limit,
                expert.activation);
        }
        admitted_ids[admitted_count++] = expert_id;
    }
}

static Result<void> complete_router_prediction(
    const CompiledModel& model,
    uint32_t layer_id,
    PendingRouterPrediction& pending,
    CpuSessionState& state,
    SessionStatistics& statistics,
    size_t token_count)
{
    if (!pending.result.valid())
        return {};
    if (pending.target_layer_id != layer_id)
    {
        if (pending.target_layer_id > layer_id)
            return {};
        return Error{
            ErrorCode::InternalError,
            "Router prediction completed after its target layer"};
    }

    const auto wait_started = std::chrono::steady_clock::now();
    Result<RouterPredictionOutcome> completed = Error{
        ErrorCode::InternalError,
        "Router prediction worker did not return a result"};
    try
    {
        completed = pending.result.get();
    }
    catch (const std::exception& error)
    {
        completed = Error{
            ErrorCode::InternalError,
            std::string("Router prediction worker failed: ") + error.what()};
    }
    catch (...)
    {
        completed = Error{
            ErrorCode::InternalError,
            "Router prediction worker failed"};
    }
    statistics.expert_route_prediction_wait_time_microseconds += elapsed_microseconds(wait_started);
    const uint32_t target_layer_id = pending.target_layer_id;
    pending.target_layer_id = std::numeric_limits<uint32_t>::max();
    if (!completed)
        return completed.error();
    ++statistics.expert_route_prediction_async_completions;
    auto applied = apply_router_prediction(
        target_layer_id,
        std::move(completed).value(),
        state,
        statistics);
    if (!applied)
        return applied.error();

    if (layer_id < model.graph.layer_plans.size())
    {
        // Admit predicted weights to the GPU queue after the host pair is ready.
        admit_ready_router_prediction(
            model,
            model.graph.layer_plans[layer_id],
            state,
            token_count);
    }
    return {};
}

static Result<void> predict_next_router_routes(
    const CompiledModel& model,
    const CompiledLayerPlan& layer,
    const CpuBatch& router_input,
    CpuSessionState& state,
    SessionStatistics& statistics,
    PendingRouterPrediction& pending)
{
    const CompiledLayerPlan* next_layer = prepare_next_router_prediction(
        model,
        layer,
        router_input,
        state,
        statistics);
    if (!next_layer)
        return {};
    const CpuLayerCache& source_cache = state.layers[layer.layer_id];
    const uint32_t prefetch_width = std::min(
        source_cache.router_prefetch_width,
        next_layer->moe.top_k);

    if (has_flag(
            model.opt.flags,
            OptionAsyncRouterPrediction))
    {
        try
        {
            if (!state.router_prediction_worker)
            {
                state.router_prediction_worker = std::make_unique<CpuTaskWorker>(2);
            }
            CpuBatch copied_input = router_input;
            auto promise = std::make_shared<
                std::promise<Result<RouterPredictionOutcome>>>();
            std::future<Result<RouterPredictionOutcome>> future = promise->get_future();
            Bfloat16BatchedLinearExecutionCounter* const bfloat16_counter = current_bfloat16_batched_linear_execution_counter();
            if (state.router_prediction_worker->try_submit(
                    [&model,
                     next_layer,
                     copied_input = std::move(copied_input),
                     prefetch_width,
                     bfloat16_counter,
                     promise]() mutable {
                        const ScopedBfloat16BatchedLinearExecutionCounter
                            bfloat16_scope(bfloat16_counter);
                        try
                        {
                            promise->set_value(run_router_prediction(
                                model,
                                *next_layer,
                                copied_input,
                                prefetch_width));
                        }
                        catch (const std::exception& error)
                        {
                            promise->set_value(Error{
                                ErrorCode::InternalError,
                                std::string("Router prediction failed: ")
                                    + error.what()});
                        }
                        catch (...)
                        {
                            promise->set_value(Error{
                                ErrorCode::InternalError,
                                "Router prediction failed"});
                        }
                    }))
            {
                pending.target_layer_id = next_layer->layer_id;
                pending.result = std::move(future);
                ++statistics.expert_route_prediction_async_submissions;
                return {};
            }
        }
        catch (...)
        {
            // Fall back to synchronous prediction.
        }
        ++statistics.expert_route_prediction_async_fallbacks;
    }

    auto outcome = run_router_prediction(
        model,
        *next_layer,
        router_input,
        prefetch_width);
    if (!outcome)
        return outcome.error();
    auto applied = apply_router_prediction(
        next_layer->layer_id,
        std::move(outcome).value(),
        state,
        statistics);
    if (!applied)
        return applied.error();
    admit_ready_router_prediction(model, *next_layer, state, router_input.rows());
    return {};
}

} // namespace moe
} // namespace ncnn
