#include "executor.h"

#include "sessionstate.h"
#include "metrics.h"
#include "expertbackend.h"
#include "expert.h"
#include "graph/router.h"
#include "graph/compiledmodel.h"
#include "kernels/attention.h"
#include "kernels/bfloat16.h"
#include "kernels/hyperconnection.h"
#include "kernels/latentattention.h"
#include "kernels/ops.h"
#include "storage/expertcache.h"
#include "backends/ncnn/vulkan.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace ncnn {
namespace moe {

static uint64_t elapsed_microseconds(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now())
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

static Result<void> execute_speculative_layer(
    const CompiledModel& model,
    const SpeculativeModelPlan::LayerNodes& execution,
    size_t layer_plan_index,
    uint64_t position_offset,
    LayerCache& cache,
    ActivationBuffer& hidden,
    SessionStatistics& statistics,
    LayerGraphState& layer_state,
    ExpertScratch& scratch,
    HyperConnectionScratch& hyper_connection_scratch,
    AttentionScratch& attention_scratch)
{
    const ExecutionGraph& graph = model.speculative.graph;
    if (layer_plan_index >= graph.layer_plans.size()
        || execution.attention >= graph.nodes.size()
        || execution.router >= graph.nodes.size()
        || execution.expert_dispatch >= graph.nodes.size()
        || execution.expert_group >= graph.nodes.size()
        || execution.combine >= graph.nodes.size())
    {
        return Error{
            ErrorCode::InternalError,
            "speculative execution graph has an incomplete layer binding"};
    }
    const CompiledLayerPlan& layer = graph.layer_plans[layer_plan_index];
    const ExecutionNode& attention_node = graph.nodes[execution.attention];
    const ExecutionNode& expert_group_node = graph.nodes[execution.expert_group];
    const ExecutionBackend attention_backend = attention_node.backend;
    const ExecutionBackend expert_backend = expert_group_node.backend;
    const bool cpu_prefetch = has_flag(expert_group_node.flags, ExecutionNodeCpuPrefetch);

    // Target and draft layers execute sequentially with the same FFN workspace.
    layer_state.reset();
    const uint32_t multiplier = model.descriptor.hyper_connection_multiplier;
    const auto attention_start = std::chrono::steady_clock::now();
    HyperConnectionMix& attention_mix = hyper_connection_scratch.transient_mix;
    const ActivationBuffer* attention_input = &hidden;
    if (multiplier > 1)
    {
        auto mixed = hyper_connection_pre(
            hidden,
            model.weights.at(layer.hyper_connection.attention_function),
            model.weights.at(layer.hyper_connection.attention_scale),
            model.weights.at(layer.hyper_connection.attention_base),
            multiplier,
            model.descriptor.hyper_connection_iterations,
            model.descriptor.norm_epsilon,
            model.descriptor.hyper_connection_epsilon,
            attention_mix,
            hyper_connection_scratch,
            model.opt.optimization_flags);
        if (!mixed)
            return mixed.error();
        attention_input = &attention_mix.reduced;
    }
    if (model.speculative.kind == SpeculativeModelKind::Mtp)
    {
        auto attention = forward_attention(
            model.weights,
            model.operators,
            layer.attention,
            attention_backend,
            model.descriptor.norm_epsilon,
            model.descriptor.kv_cache_dtype,
            position_offset,
            cache,
            attention_scratch,
            *attention_input,
            attention_scratch.output,
            model.opt.optimization_flags);
        if (!attention)
            return attention.error();
        hidden.swap(attention_scratch.output);
    }
    else
    {
        auto attention = forward_dspark_attention(
            model.weights,
            model.operators,
            layer.attention,
            attention_backend,
            model.descriptor.norm_epsilon,
            position_offset,
            cache,
            attention_scratch,
            *attention_input,
            attention_scratch.output,
            model.opt.optimization_flags);
        if (!attention)
            return attention.error();
        if (multiplier > 1)
        {
            auto connected = hyper_connection_post(
                attention_scratch.output,
                hidden,
                attention_mix,
                multiplier,
                scratch.staged_output);
            if (!connected)
                return connected.error();
            hidden.swap(scratch.staged_output);
        }
        else
        {
            add_batch_inplace(hidden, attention_scratch.output);
        }
    }
    statistics.attention_time_microseconds += elapsed_microseconds(attention_start);

    const MoeBlockPlan& moe = layer.moe;
    layer_state.router_start = std::chrono::steady_clock::now();
    if (multiplier > 1)
    {
        auto mixed = hyper_connection_pre(
            hidden,
            model.weights.at(layer.hyper_connection.ffn_function),
            model.weights.at(layer.hyper_connection.ffn_scale),
            model.weights.at(layer.hyper_connection.ffn_base),
            multiplier,
            model.descriptor.hyper_connection_iterations,
            model.descriptor.norm_epsilon,
            model.descriptor.hyper_connection_epsilon,
            layer_state.ffn_hyper_mix,
            hyper_connection_scratch,
            model.opt.optimization_flags);
        if (!mixed)
            return mixed.error();
        rms_norm_batch_into(layer_state.ffn_hyper_mix.reduced, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset, model.opt.optimization_flags);
    }
    else
    {
        rms_norm_batch_into(hidden, model.weights.at(moe.pre_ffn_norm_weight), model.descriptor.norm_epsilon, layer_state.normalized, model.descriptor.norm_weight_offset, model.opt.optimization_flags);
    }
    linear_batch_into(
        model.weights.at(moe.router_weight),
        layer_state.normalized,
        layer_state.router_logits,
        model.opt.optimization_flags,
        model.operators.find_weight(moe.router_weight));
    ExpertDispatchOptions dispatch_options;
    dispatch_options.expert_count = static_cast<uint32_t>(moe.experts.size());
    dispatch_options.top_k = moe.top_k;
    dispatch_options.score_function = moe.score_function;
    dispatch_options.normalization = moe.normalization;
    dispatch_options.routed_scaling_factor = moe.routed_scaling_factor;
    if (moe.router_selection_bias != invalid_tensor_handle)
    {
        dispatch_options.selection_bias = model.weights.at(moe.router_selection_bias).float32_values();
    }
    auto dispatched = dispatch_experts_into(layer_state.router_logits.values(), static_cast<uint32_t>(layer_state.router_logits.rows()), dispatch_options, layer_state.dispatch_plan);
    if (!dispatched)
        return dispatched.error();
    statistics.expert_assignments += layer_state.dispatch_plan.assignment_count;
    layer_state.resize_experts(layer_state.dispatch_plan.batches.size());
    for (size_t batch_index = 0; batch_index < layer_state.dispatch_plan.batches.size(); ++batch_index)
    {
        ExpertBatch& batch = layer_state.dispatch_plan.batches[batch_index];
        statistics.expert_token_counts[batch.expert_id] += batch.routes.size();
        record_expert_weight_demand(moe.experts[batch.expert_id], batch.routes.size(), statistics);
        layer_state.active_experts()[batch_index].prepare(batch);
    }
    statistics.router_time_microseconds += elapsed_microseconds(layer_state.router_start);
    layer_state.expert_start = std::chrono::steady_clock::now();

    const auto expert_engine_start = std::chrono::steady_clock::now();
    auto executed = forward_moe(
        model,
        moe,
        layer_state,
        statistics,
        scratch,
        layer.layer_id,
        expert_backend,
        cpu_prefetch);
    statistics.expert_engine_time_microseconds += elapsed_microseconds(expert_engine_start);
    if (!executed)
        return executed.error();
    const auto combine_start = std::chrono::steady_clock::now();
    if (moe.has_shared_expert && layer_state.shared_expert_output.rows() == 0)
    {
        ExpertExecutionMetrics shared_metrics;
        forward_shared_expert(
            model,
            moe,
            layer_state.normalized,
            layer_state.shared_expert_output,
            shared_metrics,
            model.opt.optimization_flags);
    }
    ActivationBuffer& moe_output = layer_state.normalized;
    const bool has_backend_aggregation = initialize_backend_aggregated_output(
        scratch,
        hidden.rows(),
        model.descriptor.hidden_size,
        moe_output);
    for (size_t active_index = 0; active_index < layer_state.active_experts().size(); ++active_index)
    {
        const ActiveExpertExecution& active = layer_state.active_experts()[active_index];
        if (has_backend_aggregation
            && active_index < scratch.backend_aggregated.size()
            && scratch.backend_aggregated[active_index] != 0)
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
        add_batch_inplace(moe_output, layer_state.shared_expert_output);
    if (multiplier > 1)
    {
        auto connected = hyper_connection_post(
            moe_output,
            hidden,
            layer_state.ffn_hyper_mix,
            multiplier,
            scratch.staged_output);
        if (!connected)
            return connected.error();
        hidden.swap(scratch.staged_output);
    }
    else
    {
        add_batch_inplace(hidden, moe_output);
    }
    const auto combine_end = std::chrono::steady_clock::now();
    statistics.expert_combine_time_microseconds += elapsed_microseconds(combine_start, combine_end);
    statistics.expert_time_microseconds += elapsed_microseconds(layer_state.expert_start, combine_end);
    return {};
}

static Result<ActivationBuffer> prepare_mtp_hidden(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    const ActivationBuffer& target_hidden)
{
    if (input_ids.empty()
        || input_ids.size() != target_hidden.rows()
        || target_hidden.columns() != model.descriptor.hidden_size)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid Qwen MTP input batch"};
    }

    ActivationBuffer embeddings;
    embedding_batch_into(
        model.weights.at(model.token_embedding),
        input_ids,
        embeddings);
    rms_norm_batch_into(
        embeddings,
        model.weights.at(
            model.speculative.mtp_embedding_norm_weight),
        model.descriptor.norm_epsilon,
        embeddings,
        model.descriptor.norm_weight_offset,
        model.opt.optimization_flags);
    ActivationBuffer normalized_hidden = rms_norm_batch(
        target_hidden,
        model.weights.at(model.speculative.mtp_hidden_norm_weight),
        model.descriptor.norm_epsilon,
        model.descriptor.norm_weight_offset,
        model.opt.optimization_flags);
    ActivationBuffer packed(
        target_hidden.rows(),
        model.descriptor.hidden_size * 2);
    for (size_t row = 0; row < target_hidden.rows(); ++row)
    {
        std::copy_n(
            embeddings.row(row),
            model.descriptor.hidden_size,
            packed.row(row));
        std::copy_n(
            normalized_hidden.row(row),
            model.descriptor.hidden_size,
            packed.row(row) + model.descriptor.hidden_size);
    }
    return linear_batch(
        model.weights.at(
            model.speculative.mtp_input_projection_weight),
        packed,
        model.opt.optimization_flags,
        model.operators.find_weight(model.speculative.mtp_input_projection_weight));
}

static Result<ActivationBuffer> execute_mtp_batch(
    const CompiledModel& model,
    const SpeculativeModelPlan::LayerNodes& execution,
    std::span<const int32_t> input_ids, const ActivationBuffer& target_hidden,
    uint64_t position_offset, SessionStatistics& statistics,
    SessionState& state)
{
    if (state.speculative_layers.size() != 1)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid Qwen MTP execution state"};
    }
    auto prepared = prepare_mtp_hidden(
        model,
        input_ids,
        target_hidden);
    if (!prepared)
        return prepared.error();
    ActivationBuffer hidden = std::move(prepared).value();
    auto executed = execute_speculative_layer(
        model,
        execution,
        0,
        position_offset,
        state.speculative_layers.front(),
        hidden,
        statistics,
        state.execution_state,
        state.expert_scratch,
        state.hyper_connection_scratch,
        state.attention_scratch);
    state.execution_state.reset();
    if (!executed)
        return executed.error();
    rms_norm_batch_into(
        hidden,
        model.weights.at(model.speculative.final_norm_weight),
        model.descriptor.norm_epsilon,
        hidden,
        model.descriptor.norm_weight_offset,
        model.opt.optimization_flags);
    return hidden;
}

static Result<void> append_mtp_context(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    const ActivationBuffer& target_hidden,
    uint64_t position_offset,
    SessionState& state)
{
    const std::vector<SpeculativeModelPlan::LayerNodes>& layer_nodes = model.speculative.layer_nodes;
    const ExecutionGraph& graph = model.speculative.graph;
    if (layer_nodes.size() != graph.layer_plans.size())
    {
        return Error{
            ErrorCode::InternalError,
            "speculative execution graph has an incomplete layer binding"};
    }
    if (layer_nodes.size() != 1
        || state.speculative_layers.size() != 1)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid Qwen MTP context state"};
    }
    auto prepared = prepare_mtp_hidden(
        model,
        input_ids,
        target_hidden);
    if (!prepared)
        return prepared.error();
    const SpeculativeModelPlan::LayerNodes& execution = layer_nodes.front();
    if (execution.attention >= graph.nodes.size())
    {
        return Error{
            ErrorCode::InternalError,
            "speculative execution graph has an incomplete layer binding"};
    }
    const CompiledLayerPlan& layer = graph.layer_plans.front();
    auto appended = append_attention_context(
        model.weights,
        model.operators,
        layer.attention,
        graph.nodes[execution.attention].backend,
        model.descriptor.norm_epsilon,
        model.descriptor.kv_cache_dtype,
        position_offset,
        state.speculative_layers.front(),
        state.attention_scratch,
        prepared.value(),
        model.opt.optimization_flags);
    return appended;
}

static Result<void> update_mtp_context(
    const CompiledModel& model,
    SessionState& state)
{
    const ActivationBuffer& target_hidden = state.speculative_main_hidden;
    if (target_hidden.rows() == 0
        || target_hidden.columns() != model.descriptor.hidden_size)
    {
        return Error{
            ErrorCode::InternalError,
            "target execution did not capture Qwen final hidden states"};
    }

    const bool direct_alignment = !state.speculative_direct_alignment_ids.empty();
    const std::vector<int32_t>& input_ids = direct_alignment
                                                ? state.speculative_direct_alignment_ids
                                                : state.speculative_input_ids;
    if (input_ids.size() != target_hidden.rows())
    {
        return Error{
            ErrorCode::InternalError,
            "Qwen MTP alignment IDs do not match target hidden states"};
    }
    if (direct_alignment
        && state.mtp_pending_target_hidden.rows() != 0)
    {
        return Error{
            ErrorCode::InternalError,
            "Qwen MTP direct alignment still has a pending target row"};
    }

    const bool has_pending = state.mtp_pending_target_hidden.rows() != 0;
    if (has_pending
        && (state.mtp_pending_target_hidden.rows() != 1
            || state.mtp_pending_target_hidden.columns()
                   != model.descriptor.hidden_size
            || state.mtp_pending_target_position + 1
                   != state.speculative_main_hidden_position))
    {
        return Error{
            ErrorCode::InternalError,
            "Qwen MTP pending target row is out of sequence"};
    }

    const size_t aligned_rows = direct_alignment
                                    ? target_hidden.rows() - 1
                                    : (has_pending ? 1 : 0) + target_hidden.rows() - 1;
    if (aligned_rows != 0)
    {
        ActivationBuffer aligned_hidden(
            aligned_rows,
            model.descriptor.hidden_size);
        std::vector<int32_t> aligned_ids;
        aligned_ids.reserve(aligned_rows);
        size_t output_row = 0;
        uint64_t aligned_position = state.speculative_main_hidden_position;
        if (!direct_alignment && has_pending)
        {
            aligned_position = state.mtp_pending_target_position;
            std::copy_n(
                state.mtp_pending_target_hidden.row(0),
                model.descriptor.hidden_size,
                aligned_hidden.row(output_row++));
            aligned_ids.push_back(input_ids.front());
        }
        for (size_t row = 0;
             row + 1 < target_hidden.rows();
             ++row)
        {
            std::copy_n(
                target_hidden.row(row),
                model.descriptor.hidden_size,
                aligned_hidden.row(output_row++));
            aligned_ids.push_back(
                input_ids[row + (direct_alignment ? 0 : 1)]);
        }
        auto aligned = append_mtp_context(
            model,
            aligned_ids,
            aligned_hidden,
            aligned_position,
            state);
        if (!aligned)
            return aligned.error();
    }

    state.mtp_pending_target_hidden.reset(
        1,
        model.descriptor.hidden_size,
        false);
    std::copy_n(
        target_hidden.row(target_hidden.rows() - 1),
        model.descriptor.hidden_size,
        state.mtp_pending_target_hidden.row(0));
    state.mtp_pending_target_position = state.speculative_main_hidden_position
                                        + target_hidden.rows() - 1;
    state.speculative_input_ids.clear();
    state.speculative_direct_alignment_ids.clear();
    return {};
}

Result<void> update_speculative_context(const CompiledModel& model, SessionStatistics& statistics, SessionState& state)
{
    if (!model.speculative.enabled()
        || !state.use_speculative_context)
        return {};
    Bfloat16BatchedLinearExecutionCounter cpu_bfloat16_execution;
    const ScopedBfloat16BatchedLinearExecutionCounter cpu_bfloat16_scope(
        &cpu_bfloat16_execution);
    const auto started = std::chrono::steady_clock::now();
    const VulkanStatistics vulkan_before = get_vulkan_statistics(model.vulkan_runtime);
    if (model.speculative.kind == SpeculativeModelKind::Mtp)
    {
        auto updated = update_mtp_context(
            model,
            state);
        if (!updated)
            return updated.error();
        const VulkanStatistics vulkan_after = get_vulkan_statistics(model.vulkan_runtime);
        record_vulkan_execution_delta(statistics, vulkan_before, vulkan_after);
        statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
        statistics.speculative_context_time_microseconds += elapsed_microseconds(started);
        return {};
    }
    const uint32_t expected_columns = model.descriptor.hidden_size
                                      * static_cast<uint32_t>(model.speculative.target_layer_ids.size());
    if (state.speculative_main_hidden.rows() == 0
        || state.speculative_main_hidden.columns() != expected_columns)
    {
        return Error{
            ErrorCode::InternalError,
            "target execution did not capture DSpark context features"};
    }
    ActivationBuffer projected = linear_batch(
        model.weights.at(model.speculative.main_projection_weight),
        state.speculative_main_hidden,
        model.opt.optimization_flags,
        model.operators.find_weight(model.speculative.main_projection_weight));
    rms_norm_batch_into(
        projected,
        model.weights.at(model.speculative.main_norm_weight),
        model.descriptor.norm_epsilon,
        projected,
        model.descriptor.norm_weight_offset,
        model.opt.optimization_flags);
    const std::vector<SpeculativeModelPlan::LayerNodes>& layer_nodes = model.speculative.layer_nodes;
    const ExecutionGraph& graph = model.speculative.graph;
    if (layer_nodes.size() != graph.layer_plans.size())
    {
        return Error{
            ErrorCode::InternalError,
            "speculative execution graph has an incomplete layer binding"};
    }
    if (state.speculative_layers.size() != layer_nodes.size())
        state.speculative_layers.resize(layer_nodes.size());
    for (size_t layer_index = 0; layer_index < layer_nodes.size(); ++layer_index)
    {
        const SpeculativeModelPlan::LayerNodes& execution = layer_nodes[layer_index];
        if (execution.attention >= graph.nodes.size())
        {
            return Error{
                ErrorCode::InternalError,
                "speculative execution graph has an incomplete layer binding"};
        }
        const CompiledLayerPlan& layer = graph.layer_plans[layer_index];
        auto appended = append_dspark_attention_context(
            model.weights,
            model.operators,
            layer.attention,
            graph.nodes[execution.attention].backend,
            model.descriptor.norm_epsilon,
            state.speculative_main_hidden_position, state.speculative_layers[layer_index], projected,
            model.opt.optimization_flags);
        if (!appended)
            return appended.error();
    }
    const VulkanStatistics vulkan_after = get_vulkan_statistics(model.vulkan_runtime);
    record_vulkan_execution_delta(statistics, vulkan_before, vulkan_after);
    statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
    statistics.speculative_context_time_microseconds += elapsed_microseconds(started);
    return {};
}

static Result<SpeculativeProposal> propose_mtp(
    const CompiledModel& model,
    int32_t input_id,
    SessionStatistics& statistics,
    SessionState& state,
    uint64_t position_offset,
    const SpeculativeSampler& sampler)
{
    if (!sampler)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "Qwen MTP proposal requires a token sampler"};
    }
    if (input_id < 0
        || static_cast<uint32_t>(input_id)
               >= model.descriptor.vocabulary_size
        || state.speculative_layers.size() != 1
        || state.mtp_pending_target_hidden.rows() != 1
        || state.mtp_pending_target_hidden.columns()
               != model.descriptor.hidden_size
        || position_offset == 0
        || state.mtp_pending_target_position + 1
               != position_offset)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid Qwen MTP proposal state"};
    }

    const auto started = std::chrono::steady_clock::now();
    const VulkanStatistics vulkan_before = get_vulkan_statistics(model.vulkan_runtime);
    ExpertCacheStatistics cache_before;
    if (model.expert_cache)
        cache_before = model.expert_cache->statistics();
    ExpertBackendStatistics backend_before;
    if (model.expert_backend)
        backend_before = model.expert_backend->statistics();
    const std::vector<SpeculativeModelPlan::LayerNodes>& layer_nodes = model.speculative.layer_nodes;
    if (layer_nodes.size() != model.speculative.graph.layer_plans.size())
    {
        return Error{
            ErrorCode::InternalError,
            "speculative execution graph has an incomplete layer binding"};
    }
    if (layer_nodes.size() != 1)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid Qwen MTP execution state"};
    }
    const SpeculativeModelPlan::LayerNodes& execution = layer_nodes.front();

    SpeculativeProposal proposal;
    proposal.token_ids.reserve(model.speculative.block_size);
    proposal.logits.reserve(model.speculative.block_size);
    proposal.confidence_logits.reserve(
        model.speculative.block_size);
    proposal.committed_context_rows = 1;
    ActivationBuffer previous_hidden = state.mtp_pending_target_hidden;
    int32_t previous_token = input_id;
    for (uint32_t row = 0;
         row < model.speculative.block_size;
         ++row)
    {
        const std::span<const int32_t> token(
            &previous_token,
            1);
        auto mtp_hidden = execute_mtp_batch(
            model, execution,
            token, previous_hidden,
            state.mtp_pending_target_position + row, statistics, state);
        if (!mtp_hidden)
            return mtp_hidden.error();
        ActivationBuffer logits = linear_batch(
            model.weights.at(model.lm_head_weight),
            mtp_hidden.value(),
            model.opt.optimization_flags,
            model.operators.find_weight(model.lm_head_weight));
        std::vector<float> row_logits(
            logits.row(0),
            logits.row(0) + logits.columns());
        auto sampled = sampler(row_logits);
        if (!sampled)
            return sampled.error();
        previous_token = sampled.value();
        proposal.token_ids.push_back(previous_token);
        proposal.logits.push_back(std::move(row_logits));
        proposal.confidence_logits.push_back(
            std::numeric_limits<float>::infinity());
        previous_hidden = std::move(mtp_hidden).value();
    }
    state.mtp_pending_target_hidden.clear();

    if (model.expert_cache)
    {
        const ExpertCacheStatistics cache_after = model.expert_cache->statistics();
        record_expert_cache_delta(statistics, cache_before, cache_after);
    }
    if (model.expert_backend)
    {
        record_expert_backend_delta(
            statistics,
            backend_before,
            model.expert_backend->statistics());
    }
    const VulkanStatistics vulkan_after = get_vulkan_statistics(model.vulkan_runtime);
    record_vulkan_execution_delta(statistics, vulkan_before, vulkan_after);
    ++statistics.speculative_proposals;
    statistics.speculative_draft_tokens += proposal.token_ids.size();
    statistics.speculative_draft_time_microseconds += elapsed_microseconds(started);
    return proposal;
}

Result<SpeculativeProposal> propose_speculative(const CompiledModel& model, int32_t input_id, SessionStatistics& statistics, SessionState& state, uint64_t position_offset,
                                                const SpeculativeSampler& sampler)
{
    if (!model.speculative.enabled())
    {
        return Error{
            ErrorCode::UnsupportedModel,
            "the model does not provide a speculative execution plan"};
    }
    Bfloat16BatchedLinearExecutionCounter cpu_bfloat16_execution;
    const ScopedBfloat16BatchedLinearExecutionCounter cpu_bfloat16_scope(
        &cpu_bfloat16_execution);
    if (model.speculative.kind == SpeculativeModelKind::Mtp)
    {
        auto proposal = propose_mtp(
            model,
            input_id,
            statistics,
            state,
            position_offset,
            sampler);
        if (proposal)
        {
            statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
        }
        return proposal;
    }
    if (!sampler)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "DSpark proposal requires a token sampler"};
    }
    const std::vector<SpeculativeModelPlan::LayerNodes>& layer_nodes = model.speculative.layer_nodes;
    if (layer_nodes.size() != model.speculative.graph.layer_plans.size())
    {
        return Error{
            ErrorCode::InternalError,
            "speculative execution graph has an incomplete layer binding"};
    }
    if (input_id < 0
        || static_cast<uint32_t>(input_id)
               >= model.descriptor.vocabulary_size
        || state.speculative_layers.size()
               != layer_nodes.size())
    {
        return Error{
            ErrorCode::InvalidArgument,
            "invalid DSpark proposal state"};
    }
    for (const LayerCache& cache : state.speculative_layers)
    {
        if (cache.latent_token_count != position_offset)
        {
            return Error{
                ErrorCode::InternalError,
                "DSpark context cache is out of sync with the target model"};
        }
    }

    const auto started = std::chrono::steady_clock::now();
    const VulkanStatistics vulkan_before = get_vulkan_statistics(model.vulkan_runtime);
    ExpertCacheStatistics cache_before;
    if (model.expert_cache)
        cache_before = model.expert_cache->statistics();
    ExpertBackendStatistics backend_before;
    if (model.expert_backend)
        backend_before = model.expert_backend->statistics();

    std::vector<int32_t> draft_input_ids(model.speculative.block_size, static_cast<int32_t>(model.speculative.noise_token_id));
    draft_input_ids.front() = input_id;
    ActivationBuffer hidden;
    embedding_batch_into(model.weights.at(model.token_embedding), draft_input_ids, hidden);
    hyper_connection_expand(hidden, model.descriptor.hyper_connection_multiplier, state.expert_scratch.staged_output);
    for (size_t layer_index = 0; layer_index < layer_nodes.size(); ++layer_index)
    {
        auto executed = execute_speculative_layer(
            model,
            layer_nodes[layer_index],
            layer_index,
            position_offset,
            state.speculative_layers[layer_index],
            hidden,
            statistics,
            state.execution_state,
            state.expert_scratch,
            state.hyper_connection_scratch,
            state.attention_scratch);
        state.execution_state.reset();
        if (!executed)
            return executed.error();
    }

    auto headed = hyper_connection_head(
        hidden,
        model.weights.at(model.speculative.hyper_head_function),
        model.weights.at(model.speculative.hyper_head_scale),
        model.weights.at(model.speculative.hyper_head_base),
        model.descriptor.hyper_connection_multiplier,
        model.descriptor.norm_epsilon,
        model.descriptor.hyper_connection_epsilon,
        state.expert_scratch.staged_output,
        state.hyper_connection_scratch,
        model.opt.optimization_flags);
    if (!headed)
        return headed.error();
    rms_norm_batch_into(
        state.expert_scratch.staged_output,
        model.weights.at(model.speculative.final_norm_weight),
        model.descriptor.norm_epsilon,
        state.final_norm,
        model.descriptor.norm_weight_offset,
        model.opt.optimization_flags);
    linear_batch_into(
        model.weights.at(model.lm_head_weight),
        state.final_norm,
        state.expert_scratch.staged_merged,
        model.opt.optimization_flags,
        model.operators.find_weight(model.lm_head_weight));
    const ActivationBuffer& head_hidden = state.expert_scratch.staged_output;
    const ActivationBuffer& base_logits = state.expert_scratch.staged_merged;

    SpeculativeProposal proposal;
    proposal.token_ids.reserve(model.speculative.block_size);
    proposal.logits.reserve(model.speculative.block_size);
    proposal.confidence_logits.reserve(model.speculative.block_size);
    int32_t previous_token = input_id;
    const TensorData& confidence = model.weights.at(model.speculative.confidence_weight);
    const std::span<const uint16_t> confidence_values = confidence.bfloat16_values();
    const TensorData& markov_embedding_weight = model.weights.at(model.speculative.markov_embedding_weight);
    const TensorData& markov_head_weight = model.weights.at(model.speculative.markov_head_weight);
    const CompiledOperator* markov_head_operator = model.operators.find_weight(model.speculative.markov_head_weight);
    ActivationBuffer markov_embedding;
    ActivationBuffer markov_logits;
    for (uint32_t row = 0; row < model.speculative.block_size; ++row)
    {
        const std::span<const int32_t> previous(&previous_token, 1);
        embedding_batch_into(markov_embedding_weight, previous, markov_embedding);
        linear_batch_into(
            markov_head_weight,
            markov_embedding,
            markov_logits,
            model.opt.optimization_flags,
            markov_head_operator);
        std::vector<float> row_logits(model.descriptor.vocabulary_size);
        for (uint32_t token_id = 0; token_id < model.descriptor.vocabulary_size; ++token_id)
        {
            row_logits[token_id] = base_logits.row(row)[token_id]
                                   + markov_logits.row(0)[token_id];
        }
        auto sampled = sampler(row_logits);
        if (!sampled)
            return sampled.error();
        const int32_t selected = sampled.value();
        proposal.token_ids.push_back(selected);
        proposal.logits.push_back(std::move(row_logits));
        previous_token = selected;

        float confidence_logit = 0.0f;
        for (uint32_t column = 0; column < model.descriptor.hidden_size; ++column)
        {
            confidence_logit += head_hidden.row(row)[column]
                                * bfloat16_to_float(confidence_values[column]);
        }
        for (uint32_t column = 0; column < model.speculative.markov_rank; ++column)
        {
            confidence_logit += markov_embedding.row(0)[column] * bfloat16_to_float(confidence_values[model.descriptor.hidden_size + column]);
        }
        proposal.confidence_logits.push_back(confidence_logit);
    }

    if (model.expert_cache)
    {
        const ExpertCacheStatistics cache_after = model.expert_cache->statistics();
        record_expert_cache_delta(statistics, cache_before, cache_after);
    }
    if (model.expert_backend)
    {
        record_expert_backend_delta(statistics, backend_before, model.expert_backend->statistics());
    }
    const VulkanStatistics vulkan_after = get_vulkan_statistics(model.vulkan_runtime);
    record_vulkan_execution_delta(statistics, vulkan_before, vulkan_after);
    statistics.cpu_bfloat16_batched_linear_dispatches += cpu_bfloat16_execution.dispatch_count();
    ++statistics.speculative_proposals;
    statistics.speculative_draft_tokens += proposal.token_ids.size();
    statistics.speculative_draft_time_microseconds += elapsed_microseconds(started);
    return proposal;
}

} // namespace moe
} // namespace ncnn
