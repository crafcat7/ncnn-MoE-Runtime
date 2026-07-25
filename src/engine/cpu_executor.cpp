#include "cpu_executor.h"

#include "kernels/cpu_attention.h"
#include "kernels/cpu_batch.h"
#include "kernels/cpu_ops.h"
#include "cpu_session_state.h"
#include "storage/expert_cache.h"
#include "backends/ncnn/ncnn_attention.h"
#include "backends/ncnn/ncnn_linear.h"

#include "ncnn/moe/expert_dispatcher.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <immintrin.h>
#endif

namespace ncnn {
namespace moe {

struct ExpertExecutionMetrics
{
    uint64_t hinted_bytes = 0;
    uint64_t mxfp4_decode_gemv_rows = 0;
    uint64_t mxfp4_prefill_gemm_rows = 0;
    uint64_t mxfp4_paired_rows = 0;
    uint64_t mxfp4_fused_gate_up_rows = 0;
};

struct ActiveExpertExecution
{
    ExpertBatch batch;
    CpuBatch output;
    ExpertExecutionMetrics metrics;
    Error error;
    bool failed = false;
};

struct LayerGraphState
{
    CpuBatch normalized;
    CpuBatch router_logits;
    std::vector<ActiveExpertExecution> active_experts;
    std::chrono::steady_clock::time_point router_start;
    std::chrono::steady_clock::time_point expert_start;
    bool experts_executed = false;
};

static constexpr size_t expert_prefetch_limit_bytes = 4 * 1024;
static constexpr size_t assumed_cache_line_bytes = 64;

static uint64_t elapsed_microseconds(std::chrono::steady_clock::time_point start)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count());
}

static void prefetch_address(const void* address)
{
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(address, 0, 3);
#else
    (void)address;
#endif
}

static uint64_t prefetch_buffer(const void* data, size_t size)
{
    if (!data || size == 0)
        return 0;
    const size_t hinted_bytes = std::min(size, expert_prefetch_limit_bytes);
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t offset = 0; offset < hinted_bytes; offset += assumed_cache_line_bytes)
        prefetch_address(bytes + offset);
    return hinted_bytes;
}

static uint64_t prefetch_tensor(const TensorData& tensor)
{
    if (tensor.dtype == DType::Float32) {
        const std::span<const float> values
            = tensor.float32_values();
        return prefetch_buffer(
            values.data(),
            values.size() * sizeof(float));
    }
    if (tensor.dtype == DType::BFloat16) {
        const std::span<const uint16_t> values
            = tensor.bfloat16_values();
        return prefetch_buffer(
            values.data(),
            values.size() * sizeof(uint16_t));
    }
    if (tensor.dtype == DType::Int8) {
        const std::span<const int8_t> values
            = tensor.int8_values();
        return prefetch_buffer(values.data(), values.size());
    }
    if (tensor.dtype == DType::MxFp4) {
        return prefetch_buffer(tensor.mxfp4_blocks.data(), tensor.mxfp4_blocks.size())
               + prefetch_buffer(tensor.mxfp4_scales.data(), tensor.mxfp4_scales.size());
    }
    return 0;
}

static uint64_t prefetch_weight(const WeightTable& weights, TensorHandle handle)
{
    return handle == invalid_tensor_handle ? 0 : prefetch_tensor(weights.at(handle));
}

static float dense_tensor_value(const TensorData& tensor, size_t index)
{
    if (tensor.dtype == DType::Float32)
        return tensor.float32_values()[index];
    if (tensor.dtype == DType::BFloat16)
        return bfloat16_to_float(tensor.bfloat16_values()[index]);
    if (tensor.dtype == DType::Int8) {
        const uint32_t columns = tensor.shape[1];
        return static_cast<float>(tensor.int8_values()[index])
               * tensor.quantization_scales[index / columns];
    }
    return 0.0f;
}

static void predict_next_layer_experts(
    const CompiledModel& model,
    size_t layer_index,
    uint64_t prediction_generation,
    std::vector<float> hidden)
{
    if (!model.expert_cache || layer_index >= model.layers.size())
        return;
    const MoeBlockPlan& moe = model.layers[layer_index].moe;
    if (moe.pre_ffn_norm_weight == invalid_tensor_handle
        || moe.router_weight == invalid_tensor_handle
        || hidden.empty()) {
        return;
    }

    const TensorData& norm_weight = model.weights.at(moe.pre_ffn_norm_weight);
    const TensorData& router_weight = model.weights.at(moe.router_weight);
    if (norm_weight.element_count() < hidden.size()
        || router_weight.shape.size() != 2
        || router_weight.shape[1] != hidden.size()) {
        return;
    }

    float square_sum = 0.0f;
    for (float value : hidden)
        square_sum += value * value;
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(hidden.size()) + model.descriptor.norm_epsilon);
    for (size_t column = 0; column < hidden.size(); ++column)
        hidden[column] *= inverse_rms * dense_tensor_value(norm_weight, column);

    const TensorData* bias = moe.router_bias == invalid_tensor_handle
                                 ? nullptr
                                 : &model.weights.at(moe.router_bias);
    std::vector<float> router_logits(router_weight.shape[0], 0.0f);
    for (uint32_t expert_id = 0; expert_id < router_weight.shape[0]; ++expert_id) {
        float score = bias ? dense_tensor_value(*bias, expert_id) : 0.0f;
        const size_t weight_offset = static_cast<size_t>(expert_id) * hidden.size();
        for (size_t column = 0; column < hidden.size(); ++column)
            score += dense_tensor_value(router_weight, weight_offset + column) * hidden[column];
        router_logits[expert_id] = score;
    }

    ExpertDispatchOptions options;
    options.expert_count = static_cast<uint32_t>(moe.experts.size());
    options.top_k = moe.top_k;
    options.normalization = moe.normalization;
    options.flags = 0;
    if (has_flag(moe.flags, MoeBlockNormalizeTopKWeights))
        options.flags |= ExpertDispatchNormalizeTopKWeights;
    ExpertDispatcher dispatcher;
    auto dispatch = dispatcher.dispatch(router_logits, 1, options);
    if (!dispatch)
        return;
    for (const ExpertBatch& batch : dispatch.value().batches) {
        if (!model.expert_cache->prediction_is_current(prediction_generation))
            return;
        if (batch.expert_id >= moe.experts.size())
            continue;
        const ExpertPlan& expert = moe.experts[batch.expert_id];
        if (expert.gate_up_weight == invalid_tensor_handle
            || expert.down_weight == invalid_tensor_handle) {
            continue;
        }
        const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
        const TensorData& down = model.weights.at(expert.down_weight);
        if (!gate_up.mxfp4_file_storage || !down.mxfp4_file_storage)
            continue;
        // Prediction is best effort: invalid metadata is rejected during exact
        // execution and capacity pressure merely skips speculative admission.
        (void)model.expert_cache->prefetch_pair(gate_up, down);
    }
}

static float activate(float value, ExpertActivation activation, float limit)
{
    switch (activation) {
    case ExpertActivation::Relu:
        return std::max(0.0f, value);
    case ExpertActivation::Silu:
        return value / (1.0f + std::exp(-value));
    case ExpertActivation::Gelu:
        return 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
    case ExpertActivation::ClampedSilu:
    {
        const float clamped = limit > 0.0f ? std::clamp(value, -limit, limit) : value;
        return clamped / (1.0f + std::exp(-clamped));
    }
    case ExpertActivation::GptOssSwiGlu:
        return value;
    }
    return value;
}

static void record_mxfp4_projection(
    const TensorData& matrix,
    const CpuBatch& input,
    ExpertExecutionMetrics& metrics)
{
    if (matrix.dtype != DType::MxFp4)
        return;
    const uint64_t rows = static_cast<uint64_t>(matrix.shape[0]) * input.rows();
    metrics.mxfp4_paired_rows
        += static_cast<uint64_t>(matrix.shape[0] / 2) * 2 * input.rows();
    if (input.rows() == 1)
        metrics.mxfp4_decode_gemv_rows += rows;
    else
        metrics.mxfp4_prefill_gemm_rows += rows;
}

static CpuBatch execute_expert_linear(
    const TensorData& matrix,
    const TensorData* bias,
    const CpuBatch& input,
    ExpertExecutionMetrics& metrics)
{
    record_mxfp4_projection(matrix, input, metrics);
    return bias ? linear_batch(matrix, *bias, input) : linear_batch(matrix, input);
}

static CpuBatch execute_expert_batch(
    const WeightTable& weights,
    const ExpertPlan& expert,
    const ExpertCacheLease* cached_weights,
    const CpuBatch& input,
    bool prefetch,
    ExpertExecutionMetrics& metrics)
{
    if (expert.gate_up_weight != invalid_tensor_handle) {
        const TensorData& gate_up_weight = cached_weights && cached_weights->gate_up
                                               ? *cached_weights->gate_up
                                               : weights.at(expert.gate_up_weight);
        if (prefetch)
            metrics.hinted_bytes += prefetch_tensor(gate_up_weight);
        const TensorData* gate_up_bias = expert.gate_up_bias == invalid_tensor_handle
                                             ? nullptr
                                             : &weights.at(expert.gate_up_bias);
        CpuBatch activated;
        if (gate_up_weight.dtype == DType::MxFp4) {
            activated = fused_mxfp4_gate_up_batch(
                gate_up_weight,
                gate_up_bias,
                input,
                expert.activation_limit);
            metrics.mxfp4_fused_gate_up_rows
                += static_cast<uint64_t>(activated.rows()) * activated.columns();
            record_mxfp4_projection(gate_up_weight, input, metrics);
        }
        else {
            CpuBatch gate_up = execute_expert_linear(
                gate_up_weight,
                gate_up_bias,
                input,
                metrics);
            activated = CpuBatch(gate_up.rows(), gate_up.columns() / 2);
            for (size_t token_index = 0; token_index < gate_up.rows(); ++token_index) {
                const float* source = gate_up.row(token_index);
                float* destination = activated.row(token_index);
                for (uint32_t column = 0; column < activated.columns(); ++column) {
                    const float gate = expert.activation_limit > 0.0f
                                           ? std::min(source[column * 2], expert.activation_limit)
                                           : source[column * 2];
                    const float linear = expert.activation_limit > 0.0f
                                             ? std::clamp(
                                                   source[column * 2 + 1],
                                                   -expert.activation_limit,
                                                   expert.activation_limit)
                                             : source[column * 2 + 1];
                    const float silu = gate / (1.0f + std::exp(-1.702f * gate));
                    destination[column] = silu * (linear + 1.0f);
                }
            }
        }
        const TensorData& down_weight = cached_weights && cached_weights->down
                                            ? *cached_weights->down
                                            : weights.at(expert.down_weight);
        if (prefetch)
            metrics.hinted_bytes += prefetch_tensor(down_weight);
        return execute_expert_linear(
            down_weight,
            expert.down_bias == invalid_tensor_handle
                ? nullptr
                : &weights.at(expert.down_bias),
            activated,
            metrics);
    }

    if (prefetch)
        metrics.hinted_bytes += prefetch_weight(weights, expert.up_weight);
    CpuBatch up = execute_expert_linear(
        weights.at(expert.up_weight),
        nullptr,
        input,
        metrics);
    if (has_flag(expert.flags, ExpertPlanGated)) {
        if (prefetch)
            metrics.hinted_bytes += prefetch_weight(weights, expert.gate_weight);
        const CpuBatch gate = execute_expert_linear(
            weights.at(expert.gate_weight),
            nullptr,
            input,
            metrics);
        for (size_t token_index = 0; token_index < up.rows(); ++token_index) {
            float* up_row = up.row(token_index);
            const float* gate_row = gate.row(token_index);
            for (uint32_t column = 0; column < up.columns(); ++column)
                up_row[column] *= activate(gate_row[column], expert.activation, expert.activation_limit);
        }
    }
    else {
        for (size_t token_index = 0; token_index < up.rows(); ++token_index) {
            float* token = up.row(token_index);
            for (uint32_t column = 0; column < up.columns(); ++column)
                token[column] = activate(token[column], expert.activation, expert.activation_limit);
        }
    }
    if (prefetch)
        metrics.hinted_bytes += prefetch_weight(weights, expert.down_weight);
    return execute_expert_linear(
        weights.at(expert.down_weight),
        nullptr,
        up,
        metrics);
}

static CpuBatch gather_tokens(
    const CpuBatch& source,
    const std::vector<ExpertRoute>& routes)
{
    CpuBatch gathered(routes.size(), source.columns());
    for (size_t route_index = 0; route_index < routes.size(); ++route_index) {
        std::copy_n(
            source.row(routes[route_index].token_index),
            source.columns(),
            gathered.row(route_index));
    }
    return gathered;
}

static Result<void> execute_active_experts(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    LayerGraphState& layer_state,
    SessionStatistics& statistics)
{
    const size_t active_expert_count = layer_state.active_experts.size();
    bool parallelize_experts = false;
    int expert_team_size = 1;
#if defined(_OPENMP)
    expert_team_size = std::min(
        static_cast<int>(active_expert_count),
        omp_get_max_threads());
    parallelize_experts = expert_team_size > 1;
#endif
    const int64_t parallel_expert_count
        = static_cast<int64_t>(active_expert_count);
#pragma omp parallel for schedule(dynamic, 1) num_threads(expert_team_size) if (parallelize_experts)
    for (int64_t expert_index = 0;
         expert_index < parallel_expert_count;
         ++expert_index) {
        ActiveExpertExecution& active
            = layer_state.active_experts[static_cast<size_t>(expert_index)];
        const uint32_t expert_id = active.batch.expert_id;
        ExpertCacheLease expert_lease;
        if (model.expert_cache) {
            const ExpertPlan& expert = moe.experts[expert_id];
            const TensorData* gate_up
                = expert.gate_up_weight == invalid_tensor_handle
                      ? nullptr
                      : &model.weights.at(expert.gate_up_weight);
            const TensorData& down = model.weights.at(expert.down_weight);
            if (gate_up
                && (gate_up->mxfp4_file_storage
                    || down.mxfp4_file_storage)) {
                auto lease = model.expert_cache->acquire_pair(*gate_up, down);
                if (!lease) {
                    active.error = lease.error();
                    active.failed = true;
                    continue;
                }
                expert_lease = std::move(lease).value();
            }
        }

        const CpuBatch expert_input
            = gather_tokens(layer_state.normalized, active.batch.routes);
        active.output = execute_expert_batch(
            model.weights,
            moe.experts[expert_id],
            expert_lease.gate_up ? &expert_lease : nullptr,
            expert_input,
            model.hybrid_mode == HybridMode::VulkanWithCpuPrefetch,
            active.metrics);
    }

    for (const ActiveExpertExecution& active : layer_state.active_experts) {
        if (active.failed)
            return active.error;
    }
    if (parallelize_experts)
        statistics.expert_parallel_tasks += active_expert_count;
    for (const ActiveExpertExecution& active : layer_state.active_experts) {
        const ExpertExecutionMetrics& metrics = active.metrics;
        if (metrics.hinted_bytes > 0) {
            ++statistics.expert_prefetches;
            statistics.expert_prefetch_bytes += metrics.hinted_bytes;
        }
        statistics.mxfp4_decode_gemv_rows += metrics.mxfp4_decode_gemv_rows;
        statistics.mxfp4_prefill_gemm_rows += metrics.mxfp4_prefill_gemm_rows;
        statistics.mxfp4_paired_rows += metrics.mxfp4_paired_rows;
        statistics.mxfp4_fused_gate_up_rows
            += metrics.mxfp4_fused_gate_up_rows;
        ++statistics.expert_batches;
    }
    layer_state.experts_executed = true;
    return {};
}

Result<std::vector<std::vector<float> > > CpuExecutor::execute(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    SessionStatistics& statistics,
    CpuSessionState& state,
    uint64_t position_offset) const
{
    const uint64_t initial_vulkan_dispatches = NcnnLinearOperator::current_thread_vulkan_dispatches();
    const uint64_t initial_vulkan_attention_blocks = NcnnVulkanAttentionOperator::current_thread_blocks();
    const NcnnVulkanRuntimeCounters initial_vulkan_runtime_counters
        = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
    for (int32_t token_id : input_ids) {
        if (token_id < 0 || static_cast<uint32_t>(token_id) >= model.descriptor.vocabulary_size)
            return Error{ErrorCode::InvalidArgument, "token id is outside the model vocabulary"};
    }

    if (statistics.expert_token_counts.size() < model.descriptor.expert_count)
        statistics.expert_token_counts.resize(model.descriptor.expert_count, 0);

    if (state.layers.size() != model.layers.size())
        state.layers.resize(model.layers.size());

    ExpertCacheStatistics execution_cache_before;
    if (model.expert_cache)
        execution_cache_before = model.expert_cache->statistics();

    CpuBatch hidden;
    std::vector<std::vector<float> > logits;
    std::vector<LayerGraphState> layer_states(model.layers.size());
    ExpertDispatcher dispatcher;
    for (const ExecutionWave& wave : model.schedule.waves) {
        for (ExecutionNodeId node_id : wave.nodes) {
            const ExecutionNode* node = model.graph.find(node_id);
            if (!node)
                return Error{ErrorCode::InternalError, "execution schedule references an invalid node"};

            if (node->type == ExecutionNodeType::TokenEmbedding) {
                hidden = embedding_batch(
                    model.weights.at(model.token_embedding),
                    input_ids);
                continue;
            }
            if (node->type == ExecutionNodeType::FinalNorm) {
                hidden = rms_norm_batch(
                    hidden,
                    model.weights.at(model.final_norm_weight),
                    model.descriptor.norm_epsilon);
                continue;
            }
            if (node->type == ExecutionNodeType::LmHead) {
                logits = batch_to_vectors(linear_batch(
                    model.weights.at(model.lm_head_weight),
                    hidden));
                continue;
            }
            if (node->layer_id >= model.layers.size())
                return Error{ErrorCode::InternalError, "execution node layer is out of range"};

            const CompiledLayerPlan& layer = model.layers[node->layer_id];
            LayerGraphState& layer_state = layer_states[node->layer_id];
            const MoeBlockPlan& moe = layer.moe;
            if (node->type == ExecutionNodeType::Attention) {
                const auto attention_start = std::chrono::steady_clock::now();
                hidden = execute_attention_block(
                    model.weights,
                    layer.attention,
                    model.descriptor.norm_epsilon,
                    model.descriptor.kv_cache_dtype,
                    position_offset,
                    state.layers[layer.layer_id],
                    hidden);
                statistics.attention_time_microseconds
                    += elapsed_microseconds(attention_start);
                continue;
            }
            if (node->type == ExecutionNodeType::Router) {
                if (model.expert_cache)
                    model.expert_cache->cancel_prediction();
                layer_state.router_start = std::chrono::steady_clock::now();
                layer_state.normalized = rms_norm_batch(
                    hidden,
                    model.weights.at(moe.pre_ffn_norm_weight),
                    model.descriptor.norm_epsilon);
                layer_state.router_logits = linear_batch(
                    model.weights.at(moe.router_weight),
                    layer_state.normalized);
                if (moe.router_bias != invalid_tensor_handle) {
                    add_bias_inplace(
                        layer_state.router_logits,
                        model.weights.at(moe.router_bias));
                }
                continue;
            }
            if (node->type == ExecutionNodeType::ExpertDispatch) {
                ExpertDispatchOptions options;
                options.expert_count
                    = static_cast<uint32_t>(moe.experts.size());
                options.top_k = moe.top_k;
                options.normalization = moe.normalization;
                options.flags = 0;
                if (has_flag(
                        moe.flags,
                        MoeBlockNormalizeTopKWeights)) {
                    options.flags |= ExpertDispatchNormalizeTopKWeights;
                }
                auto dispatch = dispatcher.dispatch(
                    layer_state.router_logits.values(),
                    static_cast<uint32_t>(layer_state.router_logits.rows()),
                    options);
                if (!dispatch)
                    return dispatch.error();

                ExpertDispatchPlan plan = std::move(dispatch).value();
                statistics.expert_assignments
                    += static_cast<uint64_t>(plan.assignment_count);
                layer_state.active_experts.clear();
                layer_state.active_experts.reserve(plan.batches.size());
                for (ExpertBatch& batch : plan.batches) {
                    statistics.expert_token_counts[batch.expert_id]
                        += static_cast<uint64_t>(batch.routes.size());
                    ActiveExpertExecution active;
                    active.batch = std::move(batch);
                    layer_state.active_experts.push_back(std::move(active));
                }
                layer_state.router_logits = {};
                statistics.router_time_microseconds
                    += elapsed_microseconds(layer_state.router_start);
                layer_state.expert_start = std::chrono::steady_clock::now();

                if (model.expert_cache) {
                    for (const ActiveExpertExecution& active : layer_state.active_experts) {
                        const ExpertPlan& expert
                            = moe.experts[active.batch.expert_id];
                        if (expert.gate_up_weight == invalid_tensor_handle)
                            continue;
                        const TensorData& gate_up
                            = model.weights.at(expert.gate_up_weight);
                        const TensorData& down
                            = model.weights.at(expert.down_weight);
                        if (!gate_up.mxfp4_file_storage
                            && !down.mxfp4_file_storage) {
                            continue;
                        }
                        auto requested
                            = model.expert_cache->request_pair(gate_up, down);
                        if (!requested)
                            return requested.error();
                    }
                    std::stable_sort(
                        layer_state.active_experts.begin(),
                        layer_state.active_experts.end(),
                        [&model, &moe](
                            const ActiveExpertExecution& left,
                            const ActiveExpertExecution& right) {
                            const ExpertPlan& left_plan
                                = moe.experts[left.batch.expert_id];
                            const ExpertPlan& right_plan
                                = moe.experts[right.batch.expert_id];
                            const bool left_ready
                                = left_plan.gate_up_weight
                                      == invalid_tensor_handle
                                  || model.expert_cache->is_ready(
                                      model.weights.at(
                                          left_plan.gate_up_weight),
                                      model.weights.at(
                                          left_plan.down_weight));
                            const bool right_ready
                                = right_plan.gate_up_weight
                                      == invalid_tensor_handle
                                  || model.expert_cache->is_ready(
                                      model.weights.at(
                                          right_plan.gate_up_weight),
                                      model.weights.at(
                                          right_plan.down_weight));
                            return left_ready && !right_ready;
                        });
                }
                continue;
            }
            if (node->type == ExecutionNodeType::Expert) {
                if (layer_state.experts_executed)
                    continue;
                auto executed = execute_active_experts(
                    model,
                    moe,
                    layer_state,
                    statistics);
                if (!executed)
                    return executed.error();
                continue;
            }
            if (node->type == ExecutionNodeType::Combine) {
                if (!layer_state.experts_executed) {
                    return Error{
                        ErrorCode::InternalError,
                        "Combine executed before its Expert wave"};
                }
                CpuBatch moe_output(hidden.rows(), model.descriptor.hidden_size);
                for (const ActiveExpertExecution& active : layer_state.active_experts) {
                    for (size_t batch_index = 0;
                         batch_index < active.batch.routes.size();
                         ++batch_index) {
                        const ExpertRoute& route
                            = active.batch.routes[batch_index];
                        float* destination
                            = moe_output.row(route.token_index);
                        const float* source = active.output.row(batch_index);
                        for (uint32_t column = 0;
                             column < model.descriptor.hidden_size;
                             ++column) {
                            destination[column]
                                += route.weight * source[column];
                        }
                    }
                }
                for (size_t token_index = 0;
                     token_index < hidden.rows();
                     ++token_index) {
                    float* hidden_row = hidden.row(token_index);
                    const float* output_row = moe_output.row(token_index);
                    for (uint32_t column = 0;
                         column < model.descriptor.hidden_size;
                         ++column) {
                        hidden_row[column] += output_row[column];
                    }
                }

                if (model.expert_cache
                    && hidden.rows() == 1
                    && node->layer_id + 1 < model.layers.size()) {
                    std::vector<float> prediction_input(
                        hidden.row(0),
                        hidden.row(0) + hidden.columns());
                    model.expert_cache->submit_prediction(
                        [&model,
                         next_layer = node->layer_id + 1,
                         hidden = std::move(prediction_input)](
                            uint64_t prediction_generation) mutable {
                            predict_next_layer_experts(
                                model,
                                next_layer,
                                prediction_generation,
                                std::move(hidden));
                        });
                }
                statistics.expert_time_microseconds
                    += elapsed_microseconds(layer_state.expert_start);
                layer_state = {};
                continue;
            }
            return Error{ErrorCode::InternalError, "unsupported execution graph node"};
        }
    }

    if (model.expert_cache) {
        const ExpertCacheStatistics after = model.expert_cache->statistics();
        statistics.expert_cache_hits += after.hits - execution_cache_before.hits;
        statistics.expert_cache_misses += after.misses - execution_cache_before.misses;
        statistics.expert_cache_evictions += after.evictions - execution_cache_before.evictions;
        statistics.expert_cache_bytes_read += after.bytes_read - execution_cache_before.bytes_read;
        statistics.expert_cache_queued_reads
            += after.queued_reads - execution_cache_before.queued_reads;
        statistics.expert_cache_speculative_reads
            += after.speculative_reads - execution_cache_before.speculative_reads;
        statistics.expert_cache_mapped_ranges
            += after.mapped_ranges - execution_cache_before.mapped_ranges;
        statistics.expert_cache_mapped_bytes
            += after.mapped_bytes - execution_cache_before.mapped_bytes;
        statistics.expert_cache_resident_bytes = after.resident_bytes;
        statistics.expert_gpu_cache_hits
            += after.victim.hits - execution_cache_before.victim.hits;
        statistics.expert_gpu_cache_misses
            += after.victim.misses - execution_cache_before.victim.misses;
        statistics.expert_gpu_cache_admissions
            += after.victim.admissions
               - execution_cache_before.victim.admissions;
        statistics.expert_gpu_cache_stores
            += after.victim.stores - execution_cache_before.victim.stores;
        statistics.expert_gpu_cache_evictions
            += after.victim.evictions
               - execution_cache_before.victim.evictions;
        statistics.expert_gpu_cache_dropped_admissions
            += after.victim.dropped_admissions
               - execution_cache_before.victim.dropped_admissions;
        statistics.expert_gpu_cache_restore_failures
            += after.victim.restore_failures
               - execution_cache_before.victim.restore_failures;
        statistics.expert_gpu_cache_bytes_uploaded
            += after.victim.bytes_uploaded
               - execution_cache_before.victim.bytes_uploaded;
        statistics.expert_gpu_cache_bytes_downloaded
            += after.victim.bytes_downloaded
               - execution_cache_before.victim.bytes_downloaded;
        statistics.expert_gpu_cache_restore_time_microseconds
            += after.victim.restore_time_microseconds
               - execution_cache_before.victim.restore_time_microseconds;
        statistics.expert_gpu_cache_mapped_stores
            += after.victim.mapped_stores
               - execution_cache_before.victim.mapped_stores;
        statistics.expert_gpu_cache_mapped_restores
            += after.victim.mapped_restores
               - execution_cache_before.victim.mapped_restores;
        statistics.expert_gpu_cache_resident_bytes
            = after.victim.resident_bytes;
        statistics.expert_gpu_cache_pending_bytes
            = after.victim.pending_bytes;
    }

    if (logits.empty())
        return Error{ErrorCode::InternalError, "execution graph did not produce logits"};
    statistics.vulkan_linear_dispatches += NcnnLinearOperator::current_thread_vulkan_dispatches()
                                           - initial_vulkan_dispatches;
    statistics.vulkan_attention_blocks += NcnnVulkanAttentionOperator::current_thread_blocks()
                                          - initial_vulkan_attention_blocks;
    const NcnnVulkanRuntimeCounters final_vulkan_runtime_counters
        = NcnnLinearOperator::current_thread_vulkan_runtime_counters();
    statistics.vulkan_compute_submissions
        += final_vulkan_runtime_counters.compute_submissions
           - initial_vulkan_runtime_counters.compute_submissions;
    statistics.vulkan_batch_uploads
        += final_vulkan_runtime_counters.batch_uploads
           - initial_vulkan_runtime_counters.batch_uploads;
    statistics.vulkan_batch_downloads
        += final_vulkan_runtime_counters.batch_downloads
           - initial_vulkan_runtime_counters.batch_downloads;
    statistics.vulkan_auxiliary_uploads
        += final_vulkan_runtime_counters.auxiliary_uploads
           - initial_vulkan_runtime_counters.auxiliary_uploads;
    statistics.vulkan_auxiliary_upload_bytes
        += final_vulkan_runtime_counters.auxiliary_upload_bytes
           - initial_vulkan_runtime_counters.auxiliary_upload_bytes;
    statistics.vulkan_staging_slot_resizes
        += final_vulkan_runtime_counters.staging_slot_resizes
           - initial_vulkan_runtime_counters.staging_slot_resizes;
    statistics.vulkan_staging_slot_reuses
        += final_vulkan_runtime_counters.staging_slot_reuses
           - initial_vulkan_runtime_counters.staging_slot_reuses;
    statistics.vulkan_staging_slot_acquisitions
        += final_vulkan_runtime_counters.staging_slot_acquisitions
           - initial_vulkan_runtime_counters.staging_slot_acquisitions;
    statistics.vulkan_staging_slot_contentions
        += final_vulkan_runtime_counters.staging_slot_contentions
           - initial_vulkan_runtime_counters.staging_slot_contentions;
    return logits;
}

} // namespace moe
} // namespace ncnn
