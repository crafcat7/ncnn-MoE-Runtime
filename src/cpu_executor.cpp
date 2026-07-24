#include "cpu_executor.h"

#include "cpu_attention.h"
#include "cpu_batch.h"
#include "cpu_ops.h"
#include "cpu_session_state.h"
#include "expert_cache.h"
#include "ncnn_attention.h"
#include "ncnn_linear.h"

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

struct RouteItem
{
    uint32_t token_index = 0;
    float weight = 0.0f;
};

struct RouteCandidate
{
    uint32_t expert_id = 0;
    float score = 0.0f;
};

struct ExpertExecutionMetrics
{
    uint64_t hinted_bytes = 0;
    uint64_t mxfp4_decode_gemv_rows = 0;
    uint64_t mxfp4_prefill_gemm_rows = 0;
    uint64_t mxfp4_paired_rows = 0;
    uint64_t mxfp4_fused_gate_up_rows = 0;
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
        return prefetch_buffer(
            tensor.float32_data.data(),
            tensor.float32_data.size() * sizeof(float));
    }
    if (tensor.dtype == DType::BFloat16) {
        return prefetch_buffer(
            tensor.bfloat16_data.data(),
            tensor.bfloat16_data.size() * sizeof(uint16_t));
    }
    if (tensor.dtype == DType::Int8)
        return prefetch_buffer(tensor.int8_data.data(), tensor.int8_data.size());
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

static bool route_precedes(const RouteCandidate& left, const RouteCandidate& right)
{
    return left.score > right.score
           || (left.score == right.score && left.expert_id < right.expert_id);
}

static void softmax(const float* logits, uint32_t size, std::vector<float>& probabilities)
{
    probabilities.resize(size);
    const float maximum = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (uint32_t index = 0; index < size; ++index) {
        probabilities[index] = std::exp(logits[index] - maximum);
        sum += probabilities[index];
    }
    for (float& probability : probabilities)
        probability /= sum;
}

static void select_topk_routes(
    const float* logits,
    uint32_t expert_count,
    uint32_t top_k,
    std::vector<RouteCandidate>& selected)
{
    selected.clear();
    for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id) {
        const RouteCandidate candidate{expert_id, logits[expert_id]};
        const auto insertion = std::lower_bound(
            selected.begin(), selected.end(), candidate, route_precedes);
        if (selected.size() == top_k && insertion == selected.end())
            continue;
        selected.insert(insertion, candidate);
        if (selected.size() > top_k)
            selected.pop_back();
    }
}

static float dense_tensor_value(const TensorData& tensor, size_t index)
{
    if (tensor.dtype == DType::Float32)
        return tensor.float32_data[index];
    if (tensor.dtype == DType::BFloat16)
        return bfloat16_to_float(tensor.bfloat16_data[index]);
    if (tensor.dtype == DType::Int8) {
        const uint32_t columns = tensor.shape[1];
        return static_cast<float>(tensor.int8_data[index])
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
    const float inverse_rms = 1.0f / std::sqrt(
        square_sum / static_cast<float>(hidden.size())
        + model.descriptor.norm_epsilon);
    for (size_t column = 0; column < hidden.size(); ++column)
        hidden[column] *= inverse_rms * dense_tensor_value(norm_weight, column);

    const TensorData* bias = moe.router_bias == invalid_tensor_handle
                                 ? nullptr
                                 : &model.weights.at(moe.router_bias);
    std::vector<RouteCandidate> selected;
    selected.reserve(moe.top_k);
    for (uint32_t expert_id = 0; expert_id < router_weight.shape[0]; ++expert_id) {
        float score = bias ? dense_tensor_value(*bias, expert_id) : 0.0f;
        const size_t weight_offset = static_cast<size_t>(expert_id) * hidden.size();
        for (size_t column = 0; column < hidden.size(); ++column)
            score += dense_tensor_value(router_weight, weight_offset + column) * hidden[column];
        const RouteCandidate candidate{expert_id, score};
        const auto insertion = std::lower_bound(
            selected.begin(), selected.end(), candidate, route_precedes);
        if (selected.size() == moe.top_k && insertion == selected.end())
            continue;
        selected.insert(insertion, candidate);
        if (selected.size() > moe.top_k)
            selected.pop_back();
    }

    for (const RouteCandidate& candidate : selected) {
        if (!model.expert_cache->prediction_is_current(prediction_generation))
            return;
        if (candidate.expert_id >= moe.experts.size())
            continue;
        const ExpertPlan& expert = moe.experts[candidate.expert_id];
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
    if (expert.gated) {
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

static CpuBatch gather_tokens(const CpuBatch& source, const std::vector<RouteItem>& routes)
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

    const uint32_t hidden_size = model.descriptor.hidden_size;
    CpuBatch hidden = embedding_batch(model.weights.at(model.token_embedding), input_ids);

    if (statistics.expert_token_counts.size() < model.descriptor.expert_count)
        statistics.expert_token_counts.resize(model.descriptor.expert_count, 0);

    if (state.layers.size() != model.layers.size())
        state.layers.resize(model.layers.size());

    ExpertCacheStatistics execution_cache_before;
    if (model.expert_cache)
        execution_cache_before = model.expert_cache->statistics();

    for (size_t layer_index = 0; layer_index < model.layers.size(); ++layer_index) {
        const CompiledLayerPlan& layer = model.layers[layer_index];
        if (layer.use_attention) {
            const auto attention_start = std::chrono::steady_clock::now();
            hidden = execute_attention_block(
                model.weights,
                layer.attention,
                model.descriptor.norm_epsilon,
                model.descriptor.kv_cache_dtype,
                position_offset,
                state.layers[layer.layer_id],
                hidden);
            statistics.attention_time_microseconds += elapsed_microseconds(attention_start);
        }
        if (model.expert_cache)
            model.expert_cache->cancel_prediction();

        const auto router_start = std::chrono::steady_clock::now();
        const MoeBlockPlan& moe = layer.moe;
        const TensorData& norm_weight = model.weights.at(moe.pre_ffn_norm_weight);
        const TensorData& router_weight = model.weights.at(moe.router_weight);
        const CpuBatch normalized = rms_norm_batch(hidden, norm_weight, model.descriptor.norm_epsilon);
        CpuBatch router_logits = linear_batch(router_weight, normalized);

        if (moe.router_bias != invalid_tensor_handle) {
            const TensorData& bias = model.weights.at(moe.router_bias);
            add_bias_inplace(router_logits, bias);
        }

        std::vector<std::vector<RouteItem> > groups(moe.experts.size());
        const bool renormalize = moe.normalize_topk_weights
                                 || moe.normalization == RouterNormalization::SelectedExperts;
        std::vector<float> probabilities;
        probabilities.reserve(router_logits.columns());
        std::vector<RouteCandidate> selected_experts;
        selected_experts.reserve(moe.top_k);
        for (uint32_t token_index = 0; token_index < router_logits.rows(); ++token_index) {
            const float* logits = router_logits.row(token_index);
            softmax(logits, router_logits.columns(), probabilities);
            select_topk_routes(
                probabilities.data(),
                router_logits.columns(),
                moe.top_k,
                selected_experts);
            float selected_sum = 0.0f;
            for (const RouteCandidate& candidate : selected_experts)
                selected_sum += candidate.score;

            for (const RouteCandidate& candidate : selected_experts) {
                const float routing_weight = renormalize
                                                 ? candidate.score / selected_sum
                                                 : candidate.score;
                groups[candidate.expert_id].push_back(RouteItem{token_index, routing_weight});
                ++statistics.expert_assignments;
                ++statistics.expert_token_counts[candidate.expert_id];
            }
        }
        statistics.router_time_microseconds += elapsed_microseconds(router_start);

        const auto expert_start = std::chrono::steady_clock::now();
        CpuBatch moe_output(hidden.rows(), hidden_size);
        size_t active_expert_count = 0;
        for (const std::vector<RouteItem>& group : groups)
            active_expert_count += group.empty() ? 0 : 1;
        bool parallelize_experts = false;
#if defined(_OPENMP)
        parallelize_experts = active_expert_count > 1 && omp_get_max_threads() > 1;
#endif
        std::vector<CpuBatch> expert_outputs(groups.size());
        std::vector<ExpertExecutionMetrics> expert_metrics(groups.size());
        std::vector<uint32_t> expert_order;
        expert_order.reserve(active_expert_count);
        for (uint32_t expert_id = 0; expert_id < groups.size(); ++expert_id) {
            if (!groups[expert_id].empty())
                expert_order.push_back(expert_id);
        }
        std::vector<Error> expert_errors(groups.size());
        std::vector<uint8_t> expert_failed(groups.size(), 0);
        if (model.expert_cache) {
            for (uint32_t expert_id : expert_order) {
                const ExpertPlan& expert = moe.experts[expert_id];
                if (expert.gate_up_weight == invalid_tensor_handle)
                    continue;
                const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                const TensorData& down = model.weights.at(expert.down_weight);
                if (!gate_up.mxfp4_file_storage && !down.mxfp4_file_storage)
                    continue;
                auto requested = model.expert_cache->request_pair(gate_up, down);
                if (!requested)
                    return requested.error();
            }
            // Ready experts run first on the serial fallback. With OpenMP this
            // also prevents all worker slots from blocking behind cold reads.
            std::stable_sort(
                expert_order.begin(),
                expert_order.end(),
                [&model, &moe](uint32_t left, uint32_t right) {
                    const ExpertPlan& left_plan = moe.experts[left];
                    const ExpertPlan& right_plan = moe.experts[right];
                    const bool left_ready = model.expert_cache->is_ready(
                        model.weights.at(left_plan.gate_up_weight),
                        model.weights.at(left_plan.down_weight));
                    const bool right_ready = model.expert_cache->is_ready(
                        model.weights.at(right_plan.gate_up_weight),
                        model.weights.at(right_plan.down_weight));
                    return left_ready && !right_ready;
                });
        }
        const int64_t parallel_expert_count = static_cast<int64_t>(expert_order.size());
#pragma omp parallel for schedule(dynamic, 1) if(parallelize_experts)
        for (int64_t expert_index = 0; expert_index < parallel_expert_count; ++expert_index) {
            const uint32_t expert_id = expert_order[static_cast<size_t>(expert_index)];
            const std::vector<RouteItem>& group = groups[expert_id];
            ExpertCacheLease expert_lease;
            if (model.expert_cache) {
                const ExpertPlan& expert = moe.experts[expert_id];
                const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                const TensorData& down = model.weights.at(expert.down_weight);
                if (gate_up.mxfp4_file_storage || down.mxfp4_file_storage) {
                    auto lease = model.expert_cache->acquire_pair(gate_up, down);
                    if (!lease) {
                        expert_errors[expert_id] = lease.error();
                        expert_failed[expert_id] = 1;
                        continue;
                    }
                    expert_lease = std::move(lease).value();
                }
            }

            const CpuBatch expert_input = gather_tokens(normalized, group);
            expert_outputs[expert_id] = execute_expert_batch(
                model.weights,
                moe.experts[expert_id],
                expert_lease.gate_up ? &expert_lease : nullptr,
                expert_input,
                model.hybrid_mode == HybridMode::VulkanWithCpuPrefetch,
                expert_metrics[expert_id]);
        }
        for (uint32_t expert_id : expert_order) {
            if (expert_failed[expert_id])
                return expert_errors[expert_id];
        }
        if (parallelize_experts)
            statistics.expert_parallel_tasks += active_expert_count;
        for (uint32_t expert_id = 0; expert_id < groups.size(); ++expert_id) {
            const std::vector<RouteItem>& group = groups[expert_id];
            if (group.empty())
                continue;
            const ExpertExecutionMetrics& metrics = expert_metrics[expert_id];
            if (metrics.hinted_bytes > 0) {
                ++statistics.expert_prefetches;
                statistics.expert_prefetch_bytes += metrics.hinted_bytes;
            }
            statistics.mxfp4_decode_gemv_rows += metrics.mxfp4_decode_gemv_rows;
            statistics.mxfp4_prefill_gemm_rows += metrics.mxfp4_prefill_gemm_rows;
            statistics.mxfp4_paired_rows += metrics.mxfp4_paired_rows;
            statistics.mxfp4_fused_gate_up_rows += metrics.mxfp4_fused_gate_up_rows;
            ++statistics.expert_batches;

            const CpuBatch& expert_output = expert_outputs[expert_id];
            for (size_t batch_index = 0; batch_index < group.size(); ++batch_index) {
                const RouteItem& route = group[batch_index];
                float* destination = moe_output.row(route.token_index);
                const float* source = expert_output.row(batch_index);
                for (uint32_t column = 0; column < hidden_size; ++column)
                    destination[column] += route.weight * source[column];
            }
        }

        for (size_t token_index = 0; token_index < hidden.rows(); ++token_index) {
            float* hidden_row = hidden.row(token_index);
            const float* output_row = moe_output.row(token_index);
            for (uint32_t column = 0; column < hidden_size; ++column)
                hidden_row[column] += output_row[column];
        }
        if (model.expert_cache
            && hidden.rows() == 1
            && layer_index + 1 < model.layers.size()) {
            std::vector<float> prediction_input(
                hidden.row(0),
                hidden.row(0) + hidden.columns());
            model.expert_cache->submit_prediction(
                [&model, next_layer = layer_index + 1, hidden = std::move(prediction_input)](
                    uint64_t prediction_generation) mutable {
                    predict_next_layer_experts(
                        model,
                        next_layer,
                        prediction_generation,
                        std::move(hidden));
                });
        }
        statistics.expert_time_microseconds += elapsed_microseconds(expert_start);
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
        statistics.expert_cache_resident_bytes = after.resident_bytes;
    }

    const TensorData& final_norm = model.weights.at(model.final_norm_weight);
    const TensorData& lm_head = model.weights.at(model.lm_head_weight);
    const CpuBatch normalized = rms_norm_batch(hidden, final_norm, model.descriptor.norm_epsilon);
    std::vector<std::vector<float> > logits = batch_to_vectors(linear_batch(lm_head, normalized));
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
