#include "cpu_executor.h"

#include "cpu_attention.h"
#include "cpu_batch.h"
#include "cpu_ops.h"
#include "cpu_session_state.h"
#include "ncnn_attention.h"
#include "ncnn_linear.h"

#include "ncnn/moe/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>

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

static CpuBatch execute_expert_batch(
    const WeightTable& weights,
    const ExpertPlan& expert,
    const CpuBatch& input,
    bool prefetch,
    uint64_t& hinted_bytes)
{
    if (expert.gate_up_weight != invalid_tensor_handle) {
        if (prefetch)
            hinted_bytes += prefetch_weight(weights, expert.gate_up_weight);
        CpuBatch gate_up = expert.gate_up_bias == invalid_tensor_handle
                               ? linear_batch(weights.at(expert.gate_up_weight), input)
                               : linear_batch(weights.at(expert.gate_up_weight), weights.at(expert.gate_up_bias), input);
        CpuBatch activated(gate_up.rows(), gate_up.columns() / 2);
        for (size_t token_index = 0; token_index < gate_up.rows(); ++token_index) {
            const float* source = gate_up.row(token_index);
            float* destination = activated.row(token_index);
            for (uint32_t column = 0; column < activated.columns(); ++column) {
                const float gate = expert.activation_limit > 0.0f
                                       ? std::min(source[column * 2], expert.activation_limit)
                                       : source[column * 2];
                const float linear = expert.activation_limit > 0.0f
                                         ? std::clamp(source[column * 2 + 1], -expert.activation_limit, expert.activation_limit)
                                         : source[column * 2 + 1];
                const float silu = gate / (1.0f + std::exp(-1.702f * gate));
                destination[column] = silu * (linear + 1.0f);
            }
        }
        if (prefetch)
            hinted_bytes += prefetch_weight(weights, expert.down_weight);
        return expert.down_bias == invalid_tensor_handle
                   ? linear_batch(weights.at(expert.down_weight), activated)
                   : linear_batch(weights.at(expert.down_weight), weights.at(expert.down_bias), activated);
    }

    if (prefetch)
        hinted_bytes += prefetch_weight(weights, expert.up_weight);
    CpuBatch up = linear_batch(weights.at(expert.up_weight), input);
    if (expert.gated) {
        if (prefetch)
            hinted_bytes += prefetch_weight(weights, expert.gate_weight);
        const CpuBatch gate = linear_batch(weights.at(expert.gate_weight), input);
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
        hinted_bytes += prefetch_weight(weights, expert.down_weight);
    return linear_batch(weights.at(expert.down_weight), up);
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

    for (const CompiledLayerPlan& layer : model.layers) {
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
        for (uint32_t expert_id = 0; expert_id < groups.size(); ++expert_id) {
            const std::vector<RouteItem>& group = groups[expert_id];
            if (group.empty())
                continue;

            const CpuBatch expert_input = gather_tokens(normalized, group);
            uint64_t hinted_bytes = 0;
            const CpuBatch expert_output = execute_expert_batch(
                model.weights,
                moe.experts[expert_id],
                expert_input,
                model.hybrid_mode == HybridMode::VulkanWithCpuPrefetch,
                hinted_bytes);
            if (hinted_bytes > 0) {
                ++statistics.expert_prefetches;
                statistics.expert_prefetch_bytes += hinted_bytes;
            }
            ++statistics.expert_batches;

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
        statistics.expert_time_microseconds += elapsed_microseconds(expert_start);
    }

    const TensorData& final_norm = model.weights.at(model.final_norm_weight);
    const TensorData& lm_head = model.weights.at(model.lm_head_weight);
    const CpuBatch normalized = rms_norm_batch(hidden, final_norm, model.descriptor.norm_epsilon);
    std::vector<std::vector<float> > logits = batch_to_vectors(linear_batch(lm_head, normalized));
    statistics.vulkan_linear_dispatches += NcnnLinearOperator::current_thread_vulkan_dispatches()
                                           - initial_vulkan_dispatches;
    statistics.vulkan_attention_blocks += NcnnVulkanAttentionOperator::current_thread_blocks()
                                          - initial_vulkan_attention_blocks;
    return logits;
}

} // namespace moe
} // namespace ncnn
