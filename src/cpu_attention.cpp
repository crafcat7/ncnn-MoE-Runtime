#include "cpu_attention.h"

#include "cpu_ops.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

namespace ncnn {
namespace moe {

static void apply_rope(
    float* vector,
    uint32_t dimension,
    uint64_t position,
    const AttentionBlockPlan& plan)
{
    const uint32_t half_dimension = dimension / 2;
    const float base = plan.rope_theta;
    const float factor = plan.rope_scaling_factor;
    float concentration = 1.0f;
    float low = 0.0f;
    float high = 0.0f;
    if (factor > 1.0f) {
        concentration = 0.1f * std::log(factor) + 1.0f;
        const float half = static_cast<float>(half_dimension);
        low = half * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_beta * 2.0f * std::numbers::pi_v<float>)) / std::log(base);
        high = half * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_alpha * 2.0f * std::numbers::pi_v<float>)) / std::log(base);
    }

    for (uint32_t index = 0; index < half_dimension; ++index) {
        const float frequency = std::pow(base, static_cast<float>(2 * index) / static_cast<float>(dimension));
        float inverse_frequency = 1.0f / frequency;
        if (factor > 1.0f) {
            const float ramp = std::clamp((static_cast<float>(index) - low) / (high - low), 0.0f, 1.0f);
            const float mask = 1.0f - ramp;
            const float interpolation = 1.0f / (factor * frequency);
            const float extrapolation = 1.0f / frequency;
            inverse_frequency = interpolation * (1.0f - mask) + extrapolation * mask;
        }

        const float angle = static_cast<float>(position) * inverse_frequency;
        const float cosine = std::cos(angle) * concentration;
        const float sine = std::sin(angle) * concentration;
        const float first = vector[index];
        const float second = vector[half_dimension + index];
        vector[index] = first * cosine - second * sine;
        vector[half_dimension + index] = second * cosine + first * sine;
    }
}

static void append_cache(CpuLayerCache& cache, const CpuBatch& key, const CpuBatch& value)
{
    const size_t old_key_size = cache.keys.size();
    const size_t old_value_size = cache.values.size();
    cache.keys.resize(old_key_size + key.rows() * key.columns());
    cache.values.resize(old_value_size + value.rows() * value.columns());
    for (size_t token_index = 0; token_index < key.rows(); ++token_index) {
        std::copy_n(key.row(token_index), key.columns(), cache.keys.data() + old_key_size + token_index * key.columns());
        std::copy_n(value.row(token_index), value.columns(), cache.values.data() + old_value_size + token_index * value.columns());
    }
    cache.token_count += key.rows();
}

static CpuBatch scaled_dot_product_attention(
    const AttentionBlockPlan& plan,
    const TensorData& sinks,
    uint64_t position_offset,
    const CpuBatch& query,
    const CpuLayerCache& cache)
{
    const uint32_t head_count = plan.head_count;
    const uint32_t kv_head_count = plan.kv_head_count;
    const uint32_t head_dimension = plan.head_dimension;
    const uint32_t heads_per_group = head_count / kv_head_count;
    const uint32_t kv_columns = kv_head_count * head_dimension;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
    CpuBatch output(query.rows(), head_count * head_dimension);
    std::vector<float> logits(cache.token_count);

    for (size_t query_index = 0; query_index < query.rows(); ++query_index) {
        const uint64_t query_position = position_offset + query_index;
        for (uint32_t query_head = 0; query_head < head_count; ++query_head) {
            const uint32_t kv_head = query_head / heads_per_group;
            const float* query_vector = query.row(query_index) + query_head * head_dimension;
            float maximum = bfloat16_to_float(sinks.bfloat16_data.empty()
                                                  ? static_cast<uint16_t>(0)
                                                  : sinks.bfloat16_data[query_head]);
            if (sinks.dtype == DType::Float32)
                maximum = sinks.float32_data[query_head];

            for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index) {
                const uint64_t key_position = cache.start_position + key_index;
                const bool future = key_position > query_position;
                const bool too_old = plan.sliding_window > 0
                                     && key_position + plan.sliding_window <= query_position;
                if (future || too_old) {
                    logits[key_index] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                const float* key_vector = cache.keys.data() + key_index * kv_columns + kv_head * head_dimension;
                float dot = 0.0f;
                for (uint32_t column = 0; column < head_dimension; ++column)
                    dot += query_vector[column] * key_vector[column];
                logits[key_index] = dot * scale;
                maximum = std::max(maximum, logits[key_index]);
            }

            const float sink_value = sinks.dtype == DType::Float32
                                         ? sinks.float32_data[query_head]
                                         : bfloat16_to_float(sinks.bfloat16_data[query_head]);
            float normalizer = std::exp(sink_value - maximum);
            for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index) {
                if (std::isfinite(logits[key_index])) {
                    logits[key_index] = std::exp(logits[key_index] - maximum);
                    normalizer += logits[key_index];
                }
                else {
                    logits[key_index] = 0.0f;
                }
            }

            float* output_vector = output.row(query_index) + query_head * head_dimension;
            for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index) {
                const float probability = logits[key_index] / normalizer;
                const float* value_vector = cache.values.data() + key_index * kv_columns + kv_head * head_dimension;
                for (uint32_t column = 0; column < head_dimension; ++column)
                    output_vector[column] += probability * value_vector[column];
            }
        }
    }
    return output;
}

static void trim_sliding_cache(CpuLayerCache& cache, const AttentionBlockPlan& plan)
{
    if (plan.sliding_window == 0)
        return;
    const uint64_t retained_tokens = plan.sliding_window > 1 ? plan.sliding_window - 1 : 0;
    if (cache.token_count <= retained_tokens)
        return;

    const uint64_t removed_tokens = cache.token_count - retained_tokens;
    const size_t columns = static_cast<size_t>(plan.kv_head_count) * plan.head_dimension;
    if (retained_tokens > 0) {
        std::move(cache.keys.begin() + removed_tokens * columns, cache.keys.end(), cache.keys.begin());
        std::move(cache.values.begin() + removed_tokens * columns, cache.values.end(), cache.values.begin());
    }
    cache.keys.resize(retained_tokens * columns);
    cache.values.resize(retained_tokens * columns);
    cache.start_position += removed_tokens;
    cache.token_count = retained_tokens;
}

CpuBatch execute_attention_block(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& hidden)
{
    const CpuBatch normalized = rms_norm_batch(hidden, weights.at(plan.pre_attention_norm_weight), norm_epsilon);
    CpuBatch query = linear_batch(weights.at(plan.query_weight), weights.at(plan.query_bias), normalized);
    CpuBatch key = linear_batch(weights.at(plan.key_weight), weights.at(plan.key_bias), normalized);
    CpuBatch value = linear_batch(weights.at(plan.value_weight), weights.at(plan.value_bias), normalized);

    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index) {
        const uint64_t position = position_offset + token_index;
        for (uint32_t head = 0; head < plan.head_count; ++head)
            apply_rope(query.row(token_index) + head * plan.head_dimension, plan.head_dimension, position, plan);
        for (uint32_t head = 0; head < plan.kv_head_count; ++head)
            apply_rope(key.row(token_index) + head * plan.head_dimension, plan.head_dimension, position, plan);
    }

    if (cache.token_count == 0)
        cache.start_position = position_offset;
    append_cache(cache, key, value);
    const CpuBatch attention = scaled_dot_product_attention(
        plan, weights.at(plan.sinks), position_offset, query, cache);
    CpuBatch projected = linear_batch(weights.at(plan.output_weight), weights.at(plan.output_bias), attention);
    CpuBatch output = hidden;
    add_batch_inplace(output, projected);
    trim_sliding_cache(cache, plan);
    return output;
}

} // namespace moe
} // namespace ncnn
