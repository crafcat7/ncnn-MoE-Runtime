#include "cpu_attention.h"

#include "cpu_ops.h"
#include "backends/ncnn/ncnn_attention.h"
#include "backends/ncnn/ncnn_linear.h"

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

static uint64_t cache_slot(const CpuLayerCache& cache, uint64_t token_index)
{
    assert(cache.capacity_tokens > 0);
    return (cache.first_slot + token_index) % cache.capacity_tokens;
}

static void configure_cache(CpuLayerCache& cache, uint32_t columns, DType dtype)
{
    if (cache.columns == columns && cache.dtype == dtype)
        return;

    cache = {};
    cache.columns = columns;
    cache.dtype = dtype;
}

static uint64_t next_capacity(uint64_t current, uint64_t required)
{
    uint64_t capacity = current == 0 ? 16 : current;
    while (capacity < required) {
        if (capacity > std::numeric_limits<uint64_t>::max() / 2)
            return required;
        capacity *= 2;
    }
    return capacity;
}

static void resize_cache(CpuLayerCache& cache, uint64_t required_tokens)
{
    if (required_tokens <= cache.capacity_tokens)
        return;

    const uint64_t new_capacity = next_capacity(cache.capacity_tokens, required_tokens);
    const size_t element_count = static_cast<size_t>(new_capacity) * cache.columns;
    if (cache.dtype == DType::BFloat16) {
        std::vector<uint16_t> keys(element_count);
        std::vector<uint16_t> values(element_count);
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index) {
            const uint64_t old_slot = cache_slot(cache, token_index);
            std::copy_n(
                cache.bfloat16_keys.data() + old_slot * cache.columns,
                cache.columns,
                keys.data() + token_index * cache.columns);
            std::copy_n(
                cache.bfloat16_values.data() + old_slot * cache.columns,
                cache.columns,
                values.data() + token_index * cache.columns);
        }
        cache.bfloat16_keys = std::move(keys);
        cache.bfloat16_values = std::move(values);
    }
    else {
        std::vector<float> keys(element_count);
        std::vector<float> values(element_count);
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index) {
            const uint64_t old_slot = cache_slot(cache, token_index);
            std::copy_n(
                cache.keys.data() + old_slot * cache.columns,
                cache.columns,
                keys.data() + token_index * cache.columns);
            std::copy_n(
                cache.values.data() + old_slot * cache.columns,
                cache.columns,
                values.data() + token_index * cache.columns);
        }
        cache.keys = std::move(keys);
        cache.values = std::move(values);
    }
    cache.first_slot = 0;
    cache.capacity_tokens = new_capacity;
}

static void compact_cache(CpuLayerCache& cache, uint64_t target_capacity)
{
    if (target_capacity >= cache.capacity_tokens)
        return;

    const uint64_t previous_capacity = cache.capacity_tokens;
    cache.capacity_tokens = 0;
    if (cache.dtype == DType::BFloat16) {
        std::vector<uint16_t> old_keys = std::move(cache.bfloat16_keys);
        std::vector<uint16_t> old_values = std::move(cache.bfloat16_values);
        cache.bfloat16_keys.assign(static_cast<size_t>(target_capacity) * cache.columns, 0);
        cache.bfloat16_values.assign(static_cast<size_t>(target_capacity) * cache.columns, 0);
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index) {
            const uint64_t old_slot = (cache.first_slot + token_index) % previous_capacity;
            std::copy_n(
                old_keys.data() + old_slot * cache.columns,
                cache.columns,
                cache.bfloat16_keys.data() + token_index * cache.columns);
            std::copy_n(
                old_values.data() + old_slot * cache.columns,
                cache.columns,
                cache.bfloat16_values.data() + token_index * cache.columns);
        }
    }
    else {
        std::vector<float> old_keys = std::move(cache.keys);
        std::vector<float> old_values = std::move(cache.values);
        cache.keys.assign(static_cast<size_t>(target_capacity) * cache.columns, 0.0f);
        cache.values.assign(static_cast<size_t>(target_capacity) * cache.columns, 0.0f);
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index) {
            const uint64_t old_slot = (cache.first_slot + token_index) % previous_capacity;
            std::copy_n(
                old_keys.data() + old_slot * cache.columns,
                cache.columns,
                cache.keys.data() + token_index * cache.columns);
            std::copy_n(
                old_values.data() + old_slot * cache.columns,
                cache.columns,
                cache.values.data() + token_index * cache.columns);
        }
    }
    cache.first_slot = 0;
    cache.capacity_tokens = target_capacity;
}

static void append_cache(
    CpuLayerCache& cache,
    DType dtype,
    const CpuBatch& key,
    const CpuBatch& value)
{
    assert(key.columns() == value.columns());
    configure_cache(cache, key.columns(), dtype);
    resize_cache(cache, cache.token_count + key.rows());
    for (size_t token_index = 0; token_index < key.rows(); ++token_index) {
        const uint64_t slot = cache_slot(cache, cache.token_count);
        if (cache.dtype == DType::BFloat16) {
            uint16_t* key_destination = cache.bfloat16_keys.data() + slot * cache.columns;
            uint16_t* value_destination = cache.bfloat16_values.data() + slot * cache.columns;
            for (uint32_t column = 0; column < cache.columns; ++column) {
                key_destination[column] = float_to_bfloat16(key.row(token_index)[column]);
                value_destination[column] = float_to_bfloat16(value.row(token_index)[column]);
            }
        }
        else {
            std::copy_n(key.row(token_index), cache.columns, cache.keys.data() + slot * cache.columns);
            std::copy_n(value.row(token_index), cache.columns, cache.values.data() + slot * cache.columns);
        }
        ++cache.token_count;
    }
}

static float cache_element(
    const CpuLayerCache& cache,
    bool key,
    uint64_t token_index,
    uint32_t column)
{
    const uint64_t slot = cache_slot(cache, token_index);
    const size_t offset = static_cast<size_t>(slot) * cache.columns + column;
    if (cache.dtype == DType::BFloat16)
        return bfloat16_to_float(key ? cache.bfloat16_keys[offset] : cache.bfloat16_values[offset]);
    return key ? cache.keys[offset] : cache.values[offset];
}

static CpuBatch scaled_dot_product_attention(
    const AttentionBlockPlan& plan,
    const TensorData* sinks,
    uint64_t position_offset,
    const CpuBatch& query,
    const CpuLayerCache& cache)
{
    const uint32_t head_count = plan.head_count;
    const uint32_t kv_head_count = plan.kv_head_count;
    const uint32_t head_dimension = plan.head_dimension;
    const uint32_t heads_per_group = head_count / kv_head_count;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
    CpuBatch output(query.rows(), head_count * head_dimension);
    std::vector<float> logits(cache.token_count);

    for (size_t query_index = 0; query_index < query.rows(); ++query_index) {
        const uint64_t query_position = position_offset + query_index;
        for (uint32_t query_head = 0; query_head < head_count; ++query_head) {
            const uint32_t kv_head = query_head / heads_per_group;
            const float* query_vector = query.row(query_index) + query_head * head_dimension;
            float maximum = -std::numeric_limits<float>::infinity();
            if (sinks) {
                maximum = sinks->dtype == DType::Float32
                              ? sinks->float32_values()[query_head]
                              : bfloat16_to_float(sinks->bfloat16_values()[query_head]);
            }

            for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index) {
                const uint64_t key_position = cache.start_position + key_index;
                const bool future = key_position > query_position;
                const bool too_old = plan.sliding_window > 0
                                     && key_position + plan.sliding_window <= query_position;
                if (future || too_old) {
                    logits[key_index] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float dot = 0.0f;
                for (uint32_t column = 0; column < head_dimension; ++column)
                    dot += query_vector[column]
                           * cache_element(cache, true, key_index, kv_head * head_dimension + column);
                logits[key_index] = dot * scale;
                maximum = std::max(maximum, logits[key_index]);
            }

            float normalizer = 0.0f;
            if (sinks) {
                const float sink_value = sinks->dtype == DType::Float32
                                             ? sinks->float32_values()[query_head]
                                             : bfloat16_to_float(sinks->bfloat16_values()[query_head]);
                normalizer = std::exp(sink_value - maximum);
            }
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
                for (uint32_t column = 0; column < head_dimension; ++column)
                    output_vector[column] += probability
                                             * cache_element(cache, false, key_index, kv_head * head_dimension + column);
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
    cache.first_slot = (cache.first_slot + removed_tokens) % cache.capacity_tokens;
    cache.start_position += removed_tokens;
    cache.token_count = retained_tokens;
    const uint64_t target_capacity = std::max<uint64_t>(16, retained_tokens * 2);
    if (cache.capacity_tokens > target_capacity * 4)
        compact_cache(cache, target_capacity);
}

static CpuBatch attention_linear(
    const WeightTable& weights,
    TensorHandle matrix,
    TensorHandle bias,
    const CpuBatch& input)
{
    return bias == invalid_tensor_handle
               ? linear_batch(weights.at(matrix), input)
               : linear_batch(weights.at(matrix), weights.at(bias), input);
}

CpuBatch execute_attention_block(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    DType kv_cache_dtype,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& hidden)
{
    CpuBatch vulkan_output;
    if (plan.vulkan_attention_operator
        && plan.vulkan_attention_operator->forward(
            position_offset, cache, hidden, vulkan_output))
        return vulkan_output;

    const CpuBatch normalized = rms_norm_batch(hidden, weights.at(plan.pre_attention_norm_weight), norm_epsilon);
    CpuBatch query;
    CpuBatch key;
    CpuBatch value;
    CpuBatch fused_qkv;
    if (plan.fused_qkv_operator
        && plan.fused_qkv_operator->forward(normalized, fused_qkv)) {
        const uint32_t query_columns = plan.head_count * plan.head_dimension;
        const uint32_t key_value_columns = plan.kv_head_count * plan.head_dimension;
        query = CpuBatch(hidden.rows(), query_columns);
        key = CpuBatch(hidden.rows(), key_value_columns);
        value = CpuBatch(hidden.rows(), key_value_columns);
        for (size_t token_index = 0; token_index < hidden.rows(); ++token_index) {
            const float* source = fused_qkv.row(token_index);
            std::copy_n(source, query_columns, query.row(token_index));
            source += query_columns;
            std::copy_n(source, key_value_columns, key.row(token_index));
            source += key_value_columns;
            std::copy_n(source, key_value_columns, value.row(token_index));
        }
    }
    else {
        query = attention_linear(
            weights, plan.query_weight, plan.query_bias, normalized);
        key = attention_linear(
            weights, plan.key_weight, plan.key_bias, normalized);
        value = attention_linear(
            weights, plan.value_weight, plan.value_bias, normalized);
    }

    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index) {
        const uint64_t position = position_offset + token_index;
        for (uint32_t head = 0; head < plan.head_count; ++head)
            apply_rope(query.row(token_index) + head * plan.head_dimension, plan.head_dimension, position, plan);
        for (uint32_t head = 0; head < plan.kv_head_count; ++head)
            apply_rope(key.row(token_index) + head * plan.head_dimension, plan.head_dimension, position, plan);
    }

    if (cache.token_count == 0)
        cache.start_position = position_offset;
    append_cache(cache, kv_cache_dtype, key, value);
    const CpuBatch attention = scaled_dot_product_attention(
        plan,
        plan.sinks == invalid_tensor_handle ? nullptr : &weights.at(plan.sinks),
        position_offset,
        query,
        cache);
    CpuBatch projected = attention_linear(
        weights, plan.output_weight, plan.output_bias, attention);
    CpuBatch output = hidden;
    add_batch_inplace(output, projected);
    trim_sliding_cache(cache, plan);
    return output;
}

} // namespace moe
} // namespace ncnn
