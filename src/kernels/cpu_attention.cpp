#include "cpu_attention.h"

#include "cpu_ops.h"
#include "cpu_bfloat16.h"
#include "cpu_state_cache.h"
#include "cpu_vector.h"
#include "backends/ncnn/ncnn_attention.h"
#include "backends/ncnn/ncnn_linear.h"
#include "ncnn/moe/runtime_config.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

namespace ncnn {
namespace moe {

static void apply_rope(float* vector, uint32_t dimension, uint64_t position, const AttentionBlockPlan& plan)
{
    const uint32_t half_dimension = dimension / 2;
    const float base = plan.rope_theta;
    const float factor = plan.rope_scaling_factor;
    float concentration = 1.0f;
    float low = 0.0f;
    float high = 0.0f;
    if (factor > 1.0f)
    {
        concentration = 0.1f * std::log(factor) + 1.0f;
        const float half = static_cast<float>(half_dimension);
        low = half * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_beta * 2.0f * std::numbers::pi_v<float>)) / std::log(base);
        high = half * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_alpha * 2.0f * std::numbers::pi_v<float>)) / std::log(base);
    }

    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const float frequency = std::pow(base, static_cast<float>(2 * index) / static_cast<float>(dimension));
        float inverse_frequency = 1.0f / frequency;
        if (factor > 1.0f)
        {
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

static bool cached_rope_coefficients_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags, RuntimeOptimizationCpuRopeCache);
}

static void prepare_rope_coefficients(
    uint32_t dimension,
    uint64_t position,
    const AttentionBlockPlan& plan,
    std::vector<float>& cosine,
    std::vector<float>& sine)
{
    const uint32_t half_dimension = dimension / 2;
    cosine.resize(half_dimension);
    sine.resize(half_dimension);
    const float base = plan.rope_theta;
    const float factor = plan.rope_scaling_factor;
    float concentration = 1.0f;
    float low = 0.0f;
    float high = 0.0f;
    if (factor > 1.0f)
    {
        concentration = 0.1f * std::log(factor) + 1.0f;
        const float half = static_cast<float>(half_dimension);
        low = half * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_beta * 2.0f * std::numbers::pi_v<float>)) / std::log(base);
        high = half * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_alpha * 2.0f * std::numbers::pi_v<float>)) / std::log(base);
    }
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const float frequency = std::pow(base, static_cast<float>(2 * index) / static_cast<float>(dimension));
        float inverse_frequency = 1.0f / frequency;
        if (factor > 1.0f)
        {
            const float ramp = std::clamp((static_cast<float>(index) - low) / (high - low), 0.0f, 1.0f);
            const float mask = 1.0f - ramp;
            const float interpolation = 1.0f / (factor * frequency);
            inverse_frequency = interpolation * (1.0f - mask) + inverse_frequency * mask;
        }
        const float angle = static_cast<float>(position) * inverse_frequency;
        cosine[index] = std::cos(angle) * concentration;
        sine[index] = std::sin(angle) * concentration;
    }
}

static void apply_prepared_rope(
    float* vector,
    uint32_t dimension,
    const std::vector<float>& cosine,
    const std::vector<float>& sine)
{
    const uint32_t half_dimension = dimension / 2;
    assert(cosine.size() >= half_dimension && sine.size() >= half_dimension);
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const float first = vector[index];
        const float second = vector[half_dimension + index];
        vector[index] = first * cosine[index] - second * sine[index];
        vector[half_dimension + index] = second * cosine[index] + first * sine[index];
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
    while (capacity < required)
    {
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
    if (cache.dtype == DType::BFloat16)
    {
        std::vector<uint16_t> keys(element_count);
        std::vector<uint16_t> values(element_count);
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index)
        {
            const uint64_t old_slot = cache_slot(cache, token_index);
            std::copy_n(cache.bfloat16_keys.data() + old_slot * cache.columns, cache.columns, keys.data() + token_index * cache.columns);
            std::copy_n(cache.bfloat16_values.data() + old_slot * cache.columns, cache.columns, values.data() + token_index * cache.columns);
        }
        cache.bfloat16_keys = std::move(keys);
        cache.bfloat16_values = std::move(values);
    }
    else
    {
        std::vector<float> keys(element_count);
        std::vector<float> values(element_count);
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index)
        {
            const uint64_t old_slot = cache_slot(cache, token_index);
            std::copy_n(cache.keys.data() + old_slot * cache.columns, cache.columns, keys.data() + token_index * cache.columns);
            std::copy_n(cache.values.data() + old_slot * cache.columns, cache.columns, values.data() + token_index * cache.columns);
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
    if (cache.dtype == DType::BFloat16)
    {
        std::vector<uint16_t> old_keys = std::move(cache.bfloat16_keys);
        std::vector<uint16_t> old_values = std::move(cache.bfloat16_values);
        cache.bfloat16_keys.assign(static_cast<size_t>(target_capacity) * cache.columns, 0);
        cache.bfloat16_values.assign(static_cast<size_t>(target_capacity) * cache.columns, 0);
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index)
        {
            const uint64_t old_slot = (cache.first_slot + token_index) % previous_capacity;
            std::copy_n(old_keys.data() + old_slot * cache.columns, cache.columns, cache.bfloat16_keys.data() + token_index * cache.columns);
            std::copy_n(old_values.data() + old_slot * cache.columns, cache.columns, cache.bfloat16_values.data() + token_index * cache.columns);
        }
    }
    else
    {
        std::vector<float> old_keys = std::move(cache.keys);
        std::vector<float> old_values = std::move(cache.values);
        cache.keys.assign(static_cast<size_t>(target_capacity) * cache.columns, 0.0f);
        cache.values.assign(static_cast<size_t>(target_capacity) * cache.columns, 0.0f);
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index)
        {
            const uint64_t old_slot = (cache.first_slot + token_index) % previous_capacity;
            std::copy_n(old_keys.data() + old_slot * cache.columns, cache.columns, cache.keys.data() + token_index * cache.columns);
            std::copy_n(old_values.data() + old_slot * cache.columns, cache.columns, cache.values.data() + token_index * cache.columns);
        }
    }
    cache.first_slot = 0;
    cache.capacity_tokens = target_capacity;
}

static void append_cache(CpuLayerCache& cache, DType dtype, const CpuBatch& key, const CpuBatch& value)
{
    assert(key.columns() == value.columns());
    configure_cache(cache, key.columns(), dtype);
    resize_cache(cache, cache.token_count + key.rows());
    for (size_t token_index = 0; token_index < key.rows(); ++token_index)
    {
        const uint64_t slot = cache_slot(cache, cache.token_count);
        if (cache.dtype == DType::BFloat16)
        {
            uint16_t* key_destination = cache.bfloat16_keys.data() + slot * cache.columns;
            uint16_t* value_destination = cache.bfloat16_values.data() + slot * cache.columns;
            float_to_bfloat16_array(
                key_destination,
                key.row(token_index),
                cache.columns);
            float_to_bfloat16_array(
                value_destination,
                value.row(token_index),
                cache.columns);
        }
        else
        {
            std::copy_n(key.row(token_index), cache.columns, cache.keys.data() + slot * cache.columns);
            std::copy_n(value.row(token_index), cache.columns, cache.values.data() + slot * cache.columns);
        }
        ++cache.token_count;
    }
    record_standard_cache_transaction_rows(cache, key.rows());
}

static bool direct_bfloat16_attention_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags, RuntimeOptimizationCpuBf16DirectAttention);
}

static bool bfloat16_attention_pair_dot_enabled(uint64_t optimization_flags) noexcept
{
    return bfloat16_pair_dot_available()
           && runtime_optimization_enabled(optimization_flags,
               RuntimeOptimizationCpuBf16AttentionDot);
}

static void scaled_dot_product_attention_into(const AttentionBlockPlan& plan, const TensorData* sinks, uint64_t position_offset, const CpuBatch& query,
                                              const CpuLayerCache& cache, CpuBatch& output, std::vector<float>& logits,
                                              std::vector<float>& key_cache, std::vector<float>& value_cache,
                                              std::vector<uint16_t>& query_bfloat16,
                                              uint64_t optimization_flags)
{
    const uint32_t head_count = plan.head_count;
    const uint32_t kv_head_count = plan.kv_head_count;
    const uint32_t head_dimension = plan.head_dimension;
    const uint32_t heads_per_group = head_count / kv_head_count;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
    output.reset(query.rows(), head_count * head_dimension, true);
    logits.resize(cache.token_count);

    const size_t cache_elements = static_cast<size_t>(cache.token_count) * cache.columns;
    const size_t cache_capacity_elements = static_cast<size_t>(cache.capacity_tokens) * cache.columns;
    const float* key_values = nullptr;
    const float* value_values = nullptr;
    const uint16_t* bfloat16_key_values = nullptr;
    const uint16_t* bfloat16_value_values = nullptr;
    const bool direct_bfloat16 = query.rows() == 1
                                 && direct_bfloat16_attention_enabled(optimization_flags)
                                 && cache.dtype == DType::BFloat16
                                 && cache.capacity_tokens > 0
                                 && cache.bfloat16_keys.size() >= cache_capacity_elements
                                 && cache.bfloat16_values.size() >= cache_capacity_elements;
    const bool direct_bfloat16_contiguous =
        direct_bfloat16
        && cache.first_slot <= cache.capacity_tokens
        && cache.token_count <= cache.capacity_tokens - cache.first_slot;
    // Decode appends the current token before SDPA.  When the cache ends at
    // that token, no key can be future; retaining at most one sliding window
    // also proves that no key is too old.  Prefill and wrapped/transactional
    // states deliberately keep the fully guarded path below.
    const bool decode_all_keys_valid =
        query.rows() == 1
        && cache.token_count != 0
        && cache.start_position <= position_offset
        && cache.token_count - 1 <= position_offset - cache.start_position
        && (plan.sliding_window == 0
            || cache.token_count <= plan.sliding_window);
    const bool pair_dot = direct_bfloat16
                          && bfloat16_attention_pair_dot_enabled(optimization_flags);
    if (pair_dot)
        query_bfloat16.resize(head_dimension);
    if (direct_bfloat16)
    {
        bfloat16_key_values = cache.bfloat16_keys.data();
        bfloat16_value_values = cache.bfloat16_values.data();
    }
    else if (cache.dtype == DType::Float32 && cache.first_slot == 0
        && cache.keys.size() >= cache_elements
        && cache.values.size() >= cache_elements)
    {
        key_values = cache.keys.data();
        value_values = cache.values.data();
    }
    else
    {
        key_cache.resize(cache_elements);
        value_cache.resize(cache_elements);
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index)
        {
            const uint64_t slot = cache_slot(cache, token_index);
            float* key_destination =
                key_cache.data() + static_cast<size_t>(token_index) * cache.columns;
            float* value_destination =
                value_cache.data() + static_cast<size_t>(token_index) * cache.columns;
            if (cache.dtype == DType::BFloat16)
            {
                const uint16_t* key_source =
                    cache.bfloat16_keys.data() + static_cast<size_t>(slot) * cache.columns;
                const uint16_t* value_source =
                    cache.bfloat16_values.data() + static_cast<size_t>(slot) * cache.columns;
                for (uint32_t column = 0; column < cache.columns; ++column)
                {
                    key_destination[column] = bfloat16_to_float(key_source[column]);
                    value_destination[column] = bfloat16_to_float(value_source[column]);
                }
            }
            else
            {
                std::copy_n(
                    cache.keys.data() + static_cast<size_t>(slot) * cache.columns,
                    cache.columns,
                    key_destination);
                std::copy_n(
                    cache.values.data() + static_cast<size_t>(slot) * cache.columns,
                    cache.columns,
                    value_destination);
            }
        }
        key_values = key_cache.data();
        value_values = value_cache.data();
    }

    for (size_t query_index = 0; query_index < query.rows(); ++query_index)
    {
        const uint64_t query_position = position_offset + query_index;
        for (uint32_t query_head = 0; query_head < head_count; ++query_head)
        {
            const uint32_t kv_head = query_head / heads_per_group;
            const float* query_vector = query.row(query_index) + query_head * head_dimension;
            if (pair_dot)
            {
                float_to_bfloat16_array(
                    query_bfloat16.data(),
                    query_vector,
                    head_dimension);
            }
            float maximum = -std::numeric_limits<float>::infinity();
            if (sinks)
            {
                maximum = sinks->dtype == DType::Float32 ? sinks->float32_values()[query_head] : bfloat16_to_float(sinks->bfloat16_values()[query_head]);
            }

            if (decode_all_keys_valid)
            {
                for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index)
                {
                    float dot = 0.0f;
                    if (direct_bfloat16)
                    {
                        const uint64_t slot = direct_bfloat16_contiguous
                                                  ? cache.first_slot + key_index
                                                  : cache_slot(cache, key_index);
                        const uint16_t* key_vector =
                            bfloat16_key_values + static_cast<size_t>(slot) * cache.columns
                            + static_cast<size_t>(kv_head) * head_dimension;
                        dot = pair_dot
                                  ? bfloat16_pair_dot(
                                        key_vector,
                                        query_bfloat16.data(),
                                        head_dimension)
                                  : bfloat16_dot(
                                        key_vector,
                                        query_vector,
                                        head_dimension);
                    }
                    else
                    {
                        const float* key_vector =
                            key_values + static_cast<size_t>(key_index) * cache.columns
                            + static_cast<size_t>(kv_head) * head_dimension;
                        dot = float_dot(query_vector, key_vector, head_dimension);
                    }
                    logits[key_index] = dot * scale;
                    maximum = std::max(maximum, logits[key_index]);
                }
            }
            else
            {
                for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index)
                {
                    const uint64_t key_position = cache.start_position + key_index;
                    const bool future = key_position > query_position;
                    const bool too_old = plan.sliding_window > 0 && key_position + plan.sliding_window <= query_position;
                    if (future || too_old)
                    {
                        logits[key_index] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    float dot = 0.0f;
                    if (direct_bfloat16)
                    {
                        const uint64_t slot = direct_bfloat16_contiguous
                                                  ? cache.first_slot + key_index
                                                  : cache_slot(cache, key_index);
                        const uint16_t* key_vector =
                            bfloat16_key_values + static_cast<size_t>(slot) * cache.columns
                            + static_cast<size_t>(kv_head) * head_dimension;
                        dot = pair_dot
                            ? bfloat16_pair_dot(
                                  key_vector,
                                  query_bfloat16.data(),
                                  head_dimension)
                            : bfloat16_dot(
                                  key_vector,
                                  query_vector,
                                  head_dimension);
                    }
                    else
                    {
                        const float* key_vector =
                            key_values + static_cast<size_t>(key_index) * cache.columns
                            + static_cast<size_t>(kv_head) * head_dimension;
                        dot = float_dot(query_vector, key_vector, head_dimension);
                    }
                    logits[key_index] = dot * scale;
                    maximum = std::max(maximum, logits[key_index]);
                }
            }

            float normalizer = 0.0f;
            if (sinks)
            {
                const float sink_value = sinks->dtype == DType::Float32 ? sinks->float32_values()[query_head] : bfloat16_to_float(sinks->bfloat16_values()[query_head]);
                normalizer = std::exp(sink_value - maximum);
            }
            for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index)
            {
                if (std::isfinite(logits[key_index]))
                {
                    logits[key_index] = std::exp(logits[key_index] - maximum);
                    normalizer += logits[key_index];
                }
                else
                {
                    logits[key_index] = 0.0f;
                }
            }

            float* output_vector = output.row(query_index) + query_head * head_dimension;
            for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index)
            {
                const float probability = logits[key_index] / normalizer;
                if (direct_bfloat16)
                {
                    const uint64_t slot = direct_bfloat16_contiguous
                                              ? cache.first_slot + key_index
                                              : cache_slot(cache, key_index);
                    const uint16_t* value_vector =
                        bfloat16_value_values + static_cast<size_t>(slot) * cache.columns
                        + static_cast<size_t>(kv_head) * head_dimension;
                    bfloat16_scaled_add(output_vector, value_vector, probability, head_dimension);
                }
                else
                {
                    const float* value_vector =
                        value_values + static_cast<size_t>(key_index) * cache.columns
                        + static_cast<size_t>(kv_head) * head_dimension;
                    float_scaled_add(output_vector, value_vector, probability, head_dimension);
                }
            }
        }
    }
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

static void attention_linear_into(const WeightStore& weights, const CompiledOperatorTable& operators, TensorHandle matrix, TensorHandle bias, const CpuBatch& input, CpuBatch& output, uint64_t optimization_flags)
{
    if (bias == invalid_tensor_handle)
    {
        linear_batch_into(weights.at(matrix), input, output, optimization_flags, &operators.at_weight(matrix));
    }
    else
    {
        linear_batch_into(weights.at(matrix), weights.at(bias), input, output, optimization_flags, &operators.at_weight(matrix));
    }
}

static float attention_weight_value(const TensorData& tensor, size_t index)
{
    if (tensor.dtype == DType::Float32)
        return tensor.float32_values()[index];
    return bfloat16_to_float(tensor.bfloat16_values()[index]);
}

static void apply_head_rms_norm(CpuBatch& batch, uint32_t head_count, uint32_t head_dimension, const TensorData& weight, float epsilon, float weight_offset, uint64_t optimization_flags)
{
    assert(weight.element_count() == head_dimension);
    for (size_t token_index = 0; token_index < batch.rows(); ++token_index)
    {
        float* token = batch.row(token_index);
        for (uint32_t head = 0; head < head_count; ++head)
        {
            float* values = token + head * head_dimension;
            const float square_sum = float_dot(values, values, head_dimension);
            const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(head_dimension) + epsilon);
            if (simd_rms_norm_enabled(optimization_flags)
                && (weight.dtype == DType::Float32
                    || weight.dtype == DType::BFloat16))
            {
                if (weight.dtype == DType::Float32)
                {
                    float_weighted_scale(
                        values,
                        values,
                        weight.float32_values().data(),
                        inverse_rms,
                        weight_offset,
                        head_dimension);
                }
                else
                {
                    bfloat16_weighted_scale(
                        values,
                        values,
                        weight.bfloat16_values().data(),
                        inverse_rms,
                        weight_offset,
                        head_dimension);
                }
                continue;
            }
            for (uint32_t column = 0; column < head_dimension; ++column)
            {
                values[column] *= inverse_rms * (attention_weight_value(weight, column) + weight_offset);
            }
        }
    }
}

Result<void> append_attention_context_into(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    DType kv_cache_dtype,
    uint64_t position_offset,
    CpuLayerCache& cache,
    CpuAttentionExecutionScratch& scratch,
    const CpuBatch& hidden,
    uint64_t optimization_flags)
{
    if (cache.vulkan_attention_cache)
    {
        const CompiledOperator& attention_operator =
            operators.at(plan.vulkan_attention_operator);
        if (cache.vulkan_attention_state_unknown
            || !attention_operator.attention
            || !attention_operator.attention->materialize_device_cache(cache))
        {
            return Error{
                ErrorCode::InternalError,
                "cannot materialize Vulkan KV cache for CPU context append"};
        }
    }
    if (cache.transaction.active && plan.sliding_window != 0)
    {
        return Error{
            ErrorCode::UnsupportedModel,
            "state cache transaction does not support sliding Attention"};
    }

    rms_norm_batch_into(hidden, weights.at(plan.pre_attention_norm_weight), norm_epsilon, scratch.normalized, plan.norm_weight_offset, optimization_flags);
    CpuBatch& key = scratch.key;
    CpuBatch& value = scratch.value;
    CpuBatch& fused_qkv = scratch.fused_qkv;
    const CompiledOperator& fused_qkv_gate_operator = operators.at(plan.fused_qkv_gate_bfloat16_operator);
    const CompiledOperator& fused_qkv_bfloat16_operator = operators.at(plan.fused_qkv_bfloat16_operator);
    const CompiledOperator& fused_qkv_linear_operator = operators.at(plan.fused_qkv_operator);
    if (backend == ExecutionBackend::Vulkan
        && ((fused_qkv_gate_operator.bfloat16
         && fused_qkv_gate_operator.bfloat16->forward(
             scratch.normalized,
             fused_qkv))
        || (fused_qkv_bfloat16_operator.bfloat16
         && fused_qkv_bfloat16_operator.bfloat16->forward(
             scratch.normalized,
             fused_qkv))
         || (fused_qkv_linear_operator.linear
             && fused_qkv_linear_operator.linear->forward(
                 scratch.normalized,
                 fused_qkv))))
    {
        const uint32_t query_columns = plan.head_count * plan.head_dimension;
        const uint32_t key_value_columns = plan.kv_head_count * plan.head_dimension;
        key.reset(hidden.rows(), key_value_columns, false);
        value.reset(hidden.rows(), key_value_columns, false);
        for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
        {
            const float* source = fused_qkv.row(token_index) + query_columns;
            std::copy_n(source, key_value_columns, key.row(token_index));
            source += key_value_columns;
            std::copy_n(source, key_value_columns, value.row(token_index));
        }
    }
    else
    {
        attention_linear_into(weights, operators, plan.key_weight, plan.key_bias, scratch.normalized, key, optimization_flags);
        attention_linear_into(weights, operators, plan.value_weight, plan.value_bias, scratch.normalized, value, optimization_flags);
    }

    if (has_flag(plan.flags, AttentionBlockQueryKeyNorm))
    {
        apply_head_rms_norm(key, plan.kv_head_count, plan.head_dimension, weights.at(plan.key_norm_weight), norm_epsilon, plan.norm_weight_offset, optimization_flags);
    }

    const uint32_t rope_dimension = plan.rope_head_dimension == 0 ? plan.head_dimension : plan.rope_head_dimension;
    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
    {
        const uint64_t position = position_offset + token_index;
        if (cached_rope_coefficients_enabled(optimization_flags))
        {
            prepare_rope_coefficients(
                rope_dimension,
                position,
                plan,
                scratch.rope_cosine,
                scratch.rope_sine);
            for (uint32_t head = 0; head < plan.kv_head_count; ++head)
            {
                apply_prepared_rope(
                    key.row(token_index) + head * plan.head_dimension,
                    rope_dimension,
                    scratch.rope_cosine,
                    scratch.rope_sine);
            }
        }
        else
        {
            for (uint32_t head = 0; head < plan.kv_head_count; ++head)
                apply_rope(key.row(token_index) + head * plan.head_dimension, rope_dimension, position, plan);
        }
    }

    if (cache.token_count == 0)
        cache.start_position = position_offset;
    append_cache(cache, kv_cache_dtype, key, value);
    trim_sliding_cache(cache, plan);
    return {};
}

Result<bool> execute_attention_block_batch_into(
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    std::span<CpuAttentionBatchEntry> entries,
    uint64_t optimization_flags)
{
    (void)optimization_flags;
    const CompiledOperator& attention_operator = operators.at(plan.vulkan_attention_operator);
    if (backend != ExecutionBackend::Vulkan
        || entries.size() < 2
        || !attention_operator.attention)
        return false;

    std::vector<NcnnVulkanAttentionBatchEntry> device_entries;
    device_entries.reserve(entries.size());
    for (CpuAttentionBatchEntry& entry : entries)
    {
        if (!entry.cache || !entry.scratch || !entry.hidden || !entry.output)
        {
            return Error{
                ErrorCode::InvalidArgument,
                "Attention batch entry is incomplete"};
        }
        if (entry.cache->transaction.active)
            return false;
        device_entries.push_back({
            entry.position_offset,
            entry.cache,
            entry.hidden,
            entry.output});
    }

    const NcnnVulkanAttentionBatchResult result =
        attention_operator.attention->forward_batch(device_entries);
    if (result == NcnnVulkanAttentionBatchResult::Executed)
        return true;
    if (result == NcnnVulkanAttentionBatchResult::Failed)
    {
        return Error{
            ErrorCode::InternalError,
            "Vulkan Attention batch failed after device KV state became authoritative"};
    }
    return false;
}

Result<void> execute_attention_block_into(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    DType kv_cache_dtype,
    uint64_t position_offset,
    CpuLayerCache& cache,
    CpuAttentionExecutionScratch& scratch,
    const CpuBatch& hidden,
    CpuBatch& output,
    uint64_t optimization_flags)
{
    if (cache.vulkan_attention_state_unknown)
    {
        return Error{
            ErrorCode::InternalError,
            "Attention state is unavailable after a failed Vulkan update"};
    }
    if (cache.transaction.active && plan.sliding_window != 0)
    {
        return Error{
            ErrorCode::UnsupportedModel,
            "state cache transaction does not support sliding Attention"};
    }

    const CompiledOperator& attention_operator = operators.at(plan.vulkan_attention_operator);
    if (backend == ExecutionBackend::Vulkan && attention_operator.attention)
    {
        if (attention_operator.attention->forward(
                position_offset,
                cache,
                hidden,
                output))
        {
            return {};
        }
        attention_operator.attention->record_cpu_fallback();
        if (cache.vulkan_attention_cache)
        {
            if (!cache.vulkan_attention_state_unknown
                && attention_operator.attention->materialize_device_cache(
                    cache))
            {
                // The failed dispatch was building a separate ring or failed
                // before mutating the old one.  The operator has restored the
                // authoritative rows to the CPU cache; continue through the
                // existing CPU Attention implementation below.
            }
            else
            {
                return Error{
                    ErrorCode::InternalError,
                    "Vulkan Attention failed and its device KV cache could not be materialized"};
            }
        }
    }

    rms_norm_batch_into(hidden, weights.at(plan.pre_attention_norm_weight), norm_epsilon, scratch.normalized, plan.norm_weight_offset, optimization_flags);
    CpuBatch& query = scratch.query;
    CpuBatch& key = scratch.key;
    CpuBatch& value = scratch.value;
    CpuBatch& fused_qkv = scratch.fused_qkv;
    const CompiledOperator& fused_qkv_gate_operator = operators.at(plan.fused_qkv_gate_bfloat16_operator);
    const CompiledOperator& fused_qkv_bfloat16_operator = operators.at(plan.fused_qkv_bfloat16_operator);
    const CompiledOperator& fused_qkv_linear_operator = operators.at(plan.fused_qkv_operator);
    const bool fused_output_gate =
        backend == ExecutionBackend::Vulkan
        && fused_qkv_gate_operator.bfloat16
        && fused_qkv_gate_operator.bfloat16->forward(
            scratch.normalized,
            fused_qkv);
    const bool fused_projection =
        fused_output_gate
        || (backend == ExecutionBackend::Vulkan
            && fused_qkv_bfloat16_operator.bfloat16
            && fused_qkv_bfloat16_operator.bfloat16->forward(
                scratch.normalized,
                fused_qkv))
        || (backend == ExecutionBackend::Vulkan
            && fused_qkv_linear_operator.linear
            && fused_qkv_linear_operator.linear->forward(
                scratch.normalized,
                fused_qkv));
    if (fused_projection)
    {
        const uint32_t query_columns = plan.head_count * plan.head_dimension;
        const uint32_t key_value_columns = plan.kv_head_count * plan.head_dimension;
        query.reset(hidden.rows(), query_columns, false);
        key.reset(hidden.rows(), key_value_columns, false);
        value.reset(hidden.rows(), key_value_columns, false);
        if (fused_output_gate)
            scratch.gate.reset(hidden.rows(), query_columns, false);
        for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
        {
            const float* source = fused_qkv.row(token_index);
            std::copy_n(source, query_columns, query.row(token_index));
            source += query_columns;
            std::copy_n(source, key_value_columns, key.row(token_index));
            source += key_value_columns;
            std::copy_n(source, key_value_columns, value.row(token_index));
            source += key_value_columns;
            if (fused_output_gate)
            {
                std::copy_n(
                    source,
                    query_columns,
                    scratch.gate.row(token_index));
            }
        }
    }
    else
    {
        attention_linear_into(weights, operators, plan.query_weight, plan.query_bias, scratch.normalized, query, optimization_flags);
        attention_linear_into(weights, operators, plan.key_weight, plan.key_bias, scratch.normalized, key, optimization_flags);
        attention_linear_into(weights, operators, plan.value_weight, plan.value_bias, scratch.normalized, value, optimization_flags);
    }

    if (has_flag(plan.flags, AttentionBlockQueryKeyNorm))
    {
        apply_head_rms_norm(query, plan.head_count, plan.head_dimension, weights.at(plan.query_norm_weight), norm_epsilon, plan.norm_weight_offset, optimization_flags);
        apply_head_rms_norm(key, plan.kv_head_count, plan.head_dimension, weights.at(plan.key_norm_weight), norm_epsilon, plan.norm_weight_offset, optimization_flags);
    }

    const uint32_t rope_dimension = plan.rope_head_dimension == 0 ? plan.head_dimension : plan.rope_head_dimension;
    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
    {
        const uint64_t position = position_offset + token_index;
        if (cached_rope_coefficients_enabled(optimization_flags))
        {
            prepare_rope_coefficients(
                rope_dimension,
                position,
                plan,
                scratch.rope_cosine,
                scratch.rope_sine);
            for (uint32_t head = 0; head < plan.head_count; ++head)
            {
                apply_prepared_rope(
                    query.row(token_index) + head * plan.head_dimension,
                    rope_dimension,
                    scratch.rope_cosine,
                    scratch.rope_sine);
            }
            for (uint32_t head = 0; head < plan.kv_head_count; ++head)
            {
                apply_prepared_rope(
                    key.row(token_index) + head * plan.head_dimension,
                    rope_dimension,
                    scratch.rope_cosine,
                    scratch.rope_sine);
            }
        }
        else
        {
            for (uint32_t head = 0; head < plan.head_count; ++head)
                apply_rope(query.row(token_index) + head * plan.head_dimension, rope_dimension, position, plan);
            for (uint32_t head = 0; head < plan.kv_head_count; ++head)
                apply_rope(key.row(token_index) + head * plan.head_dimension, rope_dimension, position, plan);
        }
    }

    if (cache.token_count == 0)
        cache.start_position = position_offset;
    append_cache(cache, kv_cache_dtype, key, value);
    scaled_dot_product_attention_into(
        plan,
        plan.sinks == invalid_tensor_handle ? nullptr : &weights.at(plan.sinks),
        position_offset,
        query,
        cache,
        scratch.attention,
        scratch.logits,
        scratch.key_cache,
        scratch.value_cache,
        scratch.query_bfloat16,
        optimization_flags);
    if (has_flag(plan.flags, AttentionBlockOutputGate))
    {
        if (!fused_output_gate)
        {
            attention_linear_into(
                weights,
                operators,
                plan.output_gate_weight,
                invalid_tensor_handle,
                scratch.normalized,
                scratch.gate,
                optimization_flags);
        }
        for (size_t token_index = 0; token_index < scratch.attention.rows(); ++token_index)
        {
            float* attention_row = scratch.attention.row(token_index);
            const float* gate_row = scratch.gate.row(token_index);
            for (uint32_t column = 0; column < scratch.attention.columns(); ++column)
            {
                attention_row[column] *= 1.0f / (1.0f + std::exp(-gate_row[column]));
            }
        }
    }
    attention_linear_into(weights, operators, plan.output_weight, plan.output_bias, scratch.attention, scratch.projected, optimization_flags);
    output = hidden;
    add_batch_inplace(output, scratch.projected);
    trim_sliding_cache(cache, plan);
    return {};
}

} // namespace moe
} // namespace ncnn
