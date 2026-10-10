#include "attention.h"

#include "fastmath.h"
#include "ops.h"
#include "bfloat16.h"
#include "statecache.h"
#include "vector.h"
#include "backends/ncnn/attention_vulkan.h"
#include "backends/ncnn/linear.h"
#include "graph/compiledoperator.h"
#include "ncnn/moe/option.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

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
    return has_flag(optimization_flags, OptimizationCpuRopeCache);
}

static void prepare_rope_coefficients(uint32_t dimension,
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

static void apply_prepared_rope(float* vector,
                                uint32_t dimension,
                                const std::vector<float>& cosine,
                                const std::vector<float>& sine)
{
    assert(cosine.size() >= dimension / 2 && sine.size() >= dimension / 2);
    float_rope_inplace(vector, cosine.data(), sine.data(), dimension);
}

static uint64_t cache_slot(const LayerCache& cache, uint64_t token_index)
{
    assert(cache.capacity_tokens > 0);
    return (cache.first_slot + token_index) % cache.capacity_tokens;
}

static void configure_cache(LayerCache& cache, uint32_t columns, DType dtype)
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

static void resize_cache(LayerCache& cache, uint64_t required_tokens)
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

static void compact_cache(LayerCache& cache, uint64_t target_capacity)
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

static void append_cache(LayerCache& cache, DType dtype, const ActivationBuffer& key, const ActivationBuffer& value)
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
            float_to_bfloat16_array(key_destination,
                                    key.row(token_index),
                                    cache.columns);
            float_to_bfloat16_array(value_destination,
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

static void scaled_dot_product_attention_into(const AttentionBlockPlan& plan, const TensorData* sinks, uint64_t position_offset, const ActivationBuffer& query,
                                              const LayerCache& cache, ActivationBuffer& output, AttentionScratch& scratch,
                                              std::span<const size_t> selected_offsets,
                                              std::span<const uint32_t> selected_indices,
                                              uint64_t optimization_flags)
{
    std::vector<float>& key_cache = scratch.key_cache;
    std::vector<float>& value_cache = scratch.value_cache;
    std::vector<float>& flash_partial_max = scratch.flash_partial_max;
    std::vector<float>& flash_partial_sum = scratch.flash_partial_sum;
    std::vector<float>& flash_partial_output = scratch.flash_partial_output;
    const uint32_t head_count = plan.head_count;
    const uint32_t kv_head_count = plan.kv_head_count;
    const uint32_t head_dimension = plan.head_dimension;
    const uint32_t heads_per_group = head_count / kv_head_count;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
    const bool has_selected_keys = selected_offsets.size() == query.rows() + 1;
    const bool flash_prefill_enabled = has_flag(optimization_flags,
                                                OptimizationCpuFlashAttention)
                                       && query.rows() > 1
                                       && cache.token_count >= 64
                                       && !has_selected_keys;
    output.reset(query.rows(), head_count * head_dimension, !flash_prefill_enabled);
    size_t maximum_selected_keys = 0;
    if (has_selected_keys)
    {
        for (size_t query_index = 0; query_index < query.rows(); ++query_index)
        {
            maximum_selected_keys = std::max(maximum_selected_keys,
                                             selected_offsets[query_index + 1]
                                                 - selected_offsets[query_index]);
        }
    }
    const size_t cache_elements = static_cast<size_t>(cache.token_count) * cache.columns;
    const size_t cache_capacity_elements = static_cast<size_t>(cache.capacity_tokens) * cache.columns;
    const float* key_values = nullptr;
    const float* value_values = nullptr;
    const uint16_t* bfloat16_key_values = nullptr;
    const uint16_t* bfloat16_value_values = nullptr;
    const bool direct_bfloat16 = (query.rows() == 1 || has_selected_keys
                                  || (flash_prefill_enabled
                                      && cache.columns > 0
                                      && cache.capacity_tokens
                                             <= std::numeric_limits<size_t>::max() / cache.columns
                                      && cache.first_slot < cache.capacity_tokens
                                      && cache.token_count <= cache.capacity_tokens))
                                 && has_flag(optimization_flags, OptimizationCpuBf16DirectAttention)
                                 && cache.dtype == DType::BFloat16
                                 && cache.capacity_tokens > 0
                                 && cache.bfloat16_keys.size() >= cache_capacity_elements
                                 && cache.bfloat16_values.size() >= cache_capacity_elements;
    const bool direct_bfloat16_contiguous = direct_bfloat16
                                            && cache.first_slot <= cache.capacity_tokens
                                            && cache.token_count <= cache.capacity_tokens - cache.first_slot;
    const bool direct_bfloat16_ring = flash_prefill_enabled
                                      && direct_bfloat16
                                      && !direct_bfloat16_contiguous;
    const bool direct_float32_contiguous = cache.dtype == DType::Float32
                                           && cache.columns > 0
                                           && cache.capacity_tokens > 0
                                           && cache.first_slot <= cache.capacity_tokens
                                           && cache.token_count <= cache.capacity_tokens - cache.first_slot
                                           && cache.first_slot + cache.token_count
                                                  <= std::min(cache.keys.size(), cache.values.size()) / cache.columns;
    const bool direct_float32_ring = cache.dtype == DType::Float32
                                     && cache.columns > 0
                                     && cache.capacity_tokens <= std::numeric_limits<size_t>::max() / cache.columns
                                     && cache.capacity_tokens > 0
                                     && cache.first_slot < cache.capacity_tokens
                                     && cache.token_count <= cache.capacity_tokens
                                     && !direct_float32_contiguous
                                     && cache.keys.size() >= cache_capacity_elements
                                     && cache.values.size() >= cache_capacity_elements;
    const uint64_t ring_tail = direct_float32_ring || direct_bfloat16_ring
                                   ? cache.capacity_tokens - cache.first_slot
                                   : 0;
    const auto float32_vector = [&](const float* values, uint64_t key_index, uint32_t kv_head) {
        if (direct_float32_ring)
        {
            uint64_t slot = key_index;
            if (slot >= ring_tail)
                slot -= ring_tail;
            else
                slot += cache.first_slot;
            return values + static_cast<size_t>(slot) * cache.columns
                   + static_cast<size_t>(kv_head) * head_dimension;
        }
        return values + static_cast<size_t>(key_index) * cache.columns
               + static_cast<size_t>(kv_head) * head_dimension;
    };
    // Decode can use the short SDPA path only when the cache has no future key.
    const bool decode_all_keys_valid = query.rows() == 1
                                       && cache.token_count != 0
                                       && cache.start_position <= position_offset
                                       && cache.token_count - 1 <= position_offset - cache.start_position
                                       && (plan.sliding_window == 0
                                           || cache.token_count <= plan.sliding_window)
                                       && !has_selected_keys;
    if (direct_bfloat16)
    {
        bfloat16_key_values = cache.bfloat16_keys.data();
        bfloat16_value_values = cache.bfloat16_values.data();
    }
    if (direct_float32_contiguous || direct_float32_ring)
    {
        if (direct_float32_contiguous)
        {
            const size_t cache_offset = static_cast<size_t>(cache.first_slot) * cache.columns;
            key_values = cache.keys.data() + cache_offset;
            value_values = cache.values.data() + cache_offset;
        }
        else
        {
            key_values = cache.keys.data();
            value_values = cache.values.data();
        }
    }
    else if (!direct_bfloat16 || flash_prefill_enabled)
    {
        if (direct_bfloat16)
        {
            key_cache.clear();
            value_cache.resize(cache_elements);
            for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index)
            {
                const uint64_t slot = cache_slot(cache, token_index);
                const uint16_t* value_source = cache.bfloat16_values.data()
                                               + static_cast<size_t>(slot) * cache.columns;
                float* value_destination = value_cache.data()
                                           + static_cast<size_t>(token_index) * cache.columns;
                for (uint32_t column = 0; column < cache.columns; ++column)
                    value_destination[column] = bfloat16_to_float(value_source[column]);
            }
        }
        else
        {
            key_cache.resize(cache_elements);
            value_cache.resize(cache_elements);
            for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index)
            {
                const uint64_t slot = cache_slot(cache, token_index);
                float* key_destination = key_cache.data() + static_cast<size_t>(token_index) * cache.columns;
                float* value_destination = value_cache.data() + static_cast<size_t>(token_index) * cache.columns;
                if (cache.dtype == DType::BFloat16)
                {
                    const uint16_t* key_source = cache.bfloat16_keys.data()
                                                 + static_cast<size_t>(slot) * cache.columns;
                    const uint16_t* value_source = cache.bfloat16_values.data()
                                                   + static_cast<size_t>(slot) * cache.columns;
                    for (uint32_t column = 0; column < cache.columns; ++column)
                    {
                        key_destination[column] = bfloat16_to_float(key_source[column]);
                        value_destination[column] = bfloat16_to_float(value_source[column]);
                    }
                }
                else
                {
                    std::copy_n(cache.keys.data() + static_cast<size_t>(slot) * cache.columns,
                                cache.columns,
                                key_destination);
                    std::copy_n(cache.values.data() + static_cast<size_t>(slot) * cache.columns,
                                cache.columns,
                                value_destination);
                }
            }
            key_values = key_cache.data();
        }
        value_values = value_cache.data();
    }

    int attention_team_size = 1;
#if defined(_OPENMP)
    if (omp_in_parallel() == 0)
        attention_team_size = static_cast<int>(cpu_linear_num_threads());
#endif
    const uint64_t split_kv_work = static_cast<uint64_t>(head_count)
                                   * static_cast<uint64_t>(head_dimension)
                                   * cache.token_count;
    // Amortize split-KV coordination over enough head-key elements.
    constexpr uint64_t minimum_split_kv_work = 256u * 1024u;
    const bool split_kv_enabled = has_flag(optimization_flags,
                                           OptimizationCpuSplitKvAttention)
                                  && query.rows() == 1
                                  && cache.token_count >= 512
                                  && attention_team_size > 1
                                  && split_kv_work >= minimum_split_kv_work
                                  && !has_selected_keys;

    if (flash_prefill_enabled || split_kv_enabled)
    {
        // Keep adjacent workers off a shared cache line, including 128-byte lines.
        constexpr size_t worker_padding = 128 / sizeof(float);
        const auto valid_key = [&](size_t query_index, uint64_t key_index) {
            if (decode_all_keys_valid)
                return true;
            const uint64_t query_position = position_offset + query_index;
            const uint64_t key_position = cache.start_position + key_index;
            return key_position <= query_position
                   && (plan.sliding_window == 0
                       || key_position + plan.sliding_window > query_position);
        };
        const auto key_dot = [&](uint32_t query_head, const float* query_vector, uint64_t key_index) {
            const uint32_t kv_head = query_head / heads_per_group;
            if (direct_bfloat16)
            {
                const uint64_t slot = direct_bfloat16_contiguous
                                          ? cache.first_slot + key_index
                                          : cache_slot(cache, key_index);
                const uint16_t* key_vector = bfloat16_key_values + static_cast<size_t>(slot) * cache.columns
                                             + static_cast<size_t>(kv_head) * head_dimension;
                return bfloat16_dot(key_vector, query_vector, head_dimension);
            }
            const float* key_vector = float32_vector(key_values, key_index, kv_head);
            return float_dot(query_vector, key_vector, head_dimension);
        };
        const auto add_value = [&](uint32_t query_head, float* destination, float weight, uint64_t key_index) {
            const uint32_t kv_head = query_head / heads_per_group;
            if (direct_bfloat16)
            {
                const uint64_t slot = direct_bfloat16_contiguous
                                          ? cache.first_slot + key_index
                                          : cache_slot(cache, key_index);
                const uint16_t* value_vector = bfloat16_value_values + static_cast<size_t>(slot) * cache.columns
                                               + static_cast<size_t>(kv_head) * head_dimension;
                bfloat16_scaled_add(destination, value_vector, weight, head_dimension);
            }
            else
            {
                const float* value_vector = float32_vector(value_values, key_index, kv_head);
                float_scaled_add(destination, value_vector, weight, head_dimension);
            }
        };
        const auto sink_value = [&](uint32_t query_head) {
            if (!sinks)
                return -std::numeric_limits<float>::infinity();
            return sinks->dtype == DType::Float32
                       ? sinks->float32_values()[query_head]
                       : bfloat16_to_float(sinks->bfloat16_values()[query_head]);
        };

        if (flash_prefill_enabled)
        {
            const uint64_t query_group_count = (query.rows() + 3) / 4;
            const uint64_t job_count = query_group_count * head_count;
            constexpr uint64_t key_tile_size = 64;
            constexpr uint32_t query_tile_size = 4;
            constexpr uint32_t key_gemm_tile_size = 8;
            const auto process_flash_group = [&](uint64_t job,
                                                 float* running,
                                                 float* tile_output,
                                                 float* tile_logits) {
                const size_t query_group = static_cast<size_t>(job / head_count) * query_tile_size;
                const uint32_t query_head = static_cast<uint32_t>(job % head_count);
                const uint32_t query_count = static_cast<uint32_t>(std::min<size_t>(query_tile_size,
                                                                                    query.rows() - query_group));
                const uint64_t last_position = position_offset + query_group + query_count - 1;
                const bool skip_future = last_position >= position_offset
                                         && cache.token_count - 1
                                                <= std::numeric_limits<uint64_t>::max() - cache.start_position;
                const uint32_t kv_head = query_head / heads_per_group;
                const float* query_values = query.row(query_group) + static_cast<size_t>(query_head) * head_dimension;
                std::array<float, query_tile_size> maximum = {};
                std::array<float, query_tile_size> normalizer = {};
                for (uint32_t query_offset = 0; query_offset < query_count; ++query_offset)
                {
                    std::fill(running + static_cast<size_t>(query_offset) * head_dimension,
                              running + static_cast<size_t>(query_offset + 1) * head_dimension,
                              0.0f);
                    maximum[query_offset] = sink_value(query_head);
                    normalizer[query_offset] = sinks ? 1.0f : 0.0f;
                }
                for (uint64_t tile_begin = 0; tile_begin < cache.token_count; tile_begin += key_tile_size)
                {
                    if (skip_future
                        && (cache.start_position > last_position
                            || tile_begin > last_position - cache.start_position))
                        break;
                    const uint64_t tile_end = std::min(cache.token_count, tile_begin + key_tile_size);
                    const uint32_t tile_width = static_cast<uint32_t>(tile_end - tile_begin);
                    std::array<float, query_tile_size> tile_maximum;
                    tile_maximum.fill(-std::numeric_limits<float>::infinity());
                    for (uint64_t key_begin = tile_begin; key_begin < tile_end; key_begin += key_gemm_tile_size)
                    {
                        const uint32_t key_count = static_cast<uint32_t>(std::min<uint64_t>(key_gemm_tile_size,
                                                                                            tile_end - key_begin));
                        const size_t tile_key_offset = static_cast<size_t>(key_begin - tile_begin);
                        const uint16_t* tile_keys_bf16 = nullptr;
                        const float* tile_keys = nullptr;
                        size_t key_stride = cache.columns;
                        if (direct_bfloat16)
                        {
                            const bool crosses_ring = direct_bfloat16_ring
                                                      && key_begin < ring_tail
                                                      && key_count > ring_tail - key_begin;
                            if (crosses_ring)
                            {
                                // Keep the GEMM tile intact: smaller tail kernels can change rounding.
                                float* packed_keys = tile_logits + query_tile_size * key_tile_size;
                                for (uint32_t key_offset = 0; key_offset < key_count; ++key_offset)
                                {
                                    const uint64_t key_index = key_begin + key_offset;
                                    const uint64_t slot = key_index < ring_tail
                                                              ? cache.first_slot + key_index
                                                              : key_index - ring_tail;
                                    const uint16_t* source = bfloat16_key_values
                                                             + static_cast<size_t>(slot) * cache.columns
                                                             + static_cast<size_t>(kv_head) * head_dimension;
                                    float* packed_key = packed_keys + static_cast<size_t>(key_offset) * head_dimension;
                                    for (uint32_t column = 0; column < head_dimension; ++column)
                                        packed_key[column] = bfloat16_to_float(source[column]);
                                }
                                tile_keys = packed_keys;
                                key_stride = head_dimension;
                            }
                            else
                            {
                                const uint64_t slot = direct_bfloat16_contiguous || key_begin < ring_tail
                                                          ? cache.first_slot + key_begin
                                                          : key_begin - ring_tail;
                                tile_keys_bf16 = bfloat16_key_values
                                                 + static_cast<size_t>(slot) * cache.columns
                                                 + static_cast<size_t>(kv_head) * head_dimension;
                            }
                        }
                        else
                        {
                            tile_keys = float32_vector(key_values, key_begin, kv_head);
                            if (direct_float32_ring && key_begin < ring_tail && key_count > ring_tail - key_begin)
                            {
                                // Keep the GEMM tile intact: smaller tail kernels can change rounding.
                                float* packed_keys = tile_logits + query_tile_size * key_tile_size;
                                for (uint32_t key_offset = 0; key_offset < key_count; ++key_offset)
                                    std::copy_n(float32_vector(key_values, key_begin + key_offset, kv_head),
                                                head_dimension, packed_keys + static_cast<size_t>(key_offset) * head_dimension);
                                tile_keys = packed_keys;
                                key_stride = head_dimension;
                            }
                        }
                        if (tile_keys_bf16)
                        {
                            bfloat16_gemm_4x8(tile_keys_bf16,
                                              key_stride,
                                              query_values,
                                              query.columns(),
                                              head_dimension,
                                              key_count,
                                              query_count,
                                              tile_logits + tile_key_offset,
                                              key_tile_size);
                        }
                        else
                        {
                            float_gemm_4x8(tile_keys,
                                           key_stride,
                                           query_values,
                                           query.columns(),
                                           head_dimension,
                                           key_count,
                                           query_count,
                                           tile_logits + tile_key_offset,
                                           key_tile_size);
                        }
                        for (uint32_t query_offset = 0; query_offset < query_count; ++query_offset)
                        {
                            for (uint32_t key_offset = 0; key_offset < key_count; ++key_offset)
                            {
                                const uint64_t key_index = key_begin + key_offset;
                                float& score = tile_logits[static_cast<size_t>(query_offset) * key_tile_size
                                                           + static_cast<size_t>(key_index - tile_begin)];
                                if (valid_key(query_group + query_offset, key_index))
                                {
                                    score *= scale;
                                    tile_maximum[query_offset] = std::max(tile_maximum[query_offset], score);
                                }
                                else
                                {
                                    score = -std::numeric_limits<float>::infinity();
                                }
                            }
                        }
                    }
                    for (uint32_t query_offset = 0; query_offset < query_count; ++query_offset)
                    {
                        if (!std::isfinite(tile_maximum[query_offset]))
                            continue;
                        float* query_tile_output = tile_output + static_cast<size_t>(query_offset) * head_dimension;
                        float tile_normalizer = 0.0f;
                        float* tile_scores = tile_logits + static_cast<size_t>(query_offset) * key_tile_size;
                        for (uint32_t key_offset = 0; key_offset < tile_width; ++key_offset)
                        {
                            const float score = tile_scores[key_offset];
                            tile_scores[key_offset] = std::isfinite(score)
                                                          ? score - tile_maximum[query_offset]
                                                          : std::numeric_limits<float>::quiet_NaN();
                        }
                        float_exp_inplace(tile_scores, tile_width);
                        for (uint32_t key_offset = 0; key_offset < tile_width; ++key_offset)
                        {
                            const float probability = tile_scores[key_offset];
                            if (std::isfinite(probability))
                                tile_normalizer += probability;
                        }
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
                        constexpr uint32_t pv_tile = 64;
#else
                        constexpr uint32_t pv_tile = 32;
#endif
                        uint32_t d = 0;
                        for (; d + pv_tile <= head_dimension; d += pv_tile)
                        {
                            std::array<float, pv_tile> a = {};
                            const size_t vo = static_cast<size_t>(kv_head) * head_dimension + d;
                            for (uint64_t b = tile_begin; b < tile_end;)
                            {
                                const uint64_t e = direct_float32_ring && b < ring_tail
                                                       ? std::min(tile_end, ring_tail)
                                                       : tile_end;
                                const float* v = float32_vector(value_values, b, 0);
                                for (uint64_t k = b; k < e; ++k)
                                {
                                    const float p = tile_scores[static_cast<size_t>(k - tile_begin)];
                                    if (std::isfinite(p))
                                    {
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
                                        for (uint32_t i = 0; i < pv_tile; ++i)
                                            a[i] += p * v[vo + i];
                                    }
                                    v += cache.columns;
                                }
                                b = e;
                            }
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
                            for (uint32_t i = 0; i < pv_tile; ++i)
                                query_tile_output[d + i] = a[i];
                        }
                        if (d < head_dimension)
                        {
                            const uint32_t n = head_dimension - d;
                            const size_t vo = static_cast<size_t>(kv_head) * head_dimension + d;
                            std::fill(query_tile_output + d, query_tile_output + head_dimension, 0.0f);
                            for (uint64_t b = tile_begin; b < tile_end;)
                            {
                                const uint64_t e = direct_float32_ring && b < ring_tail
                                                       ? std::min(tile_end, ring_tail)
                                                       : tile_end;
                                const float* v = float32_vector(value_values, b, 0);
                                for (uint64_t k = b; k < e; ++k)
                                {
                                    const float p = tile_scores[static_cast<size_t>(k - tile_begin)];
                                    if (std::isfinite(p))
                                        float_scaled_add(query_tile_output + d, v + vo, p, n);
                                    v += cache.columns;
                                }
                                b = e;
                            }
                        }
                        const float new_maximum = std::max(maximum[query_offset], tile_maximum[query_offset]);
                        const float old_scale = std::isfinite(maximum[query_offset])
                                                    ? float_approximate_exp(maximum[query_offset] - new_maximum)
                                                    : 0.0f;
                        const float tile_scale = float_approximate_exp(tile_maximum[query_offset] - new_maximum);
                        float_scale_add(running + static_cast<size_t>(query_offset) * head_dimension,
                                        old_scale,
                                        query_tile_output,
                                        tile_scale,
                                        head_dimension);
                        normalizer[query_offset] = normalizer[query_offset] * old_scale + tile_normalizer * tile_scale;
                        maximum[query_offset] = new_maximum;
                    }
                }
                for (uint32_t query_offset = 0; query_offset < query_count; ++query_offset)
                {
                    float* destination = output.row(query_group + query_offset)
                                         + static_cast<size_t>(query_head) * head_dimension;
                    if (normalizer[query_offset] > 0.0f)
                    {
                        std::copy_n(running + static_cast<size_t>(query_offset) * head_dimension,
                                    head_dimension,
                                    destination);
                        float_scale_inplace(destination, 1.0f / normalizer[query_offset], head_dimension);
                    }
                    else
                    {
                        std::fill(destination, destination + head_dimension, 0.0f);
                    }
                }
            };
            const size_t running_size = static_cast<size_t>(query_tile_size) * head_dimension;
            const size_t tile_logits_size = static_cast<size_t>(query_tile_size) * key_tile_size;
            const uint64_t worker_elements = (static_cast<uint64_t>(query_tile_size) * 2
                                              + ((direct_float32_ring || direct_bfloat16_ring)
                                                     ? key_gemm_tile_size
                                                     : 0))
                                                 * head_dimension
                                             + tile_logits_size + worker_padding;
            if (worker_elements > scratch.workspace.max_size() / attention_team_size)
                throw std::length_error("attention workspace is too large");
            const size_t worker_stride = static_cast<size_t>(worker_elements);
            const size_t worker_size = static_cast<size_t>(attention_team_size) * worker_stride;
            if (scratch.workspace.size() < worker_size)
                scratch.workspace.resize(worker_size);
#if defined(_OPENMP)
            if (attention_team_size > 1)
            {
#pragma omp parallel num_threads(attention_team_size)
                {
                    const size_t worker_offset = static_cast<size_t>(omp_get_thread_num()) * worker_stride;
                    float* running = scratch.workspace.data() + worker_offset;
                    float* tile_output = running + running_size;
                    float* tile_logits = tile_output + running_size;
#pragma omp for schedule(static)
                    for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
                        process_flash_group(static_cast<uint64_t>(job), running, tile_output, tile_logits);
                }
            }
            else
#endif
            {
                float* running = scratch.workspace.data();
                float* tile_output = running + running_size;
                float* tile_logits = tile_output + running_size;
                for (uint64_t job = 0; job < job_count; ++job)
                    process_flash_group(job, running, tile_output, tile_logits);
            }
            return;
        }

        const uint32_t split_team_size = static_cast<uint32_t>(attention_team_size);
        const uint64_t key_chunk_size = 256;
        const uint64_t chunk_count = (cache.token_count + key_chunk_size - 1) / key_chunk_size;
        flash_partial_max.assign(static_cast<size_t>(head_count) * split_team_size, -std::numeric_limits<float>::infinity());
        flash_partial_sum.assign(static_cast<size_t>(head_count) * split_team_size, 0.0f);
        flash_partial_output.assign(static_cast<size_t>(head_count) * split_team_size * head_dimension, 0.0f);
        const uint64_t worker_elements = static_cast<uint64_t>(head_dimension) + key_chunk_size + worker_padding;
        if (worker_elements > scratch.workspace.max_size() / split_team_size)
            throw std::length_error("attention workspace is too large");
        const size_t worker_stride = static_cast<size_t>(worker_elements);
        const size_t worker_size = static_cast<size_t>(split_team_size) * worker_stride;
        if (scratch.workspace.size() < worker_size)
            scratch.workspace.resize(worker_size);
        const auto compute_split_chunk = [&](uint32_t query_head,
                                             uint64_t chunk,
                                             float& local_maximum,
                                             float& local_normalizer,
                                             float* local_output,
                                             float* local_logits) {
            const uint64_t begin = chunk * key_chunk_size;
            const uint64_t end = std::min(cache.token_count, begin + key_chunk_size);
            const float* query_vector = query.row(0) + static_cast<size_t>(query_head) * head_dimension;
            local_maximum = -std::numeric_limits<float>::infinity();
            local_normalizer = 0.0f;
            std::fill(local_output, local_output + head_dimension, 0.0f);
            for (uint64_t key_index = begin; key_index < end; ++key_index)
            {
                const size_t chunk_offset = static_cast<size_t>(key_index - begin);
                if (valid_key(0, key_index))
                {
                    const float score = key_dot(query_head, query_vector, key_index) * scale;
                    local_logits[chunk_offset] = score;
                    local_maximum = std::max(local_maximum, score);
                }
                else
                {
                    local_logits[chunk_offset] = -std::numeric_limits<float>::infinity();
                }
            }
            if (!std::isfinite(local_maximum))
                return;
            for (uint64_t key_index = begin; key_index < end; ++key_index)
            {
                const size_t chunk_offset = static_cast<size_t>(key_index - begin);
                const float score = local_logits[chunk_offset];
                local_logits[chunk_offset] = std::isfinite(score)
                                                 ? score - local_maximum
                                                 : std::numeric_limits<float>::quiet_NaN();
            }
            float_exp_inplace(local_logits, static_cast<uint32_t>(end - begin));
            for (uint64_t key_index = begin; key_index < end; ++key_index)
            {
                const float probability = local_logits[static_cast<size_t>(key_index - begin)];
                if (!std::isfinite(probability))
                    continue;
                local_normalizer += probability;
                add_value(query_head, local_output, probability, key_index);
            }
        };
        const auto merge_split_chunk = [&](uint32_t query_head, uint32_t thread_index, float local_maximum, float local_normalizer, const float* local_output) {
            const size_t partial_index = static_cast<size_t>(query_head) * split_team_size + thread_index;
            float& partial_maximum = flash_partial_max[partial_index];
            float& partial_normalizer = flash_partial_sum[partial_index];
            float* partial_output = flash_partial_output.data() + partial_index * head_dimension;
            if (!std::isfinite(local_maximum))
                return;
            const float new_maximum = std::max(partial_maximum, local_maximum);
            const float old_scale = std::isfinite(partial_maximum) ? float_approximate_exp(partial_maximum - new_maximum) : 0.0f;
            const float local_scale = float_approximate_exp(local_maximum - new_maximum);
            float_scale_add(partial_output, old_scale, local_output, local_scale, head_dimension);
            partial_normalizer = partial_normalizer * old_scale + local_normalizer * local_scale;
            partial_maximum = new_maximum;
        };
        const auto reduce_split_head = [&](uint32_t query_head, uint32_t team_size) {
            float maximum = sink_value(query_head);
            float normalizer = sinks ? 1.0f : 0.0f;
            float* reduced = output.row(0) + static_cast<size_t>(query_head) * head_dimension;
            for (uint32_t thread_index = 0; thread_index < team_size; ++thread_index)
            {
                const size_t partial_index = static_cast<size_t>(query_head) * split_team_size + thread_index;
                const float local_maximum = flash_partial_max[partial_index];
                if (!std::isfinite(local_maximum))
                    continue;
                const float new_maximum = std::max(maximum, local_maximum);
                const float old_scale = std::isfinite(maximum) ? float_approximate_exp(maximum - new_maximum) : 0.0f;
                const float local_scale = float_approximate_exp(local_maximum - new_maximum);
                const float* partial_output = flash_partial_output.data() + partial_index * head_dimension;
                float_scale_add(reduced, old_scale, partial_output, local_scale, head_dimension);
                normalizer = normalizer * old_scale + flash_partial_sum[partial_index] * local_scale;
                maximum = new_maximum;
            }
            if (normalizer > 0.0f)
                float_scale_inplace(reduced, 1.0f / normalizer, head_dimension);
        };
#if defined(_OPENMP)
#pragma omp parallel num_threads(split_team_size)
        {
            const uint32_t thread_index = static_cast<uint32_t>(omp_get_thread_num());
            const uint32_t actual_split_team_size = static_cast<uint32_t>(omp_get_num_threads());
            float* local_output = scratch.workspace.data()
                                  + static_cast<size_t>(thread_index) * worker_stride;
            float* local_logits = local_output + head_dimension;
#pragma omp for schedule(static)
            for (int64_t job = 0; job < static_cast<int64_t>(head_count * chunk_count); ++job)
            {
                const uint32_t query_head = static_cast<uint32_t>(job / chunk_count);
                const uint64_t chunk = static_cast<uint64_t>(job) % chunk_count;
                float local_maximum = -std::numeric_limits<float>::infinity();
                float local_normalizer = 0.0f;
                compute_split_chunk(query_head,
                                    chunk,
                                    local_maximum,
                                    local_normalizer,
                                    local_output,
                                    local_logits);
                merge_split_chunk(query_head, thread_index, local_maximum, local_normalizer, local_output);
            }
#pragma omp single nowait
            for (uint32_t query_head = 0; query_head < head_count; ++query_head)
                reduce_split_head(query_head, actual_split_team_size);
        }
#else
        float* local_output = scratch.workspace.data();
        float* local_logits = local_output + head_dimension;
        for (uint32_t query_head = 0; query_head < head_count; ++query_head)
        {
            for (uint64_t chunk = 0; chunk < chunk_count; ++chunk)
            {
                float local_maximum = -std::numeric_limits<float>::infinity();
                float local_normalizer = 0.0f;
                compute_split_chunk(query_head,
                                    chunk,
                                    local_maximum,
                                    local_normalizer,
                                    local_output,
                                    local_logits);
                merge_split_chunk(query_head, 0, local_maximum, local_normalizer, local_output);
            }
            reduce_split_head(query_head, 1);
        }
#endif
        return;
    }

    const bool use_simd_exp = decode_all_keys_valid
                              && cache.token_count >= 4
                              && cache.token_count <= std::numeric_limits<uint32_t>::max()
                              && float_exp_simd_available();
    const int head_threads = query.rows() == 1 && !has_selected_keys
                                     && head_count >= 4 && cache.token_count >= 64
                                 ? std::min<uint32_t>(static_cast<uint32_t>(cpu_linear_team_size(2 * split_kv_work, cache.dtype)), head_count)
                                 : 1;
    const size_t logits_stride = has_selected_keys
                                     ? maximum_selected_keys
                                     : static_cast<size_t>(cache.token_count);
    if (logits_stride > scratch.logits.max_size() / head_threads)
        throw std::length_error("attention logits are too large");
    scratch.logits.resize(logits_stride * head_threads);

    for (size_t query_index = 0; query_index < query.rows(); ++query_index)
    {
        const uint64_t query_position = position_offset + query_index;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(head_threads) schedule(static) if (head_threads > 1)
#endif
        for (int64_t h = 0; h < static_cast<int64_t>(head_count); ++h)
        {
            const uint32_t query_head = static_cast<uint32_t>(h);
            size_t worker = 0;
#if defined(_OPENMP)
            if (head_threads > 1)
                worker = static_cast<size_t>(omp_get_thread_num());
#endif
            float* head_logits = logits_stride == 0 ? nullptr : scratch.logits.data() + worker * logits_stride;
            const uint32_t kv_head = query_head / heads_per_group;
            const float* query_vector = query.row(query_index) + query_head * head_dimension;
            float maximum = -std::numeric_limits<float>::infinity();
            if (sinks)
            {
                maximum = sinks->dtype == DType::Float32 ? sinks->float32_values()[query_head] : bfloat16_to_float(sinks->bfloat16_values()[query_head]);
            }

            if (has_selected_keys)
            {
                const size_t selected_begin = selected_offsets[query_index];
                const size_t selected_end = selected_offsets[query_index + 1];
                for (size_t selected_index = selected_begin;
                     selected_index < selected_end; ++selected_index)
                {
                    const uint64_t key_index = selected_indices[selected_index];
                    const uint64_t key_position = cache.start_position + key_index;
                    const bool future = key_position > query_position;
                    const bool too_old = plan.sliding_window > 0
                                         && key_position + plan.sliding_window <= query_position;
                    const size_t logit_index = selected_index - selected_begin;
                    if (key_index >= cache.token_count || future || too_old)
                    {
                        head_logits[logit_index] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    float dot = 0.0f;
                    if (direct_bfloat16)
                    {
                        const uint64_t slot = direct_bfloat16_contiguous
                                                  ? cache.first_slot + key_index
                                                  : cache_slot(cache, key_index);
                        const uint16_t* key_vector = bfloat16_key_values
                                                     + static_cast<size_t>(slot) * cache.columns
                                                     + static_cast<size_t>(kv_head) * head_dimension;
                        dot = bfloat16_dot(key_vector, query_vector, head_dimension);
                    }
                    else
                    {
                        const float* key_vector = float32_vector(key_values, key_index, kv_head);
                        dot = float_dot(query_vector, key_vector, head_dimension);
                    }
                    head_logits[logit_index] = dot * scale;
                    maximum = std::max(maximum, head_logits[logit_index]);
                }

                float normalizer = 0.0f;
                if (sinks)
                {
                    const float sink_value = sinks->dtype == DType::Float32
                                                 ? sinks->float32_values()[query_head]
                                                 : bfloat16_to_float(sinks->bfloat16_values()[query_head]);
                    normalizer = float_approximate_exp(sink_value - maximum);
                }
                for (size_t selected_index = selected_begin;
                     selected_index < selected_end; ++selected_index)
                {
                    const size_t logit_index = selected_index - selected_begin;
                    if (std::isfinite(head_logits[logit_index]))
                    {
                        head_logits[logit_index] = float_approximate_exp(head_logits[logit_index] - maximum);
                        normalizer += head_logits[logit_index];
                    }
                    else
                    {
                        head_logits[logit_index] = 0.0f;
                    }
                }

                if (normalizer > 0.0f)
                {
                    float* output_vector = output.row(query_index)
                                           + query_head * head_dimension;
                    for (size_t selected_index = selected_begin;
                         selected_index < selected_end; ++selected_index)
                    {
                        const uint64_t key_index = selected_indices[selected_index];
                        const float probability = head_logits[selected_index - selected_begin]
                                                  / normalizer;
                        if (direct_bfloat16)
                        {
                            const uint64_t slot = direct_bfloat16_contiguous
                                                      ? cache.first_slot + key_index
                                                      : cache_slot(cache, key_index);
                            const uint16_t* value_vector = bfloat16_value_values
                                                           + static_cast<size_t>(slot) * cache.columns
                                                           + static_cast<size_t>(kv_head) * head_dimension;
                            bfloat16_scaled_add(output_vector, value_vector, probability,
                                                head_dimension);
                        }
                        else
                        {
                            const float* value_vector = float32_vector(value_values, key_index, kv_head);
                            float_scaled_add(output_vector, value_vector, probability,
                                             head_dimension);
                        }
                    }
                }
                continue;
            }

            if (decode_all_keys_valid)
            {
                if (direct_bfloat16)
                {
                    for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index)
                    {
                        const uint64_t slot = direct_bfloat16_contiguous
                                                  ? cache.first_slot + key_index
                                                  : cache_slot(cache, key_index);
                        const uint16_t* key_vector = bfloat16_key_values + static_cast<size_t>(slot) * cache.columns
                                                     + static_cast<size_t>(kv_head) * head_dimension;
                        head_logits[key_index] = bfloat16_dot(key_vector, query_vector, head_dimension) * scale;
                        maximum = std::max(maximum, head_logits[key_index]);
                    }
                }
                else
                {
                    for (uint64_t begin = 0; begin < cache.token_count;)
                    {
                        const uint64_t end = direct_float32_ring && begin < ring_tail
                                                 ? std::min(cache.token_count, ring_tail)
                                                 : cache.token_count;
                        const float* keys = float32_vector(key_values, begin, kv_head);
                        for (uint64_t key_index = begin; key_index < end; ++key_index)
                        {
                            head_logits[key_index] = float_dot(query_vector, keys + static_cast<size_t>(key_index - begin) * cache.columns, head_dimension) * scale;
                            maximum = std::max(maximum, head_logits[key_index]);
                        }
                        begin = end;
                    }
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
                        head_logits[key_index] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    float dot = 0.0f;
                    if (direct_bfloat16)
                    {
                        const uint64_t slot = direct_bfloat16_contiguous
                                                  ? cache.first_slot + key_index
                                                  : cache_slot(cache, key_index);
                        const uint16_t* key_vector = bfloat16_key_values + static_cast<size_t>(slot) * cache.columns
                                                     + static_cast<size_t>(kv_head) * head_dimension;
                        dot = bfloat16_dot(key_vector, query_vector, head_dimension);
                    }
                    else
                    {
                        const float* key_vector = float32_vector(key_values, key_index, kv_head);
                        dot = float_dot(query_vector, key_vector, head_dimension);
                    }
                    head_logits[key_index] = dot * scale;
                    maximum = std::max(maximum, head_logits[key_index]);
                }
            }

            float normalizer = 0.0f;
            if (sinks)
            {
                const float sink_value = sinks->dtype == DType::Float32 ? sinks->float32_values()[query_head] : bfloat16_to_float(sinks->bfloat16_values()[query_head]);
                normalizer = float_approximate_exp(sink_value - maximum);
            }
            if (use_simd_exp)
            {
                const float negative_infinity = -std::numeric_limits<float>::infinity();
                for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index)
                {
                    const float score = head_logits[key_index];
                    head_logits[key_index] = std::isfinite(score) ? score - maximum : negative_infinity;
                }
                float_exp_inplace(head_logits, static_cast<uint32_t>(cache.token_count));
                for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index)
                    normalizer += head_logits[key_index];
            }
            else
            {
                for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index)
                {
                    if (std::isfinite(head_logits[key_index]))
                    {
                        head_logits[key_index] = float_approximate_exp(head_logits[key_index] - maximum);
                        normalizer += head_logits[key_index];
                    }
                    else
                    {
                        head_logits[key_index] = 0.0f;
                    }
                }
            }

            float* output_vector = output.row(query_index) + query_head * head_dimension;
            if (direct_bfloat16)
            {
                for (uint64_t key_index = 0; key_index < cache.token_count; ++key_index)
                {
                    const float probability = head_logits[key_index] / normalizer;
                    const uint64_t slot = direct_bfloat16_contiguous
                                              ? cache.first_slot + key_index
                                              : cache_slot(cache, key_index);
                    const uint16_t* value_vector = bfloat16_value_values + static_cast<size_t>(slot) * cache.columns
                                                   + static_cast<size_t>(kv_head) * head_dimension;
                    bfloat16_scaled_add(output_vector, value_vector, probability, head_dimension);
                }
            }
            else
            {
                for (uint64_t begin = 0; begin < cache.token_count;)
                {
                    const uint64_t end = direct_float32_ring && begin < ring_tail
                                             ? std::min(cache.token_count, ring_tail)
                                             : cache.token_count;
                    const float* values = float32_vector(value_values, begin, kv_head);
                    for (uint64_t key_index = begin; key_index < end; ++key_index)
                        float_scaled_add(output_vector, values + static_cast<size_t>(key_index - begin) * cache.columns,
                                         head_logits[key_index] / normalizer, head_dimension);
                    begin = end;
                }
            }
        }
    }
}

static void trim_sliding_cache(LayerCache& cache, const AttentionBlockPlan& plan)
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

static void attention_linear_into(const WeightStore& weights, const CompiledOperatorTable& operators, TensorHandle matrix, TensorHandle bias, const ActivationBuffer& input, ActivationBuffer& output, uint64_t optimization_flags, ExecutionBackend backend)
{
    if (bias == invalid_tensor_handle)
    {
        forward_linear(weights.at(matrix), input, output, optimization_flags, operators.find_weight(matrix), backend);
    }
    else
    {
        forward_linear(weights.at(matrix), weights.at(bias), input, output, optimization_flags, operators.find_weight(matrix), backend);
    }
}

static float attention_weight_value(const TensorData& tensor, size_t index)
{
    if (tensor.dtype == DType::Float32)
        return tensor.float32_values()[index];
    return bfloat16_to_float(tensor.bfloat16_values()[index]);
}

static void apply_head_rms_norm(ActivationBuffer& batch, uint32_t head_count, uint32_t head_dimension, const TensorData& weight, float epsilon, float weight_offset)
{
    assert(weight.element_count() == head_dimension);
    for (size_t token_index = 0; token_index < batch.rows(); ++token_index)
    {
        float* token = batch.row(token_index);
        for (uint32_t head = 0; head < head_count; ++head)
        {
            float* values = token + head * head_dimension;
            if (weight.dtype == DType::Float32)
            {
                float_rms_norm(values,
                               values,
                               weight.float32_values().data(),
                               epsilon,
                               weight_offset,
                               head_dimension);
                continue;
            }
            if (weight.dtype == DType::BFloat16)
            {
                bfloat16_rms_norm(values,
                                  values,
                                  weight.bfloat16_values().data(),
                                  epsilon,
                                  weight_offset,
                                  head_dimension);
                continue;
            }
            float square_sum = 0.0f;
            for (uint32_t column = 0; column < head_dimension; ++column)
                square_sum += values[column] * values[column];
            const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(head_dimension) + epsilon);
            for (uint32_t column = 0; column < head_dimension; ++column)
            {
                values[column] *= inverse_rms * (attention_weight_value(weight, column) + weight_offset);
            }
        }
    }
}

static Result<void> project_and_append_qsa_keys(const WeightStore& weights,
                                                const CompiledOperatorTable& operators,
                                                const AttentionBlockPlan& plan,
                                                uint64_t position_offset,
                                                const ActivationBuffer& normalized,
                                                LayerCache& cache,
                                                AttentionScratch& scratch,
                                                uint64_t optimization_flags)
{
    if (!has_flag(plan.flags, AttentionBlockQsa))
        return {};
    if (cache.transaction.active)
        return Error{ErrorCode::UnsupportedModel, "QSA cache transactions are not supported"};
    if (plan.compression_ratio == 0 || plan.index_head_dimension == 0)
        return Error{ErrorCode::InvalidModel, "invalid QSA key cache dimensions"};
    const uint64_t maximum_token_count = std::numeric_limits<uint32_t>::max();
    if (cache.token_count > maximum_token_count
        || normalized.rows() > maximum_token_count - cache.token_count)
    {
        return Error{ErrorCode::InvalidModel, "QSA key index exceeds uint32 storage"};
    }
    if (normalized.rows() != 0
        && normalized.rows() - 1 > std::numeric_limits<uint64_t>::max() - position_offset)
    {
        return Error{ErrorCode::InvalidModel, "QSA query position overflows"};
    }
    const size_t key_dimension = plan.index_head_dimension;
    const uint64_t tail_capacity_tokens = plan.compression_ratio - 1;
    if (tail_capacity_tokens > std::numeric_limits<size_t>::max() / key_dimension)
        return Error{ErrorCode::InvalidModel, "QSA key tail size overflows"};
    const size_t tail_block_size = static_cast<size_t>(tail_capacity_tokens) * key_dimension;
    const uint64_t complete_blocks = cache.token_count / plan.compression_ratio;
    const uint64_t tail_tokens = cache.token_count % plan.compression_ratio;
    if (complete_blocks > std::numeric_limits<size_t>::max() / key_dimension
        || tail_tokens > std::numeric_limits<size_t>::max() / key_dimension)
    {
        return Error{ErrorCode::InvalidModel, "QSA key cache size overflows"};
    }
    if (cache.qsa_block_keys.size() != static_cast<size_t>(complete_blocks) * key_dimension
        || cache.qsa_index_key_tail.size() != static_cast<size_t>(tail_tokens) * key_dimension)
    {
        return Error{ErrorCode::InternalError, "QSA key cache is out of sync"};
    }
    const uint64_t cache_start_position = cache.token_count == 0
                                              ? position_offset
                                              : cache.start_position;
    if (cache.token_count != 0
        && (position_offset < cache_start_position
            || position_offset - cache_start_position != cache.token_count))
    {
        return Error{ErrorCode::InternalError, "QSA key cache position is out of sync"};
    }
    attention_linear_into(weights, operators, plan.qsa_query_key_weight,
                          invalid_tensor_handle, normalized, scratch.qsa_query_key,
                          optimization_flags, ExecutionBackend::Cpu);
    const uint32_t query_columns = plan.index_head_count * plan.index_head_dimension;
    const uint32_t expected_columns = query_columns + plan.index_head_dimension;
    if (scratch.qsa_query_key.columns() != expected_columns)
        return Error{ErrorCode::InvalidModel, "invalid QSA projection output"};
    scratch.qsa_query.reset(normalized.rows(), query_columns, false);
    const uint32_t ratio = plan.compression_ratio;
    const uint32_t rope_dimension = plan.rope_head_dimension == 0
                                        ? plan.index_head_dimension
                                        : plan.rope_head_dimension;
    const TensorData& key_norm = weights.at(plan.qsa_key_norm_weight);
    const uint64_t appended_token_count = cache.token_count + normalized.rows();
    const uint64_t appended_block_count = appended_token_count / ratio;
    if (appended_block_count > std::numeric_limits<size_t>::max() / key_dimension)
        return Error{ErrorCode::InvalidModel, "QSA block cache size overflows"};
    cache.qsa_index_key_tail.reserve(tail_block_size);
    for (size_t row_index = 0; row_index < normalized.rows(); ++row_index)
    {
        const float* source = scratch.qsa_query_key.row(row_index);
        std::copy_n(source, query_columns, scratch.qsa_query.row(row_index));
        const float* index_key = source + query_columns;
        if (cache.qsa_index_key_tail.size() == tail_block_size)
        {
            const size_t block_index = cache.qsa_block_keys.size() / key_dimension;
            const size_t block_offset = cache.qsa_block_keys.size();
            cache.qsa_block_keys.resize(block_offset + key_dimension, 0.0f);
            float* pooled = cache.qsa_block_keys.data() + block_offset;
            for (size_t token = 0; token < cache.qsa_index_key_tail.size() / key_dimension; ++token)
            {
                const uint16_t* raw = cache.qsa_index_key_tail.data() + token * key_dimension;
                for (size_t column = 0; column < key_dimension; ++column)
                    pooled[column] += bfloat16_to_float(raw[column]);
            }
            for (size_t column = 0; column < key_dimension; ++column)
                pooled[column] += bfloat16_to_float(float_to_bfloat16(index_key[column]));

            const float inverse_ratio = 1.0f / static_cast<float>(ratio);
            for (size_t column = 0; column < key_dimension; ++column)
            {
                pooled[column] *= inverse_ratio;
                pooled[column] = bfloat16_to_float(float_to_bfloat16(pooled[column]));
            }
            float square_sum = 0.0f;
            for (size_t column = 0; column < key_dimension; ++column)
                square_sum += pooled[column] * pooled[column];
            const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(key_dimension) + plan.norm_epsilon);
            for (size_t column = 0; column < key_dimension; ++column)
            {
                pooled[column] *= inverse_rms
                                  * (attention_weight_value(key_norm, static_cast<uint32_t>(column))
                                     + plan.norm_weight_offset);
            }
            apply_rope(pooled,
                       rope_dimension,
                       cache_start_position + static_cast<uint64_t>(block_index) * ratio,
                       plan);
            cache.qsa_index_key_tail.clear();
        }
        else
        {
            const size_t tail_offset = cache.qsa_index_key_tail.size();
            cache.qsa_index_key_tail.resize(tail_offset + key_dimension);
            float_to_bfloat16_array(cache.qsa_index_key_tail.data() + tail_offset,
                                    index_key,
                                    plan.index_head_dimension);
        }
    }
    return {};
}

static Result<void> prepare_qsa_selection(const WeightStore& weights,
                                          const AttentionBlockPlan& plan,
                                          uint64_t position_offset,
                                          LayerCache& cache,
                                          AttentionScratch& scratch)
{
    if (!has_flag(plan.flags, AttentionBlockQsa))
    {
        scratch.qsa_selected_offsets.clear();
        scratch.qsa_selected_indices.clear();
        return {};
    }
    if (plan.compression_ratio == 0 || plan.index_head_dimension == 0)
        return Error{ErrorCode::InvalidModel, "invalid QSA key cache dimensions"};
    const uint64_t stored_complete_blocks = cache.token_count / plan.compression_ratio;
    const uint64_t tail_tokens = cache.token_count % plan.compression_ratio;
    if (stored_complete_blocks > std::numeric_limits<size_t>::max() / plan.index_head_dimension
        || tail_tokens > std::numeric_limits<size_t>::max() / plan.index_head_dimension
        || cache.qsa_block_keys.size()
               != static_cast<size_t>(stored_complete_blocks) * plan.index_head_dimension
        || cache.qsa_index_key_tail.size()
               != static_cast<size_t>(tail_tokens) * plan.index_head_dimension)
    {
        return Error{ErrorCode::InternalError, "QSA key cache is out of sync after append"};
    }
    apply_head_rms_norm(scratch.qsa_query, plan.index_head_count,
                        plan.index_head_dimension,
                        weights.at(plan.qsa_query_norm_weight), plan.norm_epsilon,
                        plan.norm_weight_offset);
    const uint32_t rope_dimension = plan.rope_head_dimension == 0
                                        ? plan.index_head_dimension
                                        : plan.rope_head_dimension;
    for (size_t row_index = 0; row_index < scratch.qsa_query.rows(); ++row_index)
    {
        for (uint32_t head = 0; head < plan.index_head_count; ++head)
        {
            apply_rope(scratch.qsa_query.row(row_index)
                           + static_cast<size_t>(head) * plan.index_head_dimension,
                       rope_dimension, position_offset + row_index, plan);
        }
    }

    if (cache.token_count > std::numeric_limits<uint32_t>::max())
        return Error{ErrorCode::InvalidModel, "QSA key index exceeds uint32 storage"};
    const uint64_t maximum_selected_per_query = static_cast<uint64_t>(plan.index_top_k) * plan.compression_ratio
                                                + plan.compression_ratio - 1;
    if (maximum_selected_per_query > std::numeric_limits<size_t>::max()
        || (scratch.qsa_query.rows() != 0
            && maximum_selected_per_query
                   > std::numeric_limits<size_t>::max()
                         / scratch.qsa_query.rows()))
    {
        return Error{ErrorCode::InvalidModel, "QSA selection storage size overflows"};
    }
    scratch.qsa_selected_offsets.assign(scratch.qsa_query.rows() + 1, 0);
    scratch.qsa_selected_indices.clear();
    scratch.qsa_selected_indices.reserve(scratch.qsa_query.rows()
                                         * static_cast<size_t>(maximum_selected_per_query));
    const float scale = 1.0f / std::sqrt(static_cast<float>(plan.index_head_dimension));
    std::vector<std::pair<float, uint32_t>>& scores = scratch.index_scores;
    for (size_t query_index = 0; query_index < scratch.qsa_query.rows(); ++query_index)
    {
        const uint64_t query_position = position_offset + query_index;
        const uint64_t visible_count = query_position < cache.start_position
                                           ? 0
                                           : std::min<uint64_t>(cache.token_count,
                                                                query_position - cache.start_position + 1);
        const uint64_t complete_blocks = visible_count / plan.compression_ratio;
        scores.clear();
        for (uint32_t block = 0; block < complete_blocks; ++block)
        {
            const float* pooled = cache.qsa_block_keys.data()
                                  + static_cast<size_t>(block) * plan.index_head_dimension;
            float score = 0.0f;
            for (uint32_t head = 0; head < plan.index_head_count; ++head)
            {
                const float* query = scratch.qsa_query.row(query_index)
                                     + static_cast<size_t>(head) * plan.index_head_dimension;
                score += std::max(0.0f,
                                  float_dot(query, pooled,
                                            plan.index_head_dimension));
            }
            scores.emplace_back(score * scale, block);
        }
        const size_t selected_count = std::min<size_t>(plan.index_top_k, scores.size());
        std::partial_sort(scores.begin(), scores.begin() + selected_count, scores.end(),
                          [](const auto& left, const auto& right) {
                              if (left.first != right.first)
                                  return left.first > right.first;
                              return left.second < right.second;
                          });
        const size_t selection_begin = scratch.qsa_selected_indices.size();
        for (size_t index = 0; index < selected_count; ++index)
        {
            const uint64_t first_token = static_cast<uint64_t>(scores[index].second)
                                         * plan.compression_ratio;
            for (uint32_t token = 0;
                 token < plan.compression_ratio; ++token)
            {
                scratch.qsa_selected_indices.push_back(static_cast<uint32_t>(first_token + token));
            }
        }
        for (uint64_t token = complete_blocks * plan.compression_ratio;
             token < visible_count; ++token)
        {
            scratch.qsa_selected_indices.push_back(static_cast<uint32_t>(token));
        }
        std::sort(scratch.qsa_selected_indices.begin() + selection_begin,
                  scratch.qsa_selected_indices.end());
        scratch.qsa_selected_offsets[query_index + 1] = scratch.qsa_selected_indices.size();
    }
    scores.clear();
    return {};
}

Result<void> append_attention_context(const WeightStore& weights,
                                      const CompiledOperatorTable& operators,
                                      const AttentionBlockPlan& plan,
                                      ExecutionBackend backend,
                                      uint64_t position_offset,
                                      LayerCache& cache,
                                      AttentionScratch& scratch,
                                      const ActivationBuffer& hidden,
                                      uint64_t optimization_flags)
{
    if (cache.vulkan_attention_cache)
    {
        const CompiledOperator& attention_operator = operators.at(plan.vulkan_attention_operator);
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

    configure_cache(cache, plan.kv_head_count * plan.head_dimension,
                    plan.kv_cache_dtype);
    const ActivationBuffer* normalized = &hidden;
    if (plan.pre_attention_norm_weight != invalid_tensor_handle)
    {
        forward_rms_norm(hidden,
                         weights.at(plan.pre_attention_norm_weight),
                         plan.norm_epsilon,
                         scratch.normalized,
                         plan.norm_weight_offset);
        normalized = &scratch.normalized;
    }
    auto qsa_status = project_and_append_qsa_keys(weights, operators, plan, position_offset,
                                                  *normalized, cache, scratch,
                                                  optimization_flags);
    if (!qsa_status)
        return qsa_status.error();
    ActivationBuffer& key = scratch.key;
    ActivationBuffer& value = scratch.value;
    ActivationBuffer& fused_qkv = scratch.fused_qkv;
    const CompiledOperator& fused_qkv_gate_operator = operators.at(plan.fused_qkv_gate_bfloat16_operator);
    const CompiledOperator& fused_qkv_operator = operators.at(plan.fused_qkv_operator);
    if (backend == ExecutionBackend::Vulkan
        && ((fused_qkv_gate_operator.bfloat16
             && fused_qkv_gate_operator.bfloat16->forward(*normalized,
                                                          fused_qkv))
            || (fused_qkv_operator.bfloat16
                && fused_qkv_operator.bfloat16->forward(*normalized,
                                                        fused_qkv))
            || (fused_qkv_operator.linear
                && fused_qkv_operator.linear->forward(*normalized,
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
        attention_linear_into(weights, operators, plan.key_weight, plan.key_bias, *normalized, key, optimization_flags, backend);
        attention_linear_into(weights, operators, plan.value_weight, plan.value_bias, *normalized, value, optimization_flags, backend);
    }

    if (has_flag(plan.flags, AttentionBlockQueryKeyNorm))
    {
        apply_head_rms_norm(key,
                            plan.kv_head_count,
                            plan.head_dimension,
                            weights.at(plan.key_norm_weight),
                            plan.norm_epsilon,
                            plan.norm_weight_offset);
    }

    const uint32_t rope_dimension = plan.rope_head_dimension == 0 ? plan.head_dimension : plan.rope_head_dimension;
    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
    {
        const uint64_t position = position_offset + token_index;
        if (cached_rope_coefficients_enabled(optimization_flags))
        {
            prepare_rope_coefficients(rope_dimension,
                                      position,
                                      plan,
                                      scratch.rope_cosine,
                                      scratch.rope_sine);
            for (uint32_t head = 0; head < plan.kv_head_count; ++head)
            {
                apply_prepared_rope(key.row(token_index) + head * plan.head_dimension,
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
    append_cache(cache, plan.kv_cache_dtype, key, value);
    trim_sliding_cache(cache, plan);
    return {};
}

Result<bool> forward_attention_batch(const CompiledOperatorTable& operators,
                                     const AttentionBlockPlan& plan,
                                     ExecutionBackend backend,
                                     std::span<AttentionBatchEntry> entries,
                                     uint64_t optimization_flags)
{
    (void)optimization_flags;
    const CompiledOperator& attention_operator = operators.at(plan.vulkan_attention_operator);
    if (backend != ExecutionBackend::Vulkan
        || entries.size() < 2
        || !attention_operator.attention)
        return false;

    std::vector<AttentionBatchEntry_vulkan> device_entries;
    device_entries.reserve(entries.size());
    for (AttentionBatchEntry& entry : entries)
    {
        if (!entry.cache || !entry.scratch || !entry.hidden || !entry.output)
        {
            return Error{
                ErrorCode::InvalidArgument,
                "Attention batch entry is incomplete"};
        }
        if (entry.cache->transaction.active || entry.cache->token_count == 0)
            return false;
        device_entries.push_back({entry.position_offset,
                                  entry.cache,
                                  entry.hidden,
                                  entry.output});
    }

    const AttentionBatchResult_vulkan result = attention_operator.attention->forward_batch(device_entries);
    if (result == AttentionBatchResult_vulkan::Executed)
        return true;
    if (result == AttentionBatchResult_vulkan::Failed)
    {
        return Error{
            ErrorCode::InternalError,
            "Vulkan Attention batch failed after device KV state became authoritative"};
    }
    return false;
}

Result<void> forward_attention(const WeightStore& weights,
                               const CompiledOperatorTable& operators,
                               const AttentionBlockPlan& plan,
                               ExecutionBackend backend,
                               uint64_t position_offset,
                               LayerCache& cache,
                               AttentionScratch& scratch,
                               const ActivationBuffer& hidden,
                               ActivationBuffer& output,
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
        if (attention_operator.attention->forward(position_offset,
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
                && attention_operator.attention->materialize_device_cache(cache))
            {
                // The failed dispatch restored the CPU cache; use CPU Attention.
            }
            else
            {
                return Error{
                    ErrorCode::InternalError,
                    "Vulkan Attention failed and its device KV cache could not be materialized"};
            }
        }
    }

    configure_cache(cache, plan.kv_head_count * plan.head_dimension,
                    plan.kv_cache_dtype);
    const ActivationBuffer* normalized = &hidden;
    if (plan.pre_attention_norm_weight != invalid_tensor_handle)
    {
        forward_rms_norm(hidden,
                         weights.at(plan.pre_attention_norm_weight),
                         plan.norm_epsilon,
                         scratch.normalized,
                         plan.norm_weight_offset);
        normalized = &scratch.normalized;
    }
    auto qsa_status = project_and_append_qsa_keys(weights, operators, plan, position_offset,
                                                  *normalized, cache, scratch,
                                                  optimization_flags);
    if (!qsa_status)
        return qsa_status.error();
    ActivationBuffer& query = scratch.query;
    ActivationBuffer& key = scratch.key;
    ActivationBuffer& value = scratch.value;
    ActivationBuffer& fused_qkv = scratch.fused_qkv;
    const CompiledOperator& fused_qkv_gate_operator = operators.at(plan.fused_qkv_gate_bfloat16_operator);
    const CompiledOperator& fused_qkv_operator = operators.at(plan.fused_qkv_operator);
    const bool fused_output_gate = backend == ExecutionBackend::Vulkan
                                   && fused_qkv_gate_operator.bfloat16
                                   && fused_qkv_gate_operator.bfloat16->forward(*normalized,
                                                                                fused_qkv);
    const bool fused_projection = fused_output_gate
                                  || (backend == ExecutionBackend::Vulkan
                                      && fused_qkv_operator.bfloat16
                                      && fused_qkv_operator.bfloat16->forward(*normalized,
                                                                              fused_qkv))
                                  || (backend == ExecutionBackend::Vulkan
                                      && fused_qkv_operator.linear
                                      && fused_qkv_operator.linear->forward(*normalized,
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
                std::copy_n(source,
                            query_columns,
                            scratch.gate.row(token_index));
            }
        }
    }
    else
    {
        attention_linear_into(weights, operators, plan.query_weight, plan.query_bias, *normalized, query, optimization_flags, backend);
        attention_linear_into(weights, operators, plan.key_weight, plan.key_bias, *normalized, key, optimization_flags, backend);
        attention_linear_into(weights, operators, plan.value_weight, plan.value_bias, *normalized, value, optimization_flags, backend);
    }

    if (has_flag(plan.flags, AttentionBlockQueryKeyNorm))
    {
        apply_head_rms_norm(query,
                            plan.head_count,
                            plan.head_dimension,
                            weights.at(plan.query_norm_weight),
                            plan.norm_epsilon,
                            plan.norm_weight_offset);
        apply_head_rms_norm(key,
                            plan.kv_head_count,
                            plan.head_dimension,
                            weights.at(plan.key_norm_weight),
                            plan.norm_epsilon,
                            plan.norm_weight_offset);
    }

    const uint32_t rope_dimension = plan.rope_head_dimension == 0 ? plan.head_dimension : plan.rope_head_dimension;
    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
    {
        const uint64_t position = position_offset + token_index;
        if (cached_rope_coefficients_enabled(optimization_flags))
        {
            prepare_rope_coefficients(rope_dimension,
                                      position,
                                      plan,
                                      scratch.rope_cosine,
                                      scratch.rope_sine);
            for (uint32_t head = 0; head < plan.head_count; ++head)
            {
                apply_prepared_rope(query.row(token_index) + head * plan.head_dimension,
                                    rope_dimension,
                                    scratch.rope_cosine,
                                    scratch.rope_sine);
            }
            for (uint32_t head = 0; head < plan.kv_head_count; ++head)
            {
                apply_prepared_rope(key.row(token_index) + head * plan.head_dimension,
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
    append_cache(cache, plan.kv_cache_dtype, key, value);
    qsa_status = prepare_qsa_selection(weights, plan, position_offset, cache, scratch);
    if (!qsa_status)
        return qsa_status.error();
    scaled_dot_product_attention_into(plan,
                                      plan.sinks == invalid_tensor_handle ? nullptr : &weights.at(plan.sinks),
                                      position_offset,
                                      query,
                                      cache,
                                      scratch.attention,
                                      scratch,
                                      scratch.qsa_selected_offsets,
                                      scratch.qsa_selected_indices,
                                      optimization_flags);
    if (has_flag(plan.flags, AttentionBlockOutputGate))
    {
        if (!fused_output_gate)
        {
            attention_linear_into(weights,
                                  operators,
                                  plan.output_gate_weight,
                                  invalid_tensor_handle,
                                  *normalized,
                                  scratch.gate,
                                  optimization_flags,
                                  backend);
        }
        for (size_t token_index = 0; token_index < scratch.attention.rows(); ++token_index)
        {
            float* attention_row = scratch.attention.row(token_index);
            const float* gate_row = scratch.gate.row(token_index);
            if (has_flag(optimization_flags, OptimizationCpuFastSilu))
            {
                float_sigmoid_mul(attention_row,
                                  gate_row,
                                  attention_row,
                                  scratch.attention.columns());
            }
            else
            {
                for (uint32_t column = 0; column < scratch.attention.columns(); ++column)
                    attention_row[column] *= 1.0f / (1.0f + float_approximate_exp(-gate_row[column]));
            }
        }
    }
    ActivationBuffer& projected = (&output == &hidden || &output == &scratch.attention)
                                      ? scratch.projected
                                      : output;
    attention_linear_into(weights,
                          operators,
                          plan.output_weight,
                          plan.output_bias,
                          scratch.attention,
                          projected,
                          optimization_flags,
                          backend);
    if (!has_flag(plan.flags, AttentionBlockExternalResidual))
        add_batch_inplace(projected, hidden);
    if (&projected != &output)
        output.swap(projected);
    trim_sliding_cache(cache, plan);
    return {};
}

} // namespace moe
} // namespace ncnn
