#include "attention_internal.h"

#include "bfloat16.h"
#include "engine/cpu.h"
#include "fastmath.h"
#include "ops.h"
#include "vector.h"
#include "ncnn/moe/option.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

void sdpa_into(const AttentionBlockPlan& plan, const TensorData* sinks, uint64_t position_offset, const ActivationBuffer& query,
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
    const float scale = plan.attn_scale > 0.0f
                            ? plan.attn_scale
                            : 1.0f / std::sqrt(static_cast<float>(head_dimension));
    const bool has_selected_keys = selected_offsets.size() == query.rows() + 1;
    // Decode can use the short SDPA path only when the cache has no future key.
    const bool decode_all_keys_valid = query.rows() == 1
                                       && cache.token_count != 0
                                       && cache.start_position <= position_offset
                                       && cache.token_count - 1 <= position_offset - cache.start_position
                                       && (plan.sliding_window == 0
                                           || cache.token_count <= plan.sliding_window)
                                       && !has_selected_keys;
    const bool valid_capacity = cache.columns > 0
                                && cache.capacity_tokens > 0
                                && cache.capacity_tokens
                                       <= std::numeric_limits<size_t>::max() / cache.columns;
    const size_t cache_capacity_elements = valid_capacity
                                               ? static_cast<size_t>(cache.capacity_tokens) * cache.columns
                                               : 0;
    const bool valid_bf16 = cache.dtype == DType::BFloat16
                            && valid_capacity
                            && cache.first_slot < cache.capacity_tokens
                            && cache.token_count <= cache.capacity_tokens
                            && cache.bfloat16_keys.size() >= cache_capacity_elements
                            && cache.bfloat16_values.size() >= cache_capacity_elements;
    const bool use_flash = has_flag(optimization_flags, OptimizationCpuFlashAttention)
                           && !has_selected_keys
                           && cache.token_count >= 64
                           && (query.rows() > 1
                               || (query.rows() == 1
                                   && cache.token_count < 512
                                   && decode_all_keys_valid
                                   && has_flag(optimization_flags,
                                               OptimizationCpuBf16DirectAttention)
                                   && valid_bf16));
    output.reset(query.rows(), head_count * head_dimension, !use_flash);
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
    const float* key_values = nullptr;
    const float* value_values = nullptr;
    const uint16_t* bfloat16_key_values = nullptr;
    const uint16_t* bfloat16_value_values = nullptr;
    const bool direct_bfloat16 = (query.rows() == 1 || has_selected_keys
                                  || (use_flash && valid_bf16))
                                 && has_flag(optimization_flags, OptimizationCpuBf16DirectAttention)
                                 && cache.dtype == DType::BFloat16
                                 && valid_capacity
                                 && cache.bfloat16_keys.size() >= cache_capacity_elements
                                 && cache.bfloat16_values.size() >= cache_capacity_elements;
    const bool direct_bfloat16_contiguous = direct_bfloat16
                                            && cache.first_slot <= cache.capacity_tokens
                                            && cache.token_count <= cache.capacity_tokens - cache.first_slot;
    const bool direct_bfloat16_ring = use_flash
                                      && direct_bfloat16
                                      && !direct_bfloat16_contiguous;
    const bool bf16_pv = use_flash && query.rows() == 1 && direct_bfloat16;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    constexpr bool f16_hw = true;
#else
    constexpr bool f16_hw = false;
#endif
    const bool f16 = use_flash
                     && query.rows() > 1
                     && has_flag(optimization_flags, OptimizationCpuFlashAttentionF16)
                     && f16_hw
                     && !has_selected_keys
                     && (cache.dtype == DType::Float32
                         || (cache.dtype == DType::BFloat16
                             && valid_capacity
                             && cache.bfloat16_keys.size() >= cache_capacity_elements
                             && cache.bfloat16_values.size() >= cache_capacity_elements))
                     && cache.token_count != 0
                     && cache.columns != 0;
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
    else if (!f16 && (!direct_bfloat16 || (use_flash && query.rows() > 1)))
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
    const int head_threads = !split_kv_enabled && query.rows() == 1 && !has_selected_keys
                                     && head_count >= 4 && cache.token_count >= 64
                                 ? std::min<uint32_t>(static_cast<uint32_t>(cpu_linear_team_size(2 * split_kv_work, cache.dtype)), head_count)
                                 : 1;

    if (f16)
    {
        // Convert KV once into head-major FP16 tiles; each flash job reads
        // one head's K/V rows instead of striding over all heads.
        const size_t elems = static_cast<size_t>(cache.token_count) * cache.columns;
        scratch.f16_key.resize(elems);
        scratch.f16_val.resize(elems);
        const uint32_t kv_heads = head_dimension != 0 ? cache.columns / head_dimension : 0;
        const bool bf16_src = cache.dtype == DType::BFloat16;
        for (uint64_t token_index = 0; token_index < cache.token_count; ++token_index)
        {
            const uint64_t slot = cache_slot(cache, token_index);
            for (uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head)
            {
                const size_t destination = (static_cast<size_t>(kv_head) * cache.token_count + token_index)
                                           * head_dimension;
                const size_t source = static_cast<size_t>(slot) * cache.columns
                                      + static_cast<size_t>(kv_head) * head_dimension;
                if (bf16_src)
                {
                    bf16_to_f16(scratch.f16_key.data() + destination,
                                cache.bfloat16_keys.data() + source,
                                head_dimension);
                    bf16_to_f16(scratch.f16_val.data() + destination,
                                cache.bfloat16_values.data() + source,
                                head_dimension);
                }
                else
                {
                    f32_to_f16(scratch.f16_key.data() + destination,
                               cache.keys.data() + source,
                               head_dimension);
                    f32_to_f16(scratch.f16_val.data() + destination,
                               cache.values.data() + source,
                               head_dimension);
                }
            }
        }
    }
    const bool gqa_decode = query.rows() == 1 && split_kv_enabled && heads_per_group > 1
                            && cache.token_count > 6144
                            && !has_selected_keys && sinks == nullptr && direct_bfloat16
                            && direct_bfloat16_contiguous;
    if (gqa_decode)
    {
        const uint32_t batch_heads = cache.token_count >= 32768 ? 2 : heads_per_group;
        const uint64_t chunk_size = cache.token_count <= 4096   ? 256
                                    : cache.token_count <= 8192 ? 1024
                                                                : 512;
        const uint32_t chunks = static_cast<uint32_t>((cache.token_count + chunk_size - 1) / chunk_size);
        const uint32_t head_groups = (heads_per_group + batch_heads - 1) / batch_heads;
        const uint32_t job_count = kv_head_count * head_groups * chunks;
        const uint32_t team = static_cast<uint32_t>(std::min<int>(attention_team_size, job_count));
        const size_t score_stride = static_cast<size_t>(batch_heads) * chunk_size;
        if (team == 0 || score_stride > scratch.workspace.max_size() / team)
            throw std::length_error("GQA decode workspace is too large");
        scratch.workspace.resize(static_cast<size_t>(team) * score_stride);
        scratch.flash_partial_max.assign(static_cast<size_t>(head_count) * chunks,
                                         -std::numeric_limits<float>::infinity());
        scratch.flash_partial_sum.assign(static_cast<size_t>(head_count) * chunks, 0.0f);
        scratch.flash_partial_output.assign(static_cast<size_t>(head_count) * chunks * head_dimension, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static, 1) num_threads(team)
#endif
        for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
        {
            const uint32_t thread_id =
#if defined(_OPENMP)
                static_cast<uint32_t>(omp_get_thread_num());
#else
                0;
#endif
            const uint32_t per_kv_head = head_groups * chunks;
            const uint32_t kv_head = static_cast<uint32_t>(job) / per_kv_head;
            const uint32_t within_kv_head = static_cast<uint32_t>(job) % per_kv_head;
            const uint32_t head_group = within_kv_head / chunks;
            const uint32_t chunk = within_kv_head % chunks;
            const uint64_t key_begin = static_cast<uint64_t>(chunk) * chunk_size;
            const uint32_t key_count = static_cast<uint32_t>(std::min<uint64_t>(chunk_size,
                                                                                cache.token_count - key_begin));
            const uint32_t head_offset = head_group * batch_heads;
            const uint32_t group_heads = std::min(batch_heads, heads_per_group - head_offset);
            const uint32_t head_begin = kv_head * heads_per_group + head_offset;
            float* scores = scratch.workspace.data() + static_cast<size_t>(thread_id) * score_stride;
            const uint16_t* keys = cache.bfloat16_keys.data() + static_cast<size_t>(cache_slot(cache, key_begin)) * cache.columns
                                   + static_cast<size_t>(kv_head) * head_dimension;
            const float* query_group = query.row(0) + static_cast<size_t>(head_begin) * head_dimension;
            for (uint32_t row = 0; row < group_heads; row += 4)
            {
                const uint32_t rows = std::min(4u, group_heads - row);
                bfloat16_gemm_4x8(keys,
                                  cache.columns,
                                  query_group + static_cast<size_t>(row) * head_dimension,
                                  head_dimension,
                                  head_dimension,
                                  key_count,
                                  rows,
                                  scores + static_cast<size_t>(row) * chunk_size,
                                  chunk_size);
            }
            const uint16_t* values = cache.bfloat16_values.data() + static_cast<size_t>(cache_slot(cache, key_begin)) * cache.columns
                                     + static_cast<size_t>(kv_head) * head_dimension;
            for (uint32_t head = 0; head < group_heads; ++head)
            {
                float* row_scores = scores + static_cast<size_t>(head) * chunk_size;
                for (uint32_t key = 0; key < key_count; ++key)
                {
                    const bool visible = key_visible(cache.start_position + key_begin + key,
                                                     position_offset, plan.sliding_window);
                    row_scores[key] = visible ? row_scores[key] * scale
                                              : -std::numeric_limits<float>::infinity();
                }
                const float maximum = f32_scale_max(row_scores, 1.0f, key_count);
                const size_t partial = static_cast<size_t>(head_begin + head) * chunks + chunk;
                // A fully masked chunk has no finite maximum; exp(-inf - -inf)
                // is NaN, so leave a neutral partial (output is zeroed) and let
                // the reduction weight it by zero.
                if (!std::isfinite(maximum))
                {
                    scratch.flash_partial_max[partial] = -std::numeric_limits<float>::infinity();
                    scratch.flash_partial_sum[partial] = 0.0f;
                    continue;
                }
                const float normalizer = f32_softmax(row_scores, key_count, maximum);
                scratch.flash_partial_max[partial] = maximum;
                scratch.flash_partial_sum[partial] = normalizer;
                float* output_partial = scratch.flash_partial_output.data() + partial * head_dimension;
                pv_bf16(output_partial,
                        values,
                        cache.columns,
                        row_scores,
                        key_count,
                        nullptr,
                        nullptr,
                        0,
                        head_dimension);
            }
        }
        for (uint32_t head = 0; head < head_count; ++head)
        {
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t chunk = 0; chunk < chunks; ++chunk)
                maximum = std::max(maximum, scratch.flash_partial_max[static_cast<size_t>(head) * chunks + chunk]);
            float normalizer = 0.0f;
            float* destination = output.row(0) + static_cast<size_t>(head) * head_dimension;
            std::fill(destination, destination + head_dimension, 0.0f);
            for (uint32_t chunk = 0; chunk < chunks; ++chunk)
            {
                const size_t partial = static_cast<size_t>(head) * chunks + chunk;
                const float weight = std::isfinite(maximum)
                                         ? float_approximate_exp(scratch.flash_partial_max[partial] - maximum)
                                         : 0.0f;
                normalizer += weight * scratch.flash_partial_sum[partial];
                const float* source = scratch.flash_partial_output.data() + partial * head_dimension;
                for (uint32_t column = 0; column < head_dimension; ++column)
                    destination[column] += weight * source[column];
            }
            if (normalizer > 0.0f)
                float_scale_inplace(destination, 1.0f / normalizer, head_dimension);
            else
                std::fill(destination, destination + head_dimension, 0.0f);
        }
        return;
    }

    if (use_flash || split_kv_enabled)
    {
        // Keep adjacent workers off a shared cache line, including 128-byte lines.
        constexpr size_t worker_padding = 128 / sizeof(float);
        const auto valid_key = [&](size_t query_index, uint64_t key_index) {
            if (decode_all_keys_valid)
                return true;
            const uint64_t query_position = position_offset + query_index;
            return key_visible(cache.start_position + key_index, query_position, plan.sliding_window);
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

        if (use_flash)
        {
            const int num_threads = query.rows() == 1 ? head_threads : attention_team_size;
            constexpr uint32_t query_tile_size = 64;
            const uint32_t rows = query.rows() == 1 ? 1 : query_tile_size;
            const uint64_t query_group_count = (query.rows() + rows - 1) / rows;
            const uint64_t job_count = query_group_count * head_count;
            constexpr uint64_t key_tile_size = 64;
            constexpr uint32_t key_gemm_tile_size = 8;
            const auto process_flash_group = [&](uint64_t job,
                                                 float* running,
                                                 float* tile_output,
                                                 float* tile_logits,
                                                 uint16_t* q16,
                                                 uint16_t* s16) {
                const size_t query_group = static_cast<size_t>(job / head_count) * rows;
                const uint32_t query_head = static_cast<uint32_t>(job % head_count);
                const uint32_t query_count = static_cast<uint32_t>(std::min<size_t>(rows,
                                                                                    query.rows() - query_group));
                const uint64_t last_position = position_offset + query_group + query_count - 1;
                const bool skip_future = last_position >= position_offset
                                         && cache.token_count - 1
                                                <= std::numeric_limits<uint64_t>::max() - cache.start_position;
                const uint32_t kv_head = query_head / heads_per_group;
                const float* query_values = query.row(query_group) + static_cast<size_t>(query_head) * head_dimension;
                if (f16)
                {
                    for (uint32_t query_offset = 0; query_offset < query_count; ++query_offset)
                    {
                        f32_to_f16(q16 + static_cast<size_t>(query_offset) * head_dimension,
                                   query.row(query_group + query_offset)
                                       + static_cast<size_t>(query_head) * head_dimension,
                                   head_dimension);
                    }
                }
                std::array<float, query_tile_size> maximum = {};
                std::array<float, query_tile_size> normalizer = {};
                std::array<int64_t, query_tile_size> first_key = {};
                std::array<int64_t, query_tile_size> last_key = {};
                for (uint32_t query_offset = 0; query_offset < query_count; ++query_offset)
                {
                    std::fill(running + static_cast<size_t>(query_offset) * head_dimension,
                              running + static_cast<size_t>(query_offset + 1) * head_dimension,
                              0.0f);
                    maximum[query_offset] = sink_value(query_head);
                    normalizer[query_offset] = sinks ? 1.0f : 0.0f;
                    const int64_t pos = static_cast<int64_t>(position_offset) + static_cast<int64_t>(query_group)
                                        + static_cast<int64_t>(query_offset)
                                        - static_cast<int64_t>(cache.start_position);
                    if (pos >= 0)
                    {
                        last_key[query_offset] = std::min<int64_t>(static_cast<int64_t>(cache.token_count), pos + 1);
                        if (plan.sliding_window != 0)
                            first_key[query_offset] = std::max<int64_t>(0, pos - static_cast<int64_t>(plan.sliding_window) + 1);
                    }
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
                                float* packed_keys = tile_logits + rows * key_tile_size;
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
                        else if (!f16)
                        {
                            // A BF16 cache with the F16 path leaves key_values unset;
                            // only the F16 tile loader below may read this row.
                            tile_keys = float32_vector(key_values, key_begin, kv_head);
                            if (direct_float32_ring && key_begin < ring_tail && key_count > ring_tail - key_begin)
                            {
                                // Keep the GEMM tile intact: smaller tail kernels can change rounding.
                                float* packed_keys = tile_logits + rows * key_tile_size;
                                for (uint32_t key_offset = 0; key_offset < key_count; ++key_offset)
                                    std::copy_n(float32_vector(key_values, key_begin + key_offset, kv_head),
                                                head_dimension, packed_keys + static_cast<size_t>(key_offset) * head_dimension);
                                tile_keys = packed_keys;
                                key_stride = head_dimension;
                            }
                        }
                        if (f16)
                        {
                            const uint16_t* f16_key_tile = scratch.f16_key.data()
                                                           + (static_cast<size_t>(kv_head) * cache.token_count
                                                              + key_begin)
                                                                 * head_dimension;
                            for (uint32_t sub = 0; sub < query_count; sub += 4)
                            {
                                const uint32_t sub_count = std::min(4u, query_count - sub);
                                f16_gemm_4x8(f16_key_tile,
                                             head_dimension,
                                             q16 + static_cast<size_t>(sub) * head_dimension,
                                             head_dimension,
                                             head_dimension,
                                             key_count,
                                             sub_count,
                                             tile_logits + static_cast<size_t>(sub) * key_tile_size + tile_key_offset,
                                             key_tile_size);
                            }
                        }
                        else
                        {
                            for (uint32_t sub = 0; sub < query_count; sub += 4)
                            {
                                const uint32_t sub_count = std::min(4u, query_count - sub);
                                float* sub_logits = tile_logits
                                                    + static_cast<size_t>(sub) * key_tile_size
                                                    + tile_key_offset;
                                if (tile_keys_bf16)
                                {
                                    bfloat16_gemm_4x8(tile_keys_bf16,
                                                      key_stride,
                                                      query_values + static_cast<size_t>(sub) * query.columns(),
                                                      query.columns(),
                                                      head_dimension,
                                                      key_count,
                                                      sub_count,
                                                      sub_logits,
                                                      key_tile_size);
                                }
                                else
                                {
                                    float_gemm_4x8(tile_keys,
                                                   key_stride,
                                                   query_values + static_cast<size_t>(sub) * query.columns(),
                                                   query.columns(),
                                                   head_dimension,
                                                   key_count,
                                                   sub_count,
                                                   sub_logits,
                                                   key_tile_size);
                                }
                            }
                        }
                        for (uint32_t query_offset = 0; query_offset < query_count; ++query_offset)
                        {
                            const int64_t first_valid = first_key[query_offset];
                            const int64_t last_valid = last_key[query_offset];
                            const int64_t tile_begin_index = static_cast<int64_t>(key_begin);
                            const int64_t tile_end_index = tile_begin_index + static_cast<int64_t>(key_count);
                            // Clamp to this tile so the fill cannot touch
                            // neighbouring scores.
                            const int64_t valid_begin = std::min(tile_end_index,
                                                                 std::max(tile_begin_index, first_valid));
                            const int64_t valid_end = std::max(valid_begin,
                                                               std::min(tile_end_index, last_valid));
                            const size_t tile_base = static_cast<size_t>(tile_begin);
                            const size_t begin_offset = static_cast<size_t>(tile_begin_index - tile_base);
                            const size_t valid_offset = static_cast<size_t>(valid_begin - tile_base);
                            const size_t end_offset = static_cast<size_t>(tile_end_index - tile_base);
                            float* row_scores = tile_logits
                                                + static_cast<size_t>(query_offset) * key_tile_size;
                            // Zero-length fills/scales are no-ops.
                            std::fill(row_scores + begin_offset, row_scores + valid_offset,
                                      -std::numeric_limits<float>::infinity());
                            std::fill(row_scores + static_cast<size_t>(valid_end - tile_base),
                                      row_scores + end_offset, -std::numeric_limits<float>::infinity());
                            tile_maximum[query_offset] = std::max(tile_maximum[query_offset],
                                                                  f32_scale_max(row_scores + valid_offset,
                                                                                scale,
                                                                                static_cast<uint32_t>(valid_end - valid_begin)));
                        }
                    }
                    if (f16)
                    {
                        const uint16_t* value_tile = scratch.f16_val.data()
                                                     + (static_cast<size_t>(kv_head) * cache.token_count
                                                        + tile_begin)
                                                           * head_dimension;
                        const auto prepare_f16_row = [&](uint32_t query_offset,
                                                         float& old_scale,
                                                         float& tile_scale,
                                                         float& tile_normalizer,
                                                         float& new_maximum) {
                            float* scores = tile_logits
                                            + static_cast<size_t>(query_offset) * key_tile_size;
                            if (!std::isfinite(tile_maximum[query_offset]))
                            {
                                std::fill(scores, scores + tile_width, 0.0f);
                                return false;
                            }
                            const float score_maximum = tile_maximum[query_offset];
                            tile_normalizer = f32_softmax(scores, tile_width, score_maximum);
                            new_maximum = std::max(maximum[query_offset], score_maximum);
                            old_scale = std::isfinite(maximum[query_offset])
                                            ? float_approximate_exp(maximum[query_offset] - new_maximum)
                                            : 0.0f;
                            tile_scale = float_approximate_exp(score_maximum - new_maximum);
                            return true;
                        };
                        uint32_t query_offset = 0;
                        for (; query_offset + 4 <= query_count; query_offset += 4)
                        {
                            std::array<float, 4> old_scale = {};
                            std::array<float, 4> tile_scale = {};
                            std::array<float, 4> tile_normalizer = {};
                            std::array<float, 4> new_maximum = {};
                            std::array<bool, 4> valid = {};
                            for (uint32_t row = 0; row < 4; ++row)
                                valid[row] = prepare_f16_row(query_offset + row, old_scale[row],
                                                             tile_scale[row], tile_normalizer[row],
                                                             new_maximum[row]);
                            float* tile_group = tile_output + static_cast<size_t>(query_offset) * head_dimension;
                            f16_pv4(tile_group,
                                    value_tile,
                                    head_dimension,
                                    tile_logits + static_cast<size_t>(query_offset) * key_tile_size,
                                    tile_logits + static_cast<size_t>(query_offset + 1) * key_tile_size,
                                    tile_logits + static_cast<size_t>(query_offset + 2) * key_tile_size,
                                    tile_logits + static_cast<size_t>(query_offset + 3) * key_tile_size,
                                    tile_width,
                                    head_dimension);
                            for (uint32_t row = 0; row < 4; ++row)
                            {
                                if (!valid[row])
                                    continue;
                                float_scale_add(running + static_cast<size_t>(query_offset + row) * head_dimension,
                                                old_scale[row],
                                                tile_group + static_cast<size_t>(row) * head_dimension,
                                                tile_scale[row],
                                                head_dimension);
                                normalizer[query_offset + row] = normalizer[query_offset + row] * old_scale[row]
                                                                 + tile_normalizer[row] * tile_scale[row];
                                maximum[query_offset + row] = new_maximum[row];
                            }
                        }
                        for (; query_offset < query_count; ++query_offset)
                        {
                            float old_scale = 0.0f;
                            float tile_scale = 0.0f;
                            float tile_normalizer = 0.0f;
                            float new_maximum = 0.0f;
                            if (!prepare_f16_row(query_offset, old_scale, tile_scale,
                                                 tile_normalizer, new_maximum))
                                continue;
                            float* row_scores = tile_logits
                                                + static_cast<size_t>(query_offset) * key_tile_size;
                            f32_to_f16(s16, row_scores, tile_width);
                            float* row_output = tile_output
                                                + static_cast<size_t>(query_offset) * head_dimension;
                            f16_pv(row_output,
                                   value_tile,
                                   head_dimension,
                                   s16,
                                   tile_width,
                                   head_dimension);
                            float_scale_add(running + static_cast<size_t>(query_offset) * head_dimension,
                                            old_scale,
                                            row_output,
                                            tile_scale,
                                            head_dimension);
                            normalizer[query_offset] = normalizer[query_offset] * old_scale
                                                       + tile_normalizer * tile_scale;
                            maximum[query_offset] = new_maximum;
                        }
                    }
                    else
                    {
                        for (uint32_t query_offset = 0; query_offset < query_count; ++query_offset)
                        {
                            if (!std::isfinite(tile_maximum[query_offset]))
                                continue;
                            float* query_tile_output = running + static_cast<size_t>(query_offset) * head_dimension;
                            if (!bf16_pv)
                                query_tile_output = tile_output + static_cast<size_t>(query_offset) * head_dimension;
                            float tile_normalizer = 0.0f;
                            float* tile_scores = tile_logits + static_cast<size_t>(query_offset) * key_tile_size;
                            float new_maximum = 0.0f;
                            float old_scale = 0.0f;
                            if (bf16_pv)
                            {
                                new_maximum = std::max(maximum[query_offset], tile_maximum[query_offset]);
                                old_scale = std::isfinite(maximum[query_offset])
                                                ? float_approximate_exp(maximum[query_offset] - new_maximum)
                                                : 0.0f;
                                if (normalizer[query_offset] != 0.0f && old_scale != 1.0f)
                                    float_scale_inplace(query_tile_output, old_scale, head_dimension);
                                normalizer[query_offset] *= old_scale;
                            }
                            const float score_maximum = bf16_pv ? new_maximum : tile_maximum[query_offset];
                            for (uint32_t key_offset = 0; key_offset < tile_width; ++key_offset)
                            {
                                const float score = tile_scores[key_offset];
                                tile_scores[key_offset] = std::isfinite(score)
                                                              ? score - score_maximum
                                                              : std::numeric_limits<float>::quiet_NaN();
                            }
                            float_exp_inplace(tile_scores, tile_width);
                            for (uint32_t key_offset = 0; key_offset < tile_width; ++key_offset)
                            {
                                const float probability = tile_scores[key_offset];
                                if (std::isfinite(probability))
                                {
                                    tile_normalizer += probability;
                                }
                                else
                                {
                                    // Fold masked/future keys to an additive zero so
                                    // the probability-value kernel can run branch-free.
                                    tile_scores[key_offset] = 0.0f;
                                }
                            }
                            if (bf16_pv)
                            {
                                const bool crosses_ring = direct_bfloat16_ring
                                                          && tile_begin < ring_tail
                                                          && tile_end > ring_tail;
                                const uint64_t middle = crosses_ring ? ring_tail : tile_end;
                                const auto value_rows = [&](uint64_t key_index) {
                                    const uint64_t slot = direct_bfloat16_contiguous || key_index < ring_tail
                                                              ? cache.first_slot + key_index
                                                              : key_index - ring_tail;
                                    return bfloat16_value_values
                                           + static_cast<size_t>(slot) * cache.columns
                                           + static_cast<size_t>(kv_head) * head_dimension;
                                };
                                const uint16_t* first_values = value_rows(tile_begin);
                                const uint16_t* second_values = crosses_ring ? value_rows(middle) : first_values;
                                pv_bf16(query_tile_output,
                                        first_values,
                                        cache.columns,
                                        tile_scores,
                                        static_cast<uint32_t>(middle - tile_begin),
                                        second_values,
                                        tile_scores + static_cast<size_t>(middle - tile_begin),
                                        static_cast<uint32_t>(tile_end - middle),
                                        head_dimension);
                            }
                            else
                            {
                                const size_t value_offset = static_cast<size_t>(kv_head) * head_dimension;
                                if (f16)
                                {
                                    f32_to_f16(s16, tile_scores, tile_width);
                                    f16_pv(query_tile_output,
                                           scratch.f16_val.data()
                                               + (static_cast<size_t>(kv_head) * cache.token_count
                                                  + tile_begin)
                                                     * head_dimension,
                                           head_dimension,
                                           s16,
                                           tile_width,
                                           head_dimension);
                                }
                                else
                                {
                                    std::fill(query_tile_output,
                                              query_tile_output + head_dimension,
                                              0.0f);
                                    const bool crosses_ring = direct_float32_ring
                                                              && tile_begin < ring_tail
                                                              && tile_end > ring_tail;
                                    const uint64_t middle = crosses_ring ? ring_tail : tile_end;
                                    const float* first_values = float32_vector(value_values, tile_begin, 0) + value_offset;
                                    const float* second_values = crosses_ring
                                                                     ? float32_vector(value_values, middle, 0) + value_offset
                                                                     : first_values;
                                    pv_f32(query_tile_output,
                                           first_values,
                                           cache.columns,
                                           tile_scores,
                                           static_cast<uint32_t>(middle - tile_begin),
                                           second_values,
                                           tile_scores + static_cast<size_t>(middle - tile_begin),
                                           static_cast<uint32_t>(tile_end - middle),
                                           head_dimension);
                                }
                            }
                            if (bf16_pv)
                            {
                                normalizer[query_offset] += tile_normalizer;
                                maximum[query_offset] = new_maximum;
                            }
                            else
                            {
                                new_maximum = std::max(maximum[query_offset], tile_maximum[query_offset]);
                                old_scale = std::isfinite(maximum[query_offset])
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
            const size_t running_size = static_cast<size_t>(rows) * head_dimension;
            const size_t output_size = bf16_pv ? 0 : running_size;
            const size_t tile_logits_size = static_cast<size_t>(rows) * key_tile_size;
            const uint64_t worker_elements = (static_cast<uint64_t>(rows)
                                              + (bf16_pv ? 0 : rows)
                                              + ((direct_float32_ring || direct_bfloat16_ring)
                                                     ? key_gemm_tile_size
                                                     : 0))
                                                 * head_dimension
                                             + tile_logits_size + worker_padding;
            if (worker_elements > scratch.workspace.max_size() / num_threads)
                throw std::length_error("attention workspace is too large");
            const size_t worker_stride = static_cast<size_t>(worker_elements);
            const size_t worker_size = static_cast<size_t>(num_threads) * worker_stride;
            if (scratch.workspace.size() < worker_size)
                scratch.workspace.resize(worker_size);
            const size_t q16_elems = static_cast<size_t>(rows) * head_dimension;
            constexpr size_t f16_align = 128 / sizeof(uint16_t);
            const size_t f16_stride = (q16_elems + key_tile_size + f16_align) & ~(f16_align - 1);
            uint16_t* f16_work = nullptr;
            if (f16)
            {
                const size_t f16_size = static_cast<size_t>(num_threads) * f16_stride;
                if (f16_size > scratch.f16_work.max_size() - f16_align)
                    throw std::length_error("attention f16 workspace is too large");
                if (scratch.f16_work.size() < f16_size + f16_align)
                    scratch.f16_work.resize(f16_size + f16_align);
                const uintptr_t raw = reinterpret_cast<uintptr_t>(scratch.f16_work.data());
                f16_work = reinterpret_cast<uint16_t*>((raw + 127) & ~static_cast<uintptr_t>(127));
            }
#if defined(_OPENMP)
            if (num_threads > 1)
            {
#pragma omp parallel num_threads(num_threads)
                {
                    cpu_bind_thread();
                    const size_t worker_offset = static_cast<size_t>(omp_get_thread_num()) * worker_stride;
                    float* running = scratch.workspace.data() + worker_offset;
                    float* tile_output = running + (bf16_pv ? 0 : running_size);
                    float* tile_logits = running + running_size + output_size;
                    uint16_t* q16 = nullptr;
                    uint16_t* s16 = nullptr;
                    if (f16_work != nullptr)
                    {
                        uint16_t* slot = f16_work + static_cast<size_t>(omp_get_thread_num()) * f16_stride;
                        q16 = slot;
                        s16 = slot + q16_elems;
                    }
#pragma omp for schedule(static, 1)
                    for (int64_t job = 0; job < static_cast<int64_t>(job_count); ++job)
                        process_flash_group(static_cast<uint64_t>(job), running, tile_output, tile_logits,
                                            q16, s16);
                }
            }
            else
#endif
            {
                float* running = scratch.workspace.data();
                float* tile_output = running + (bf16_pv ? 0 : running_size);
                float* tile_logits = running + running_size + output_size;
                uint16_t* q16 = f16_work;
                uint16_t* s16 = f16_work == nullptr ? nullptr : f16_work + q16_elems;
                for (uint64_t job = 0; job < job_count; ++job)
                    process_flash_group(job, running, tile_output, tile_logits, q16, s16);
            }
            return;
        }

        const uint32_t split_team_size = static_cast<uint32_t>(attention_team_size);
        const uint64_t key_chunk_size = cache.token_count >= 8192 ? 1024 : 256;
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
            const uint32_t kv_head = query_head / heads_per_group;
            local_maximum = -std::numeric_limits<float>::infinity();
            local_normalizer = 0.0f;
            std::fill(local_output, local_output + head_dimension, 0.0f);
            if (direct_bfloat16 && direct_bfloat16_contiguous)
            {
                const uint32_t count = static_cast<uint32_t>(end - begin);
                const uint16_t* keys = cache.bfloat16_keys.data() + static_cast<size_t>(cache_slot(cache, begin)) * cache.columns
                                       + static_cast<size_t>(kv_head) * head_dimension;
                bfloat16_gemm_4x8(keys,
                                  cache.columns,
                                  query_vector,
                                  head_dimension,
                                  head_dimension,
                                  count,
                                  1,
                                  local_logits,
                                  count);
                for (uint32_t offset = 0; offset < count; ++offset)
                {
                    const bool visible = key_visible(cache.start_position + begin + offset,
                                                     position_offset, plan.sliding_window);
                    const float score = local_logits[offset];
                    local_logits[offset] = visible ? score * scale : -std::numeric_limits<float>::infinity();
                    local_maximum = std::max(local_maximum, local_logits[offset]);
                }
                if (!std::isfinite(local_maximum))
                    return;
                local_normalizer = f32_softmax(local_logits, count, local_maximum);
                const uint16_t* values = cache.bfloat16_values.data() + static_cast<size_t>(cache_slot(cache, begin)) * cache.columns
                                         + static_cast<size_t>(kv_head) * head_dimension;
                pv_bf16(local_output,
                        values,
                        cache.columns,
                        local_logits,
                        count,
                        nullptr,
                        nullptr,
                        0,
                        head_dimension);
                return;
            }
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
                    for (uint64_t key_index = 0; key_index < cache.token_count;)
                    {
                        const uint64_t slot = direct_bfloat16_contiguous
                                                  ? cache.first_slot + key_index
                                                  : cache_slot(cache, key_index);
#if defined(__aarch64__) || defined(_M_ARM64)
                        if (cache.token_count - key_index >= 4
                            && cache.capacity_tokens - slot >= 4)
                        {
                            const uint16_t* key_rows = bfloat16_key_values
                                                       + static_cast<size_t>(slot) * cache.columns
                                                       + static_cast<size_t>(kv_head) * head_dimension;
                            bfloat16_gemm_4x8(key_rows,
                                              cache.columns,
                                              query_vector,
                                              head_dimension,
                                              head_dimension,
                                              4,
                                              1,
                                              head_logits + key_index,
                                              4);
                            for (uint32_t row = 0; row < 4; ++row)
                            {
                                const uint64_t logit_index = key_index + row;
                                head_logits[logit_index] *= scale;
                                maximum = std::max(maximum, head_logits[logit_index]);
                            }
                            key_index += 4;
                            continue;
                        }
#endif
                        const uint16_t* key_vector = bfloat16_key_values + static_cast<size_t>(slot) * cache.columns
                                                     + static_cast<size_t>(kv_head) * head_dimension;
                        head_logits[key_index] = bfloat16_dot(key_vector, query_vector, head_dimension) * scale;
                        maximum = std::max(maximum, head_logits[key_index]);
                        ++key_index;
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

} // namespace moe
} // namespace ncnn
