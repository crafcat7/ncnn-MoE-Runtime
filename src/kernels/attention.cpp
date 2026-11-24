#include "attention.h"
#include "attention_internal.h"
#include "qnk.h"

#include "engine/cpu.h"
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
    const uint32_t active_half_dimension = plan.rope_active != 0
                                               ? plan.rope_active / 2
                                               : half_dimension;
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

    for (uint32_t index = 0; index < active_half_dimension; ++index)
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

static void prepare_rope_coefficients(uint32_t dimension,
                                      uint64_t position,
                                      const AttentionBlockPlan& plan,
                                      float* cosine,
                                      float* sine) noexcept
{
    const uint32_t half_dimension = dimension / 2;
    const uint32_t active_half_dimension = plan.rope_active != 0
                                               ? plan.rope_active / 2
                                               : half_dimension;
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
    for (uint32_t index = 0; index < active_half_dimension; ++index)
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
    // Inactive rotary pairs pass through untouched.
    for (uint32_t index = active_half_dimension; index < half_dimension; ++index)
    {
        cosine[index] = 1.0f;
        sine[index] = 0.0f;
    }
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

static void attention_linear_into(const WeightStore& weights, const CompiledOperatorTable& operators, TensorHandle matrix, TensorHandle bias, const ActivationBuffer& input, ActivationBuffer& output, uint64_t optimization_flags, ExecutionBackend backend, ActivationBuffer* quantized_input, QnkQ8Input* q8_input)
{
    const TensorData& weight = weights.at(matrix);
    const bool needs_input_scratch = !weight.input_column_indices.empty()
                                     || weight.dtype == DType::Float8E4M3;
    ActivationBuffer* scratch = needs_input_scratch && quantized_input
                                        && quantized_input->dtype() == DType::Float32
                                        && quantized_input != &input && quantized_input != &output
                                    ? quantized_input
                                    : nullptr;
    if (bias == invalid_tensor_handle)
    {
        forward_linear(weight, input, output, optimization_flags, operators.find_weight(matrix), backend, scratch, q8_input);
    }
    else
    {
        forward_linear(weight, weights.at(bias), input, output, optimization_flags, operators.find_weight(matrix), backend, scratch, q8_input);
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
    const size_t rows = batch.rows();
    const int norm_threads = rows > 1
                                 ? static_cast<int>(std::min<size_t>(rows, static_cast<size_t>(cpu_linear_num_threads())))
                                 : 1;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(norm_threads) if (norm_threads > 1)
#endif
    for (int64_t token_index = 0; token_index < static_cast<int64_t>(rows); ++token_index)
    {
        float* token = batch.row(static_cast<size_t>(token_index));
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
                                                ActivationBuffer* quantized_input,
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
                          optimization_flags, ExecutionBackend::Cpu, quantized_input, nullptr);
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
    if (backend == ExecutionBackend::Vulkan && plan.kind == AttentionKind::Standard
        && !has_flag(plan.flags, AttentionBlockQsa | AttentionBlockExternalResidual))
    {
        return forward_attention(weights, operators, plan, backend, position_offset,
                                 cache, scratch, hidden, scratch.output, optimization_flags);
    }
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
    ActivationBuffer* quantized_input = scratch.quantized_input.dtype() == DType::Float32
                                                && &scratch.quantized_input != &hidden
                                            ? &scratch.quantized_input
                                            : nullptr;
    const ActivationBuffer* normalized = &hidden;
    if (plan.pre_attention_norm_bias != invalid_tensor_handle)
    {
        forward_layer_norm(hidden,
                           weights.at(plan.pre_attention_norm_weight),
                           weights.at(plan.pre_attention_norm_bias),
                           plan.norm_epsilon,
                           scratch.normalized);
        normalized = &scratch.normalized;
    }
    else if (plan.pre_attention_norm_weight != invalid_tensor_handle)
    {
        forward_rms_norm(hidden,
                         weights.at(plan.pre_attention_norm_weight),
                         plan.norm_epsilon,
                         scratch.normalized,
                         plan.norm_weight_offset);
        normalized = &scratch.normalized;
    }
    auto qsa_status = project_and_append_qsa_keys(weights, operators, plan, position_offset,
                                                  *normalized, cache, scratch, quantized_input,
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
            || (fused_qkv_operator.qnk
                && fused_qkv_operator.qnk->forward(*normalized, fused_qkv))
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
    else if (backend != ExecutionBackend::Cpu
             || plan.key_bias != invalid_tensor_handle
             || plan.value_bias != invalid_tensor_handle
             || !forward_linear({&weights.at(plan.key_weight),
                                 &weights.at(plan.value_weight)},
                                *normalized,
                                {&key, &value},
                                optimization_flags,
                                {operators.find_weight(plan.key_weight),
                                 operators.find_weight(plan.value_weight)},
                                quantized_input))
    {
        attention_linear_into(weights, operators, plan.key_weight, plan.key_bias, *normalized, key, optimization_flags, backend, quantized_input, nullptr);
        attention_linear_into(weights, operators, plan.value_weight, plan.value_bias, *normalized, value, optimization_flags, backend, quantized_input, nullptr);
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
    if (has_flag(plan.flags, AttentionBlockValueNorm))
        apply_head_rms_norm(value,
                            plan.kv_head_count,
                            plan.head_dimension,
                            weights.at(plan.v_norm),
                            plan.norm_epsilon,
                            0.0f);

    const uint32_t rope_dimension = plan.rope_head_dimension == 0 ? plan.head_dimension : plan.rope_head_dimension;
    scratch.rope_cosine.resize(rope_dimension / 2);
    scratch.rope_sine.resize(rope_dimension / 2);
    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
    {
        const uint64_t position = position_offset + token_index;
        prepare_rope_coefficients(rope_dimension,
                                  position,
                                  plan,
                                  scratch.rope_cosine.data(),
                                  scratch.rope_sine.data());
        for (uint32_t head = 0; head < plan.kv_head_count; ++head)
        {
            float_rope_inplace(key.row(token_index) + head * plan.head_dimension,
                               scratch.rope_cosine.data(), scratch.rope_sine.data(), rope_dimension);
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
    const CompiledOperator& attention_operator = operators.at(plan.vulkan_attention_operator);
    if (backend != ExecutionBackend::Vulkan
        || entries.size() < 2
        || !has_flag(optimization_flags, OptimizationVulkanAttentionBatch)
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
    if (cache.transaction.active && has_flag(plan.flags, AttentionBlockQsa))
        return Error{ErrorCode::UnsupportedModel, "QSA cache transactions are not supported"};

    const CompiledOperator& attention_operator = operators.at(plan.vulkan_attention_operator);
    if (backend == ExecutionBackend::Vulkan
        && plan.kind == AttentionKind::Standard
        && !attention_operator.attention)
    {
        return Error{
            ErrorCode::InternalError,
            "Vulkan Standard Attention was selected without a complete attention operator"};
    }
    if (backend == ExecutionBackend::Vulkan && attention_operator.attention)
    {
        if (attention_operator.attention->forward(position_offset,
                                                  cache,
                                                  hidden,
                                                  output))
        {
            return {};
        }
        if (cache.vulkan_attention_cache
            && (cache.vulkan_attention_state_unknown
                || !attention_operator.attention->materialize_device_cache(cache)))
        {
            return Error{
                ErrorCode::InternalError,
                "Vulkan Attention failed and its device KV cache could not be materialized"};
        }
        // Keep the host shadow current for a later retry or explicit backend
        // change, but do not run the selected Vulkan block on the CPU.
        return Error{
            ErrorCode::InternalError,
            "Vulkan Attention execution failed"};
    }

    configure_cache(cache, plan.kv_head_count * plan.head_dimension,
                    plan.kv_cache_dtype);
    ActivationBuffer* quantized_input = scratch.quantized_input.dtype() == DType::Float32
                                                && &scratch.quantized_input != &hidden
                                            ? &scratch.quantized_input
                                            : nullptr;
    const ActivationBuffer* normalized = &hidden;
    if (plan.pre_attention_norm_bias != invalid_tensor_handle)
    {
        forward_layer_norm(hidden,
                           weights.at(plan.pre_attention_norm_weight),
                           weights.at(plan.pre_attention_norm_bias),
                           plan.norm_epsilon,
                           scratch.normalized);
        normalized = &scratch.normalized;
    }
    else if (plan.pre_attention_norm_weight != invalid_tensor_handle)
    {
        forward_rms_norm(hidden,
                         weights.at(plan.pre_attention_norm_weight),
                         plan.norm_epsilon,
                         scratch.normalized,
                         plan.norm_weight_offset);
        normalized = &scratch.normalized;
    }
    auto qsa_status = project_and_append_qsa_keys(weights, operators, plan, position_offset,
                                                  *normalized, cache, scratch, quantized_input,
                                                  optimization_flags);
    if (!qsa_status)
        return qsa_status.error();
    ActivationBuffer& query = scratch.query;
    ActivationBuffer& key = scratch.key;
    ActivationBuffer& value = scratch.value;
    ActivationBuffer& fused_qkv = scratch.fused_qkv;
    const CompiledOperator& fused_qkv_gate_operator = operators.at(plan.fused_qkv_gate_bfloat16_operator);
    const CompiledOperator& fused_qkv_operator = operators.at(plan.fused_qkv_operator);
    // Q/K/V and the output gate share one activation here, so quantize it once
    // for the int8 kernels instead of once per projection.
    QnkQ8Input q8;
    if (hidden.rows() > 0 && backend == ExecutionBackend::Cpu)
    {
        const DType qkv_dtypes[3] = {weights.at(plan.query_weight).dtype,
                                     weights.at(plan.key_weight).dtype,
                                     weights.at(plan.value_weight).dtype};
        if ((is_qnk_dtype(qkv_dtypes[0]) || is_qnk_dtype(qkv_dtypes[1]) || is_qnk_dtype(qkv_dtypes[2]))
            && qnk_q8_act(qkv_dtypes, 3))
            qnk_quantize_input(*normalized, q8);
    }
    bool gate_ready = backend == ExecutionBackend::Vulkan
                      && fused_qkv_gate_operator.bfloat16
                      && fused_qkv_gate_operator.bfloat16->forward(*normalized,
                                                                   fused_qkv);
    const bool fused_projection = gate_ready
                                  || (backend == ExecutionBackend::Vulkan
                                      && fused_qkv_operator.qnk
                                      && fused_qkv_operator.qnk->forward(*normalized, fused_qkv))
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
        if (gate_ready)
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
            if (gate_ready)
            {
                std::copy_n(source,
                            query_columns,
                            scratch.gate.row(token_index));
            }
        }
    }
    else
    {
        gate_ready = backend == ExecutionBackend::Cpu
                     && has_flag(plan.flags, AttentionBlockOutputGate)
                     && plan.query_bias == invalid_tensor_handle
                     && plan.key_bias == invalid_tensor_handle
                     && plan.value_bias == invalid_tensor_handle
                     && forward_linear({&weights.at(plan.query_weight),
                                        &weights.at(plan.key_weight),
                                        &weights.at(plan.value_weight),
                                        &weights.at(plan.output_gate_weight)},
                                       *normalized,
                                       {&query, &key, &value, &scratch.gate},
                                       optimization_flags,
                                       {operators.find_weight(plan.query_weight),
                                        operators.find_weight(plan.key_weight),
                                        operators.find_weight(plan.value_weight),
                                        operators.find_weight(plan.output_gate_weight)},
                                       quantized_input, &q8);
        if (!gate_ready)
        {
            gate_ready = backend == ExecutionBackend::Cpu
                         && has_flag(plan.flags, AttentionBlockOutputGate)
                         && plan.query_bias == invalid_tensor_handle
                         && forward_linear({&weights.at(plan.query_weight),
                                            &weights.at(plan.output_gate_weight)},
                                           *normalized,
                                           {&query, &scratch.gate}, optimization_flags,
                                           {operators.find_weight(plan.query_weight),
                                            operators.find_weight(plan.output_gate_weight)},
                                           quantized_input, &q8);
            if (!gate_ready)
                attention_linear_into(weights, operators, plan.query_weight, plan.query_bias, *normalized, query, optimization_flags, backend, quantized_input, &q8);
            if (backend != ExecutionBackend::Cpu
                || plan.key_bias != invalid_tensor_handle
                || plan.value_bias != invalid_tensor_handle
                || !forward_linear({&weights.at(plan.key_weight),
                                    &weights.at(plan.value_weight)},
                                   *normalized,
                                   {&key, &value},
                                   optimization_flags,
                                   {operators.find_weight(plan.key_weight),
                                    operators.find_weight(plan.value_weight)},
                                   quantized_input))
            {
                attention_linear_into(weights, operators, plan.key_weight, plan.key_bias, *normalized, key, optimization_flags, backend, quantized_input, &q8);
                attention_linear_into(weights, operators, plan.value_weight, plan.value_bias, *normalized, value, optimization_flags, backend, quantized_input, &q8);
            }
        }
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
    if (has_flag(plan.flags, AttentionBlockValueNorm))
        apply_head_rms_norm(value,
                            plan.kv_head_count,
                            plan.head_dimension,
                            weights.at(plan.v_norm),
                            plan.norm_epsilon,
                            0.0f);

    const uint32_t rope_dimension = plan.rope_head_dimension == 0 ? plan.head_dimension : plan.rope_head_dimension;
    const uint32_t rope_half_dimension = rope_dimension / 2;
    const size_t rope_rows = hidden.rows();
    const bool rope_cache_hit = scratch.rope_cache_rows == rope_rows
                                && scratch.rope_cache_half_dim == rope_half_dimension
                                && scratch.rope_cache_position == position_offset
                                && scratch.rope_cache_theta == plan.rope_theta
                                && scratch.rope_cache_factor == plan.rope_scaling_factor
                                && scratch.rope_cache_ntk_alpha == plan.rope_ntk_alpha
                                && scratch.rope_cache_ntk_beta == plan.rope_ntk_beta
                                && scratch.rope_cache_active == plan.rope_active
                                && scratch.rope_cache_context == plan.initial_context_length;
    if (!rope_cache_hit)
    {
        scratch.rope_cosine_cache.resize(rope_rows * rope_half_dimension);
        scratch.rope_sine_cache.resize(rope_rows * rope_half_dimension);
        for (size_t token_index = 0; token_index < rope_rows; ++token_index)
        {
            prepare_rope_coefficients(rope_dimension,
                                      position_offset + token_index,
                                      plan,
                                      scratch.rope_cosine_cache.data() + token_index * rope_half_dimension,
                                      scratch.rope_sine_cache.data() + token_index * rope_half_dimension);
        }
        scratch.rope_cache_rows = static_cast<uint32_t>(rope_rows);
        scratch.rope_cache_half_dim = rope_half_dimension;
        scratch.rope_cache_position = position_offset;
        scratch.rope_cache_theta = plan.rope_theta;
        scratch.rope_cache_factor = plan.rope_scaling_factor;
        scratch.rope_cache_ntk_alpha = plan.rope_ntk_alpha;
        scratch.rope_cache_ntk_beta = plan.rope_ntk_beta;
        scratch.rope_cache_active = plan.rope_active;
        scratch.rope_cache_context = plan.initial_context_length;
    }
    const int rope_threads = rope_rows > 1
                                 ? static_cast<int>(std::min<size_t>(rope_rows, static_cast<size_t>(cpu_linear_num_threads())))
                                 : 1;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(rope_threads) if (rope_threads > 1)
#endif
    for (int64_t token_index = 0; token_index < static_cast<int64_t>(rope_rows); ++token_index)
    {
        const size_t token = static_cast<size_t>(token_index);
        const float* cosine = scratch.rope_cosine_cache.data() + token * rope_half_dimension;
        const float* sine = scratch.rope_sine_cache.data() + token * rope_half_dimension;
        for (uint32_t head = 0; head < plan.head_count; ++head)
        {
            float_rope_inplace(query.row(token) + head * plan.head_dimension,
                               cosine, sine, rope_dimension);
        }
        for (uint32_t head = 0; head < plan.kv_head_count; ++head)
        {
            float_rope_inplace(key.row(token) + head * plan.head_dimension,
                               cosine, sine, rope_dimension);
        }
    }

    if (cache.token_count == 0)
        cache.start_position = position_offset;
    append_cache(cache, plan.kv_cache_dtype, key, value);
    qsa_status = prepare_qsa_selection(weights, plan, position_offset, cache, scratch);
    if (!qsa_status)
        return qsa_status.error();
    sdpa_into(plan,
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
        if (!gate_ready)
        {
            attention_linear_into(weights,
                                  operators,
                                  plan.output_gate_weight,
                                  invalid_tensor_handle,
                                  *normalized,
                                  scratch.gate,
                                  optimization_flags,
                                  backend, quantized_input, &q8);
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
                          backend, quantized_input, nullptr);
    if (has_flag(plan.flags, AttentionBlockPostNorm))
    {
        forward_rms_norm(projected,
                         weights.at(plan.attn_post_norm),
                         plan.norm_epsilon,
                         scratch.normalized,
                         0.0f);
        projected.swap(scratch.normalized);
    }
    if (!has_flag(plan.flags, AttentionBlockExternalResidual))
        add_batch_inplace(projected, hidden);
    if (&projected != &output)
        output.swap(projected);
    trim_sliding_cache(cache, plan);
    return {};
}

} // namespace moe
} // namespace ncnn
