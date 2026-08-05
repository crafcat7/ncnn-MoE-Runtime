#include "cpu_latent_attention.h"

#include "backends/ncnn/ncnn_linear.h"
#include "cpu_float8.h"
#include "cpu_ops.h"
#include "cpu_vector.h"
#include "engine/cpu_session_state.h"
#include "ncnn/moe/runtime_config.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif
namespace ncnn {
namespace moe {

static CpuLatentVectorUndo capture_vector_undo(
    const std::vector<float>& values,
    size_t expected_size,
    size_t offset,
    size_t count)
{
    CpuLatentVectorUndo undo;
    undo.original_size = values.size();
    undo.captured = true;
    if (values.size() != expected_size)
    {
        undo.values = values;
        return undo;
    }
    undo.offset = offset;
    const size_t available = offset < values.size() ? values.size() - offset : 0;
    const size_t copied = std::min(count, available);
    undo.values.assign(
        values.begin() + std::min(offset, values.size()),
        values.begin() + std::min(offset + copied, values.size()));
    return undo;
}

static CpuLatentVectorUndo capture_whole_vector_undo(const std::vector<float>& values)
{
    CpuLatentVectorUndo undo;
    undo.original_size = values.size();
    undo.values = values;
    undo.captured = true;
    return undo;
}

static void restore_vector_undo(std::vector<float>& values, const CpuLatentVectorUndo& undo)
{
    if (!undo.captured)
        return;
    values.resize(undo.original_size);
    if (!undo.values.empty())
    {
        std::copy(
            undo.values.begin(),
            undo.values.end(),
            values.begin() + undo.offset);
    }
}

static void record_latent_cache_undo(CpuLayerCache& cache, const AttentionBlockPlan& plan, uint64_t position)
{
    if (!cache.transaction.latent_active)
        return;

    CpuLatentCacheUndo undo;
    undo.latent_token_count = cache.latent_token_count;
    undo.latent_compressed_size = cache.latent_compressed.size();
    undo.latent_index_compressed_size = cache.latent_index_compressed.size();
    const size_t window_size = static_cast<size_t>(plan.sliding_window) * plan.head_dimension;
    const size_t window_offset = static_cast<size_t>(position % plan.sliding_window) * plan.head_dimension;
    undo.latent_window = capture_vector_undo(cache.latent_window, window_size, window_offset, plan.head_dimension);

    if (plan.compression_ratio != 0)
    {
        const uint32_t ratio = plan.compression_ratio;
        const uint32_t projection_multiplier = ratio == 4 ? 2 : 1;
        const size_t slot = position % ratio;
        const size_t pending_columns = static_cast<size_t>(projection_multiplier) * plan.head_dimension;
        const size_t pending_size = static_cast<size_t>(ratio) * pending_columns;
        const size_t pending_offset = slot * pending_columns;
        undo.compressor_pending_values = capture_vector_undo(cache.compressor_pending_values, pending_size, pending_offset, pending_columns);
        undo.compressor_pending_scores = capture_vector_undo(cache.compressor_pending_scores, pending_size, pending_offset, pending_columns);
        if (ratio == 4 && slot + 1 == ratio)
        {
            undo.compressor_previous_values = capture_whole_vector_undo(cache.compressor_previous_values);
            undo.compressor_previous_scores = capture_whole_vector_undo(cache.compressor_previous_scores);
        }
        if (ratio == 4)
        {
            const size_t index_pending_columns = static_cast<size_t>(projection_multiplier) * plan.index_head_dimension;
            const size_t index_pending_size = static_cast<size_t>(ratio) * index_pending_columns;
            const size_t index_pending_offset = slot * index_pending_columns;
            undo.index_compressor_pending_values = capture_vector_undo(cache.index_compressor_pending_values, index_pending_size, index_pending_offset, index_pending_columns);
            undo.index_compressor_pending_scores = capture_vector_undo(cache.index_compressor_pending_scores, index_pending_size, index_pending_offset, index_pending_columns);
            if (slot + 1 == ratio)
            {
                undo.index_compressor_previous_values = capture_whole_vector_undo(cache.index_compressor_previous_values);
                undo.index_compressor_previous_scores = capture_whole_vector_undo(cache.index_compressor_previous_scores);
            }
        }
    }
    cache.transaction.latent_undo.push_back(std::move(undo));
}

static void restore_latent_cache_undo(CpuLayerCache& cache, const CpuLatentCacheUndo& undo)
{
    restore_vector_undo(cache.latent_window, undo.latent_window);
    cache.latent_compressed.resize(undo.latent_compressed_size);
    cache.latent_index_compressed.resize(undo.latent_index_compressed_size);
    restore_vector_undo(cache.compressor_pending_values, undo.compressor_pending_values);
    restore_vector_undo(cache.compressor_pending_scores, undo.compressor_pending_scores);
    if (undo.compressor_previous_values.captured)
    {
        restore_vector_undo(cache.compressor_previous_values, undo.compressor_previous_values);
        restore_vector_undo(cache.compressor_previous_scores, undo.compressor_previous_scores);
    }
    restore_vector_undo(cache.index_compressor_pending_values, undo.index_compressor_pending_values);
    restore_vector_undo(cache.index_compressor_pending_scores, undo.index_compressor_pending_scores);
    if (undo.index_compressor_previous_values.captured)
    {
        restore_vector_undo(cache.index_compressor_previous_values, undo.index_compressor_previous_values);
        restore_vector_undo(cache.index_compressor_previous_scores, undo.index_compressor_previous_scores);
    }
    cache.latent_token_count = undo.latent_token_count;
}

void begin_latent_cache_transaction(std::span<CpuLayerCache> caches)
{
    for (CpuLayerCache& cache : caches)
    {
        cache.transaction.latent_undo.clear();
        cache.transaction.latent_active = true;
    }
}

Result<void> finish_latent_cache_transaction(std::span<CpuLayerCache> caches, size_t committed_rows)
{
    for (const CpuLayerCache& cache : caches)
    {
        if (cache.transaction.latent_active && !cache.transaction.latent_undo.empty()
            && cache.transaction.latent_undo.size() < committed_rows)
        {
            return Error{
                ErrorCode::InternalError,
                "latent cache transaction is missing committed rows"};
        }
    }
    for (CpuLayerCache& cache : caches)
    {
        if (!cache.transaction.latent_active)
            continue;
        while (cache.transaction.latent_undo.size() > committed_rows)
        {
            restore_latent_cache_undo(cache, cache.transaction.latent_undo.back());
            cache.transaction.latent_undo.pop_back();
        }
        cache.transaction.latent_undo.clear();
        cache.transaction.latent_active = false;
    }
    return {};
}

static void apply_rope(float* values, uint32_t dimension, uint64_t position, const AttentionBlockPlan& plan, bool inverse)
{
    const bool yarn = plan.compression_ratio != 0;
    const float base = yarn ? plan.compressed_rope_theta : plan.rope_theta;
    int correction_low = 0;
    int correction_high = 0;
    if (yarn)
    {
        const float denominator = 2.0f * std::log(base);
        const float rotations_low = static_cast<float>(dimension) * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_beta * 2.0f * 3.14159265358979323846f)) / denominator;
        const float rotations_high = static_cast<float>(dimension) * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_alpha * 2.0f * 3.14159265358979323846f)) / denominator;
        correction_low = std::max(0, static_cast<int>(std::floor(rotations_low)));
        correction_high = std::min(static_cast<int>(dimension) - 1, static_cast<int>(std::ceil(rotations_high)));
    }

    for (uint32_t pair = 0; pair < dimension / 2; ++pair)
    {
        float frequency = 1.0f / std::pow(base, static_cast<float>(pair * 2) / static_cast<float>(dimension));
        if (yarn)
        {
            const float ramp_denominator = correction_low == correction_high ? 0.001f : static_cast<float>(correction_high - correction_low);
            const float ramp = std::clamp((static_cast<float>(pair) - static_cast<float>(correction_low)) / ramp_denominator, 0.0f, 1.0f);
            const float smooth = 1.0f - ramp;
            frequency = frequency / plan.rope_scaling_factor * (1.0f - smooth) + frequency * smooth;
        }
        float angle = static_cast<float>(position) * frequency;
        if (inverse)
            angle = -angle;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float real = values[pair * 2];
        const float imaginary = values[pair * 2 + 1];
        values[pair * 2] = real * cosine - imaginary * sine;
        values[pair * 2 + 1] = real * sine + imaginary * cosine;
    }
}

static bool prepared_latent_rope_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags,
        RuntimeOptimizationCpuLatentPreparedRope);
}

static bool simd_latent_norm_enabled(uint64_t optimization_flags) noexcept
{
    return simd_rms_norm_enabled(optimization_flags)
           && runtime_optimization_enabled(optimization_flags,
               RuntimeOptimizationCpuLatentSimdNorm);
}

static bool online_latent_softmax_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags,
        RuntimeOptimizationCpuLatentOnlineSoftmax);
}

static bool vector_latent_softmax_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags,
               RuntimeOptimizationCpuLatentVectorSoftmax)
           && float_exp_simd_available();
}

static constexpr uint32_t vector_latent_softmax_min_candidates = 64;

static bool parallel_latent_output_groups_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags,
        RuntimeOptimizationCpuLatentOutputGroups);
}

static void prepare_rope_coefficients(
    uint32_t dimension,
    uint64_t position,
    const AttentionBlockPlan& plan,
    std::vector<float>& cosines,
    std::vector<float>& sines)
{
    const bool yarn = plan.compression_ratio != 0;
    const float base = yarn ? plan.compressed_rope_theta : plan.rope_theta;
    int correction_low = 0;
    int correction_high = 0;
    if (yarn)
    {
        const float denominator = 2.0f * std::log(base);
        const float rotations_low = static_cast<float>(dimension) * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_beta * 2.0f * 3.14159265358979323846f)) / denominator;
        const float rotations_high = static_cast<float>(dimension) * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_alpha * 2.0f * 3.14159265358979323846f)) / denominator;
        correction_low = std::max(0, static_cast<int>(std::floor(rotations_low)));
        correction_high = std::min(static_cast<int>(dimension) - 1, static_cast<int>(std::ceil(rotations_high)));
    }

    const uint32_t pair_count = dimension / 2;
    cosines.resize(pair_count);
    sines.resize(pair_count);
    for (uint32_t pair = 0; pair < pair_count; ++pair)
    {
        float frequency = 1.0f / std::pow(
            base,
            static_cast<float>(pair * 2) / static_cast<float>(dimension));
        if (yarn)
        {
            const float ramp_denominator = correction_low == correction_high
                                               ? 0.001f
                                               : static_cast<float>(correction_high - correction_low);
            const float ramp = std::clamp(
                (static_cast<float>(pair) - static_cast<float>(correction_low))
                    / ramp_denominator,
                0.0f,
                1.0f);
            const float smooth = 1.0f - ramp;
            frequency = frequency / plan.rope_scaling_factor * (1.0f - smooth)
                        + frequency * smooth;
        }
        const float angle = static_cast<float>(position) * frequency;
        cosines[pair] = std::cos(angle);
        sines[pair] = std::sin(angle);
    }
}

static void apply_prepared_rope(
    float* values,
    uint32_t dimension,
    std::span<const float> cosines,
    std::span<const float> sines,
    bool inverse)
{
    const uint32_t pair_count = dimension / 2;
    assert(cosines.size() >= pair_count && sines.size() >= pair_count);
    for (uint32_t pair = 0; pair < pair_count; ++pair)
    {
        const float cosine = cosines[pair];
        const float sine = inverse ? -sines[pair] : sines[pair];
        const float real = values[pair * 2];
        const float imaginary = values[pair * 2 + 1];
        values[pair * 2] = real * cosine - imaginary * sine;
        values[pair * 2 + 1] = real * sine + imaginary * cosine;
    }
}

static void hadamard_rotate(float* values, uint32_t count)
{
    for (uint32_t stride = 1; stride < count; stride *= 2)
    {
        for (uint32_t begin = 0; begin < count; begin += stride * 2)
        {
            for (uint32_t offset = 0; offset < stride; ++offset)
            {
                const float left = values[begin + offset];
                const float right = values[begin + stride + offset];
                values[begin + offset] = left + right;
                values[begin + stride + offset] = left - right;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(count));
    for (uint32_t index = 0; index < count; ++index)
        values[index] *= scale;
}

static void normalize_vector(float* values, uint32_t count, const TensorData& weight, float epsilon, uint64_t optimization_flags)
{
    float square_sum = 0.0f;
    if (simd_latent_norm_enabled(optimization_flags))
        square_sum = float_dot(values, values, count);
    else
    {
        for (uint32_t index = 0; index < count; ++index)
            square_sum += values[index] * values[index];
    }
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    if (weight.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> weights = weight.bfloat16_values();
        if (simd_latent_norm_enabled(optimization_flags))
            bfloat16_weighted_scale(values, values, weights.data(), inverse_rms, 0.0f, count);
        else
        {
            for (uint32_t index = 0; index < count; ++index)
                values[index] *= inverse_rms * bfloat16_to_float(weights[index]);
        }
    }
    else
    {
        const std::span<const float> weights = weight.float32_values();
        if (simd_latent_norm_enabled(optimization_flags))
            float_weighted_scale(values, values, weights.data(), inverse_rms, 0.0f, count);
        else
        {
            for (uint32_t index = 0; index < count; ++index)
                values[index] *= inverse_rms * weights[index];
        }
    }
}

static void normalize_unit(float* values, uint32_t count, float epsilon, uint64_t optimization_flags)
{
    float square_sum = 0.0f;
    if (simd_latent_norm_enabled(optimization_flags))
        square_sum = float_dot(values, values, count);
    else
    {
        for (uint32_t index = 0; index < count; ++index)
            square_sum += values[index] * values[index];
    }
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
        values[index] *= inverse_rms;
}

static void normalize_unit_prepared_rope(
    float* values,
    uint32_t count,
    uint32_t rope_dimension,
    float epsilon,
    std::span<const float> cosines,
    std::span<const float> sines,
    uint64_t optimization_flags)
{
    const float square_sum = simd_latent_norm_enabled(optimization_flags)
                                 ? float_dot(values, values, count)
                                 : [&]() {
                                       float sum = 0.0f;
                                       for (uint32_t index = 0; index < count; ++index)
                                           sum += values[index] * values[index];
                                       return sum;
                                   }();
    const float inverse_rms =
        1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    const uint32_t rope_offset = count - rope_dimension;
    for (uint32_t index = 0; index < rope_offset; ++index)
        values[index] *= inverse_rms;

    const uint32_t pair_count = rope_dimension / 2;
    assert(cosines.size() >= pair_count && sines.size() >= pair_count);
    for (uint32_t pair = 0; pair < pair_count; ++pair)
    {
        const float cosine = cosines[pair];
        const float sine = sines[pair];
        const float real = values[rope_offset + pair * 2] * inverse_rms;
        const float imaginary = values[rope_offset + pair * 2 + 1] * inverse_rms;
        values[rope_offset + pair * 2] = real * cosine - imaginary * sine;
        values[rope_offset + pair * 2 + 1] = real * sine + imaginary * cosine;
    }
}

static bool scored_index_precedes(
    const CpuLayerCache::LatentScoredIndex& left,
    const CpuLayerCache::LatentScoredIndex& right)
{
    return left.score > right.score || (left.score == right.score && left.index < right.index);
}

static bool vulkan_latent_compressor_enabled(
    ExecutionBackend backend,
    uint64_t optimization_flags) noexcept
{
    return backend == ExecutionBackend::Vulkan
           && runtime_optimization_enabled(optimization_flags,
        RuntimeOptimizationVulkanLatentCompressor);
}

static void append_compressed_value(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    const CpuBatch& input,
    uint64_t position,
    float norm_epsilon,
    bool indexer,
    CpuLayerCache& cache,
    uint64_t optimization_flags,
    const float* projected_values_override = nullptr,
    const float* projected_scores_override = nullptr)
{
    const uint32_t ratio = plan.compression_ratio;
    const uint32_t dimension = indexer ? plan.index_head_dimension : plan.head_dimension;
    const TensorHandle value_handle = indexer ? plan.indexer_compressor_key_value_weight : plan.compressor_key_value_weight;
    const TensorHandle gate_handle = indexer ? plan.indexer_compressor_gate_weight : plan.compressor_gate_weight;
    const TensorHandle position_handle = indexer ? plan.indexer_compressor_position : plan.compressor_position;
    const TensorHandle norm_handle = indexer ? plan.indexer_compressor_norm_weight : plan.compressor_norm_weight;
    const uint32_t projection_multiplier = ratio == 4 ? 2 : 1;
    const TensorData& value_weight = weights.at(value_handle);
    const TensorData& gate_weight = weights.at(gate_handle);
    const CompiledOperator& value_operator = operators.at_weight(value_handle);
    const CompiledOperator& gate_operator = operators.at_weight(gate_handle);
    const bool has_projected_pair =
        projected_values_override && projected_scores_override;
    const bool used_vulkan_pair =
        !has_projected_pair
        && vulkan_latent_compressor_enabled(backend, optimization_flags)
        && value_operator.bfloat16
        && gate_operator.bfloat16
        && value_operator.bfloat16->forward_parallel(
            input,
            *gate_operator.bfloat16,
            cache.compressor_values,
            cache.compressor_scores);
    if (!has_projected_pair
        && !used_vulkan_pair
        && !float8_linear_pair_batch_into(
            value_weight, gate_weight, input,
            cache.compressor_values, cache.compressor_scores, optimization_flags,
            &value_operator, &gate_operator))
    {
        linear_batch_into(value_weight, input, cache.compressor_values, optimization_flags, &value_operator, backend);
        linear_batch_into(gate_weight, input, cache.compressor_scores, optimization_flags, &gate_operator, backend);
    }

    std::vector<float>& pending_values = indexer ? cache.index_compressor_pending_values : cache.compressor_pending_values;
    std::vector<float>& pending_scores = indexer ? cache.index_compressor_pending_scores : cache.compressor_pending_scores;
    std::vector<float>& previous_values = indexer ? cache.index_compressor_previous_values : cache.compressor_previous_values;
    std::vector<float>& previous_scores = indexer ? cache.index_compressor_previous_scores : cache.compressor_previous_scores;
    std::vector<float>& compressed = indexer ? cache.latent_index_compressed : cache.latent_compressed;
    if (pending_values.size() != static_cast<size_t>(ratio) * projection_multiplier * dimension)
    {
        pending_values.assign(static_cast<size_t>(ratio) * projection_multiplier * dimension, 0.0f);
        pending_scores.assign(static_cast<size_t>(ratio) * projection_multiplier * dimension, 0.0f);
    }
    const uint32_t slot = static_cast<uint32_t>(position % ratio);
    const float* projected_values = has_projected_pair
                                        ? projected_values_override
                                        : cache.compressor_values.row(0);
    const float* projected_scores = has_projected_pair
                                        ? projected_scores_override
                                        : cache.compressor_scores.row(0);
    const std::span<const float> positional = weights.at(position_handle).float32_values();
    for (uint32_t column = 0; column < projection_multiplier * dimension; ++column)
    {
        const size_t destination = (static_cast<size_t>(slot) * projection_multiplier * dimension) + column;
        pending_values[destination] = projected_values[column];
        pending_scores[destination] = projected_scores[column] + positional[static_cast<size_t>(slot) * projection_multiplier * dimension + column];
    }
    if (slot + 1 != ratio)
        return;

    std::vector<float>& pooled = cache.compressor_pooled;
    if (pooled.size() < dimension)
        pooled.resize(dimension);
    std::fill_n(pooled.data(), dimension, 0.0f);
    const bool overlap = ratio == 4;
    const bool has_previous = overlap && !previous_values.empty();
    const uint32_t candidate_count = ratio + (has_previous ? ratio : 0);
    std::vector<float>& exponentials = cache.compressor_exponentials;
    exponentials.resize(candidate_count);
    for (uint32_t column = 0; column < dimension; ++column)
    {
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t token = 0; token < candidate_count; ++token)
        {
            float score = 0.0f;
            if (has_previous && token < ratio)
                score = previous_scores[static_cast<size_t>(token) * dimension + column];
            else
            {
                const uint32_t current_token = has_previous ? token - ratio : token;
                const uint32_t component = overlap ? 1 : 0;
                score = pending_scores[(static_cast<size_t>(current_token) * projection_multiplier + component) * dimension + column];
            }
            exponentials[token] = score;
            maximum = std::max(maximum, score);
        }
        float denominator = 0.0f;
        for (float& score : exponentials)
        {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (uint32_t token = 0; token < candidate_count; ++token)
        {
            float value = 0.0f;
            if (has_previous && token < ratio)
                value = previous_values[static_cast<size_t>(token) * dimension + column];
            else
            {
                const uint32_t current_token = has_previous ? token - ratio : token;
                const uint32_t component = overlap ? 1 : 0;
                value = pending_values[(static_cast<size_t>(current_token) * projection_multiplier + component) * dimension + column];
            }
            pooled[column] += value * exponentials[token] / denominator;
        }
    }
    if (overlap)
    {
        previous_values.resize(static_cast<size_t>(ratio) * dimension);
        previous_scores.resize(static_cast<size_t>(ratio) * dimension);
        for (uint32_t token = 0; token < ratio; ++token)
        {
            std::copy_n(pending_values.data() + static_cast<size_t>(token) * projection_multiplier * dimension, dimension, previous_values.data() + static_cast<size_t>(token) * dimension);
            std::copy_n(pending_scores.data() + static_cast<size_t>(token) * projection_multiplier * dimension, dimension, previous_scores.data() + static_cast<size_t>(token) * dimension);
        }
    }

    normalize_vector(pooled.data(), dimension, weights.at(norm_handle), norm_epsilon, optimization_flags);
    apply_rope(pooled.data() + dimension - plan.rope_head_dimension, plan.rope_head_dimension, position + 1 - ratio, plan, false);
    if (indexer)
    {
        hadamard_rotate(pooled.data(), dimension);
        quantize_float4_e2m1_inplace(pooled.data(), dimension, 32);
    }
    else
    {
        quantize_float8_e4m3_inplace(pooled.data(), dimension - plan.rope_head_dimension, 64, true, optimization_flags);
    }
    compressed.insert(compressed.end(), pooled.begin(), pooled.begin() + dimension);
}

static bool select_compressed_indices(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    const CpuBatch& normalized,
    const CpuBatch& query_rank,
    uint64_t position,
    CpuLayerCache& cache,
    uint64_t optimization_flags)
{
    const uint32_t compressed_count = static_cast<uint32_t>(cache.latent_compressed.size() / plan.head_dimension);
    cache.latent_selected_indices.clear();
    if (plan.compression_ratio != 4 || compressed_count <= plan.index_top_k)
        return false;

    CpuBatch& query = cache.latent_index_query;
    linear_batch_into(weights.at(plan.indexer_query_weight), query_rank, query, optimization_flags, &operators.at_weight(plan.indexer_query_weight), backend);
    for (uint32_t head = 0; head < plan.index_head_count; ++head)
    {
        float* values = query.row(0) + static_cast<size_t>(head) * plan.index_head_dimension;
        apply_rope(values + plan.index_head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, false);
        hadamard_rotate(values, plan.index_head_dimension);
        quantize_float4_e2m1_inplace(values, plan.index_head_dimension, 32);
    }
    CpuBatch& projected_weights = cache.latent_index_projected_weights;
    linear_batch_into(weights.at(plan.indexer_weights_weight), normalized, projected_weights, optimization_flags, &operators.at_weight(plan.indexer_weights_weight), backend);
    const float index_scale = 1.0f / std::sqrt(static_cast<float>(plan.index_head_dimension * plan.index_head_count));
    std::vector<float>& scores = cache.latent_index_scores;
    scores.assign(compressed_count, 0.0f);
    for (uint32_t compressed_index = 0; compressed_index < compressed_count; ++compressed_index)
    {
        const float* key = cache.latent_index_compressed.data() + static_cast<size_t>(compressed_index) * plan.index_head_dimension;
        for (uint32_t head = 0; head < plan.index_head_count; ++head)
        {
            const float* query_head = query.row(0) + static_cast<size_t>(head) * plan.index_head_dimension;
            const float dot = float_dot(query_head, key, plan.index_head_dimension);
            scores[compressed_index] += std::max(0.0f, dot) * projected_weights.row(0)[head] * index_scale;
        }
    }
    std::vector<CpuLayerCache::LatentScoredIndex>& scored = cache.latent_scored_indices;
    scored.resize(compressed_count);
    for (uint32_t index = 0; index < compressed_count; ++index)
        scored[index] = {index, scores[index]};
    const uint32_t selected_count = std::min(plan.index_top_k, compressed_count);
    // The selector only needs the best K entries.  partial_sort performs an
    // O(N log K) heap maintenance pass over the complete compressed history;
    // nth_element followed by sorting the retained prefix is O(N + K log K)
    // and keeps the same deterministic score/index ordering for the prefix.
    if (selected_count != 0)
    {
        std::nth_element(scored.begin(), scored.begin() + selected_count, scored.end(), scored_index_precedes);
        std::sort(scored.begin(), scored.begin() + selected_count, scored_index_precedes);
    }
    std::vector<uint32_t>& selected_indices = cache.latent_selected_indices;
    selected_indices.resize(selected_count);
    for (uint32_t index = 0; index < selected_count; ++index)
        selected_indices[index] = scored[index].index;
    return true;
}

static void fp8_matrix_rows_dot(
    const TensorData& matrix,
    uint32_t first_row,
    uint32_t row_count,
    const float* input,
    float* output)
{
    const uint32_t columns = matrix.shape[1];
    const uint32_t input_blocks = (columns + 127) / 128;
    uint32_t processed = 0;
    while (processed < row_count)
    {
        const uint32_t row = first_row + processed;
        // Scales are shared by 128 output rows.  Do not let the small
        // multi-row kernel cross that boundary.
        const uint32_t scale_rows = 128 - row % 128;
        const uint32_t rows = std::min<uint32_t>(
            {4, row_count - processed, scale_rows});
        float8_e4m3_block_dot_rows4(
            matrix.float8_values().data()
                + static_cast<size_t>(row) * columns,
            columns,
            matrix.quantization_scales.data()
                + static_cast<size_t>(row / 128) * input_blocks,
            input,
            columns,
            128,
            rows,
            output + processed);
        processed += rows;
    }
}

static int latent_output_group_team_size(
    size_t row_count,
    uint32_t group_count,
    uint32_t group_columns,
    uint32_t rank,
    uint64_t optimization_flags) noexcept
{
#if defined(_OPENMP)
    const uint64_t operations =
        static_cast<uint64_t>(row_count) * group_count * group_columns
        * rank;
    if (!parallel_latent_output_groups_enabled(optimization_flags)
        || operations < 1024 * 1024)
        return 1;
    const uint32_t maximum = float8_linear_thread_limit();
    return std::max(
        1,
        std::min({static_cast<int>(maximum),
                  static_cast<int>(row_count * group_count),
                  omp_get_max_threads()}));
#else
    (void)row_count;
    (void)group_count;
    (void)group_columns;
    (void)rank;
    return 1;
#endif
}

static Result<CpuBatch> execute_latent_attention_rows(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    std::span<const uint64_t> positions,
    std::span<CpuLayerCache* const> caches,
    const CpuBatch& input,
    uint64_t optimization_flags)
{
    if (!has_flag(plan.flags, AttentionBlockLatent)
        || input.columns() != weights.at(plan.pre_attention_norm_weight).shape[0]
        || input.rows() != positions.size()
        || input.rows() != caches.size()
        || plan.sliding_window == 0)
        return Error{ErrorCode::InvalidArgument, "invalid latent attention execution plan"};
    for (const CpuLayerCache* cache : caches)
    {
        if (!cache)
            return Error{ErrorCode::InvalidArgument, "latent attention cache cannot be null"};
    }

    CpuBatch normalized;
    if (plan.compression_ratio != 0)
        normalized = rms_norm_batch(input, weights.at(plan.pre_attention_norm_weight), norm_epsilon, 0.0f, optimization_flags);
    CpuBatch query_rank;
    CpuBatch query;
    const TensorData& query_a = weights.at(plan.query_a_weight);
    const TensorData& query_b = weights.at(plan.query_b_weight);
    const CompiledOperator& query_a_operator = operators.at_weight(plan.query_a_weight);
    const CompiledOperator& query_b_operator = operators.at_weight(plan.query_b_weight);
    std::vector<std::pair<const CpuLayerCache*, uint32_t>> projected_compressed_counts;
    projected_compressed_counts.reserve(caches.size());
    uint32_t maximum_projected_compressed_count = 0;
    for (size_t row = 0; row < caches.size(); ++row)
    {
        const CpuLayerCache* cache = caches[row];
        auto projected = std::find_if(
            projected_compressed_counts.begin(),
            projected_compressed_counts.end(),
            [cache](const auto& item) {
                return item.first == cache;
            });
        if (projected == projected_compressed_counts.end())
        {
            const uint32_t current_count = plan.head_dimension == 0
                                               ? 0u
                                               : static_cast<uint32_t>(
                                                     cache->latent_compressed.size()
                                                     / plan.head_dimension);
            projected_compressed_counts.push_back({cache, current_count});
            projected = projected_compressed_counts.end() - 1;
        }
        if (plan.compression_ratio != 0
            && positions[row] % plan.compression_ratio + 1 == plan.compression_ratio)
        {
            ++projected->second;
        }
        maximum_projected_compressed_count = std::max(maximum_projected_compressed_count, projected->second);
    }
    const bool query_rank_not_required = plan.compression_ratio != 4 || maximum_projected_compressed_count <= plan.index_top_k;
    const TensorData& key_value_weight = weights.at(plan.key_value_weight);
    CpuBatch key_value;
    CpuBatch fused_compressor_values;
    CpuBatch fused_compressor_scores;
    CpuBatch fused_index_compressor_values;
    CpuBatch fused_index_compressor_scores;
    std::array<const NcnnVulkanBfloat16Operator*, 4> fused_compressor_operators{};
    std::array<ActivationBuffer*, 4> fused_compressor_outputs{};
    size_t fused_compressor_count = 0;
    if (vulkan_latent_compressor_enabled(backend, optimization_flags)
        && plan.compression_ratio == 4)
    {
        const auto add_compressor_pair = [&](TensorHandle value_handle,
                                              TensorHandle gate_handle,
                                              CpuBatch& value_output,
                                              CpuBatch& gate_output) {
            if (value_handle == invalid_tensor_handle
                || gate_handle == invalid_tensor_handle)
                return;
            const CompiledOperator& value_operator = operators.at_weight(value_handle);
            const CompiledOperator& gate_operator = operators.at_weight(gate_handle);
            if (!value_operator.bfloat16
                || !gate_operator.bfloat16
                || fused_compressor_count + 2 > fused_compressor_operators.size())
                return;
            fused_compressor_operators[fused_compressor_count] =
                value_operator.bfloat16.get();
            fused_compressor_outputs[fused_compressor_count] = &value_output;
            ++fused_compressor_count;
            fused_compressor_operators[fused_compressor_count] =
                gate_operator.bfloat16.get();
            fused_compressor_outputs[fused_compressor_count] = &gate_output;
            ++fused_compressor_count;
        };
        add_compressor_pair(
            plan.compressor_key_value_weight,
            plan.compressor_gate_weight,
            fused_compressor_values,
            fused_compressor_scores);
        add_compressor_pair(
            plan.indexer_compressor_key_value_weight,
            plan.indexer_compressor_gate_weight,
            fused_index_compressor_values,
            fused_index_compressor_scores);
        if (fused_compressor_count != 4)
            fused_compressor_count = 0;
    }
    const CompiledOperator& key_value_operator = operators.at_weight(plan.key_value_weight);
    bool chained_query = false;
    bool fused_compressor = false;
    if (backend == ExecutionBackend::Vulkan
        && query_rank_not_required
        && query_a_operator.float8
        && query_b_operator.float8)
    {
        if (key_value_operator.float8)
        {
            if (fused_compressor_count != 0)
            {
                chained_query = query_a_operator.float8->forward_rms_norm_chain_parallel_bfloat16(
                    normalized,
                    *query_b_operator.float8,
                    *key_value_operator.float8,
                    std::span<const NcnnVulkanBfloat16Operator*>(
                        fused_compressor_operators.data(),
                        fused_compressor_count),
                    std::span<ActivationBuffer*>(
                        fused_compressor_outputs.data(),
                        fused_compressor_count),
                    query,
                    key_value);
                fused_compressor = chained_query;
            }
            else
            {
                if (plan.compression_ratio == 0
                    && runtime_optimization_enabled(optimization_flags,
                        RuntimeOptimizationVulkanLatentInputRmsNorm))
                {
                    chained_query = query_a_operator.float8->forward_input_rms_norm_chain_parallel(
                        input,
                        *query_b_operator.float8,
                        *key_value_operator.float8,
                        query,
                        key_value);
                }
                if (!chained_query)
                {
                    if (normalized.rows() == 0)
                        normalized = rms_norm_batch(
                            input,
                            weights.at(plan.pre_attention_norm_weight),
                            norm_epsilon, 0.0f, optimization_flags);
                    chained_query = query_a_operator.float8->forward_rms_norm_chain_parallel(
                        normalized,
                        *query_b_operator.float8,
                        *key_value_operator.float8,
                        query,
                        key_value);
                }
            }
        }
        else
        {
            if (normalized.rows() == 0)
                normalized = rms_norm_batch(
                    input,
                    weights.at(plan.pre_attention_norm_weight),
                    norm_epsilon, 0.0f, optimization_flags);
                chained_query = query_a_operator.float8->forward_rms_norm_chain(normalized, *query_b_operator.float8, query);
        }
    }
    if (!chained_query
        && backend == ExecutionBackend::Vulkan
        && runtime_optimization_enabled(optimization_flags,
            RuntimeOptimizationVulkanCommandGraph)
        && query_rank_not_required
        && query_a_operator.linear
        && query_b_operator.linear
        && key_value_operator.linear)
    {
        if (normalized.rows() == 0)
            normalized = rms_norm_batch(
                input,
                weights.at(plan.pre_attention_norm_weight),
                norm_epsilon, 0.0f, optimization_flags);
        auto graph = NcnnVulkanCommandGraph::create(
            *query_a_operator.linear);
        NcnnVulkanDeviceTensor normalized_device;
        NcnnVulkanDeviceTensor query_rank_device;
        NcnnVulkanDeviceTensor query_device;
        NcnnVulkanDeviceTensor key_value_device;
        chained_query = graph
                        && graph->upload(normalized, normalized_device)
                        && graph->linear(
                            *query_a_operator.linear,
                            normalized_device,
                            query_rank_device)
                        && graph->linear(
                            *query_b_operator.linear,
                            query_rank_device,
                            query_device)
                        && graph->linear(
                            *key_value_operator.linear,
                            normalized_device,
                            key_value_device)
                        && graph->download(query_device, query)
                        && graph->download(key_value_device, key_value)
                        && graph->submit()
                        && graph->wait();
    }
    if (!chained_query)
    {
        if (normalized.rows() == 0)
            normalized = rms_norm_batch(
                input,
                weights.at(plan.pre_attention_norm_weight),
                norm_epsilon, 0.0f, optimization_flags);
        if (!float8_linear_pair_batch_into(
                query_a, key_value_weight, normalized, query_rank,
                key_value, optimization_flags, &query_a_operator, &key_value_operator))
        {
            key_value = CpuBatch();
            query_rank = linear_batch(query_a, normalized, optimization_flags, &query_a_operator, backend);
        }
        if (query_rank_not_required
            && float8_linear_rms_norm_batch_into(
                query_b, query_rank,
                weights.at(plan.query_norm_weight), norm_epsilon, query,
                optimization_flags, &query_b_operator))
        {
            // The rank vector is not needed by the indexer in this branch;
            // avoid retaining and copying a second intermediate batch.
            query_rank.clear();
        }
        else
        {
            query_rank = rms_norm_batch(
                query_rank, weights.at(plan.query_norm_weight),
                norm_epsilon, 0.0f, optimization_flags);
            query = linear_batch(query_b, query_rank, optimization_flags, &query_b_operator, backend);
        }
    }
    if (key_value.rows() == 0)
        key_value = linear_batch(key_value_weight, normalized, optimization_flags, &key_value_operator, backend);
    key_value = rms_norm_batch(key_value, weights.at(plan.key_value_norm_weight), norm_epsilon, 0.0f, optimization_flags);
    CpuBatch attention_output(input.rows(), plan.head_count * plan.head_dimension);

    const std::span<const float> sinks = weights.at(plan.sinks).float32_values();
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(plan.head_dimension));
    const bool use_online_latent_softmax = online_latent_softmax_enabled(optimization_flags);
    const bool use_vector_latent_softmax = vector_latent_softmax_enabled(optimization_flags);

    struct AttentionRowContext
    {
        uint64_t window_begin = 0;
        uint32_t window_count = 0;
        uint32_t compressed_count = 0;
        bool selected_compressed_indices = false;
        std::span<const uint32_t> compressed_indices;
        std::span<float> logits;
    };
    std::vector<AttentionRowContext> row_contexts(input.rows());

    auto prepare_attention_row = [&](size_t row_index) {
        {
            CpuLayerCache& cache = *caches[row_index];
            const uint64_t position = positions[row_index];
            const size_t window_elements = static_cast<size_t>(plan.sliding_window)
                                           * plan.head_dimension;
            if (cache.latent_window.size() != window_elements)
                cache.latent_window.assign(window_elements, 0.0f);
            cache.columns = plan.head_dimension;
            cache.capacity_tokens = plan.sliding_window;
            cache.latent_cache = true;
            record_latent_cache_undo(cache, plan, position);
            cache.latent_token_count = position + 1;
            float* key = key_value.row(row_index);
            if (prepared_latent_rope_enabled(optimization_flags))
            {
                prepare_rope_coefficients(
                    plan.rope_head_dimension,
                    position,
                    plan,
                    cache.latent_rope_cosines,
                    cache.latent_rope_sines);
                apply_prepared_rope(
                    key + plan.head_dimension - plan.rope_head_dimension,
                    plan.rope_head_dimension,
                    cache.latent_rope_cosines,
                    cache.latent_rope_sines,
                    false);
            }
            else
            {
                apply_rope(
                    key + plan.head_dimension - plan.rope_head_dimension,
                    plan.rope_head_dimension,
                    position,
                    plan,
                    false);
            }
            quantize_float8_e4m3_inplace(key, plan.head_dimension - plan.rope_head_dimension, 64, true, optimization_flags);
            std::copy_n(key, plan.head_dimension, cache.latent_window.data() + static_cast<size_t>(position % plan.sliding_window) * plan.head_dimension);

            CpuBatch& token_input = cache.latent_token_input;
            CpuBatch& token_rank = cache.latent_token_rank;
            if (plan.compression_ratio != 0)
            {
                token_input.reset(1, normalized.columns(), false);
                std::copy_n(normalized.row(row_index), normalized.columns(), token_input.row(0));
                token_rank.clear();
                if (query_rank.rows() != 0)
                {
                    token_rank.reset(1, query_rank.columns(), false);
                    std::copy_n(query_rank.row(row_index), query_rank.columns(), token_rank.row(0));
                }
                const float* fused_values = nullptr;
                const float* fused_scores = nullptr;
                if (fused_compressor)
                {
                    fused_values = fused_compressor_values.row(row_index);
                    fused_scores = fused_compressor_scores.row(row_index);
                }
                append_compressed_value(
                    weights,
                    operators,
                    plan,
                    backend,
                    token_input,
                    position,
                    norm_epsilon,
                    false,
                    cache,
                    optimization_flags,
                    fused_values,
                    fused_scores);
                if (plan.compression_ratio == 4)
                {
                    fused_values = nullptr;
                    fused_scores = nullptr;
                    if (fused_compressor)
                    {
                        fused_values = fused_index_compressor_values.row(row_index);
                        fused_scores = fused_index_compressor_scores.row(row_index);
                    }
                append_compressed_value(
                    weights,
                    operators,
                    plan,
                    backend,
                    token_input,
                    position,
                    norm_epsilon,
                    true,
                    cache,
                    optimization_flags,
                        fused_values,
                        fused_scores);
                }
            }
            AttentionRowContext& context = row_contexts[row_index];
            context.compressed_count = static_cast<uint32_t>(cache.latent_compressed.size() / plan.head_dimension);
            context.selected_compressed_indices = select_compressed_indices(weights, operators, plan, backend, token_input, token_rank, position, cache, optimization_flags);
            if (context.selected_compressed_indices)
                context.compressed_indices = cache.latent_selected_indices;
            else
                context.compressed_indices = {};
            context.window_begin = position + 1 > plan.sliding_window ? position + 1 - plan.sliding_window : 0;
            context.window_count = static_cast<uint32_t>(position + 1 - context.window_begin);
            const uint32_t selected_count = context.selected_compressed_indices ? static_cast<uint32_t>(context.compressed_indices.size()) : context.compressed_count;
            const uint32_t candidate_count = context.window_count + selected_count;
            const bool vector_softmax_for_row =
                use_vector_latent_softmax
                && candidate_count >= vector_latent_softmax_min_candidates;
            if (use_online_latent_softmax && !vector_softmax_for_row)
            {
                cache.latent_attention_logits.clear();
                context.logits = {};
            }
            else
            {
                cache.latent_attention_logits.resize(
                    static_cast<size_t>(plan.head_count) * candidate_count);
                context.logits = cache.latent_attention_logits;
            }
        }
    };

    auto execute_attention_head = [&](size_t row_index, uint32_t head) {
        CpuLayerCache& cache = *caches[row_index];
        const uint64_t position = positions[row_index];
        AttentionRowContext& context = row_contexts[row_index];
        const uint32_t selected_count = context.selected_compressed_indices ? static_cast<uint32_t>(context.compressed_indices.size()) : context.compressed_count;
        const uint32_t candidate_count = context.window_count + selected_count;
        float* query_head = query.row(row_index) + static_cast<size_t>(head) * plan.head_dimension;
        if (prepared_latent_rope_enabled(optimization_flags))
        {
            normalize_unit_prepared_rope(
                query_head,
                plan.head_dimension,
                plan.rope_head_dimension,
                norm_epsilon,
                cache.latent_rope_cosines,
                cache.latent_rope_sines,
                optimization_flags);
        }
        else
        {
            normalize_unit(query_head, plan.head_dimension, norm_epsilon, optimization_flags);
            apply_rope(
                query_head + plan.head_dimension - plan.rope_head_dimension,
                plan.rope_head_dimension,
                position,
                plan,
                false);
        }
        float* output_head = attention_output.row(row_index)
                             + static_cast<size_t>(head) * plan.head_dimension;
        const bool use_vector_softmax =
            use_vector_latent_softmax
            && candidate_count >= vector_latent_softmax_min_candidates;
        if (use_online_latent_softmax && !use_vector_softmax)
        {
            // The latent attention value is the same vector used as its key.
            // Online softmax therefore avoids materializing logits and a
            // second candidate traversal while retaining the sink mass.
            float maximum = sinks[head];
            float denominator = 1.0f;
            for (uint32_t candidate = 0; candidate < candidate_count;
                 ++candidate)
            {
                const float* candidate_key = nullptr;
                if (candidate < context.window_count)
                {
                    const uint64_t candidate_position =
                        context.window_begin + candidate;
                    candidate_key = cache.latent_window.data()
                                    + static_cast<size_t>(
                                          candidate_position
                                          % plan.sliding_window)
                                          * plan.head_dimension;
                }
                else
                {
                    const uint32_t compressed_offset =
                        candidate - context.window_count;
                    const uint32_t compressed_index =
                        context.selected_compressed_indices
                            ? context.compressed_indices[compressed_offset]
                            : compressed_offset;
                    candidate_key = cache.latent_compressed.data()
                                    + static_cast<size_t>(compressed_index)
                                          * plan.head_dimension;
                }
                const float score = float_dot(
                                       query_head, candidate_key,
                                       plan.head_dimension)
                                    * softmax_scale;
                if (score > maximum)
                {
                    const float rescale = std::exp(maximum - score);
                    float_scale_inplace(
                        output_head, rescale, plan.head_dimension);
                    denominator *= rescale;
                    maximum = score;
                }
                const float weight = std::exp(score - maximum);
                denominator += weight;
                float_scaled_add(
                    output_head, candidate_key, weight,
                    plan.head_dimension);
            }
            float_scale_inplace(
                output_head, 1.0f / denominator, plan.head_dimension);
            if (prepared_latent_rope_enabled(optimization_flags))
            {
                apply_prepared_rope(
                    output_head + plan.head_dimension
                        - plan.rope_head_dimension,
                    plan.rope_head_dimension,
                    cache.latent_rope_cosines,
                    cache.latent_rope_sines,
                    true);
            }
            else
            {
                apply_rope(
                    output_head + plan.head_dimension
                        - plan.rope_head_dimension,
                    plan.rope_head_dimension,
                    position,
                    plan,
                    true);
            }
            return;
        }
        float* logits = context.logits.data()
                        + static_cast<size_t>(head) * candidate_count;
        float maximum = sinks[head];
        for (uint32_t candidate = 0; candidate < candidate_count; ++candidate)
        {
            const float* candidate_key = nullptr;
            if (candidate < context.window_count)
            {
                const uint64_t candidate_position = context.window_begin + candidate;
                candidate_key = cache.latent_window.data() + static_cast<size_t>(candidate_position % plan.sliding_window) * plan.head_dimension;
            }
            else
            {
                const uint32_t compressed_offset = candidate - context.window_count;
                const uint32_t compressed_index = context.selected_compressed_indices ? context.compressed_indices[compressed_offset] : compressed_offset;
                candidate_key = cache.latent_compressed.data() + static_cast<size_t>(compressed_index) * plan.head_dimension;
            }
            const float dot = float_dot(query_head, candidate_key, plan.head_dimension);
            logits[candidate] = dot * softmax_scale;
            maximum = std::max(maximum, logits[candidate]);
        }
        float denominator = std::exp(sinks[head] - maximum);
        for (uint32_t candidate = 0; candidate < candidate_count; ++candidate)
            logits[candidate] -= maximum;
        if (use_vector_softmax)
            float_exp_inplace(logits, candidate_count);
        else
        {
            for (uint32_t candidate = 0; candidate < candidate_count; ++candidate)
                logits[candidate] = std::exp(logits[candidate]);
        }
        for (uint32_t candidate = 0; candidate < candidate_count; ++candidate)
            denominator += logits[candidate];
        for (uint32_t candidate = 0; candidate < candidate_count; ++candidate)
        {
            const float* candidate_key = nullptr;
            if (candidate < context.window_count)
            {
                const uint64_t candidate_position = context.window_begin + candidate;
                candidate_key = cache.latent_window.data() + static_cast<size_t>(candidate_position % plan.sliding_window) * plan.head_dimension;
            }
            else
            {
                const uint32_t compressed_offset = candidate - context.window_count;
                const uint32_t compressed_index = context.selected_compressed_indices ? context.compressed_indices[compressed_offset] : compressed_offset;
                candidate_key = cache.latent_compressed.data() + static_cast<size_t>(compressed_index) * plan.head_dimension;
            }
            const float probability = logits[candidate] / denominator;
            float_scaled_add(output_head, candidate_key, probability, plan.head_dimension);
        }
        if (prepared_latent_rope_enabled(optimization_flags))
        {
            apply_prepared_rope(
                output_head + plan.head_dimension - plan.rope_head_dimension,
                plan.rope_head_dimension,
                cache.latent_rope_cosines,
                cache.latent_rope_sines,
                true);
        }
        else
        {
            apply_rope(
                output_head + plan.head_dimension - plan.rope_head_dimension,
                plan.rope_head_dimension,
                position,
                plan,
                true);
        }
    };

    bool independent_caches = true;
    for (size_t row_index = 0; row_index < caches.size() && independent_caches; ++row_index)
    {
        for (size_t previous = 0; previous < row_index; ++previous)
        {
            if (caches[previous] == caches[row_index])
            {
                independent_caches = false;
                break;
            }
        }
    }
    if (!independent_caches)
    {
        for (size_t row_index = 0; row_index < input.rows(); ++row_index)
        {
            prepare_attention_row(row_index);
            for (uint32_t head = 0; head < plan.head_count; ++head)
                execute_attention_head(row_index, head);
        }
    }
    else
    {
        for (size_t row_index = 0; row_index < input.rows(); ++row_index)
            prepare_attention_row(row_index);
        const int64_t task_count = static_cast<int64_t>(input.rows()) * plan.head_count;
#if defined(_OPENMP)
        const int attention_team_size = std::max(1, std::min(static_cast<int>(task_count), omp_get_max_threads()));
#pragma omp parallel for schedule(static) num_threads(attention_team_size) if (attention_team_size > 1)
#endif
        for (int64_t task = 0; task < task_count; ++task)
        {
            const size_t row_index = static_cast<size_t>(task) / plan.head_count;
            const uint32_t head = static_cast<uint32_t>(task % plan.head_count);
            execute_attention_head(row_index, head);
        }
    }

    const TensorData& output_a = weights.at(plan.output_a_weight);
    const TensorData& output_b = weights.at(plan.output_b_weight);
    const CompiledOperator& output_a_operator = operators.at_weight(plan.output_a_weight);
    const CompiledOperator& output_b_operator = operators.at_weight(plan.output_b_weight);
    if (backend == ExecutionBackend::Vulkan
        && output_a_operator.float8
        && output_b_operator.float8)
    {
        CpuBatch chained_output;
        if (output_a_operator.float8->forward_chain(attention_output, *output_b_operator.float8, chained_output))
            return chained_output;
    }
    CpuBatch output_rank;
    if (backend == ExecutionBackend::Vulkan && output_a_operator.float8)
    {
        linear_batch_into(output_a, attention_output, output_rank, optimization_flags, &output_a_operator, backend);
    }
    else
    {
        output_rank.reset(input.rows(), plan.output_group_count * plan.output_lora_rank, false);
        const uint32_t heads_per_group = plan.head_count / plan.output_group_count;
        const uint32_t group_columns = heads_per_group * plan.head_dimension;
#if defined(_OPENMP)
        const int output_team_size = latent_output_group_team_size(
            input.rows(), plan.output_group_count, group_columns,
            plan.output_lora_rank, optimization_flags);
        const int64_t output_tasks = static_cast<int64_t>(input.rows())
                                     * plan.output_group_count;
#pragma omp parallel for schedule(static) num_threads(output_team_size) if(output_team_size > 1)
#else
        const int64_t output_tasks = static_cast<int64_t>(input.rows())
                                     * plan.output_group_count;
#endif
        for (int64_t task = 0; task < output_tasks; ++task)
        {
            const size_t row_index = static_cast<size_t>(task)
                                     / plan.output_group_count;
            const uint32_t group = static_cast<uint32_t>(task)
                                   % plan.output_group_count;
            const float* group_input = attention_output.row(row_index)
                                       + static_cast<size_t>(group)
                                             * group_columns;
            float* group_output = output_rank.row(row_index)
                                  + static_cast<size_t>(group)
                                        * plan.output_lora_rank;
            fp8_matrix_rows_dot(
                output_a,
                group * plan.output_lora_rank,
                plan.output_lora_rank,
                group_input,
                group_output);
        }
    }
    return linear_batch(output_b, output_rank, optimization_flags, &output_b_operator, backend);
}

Result<CpuBatch> execute_latent_attention(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& input,
    uint64_t optimization_flags)
{
    if (input.rows() == 1)
    {
        const std::array<uint64_t, 1> positions = {position_offset};
        std::array<CpuLayerCache*, 1> caches = {&cache};
        return execute_latent_attention_rows(weights, operators, plan, backend, norm_epsilon, positions, caches, input, optimization_flags);
    }
    std::vector<uint64_t> positions(input.rows());
    std::vector<CpuLayerCache*> caches(input.rows(), &cache);
    for (size_t row = 0; row < input.rows(); ++row)
        positions[row] = position_offset + row;
    return execute_latent_attention_rows(weights, operators, plan, backend, norm_epsilon, positions, caches, input, optimization_flags);
}

Result<CpuBatch> execute_latent_attention_batch(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    std::span<const uint64_t> positions,
    std::span<CpuLayerCache* const> caches,
    const CpuBatch& input,
    uint64_t optimization_flags)
{
    return execute_latent_attention_rows(weights, operators, plan, backend, norm_epsilon, positions, caches, input, optimization_flags);
}

Result<void> append_dspark_attention_context(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& input,
    uint64_t optimization_flags)
{
    if (!has_flag(plan.flags, AttentionBlockLatent)
        || plan.compression_ratio != 0
        || plan.sliding_window == 0
        || input.columns() != weights.at(plan.key_value_weight).shape[1])
    {
        return Error{ErrorCode::InvalidArgument, "invalid DSpark context append"};
    }
    CpuBatch key_value = linear_batch(
        weights.at(plan.key_value_weight),
        input,
        optimization_flags,
        &operators.at_weight(plan.key_value_weight),
        backend);
    key_value = rms_norm_batch(key_value, weights.at(plan.key_value_norm_weight), norm_epsilon, 0.0f, optimization_flags);
    const size_t window_elements = static_cast<size_t>(plan.sliding_window) * plan.head_dimension;
    if (cache.latent_window.size() != window_elements)
        cache.latent_window.assign(window_elements, 0.0f);
    cache.columns = plan.head_dimension;
    cache.capacity_tokens = plan.sliding_window;
    cache.latent_cache = true;
    for (size_t row = 0; row < input.rows(); ++row)
    {
        const uint64_t position = position_offset + row;
        record_latent_cache_undo(cache, plan, position);
        float* key = key_value.row(row);
        apply_rope(key + plan.head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, false);
        quantize_float8_e4m3_inplace(key, plan.head_dimension - plan.rope_head_dimension, 64, true, optimization_flags);
        std::copy_n(key, plan.head_dimension, cache.latent_window.data() + static_cast<size_t>(position % plan.sliding_window) * plan.head_dimension);
        cache.latent_token_count = position + 1;
    }
    return {};
}

Result<CpuBatch> execute_dspark_attention(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    uint64_t position_offset,
    const CpuLayerCache& cache,
    const CpuBatch& input,
    uint64_t optimization_flags)
{
    if (!has_flag(plan.flags, AttentionBlockLatent)
        || plan.compression_ratio != 0
        || plan.sliding_window == 0
        || input.rows() == 0
        || input.columns() != weights.at(plan.pre_attention_norm_weight).shape[0]
        || cache.latent_window.size() != static_cast<size_t>(plan.sliding_window) * plan.head_dimension)
    {
        return Error{ErrorCode::InvalidArgument, "invalid DSpark attention execution"};
    }

    CpuBatch normalized;
    CpuBatch query;
    CpuBatch key_value;
    const TensorData& query_a = weights.at(plan.query_a_weight);
    const TensorData& query_b = weights.at(plan.query_b_weight);
    const TensorData& key_value_weight = weights.at(plan.key_value_weight);
    const CompiledOperator& query_a_operator = operators.at_weight(plan.query_a_weight);
    const CompiledOperator& query_b_operator = operators.at_weight(plan.query_b_weight);
    const CompiledOperator& key_value_operator = operators.at_weight(plan.key_value_weight);
    bool chained_query = false;
    if (backend == ExecutionBackend::Vulkan
        && query_a_operator.float8
        && query_b_operator.float8)
    {
        if (key_value_operator.float8)
        {
            if (runtime_optimization_enabled(optimization_flags,
                    RuntimeOptimizationVulkanLatentInputRmsNorm))
            {
                chained_query = query_a_operator.float8->forward_input_rms_norm_chain_parallel(
                    input,
                    *query_b_operator.float8,
                    *key_value_operator.float8,
                    query,
                    key_value);
            }
            if (!chained_query)
            {
                if (normalized.rows() == 0)
                    normalized = rms_norm_batch(
                        input,
                        weights.at(plan.pre_attention_norm_weight),
                        norm_epsilon, 0.0f, optimization_flags);
                chained_query = query_a_operator.float8->forward_rms_norm_chain_parallel(
                    normalized,
                    *query_b_operator.float8,
                    *key_value_operator.float8,
                    query,
                    key_value);
            }
        }
        else
        {
            normalized = rms_norm_batch(
                input,
                weights.at(plan.pre_attention_norm_weight),
                norm_epsilon, 0.0f, optimization_flags);
            chained_query = query_a_operator.float8->forward_rms_norm_chain(normalized, *query_b_operator.float8, query);
        }
    }
    if (!chained_query)
    {
        if (normalized.rows() == 0)
            normalized = rms_norm_batch(
                input,
                weights.at(plan.pre_attention_norm_weight),
                norm_epsilon, 0.0f, optimization_flags);
        CpuBatch query_rank;
        if (!float8_linear_pair_batch_into(
                query_a, key_value_weight, normalized, query_rank,
                key_value, optimization_flags, &query_a_operator, &key_value_operator))
        {
            query_rank = linear_batch(query_a, normalized, optimization_flags, &query_a_operator, backend);
        }
        if (!float8_linear_rms_norm_batch_into(
                query_b, query_rank, weights.at(plan.query_norm_weight),
                norm_epsilon, query, optimization_flags, &query_b_operator))
        {
            query_rank = rms_norm_batch(
                query_rank, weights.at(plan.query_norm_weight),
                norm_epsilon, 0.0f, optimization_flags);
            query = linear_batch(query_b, query_rank, optimization_flags, &query_b_operator, backend);
        }
    }
    if (key_value.rows() == 0)
        key_value = linear_batch(key_value_weight, normalized, optimization_flags, &key_value_operator, backend);
    key_value = rms_norm_batch(key_value, weights.at(plan.key_value_norm_weight), norm_epsilon, 0.0f, optimization_flags);

    for (size_t row = 0; row < input.rows(); ++row)
    {
        const uint64_t position = position_offset + row;
        float* key = key_value.row(row);
        apply_rope(key + plan.head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, false);
        quantize_float8_e4m3_inplace(key, plan.head_dimension - plan.rope_head_dimension, 64, true, optimization_flags);
    }

    CpuBatch attention_output(input.rows(), plan.head_count * plan.head_dimension);
    const std::span<const float> sinks = weights.at(plan.sinks).float32_values();
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(plan.head_dimension));
    const uint32_t window_count = static_cast<uint32_t>(std::min<uint64_t>(cache.latent_token_count, plan.sliding_window));
    const uint64_t window_begin = cache.latent_token_count - window_count;
    const uint32_t candidate_count = window_count + static_cast<uint32_t>(input.rows());
    const bool use_vector_softmax =
        vector_latent_softmax_enabled(optimization_flags)
        && candidate_count >= vector_latent_softmax_min_candidates;
    std::vector<float> logits(candidate_count);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        const uint64_t position = position_offset + row;
        for (uint32_t head = 0; head < plan.head_count; ++head)
        {
            float* query_head = query.row(row) + static_cast<size_t>(head) * plan.head_dimension;
            normalize_unit(query_head, plan.head_dimension, norm_epsilon, optimization_flags);
            apply_rope(query_head + plan.head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, false);
            float maximum = sinks[head];
            for (uint32_t candidate = 0; candidate < candidate_count; ++candidate)
            {
                const float* key = candidate < window_count
                                       ? cache.latent_window.data()
                                             + static_cast<size_t>(
                                                   (window_begin + candidate)
                                                   % plan.sliding_window)
                                                   * plan.head_dimension
                                       : key_value.row(candidate - window_count);
                const float dot = float_dot(query_head, key, plan.head_dimension);
                logits[candidate] = dot * softmax_scale;
                maximum = std::max(maximum, logits[candidate]);
            }
            float denominator = std::exp(sinks[head] - maximum);
            for (float& logit : logits)
                logit -= maximum;
            if (use_vector_softmax)
                float_exp_inplace(logits.data(), candidate_count);
            else
            {
                for (float& logit : logits)
                    logit = std::exp(logit);
            }
            for (float logit : logits)
                denominator += logit;
            float* output = attention_output.row(row)
                            + static_cast<size_t>(head) * plan.head_dimension;
            for (uint32_t candidate = 0; candidate < candidate_count; ++candidate)
            {
                const float* key = candidate < window_count
                                       ? cache.latent_window.data()
                                             + static_cast<size_t>(
                                                   (window_begin + candidate)
                                                   % plan.sliding_window)
                                                   * plan.head_dimension
                                       : key_value.row(candidate - window_count);
                const float probability = logits[candidate] / denominator;
                float_scaled_add(output, key, probability, plan.head_dimension);
            }
            apply_rope(output + plan.head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, true);
        }
    }

    const TensorData& output_a = weights.at(plan.output_a_weight);
    const TensorData& output_b = weights.at(plan.output_b_weight);
    const CompiledOperator& output_a_operator = operators.at_weight(plan.output_a_weight);
    const CompiledOperator& output_b_operator = operators.at_weight(plan.output_b_weight);
    if (backend == ExecutionBackend::Vulkan
        && output_a_operator.float8
        && output_b_operator.float8)
    {
        CpuBatch chained_output;
        if (output_a_operator.float8->forward_chain(attention_output, *output_b_operator.float8, chained_output))
        {
            return chained_output;
        }
    }
    CpuBatch output_rank;
    if (backend == ExecutionBackend::Vulkan && output_a_operator.float8)
    {
        linear_batch_into(output_a, attention_output, output_rank, optimization_flags, &output_a_operator, backend);
    }
    else
    {
        output_rank.reset(input.rows(), plan.output_group_count * plan.output_lora_rank, false);
        const uint32_t heads_per_group = plan.head_count / plan.output_group_count;
        const uint32_t group_columns = heads_per_group * plan.head_dimension;
#if defined(_OPENMP)
        const int output_team_size = latent_output_group_team_size(
            input.rows(), plan.output_group_count, group_columns,
            plan.output_lora_rank, optimization_flags);
        const int64_t output_tasks = static_cast<int64_t>(input.rows())
                                     * plan.output_group_count;
#pragma omp parallel for schedule(static) num_threads(output_team_size) if(output_team_size > 1)
#else
        const int64_t output_tasks = static_cast<int64_t>(input.rows())
                                     * plan.output_group_count;
#endif
        for (int64_t task = 0; task < output_tasks; ++task)
        {
            const size_t row = static_cast<size_t>(task)
                               / plan.output_group_count;
            const uint32_t group = static_cast<uint32_t>(task)
                                   % plan.output_group_count;
            const float* group_input = attention_output.row(row)
                                       + static_cast<size_t>(group)
                                             * group_columns;
            float* group_output = output_rank.row(row)
                                  + static_cast<size_t>(group)
                                        * plan.output_lora_rank;
            fp8_matrix_rows_dot(
                output_a,
                group * plan.output_lora_rank,
                plan.output_lora_rank,
                group_input,
                group_output);
        }
    }
    return linear_batch(
        output_b,
        output_rank,
        optimization_flags,
        &output_b_operator,
        backend);
}

} // namespace moe
} // namespace ncnn
