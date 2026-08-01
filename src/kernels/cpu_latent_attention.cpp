#include "cpu_latent_attention.h"

#include "backends/ncnn/ncnn_linear.h"
#include "cpu_float8.h"
#include "cpu_ops.h"
#include "cpu_vector.h"
#include "engine/cpu_session_state.h"

#include <algorithm>
#include <array>
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
    if (!cache.latent_transaction_active)
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
    cache.latent_transaction_undo.push_back(std::move(undo));
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
        cache.latent_transaction_undo.clear();
        cache.latent_transaction_active = true;
    }
}

Result<void> finish_latent_cache_transaction(std::span<CpuLayerCache> caches, size_t committed_rows)
{
    for (const CpuLayerCache& cache : caches)
    {
        if (cache.latent_transaction_active && !cache.latent_transaction_undo.empty()
            && cache.latent_transaction_undo.size() < committed_rows)
        {
            return Error{
                ErrorCode::InternalError,
                "latent cache transaction is missing committed rows"};
        }
    }
    for (CpuLayerCache& cache : caches)
    {
        if (!cache.latent_transaction_active)
            continue;
        while (cache.latent_transaction_undo.size() > committed_rows)
        {
            restore_latent_cache_undo(cache, cache.latent_transaction_undo.back());
            cache.latent_transaction_undo.pop_back();
        }
        cache.latent_transaction_undo.clear();
        cache.latent_transaction_active = false;
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

static void normalize_vector(float* values, uint32_t count, const TensorData& weight, float epsilon)
{
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += values[index] * values[index];
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    if (weight.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> weights = weight.bfloat16_values();
        for (uint32_t index = 0; index < count; ++index)
            values[index] *= inverse_rms * bfloat16_to_float(weights[index]);
    }
    else
    {
        const std::span<const float> weights = weight.float32_values();
        for (uint32_t index = 0; index < count; ++index)
            values[index] *= inverse_rms * weights[index];
    }
}

static void normalize_unit(float* values, uint32_t count, float epsilon)
{
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += values[index] * values[index];
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
        values[index] *= inverse_rms;
}

static bool scored_index_precedes(
    const CpuLayerCache::LatentScoredIndex& left,
    const CpuLayerCache::LatentScoredIndex& right)
{
    return left.score > right.score || (left.score == right.score && left.index < right.index);
}

static void append_compressed_value(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    const CpuBatch& input,
    uint64_t position,
    float norm_epsilon,
    bool indexer,
    CpuLayerCache& cache)
{
    const uint32_t ratio = plan.compression_ratio;
    const uint32_t dimension = indexer ? plan.index_head_dimension : plan.head_dimension;
    const TensorHandle value_handle = indexer ? plan.indexer_compressor_key_value_weight : plan.compressor_key_value_weight;
    const TensorHandle gate_handle = indexer ? plan.indexer_compressor_gate_weight : plan.compressor_gate_weight;
    const TensorHandle position_handle = indexer ? plan.indexer_compressor_position : plan.compressor_position;
    const TensorHandle norm_handle = indexer ? plan.indexer_compressor_norm_weight : plan.compressor_norm_weight;
    const uint32_t projection_multiplier = ratio == 4 ? 2 : 1;
    linear_batch_into(weights.at(value_handle), input, cache.compressor_values);
    linear_batch_into(weights.at(gate_handle), input, cache.compressor_scores);

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
    const float* projected_values = cache.compressor_values.row(0);
    const float* projected_scores = cache.compressor_scores.row(0);
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

    normalize_vector(pooled.data(), dimension, weights.at(norm_handle), norm_epsilon);
    apply_rope(pooled.data() + dimension - plan.rope_head_dimension, plan.rope_head_dimension, position + 1 - ratio, plan, false);
    if (indexer)
    {
        hadamard_rotate(pooled.data(), dimension);
        quantize_float4_e2m1_inplace(pooled.data(), dimension, 32);
    }
    else
    {
        quantize_float8_e4m3_inplace(pooled.data(), dimension - plan.rope_head_dimension, 64, true);
    }
    compressed.insert(compressed.end(), pooled.begin(), pooled.begin() + dimension);
}

static bool select_compressed_indices(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    const CpuBatch& normalized,
    const CpuBatch& query_rank,
    uint64_t position,
    CpuLayerCache& cache)
{
    const uint32_t compressed_count = static_cast<uint32_t>(cache.latent_compressed.size() / plan.head_dimension);
    cache.latent_selected_indices.clear();
    if (plan.compression_ratio != 4 || compressed_count <= plan.index_top_k)
        return false;

    CpuBatch& query = cache.latent_index_query;
    linear_batch_into(weights.at(plan.indexer_query_weight), query_rank, query);
    for (uint32_t head = 0; head < plan.index_head_count; ++head)
    {
        float* values = query.row(0) + static_cast<size_t>(head) * plan.index_head_dimension;
        apply_rope(values + plan.index_head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, false);
        hadamard_rotate(values, plan.index_head_dimension);
        quantize_float4_e2m1_inplace(values, plan.index_head_dimension, 32);
    }
    CpuBatch& projected_weights = cache.latent_index_projected_weights;
    linear_batch_into(weights.at(plan.indexer_weights_weight), normalized, projected_weights);
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
    std::partial_sort(scored.begin(), scored.begin() + selected_count, scored.end(), scored_index_precedes);
    std::vector<uint32_t>& selected_indices = cache.latent_selected_indices;
    selected_indices.resize(selected_count);
    for (uint32_t index = 0; index < selected_count; ++index)
        selected_indices[index] = scored[index].index;
    return true;
}

static float fp8_matrix_row_dot(const TensorData& matrix, uint32_t row, const float* input)
{
    const uint32_t columns = matrix.shape[1];
    const uint32_t input_blocks = (columns + 127) / 128;
    return float8_e4m3_block_dot(
        matrix.float8_values().data() + static_cast<size_t>(row) * columns,
        matrix.quantization_scales.data() + static_cast<size_t>(row / 128) * input_blocks,
        input,
        columns,
        128);
}

static Result<CpuBatch> execute_latent_attention_rows(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    std::span<const uint64_t> positions,
    std::span<CpuLayerCache* const> caches,
    const CpuBatch& input)
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

    CpuBatch normalized = rms_norm_batch(input, weights.at(plan.pre_attention_norm_weight), norm_epsilon);
    CpuBatch query_rank;
    CpuBatch query;
    const TensorData& query_a = weights.at(plan.query_a_weight);
    const TensorData& query_b = weights.at(plan.query_b_weight);
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
    bool chained_query = false;
    if (query_rank_not_required && query_a.float8_linear_operator && query_b.float8_linear_operator)
    {
        if (key_value_weight.float8_linear_operator)
        {
            chained_query = query_a.float8_linear_operator->forward_rms_norm_chain_parallel(
                normalized,
                *query_b.float8_linear_operator,
                *key_value_weight.float8_linear_operator,
                query,
                key_value);
        }
        else
        {
            chained_query = query_a.float8_linear_operator->forward_rms_norm_chain(normalized, *query_b.float8_linear_operator, query);
        }
    }
    if (!chained_query)
    {
        key_value = CpuBatch();
        query_rank = linear_batch(query_a, normalized);
        query_rank = rms_norm_batch(query_rank, weights.at(plan.query_norm_weight), norm_epsilon);
        query = linear_batch(query_b, query_rank);
    }
    if (key_value.rows() == 0)
        key_value = linear_batch(key_value_weight, normalized);
    key_value = rms_norm_batch(key_value, weights.at(plan.key_value_norm_weight), norm_epsilon);
    CpuBatch attention_output(input.rows(), plan.head_count * plan.head_dimension);

    const std::span<const float> sinks = weights.at(plan.sinks).float32_values();
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(plan.head_dimension));

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
            apply_rope(key + plan.head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, false);
            quantize_float8_e4m3_inplace(key, plan.head_dimension - plan.rope_head_dimension, 64, true);
            std::copy_n(key, plan.head_dimension, cache.latent_window.data() + static_cast<size_t>(position % plan.sliding_window) * plan.head_dimension);

            CpuBatch& token_input = cache.latent_token_input;
            token_input.reset(1, normalized.columns(), false);
            std::copy_n(normalized.row(row_index), normalized.columns(), token_input.row(0));
            CpuBatch& token_rank = cache.latent_token_rank;
            token_rank.clear();
            if (query_rank.rows() != 0)
            {
                token_rank.reset(1, query_rank.columns(), false);
                std::copy_n(query_rank.row(row_index), query_rank.columns(), token_rank.row(0));
            }
            if (plan.compression_ratio != 0)
            {
                append_compressed_value(weights, plan, token_input, position, norm_epsilon, false, cache);
                if (plan.compression_ratio == 4)
                    append_compressed_value(weights, plan, token_input, position, norm_epsilon, true, cache);
            }
            AttentionRowContext& context = row_contexts[row_index];
            context.compressed_count = static_cast<uint32_t>(cache.latent_compressed.size() / plan.head_dimension);
            context.selected_compressed_indices = select_compressed_indices(weights, plan, token_input, token_rank, position, cache);
            if (context.selected_compressed_indices)
                context.compressed_indices = cache.latent_selected_indices;
            else
                context.compressed_indices = {};
            context.window_begin = position + 1 > plan.sliding_window ? position + 1 - plan.sliding_window : 0;
            context.window_count = static_cast<uint32_t>(position + 1 - context.window_begin);
            const uint32_t selected_count = context.selected_compressed_indices ? static_cast<uint32_t>(context.compressed_indices.size()) : context.compressed_count;
            const uint32_t candidate_count = context.window_count + selected_count;
            cache.latent_attention_logits.resize(static_cast<size_t>(plan.head_count) * candidate_count);
            context.logits = cache.latent_attention_logits;
        }
    };

    auto execute_attention_head = [&](size_t row_index, uint32_t head) {
        CpuLayerCache& cache = *caches[row_index];
        const uint64_t position = positions[row_index];
        AttentionRowContext& context = row_contexts[row_index];
        const uint32_t selected_count = context.selected_compressed_indices ? static_cast<uint32_t>(context.compressed_indices.size()) : context.compressed_count;
        const uint32_t candidate_count = context.window_count + selected_count;
        float* logits = context.logits.data() + static_cast<size_t>(head) * candidate_count;
        float* query_head = query.row(row_index) + static_cast<size_t>(head) * plan.head_dimension;
        normalize_unit(query_head, plan.head_dimension, norm_epsilon);
        apply_rope(query_head + plan.head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, false);
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
        {
            logits[candidate] = std::exp(logits[candidate] - maximum);
            denominator += logits[candidate];
        }
        float* output_head = attention_output.row(row_index) + static_cast<size_t>(head) * plan.head_dimension;
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
        apply_rope(output_head + plan.head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, true);
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
    if (output_a.float8_linear_operator && output_b.float8_linear_operator)
    {
        CpuBatch chained_output;
        if (output_a.float8_linear_operator->forward_chain(attention_output, *output_b.float8_linear_operator, chained_output))
            return chained_output;
    }
    CpuBatch output_rank;
    if (output_a.float8_linear_operator)
    {
        linear_batch_into(output_a, attention_output, output_rank);
    }
    else
    {
        output_rank.reset(input.rows(), plan.output_group_count * plan.output_lora_rank, false);
        const uint32_t heads_per_group = plan.head_count / plan.output_group_count;
        const uint32_t group_columns = heads_per_group * plan.head_dimension;
        for (size_t row_index = 0; row_index < input.rows(); ++row_index)
        {
            for (uint32_t group = 0; group < plan.output_group_count; ++group)
            {
                const float* group_input = attention_output.row(row_index) + static_cast<size_t>(group) * group_columns;
                for (uint32_t rank = 0; rank < plan.output_lora_rank; ++rank)
                {
                    const uint32_t matrix_row = group * plan.output_lora_rank + rank;
                    output_rank.row(row_index)[matrix_row] = fp8_matrix_row_dot(output_a, matrix_row, group_input);
                }
            }
        }
    }
    return linear_batch(output_b, output_rank);
}

Result<CpuBatch> execute_latent_attention(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& input)
{
    if (input.rows() == 1)
    {
        const std::array<uint64_t, 1> positions = {position_offset};
        std::array<CpuLayerCache*, 1> caches = {&cache};
        return execute_latent_attention_rows(weights, plan, norm_epsilon, positions, caches, input);
    }
    std::vector<uint64_t> positions(input.rows());
    std::vector<CpuLayerCache*> caches(input.rows(), &cache);
    for (size_t row = 0; row < input.rows(); ++row)
        positions[row] = position_offset + row;
    return execute_latent_attention_rows(weights, plan, norm_epsilon, positions, caches, input);
}

Result<CpuBatch> execute_latent_attention_batch(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    std::span<const uint64_t> positions,
    std::span<CpuLayerCache* const> caches,
    const CpuBatch& input)
{
    return execute_latent_attention_rows(weights, plan, norm_epsilon, positions, caches, input);
}

Result<void> append_dspark_attention_context(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& input)
{
    if (!has_flag(plan.flags, AttentionBlockLatent)
        || plan.compression_ratio != 0
        || plan.sliding_window == 0
        || input.columns() != weights.at(plan.key_value_weight).shape[1])
    {
        return Error{ErrorCode::InvalidArgument, "invalid DSpark context append"};
    }
    CpuBatch key_value = linear_batch(weights.at(plan.key_value_weight), input);
    key_value = rms_norm_batch(key_value, weights.at(plan.key_value_norm_weight), norm_epsilon);
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
        quantize_float8_e4m3_inplace(key, plan.head_dimension - plan.rope_head_dimension, 64, true);
        std::copy_n(key, plan.head_dimension, cache.latent_window.data() + static_cast<size_t>(position % plan.sliding_window) * plan.head_dimension);
        cache.latent_token_count = position + 1;
    }
    return {};
}

Result<CpuBatch> execute_dspark_attention(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    uint64_t position_offset,
    const CpuLayerCache& cache,
    const CpuBatch& input)
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

    CpuBatch normalized = rms_norm_batch(input, weights.at(plan.pre_attention_norm_weight), norm_epsilon);
    CpuBatch query;
    CpuBatch key_value;
    const TensorData& query_a = weights.at(plan.query_a_weight);
    const TensorData& query_b = weights.at(plan.query_b_weight);
    const TensorData& key_value_weight = weights.at(plan.key_value_weight);
    bool chained_query = false;
    if (query_a.float8_linear_operator && query_b.float8_linear_operator)
    {
        if (key_value_weight.float8_linear_operator)
        {
            chained_query = query_a.float8_linear_operator->forward_rms_norm_chain_parallel(
                normalized,
                *query_b.float8_linear_operator,
                *key_value_weight.float8_linear_operator,
                query,
                key_value);
        }
        else
        {
            chained_query = query_a.float8_linear_operator->forward_rms_norm_chain(normalized, *query_b.float8_linear_operator, query);
        }
    }
    if (!chained_query)
    {
        CpuBatch query_rank = linear_batch(query_a, normalized);
        query_rank = rms_norm_batch(query_rank, weights.at(plan.query_norm_weight), norm_epsilon);
        query = linear_batch(query_b, query_rank);
    }
    if (key_value.rows() == 0)
        key_value = linear_batch(key_value_weight, normalized);
    key_value = rms_norm_batch(key_value, weights.at(plan.key_value_norm_weight), norm_epsilon);

    for (size_t row = 0; row < input.rows(); ++row)
    {
        const uint64_t position = position_offset + row;
        float* key = key_value.row(row);
        apply_rope(key + plan.head_dimension - plan.rope_head_dimension, plan.rope_head_dimension, position, plan, false);
        quantize_float8_e4m3_inplace(key, plan.head_dimension - plan.rope_head_dimension, 64, true);
    }

    CpuBatch attention_output(input.rows(), plan.head_count * plan.head_dimension);
    const std::span<const float> sinks = weights.at(plan.sinks).float32_values();
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(plan.head_dimension));
    const uint32_t window_count = static_cast<uint32_t>(std::min<uint64_t>(cache.latent_token_count, plan.sliding_window));
    const uint64_t window_begin = cache.latent_token_count - window_count;
    const uint32_t candidate_count = window_count + static_cast<uint32_t>(input.rows());
    std::vector<float> logits(candidate_count);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        const uint64_t position = position_offset + row;
        for (uint32_t head = 0; head < plan.head_count; ++head)
        {
            float* query_head = query.row(row) + static_cast<size_t>(head) * plan.head_dimension;
            normalize_unit(query_head, plan.head_dimension, norm_epsilon);
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
            {
                logit = std::exp(logit - maximum);
                denominator += logit;
            }
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
    if (output_a.float8_linear_operator && output_b.float8_linear_operator)
    {
        CpuBatch chained_output;
        if (output_a.float8_linear_operator->forward_chain(attention_output, *output_b.float8_linear_operator, chained_output))
        {
            return chained_output;
        }
    }
    CpuBatch output_rank;
    if (output_a.float8_linear_operator)
    {
        linear_batch_into(output_a, attention_output, output_rank);
    }
    else
    {
        output_rank.reset(input.rows(), plan.output_group_count * plan.output_lora_rank, false);
        const uint32_t heads_per_group = plan.head_count / plan.output_group_count;
        const uint32_t group_columns = heads_per_group * plan.head_dimension;
        for (size_t row = 0; row < input.rows(); ++row)
        {
            for (uint32_t group = 0; group < plan.output_group_count; ++group)
            {
                const float* group_input = attention_output.row(row) + static_cast<size_t>(group) * group_columns;
                for (uint32_t rank = 0; rank < plan.output_lora_rank; ++rank)
                {
                    const uint32_t matrix_row = group * plan.output_lora_rank + rank;
                    output_rank.row(row)[matrix_row] = fp8_matrix_row_dot(output_a, matrix_row, group_input);
                }
            }
        }
    }
    return linear_batch(output_b, output_rank);
}

} // namespace moe
} // namespace ncnn
