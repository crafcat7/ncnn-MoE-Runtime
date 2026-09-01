#include "cpu_ple.h"

#include "cpu_ops.h"
#include "engine/cpu_session_state.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

static float ple_sigmoid(float value) noexcept
{
    return 1.0f / (1.0f + std::exp(-value));
}

static void grouped_rms_norm_into(const CpuBatch& input, const TensorData& weight,
                           uint32_t group_size, float epsilon,
                           float weight_offset, CpuBatch& output)
{
    output.reset(input.rows(), input.columns(), false);
    const std::span<const uint16_t> norm = weight.bfloat16_values();
    const uint32_t group_count = input.columns() / group_size;
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* source = input.row(row_index);
        float* destination = output.row(row_index);
        for (uint32_t group = 0; group < group_count; ++group)
        {
            const size_t offset = static_cast<size_t>(group) * group_size;
            float mean_square = 0.0f;
            for (uint32_t column = 0; column < group_size; ++column)
                mean_square += source[offset + column] * source[offset + column];
            const float scale = 1.0f / std::sqrt(mean_square / static_cast<float>(group_size) + epsilon);
            for (uint32_t column = 0; column < group_size; ++column)
            {
                destination[offset + column] = source[offset + column] * scale
                                               * (bfloat16_to_float(norm[offset + column]) + weight_offset);
            }
        }
    }
}

static int64_t wrapped_product(int64_t left, int64_t right) noexcept
{
    const uint64_t product = static_cast<uint64_t>(left) * static_cast<uint64_t>(right);
    return std::bit_cast<int64_t>(product);
}

static int64_t floor_remainder(int64_t value, int64_t divisor) noexcept
{
    const int64_t remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}

static Result<const uint16_t*> embedding_row(const WeightStore& weights,
                                      const PleBlockPlan& plan,
                                      uint64_t global_row,
                                      uint32_t head_dimension)
{
    uint64_t row = global_row;
    for (TensorHandle handle : plan.embedding_shards)
    {
        const TensorData& shard = weights.at(handle);
        if (row < shard.shape[0])
            return shard.bfloat16_values().data() + static_cast<size_t>(row) * head_dimension;
        row -= shard.shape[0];
    }
    return Error{ErrorCode::InvalidModel, "PLE hash index exceeds embedding shards"};
}

Result<void> execute_ple_into(
    const WeightStore& weights,
    const PleBlockPlan& plan,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    std::span<const int32_t> input_ids,
    CpuLayerCache& cache,
    CpuBatch& hidden,
    uint64_t optimization_flags)
{
    if (!plan.enabled())
        return {};
    const uint32_t expanded_size = multiplier * hidden_size;
    const uint32_t head_count = (plan.ngram_size - 1) * plan.heads_per_ngram;
    if (multiplier == 0 || hidden.rows() != input_ids.size()
        || hidden.columns() != expanded_size || plan.ngram_size < 2
        || head_count == 0 || plan.embedding_dimension % head_count != 0)
    {
        return Error{ErrorCode::InvalidArgument, "invalid PLE execution dimensions"};
    }
    const uint32_t head_dimension = plan.embedding_dimension / head_count;
    const TensorData& multipliers = weights.at(plan.hash_multipliers);
    const TensorData& vocabulary_sizes = weights.at(plan.head_vocabulary_sizes);
    const TensorData& offsets = weights.at(plan.head_offsets);
    if (multipliers.int64_values().size() != plan.ngram_size
        || vocabulary_sizes.int64_values().size() != head_count
        || offsets.int64_values().size() != head_count)
    {
        return Error{ErrorCode::InvalidModel, "invalid PLE hash metadata"};
    }

    const uint32_t context_length = plan.ngram_size - 1;
    if (cache.ple_token_history.size() > context_length)
        return Error{ErrorCode::InternalError, "invalid PLE token history"};
    std::vector<int32_t> token_history(context_length, static_cast<int32_t>(plan.eos_token_id));
    const size_t existing_offset = context_length - cache.ple_token_history.size();
    std::copy(cache.ple_token_history.begin(), cache.ple_token_history.end(),
              token_history.begin() + static_cast<ptrdiff_t>(existing_offset));
    token_history.insert(token_history.end(), input_ids.begin(), input_ids.end());

    CpuBatch embeddings(input_ids.size(), plan.embedding_dimension);
    for (size_t row_index = 0; row_index < input_ids.size(); ++row_index)
    {
        const size_t current = context_length + row_index;
        std::vector<int64_t> shifted(plan.ngram_size, plan.eos_token_id);
        shifted[0] = token_history[current];
        bool crossed_eos = false;
        for (uint32_t shift = 1; shift < plan.ngram_size; ++shift)
        {
            const size_t source = current - shift;
            crossed_eos = crossed_eos || token_history[source] == static_cast<int32_t>(plan.eos_token_id);
            shifted[shift] = crossed_eos ? static_cast<int64_t>(plan.eos_token_id)
                                         : static_cast<int64_t>(token_history[source]);
        }
        uint32_t head = 0;
        for (uint32_t ngram = 2; ngram <= plan.ngram_size; ++ngram)
        {
            int64_t mixed = wrapped_product(shifted[0], multipliers.int64_values()[0]);
            for (uint32_t position = 1; position < ngram; ++position)
            {
                const int64_t product = wrapped_product(
                    shifted[position], multipliers.int64_values()[position]);
                mixed = std::bit_cast<int64_t>(std::bit_cast<uint64_t>(mixed)
                                               ^ std::bit_cast<uint64_t>(product));
            }
            for (uint32_t local_head = 0; local_head < plan.heads_per_ngram; ++local_head, ++head)
            {
                const int64_t vocabulary_size = vocabulary_sizes.int64_values()[head];
                const int64_t offset = offsets.int64_values()[head];
                if (vocabulary_size <= 0 || offset < 0)
                    return Error{ErrorCode::InvalidModel, "invalid PLE head vocabulary metadata"};
                const uint64_t index = static_cast<uint64_t>(floor_remainder(mixed, vocabulary_size) + offset);
                auto source = embedding_row(weights, plan, index, head_dimension);
                if (!source)
                    return source.error();
                float* destination = embeddings.row(row_index) + static_cast<size_t>(head) * head_dimension;
                for (uint32_t column = 0; column < head_dimension; ++column)
                    destination[column] = bfloat16_to_float(source.value()[column]);
            }
        }
    }
    cache.ple_token_history.assign(token_history.end() - context_length,
                                   token_history.end());

    CpuBatch key = linear_batch(weights.at(plan.key_weight), embeddings, optimization_flags);
    CpuBatch value = linear_batch(weights.at(plan.value_weight), embeddings, optimization_flags);
    CpuBatch key_normed;
    CpuBatch query_normed;
    grouped_rms_norm_into(key, weights.at(plan.key_norm_weight), hidden_size,
                          norm_epsilon, norm_weight_offset, key_normed);
    grouped_rms_norm_into(hidden, weights.at(plan.query_norm_weight), hidden_size,
                          norm_epsilon, norm_weight_offset, query_normed);

    CpuBatch gated(input_ids.size(), expanded_size);
    const float inverse_sqrt_hidden = 1.0f / std::sqrt(static_cast<float>(hidden_size));
    for (size_t row_index = 0; row_index < input_ids.size(); ++row_index)
    {
        for (uint32_t copy = 0; copy < multiplier; ++copy)
        {
            const size_t offset = static_cast<size_t>(copy) * hidden_size;
            float score = 0.0f;
            for (uint32_t column = 0; column < hidden_size; ++column)
                score += key_normed.row(row_index)[offset + column]
                         * query_normed.row(row_index)[offset + column];
            score *= inverse_sqrt_hidden;
            if (score > 0.0f)
                score = std::sqrt(std::max(score, 1e-6f));
            else if (score < 0.0f)
                score = -std::sqrt(std::max(-score, 1e-6f));
            const float gate = ple_sigmoid(score);
            for (uint32_t column = 0; column < hidden_size; ++column)
                gated.row(row_index)[offset + column] = gate * value.row(row_index)[column];
        }
    }

    CpuBatch convolution_input;
    grouped_rms_norm_into(gated, weights.at(plan.convolution_norm_weight),
                          hidden_size, norm_epsilon, norm_weight_offset,
                          convolution_input);
    const uint32_t state_length = (plan.convolution_kernel_size - 1) * plan.ngram_size;
    const size_t state_elements = static_cast<size_t>(state_length) * expanded_size;
    if (cache.ple_convolution_state.empty())
        cache.ple_convolution_state.assign(state_elements, 0.0f);
    else if (cache.ple_convolution_state.size() != state_elements)
        return Error{ErrorCode::InternalError, "invalid PLE convolution state"};
    const std::span<const uint16_t> convolution_weight = weights.at(plan.convolution_weight).bfloat16_values();
    for (size_t row_index = 0; row_index < input_ids.size(); ++row_index)
    {
        const float* current = convolution_input.row(row_index);
        for (uint32_t channel = 0; channel < expanded_size; ++channel)
        {
            float convolution = 0.0f;
            for (uint32_t kernel = 0; kernel < plan.convolution_kernel_size; ++kernel)
            {
                const uint32_t lag = (plan.convolution_kernel_size - 1 - kernel) * plan.ngram_size;
                const float sample = lag == 0
                                         ? current[channel]
                                         : cache.ple_convolution_state[static_cast<size_t>(state_length - lag) * expanded_size + channel];
                convolution += sample * bfloat16_to_float(
                                            convolution_weight[static_cast<size_t>(channel) * plan.convolution_kernel_size + kernel]);
            }
            hidden.row(row_index)[channel] += gated.row(row_index)[channel]
                                              + scaled_silu(convolution, 1.0f, optimization_flags);
        }
        if (state_length != 0)
        {
            std::move(cache.ple_convolution_state.begin() + expanded_size,
                      cache.ple_convolution_state.end(),
                      cache.ple_convolution_state.begin());
            std::copy_n(current, expanded_size,
                        cache.ple_convolution_state.end() - expanded_size);
        }
    }
    return {};
}

} // namespace moe
} // namespace ncnn
