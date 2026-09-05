#include "gatedresidual.h"

#include "fastmath.h"
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

static float gated_sigmoid(float value) noexcept
{
    return 1.0f / (1.0f + float_approximate_exp(-value));
}

static bool valid_bfloat16_matrix(const TensorData& tensor, uint32_t rows, uint32_t columns) noexcept
{
    return tensor.dtype == DType::BFloat16
           && tensor.shape == std::vector<uint32_t>{rows, columns}
           && tensor.bfloat16_values().size() == tensor.element_count();
}

static bool valid_bfloat16_vector(const TensorData& tensor, uint32_t size) noexcept
{
    return tensor.dtype == DType::BFloat16
           && tensor.shape == std::vector<uint32_t>{size}
           && tensor.bfloat16_values().size() == tensor.element_count();
}

static Result<CpuHyperConnectionMix> gated_residual_pre_impl(
    const CpuBatch& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    const TensorData* inject_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    uint64_t optimization_flags)
{
    if (multiplier == 0 || hidden_size == 0
        || input.columns() != static_cast<size_t>(multiplier) * hidden_size
        || norm_epsilon <= 0.0f)
    {
        return Error{ErrorCode::InvalidArgument, "invalid gated-residual input dimensions"};
    }
    const uint32_t expanded_size = multiplier * hidden_size;
    if (!valid_bfloat16_vector(norm_weight, expanded_size))
        return Error{ErrorCode::InvalidModel, "invalid gated-residual normalization tensor"};
    if (mix_down_weight.dtype != DType::BFloat16
        || mix_down_weight.shape.size() != 2
        || mix_down_weight.shape[1] != expanded_size
        || mix_down_weight.shape[0] == 0
        || mix_down_weight.bfloat16_values().size() != mix_down_weight.element_count())
    {
        return Error{ErrorCode::InvalidModel, "invalid gated-residual down projection"};
    }
    const uint32_t low_rank = mix_down_weight.shape[0];
    if (!valid_bfloat16_matrix(mix_up_weight, expanded_size, low_rank))
        return Error{ErrorCode::InvalidModel, "invalid gated-residual up projection"};
    if (inject_weight && !valid_bfloat16_matrix(*inject_weight, multiplier, expanded_size))
        return Error{ErrorCode::InvalidModel, "invalid gated-residual block injection projection"};

    CpuBatch normalized(input.rows(), expanded_size);
    const std::span<const uint16_t> norm = norm_weight.bfloat16_values();
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* source = input.row(row_index);
        float* destination = normalized.row(row_index);
        for (uint32_t copy = 0; copy < multiplier; ++copy)
        {
            const size_t offset = static_cast<size_t>(copy) * hidden_size;
            float mean_square = 0.0f;
            for (uint32_t column = 0; column < hidden_size; ++column)
                mean_square += source[offset + column] * source[offset + column];
            const float scale = 1.0f / std::sqrt(mean_square / static_cast<float>(hidden_size) + norm_epsilon);
            for (uint32_t column = 0; column < hidden_size; ++column)
            {
                destination[offset + column] = source[offset + column] * scale
                                               * (bfloat16_to_float(norm[offset + column]) + norm_weight_offset);
            }
        }
    }

    CpuBatch low_rank_mix = linear_batch(mix_down_weight, normalized, optimization_flags);
    const float inverse_multiplier = 1.0f / static_cast<float>(multiplier);
    for (size_t row_index = 0; row_index < low_rank_mix.rows(); ++row_index)
    {
        float* row = low_rank_mix.row(row_index);
        for (uint32_t column = 0; column < low_rank_mix.columns(); ++column)
        {
            row[column] *= inverse_multiplier;
            row[column] = row[column] * gated_sigmoid(row[column]);
        }
    }
    CpuBatch input_mix = linear_batch(mix_up_weight, low_rank_mix, optimization_flags);

    CpuHyperConnectionMix result;
    result.reduced.reset(input.rows(), hidden_size, true);
    if (inject_weight)
    {
        CpuBatch injection = linear_batch(*inject_weight, normalized, optimization_flags);
        result.post.resize(input.rows() * multiplier);
        for (size_t row_index = 0; row_index < input.rows(); ++row_index)
        {
            const float* row = injection.row(row_index);
            float* output = result.post.data() + row_index * multiplier;
            for (uint32_t copy = 0; copy < multiplier; ++copy)
                output[copy] = 2.0f * gated_sigmoid(row[copy] * inverse_multiplier);
        }
    }

    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* normalized_row = normalized.row(row_index);
        const float* mix_row = input_mix.row(row_index);
        float* reduced = result.reduced.row(row_index);
        for (uint32_t copy = 0; copy < multiplier; ++copy)
        {
            const size_t offset = static_cast<size_t>(copy) * hidden_size;
            for (uint32_t column = 0; column < hidden_size; ++column)
            {
                reduced[column] += normalized_row[offset + column]
                                   * gated_sigmoid(mix_row[offset + column])
                                   * inverse_multiplier;
            }
        }
    }
    return result;
}

Result<CpuHyperConnectionMix> gated_residual_pre(
    const CpuBatch& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    const TensorData& inject_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    uint64_t optimization_flags)
{
    return gated_residual_pre_impl(input, norm_weight, mix_down_weight,
                                   mix_up_weight, &inject_weight, multiplier,
                                   hidden_size, norm_epsilon,
                                   norm_weight_offset, optimization_flags);
}

Result<CpuBatch> gated_residual_post(
    const CpuBatch& branch,
    const CpuBatch& residual,
    const CpuHyperConnectionMix& mix,
    uint32_t multiplier)
{
    if (multiplier == 0 || branch.rows() != residual.rows()
        || residual.columns() != branch.columns() * multiplier
        || mix.post.size() != branch.rows() * multiplier)
    {
        return Error{ErrorCode::InvalidArgument, "gated-residual post tensors have incompatible shapes"};
    }
    CpuBatch output(residual.rows(), residual.columns());
    for (size_t row_index = 0; row_index < branch.rows(); ++row_index)
    {
        const float* branch_row = branch.row(row_index);
        const float* residual_row = residual.row(row_index);
        const float* injection = mix.post.data() + row_index * multiplier;
        float* destination = output.row(row_index);
        for (uint32_t copy = 0; copy < multiplier; ++copy)
        {
            const size_t offset = static_cast<size_t>(copy) * branch.columns();
            for (size_t column = 0; column < branch.columns(); ++column)
                destination[offset + column] = residual_row[offset + column] + branch_row[column] * injection[copy];
        }
    }
    return output;
}

Result<CpuBatch> gated_residual_head(
    const CpuBatch& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    uint64_t optimization_flags)
{
    auto mixed = gated_residual_pre_impl(input, norm_weight, mix_down_weight,
                                         mix_up_weight, nullptr, multiplier,
                                         hidden_size, norm_epsilon,
                                         norm_weight_offset,
                                         optimization_flags);
    if (!mixed)
        return mixed.error();
    return std::move(mixed).value().reduced;
}

} // namespace moe
} // namespace ncnn
