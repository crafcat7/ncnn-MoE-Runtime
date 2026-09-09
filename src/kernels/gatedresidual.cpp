#include "gatedresidual.h"

#include "fastmath.h"
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <span>

namespace ncnn {
namespace moe {

static float gated_sigmoid(float value) noexcept
{
    return 1.0f / (1.0f + float_approximate_exp(-value));
}

static bool valid_bfloat16_matrix(const TensorData& tensor, uint32_t rows, uint32_t columns) noexcept
{
    return tensor.dtype == DType::BFloat16
           && tensor.shape.size() == 2
           && tensor.shape[0] == rows
           && tensor.shape[1] == columns
           && tensor.bfloat16_values().size() == tensor.element_count();
}

static bool valid_bfloat16_vector(const TensorData& tensor, uint32_t size) noexcept
{
    return tensor.dtype == DType::BFloat16
           && tensor.shape.size() == 1
           && tensor.shape[0] == size
           && tensor.bfloat16_values().size() == tensor.element_count();
}

static Result<void> gated_residual_pre_impl(
    const ActivationBuffer& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    const TensorData* inject_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    ActivationBuffer& reduced_output,
    std::vector<float>* post_output,
    HyperConnectionScratch& scratch,
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

    ActivationBuffer& normalized = scratch.normalized;
    normalized.reset(input.rows(), expanded_size, false);
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

    linear_batch_into(mix_down_weight, normalized, scratch.projection, optimization_flags);
    const float inverse_multiplier = 1.0f / static_cast<float>(multiplier);
    for (size_t row_index = 0; row_index < scratch.projection.rows(); ++row_index)
    {
        float* row = scratch.projection.row(row_index);
        for (uint32_t column = 0; column < scratch.projection.columns(); ++column)
        {
            row[column] *= inverse_multiplier;
            row[column] = row[column] * gated_sigmoid(row[column]);
        }
    }
    linear_batch_into(mix_up_weight, scratch.projection, scratch.auxiliary, optimization_flags);

    reduced_output.reset(input.rows(), hidden_size, true);
    if (inject_weight)
    {
        if (!post_output)
            return Error{ErrorCode::InternalError, "gated-residual post output is unavailable"};
        linear_batch_into(*inject_weight, normalized, scratch.projection, optimization_flags);
        post_output->resize(input.rows() * multiplier);
        for (size_t row_index = 0; row_index < input.rows(); ++row_index)
        {
            const float* row = scratch.projection.row(row_index);
            float* output = post_output->data() + row_index * multiplier;
            for (uint32_t copy = 0; copy < multiplier; ++copy)
                output[copy] = 2.0f * gated_sigmoid(row[copy] * inverse_multiplier);
        }
    }
    else if (post_output)
    {
        post_output->resize(0);
    }

    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* normalized_row = normalized.row(row_index);
        const float* mix_row = scratch.auxiliary.row(row_index);
        float* reduced = reduced_output.row(row_index);
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
    return {};
}

Result<void> gated_residual_pre(
    const ActivationBuffer& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    const TensorData& inject_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    HyperConnectionMix& result,
    HyperConnectionScratch& scratch,
    uint64_t optimization_flags)
{
    result.combine.resize(0);
    return gated_residual_pre_impl(
        input,
        norm_weight,
        mix_down_weight,
        mix_up_weight,
        &inject_weight,
        multiplier,
        hidden_size,
        norm_epsilon,
        norm_weight_offset,
        result.reduced,
        &result.post,
        scratch,
        optimization_flags);
}

Result<void> gated_residual_post(
    const ActivationBuffer& branch,
    const ActivationBuffer& residual,
    const HyperConnectionMix& mix,
    uint32_t multiplier,
    ActivationBuffer& output)
{
    if (multiplier == 0 || branch.rows() != residual.rows()
        || residual.columns() != branch.columns() * multiplier
        || mix.post.size() != branch.rows() * multiplier)
    {
        return Error{ErrorCode::InvalidArgument, "gated-residual post tensors have incompatible shapes"};
    }
    if (&output == &branch)
        return Error{ErrorCode::InvalidArgument, "gated-residual post output must not alias branch"};
    output.reset(residual.rows(), residual.columns(), false);
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
    return {};
}

Result<void> gated_residual_head(
    const ActivationBuffer& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    ActivationBuffer& output,
    HyperConnectionScratch& scratch,
    uint64_t optimization_flags)
{
    if (&output == &input)
        return Error{ErrorCode::InvalidArgument, "gated-residual head output must not alias input"};
    auto mixed = gated_residual_pre_impl(
        input,
        norm_weight,
        mix_down_weight,
        mix_up_weight,
        nullptr,
        multiplier,
        hidden_size,
        norm_epsilon,
        norm_weight_offset,
        output,
        nullptr,
        scratch,
        optimization_flags);
    if (!mixed)
        return mixed.error();
    return {};
}

} // namespace moe
} // namespace ncnn
