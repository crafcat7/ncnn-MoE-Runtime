#include "cpu_hyper_connection.h"

#include "cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ncnn {
namespace moe {

static float hyper_sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

static Result<void> validate_hyper_tensors(const CpuBatch& input, const TensorData& function, const TensorData& scale, const TensorData& base,
                                           uint32_t multiplier, uint32_t output_count)
{
    if (multiplier == 0 || input.columns() == 0 || input.columns() % multiplier != 0)
        return Error{ErrorCode::InvalidArgument, "hyper-connection input must contain multiplier hidden-state copies"};
    if (function.dtype != DType::Float32 || function.shape != std::vector<uint32_t>{output_count, input.columns()}
        || function.float32_values().size() != function.element_count())
        return Error{ErrorCode::InvalidModel, "invalid hyper-connection function tensor"};
    if (base.dtype != DType::Float32 || base.shape != std::vector<uint32_t>{output_count}
        || base.float32_values().size() != output_count)
        return Error{ErrorCode::InvalidModel, "invalid hyper-connection base tensor"};
    if (scale.dtype != DType::Float32 || scale.float32_values().empty())
        return Error{ErrorCode::InvalidModel, "invalid hyper-connection scale tensor"};
    return {};
}

Result<CpuHyperConnectionMix> hyper_connection_pre(const CpuBatch& input, const TensorData& function, const TensorData& scale, const TensorData& base,
                                                   uint32_t multiplier, uint32_t sinkhorn_iterations, float norm_epsilon, float hyper_epsilon)
{
    const uint32_t mix_count = (2 + multiplier) * multiplier;
    auto valid = validate_hyper_tensors(input, function, scale, base, multiplier, mix_count);
    if (!valid)
        return valid.error();
    if (scale.float32_values().size() != 3 || sinkhorn_iterations == 0 || norm_epsilon <= 0.0f || hyper_epsilon <= 0.0f)
        return Error{ErrorCode::InvalidArgument, "invalid hyper-connection mixing parameters"};

    const uint32_t hidden_size = input.columns() / multiplier;
    CpuBatch normalized(input.rows(), input.columns());
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* source = input.row(row_index);
        float* target = normalized.row(row_index);
        float square_sum = 0.0f;
        for (uint32_t column = 0; column < input.columns(); ++column)
            square_sum += source[column] * source[column];
        const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(input.columns()) + norm_epsilon);
        for (uint32_t column = 0; column < input.columns(); ++column)
            target[column] = source[column] * inverse_rms;
    }
    CpuBatch mixes = linear_batch(function, normalized);
    const std::span<const float> scales = scale.float32_values();
    const std::span<const float> bases = base.float32_values();

    CpuHyperConnectionMix result;
    result.reduced.reset(input.rows(), hidden_size, true);
    result.post.resize(input.rows() * multiplier);
    result.combine.resize(input.rows() * multiplier * multiplier);
    std::vector<float> sums(multiplier);
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* source = input.row(row_index);
        const float* mixed = mixes.row(row_index);
        float* reduced = result.reduced.row(row_index);
        float* post = result.post.data() + row_index * multiplier;
        float* combine = result.combine.data() + row_index * multiplier * multiplier;
        for (uint32_t copy = 0; copy < multiplier; ++copy)
        {
            const float pre = hyper_sigmoid(mixed[copy] * scales[0] + bases[copy]) + hyper_epsilon;
            post[copy] = 2.0f * hyper_sigmoid(mixed[multiplier + copy] * scales[1] + bases[multiplier + copy]);
            for (uint32_t column = 0; column < hidden_size; ++column)
                reduced[column] += pre * source[static_cast<size_t>(copy) * hidden_size + column];
        }

        const uint32_t combine_offset = 2 * multiplier;
        for (uint32_t output = 0; output < multiplier; ++output)
        {
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t residual = 0; residual < multiplier; ++residual)
            {
                const uint32_t index = output * multiplier + residual;
                combine[index] = mixed[combine_offset + index] * scales[2] + bases[combine_offset + index];
                maximum = std::max(maximum, combine[index]);
            }
            float sum = 0.0f;
            for (uint32_t residual = 0; residual < multiplier; ++residual)
            {
                const uint32_t index = output * multiplier + residual;
                combine[index] = std::exp(combine[index] - maximum);
                sum += combine[index];
            }
            for (uint32_t residual = 0; residual < multiplier; ++residual)
                combine[output * multiplier + residual] = combine[output * multiplier + residual] / sum + hyper_epsilon;
        }

        for (uint32_t residual = 0; residual < multiplier; ++residual)
        {
            float sum = 0.0f;
            for (uint32_t output = 0; output < multiplier; ++output)
                sum += combine[output * multiplier + residual];
            sums[residual] = sum;
        }
        for (uint32_t output = 0; output < multiplier; ++output)
        {
            for (uint32_t residual = 0; residual < multiplier; ++residual)
                combine[output * multiplier + residual] /= sums[residual] + hyper_epsilon;
        }
        for (uint32_t iteration = 1; iteration < sinkhorn_iterations; ++iteration)
        {
            for (uint32_t output = 0; output < multiplier; ++output)
            {
                float sum = 0.0f;
                for (uint32_t residual = 0; residual < multiplier; ++residual)
                    sum += combine[output * multiplier + residual];
                sums[output] = sum;
            }
            for (uint32_t output = 0; output < multiplier; ++output)
            {
                for (uint32_t residual = 0; residual < multiplier; ++residual)
                    combine[output * multiplier + residual] /= sums[output] + hyper_epsilon;
            }
            for (uint32_t residual = 0; residual < multiplier; ++residual)
            {
                float sum = 0.0f;
                for (uint32_t output = 0; output < multiplier; ++output)
                    sum += combine[output * multiplier + residual];
                sums[residual] = sum;
            }
            for (uint32_t output = 0; output < multiplier; ++output)
            {
                for (uint32_t residual = 0; residual < multiplier; ++residual)
                    combine[output * multiplier + residual] /= sums[residual] + hyper_epsilon;
            }
        }
    }
    return result;
}

Result<CpuBatch> hyper_connection_post(const CpuBatch& branch, const CpuBatch& residual, const CpuHyperConnectionMix& mix, uint32_t multiplier)
{
    if (multiplier == 0
        || branch.rows() != residual.rows()
        || residual.columns() != branch.columns() * multiplier
        || mix.post.size() != branch.rows() * multiplier
        || mix.combine.size() != branch.rows() * multiplier * multiplier)
        return Error{ErrorCode::InvalidArgument, "hyper-connection post tensors have incompatible shapes"};

    CpuBatch output(branch.rows(), residual.columns());
    for (size_t row_index = 0; row_index < branch.rows(); ++row_index)
    {
        const float* branch_row = branch.row(row_index);
        const float* residual_row = residual.row(row_index);
        const float* post = mix.post.data() + row_index * multiplier;
        const float* combine = mix.combine.data() + row_index * multiplier * multiplier;
        float* destination = output.row(row_index);
        for (uint32_t copy = 0; copy < multiplier; ++copy)
        {
            for (uint32_t column = 0; column < branch.columns(); ++column)
            {
                float value = post[copy] * branch_row[column];
                for (uint32_t residual_copy = 0; residual_copy < multiplier; ++residual_copy)
                    value += combine[residual_copy * multiplier + copy] * residual_row[static_cast<size_t>(residual_copy) * branch.columns() + column];
                destination[static_cast<size_t>(copy) * branch.columns() + column] = value;
            }
        }
    }
    return output;
}

Result<CpuBatch> hyper_connection_head(const CpuBatch& input, const TensorData& function, const TensorData& scale, const TensorData& base, uint32_t multiplier,
                                       float norm_epsilon, float hyper_epsilon)
{
    auto valid = validate_hyper_tensors(input, function, scale, base, multiplier, multiplier);
    if (!valid)
        return valid.error();
    if (scale.float32_values().size() != 1 || norm_epsilon <= 0.0f || hyper_epsilon <= 0.0f)
        return Error{ErrorCode::InvalidArgument, "invalid hyper-connection head parameters"};

    const uint32_t hidden_size = input.columns() / multiplier;
    CpuBatch normalized(input.rows(), input.columns());
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* source = input.row(row_index);
        float* target = normalized.row(row_index);
        float square_sum = 0.0f;
        for (uint32_t column = 0; column < input.columns(); ++column)
            square_sum += source[column] * source[column];
        const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(input.columns()) + norm_epsilon);
        for (uint32_t column = 0; column < input.columns(); ++column)
            target[column] = source[column] * inverse_rms;
    }
    CpuBatch mixes = linear_batch(function, normalized);
    CpuBatch output(input.rows(), hidden_size);
    const float scale_value = scale.float32_values()[0];
    const std::span<const float> bases = base.float32_values();
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* source = input.row(row_index);
        const float* mixed = mixes.row(row_index);
        float* destination = output.row(row_index);
        for (uint32_t copy = 0; copy < multiplier; ++copy)
        {
            const float pre = hyper_sigmoid(mixed[copy] * scale_value + bases[copy]) + hyper_epsilon;
            for (uint32_t column = 0; column < hidden_size; ++column)
                destination[column] += pre * source[static_cast<size_t>(copy) * hidden_size + column];
        }
    }
    return output;
}

} // namespace moe
} // namespace ncnn
