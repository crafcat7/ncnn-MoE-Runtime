#include "hyperconnection.h"

#include "fastmath.h"
#include "ops.h"
#include "vector.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ncnn {
namespace moe {

void hyper_connection_expand(CpuBatch& hidden, uint32_t multiplier, CpuBatch& scratch)
{
    if (multiplier <= 1)
        return;
    scratch.reset(hidden.rows(), hidden.columns() * multiplier, false);
    for (size_t row_index = 0; row_index < hidden.rows(); ++row_index)
    {
        for (uint32_t copy = 0; copy < multiplier; ++copy)
            std::copy_n(hidden.row(row_index), hidden.columns(), scratch.row(row_index) + static_cast<size_t>(copy) * hidden.columns());
    }
    hidden.swap(scratch);
}

static float hyper_sigmoid(float value)
{
    return 1.0f / (1.0f + float_approximate_exp(-value));
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
                                                   uint32_t multiplier, uint32_t sinkhorn_iterations, float norm_epsilon, float hyper_epsilon,
                                                   uint64_t optimization_flags)
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
        std::copy_n(source, input.columns(), target);
        float_rms_scale_inplace(target, norm_epsilon, input.columns());
    }
    CpuBatch mixes = linear_batch(function, normalized, optimization_flags);
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
        if (multiplier == 4)
        {
            const float pre0 = hyper_sigmoid(mixed[0] * scales[0] + bases[0]) + hyper_epsilon;
            const float pre1 = hyper_sigmoid(mixed[1] * scales[0] + bases[1]) + hyper_epsilon;
            const float pre2 = hyper_sigmoid(mixed[2] * scales[0] + bases[2]) + hyper_epsilon;
            const float pre3 = hyper_sigmoid(mixed[3] * scales[0] + bases[3]) + hyper_epsilon;
            post[0] = 2.0f * hyper_sigmoid(mixed[4] * scales[1] + bases[4]);
            post[1] = 2.0f * hyper_sigmoid(mixed[5] * scales[1] + bases[5]);
            post[2] = 2.0f * hyper_sigmoid(mixed[6] * scales[1] + bases[6]);
            post[3] = 2.0f * hyper_sigmoid(mixed[7] * scales[1] + bases[7]);
            float_hc_pre_4(reduced, source, pre0, pre1, pre2, pre3, hidden_size);
        }
        else
        {
            for (uint32_t copy = 0; copy < multiplier; ++copy)
            {
                const float pre = hyper_sigmoid(mixed[copy] * scales[0] + bases[copy]) + hyper_epsilon;
                post[copy] = 2.0f * hyper_sigmoid(mixed[multiplier + copy] * scales[1] + bases[multiplier + copy]);
                float_scaled_add(
                    reduced,
                    source + static_cast<size_t>(copy) * hidden_size,
                    pre,
                    hidden_size);
            }
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
                combine[index] = float_approximate_exp(combine[index] - maximum);
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
        if (multiplier == 4)
        {
            float_hc_post_4(destination, branch_row, residual_row, post, combine, branch.columns());
        }
        else
        {
            for (uint32_t copy = 0; copy < multiplier; ++copy)
            {
                float* output_row = destination + static_cast<size_t>(copy) * branch.columns();
                std::copy_n(branch_row, branch.columns(), output_row);
                float_scale_inplace(output_row, post[copy], branch.columns());
                for (uint32_t residual_copy = 0; residual_copy < multiplier; ++residual_copy)
                    float_scaled_add(
                        output_row,
                        residual_row + static_cast<size_t>(residual_copy) * branch.columns(),
                        combine[residual_copy * multiplier + copy],
                        branch.columns());
            }
        }
    }
    return output;
}

Result<CpuBatch> hyper_connection_head(const CpuBatch& input, const TensorData& function, const TensorData& scale, const TensorData& base, uint32_t multiplier,
                                       float norm_epsilon, float hyper_epsilon, uint64_t optimization_flags)
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
        std::copy_n(source, input.columns(), target);
        float_rms_scale_inplace(target, norm_epsilon, input.columns());
    }
    CpuBatch mixes = linear_batch(function, normalized, optimization_flags);
    CpuBatch output(input.rows(), hidden_size);
    const float scale_value = scale.float32_values()[0];
    const std::span<const float> bases = base.float32_values();
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* source = input.row(row_index);
        const float* mixed = mixes.row(row_index);
        float* destination = output.row(row_index);
        if (multiplier == 4)
        {
            const float pre0 = hyper_sigmoid(mixed[0] * scale_value + bases[0]) + hyper_epsilon;
            const float pre1 = hyper_sigmoid(mixed[1] * scale_value + bases[1]) + hyper_epsilon;
            const float pre2 = hyper_sigmoid(mixed[2] * scale_value + bases[2]) + hyper_epsilon;
            const float pre3 = hyper_sigmoid(mixed[3] * scale_value + bases[3]) + hyper_epsilon;
            float_hc_pre_4(destination, source, pre0, pre1, pre2, pre3, hidden_size);
        }
        else
        {
            for (uint32_t copy = 0; copy < multiplier; ++copy)
            {
                const float pre = hyper_sigmoid(mixed[copy] * scale_value + bases[copy]) + hyper_epsilon;
                float_scaled_add(
                    destination,
                    source + static_cast<size_t>(copy) * hidden_size,
                    pre,
                    hidden_size);
            }
        }
    }
    return output;
}

} // namespace moe
} // namespace ncnn
