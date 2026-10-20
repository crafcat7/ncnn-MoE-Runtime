#include "int4.h"

#include "ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace ncnn {
namespace moe {

Result<void> validate_input_column_indices(const TensorData& tensor)
{
    if (tensor.input_column_indices.empty())
        return {};
    if ((tensor.dtype != DType::Int4 && !is_qnk_dtype(tensor.dtype)) || tensor.shape.size() != 2
        || tensor.input_column_indices.size() != tensor.shape[1])
    {
        return Error{ErrorCode::InvalidModel, "quantized input column map must match the input dimension"};
    }
    std::array<uint8_t, 8192> small_seen;
    std::vector<uint8_t> large_seen;
    uint8_t* seen = small_seen.data();
    if (tensor.shape[1] <= small_seen.size())
    {
        std::fill_n(seen, tensor.shape[1], uint8_t{0});
    }
    else
    {
        large_seen.assign(tensor.shape[1], 0);
        seen = large_seen.data();
    }
    for (uint32_t column : tensor.input_column_indices)
    {
        if (column >= tensor.shape[1] || seen[column] != 0)
            return Error{ErrorCode::InvalidModel, "quantized input column map must be a permutation"};
        seen[column] = 1;
    }
    return {};
}

static Result<void> validate_int4_layout(const TensorData& tensor)
{
    if (tensor.dtype != DType::Int4 || tensor.shape.size() != 2
        || tensor.shape[0] == 0 || tensor.shape[1] == 0)
    {
        return Error{ErrorCode::InvalidModel, "INT4 weight must have a nonempty two-dimensional shape"};
    }
    if (tensor.quantization_group_size == 0)
        return Error{ErrorCode::InvalidModel, "INT4 weight must specify a positive group size"};
    if (!tensor.float32_data.empty() || !tensor.bfloat16_data.empty()
        || !tensor.int8_data.empty() || !tensor.int64_data.empty()
        || !tensor.mxfp4_blocks.empty() || !tensor.mxfp4_scales.empty()
        || tensor.mxfp4_file_storage || (tensor.mapped_data && !tensor.quantized_data.empty())
        || (!tensor.mapped_data && tensor.mapped_size != 0))
    {
        return Error{ErrorCode::InvalidModel, "INT4 weight contains duplicate or unrelated storage"};
    }
    const uint64_t rows = tensor.shape[0];
    const uint64_t columns = tensor.shape[1];
    const uint64_t row_bytes = (columns + 1) / 2;
    const uint64_t groups = (columns + tensor.quantization_group_size - 1) / tensor.quantization_group_size;
    if (tensor.int4_values().size() != rows * row_bytes)
        return Error{ErrorCode::InvalidModel, "INT4 packed weight byte count does not match its shape"};
    if (tensor.quantization_scales.size() != rows * groups
        || tensor.quantization_zero_points.size() != rows * groups)
    {
        return Error{ErrorCode::InvalidModel, "INT4 scale and zero-point counts do not match its groups"};
    }
    if (!tensor.quantization_group_indices.empty() && tensor.quantization_group_indices.size() != columns)
        return Error{ErrorCode::InvalidModel, "INT4 group-index count does not match its input columns"};
    if (!tensor.input_column_indices.empty() && tensor.input_column_indices.size() != columns)
        return Error{ErrorCode::InvalidModel, "INT4 input-column count does not match its input columns"};
    return {};
}

Result<void> validate_int4_tensor(const TensorData& tensor)
{
    auto status = validate_int4_layout(tensor);
    if (!status)
        return status.error();
    for (float scale : tensor.quantization_scales)
    {
        if (!std::isfinite(scale) || scale <= 0.0f)
            return Error{ErrorCode::InvalidModel, "INT4 scales must be finite and positive"};
    }
    for (uint8_t zero : tensor.quantization_zero_points)
    {
        if (zero > 16)
            return Error{ErrorCode::InvalidModel, "INT4 zero point is outside the supported range [0, 16]"};
    }
    if (!tensor.quantization_group_indices.empty())
    {
        const uint64_t groups = (static_cast<uint64_t>(tensor.shape[1]) + tensor.quantization_group_size - 1) / tensor.quantization_group_size;
        for (uint32_t group : tensor.quantization_group_indices)
        {
            if (group >= groups)
                return Error{ErrorCode::InvalidModel, "INT4 group index is outside its scale groups"};
        }
    }
    return validate_input_column_indices(tensor);
}

uint64_t int4_storage_bytes(const TensorData& tensor) noexcept
{
    if (tensor.dtype != DType::Int4)
        return 0;
    return static_cast<uint64_t>(tensor.int4_values().size())
           + static_cast<uint64_t>(tensor.quantization_scales.size()) * sizeof(float)
           + static_cast<uint64_t>(tensor.quantization_zero_points.size())
           + static_cast<uint64_t>(tensor.quantization_group_indices.size()) * sizeof(uint32_t)
           + static_cast<uint64_t>(tensor.input_column_indices.size()) * sizeof(uint32_t);
}

bool int4_dequantize_row(const TensorData& matrix, uint32_t row, float* output)
{
    if (!validate_int4_layout(matrix) || row >= matrix.shape[0] || !output)
        return false;
    const uint32_t columns = matrix.shape[1];
    const size_t row_bytes = columns / 2 + columns % 2;
    const size_t groups = (static_cast<uint64_t>(columns) + matrix.quantization_group_size - 1) / matrix.quantization_group_size;
    const uint8_t* packed = matrix.int4_values().data() + static_cast<size_t>(row) * row_bytes;
    const float* scales = matrix.quantization_scales.data() + static_cast<size_t>(row) * groups;
    const uint8_t* zeros = matrix.quantization_zero_points.data() + static_cast<size_t>(row) * groups;
    for (uint32_t column = 0; column < columns; ++column)
    {
        const uint32_t group = matrix.quantization_group_indices.empty()
                                   ? column / matrix.quantization_group_size
                                   : matrix.quantization_group_indices[column];
        const uint8_t code = (packed[column / 2] >> ((column & 1u) * 4)) & 15u;
        const uint32_t destination = matrix.input_column_indices.empty() ? column : matrix.input_column_indices[column];
        output[destination] = (static_cast<int>(code) - zeros[group]) * scales[group];
    }
    return true;
}

static void int4_dot_group(const uint8_t* weights,
                           uint32_t first_column,
                           uint32_t last_column,
                           float scale,
                           uint8_t zero,
                           const std::array<const float*, 4>& inputs,
                           uint32_t token_count,
                           std::array<float, 4>& sums) noexcept
{
    uint32_t column = first_column;
    if ((column & 1u) != 0 && column < last_column)
    {
        const float weight = (static_cast<int>(weights[column / 2] >> 4) - zero) * scale;
        for (uint32_t token = 0; token < token_count; ++token)
            sums[token] += weight * inputs[token][column];
        ++column;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    std::array<float32x4_t, 4> accumulators;
    for (uint32_t token = 0; token < token_count; ++token)
        accumulators[token] = vdupq_n_f32(0.0f);
    const float32x4_t zeros = vdupq_n_f32(static_cast<float>(zero));
    for (; last_column - column >= 16; column += 16)
    {
        const uint8x8_t packed = vld1_u8(weights + column / 2);
        const uint8x8_t low = vand_u8(packed, vdup_n_u8(15));
        const uint8x8_t high = vshr_n_u8(packed, 4);
        const uint16x8_t first = vmovl_u8(vzip1_u8(low, high));
        const uint16x8_t second = vmovl_u8(vzip2_u8(low, high));
        const float32x4_t w0 = vmulq_n_f32(vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(first))), zeros), scale);
        const float32x4_t w1 = vmulq_n_f32(vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(first))), zeros), scale);
        const float32x4_t w2 = vmulq_n_f32(vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(second))), zeros), scale);
        const float32x4_t w3 = vmulq_n_f32(vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(second))), zeros), scale);
        for (uint32_t token = 0; token < token_count; ++token)
        {
            float32x4_t sum = accumulators[token];
            sum = vfmaq_f32(sum, w0, vld1q_f32(inputs[token] + column));
            sum = vfmaq_f32(sum, w1, vld1q_f32(inputs[token] + column + 4));
            sum = vfmaq_f32(sum, w2, vld1q_f32(inputs[token] + column + 8));
            accumulators[token] = vfmaq_f32(sum, w3, vld1q_f32(inputs[token] + column + 12));
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        sums[token] += vaddvq_f32(accumulators[token]);
#endif
    for (; column < last_column; ++column)
    {
        const uint8_t code = (weights[column / 2] >> ((column & 1u) * 4)) & 15u;
        const float weight = (static_cast<int>(code) - zero) * scale;
        for (uint32_t token = 0; token < token_count; ++token)
            sums[token] += weight * inputs[token][column];
    }
}

bool forward_linear_int4(const TensorData& matrix,
                         const ActivationBuffer& input,
                         ActivationBuffer& output)
{
    if (!validate_int4_layout(matrix) || input.dtype() != DType::Float32
        || output.dtype() != DType::Float32 || &input == &output
        || input.columns() != matrix.shape[1])
    {
        return false;
    }
    const uint32_t rows = matrix.shape[0];
    const uint32_t columns = matrix.shape[1];
    const size_t row_bytes = columns / 2 + columns % 2;
    const size_t groups = (static_cast<uint64_t>(columns) + matrix.quantization_group_size - 1) / matrix.quantization_group_size;
    const std::span<const uint8_t> packed = matrix.int4_values();
    const bool indexed = !matrix.quantization_group_indices.empty();
    output.reset(input.rows(), rows, false);
    const uint64_t operations = static_cast<uint64_t>(rows) * columns * input.rows();
    const int team_size = cpu_linear_team_size(operations, DType::Int4);
    const int64_t row_count = rows;
    const auto forward_row = [&](int64_t output_row) {
        const uint8_t* weights = packed.data() + static_cast<size_t>(output_row) * row_bytes;
        const float* scales = matrix.quantization_scales.data() + static_cast<size_t>(output_row) * groups;
        const uint8_t* zeros = matrix.quantization_zero_points.data() + static_cast<size_t>(output_row) * groups;
        for (size_t first_token = 0; first_token < input.rows(); first_token += 4)
        {
            const uint32_t token_count = static_cast<uint32_t>(std::min<size_t>(4, input.rows() - first_token));
            std::array<const float*, 4> inputs = {};
            std::array<float, 4> sums = {};
            for (uint32_t token = 0; token < token_count; ++token)
                inputs[token] = input.row(first_token + token);
            if (indexed)
            {
                for (uint32_t column = 0; column < columns; ++column)
                {
                    const uint32_t group = matrix.quantization_group_indices[column];
                    const uint8_t code = (weights[column / 2] >> ((column & 1u) * 4)) & 15u;
                    const float weight = (static_cast<int>(code) - zeros[group]) * scales[group];
                    for (uint32_t token = 0; token < token_count; ++token)
                        sums[token] += weight * inputs[token][column];
                }
            }
            else
            {
                for (size_t group = 0; group < groups; ++group)
                {
                    const uint32_t first = static_cast<uint32_t>(group * matrix.quantization_group_size);
                    const uint32_t last = first + std::min(matrix.quantization_group_size, columns - first);
                    int4_dot_group(weights, first, last, scales[group], zeros[group], inputs, token_count, sums);
                }
            }
            for (uint32_t token = 0; token < token_count; ++token)
                output.row(first_token + token)[output_row] = sums[token];
        }
    };
    if (team_size == 1)
    {
        for (int64_t row = 0; row < row_count; ++row)
            forward_row(row);
        return true;
    }
#pragma omp parallel for num_threads(team_size)
    for (int64_t row = 0; row < row_count; ++row)
        forward_row(row);
    return true;
}

} // namespace moe
} // namespace ncnn
