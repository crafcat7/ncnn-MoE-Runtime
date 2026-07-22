#include "cpu_ops.h"

#include "ncnn_linear.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <numeric>

namespace ncnn {
namespace moe {

float bfloat16_to_float(uint16_t value) noexcept
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t float_to_bfloat16(float value) noexcept
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

static float tensor_value(const TensorData& tensor, size_t index)
{
    if (tensor.dtype == DType::Float32)
        return tensor.float32_data[index];
    if (tensor.dtype == DType::BFloat16)
        return bfloat16_to_float(tensor.bfloat16_data[index]);
    assert(false && "tensor_value only supports float32 and bfloat16");
    return 0.0f;
}

static const std::array<float, 256>& mxfp4_scale_table()
{
    static const std::array<float, 256> values = [] {
        std::array<float, 256> table = {};
        for (uint32_t index = 0; index < table.size(); ++index)
            table[index] = std::ldexp(1.0f, static_cast<int>(index) - 127);
        return table;
    }();
    return values;
}

CpuBatch embedding_batch(const TensorData& embedding, std::span<const int32_t> input_ids)
{
    assert(embedding.shape.size() == 2);
    const uint32_t hidden_size = embedding.shape[1];
    CpuBatch output(input_ids.size(), hidden_size);
    for (size_t token_index = 0; token_index < input_ids.size(); ++token_index) {
        const size_t offset = static_cast<size_t>(input_ids[token_index]) * hidden_size;
        float* destination = output.row(token_index);
        for (uint32_t column = 0; column < hidden_size; ++column)
            destination[column] = tensor_value(embedding, offset + column);
    }
    return output;
}

CpuBatch linear_batch(const TensorData& matrix, const CpuBatch& input)
{
    assert(matrix.shape.size() == 2);
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    assert(input.columns() == input_columns);

    CpuBatch output;
    if (matrix.linear_operator && matrix.linear_operator->forward(input, output))
        return output;
    output = CpuBatch(input.rows(), output_columns);
    const int64_t parallel_output_columns = static_cast<int64_t>(output_columns);
    if (matrix.dtype == DType::Float32) {
#pragma omp parallel for
        for (int64_t output_column = 0; output_column < parallel_output_columns; ++output_column) {
            const float* weights = matrix.float32_data.data() + static_cast<size_t>(output_column) * input_columns;
            for (size_t token_index = 0; token_index < input.rows(); ++token_index) {
                const float* token = input.row(token_index);
                output.row(token_index)[output_column] = std::inner_product(
                    weights, weights + input_columns, token, 0.0f);
            }
        }
    }
    else if (matrix.dtype == DType::BFloat16) {
#pragma omp parallel for
        for (int64_t output_column = 0; output_column < parallel_output_columns; ++output_column) {
            const uint16_t* weights = matrix.bfloat16_data.data() + static_cast<size_t>(output_column) * input_columns;
            for (size_t token_index = 0; token_index < input.rows(); ++token_index) {
                const float* token = input.row(token_index);
                float sum = 0.0f;
                for (uint32_t input_column = 0; input_column < input_columns; ++input_column)
                    sum += bfloat16_to_float(weights[input_column]) * token[input_column];
                output.row(token_index)[output_column] = sum;
            }
        }
    }
    else if (matrix.dtype == DType::Int8) {
#pragma omp parallel for
        for (int64_t output_column = 0; output_column < parallel_output_columns; ++output_column) {
            const int8_t* weights = matrix.int8_data.data() + static_cast<size_t>(output_column) * input_columns;
            const float scale = matrix.quantization_scales[output_column];
            for (size_t token_index = 0; token_index < input.rows(); ++token_index) {
                const float* token = input.row(token_index);
                float sum = 0.0f;
                for (uint32_t input_column = 0; input_column < input_columns; ++input_column)
                    sum += static_cast<float>(weights[input_column]) * token[input_column];
                output.row(token_index)[output_column] = sum * scale;
            }
        }
    }
    else if (matrix.dtype == DType::MxFp4) {
        static constexpr float values[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        const uint32_t blocks_per_row = input_columns / 32;
        const std::array<float, 256>& scale_table = mxfp4_scale_table();
#pragma omp parallel for
        for (int64_t output_column = 0; output_column < parallel_output_columns; ++output_column) {
            const uint8_t* blocks = matrix.mxfp4_blocks.data() + static_cast<size_t>(output_column) * input_columns / 2;
            const uint8_t* scales = matrix.mxfp4_scales.data()
                                    + static_cast<size_t>(output_column) * blocks_per_row;
            for (size_t token_index = 0; token_index < input.rows(); ++token_index) {
                const float* token = input.row(token_index);
                float sum = 0.0f;
                for (uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
                    const float scale = scale_table[scales[block_index]];
                    const uint8_t* packed = blocks + block_index * 16;
                    const float* input_block = token + block_index * 32;
                    float block_sum = 0.0f;
                    for (uint32_t byte_index = 0; byte_index < 16; ++byte_index) {
                        const uint8_t byte = packed[byte_index];
                        block_sum += values[byte & 0x0f] * input_block[byte_index * 2];
                        block_sum += values[byte >> 4] * input_block[byte_index * 2 + 1];
                    }
                    sum += block_sum * scale;
                }
                output.row(token_index)[output_column] = sum;
            }
        }
    }
    else {
        assert(false && "unsupported matrix dtype");
    }
    return output;
}

CpuBatch linear_batch(const TensorData& matrix, const TensorData& bias, const CpuBatch& input)
{
    CpuBatch accelerated;
    if (matrix.linear_operator && matrix.linear_operator->forward(input, accelerated))
        return accelerated;
    CpuBatch output = linear_batch(matrix, input);
    assert(bias.shape.size() == 1 && bias.shape[0] == output.columns());
    for (size_t token_index = 0; token_index < output.rows(); ++token_index) {
        float* row = output.row(token_index);
        for (uint32_t column = 0; column < output.columns(); ++column)
            row[column] += tensor_value(bias, column);
    }
    return output;
}

CpuBatch rms_norm_batch(const CpuBatch& input, const TensorData& weight, float epsilon)
{
    assert(weight.element_count() == input.columns());
    CpuBatch output(input.rows(), input.columns());
    for (size_t token_index = 0; token_index < input.rows(); ++token_index) {
        const float* source = input.row(token_index);
        float* destination = output.row(token_index);
        float sum_of_squares = 0.0f;
        for (uint32_t column = 0; column < input.columns(); ++column)
            sum_of_squares += source[column] * source[column];
        const float inverse_rms = 1.0f / std::sqrt(sum_of_squares / static_cast<float>(input.columns()) + epsilon);
        for (uint32_t column = 0; column < input.columns(); ++column)
            destination[column] = source[column] * inverse_rms * tensor_value(weight, column);
    }
    return output;
}

void add_bias_inplace(CpuBatch& destination, const TensorData& bias)
{
    assert(bias.shape.size() == 1 && bias.shape[0] == destination.columns());
    for (size_t row_index = 0; row_index < destination.rows(); ++row_index) {
        float* output = destination.row(row_index);
        for (uint32_t column = 0; column < destination.columns(); ++column)
            output[column] += tensor_value(bias, column);
    }
}

void add_batch_inplace(CpuBatch& destination, const CpuBatch& source)
{
    assert(destination.rows() == source.rows() && destination.columns() == source.columns());
    for (size_t row_index = 0; row_index < destination.rows(); ++row_index) {
        float* output = destination.row(row_index);
        const float* input = source.row(row_index);
        for (uint32_t column = 0; column < destination.columns(); ++column)
            output[column] += input[column];
    }
}

std::vector<std::vector<float> > batch_to_vectors(const CpuBatch& batch)
{
    std::vector<std::vector<float> > output(batch.rows(), std::vector<float>(batch.columns()));
    for (size_t row_index = 0; row_index < batch.rows(); ++row_index)
        std::copy_n(batch.row(row_index), batch.columns(), output[row_index].begin());
    return output;
}

} // namespace moe
} // namespace ncnn
