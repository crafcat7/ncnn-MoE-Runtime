#include "cpu_ops.h"

#include "cpu_mxfp4.h"
#include "ncnn_linear.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <numeric>

#if defined(_OPENMP)
#include <omp.h>
#endif

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

static bool allow_openmp_parallel_region() noexcept
{
#if defined(_OPENMP)
    return omp_in_parallel() == 0;
#else
    return false;
#endif
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
#pragma omp parallel for if(allow_openmp_parallel_region())
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
#pragma omp parallel for if(allow_openmp_parallel_region())
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
#pragma omp parallel for if(allow_openmp_parallel_region())
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
        const uint32_t blocks_per_row = input_columns / 32;
        const int64_t row_pair_count
            = static_cast<int64_t>((output_columns + 1) / 2);
#pragma omp parallel for if(allow_openmp_parallel_region())
        for (int64_t row_pair = 0; row_pair < row_pair_count; ++row_pair) {
            const uint32_t first_row = static_cast<uint32_t>(row_pair) * 2;
            const uint32_t second_row = first_row + 1;
            const uint8_t* first_blocks
                = matrix.mxfp4_blocks.data()
                  + static_cast<size_t>(first_row) * input_columns / 2;
            const uint8_t* first_scales
                = matrix.mxfp4_scales.data()
                  + static_cast<size_t>(first_row) * blocks_per_row;
            if (second_row < output_columns) {
                const uint8_t* second_blocks
                    = matrix.mxfp4_blocks.data()
                      + static_cast<size_t>(second_row) * input_columns / 2;
                const uint8_t* second_scales
                    = matrix.mxfp4_scales.data()
                      + static_cast<size_t>(second_row) * blocks_per_row;
                mxfp4_matmul_rows2(
                    first_blocks,
                    first_scales,
                    second_blocks,
                    second_scales,
                    blocks_per_row,
                    input.row(0),
                    input.columns(),
                    input.rows(),
                    output.row(0) + first_row,
                    output.columns(),
                    output.row(0) + second_row,
                    output.columns());
            }
            else if (input.rows() == 1) {
                output.row(0)[first_row] = mxfp4_dot(
                    first_blocks,
                    first_scales,
                    blocks_per_row,
                    input.row(0));
            }
            else {
                mxfp4_gemm_row(
                    first_blocks,
                    first_scales,
                    blocks_per_row,
                    input.row(0),
                    input.columns(),
                    input.rows(),
                    output.row(0) + first_row,
                    output.columns());
            }
        }
    }
    else {
        assert(false && "unsupported matrix dtype");
    }
    return output;
}

CpuBatch fused_mxfp4_gate_up_batch(
    const TensorData& matrix,
    const TensorData* bias,
    const CpuBatch& input,
    float activation_limit)
{
    assert(matrix.dtype == DType::MxFp4 && matrix.shape.size() == 2);
    assert(matrix.shape[0] % 2 == 0 && matrix.shape[1] == input.columns());
    const uint32_t intermediate_size = matrix.shape[0] / 2;
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t blocks_per_row = input_columns / 32;
    if (bias)
        assert(bias->shape == std::vector<uint32_t>{matrix.shape[0]});

    CpuBatch output(input.rows(), intermediate_size);
    const int64_t parallel_columns = static_cast<int64_t>(intermediate_size);
#pragma omp parallel for if(allow_openmp_parallel_region())
    for (int64_t column = 0; column < parallel_columns; ++column) {
        const size_t gate_row = static_cast<size_t>(column) * 2;
        const size_t up_row = gate_row + 1;
        const uint8_t* gate_blocks
            = matrix.mxfp4_blocks.data() + gate_row * input_columns / 2;
        const uint8_t* up_blocks
            = matrix.mxfp4_blocks.data() + up_row * input_columns / 2;
        const uint8_t* gate_scales
            = matrix.mxfp4_scales.data() + gate_row * blocks_per_row;
        const uint8_t* up_scales
            = matrix.mxfp4_scales.data() + up_row * blocks_per_row;
        const float gate_bias = bias ? tensor_value(*bias, gate_row) : 0.0f;
        const float up_bias = bias ? tensor_value(*bias, up_row) : 0.0f;
        if (input.rows() == 1) {
            float gate = 0.0f;
            float linear = 0.0f;
            mxfp4_matmul_rows2(
                gate_blocks,
                gate_scales,
                up_blocks,
                up_scales,
                blocks_per_row,
                input.row(0),
                input.columns(),
                1,
                &gate,
                1,
                &linear,
                1);
            gate += gate_bias;
            linear += up_bias;
            if (activation_limit > 0.0f) {
                gate = std::min(gate, activation_limit);
                linear = std::clamp(linear, -activation_limit, activation_limit);
            }
            const float silu = gate / (1.0f + std::exp(-1.702f * gate));
            output.row(0)[column] = silu * (linear + 1.0f);
        }
        else {
            static thread_local std::vector<float> linear;
            linear.resize(input.rows());
            mxfp4_matmul_rows2(
                gate_blocks,
                gate_scales,
                up_blocks,
                up_scales,
                blocks_per_row,
                input.row(0),
                input.columns(),
                input.rows(),
                output.row(0) + column,
                output.columns(),
                linear.data(),
                1);
            for (size_t token_index = 0; token_index < input.rows(); ++token_index) {
                float gate = output.row(token_index)[column] + gate_bias;
                float up = linear[token_index] + up_bias;
                if (activation_limit > 0.0f) {
                    gate = std::min(gate, activation_limit);
                    up = std::clamp(up, -activation_limit, activation_limit);
                }
                const float silu = gate / (1.0f + std::exp(-1.702f * gate));
                output.row(token_index)[column] = silu * (up + 1.0f);
            }
        }
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
