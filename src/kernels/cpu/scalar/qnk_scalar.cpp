#include "kernels/cpu/scalar/scalar_internal.h"

#include "kernels/qnk.h"
#include "kernels/vector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace ncnn {
namespace moe {

static uint16_t load_u16_scalar(const uint8_t* source) noexcept
{
    uint16_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

static float load_f16_scalar(const uint8_t* source) noexcept
{
    return f16_to_f32(load_u16_scalar(source));
}

static uint32_t load_u32_scalar(const uint8_t* source) noexcept
{
    uint32_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

static void scalar_qnk_dequantize_block_32(DType dtype, const uint8_t* block, float* output) noexcept
{
    if (dtype == DType::Q6K)
    {
        std::fill_n(output, qnk_block_elements, 0.0f);
        return;
    }
    static constexpr int8_t iq4_nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const float scale = load_f16_scalar(block);
    if (dtype == DType::Q4_0 || dtype == DType::Q4_1)
    {
        const bool affine = dtype == DType::Q4_1;
        const float minimum = affine ? load_f16_scalar(block + 2) : 0.0f;
        const uint8_t* values = block + (affine ? 4 : 2);
        for (uint32_t index = 0; index < 16; ++index)
        {
            output[index] = scale * static_cast<float>(static_cast<int>(values[index] & 15u) - (affine ? 0 : 8)) + minimum;
            output[index + 16] = scale * static_cast<float>(static_cast<int>(values[index] >> 4) - (affine ? 0 : 8)) + minimum;
        }
        return;
    }
    if (dtype == DType::Q5_0)
    {
        const uint32_t high_bits = load_u32_scalar(block + 2);
        const uint8_t* values = block + 6;
        for (uint32_t index = 0; index < 16; ++index)
        {
            const int low = static_cast<int>((values[index] & 15u) | (((high_bits >> index) & 1u) << 4)) - 16;
            const int high = static_cast<int>((values[index] >> 4) | (((high_bits >> (index + 16)) & 1u) << 4)) - 16;
            output[index] = scale * static_cast<float>(low);
            output[index + 16] = scale * static_cast<float>(high);
        }
        return;
    }
    if (dtype == DType::Q5_1)
    {
        const float minimum = load_f16_scalar(block + 2);
        const uint32_t high = load_u32_scalar(block + 4);
        for (uint32_t index = 0; index < 32; ++index)
        {
            const uint32_t value = ((block[8 + index % 16] >> (index / 16 * 4)) & 15u) | (((high >> index) & 1u) << 4);
            output[index] = scale * static_cast<float>(value) + minimum;
        }
        return;
    }
    for (uint32_t index = 0; index < 32; ++index)
    {
        const int8_t value = dtype == DType::IQ4NL
                                 ? iq4_nl_values[(block[2 + index % 16] >> (index / 16 * 4)) & 15u]
                                 : reinterpret_cast<const int8_t*>(block + 2)[index];
        output[index] = scale * static_cast<float>(value);
    }
}

static float scalar_qnk_dot_block(DType dtype, const uint8_t* block, const float* input) noexcept
{
    if (qnk_block_size(dtype) == 32)
    {
        static constexpr int8_t iq4_nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        const float scale = load_f16_scalar(block);
        float sum = 0.0f;
        if (dtype == DType::Q5_1)
        {
            const float minimum = load_f16_scalar(block + 2);
            const uint32_t high = load_u32_scalar(block + 4);
            for (uint32_t index = 0; index < 32; ++index)
            {
                const uint32_t value = ((block[8 + index % 16] >> (index / 16 * 4)) & 15u) | (((high >> index) & 1u) << 4);
                sum += (scale * static_cast<float>(value) + minimum) * input[index];
            }
            return sum;
        }
        if (dtype == DType::Q4_0 || dtype == DType::Q4_1)
        {
            const bool affine = dtype == DType::Q4_1;
            const float minimum = affine ? load_f16_scalar(block + 2) : 0.0f;
            const uint8_t* values = block + (affine ? 4 : 2);
            for (uint32_t index = 0; index < 16; ++index)
            {
                const float low = scale * static_cast<float>(static_cast<int>(values[index] & 15u) - (affine ? 0 : 8)) + minimum;
                const float high = scale * static_cast<float>(static_cast<int>(values[index] >> 4) - (affine ? 0 : 8)) + minimum;
                sum += low * input[index] + high * input[index + 16];
            }
            return sum;
        }
        if (dtype == DType::Q5_0)
        {
            const uint32_t high_bits = load_u32_scalar(block + 2);
            const uint8_t* values = block + 6;
            for (uint32_t index = 0; index < 16; ++index)
            {
                const int low = static_cast<int>((values[index] & 15u) | (((high_bits >> index) & 1u) << 4)) - 16;
                const int high = static_cast<int>((values[index] >> 4) | (((high_bits >> (index + 16)) & 1u) << 4)) - 16;
                sum += scale * static_cast<float>(low) * input[index] + scale * static_cast<float>(high) * input[index + 16];
            }
            return sum;
        }
        for (uint32_t index = 0; index < 32; ++index)
        {
            const int8_t value = dtype == DType::IQ4NL
                                     ? iq4_nl_values[(block[2 + index % 16] >> (index / 16 * 4)) & 15u]
                                     : reinterpret_cast<const int8_t*>(block + 2)[index];
            sum += (scale * static_cast<float>(value)) * input[index];
        }
        return sum;
    }
    alignas(64) float decoded[qnk_block_elements];
    qnk_dequantize_block(dtype, block, decoded);
    float sum = 0.0f;
    for (uint32_t index = 0; index < qnk_block_elements; ++index)
        sum += decoded[index] * input[index];
    return sum;
}

static float scalar_qnk_dot_row(DType dtype,
                                const uint8_t* weights,
                                size_t block_stride,
                                uint32_t block_count,
                                const float* input) noexcept
{
    float sum = 0.0f;
    const uint32_t block_size = qnk_block_size(dtype);
    for (uint32_t block = 0; block < block_count; ++block)
        sum += qnk_dot_block(dtype, weights + static_cast<size_t>(block) * block_stride, input + static_cast<size_t>(block) * block_size);
    return sum;
}

static void scalar_qnk_gemm(const QnKPack& weights,
                            const float* input,
                            size_t input_stride,
                            size_t token_count,
                            float* output,
                            size_t output_stride) noexcept
{
    const uint32_t block_size = qnk_block_size(weights.dtype);
    for (size_t row = 0; row < weights.rows; ++row)
    {
        for (size_t token = 0; token < token_count; ++token)
        {
            const float* token_input = input + token * input_stride;
            float sum = 0.0f;
            for (uint32_t block = 0; block < weights.block_count; ++block)
                sum += qnk_dot_block(weights.dtype, qnk_packed_block(weights, row, block), token_input + static_cast<size_t>(block) * block_size);
            output[token * output_stride + row] = sum;
        }
    }
}

static void scalar_qnk_q8k_quantize(const float* source, uint8_t* output, uint32_t columns) noexcept
{
    const uint32_t block_count = columns / qnk_block_elements;
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const float* values = source + static_cast<size_t>(block_index) * qnk_block_elements;
        uint8_t* encoded = output + static_cast<size_t>(block_index) * qnk_block_bytes(DType::Q8K);
        float maximum = 0.0f;
        float signed_maximum = 0.0f;
        for (uint32_t index = 0; index < qnk_block_elements; ++index)
        {
            const float value = values[index];
            const float absolute = std::fabs(value);
            if (absolute > maximum)
            {
                maximum = absolute;
                signed_maximum = value;
            }
        }
        if (maximum == 0.0f)
        {
            std::memset(encoded, 0, qnk_block_bytes(DType::Q8K));
            continue;
        }
        const float inverse_scale = -127.0f / signed_maximum;
        const float scale = 1.0f / inverse_scale;
        std::memcpy(encoded, &scale, sizeof(scale));
        int8_t* quantized = reinterpret_cast<int8_t*>(encoded + 4);
        for (uint32_t index = 0; index < qnk_block_elements; ++index)
        {
            const long rounded = std::lrintf(inverse_scale * values[index]);
            quantized[index] = static_cast<int8_t>(std::clamp<long>(rounded, -127, 127));
        }
        for (uint32_t sum_index = 0; sum_index < qnk_block_elements / 16; ++sum_index)
        {
            int16_t sum = 0;
            for (uint32_t lane = 0; lane < 16; ++lane)
                sum = static_cast<int16_t>(sum + quantized[sum_index * 16 + lane]);
            std::memcpy(encoded + 260 + sum_index * sizeof(int16_t), &sum, sizeof(sum));
        }
    }
}

static void scalar_qnk_q8_quant(const float*, int8_t*, float*, int32_t*, uint32_t) noexcept
{
}

static float scalar_qnk_q8_row(DType, const uint8_t*, size_t, uint32_t, const int8_t*, const float*, const int32_t*) noexcept
{
    return 0.0f;
}

void register_scalar_qnk_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept
{
    ops.qnk_dequantize_block_32 = scalar_qnk_dequantize_block_32;
    ops.qnk_dot_block = scalar_qnk_dot_block;
    ops.qnk_dot_row = scalar_qnk_dot_row;
    ops.qnk_gemm = scalar_qnk_gemm;
    ops.qnk_q8k_quantize = scalar_qnk_q8k_quantize;
    ops.qnk_q8_quant = scalar_qnk_q8_quant;
    ops.qnk_q8_row = scalar_qnk_q8_row;
    caps.qnk_q8_supported = false;
    caps.qnk_i8mm_supported = false;
}

} // namespace moe
} // namespace ncnn
