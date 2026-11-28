#include "kernels/float8.h"

#include "kernels/cpu/cpu_ops.h"
#include "ncnn/moe/option.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace ncnn {
namespace moe {

static std::array<float, 256> make_float8_table()
{
    std::array<float, 256> table = {};
    for (uint32_t encoded = 0; encoded < table.size(); ++encoded)
    {
        const uint32_t sign = (encoded & 0x80u) << 24;
        const uint32_t exponent = (encoded >> 3) & 0x0fu;
        const uint32_t mantissa = encoded & 0x07u;
        uint32_t bits = sign;
        if (exponent == 0x0fu && mantissa == 0x07u)
            bits |= 0x7fc00000u;
        else if (exponent != 0)
            bits |= ((exponent + 120u) << 23) | (mantissa << 20);
        else if (mantissa != 0)
        {
            const float value = static_cast<float>(mantissa) / 512.0f;
            bits |= std::bit_cast<uint32_t>(value);
        }
        table[encoded] = std::bit_cast<float>(bits);
    }
    return table;
}

static const std::array<float, 256>& float8_table()
{
    static const std::array<float, 256> table = make_float8_table();
    return table;
}

float float8_e4m3_to_float(uint8_t value) noexcept
{
    return float8_table()[value];
}

uint8_t float_to_float8_e4m3(float value) noexcept
{
    if (std::isnan(value))
        return 0x7fu;
    const uint32_t sign = std::signbit(value) ? 0x80u : 0x00u;
    const float magnitude = std::min(std::fabs(value), 448.0f);
    if (magnitude == 0.0f)
        return static_cast<uint8_t>(sign);
    if (magnitude < 1.0f / 64.0f)
    {
        const uint32_t mantissa = static_cast<uint32_t>(std::min(7.0f, std::floor(magnitude * 512.0f + 0.5f)));
        return static_cast<uint8_t>(sign | mantissa);
    }
    uint32_t bits = std::bit_cast<uint32_t>(magnitude) + 0x00080000u;
    const uint32_t exponent = (bits >> 23) - 120u;
    const uint32_t mantissa = (bits >> 20) & 0x07u;
    if (exponent >= 0x0fu && mantissa >= 0x07u)
        return static_cast<uint8_t>(sign | 0x7eu);
    return static_cast<uint8_t>(sign | ((exponent & 0x0fu) << 3) | mantissa);
}

float float8_e4m3_block_dot(const uint8_t* weights,
                            const float* scales,
                            const float* input,
                            uint32_t count,
                            uint32_t block_size) noexcept
{
    return cpu_ops().float8_e4m3_block_dot(weights, scales, input, count, block_size);
}

void float8_e4m3_block_dot_rows4(const uint8_t* weights,
                                 uint32_t weight_row_stride,
                                 const float* scales,
                                 const float* input,
                                 uint32_t count,
                                 uint32_t block_size,
                                 uint32_t row_count,
                                 float* output) noexcept
{
    cpu_ops().float8_e4m3_block_dot_rows4(weights, weight_row_stride, scales, input, count, block_size, row_count, output);
}

static bool use_bfloat16_float8_linear_dot(uint64_t optimization_flags) noexcept
{
    const CpuBackendCaps& caps = cpu_backend_caps();
    return caps.bf16_fast_dp
           && cpu_ops().bfloat16_float8_e4m3_block_dot != nullptr
           && has_flag(optimization_flags, OptimizationCpuFloat8Bf16Dot);
}

static bool use_float8_batch_tile(uint64_t optimization_flags) noexcept
{
    return has_flag(optimization_flags, OptimizationCpuFloat8BatchTile);
}

float float8_e4m3_quantized_input_dot(const uint8_t* weights,
                                      const float* scales,
                                      const float* input,
                                      uint32_t count,
                                      uint32_t block_size,
                                      uint64_t optimization_flags) noexcept
{
    if (use_bfloat16_float8_linear_dot(optimization_flags))
        return cpu_ops().bfloat16_float8_e4m3_block_dot(weights, scales, input, count, block_size);
    return float8_e4m3_block_dot(weights, scales, input, count, block_size);
}

void float8_e4m3_quantized_input_dot_rows(const uint8_t* weights,
                                          uint32_t weight_row_stride,
                                          const float* scales,
                                          const float* input,
                                          uint32_t count,
                                          uint32_t block_size,
                                          uint32_t row_count,
                                          float* output,
                                          uint64_t optimization_flags) noexcept
{
    if (use_bfloat16_float8_linear_dot(optimization_flags))
    {
        const CpuKernelOps& ops = cpu_ops();
        for (uint32_t first_row = 0; first_row < row_count; first_row += 8)
        {
            ops.bfloat16_float8_e4m3_block_dot_rows(weights + static_cast<size_t>(first_row) * weight_row_stride,
                                                    weight_row_stride, scales, input, count, block_size,
                                                    std::min<uint32_t>(8, row_count - first_row),
                                                    output + first_row);
        }
        return;
    }
    for (uint32_t first_row = 0; first_row < row_count; first_row += 4)
    {
        float8_e4m3_block_dot_rows4(weights + static_cast<size_t>(first_row) * weight_row_stride,
                                    weight_row_stride, scales, input, count, block_size,
                                    std::min<uint32_t>(4, row_count - first_row), output + first_row);
    }
}

static void float8_e4m3_quantized_input_dot_rows_batch_tile(const uint8_t* weights,
                                                            uint32_t weight_row_stride,
                                                            const float* scales,
                                                            const float* input,
                                                            size_t input_stride,
                                                            uint32_t count,
                                                            uint32_t block_size,
                                                            uint32_t row_count,
                                                            size_t output_stride,
                                                            size_t token_count,
                                                            float* output) noexcept
{
    const uint32_t input_blocks = (count + block_size - 1) / block_size;
    for (uint32_t first_row = 0; first_row < row_count; first_row += 4)
    {
        const uint32_t rows = std::min<uint32_t>(4, row_count - first_row);
        for (size_t first_token = 0; first_token < token_count; first_token += 4)
        {
            const size_t tokens = std::min<size_t>(4, token_count - first_token);
            float accum[4][4] = {};
            for (uint32_t block = 0; block < input_blocks; ++block)
            {
                const uint32_t block_begin = block * block_size;
                const uint32_t block_end = std::min(count, block_begin + block_size);
                for (uint32_t column = block_begin; column < block_end; ++column)
                {
                    float inputs[4] = {};
                    for (size_t token = 0; token < tokens; ++token)
                        inputs[token] = input[(first_token + token) * input_stride + column];
                    for (uint32_t row = 0; row < rows; ++row)
                    {
                        const uint32_t matrix_row = first_row + row;
                        const float weight = float8_e4m3_to_float(weights[static_cast<size_t>(matrix_row) * weight_row_stride + column]);
                        const float scale = scales[static_cast<size_t>(matrix_row / block_size) * input_blocks + block];
                        for (size_t token = 0; token < tokens; ++token)
                            accum[token][row] += weight * scale * inputs[token];
                    }
                }
            }
            for (size_t token = 0; token < tokens; ++token)
                for (uint32_t row = 0; row < rows; ++row)
                    output[(first_token + token) * output_stride + first_row + row] = accum[token][row];
        }
    }
}

void float8_e4m3_quantized_input_dot_rows_batch(const uint8_t* weights,
                                                uint32_t weight_row_stride,
                                                const float* scales,
                                                const float* input,
                                                size_t input_stride,
                                                uint32_t count,
                                                uint32_t block_size,
                                                uint32_t row_count,
                                                size_t output_stride,
                                                size_t token_count,
                                                float* output,
                                                uint64_t optimization_flags) noexcept
{
    if (row_count == 0 || token_count == 0)
        return;
    for (size_t token = 0; token < token_count; ++token)
    {
        for (uint32_t row = 0; row < row_count; ++row)
            output[token * output_stride + row] = 0.0f;
    }

    if (!cpu_backend_caps().float8_native
        && use_float8_batch_tile(optimization_flags)
        && token_count >= 4
        && row_count >= 4
        && count >= block_size)
    {
        float8_e4m3_quantized_input_dot_rows_batch_tile(weights,
                                                        weight_row_stride,
                                                        scales,
                                                        input,
                                                        input_stride,
                                                        count,
                                                        block_size,
                                                        row_count,
                                                        output_stride,
                                                        token_count,
                                                        output);
        return;
    }

    if (!use_float8_batch_tile(optimization_flags))
    {
        for (uint32_t first_row = 0; first_row < row_count; first_row += 4)
        {
            const uint32_t rows = std::min<uint32_t>(4, row_count - first_row);
            for (size_t token = 0; token < token_count; ++token)
            {
                float8_e4m3_quantized_input_dot_rows(weights + static_cast<size_t>(first_row) * weight_row_stride,
                                                     weight_row_stride, scales, input + token * input_stride,
                                                     count, block_size, rows,
                                                     output + token * output_stride + first_row, optimization_flags);
            }
        }
        return;
    }

    const CpuKernelOps& ops = cpu_ops();
    const bool use_bf16_batch = use_bfloat16_float8_linear_dot(optimization_flags)
                                && ops.bfloat16_float8_e4m3_block_dot_rows_batch != nullptr;
    for (uint32_t first_row = 0; first_row < row_count; first_row += 4)
    {
        const uint32_t rows = std::min<uint32_t>(4, row_count - first_row);
        if (use_bf16_batch)
        {
            for (size_t first_token = 0; first_token < token_count; first_token += 4)
            {
                const size_t tokens = std::min<size_t>(4, token_count - first_token);
                ops.bfloat16_float8_e4m3_block_dot_rows_batch(weights + static_cast<size_t>(first_row) * weight_row_stride,
                                                              weight_row_stride, scales,
                                                              input + first_token * input_stride, input_stride, count,
                                                              block_size, rows, output_stride, tokens,
                                                              output + first_token * output_stride + first_row);
            }
            continue;
        }
        for (size_t token = 0; token < token_count; ++token)
        {
            float8_e4m3_quantized_input_dot_rows(weights + static_cast<size_t>(first_row) * weight_row_stride,
                                                 weight_row_stride, scales, input + token * input_stride,
                                                 count, block_size, rows,
                                                 output + token * output_stride + first_row, optimization_flags);
        }
    }
}

const char* float8_kernel_name() noexcept
{
    return cpu_backend_caps().float8_name;
}

const char* float8_linear_kernel_name(uint64_t optimization_flags) noexcept
{
    return use_bfloat16_float8_linear_dot(optimization_flags)
               ? "x86-avx512-bf16-dpbf16"
               : float8_kernel_name();
}

uint32_t float8_linear_row_group_size(uint64_t optimization_flags) noexcept
{
    return use_bfloat16_float8_linear_dot(optimization_flags) ? 8 : 4;
}

void quantize_float8_e4m3(const float* source,
                          float* values,
                          uint32_t count,
                          uint32_t block_size,
                          bool power_of_two_scale,
                          uint64_t optimization_flags) noexcept
{
    if (has_flag(optimization_flags, OptimizationCpuFloat8SimdQuantize))
    {
        cpu_ops().quantize_float8_e4m3(source, values, count, block_size, power_of_two_scale);
        return;
    }
    for (uint32_t block_begin = 0; block_begin < count; block_begin += block_size)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        float maximum = 1e-4f;
        for (uint32_t index = block_begin; index < block_end; ++index)
            maximum = std::max(maximum, std::fabs(source[index]));
        float scale = maximum / 448.0f;
        if (power_of_two_scale)
            scale = std::exp2(std::ceil(std::log2(scale)));
        for (uint32_t index = block_begin; index < block_end; ++index)
        {
            const float normalized = std::clamp(source[index] / scale, -448.0f, 448.0f);
            values[index] = float8_e4m3_to_float(float_to_float8_e4m3(normalized)) * scale;
        }
    }
}

void quantize_float8_e4m3_inplace(float* values,
                                  uint32_t count,
                                  uint32_t block_size,
                                  bool power_of_two_scale,
                                  uint64_t optimization_flags) noexcept
{
    quantize_float8_e4m3(values, values, count, block_size, power_of_two_scale, optimization_flags);
}

void quantize_float4_e2m1_inplace(float* values, uint32_t count, uint32_t block_size) noexcept
{
    static constexpr std::array<float, 8> positive_e2m1 = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (uint32_t block_begin = 0; block_begin < count; block_begin += block_size)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        float maximum = 1e-4f;
        for (uint32_t index = block_begin; index < block_end; ++index)
            maximum = std::max(maximum, std::fabs(values[index]));
        const float scale = maximum / 6.0f;
        for (uint32_t index = block_begin; index < block_end; ++index)
        {
            const float value = values[index];
            const float target = std::fabs(value / scale);
            float nearest = 0.0f;
            float best_error = std::numeric_limits<float>::infinity();
            for (float candidate : positive_e2m1)
            {
                const float error = std::fabs(target - candidate);
                if (error < best_error)
                {
                    best_error = error;
                    nearest = candidate;
                }
            }
            values[index] = (std::signbit(value) ? -nearest : nearest) * scale;
        }
    }
}

} // namespace moe
} // namespace ncnn
