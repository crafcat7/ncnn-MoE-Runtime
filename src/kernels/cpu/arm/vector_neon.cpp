#if defined(__aarch64__) || defined(_M_ARM64)

#include "kernels/cpu/arm/vector_arm_internal.h"

#include "kernels/fastmath.h"
#include "kernels/vector.h"

#include <algorithm>
#include <arm_neon.h>
#include <bit>
#include <cmath>
#include <limits>

namespace ncnn {
namespace moe {

static float32x4_t neon_expf(float32x4_t values) noexcept
{
    // Match float_approximate_exp: negative inputs use reciprocal positive exp,
    // NaNs pass through, and +/-104 keep their explicit scalar boundaries.
    const uint32x4_t input_bits = vreinterpretq_u32_f32(values);
    const uint32x4_t magnitude_bits = vandq_u32(input_bits, vdupq_n_u32(0x7fffffffu));
    const uint32x4_t is_nan = vcgtq_u32(magnitude_bits, vdupq_n_u32(0x7f800000u));
    const uint32x4_t is_negative = vandq_u32(vcgtq_u32(input_bits, vdupq_n_u32(0x7fffffffu)),
                                             vcgtq_u32(magnitude_bits, vdupq_n_u32(0u)));
    const uint32x4_t below_limit = vcltq_u32(magnitude_bits, vdupq_n_u32(0x42d00000u));
    const float32x4_t magnitude = vreinterpretq_f32_u32(vandq_u32(magnitude_bits, below_limit));

    const float32x4_t rounding = vdupq_n_f32(0x1.8p23f);
    float32x4_t exponent = vfmaq_f32(rounding, magnitude, vdupq_n_f32(0x1.715476p+0f));
    exponent = vsubq_f32(exponent, rounding);
    float32x4_t remainder = vfmaq_f32(magnitude, exponent, vdupq_n_f32(-0x1.7f7d1cp-20f));
    remainder = vfmaq_f32(remainder, exponent, vdupq_n_f32(-0x1.62e4p-1f));

    float32x4_t polynomial = vdupq_n_f32(1.37805939e-3f);
    polynomial = vfmaq_f32(vdupq_n_f32(8.37312452e-3f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(4.16695364e-2f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(1.66664720e-1f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(4.99999851e-1f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f), polynomial, remainder);
    polynomial = vfmaq_f32(vdupq_n_f32(1.0f), polynomial, remainder);

    const int32x4_t exponent_integer = vcvtq_s32_f32(exponent);
    const uint32x4_t exponent_bits = vshlq_n_u32(vreinterpretq_u32_s32(exponent_integer), 23);
    const uint32x4_t has_positive_exponent = vcgtq_s32(exponent_integer, vdupq_n_s32(0));
    const uint32x4_t underflow_bias = vbslq_u32(has_positive_exponent,
                                                vdupq_n_u32(0u),
                                                vdupq_n_u32(0x83000000u));
    const float32x4_t scale_high = vreinterpretq_f32_u32(vaddq_u32(vdupq_n_u32(0x7f000000u),
                                                                   underflow_bias));
    const float32x4_t scale_low = vreinterpretq_f32_u32(vsubq_u32(exponent_bits, underflow_bias));
    const float32x4_t positive_result = vmulq_f32(vmulq_f32(polynomial, scale_high), scale_low);
    float32x4_t result = vbslq_f32(is_negative,
                                   vdivq_f32(vdupq_n_f32(1.0f), positive_result),
                                   positive_result);

    const uint32x4_t at_limit = vcgeq_u32(magnitude_bits, vdupq_n_u32(0x42d00000u));
    const uint32x4_t positive_limit = vandq_u32(vandq_u32(at_limit, vmvnq_u32(is_negative)),
                                                vmvnq_u32(is_nan));
    const uint32x4_t negative_limit = vandq_u32(vandq_u32(at_limit, is_negative), vmvnq_u32(is_nan));
    result = vbslq_f32(positive_limit, vdupq_n_f32(INFINITY), result);
    result = vbslq_f32(negative_limit, vdupq_n_f32(0.0f), result);
    return vbslq_f32(is_nan, values, result);
}

static float neon_bfloat16_dot_scalar_tail(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    float result = 0.0f;
    uint32_t index = 0;
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (; index + 4 <= count; index += 4)
    {
        const float32x4_t w = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(weights + index), 16));
        sum = vfmaq_f32(sum, vld1q_f32(input + index), w);
    }
    result = vaddvq_f32(sum);
    for (; index < count; ++index)
        result += input[index] * std::bit_cast<float>(static_cast<uint32_t>(weights[index]) << 16);
    return result;
}

float neon_float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    uint32_t index = 0;
    for (; count - index >= 16; index += 16)
    {
        sum0 = vfmaq_f32(sum0, vld1q_f32(left + index), vld1q_f32(right + index));
        sum1 = vfmaq_f32(sum1, vld1q_f32(left + index + 4), vld1q_f32(right + index + 4));
        sum2 = vfmaq_f32(sum2, vld1q_f32(left + index + 8), vld1q_f32(right + index + 8));
        sum3 = vfmaq_f32(sum3, vld1q_f32(left + index + 12), vld1q_f32(right + index + 12));
    }
    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    for (; count - index >= 4; index += 4)
        sum0 = vfmaq_f32(sum0, vld1q_f32(left + index), vld1q_f32(right + index));
    float result = vaddvq_f32(sum0);
    for (; index < count; ++index)
        result += left[index] * right[index];
    return result;
}

static void neon_float_gemm_tile(const float* weights,
                                 size_t weight_stride,
                                 const float* input,
                                 size_t input_stride,
                                 uint32_t input_columns,
                                 uint32_t output_count,
                                 uint32_t token_count,
                                 float* output,
                                 size_t output_stride) noexcept
{
    // Four outputs keep the sixteen vector accumulators in registers.
    uint32_t first_output = 0;
    if (token_count == 4)
    {
        for (; output_count - first_output >= 4; first_output += 4)
        {
            const float* tile_weights = weights + static_cast<size_t>(first_output) * weight_stride;
            float32x4_t sums[4][4] = {};
            uint32_t column = 0;
            for (; input_columns - column >= 4; column += 4)
            {
                float32x4_t inputs[4];
                float32x4_t weight_rows[4];
                for (uint32_t row = 0; row < 4; ++row)
                {
                    inputs[row] = vld1q_f32(input + static_cast<size_t>(row) * input_stride + column);
                    weight_rows[row] = vld1q_f32(tile_weights + static_cast<size_t>(row) * weight_stride + column);
                }
                for (uint32_t token = 0; token < 4; ++token)
                    for (uint32_t row = 0; row < 4; ++row)
                        sums[token][row] = vfmaq_f32(sums[token][row], inputs[token], weight_rows[row]);
            }
            const uint32_t remain = input_columns & 3u;
            for (uint32_t token = 0; token < 4; ++token)
            {
                for (uint32_t row = 0; row < 4; ++row)
                {
                    float sum = vaddvq_f32(sums[token][row]);
                    for (uint32_t tail = 0; tail < remain; ++tail)
                    {
                        const uint32_t index = column + tail;
                        sum += input[static_cast<size_t>(token) * input_stride + index]
                               * tile_weights[static_cast<size_t>(row) * weight_stride + index];
                    }
                    output[static_cast<size_t>(token) * output_stride + first_output + row] = sum;
                }
            }
        }
    }
    // Partial output/token tiles must not read padding or adjacent rows.
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t row = first_output; row < output_count; ++row)
            output[static_cast<size_t>(token) * output_stride + row] = neon_float_dot(weights + static_cast<size_t>(row) * weight_stride,
                                                                                      input + static_cast<size_t>(token) * input_stride,
                                                                                      input_columns);
}

void neon_float_gemm(const float* weights,
                     size_t weight_stride,
                     const float* input,
                     size_t input_stride,
                     uint32_t input_columns,
                     uint32_t output_count,
                     uint32_t token_count,
                     float* output,
                     size_t output_stride) noexcept
{
    if (token_count == 0 || output_count == 0)
        return;
    for (uint32_t token = 0; token < token_count; token += 4)
    {
        neon_float_gemm_tile(weights, weight_stride,
                             input + static_cast<size_t>(token) * input_stride, input_stride,
                             input_columns, output_count, std::min(4u, token_count - token),
                             output + static_cast<size_t>(token) * output_stride, output_stride);
    }
}

static void neon_bfloat16_gemm_tile(const uint16_t* weights,
                                    size_t weight_stride,
                                    const float* input,
                                    size_t input_stride,
                                    uint32_t input_columns,
                                    uint32_t output_count,
                                    uint32_t token_count,
                                    float* output,
                                    size_t output_stride) noexcept
{
    uint32_t first_output = 0;
    // Share input loads across four output channels during decode.
    if (token_count == 1)
    {
        for (; output_count - first_output >= 4; first_output += 4)
        {
            const uint16_t* w = weights + static_cast<size_t>(first_output) * weight_stride;
            float32x4_t sums[4][4] = {};
            uint32_t k = 0;
            for (; input_columns - k >= 16; k += 16)
            {
                float32x4_t x[4];
                for (uint32_t j = 0; j < 4; ++j)
                    x[j] = vld1q_f32(input + k + j * 4);
                for (uint32_t r = 0; r < 4; ++r)
                {
                    const uint16_t* wr = w + static_cast<size_t>(r) * weight_stride + k;
                    const uint16x8_t w0 = vld1q_u16(wr);
                    const uint16x8_t w1 = vld1q_u16(wr + 8);
                    sums[r][0] = vfmaq_f32(sums[r][0], x[0], vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w0), 16)));
                    sums[r][1] = vfmaq_f32(sums[r][1], x[1], vreinterpretq_f32_u32(vshll_high_n_u16(w0, 16)));
                    sums[r][2] = vfmaq_f32(sums[r][2], x[2], vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w1), 16)));
                    sums[r][3] = vfmaq_f32(sums[r][3], x[3], vreinterpretq_f32_u32(vshll_high_n_u16(w1, 16)));
                }
            }
            for (uint32_t r = 0; r < 4; ++r)
            {
                float32x4_t sum = vaddq_f32(vaddq_f32(sums[r][0], sums[r][1]), vaddq_f32(sums[r][2], sums[r][3]));
                uint32_t j = k;
                for (; input_columns - j >= 4; j += 4)
                    sum = vfmaq_f32(sum, vld1q_f32(input + j), vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + r * weight_stride + j), 16)));
                float value = vaddvq_f32(sum);
                for (; j < input_columns; ++j)
                    value += input[j] * std::bit_cast<float>(static_cast<uint32_t>(w[r * weight_stride + j]) << 16);
                output[first_output + r] = value;
            }
        }
    }
    if (token_count == 4)
    {
        for (; output_count - first_output >= 4; first_output += 4)
        {
            const uint16_t* tile_weights = weights + static_cast<size_t>(first_output) * weight_stride;
            float32x4_t sums[4][4] = {};
            uint32_t column = 0;
            for (; input_columns - column >= 4; column += 4)
            {
                float32x4_t inputs[4];
                float32x4_t weight_rows[4];
                for (uint32_t row = 0; row < 4; ++row)
                {
                    inputs[row] = vld1q_f32(input + static_cast<size_t>(row) * input_stride + column);
                    weight_rows[row] = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(tile_weights + static_cast<size_t>(row) * weight_stride + column), 16));
                }
                for (uint32_t token = 0; token < 4; ++token)
                    for (uint32_t row = 0; row < 4; ++row)
                        sums[token][row] = vfmaq_f32(sums[token][row], inputs[token], weight_rows[row]);
            }
            for (uint32_t token = 0; token < 4; ++token)
            {
                for (uint32_t row = 0; row < 4; ++row)
                {
                    float sum = vaddvq_f32(sums[token][row]);
                    for (uint32_t tail = column; tail < input_columns; ++tail)
                        sum += input[static_cast<size_t>(token) * input_stride + tail]
                               * std::bit_cast<float>(static_cast<uint32_t>(tile_weights[static_cast<size_t>(row) * weight_stride + tail]) << 16);
                    output[static_cast<size_t>(token) * output_stride + first_output + row] = sum;
                }
            }
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t row = first_output; row < output_count; ++row)
            output[static_cast<size_t>(token) * output_stride + row] = neon_bfloat16_dot_scalar_tail(weights + static_cast<size_t>(row) * weight_stride,
                                                                                                     input + static_cast<size_t>(token) * input_stride,
                                                                                                     input_columns);
}

void neon_bfloat16_gemm(const uint16_t* weights,
                        size_t weight_stride,
                        const float* input,
                        size_t input_stride,
                        uint32_t input_columns,
                        uint32_t output_count,
                        uint32_t token_count,
                        float* output,
                        size_t output_stride) noexcept
{
    if (token_count == 0 || output_count == 0)
        return;
    for (uint32_t token = 0; token < token_count; token += 4)
    {
        neon_bfloat16_gemm_tile(weights, weight_stride,
                                input + static_cast<size_t>(token) * input_stride, input_stride,
                                input_columns, output_count, std::min(4u, token_count - token),
                                output + static_cast<size_t>(token) * output_stride, output_stride);
    }
}

void neon_float_rms_norm(float* output,
                         const float* input,
                         const float* weight,
                         float epsilon,
                         float weight_offset,
                         uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = neon_float_dot(input, input, count);
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
        output[index] = input[index] * inverse_rms * (weight[index] + weight_offset);
}

void neon_bfloat16_rms_norm(float* output,
                            const float* input,
                            const uint16_t* weight,
                            float epsilon,
                            float weight_offset,
                            uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = neon_float_dot(input, input, count);
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
    {
        const float weight_value = std::bit_cast<float>(static_cast<uint32_t>(weight[index]) << 16);
        output[index] = input[index] * inverse_rms * (weight_value + weight_offset);
    }
}

void neon_float_rope_inplace(float* values,
                             const float* cosine,
                             const float* sine,
                             uint32_t dimension) noexcept
{
    const uint32_t half_dimension = dimension / 2;
    uint32_t index = 0;
    for (; index + 4 <= half_dimension; index += 4)
    {
        const float32x4_t first = vld1q_f32(values + index);
        const float32x4_t second = vld1q_f32(values + half_dimension + index);
        const float32x4_t cosine_values = vld1q_f32(cosine + index);
        const float32x4_t sine_values = vld1q_f32(sine + index);
        vst1q_f32(values + index,
                  vsubq_f32(vmulq_f32(first, cosine_values), vmulq_f32(second, sine_values)));
        vst1q_f32(values + half_dimension + index,
                  vaddq_f32(vmulq_f32(second, cosine_values), vmulq_f32(first, sine_values)));
    }
    for (; index < half_dimension; ++index)
    {
        const float first = values[index];
        const float second = values[half_dimension + index];
        values[index] = first * cosine[index] - second * sine[index];
        values[half_dimension + index] = second * cosine[index] + first * sine[index];
    }
}

float neon_f32_scale_max(float* values, float scale, uint32_t count) noexcept
{
    uint32_t index = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    const float32x4_t scale_values = vdupq_n_f32(scale);
    float32x4_t maximum_values = vdupq_n_f32(maximum);
    for (; index + 4 <= count; index += 4)
    {
        const float32x4_t scaled = vmulq_f32(vld1q_f32(values + index), scale_values);
        vst1q_f32(values + index, scaled);
        maximum_values = vmaxq_f32(maximum_values, scaled);
    }
    maximum = vmaxvq_f32(maximum_values);
    for (; index < count; ++index)
    {
        const float scaled = values[index] * scale;
        values[index] = scaled;
        maximum = std::max(maximum, scaled);
    }
    return maximum;
}

float neon_f32_softmax(float* values, uint32_t count, float maximum) noexcept
{
    uint32_t index = 0;
    const float32x4_t maximum_values = vdupq_n_f32(maximum);
    float32x4_t sum_values = vdupq_n_f32(0.0f);
    for (; index + 4 <= count; index += 4)
    {
        const float32x4_t differences = vsubq_f32(vld1q_f32(values + index), maximum_values);
        const float32x4_t exponentials = neon_expf(differences);
        vst1q_f32(values + index, exponentials);
        sum_values = vaddq_f32(sum_values, exponentials);
    }
    float sum = vaddvq_f32(sum_values);
    for (; index < count; ++index)
    {
        const float exponential = float_approximate_exp(values[index] - maximum);
        values[index] = exponential;
        sum += exponential;
    }
    return sum;
}

void neon_float_exp_inplace(float* values, uint32_t count) noexcept
{
    uint32_t index = 0;
    for (; count - index >= 4; index += 4)
        vst1q_f32(values + index, neon_expf(vld1q_f32(values + index)));
    for (; index < count; ++index)
        values[index] = float_approximate_exp(values[index]);
}

void neon_float_sigmoid_mul(float* output,
                            const float* gate,
                            const float* input,
                            uint32_t count) noexcept
{
    uint32_t index = 0;
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; count - index >= 4; index += 4)
    {
        const float32x4_t gate_values = vld1q_f32(gate + index);
        const float32x4_t input_values = vld1q_f32(input + index);
        const float32x4_t exponentials = neon_expf(vnegq_f32(gate_values));
        const float32x4_t result = vdivq_f32(input_values, vaddq_f32(one, exponentials));
        vst1q_f32(output + index, result);
    }
    for (; index < count; ++index)
        output[index] = input[index] / (1.0f + float_approximate_exp(-gate[index]));
}

void neon_float_silu_mul(float* output,
                         const float* gate,
                         const float* up,
                         float sigmoid_scale,
                         float up_offset,
                         uint32_t count) noexcept
{
    uint32_t index = 0;
    const float32x4_t scale = vdupq_n_f32(sigmoid_scale);
    const float32x4_t offset = vdupq_n_f32(up_offset);
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; count - index >= 4; index += 4)
    {
        const float32x4_t gate_values = vld1q_f32(gate + index);
        const float32x4_t up_values = vld1q_f32(up + index);
        const float32x4_t exponentials = neon_expf(vnegq_f32(vmulq_f32(scale, gate_values)));
        const float32x4_t silu = vdivq_f32(gate_values, vaddq_f32(one, exponentials));
        const float32x4_t biased_up = up_offset == 0.0f ? up_values : vaddq_f32(up_values, offset);
        vst1q_f32(output + index, vmulq_f32(silu, biased_up));
    }
    for (; index < count; ++index)
    {
        const float gate_value = gate[index];
        const float biased_up = up_offset == 0.0f ? up[index] : up[index] + up_offset;
        output[index] = gate_value / (1.0f + float_approximate_exp(-sigmoid_scale * gate_value))
                        * biased_up;
    }
}

void neon_float_silu_inplace(float* values, uint32_t count) noexcept
{
    uint32_t index = 0;
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; count - index >= 4; index += 4)
    {
        const float32x4_t input = vld1q_f32(values + index);
        const float32x4_t exponentials = neon_expf(vnegq_f32(input));
        vst1q_f32(values + index, vdivq_f32(input, vaddq_f32(one, exponentials)));
    }
    for (; index < count; ++index)
    {
        const float value = values[index];
        values[index] = value / (1.0f + float_approximate_exp(-value));
    }
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static float f16_sum(float16x8_t values) noexcept
{
    const float32x4_t low = vcvt_f32_f16(vget_low_f16(values));
    const float32x4_t high = vcvt_f32_f16(vget_high_f16(values));
    return vaddvq_f32(vaddq_f32(low, high));
}

void neon_f32_to_f16(uint16_t* output, const float* input, size_t count) noexcept
{
    size_t index = 0;
    const float32x4_t f16_maximum = vdupq_n_f32(65504.0f);
    const float32x4_t f16_minimum = vdupq_n_f32(-65504.0f);
    for (; index + 4 <= count; index += 4)
    {
        // vcvt_f16_f32 yields inf past 65504; saturate like the scalar tail so
        // KV outliers cannot poison the F16 attention tiles.
        const float32x4_t values = vmaxq_f32(vminq_f32(vld1q_f32(input + index), f16_maximum), f16_minimum);
        vst1_f16(reinterpret_cast<__fp16*>(output + index), vcvt_f16_f32(values));
    }
    for (; index < count; ++index)
        output[index] = float_to_f16(input[index]);
}

void neon_bf16_to_f16(uint16_t* output, const uint16_t* input, size_t count) noexcept
{
    size_t index = 0;
    const float32x4_t f16_maximum = vdupq_n_f32(65504.0f);
    const float32x4_t f16_minimum = vdupq_n_f32(-65504.0f);
    for (; index + 4 <= count; index += 4)
    {
        const uint32x4_t bits = vshlq_n_u32(vmovl_u16(vld1_u16(input + index)), 16);
        const float32x4_t values = vmaxq_f32(vminq_f32(vreinterpretq_f32_u32(bits), f16_maximum), f16_minimum);
        vst1_f16(reinterpret_cast<__fp16*>(output + index), vcvt_f16_f32(values));
    }
    for (; index < count; ++index)
    {
        const float value = std::bit_cast<float>(static_cast<uint32_t>(input[index]) << 16);
        output[index] = float_to_f16(value);
    }
}

void neon_f16_gemm_4x8(const uint16_t* weights,
                       size_t weight_stride,
                       const uint16_t* input,
                       size_t input_stride,
                       uint32_t input_columns,
                       uint32_t output_count,
                       uint32_t token_count,
                       float* output,
                       size_t output_stride) noexcept
{
    for (uint32_t row_base = 0; row_base < token_count; row_base += 4)
    {
        const uint32_t rows = std::min(4u, token_count - row_base);
        for (uint32_t column_base = 0; column_base < output_count; column_base += 4)
        {
            const uint32_t columns = std::min(4u, output_count - column_base);
            float16x8_t accumulators[4][4];
            for (uint32_t row = 0; row < rows; ++row)
                for (uint32_t column = 0; column < columns; ++column)
                    accumulators[row][column] = vdupq_n_f16(0.0f);
            uint32_t offset = 0;
            for (; offset + 8 <= input_columns; offset += 8)
            {
                float16x8_t queries[4];
                for (uint32_t row = 0; row < rows; ++row)
                    queries[row] = vld1q_f16(reinterpret_cast<const __fp16*>(input + (row_base + row) * input_stride + offset));
                for (uint32_t column = 0; column < columns; ++column)
                {
                    const float16x8_t key = vld1q_f16(reinterpret_cast<const __fp16*>(weights + (column_base + column) * weight_stride + offset));
                    for (uint32_t row = 0; row < rows; ++row)
                        accumulators[row][column] = vfmaq_f16(accumulators[row][column], queries[row], key);
                }
            }
            for (uint32_t row = 0; row < rows; ++row)
            {
                for (uint32_t column = 0; column < columns; ++column)
                {
                    float sum = f16_sum(accumulators[row][column]);
                    for (uint32_t tail = offset; tail < input_columns; ++tail)
                    {
                        sum += f16_to_f32(input[(row_base + row) * input_stride + tail])
                               * f16_to_f32(weights[(column_base + column) * weight_stride + tail]);
                    }
                    output[(row_base + row) * output_stride + column_base + column] = sum;
                }
            }
        }
    }
}

void neon_f16_pv(float* output,
                 const uint16_t* values,
                 size_t value_stride,
                 const uint16_t* weights,
                 uint32_t count,
                 uint32_t columns) noexcept
{
    uint32_t column = 0;
    // One broadcast feeds four 8-lane chunks (32 values).
    for (; column + 32 <= columns; column += 32)
    {
        float16x8_t accumulator0 = vdupq_n_f16(0.0f);
        float16x8_t accumulator1 = vdupq_n_f16(0.0f);
        float16x8_t accumulator2 = vdupq_n_f16(0.0f);
        float16x8_t accumulator3 = vdupq_n_f16(0.0f);
        for (uint32_t index = 0; index < count; ++index)
        {
            const float16x8_t weight = vdupq_n_f16(*reinterpret_cast<const __fp16*>(weights + index));
            const uint16_t* row = values + static_cast<size_t>(index) * value_stride + column;
            accumulator0 = vfmaq_f16(accumulator0, weight, vld1q_f16(reinterpret_cast<const __fp16*>(row)));
            accumulator1 = vfmaq_f16(accumulator1, weight, vld1q_f16(reinterpret_cast<const __fp16*>(row + 8)));
            accumulator2 = vfmaq_f16(accumulator2, weight, vld1q_f16(reinterpret_cast<const __fp16*>(row + 16)));
            accumulator3 = vfmaq_f16(accumulator3, weight, vld1q_f16(reinterpret_cast<const __fp16*>(row + 24)));
        }
        vst1q_f32(output + column, vcvt_f32_f16(vget_low_f16(accumulator0)));
        vst1q_f32(output + column + 4, vcvt_f32_f16(vget_high_f16(accumulator0)));
        vst1q_f32(output + column + 8, vcvt_f32_f16(vget_low_f16(accumulator1)));
        vst1q_f32(output + column + 12, vcvt_f32_f16(vget_high_f16(accumulator1)));
        vst1q_f32(output + column + 16, vcvt_f32_f16(vget_low_f16(accumulator2)));
        vst1q_f32(output + column + 20, vcvt_f32_f16(vget_high_f16(accumulator2)));
        vst1q_f32(output + column + 24, vcvt_f32_f16(vget_low_f16(accumulator3)));
        vst1q_f32(output + column + 28, vcvt_f32_f16(vget_high_f16(accumulator3)));
    }
    for (; column + 8 <= columns; column += 8)
    {
        float16x8_t accumulator = vdupq_n_f16(0.0f);
        for (uint32_t index = 0; index < count; ++index)
        {
            const float16x8_t weight = vdupq_n_f16(*reinterpret_cast<const __fp16*>(weights + index));
            accumulator = vfmaq_f16(accumulator,
                                    weight,
                                    vld1q_f16(reinterpret_cast<const __fp16*>(values + static_cast<size_t>(index) * value_stride + column)));
        }
        vst1q_f32(output + column, vcvt_f32_f16(vget_low_f16(accumulator)));
        vst1q_f32(output + column + 4, vcvt_f32_f16(vget_high_f16(accumulator)));
    }
    for (; column < columns; ++column)
    {
        float sum = 0.0f;
        for (uint32_t index = 0; index < count; ++index)
        {
            sum += f16_to_f32(values[static_cast<size_t>(index) * value_stride + column])
                   * f16_to_f32(weights[index]);
        }
        output[column] = sum;
    }
}

void neon_f16_pv4(float* output,
                  const uint16_t* values,
                  size_t value_stride,
                  const float* weights0,
                  const float* weights1,
                  const float* weights2,
                  const float* weights3,
                  uint32_t count,
                  uint32_t columns) noexcept
{
    uint32_t column = 0;
    for (; column + 32 <= columns; column += 32)
    {
        float16x8_t accumulator00 = vdupq_n_f16(0.0f);
        float16x8_t accumulator01 = vdupq_n_f16(0.0f);
        float16x8_t accumulator02 = vdupq_n_f16(0.0f);
        float16x8_t accumulator03 = vdupq_n_f16(0.0f);
        float16x8_t accumulator10 = vdupq_n_f16(0.0f);
        float16x8_t accumulator11 = vdupq_n_f16(0.0f);
        float16x8_t accumulator12 = vdupq_n_f16(0.0f);
        float16x8_t accumulator13 = vdupq_n_f16(0.0f);
        float16x8_t accumulator20 = vdupq_n_f16(0.0f);
        float16x8_t accumulator21 = vdupq_n_f16(0.0f);
        float16x8_t accumulator22 = vdupq_n_f16(0.0f);
        float16x8_t accumulator23 = vdupq_n_f16(0.0f);
        float16x8_t accumulator30 = vdupq_n_f16(0.0f);
        float16x8_t accumulator31 = vdupq_n_f16(0.0f);
        float16x8_t accumulator32 = vdupq_n_f16(0.0f);
        float16x8_t accumulator33 = vdupq_n_f16(0.0f);
        for (uint32_t index = 0; index < count; ++index)
        {
            const float16x8_t weight0 = vdupq_n_f16(static_cast<__fp16>(weights0[index]));
            const float16x8_t weight1 = vdupq_n_f16(static_cast<__fp16>(weights1[index]));
            const float16x8_t weight2 = vdupq_n_f16(static_cast<__fp16>(weights2[index]));
            const float16x8_t weight3 = vdupq_n_f16(static_cast<__fp16>(weights3[index]));
            const uint16_t* row = values + static_cast<size_t>(index) * value_stride + column;
            const float16x8_t value0 = vld1q_f16(reinterpret_cast<const __fp16*>(row));
            const float16x8_t value1 = vld1q_f16(reinterpret_cast<const __fp16*>(row + 8));
            const float16x8_t value2 = vld1q_f16(reinterpret_cast<const __fp16*>(row + 16));
            const float16x8_t value3 = vld1q_f16(reinterpret_cast<const __fp16*>(row + 24));
            accumulator00 = vfmaq_f16(accumulator00, weight0, value0);
            accumulator01 = vfmaq_f16(accumulator01, weight0, value1);
            accumulator02 = vfmaq_f16(accumulator02, weight0, value2);
            accumulator03 = vfmaq_f16(accumulator03, weight0, value3);
            accumulator10 = vfmaq_f16(accumulator10, weight1, value0);
            accumulator11 = vfmaq_f16(accumulator11, weight1, value1);
            accumulator12 = vfmaq_f16(accumulator12, weight1, value2);
            accumulator13 = vfmaq_f16(accumulator13, weight1, value3);
            accumulator20 = vfmaq_f16(accumulator20, weight2, value0);
            accumulator21 = vfmaq_f16(accumulator21, weight2, value1);
            accumulator22 = vfmaq_f16(accumulator22, weight2, value2);
            accumulator23 = vfmaq_f16(accumulator23, weight2, value3);
            accumulator30 = vfmaq_f16(accumulator30, weight3, value0);
            accumulator31 = vfmaq_f16(accumulator31, weight3, value1);
            accumulator32 = vfmaq_f16(accumulator32, weight3, value2);
            accumulator33 = vfmaq_f16(accumulator33, weight3, value3);
        }
        for (uint32_t row = 0; row < 4; ++row)
        {
            float* row_output = output + static_cast<size_t>(row) * columns + column;
            const float16x8_t a0 = row == 0 ? accumulator00 : row == 1 ? accumulator10
                                                          : row == 2   ? accumulator20
                                                                       : accumulator30;
            const float16x8_t a1 = row == 0 ? accumulator01 : row == 1 ? accumulator11
                                                          : row == 2   ? accumulator21
                                                                       : accumulator31;
            const float16x8_t a2 = row == 0 ? accumulator02 : row == 1 ? accumulator12
                                                          : row == 2   ? accumulator22
                                                                       : accumulator32;
            const float16x8_t a3 = row == 0 ? accumulator03 : row == 1 ? accumulator13
                                                          : row == 2   ? accumulator23
                                                                       : accumulator33;
            vst1q_f32(row_output, vcvt_f32_f16(vget_low_f16(a0)));
            vst1q_f32(row_output + 4, vcvt_f32_f16(vget_high_f16(a0)));
            vst1q_f32(row_output + 8, vcvt_f32_f16(vget_low_f16(a1)));
            vst1q_f32(row_output + 12, vcvt_f32_f16(vget_high_f16(a1)));
            vst1q_f32(row_output + 16, vcvt_f32_f16(vget_low_f16(a2)));
            vst1q_f32(row_output + 20, vcvt_f32_f16(vget_high_f16(a2)));
            vst1q_f32(row_output + 24, vcvt_f32_f16(vget_low_f16(a3)));
            vst1q_f32(row_output + 28, vcvt_f32_f16(vget_high_f16(a3)));
        }
    }
    for (; column + 8 <= columns; column += 8)
    {
        float16x8_t accumulator0 = vdupq_n_f16(0.0f);
        float16x8_t accumulator1 = vdupq_n_f16(0.0f);
        float16x8_t accumulator2 = vdupq_n_f16(0.0f);
        float16x8_t accumulator3 = vdupq_n_f16(0.0f);
        for (uint32_t index = 0; index < count; ++index)
        {
            const float16x8_t value = vld1q_f16(reinterpret_cast<const __fp16*>(values + static_cast<size_t>(index) * value_stride + column));
            accumulator0 = vfmaq_f16(accumulator0, vdupq_n_f16(static_cast<__fp16>(weights0[index])), value);
            accumulator1 = vfmaq_f16(accumulator1, vdupq_n_f16(static_cast<__fp16>(weights1[index])), value);
            accumulator2 = vfmaq_f16(accumulator2, vdupq_n_f16(static_cast<__fp16>(weights2[index])), value);
            accumulator3 = vfmaq_f16(accumulator3, vdupq_n_f16(static_cast<__fp16>(weights3[index])), value);
        }
        vst1q_f32(output + column, vcvt_f32_f16(vget_low_f16(accumulator0)));
        vst1q_f32(output + column + 4, vcvt_f32_f16(vget_high_f16(accumulator0)));
        vst1q_f32(output + columns + column, vcvt_f32_f16(vget_low_f16(accumulator1)));
        vst1q_f32(output + columns + column + 4, vcvt_f32_f16(vget_high_f16(accumulator1)));
        vst1q_f32(output + 2 * columns + column, vcvt_f32_f16(vget_low_f16(accumulator2)));
        vst1q_f32(output + 2 * columns + column + 4, vcvt_f32_f16(vget_high_f16(accumulator2)));
        vst1q_f32(output + 3 * columns + column, vcvt_f32_f16(vget_low_f16(accumulator3)));
        vst1q_f32(output + 3 * columns + column + 4, vcvt_f32_f16(vget_high_f16(accumulator3)));
    }
    for (; column < columns; ++column)
    {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        for (uint32_t index = 0; index < count; ++index)
        {
            const float value = f16_to_f32(values[static_cast<size_t>(index) * value_stride + column]);
            sum0 += weights0[index] * value;
            sum1 += weights1[index] * value;
            sum2 += weights2[index] * value;
            sum3 += weights3[index] * value;
        }
        output[column] = sum0;
        output[columns + column] = sum1;
        output[2 * columns + column] = sum2;
        output[3 * columns + column] = sum3;
    }
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace moe
} // namespace ncnn

#endif // ARM64
