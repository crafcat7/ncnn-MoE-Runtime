#include "vector.h"

#include "fastmath.h"
#include "bfloat16.h"
#include "engine/cpu.h"
#include "ncnn/moe/runtime.h"

#include <bit>
#include <cmath>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if defined(NCNN_MOE_X86_SIMD) \
    && (defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
#define NCNN_MOE_VECTOR_X86_SIMD 1
#include "vector_x86.h"
#endif

namespace ncnn {
namespace moe {

using FloatDotFunction = float (*)(const float*, const float*, uint32_t) noexcept;
using FloatExpInplaceFunction = void (*)(float*, uint32_t) noexcept;
using Int8FloatDotFunction = float (*)(const int8_t*, const float*, uint32_t) noexcept;
using FloatScaleFunction = void (*)(float*, float, uint32_t) noexcept;
using FloatScaledAddFunction = void (*)(float*, const float*, float, uint32_t) noexcept;
using FloatScaleAddFunction = void (*)(float*, float, const float*, float, uint32_t) noexcept;
using FloatScaleInplaceAndScaledAddFunction = void (*)(float*, float, float*, float, uint32_t) noexcept;
using FloatScaleInplaceAndScaledAddAndAccumulateFunction = void (*)(float*, float, const float*, float, float*, float, uint32_t) noexcept;
using FloatWeightedScaleFunction = void (*)(float*, const float*, const float*, float, float, uint32_t) noexcept;
using Bfloat16WeightedScaleFunction = void (*)(float*, const float*, const uint16_t*, float, float, uint32_t) noexcept;
using FloatRmsScaleFunction = void (*)(float*, float, uint32_t) noexcept;
using FloatL2ScaleFunction = void (*)(float*, float, uint32_t) noexcept;
using FloatRmsNormFunction = void (*)(float*, const float*, const float*, float, float, uint32_t) noexcept;
using Bfloat16RmsNormFunction = void (*)(float*, const float*, const uint16_t*, float, float, uint32_t) noexcept;
using FloatRopeFunction = void (*)(float*, const float*, const float*, uint32_t) noexcept;
using FloatHcPre4Function = void (*)(float*, const float*, float, float, float, float, uint32_t) noexcept;
using FloatHcPost4Function = void (*)(float*, const float*, const float*, const float*, const float*, uint32_t) noexcept;

static float scalar_float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    float result = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        result += left[index] * right[index];
    return result;
}

#if defined(__aarch64__) || defined(_M_ARM64)
static float neon_float_dot(const float* left, const float* right, uint32_t count) noexcept
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

static void neon_float_gemm(const float* weights,
                            size_t weight_stride,
                            const float* input,
                            size_t input_stride,
                            uint32_t input_columns,
                            uint32_t output_count,
                            uint32_t token_count,
                            float* output,
                            size_t output_stride) noexcept
{
    // Four outputs keep the sixteen vector accumulators in registers. The
    // 4x8 entry point reuses this tile instead of doubling register pressure.
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

static void neon_bfloat16_gemm(const uint16_t* weights,
                               size_t weight_stride,
                               const float* input,
                               size_t input_stride,
                               uint32_t input_columns,
                               uint32_t output_count,
                               uint32_t token_count,
                               float* output,
                               size_t output_stride) noexcept
{
    // Expand each BF16 weight vector once for four input rows. Accumulation
    // and activations stay FP32, requiring only the baseline ARM64 NEON ISA.
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
                    for (uint32_t j = 0; j < 4; ++j)
                        sums[r][j] = vfmaq_f32(sums[r][j], x[j], vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + r * weight_stride + k + j * 4), 16)));
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
            output[static_cast<size_t>(token) * output_stride + row] = bfloat16_dot(weights + static_cast<size_t>(row) * weight_stride,
                                                                                    input + static_cast<size_t>(token) * input_stride,
                                                                                    input_columns);
}

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
#endif

static void scalar_float_gemm_4x4(const float* weights,
                                  size_t weight_stride,
                                  const float* input,
                                  size_t input_stride,
                                  uint32_t input_columns,
                                  uint32_t output_count,
                                  uint32_t token_count,
                                  float* output,
                                  size_t output_stride) noexcept
{
    float accumulators[4][4] = {};
    for (uint32_t column = 0; column < input_columns; ++column)
    {
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const float value = input[static_cast<size_t>(token) * input_stride + column];
            for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            {
                accumulators[token][output_index] += value * weights[static_cast<size_t>(output_index) * weight_stride + column];
            }
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = accumulators[token][output_index];
}

static void scalar_float_gemm_4x8(const float* weights,
                                  size_t weight_stride,
                                  const float* input,
                                  size_t input_stride,
                                  uint32_t input_columns,
                                  uint32_t output_count,
                                  uint32_t token_count,
                                  float* output,
                                  size_t output_stride) noexcept
{
    float accumulators[4][8] = {};
    for (uint32_t column = 0; column < input_columns; ++column)
    {
        float input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
            input_values[token] = input[static_cast<size_t>(token) * input_stride + column];
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const float weight = weights[static_cast<size_t>(output_index) * weight_stride + column];
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] += input_values[token] * weight;
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = accumulators[token][output_index];
}

static void scalar_bfloat16_gemm_4x8(const uint16_t* weights,
                                     size_t weight_stride,
                                     const float* input,
                                     size_t input_stride,
                                     uint32_t input_columns,
                                     uint32_t output_count,
                                     uint32_t token_count,
                                     float* output,
                                     size_t output_stride) noexcept
{
    float accumulators[4][8] = {};
    for (uint32_t column = 0; column < input_columns; ++column)
    {
        float input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
            input_values[token] = input[static_cast<size_t>(token) * input_stride + column];
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const uint32_t bits = static_cast<uint32_t>(weights[static_cast<size_t>(output_index) * weight_stride + column])
                                  << 16;
            const float weight = std::bit_cast<float>(bits);
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] += input_values[token] * weight;
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = accumulators[token][output_index];
}

static float scalar_int8_float_dot(const int8_t* left,
                                   const float* right,
                                   uint32_t count) noexcept
{
    float result = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        result += static_cast<float>(left[index]) * right[index];
    return result;
}

static void scalar_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] += scale * input[index];
}

static void scalar_float_scale_inplace(float* values, float scale, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        values[index] *= scale;
}

static void scalar_float_rms_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    if (count == 0)
        return;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += values[index] * values[index];
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
        values[index] *= inverse_rms;
}

static void scalar_float_l2_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    if (count == 0)
        return;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += values[index] * values[index];
    const float inverse_norm = 1.0f / std::sqrt(square_sum + epsilon);
    for (uint32_t index = 0; index < count; ++index)
        values[index] *= inverse_norm;
}

static void scalar_float_scale_add(float* output,
                                   float output_scale,
                                   const float* input,
                                   float input_scale,
                                   uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] = output[index] * output_scale + input[index] * input_scale;
}

static void scalar_float_scale_inplace_and_scaled_add(float* values,
                                                      float value_scale,
                                                      float* output,
                                                      float output_scale,
                                                      uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        values[index] *= value_scale;
        output[index] += output_scale * values[index];
    }
}

static void scalar_float_scale_inplace_and_scaled_add_and_accumulate(float* values,
                                                                     float value_scale,
                                                                     const float* input,
                                                                     float input_scale,
                                                                     float* output,
                                                                     float output_scale,
                                                                     uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        values[index] = values[index] * value_scale
                        + input[index] * input_scale;
        output[index] += output_scale * values[index];
    }
}

static void scalar_float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] = input[index] * scale * (weight[index] + weight_offset);
}

static void scalar_bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        const float value = std::bit_cast<float>(static_cast<uint32_t>(weight[index]) << 16);
        output[index] = input[index] * scale * (value + weight_offset);
    }
}

static void scalar_float_rms_norm(float* output,
                                  const float* input,
                                  const float* weight,
                                  float epsilon,
                                  float weight_offset,
                                  uint32_t count) noexcept
{
    if (count == 0)
        return;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += input[index] * input[index];
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
        output[index] = input[index] * inverse_rms * (weight[index] + weight_offset);
}

static void scalar_bfloat16_rms_norm(float* output,
                                     const float* input,
                                     const uint16_t* weight,
                                     float epsilon,
                                     float weight_offset,
                                     uint32_t count) noexcept
{
    if (count == 0)
        return;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
        square_sum += input[index] * input[index];
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    for (uint32_t index = 0; index < count; ++index)
    {
        const float weight_value = std::bit_cast<float>(static_cast<uint32_t>(weight[index]) << 16);
        output[index] = input[index] * inverse_rms * (weight_value + weight_offset);
    }
}

static void scalar_float_rope_inplace(float* values,
                                      const float* cosine,
                                      const float* sine,
                                      uint32_t dimension) noexcept
{
    const uint32_t half_dimension = dimension / 2;
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const float first = values[index];
        const float second = values[half_dimension + index];
        values[index] = first * cosine[index] - second * sine[index];
        values[half_dimension + index] = second * cosine[index] + first * sine[index];
    }
}

static void scalar_float_hc_pre_4(float* output,
                                  const float* input,
                                  float scale0,
                                  float scale1,
                                  float scale2,
                                  float scale3,
                                  uint32_t hidden_size) noexcept
{
    const float* input1 = input + hidden_size;
    const float* input2 = input1 + hidden_size;
    const float* input3 = input2 + hidden_size;
    for (uint32_t index = 0; index < hidden_size; ++index)
        output[index] = input[index] * scale0
                        + input1[index] * scale1
                        + input2[index] * scale2
                        + input3[index] * scale3;
}

static void scalar_float_hc_post_4(float* output,
                                   const float* branch,
                                   const float* residual,
                                   const float* post,
                                   const float* combine,
                                   uint32_t hidden_size) noexcept
{
    for (uint32_t index = 0; index < hidden_size; ++index)
    {
        for (uint32_t output_index = 0; output_index < 4; ++output_index)
        {
            float value = branch[index] * post[output_index];
            for (uint32_t residual_index = 0; residual_index < 4; ++residual_index)
            {
                value += residual[static_cast<size_t>(residual_index) * hidden_size + index]
                         * combine[residual_index * 4 + output_index];
            }
            output[static_cast<size_t>(output_index) * hidden_size + index] = value;
        }
    }
}

static void scalar_float_sigmoid_mul(float* output,
                                     const float* gate,
                                     const float* input,
                                     uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
        output[index] = input[index] / (1.0f + float_approximate_exp(-gate[index]));
}

static void scalar_float_silu_mul(float* output, const float* gate, const float* up,
                                  float sigmoid_scale, float up_offset, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        const float gate_value = gate[index];
        const float up_value = up[index];
        const float biased_up = up_offset == 0.0f ? up_value : up_value + up_offset;
        output[index] = gate_value / (1.0f + float_approximate_exp(-sigmoid_scale * gate_value))
                        * biased_up;
    }
}

static void scalar_float_silu_inplace(float* values, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        const float value = values[index];
        values[index] = value / (1.0f + float_approximate_exp(-value));
    }
}

static FloatDotFunction select_float_dot() noexcept
{
#if defined(__aarch64__) || defined(_M_ARM64)
    return neon_float_dot;
#elif defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_dot;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_dot;
#endif
    return scalar_float_dot;
}

static FloatExpInplaceFunction select_float_exp_inplace() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_exp_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_exp_inplace;
#endif
    return nullptr;
}

static Int8FloatDotFunction select_int8_float_dot() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_int8_float_dot;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_int8_float_dot;
#endif
    return scalar_int8_float_dot;
}

static FloatScaledAddFunction select_float_scaled_add() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_scaled_add;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_scaled_add;
#endif
    return scalar_float_scaled_add;
}

static FloatScaleFunction select_float_scale() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_scale_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_scale_inplace;
#endif
    return scalar_float_scale_inplace;
}

static FloatScaleAddFunction select_float_scale_add() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_scale_add;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_scale_add;
#endif
    return scalar_float_scale_add;
}

static FloatScaleInplaceAndScaledAddFunction select_float_scale_inplace_and_scaled_add() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_scale_inplace_and_scaled_add;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_scale_inplace_and_scaled_add;
#endif
    return scalar_float_scale_inplace_and_scaled_add;
}

static FloatScaleInplaceAndScaledAddAndAccumulateFunction
select_float_scale_inplace_and_scaled_add_and_accumulate() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_scale_inplace_and_scaled_add_and_accumulate;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_scale_inplace_and_scaled_add_and_accumulate;
#endif
    return scalar_float_scale_inplace_and_scaled_add_and_accumulate;
}

static FloatWeightedScaleFunction select_float_weighted_scale() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_weighted_scale;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_weighted_scale;
#endif
    return scalar_float_weighted_scale;
}

static Bfloat16WeightedScaleFunction select_bfloat16_weighted_scale() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_bfloat16_weighted_scale;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_bfloat16_weighted_scale;
#endif
    return scalar_bfloat16_weighted_scale;
}

static FloatRmsScaleFunction select_float_rms_scale() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_rms_scale_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_rms_scale_inplace;
#endif
    return scalar_float_rms_scale_inplace;
}

static FloatL2ScaleFunction select_float_l2_scale() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_l2_scale_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_l2_scale_inplace;
#endif
    return scalar_float_l2_scale_inplace;
}

static FloatRmsNormFunction select_float_rms_norm() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_rms_norm;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_rms_norm;
#endif
    return scalar_float_rms_norm;
}

static Bfloat16RmsNormFunction select_bfloat16_rms_norm() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_bfloat16_rms_norm;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_bfloat16_rms_norm;
#endif
    return scalar_bfloat16_rms_norm;
}

static FloatRopeFunction select_float_rope() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_rope_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_rope_inplace;
#endif
    return scalar_float_rope_inplace;
}

static FloatHcPre4Function select_float_hc_pre_4() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_hc_pre_4;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_hc_pre_4;
#endif
    return scalar_float_hc_pre_4;
}

static FloatHcPost4Function select_float_hc_post_4() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_hc_post_4;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_hc_post_4;
#endif
    return scalar_float_hc_post_4;
}

using FloatSigmoidMulFunction = void (*)(float*, const float*, const float*, uint32_t) noexcept;
using FloatSiluMulFunction = void (*)(float*, const float*, const float*, float, float, uint32_t) noexcept;
using FloatSiluInplaceFunction = void (*)(float*, uint32_t) noexcept;

static FloatSigmoidMulFunction select_float_sigmoid_mul() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_sigmoid_mul;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_sigmoid_mul;
#endif
    return scalar_float_sigmoid_mul;
}

static FloatSiluMulFunction select_float_silu_mul() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_silu_mul;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_silu_mul;
#endif
    return scalar_float_silu_mul;
}

static FloatSiluInplaceFunction select_float_silu_inplace() noexcept
{
#if defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_float_silu_inplace;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_silu_inplace;
#endif
    return scalar_float_silu_inplace;
}

float float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    static const FloatDotFunction function = select_float_dot();
    return function(left, right, count);
}

void float_gemm_4x4(const float* weights,
                    size_t weight_stride,
                    const float* input,
                    size_t input_stride,
                    uint32_t input_columns,
                    uint32_t output_count,
                    uint32_t token_count,
                    float* output,
                    size_t output_stride) noexcept
{
#if defined(__aarch64__) || defined(_M_ARM64)
    neon_float_gemm(weights, weight_stride, input, input_stride, input_columns,
                    output_count, token_count, output, output_stride);
    return;
#elif defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
    {
        avx512_float_gemm_4x4(weights,
                              weight_stride,
                              input,
                              input_stride,
                              input_columns,
                              output_count,
                              token_count,
                              output,
                              output_stride);
        return;
    }
    if ((isa & CpuIsaX86Avx2Fma) != 0)
    {
        avx2_float_gemm_4x4(weights,
                            weight_stride,
                            input,
                            input_stride,
                            input_columns,
                            output_count,
                            token_count,
                            output,
                            output_stride);
        return;
    }
#endif
    scalar_float_gemm_4x4(weights,
                          weight_stride,
                          input,
                          input_stride,
                          input_columns,
                          output_count,
                          token_count,
                          output,
                          output_stride);
}

void float_gemm_4x8(const float* weights,
                    size_t weight_stride,
                    const float* input,
                    size_t input_stride,
                    uint32_t input_columns,
                    uint32_t output_count,
                    uint32_t token_count,
                    float* output,
                    size_t output_stride) noexcept
{
#if defined(__aarch64__) || defined(_M_ARM64)
    neon_float_gemm(weights, weight_stride, input, input_stride, input_columns,
                    output_count, token_count, output, output_stride);
    return;
#elif defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
    {
        avx512_float_gemm_4x8(weights, weight_stride, input, input_stride, input_columns,
                              output_count, token_count, output, output_stride);
        return;
    }
    if ((isa & CpuIsaX86Avx2Fma) != 0)
    {
        avx2_float_gemm_4x8(weights, weight_stride, input, input_stride, input_columns,
                            output_count, token_count, output, output_stride);
        return;
    }
#endif
    scalar_float_gemm_4x8(weights, weight_stride, input, input_stride, input_columns,
                          output_count, token_count, output, output_stride);
}

void bfloat16_gemm_4x8(const uint16_t* weights,
                       size_t weight_stride,
                       const float* input,
                       size_t input_stride,
                       uint32_t input_columns,
                       uint32_t output_count,
                       uint32_t token_count,
                       float* output,
                       size_t output_stride) noexcept
{
#if defined(__aarch64__) || defined(_M_ARM64)
    neon_bfloat16_gemm(weights, weight_stride, input, input_stride, input_columns,
                       output_count, token_count, output, output_stride);
    return;
#elif defined(NCNN_MOE_VECTOR_X86_SIMD)
    const uint64_t isa = cpu_isa_flags();
    if ((isa & CpuIsaX86Avx512) != 0)
    {
        avx512_bfloat16_gemm_4x8(weights, weight_stride, input, input_stride, input_columns,
                                 output_count, token_count, output, output_stride);
        return;
    }
    if ((isa & CpuIsaX86Avx2Fma) != 0)
    {
        avx2_bfloat16_gemm_4x8(weights, weight_stride, input, input_stride, input_columns,
                               output_count, token_count, output, output_stride);
        return;
    }
#endif
    scalar_bfloat16_gemm_4x8(weights, weight_stride, input, input_stride, input_columns,
                             output_count, token_count, output, output_stride);
}

void float_exp_inplace(float* values, uint32_t count) noexcept
{
    static const FloatExpInplaceFunction function = select_float_exp_inplace();
    uint32_t index = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    for (; count - index >= 4; index += 4)
        vst1q_f32(values + index, neon_expf(vld1q_f32(values + index)));
#endif
    if (function && index < count)
    {
        function(values + index, count - index);
        return;
    }
    for (; index < count; ++index)
        values[index] = float_approximate_exp(values[index]);
}

bool float_exp_simd_available() noexcept
{
#if defined(__aarch64__) || defined(_M_ARM64)
    return true;
#else
    static const FloatExpInplaceFunction function = select_float_exp_inplace();
    return function != nullptr;
#endif
}

float int8_float_dot(const int8_t* left, const float* right, uint32_t count) noexcept
{
    static const Int8FloatDotFunction function = select_int8_float_dot();
    return function(left, right, count);
}

void float_l2_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    static const FloatL2ScaleFunction function = select_float_l2_scale();
    function(values, epsilon, count);
}

void float_rms_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    static const FloatRmsScaleFunction function = select_float_rms_scale();
    function(values, epsilon, count);
}

void float_rms_norm(float* output, const float* input, const float* weight,
                    float epsilon, float weight_offset, uint32_t count) noexcept
{
    static const FloatRmsNormFunction function = select_float_rms_norm();
    function(output, input, weight, epsilon, weight_offset, count);
}

void bfloat16_rms_norm(float* output, const float* input, const uint16_t* weight,
                       float epsilon, float weight_offset, uint32_t count) noexcept
{
    static const Bfloat16RmsNormFunction function = select_bfloat16_rms_norm();
    function(output, input, weight, epsilon, weight_offset, count);
}

void float_rope_inplace(float* values, const float* cosine, const float* sine,
                        uint32_t dimension) noexcept
{
    static const FloatRopeFunction function = select_float_rope();
    function(values, cosine, sine, dimension);
}

void float_scale_inplace(float* values, float scale, uint32_t count) noexcept
{
    static const FloatScaleFunction function = select_float_scale();
    function(values, scale, count);
}

void float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
{
    static const FloatScaledAddFunction function = select_float_scaled_add();
    function(output, input, scale, count);
}

void float_scale_add(float* output,
                     float output_scale,
                     const float* input,
                     float input_scale,
                     uint32_t count) noexcept
{
    static const FloatScaleAddFunction function = select_float_scale_add();
    function(output, output_scale, input, input_scale, count);
}

void float_scale_inplace_and_scaled_add(float* values,
                                        float value_scale,
                                        float* output,
                                        float output_scale,
                                        uint32_t count) noexcept
{
    static const FloatScaleInplaceAndScaledAddFunction function = select_float_scale_inplace_and_scaled_add();
    function(values, value_scale, output, output_scale, count);
}

void float_scale_inplace_and_scaled_add_and_accumulate(float* values,
                                                       float value_scale,
                                                       const float* input,
                                                       float input_scale,
                                                       float* output,
                                                       float output_scale,
                                                       uint32_t count) noexcept
{
    static const FloatScaleInplaceAndScaledAddAndAccumulateFunction function = select_float_scale_inplace_and_scaled_add_and_accumulate();
    function(values, value_scale, input, input_scale, output, output_scale, count);
}

void float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    static const FloatWeightedScaleFunction function = select_float_weighted_scale();
    function(output, input, weight, scale, weight_offset, count);
}

void bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    static const Bfloat16WeightedScaleFunction function = select_bfloat16_weighted_scale();
    function(output, input, weight, scale, weight_offset, count);
}

void float_sigmoid_mul(float* output,
                       const float* gate,
                       const float* input,
                       uint32_t count) noexcept
{
    static const FloatSigmoidMulFunction function = select_float_sigmoid_mul();
    uint32_t index = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; count - index >= 4; index += 4)
    {
        const float32x4_t gate_values = vld1q_f32(gate + index);
        const float32x4_t input_values = vld1q_f32(input + index);
        const float32x4_t exponentials = neon_expf(vnegq_f32(gate_values));
        const float32x4_t result = vdivq_f32(input_values, vaddq_f32(one, exponentials));
        vst1q_f32(output + index, result);
    }
#endif
    if (index < count)
        function(output + index, gate + index, input + index, count - index);
}

void float_silu_mul(float* output, const float* gate, const float* up,
                    float sigmoid_scale, float up_offset, uint32_t count) noexcept
{
    static const FloatSiluMulFunction function = select_float_silu_mul();
    uint32_t index = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
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
#endif
    if (index < count)
        function(output + index,
                 gate + index,
                 up + index,
                 sigmoid_scale,
                 up_offset,
                 count - index);
}

void float_silu_inplace(float* values, uint32_t count) noexcept
{
    static const FloatSiluInplaceFunction function = select_float_silu_inplace();
    uint32_t index = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; count - index >= 4; index += 4)
    {
        const float32x4_t input = vld1q_f32(values + index);
        const float32x4_t exponentials = neon_expf(vnegq_f32(input));
        vst1q_f32(values + index, vdivq_f32(input, vaddq_f32(one, exponentials)));
    }
#endif
    if (index < count)
        function(values + index, count - index);
}

void float_hc_pre_4(float* output, const float* input,
                    float scale0, float scale1, float scale2, float scale3,
                    uint32_t hidden_size) noexcept
{
    static const FloatHcPre4Function function = select_float_hc_pre_4();
    function(output, input, scale0, scale1, scale2, scale3, hidden_size);
}

void float_hc_post_4(float* output, const float* branch, const float* residual,
                     const float* post, const float* combine,
                     uint32_t hidden_size) noexcept
{
    static const FloatHcPost4Function function = select_float_hc_post_4();
    function(output, branch, residual, post, combine, hidden_size);
}

} // namespace moe
} // namespace ncnn
