#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)

#include "vector_x86.h"

#include "fastmath.h"
#include <bit>
#include <cmath>
#include <immintrin.h>
#include <limits>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl")
#endif

namespace ncnn {
namespace moe {

static __m512 avx512_expf(__m512 value) noexcept
{
    // Preserve scalar boundary and non-finite behavior outside the fast range.

    if (_mm512_cmp_ps_mask(_mm512_abs_ps(value), _mm512_set1_ps(80.0f), _CMP_NLT_UQ) != 0)
    {
        alignas(64) float lanes[16];
        _mm512_store_ps(lanes, value);
        for (float& lane : lanes)
            lane = float_approximate_exp(lane);
        return _mm512_load_ps(lanes);
    }

    const __m512 rounding = _mm512_set1_ps(0x1.8p23f);
    const __m512 scaled = _mm512_fmadd_ps(value, _mm512_set1_ps(0x1.715476p+0f), rounding);
    const __m512 exponent = _mm512_sub_ps(scaled, rounding);
    const __m512 remainder = _mm512_fnmadd_ps(exponent, _mm512_set1_ps(0x1.7f7d1cp-20f),
                                              _mm512_fnmadd_ps(exponent, _mm512_set1_ps(0x1.62e4p-1f), value));
    const __m512 remainder_squared = _mm512_mul_ps(remainder, remainder);
    const __m512 polynomial = _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), remainder,
                                                                              _mm512_set1_ps(0x1.573e2ep-5f)),
                                                              remainder_squared,
                                                              _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), remainder,
                                                                              _mm512_set1_ps(0x1.fffdb6p-2f))),
                                              remainder_squared,
                                              _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), remainder,
                                                              _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(polynomial, exponent);
}

void avx512_float_sigmoid_mul(float* output,
                              const float* gate,
                              const float* input,
                              uint32_t count) noexcept
{
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 gate_values = _mm512_loadu_ps(gate + index);
        const __m512 sigmoid = _mm512_div_ps(one,
                                             _mm512_add_ps(one, avx512_expf(_mm512_sub_ps(zero, gate_values))));
        _mm512_storeu_ps(output + index,
                         _mm512_mul_ps(sigmoid, _mm512_loadu_ps(input + index)));
    }
    for (; index < count; ++index)
        output[index] = input[index] / (1.0f + float_approximate_exp(-gate[index]));
}

void avx512_float_silu_mul(float* output, const float* gate, const float* up,
                           float sigmoid_scale, float up_offset, uint32_t count) noexcept
{
    const __m512 scale = _mm512_set1_ps(sigmoid_scale);
    const __m512 offset = _mm512_set1_ps(up_offset);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 gate_values = _mm512_loadu_ps(gate + index);
        const __m512 silu = _mm512_div_ps(gate_values,
                                          _mm512_add_ps(one, avx512_expf(_mm512_sub_ps(zero, _mm512_mul_ps(scale, gate_values)))));
        const __m512 up_values = _mm512_loadu_ps(up + index);
        const __m512 biased_up = up_offset == 0.0f ? up_values : _mm512_add_ps(up_values, offset);
        _mm512_storeu_ps(output + index, _mm512_mul_ps(silu, biased_up));
    }
    for (; index < count; ++index)
    {
        const float gate_value = gate[index];
        const float biased_up = up_offset == 0.0f ? up[index] : up[index] + up_offset;
        output[index] = gate_value / (1.0f + float_approximate_exp(-sigmoid_scale * gate_value))
                        * biased_up;
    }
}

void avx512_float_silu_inplace(float* values, uint32_t count) noexcept
{
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 input = _mm512_loadu_ps(values + index);
        const __m512 silu = _mm512_div_ps(input,
                                          _mm512_add_ps(one, avx512_expf(_mm512_sub_ps(zero, input))));
        _mm512_storeu_ps(values + index, silu);
    }
    for (; index < count; ++index)
    {
        const float value = values[index];
        values[index] = value / (1.0f + float_approximate_exp(-value));
    }
}

float avx512_float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    __m512 accumulator0 = _mm512_setzero_ps();
    __m512 accumulator1 = _mm512_setzero_ps();
    __m512 accumulator2 = _mm512_setzero_ps();
    __m512 accumulator3 = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 64 <= count; index += 64)
    {
        accumulator0 = _mm512_fmadd_ps(_mm512_loadu_ps(left + index), _mm512_loadu_ps(right + index), accumulator0);
        accumulator1 = _mm512_fmadd_ps(_mm512_loadu_ps(left + index + 16), _mm512_loadu_ps(right + index + 16), accumulator1);
        accumulator2 = _mm512_fmadd_ps(_mm512_loadu_ps(left + index + 32), _mm512_loadu_ps(right + index + 32), accumulator2);
        accumulator3 = _mm512_fmadd_ps(_mm512_loadu_ps(left + index + 48), _mm512_loadu_ps(right + index + 48), accumulator3);
    }
    const __m512 sum = _mm512_add_ps(_mm512_add_ps(accumulator0, accumulator1), _mm512_add_ps(accumulator2, accumulator3));
    float result = _mm512_reduce_add_ps(sum);
    for (; index < count; ++index)
        result += left[index] * right[index];
    return result;
}

void avx512_float_l2_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = avx512_float_dot(values, values, count);
    const float inverse_norm = 1.0f / std::sqrt(square_sum + epsilon);
    const __m512 scale = _mm512_set1_ps(inverse_norm);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
        _mm512_storeu_ps(values + index, _mm512_mul_ps(_mm512_loadu_ps(values + index), scale));
    for (; index < count; ++index)
        values[index] *= inverse_norm;
}

void avx512_float_rms_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = avx512_float_dot(values, values, count);
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    const __m512 scale = _mm512_set1_ps(inverse_rms);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
        _mm512_storeu_ps(values + index, _mm512_mul_ps(_mm512_loadu_ps(values + index), scale));
    for (; index < count; ++index)
        values[index] *= inverse_rms;
}

void avx512_float_rms_norm(float* output,
                           const float* input,
                           const float* weight,
                           float epsilon,
                           float weight_offset,
                           uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = avx512_float_dot(input, input, count);
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    const __m512 inverse_values = _mm512_set1_ps(inverse_rms);
    const __m512 offset_values = _mm512_set1_ps(weight_offset);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 normalized = _mm512_mul_ps(_mm512_loadu_ps(input + index), inverse_values);
        const __m512 weighted = _mm512_add_ps(_mm512_loadu_ps(weight + index), offset_values);
        _mm512_storeu_ps(output + index, _mm512_mul_ps(normalized, weighted));
    }
    for (; index < count; ++index)
        output[index] = input[index] * inverse_rms * (weight[index] + weight_offset);
}

void avx512_bfloat16_rms_norm(float* output,
                              const float* input,
                              const uint16_t* weight,
                              float epsilon,
                              float weight_offset,
                              uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = avx512_float_dot(input, input, count);
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    const __m512 inverse_values = _mm512_set1_ps(inverse_rms);
    const __m512 offset_values = _mm512_set1_ps(weight_offset);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight + index));
        const __m512i weight_bits = _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16);
        const __m512 normalized = _mm512_mul_ps(_mm512_loadu_ps(input + index), inverse_values);
        const __m512 weighted = _mm512_add_ps(_mm512_castsi512_ps(weight_bits), offset_values);
        _mm512_storeu_ps(output + index, _mm512_mul_ps(normalized, weighted));
    }
    for (; index < count; ++index)
    {
        const float weight_value = std::bit_cast<float>(static_cast<uint32_t>(weight[index]) << 16);
        output[index] = input[index] * inverse_rms * (weight_value + weight_offset);
    }
}

void avx512_float_rope_inplace(float* values,
                               const float* cosine,
                               const float* sine,
                               uint32_t dimension) noexcept
{
    const uint32_t half_dimension = dimension / 2;
    uint32_t index = 0;
    for (; index + 16 <= half_dimension; index += 16)
    {
        const __m512 first = _mm512_loadu_ps(values + index);
        const __m512 second = _mm512_loadu_ps(values + half_dimension + index);
        const __m512 cosine_values = _mm512_loadu_ps(cosine + index);
        const __m512 sine_values = _mm512_loadu_ps(sine + index);
        const __m512 first_result = _mm512_fmsub_ps(first, cosine_values, _mm512_mul_ps(second, sine_values));
        const __m512 second_result = _mm512_fmadd_ps(second, cosine_values, _mm512_mul_ps(first, sine_values));
        _mm512_storeu_ps(values + index, first_result);
        _mm512_storeu_ps(values + half_dimension + index, second_result);
    }
    for (; index < half_dimension; ++index)
    {
        const float first = values[index];
        const float second = values[half_dimension + index];
        values[index] = first * cosine[index] - second * sine[index];
        values[half_dimension + index] = second * cosine[index] + first * sine[index];
    }
}

void avx512_float_hc_pre_4(float* output,
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
    const __m512 scale0_values = _mm512_set1_ps(scale0);
    const __m512 scale1_values = _mm512_set1_ps(scale1);
    const __m512 scale2_values = _mm512_set1_ps(scale2);
    const __m512 scale3_values = _mm512_set1_ps(scale3);
    uint32_t index = 0;
    for (; index + 16 <= hidden_size; index += 16)
    {
        __m512 result = _mm512_mul_ps(_mm512_loadu_ps(input + index), scale0_values);
        result = _mm512_fmadd_ps(_mm512_loadu_ps(input1 + index), scale1_values, result);
        result = _mm512_fmadd_ps(_mm512_loadu_ps(input2 + index), scale2_values, result);
        result = _mm512_fmadd_ps(_mm512_loadu_ps(input3 + index), scale3_values, result);
        _mm512_storeu_ps(output + index, result);
    }
    for (; index < hidden_size; ++index)
        output[index] = input[index] * scale0
                        + input1[index] * scale1
                        + input2[index] * scale2
                        + input3[index] * scale3;
}

void avx512_float_hc_post_4(float* output,
                            const float* branch,
                            const float* residual,
                            const float* post,
                            const float* combine,
                            uint32_t hidden_size) noexcept
{
    const float* residual1 = residual + hidden_size;
    const float* residual2 = residual1 + hidden_size;
    const float* residual3 = residual2 + hidden_size;
    const __m512 post_values[4] = {
        _mm512_set1_ps(post[0]), _mm512_set1_ps(post[1]),
        _mm512_set1_ps(post[2]), _mm512_set1_ps(post[3])};
    const __m512 combine_values[4][4] = {
        {_mm512_set1_ps(combine[0]), _mm512_set1_ps(combine[1]),
         _mm512_set1_ps(combine[2]), _mm512_set1_ps(combine[3])},
        {_mm512_set1_ps(combine[4]), _mm512_set1_ps(combine[5]),
         _mm512_set1_ps(combine[6]), _mm512_set1_ps(combine[7])},
        {_mm512_set1_ps(combine[8]), _mm512_set1_ps(combine[9]),
         _mm512_set1_ps(combine[10]), _mm512_set1_ps(combine[11])},
        {_mm512_set1_ps(combine[12]), _mm512_set1_ps(combine[13]),
         _mm512_set1_ps(combine[14]), _mm512_set1_ps(combine[15])}};
    uint32_t index = 0;
    for (; index + 16 <= hidden_size; index += 16)
    {
        const __m512 branch_values = _mm512_loadu_ps(branch + index);
        const __m512 residual_values[4] = {
            _mm512_loadu_ps(residual + index), _mm512_loadu_ps(residual1 + index),
            _mm512_loadu_ps(residual2 + index), _mm512_loadu_ps(residual3 + index)};
        for (uint32_t output_index = 0; output_index < 4; ++output_index)
        {
            __m512 result = _mm512_mul_ps(branch_values, post_values[output_index]);
            result = _mm512_fmadd_ps(residual_values[0], combine_values[0][output_index], result);
            result = _mm512_fmadd_ps(residual_values[1], combine_values[1][output_index], result);
            result = _mm512_fmadd_ps(residual_values[2], combine_values[2][output_index], result);
            result = _mm512_fmadd_ps(residual_values[3], combine_values[3][output_index], result);
            _mm512_storeu_ps(output + static_cast<size_t>(output_index) * hidden_size + index, result);
        }
    }
    for (; index < hidden_size; ++index)
    {
        for (uint32_t output_index = 0; output_index < 4; ++output_index)
        {
            float result = branch[index] * post[output_index];
            result += residual[index] * combine[output_index];
            result += residual1[index] * combine[4 + output_index];
            result += residual2[index] * combine[8 + output_index];
            result += residual3[index] * combine[12 + output_index];
            output[static_cast<size_t>(output_index) * hidden_size + index] = result;
        }
    }
}

void avx512_float_gemm_4x4(const float* weights,
                           size_t weight_stride,
                           const float* input,
                           size_t input_stride,
                           uint32_t input_columns,
                           uint32_t output_count,
                           uint32_t token_count,
                           float* output,
                           size_t output_stride) noexcept
{
    __m512 accumulators[4][4] = {};
    uint32_t column = 0;
    for (; column + 16 <= input_columns; column += 16)
    {
        __m512 input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
        {
            input_values[token] = _mm512_loadu_ps(input + static_cast<size_t>(token) * input_stride + column);
        }
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const __m512 weight_values = _mm512_loadu_ps(weights + static_cast<size_t>(output_index) * weight_stride + column);
            for (uint32_t token = 0; token < token_count; ++token)
            {
                accumulators[token][output_index] = _mm512_fmadd_ps(input_values[token],
                                                                    weight_values,
                                                                    accumulators[token][output_index]);
            }
        }
    }

    float results[4][4] = {};
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            results[token][output_index] = _mm512_reduce_add_ps(accumulators[token][output_index]);

    for (; column < input_columns; ++column)
    {
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const float value = input[static_cast<size_t>(token) * input_stride + column];
            for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            {
                results[token][output_index] += value
                                                * weights[static_cast<size_t>(output_index) * weight_stride + column];
            }
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = results[token][output_index];
}

void avx512_float_gemm_4x8(const float* weights,
                           size_t weight_stride,
                           const float* input,
                           size_t input_stride,
                           uint32_t input_columns,
                           uint32_t output_count,
                           uint32_t token_count,
                           float* output,
                           size_t output_stride) noexcept
{
    __m512 accumulators[4][8] = {};
    uint32_t column = 0;
    for (; column + 16 <= input_columns; column += 16)
    {
        __m512 input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
            input_values[token] = _mm512_loadu_ps(input + static_cast<size_t>(token) * input_stride + column);
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const __m512 weight_values = _mm512_loadu_ps(weights + static_cast<size_t>(output_index) * weight_stride + column);
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] = _mm512_fmadd_ps(input_values[token], weight_values,
                                                                    accumulators[token][output_index]);
        }
    }

    float results[4][8] = {};
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            results[token][output_index] = _mm512_reduce_add_ps(accumulators[token][output_index]);

    for (; column < input_columns; ++column)
    {
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const float value = input[static_cast<size_t>(token) * input_stride + column];
            for (uint32_t output_index = 0; output_index < output_count; ++output_index)
                results[token][output_index] += value
                                                * weights[static_cast<size_t>(output_index) * weight_stride + column];
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = results[token][output_index];
}

void avx512_bfloat16_gemm_4x8(const uint16_t* weights,
                              size_t weight_stride,
                              const float* input,
                              size_t input_stride,
                              uint32_t input_columns,
                              uint32_t output_count,
                              uint32_t token_count,
                              float* output,
                              size_t output_stride) noexcept
{
    __m512 accumulators[4][8] = {};
    uint32_t column = 0;
    for (; column + 16 <= input_columns; column += 16)
    {
        __m512 input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
            input_values[token] = _mm512_loadu_ps(input + static_cast<size_t>(token) * input_stride + column);
        __m512 weight_values[8] = {};
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + static_cast<size_t>(output_index) * weight_stride + column));
            weight_values[output_index] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] = _mm512_fmadd_ps(input_values[token], weight_values[output_index],
                                                                    accumulators[token][output_index]);
        }
    }

    float results[4][8] = {};
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            results[token][output_index] = _mm512_reduce_add_ps(accumulators[token][output_index]);

    for (; column < input_columns; ++column)
    {
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const float value = input[static_cast<size_t>(token) * input_stride + column];
            for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            {
                const float weight = std::bit_cast<float>(static_cast<uint32_t>(weights[static_cast<size_t>(output_index) * weight_stride + column])
                                                          << 16);
                results[token][output_index] += value * weight;
            }
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = results[token][output_index];
}

void avx512_float_exp_inplace(float* values, uint32_t count) noexcept
{
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
        _mm512_storeu_ps(values + index, avx512_expf(_mm512_loadu_ps(values + index)));
    for (; index < count; ++index)
        values[index] = float_approximate_exp(values[index]);
}

float avx512_int8_float_dot(const int8_t* left,
                            const float* right,
                            uint32_t count) noexcept
{
    __m512 accumulator0 = _mm512_setzero_ps();
    __m512 accumulator1 = _mm512_setzero_ps();
    __m512 accumulator2 = _mm512_setzero_ps();
    __m512 accumulator3 = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 64 <= count; index += 64)
    {
        accumulator0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index)))),
                                       _mm512_loadu_ps(right + index),
                                       accumulator0);
        accumulator1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index + 16)))),
                                       _mm512_loadu_ps(right + index + 16),
                                       accumulator1);
        accumulator2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index + 32)))),
                                       _mm512_loadu_ps(right + index + 32),
                                       accumulator2);
        accumulator3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index + 48)))),
                                       _mm512_loadu_ps(right + index + 48),
                                       accumulator3);
    }
    __m512 sum = _mm512_add_ps(_mm512_add_ps(accumulator0, accumulator1),
                               _mm512_add_ps(accumulator2, accumulator3));
    float result = _mm512_reduce_add_ps(sum);
    for (; index + 16 <= count; index += 16)
    {
        result += _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index)))),
                                                     _mm512_loadu_ps(right + index)));
    }
    for (; index < count; ++index)
        result += static_cast<float>(left[index]) * right[index];
    return result;
}

void avx512_float_scale_inplace(float* values, float scale, uint32_t count) noexcept
{
    const __m512 scale_values = _mm512_set1_ps(scale);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
        _mm512_storeu_ps(values + index, _mm512_mul_ps(_mm512_loadu_ps(values + index), scale_values));
    for (; index < count; ++index)
        values[index] *= scale;
}

void avx512_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
{
    const __m512 scale_values = _mm512_set1_ps(scale);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 product = _mm512_mul_ps(scale_values, _mm512_loadu_ps(input + index));
        const __m512 sum = _mm512_add_ps(_mm512_loadu_ps(output + index), product);
        _mm512_storeu_ps(output + index, sum);
    }
    for (; index < count; ++index)
        output[index] += scale * input[index];
}

void avx512_float_scale_add(float* output,
                            float output_scale,
                            const float* input,
                            float input_scale,
                            uint32_t count) noexcept
{
    const __m512 output_scale_values = _mm512_set1_ps(output_scale);
    const __m512 input_scale_values = _mm512_set1_ps(input_scale);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 scaled_output = _mm512_mul_ps(_mm512_loadu_ps(output + index), output_scale_values);
        const __m512 scaled_input = _mm512_mul_ps(_mm512_loadu_ps(input + index), input_scale_values);
        _mm512_storeu_ps(output + index, _mm512_add_ps(scaled_output, scaled_input));
    }
    for (; index < count; ++index)
        output[index] = output[index] * output_scale + input[index] * input_scale;
}

void avx512_float_scale_inplace_and_scaled_add(float* values,
                                               float value_scale,
                                               float* output,
                                               float output_scale,
                                               uint32_t count) noexcept
{
    const __m512 value_scale_values = _mm512_set1_ps(value_scale);
    const __m512 output_scale_values = _mm512_set1_ps(output_scale);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 scaled_values = _mm512_mul_ps(_mm512_loadu_ps(values + index), value_scale_values);
        _mm512_storeu_ps(values + index, scaled_values);
        const __m512 product = _mm512_mul_ps(scaled_values, output_scale_values);
        _mm512_storeu_ps(output + index,
                         _mm512_add_ps(_mm512_loadu_ps(output + index), product));
    }
    for (; index < count; ++index)
    {
        values[index] *= value_scale;
        output[index] += output_scale * values[index];
    }
}

void avx512_float_scaled_add_and_accumulate(float* values,
                                            const float* input,
                                            float input_scale,
                                            float* output,
                                            float output_scale,
                                            uint32_t count) noexcept
{
    const __m512 input_scale_values = _mm512_set1_ps(input_scale);
    const __m512 output_scale_values = _mm512_set1_ps(output_scale);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 input_values = _mm512_mul_ps(_mm512_loadu_ps(input + index), input_scale_values);
        const __m512 updated_values = _mm512_add_ps(_mm512_loadu_ps(values + index), input_values);
        _mm512_storeu_ps(values + index, updated_values);
        _mm512_storeu_ps(output + index,
                         _mm512_add_ps(_mm512_loadu_ps(output + index),
                                       _mm512_mul_ps(updated_values, output_scale_values)));
    }
    for (; index < count; ++index)
    {
        const float product = input[index] * input_scale;
        values[index] += product;
        output[index] += output_scale * values[index];
    }
}

void avx512_float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    const __m512 scale_values = _mm512_set1_ps(scale);
    const __m512 offset_values = _mm512_set1_ps(weight_offset);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 scaled = _mm512_mul_ps(_mm512_loadu_ps(input + index), scale_values);
        const __m512 weighted = _mm512_add_ps(_mm512_loadu_ps(weight + index), offset_values);
        _mm512_storeu_ps(output + index, _mm512_mul_ps(scaled, weighted));
    }
    for (; index < count; ++index)
        output[index] = input[index] * scale * (weight[index] + weight_offset);
}

void avx512_bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    const __m512 scale_values = _mm512_set1_ps(scale);
    const __m512 offset_values = _mm512_set1_ps(weight_offset);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        __m512i weight_bits = _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight + index)));
        weight_bits = _mm512_slli_epi32(weight_bits, 16);
        const __m512 scaled = _mm512_mul_ps(_mm512_loadu_ps(input + index), scale_values);
        const __m512 weighted = _mm512_add_ps(_mm512_castsi512_ps(weight_bits), offset_values);
        _mm512_storeu_ps(output + index, _mm512_mul_ps(scaled, weighted));
    }
    for (; index < count; ++index)
    {
        const float value = std::bit_cast<float>(static_cast<uint32_t>(weight[index]) << 16);
        output[index] = input[index] * scale * (value + weight_offset);
    }
}

size_t avx512_float_argmax(const float* values, size_t count) noexcept
{
    if (count == 0)
        return 0;

    const __m512 negative_infinity = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    __m512 maxima = negative_infinity;
    size_t index = 0;
    for (; count - index >= 16; index += 16)
    {
        const __m512 current = _mm512_loadu_ps(values + index);
        const __m512i exponent = _mm512_and_si512(_mm512_castps_si512(current),
                                                  _mm512_set1_epi32(0x7f800000));
        const __mmask16 nonfinite = _mm512_cmpeq_epi32_mask(exponent,
                                                            _mm512_set1_epi32(0x7f800000));
        const __mmask16 finite = static_cast<__mmask16>(~nonfinite);
        maxima = _mm512_max_ps(maxima,
                               _mm512_mask_mov_ps(negative_infinity, finite, current));
    }

    float maximum = _mm512_reduce_max_ps(maxima);
    for (; index < count; ++index)
    {
        const float current = values[index];
        if (std::isfinite(current) && current > maximum)
            maximum = current;
    }
    if (maximum == -std::numeric_limits<float>::infinity())
        return count;

    const __m512 target = _mm512_set1_ps(maximum);
    index = 0;
    for (; count - index >= 16; index += 16)
    {
        const __m512 current = _mm512_loadu_ps(values + index);
        const unsigned lanes_equal = static_cast<unsigned>(_mm512_cmp_ps_mask(current, target, _CMP_EQ_OQ));
        if (lanes_equal != 0)
            return index + std::countr_zero(lanes_equal);
    }
    for (; index < count; ++index)
        if (values[index] == maximum)
            return index;
    return count;
}

void avx512_gdn_step(float* state,
                     const float* key,
                     const float* query,
                     const float* value,
                     uint32_t key_dim,
                     uint32_t value_dim,
                     float decay,
                     float beta,
                     float scale,
                     float* memory,
                     float* delta,
                     float* output) noexcept
{
    const __m512 decay_values = _mm512_set1_ps(decay);
    const __m512 beta_values = _mm512_set1_ps(beta);
    const __m512 scale_values = _mm512_set1_ps(scale);
    const __m512 zero = _mm512_setzero_ps();

    if (value_dim == 128)
    {
        __m512 memory_regs[8];
        __m512 delta_regs[8];
        __m512 output_regs[8];
        memory_regs[0] = zero;
        memory_regs[1] = zero;
        memory_regs[2] = zero;
        memory_regs[3] = zero;
        memory_regs[4] = zero;
        memory_regs[5] = zero;
        memory_regs[6] = zero;
        memory_regs[7] = zero;
        for (uint32_t row = 0; row < key_dim; ++row)
        {
            const float* state_row = state + static_cast<size_t>(row) * 128;
            const __m512 key_values = _mm512_set1_ps(key[row]);
            const __m512 decayed0 = _mm512_mul_ps(_mm512_loadu_ps(state_row + 0), decay_values);
            memory_regs[0] = _mm512_fmadd_ps(decayed0, key_values, memory_regs[0]);
            const __m512 decayed1 = _mm512_mul_ps(_mm512_loadu_ps(state_row + 16), decay_values);
            memory_regs[1] = _mm512_fmadd_ps(decayed1, key_values, memory_regs[1]);
            const __m512 decayed2 = _mm512_mul_ps(_mm512_loadu_ps(state_row + 32), decay_values);
            memory_regs[2] = _mm512_fmadd_ps(decayed2, key_values, memory_regs[2]);
            const __m512 decayed3 = _mm512_mul_ps(_mm512_loadu_ps(state_row + 48), decay_values);
            memory_regs[3] = _mm512_fmadd_ps(decayed3, key_values, memory_regs[3]);
            const __m512 decayed4 = _mm512_mul_ps(_mm512_loadu_ps(state_row + 64), decay_values);
            memory_regs[4] = _mm512_fmadd_ps(decayed4, key_values, memory_regs[4]);
            const __m512 decayed5 = _mm512_mul_ps(_mm512_loadu_ps(state_row + 80), decay_values);
            memory_regs[5] = _mm512_fmadd_ps(decayed5, key_values, memory_regs[5]);
            const __m512 decayed6 = _mm512_mul_ps(_mm512_loadu_ps(state_row + 96), decay_values);
            memory_regs[6] = _mm512_fmadd_ps(decayed6, key_values, memory_regs[6]);
            const __m512 decayed7 = _mm512_mul_ps(_mm512_loadu_ps(state_row + 112), decay_values);
            memory_regs[7] = _mm512_fmadd_ps(decayed7, key_values, memory_regs[7]);
        }
        delta_regs[0] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(value + 0), memory_regs[0]), beta_values);
        output_regs[0] = zero;
        delta_regs[1] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(value + 16), memory_regs[1]), beta_values);
        output_regs[1] = zero;
        delta_regs[2] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(value + 32), memory_regs[2]), beta_values);
        output_regs[2] = zero;
        delta_regs[3] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(value + 48), memory_regs[3]), beta_values);
        output_regs[3] = zero;
        delta_regs[4] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(value + 64), memory_regs[4]), beta_values);
        output_regs[4] = zero;
        delta_regs[5] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(value + 80), memory_regs[5]), beta_values);
        output_regs[5] = zero;
        delta_regs[6] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(value + 96), memory_regs[6]), beta_values);
        output_regs[6] = zero;
        delta_regs[7] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(value + 112), memory_regs[7]), beta_values);
        output_regs[7] = zero;
        for (uint32_t row = 0; row < key_dim; ++row)
        {
            float* state_row = state + static_cast<size_t>(row) * 128;
            const __m512 key_values = _mm512_set1_ps(key[row]);
            const __m512 query_values = _mm512_set1_ps(query[row]);
            const __m512 updated0 = _mm512_fmadd_ps(key_values, delta_regs[0],
                                                    _mm512_mul_ps(_mm512_loadu_ps(state_row + 0), decay_values));
            _mm512_storeu_ps(state_row + 0, updated0);
            output_regs[0] = _mm512_fmadd_ps(updated0, query_values, output_regs[0]);
            const __m512 updated1 = _mm512_fmadd_ps(key_values, delta_regs[1],
                                                    _mm512_mul_ps(_mm512_loadu_ps(state_row + 16), decay_values));
            _mm512_storeu_ps(state_row + 16, updated1);
            output_regs[1] = _mm512_fmadd_ps(updated1, query_values, output_regs[1]);
            const __m512 updated2 = _mm512_fmadd_ps(key_values, delta_regs[2],
                                                    _mm512_mul_ps(_mm512_loadu_ps(state_row + 32), decay_values));
            _mm512_storeu_ps(state_row + 32, updated2);
            output_regs[2] = _mm512_fmadd_ps(updated2, query_values, output_regs[2]);
            const __m512 updated3 = _mm512_fmadd_ps(key_values, delta_regs[3],
                                                    _mm512_mul_ps(_mm512_loadu_ps(state_row + 48), decay_values));
            _mm512_storeu_ps(state_row + 48, updated3);
            output_regs[3] = _mm512_fmadd_ps(updated3, query_values, output_regs[3]);
            const __m512 updated4 = _mm512_fmadd_ps(key_values, delta_regs[4],
                                                    _mm512_mul_ps(_mm512_loadu_ps(state_row + 64), decay_values));
            _mm512_storeu_ps(state_row + 64, updated4);
            output_regs[4] = _mm512_fmadd_ps(updated4, query_values, output_regs[4]);
            const __m512 updated5 = _mm512_fmadd_ps(key_values, delta_regs[5],
                                                    _mm512_mul_ps(_mm512_loadu_ps(state_row + 80), decay_values));
            _mm512_storeu_ps(state_row + 80, updated5);
            output_regs[5] = _mm512_fmadd_ps(updated5, query_values, output_regs[5]);
            const __m512 updated6 = _mm512_fmadd_ps(key_values, delta_regs[6],
                                                    _mm512_mul_ps(_mm512_loadu_ps(state_row + 96), decay_values));
            _mm512_storeu_ps(state_row + 96, updated6);
            output_regs[6] = _mm512_fmadd_ps(updated6, query_values, output_regs[6]);
            const __m512 updated7 = _mm512_fmadd_ps(key_values, delta_regs[7],
                                                    _mm512_mul_ps(_mm512_loadu_ps(state_row + 112), decay_values));
            _mm512_storeu_ps(state_row + 112, updated7);
            output_regs[7] = _mm512_fmadd_ps(updated7, query_values, output_regs[7]);
        }
        _mm512_storeu_ps(output + 0, _mm512_mul_ps(output_regs[0], scale_values));
        _mm512_storeu_ps(output + 16, _mm512_mul_ps(output_regs[1], scale_values));
        _mm512_storeu_ps(output + 32, _mm512_mul_ps(output_regs[2], scale_values));
        _mm512_storeu_ps(output + 48, _mm512_mul_ps(output_regs[3], scale_values));
        _mm512_storeu_ps(output + 64, _mm512_mul_ps(output_regs[4], scale_values));
        _mm512_storeu_ps(output + 80, _mm512_mul_ps(output_regs[5], scale_values));
        _mm512_storeu_ps(output + 96, _mm512_mul_ps(output_regs[6], scale_values));
        _mm512_storeu_ps(output + 112, _mm512_mul_ps(output_regs[7], scale_values));
        return;
    }

    uint32_t column = 0;
    for (; column + 16 <= value_dim; column += 16)
        _mm512_storeu_ps(memory + column, zero);
    for (; column < value_dim; ++column)
        memory[column] = 0.0f;

    // Pass 1: decay the state into the memory vector.  The state is only read
    // here; the updated value is recomputed in pass 2 and written once.
    for (uint32_t row = 0; row < key_dim; ++row)
    {
        const float* state_row = state + static_cast<size_t>(row) * value_dim;
        const __m512 key_values = _mm512_set1_ps(key[row]);
        column = 0;
        for (; column + 16 <= value_dim; column += 16)
        {
            const __m512 decayed = _mm512_mul_ps(_mm512_loadu_ps(state_row + column), decay_values);
            _mm512_storeu_ps(memory + column,
                             _mm512_fmadd_ps(decayed, key_values, _mm512_loadu_ps(memory + column)));
        }
        for (; column < value_dim; ++column)
            memory[column] += state_row[column] * decay * key[row];
    }

    column = 0;
    for (; column + 16 <= value_dim; column += 16)
        _mm512_storeu_ps(delta + column,
                         _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(value + column),
                                                     _mm512_loadu_ps(memory + column)),
                                       beta_values));
    for (; column < value_dim; ++column)
        delta[column] = (value[column] - memory[column]) * beta;

    column = 0;
    for (; column + 16 <= value_dim; column += 16)
        _mm512_storeu_ps(output + column, zero);
    for (; column < value_dim; ++column)
        output[column] = 0.0f;

    // Pass 2: apply the rank-one update in place and project it onto the
    // query vector.  Fusing both passes keeps the 16K-element state resident
    // while dropping the per-key vector helper call overhead.
    for (uint32_t row = 0; row < key_dim; ++row)
    {
        float* state_row = state + static_cast<size_t>(row) * value_dim;
        const __m512 key_values = _mm512_set1_ps(key[row]);
        const __m512 query_values = _mm512_set1_ps(query[row]);
        column = 0;
        for (; column + 16 <= value_dim; column += 16)
        {
            const __m512 updated = _mm512_fmadd_ps(key_values,
                                                   _mm512_loadu_ps(delta + column),
                                                   _mm512_mul_ps(_mm512_loadu_ps(state_row + column),
                                                                 decay_values));
            _mm512_storeu_ps(state_row + column, updated);
            _mm512_storeu_ps(output + column,
                             _mm512_fmadd_ps(updated,
                                             query_values,
                                             _mm512_loadu_ps(output + column)));
        }
        for (; column < value_dim; ++column)
        {
            const float updated = state_row[column] * decay + key[row] * delta[column];
            state_row[column] = updated;
            output[column] += updated * query[row];
        }
    }

    column = 0;
    for (; column + 16 <= value_dim; column += 16)
        _mm512_storeu_ps(output + column, _mm512_mul_ps(_mm512_loadu_ps(output + column), scale_values));
    for (; column < value_dim; ++column)
        output[column] *= scale;
}

void avx512_gdn_conv4(float* state,
                      float* values,
                      size_t value_stride,
                      size_t first,
                      size_t row_count,
                      uint32_t begin,
                      uint32_t end,
                      uint32_t columns,
                      const float* taps) noexcept
{
    const float* tap0 = taps;
    const float* tap1 = taps + columns;
    const float* tap2 = taps + static_cast<size_t>(columns) * 2;
    const float* tap3 = taps + static_cast<size_t>(columns) * 3;
    for (size_t token = 0; token < row_count; ++token)
    {
        float* row = values + (first + token) * value_stride;
        uint32_t channel = begin;
        for (; channel + 16 <= end; channel += 16)
        {
            const __m512 x = _mm512_loadu_ps(row + channel);
            const __m512 h1 = _mm512_loadu_ps(state + columns + channel);
            const __m512 h2 = _mm512_loadu_ps(state + static_cast<size_t>(columns) * 2 + channel);
            const __m512 h3 = _mm512_loadu_ps(state + static_cast<size_t>(columns) * 3 + channel);
            __m512 sum = _mm512_mul_ps(h1, _mm512_loadu_ps(tap0 + channel));
            sum = _mm512_fmadd_ps(h2, _mm512_loadu_ps(tap1 + channel), sum);
            sum = _mm512_fmadd_ps(h3, _mm512_loadu_ps(tap2 + channel), sum);
            sum = _mm512_fmadd_ps(x, _mm512_loadu_ps(tap3 + channel), sum);
            _mm512_storeu_ps(state + channel, h1);
            _mm512_storeu_ps(state + columns + channel, h2);
            _mm512_storeu_ps(state + static_cast<size_t>(columns) * 2 + channel, h3);
            _mm512_storeu_ps(state + static_cast<size_t>(columns) * 3 + channel, x);
            _mm512_storeu_ps(row + channel, sum);
        }
        for (; channel < end; ++channel)
        {
            const float x = row[channel];
            const float h1 = state[columns + channel];
            const float h2 = state[static_cast<size_t>(columns) * 2 + channel];
            const float h3 = state[static_cast<size_t>(columns) * 3 + channel];
            row[channel] = h1 * tap0[channel]
                           + h2 * tap1[channel]
                           + h3 * tap2[channel]
                           + x * tap3[channel];
            state[channel] = h1;
            state[columns + channel] = h2;
            state[static_cast<size_t>(columns) * 2 + channel] = h3;
            state[static_cast<size_t>(columns) * 3 + channel] = x;
        }
    }
}

void avx512_pv_f32(float* output,
                   const float* values0,
                   size_t value_stride,
                   const float* weights0,
                   uint32_t count0,
                   const float* values1,
                   const float* weights1,
                   uint32_t count1,
                   uint32_t columns) noexcept
{
    uint32_t column = 0;
    for (; column + 256 <= columns; column += 256)
    {
        __m512 a0 = _mm512_setzero_ps();
        __m512 a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps();
        __m512 a3 = _mm512_setzero_ps();
        __m512 a4 = _mm512_setzero_ps();
        __m512 a5 = _mm512_setzero_ps();
        __m512 a6 = _mm512_setzero_ps();
        __m512 a7 = _mm512_setzero_ps();
        __m512 a8 = _mm512_setzero_ps();
        __m512 a9 = _mm512_setzero_ps();
        __m512 a10 = _mm512_setzero_ps();
        __m512 a11 = _mm512_setzero_ps();
        __m512 a12 = _mm512_setzero_ps();
        __m512 a13 = _mm512_setzero_ps();
        __m512 a14 = _mm512_setzero_ps();
        __m512 a15 = _mm512_setzero_ps();
        const float* row = values0 + column;
        for (uint32_t key = 0; key < count0; ++key)
        {
            const __m512 weight = _mm512_set1_ps(weights0[key]);
            a0 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 0), a0);
            a1 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 16), a1);
            a2 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 32), a2);
            a3 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 48), a3);
            a4 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 64), a4);
            a5 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 80), a5);
            a6 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 96), a6);
            a7 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 112), a7);
            a8 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 128), a8);
            a9 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 144), a9);
            a10 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 160), a10);
            a11 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 176), a11);
            a12 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 192), a12);
            a13 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 208), a13);
            a14 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 224), a14);
            a15 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 240), a15);
            row += value_stride;
        }
        row = values1 + column;
        for (uint32_t key = 0; key < count1; ++key)
        {
            const __m512 weight = _mm512_set1_ps(weights1[key]);
            a0 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 0), a0);
            a1 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 16), a1);
            a2 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 32), a2);
            a3 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 48), a3);
            a4 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 64), a4);
            a5 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 80), a5);
            a6 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 96), a6);
            a7 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 112), a7);
            a8 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 128), a8);
            a9 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 144), a9);
            a10 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 160), a10);
            a11 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 176), a11);
            a12 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 192), a12);
            a13 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 208), a13);
            a14 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 224), a14);
            a15 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 240), a15);
            row += value_stride;
        }
        _mm512_storeu_ps(output + column + 0, _mm512_add_ps(_mm512_loadu_ps(output + column + 0), a0));
        _mm512_storeu_ps(output + column + 16, _mm512_add_ps(_mm512_loadu_ps(output + column + 16), a1));
        _mm512_storeu_ps(output + column + 32, _mm512_add_ps(_mm512_loadu_ps(output + column + 32), a2));
        _mm512_storeu_ps(output + column + 48, _mm512_add_ps(_mm512_loadu_ps(output + column + 48), a3));
        _mm512_storeu_ps(output + column + 64, _mm512_add_ps(_mm512_loadu_ps(output + column + 64), a4));
        _mm512_storeu_ps(output + column + 80, _mm512_add_ps(_mm512_loadu_ps(output + column + 80), a5));
        _mm512_storeu_ps(output + column + 96, _mm512_add_ps(_mm512_loadu_ps(output + column + 96), a6));
        _mm512_storeu_ps(output + column + 112, _mm512_add_ps(_mm512_loadu_ps(output + column + 112), a7));
        _mm512_storeu_ps(output + column + 128, _mm512_add_ps(_mm512_loadu_ps(output + column + 128), a8));
        _mm512_storeu_ps(output + column + 144, _mm512_add_ps(_mm512_loadu_ps(output + column + 144), a9));
        _mm512_storeu_ps(output + column + 160, _mm512_add_ps(_mm512_loadu_ps(output + column + 160), a10));
        _mm512_storeu_ps(output + column + 176, _mm512_add_ps(_mm512_loadu_ps(output + column + 176), a11));
        _mm512_storeu_ps(output + column + 192, _mm512_add_ps(_mm512_loadu_ps(output + column + 192), a12));
        _mm512_storeu_ps(output + column + 208, _mm512_add_ps(_mm512_loadu_ps(output + column + 208), a13));
        _mm512_storeu_ps(output + column + 224, _mm512_add_ps(_mm512_loadu_ps(output + column + 224), a14));
        _mm512_storeu_ps(output + column + 240, _mm512_add_ps(_mm512_loadu_ps(output + column + 240), a15));
    }
    for (; column + 128 <= columns; column += 128)
    {
        __m512 a0 = _mm512_setzero_ps();
        __m512 a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps();
        __m512 a3 = _mm512_setzero_ps();
        __m512 a4 = _mm512_setzero_ps();
        __m512 a5 = _mm512_setzero_ps();
        __m512 a6 = _mm512_setzero_ps();
        __m512 a7 = _mm512_setzero_ps();
        const float* row = values0 + column;
        for (uint32_t key = 0; key < count0; ++key)
        {
            const __m512 weight = _mm512_set1_ps(weights0[key]);
            a0 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 0), a0);
            a1 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 16), a1);
            a2 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 32), a2);
            a3 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 48), a3);
            a4 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 64), a4);
            a5 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 80), a5);
            a6 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 96), a6);
            a7 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 112), a7);
            row += value_stride;
        }
        row = values1 + column;
        for (uint32_t key = 0; key < count1; ++key)
        {
            const __m512 weight = _mm512_set1_ps(weights1[key]);
            a0 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 0), a0);
            a1 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 16), a1);
            a2 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 32), a2);
            a3 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 48), a3);
            a4 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 64), a4);
            a5 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 80), a5);
            a6 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 96), a6);
            a7 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(row + 112), a7);
            row += value_stride;
        }
        _mm512_storeu_ps(output + column + 0, _mm512_add_ps(_mm512_loadu_ps(output + column + 0), a0));
        _mm512_storeu_ps(output + column + 16, _mm512_add_ps(_mm512_loadu_ps(output + column + 16), a1));
        _mm512_storeu_ps(output + column + 32, _mm512_add_ps(_mm512_loadu_ps(output + column + 32), a2));
        _mm512_storeu_ps(output + column + 48, _mm512_add_ps(_mm512_loadu_ps(output + column + 48), a3));
        _mm512_storeu_ps(output + column + 64, _mm512_add_ps(_mm512_loadu_ps(output + column + 64), a4));
        _mm512_storeu_ps(output + column + 80, _mm512_add_ps(_mm512_loadu_ps(output + column + 80), a5));
        _mm512_storeu_ps(output + column + 96, _mm512_add_ps(_mm512_loadu_ps(output + column + 96), a6));
        _mm512_storeu_ps(output + column + 112, _mm512_add_ps(_mm512_loadu_ps(output + column + 112), a7));
    }
    for (; column + 16 <= columns; column += 16)
    {
        __m512 accumulator = _mm512_setzero_ps();
        const float* row = values0 + column;
        for (uint32_t key = 0; key < count0; ++key)
        {
            accumulator = _mm512_fmadd_ps(_mm512_set1_ps(weights0[key]), _mm512_loadu_ps(row), accumulator);
            row += value_stride;
        }
        row = values1 + column;
        for (uint32_t key = 0; key < count1; ++key)
        {
            accumulator = _mm512_fmadd_ps(_mm512_set1_ps(weights1[key]), _mm512_loadu_ps(row), accumulator);
            row += value_stride;
        }
        _mm512_storeu_ps(output + column, _mm512_add_ps(_mm512_loadu_ps(output + column), accumulator));
    }
    for (; column < columns; ++column)
    {
        const float* row = values0 + column;
        float sum = 0.0f;
        for (uint32_t key = 0; key < count0; ++key)
        {
            sum += weights0[key] * row[0];
            row += value_stride;
        }
        row = values1 + column;
        for (uint32_t key = 0; key < count1; ++key)
        {
            sum += weights1[key] * row[0];
            row += value_stride;
        }
        output[column] += sum;
    }
}

} // namespace moe
} // namespace ncnn

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // x86
