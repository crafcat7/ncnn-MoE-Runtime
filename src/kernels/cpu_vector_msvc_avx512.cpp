#include "cpu_vector_msvc.h"

#include "cpu_fast_math.h"
#include <bit>
#include <cmath>
#include <immintrin.h>

namespace ncnn {
namespace moe {

static __m512 msvc_avx512_expf(__m512 value) noexcept
{
    const __m512 rounding = _mm512_set1_ps(0x1.8p23f);
    const __m512 scaled = _mm512_fmadd_ps(value, _mm512_set1_ps(0x1.715476p+0f), rounding);
    const __m512 exponent = _mm512_sub_ps(scaled, rounding);
    const __m512 remainder = _mm512_fnmadd_ps(
        exponent, _mm512_set1_ps(0x1.7f7d1cp-20f),
        _mm512_fnmadd_ps(exponent, _mm512_set1_ps(0x1.62e4p-1f), value));
    const __m512i exponent_bits = _mm512_slli_epi32(_mm512_castps_si512(scaled), 23);
    const __m512 power = _mm512_castsi512_ps(
        _mm512_add_epi32(exponent_bits, _mm512_castps_si512(_mm512_set1_ps(1.0f))));
    const __mmask16 large = _mm512_cmp_ps_mask(
        _mm512_abs_ps(exponent), _mm512_set1_ps(126.0f), _CMP_GT_OQ);
    const __m512 remainder_squared = _mm512_mul_ps(remainder, remainder);
    const __m512 polynomial = _mm512_fmadd_ps(
        _mm512_fmadd_ps(
            _mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), remainder,
                            _mm512_set1_ps(0x1.573e2ep-5f)),
            remainder_squared,
            _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), remainder,
                            _mm512_set1_ps(0x1.fffdb6p-2f))),
        remainder_squared,
        _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), remainder,
                        _mm512_set1_ps(1.0f)));
    const __m512 result = _mm512_scalef_ps(polynomial, exponent);
    if (_mm512_kortestz(large, large))
        return result;

    const __m512 zero = _mm512_setzero_ps();
    const __m512 alternate = _mm512_mask_blend_ps(
        _mm512_cmp_ps_mask(exponent, zero, _CMP_LE_OQ),
        _mm512_set1_ps(INFINITY), zero);
    return _mm512_mask_blend_ps(large, result, alternate);
}

void msvc_avx512_float_sigmoid_mul(
    float* output,
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
        const __m512 sigmoid = _mm512_div_ps(
            one,
            _mm512_add_ps(one, msvc_avx512_expf(_mm512_sub_ps(zero, gate_values))));
        _mm512_storeu_ps(output + index,
                         _mm512_mul_ps(sigmoid, _mm512_loadu_ps(input + index)));
    }
    for (; index < count; ++index)
        output[index] = input[index] / (1.0f + float_approximate_exp(-gate[index]));
}

void msvc_avx512_float_silu_mul(float* output, const float* gate, const float* up,
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
        const __m512 silu = _mm512_div_ps(
            gate_values,
            _mm512_add_ps(one, msvc_avx512_expf(_mm512_sub_ps(
                                   zero, _mm512_mul_ps(scale, gate_values)))));
        _mm512_storeu_ps(output + index,
                         _mm512_mul_ps(silu, _mm512_add_ps(_mm512_loadu_ps(up + index), offset)));
    }
    for (; index < count; ++index)
    {
        const float gate_value = gate[index];
        output[index] = gate_value / (1.0f + float_approximate_exp(-sigmoid_scale * gate_value))
                        * (up[index] + up_offset);
    }
}

void msvc_avx512_float_silu_inplace(float* values, uint32_t count) noexcept
{
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 input = _mm512_loadu_ps(values + index);
        const __m512 silu = _mm512_div_ps(
            input,
            _mm512_add_ps(one, msvc_avx512_expf(_mm512_sub_ps(zero, input))));
        _mm512_storeu_ps(values + index, silu);
    }
    for (; index < count; ++index)
    {
        const float value = values[index];
        values[index] = value / (1.0f + float_approximate_exp(-value));
    }
}

float msvc_avx512_float_dot(const float* left, const float* right, uint32_t count) noexcept
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

void msvc_avx512_float_l2_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = msvc_avx512_float_dot(values, values, count);
    const float inverse_norm = 1.0f / std::sqrt(square_sum + epsilon);
    const __m512 scale = _mm512_set1_ps(inverse_norm);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
        _mm512_storeu_ps(values + index, _mm512_mul_ps(_mm512_loadu_ps(values + index), scale));
    for (; index < count; ++index)
        values[index] *= inverse_norm;
}

void msvc_avx512_float_rms_scale_inplace(float* values, float epsilon, uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = msvc_avx512_float_dot(values, values, count);
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(count) + epsilon);
    const __m512 scale = _mm512_set1_ps(inverse_rms);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
        _mm512_storeu_ps(values + index, _mm512_mul_ps(_mm512_loadu_ps(values + index), scale));
    for (; index < count; ++index)
        values[index] *= inverse_rms;
}

void msvc_avx512_float_rms_norm(
    float* output,
    const float* input,
    const float* weight,
    float epsilon,
    float weight_offset,
    uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = msvc_avx512_float_dot(input, input, count);
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

void msvc_avx512_bfloat16_rms_norm(
    float* output,
    const float* input,
    const uint16_t* weight,
    float epsilon,
    float weight_offset,
    uint32_t count) noexcept
{
    if (count == 0)
        return;
    const float square_sum = msvc_avx512_float_dot(input, input, count);
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

void msvc_avx512_float_rope_inplace(
    float* values,
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

void msvc_avx512_float_hc_pre_4(
    float* output,
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

void msvc_avx512_float_hc_post_4(
    float* output,
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

void msvc_avx512_float_gemm_4x4(
    const float* weights,
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
            input_values[token] = _mm512_loadu_ps(
                input + static_cast<size_t>(token) * input_stride + column);
        }
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const __m512 weight_values = _mm512_loadu_ps(
                weights + static_cast<size_t>(output_index) * weight_stride + column);
            for (uint32_t token = 0; token < token_count; ++token)
            {
                accumulators[token][output_index] = _mm512_fmadd_ps(
                    input_values[token],
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

void msvc_avx512_float_gemm_4x8(
    const float* weights,
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
            input_values[token] = _mm512_loadu_ps(
                input + static_cast<size_t>(token) * input_stride + column);
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const __m512 weight_values = _mm512_loadu_ps(
                weights + static_cast<size_t>(output_index) * weight_stride + column);
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] = _mm512_fmadd_ps(
                    input_values[token], weight_values,
                    accumulators[token][output_index]);
        }
    }

    float results[4][8] = {};
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            results[token][output_index] = _mm512_reduce_add_ps(
                accumulators[token][output_index]);

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

void msvc_avx512_bfloat16_gemm_4x8(
    const uint16_t* weights,
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
            input_values[token] = _mm512_loadu_ps(
                input + static_cast<size_t>(token) * input_stride + column);
        __m512 weight_values[8] = {};
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                weights + static_cast<size_t>(output_index) * weight_stride + column));
            weight_values[output_index] = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(packed), 16));
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] = _mm512_fmadd_ps(
                    input_values[token], weight_values[output_index],
                    accumulators[token][output_index]);
        }
    }

    float results[4][8] = {};
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            results[token][output_index] = _mm512_reduce_add_ps(
                accumulators[token][output_index]);

    for (; column < input_columns; ++column)
    {
        for (uint32_t token = 0; token < token_count; ++token)
        {
            const float value = input[static_cast<size_t>(token) * input_stride + column];
            for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            {
                const float weight = std::bit_cast<float>(static_cast<uint32_t>(
                                                              weights[static_cast<size_t>(output_index) * weight_stride + column])
                                                          << 16);
                results[token][output_index] += value * weight;
            }
        }
    }
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            output[static_cast<size_t>(token) * output_stride + output_index] = results[token][output_index];
}

void msvc_avx512_float_exp_inplace(float* values, uint32_t count) noexcept
{
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
        _mm512_storeu_ps(values + index, msvc_avx512_expf(_mm512_loadu_ps(values + index)));
    for (; index < count; ++index)
        values[index] = float_approximate_exp(values[index]);
}

float msvc_avx512_int8_float_dot(
    const int8_t* left,
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
        accumulator0 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index)))),
            _mm512_loadu_ps(right + index),
            accumulator0);
        accumulator1 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index + 16)))),
            _mm512_loadu_ps(right + index + 16),
            accumulator1);
        accumulator2 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index + 32)))),
            _mm512_loadu_ps(right + index + 32),
            accumulator2);
        accumulator3 = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index + 48)))),
            _mm512_loadu_ps(right + index + 48),
            accumulator3);
    }
    __m512 sum = _mm512_add_ps(
        _mm512_add_ps(accumulator0, accumulator1),
        _mm512_add_ps(accumulator2, accumulator3));
    float result = _mm512_reduce_add_ps(sum);
    for (; index + 16 <= count; index += 16)
    {
        result += _mm512_reduce_add_ps(_mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(left + index)))),
            _mm512_loadu_ps(right + index)));
    }
    for (; index < count; ++index)
        result += static_cast<float>(left[index]) * right[index];
    return result;
}

void msvc_avx512_float_scale_inplace(float* values, float scale, uint32_t count) noexcept
{
    const __m512 scale_values = _mm512_set1_ps(scale);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
        _mm512_storeu_ps(values + index, _mm512_mul_ps(_mm512_loadu_ps(values + index), scale_values));
    for (; index < count; ++index)
        values[index] *= scale;
}

void msvc_avx512_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
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

void msvc_avx512_float_scale_add(
    float* output,
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
        const __m512 scaled_output = _mm512_mul_ps(
            _mm512_loadu_ps(output + index), output_scale_values);
        const __m512 scaled_input = _mm512_mul_ps(
            _mm512_loadu_ps(input + index), input_scale_values);
        _mm512_storeu_ps(output + index, _mm512_add_ps(scaled_output, scaled_input));
    }
    for (; index < count; ++index)
        output[index] = output[index] * output_scale + input[index] * input_scale;
}

void msvc_avx512_float_scale_inplace_and_scaled_add(
    float* values,
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
        const __m512 scaled_values = _mm512_mul_ps(
            _mm512_loadu_ps(values + index), value_scale_values);
        _mm512_storeu_ps(values + index, scaled_values);
        const __m512 product = _mm512_mul_ps(scaled_values, output_scale_values);
        _mm512_storeu_ps(
            output + index,
            _mm512_add_ps(_mm512_loadu_ps(output + index), product));
    }
    for (; index < count; ++index)
    {
        values[index] *= value_scale;
        output[index] += output_scale * values[index];
    }
}

void msvc_avx512_float_scale_inplace_and_scaled_add_and_accumulate(
    float* values,
    float value_scale,
    const float* input,
    float input_scale,
    float* output,
    float output_scale,
    uint32_t count) noexcept
{
    const __m512 value_scale_values = _mm512_set1_ps(value_scale);
    const __m512 input_scale_values = _mm512_set1_ps(input_scale);
    const __m512 output_scale_values = _mm512_set1_ps(output_scale);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m512 scaled_values = _mm512_mul_ps(
            _mm512_loadu_ps(values + index), value_scale_values);
        const __m512 input_values = _mm512_mul_ps(
            _mm512_loadu_ps(input + index), input_scale_values);
        const __m512 updated_values = _mm512_add_ps(
            scaled_values, input_values);
        _mm512_storeu_ps(values + index, updated_values);
        _mm512_storeu_ps(
            output + index,
            _mm512_add_ps(
                _mm512_loadu_ps(output + index),
                _mm512_mul_ps(updated_values, output_scale_values)));
    }
    for (; index < count; ++index)
    {
        values[index] = values[index] * value_scale
                        + input[index] * input_scale;
        output[index] += output_scale * values[index];
    }
}

void msvc_avx512_float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept
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

void msvc_avx512_bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept
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

} // namespace moe
} // namespace ncnn
