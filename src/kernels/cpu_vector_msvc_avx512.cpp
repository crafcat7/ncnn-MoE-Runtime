#include "cpu_vector_msvc.h"

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
        output[index] = gate_value / (1.0f + std::exp(-sigmoid_scale * gate_value))
                        * (up[index] + up_offset);
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

void msvc_avx512_float_exp_inplace(float* values, uint32_t count) noexcept
{
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
        _mm512_storeu_ps(values + index, msvc_avx512_expf(_mm512_loadu_ps(values + index)));
    for (; index < count; ++index)
        values[index] = std::exp(values[index]);
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
