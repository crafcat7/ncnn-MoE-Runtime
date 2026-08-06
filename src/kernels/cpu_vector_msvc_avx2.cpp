#include "cpu_vector_msvc.h"

#include <bit>
#include <cmath>
#include <immintrin.h>

namespace ncnn {
namespace moe {

static __m256 msvc_avx2_expf(__m256 value) noexcept
{
    const __m256 rounding = _mm256_set1_ps(0x1.8p23f);
    const __m256 scaled = _mm256_fmadd_ps(value, _mm256_set1_ps(0x1.715476p+0f), rounding);
    const __m256 exponent = _mm256_sub_ps(scaled, rounding);
    const __m256 remainder = _mm256_fnmadd_ps(
        exponent, _mm256_set1_ps(0x1.7f7d1cp-20f),
        _mm256_fnmadd_ps(exponent, _mm256_set1_ps(0x1.62e4p-1f), value));
    const __m256i exponent_bits = _mm256_slli_epi32(_mm256_castps_si256(scaled), 23);
    const __m256 power = _mm256_castsi256_ps(
        _mm256_add_epi32(exponent_bits, _mm256_castps_si256(_mm256_set1_ps(1.0f))));
    const __m256 large = _mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), exponent),
        _mm256_set1_ps(126.0f), _CMP_GT_OQ);
    const __m256 remainder_squared = _mm256_mul_ps(remainder, remainder);
    const __m256 polynomial = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), remainder,
                            _mm256_set1_ps(0x1.573e2ep-5f)),
            remainder_squared,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), remainder,
                            _mm256_set1_ps(0x1.fffdb6p-2f))),
        remainder_squared,
        _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), remainder));
    if (_mm256_movemask_ps(large) == 0)
        return _mm256_fmadd_ps(polynomial, power, power);

    const __m256i negative = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(exponent, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x82000000u));
    const __m256 scale1 = _mm256_castsi256_ps(
        _mm256_add_epi32(negative, _mm256_set1_epi32(0x7f000000u)));
    const __m256 scale2 = _mm256_castsi256_ps(
        _mm256_sub_epi32(exponent_bits, negative));
    const __m256 huge = _mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f), exponent),
        _mm256_set1_ps(192.0f), _CMP_GT_OQ);
    return _mm256_or_ps(
        _mm256_and_ps(huge, _mm256_mul_ps(scale1, scale1)),
        _mm256_andnot_ps(
            huge,
            _mm256_or_ps(
                _mm256_and_ps(large,
                              _mm256_mul_ps(_mm256_fmadd_ps(scale2, polynomial, scale2), scale1)),
                _mm256_andnot_ps(large,
                                 _mm256_fmadd_ps(power, polynomial, power)))));
}

void msvc_avx2_float_silu_mul(float* output, const float* gate, const float* up,
                              float sigmoid_scale, float up_offset, uint32_t count) noexcept
{
    const __m256 scale = _mm256_set1_ps(sigmoid_scale);
    const __m256 offset = _mm256_set1_ps(up_offset);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        const __m256 gate_values = _mm256_loadu_ps(gate + index);
        const __m256 silu = _mm256_div_ps(
            gate_values,
            _mm256_add_ps(one, msvc_avx2_expf(_mm256_sub_ps(
                                   zero, _mm256_mul_ps(scale, gate_values)))));
        _mm256_storeu_ps(output + index,
                         _mm256_mul_ps(silu, _mm256_add_ps(_mm256_loadu_ps(up + index), offset)));
    }
    for (; index < count; ++index)
    {
        const float gate_value = gate[index];
        output[index] = gate_value / (1.0f + std::exp(-sigmoid_scale * gate_value))
                        * (up[index] + up_offset);
    }
}

float msvc_avx2_float_dot(const float* left, const float* right, uint32_t count) noexcept
{
    __m256 accumulator0 = _mm256_setzero_ps();
    __m256 accumulator1 = _mm256_setzero_ps();
    __m256 accumulator2 = _mm256_setzero_ps();
    __m256 accumulator3 = _mm256_setzero_ps();
    uint32_t index = 0;
    for (; index + 32 <= count; index += 32)
    {
        accumulator0 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index), _mm256_loadu_ps(right + index), accumulator0);
        accumulator1 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 8), _mm256_loadu_ps(right + index + 8), accumulator1);
        accumulator2 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 16), _mm256_loadu_ps(right + index + 16), accumulator2);
        accumulator3 = _mm256_fmadd_ps(_mm256_loadu_ps(left + index + 24), _mm256_loadu_ps(right + index + 24), accumulator3);
    }
    __m256 sum = _mm256_add_ps(_mm256_add_ps(accumulator0, accumulator1), _mm256_add_ps(accumulator2, accumulator3));
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    low = _mm_add_ps(low, high);
    low = _mm_add_ps(low, _mm_movehl_ps(low, low));
    low = _mm_add_ss(low, _mm_movehdup_ps(low));
    float result = _mm_cvtss_f32(low);
    for (; index < count; ++index)
        result += left[index] * right[index];
    return result;
}

static float msvc_avx2_horizontal_sum(__m256 values) noexcept
{
    __m128 low = _mm256_castps256_ps128(values);
    __m128 high = _mm256_extractf128_ps(values, 1);
    low = _mm_add_ps(low, high);
    low = _mm_add_ps(low, _mm_movehl_ps(low, low));
    low = _mm_add_ss(low, _mm_movehdup_ps(low));
    return _mm_cvtss_f32(low);
}

void msvc_avx2_float_gemm_4x4(
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
    __m256 accumulators[4][4] = {};
    uint32_t column = 0;
    for (; column + 8 <= input_columns; column += 8)
    {
        __m256 input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
        {
            input_values[token] = _mm256_loadu_ps(
                input + static_cast<size_t>(token) * input_stride + column);
        }
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const __m256 weight_values = _mm256_loadu_ps(
                weights + static_cast<size_t>(output_index) * weight_stride + column);
            for (uint32_t token = 0; token < token_count; ++token)
            {
                accumulators[token][output_index] = _mm256_fmadd_ps(
                    input_values[token],
                    weight_values,
                    accumulators[token][output_index]);
            }
        }
    }

    float results[4][4] = {};
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            results[token][output_index] = msvc_avx2_horizontal_sum(accumulators[token][output_index]);

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

void msvc_avx2_float_gemm_4x8(
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
    __m256 accumulators[4][8] = {};
    uint32_t column = 0;
    for (; column + 8 <= input_columns; column += 8)
    {
        __m256 input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
            input_values[token] = _mm256_loadu_ps(
                input + static_cast<size_t>(token) * input_stride + column);
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const __m256 weight_values = _mm256_loadu_ps(
                weights + static_cast<size_t>(output_index) * weight_stride + column);
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] = _mm256_fmadd_ps(
                    input_values[token], weight_values,
                    accumulators[token][output_index]);
        }
    }

    float results[4][8] = {};
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            results[token][output_index] = msvc_avx2_horizontal_sum(
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

void msvc_avx2_bfloat16_gemm_4x8(
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
    __m256 accumulators[4][8] = {};
    uint32_t column = 0;
    for (; column + 8 <= input_columns; column += 8)
    {
        __m256 input_values[4] = {};
        for (uint32_t token = 0; token < token_count; ++token)
            input_values[token] = _mm256_loadu_ps(
                input + static_cast<size_t>(token) * input_stride + column);
        __m256 weight_values[8] = {};
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
        {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                weights + static_cast<size_t>(output_index) * weight_stride + column));
            weight_values[output_index] = _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_cvtepu16_epi32(packed), 16));
            for (uint32_t token = 0; token < token_count; ++token)
                accumulators[token][output_index] = _mm256_fmadd_ps(
                    input_values[token], weight_values[output_index],
                    accumulators[token][output_index]);
        }
    }

    float results[4][8] = {};
    for (uint32_t token = 0; token < token_count; ++token)
        for (uint32_t output_index = 0; output_index < output_count; ++output_index)
            results[token][output_index] = msvc_avx2_horizontal_sum(
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

void msvc_avx2_float_exp_inplace(float* values, uint32_t count) noexcept
{
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
        _mm256_storeu_ps(values + index, msvc_avx2_expf(_mm256_loadu_ps(values + index)));
    for (; index < count; ++index)
        values[index] = std::exp(values[index]);
}

float msvc_avx2_int8_float_dot(
    const int8_t* left,
    const float* right,
    uint32_t count) noexcept
{
    __m256 accumulator0 = _mm256_setzero_ps();
    __m256 accumulator1 = _mm256_setzero_ps();
    __m256 accumulator2 = _mm256_setzero_ps();
    __m256 accumulator3 = _mm256_setzero_ps();
    uint32_t index = 0;
    for (; index + 32 <= count; index += 32)
    {
        accumulator0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(left + index)))),
            _mm256_loadu_ps(right + index),
            accumulator0);
        accumulator1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(left + index + 8)))),
            _mm256_loadu_ps(right + index + 8),
            accumulator1);
        accumulator2 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(left + index + 16)))),
            _mm256_loadu_ps(right + index + 16),
            accumulator2);
        accumulator3 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(left + index + 24)))),
            _mm256_loadu_ps(right + index + 24),
            accumulator3);
    }
    __m256 sum = _mm256_add_ps(
        _mm256_add_ps(accumulator0, accumulator1),
        _mm256_add_ps(accumulator2, accumulator3));
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    low = _mm_add_ps(low, high);
    low = _mm_add_ps(low, _mm_movehl_ps(low, low));
    low = _mm_add_ss(low, _mm_movehdup_ps(low));
    float result = _mm_cvtss_f32(low);
    for (; index + 8 <= count; index += 8)
    {
        result += _mm_cvtss_f32(_mm_dp_ps(
            _mm256_castps256_ps128(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(left + index))))),
            _mm_loadu_ps(right + index),
            0xff));
    }
    for (; index < count; ++index)
        result += static_cast<float>(left[index]) * right[index];
    return result;
}

void msvc_avx2_float_scale_inplace(float* values, float scale, uint32_t count) noexcept
{
    const __m256 scale_values = _mm256_set1_ps(scale);
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
        _mm256_storeu_ps(values + index, _mm256_mul_ps(_mm256_loadu_ps(values + index), scale_values));
    for (; index < count; ++index)
        values[index] *= scale;
}

void msvc_avx2_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept
{
    const __m256 scale_values = _mm256_set1_ps(scale);
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        const __m256 product = _mm256_mul_ps(scale_values, _mm256_loadu_ps(input + index));
        const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(output + index), product);
        _mm256_storeu_ps(output + index, sum);
    }
    for (; index < count; ++index)
        output[index] += scale * input[index];
}

void msvc_avx2_float_scale_inplace_and_scaled_add(
    float* values,
    float value_scale,
    float* output,
    float output_scale,
    uint32_t count) noexcept
{
    const __m256 value_scale_values = _mm256_set1_ps(value_scale);
    const __m256 output_scale_values = _mm256_set1_ps(output_scale);
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        const __m256 scaled_values = _mm256_mul_ps(
            _mm256_loadu_ps(values + index), value_scale_values);
        _mm256_storeu_ps(values + index, scaled_values);
        const __m256 product = _mm256_mul_ps(scaled_values, output_scale_values);
        _mm256_storeu_ps(
            output + index,
            _mm256_add_ps(_mm256_loadu_ps(output + index), product));
    }
    for (; index < count; ++index)
    {
        values[index] *= value_scale;
        output[index] += output_scale * values[index];
    }
}

void msvc_avx2_float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    const __m256 scale_values = _mm256_set1_ps(scale);
    const __m256 offset_values = _mm256_set1_ps(weight_offset);
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(input + index), scale_values);
        const __m256 weighted = _mm256_add_ps(_mm256_loadu_ps(weight + index), offset_values);
        _mm256_storeu_ps(output + index, _mm256_mul_ps(scaled, weighted));
    }
    for (; index < count; ++index)
        output[index] = input[index] * scale * (weight[index] + weight_offset);
}

void msvc_avx2_bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept
{
    const __m256 scale_values = _mm256_set1_ps(scale);
    const __m256 offset_values = _mm256_set1_ps(weight_offset);
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        __m256i weight_bits = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + index)));
        weight_bits = _mm256_slli_epi32(weight_bits, 16);
        const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(input + index), scale_values);
        const __m256 weighted = _mm256_add_ps(_mm256_castsi256_ps(weight_bits), offset_values);
        _mm256_storeu_ps(output + index, _mm256_mul_ps(scaled, weighted));
    }
    for (; index < count; ++index)
    {
        const float value = std::bit_cast<float>(static_cast<uint32_t>(weight[index]) << 16);
        output[index] = input[index] * scale * (value + weight_offset);
    }
}

} // namespace moe
} // namespace ncnn
