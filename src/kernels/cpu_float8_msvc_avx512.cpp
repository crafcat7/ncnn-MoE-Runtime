#include "cpu_float8_msvc.h"

#include "cpu_float8.h"

#include <algorithm>
#include <immintrin.h>

namespace ncnn {
namespace moe {

static __m512 decode_float8_e4m3(__m128i packed) noexcept
{
    const __m512i bytes = _mm512_cvtepu8_epi32(packed);
    const __m512i exponent = _mm512_and_si512(_mm512_srli_epi32(bytes, 3), _mm512_set1_epi32(15));
    const __m512i mantissa = _mm512_and_si512(bytes, _mm512_set1_epi32(7));
    const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(bytes, _mm512_set1_epi32(128)), 24);

    const __m512i normal_bits = _mm512_or_si512(
        sign,
        _mm512_or_si512(
            _mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(120)), 23),
            _mm512_slli_epi32(mantissa, 20)));
    const __m512 normal = _mm512_castsi512_ps(normal_bits);
    const __m512 subnormal_magnitude = _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa), _mm512_set1_ps(1.0f / 512.0f));
    const __m512 subnormal = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(subnormal_magnitude), sign));
    const __mmask16 exponent_zero = _mm512_cmpeq_epi32_mask(exponent, _mm512_setzero_si512());
    __m512 decoded = _mm512_mask_blend_ps(exponent_zero, normal, subnormal);

    const __mmask16 nan_mask = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(15))
                               & _mm512_cmpeq_epi32_mask(mantissa, _mm512_set1_epi32(7));
    const __m512 nan_value = _mm512_castsi512_ps(_mm512_or_si512(sign, _mm512_set1_epi32(0x7fc00000)));
    decoded = _mm512_mask_blend_ps(nan_mask, decoded, nan_value);
    return decoded;
}

float msvc_avx512_float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept
{
    float result = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulator = _mm512_setzero_ps();
        uint32_t index = block_begin;
        for (; index + 16 <= block_end; index += 16)
        {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
            accumulator = _mm512_fmadd_ps(decode_float8_e4m3(packed), _mm512_loadu_ps(input + index), accumulator);
        }
        float partial = _mm512_reduce_add_ps(accumulator);
        for (; index < block_end; ++index)
            partial += float8_e4m3_to_float(weights[index]) * input[index];
        result += partial * scales[block];
    }
    return result;
}

void msvc_avx512_float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count,
                                             uint32_t block_size, uint32_t row_count, float* output) noexcept
{
    if (row_count == 0 || row_count > 4)
        return;
    for (uint32_t row = 0; row < row_count; ++row)
        output[row] = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulators[4] = {
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
        };
        uint32_t index = block_begin;
        for (; index + 16 <= block_end; index += 16)
        {
            const __m512 input_values = _mm512_loadu_ps(input + index);
            for (uint32_t row = 0; row < row_count; ++row)
            {
                const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + static_cast<size_t>(row) * weight_row_stride + index));
                accumulators[row] = _mm512_fmadd_ps(decode_float8_e4m3(packed), input_values, accumulators[row]);
            }
        }
        for (uint32_t row = 0; row < row_count; ++row)
        {
            float partial = _mm512_reduce_add_ps(accumulators[row]);
            for (uint32_t tail = index; tail < block_end; ++tail)
                partial += float8_e4m3_to_float(weights[static_cast<size_t>(row) * weight_row_stride + tail]) * input[tail];
            output[row] += partial * scales[block];
        }
    }
}

} // namespace moe
} // namespace ncnn
