#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)

#include "kernels/cpu/x86/x86_internal.h"

#include <cstring>
#include <immintrin.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#endif

namespace ncnn {
namespace moe {

static float avx2_horizontal_sum(__m256 values) noexcept
{
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(values), _mm256_extractf128_ps(values, 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

float avx2_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    __m256 accumulator0 = _mm256_setzero_ps();
    __m256 accumulator1 = _mm256_setzero_ps();
    __m256 accumulator2 = _mm256_setzero_ps();
    __m256 accumulator3 = _mm256_setzero_ps();
    uint32_t index = 0;
    for (; index + 32 <= count; index += 32)
    {
        const __m128i packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
        const __m128i packed1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index + 8));
        const __m128i packed2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index + 16));
        const __m128i packed3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index + 24));
        accumulator0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed0), 16)),
                                       _mm256_loadu_ps(input + index),
                                       accumulator0);
        accumulator1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed1), 16)),
                                       _mm256_loadu_ps(input + index + 8),
                                       accumulator1);
        accumulator2 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed2), 16)),
                                       _mm256_loadu_ps(input + index + 16),
                                       accumulator2);
        accumulator3 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed3), 16)),
                                       _mm256_loadu_ps(input + index + 24),
                                       accumulator3);
    }
    float sum = avx2_horizontal_sum(_mm256_add_ps(_mm256_add_ps(accumulator0, accumulator1),
                                                  _mm256_add_ps(accumulator2, accumulator3)));
    for (; index + 8 <= count; index += 8)
    {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
        const __m256 values = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        sum += avx2_horizontal_sum(_mm256_mul_ps(values, _mm256_loadu_ps(input + index)));
    }
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(weights[index]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[index];
    }
    return sum;
}

void avx2_float_to_bfloat16_array(uint16_t* output,
                                  const float* input,
                                  uint32_t count) noexcept
{
    const __m256i rounding_bias = _mm256_set1_epi32(0x7fff);
    const __m256i low_bit_mask = _mm256_set1_epi32(1);
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(input + index));
        const __m256i rounding = _mm256_add_epi32(rounding_bias,
                                                  _mm256_and_si256(_mm256_srli_epi32(bits, 16), low_bit_mask));
        const __m256i high = _mm256_srli_epi32(_mm256_add_epi32(bits, rounding), 16);
        const __m128i shuffle_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13,
                                                   -128, -128, -128, -128, -128, -128, -128, -128);
        const __m128i packed_low = _mm_shuffle_epi8(_mm256_castsi256_si128(high), shuffle_mask);
        const __m128i packed_high = _mm_shuffle_epi8(_mm256_extracti128_si256(high, 1), shuffle_mask);
        const __m128i packed = _mm_unpacklo_epi64(packed_low, packed_high);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + index), packed);
    }
    for (; index < count; ++index)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, input + index, sizeof(bits));
        const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
        output[index] = static_cast<uint16_t>((bits + rounding) >> 16);
    }
}

void avx2_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
{
    const __m256 scale_vector = _mm256_set1_ps(scale);
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + index));
        const __m256 values = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        _mm256_storeu_ps(output + index, _mm256_fmadd_ps(values, scale_vector, _mm256_loadu_ps(output + index)));
    }
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(input[index]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        output[index] += scale * value;
    }
}

} // namespace moe
} // namespace ncnn

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif
