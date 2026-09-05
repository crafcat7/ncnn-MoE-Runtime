#include "float8_msvc.h"

#include "float8.h"

#include <algorithm>
#include <immintrin.h>

namespace ncnn {
namespace moe {

static float horizontal_sum(__m256 values) noexcept
{
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(values), _mm256_extractf128_ps(values, 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

static __m256 decode_float8_e4m3(__m128i packed) noexcept
{
    const __m256i bytes = _mm256_cvtepu8_epi32(packed);
    const __m256i exponent = _mm256_and_si256(_mm256_srli_epi32(bytes, 3), _mm256_set1_epi32(15));
    const __m256i mantissa = _mm256_and_si256(bytes, _mm256_set1_epi32(7));
    const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(bytes, _mm256_set1_epi32(128)), 24);

    const __m256i normal_bits = _mm256_or_si256(
        sign,
        _mm256_or_si256(
            _mm256_slli_epi32(_mm256_add_epi32(exponent, _mm256_set1_epi32(120)), 23),
            _mm256_slli_epi32(mantissa, 20)));
    const __m256 normal = _mm256_castsi256_ps(normal_bits);
    const __m256 subnormal_magnitude = _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa), _mm256_set1_ps(1.0f / 512.0f));
    const __m256 subnormal = _mm256_xor_ps(subnormal_magnitude, _mm256_castsi256_ps(sign));
    const __m256 exponent_zero = _mm256_castsi256_ps(_mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
    __m256 decoded = _mm256_blendv_ps(normal, subnormal, exponent_zero);

    const __m256i nan_mask = _mm256_and_si256(
        _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(15)),
        _mm256_cmpeq_epi32(mantissa, _mm256_set1_epi32(7)));
    const __m256 nan_value = _mm256_castsi256_ps(_mm256_or_si256(sign, _mm256_set1_epi32(0x7fc00000)));
    decoded = _mm256_blendv_ps(decoded, nan_value, _mm256_castsi256_ps(nan_mask));
    return decoded;
}

float msvc_avx2_float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept
{
    float result = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m256 accumulator = _mm256_setzero_ps();
        uint32_t index = block_begin;
        for (; index + 8 <= block_end; index += 8)
        {
            const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights + index));
            accumulator = _mm256_fmadd_ps(decode_float8_e4m3(packed), _mm256_loadu_ps(input + index), accumulator);
        }
        float partial = horizontal_sum(accumulator);
        for (; index < block_end; ++index)
            partial += float8_e4m3_to_float(weights[index]) * input[index];
        result += partial * scales[block];
    }
    return result;
}

void msvc_avx2_float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count,
                                           uint32_t block_size, uint32_t row_count, float* output) noexcept
{
    if (row_count == 0 || row_count > 4)
        return;
    for (uint32_t row = 0; row < row_count; ++row)
        output[row] = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m256 accumulators[4] = {
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        };
        uint32_t index = block_begin;
        for (; index + 8 <= block_end; index += 8)
        {
            const __m256 input_values = _mm256_loadu_ps(input + index);
            for (uint32_t row = 0; row < row_count; ++row)
            {
                const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights + static_cast<size_t>(row) * weight_row_stride + index));
                accumulators[row] = _mm256_fmadd_ps(decode_float8_e4m3(packed), input_values, accumulators[row]);
            }
        }
        for (uint32_t row = 0; row < row_count; ++row)
        {
            float partial = horizontal_sum(accumulators[row]);
            for (uint32_t tail = index; tail < block_end; ++tail)
                partial += float8_e4m3_to_float(weights[static_cast<size_t>(row) * weight_row_stride + tail]) * input[tail];
            output[row] += partial * scales[block];
        }
    }
}

} // namespace moe
} // namespace ncnn
