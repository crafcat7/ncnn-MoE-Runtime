#include "cpu_vector_msvc.h"

#include <immintrin.h>

namespace ncnn {
namespace moe {

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

} // namespace moe
} // namespace ncnn
