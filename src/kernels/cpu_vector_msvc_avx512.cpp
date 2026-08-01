#include "cpu_vector_msvc.h"

#include <immintrin.h>

namespace ncnn {
namespace moe {

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

} // namespace moe
} // namespace ncnn
