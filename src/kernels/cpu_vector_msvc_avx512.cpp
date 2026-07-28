#include "cpu_vector_msvc.h"

#include <immintrin.h>

namespace ncnn {
namespace moe {

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
