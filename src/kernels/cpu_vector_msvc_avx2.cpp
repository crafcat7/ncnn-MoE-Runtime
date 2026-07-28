#include "cpu_vector_msvc.h"

#include <immintrin.h>

namespace ncnn {
namespace moe {

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
