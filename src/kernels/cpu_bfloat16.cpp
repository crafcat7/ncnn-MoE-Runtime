#include "cpu_bfloat16.h"

#include "cpu_mxfp4.h"

#if defined(NCNN_MOE_MSVC_X86_SIMD)
#include "cpu_mxfp4_msvc.h"
#endif

#include <cstring>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace ncnn {
namespace moe {

static float scalar_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    float sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(weights[index]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[index];
    }
    return sum;
}

#if defined(__aarch64__) || defined(_M_ARM64)
static float neon_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    uint32_t index = 0;
    for (; index + 4 <= count; index += 4)
    {
        const uint32x4_t expanded = vshlq_n_u32(vmovl_u16(vld1_u16(weights + index)), 16);
        accumulator = vfmaq_f32(accumulator, vreinterpretq_f32_u32(expanded), vld1q_f32(input + index));
    }
    float sum = vaddvq_f32(accumulator);
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(weights[index]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[index];
    }
    return sum;
}
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma"))) static float avx2_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    __m256 accumulator = _mm256_setzero_ps();
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
        const __m256 values = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        accumulator = _mm256_fmadd_ps(values, _mm256_loadu_ps(input + index), accumulator);
    }
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(accumulator), _mm256_extractf128_ps(accumulator, 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    float sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(weights[index]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[index];
    }
    return sum;
}

__attribute__((target("avx512f,avx512bw"))) static float avx512_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    __m512 accumulator = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + index));
        const __m512 values = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
        accumulator = _mm512_fmadd_ps(values, _mm512_loadu_ps(input + index), accumulator);
    }
    float sum = _mm512_reduce_add_ps(accumulator);
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(weights[index]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[index];
    }
    return sum;
}
#endif

float bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    switch (mxfp4_kernel_kind())
    {
#if defined(__aarch64__) || defined(_M_ARM64)
    case MxFp4KernelKind::ArmNeon:
    case MxFp4KernelKind::ArmSve2: return neon_bfloat16_dot(weights, input, count);
#elif defined(NCNN_MOE_MSVC_X86_SIMD)
    case MxFp4KernelKind::X86Avx512: return msvc_avx512_bfloat16_dot(weights, input, count);
    case MxFp4KernelKind::X86Avx2: return msvc_avx2_bfloat16_dot(weights, input, count);
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    case MxFp4KernelKind::X86Avx512: return avx512_bfloat16_dot(weights, input, count);
    case MxFp4KernelKind::X86Avx2: return avx2_bfloat16_dot(weights, input, count);
#endif
    default: return scalar_bfloat16_dot(weights, input, count);
    }
}

} // namespace moe
} // namespace ncnn
