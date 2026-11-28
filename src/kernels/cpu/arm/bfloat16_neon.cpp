#if defined(__aarch64__) || defined(_M_ARM64)

#include "kernels/cpu/arm/arm_internal.h"

#include <arm_neon.h>
#include <cstring>

namespace ncnn {
namespace moe {

static void scalar_float_to_bfloat16_array(uint16_t* output,
                                           const float* input,
                                           uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, input + index, sizeof(bits));
        const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
        output[index] = static_cast<uint16_t>((bits + rounding) >> 16);
    }
}

void neon_float_to_bfloat16_array(uint16_t* output,
                                  const float* input,
                                  uint32_t count) noexcept
{
    const uint32x4_t rounding_bias = vdupq_n_u32(0x7fffu);
    const uint32x4_t low_bit_mask = vdupq_n_u32(1);
    uint32_t index = 0;
    for (; index + 4 <= count; index += 4)
    {
        const uint32x4_t bits = vreinterpretq_u32_f32(vld1q_f32(input + index));
        const uint32x4_t rounding = vaddq_u32(rounding_bias,
                                              vandq_u32(vshrq_n_u32(bits, 16), low_bit_mask));
        const uint32x4_t high = vshrq_n_u32(vaddq_u32(bits, rounding), 16);
        vst1_u16(output + index, vmovn_u32(high));
    }
    scalar_float_to_bfloat16_array(output + index, input + index, count - index);
}

float neon_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    uint32_t index = 0;
    if (count >= 128)
    {
        float32x4_t sum4 = vdupq_n_f32(0.0f);
        float32x4_t sum5 = vdupq_n_f32(0.0f);
        float32x4_t sum6 = vdupq_n_f32(0.0f);
        float32x4_t sum7 = vdupq_n_f32(0.0f);
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#elif defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC unroll 2
#endif
        for (; count - index >= 32; index += 32)
        {
            const uint16x8_t weights0 = vld1q_u16(weights + index);
            const uint16x8_t weights1 = vld1q_u16(weights + index + 8);
            const uint16x8_t weights2 = vld1q_u16(weights + index + 16);
            const uint16x8_t weights3 = vld1q_u16(weights + index + 24);
            sum0 = vfmaq_f32(sum0,
                             vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(weights0), 16)),
                             vld1q_f32(input + index));
            sum1 = vfmaq_f32(sum1,
                             vreinterpretq_f32_u32(vshll_high_n_u16(weights0, 16)),
                             vld1q_f32(input + index + 4));
            sum2 = vfmaq_f32(sum2,
                             vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(weights1), 16)),
                             vld1q_f32(input + index + 8));
            sum3 = vfmaq_f32(sum3,
                             vreinterpretq_f32_u32(vshll_high_n_u16(weights1, 16)),
                             vld1q_f32(input + index + 12));
            sum4 = vfmaq_f32(sum4,
                             vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(weights2), 16)),
                             vld1q_f32(input + index + 16));
            sum5 = vfmaq_f32(sum5,
                             vreinterpretq_f32_u32(vshll_high_n_u16(weights2, 16)),
                             vld1q_f32(input + index + 20));
            sum6 = vfmaq_f32(sum6,
                             vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(weights3), 16)),
                             vld1q_f32(input + index + 24));
            sum7 = vfmaq_f32(sum7,
                             vreinterpretq_f32_u32(vshll_high_n_u16(weights3, 16)),
                             vld1q_f32(input + index + 28));
        }
        sum0 = vaddq_f32(sum0, sum4);
        sum1 = vaddq_f32(sum1, sum5);
        sum2 = vaddq_f32(sum2, sum6);
        sum3 = vaddq_f32(sum3, sum7);
    }
#if defined(__clang__)
#pragma clang loop unroll_count(4)
#elif defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC unroll 4
#endif
    for (; count - index >= 16; index += 16)
    {
        const uint16x8_t weights0 = vld1q_u16(weights + index);
        const uint16x8_t weights1 = vld1q_u16(weights + index + 8);
        const uint32x4_t expanded0 = vshll_n_u16(vget_low_u16(weights0), 16);
        const uint32x4_t expanded1 = vshll_n_u16(vget_high_u16(weights0), 16);
        const uint32x4_t expanded2 = vshll_n_u16(vget_low_u16(weights1), 16);
        const uint32x4_t expanded3 = vshll_n_u16(vget_high_u16(weights1), 16);
        sum0 = vfmaq_f32(sum0,
                         vreinterpretq_f32_u32(expanded0),
                         vld1q_f32(input + index));
        sum1 = vfmaq_f32(sum1,
                         vreinterpretq_f32_u32(expanded1),
                         vld1q_f32(input + index + 4));
        sum2 = vfmaq_f32(sum2,
                         vreinterpretq_f32_u32(expanded2),
                         vld1q_f32(input + index + 8));
        sum3 = vfmaq_f32(sum3,
                         vreinterpretq_f32_u32(expanded3),
                         vld1q_f32(input + index + 12));
    }
    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    for (; count - index >= 4; index += 4)
    {
        const uint32x4_t expanded = vshlq_n_u32(vmovl_u16(vld1_u16(weights + index)), 16);
        sum0 = vfmaq_f32(sum0,
                         vreinterpretq_f32_u32(expanded),
                         vld1q_f32(input + index));
    }
    float sum = vaddvq_f32(sum0);
    const uint32_t tail_count = count & 3u;
    for (uint32_t tail = 0; tail < tail_count; ++tail)
    {
        const uint32_t i = index + tail;
        const uint32_t bits = static_cast<uint32_t>(weights[i]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[i];
    }
    return sum;
}

void neon_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
{
    const float32x4_t scale_vector = vdupq_n_f32(scale);
    uint32_t index = 0;
    for (; index + 4 <= count; index += 4)
    {
        const uint32x4_t expanded = vshlq_n_u32(vmovl_u16(vld1_u16(input + index)), 16);
        const float32x4_t values = vreinterpretq_f32_u32(expanded);
        vst1q_f32(output + index, vfmaq_f32(vld1q_f32(output + index), values, scale_vector));
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

#endif
