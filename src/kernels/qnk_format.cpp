#include "qnk.h"

#include "kernels/cpu/cpu_ops.h"
#include "vector.h"
#include "ncnn/moe/runtime.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace ncnn {
namespace moe {

static uint16_t load_u16(const uint8_t* source) noexcept
{
    uint16_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

static float load_f16(const uint8_t* source) noexcept
{
    return f16_to_f32(load_u16(source));
}

static uint32_t load_u32(const uint8_t* source) noexcept
{
    uint32_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

struct QnKOutput
{
    float* output;
    const float* input;
#if defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t sum0;
    float32x4_t sum1;
    float32x4_t sum2;
    float32x4_t sum3;
#endif
};

static void get_scale_min_k4(int index,
                             const uint8_t* scales,
                             uint8_t& scale,
                             uint8_t& minimum) noexcept
{
    if (index < 4)
    {
        scale = scales[index] & 63u;
        minimum = scales[index + 4] & 63u;
    }
    else
    {
        scale = static_cast<uint8_t>((scales[index + 4] & 0x0fu) | ((scales[index - 4] >> 6) << 4));
        minimum = static_cast<uint8_t>((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)
static inline void qnk_i8_to_f32(int8x8_t values,
                                 float scale,
                                 float minimum,
                                 bool affine,
                                 QnKOutput& output,
                                 uint32_t index) noexcept
{
    const int16x8_t wide = vmovl_s8(values);
    float32x4_t low = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide))), scale);
    float32x4_t high = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide))), scale);
    if (affine)
    {
        const float32x4_t min_values = vdupq_n_f32(minimum);
        low = vsubq_f32(low, min_values);
        high = vsubq_f32(high, min_values);
    }
    if (!output.input)
    {
        vst1q_f32(output.output + index, low);
        vst1q_f32(output.output + index + 4, high);
    }
    else if ((index & 8u) == 0)
    {
        output.sum0 = vfmaq_f32(output.sum0, low, vld1q_f32(output.input + index));
        output.sum1 = vfmaq_f32(output.sum1, high, vld1q_f32(output.input + index + 4));
    }
    else
    {
        output.sum2 = vfmaq_f32(output.sum2, low, vld1q_f32(output.input + index));
        output.sum3 = vfmaq_f32(output.sum3, high, vld1q_f32(output.input + index + 4));
    }
}
#endif

static void dequantize_q2(const uint8_t* block, QnKOutput& output) noexcept
{
    const float d = load_f16(block + 80);
    const float minimum = load_f16(block + 82);
    const uint8_t* scales = block;
    const uint8_t* q = block + 16;
    int scale_index = 0;
    for (uint32_t n = 0; n < qnk_block_elements; n += 128)
    {
        int shift = 0;
        for (int group = 0; group < 4; ++group)
        {
            for (int half = 0; half < 2; ++half)
            {
                const uint8_t scale_min = scales[scale_index++];
                const float dl = d * (scale_min & 0x0fu);
                const float ml = minimum * (scale_min >> 4);
                const uint8_t* source = q + n / 4 + half * 16;
                const uint32_t index = n + group * 32 + half * 16;
#if defined(__aarch64__) || defined(_M_ARM64)
                for (int lane = 0; lane < 16; lane += 8)
                {
                    const int8x8_t shift_count = vdup_n_s8(static_cast<int8_t>(-shift));
                    const uint8x8_t values = vand_u8(vshl_u8(vld1_u8(source + lane), shift_count),
                                                     vdup_n_u8(3));
                    qnk_i8_to_f32(vreinterpret_s8_u8(values), dl, ml, true,
                                  output, index + lane);
                }
#else
                for (int lane = 0; lane < 16; ++lane)
                    output.output[index + lane] = dl * static_cast<float>((source[lane] >> shift) & 3u) - ml;
#endif
            }
            shift += 2;
        }
    }
}

static void dequantize_q3(const uint8_t* block, QnKOutput& output) noexcept
{
    constexpr uint32_t kmask1 = 0x03030303u;
    constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t auxiliary[4] = {};
    std::memcpy(auxiliary, block + 96, 12);
    const uint32_t temporary = auxiliary[2];
    auxiliary[2] = ((auxiliary[0] >> 4) & kmask2) | (((temporary >> 4) & kmask1) << 4);
    auxiliary[3] = ((auxiliary[1] >> 4) & kmask2) | (((temporary >> 6) & kmask1) << 4);
    auxiliary[0] = (auxiliary[0] & kmask2) | (((temporary >> 0) & kmask1) << 4);
    auxiliary[1] = (auxiliary[1] & kmask2) | (((temporary >> 2) & kmask1) << 4);
    int8_t scales[16] = {};
    std::memcpy(scales, auxiliary, sizeof(scales));
    const float d = load_f16(block + 108);
    const uint8_t* q = block + 32;
    const uint8_t* high_bits = block;
    uint8_t mask = 1;
    int scale_index = 0;
    for (uint32_t n = 0; n < qnk_block_elements; n += 128)
    {
        int shift = 0;
        for (int group = 0; group < 4; ++group)
        {
            for (int half = 0; half < 2; ++half)
            {
                const float scale = d * static_cast<float>(scales[scale_index++] - 32);
#if defined(__aarch64__) || defined(_M_ARM64)
                for (int lane = 0; lane < 16; lane += 8)
                {
                    const uint8_t* source = q + n / 4 + half * 16 + lane;
                    const uint8x8_t values = vand_u8(vshl_u8(vld1_u8(source),
                                                             vdup_n_s8(static_cast<int8_t>(-shift))),
                                                     vdup_n_u8(3));
                    const uint8x8_t bits = vand_u8(vld1_u8(high_bits + half * 16 + lane),
                                                   vdup_n_u8(mask));
                    const uint8x8_t high = vand_u8(vceq_u8(bits, vdup_n_u8(0)),
                                                   vdup_n_u8(4));
                    qnk_i8_to_f32(vreinterpret_s8_u8(vsub_u8(values, high)),
                                  scale, 0.0f, false,
                                  output, n + group * 32 + half * 16 + lane);
                }
#else
                for (int lane = 0; lane < 16; ++lane)
                {
                    const int high = (high_bits[half * 16 + lane] & mask) != 0 ? 0 : 4;
                    const int value = static_cast<int>((q[n / 4 + half * 16 + lane] >> shift) & 3u) - high;
                    output.output[n + group * 32 + half * 16 + lane] = scale * value;
                }
#endif
            }
            shift += 2;
            mask = static_cast<uint8_t>(mask << 1);
        }
    }
}

static void dequantize_q4(const uint8_t* block, QnKOutput& output) noexcept
{
    const float d = load_f16(block);
    const float minimum = load_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* q = block + 16;
    int scale_index = 0;
    for (uint32_t n = 0; n < qnk_block_elements; n += 64)
    {
        const uint8_t* source = q + n / 2;
        for (int half = 0; half < 2; ++half)
        {
            uint8_t scale_value = 0;
            uint8_t min_value = 0;
            get_scale_min_k4(scale_index++, scales, scale_value, min_value);
            const float scale = d * scale_value;
            const float minimum_value = minimum * min_value;
            const uint32_t index = n + half * 32;
#if defined(__aarch64__) || defined(_M_ARM64)
            for (int lane = 0; lane < 32; lane += 8)
            {
                const uint8x8_t packed = vld1_u8(source + lane);
                const uint8x8_t values = half == 0 ? vand_u8(packed, vdup_n_u8(0x0fu)) : vshr_n_u8(packed, 4);
                qnk_i8_to_f32(vreinterpret_s8_u8(values), scale, minimum_value, true, output, index + lane);
            }
#else
            for (int lane = 0; lane < 32; ++lane)
            {
                const uint8_t packed = source[lane];
                const uint8_t value = half == 0 ? packed & 0x0fu : packed >> 4;
                output.output[index + lane] = scale * static_cast<float>(value) - minimum_value;
            }
#endif
        }
    }
}

static void dequantize_q5(const uint8_t* block, QnKOutput& output) noexcept
{
    const float d = load_f16(block);
    const float minimum = load_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* high_bits = block + 16;
    const uint8_t* low_bits = block + 48;
    int scale_index = 0;
    uint8_t first_mask = 1;
    uint8_t second_mask = 2;
    for (uint32_t n = 0; n < qnk_block_elements; n += 64)
    {
        for (int half = 0; half < 2; ++half)
        {
            uint8_t scale_value = 0;
            uint8_t min_value = 0;
            get_scale_min_k4(scale_index++, scales, scale_value, min_value);
            const float scale = d * scale_value;
            const float minimum_value = minimum * min_value;
            const uint8_t mask = half == 0 ? first_mask : second_mask;
            const uint32_t index = n + half * 32;
#if defined(__aarch64__) || defined(_M_ARM64)
            for (int lane = 0; lane < 32; lane += 8)
            {
                const uint8x8_t low = vld1_u8(low_bits + n / 2 + lane);
                const uint8x8_t high = vld1_u8(high_bits + lane);
                const uint8x8_t unpacked = half == 0 ? vand_u8(low, vdup_n_u8(0x0fu)) : vshr_n_u8(low, 4);
                const uint8x8_t high_value = vand_u8(vcgt_u8(vand_u8(high, vdup_n_u8(mask)),
                                                             vdup_n_u8(0)),
                                                     vdup_n_u8(16));
                qnk_i8_to_f32(vreinterpret_s8_u8(vadd_u8(unpacked, high_value)),
                              scale, minimum_value, true, output, index + lane);
            }
#else
            for (int lane = 0; lane < 32; ++lane)
            {
                const uint8_t low = low_bits[n / 2 + lane];
                const int high = (high_bits[lane] & mask) != 0 ? 16 : 0;
                const uint8_t value = half == 0 ? low & 0x0fu : low >> 4;
                output.output[index + lane] = scale * static_cast<float>(value + high) - minimum_value;
            }
#endif
        }
        first_mask = static_cast<uint8_t>(first_mask << 2);
        second_mask = static_cast<uint8_t>(second_mask << 2);
    }
}

// Keep the decoder in the caller so fused accumulators remain local.
#if defined(_MSC_VER) && defined(_M_ARM64)
static __forceinline void dequantize_q6(const uint8_t* block, QnKOutput& output) noexcept
#else
#if defined(__aarch64__)
__attribute__((always_inline))
#endif
static inline void dequantize_q6(const uint8_t* block, QnKOutput& output) noexcept
#endif
{
    const float d = load_f16(block + 208);
    const uint8_t* low_bits = block;
    const uint8_t* high_bits = block + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
    for (uint32_t n = 0; n < qnk_block_elements; n += 128)
    {
        const uint8_t* low = low_bits + n / 2;
        const uint8_t* high = high_bits + n / 4;
        const int scale_base = static_cast<int>(n / 16);
#if defined(__aarch64__) || defined(_M_ARM64)
        for (int quarter = 0; quarter < 4; ++quarter)
        {
            for (int lane = 0; lane < 32; lane += 8)
            {
                const uint8x8_t packed_low = vld1_u8(low + (quarter & 1) * 32 + lane);
                const uint8x8_t low_value = quarter < 2
                                                ? vand_u8(packed_low, vdup_n_u8(0x0fu))
                                                : vshr_n_u8(packed_low, 4);
                const uint8x8_t high_value = vand_u8(vshl_u8(vld1_u8(high + lane),
                                                             vdup_n_s8(static_cast<int8_t>(-2 * quarter))),
                                                     vdup_n_u8(3));
                const uint8x8_t value = vsub_u8(vorr_u8(low_value, vshl_n_u8(high_value, 4)), vdup_n_u8(32));
                const float scale = d * scales[scale_base + 2 * quarter + lane / 16];
                qnk_i8_to_f32(vreinterpret_s8_u8(value), scale, 0.0f, false, output,
                              n + quarter * 32 + lane);
            }
        }
#else
        for (int lane = 0; lane < 32; ++lane)
        {
            const int q1 = static_cast<int>((low[lane] & 0x0fu) | (((high[lane] >> 0) & 3u) << 4)) - 32;
            const int q2 = static_cast<int>((low[lane + 32] & 0x0fu) | (((high[lane] >> 2) & 3u) << 4)) - 32;
            const int q3 = static_cast<int>((low[lane] >> 4) | (((high[lane] >> 4) & 3u) << 4)) - 32;
            const int q4 = static_cast<int>((low[lane + 32] >> 4) | (((high[lane] >> 6) & 3u) << 4)) - 32;
            output.output[n + lane] = d * scales[scale_base + lane / 16] * q1;
            output.output[n + lane + 32] = d * scales[scale_base + lane / 16 + 2] * q2;
            output.output[n + lane + 64] = d * scales[scale_base + lane / 16 + 4] * q3;
            output.output[n + lane + 96] = d * scales[scale_base + lane / 16 + 6] * q4;
        }
#endif
    }
}

static void dequantize_q8(const uint8_t* block, QnKOutput& output) noexcept
{
    float scale = 0.0f;
    std::memcpy(&scale, block, sizeof(scale));
    const int8_t* values = reinterpret_cast<const int8_t*>(block + 4);
#if defined(__aarch64__) || defined(_M_ARM64)
    for (uint32_t index = 0; index < qnk_block_elements; index += 16)
    {
        const int8x16_t packed = vld1q_s8(values + index);
        qnk_i8_to_f32(vget_low_s8(packed), scale, 0.0f, false, output, index);
        qnk_i8_to_f32(vget_high_s8(packed), scale, 0.0f, false, output, index + 8);
    }
#else
    for (uint32_t index = 0; index < qnk_block_elements; ++index)
        output.output[index] = scale * static_cast<float>(values[index]);
#endif
}

static void dequantize_q8_0(const uint8_t* block, QnKOutput& output) noexcept
{
    const float scale = load_f16(block);
    const int8_t* values = reinterpret_cast<const int8_t*>(block + 2);
#if defined(__aarch64__) || defined(_M_ARM64)
    for (uint32_t index = 0; index < 32; index += 16)
    {
        const int8x16_t packed = vld1q_s8(values + index);
        qnk_i8_to_f32(vget_low_s8(packed), scale, 0.0f, false, output, index);
        qnk_i8_to_f32(vget_high_s8(packed), scale, 0.0f, false, output, index + 8);
    }
#else
    for (uint32_t index = 0; index < 32; ++index)
        output.output[index] = scale * static_cast<float>(values[index]);
#endif
}

static float ue4m3_to_float(uint8_t value) noexcept
{
    if (value == 0 || value == 0x7fu)
        return 0.0f;
    const uint32_t exponent = (value >> 3) & 15u;
    const uint32_t mantissa = value & 7u;
    if (exponent == 0)
        return static_cast<float>(mantissa) * 0.0009765625f;
    const uint32_t bits = (exponent + 119u) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return scale * (1.0f + static_cast<float>(mantissa) * 0.125f);
}

static void dequantize_nvfp4(const uint8_t* block, QnKOutput& output) noexcept
{
    static constexpr float doubled_e2m1[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
                                               -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};
    for (uint32_t subblock = 0; subblock < 4; ++subblock)
    {
        const float scale = ue4m3_to_float(block[subblock]);
        const uint8_t* codes = block + 4 + subblock * 8;
        float* decoded = output.output + subblock * 16;
        for (uint32_t index = 0; index < 8; ++index)
        {
            const uint8_t byte = codes[index];
            decoded[index] = doubled_e2m1[byte & 15u] * scale;
            decoded[index + 8] = doubled_e2m1[byte >> 4] * scale;
        }
    }
}

static void dequantize_q4(DType dtype, const uint8_t* block, QnKOutput& output) noexcept
{
    const bool affine = dtype == DType::Q4_1;
    const float scale = load_f16(block);
    const float minimum = affine ? -load_f16(block + 2) : 0.0f;
    const uint8_t* values = block + (affine ? 4 : 2);
#if defined(__aarch64__) || defined(_M_ARM64)
    std::array<int8_t, 16> low;
    std::array<int8_t, 16> high;
    for (uint32_t index = 0; index < 16; ++index)
    {
        low[index] = static_cast<int8_t>(static_cast<int>(values[index] & 15u) - (affine ? 0 : 8));
        high[index] = static_cast<int8_t>(static_cast<int>(values[index] >> 4) - (affine ? 0 : 8));
    }
    for (uint32_t index = 0; index < 16; index += 8)
    {
        qnk_i8_to_f32(vld1_s8(low.data() + index), scale, minimum, affine, output, index);
        qnk_i8_to_f32(vld1_s8(high.data() + index), scale, minimum, affine, output, index + 16);
    }
#else
    for (uint32_t index = 0; index < 16; ++index)
    {
        const int low = static_cast<int>(values[index] & 15u) - (affine ? 0 : 8);
        const int high = static_cast<int>(values[index] >> 4) - (affine ? 0 : 8);
        output.output[index] = scale * static_cast<float>(low) - minimum;
        output.output[index + 16] = scale * static_cast<float>(high) - minimum;
    }
#endif
}

static void dequantize_q5_0(const uint8_t* block, QnKOutput& output) noexcept
{
    const float scale = load_f16(block);
    const uint32_t high_bits = load_u32(block + 2);
    const uint8_t* values = block + 6;
#if defined(__aarch64__) || defined(_M_ARM64)
    std::array<int8_t, 16> low;
    std::array<int8_t, 16> high;
    for (uint32_t index = 0; index < 16; ++index)
    {
        low[index] = static_cast<int8_t>(static_cast<int>((values[index] & 15u) | (((high_bits >> index) & 1u) << 4)) - 16);
        high[index] = static_cast<int8_t>(static_cast<int>((values[index] >> 4) | (((high_bits >> (index + 16)) & 1u) << 4)) - 16);
    }
    for (uint32_t index = 0; index < 16; index += 8)
    {
        qnk_i8_to_f32(vld1_s8(low.data() + index), scale, 0.0f, false, output, index);
        qnk_i8_to_f32(vld1_s8(high.data() + index), scale, 0.0f, false, output, index + 16);
    }
#else
    for (uint32_t index = 0; index < 16; ++index)
    {
        const int low = static_cast<int>((values[index] & 15u) | (((high_bits >> index) & 1u) << 4)) - 16;
        const int high = static_cast<int>((values[index] >> 4) | (((high_bits >> (index + 16)) & 1u) << 4)) - 16;
        output.output[index] = scale * static_cast<float>(low);
        output.output[index + 16] = scale * static_cast<float>(high);
    }
#endif
}

static constexpr std::array<int8_t, 16> iq4_nl_values = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static void dequantize_iq4_nl(const uint8_t* block, QnKOutput& output) noexcept
{
    const float scale = load_f16(block);
    const uint8_t* values = block + 2;
#if defined(__aarch64__) || defined(_M_ARM64)
    const int8x16_t table = vld1q_s8(iq4_nl_values.data());
    const uint8x16_t packed = vld1q_u8(values);
    const int8x16_t low = vqtbl1q_s8(table, vandq_u8(packed, vdupq_n_u8(15)));
    const int8x16_t high = vqtbl1q_s8(table, vshrq_n_u8(packed, 4));
    qnk_i8_to_f32(vget_low_s8(low), scale, 0.0f, false, output, 0);
    qnk_i8_to_f32(vget_high_s8(low), scale, 0.0f, false, output, 8);
    qnk_i8_to_f32(vget_low_s8(high), scale, 0.0f, false, output, 16);
    qnk_i8_to_f32(vget_high_s8(high), scale, 0.0f, false, output, 24);
#else
    for (uint32_t index = 0; index < 16; ++index)
    {
        output.output[index] = scale * static_cast<float>(iq4_nl_values[values[index] & 15u]);
        output.output[index + 16] = scale * static_cast<float>(iq4_nl_values[values[index] >> 4]);
    }
#endif
}

static void dequantize_iq4_xs(const uint8_t* block, QnKOutput& output) noexcept
{
    const float scale = load_f16(block);
    const uint16_t scales_high = load_u16(block + 2);
    const uint8_t* scales_low = block + 4;
    const uint8_t* values = block + 8;
    for (uint32_t group = 0; group < 8; ++group)
    {
        const uint32_t packed_scale = ((scales_low[group / 2] >> ((group & 1u) * 4)) & 15u)
                                      | (((scales_high >> (group * 2)) & 3u) << 4);
        const float group_scale = scale * static_cast<float>(static_cast<int>(packed_scale) - 32);
        const uint8_t* packed = values + group * 16;
        for (uint32_t index = 0; index < 16; ++index)
        {
            output.output[group * 32 + index] = group_scale * iq4_nl_values[packed[index] & 15u];
            output.output[group * 32 + index + 16] = group_scale * iq4_nl_values[packed[index] >> 4];
        }
    }
}

static void dequantize_q5_1(const uint8_t* block, QnKOutput& output) noexcept
{
    const float scale = load_f16(block);
    const float minimum = load_f16(block + 2);
    const uint32_t high = load_u32(block + 4);
    const uint8_t* values = block + 8;
#if defined(__aarch64__) || defined(_M_ARM64)
    std::array<int8_t, 32> decoded;
    for (uint32_t index = 0; index < 32; ++index)
        decoded[index] = static_cast<int8_t>(((values[index % 16] >> (index / 16 * 4)) & 15u)
                                             | (((high >> index) & 1u) << 4));
    for (uint32_t index = 0; index < 32; index += 8)
        qnk_i8_to_f32(vld1_s8(decoded.data() + index), scale, -minimum, true, output, index);
#else
    for (uint32_t index = 0; index < 32; ++index)
    {
        const uint32_t value = ((values[index % 16] >> (index / 16 * 4)) & 15u)
                               | (((high >> index) & 1u) << 4);
        output.output[index] = scale * static_cast<float>(value) + minimum;
    }
#endif
}

// First block of one weight row plus the byte distance to the next block.
// Packed tiles interleave eight rows, so their blocks are eight slots apart.
uint32_t qnk_block_size(DType dtype) noexcept
{
    if (dtype == DType::NvFp4)
        return 64;
    if (dtype == DType::MxFp4)
        return 32;
    return dtype == DType::Q8_0 || dtype == DType::IQ4NL || dtype == DType::Q5_1
                   || dtype == DType::Q4_0 || dtype == DType::Q4_1 || dtype == DType::Q5_0
               ? 32
               : (is_qnk_dtype(dtype) ? qnk_block_elements : 0);
}

size_t qnk_block_bytes(DType dtype) noexcept
{
    switch (dtype)
    {
    case DType::Q2K: return 84;
    case DType::Q3K: return 110;
    case DType::Q4K: return 144;
    case DType::Q5K: return 176;
    case DType::Q6K: return 210;
    case DType::Q8K: return 292;
    case DType::Q8_0: return 34;
    case DType::IQ4NL: return 18;
    case DType::Q5_1: return 24;
    case DType::Q4_0: return 18;
    case DType::Q4_1: return 20;
    case DType::Q5_0: return 22;
    case DType::IQ4XS: return 136;
    case DType::MxFp4: return 17;
    case DType::NvFp4: return 36;
    default: return 0;
    }
}

uint64_t qnk_storage_bytes(DType dtype, size_t rows, uint32_t columns) noexcept
{
    if (!qnk_shape_supported(dtype, rows, columns))
        return 0;
    const uint64_t row_blocks = static_cast<uint64_t>(columns / qnk_block_size(dtype));
    const uint64_t bytes_per_block = qnk_block_bytes(dtype);
    if (row_blocks != 0 && bytes_per_block > std::numeric_limits<uint64_t>::max() / row_blocks)
        return 0;
    const uint64_t row_bytes = row_blocks * bytes_per_block;
    if (rows != 0 && row_bytes > std::numeric_limits<uint64_t>::max() / rows)
        return 0;
    return row_bytes * rows;
}

uint64_t qnk_packed_storage_bytes(DType dtype, size_t rows, uint32_t columns) noexcept
{
    if (!qnk_shape_supported(dtype, rows, columns)
        || rows > std::numeric_limits<size_t>::max() - 7)
    {
        return 0;
    }
    const uint64_t padded_rows = (rows + 7) / 8 * 8;
    const uint64_t block_count = columns / qnk_block_size(dtype);
    const uint64_t block_bytes = qnk_block_bytes(dtype);
    if (padded_rows > std::numeric_limits<uint64_t>::max() / block_count
        || padded_rows * block_count > std::numeric_limits<uint64_t>::max() / block_bytes)
    {
        return 0;
    }
    return padded_rows * block_count * block_bytes;
}

bool qnk_shape_supported(DType dtype, size_t rows, uint32_t columns) noexcept
{
    return is_qnk_dtype(dtype) && rows != 0 && columns != 0 && columns % qnk_block_size(dtype) == 0
           && qnk_block_bytes(dtype) != 0;
}

static inline float forward_qnk(DType dtype,
                                const uint8_t* block,
                                const float* input,
                                float* output) noexcept
{
    QnKOutput local;
    local.output = output;
    local.input = input;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (local.input)
    {
        local.sum0 = vdupq_n_f32(0.0f);
        local.sum1 = vdupq_n_f32(0.0f);
        local.sum2 = vdupq_n_f32(0.0f);
        local.sum3 = vdupq_n_f32(0.0f);
    }
#endif
    switch (dtype)
    {
    case DType::Q2K: dequantize_q2(block, local); break;
    case DType::Q3K: dequantize_q3(block, local); break;
    case DType::Q4K: dequantize_q4(block, local); break;
    case DType::Q5K: dequantize_q5(block, local); break;
    case DType::Q6K: dequantize_q6(block, local); break;
    case DType::Q8K: dequantize_q8(block, local); break;
    case DType::Q8_0: dequantize_q8_0(block, local); break;
    case DType::IQ4NL: dequantize_iq4_nl(block, local); break;
    case DType::Q5_1: dequantize_q5_1(block, local); break;
    case DType::Q4_0:
    case DType::Q4_1: dequantize_q4(dtype, block, local); break;
    case DType::Q5_0: dequantize_q5_0(block, local); break;
    case DType::IQ4XS: dequantize_iq4_xs(block, local); break;
    case DType::NvFp4: dequantize_nvfp4(block, local); break;
    default:
        if (output)
            std::fill_n(output, qnk_block_elements, 0.0f);
        return 0.0f;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    if (local.input)
    {
        local.sum0 = vaddq_f32(vaddq_f32(local.sum0, local.sum1),
                               vaddq_f32(local.sum2, local.sum3));
        return vaddvq_f32(local.sum0);
    }
#endif
    return 0.0f;
}

void qnk_dequantize_block(DType dtype, const uint8_t* block, float* output) noexcept
{
    if (!block || !output || !is_qnk_dtype(dtype))
        return;
    if (qnk_block_size(dtype) == 32
        && (cpu_backend_caps().active_isa_flags & (CpuIsaX86Avx2Fma | CpuIsaX86Avx512)) != 0)
    {
        cpu_ops().qnk_dequantize_block_32(dtype, block, output);
        return;
    }
    if (dtype == DType::Q6K)
    {
        QnKOutput destination;
        destination.output = output;
        destination.input = nullptr;
        dequantize_q6(block, destination);
        return;
    }
    forward_qnk(dtype, block, nullptr, output);
}

float qnk_dot_block(DType dtype, const uint8_t* block, const float* input) noexcept
{
    if (!block || !input || !is_qnk_dtype(dtype))
        return 0.0f;
    if (dtype == DType::NvFp4)
    {
        std::array<float, 64> decoded;
        qnk_dequantize_block(dtype, block, decoded.data());
        return float_dot(decoded.data(), input, static_cast<uint32_t>(decoded.size()));
    }
    const CpuBackendCaps& caps = cpu_backend_caps();
    const CpuKernelOps& ops = cpu_ops();
    const bool has_x86_simd = (caps.active_isa_flags & (CpuIsaX86Avx2Fma | CpuIsaX86Avx512)) != 0;
    const bool has_arm_neon = (caps.active_isa_flags & CpuIsaArmNeon) != 0;
    if (qnk_block_size(dtype) == 32)
    {
        if (has_x86_simd)
            return ops.qnk_dot_row(dtype, block, 0, 1, input);
        if (has_arm_neon)
            return forward_qnk(dtype, block, input, nullptr);
        return ops.qnk_dot_block(dtype, block, input);
    }
    if (qnk_block_size(dtype) == qnk_block_elements && dtype != DType::IQ4XS && has_x86_simd)
        return ops.qnk_dot_block(dtype, block, input);
    if (has_arm_neon
        && (dtype == DType::Q2K || dtype == DType::Q3K || dtype == DType::Q4K
            || dtype == DType::Q5K || dtype == DType::Q6K || dtype == DType::Q8K))
    {
        return forward_qnk(dtype, block, input, nullptr);
    }
    alignas(64) float decoded[qnk_block_elements];
    qnk_dequantize_block(dtype, block, decoded);
    return float_dot(decoded, input, qnk_block_elements);
}

void qnk_q8k_quantize(const float* source, uint8_t* output, uint32_t columns) noexcept
{
    if (!source || !output || columns == 0 || columns % qnk_block_elements != 0)
        return;
    cpu_ops().qnk_q8k_quantize(source, output, columns);
}

} // namespace moe
} // namespace ncnn
