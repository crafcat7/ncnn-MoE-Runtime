#if defined(__aarch64__) || defined(_M_ARM64)

#include "kernels/cpu/arm/arm_internal.h"

#include <algorithm>
#include <arm_neon.h>
#include <array>
#include <bit>
#include <cstring>

namespace ncnn {
namespace moe {

static std::array<float, 256> make_scale_table()
{
    std::array<float, 256> table = {};
    for (uint32_t index = 0; index < table.size(); ++index)
    {
        // GGML E8M0 scale is a half-scale because its MXFP4 codebook is
        // doubled.  Construct the exact bit pattern so 0 and 255 remain
        // finite subnormal/normal powers of two instead of underflowing or
        // becoming infinity.
        const uint32_t bits = index < 2 ? (UINT32_C(0x00200000) << index)
                                        : (static_cast<uint32_t>(index - 1) << 23);
        std::memcpy(&table[index], &bits, sizeof(bits));
    }
    return table;
}

static const std::array<float, 256>& scale_table()
{
    static const std::array<float, 256> values = make_scale_table();
    return values;
}

#define NCNN_MOE_DEFINE_MXFP4_ROW_PAIR_KERNEL(function_name, row_pair_function)                                                                     \
    void function_name(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,             \
                       size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,           \
                       float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept                                        \
    {                                                                                                                                               \
        const size_t packed_row_bytes = static_cast<size_t>(block_count) * 16;                                                                      \
        for (uint32_t pair = 0; pair < row_pair_count; ++pair)                                                                                      \
        {                                                                                                                                           \
            const size_t first_row = static_cast<size_t>(pair) * 2;                                                                                 \
            row_pair_function(packed + first_row * packed_row_bytes, scales + first_row * block_count, packed + (first_row + 1) * packed_row_bytes, \
                              scales + (first_row + 1) * block_count, block_count, input, input_stride, token_count,                                \
                              first_output + static_cast<size_t>(pair) * first_pair_stride, first_token_stride,                                     \
                              second_output + static_cast<size_t>(pair) * second_pair_stride, second_token_stride);                                 \
        }                                                                                                                                           \
    }

float neon_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
{
    static constexpr int8_t value_bytes[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int8x16_t value_table = vld1q_s8(value_bytes);
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const std::array<float, 256>& scales_by_exponent = scale_table();
    float sum = 0.0f;

    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const uint8_t* block = packed + static_cast<size_t>(block_index) * 16;
        const float* input_block = input + static_cast<size_t>(block_index) * 32;
        const uint8x16_t bytes = vld1q_u8(block);
        const uint8x16_t low = vandq_u8(bytes, nibble_mask);
        const uint8x16_t high = vandq_u8(vshrq_n_u8(bytes, 4), nibble_mask);
        const uint8x16x2_t interleaved = vzipq_u8(low, high);
        const int8x16_t decoded_low = vqtbl1q_s8(value_table, interleaved.val[0]);
        const int8x16_t decoded_high = vqtbl1q_s8(value_table, interleaved.val[1]);

        float32x4_t accumulator = vdupq_n_f32(0.0f);
        const int16x8_t low_16 = vmovl_s8(vget_low_s8(decoded_low));
        const int16x8_t low_high_16 = vmovl_s8(vget_high_s8(decoded_low));
        const int16x8_t high_16 = vmovl_s8(vget_low_s8(decoded_high));
        const int16x8_t high_high_16 = vmovl_s8(vget_high_s8(decoded_high));
#define NCNN_MOE_ACCUMULATE_MXFP4_NEON(values16, input_offset)                                                                     \
    accumulator = vfmaq_f32(accumulator, vcvtq_f32_s32(vmovl_s16(vget_low_s16(values16))), vld1q_f32(input_block + input_offset)); \
    accumulator = vfmaq_f32(accumulator, vcvtq_f32_s32(vmovl_s16(vget_high_s16(values16))), vld1q_f32(input_block + input_offset + 4))
        NCNN_MOE_ACCUMULATE_MXFP4_NEON(low_16, 0);
        NCNN_MOE_ACCUMULATE_MXFP4_NEON(low_high_16, 8);
        NCNN_MOE_ACCUMULATE_MXFP4_NEON(high_16, 16);
        NCNN_MOE_ACCUMULATE_MXFP4_NEON(high_high_16, 24);
#undef NCNN_MOE_ACCUMULATE_MXFP4_NEON
        sum += vaddvq_f32(accumulator) * (scales_by_exponent[scales[block_index]]);
    }
    return sum;
}

void neon_mxfp4_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
                         float* output, size_t output_stride) noexcept
{
    static constexpr int8_t value_bytes[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int8x16_t value_table = vld1q_s8(value_bytes);
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const std::array<float, 256>& scales_by_exponent = scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
        output[token_index * output_stride] = 0.0f;

    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const uint8x16_t bytes = vld1q_u8(packed + static_cast<size_t>(block_index) * 16);
        const uint8x16_t low = vandq_u8(bytes, nibble_mask);
        const uint8x16_t high = vandq_u8(vshrq_n_u8(bytes, 4), nibble_mask);
        const uint8x16x2_t interleaved = vzipq_u8(low, high);
        const int8x16_t decoded_low = vqtbl1q_s8(value_table, interleaved.val[0]);
        const int8x16_t decoded_high = vqtbl1q_s8(value_table, interleaved.val[1]);
        const int16x8_t decoded_16[4] = {
            vmovl_s8(vget_low_s8(decoded_low)),
            vmovl_s8(vget_high_s8(decoded_low)),
            vmovl_s8(vget_low_s8(decoded_high)),
            vmovl_s8(vget_high_s8(decoded_high)),
        };
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float scale = scales_by_exponent[scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index)
        {
            const float* token = input + token_index * input_stride + input_offset;
            float32x4_t accumulator = vdupq_n_f32(0.0f);
            for (uint32_t group = 0; group < 4; ++group)
            {
                accumulator = vfmaq_f32(accumulator, vcvtq_f32_s32(vmovl_s16(vget_low_s16(decoded_16[group]))), vld1q_f32(token + group * 8));
                accumulator = vfmaq_f32(accumulator, vcvtq_f32_s32(vmovl_s16(vget_high_s16(decoded_16[group]))), vld1q_f32(token + group * 8 + 4));
            }
            output[token_index * output_stride] += vaddvq_f32(accumulator) * scale;
        }
    }
}

void neon_mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                             uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output,
                             size_t first_output_stride, float* second_output, size_t second_output_stride) noexcept
{
    static constexpr int8_t value_bytes[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int8x16_t value_table = vld1q_s8(value_bytes);
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const std::array<float, 256>& scales_by_exponent = scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
    {
        first_output[token_index * first_output_stride] = 0.0f;
        second_output[token_index * second_output_stride] = 0.0f;
    }

    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const uint8x16_t packed_rows[2] = {
            vld1q_u8(first_packed + static_cast<size_t>(block_index) * 16),
            vld1q_u8(second_packed + static_cast<size_t>(block_index) * 16),
        };
        int16x8_t decoded_rows[2][4];
        for (uint32_t row = 0; row < 2; ++row)
        {
            const uint8x16_t low = vandq_u8(packed_rows[row], nibble_mask);
            const uint8x16_t high = vandq_u8(vshrq_n_u8(packed_rows[row], 4), nibble_mask);
            const uint8x16x2_t interleaved = vzipq_u8(low, high);
            const int8x16_t decoded_low = vqtbl1q_s8(value_table, interleaved.val[0]);
            const int8x16_t decoded_high = vqtbl1q_s8(value_table, interleaved.val[1]);
            decoded_rows[row][0] = vmovl_s8(vget_low_s8(decoded_low));
            decoded_rows[row][1] = vmovl_s8(vget_high_s8(decoded_low));
            decoded_rows[row][2] = vmovl_s8(vget_low_s8(decoded_high));
            decoded_rows[row][3] = vmovl_s8(vget_high_s8(decoded_high));
        }
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float first_scale = scales_by_exponent[first_scales[block_index]];
        const float second_scale = scales_by_exponent[second_scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index)
        {
            const float* token = input + token_index * input_stride + input_offset;
            float32x4_t first_accumulator = vdupq_n_f32(0.0f);
            float32x4_t second_accumulator = vdupq_n_f32(0.0f);
            for (uint32_t group = 0; group < 4; ++group)
            {
                const float32x4_t input_low = vld1q_f32(token + group * 8);
                const float32x4_t input_high = vld1q_f32(token + group * 8 + 4);
                first_accumulator = vfmaq_f32(first_accumulator, vcvtq_f32_s32(vmovl_s16(vget_low_s16(decoded_rows[0][group]))), input_low);
                first_accumulator = vfmaq_f32(first_accumulator, vcvtq_f32_s32(vmovl_s16(vget_high_s16(decoded_rows[0][group]))), input_high);
                second_accumulator = vfmaq_f32(second_accumulator, vcvtq_f32_s32(vmovl_s16(vget_low_s16(decoded_rows[1][group]))), input_low);
                second_accumulator = vfmaq_f32(second_accumulator, vcvtq_f32_s32(vmovl_s16(vget_high_s16(decoded_rows[1][group]))), input_high);
            }
            first_output[token_index * first_output_stride] += vaddvq_f32(first_accumulator) * first_scale;
            second_output[token_index * second_output_stride] += vaddvq_f32(second_accumulator) * second_scale;
        }
    }
}

NCNN_MOE_DEFINE_MXFP4_ROW_PAIR_KERNEL(neon_mxfp4_matmul_row_pairs, neon_mxfp4_matmul_rows2)

} // namespace moe
} // namespace ncnn

#endif
