#include "mxfp4_sve2.h"

#include <arm_sve.h>

#include <bit>
#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

static float e8m0_scale(uint8_t exponent) noexcept
{
    const uint32_t bits = exponent == 0 ? UINT32_C(0x00400000) : static_cast<uint32_t>(exponent) << 23;
    return std::bit_cast<float>(bits);
}

static void decode_mxfp4_block(const uint8_t* packed, int8_t* decoded) noexcept
{
    static constexpr int8_t values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (uint32_t index = 0; index < 16; ++index)
    {
        const uint8_t value = packed[index];
        decoded[index * 2] = values[value & 0x0f];
        decoded[index * 2 + 1] = values[value >> 4];
    }
}

static svfloat32_t accumulate_decoded(const int8_t* decoded, const float* input, float scale, svfloat32_t accumulator) noexcept
{
    uint64_t column = 0;
    while (column < 32)
    {
        const svbool_t predicate = svwhilelt_b32(column, UINT64_C(32));
        const svint32_t weights = svld1sb_s32(predicate, decoded + column);
        const svfloat32_t values = svcvt_f32_s32_x(predicate, weights);
        const svfloat32_t activations = svld1_f32(predicate, input + column);
        accumulator = svmla_f32_x(predicate, accumulator, svmul_n_f32_x(predicate, values, scale), activations);
        column += svcntw();
    }
    return accumulator;
}

float sve2_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
{
    svfloat32_t accumulator = svdup_f32(0.0f);
    alignas(64) int8_t decoded[32];
    for (uint32_t block = 0; block < block_count; ++block)
    {
        decode_mxfp4_block(packed + static_cast<size_t>(block) * 16, decoded);
        accumulator = accumulate_decoded(decoded, input + static_cast<size_t>(block) * 32, 0.5f * e8m0_scale(scales[block]), accumulator);
    }
    return svaddv_f32(svptrue_b32(), accumulator);
}

void sve2_mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                             uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output, size_t first_output_stride,
                             float* second_output, size_t second_output_stride) noexcept
{
    alignas(64) int8_t first_decoded[32];
    alignas(64) int8_t second_decoded[32];
    for (size_t token = 0; token < token_count; ++token)
    {
        svfloat32_t first_accumulator = svdup_f32(0.0f);
        svfloat32_t second_accumulator = svdup_f32(0.0f);
        const float* token_input = input + token * input_stride;
        for (uint32_t block = 0; block < block_count; ++block)
        {
            decode_mxfp4_block(first_packed + static_cast<size_t>(block) * 16, first_decoded);
            decode_mxfp4_block(second_packed + static_cast<size_t>(block) * 16, second_decoded);
            const float* block_input = token_input + static_cast<size_t>(block) * 32;
            first_accumulator = accumulate_decoded(first_decoded, block_input, 0.5f * e8m0_scale(first_scales[block]), first_accumulator);
            second_accumulator = accumulate_decoded(second_decoded, block_input, 0.5f * e8m0_scale(second_scales[block]), second_accumulator);
        }
        first_output[token * first_output_stride] = svaddv_f32(svptrue_b32(), first_accumulator);
        second_output[token * second_output_stride] = svaddv_f32(svptrue_b32(), second_accumulator);
    }
}

void sve2_mxfp4_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,
                                 size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                                 float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept
{
    const size_t packed_row_bytes = static_cast<size_t>(block_count) * 16;
    for (uint32_t pair = 0; pair < row_pair_count; ++pair)
    {
        const size_t first_row = static_cast<size_t>(pair) * 2;
        sve2_mxfp4_matmul_rows2(packed + first_row * packed_row_bytes, scales + first_row * block_count, packed + (first_row + 1) * packed_row_bytes,
                                scales + (first_row + 1) * block_count, block_count, input, input_stride, token_count,
                                first_output + static_cast<size_t>(pair) * first_pair_stride, first_token_stride,
                                second_output + static_cast<size_t>(pair) * second_pair_stride, second_token_stride);
    }
}

} // namespace moe
} // namespace ncnn
