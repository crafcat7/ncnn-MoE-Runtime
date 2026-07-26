#include "cpu_mxfp4.h"
#include "engine/cpu_features.h"

#if defined(NCNN_MOE_MSVC_X86_SIMD)
#include "cpu_mxfp4_msvc.h"
#include <intrin.h>
#endif
#if defined(NCNN_MOE_ARM_SVE2_KERNEL)
#include "cpu_mxfp4_sve2.h"
#include "ncnn/moe/runtime.h"
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace ncnn {
namespace moe {

static volatile float mxfp4_benchmark_sink = 0.0f;

static std::array<float, 256> make_scale_table()
{
    std::array<float, 256> table = {};
    for (uint32_t index = 0; index < table.size(); ++index)
        table[index] = std::ldexp(1.0f, static_cast<int>(index) - 127);
    return table;
}

static const std::array<float, 256>& scale_table()
{
    static const std::array<float, 256> values = make_scale_table();
    return values;
}

static float scalar_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
{
    static constexpr float values[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    const std::array<float, 256>& scales_by_exponent = scale_table();
    float sum = 0.0f;
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const uint8_t* block = packed + static_cast<size_t>(block_index) * 16;
        const float* input_block = input + static_cast<size_t>(block_index) * 32;
        float block_sum = 0.0f;
        for (uint32_t byte_index = 0; byte_index < 16; ++byte_index)
        {
            const uint8_t byte = block[byte_index];
            block_sum += values[byte & 0x0f] * input_block[byte_index * 2];
            block_sum += values[byte >> 4] * input_block[byte_index * 2 + 1];
        }
        sum += block_sum * scales_by_exponent[scales[block_index]];
    }
    return sum;
}

static void scalar_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
                            float* output, size_t output_stride) noexcept
{
    static constexpr float values[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    const std::array<float, 256>& scales_by_exponent = scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
        output[token_index * output_stride] = 0.0f;

    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const uint8_t* block = packed + static_cast<size_t>(block_index) * 16;
        const float scale = scales_by_exponent[scales[block_index]];
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        for (uint32_t byte_index = 0; byte_index < 16; ++byte_index)
        {
            const uint8_t byte = block[byte_index];
            const float low = values[byte & 0x0f] * scale;
            const float high = values[byte >> 4] * scale;
            const size_t column = input_offset + byte_index * 2;
            for (size_t token_index = 0; token_index < token_count; ++token_index)
            {
                const float* token = input + token_index * input_stride;
                output[token_index * output_stride] += low * token[column] + high * token[column + 1];
            }
        }
    }
}

static void scalar_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                                uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output,
                                size_t first_output_stride, float* second_output, size_t second_output_stride) noexcept
{
    static constexpr float values[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    const std::array<float, 256>& scales_by_exponent = scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
    {
        first_output[token_index * first_output_stride] = 0.0f;
        second_output[token_index * second_output_stride] = 0.0f;
    }

    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const uint8_t* first_block = first_packed + static_cast<size_t>(block_index) * 16;
        const uint8_t* second_block = second_packed + static_cast<size_t>(block_index) * 16;
        const float first_scale = scales_by_exponent[first_scales[block_index]];
        const float second_scale = scales_by_exponent[second_scales[block_index]];
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        for (uint32_t byte_index = 0; byte_index < 16; ++byte_index)
        {
            const uint8_t first_byte = first_block[byte_index];
            const uint8_t second_byte = second_block[byte_index];
            const float first_low = values[first_byte & 0x0f] * first_scale;
            const float first_high = values[first_byte >> 4] * first_scale;
            const float second_low = values[second_byte & 0x0f] * second_scale;
            const float second_high = values[second_byte >> 4] * second_scale;
            const size_t column = input_offset + byte_index * 2;
            for (size_t token_index = 0; token_index < token_count; ++token_index)
            {
                const float* token = input + token_index * input_stride;
                const float low_input = token[column];
                const float high_input = token[column + 1];
                first_output[token_index * first_output_stride] += first_low * low_input + first_high * high_input;
                second_output[token_index * second_output_stride] += second_low * low_input + second_high * high_input;
            }
        }
    }
}

#define NCNN_MOE_DEFINE_MXFP4_ROW_PAIR_KERNEL(function_name, row_pair_function)                                                                     \
    static void function_name(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,      \
                              size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,    \
                              float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept                                 \
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

NCNN_MOE_DEFINE_MXFP4_ROW_PAIR_KERNEL(scalar_matmul_row_pairs, scalar_matmul_rows2)

using DotFunction = decltype(&scalar_dot);
using GemmRowFunction = decltype(&scalar_gemm_row);
using MatmulRows2Function = decltype(&scalar_matmul_rows2);
using MatmulRowPairsFunction = decltype(&scalar_matmul_row_pairs);

#if defined(__aarch64__) || defined(_M_ARM64)
static float neon_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
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
        sum += vaddvq_f32(accumulator) * (0.5f * scales_by_exponent[scales[block_index]]);
    }
    return sum;
}

static void neon_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
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
        const float scale = 0.5f * scales_by_exponent[scales[block_index]];
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

static void neon_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
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
        const float first_scale = 0.5f * scales_by_exponent[first_scales[block_index]];
        const float second_scale = 0.5f * scales_by_exponent[second_scales[block_index]];
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

NCNN_MOE_DEFINE_MXFP4_ROW_PAIR_KERNEL(neon_matmul_row_pairs, neon_matmul_rows2)
#endif

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma,ssse3"))) static float avx2_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const std::array<float, 256>& scales_by_exponent = scale_table();
    float sum = 0.0f;
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + static_cast<size_t>(block_index) * 16));
        const __m128i low = _mm_and_si128(bytes, nibble_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
        const __m128i indices_low = _mm_unpacklo_epi8(low, high);
        const __m128i indices_high = _mm_unpackhi_epi8(low, high);
        const __m128i decoded_low = _mm_shuffle_epi8(value_table, indices_low);
        const __m128i decoded_high = _mm_shuffle_epi8(value_table, indices_high);
        const float* input_block = input + static_cast<size_t>(block_index) * 32;
        __m256 accumulator = _mm256_setzero_ps();
#define NCNN_MOE_ACCUMULATE_MXFP4_AVX2(values8, input_offset) \
    accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(values8)), _mm256_loadu_ps(input_block + input_offset), accumulator)
        NCNN_MOE_ACCUMULATE_MXFP4_AVX2(decoded_low, 0);
        NCNN_MOE_ACCUMULATE_MXFP4_AVX2(_mm_srli_si128(decoded_low, 8), 8);
        NCNN_MOE_ACCUMULATE_MXFP4_AVX2(decoded_high, 16);
        NCNN_MOE_ACCUMULATE_MXFP4_AVX2(_mm_srli_si128(decoded_high, 8), 24);
#undef NCNN_MOE_ACCUMULATE_MXFP4_AVX2
        const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(accumulator), _mm256_extractf128_ps(accumulator, 1));
        const __m128 pairs = _mm_hadd_ps(halves, halves);
        const float block_sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
        sum += block_sum * (0.5f * scales_by_exponent[scales[block_index]]);
    }
    return sum;
}

__attribute__((target("avx2,fma,ssse3"))) static void avx2_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input,
                                                                    size_t input_stride, size_t token_count, float* output, size_t output_stride) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const std::array<float, 256>& scales_by_exponent = scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
        output[token_index * output_stride] = 0.0f;
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + static_cast<size_t>(block_index) * 16));
        const __m128i low = _mm_and_si128(bytes, nibble_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
        const __m128i decoded[2] = {
            _mm_shuffle_epi8(value_table, _mm_unpacklo_epi8(low, high)),
            _mm_shuffle_epi8(value_table, _mm_unpackhi_epi8(low, high)),
        };
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float scale = 0.5f * scales_by_exponent[scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index)
        {
            const float* token = input + token_index * input_stride + input_offset;
            __m256 accumulator = _mm256_setzero_ps();
            for (uint32_t half = 0; half < 2; ++half)
            {
                accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(decoded[half])), _mm256_loadu_ps(token + half * 16), accumulator);
                accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(decoded[half], 8))),
                                              _mm256_loadu_ps(token + half * 16 + 8), accumulator);
            }
            const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(accumulator), _mm256_extractf128_ps(accumulator, 1));
            const __m128 pairs = _mm_hadd_ps(halves, halves);
            output[token_index * output_stride] += _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs)) * scale;
        }
    }
}

__attribute__((target("avx2,fma,ssse3"))) static void avx2_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed,
                                                                        const uint8_t* second_scales, uint32_t block_count, const float* input,
                                                                        size_t input_stride, size_t token_count, float* first_output,
                                                                        size_t first_output_stride, float* second_output, size_t second_output_stride) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const std::array<float, 256>& scales_by_exponent = scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
    {
        first_output[token_index * first_output_stride] = 0.0f;
        second_output[token_index * second_output_stride] = 0.0f;
    }

    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const __m128i packed_rows[2] = {
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(first_packed + static_cast<size_t>(block_index) * 16)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(second_packed + static_cast<size_t>(block_index) * 16)),
        };
        __m128i decoded_rows[2][2];
        for (uint32_t row = 0; row < 2; ++row)
        {
            const __m128i low = _mm_and_si128(packed_rows[row], nibble_mask);
            const __m128i high = _mm_and_si128(_mm_srli_epi16(packed_rows[row], 4), nibble_mask);
            decoded_rows[row][0] = _mm_shuffle_epi8(value_table, _mm_unpacklo_epi8(low, high));
            decoded_rows[row][1] = _mm_shuffle_epi8(value_table, _mm_unpackhi_epi8(low, high));
        }
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float first_scale = 0.5f * scales_by_exponent[first_scales[block_index]];
        const float second_scale = 0.5f * scales_by_exponent[second_scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index)
        {
            const float* token = input + token_index * input_stride + input_offset;
            __m256 first_accumulator = _mm256_setzero_ps();
            __m256 second_accumulator = _mm256_setzero_ps();
            for (uint32_t half = 0; half < 2; ++half)
            {
                const __m256 input_low = _mm256_loadu_ps(token + half * 16);
                const __m256 input_high = _mm256_loadu_ps(token + half * 16 + 8);
                first_accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(decoded_rows[0][half])), input_low, first_accumulator);
                first_accumulator = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(decoded_rows[0][half], 8))), input_high, first_accumulator);
                second_accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(decoded_rows[1][half])), input_low, second_accumulator);
                second_accumulator = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(decoded_rows[1][half], 8))), input_high, second_accumulator);
            }
            const __m128 first_halves = _mm_add_ps(_mm256_castps256_ps128(first_accumulator), _mm256_extractf128_ps(first_accumulator, 1));
            const __m128 first_pairs = _mm_hadd_ps(first_halves, first_halves);
            const float first_sum = _mm_cvtss_f32(_mm_hadd_ps(first_pairs, first_pairs));
            const __m128 second_halves = _mm_add_ps(_mm256_castps256_ps128(second_accumulator), _mm256_extractf128_ps(second_accumulator, 1));
            const __m128 second_pairs = _mm_hadd_ps(second_halves, second_halves);
            const float second_sum = _mm_cvtss_f32(_mm_hadd_ps(second_pairs, second_pairs));
            first_output[token_index * first_output_stride] += first_sum * first_scale;
            second_output[token_index * second_output_stride] += second_sum * second_scale;
        }
    }
}

__attribute__((target("avx2,fma,ssse3"))) NCNN_MOE_DEFINE_MXFP4_ROW_PAIR_KERNEL(avx2_matmul_row_pairs, avx2_matmul_rows2)

    __attribute__((target("avx512f,avx512bw,avx512vl,ssse3,fma"))) static float avx512_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count,
                                                                                           const float* input) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const std::array<float, 256>& scales_by_exponent = scale_table();
    float sum = 0.0f;
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + static_cast<size_t>(block_index) * 16));
        const __m128i low = _mm_and_si128(bytes, nibble_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
        const __m128i decoded_low = _mm_shuffle_epi8(value_table, _mm_unpacklo_epi8(low, high));
        const __m128i decoded_high = _mm_shuffle_epi8(value_table, _mm_unpackhi_epi8(low, high));
        const float* input_block = input + static_cast<size_t>(block_index) * 32;
        __m512 accumulator = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(decoded_low)), _mm512_loadu_ps(input_block));
        accumulator = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(decoded_high)), _mm512_loadu_ps(input_block + 16), accumulator);
        sum += _mm512_reduce_add_ps(accumulator) * (0.5f * scales_by_exponent[scales[block_index]]);
    }
    return sum;
}

__attribute__((target("avx512f,avx512bw,avx512vl,ssse3,fma"))) static void avx512_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count,
                                                                                           const float* input, size_t input_stride, size_t token_count,
                                                                                           float* output, size_t output_stride) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const std::array<float, 256>& scales_by_exponent = scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
        output[token_index * output_stride] = 0.0f;
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + static_cast<size_t>(block_index) * 16));
        const __m128i low = _mm_and_si128(bytes, nibble_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
        const __m128i decoded_low = _mm_shuffle_epi8(value_table, _mm_unpacklo_epi8(low, high));
        const __m128i decoded_high = _mm_shuffle_epi8(value_table, _mm_unpackhi_epi8(low, high));
        const __m512 weights = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(decoded_low));
        const __m512 weights_high = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(decoded_high));
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float scale = 0.5f * scales_by_exponent[scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index)
        {
            const float* token = input + token_index * input_stride + input_offset;
            const __m512 accumulator = _mm512_fmadd_ps(weights_high, _mm512_loadu_ps(token + 16), _mm512_mul_ps(weights, _mm512_loadu_ps(token)));
            output[token_index * output_stride] += _mm512_reduce_add_ps(accumulator) * scale;
        }
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,ssse3,fma"))) static void avx512_matmul_rows2(
    const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales, uint32_t block_count,
    const float* input, size_t input_stride, size_t token_count, float* first_output, size_t first_output_stride, float* second_output,
    size_t second_output_stride) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const std::array<float, 256>& scales_by_exponent = scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
    {
        first_output[token_index * first_output_stride] = 0.0f;
        second_output[token_index * second_output_stride] = 0.0f;
    }

    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        const __m128i packed_rows[2] = {
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(first_packed + static_cast<size_t>(block_index) * 16)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(second_packed + static_cast<size_t>(block_index) * 16)),
        };
        __m512 decoded_low[2];
        __m512 decoded_high[2];
        for (uint32_t row = 0; row < 2; ++row)
        {
            const __m128i low = _mm_and_si128(packed_rows[row], nibble_mask);
            const __m128i high = _mm_and_si128(_mm_srli_epi16(packed_rows[row], 4), nibble_mask);
            decoded_low[row] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(value_table, _mm_unpacklo_epi8(low, high))));
            decoded_high[row] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(value_table, _mm_unpackhi_epi8(low, high))));
        }
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float first_scale = 0.5f * scales_by_exponent[first_scales[block_index]];
        const float second_scale = 0.5f * scales_by_exponent[second_scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index)
        {
            const float* token = input + token_index * input_stride + input_offset;
            const __m512 input_low = _mm512_loadu_ps(token);
            const __m512 input_high = _mm512_loadu_ps(token + 16);
            const __m512 first_accumulator = _mm512_fmadd_ps(decoded_high[0], input_high, _mm512_mul_ps(decoded_low[0], input_low));
            const __m512 second_accumulator = _mm512_fmadd_ps(decoded_high[1], input_high, _mm512_mul_ps(decoded_low[1], input_low));
            first_output[token_index * first_output_stride] += _mm512_reduce_add_ps(first_accumulator) * first_scale;
            second_output[token_index * second_output_stride] += _mm512_reduce_add_ps(second_accumulator) * second_scale;
        }
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,ssse3,fma"))) NCNN_MOE_DEFINE_MXFP4_ROW_PAIR_KERNEL(avx512_matmul_row_pairs, avx512_matmul_rows2)
#endif

#undef NCNN_MOE_DEFINE_MXFP4_ROW_PAIR_KERNEL

    struct KernelDispatch
{
    MxFp4KernelKind kind = MxFp4KernelKind::Scalar;
    DotFunction function = scalar_dot;
    GemmRowFunction gemm_row = scalar_gemm_row;
    MatmulRows2Function matmul_rows2 = scalar_matmul_rows2;
    MatmulRowPairsFunction matmul_row_pairs = scalar_matmul_row_pairs;
};

#if defined(NCNN_MOE_MSVC_X86_SIMD)
static bool msvc_cpu_supports_avx(bool require_avx512) noexcept
{
    int registers[4] = {};
    __cpuid(registers, 0);
    const int maximum_leaf = registers[0];
    if (maximum_leaf < 1)
        return false;

    __cpuidex(registers, 1, 0);
    const uint32_t feature_ecx = static_cast<uint32_t>(registers[2]);
    const uint32_t required_ecx = (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_SSSE3_BIT) | (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_FMA_BIT)
                                  | (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_OSXSAVE_BIT) | (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_AVX_BIT);
    if ((feature_ecx & required_ecx) != required_ecx)
        return false;

    const uint64_t enabled_xstate = _xgetbv(0);
    const uint64_t avx_state_mask = (UINT64_C(1) << NCNN_MOE_XSTATE_XMM_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_YMM_BIT);
    if ((enabled_xstate & avx_state_mask) != avx_state_mask || maximum_leaf < 7)
    {
        return false;
    }

    __cpuidex(registers, 7, 0);
    const uint32_t feature_ebx = static_cast<uint32_t>(registers[1]);
    if ((feature_ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX2_BIT)) == 0)
        return false;
    if (!require_avx512)
        return true;

    const uint32_t required_ebx = (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512F_BIT) | (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512BW_BIT)
                                  | (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512VL_BIT);
    const uint64_t avx512_state_mask = avx_state_mask | (UINT64_C(1) << NCNN_MOE_XSTATE_OPMASK_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_ZMM_HI256_BIT)
                                       | (UINT64_C(1) << NCNN_MOE_XSTATE_HI16_ZMM_BIT);
    return (enabled_xstate & avx512_state_mask) == avx512_state_mask && (feature_ebx & required_ebx) == required_ebx;
}
#endif

static KernelDispatch select_kernel() noexcept
{
    std::array<KernelDispatch, 4> candidates = {};
    size_t candidate_count = 0;
    candidates[candidate_count++] = {};
#if defined(__aarch64__) || defined(_M_ARM64)
    candidates[candidate_count++] = {MxFp4KernelKind::ArmNeon, neon_dot, neon_gemm_row, neon_matmul_rows2, neon_matmul_row_pairs};
#if defined(NCNN_MOE_ARM_SVE2_KERNEL)
    if ((detect_cpu_isa_capabilities().flags & CpuIsaArmSve2) != 0)
    {
        candidates[candidate_count++] = {MxFp4KernelKind::ArmSve2, sve2_mxfp4_dot, neon_gemm_row, sve2_mxfp4_matmul_rows2, sve2_mxfp4_matmul_row_pairs};
    }
#endif
#elif defined(NCNN_MOE_MSVC_X86_SIMD)
    if (msvc_cpu_supports_avx(true))
    {
        candidates[candidate_count++] = {MxFp4KernelKind::X86Avx512, msvc_avx512_mxfp4_dot, msvc_avx512_mxfp4_gemm_row, msvc_avx512_mxfp4_matmul_rows2,
                                         msvc_avx512_mxfp4_matmul_row_pairs};
    }
    if (msvc_cpu_supports_avx(false))
    {
        candidates[candidate_count++] = {
            MxFp4KernelKind::X86Avx2,
            msvc_avx2_mxfp4_dot,
            msvc_avx2_mxfp4_gemm_row,
            msvc_avx2_mxfp4_matmul_rows2,
            msvc_avx2_mxfp4_matmul_row_pairs,
        };
    }
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("ssse3")
        && __builtin_cpu_supports("fma"))
    {
        candidates[candidate_count++] = {MxFp4KernelKind::X86Avx512, avx512_dot, avx512_gemm_row, avx512_matmul_rows2, avx512_matmul_row_pairs};
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("ssse3") && __builtin_cpu_supports("fma"))
    {
        candidates[candidate_count++] = {MxFp4KernelKind::X86Avx2, avx2_dot, avx2_gemm_row, avx2_matmul_rows2, avx2_matmul_row_pairs};
    }
#endif

    const char* override_name = nullptr;
#if defined(_MSC_VER)
    std::array<char, 16> override_storage = {};
    size_t override_length = 0;
    if (getenv_s(&override_length, override_storage.data(), override_storage.size(), "NCNN_MOE_MXFP4_KERNEL") == 0 && override_length > 1
        && override_length <= override_storage.size())
    {
        override_name = override_storage.data();
    }
#else
    override_name = std::getenv("NCNN_MOE_MXFP4_KERNEL");
#endif
    if (override_name && override_name[0] != '\0')
    {
        for (size_t index = 0; index < candidate_count; ++index)
        {
            const bool selected = (std::strcmp(override_name, "scalar") == 0 && candidates[index].kind == MxFp4KernelKind::Scalar)
                                  || (std::strcmp(override_name, "neon") == 0 && candidates[index].kind == MxFp4KernelKind::ArmNeon)
                                  || (std::strcmp(override_name, "sve2") == 0 && candidates[index].kind == MxFp4KernelKind::ArmSve2)
                                  || (std::strcmp(override_name, "avx2") == 0 && candidates[index].kind == MxFp4KernelKind::X86Avx2)
                                  || (std::strcmp(override_name, "avx512") == 0 && candidates[index].kind == MxFp4KernelKind::X86Avx512);
            if (selected)
                return candidates[index];
        }
    }
    if (candidate_count == 1)
        return candidates[0];

    static constexpr uint32_t benchmark_blocks = 64;
    static constexpr uint32_t benchmark_repeats = 96;
    std::array<uint8_t, benchmark_blocks * 16 * 2> packed = {};
    std::array<uint8_t, benchmark_blocks * 2> scales = {};
    std::array<float, benchmark_blocks * 32> input = {};
    for (size_t index = 0; index < packed.size(); ++index)
    {
        packed[index] = static_cast<uint8_t>(((index * 5 + 1) & 0x0f) | (((index * 7 + 3) & 0x0f) << 4));
    }
    for (size_t index = 0; index < scales.size(); ++index)
        scales[index] = static_cast<uint8_t>(124 + index % 7);
    for (size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(static_cast<int>(index % 31) - 15) * 0.03125f;
    }

    std::array<int64_t, 3> elapsed = {};
    volatile float sink = 0.0f;
    for (uint32_t round = 0; round < 5; ++round)
    {
        for (size_t order = 0; order < candidate_count; ++order)
        {
            const size_t index = round % 2 == 0 ? order : candidate_count - order - 1;
            float first = 0.0f;
            float second = 0.0f;
            const auto started = std::chrono::steady_clock::now();
            for (uint32_t repeat = 0; repeat < benchmark_repeats; ++repeat)
            {
                candidates[index].matmul_rows2(packed.data(), scales.data(), packed.data() + benchmark_blocks * 16, scales.data() + benchmark_blocks,
                                               benchmark_blocks, input.data(), input.size(), 1, &first, 1, &second, 1);
                sink = first + second;
            }
            elapsed[index] += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
        }
    }
    (void)sink;
    size_t selected_index = 0;
    for (size_t index = 1; index < candidate_count; ++index)
    {
        if (elapsed[index] < elapsed[selected_index])
            selected_index = index;
    }
    return candidates[selected_index];
}

static const KernelDispatch& kernel_dispatch() noexcept
{
    static const KernelDispatch dispatch = select_kernel();
    return dispatch;
}

MxFp4KernelKind mxfp4_kernel_kind() noexcept
{
    return kernel_dispatch().kind;
}

const char* mxfp4_kernel_name() noexcept
{
    switch (kernel_dispatch().kind)
    {
    case MxFp4KernelKind::ArmNeon: return "arm-neon";
    case MxFp4KernelKind::ArmSve2: return "arm-sve2";
    case MxFp4KernelKind::X86Avx2: return "x86-avx2-fma";
    case MxFp4KernelKind::X86Avx512: return "x86-avx512";
    case MxFp4KernelKind::Scalar: return "scalar";
    }
    return "scalar";
}

static int64_t measure_decode_group(bool grouped, bool parallel, const KernelDispatch& dispatch, const std::vector<uint8_t>& packed,
                                    const std::vector<uint8_t>& scales, uint32_t block_count, uint32_t pair_count, uint32_t repeats, size_t packed_row_bytes,
                                    const float* input, size_t input_count, std::vector<float>& first, std::vector<float>& second)
{
    const auto started = std::chrono::steady_clock::now();
    for (uint32_t repeat = 0; repeat < repeats; ++repeat)
    {
        if (grouped)
        {
            const int64_t group_count = pair_count / 2;
#pragma omp parallel for schedule(static) if (parallel)
            for (int64_t group = 0; group < group_count; ++group)
            {
                const uint32_t pair = static_cast<uint32_t>(group) * 2;
                const size_t first_row = static_cast<size_t>(pair) * 2;
                dispatch.matmul_row_pairs(packed.data() + first_row * packed_row_bytes, scales.data() + first_row * block_count, block_count, 2, input,
                                          input_count, 1, first.data() + pair, 1, 1, second.data() + pair, 1, 1);
            }
        }
        else
        {
            const int64_t group_count = pair_count;
#pragma omp parallel for schedule(static) if (parallel)
            for (int64_t group = 0; group < group_count; ++group)
            {
                const uint32_t pair = static_cast<uint32_t>(group);
                const size_t first_row = static_cast<size_t>(pair) * 2;
                dispatch.matmul_rows2(packed.data() + first_row * packed_row_bytes, scales.data() + first_row * block_count,
                                      packed.data() + (first_row + 1) * packed_row_bytes, scales.data() + (first_row + 1) * block_count, block_count, input,
                                      input_count, 1, first.data() + pair, 1, second.data() + pair, 1);
            }
        }
    }
    mxfp4_benchmark_sink = first.back() + second.back();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
}

static uint32_t select_decode_group_size() noexcept
{
    const char* override_name = nullptr;
#if defined(_MSC_VER)
    std::array<char, 8> override_storage = {};
    size_t override_length = 0;
    if (getenv_s(&override_length, override_storage.data(), override_storage.size(), "NCNN_MOE_MXFP4_DECODE_GROUP") == 0 && override_length > 1
        && override_length <= override_storage.size())
    {
        override_name = override_storage.data();
    }
#else
    override_name = std::getenv("NCNN_MOE_MXFP4_DECODE_GROUP");
#endif
    if (override_name)
    {
        if (std::strcmp(override_name, "1") == 0)
            return 1;
        if (std::strcmp(override_name, "2") == 0)
            return 2;
    }

    constexpr uint32_t block_count = 90;
    constexpr uint32_t pair_count = 128;
    constexpr uint32_t rounds = 5;
    constexpr uint32_t repeats = 8;
    const size_t packed_row_bytes = static_cast<size_t>(block_count) * 16;
    std::vector<uint8_t> packed(static_cast<size_t>(pair_count) * 2 * packed_row_bytes);
    std::vector<uint8_t> scales(static_cast<size_t>(pair_count) * 2 * block_count);
    std::array<float, block_count * 32> input = {};
    std::vector<float> first(pair_count);
    std::vector<float> second(pair_count);
    for (size_t index = 0; index < packed.size(); ++index)
    {
        packed[index] = static_cast<uint8_t>(((index * 5 + 1) & 0x0f) | (((index * 7 + 3) & 0x0f) << 4));
    }
    for (size_t index = 0; index < scales.size(); ++index)
        scales[index] = static_cast<uint8_t>(124 + index % 7);
    for (size_t index = 0; index < input.size(); ++index)
        input[index] = static_cast<float>(static_cast<int>(index % 31) - 15) * 0.03125f;

    const KernelDispatch& dispatch = kernel_dispatch();
    bool parallel = false;
#if defined(_OPENMP)
    parallel = omp_in_parallel() == 0 && omp_get_max_threads() > 1;
#endif
    int64_t scalar_pair_time = 0;
    int64_t grouped_pair_time = 0;
    for (uint32_t round = 0; round < rounds; ++round)
    {
        if (round % 2 == 0)
        {
            scalar_pair_time += measure_decode_group(false, parallel, dispatch, packed, scales, block_count, pair_count, repeats, packed_row_bytes,
                                                     input.data(), input.size(), first, second);
            grouped_pair_time += measure_decode_group(true, parallel, dispatch, packed, scales, block_count, pair_count, repeats, packed_row_bytes,
                                                      input.data(), input.size(), first, second);
        }
        else
        {
            grouped_pair_time += measure_decode_group(true, parallel, dispatch, packed, scales, block_count, pair_count, repeats, packed_row_bytes,
                                                      input.data(), input.size(), first, second);
            scalar_pair_time += measure_decode_group(false, parallel, dispatch, packed, scales, block_count, pair_count, repeats, packed_row_bytes,
                                                     input.data(), input.size(), first, second);
        }
    }
    // Five percent hysteresis prevents noisy decode-strategy changes.
    return grouped_pair_time < scalar_pair_time - scalar_pair_time / 20 ? 2 : 1;
}

uint32_t mxfp4_decode_row_pair_group_size() noexcept
{
    static const uint32_t selected = select_decode_group_size();
    return selected;
}

float mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
{
    return kernel_dispatch().function(packed, scales, block_count, input);
}

void mxfp4_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
                    float* output, size_t output_stride) noexcept
{
    kernel_dispatch().gemm_row(packed, scales, block_count, input, input_stride, token_count, output, output_stride);
}

void mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                        uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output, size_t first_output_stride,
                        float* second_output, size_t second_output_stride) noexcept
{
    kernel_dispatch().matmul_rows2(first_packed, first_scales, second_packed, second_scales, block_count, input, input_stride, token_count, first_output,
                                   first_output_stride, second_output, second_output_stride);
}

void mxfp4_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,
                            size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                            float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept
{
    kernel_dispatch().matmul_row_pairs(packed, scales, block_count, row_pair_count, input, input_stride, token_count, first_output, first_pair_stride,
                                       first_token_stride, second_output, second_pair_stride, second_token_stride);
}

} // namespace moe
} // namespace ncnn
