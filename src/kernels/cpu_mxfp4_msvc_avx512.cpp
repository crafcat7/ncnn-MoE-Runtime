#include "cpu_mxfp4_msvc.h"

#include <array>
#include <cmath>
#include <immintrin.h>

namespace ncnn {
namespace moe {

static const std::array<float, 256>& avx512_scale_table()
{
    static const std::array<float, 256> values = [] {
        std::array<float, 256> table = {};
        for (uint32_t index = 0; index < table.size(); ++index)
            table[index] = std::ldexp(1.0f, static_cast<int>(index) - 127);
        return table;
    }();
    return values;
}

static __m512 avx512_decode_half(
    __m128i values,
    __m128i indices) noexcept
{
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
        _mm_shuffle_epi8(values, indices)));
}

static void avx512_decode_block(
    const uint8_t* packed,
    __m512 decoded[2]) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i bytes = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(packed));
    const __m128i low = _mm_and_si128(bytes, nibble_mask);
    const __m128i high = _mm_and_si128(
        _mm_srli_epi16(bytes, 4), nibble_mask);
    decoded[0] = avx512_decode_half(
        value_table, _mm_unpacklo_epi8(low, high));
    decoded[1] = avx512_decode_half(
        value_table, _mm_unpackhi_epi8(low, high));
}

float msvc_avx512_mxfp4_dot(
    const uint8_t* packed,
    const uint8_t* scales,
    uint32_t block_count,
    const float* input) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx512_scale_table();
    __m512 total = _mm512_setzero_ps();
    for (uint32_t block_index = 0; block_index < block_count; ++block_index) {
        __m512 decoded[2];
        avx512_decode_block(
            packed + static_cast<size_t>(block_index) * 16,
            decoded);
        const float* input_block
            = input + static_cast<size_t>(block_index) * 32;
        const __m512 accumulator = _mm512_fmadd_ps(
            decoded[1],
            _mm512_loadu_ps(input_block + 16),
            _mm512_mul_ps(decoded[0], _mm512_loadu_ps(input_block)));
        total = _mm512_fmadd_ps(
            accumulator,
            _mm512_set1_ps(
                0.5f * scales_by_exponent[scales[block_index]]),
            total);
    }
    return _mm512_reduce_add_ps(total);
}

void msvc_avx512_mxfp4_gemm_row(
    const uint8_t* packed,
    const uint8_t* scales,
    uint32_t block_count,
    const float* input,
    size_t input_stride,
    size_t token_count,
    float* output,
    size_t output_stride) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx512_scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
        output[token_index * output_stride] = 0.0f;
    if (token_count == 1) {
        output[0] = msvc_avx512_mxfp4_dot(
            packed,
            scales,
            block_count,
            input);
        return;
    }

    for (uint32_t block_index = 0; block_index < block_count; ++block_index) {
        __m512 decoded[2];
        avx512_decode_block(
            packed + static_cast<size_t>(block_index) * 16,
            decoded);
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float scale
            = 0.5f * scales_by_exponent[scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index) {
            const float* token
                = input + token_index * input_stride + input_offset;
            const __m512 accumulator = _mm512_fmadd_ps(
                decoded[1],
                _mm512_loadu_ps(token + 16),
                _mm512_mul_ps(decoded[0], _mm512_loadu_ps(token)));
            output[token_index * output_stride]
                += _mm512_reduce_add_ps(accumulator) * scale;
        }
    }
}

void msvc_avx512_mxfp4_matmul_rows2(
    const uint8_t* first_packed,
    const uint8_t* first_scales,
    const uint8_t* second_packed,
    const uint8_t* second_scales,
    uint32_t block_count,
    const float* input,
    size_t input_stride,
    size_t token_count,
    float* first_output,
    size_t first_output_stride,
    float* second_output,
    size_t second_output_stride) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx512_scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index) {
        first_output[token_index * first_output_stride] = 0.0f;
        second_output[token_index * second_output_stride] = 0.0f;
    }
    if (token_count == 1) {
        __m512 first_total = _mm512_setzero_ps();
        __m512 second_total = _mm512_setzero_ps();
        for (uint32_t block_index = 0;
             block_index < block_count;
             ++block_index) {
            __m512 decoded_rows[2][2];
            avx512_decode_block(
                first_packed + static_cast<size_t>(block_index) * 16,
                decoded_rows[0]);
            avx512_decode_block(
                second_packed + static_cast<size_t>(block_index) * 16,
                decoded_rows[1]);
            const float* input_block
                = input + static_cast<size_t>(block_index) * 32;
            const __m512 input_low = _mm512_loadu_ps(input_block);
            const __m512 input_high
                = _mm512_loadu_ps(input_block + 16);
            const __m512 first_block = _mm512_fmadd_ps(
                decoded_rows[0][1],
                input_high,
                _mm512_mul_ps(decoded_rows[0][0], input_low));
            const __m512 second_block = _mm512_fmadd_ps(
                decoded_rows[1][1],
                input_high,
                _mm512_mul_ps(decoded_rows[1][0], input_low));
            first_total = _mm512_fmadd_ps(
                first_block,
                _mm512_set1_ps(
                    0.5f
                    * scales_by_exponent[first_scales[block_index]]),
                first_total);
            second_total = _mm512_fmadd_ps(
                second_block,
                _mm512_set1_ps(
                    0.5f
                    * scales_by_exponent[second_scales[block_index]]),
                second_total);
        }
        first_output[0] = _mm512_reduce_add_ps(first_total);
        second_output[0] = _mm512_reduce_add_ps(second_total);
        return;
    }

    for (uint32_t block_index = 0; block_index < block_count; ++block_index) {
        __m512 decoded_rows[2][2];
        avx512_decode_block(
            first_packed + static_cast<size_t>(block_index) * 16,
            decoded_rows[0]);
        avx512_decode_block(
            second_packed + static_cast<size_t>(block_index) * 16,
            decoded_rows[1]);
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float first_scale
            = 0.5f * scales_by_exponent[first_scales[block_index]];
        const float second_scale
            = 0.5f * scales_by_exponent[second_scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index) {
            const float* token
                = input + token_index * input_stride + input_offset;
            const __m512 input_low = _mm512_loadu_ps(token);
            const __m512 input_high = _mm512_loadu_ps(token + 16);
            const __m512 first_accumulator = _mm512_fmadd_ps(
                decoded_rows[0][1],
                input_high,
                _mm512_mul_ps(decoded_rows[0][0], input_low));
            const __m512 second_accumulator = _mm512_fmadd_ps(
                decoded_rows[1][1],
                input_high,
                _mm512_mul_ps(decoded_rows[1][0], input_low));
            first_output[token_index * first_output_stride]
                += _mm512_reduce_add_ps(first_accumulator) * first_scale;
            second_output[token_index * second_output_stride]
                += _mm512_reduce_add_ps(second_accumulator) * second_scale;
        }
    }
}

} // namespace moe
} // namespace ncnn
