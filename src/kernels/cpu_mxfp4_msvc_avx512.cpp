#include "cpu_mxfp4_msvc.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

namespace ncnn {
namespace moe {

static std::array<float, 256> make_avx512_scale_table()
{
    std::array<float, 256> table = {};
    for (uint32_t index = 0; index < table.size(); ++index)
        table[index] = std::ldexp(1.0f, static_cast<int>(index) - 127);
    return table;
}

static const std::array<float, 256>& avx512_scale_table()
{
    static const std::array<float, 256> values = make_avx512_scale_table();
    return values;
}

static bool avx512_batch2_enabled() noexcept
{
    char value[8] = {};
    size_t length = 0;
    if (getenv_s(&length, value, sizeof(value), "NCNN_MOE_MXFP4_BATCH2_ROW_GROUP") == 0 && length > 1)
    {
        return value[0] != '0';
    }
    return true;
}

static bool use_avx512_batch2_row_group() noexcept
{
    static const bool enabled = avx512_batch2_enabled();
    return enabled;
}

static __m512 avx512_decode_half(__m128i indices) noexcept
{
    const __m512 values = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f);
    return _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(indices), values);
}

static void avx512_decode_block(const uint8_t* packed, __m512 decoded[2]) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i low = _mm_and_si128(bytes, nibble_mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
    decoded[0] = avx512_decode_half(_mm_unpacklo_epi8(low, high));
    decoded[1] = avx512_decode_half(_mm_unpackhi_epi8(low, high));
}

static __forceinline void avx512_accumulate_contiguous_rows4_block(const uint8_t* packed, const uint8_t* scales, size_t packed_row_bytes, uint32_t block_count,
                                                                   size_t first_row, uint32_t block_index, const float* input,
                                                                   const std::array<float, 256>& scales_by_exponent, __m512 (&totals)[4]) noexcept
{
    const float* input_block = input + static_cast<size_t>(block_index) * 32;
    const __m512 input_low = _mm512_loadu_ps(input_block);
    const __m512 input_high = _mm512_loadu_ps(input_block + 16);
    for (size_t row = 0; row < 4; ++row)
    {
        const size_t matrix_row = first_row + row;
        __m512 decoded[2];
        avx512_decode_block(packed + matrix_row * packed_row_bytes + static_cast<size_t>(block_index) * 16, decoded);
        const __m512 block = _mm512_fmadd_ps(decoded[1], input_high, _mm512_mul_ps(decoded[0], input_low));
        totals[row] = _mm512_fmadd_ps(block, _mm512_set1_ps(0.5f * scales_by_exponent[scales[matrix_row * block_count + block_index]]), totals[row]);
    }
}

static __forceinline void avx512_accumulate_contiguous_rows4_tokens2_block(const uint8_t* packed, const uint8_t* scales, size_t packed_row_bytes,
                                                                           uint32_t block_count, size_t first_row, uint32_t block_index, const float* input,
                                                                           size_t input_stride, const std::array<float, 256>& scales_by_exponent,
                                                                           __m512 (&totals)[8]) noexcept
{
    const size_t input_offset = static_cast<size_t>(block_index) * 32;
    const float* first_input = input + input_offset;
    const float* second_input = input + input_stride + input_offset;
    const __m512 first_input_low = _mm512_loadu_ps(first_input);
    const __m512 first_input_high = _mm512_loadu_ps(first_input + 16);
    const __m512 second_input_low = _mm512_loadu_ps(second_input);
    const __m512 second_input_high = _mm512_loadu_ps(second_input + 16);
    for (size_t row = 0; row < 4; ++row)
    {
        const size_t matrix_row = first_row + row;
        __m512 decoded[2];
        avx512_decode_block(packed + matrix_row * packed_row_bytes + static_cast<size_t>(block_index) * 16, decoded);
        const __m512 scale = _mm512_set1_ps(0.5f * scales_by_exponent[scales[matrix_row * block_count + block_index]]);
        totals[row] = _mm512_fmadd_ps(_mm512_fmadd_ps(decoded[1], first_input_high, _mm512_mul_ps(decoded[0], first_input_low)), scale, totals[row]);
        totals[4 + row] = _mm512_fmadd_ps(_mm512_fmadd_ps(decoded[1], second_input_high, _mm512_mul_ps(decoded[0], second_input_low)), scale, totals[4 + row]);
    }
}

float msvc_avx512_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
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

float msvc_avx512_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx512_scale_table();
    __m512 total = _mm512_setzero_ps();
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        __m512 decoded[2];
        avx512_decode_block(packed + static_cast<size_t>(block_index) * 16, decoded);
        const float* input_block = input + static_cast<size_t>(block_index) * 32;
        const __m512 accumulator = _mm512_fmadd_ps(decoded[1], _mm512_loadu_ps(input_block + 16), _mm512_mul_ps(decoded[0], _mm512_loadu_ps(input_block)));
        total = _mm512_fmadd_ps(accumulator, _mm512_set1_ps(0.5f * scales_by_exponent[scales[block_index]]), total);
    }
    return _mm512_reduce_add_ps(total);
}

void msvc_avx512_mxfp4_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
                                float* output, size_t output_stride) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx512_scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
        output[token_index * output_stride] = 0.0f;
    if (token_count == 1)
    {
        output[0] = msvc_avx512_mxfp4_dot(packed, scales, block_count, input);
        return;
    }
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        __m512 decoded[2];
        avx512_decode_block(packed + static_cast<size_t>(block_index) * 16, decoded);
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float scale = 0.5f * scales_by_exponent[scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index)
        {
            const float* token = input + token_index * input_stride + input_offset;
            const __m512 accumulator = _mm512_fmadd_ps(decoded[1], _mm512_loadu_ps(token + 16), _mm512_mul_ps(decoded[0], _mm512_loadu_ps(token)));
            output[token_index * output_stride] += _mm512_reduce_add_ps(accumulator) * scale;
        }
    }
}

void msvc_avx512_mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                                    uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output,
                                    size_t first_output_stride, float* second_output, size_t second_output_stride) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx512_scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
    {
        first_output[token_index * first_output_stride] = 0.0f;
        second_output[token_index * second_output_stride] = 0.0f;
    }
    if (token_count == 1)
    {
        __m512 first_total = _mm512_setzero_ps();
        __m512 second_total = _mm512_setzero_ps();
        for (uint32_t block_index = 0; block_index < block_count; ++block_index)
        {
            __m512 decoded_rows[2][2];
            avx512_decode_block(first_packed + static_cast<size_t>(block_index) * 16, decoded_rows[0]);
            avx512_decode_block(second_packed + static_cast<size_t>(block_index) * 16, decoded_rows[1]);
            const float* input_block = input + static_cast<size_t>(block_index) * 32;
            const __m512 input_low = _mm512_loadu_ps(input_block);
            const __m512 input_high = _mm512_loadu_ps(input_block + 16);
            const __m512 first_block = _mm512_fmadd_ps(decoded_rows[0][1], input_high, _mm512_mul_ps(decoded_rows[0][0], input_low));
            const __m512 second_block = _mm512_fmadd_ps(decoded_rows[1][1], input_high, _mm512_mul_ps(decoded_rows[1][0], input_low));
            first_total = _mm512_fmadd_ps(first_block, _mm512_set1_ps(0.5f * scales_by_exponent[first_scales[block_index]]), first_total);
            second_total = _mm512_fmadd_ps(second_block, _mm512_set1_ps(0.5f * scales_by_exponent[second_scales[block_index]]), second_total);
        }
        first_output[0] = _mm512_reduce_add_ps(first_total);
        second_output[0] = _mm512_reduce_add_ps(second_total);
        return;
    }
    if (token_count <= 4)
    {
        __m512 first_totals[4] = {
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
        };
        __m512 second_totals[4] = {
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
        };
        for (uint32_t block_index = 0; block_index < block_count; ++block_index)
        {
            __m512 decoded_rows[2][2];
            avx512_decode_block(first_packed + static_cast<size_t>(block_index) * 16, decoded_rows[0]);
            avx512_decode_block(second_packed + static_cast<size_t>(block_index) * 16, decoded_rows[1]);
            const __m512 first_scale = _mm512_set1_ps(0.5f * scales_by_exponent[first_scales[block_index]]);
            const __m512 second_scale = _mm512_set1_ps(0.5f * scales_by_exponent[second_scales[block_index]]);
            const size_t input_offset = static_cast<size_t>(block_index) * 32;
            for (size_t token_index = 0; token_index < token_count; ++token_index)
            {
                const float* token = input + token_index * input_stride + input_offset;
                const __m512 input_low = _mm512_loadu_ps(token);
                const __m512 input_high = _mm512_loadu_ps(token + 16);
                const __m512 first_block = _mm512_fmadd_ps(decoded_rows[0][1], input_high, _mm512_mul_ps(decoded_rows[0][0], input_low));
                const __m512 second_block = _mm512_fmadd_ps(decoded_rows[1][1], input_high, _mm512_mul_ps(decoded_rows[1][0], input_low));
                first_totals[token_index] = _mm512_fmadd_ps(first_block, first_scale, first_totals[token_index]);
                second_totals[token_index] = _mm512_fmadd_ps(second_block, second_scale, second_totals[token_index]);
            }
        }
        for (size_t token_index = 0; token_index < token_count; ++token_index)
        {
            first_output[token_index * first_output_stride] = _mm512_reduce_add_ps(first_totals[token_index]);
            second_output[token_index * second_output_stride] = _mm512_reduce_add_ps(second_totals[token_index]);
        }
        return;
    }

    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        __m512 decoded_rows[2][2];
        avx512_decode_block(first_packed + static_cast<size_t>(block_index) * 16, decoded_rows[0]);
        avx512_decode_block(second_packed + static_cast<size_t>(block_index) * 16, decoded_rows[1]);
        const size_t input_offset = static_cast<size_t>(block_index) * 32;
        const float first_scale = 0.5f * scales_by_exponent[first_scales[block_index]];
        const float second_scale = 0.5f * scales_by_exponent[second_scales[block_index]];
        for (size_t token_index = 0; token_index < token_count; ++token_index)
        {
            const float* token = input + token_index * input_stride + input_offset;
            const __m512 input_low = _mm512_loadu_ps(token);
            const __m512 input_high = _mm512_loadu_ps(token + 16);
            const __m512 first_accumulator = _mm512_fmadd_ps(decoded_rows[0][1], input_high, _mm512_mul_ps(decoded_rows[0][0], input_low));
            const __m512 second_accumulator = _mm512_fmadd_ps(decoded_rows[1][1], input_high, _mm512_mul_ps(decoded_rows[1][0], input_low));
            first_output[token_index * first_output_stride] += _mm512_reduce_add_ps(first_accumulator) * first_scale;
            second_output[token_index * second_output_stride] += _mm512_reduce_add_ps(second_accumulator) * second_scale;
        }
    }
}

void msvc_avx512_mxfp4_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,
                                        size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                                        float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept
{
    const size_t packed_row_bytes = static_cast<size_t>(block_count) * 16;
    uint32_t pair = 0;
    if (token_count == 1)
    {
        const std::array<float, 256>& scales_by_exponent = avx512_scale_table();
        static constexpr uint32_t pairs_per_group = 2;
        for (; pair + pairs_per_group <= row_pair_count; pair += pairs_per_group)
        {
            __m512 totals[pairs_per_group * 2];
            __m512 alternate_totals[pairs_per_group * 2];
            for (__m512& total : totals)
                total = _mm512_setzero_ps();
            for (__m512& total : alternate_totals)
                total = _mm512_setzero_ps();
            const size_t first_group_row = static_cast<size_t>(pair) * 2;
            uint32_t block_index = 0;
            for (; block_index + 1 < block_count; block_index += 2)
            {
                avx512_accumulate_contiguous_rows4_block(packed, scales, packed_row_bytes, block_count, first_group_row, block_index, input, scales_by_exponent,
                                                         totals);
                avx512_accumulate_contiguous_rows4_block(packed, scales, packed_row_bytes, block_count, first_group_row, block_index + 1, input,
                                                         scales_by_exponent, alternate_totals);
            }
            if (block_index < block_count)
            {
                avx512_accumulate_contiguous_rows4_block(packed, scales, packed_row_bytes, block_count, first_group_row, block_index, input, scales_by_exponent,
                                                         totals);
            }
            for (uint32_t local_pair = 0; local_pair < pairs_per_group; ++local_pair)
            {
                first_output[static_cast<size_t>(pair + local_pair) * first_pair_stride] = _mm512_reduce_add_ps(
                    _mm512_add_ps(totals[local_pair * 2], alternate_totals[local_pair * 2]));
                second_output[static_cast<size_t>(pair + local_pair) * second_pair_stride] = _mm512_reduce_add_ps(
                    _mm512_add_ps(totals[local_pair * 2 + 1], alternate_totals[local_pair * 2 + 1]));
            }
        }
    }
    else if (token_count == 2 && use_avx512_batch2_row_group())
    {
        const std::array<float, 256>& scales_by_exponent = avx512_scale_table();
        static constexpr uint32_t pairs_per_group = 2;
        for (; pair + pairs_per_group <= row_pair_count; pair += pairs_per_group)
        {
            __m512 totals[pairs_per_group * 4];
            for (__m512& total : totals)
                total = _mm512_setzero_ps();
            const size_t first_group_row = static_cast<size_t>(pair) * 2;
            for (uint32_t block_index = 0; block_index < block_count; ++block_index)
            {
                avx512_accumulate_contiguous_rows4_tokens2_block(packed, scales, packed_row_bytes, block_count, first_group_row, block_index, input,
                                                                 input_stride, scales_by_exponent, totals);
            }
            for (size_t token_index = 0; token_index < 2; ++token_index)
            {
                const size_t token_offset = token_index * 4;
                for (uint32_t local_pair = 0; local_pair < pairs_per_group; ++local_pair)
                {
                    first_output[static_cast<size_t>(pair + local_pair) * first_pair_stride + token_index * first_token_stride] = _mm512_reduce_add_ps(totals[token_offset + local_pair * 2]);
                    second_output[static_cast<size_t>(pair + local_pair) * second_pair_stride + token_index * second_token_stride] = _mm512_reduce_add_ps(totals[token_offset + local_pair * 2 + 1]);
                }
            }
        }
    }
    for (; pair < row_pair_count; ++pair)
    {
        const size_t first_row = static_cast<size_t>(pair) * 2;
        msvc_avx512_mxfp4_matmul_rows2(packed + first_row * packed_row_bytes, scales + first_row * block_count, packed + (first_row + 1) * packed_row_bytes,
                                       scales + (first_row + 1) * block_count, block_count, input, input_stride, token_count,
                                       first_output + static_cast<size_t>(pair) * first_pair_stride, first_token_stride,
                                       second_output + static_cast<size_t>(pair) * second_pair_stride, second_token_stride);
    }
}

} // namespace moe
} // namespace ncnn
