#include "cpu_mxfp4_msvc.h"

#include <bit>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <vector>

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
    return true;
}

static bool use_avx512_batch2_row_group() noexcept
{
    return avx512_batch2_enabled();
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
    __m512 accumulator0 = _mm512_setzero_ps();
    __m512 accumulator1 = _mm512_setzero_ps();
    __m512 accumulator2 = _mm512_setzero_ps();
    __m512 accumulator3 = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 64 <= count; index += 64)
    {
        const __m256i packed0 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights + index));
        const __m256i packed1 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights + index + 16));
        const __m256i packed2 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights + index + 32));
        const __m256i packed3 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weights + index + 48));
        const __m512 values0 = _mm512_castsi512_ps(
            _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(packed0),
                16));
        const __m512 values1 = _mm512_castsi512_ps(
            _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(packed1),
                16));
        const __m512 values2 = _mm512_castsi512_ps(
            _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(packed2),
                16));
        const __m512 values3 = _mm512_castsi512_ps(
            _mm512_slli_epi32(
                _mm512_cvtepu16_epi32(packed3),
                16));
        accumulator0 = _mm512_fmadd_ps(
            values0,
            _mm512_loadu_ps(input + index),
            accumulator0);
        accumulator1 = _mm512_fmadd_ps(
            values1,
            _mm512_loadu_ps(input + index + 16),
            accumulator1);
        accumulator2 = _mm512_fmadd_ps(
            values2,
            _mm512_loadu_ps(input + index + 32),
            accumulator2);
        accumulator3 = _mm512_fmadd_ps(
            values3,
            _mm512_loadu_ps(input + index + 48),
            accumulator3);
    }
    __m512 accumulator = _mm512_add_ps(
        _mm512_add_ps(accumulator0, accumulator1),
        _mm512_add_ps(accumulator2, accumulator3));
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

void msvc_avx512_float_to_bfloat16_array(
    uint16_t* output,
    const float* input,
    uint32_t count) noexcept
{
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m256i packed = (__m256i)_mm512_cvtneps_pbh(
            _mm512_loadu_ps(input + index));
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(output + index),
            packed);
    }
    for (; index < count; ++index)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, input + index, sizeof(bits));
        const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
        output[index] = static_cast<uint16_t>((bits + rounding) >> 16);
    }
}

float msvc_avx512_bfloat16_pair_dot(
    const uint16_t* left,
    const uint16_t* right,
    uint32_t count) noexcept
{
    __m512 accumulator = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 32 <= count; index += 32)
    {
        accumulator = _mm512_dpbf16_ps(
            accumulator,
            (__m512bh)_mm512_loadu_si512(left + index),
            (__m512bh)_mm512_loadu_si512(right + index));
    }
    float sum = _mm512_reduce_add_ps(accumulator);
    for (; index < count; ++index)
    {
        const float left_value =
            std::bit_cast<float>(static_cast<uint32_t>(left[index]) << 16);
        const float right_value =
            std::bit_cast<float>(static_cast<uint32_t>(right[index]) << 16);
        sum += left_value * right_value;
    }
    return sum;
}

void msvc_avx512_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
{
    const __m512 scale_vector = _mm512_set1_ps(scale);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + index));
        const __m512 values = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
        _mm512_storeu_ps(output + index, _mm512_fmadd_ps(values, scale_vector, _mm512_loadu_ps(output + index)));
    }
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(input[index]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        output[index] += scale * value;
    }
}

static __forceinline void avx512_bfloat16_linear_tile4x4(const uint16_t* weights,
                                                         const uint16_t* input,
                                                         size_t input_stride,
                                                         uint32_t input_columns,
                                                         float* output,
                                                         size_t output_stride,
                                                         size_t valid_rows) noexcept
{
    __m512 sum00 = _mm512_setzero_ps();
    __m512 sum01 = _mm512_setzero_ps();
    __m512 sum02 = _mm512_setzero_ps();
    __m512 sum03 = _mm512_setzero_ps();
    __m512 sum10 = _mm512_setzero_ps();
    __m512 sum11 = _mm512_setzero_ps();
    __m512 sum12 = _mm512_setzero_ps();
    __m512 sum13 = _mm512_setzero_ps();
    __m512 sum20 = _mm512_setzero_ps();
    __m512 sum21 = _mm512_setzero_ps();
    __m512 sum22 = _mm512_setzero_ps();
    __m512 sum23 = _mm512_setzero_ps();
    __m512 sum30 = _mm512_setzero_ps();
    __m512 sum31 = _mm512_setzero_ps();
    __m512 sum32 = _mm512_setzero_ps();
    __m512 sum33 = _mm512_setzero_ps();
    for (uint32_t column = 0; column < input_columns; column += 32)
    {
        const __m512bh input0 = (__m512bh)_mm512_loadu_si512(input + column);
        const __m512bh input1 = (__m512bh)_mm512_loadu_si512(input + input_stride + column);
        const __m512bh input2 = (__m512bh)_mm512_loadu_si512(input + input_stride * 2 + column);
        const __m512bh input3 = (__m512bh)_mm512_loadu_si512(input + input_stride * 3 + column);
        const __m512bh weight0 = (__m512bh)_mm512_loadu_si512(weights + column);
        const __m512bh weight1 = (__m512bh)_mm512_loadu_si512(weights + input_columns + column);
        const __m512bh weight2 = (__m512bh)_mm512_loadu_si512(weights + static_cast<size_t>(input_columns) * 2 + column);
        const __m512bh weight3 = (__m512bh)_mm512_loadu_si512(weights + static_cast<size_t>(input_columns) * 3 + column);
        sum00 = _mm512_dpbf16_ps(sum00, input0, weight0);
        sum01 = _mm512_dpbf16_ps(sum01, input0, weight1);
        sum02 = _mm512_dpbf16_ps(sum02, input0, weight2);
        sum03 = _mm512_dpbf16_ps(sum03, input0, weight3);
        sum10 = _mm512_dpbf16_ps(sum10, input1, weight0);
        sum11 = _mm512_dpbf16_ps(sum11, input1, weight1);
        sum12 = _mm512_dpbf16_ps(sum12, input1, weight2);
        sum13 = _mm512_dpbf16_ps(sum13, input1, weight3);
        sum20 = _mm512_dpbf16_ps(sum20, input2, weight0);
        sum21 = _mm512_dpbf16_ps(sum21, input2, weight1);
        sum22 = _mm512_dpbf16_ps(sum22, input2, weight2);
        sum23 = _mm512_dpbf16_ps(sum23, input2, weight3);
        sum30 = _mm512_dpbf16_ps(sum30, input3, weight0);
        sum31 = _mm512_dpbf16_ps(sum31, input3, weight1);
        sum32 = _mm512_dpbf16_ps(sum32, input3, weight2);
        sum33 = _mm512_dpbf16_ps(sum33, input3, weight3);
    }
    output[0] = _mm512_reduce_add_ps(sum00);
    output[1] = _mm512_reduce_add_ps(sum01);
    output[2] = _mm512_reduce_add_ps(sum02);
    output[3] = _mm512_reduce_add_ps(sum03);
    if (valid_rows > 1)
    {
        output[output_stride] = _mm512_reduce_add_ps(sum10);
        output[output_stride + 1] = _mm512_reduce_add_ps(sum11);
        output[output_stride + 2] = _mm512_reduce_add_ps(sum12);
        output[output_stride + 3] = _mm512_reduce_add_ps(sum13);
    }
    if (valid_rows > 2)
    {
        output[output_stride * 2] = _mm512_reduce_add_ps(sum20);
        output[output_stride * 2 + 1] = _mm512_reduce_add_ps(sum21);
        output[output_stride * 2 + 2] = _mm512_reduce_add_ps(sum22);
        output[output_stride * 2 + 3] = _mm512_reduce_add_ps(sum23);
    }
    if (valid_rows > 3)
    {
        output[output_stride * 3] = _mm512_reduce_add_ps(sum30);
        output[output_stride * 3 + 1] = _mm512_reduce_add_ps(sum31);
        output[output_stride * 3 + 2] = _mm512_reduce_add_ps(sum32);
        output[output_stride * 3 + 3] = _mm512_reduce_add_ps(sum33);
    }
}

static __forceinline void avx512_bfloat16_linear_tile1x4(
    const uint16_t* weights,
    const uint16_t* input,
    uint32_t input_columns,
    float* output) noexcept
{
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    for (uint32_t column = 0; column < input_columns; column += 32)
    {
        const __m512bh input_values =
            (__m512bh)_mm512_loadu_si512(input + column);
        const __m512bh weight0 =
            (__m512bh)_mm512_loadu_si512(weights + column);
        const __m512bh weight1 = (__m512bh)_mm512_loadu_si512(
            weights + input_columns + column);
        const __m512bh weight2 = (__m512bh)_mm512_loadu_si512(
            weights + static_cast<size_t>(input_columns) * 2 + column);
        const __m512bh weight3 = (__m512bh)_mm512_loadu_si512(
            weights + static_cast<size_t>(input_columns) * 3 + column);
        sum0 = _mm512_dpbf16_ps(sum0, input_values, weight0);
        sum1 = _mm512_dpbf16_ps(sum1, input_values, weight1);
        sum2 = _mm512_dpbf16_ps(sum2, input_values, weight2);
        sum3 = _mm512_dpbf16_ps(sum3, input_values, weight3);
    }
    output[0] = _mm512_reduce_add_ps(sum0);
    output[1] = _mm512_reduce_add_ps(sum1);
    output[2] = _mm512_reduce_add_ps(sum2);
    output[3] = _mm512_reduce_add_ps(sum3);
}

void msvc_avx512_bfloat16_batched_linear(const uint16_t* weights,
                                         const float* input,
                                         size_t input_stride,
                                         size_t token_count,
                                         uint32_t output_columns,
                                         uint32_t input_columns,
                                         float* output,
                                         size_t output_stride,
                                         int thread_count,
                                         std::vector<uint16_t>& packed_input)
{
    const size_t padded_token_count = token_count == 1
        ? size_t{1}
        : (token_count + 3) & ~size_t{3};
    packed_input.resize(padded_token_count * input_columns);
    for (size_t token = 0; token < token_count; ++token)
    {
        const float* source = input + token * input_stride;
        uint16_t* destination = packed_input.data() + token * input_columns;
        for (uint32_t column = 0; column < input_columns; column += 16)
        {
            const __m256i packed = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(source + column));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + column), packed);
        }
    }
    std::fill(packed_input.begin() + token_count * input_columns,
              packed_input.end(),
              uint16_t{0});

    const uint16_t* packed_input_data = packed_input.data();
    const int64_t output_groups = output_columns / 4;
    if (token_count == 1)
    {
#pragma omp parallel for num_threads(thread_count) if (thread_count > 1)
        for (int64_t output_group = 0; output_group < output_groups;
             ++output_group)
        {
            const uint32_t first_output =
                static_cast<uint32_t>(output_group) * 4;
            avx512_bfloat16_linear_tile1x4(
                weights + static_cast<size_t>(first_output) * input_columns,
                packed_input_data,
                input_columns,
                output + first_output);
        }
        return;
    }
#pragma omp parallel for num_threads(thread_count) if (thread_count > 1)
    for (int64_t output_group = 0; output_group < output_groups; ++output_group)
    {
        const uint32_t first_output = static_cast<uint32_t>(output_group) * 4;
        const uint16_t* group_weights = weights + static_cast<size_t>(first_output) * input_columns;
        for (size_t token = 0; token < token_count; token += 4)
        {
            const size_t valid_rows = std::min<size_t>(4, token_count - token);
            avx512_bfloat16_linear_tile4x4(group_weights,
                                           packed_input_data + token * input_columns,
                                           input_columns,
                                           input_columns,
                                           output + token * output_stride + first_output,
                                           output_stride,
                                           valid_rows);
        }
    }
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
