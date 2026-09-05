#include "mxfp4_msvc.h"

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

void msvc_avx512_mxfp4_q8_quantize(
    const float* source,
    int8_t* values,
    float* scales,
    uint32_t columns) noexcept
{
    if (!source || !values || !scales || columns == 0)
        return;

    const __m512 zero = _mm512_setzero_ps();
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);
    const __m512 lower_bound = _mm512_set1_ps(-127.0f);
    const __m512 upper_bound = _mm512_set1_ps(127.0f);
    const uint32_t block_count = (columns + 31) / 32;
    for (uint32_t block = 0; block < block_count; ++block)
    {
        const uint32_t begin = block * 32;
        const uint32_t end = (columns < begin + 32) ? columns : begin + 32;
        const uint32_t count = end - begin;
        __m512 maximum_values = zero;
        uint32_t index = 0;
        for (; index + 16 <= count; index += 16)
        {
            const __m512 current = _mm512_loadu_ps(source + begin + index);
            maximum_values = _mm512_max_ps(
                maximum_values,
                _mm512_andnot_ps(sign_mask, current));
        }
        float maximum = _mm512_reduce_max_ps(maximum_values);
        for (; index < count; ++index)
            maximum = std::max(maximum, std::fabs(source[begin + index]));

        const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scales[block] = scale;
        const __m512 inverse_scale = _mm512_set1_ps(1.0f / scale);
        index = 0;
        for (; index + 16 <= count; index += 16)
        {
            __m512 normalized = _mm512_mul_ps(
                _mm512_loadu_ps(source + begin + index),
                inverse_scale);
            normalized = _mm512_max_ps(
                lower_bound,
                _mm512_min_ps(upper_bound, normalized));
            const __m512i quantized = _mm512_cvtps_epi32(normalized);
            alignas(64) int32_t quantized_values[16];
            _mm512_storeu_si512(quantized_values, quantized);
            for (uint32_t lane = 0; lane < 16; ++lane)
                values[begin + index + lane] = static_cast<int8_t>(quantized_values[lane]);
        }
        for (; index < count; ++index)
        {
            const float normalized = std::clamp(
                source[begin + index] / scale,
                -127.0f,
                127.0f);
            values[begin + index] = static_cast<int8_t>(std::lrintf(normalized));
        }
    }
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

static __forceinline int32_t avx512_reduce_8_epi32(__m256i values) noexcept
{
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(values),
        _mm256_extracti128_si256(values, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(sum);
}

// The persistent sidecar stores eight rows per MXFP4 block as
// [8 scales][8 bytes for row 0..7 of chunk 0][8 bytes for row 0..7 of chunk 1].
// Decode the two rows in each 128-bit lane together, then use madd_epi16 to
// produce two independent eight-term integer dot products.
static __forceinline void avx512_mxfp4_q8_packed_chunk_dot(
    const uint8_t* packed,
    const int8_t* input,
    int32_t (&dots)[8]) noexcept
{
    const __m512i bytes = _mm512_loadu_si512(reinterpret_cast<const void*>(packed));
    const __m512i nibble_mask = _mm512_set1_epi8(0x0f);
    const __m512i low = _mm512_and_si512(bytes, nibble_mask);
    const __m512i high = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), nibble_mask);
    const __m128i value_table_128 = _mm_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12);
    const __m512i value_table = _mm512_broadcast_i32x4(value_table_128);
    const __m512i even_rows = _mm512_shuffle_epi8(
        value_table,
        _mm512_unpacklo_epi8(low, high));
    const __m512i odd_rows = _mm512_shuffle_epi8(
        value_table,
        _mm512_unpackhi_epi8(low, high));
    const __m128i input_values_128 = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(input));
    const __m512i input_values = _mm512_cvtepi8_epi16(
        _mm256_set_m128i(input_values_128, input_values_128));

#define NCNN_MOE_AVX512_PACKED_ROW_LANE(lane)                                   \
    {                                                                           \
        const __m128i even_values = _mm512_extracti32x4_epi32(even_rows, lane); \
        const __m128i odd_values = _mm512_extracti32x4_epi32(odd_rows, lane);   \
        const __m256i values = _mm256_set_m128i(odd_values, even_values);       \
        const __m512i products = _mm512_madd_epi16(                             \
            _mm512_cvtepi8_epi16(values),                                       \
            input_values);                                                      \
        dots[(lane) * 2] = avx512_reduce_8_epi32(                               \
            _mm512_castsi512_si256(products));                                  \
        dots[(lane) * 2 + 1] = avx512_reduce_8_epi32(                           \
            _mm512_extracti64x4_epi64(products, 1));                            \
    }
    NCNN_MOE_AVX512_PACKED_ROW_LANE(0);
    NCNN_MOE_AVX512_PACKED_ROW_LANE(1);
    NCNN_MOE_AVX512_PACKED_ROW_LANE(2);
    NCNN_MOE_AVX512_PACKED_ROW_LANE(3);
#undef NCNN_MOE_AVX512_PACKED_ROW_LANE
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
        const float left_value = std::bit_cast<float>(static_cast<uint32_t>(left[index]) << 16);
        const float right_value = std::bit_cast<float>(static_cast<uint32_t>(right[index]) << 16);
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
        const __m512bh input_values = (__m512bh)_mm512_loadu_si512(input + column);
        const __m512bh weight0 = (__m512bh)_mm512_loadu_si512(weights + column);
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
            const uint32_t first_output = static_cast<uint32_t>(output_group) * 4;
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

void msvc_avx512_bfloat16_single_token_linear(const uint16_t* weights,
                                              const float* input,
                                              uint32_t output_columns,
                                              uint32_t input_columns,
                                              float* output,
                                              int thread_count)
{
    const int64_t output_groups = output_columns / 4;
#pragma omp parallel for num_threads(thread_count) if (thread_count > 1)
    for (int64_t output_group = 0; output_group < output_groups;
         ++output_group)
    {
        const uint32_t first_output = static_cast<uint32_t>(output_group) * 4;
        const uint16_t* group_weights = weights + static_cast<size_t>(first_output) * input_columns;
        __m512 accumulator00 = _mm512_setzero_ps();
        __m512 accumulator01 = _mm512_setzero_ps();
        __m512 accumulator02 = _mm512_setzero_ps();
        __m512 accumulator03 = _mm512_setzero_ps();
        __m512 accumulator10 = _mm512_setzero_ps();
        __m512 accumulator11 = _mm512_setzero_ps();
        __m512 accumulator12 = _mm512_setzero_ps();
        __m512 accumulator13 = _mm512_setzero_ps();
        __m512 accumulator20 = _mm512_setzero_ps();
        __m512 accumulator21 = _mm512_setzero_ps();
        __m512 accumulator22 = _mm512_setzero_ps();
        __m512 accumulator23 = _mm512_setzero_ps();
        __m512 accumulator30 = _mm512_setzero_ps();
        __m512 accumulator31 = _mm512_setzero_ps();
        __m512 accumulator32 = _mm512_setzero_ps();
        __m512 accumulator33 = _mm512_setzero_ps();
        uint32_t column = 0;
        for (; column + 64 <= input_columns; column += 64)
        {
            const __m512 input0 = _mm512_loadu_ps(input + column);
            const __m512 input1 = _mm512_loadu_ps(input + column + 16);
            const __m512 input2 = _mm512_loadu_ps(input + column + 32);
            const __m512 input3 = _mm512_loadu_ps(input + column + 48);
            const __m256i packed00 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + column));
            const __m256i packed01 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + column + 16));
            const __m256i packed02 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + column + 32));
            const __m256i packed03 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + column + 48));
            const __m256i packed10 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + input_columns + column));
            const __m256i packed11 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + input_columns + column + 16));
            const __m256i packed12 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + input_columns + column + 32));
            const __m256i packed13 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + input_columns + column + 48));
            const __m256i packed20 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column));
            const __m256i packed21 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column + 16));
            const __m256i packed22 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column + 32));
            const __m256i packed23 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column + 48));
            const __m256i packed30 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column));
            const __m256i packed31 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column + 16));
            const __m256i packed32 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column + 32));
            const __m256i packed33 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column + 48));
#define NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator, packed, input_values) \
    accumulator = _mm512_fmadd_ps(                                        \
        _mm512_castsi512_ps(_mm512_slli_epi32(                            \
            _mm512_cvtepu16_epi32(packed), 16)),                          \
        input_values,                                                     \
        accumulator)
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator00, packed00, input0);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator01, packed01, input1);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator02, packed02, input2);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator03, packed03, input3);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator10, packed10, input0);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator11, packed11, input1);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator12, packed12, input2);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator13, packed13, input3);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator20, packed20, input0);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator21, packed21, input1);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator22, packed22, input2);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator23, packed23, input3);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator30, packed30, input0);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator31, packed31, input1);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator32, packed32, input2);
            NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator33, packed33, input3);
#undef NCNN_MOE_BF16_SINGLE_TOKEN_FMA
        }
        __m512 sum0 = _mm512_add_ps(
            _mm512_add_ps(accumulator00, accumulator01),
            _mm512_add_ps(accumulator02, accumulator03));
        __m512 sum1 = _mm512_add_ps(
            _mm512_add_ps(accumulator10, accumulator11),
            _mm512_add_ps(accumulator12, accumulator13));
        __m512 sum2 = _mm512_add_ps(
            _mm512_add_ps(accumulator20, accumulator21),
            _mm512_add_ps(accumulator22, accumulator23));
        __m512 sum3 = _mm512_add_ps(
            _mm512_add_ps(accumulator30, accumulator31),
            _mm512_add_ps(accumulator32, accumulator33));
        for (; column + 16 <= input_columns; column += 16)
        {
            const __m512 input_values = _mm512_loadu_ps(input + column);
            const __m256i packed0 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + column));
            const __m256i packed1 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + input_columns + column));
            const __m256i packed2 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column));
            const __m256i packed3 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column));
            sum0 = _mm512_fmadd_ps(
                _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(packed0), 16)),
                input_values,
                sum0);
            sum1 = _mm512_fmadd_ps(
                _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(packed1), 16)),
                input_values,
                sum1);
            sum2 = _mm512_fmadd_ps(
                _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(packed2), 16)),
                input_values,
                sum2);
            sum3 = _mm512_fmadd_ps(
                _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(packed3), 16)),
                input_values,
                sum3);
        }
        output[first_output] = _mm512_reduce_add_ps(sum0);
        output[first_output + 1] = _mm512_reduce_add_ps(sum1);
        output[first_output + 2] = _mm512_reduce_add_ps(sum2);
        output[first_output + 3] = _mm512_reduce_add_ps(sum3);
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

void msvc_avx512_mxfp4_q8_packed_gemm(
    const uint8_t* packed,
    uint32_t row_count,
    uint32_t block_count,
    uint32_t tile_rows,
    const int8_t* input,
    size_t input_stride,
    const float* input_scales,
    size_t scale_stride,
    size_t token_count,
    float* output,
    size_t output_stride) noexcept
{
    if (!packed || !input || !input_scales || !output || row_count == 0
        || block_count == 0 || tile_rows != 8 || token_count == 0)
        return;

    constexpr size_t block_bytes = 8 * 17;
    const size_t group_stride = static_cast<size_t>(block_count) * block_bytes;
    const size_t group_count = (static_cast<size_t>(row_count) + 7) / 8;
    const std::array<float, 256>& scales_by_exponent = avx512_scale_table();
    for (size_t token = 0; token < token_count; ++token)
    {
        std::fill(
            output + token * output_stride,
            output + token * output_stride + row_count,
            0.0f);
    }

    for (size_t group = 0; group < group_count; ++group)
    {
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* packed_block = packed
                                          + group * group_stride
                                          + static_cast<size_t>(block) * block_bytes;
            const uint8_t* packed_values = packed_block + 8;
            for (size_t token = 0; token < token_count; ++token)
            {
                const int8_t* input_block = input
                                            + token * input_stride
                                            + static_cast<size_t>(block) * 32;
                const float input_scale = input_scales[token * scale_stride + block];
                for (uint32_t chunk = 0; chunk < 2; ++chunk)
                {
                    int32_t dots[8] = {};
                    avx512_mxfp4_q8_packed_chunk_dot(
                        packed_values + static_cast<size_t>(chunk) * 64,
                        input_block + chunk * 16,
                        dots);
                    for (uint32_t row = 0; row < 8; ++row)
                    {
                        const size_t matrix_row = group * 8 + row;
                        if (matrix_row >= row_count)
                            continue;
                        output[token * output_stride + matrix_row] += static_cast<float>(dots[row])
                                                                      * (0.5f * scales_by_exponent[packed_block[row]])
                                                                      * input_scale;
                    }
                }
            }
        }
    }
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
