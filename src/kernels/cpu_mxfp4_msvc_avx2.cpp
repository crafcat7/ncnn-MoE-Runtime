#include "cpu_mxfp4_msvc.h"

#include <array>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace ncnn {
namespace moe {

static std::array<float, 256> make_avx2_scale_table()
{
    std::array<float, 256> table = {};
    for (uint32_t index = 0; index < table.size(); ++index)
        table[index] = std::ldexp(1.0f, static_cast<int>(index) - 127);
    return table;
}

static const std::array<float, 256>& avx2_scale_table()
{
    static const std::array<float, 256> values = make_avx2_scale_table();
    return values;
}

static void avx2_decode_block(const uint8_t* packed, __m128i decoded[2]) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i low = _mm_and_si128(bytes, nibble_mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
    decoded[0] = _mm_shuffle_epi8(value_table, _mm_unpacklo_epi8(low, high));
    decoded[1] = _mm_shuffle_epi8(value_table, _mm_unpackhi_epi8(low, high));
}

static float avx2_horizontal_sum(__m256 values) noexcept
{
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(values), _mm256_extractf128_ps(values, 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

static int32_t avx2_horizontal_sum_epi32(__m256i values) noexcept
{
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(values),
        _mm256_extracti128_si256(values, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(sum);
}

float msvc_avx2_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    __m256 accumulator0 = _mm256_setzero_ps();
    __m256 accumulator1 = _mm256_setzero_ps();
    __m256 accumulator2 = _mm256_setzero_ps();
    __m256 accumulator3 = _mm256_setzero_ps();
    uint32_t index = 0;
    for (; index + 32 <= count; index += 32)
    {
        const __m128i packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
        const __m128i packed1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index + 8));
        const __m128i packed2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index + 16));
        const __m128i packed3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index + 24));
        accumulator0 = _mm256_fmadd_ps(
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed0), 16)),
            _mm256_loadu_ps(input + index),
            accumulator0);
        accumulator1 = _mm256_fmadd_ps(
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed1), 16)),
            _mm256_loadu_ps(input + index + 8),
            accumulator1);
        accumulator2 = _mm256_fmadd_ps(
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed2), 16)),
            _mm256_loadu_ps(input + index + 16),
            accumulator2);
        accumulator3 = _mm256_fmadd_ps(
            _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed3), 16)),
            _mm256_loadu_ps(input + index + 24),
            accumulator3);
    }
    float sum = avx2_horizontal_sum(
        _mm256_add_ps(
            _mm256_add_ps(accumulator0, accumulator1),
            _mm256_add_ps(accumulator2, accumulator3)));
    for (; index + 8 <= count; index += 8)
    {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
        const __m256 values = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        sum += avx2_horizontal_sum(
            _mm256_mul_ps(values, _mm256_loadu_ps(input + index)));
    }
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(weights[index]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[index];
    }
    return sum;
}

void msvc_avx2_float_to_bfloat16_array(
    uint16_t* output,
    const float* input,
    uint32_t count) noexcept
{
    const __m256i rounding_bias = _mm256_set1_epi32(0x7fff);
    const __m256i low_bit_mask = _mm256_set1_epi32(1);
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(input + index));
        const __m256i rounding = _mm256_add_epi32(
            rounding_bias,
            _mm256_and_si256(_mm256_srli_epi32(bits, 16), low_bit_mask));
        const __m256i high = _mm256_srli_epi32(
            _mm256_add_epi32(bits, rounding), 16);
        const __m128i shuffle_mask = _mm_setr_epi8(
            0, 1, 4, 5, 8, 9, 12, 13,
            -128, -128, -128, -128, -128, -128, -128, -128);
        const __m128i packed_low = _mm_shuffle_epi8(
            _mm256_castsi256_si128(high), shuffle_mask);
        const __m128i packed_high = _mm_shuffle_epi8(
            _mm256_extracti128_si256(high, 1), shuffle_mask);
        const __m128i packed = _mm_unpacklo_epi64(packed_low, packed_high);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + index), packed);
    }
    for (; index < count; ++index)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, input + index, sizeof(bits));
        const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
        output[index] = static_cast<uint16_t>((bits + rounding) >> 16);
    }
}

void msvc_avx2_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
{
    const __m256 scale_vector = _mm256_set1_ps(scale);
    uint32_t index = 0;
    for (; index + 8 <= count; index += 8)
    {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + index));
        const __m256 values = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        _mm256_storeu_ps(output + index, _mm256_fmadd_ps(values, scale_vector, _mm256_loadu_ps(output + index)));
    }
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(input[index]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        output[index] += scale * value;
    }
}

float msvc_avx2_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx2_scale_table();
    __m256 total = _mm256_setzero_ps();
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        __m128i decoded[2];
        avx2_decode_block(packed + static_cast<size_t>(block_index) * 16, decoded);
        const float* input_block = input + static_cast<size_t>(block_index) * 32;
        __m256 accumulator = _mm256_setzero_ps();
        for (uint32_t half = 0; half < 2; ++half)
        {
            accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(decoded[half])), _mm256_loadu_ps(input_block + half * 16), accumulator);
            accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(decoded[half], 8))),
                                          _mm256_loadu_ps(input_block + half * 16 + 8), accumulator);
        }
        total = _mm256_fmadd_ps(accumulator, _mm256_set1_ps(0.5f * scales_by_exponent[scales[block_index]]), total);
    }
    return avx2_horizontal_sum(total);
}

float msvc_avx2_mxfp4_q8_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count,
                             const int8_t* input, const float* input_scales) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const std::array<float, 256>& scales_by_exponent = avx2_scale_table();
    float sum = 0.0f;
    for (uint32_t block = 0; block < block_count; ++block)
    {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + static_cast<size_t>(block) * 16));
        const __m128i low = _mm_and_si128(bytes, nibble_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
        const __m128i decoded_low = _mm_shuffle_epi8(value_table, _mm_unpacklo_epi8(low, high));
        const __m128i decoded_high = _mm_shuffle_epi8(value_table, _mm_unpackhi_epi8(low, high));
        const int8_t* input_block = input + static_cast<size_t>(block) * 32;
        const __m256i input_low = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input_block)));
        const __m256i input_high = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input_block + 16)));
        const __m256i product_low = _mm256_madd_epi16(_mm256_cvtepi8_epi16(decoded_low), input_low);
        const __m256i product_high = _mm256_madd_epi16(_mm256_cvtepi8_epi16(decoded_high), input_high);
        sum += static_cast<float>(avx2_horizontal_sum_epi32(_mm256_add_epi32(product_low, product_high)))
               * (0.5f * scales_by_exponent[scales[block]])
               * input_scales[block];
    }
    return sum;
}

void msvc_avx2_mxfp4_q8_matmul_rows2(
    const uint8_t* first_packed,
    const uint8_t* first_scales,
    const uint8_t* second_packed,
    const uint8_t* second_scales,
    uint32_t block_count,
    const int8_t* input,
    size_t input_stride,
    const float* input_scales,
    size_t scale_stride,
    size_t token_count,
    float* first_output,
    size_t first_output_stride,
    float* second_output,
    size_t second_output_stride) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx2_scale_table();
    for (size_t token = 0; token < token_count; ++token)
    {
        first_output[token * first_output_stride] = 0.0f;
        second_output[token * second_output_stride] = 0.0f;
    }

    for (uint32_t block = 0; block < block_count; ++block)
    {
        __m128i first_decoded[2];
        __m128i second_decoded[2];
        avx2_decode_block(
            first_packed + static_cast<size_t>(block) * 16,
            first_decoded);
        avx2_decode_block(
            second_packed + static_cast<size_t>(block) * 16,
            second_decoded);
        const float first_weight_scale =
            0.5f * scales_by_exponent[first_scales[block]];
        const float second_weight_scale =
            0.5f * scales_by_exponent[second_scales[block]];
        const __m256i first_low_weights = _mm256_cvtepi8_epi16(first_decoded[0]);
        const __m256i first_high_weights = _mm256_cvtepi8_epi16(first_decoded[1]);
        const __m256i second_low_weights = _mm256_cvtepi8_epi16(second_decoded[0]);
        const __m256i second_high_weights = _mm256_cvtepi8_epi16(second_decoded[1]);
        for (size_t token = 0; token < token_count; ++token)
        {
            const int8_t* input_block =
                input + token * input_stride + static_cast<size_t>(block) * 32;
            const __m256i input_low = _mm256_cvtepi8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_block)));
            const __m256i input_high = _mm256_cvtepi8_epi16(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_block + 16)));
            const int32_t first_integer_sum = avx2_horizontal_sum_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(first_low_weights, input_low),
                    _mm256_madd_epi16(first_high_weights, input_high)));
            const int32_t second_integer_sum = avx2_horizontal_sum_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(second_low_weights, input_low),
                    _mm256_madd_epi16(second_high_weights, input_high)));
            const float input_scale =
                input_scales[token * scale_stride + block];
            first_output[token * first_output_stride] +=
                static_cast<float>(first_integer_sum)
                * first_weight_scale * input_scale;
            second_output[token * second_output_stride] +=
                static_cast<float>(second_integer_sum)
                * second_weight_scale * input_scale;
        }
    }
}

static int32_t msvc_avx2_mxfp4_q8_packed_chunk_dot(const uint8_t* packed, const int8_t* input) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed));
    const __m128i low = _mm_and_si128(bytes, nibble_mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
    const __m128i decoded = _mm_shuffle_epi8(value_table, _mm_unpacklo_epi8(low, high));
    const __m256i weights = _mm256_cvtepi8_epi16(decoded);
    const __m256i input_values = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input)));
    return avx2_horizontal_sum_epi32(_mm256_madd_epi16(weights, input_values));
}

void msvc_avx2_mxfp4_q8_packed_gemm(
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
    if (!packed || !input || !input_scales || !output || row_count == 0 || block_count == 0 || tile_rows != 8 || token_count == 0)
        return;

    const size_t block_bytes = 8 * 17;
    const size_t group_stride = static_cast<size_t>(block_count) * block_bytes;
    const std::array<float, 256>& scales_by_exponent = avx2_scale_table();
    const size_t group_count = (static_cast<size_t>(row_count) + 7) / 8;
    if (token_count == 1)
    {
        for (size_t group = 0; group < group_count; ++group)
        {
            float accumulators[8] = {};
            for (uint32_t block = 0; block < block_count; ++block)
            {
                const uint8_t* packed_block = packed + group * group_stride + static_cast<size_t>(block) * block_bytes;
                const uint8_t* packed_values = packed_block + 8;
                const float input_scale = input_scales[block];
                for (uint32_t chunk = 0; chunk < 2; ++chunk)
                {
                    const int8_t* input_chunk = input + static_cast<size_t>(block) * 32 + chunk * 16;
                    for (uint32_t row = 0; row < 8; ++row)
                    {
                        const size_t matrix_row = group * 8 + row;
                        if (matrix_row >= row_count)
                            continue;
                        const uint8_t* row_values = packed_values + (static_cast<size_t>(chunk) * 8 + row) * 8;
                        accumulators[row] += static_cast<float>(msvc_avx2_mxfp4_q8_packed_chunk_dot(row_values, input_chunk))
                                             * (0.5f * scales_by_exponent[packed_block[row]]) * input_scale;
                    }
                }
            }
            for (uint32_t row = 0; row < 8; ++row)
            {
                const size_t matrix_row = group * 8 + row;
                if (matrix_row < row_count)
                    output[matrix_row] = accumulators[row];
            }
        }
        return;
    }

    for (size_t token = 0; token < token_count; ++token)
        for (uint32_t row = 0; row < row_count; ++row)
            output[token * output_stride + row] = 0.0f;

    for (size_t group = 0; group < group_count; ++group)
    {
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* packed_block = packed + group * group_stride + static_cast<size_t>(block) * block_bytes;
            const uint8_t* packed_values = packed_block + 8;
            for (size_t token = 0; token < token_count; ++token)
            {
                const int8_t* input_block = input + token * input_stride + static_cast<size_t>(block) * 32;
                const float input_scale = input_scales[token * scale_stride + block];
                for (uint32_t chunk = 0; chunk < 2; ++chunk)
                {
                    const int8_t* input_chunk = input_block + chunk * 16;
                    for (uint32_t row = 0; row < 8; ++row)
                    {
                        const size_t matrix_row = group * 8 + row;
                        if (matrix_row >= row_count)
                            continue;
                        const uint8_t* row_values = packed_values + (static_cast<size_t>(chunk) * 8 + row) * 8;
                        output[token * output_stride + matrix_row] +=
                            static_cast<float>(msvc_avx2_mxfp4_q8_packed_chunk_dot(row_values, input_chunk))
                            * (0.5f * scales_by_exponent[packed_block[row]]) * input_scale;
                    }
                }
            }
        }
    }
}

void msvc_avx2_mxfp4_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
                              float* output, size_t output_stride) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx2_scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
        output[token_index * output_stride] = 0.0f;
    if (token_count == 1)
    {
        output[0] = msvc_avx2_mxfp4_dot(packed, scales, block_count, input);
        return;
    }
    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        __m128i decoded[2];
        avx2_decode_block(packed + static_cast<size_t>(block_index) * 16, decoded);
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
            output[token_index * output_stride] += avx2_horizontal_sum(accumulator) * scale;
        }
    }
}

void msvc_avx2_mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                                  uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output,
                                  size_t first_output_stride, float* second_output, size_t second_output_stride) noexcept
{
    const std::array<float, 256>& scales_by_exponent = avx2_scale_table();
    for (size_t token_index = 0; token_index < token_count; ++token_index)
    {
        first_output[token_index * first_output_stride] = 0.0f;
        second_output[token_index * second_output_stride] = 0.0f;
    }
    if (token_count == 1)
    {
        __m256 first_total = _mm256_setzero_ps();
        __m256 second_total = _mm256_setzero_ps();
        for (uint32_t block_index = 0; block_index < block_count; ++block_index)
        {
            __m128i decoded_rows[2][2];
            avx2_decode_block(first_packed + static_cast<size_t>(block_index) * 16, decoded_rows[0]);
            avx2_decode_block(second_packed + static_cast<size_t>(block_index) * 16, decoded_rows[1]);
            const float* input_block = input + static_cast<size_t>(block_index) * 32;
            __m256 first_block = _mm256_setzero_ps();
            __m256 second_block = _mm256_setzero_ps();
            for (uint32_t half = 0; half < 2; ++half)
            {
                const __m256 input_low = _mm256_loadu_ps(input_block + half * 16);
                const __m256 input_high = _mm256_loadu_ps(input_block + half * 16 + 8);
                first_block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(decoded_rows[0][half])), input_low, first_block);
                first_block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(decoded_rows[0][half], 8))), input_high, first_block);
                second_block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(decoded_rows[1][half])), input_low, second_block);
                second_block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(decoded_rows[1][half], 8))), input_high, second_block);
            }
            first_total = _mm256_fmadd_ps(first_block, _mm256_set1_ps(0.5f * scales_by_exponent[first_scales[block_index]]), first_total);
            second_total = _mm256_fmadd_ps(second_block, _mm256_set1_ps(0.5f * scales_by_exponent[second_scales[block_index]]), second_total);
        }
        first_output[0] = avx2_horizontal_sum(first_total);
        second_output[0] = avx2_horizontal_sum(second_total);
        return;
    }
    if (token_count == 2)
    {
        __m256 first_totals[2] = {
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        };
        __m256 second_totals[2] = {
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        };
        for (uint32_t block_index = 0; block_index < block_count; ++block_index)
        {
            __m128i decoded_rows[2][2];
            avx2_decode_block(first_packed + static_cast<size_t>(block_index) * 16, decoded_rows[0]);
            avx2_decode_block(second_packed + static_cast<size_t>(block_index) * 16, decoded_rows[1]);
            const __m256 first_scale = _mm256_set1_ps(0.5f * scales_by_exponent[first_scales[block_index]]);
            const __m256 second_scale = _mm256_set1_ps(0.5f * scales_by_exponent[second_scales[block_index]]);
            const size_t input_offset = static_cast<size_t>(block_index) * 32;
            for (size_t token_index = 0; token_index < 2; ++token_index)
            {
                const float* token = input + token_index * input_stride + input_offset;
                __m256 first_block = _mm256_setzero_ps();
                __m256 second_block = _mm256_setzero_ps();
                for (uint32_t half = 0; half < 2; ++half)
                {
                    const __m256 input_low = _mm256_loadu_ps(token + half * 16);
                    const __m256 input_high = _mm256_loadu_ps(token + half * 16 + 8);
                    first_block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(decoded_rows[0][half])), input_low, first_block);
                    first_block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(decoded_rows[0][half], 8))), input_high, first_block);
                    second_block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(decoded_rows[1][half])), input_low, second_block);
                    second_block = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(decoded_rows[1][half], 8))), input_high, second_block);
                }
                first_totals[token_index] = _mm256_fmadd_ps(first_block, first_scale, first_totals[token_index]);
                second_totals[token_index] = _mm256_fmadd_ps(second_block, second_scale, second_totals[token_index]);
            }
        }
        for (size_t token_index = 0; token_index < 2; ++token_index)
        {
            first_output[token_index * first_output_stride] = avx2_horizontal_sum(first_totals[token_index]);
            second_output[token_index * second_output_stride] = avx2_horizontal_sum(second_totals[token_index]);
        }
        return;
    }

    for (uint32_t block_index = 0; block_index < block_count; ++block_index)
    {
        __m128i decoded_rows[2][2];
        avx2_decode_block(first_packed + static_cast<size_t>(block_index) * 16, decoded_rows[0]);
        avx2_decode_block(second_packed + static_cast<size_t>(block_index) * 16, decoded_rows[1]);
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
            first_output[token_index * first_output_stride] += avx2_horizontal_sum(first_accumulator) * first_scale;
            second_output[token_index * second_output_stride] += avx2_horizontal_sum(second_accumulator) * second_scale;
        }
    }
}

void msvc_avx2_mxfp4_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,
                                      size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                                      float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept
{
    const size_t packed_row_bytes = static_cast<size_t>(block_count) * 16;
    uint32_t pair = 0;
    if (token_count == 1)
    {
        const std::array<float, 256>& scales_by_exponent = avx2_scale_table();
        static constexpr uint32_t pairs_per_group = 2;
        for (; pair + pairs_per_group <= row_pair_count; pair += pairs_per_group)
        {
            __m256 totals[pairs_per_group * 2];
            for (__m256& total : totals)
                total = _mm256_setzero_ps();
            const size_t first_group_row = static_cast<size_t>(pair) * 2;
            for (uint32_t block_index = 0; block_index < block_count; ++block_index)
            {
                const float* input_block = input + static_cast<size_t>(block_index) * 32;
                const __m256 input_blocks[4] = {
                    _mm256_loadu_ps(input_block),
                    _mm256_loadu_ps(input_block + 8),
                    _mm256_loadu_ps(input_block + 16),
                    _mm256_loadu_ps(input_block + 24),
                };
                for (uint32_t row = 0; row < pairs_per_group * 2; ++row)
                {
                    const size_t matrix_row = first_group_row + row;
                    __m128i decoded[2];
                    avx2_decode_block(packed + matrix_row * packed_row_bytes + static_cast<size_t>(block_index) * 16, decoded);
                    __m256 block = _mm256_setzero_ps();
                    for (uint32_t half = 0; half < 2; ++half)
                    {
                        block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(decoded[half])), input_blocks[half * 2], block);
                        block = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(decoded[half], 8))), input_blocks[half * 2 + 1], block);
                    }
                    totals[row] = _mm256_fmadd_ps(
                        block, _mm256_set1_ps(0.5f * scales_by_exponent[scales[matrix_row * block_count + block_index]]), totals[row]);
                }
            }
            for (uint32_t local_pair = 0; local_pair < pairs_per_group; ++local_pair)
            {
                first_output[static_cast<size_t>(pair + local_pair) * first_pair_stride] = avx2_horizontal_sum(totals[local_pair * 2]);
                second_output[static_cast<size_t>(pair + local_pair) * second_pair_stride] = avx2_horizontal_sum(totals[local_pair * 2 + 1]);
            }
        }
    }
    for (; pair < row_pair_count; ++pair)
    {
        const size_t first_row = static_cast<size_t>(pair) * 2;
        msvc_avx2_mxfp4_matmul_rows2(packed + first_row * packed_row_bytes, scales + first_row * block_count, packed + (first_row + 1) * packed_row_bytes,
                                     scales + (first_row + 1) * block_count, block_count, input, input_stride, token_count,
                                     first_output + static_cast<size_t>(pair) * first_pair_stride, first_token_stride,
                                     second_output + static_cast<size_t>(pair) * second_pair_stride, second_token_stride);
    }
}

} // namespace moe
} // namespace ncnn
