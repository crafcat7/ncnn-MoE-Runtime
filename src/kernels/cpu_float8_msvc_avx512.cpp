#include "cpu_float8_msvc.h"

#include "cpu_float8.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace ncnn {
namespace moe {

static __m512 decode_float8_e4m3(__m128i packed) noexcept
{
    const __m512i bytes = _mm512_cvtepu8_epi32(packed);
    const __m512i exponent = _mm512_and_si512(_mm512_srli_epi32(bytes, 3), _mm512_set1_epi32(15));
    const __m512i mantissa = _mm512_and_si512(bytes, _mm512_set1_epi32(7));
    const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(bytes, _mm512_set1_epi32(128)), 24);

    const __m512i normal_bits = _mm512_or_si512(
        sign,
        _mm512_or_si512(
            _mm512_slli_epi32(_mm512_add_epi32(exponent, _mm512_set1_epi32(120)), 23),
            _mm512_slli_epi32(mantissa, 20)));
    const __m512 normal = _mm512_castsi512_ps(normal_bits);
    const __m512 subnormal_magnitude = _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa), _mm512_set1_ps(1.0f / 512.0f));
    const __m512 subnormal = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(subnormal_magnitude), sign));
    const __mmask16 exponent_zero = _mm512_cmpeq_epi32_mask(exponent, _mm512_setzero_si512());
    __m512 decoded = _mm512_mask_blend_ps(exponent_zero, normal, subnormal);

    const __mmask16 nan_mask = _mm512_cmpeq_epi32_mask(exponent, _mm512_set1_epi32(15))
                               & _mm512_cmpeq_epi32_mask(mantissa, _mm512_set1_epi32(7));
    const __m512 nan_value = _mm512_castsi512_ps(_mm512_or_si512(sign, _mm512_set1_epi32(0x7fc00000)));
    decoded = _mm512_mask_blend_ps(nan_mask, decoded, nan_value);
    return decoded;
}

static __forceinline __m512bh decode_float8_e4m3_bfloat16(
    __m256i packed) noexcept
{
    const __m512i bytes = _mm512_cvtepu8_epi16(packed);
    const __m512i exponent = _mm512_and_si512(
        _mm512_srli_epi16(bytes, 3), _mm512_set1_epi16(15));
    const __m512i mantissa =
        _mm512_and_si512(bytes, _mm512_set1_epi16(7));
    const __m512i sign = _mm512_slli_epi16(
        _mm512_and_si512(bytes, _mm512_set1_epi16(128)), 8);
    const __m512i normal = _mm512_or_si512(
        sign,
        _mm512_or_si512(
            _mm512_slli_epi16(
                _mm512_add_epi16(exponent, _mm512_set1_epi16(120)), 7),
            _mm512_slli_epi16(mantissa, 4)));
    const __m512i subnormal_table = _mm512_setr_epi16(
        0x0000, 0x3b00, 0x3b80, 0x3bc0,
        0x3c00, 0x3c20, 0x3c40, 0x3c60,
        0x0000, 0x3b00, 0x3b80, 0x3bc0,
        0x3c00, 0x3c20, 0x3c40, 0x3c60,
        0x0000, 0x3b00, 0x3b80, 0x3bc0,
        0x3c00, 0x3c20, 0x3c40, 0x3c60,
        0x0000, 0x3b00, 0x3b80, 0x3bc0,
        0x3c00, 0x3c20, 0x3c40, 0x3c60);
    const __m512i subnormal = _mm512_or_si512(
        sign, _mm512_permutexvar_epi16(mantissa, subnormal_table));
    const __mmask32 exponent_zero =
        _mm512_cmpeq_epi16_mask(exponent, _mm512_setzero_si512());
    __m512i decoded =
        _mm512_mask_blend_epi16(exponent_zero, normal, subnormal);
    const __mmask32 nan_mask =
        _mm512_cmpeq_epi16_mask(exponent, _mm512_set1_epi16(15))
        & _mm512_cmpeq_epi16_mask(mantissa, _mm512_set1_epi16(7));
    decoded = _mm512_mask_blend_epi16(
        nan_mask, decoded,
        _mm512_or_si512(sign, _mm512_set1_epi16(0x7fc0)));
    return (__m512bh)decoded;
}

static __forceinline __m512bh pack_input_bfloat16(
    const float* input) noexcept
{
    return _mm512_cvtne2ps_pbh(_mm512_loadu_ps(input + 16),
                               _mm512_loadu_ps(input));
}

void msvc_avx512_quantize_float8_e4m3(
    const float* source,
    float* values,
    uint32_t count,
    uint32_t block_size,
    bool power_of_two_scale) noexcept
{
    const __m512 absolute_mask =
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512 maximum_value = _mm512_set1_ps(448.0f);
    const __m512 minimum_normal = _mm512_set1_ps(1.0f / 64.0f);
    const __m512 subnormal_scale = _mm512_set1_ps(512.0f);
    const __m512 subnormal_inverse_scale = _mm512_set1_ps(1.0f / 512.0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 subnormal_maximum = _mm512_set1_ps(7.0f);
    const __m512 canonical_nan =
        _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000));
    const __m512i sign_mask = _mm512_set1_epi32(0x80000000u);
    const __m512i rounding_bias = _mm512_set1_epi32(0x00080000);
    const __m512i e4m3_value_mask = _mm512_set1_epi32(0xfff00000u);

    for (uint32_t block_begin = 0; block_begin < count;
         block_begin += block_size)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 vector_maximum = _mm512_set1_ps(1e-4f);
        uint32_t index = block_begin;
        for (; index + 16 <= block_end; index += 16)
        {
            const __m512 magnitude = _mm512_and_ps(
                _mm512_loadu_ps(source + index), absolute_mask);
            const __mmask16 greater = _mm512_cmp_ps_mask(
                magnitude, vector_maximum, _CMP_GT_OQ);
            vector_maximum = _mm512_mask_mov_ps(
                vector_maximum, greater, magnitude);
        }
        float maximum = _mm512_reduce_max_ps(vector_maximum);
        for (uint32_t tail = index; tail < block_end; ++tail)
            maximum = std::max(maximum, std::fabs(source[tail]));

        float scale = maximum / 448.0f;
        if (power_of_two_scale)
            scale = std::exp2(std::ceil(std::log2(scale)));
        const __m512 scale_vector = _mm512_set1_ps(scale);

        index = block_begin;
        for (; index + 16 <= block_end; index += 16)
        {
            const __m512 source_values = _mm512_loadu_ps(source + index);
            const __m512 normalized = _mm512_div_ps(source_values, scale_vector);
            const __m512i normalized_bits = _mm512_castps_si512(normalized);
            const __m512i signs = _mm512_and_si512(normalized_bits, sign_mask);
            __m512 magnitude = _mm512_and_ps(normalized, absolute_mask);
            const __mmask16 nan_mask =
                _mm512_cmp_ps_mask(magnitude, magnitude, _CMP_UNORD_Q);
            const __mmask16 above_maximum = _mm512_cmp_ps_mask(
                magnitude, maximum_value, _CMP_GT_OQ);
            magnitude = _mm512_mask_mov_ps(
                magnitude, above_maximum, maximum_value);

            const __m512i rounded_bits = _mm512_and_si512(
                _mm512_add_epi32(
                    _mm512_castps_si512(magnitude), rounding_bias),
                e4m3_value_mask);
            const __m512 normal = _mm512_castsi512_ps(rounded_bits);
            __m512 subnormal = _mm512_floor_ps(_mm512_add_ps(
                _mm512_mul_ps(magnitude, subnormal_scale), half));
            const __mmask16 above_subnormal_maximum = _mm512_cmp_ps_mask(
                subnormal, subnormal_maximum, _CMP_GT_OQ);
            subnormal = _mm512_mask_mov_ps(
                subnormal, above_subnormal_maximum, subnormal_maximum);
            subnormal = _mm512_mul_ps(subnormal, subnormal_inverse_scale);
            const __mmask16 normal_mask = _mm512_cmp_ps_mask(
                magnitude, minimum_normal, _CMP_GE_OQ);
            __m512 quantized = _mm512_mask_blend_ps(
                normal_mask, subnormal, normal);
            quantized = _mm512_castsi512_ps(_mm512_xor_si512(
                _mm512_castps_si512(quantized), signs));
            quantized = _mm512_mul_ps(quantized, scale_vector);
            quantized = _mm512_mask_mov_ps(
                quantized, nan_mask, canonical_nan);
            _mm512_storeu_ps(values + index, quantized);
        }
        for (; index < block_end; ++index)
        {
            const float normalized =
                std::clamp(source[index] / scale, -448.0f, 448.0f);
            values[index] = float8_e4m3_to_float(
                                float_to_float8_e4m3(normalized))
                            * scale;
        }
    }
}

float msvc_avx512_float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept
{
    float result = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulator = _mm512_setzero_ps();
        uint32_t index = block_begin;
        for (; index + 16 <= block_end; index += 16)
        {
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
            accumulator = _mm512_fmadd_ps(decode_float8_e4m3(packed), _mm512_loadu_ps(input + index), accumulator);
        }
        float partial = _mm512_reduce_add_ps(accumulator);
        for (; index < block_end; ++index)
            partial += float8_e4m3_to_float(weights[index]) * input[index];
        result += partial * scales[block];
    }
    return result;
}

void msvc_avx512_float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count,
                                             uint32_t block_size, uint32_t row_count, float* output) noexcept
{
    if (row_count == 0 || row_count > 4)
        return;
    for (uint32_t row = 0; row < row_count; ++row)
        output[row] = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count; block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulators[4];
        for (uint32_t row = 0; row < row_count; ++row)
            accumulators[row] = _mm512_setzero_ps();
        uint32_t index = block_begin;
        for (; index + 16 <= block_end; index += 16)
        {
            const __m512 input_values = _mm512_loadu_ps(input + index);
            for (uint32_t row = 0; row < row_count; ++row)
            {
                const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + static_cast<size_t>(row) * weight_row_stride + index));
                accumulators[row] = _mm512_fmadd_ps(decode_float8_e4m3(packed), input_values, accumulators[row]);
            }
        }
        for (uint32_t row = 0; row < row_count; ++row)
        {
            float partial = _mm512_reduce_add_ps(accumulators[row]);
            for (uint32_t tail = index; tail < block_end; ++tail)
                partial += float8_e4m3_to_float(weights[static_cast<size_t>(row) * weight_row_stride + tail]) * input[tail];
            output[row] += partial * scales[block];
        }
    }
}

float msvc_avx512_bfloat16_float8_e4m3_block_dot(
    const uint8_t* weights,
    const float* scales,
    const float* input,
    uint32_t count,
    uint32_t block_size) noexcept
{
    float result = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count;
         block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulator = _mm512_setzero_ps();
        uint32_t index = block_begin;
        for (; index + 32 <= block_end; index += 32)
        {
            const __m256i packed = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(weights + index));
            accumulator = _mm512_dpbf16_ps(
                accumulator,
                decode_float8_e4m3_bfloat16(packed),
                pack_input_bfloat16(input + index));
        }
        float partial = _mm512_reduce_add_ps(accumulator);
        for (; index < block_end; ++index)
        {
            partial += float8_e4m3_to_float(weights[index]) * input[index];
        }
        result += partial * scales[block];
    }
    return result;
}

static void bfloat16_float8_e4m3_block_dot_rows8(
    const uint8_t* weights,
    uint32_t weight_row_stride,
    const float* scales,
    const float* input,
    uint32_t count,
    uint32_t block_size,
    float* output) noexcept
{
    std::fill_n(output, 8, 0.0f);
    for (uint32_t block_begin = 0, block = 0; block_begin < count;
         block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulator0 = _mm512_setzero_ps();
        __m512 accumulator1 = _mm512_setzero_ps();
        __m512 accumulator2 = _mm512_setzero_ps();
        __m512 accumulator3 = _mm512_setzero_ps();
        __m512 accumulator4 = _mm512_setzero_ps();
        __m512 accumulator5 = _mm512_setzero_ps();
        __m512 accumulator6 = _mm512_setzero_ps();
        __m512 accumulator7 = _mm512_setzero_ps();
        uint32_t index = block_begin;
        for (; index + 32 <= block_end; index += 32)
        {
            const __m512bh input_values = pack_input_bfloat16(input + index);
#define NCNN_MOE_ACCUMULATE_FLOAT8_ROW(row)                                \
            accumulator##row = _mm512_dpbf16_ps(                           \
                accumulator##row,                                         \
                decode_float8_e4m3_bfloat16(_mm256_loadu_si256(            \
                    reinterpret_cast<const __m256i*>(                      \
                        weights + static_cast<size_t>(row)                 \
                                      * weight_row_stride                  \
                                + index))),                                \
                input_values)
            NCNN_MOE_ACCUMULATE_FLOAT8_ROW(0);
            NCNN_MOE_ACCUMULATE_FLOAT8_ROW(1);
            NCNN_MOE_ACCUMULATE_FLOAT8_ROW(2);
            NCNN_MOE_ACCUMULATE_FLOAT8_ROW(3);
            NCNN_MOE_ACCUMULATE_FLOAT8_ROW(4);
            NCNN_MOE_ACCUMULATE_FLOAT8_ROW(5);
            NCNN_MOE_ACCUMULATE_FLOAT8_ROW(6);
            NCNN_MOE_ACCUMULATE_FLOAT8_ROW(7);
#undef NCNN_MOE_ACCUMULATE_FLOAT8_ROW
        }
        float partials[8] = {
            _mm512_reduce_add_ps(accumulator0),
            _mm512_reduce_add_ps(accumulator1),
            _mm512_reduce_add_ps(accumulator2),
            _mm512_reduce_add_ps(accumulator3),
            _mm512_reduce_add_ps(accumulator4),
            _mm512_reduce_add_ps(accumulator5),
            _mm512_reduce_add_ps(accumulator6),
            _mm512_reduce_add_ps(accumulator7),
        };
        for (uint32_t row = 0; row < 8; ++row)
        {
            for (uint32_t tail = index; tail < block_end; ++tail)
            {
                partials[row] += float8_e4m3_to_float(
                                     weights[static_cast<size_t>(row)
                                                 * weight_row_stride
                                             + tail])
                                 * input[tail];
            }
            output[row] += partials[row] * scales[block];
        }
    }
}

static bool use_unrolled_bfloat16_float8_rows8() noexcept
{
    return true;
}

void msvc_avx512_bfloat16_float8_e4m3_block_dot_rows(
    const uint8_t* weights,
    uint32_t weight_row_stride,
    const float* scales,
    const float* input,
    uint32_t count,
    uint32_t block_size,
    uint32_t row_count,
    float* output) noexcept
{
    if (row_count == 0 || row_count > 8)
        return;
    if (row_count == 8 && use_unrolled_bfloat16_float8_rows8())
    {
        bfloat16_float8_e4m3_block_dot_rows8(
            weights, weight_row_stride, scales, input, count, block_size,
            output);
        return;
    }
    for (uint32_t row = 0; row < row_count; ++row)
        output[row] = 0.0f;
    for (uint32_t block_begin = 0, block = 0; block_begin < count;
         block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulators[8];
        for (uint32_t row = 0; row < row_count; ++row)
            accumulators[row] = _mm512_setzero_ps();
        uint32_t index = block_begin;
        for (; index + 32 <= block_end; index += 32)
        {
            const __m512bh input_values = pack_input_bfloat16(input + index);
            for (uint32_t row = 0; row < row_count; ++row)
            {
                const __m256i packed = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(
                        weights + static_cast<size_t>(row) * weight_row_stride
                        + index));
                accumulators[row] = _mm512_dpbf16_ps(
                    accumulators[row],
                    decode_float8_e4m3_bfloat16(packed), input_values);
            }
        }
        for (uint32_t row = 0; row < row_count; ++row)
        {
            float partial = _mm512_reduce_add_ps(accumulators[row]);
            for (uint32_t tail = index; tail < block_end; ++tail)
            {
                partial += float8_e4m3_to_float(
                               weights[static_cast<size_t>(row)
                                           * weight_row_stride
                                       + tail])
                           * input[tail];
            }
            output[row] += partial * scales[block];
        }
    }
}

void msvc_avx512_bfloat16_float8_e4m3_block_dot_rows_batch(
    const uint8_t* weights,
    uint32_t weight_row_stride,
    const float* scales,
    const float* input,
    size_t input_stride,
    uint32_t count,
    uint32_t block_size,
    uint32_t row_count,
    size_t output_stride,
    size_t token_count,
    float* output) noexcept
{
    if (row_count == 0 || row_count > 4 || token_count == 0
        || token_count > 4)
    {
        return;
    }

    for (uint32_t block_begin = 0, block = 0; block_begin < count;
         block_begin += block_size, ++block)
    {
        const uint32_t block_end = std::min(count, block_begin + block_size);
        __m512 accumulators[4][4];
        for (uint32_t row = 0; row < row_count; ++row)
        {
            for (size_t token = 0; token < token_count; ++token)
                accumulators[row][token] = _mm512_setzero_ps();
        }

        uint32_t index = block_begin;
        for (; index + 32 <= block_end; index += 32)
        {
            __m512bh input_values[4];
            for (size_t token = 0; token < token_count; ++token)
            {
                input_values[token] = pack_input_bfloat16(
                    input + token * input_stride + index);
            }
            __m512bh weight_values[4];
            for (uint32_t row = 0; row < row_count; ++row)
            {
                const uint8_t* row_weights =
                    weights + static_cast<size_t>(row) * weight_row_stride
                    + index;
                weight_values[row] = decode_float8_e4m3_bfloat16(
                    _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(row_weights)));
            }
            for (uint32_t row = 0; row < row_count; ++row)
            {
                for (size_t token = 0; token < token_count; ++token)
                {
                    accumulators[row][token] = _mm512_dpbf16_ps(
                        accumulators[row][token], weight_values[row],
                        input_values[token]);
                }
            }
        }

        for (uint32_t row = 0; row < row_count; ++row)
        {
            const uint8_t* row_weights =
                weights + static_cast<size_t>(row) * weight_row_stride;
            for (size_t token = 0; token < token_count; ++token)
            {
                float partial =
                    _mm512_reduce_add_ps(accumulators[row][token]);
                const float* token_input = input + token * input_stride;
                for (uint32_t tail = index; tail < block_end; ++tail)
                {
                    partial += float8_e4m3_to_float(row_weights[tail])
                               * token_input[tail];
                }
                output[token * output_stride + row] +=
                    partial * scales[block];
            }
        }
    }
}

} // namespace moe
} // namespace ncnn
