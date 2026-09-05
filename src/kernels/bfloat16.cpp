#include "bfloat16.h"

#include "engine/cpu.h"
#include "ncnn/moe/runtime.h"

#if defined(NCNN_MOE_MSVC_X86_SIMD)
#include "mxfp4_msvc.h"
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f,avx512bw"))) static void avx512_bfloat16_single_token_linear(
    const uint16_t* weights,
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
#endif

namespace ncnn {
namespace moe {

static void scalar_float_to_bfloat16_array(
    uint16_t* output,
    const float* input,
    uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, input + index, sizeof(bits));
        const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
        output[index] = static_cast<uint16_t>((bits + rounding) >> 16);
    }
}

static float scalar_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    float sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(weights[index]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[index];
    }
    return sum;
}

static void scalar_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
{
    for (uint32_t index = 0; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(input[index]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        output[index] += scale * value;
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)
static void neon_float_to_bfloat16_array(
    uint16_t* output,
    const float* input,
    uint32_t count) noexcept
{
    const uint32x4_t rounding_bias = vdupq_n_u32(0x7fffu);
    const uint32x4_t low_bit_mask = vdupq_n_u32(1);
    uint32_t index = 0;
    for (; index + 4 <= count; index += 4)
    {
        const uint32x4_t bits = vreinterpretq_u32_f32(vld1q_f32(input + index));
        const uint32x4_t rounding = vaddq_u32(
            rounding_bias,
            vandq_u32(vshrq_n_u32(bits, 16), low_bit_mask));
        const uint32x4_t high = vshrq_n_u32(vaddq_u32(bits, rounding), 16);
        vst1_u16(output + index, vmovn_u32(high));
    }
    scalar_float_to_bfloat16_array(output + index, input + index, count - index);
}

static float neon_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    uint32_t index = 0;
    for (; index + 4 <= count; index += 4)
    {
        const uint32x4_t expanded = vshlq_n_u32(vmovl_u16(vld1_u16(weights + index)), 16);
        accumulator = vfmaq_f32(accumulator, vreinterpretq_f32_u32(expanded), vld1q_f32(input + index));
    }
    float sum = vaddvq_f32(accumulator);
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(weights[index]) << 16;
        float weight = 0.0f;
        std::memcpy(&weight, &bits, sizeof(weight));
        sum += weight * input[index];
    }
    return sum;
}

static void neon_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
{
    const float32x4_t scale_vector = vdupq_n_f32(scale);
    uint32_t index = 0;
    for (; index + 4 <= count; index += 4)
    {
        const uint32x4_t expanded = vshlq_n_u32(vmovl_u16(vld1_u16(input + index)), 16);
        const float32x4_t values = vreinterpretq_f32_u32(expanded);
        vst1q_f32(output + index, vfmaq_f32(vld1q_f32(output + index), values, scale_vector));
    }
    for (; index < count; ++index)
    {
        const uint32_t bits = static_cast<uint32_t>(input[index]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        output[index] += scale * value;
    }
}
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) static void avx2_float_to_bfloat16_array(
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
    scalar_float_to_bfloat16_array(output + index, input + index, count - index);
}

__attribute__((target("avx512bf16,avx512f"))) static void avx512bf16_float_to_bfloat16_array(
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
    scalar_float_to_bfloat16_array(output + index, input + index, count - index);
}

__attribute__((target("avx2,fma"))) static float avx2_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
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
    const __m256 accumulator = _mm256_add_ps(
        _mm256_add_ps(accumulator0, accumulator1),
        _mm256_add_ps(accumulator2, accumulator3));
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(accumulator), _mm256_extractf128_ps(accumulator, 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    float sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
    for (; index + 8 <= count; index += 8)
    {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
        const __m256 values = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
        const __m256 product = _mm256_mul_ps(values, _mm256_loadu_ps(input + index));
        const __m128 product_halves = _mm_add_ps(
            _mm256_castps256_ps128(product),
            _mm256_extractf128_ps(product, 1));
        const __m128 product_pairs = _mm_hadd_ps(product_halves, product_halves);
        sum += _mm_cvtss_f32(_mm_hadd_ps(product_pairs, product_pairs));
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

__attribute__((target("avx2,fma"))) static void avx2_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
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

__attribute__((target("avx512f,avx512bw"))) static float avx512_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
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

__attribute__((target("avx512f,avx512bw"))) static void avx512_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
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

__attribute__((target("avx512bf16,avx512f,avx512bw"))) static float avx512bf16_pair_dot(
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
        const uint32_t left_bits = static_cast<uint32_t>(left[index]) << 16;
        const uint32_t right_bits = static_cast<uint32_t>(right[index]) << 16;
        float left_value = 0.0f;
        float right_value = 0.0f;
        std::memcpy(&left_value, &left_bits, sizeof(left_value));
        std::memcpy(&right_value, &right_bits, sizeof(right_value));
        sum += left_value * right_value;
    }
    return sum;
}

__attribute__((target("avx512bf16,avx512f,avx512bw"), always_inline)) static inline void avx512bf16_linear_tile4x4(const uint16_t* weights,
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

__attribute__((target("avx512bf16,avx512f,avx512bw"), always_inline)) static inline void avx512bf16_linear_tile1x4(
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

__attribute__((target("avx512bf16,avx512f,avx512bw"))) static void avx512bf16_batched_linear(const uint16_t* weights,
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
            avx512bf16_linear_tile1x4(
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
            avx512bf16_linear_tile4x4(group_weights,
                                      packed_input_data + token * input_columns,
                                      input_columns,
                                      input_columns,
                                      output + token * output_stride + first_output,
                                      output_stride,
                                      valid_rows);
        }
    }
}

#endif

using FloatToBfloat16Function = void (*)(uint16_t*, const float*, uint32_t) noexcept;
using Bfloat16DotFunction = float (*)(const uint16_t*, const float*, uint32_t) noexcept;
using Bfloat16ScaledAddFunction = void (*)(float*, const uint16_t*, float, uint32_t) noexcept;

static float scalar_bfloat16_pair_dot(
    const uint16_t* left,
    const uint16_t* right,
    uint32_t count) noexcept
{
    float sum = 0.0f;
    for (uint32_t index = 0; index < count; ++index)
    {
        const uint32_t left_bits = static_cast<uint32_t>(left[index]) << 16;
        const uint32_t right_bits = static_cast<uint32_t>(right[index]) << 16;
        float left_value = 0.0f;
        float right_value = 0.0f;
        std::memcpy(&left_value, &left_bits, sizeof(left_value));
        std::memcpy(&right_value, &right_bits, sizeof(right_value));
        sum += left_value * right_value;
    }
    return sum;
}

// Scoped instrumentation context only. Model weights, activation scratch,
// scheduling, and session state never use thread-local storage.
static thread_local Bfloat16BatchedLinearExecutionCounter*
    current_bfloat16_execution_counter = nullptr;

uint64_t Bfloat16BatchedLinearExecutionCounter::dispatch_count() const noexcept
{
    return count.load(std::memory_order_relaxed);
}

ScopedBfloat16BatchedLinearExecutionCounter::
    ScopedBfloat16BatchedLinearExecutionCounter(
        Bfloat16BatchedLinearExecutionCounter* counter) noexcept
    : previous(current_bfloat16_execution_counter)
{
    current_bfloat16_execution_counter = counter;
}

ScopedBfloat16BatchedLinearExecutionCounter::
    ~ScopedBfloat16BatchedLinearExecutionCounter()
{
    current_bfloat16_execution_counter = previous;
}

Bfloat16BatchedLinearExecutionCounter*
current_bfloat16_batched_linear_execution_counter() noexcept
{
    return current_bfloat16_execution_counter;
}

void record_bfloat16_batched_linear_dispatch() noexcept
{
    if (current_bfloat16_execution_counter)
    {
        current_bfloat16_execution_counter->count.fetch_add(
            1,
            std::memory_order_relaxed);
    }
}

static Bfloat16DotFunction select_bfloat16_dot() noexcept
{
    const uint64_t isa = cpu_isa_flags();
#if defined(__aarch64__) || defined(_M_ARM64)
    if ((isa & CpuIsaArmNeon) != 0)
        return neon_bfloat16_dot;
#elif defined(NCNN_MOE_MSVC_X86_SIMD)
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_bfloat16_dot;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_bfloat16_dot;
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_bfloat16_dot;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_bfloat16_dot;
#endif
    return scalar_bfloat16_dot;
}

static FloatToBfloat16Function select_float_to_bfloat16_array() noexcept
{
    const uint64_t isa = cpu_isa_flags();
#if defined(__aarch64__) || defined(_M_ARM64)
    if ((isa & CpuIsaArmNeon) != 0)
        return neon_float_to_bfloat16_array;
#elif defined(NCNN_MOE_MSVC_X86_SIMD)
    if ((isa & CpuIsaX86Avx512Bf16) != 0)
        return msvc_avx512_float_to_bfloat16_array;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_float_to_bfloat16_array;
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if ((isa & CpuIsaX86Avx512Bf16) != 0)
        return avx512bf16_float_to_bfloat16_array;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_float_to_bfloat16_array;
#endif
    return scalar_float_to_bfloat16_array;
}

static Bfloat16ScaledAddFunction select_bfloat16_scaled_add() noexcept
{
    const uint64_t isa = cpu_isa_flags();
#if defined(__aarch64__) || defined(_M_ARM64)
    if ((isa & CpuIsaArmNeon) != 0)
        return neon_bfloat16_scaled_add;
#elif defined(NCNN_MOE_MSVC_X86_SIMD)
    if ((isa & CpuIsaX86Avx512) != 0)
        return msvc_avx512_bfloat16_scaled_add;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return msvc_avx2_bfloat16_scaled_add;
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if ((isa & CpuIsaX86Avx512) != 0)
        return avx512_bfloat16_scaled_add;
    if ((isa & CpuIsaX86Avx2Fma) != 0)
        return avx2_bfloat16_scaled_add;
#endif
    return scalar_bfloat16_scaled_add;
}

const char* bfloat16_dot_kernel_name() noexcept
{
    const Bfloat16DotFunction function = select_bfloat16_dot();
#if defined(__aarch64__) || defined(_M_ARM64)
    if (function == neon_bfloat16_dot)
        return "arm-neon-fp32-fma";
#elif defined(NCNN_MOE_MSVC_X86_SIMD)
    if (function == msvc_avx512_bfloat16_dot)
        return "x86-avx512-fp32-fma";
    if (function == msvc_avx2_bfloat16_dot)
        return "x86-avx2-fp32-fma";
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (function == avx512_bfloat16_dot)
        return "x86-avx512-fp32-fma";
    if (function == avx2_bfloat16_dot)
        return "x86-avx2-fp32-fma";
#endif
    return "scalar-fp32";
}

float bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    static const Bfloat16DotFunction function = select_bfloat16_dot();
    return function(weights, input, count);
}

void float_to_bfloat16_array(
    uint16_t* output,
    const float* input,
    uint32_t count) noexcept
{
    static const FloatToBfloat16Function function = select_float_to_bfloat16_array();
    function(output, input, count);
}

bool bfloat16_pair_dot_available() noexcept
{
    return (cpu_isa_flags() & CpuIsaX86Avx512Bf16) != 0;
}

float bfloat16_pair_dot(
    const uint16_t* left,
    const uint16_t* right,
    uint32_t count) noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (bfloat16_pair_dot_available())
        return msvc_avx512_bfloat16_pair_dot(left, right, count);
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if (bfloat16_pair_dot_available())
        return avx512bf16_pair_dot(left, right, count);
#endif
    return scalar_bfloat16_pair_dot(left, right, count);
}

void bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
{
    static const Bfloat16ScaledAddFunction function = select_bfloat16_scaled_add();
    function(output, input, scale, count);
}

const char* bfloat16_batched_linear_kernel_name(uint64_t optimization_flags) noexcept
{
    if (!has_flag(optimization_flags, OptimizationCpuBfloat16Batched))
        return "unavailable";
    const uint64_t isa = cpu_isa_flags();
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if ((isa & CpuIsaX86Avx512Bf16) != 0)
        return "x86-avx512-bf16-dpbf16-1x4+4x4";
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if ((isa & CpuIsaX86Avx512Bf16) != 0)
        return "x86-avx512-bf16-dpbf16-1x4+4x4";
#endif
    (void)isa;
    return "unavailable";
}

bool bfloat16_batched_linear(const uint16_t* weights,
                             const float* input,
                             size_t input_stride,
                             size_t token_count,
                             uint32_t output_columns,
                             uint32_t input_columns,
                             float* output,
                             size_t output_stride,
                             int thread_count,
                             uint64_t optimization_flags)
{
    if (!weights || !input || !output || token_count == 0
        || output_columns < 4 || output_columns % 4 != 0
        || input_columns == 0 || input_columns % 32 != 0 || input_stride < input_columns || output_stride < output_columns)
    {
        return false;
    }
    if (token_count > std::numeric_limits<size_t>::max() - 3)
        return false;
    const size_t padded_token_count = (token_count + 3) & ~size_t{3};
    if (padded_token_count > std::numeric_limits<size_t>::max() / input_columns)
        return false;

    if (!has_flag(optimization_flags, OptimizationCpuBfloat16Batched))
        return false;
    if (token_count > std::numeric_limits<uint64_t>::max() / output_columns / input_columns)
        return false;
    const uint64_t operation_count = static_cast<uint64_t>(token_count) * output_columns * input_columns;
    if (output_columns < 128 || operation_count < 1024 * 1024)
        return false;
    std::vector<uint16_t> packed_input;

    const uint64_t isa = cpu_isa_flags();
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if ((isa & CpuIsaX86Avx512Bf16) != 0)
    {
        if (token_count == 1)
        {
            msvc_avx512_bfloat16_single_token_linear(
                weights,
                input,
                output_columns,
                input_columns,
                output,
                std::max(1, thread_count));
            record_bfloat16_batched_linear_dispatch();
            return true;
        }
        msvc_avx512_bfloat16_batched_linear(weights,
                                            input,
                                            input_stride,
                                            token_count,
                                            output_columns,
                                            input_columns,
                                            output,
                                            output_stride,
                                            std::max(1, thread_count),
                                            packed_input);
        record_bfloat16_batched_linear_dispatch();
        return true;
    }
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if ((isa & CpuIsaX86Avx512Bf16) != 0)
    {
        if (token_count == 1)
        {
            avx512_bfloat16_single_token_linear(
                weights,
                input,
                output_columns,
                input_columns,
                output,
                std::max(1, thread_count));
            record_bfloat16_batched_linear_dispatch();
            return true;
        }
        avx512bf16_batched_linear(weights,
                                  input,
                                  input_stride,
                                  token_count,
                                  output_columns,
                                  input_columns,
                                  output,
                                  output_stride,
                                  std::max(1, thread_count),
                                  packed_input);
        record_bfloat16_batched_linear_dispatch();
        return true;
    }
#else
    (void)thread_count;
#endif
    (void)isa;
    return false;
}

} // namespace moe
} // namespace ncnn
