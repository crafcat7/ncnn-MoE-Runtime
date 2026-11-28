#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)

#include "kernels/cpu/x86/x86_internal.h"

#include "engine/cpu.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <immintrin.h>
#include <vector>

#if defined(_MSC_VER)
#define NCNN_MOE_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define NCNN_MOE_FORCE_INLINE inline __attribute__((always_inline))
#else
#define NCNN_MOE_FORCE_INLINE inline
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl")
#endif

namespace ncnn {
namespace moe {

float avx512_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept
{
    __m512 accumulator0 = _mm512_setzero_ps();
    __m512 accumulator1 = _mm512_setzero_ps();
    __m512 accumulator2 = _mm512_setzero_ps();
    __m512 accumulator3 = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 64 <= count; index += 64)
    {
        const __m256i packed0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + index));
        const __m256i packed1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + index + 16));
        const __m256i packed2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + index + 32));
        const __m256i packed3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + index + 48));
        const __m512 values0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed0),
                                                                     16));
        const __m512 values1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed1),
                                                                     16));
        const __m512 values2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed2),
                                                                     16));
        const __m512 values3 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed3),
                                                                     16));
        accumulator0 = _mm512_fmadd_ps(values0,
                                       _mm512_loadu_ps(input + index),
                                       accumulator0);
        accumulator1 = _mm512_fmadd_ps(values1,
                                       _mm512_loadu_ps(input + index + 16),
                                       accumulator1);
        accumulator2 = _mm512_fmadd_ps(values2,
                                       _mm512_loadu_ps(input + index + 32),
                                       accumulator2);
        accumulator3 = _mm512_fmadd_ps(values3,
                                       _mm512_loadu_ps(input + index + 48),
                                       accumulator3);
    }
    __m512 accumulator = _mm512_add_ps(_mm512_add_ps(accumulator0, accumulator1),
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

void avx512_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept
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

void avx512_pv_bf16(float* output,
                    const uint16_t* values0,
                    size_t value_stride,
                    const float* weights0,
                    uint32_t count0,
                    const uint16_t* values1,
                    const float* weights1,
                    uint32_t count1,
                    uint32_t columns) noexcept
{
    uint32_t column = 0;
#define NCNN_MOE_BF16_ROW2_LOAD(ptr, off) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>((ptr) + (off)))), 16))
    for (; column + 256 <= columns; column += 256)
    {
        __m512 a0 = _mm512_setzero_ps();
        __m512 a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps();
        __m512 a3 = _mm512_setzero_ps();
        __m512 a4 = _mm512_setzero_ps();
        __m512 a5 = _mm512_setzero_ps();
        __m512 a6 = _mm512_setzero_ps();
        __m512 a7 = _mm512_setzero_ps();
        __m512 a8 = _mm512_setzero_ps();
        __m512 a9 = _mm512_setzero_ps();
        __m512 a10 = _mm512_setzero_ps();
        __m512 a11 = _mm512_setzero_ps();
        __m512 a12 = _mm512_setzero_ps();
        __m512 a13 = _mm512_setzero_ps();
        __m512 a14 = _mm512_setzero_ps();
        __m512 a15 = _mm512_setzero_ps();
        const uint16_t* row = values0 + column;
        for (uint32_t key = 0; key < count0; ++key)
        {
            const __m512 weight = _mm512_set1_ps(weights0[key]);
            a0 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 0), a0);
            a1 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 16), a1);
            a2 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 32), a2);
            a3 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 48), a3);
            a4 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 64), a4);
            a5 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 80), a5);
            a6 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 96), a6);
            a7 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 112), a7);
            a8 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 128), a8);
            a9 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 144), a9);
            a10 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 160), a10);
            a11 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 176), a11);
            a12 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 192), a12);
            a13 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 208), a13);
            a14 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 224), a14);
            a15 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 240), a15);
            row += value_stride;
        }
        row = values1 + column;
        for (uint32_t key = 0; key < count1; ++key)
        {
            const __m512 weight = _mm512_set1_ps(weights1[key]);
            a0 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 0), a0);
            a1 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 16), a1);
            a2 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 32), a2);
            a3 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 48), a3);
            a4 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 64), a4);
            a5 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 80), a5);
            a6 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 96), a6);
            a7 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 112), a7);
            a8 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 128), a8);
            a9 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 144), a9);
            a10 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 160), a10);
            a11 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 176), a11);
            a12 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 192), a12);
            a13 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 208), a13);
            a14 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 224), a14);
            a15 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 240), a15);
            row += value_stride;
        }
        _mm512_storeu_ps(output + column + 0, _mm512_add_ps(_mm512_loadu_ps(output + column + 0), a0));
        _mm512_storeu_ps(output + column + 16, _mm512_add_ps(_mm512_loadu_ps(output + column + 16), a1));
        _mm512_storeu_ps(output + column + 32, _mm512_add_ps(_mm512_loadu_ps(output + column + 32), a2));
        _mm512_storeu_ps(output + column + 48, _mm512_add_ps(_mm512_loadu_ps(output + column + 48), a3));
        _mm512_storeu_ps(output + column + 64, _mm512_add_ps(_mm512_loadu_ps(output + column + 64), a4));
        _mm512_storeu_ps(output + column + 80, _mm512_add_ps(_mm512_loadu_ps(output + column + 80), a5));
        _mm512_storeu_ps(output + column + 96, _mm512_add_ps(_mm512_loadu_ps(output + column + 96), a6));
        _mm512_storeu_ps(output + column + 112, _mm512_add_ps(_mm512_loadu_ps(output + column + 112), a7));
        _mm512_storeu_ps(output + column + 128, _mm512_add_ps(_mm512_loadu_ps(output + column + 128), a8));
        _mm512_storeu_ps(output + column + 144, _mm512_add_ps(_mm512_loadu_ps(output + column + 144), a9));
        _mm512_storeu_ps(output + column + 160, _mm512_add_ps(_mm512_loadu_ps(output + column + 160), a10));
        _mm512_storeu_ps(output + column + 176, _mm512_add_ps(_mm512_loadu_ps(output + column + 176), a11));
        _mm512_storeu_ps(output + column + 192, _mm512_add_ps(_mm512_loadu_ps(output + column + 192), a12));
        _mm512_storeu_ps(output + column + 208, _mm512_add_ps(_mm512_loadu_ps(output + column + 208), a13));
        _mm512_storeu_ps(output + column + 224, _mm512_add_ps(_mm512_loadu_ps(output + column + 224), a14));
        _mm512_storeu_ps(output + column + 240, _mm512_add_ps(_mm512_loadu_ps(output + column + 240), a15));
    }
    for (; column + 128 <= columns; column += 128)
    {
        __m512 a0 = _mm512_setzero_ps();
        __m512 a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps();
        __m512 a3 = _mm512_setzero_ps();
        __m512 a4 = _mm512_setzero_ps();
        __m512 a5 = _mm512_setzero_ps();
        __m512 a6 = _mm512_setzero_ps();
        __m512 a7 = _mm512_setzero_ps();
        const uint16_t* row = values0 + column;
        for (uint32_t key = 0; key < count0; ++key)
        {
            const __m512 weight = _mm512_set1_ps(weights0[key]);
            a0 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 0), a0);
            a1 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 16), a1);
            a2 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 32), a2);
            a3 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 48), a3);
            a4 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 64), a4);
            a5 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 80), a5);
            a6 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 96), a6);
            a7 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 112), a7);
            row += value_stride;
        }
        row = values1 + column;
        for (uint32_t key = 0; key < count1; ++key)
        {
            const __m512 weight = _mm512_set1_ps(weights1[key]);
            a0 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 0), a0);
            a1 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 16), a1);
            a2 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 32), a2);
            a3 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 48), a3);
            a4 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 64), a4);
            a5 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 80), a5);
            a6 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 96), a6);
            a7 = _mm512_fmadd_ps(weight, NCNN_MOE_BF16_ROW2_LOAD(row, 112), a7);
            row += value_stride;
        }
        _mm512_storeu_ps(output + column + 0, _mm512_add_ps(_mm512_loadu_ps(output + column + 0), a0));
        _mm512_storeu_ps(output + column + 16, _mm512_add_ps(_mm512_loadu_ps(output + column + 16), a1));
        _mm512_storeu_ps(output + column + 32, _mm512_add_ps(_mm512_loadu_ps(output + column + 32), a2));
        _mm512_storeu_ps(output + column + 48, _mm512_add_ps(_mm512_loadu_ps(output + column + 48), a3));
        _mm512_storeu_ps(output + column + 64, _mm512_add_ps(_mm512_loadu_ps(output + column + 64), a4));
        _mm512_storeu_ps(output + column + 80, _mm512_add_ps(_mm512_loadu_ps(output + column + 80), a5));
        _mm512_storeu_ps(output + column + 96, _mm512_add_ps(_mm512_loadu_ps(output + column + 96), a6));
        _mm512_storeu_ps(output + column + 112, _mm512_add_ps(_mm512_loadu_ps(output + column + 112), a7));
    }
    for (; column + 16 <= columns; column += 16)
    {
        __m512 accumulator = _mm512_setzero_ps();
        const uint16_t* row = values0 + column;
        for (uint32_t key = 0; key < count0; ++key)
        {
            accumulator = _mm512_fmadd_ps(_mm512_set1_ps(weights0[key]), NCNN_MOE_BF16_ROW2_LOAD(row, 0), accumulator);
            row += value_stride;
        }
        row = values1 + column;
        for (uint32_t key = 0; key < count1; ++key)
        {
            accumulator = _mm512_fmadd_ps(_mm512_set1_ps(weights1[key]), NCNN_MOE_BF16_ROW2_LOAD(row, 0), accumulator);
            row += value_stride;
        }
        _mm512_storeu_ps(output + column, _mm512_add_ps(_mm512_loadu_ps(output + column), accumulator));
    }
#undef NCNN_MOE_BF16_ROW2_LOAD
    for (; column < columns; ++column)
    {
        const uint16_t* row = values0 + column;
        float sum = 0.0f;
        for (uint32_t key = 0; key < count0; ++key)
        {
            const uint32_t bits = static_cast<uint32_t>(row[0]) << 16;
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            sum += weights0[key] * value;
            row += value_stride;
        }
        row = values1 + column;
        for (uint32_t key = 0; key < count1; ++key)
        {
            const uint32_t bits = static_cast<uint32_t>(row[0]) << 16;
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            sum += weights1[key] * value;
            row += value_stride;
        }
        output[column] += sum;
    }
}

} // namespace moe
} // namespace ncnn

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512bf16")
#endif

namespace ncnn {
namespace moe {

void avx512_float_to_bfloat16_array(uint16_t* output,
                                    const float* input,
                                    uint32_t count) noexcept
{
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16)
    {
        const __m256i packed = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(input + index));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + index),
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

float avx512_bfloat16_pair_dot(const uint16_t* left,
                               const uint16_t* right,
                               uint32_t count) noexcept
{
    __m512 accumulator = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 32 <= count; index += 32)
    {
        accumulator = _mm512_dpbf16_ps(accumulator,
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

static NCNN_MOE_FORCE_INLINE void avx512_bfloat16_linear_tile1x4(const uint16_t* weights,
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
        const __m512bh weight1 = (__m512bh)_mm512_loadu_si512(weights + input_columns + column);
        const __m512bh weight2 = (__m512bh)_mm512_loadu_si512(weights + static_cast<size_t>(input_columns) * 2 + column);
        const __m512bh weight3 = (__m512bh)_mm512_loadu_si512(weights + static_cast<size_t>(input_columns) * 3 + column);
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

static NCNN_MOE_FORCE_INLINE void avx512_bfloat16_linear_tile4x6(const uint16_t* weights,
                                                                 const uint16_t* input,
                                                                 uint32_t input_columns,
                                                                 float* output,
                                                                 size_t output_stride,
                                                                 size_t valid_rows) noexcept
{
    __m512 a00 = _mm512_setzero_ps();
    __m512 a01 = a00, a02 = a00, a03 = a00, a04 = a00, a05 = a00;
    __m512 a10 = a00, a11 = a00, a12 = a00, a13 = a00, a14 = a00, a15 = a00;
    __m512 a20 = a00, a21 = a00, a22 = a00, a23 = a00, a24 = a00, a25 = a00;
    __m512 a30 = a00, a31 = a00, a32 = a00, a33 = a00, a34 = a00, a35 = a00;
    for (uint32_t column = 0; column < input_columns; column += 32)
    {
        const __m512bh input0 = (__m512bh)_mm512_loadu_si512(input + column);
        const __m512bh input1 = (__m512bh)_mm512_loadu_si512(input + input_columns + column);
        const __m512bh input2 = (__m512bh)_mm512_loadu_si512(input + static_cast<size_t>(input_columns) * 2 + column);
        const __m512bh input3 = (__m512bh)_mm512_loadu_si512(input + static_cast<size_t>(input_columns) * 3 + column);
#define NCNN_MOE_BF16_TILE4X6_STEP(index)                                                                                    \
    do                                                                                                                       \
    {                                                                                                                        \
        const __m512bh weight = (__m512bh)_mm512_loadu_si512(weights + static_cast<size_t>(index) * input_columns + column); \
        a0##index = _mm512_dpbf16_ps(a0##index, input0, weight);                                                             \
        a1##index = _mm512_dpbf16_ps(a1##index, input1, weight);                                                             \
        a2##index = _mm512_dpbf16_ps(a2##index, input2, weight);                                                             \
        a3##index = _mm512_dpbf16_ps(a3##index, input3, weight);                                                             \
    } while (0)
        NCNN_MOE_BF16_TILE4X6_STEP(0);
        NCNN_MOE_BF16_TILE4X6_STEP(1);
        NCNN_MOE_BF16_TILE4X6_STEP(2);
        NCNN_MOE_BF16_TILE4X6_STEP(3);
        NCNN_MOE_BF16_TILE4X6_STEP(4);
        NCNN_MOE_BF16_TILE4X6_STEP(5);
#undef NCNN_MOE_BF16_TILE4X6_STEP
    }
    output[0] = _mm512_reduce_add_ps(a00);
    output[1] = _mm512_reduce_add_ps(a01);
    output[2] = _mm512_reduce_add_ps(a02);
    output[3] = _mm512_reduce_add_ps(a03);
    output[4] = _mm512_reduce_add_ps(a04);
    output[5] = _mm512_reduce_add_ps(a05);
    if (valid_rows > 1)
    {
        float* row = output + output_stride;
        row[0] = _mm512_reduce_add_ps(a10);
        row[1] = _mm512_reduce_add_ps(a11);
        row[2] = _mm512_reduce_add_ps(a12);
        row[3] = _mm512_reduce_add_ps(a13);
        row[4] = _mm512_reduce_add_ps(a14);
        row[5] = _mm512_reduce_add_ps(a15);
    }
    if (valid_rows > 2)
    {
        float* row = output + output_stride * 2;
        row[0] = _mm512_reduce_add_ps(a20);
        row[1] = _mm512_reduce_add_ps(a21);
        row[2] = _mm512_reduce_add_ps(a22);
        row[3] = _mm512_reduce_add_ps(a23);
        row[4] = _mm512_reduce_add_ps(a24);
        row[5] = _mm512_reduce_add_ps(a25);
    }
    if (valid_rows > 3)
    {
        float* row = output + output_stride * 3;
        row[0] = _mm512_reduce_add_ps(a30);
        row[1] = _mm512_reduce_add_ps(a31);
        row[2] = _mm512_reduce_add_ps(a32);
        row[3] = _mm512_reduce_add_ps(a33);
        row[4] = _mm512_reduce_add_ps(a34);
        row[5] = _mm512_reduce_add_ps(a35);
    }
}

static inline float bf16_pair_dot(const uint16_t* left, const uint16_t* right, uint32_t count) noexcept
{
    __m512 accumulator = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 32 <= count; index += 32)
        accumulator = _mm512_dpbf16_ps(accumulator,
                                       (__m512bh)_mm512_loadu_si512(left + index),
                                       (__m512bh)_mm512_loadu_si512(right + index));
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

void avx512_bfloat16_batched_linear(const uint16_t* weights,
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
    const int64_t output_groups6 = output_columns / 6;
    const uint32_t remainder_columns = output_columns - static_cast<uint32_t>(output_groups6) * 6;
    if (token_count == 1)
    {
#pragma omp parallel num_threads(thread_count) if (thread_count > 1)
        {
            cpu_bind_thread();
#pragma omp for schedule(static)
            for (int64_t output_group = 0; output_group < output_groups;
                 ++output_group)
            {
                const uint32_t first_output = static_cast<uint32_t>(output_group) * 4;
                avx512_bfloat16_linear_tile1x4(weights + static_cast<size_t>(first_output) * input_columns,
                                               packed_input_data,
                                               input_columns,
                                               output + first_output);
            }
        }
        return;
    }
    // Walk a token block for every output group before advancing.  The packed
    // activation block then stays resident while the weight panel streams, and
    // each activation element is fetched from L2 instead of once per output
    // group from L3.
    const size_t token_block = 32;
#pragma omp parallel num_threads(thread_count) if (thread_count > 1)
    {
        cpu_bind_thread();
#pragma omp for schedule(static)
        for (int64_t output_group = 0; output_group < output_groups6; ++output_group)
        {
            const uint32_t first_output = static_cast<uint32_t>(output_group) * 6;
            const uint16_t* group_weights = weights + static_cast<size_t>(first_output) * input_columns;
            for (size_t block = 0; block < token_count; block += token_block)
            {
                const size_t block_end = std::min(token_count, block + token_block);
                for (size_t token = block; token < block_end; token += 4)
                {
                    const size_t valid_rows = std::min<size_t>(4, token_count - token);
                    avx512_bfloat16_linear_tile4x6(group_weights,
                                                   packed_input_data + token * input_columns,
                                                   input_columns,
                                                   output + token * output_stride + first_output,
                                                   output_stride,
                                                   valid_rows);
                }
            }
        }
        if (remainder_columns != 0)
        {
#pragma omp for schedule(static)
            for (int64_t output_group = 0; output_group < static_cast<int64_t>(remainder_columns); ++output_group)
            {
                const uint32_t output_column = static_cast<uint32_t>(output_groups6) * 6
                                               + static_cast<uint32_t>(output_group);
                const uint16_t* column_weights = weights + static_cast<size_t>(output_column) * input_columns;
                for (size_t token = 0; token < token_count; ++token)
                {
                    const uint16_t* token_input = packed_input_data + token * input_columns;
                    output[token * output_stride + output_column] = bf16_pair_dot(column_weights, token_input, input_columns);
                }
            }
        }
    }
}

void avx512_bfloat16_single_token_linear(const uint16_t* weights,
                                         const float* input,
                                         uint32_t output_columns,
                                         uint32_t input_columns,
                                         float* output,
                                         int thread_count) noexcept
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
            const __m256i packed00 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + column));
            const __m256i packed01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + column + 16));
            const __m256i packed02 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + column + 32));
            const __m256i packed03 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + column + 48));
            const __m256i packed10 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + input_columns + column));
            const __m256i packed11 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + input_columns + column + 16));
            const __m256i packed12 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + input_columns + column + 32));
            const __m256i packed13 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + input_columns + column + 48));
            const __m256i packed20 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column));
            const __m256i packed21 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column + 16));
            const __m256i packed22 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column + 32));
            const __m256i packed23 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column + 48));
            const __m256i packed30 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column));
            const __m256i packed31 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column + 16));
            const __m256i packed32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column + 32));
            const __m256i packed33 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column + 48));
#define NCNN_MOE_BF16_SINGLE_TOKEN_FMA(accumulator, packed, input_values)                                    \
    accumulator = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16)), \
                                  input_values,                                                              \
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
        __m512 sum0 = _mm512_add_ps(_mm512_add_ps(accumulator00, accumulator01),
                                    _mm512_add_ps(accumulator02, accumulator03));
        __m512 sum1 = _mm512_add_ps(_mm512_add_ps(accumulator10, accumulator11),
                                    _mm512_add_ps(accumulator12, accumulator13));
        __m512 sum2 = _mm512_add_ps(_mm512_add_ps(accumulator20, accumulator21),
                                    _mm512_add_ps(accumulator22, accumulator23));
        __m512 sum3 = _mm512_add_ps(_mm512_add_ps(accumulator30, accumulator31),
                                    _mm512_add_ps(accumulator32, accumulator33));
        for (; column + 16 <= input_columns; column += 16)
        {
            const __m512 input_values = _mm512_loadu_ps(input + column);
            const __m256i packed0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + column));
            const __m256i packed1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + input_columns + column));
            const __m256i packed2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 2 + column));
            const __m256i packed3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_weights + static_cast<size_t>(input_columns) * 3 + column));
            sum0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed0), 16)),
                                   input_values,
                                   sum0);
            sum1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed1), 16)),
                                   input_values,
                                   sum1);
            sum2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed2), 16)),
                                   input_values,
                                   sum2);
            sum3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(packed3), 16)),
                                   input_values,
                                   sum3);
        }
        output[first_output] = _mm512_reduce_add_ps(sum0);
        output[first_output + 1] = _mm512_reduce_add_ps(sum1);
        output[first_output + 2] = _mm512_reduce_add_ps(sum2);
        output[first_output + 3] = _mm512_reduce_add_ps(sum3);
    }
}

} // namespace moe
} // namespace ncnn

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif
