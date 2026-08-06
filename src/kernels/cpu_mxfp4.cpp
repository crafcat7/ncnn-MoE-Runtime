#include "cpu_mxfp4.h"
#include "cpu_bfloat16.h"
#include "engine/cpu_features.h"
#include "engine/cpu_thread_budget.h"
#include "ncnn/moe/runtime.h"

#if defined(NCNN_MOE_MSVC_X86_SIMD)
#include "cpu_mxfp4_msvc.h"
#include <intrin.h>
#endif
#if defined(NCNN_MOE_ARM_SVE2_KERNEL)
#include "cpu_mxfp4_sve2.h"
#include "ncnn/moe/runtime.h"
#endif

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

static constexpr int8_t mxfp4_integer_values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12};

void Mxfp4Q8Batch::reset(size_t row_count, uint32_t column_count)
{
    rows = row_count;
    columns = column_count;
    const size_t block_count = (static_cast<size_t>(column_count) + 31) / 32;
    values.resize(row_count * column_count);
    scales.resize(row_count * block_count);
}

void mxfp4_q8_quantize(const float* source, int8_t* values, float* scales, uint32_t columns) noexcept
{
    const uint32_t block_count = (columns + 31) / 32;
    for (uint32_t block = 0; block < block_count; ++block)
    {
        const uint32_t begin = block * 32;
        const uint32_t end = std::min(columns, begin + 32);
        float maximum = 0.0f;
        for (uint32_t column = begin; column < end; ++column)
            maximum = std::max(maximum, std::fabs(source[column]));
        const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scales[block] = scale;
        const float inverse_scale = 1.0f / scale;
        for (uint32_t column = begin; column < end; ++column)
        {
            const float normalized = std::clamp(source[column] * inverse_scale, -127.0f, 127.0f);
            values[column] = static_cast<int8_t>(std::lrintf(normalized));
        }
        for (uint32_t column = end; column < begin + 32 && column < columns; ++column)
            values[column] = 0;
    }
}

void mxfp4_q8_quantize_batch(const float* source, size_t input_stride, size_t rows, uint32_t columns, Mxfp4Q8Batch& output) noexcept
{
    output.reset(rows, columns);
    const size_t block_count = (static_cast<size_t>(columns) + 31) / 32;
    for (size_t row = 0; row < rows; ++row)
    {
        mxfp4_q8_quantize(
            source + row * input_stride,
            output.row(row),
            output.scales.data() + row * block_count,
            columns);
    }
}

static float scalar_mxfp4_q8_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count,
                                 const int8_t* input, const float* input_scales) noexcept
{
    const std::array<float, 256>& scales_by_exponent = scale_table();
    float sum = 0.0f;
    for (uint32_t block = 0; block < block_count; ++block)
    {
        const uint8_t* packed_block = packed + static_cast<size_t>(block) * 16;
        const int8_t* input_block = input + static_cast<size_t>(block) * 32;
        int32_t integer_sum = 0;
        for (uint32_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t packed_value = packed_block[byte];
            integer_sum += static_cast<int32_t>(mxfp4_integer_values[packed_value & 0x0f]) * input_block[byte * 2];
            integer_sum += static_cast<int32_t>(mxfp4_integer_values[packed_value >> 4]) * input_block[byte * 2 + 1];
        }
        sum += static_cast<float>(integer_sum)
               * (0.5f * scales_by_exponent[scales[block]])
               * input_scales[block];
    }
    return sum;
}

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,ssse3"))) static int32_t avx2_horizontal_sum_epi32(__m256i values) noexcept
{
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(values),
        _mm256_extracti128_si256(values, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(sum);
}

__attribute__((target("avx2,ssse3"))) static float avx2_mxfp4_q8_dot(
    const uint8_t* packed,
    const uint8_t* scales,
    uint32_t block_count,
    const int8_t* input,
    const float* input_scales) noexcept
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i value_table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const std::array<float, 256>& scales_by_exponent = scale_table();
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
        const __m256i product_low = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(decoded_low),
            input_low);
        const __m256i product_high = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(decoded_high),
            input_high);
        const int32_t integer_sum = avx2_horizontal_sum_epi32(_mm256_add_epi32(product_low, product_high));
        sum += static_cast<float>(integer_sum)
               * (0.5f * scales_by_exponent[scales[block]])
               * input_scales[block];
    }
    return sum;
}
#endif

float mxfp4_q8_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count,
                   const int8_t* input, const float* input_scales) noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512
        || mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
        return msvc_avx2_mxfp4_q8_dot(packed, scales, block_count, input, input_scales);
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    if ((detect_cpu_isa_capabilities().flags & CpuIsaX86Avx2Fma) != 0)
        return avx2_mxfp4_q8_dot(packed, scales, block_count, input, input_scales);
#endif
    return scalar_mxfp4_q8_dot(packed, scales, block_count, input, input_scales);
}

void mxfp4_q8_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const int8_t* input,
                       size_t input_stride, const float* input_scales, size_t scale_stride, size_t token_count,
                       float* output, size_t output_stride) noexcept
{
    for (size_t token = 0; token < token_count; ++token)
    {
        output[token * output_stride] = mxfp4_q8_dot(
            packed,
            scales,
            block_count,
            input + token * input_stride,
            input_scales + token * scale_stride);
    }
}

void mxfp4_q8_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed,
                           const uint8_t* second_scales, uint32_t block_count, const int8_t* input, size_t input_stride,
                           const float* input_scales, size_t scale_stride, size_t token_count, float* first_output,
                           size_t first_output_stride, float* second_output, size_t second_output_stride) noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512
        || mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
    {
        msvc_avx2_mxfp4_q8_matmul_rows2(
            first_packed,
            first_scales,
            second_packed,
            second_scales,
            block_count,
            input,
            input_stride,
            input_scales,
            scale_stride,
            token_count,
            first_output,
            first_output_stride,
            second_output,
            second_output_stride);
        return;
    }
#endif
    for (size_t token = 0; token < token_count; ++token)
    {
        const int8_t* input_row = input + token * input_stride;
        const float* scale_row = input_scales + token * scale_stride;
        first_output[token * first_output_stride] = mxfp4_q8_dot(
            first_packed, first_scales, block_count, input_row, scale_row);
        second_output[token * second_output_stride] = mxfp4_q8_dot(
            second_packed, second_scales, block_count, input_row, scale_row);
    }
}

void mxfp4_q8_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count,
                               const int8_t* input, size_t input_stride, const float* input_scales, size_t scale_stride,
                               size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                               float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept
{
    const size_t packed_row_bytes = static_cast<size_t>(block_count) * 16;
    for (uint32_t pair = 0; pair < row_pair_count; ++pair)
    {
        const size_t first_row = static_cast<size_t>(pair) * 2;
        mxfp4_q8_matmul_rows2(
            packed + first_row * packed_row_bytes,
            scales + first_row * block_count,
            packed + (first_row + 1) * packed_row_bytes,
            scales + (first_row + 1) * block_count,
            block_count,
            input,
            input_stride,
            input_scales,
            scale_stride,
            token_count,
            first_output + static_cast<size_t>(pair) * first_pair_stride,
            first_token_stride,
            second_output + static_cast<size_t>(pair) * second_pair_stride,
            second_token_stride);
    }
}

static constexpr uint32_t mxfp4_q8_packed_block_bytes(uint32_t tile_rows) noexcept
{
    return tile_rows * (1 + 16);
}

static constexpr uint32_t mxfp4_q8_packed_chunk_bytes(uint32_t tile_rows) noexcept
{
    return tile_rows;
}

static int32_t scalar_packed_chunk_dot(const uint8_t* packed, const int8_t* input, uint32_t chunk_bytes) noexcept
{
    int32_t sum = 0;
    for (uint32_t byte = 0; byte < chunk_bytes; ++byte)
    {
        const uint8_t value = packed[byte];
        sum += static_cast<int32_t>(mxfp4_integer_values[value & 0x0f]) * input[byte * 2];
        sum += static_cast<int32_t>(mxfp4_integer_values[value >> 4]) * input[byte * 2 + 1];
    }
    return sum;
}

static void scalar_packed_gemv(const Mxfp4Q8PackedMatrix& weights, const int8_t* input, const float* input_scales,
                               float* output) noexcept
{
    const uint32_t tile_rows = weights.tile_rows;
    const uint32_t chunk_bytes = mxfp4_q8_packed_chunk_bytes(tile_rows);
    const uint32_t chunk_count = 16 / chunk_bytes;
    const size_t block_bytes = mxfp4_q8_packed_block_bytes(tile_rows);
    const size_t group_stride = static_cast<size_t>(weights.block_count) * block_bytes;
    const std::array<float, 256>& scales_by_exponent = scale_table();

    for (size_t row = 0; row < weights.rows; ++row)
        output[row] = 0.0f;

    for (size_t group = 0; group < weights.group_count(); ++group)
    {
        float accumulators[8] = {};
        for (uint32_t block = 0; block < weights.block_count; ++block)
        {
            const uint8_t* packed_block = weights.storage.data() + group * group_stride + static_cast<size_t>(block) * block_bytes;
            const uint8_t* packed_values = packed_block + tile_rows;
            const float input_scale = input_scales[block];
            for (uint32_t chunk = 0; chunk < chunk_count; ++chunk)
            {
                const int8_t* input_chunk = input + static_cast<size_t>(block) * 32 + chunk * chunk_bytes * 2;
                for (uint32_t row = 0; row < tile_rows; ++row)
                {
                    const size_t matrix_row = group * tile_rows + row;
                    if (matrix_row >= weights.rows)
                        continue;
                    const uint8_t* row_values = packed_values + (static_cast<size_t>(chunk) * tile_rows + row) * chunk_bytes;
                    accumulators[row] += static_cast<float>(scalar_packed_chunk_dot(row_values, input_chunk, chunk_bytes))
                                         * (0.5f * scales_by_exponent[packed_block[row]]) * input_scale;
                }
            }
        }
        for (uint32_t row = 0; row < tile_rows; ++row)
        {
            const size_t matrix_row = group * tile_rows + row;
            if (matrix_row < weights.rows)
                output[matrix_row] = accumulators[row];
        }
    }
}

static void scalar_packed_gemm(const Mxfp4Q8PackedMatrix& weights, const int8_t* input, size_t input_stride,
                               const float* input_scales, size_t scale_stride, size_t token_count, float* output,
                               size_t output_stride) noexcept
{
    if (token_count == 0 || !weights.valid())
        return;
    if (token_count == 1)
    {
        scalar_packed_gemv(weights, input, input_scales, output);
        return;
    }

    for (size_t token = 0; token < token_count; ++token)
        std::fill(output + token * output_stride, output + token * output_stride + weights.rows, 0.0f);

    const uint32_t tile_rows = weights.tile_rows;
    const uint32_t chunk_bytes = mxfp4_q8_packed_chunk_bytes(tile_rows);
    const uint32_t chunk_count = 16 / chunk_bytes;
    const size_t block_bytes = mxfp4_q8_packed_block_bytes(tile_rows);
    const size_t group_stride = static_cast<size_t>(weights.block_count) * block_bytes;
    const std::array<float, 256>& scales_by_exponent = scale_table();
    for (size_t group = 0; group < weights.group_count(); ++group)
    {
        for (uint32_t block = 0; block < weights.block_count; ++block)
        {
            const uint8_t* packed_block = weights.storage.data() + group * group_stride + static_cast<size_t>(block) * block_bytes;
            const uint8_t* packed_values = packed_block + tile_rows;
            for (size_t token = 0; token < token_count; ++token)
            {
                const int8_t* input_block = input + token * input_stride + static_cast<size_t>(block) * 32;
                const float input_scale = input_scales[token * scale_stride + block];
                for (uint32_t chunk = 0; chunk < chunk_count; ++chunk)
                {
                    const int8_t* input_chunk = input_block + chunk * chunk_bytes * 2;
                    for (uint32_t row = 0; row < tile_rows; ++row)
                    {
                        const size_t matrix_row = group * tile_rows + row;
                        if (matrix_row >= weights.rows)
                            continue;
                        const uint8_t* row_values = packed_values + (static_cast<size_t>(chunk) * tile_rows + row) * chunk_bytes;
                        output[token * output_stride + matrix_row] += static_cast<float>(scalar_packed_chunk_dot(row_values, input_chunk, chunk_bytes))
                                                                      * (0.5f * scales_by_exponent[packed_block[row]]) * input_scale;
                    }
                }
            }
        }
    }
}

uint32_t mxfp4_q8_packed_tile_rows(size_t row_count) noexcept
{
    if (row_count >= 8 && (mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2 || mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512))
        return 8;
    return 4;
}

bool mxfp4_q8_packed_kernel_available() noexcept
{
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    return mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2
           || mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512;
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    // Keep the interleaved sidecar available until the compiler-specific kernel lands.
    return false;
#else
    return false;
#endif
}

bool mxfp4_q8_pack_weights(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, size_t row_count,
                           Mxfp4Q8PackedMatrix& output, uint32_t tile_rows)
{
    if (!packed || !scales || block_count == 0 || row_count == 0)
    {
        output.clear();
        return false;
    }
    if (tile_rows == 0)
        tile_rows = mxfp4_q8_packed_tile_rows(row_count);
    if (tile_rows != 4 && tile_rows != 8)
    {
        output.clear();
        return false;
    }

    const size_t group_count = (row_count + tile_rows - 1) / tile_rows;
    const size_t block_bytes = mxfp4_q8_packed_block_bytes(tile_rows);
    const size_t group_stride = static_cast<size_t>(block_count) * block_bytes;
    try
    {
        output.storage.assign(group_count * group_stride, 0);
    }
    catch (...)
    {
        output.clear();
        return false;
    }
    output.rows = row_count;
    output.block_count = block_count;
    output.tile_rows = tile_rows;

    const uint32_t chunk_bytes = mxfp4_q8_packed_chunk_bytes(tile_rows);
    const uint32_t chunk_count = 16 / chunk_bytes;
    const size_t source_row_bytes = static_cast<size_t>(block_count) * 16;
    for (size_t group = 0; group < group_count; ++group)
    {
        for (uint32_t block = 0; block < block_count; ++block)
        {
            uint8_t* destination = output.storage.data() + group * group_stride + static_cast<size_t>(block) * block_bytes;
            for (uint32_t row = 0; row < tile_rows; ++row)
            {
                const size_t source_row = group * tile_rows + row;
                if (source_row >= row_count)
                    continue;
                destination[row] = scales[source_row * block_count + block];
                const uint8_t* source = packed + source_row * source_row_bytes + static_cast<size_t>(block) * 16;
                for (uint32_t chunk = 0; chunk < chunk_count; ++chunk)
                {
                    std::memcpy(destination + tile_rows + (static_cast<size_t>(chunk) * tile_rows + row) * chunk_bytes,
                                source + chunk * chunk_bytes, chunk_bytes);
                }
            }
        }
    }
    return true;
}

void mxfp4_q8_packed_gemv(const Mxfp4Q8PackedMatrix& weights, const int8_t* input, const float* input_scales,
                          float* output) noexcept
{
    if (!input || !input_scales || !output || !weights.valid())
        return;
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (weights.tile_rows == 8 && mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512)
    {
        msvc_avx512_mxfp4_q8_packed_gemm(weights.storage.data(), static_cast<uint32_t>(weights.rows), weights.block_count, weights.tile_rows,
                                         input, 0, input_scales, 0, 1, output, weights.rows);
        return;
    }
    if (weights.tile_rows == 8 && mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
    {
        msvc_avx2_mxfp4_q8_packed_gemm(weights.storage.data(), static_cast<uint32_t>(weights.rows), weights.block_count, weights.tile_rows,
                                       input, 0, input_scales, 0, 1, output, weights.rows);
        return;
    }
#endif
    scalar_packed_gemv(weights, input, input_scales, output);
}

void mxfp4_q8_packed_gemm(const Mxfp4Q8PackedMatrix& weights, const int8_t* input, size_t input_stride,
                          const float* input_scales, size_t scale_stride, size_t token_count, float* output,
                          size_t output_stride) noexcept
{
    if (!input || !input_scales || !output || !weights.valid())
        return;
    if (token_count == 1)
    {
        mxfp4_q8_packed_gemv(weights, input, input_scales, output);
        return;
    }
#if defined(NCNN_MOE_MSVC_X86_SIMD)
    if (weights.tile_rows == 8 && mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512)
    {
        msvc_avx512_mxfp4_q8_packed_gemm(weights.storage.data(), static_cast<uint32_t>(weights.rows), weights.block_count, weights.tile_rows,
                                         input, input_stride, input_scales, scale_stride, token_count, output, output_stride);
        return;
    }
    if (weights.tile_rows == 8 && mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2)
    {
        msvc_avx2_mxfp4_q8_packed_gemm(weights.storage.data(), static_cast<uint32_t>(weights.rows), weights.block_count, weights.tile_rows,
                                       input, input_stride, input_scales, scale_stride, token_count, output, output_stride);
        return;
    }
#endif
    scalar_packed_gemm(weights, input, input_stride, input_scales, scale_stride, token_count, output, output_stride);
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

    // Select the highest supported ISA without an initialization benchmark.
    constexpr std::array<MxFp4KernelKind, 5> preference = {
        MxFp4KernelKind::X86Avx512,
        MxFp4KernelKind::ArmSve2,
        MxFp4KernelKind::X86Avx2,
        MxFp4KernelKind::ArmNeon,
        MxFp4KernelKind::Scalar,
    };
    for (const MxFp4KernelKind preferred : preference)
    {
        for (size_t index = 0; index < candidate_count; ++index)
        {
            if (candidates[index].kind == preferred)
                return candidates[index];
        }
    }
    return candidates[0];
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

uint32_t mxfp4_decode_row_pair_group_size() noexcept
{
    // Keep the row-pair tile static; decode must not benchmark on first use.
    return mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx2
                   || mxfp4_kernel_kind() == MxFp4KernelKind::X86Avx512
               ? 2
               : 1;
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
