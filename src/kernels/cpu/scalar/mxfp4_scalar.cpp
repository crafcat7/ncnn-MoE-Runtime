#include "kernels/cpu/scalar/scalar_internal.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__)
#include <immintrin.h>
#endif

namespace ncnn {
namespace moe {

static void unpack_gguf_mxfp4_rows_baseline(const uint8_t* source, uint8_t* blocks, uint8_t* scales,
                                            size_t rows, size_t blocks_per_row,
                                            size_t block_row_stride, size_t scale_row_stride) noexcept
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__)
    const __m128i low_mask = _mm_set1_epi16(0x000f);
    const __m128i high_mask = _mm_set1_epi16(0x00f0);
#endif
    for (size_t row = 0; row < rows; ++row)
    {
        const uint8_t* input = source + row * blocks_per_row * 17;
        uint8_t* packed_row = blocks + row * block_row_stride;
        uint8_t* scale_row = scales + row * scale_row_stride;
        for (size_t block = 0; block < blocks_per_row; ++block)
        {
            const uint8_t* codes = input + block * 17 + 1;
            uint8_t* output = packed_row + block * 16;
            scale_row[block] = input[block * 17];
#if defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__)
            // Each 16-bit lane holds consecutive GGUF codes. Join their low
            // nibbles, then their high nibbles, without expanding to 32 values.
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));
            const __m128i low = _mm_or_si128(_mm_and_si128(packed, low_mask),
                                             _mm_and_si128(_mm_srli_epi16(packed, 4), high_mask));
            const __m128i high = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(packed, 4), low_mask),
                                              _mm_and_si128(_mm_srli_epi16(packed, 8), high_mask));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output), _mm_packus_epi16(low, high));
#else
            for (size_t pair = 0; pair < 8; ++pair)
            {
                const uint8_t first = codes[pair * 2];
                const uint8_t second = codes[pair * 2 + 1];
                output[pair] = static_cast<uint8_t>((first & 15u) | (second & 15u) << 4);
                output[pair + 8] = static_cast<uint8_t>((first >> 4) | (second & 240u));
            }
#endif
        }
    }
}

static std::array<float, 256> make_scale_table()
{
    std::array<float, 256> table = {};
    for (uint32_t index = 0; index < table.size(); ++index)
    {
        // GGML E8M0 scale is a half-scale because its MXFP4 codebook is
        // doubled.  Construct the exact bit pattern so 0 and 255 remain
        // finite subnormal/normal powers of two instead of underflowing or
        // becoming infinity.
        const uint32_t bits = index < 2 ? (UINT32_C(0x00200000) << index)
                                        : (static_cast<uint32_t>(index - 1) << 23);
        std::memcpy(&table[index], &bits, sizeof(bits));
    }
    return table;
}

static const std::array<float, 256>& scale_table()
{
    static const std::array<float, 256> values = make_scale_table();
    return values;
}

static float scalar_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
{
    static constexpr float values[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f, -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};
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
    static constexpr float values[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f, -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};
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
    static constexpr float values[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f, -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};
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

static void scalar_mxfp4_q8_quantize(const float* source, int8_t* values, float* scales, uint32_t columns) noexcept
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
               * (scales_by_exponent[scales[block]])
               * input_scales[block];
    }
    return sum;
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

static void scalar_mxfp4_q8_gemm_row(const uint8_t* packed,
                                     const uint8_t* scales,
                                     uint32_t block_count,
                                     const int8_t* input,
                                     size_t input_stride,
                                     const float* input_scales,
                                     size_t scale_stride,
                                     size_t token_count,
                                     float* output,
                                     size_t output_stride) noexcept
{
    const auto dot_fn = cpu_ops().mxfp4_q8_dot;
    for (size_t token = 0; token < token_count; ++token)
        output[token * output_stride] = dot_fn(packed, scales, block_count, input + token * input_stride, input_scales + token * scale_stride);
}

static void scalar_mxfp4_q8_matmul_rows2(const uint8_t* first_packed,
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
    const auto dot_fn = cpu_ops().mxfp4_q8_dot;
    for (size_t token = 0; token < token_count; ++token)
    {
        const int8_t* input_row = input + token * input_stride;
        const float* scale_row = input_scales + token * scale_stride;
        first_output[token * first_output_stride] = dot_fn(first_packed, first_scales, block_count, input_row, scale_row);
        second_output[token * second_output_stride] = dot_fn(second_packed, second_scales, block_count, input_row, scale_row);
    }
}

static void scalar_mxfp4_q8_matmul_row_pairs(const uint8_t* packed,
                                             const uint8_t* scales,
                                             uint32_t block_count,
                                             uint32_t row_pair_count,
                                             const int8_t* input,
                                             size_t input_stride,
                                             const float* input_scales,
                                             size_t scale_stride,
                                             size_t token_count,
                                             float* first_output,
                                             size_t first_pair_stride,
                                             size_t first_token_stride,
                                             float* second_output,
                                             size_t second_pair_stride,
                                             size_t second_token_stride) noexcept
{
    const size_t packed_row_bytes = static_cast<size_t>(block_count) * 16;
    const auto rows2_fn = cpu_ops().mxfp4_q8_matmul_rows2;
    for (uint32_t pair = 0; pair < row_pair_count; ++pair)
    {
        const size_t first_row = static_cast<size_t>(pair) * 2;
        rows2_fn(packed + first_row * packed_row_bytes,
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

void scalar_mxfp4_q8_packed_gemm(const uint8_t* packed,
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
    if (!packed || !input || !input_scales || !output || row_count == 0 || block_count == 0
        || (tile_rows != 4 && tile_rows != 8) || token_count == 0)
        return;

    const uint32_t chunk_bytes = mxfp4_q8_packed_chunk_bytes(tile_rows);
    const uint32_t chunk_count = 16 / chunk_bytes;
    const size_t block_bytes = mxfp4_q8_packed_block_bytes(tile_rows);
    const size_t group_stride = static_cast<size_t>(block_count) * block_bytes;
    const size_t group_count = (static_cast<size_t>(row_count) + tile_rows - 1) / tile_rows;
    const std::array<float, 256>& scales_by_exponent = scale_table();

    if (token_count == 1)
    {
        std::fill_n(output, row_count, 0.0f);
        for (size_t group = 0; group < group_count; ++group)
        {
            float accumulators[8] = {};
            for (uint32_t block = 0; block < block_count; ++block)
            {
                const uint8_t* packed_block = packed + group * group_stride + static_cast<size_t>(block) * block_bytes;
                const uint8_t* packed_values = packed_block + tile_rows;
                const float input_scale = input_scales[block];
                for (uint32_t chunk = 0; chunk < chunk_count; ++chunk)
                {
                    const int8_t* input_chunk = input + static_cast<size_t>(block) * 32 + chunk * chunk_bytes * 2;
                    for (uint32_t row = 0; row < tile_rows; ++row)
                    {
                        const size_t matrix_row = group * tile_rows + row;
                        if (matrix_row >= row_count)
                            continue;
                        const uint8_t* row_values = packed_values + (static_cast<size_t>(chunk) * tile_rows + row) * chunk_bytes;
                        accumulators[row] += static_cast<float>(scalar_packed_chunk_dot(row_values, input_chunk, chunk_bytes))
                                             * (scales_by_exponent[packed_block[row]]) * input_scale;
                    }
                }
            }
            for (uint32_t row = 0; row < tile_rows; ++row)
            {
                const size_t matrix_row = group * tile_rows + row;
                if (matrix_row < row_count)
                    output[matrix_row] = accumulators[row];
            }
        }
        return;
    }

    for (size_t token = 0; token < token_count; ++token)
        std::fill(output + token * output_stride, output + token * output_stride + row_count, 0.0f);

    for (size_t group = 0; group < group_count; ++group)
    {
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t* packed_block = packed + group * group_stride + static_cast<size_t>(block) * block_bytes;
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
                        if (matrix_row >= row_count)
                            continue;
                        const uint8_t* row_values = packed_values + (static_cast<size_t>(chunk) * tile_rows + row) * chunk_bytes;
                        output[token * output_stride + matrix_row] += static_cast<float>(scalar_packed_chunk_dot(row_values, input_chunk, chunk_bytes))
                                                                      * (scales_by_exponent[packed_block[row]]) * input_scale;
                    }
                }
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

void register_scalar_mxfp4_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept
{
    ops.unpack_gguf_mxfp4_rows = unpack_gguf_mxfp4_rows_baseline;
    ops.mxfp4_dot = scalar_dot;
    ops.mxfp4_gemm_row = scalar_gemm_row;
    ops.mxfp4_matmul_rows2 = scalar_matmul_rows2;
    ops.mxfp4_matmul_row_pairs = scalar_matmul_row_pairs;
    ops.mxfp4_q8_dot = scalar_mxfp4_q8_dot;
    ops.mxfp4_q8_quantize = scalar_mxfp4_q8_quantize;
    ops.mxfp4_q8_gemm_row = scalar_mxfp4_q8_gemm_row;
    ops.mxfp4_q8_matmul_rows2 = scalar_mxfp4_q8_matmul_rows2;
    ops.mxfp4_q8_matmul_row_pairs = scalar_mxfp4_q8_matmul_row_pairs;
    ops.mxfp4_q8_packed_gemm = scalar_mxfp4_q8_packed_gemm;
    caps.mxfp4_kind = MxFp4KernelKind::Scalar;
    caps.mxfp4_name = "scalar";
    caps.mxfp4_decode_pair_group = 1;
    caps.mxfp4_q8_ready = false;
}

} // namespace moe
} // namespace ncnn
