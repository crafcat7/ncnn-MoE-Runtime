#include "kernels/mxfp4.h"

#include "kernels/cpu/cpu_ops.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>

namespace ncnn {
namespace moe {

static bool gguf_mxfp4_overlap(const uint8_t* first,
                               size_t first_size,
                               const uint8_t* second,
                               size_t second_size) noexcept
{
    const uintptr_t first_address = reinterpret_cast<uintptr_t>(first);
    const uintptr_t second_address = reinterpret_cast<uintptr_t>(second);
    return first_address <= second_address ? second_address - first_address < first_size
                                           : first_address - second_address < second_size;
}

bool unpack_gguf_mxfp4_rows(std::span<const uint8_t> source,
                            size_t blocks_per_row,
                            size_t block_row_stride,
                            size_t scale_row_stride,
                            std::span<uint8_t> blocks,
                            std::span<uint8_t> scales) noexcept
{
    if (source.empty() || source.size() % 17 != 0 || blocks_per_row == 0
        || source.size() / 17 % blocks_per_row != 0)
        return false;
    const size_t rows = source.size() / 17 / blocks_per_row;
    const size_t block_row_bytes = blocks_per_row * 16;
    if (block_row_stride < block_row_bytes || scale_row_stride < blocks_per_row
        || blocks.size() < block_row_bytes || scales.size() < blocks_per_row
        || rows - 1 > (blocks.size() - block_row_bytes) / block_row_stride
        || rows - 1 > (scales.size() - blocks_per_row) / scale_row_stride
        || gguf_mxfp4_overlap(source.data(), source.size(), blocks.data(), blocks.size())
        || gguf_mxfp4_overlap(source.data(), source.size(), scales.data(), scales.size())
        || gguf_mxfp4_overlap(blocks.data(), blocks.size(), scales.data(), scales.size()))
        return false;
    if (blocks_per_row == 1)
    {
        for (size_t row = 0; row < rows; ++row)
        {
            const uint8_t* input = source.data() + row * 17;
            const uint8_t* codes = input + 1;
            uint8_t* output = blocks.data() + row * block_row_stride;
            scales[row * scale_row_stride] = input[0];
            for (size_t pair = 0; pair < 8; ++pair)
            {
                const uint8_t first = codes[pair * 2];
                const uint8_t second = codes[pair * 2 + 1];
                output[pair] = static_cast<uint8_t>((first & 15u) | ((second & 15u) << 4));
                output[pair + 8] = static_cast<uint8_t>((first >> 4) | (second & 240u));
            }
        }
        return true;
    }
    cpu_ops().unpack_gguf_mxfp4_rows(source.data(), blocks.data(), scales.data(), rows,
                                     blocks_per_row, block_row_stride, scale_row_stride);
    return true;
}

bool unpack_gguf_mxfp4(std::span<const uint8_t> source,
                       std::span<uint8_t> blocks,
                       std::span<uint8_t> scales) noexcept
{
    if (source.empty() || source.size() % 17 != 0
        || scales.size() != source.size() / 17 || blocks.size() != scales.size() * 16)
        return false;
    return unpack_gguf_mxfp4_rows(source, scales.size(), blocks.size(), scales.size(), blocks, scales);
}

bool pack_gguf_mxfp4(std::span<const uint8_t> blocks,
                     std::span<const uint8_t> scales,
                     std::span<uint8_t> destination) noexcept
{
    if (scales.empty() || scales.size() > std::numeric_limits<size_t>::max() / 17
        || blocks.size() != scales.size() * 16 || destination.size() != scales.size() * 17
        || gguf_mxfp4_overlap(blocks.data(), blocks.size(), scales.data(), scales.size())
        || gguf_mxfp4_overlap(blocks.data(), blocks.size(), destination.data(), destination.size())
        || gguf_mxfp4_overlap(scales.data(), scales.size(), destination.data(), destination.size()))
        return false;

    for (size_t block = 0; block < scales.size(); ++block)
    {
        const uint8_t* packed = blocks.data() + block * 16;
        uint8_t* encoded = destination.data() + block * 17;
        encoded[0] = scales[block];
        for (size_t pair = 0; pair < 8; ++pair)
        {
            encoded[1 + pair * 2] = static_cast<uint8_t>((packed[pair] & 15u)
                                                         | ((packed[pair + 8] & 15u) << 4));
            encoded[2 + pair * 2] = static_cast<uint8_t>((packed[pair] >> 4)
                                                         | (packed[pair + 8] & 240u));
        }
    }
    return true;
}

void mxfp4_dequantize_block(const uint8_t* packed, uint8_t scale, float* output) noexcept
{
    if (!packed || !output)
        return;
    static constexpr float doubled_e2m1[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
                                               -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};
    const uint32_t scale_bits = scale < 2 ? (UINT32_C(0x00200000) << scale)
                                          : (static_cast<uint32_t>(scale - 1) << 23);
    float block_scale = 0.0f;
    std::memcpy(&block_scale, &scale_bits, sizeof(block_scale));
    for (uint32_t index = 0; index < 16; ++index)
    {
        const uint8_t byte = packed[index];
        output[index * 2] = doubled_e2m1[byte & 0x0fu] * block_scale;
        output[index * 2 + 1] = doubled_e2m1[byte >> 4] * block_scale;
    }
}

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
    cpu_ops().mxfp4_q8_quantize(source, values, scales, columns);
}

void mxfp4_q8_quantize_batch(const float* source, size_t input_stride, size_t rows, uint32_t columns, Mxfp4Q8Batch& output) noexcept
{
    output.reset(rows, columns);
    const size_t block_count = (static_cast<size_t>(columns) + 31) / 32;
    for (size_t row = 0; row < rows; ++row)
    {
        mxfp4_q8_quantize(source + row * input_stride,
                          output.row(row),
                          output.scales.data() + row * block_count,
                          columns);
    }
}

float mxfp4_q8_dot(const uint8_t* packed,
                   const uint8_t* scales,
                   uint32_t block_count,
                   const int8_t* input,
                   const float* input_scales) noexcept
{
    return cpu_ops().mxfp4_q8_dot(packed, scales, block_count, input, input_scales);
}

void mxfp4_q8_gemm_row(const uint8_t* packed,
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
    cpu_ops().mxfp4_q8_gemm_row(packed, scales, block_count, input, input_stride, input_scales, scale_stride, token_count, output, output_stride);
}

void mxfp4_q8_matmul_rows2(const uint8_t* first_packed,
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
    cpu_ops().mxfp4_q8_matmul_rows2(first_packed, first_scales, second_packed, second_scales, block_count,
                                    input, input_stride, input_scales, scale_stride, token_count,
                                    first_output, first_output_stride, second_output, second_output_stride);
}

void mxfp4_q8_matmul_row_pairs(const uint8_t* packed,
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
    cpu_ops().mxfp4_q8_matmul_row_pairs(packed, scales, block_count, row_pair_count,
                                        input, input_stride, input_scales, scale_stride, token_count,
                                        first_output, first_pair_stride, first_token_stride,
                                        second_output, second_pair_stride, second_token_stride);
}

static uint32_t mxfp4_q8_packed_tile_rows(size_t row_count) noexcept
{
    const MxFp4KernelKind kind = cpu_backend_caps().mxfp4_kind;
    if (row_count >= 8 && (kind == MxFp4KernelKind::X86Avx2 || kind == MxFp4KernelKind::X86Avx512))
        return 8;
    return 4;
}

uint64_t mxfp4_q8_packed_storage_bytes(size_t row_count,
                                       uint32_t block_count,
                                       uint32_t tile_rows) noexcept
{
    if (row_count == 0 || block_count == 0)
        return 0;
    if (tile_rows == 0)
        tile_rows = mxfp4_q8_packed_tile_rows(row_count);
    if ((tile_rows != 4 && tile_rows != 8)
        || row_count > std::numeric_limits<size_t>::max() - (tile_rows - 1))
    {
        return 0;
    }
    const uint64_t group_count = (row_count + tile_rows - 1) / tile_rows;
    const uint64_t bytes_per_block = static_cast<uint64_t>(tile_rows) * 17;
    if (group_count > std::numeric_limits<uint64_t>::max() / block_count
        || group_count * block_count > std::numeric_limits<uint64_t>::max() / bytes_per_block)
    {
        return 0;
    }
    return group_count * block_count * bytes_per_block;
}

bool mxfp4_q8_ready() noexcept
{
    return cpu_backend_caps().mxfp4_q8_ready;
}

bool mxfp4_q8_kernel_available() noexcept
{
    return mxfp4_q8_ready();
}

bool mxfp4_q8_pack_weights(const uint8_t* packed,
                           const uint8_t* scales,
                           uint32_t block_count,
                           size_t row_count,
                           Mxfp4Q8PackedMatrix& output,
                           uint32_t tile_rows)
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

    const uint64_t storage_bytes = mxfp4_q8_packed_storage_bytes(row_count,
                                                                 block_count,
                                                                 tile_rows);
    if (storage_bytes == 0 || storage_bytes > std::numeric_limits<size_t>::max())
    {
        output.clear();
        return false;
    }

    const size_t group_count = (row_count + tile_rows - 1) / tile_rows;
    const size_t block_bytes = static_cast<size_t>(tile_rows) * 17;
    const size_t group_stride = static_cast<size_t>(block_count) * block_bytes;
    try
    {
        output.storage.assign(static_cast<size_t>(storage_bytes), 0);
    }
    catch (...)
    {
        output.clear();
        return false;
    }
    output.rows = row_count;
    output.block_count = block_count;
    output.tile_rows = tile_rows;

    const uint32_t chunk_bytes = tile_rows;
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

void mxfp4_q8_packed_gemv(const Mxfp4Q8PackedMatrix& weights,
                          const int8_t* input,
                          const float* input_scales,
                          float* output) noexcept
{
    if (!input || !input_scales || !output || !weights.valid())
        return;
    cpu_ops().mxfp4_q8_packed_gemm(weights.storage.data(),
                                   static_cast<uint32_t>(weights.rows),
                                   weights.block_count,
                                   weights.tile_rows,
                                   input,
                                   0,
                                   input_scales,
                                   0,
                                   1,
                                   output,
                                   weights.rows);
}

void mxfp4_q8_packed_gemm(const Mxfp4Q8PackedMatrix& weights,
                          const int8_t* input,
                          size_t input_stride,
                          const float* input_scales,
                          size_t scale_stride,
                          size_t token_count,
                          float* output,
                          size_t output_stride) noexcept
{
    if (!input || !input_scales || !output || !weights.valid())
        return;
    if (token_count == 1)
    {
        mxfp4_q8_packed_gemv(weights, input, input_scales, output);
        return;
    }
    cpu_ops().mxfp4_q8_packed_gemm(weights.storage.data(),
                                   static_cast<uint32_t>(weights.rows),
                                   weights.block_count,
                                   weights.tile_rows,
                                   input,
                                   input_stride,
                                   input_scales,
                                   scale_stride,
                                   token_count,
                                   output,
                                   output_stride);
}

MxFp4KernelKind mxfp4_kernel_kind() noexcept
{
    return cpu_backend_caps().mxfp4_kind;
}

const char* mxfp4_kernel_name() noexcept
{
    return cpu_backend_caps().mxfp4_name;
}

uint32_t mxfp4_decode_row_pair_group_size() noexcept
{
    return cpu_backend_caps().mxfp4_decode_pair_group;
}

float mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept
{
    return cpu_ops().mxfp4_dot(packed, scales, block_count, input);
}

void mxfp4_gemm_row(const uint8_t* packed,
                    const uint8_t* scales,
                    uint32_t block_count,
                    const float* input,
                    size_t input_stride,
                    size_t token_count,
                    float* output,
                    size_t output_stride) noexcept
{
    cpu_ops().mxfp4_gemm_row(packed, scales, block_count, input, input_stride, token_count, output, output_stride);
}

void mxfp4_matmul_rows2(const uint8_t* first_packed,
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
    cpu_ops().mxfp4_matmul_rows2(first_packed, first_scales, second_packed, second_scales, block_count,
                                 input, input_stride, token_count, first_output, first_output_stride,
                                 second_output, second_output_stride);
}

void mxfp4_matmul_row_pairs(const uint8_t* packed,
                            const uint8_t* scales,
                            uint32_t block_count,
                            uint32_t row_pair_count,
                            const float* input,
                            size_t input_stride,
                            size_t token_count,
                            float* first_output,
                            size_t first_pair_stride,
                            size_t first_token_stride,
                            float* second_output,
                            size_t second_pair_stride,
                            size_t second_token_stride) noexcept
{
    cpu_ops().mxfp4_matmul_row_pairs(packed, scales, block_count, row_pair_count, input, input_stride,
                                     token_count, first_output, first_pair_stride, first_token_stride,
                                     second_output, second_pair_stride, second_token_stride);
}

} // namespace moe
} // namespace ncnn
