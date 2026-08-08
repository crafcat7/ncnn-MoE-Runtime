#ifndef NCNN_MOE_CPU_QNK_H
#define NCNN_MOE_CPU_QNK_H

#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

class CpuBatch;

// These sizes are the wire-compatible llama.cpp QK_K=256 layouts.  Q8_K is
// included because it is the canonical activation format used by the ggml
// Q2_K..Q6_K dot kernels, even though this runtime normally keeps activations
// as float32 CpuBatch rows.
inline constexpr uint32_t qnk_block_elements = 256;

[[nodiscard]] size_t qnk_block_bytes(DType dtype) noexcept;
[[nodiscard]] uint64_t qnk_storage_bytes(DType dtype, size_t rows, uint32_t columns) noexcept;
[[nodiscard]] bool qnk_shape_supported(DType dtype, size_t rows, uint32_t columns) noexcept;

// The persistent CPU layout is group-major:
//   [8-row tile][super-block][row in tile][raw Qn_K block]
// Rows in the final tile are zero-padded.  Keeping the raw block bytes in the
// sidecar means the pack is lossless and can be discarded/rebuilt per ISA.
struct QnKPack
{
    std::vector<uint8_t> storage;
    DType dtype = DType::Q4K;
    size_t rows = 0;
    uint32_t columns = 0;
    uint32_t block_count = 0;
    uint32_t tile_rows = 8;

    void clear() noexcept
    {
        storage.clear();
        rows = 0;
        columns = 0;
        block_count = 0;
        tile_rows = 8;
    }

    [[nodiscard]] bool valid() const noexcept
    {
        return is_qnk_dtype(dtype) && rows != 0 && columns != 0 && block_count != 0
               && tile_rows == 8 && !storage.empty();
    }
};

[[nodiscard]] inline const uint8_t* qnk_packed_block(
    const QnKPack& weights,
    size_t row,
    uint32_t block) noexcept
{
    const size_t tile = row / weights.tile_rows;
    const size_t lane = row % weights.tile_rows;
    const size_t block_bytes = qnk_block_bytes(weights.dtype);
    const size_t offset = (((tile * weights.block_count + block) * weights.tile_rows + lane) * block_bytes);
    return weights.storage.data() + offset;
}

// Decodes exactly one llama.cpp-compatible super-block.  This is also the
// scalar correctness oracle used by the x86 SIMD implementations.
void qnk_dequantize_block(DType dtype, const uint8_t* block, float* output) noexcept;

[[nodiscard]] float qnk_dot_block(
    DType dtype,
    const uint8_t* block,
    const float* input) noexcept;

// Quantizes float32 activation rows into the exact Q8_K wire layout used by
// llama.cpp's Q2_K..Q6_K dot kernels.  The output contains
// [float d][256 int8 qs][16 int16 bsums] for every 256-value block.
void qnk_q8k_quantize(
    const float* source,
    uint8_t* output,
    uint32_t columns) noexcept;

void qnk_q8k_quantize_batch(
    const float* source,
    size_t input_stride,
    size_t rows,
    uint32_t columns,
    std::vector<uint8_t>& output) noexcept;

[[nodiscard]] bool qnk_pack_weights(
    const uint8_t* raw,
    size_t raw_bytes,
    DType dtype,
    size_t rows,
    uint32_t columns,
    QnKPack& output) noexcept;

[[nodiscard]] bool qnk_linear_batch_into(
    const TensorData& matrix,
    const CpuBatch& input,
    CpuBatch& output) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_QNK_H
