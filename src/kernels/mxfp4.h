#ifndef NCNN_MOE_MXFP4_H
#define NCNN_MOE_MXFP4_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

enum class MxFp4KernelKind
{
    Scalar,
    ArmNeon,
    ArmSve2,
    X86Avx2,
    X86Avx512
};

[[nodiscard]] MxFp4KernelKind mxfp4_kernel_kind() noexcept;
[[nodiscard]] const char* mxfp4_kernel_name() noexcept;
[[nodiscard]] uint32_t mxfp4_decode_row_pair_group_size() noexcept;

[[nodiscard]] float mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept;

void mxfp4_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
                    float* output, size_t output_stride) noexcept;

void mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                        uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output, size_t first_output_stride,
                        float* second_output, size_t second_output_stride) noexcept;

void mxfp4_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,
                            size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                            float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept;

struct Mxfp4Q8Batch
{
    std::vector<int8_t> values;
    std::vector<float> scales;
    size_t rows = 0;
    uint32_t columns = 0;

    void reset(size_t row_count, uint32_t column_count);
    [[nodiscard]] int8_t* row(size_t index) noexcept
    {
        return values.data() + index * columns;
    }
    [[nodiscard]] const int8_t* row(size_t index) const noexcept
    {
        return values.data() + index * columns;
    }
    [[nodiscard]] const float* row_scales(size_t index) const noexcept
    {
        return scales.data() + index * ((columns + 31) / 32);
    }
};

void mxfp4_q8_quantize(const float* source, int8_t* values, float* scales, uint32_t columns) noexcept;
void mxfp4_q8_quantize_batch(const float* source, size_t input_stride, size_t rows, uint32_t columns, Mxfp4Q8Batch& output) noexcept;
[[nodiscard]] bool mxfp4_q8_kernel_available() noexcept;
[[nodiscard]] float mxfp4_q8_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count,
                                 const int8_t* input, const float* input_scales) noexcept;
void mxfp4_q8_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const int8_t* input,
                       size_t input_stride, const float* input_scales, size_t scale_stride, size_t token_count,
                       float* output, size_t output_stride) noexcept;
void mxfp4_q8_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed,
                           const uint8_t* second_scales, uint32_t block_count, const int8_t* input, size_t input_stride,
                           const float* input_scales, size_t scale_stride, size_t token_count, float* first_output,
                           size_t first_output_stride, float* second_output, size_t second_output_stride) noexcept;
void mxfp4_q8_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count,
                               const int8_t* input, size_t input_stride, const float* input_scales, size_t scale_stride,
                               size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                               float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept;

// Persistent MXFP4-Q8 weight layout uses 4x4/8x8 interleaved blocks.  The
// object owns the reordered storage so callers can
// pack a model weight once and reuse it across decode/prefill calls.
struct Mxfp4Q8PackedMatrix
{
    std::vector<uint8_t> storage;
    size_t rows = 0;
    uint32_t block_count = 0;
    uint32_t tile_rows = 0;

    void clear() noexcept
    {
        storage.clear();
        rows = 0;
        block_count = 0;
        tile_rows = 0;
    }

    [[nodiscard]] bool valid() const noexcept
    {
        return rows != 0 && block_count != 0 && (tile_rows == 4 || tile_rows == 8) && !storage.empty();
    }

    [[nodiscard]] size_t group_count() const noexcept
    {
        return tile_rows == 0 ? 0 : (rows + tile_rows - 1) / tile_rows;
    }
};

// Returns the default interleave width for a matrix with row_count outputs.
// x86 AVX2/AVX512 uses 8-row blocks when possible; scalar/ARM and small tails
// use the 4-row layout.  Passing an explicit tile_rows to pack_weights must
// be either 4 or 8.
[[nodiscard]] uint32_t mxfp4_q8_packed_tile_rows(size_t row_count) noexcept;
[[nodiscard]] uint64_t mxfp4_q8_packed_storage_bytes(
    size_t row_count,
    uint32_t block_count,
    uint32_t tile_rows = 0) noexcept;

// True only when the current ISA has a tuned implementation for the
// interleaved layout.  Keeping this separate from the packer lets callers
// cache the sidecar without accidentally selecting a slower fallback.
[[nodiscard]] bool mxfp4_q8_packed_kernel_available() noexcept;

// Reorders row-major MXFP4 weights (16 bytes and one E8M0 scale per 32-value
// block) into persistent 4x4 or 8x8 interleaved blocks.  The source buffers
// remain owned by the caller and may be released after this returns.
[[nodiscard]] bool mxfp4_q8_pack_weights(const uint8_t* packed, const uint8_t* scales, uint32_t block_count,
                                         size_t row_count, Mxfp4Q8PackedMatrix& output, uint32_t tile_rows = 0);

// Output layout is token-major: output[token * output_stride + row].  GEMV
// writes one contiguous output vector; GEMM is the multi-token path.  Neither
// function repacks weights; the packed object must outlive the call.
void mxfp4_q8_packed_gemv(const Mxfp4Q8PackedMatrix& weights, const int8_t* input, const float* input_scales,
                          float* output) noexcept;
void mxfp4_q8_packed_gemm(const Mxfp4Q8PackedMatrix& weights, const int8_t* input, size_t input_stride,
                          const float* input_scales, size_t scale_stride, size_t token_count, float* output,
                          size_t output_stride) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MXFP4_H
