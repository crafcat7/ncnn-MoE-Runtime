#ifndef NCNN_MOE_QNK_H
#define NCNN_MOE_QNK_H

#include "activationbuffer.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

// Qn_K uses 256-value blocks; Q8_K is the activation format for its dot products.
// This is also the maximum scratch size. Native row strides use qnk_block_size.
inline constexpr uint32_t qnk_block_elements = 256;

[[nodiscard]] uint32_t qnk_block_size(DType dtype) noexcept;
[[nodiscard]] size_t qnk_block_bytes(DType dtype) noexcept;
[[nodiscard]] uint64_t qnk_storage_bytes(DType dtype, size_t rows, uint32_t columns) noexcept;
[[nodiscard]] uint64_t qnk_packed_storage_bytes(DType dtype, size_t rows, uint32_t columns) noexcept;
[[nodiscard]] bool qnk_shape_supported(DType dtype, size_t rows, uint32_t columns) noexcept;

// Packed layout: [8-row tile][block][row][raw block].
struct QnKPack
{
    std::vector<uint8_t> storage;
    DType dtype = DType::Q4K;
    size_t rows = 0;
    uint32_t columns = 0;
    uint32_t block_count = 0;

    void clear() noexcept
    {
        storage.clear();
        rows = 0;
        columns = 0;
        block_count = 0;
    }

    [[nodiscard]] bool valid() const noexcept
    {
        return is_qnk_dtype(dtype) && rows != 0 && columns != 0 && block_count != 0
               && !storage.empty();
    }
};

[[nodiscard]] inline const uint8_t* qnk_packed_block(const QnKPack& weights,
                                                     size_t row,
                                                     uint32_t block) noexcept
{
    const size_t tile = row / 8;
    const size_t lane = row % 8;
    const size_t block_bytes = qnk_block_bytes(weights.dtype);
    const size_t offset = (((tile * weights.block_count + block) * 8 + lane) * block_bytes);
    return weights.storage.data() + offset;
}

void qnk_dequantize_block(DType dtype, const uint8_t* block, float* output) noexcept;

[[nodiscard]] float qnk_dot_block(DType dtype,
                                  const uint8_t* block,
                                  const float* input) noexcept;

// Quantize float32 activations to Q8_K blocks.
void qnk_q8k_quantize(const float* source,
                      uint8_t* output,
                      uint32_t columns) noexcept;

#if defined(__aarch64__) || defined(_M_ARM64)
// ARM64 counterpart of the x86 int8-activation path for 32-value blocks.
[[nodiscard]] bool neon_qnk_q8_supported() noexcept;
[[nodiscard]] bool neon_qnk_i8mm_supported() noexcept;

void neon_qnk_prefill_q8_0_gemm12(const uint8_t* weight_pack,
                                        size_t rows,
                                        uint32_t columns,
                                        const uint8_t* activation_pack,
                                        const float* activation_scales,
                                        size_t token_count,
                                        float* output,
                                        size_t output_stride) noexcept;

void neon_qnk_prefill_q8_0_gemm32(const uint8_t* weight_pack,
                                        size_t rows,
                                        uint32_t columns,
                                        const uint8_t* activation_pack,
                                        const float* activation_scales,
                                        size_t token_count,
                                        float* output,
                                        size_t output_stride) noexcept;

void neon_qnk_prefill_q8_0_fused16(const float* input,
                                   size_t tokens,
                                   uint32_t columns,
                                   const uint8_t* weight_pack,
                                   size_t rows,
                                   float* output,
                                   size_t output_stride) noexcept;

void neon_qnk_prefill_q8_0_gemm16(const uint8_t* weight_pack,
                                        size_t rows,
                                        uint32_t columns,
                                        const uint8_t* activation_pack,
                                        const float* activation_scales,
                                        size_t token_count,
                                        float* output,
                                        size_t output_stride) noexcept;

void neon_qnk_prefill_q8_0_gemm8(const uint8_t* weight_pack,
                                       size_t rows,
                                       uint32_t columns,
                                       const uint8_t* activation_pack,
                                       const float* activation_scales,
                                       size_t token_count,
                                       float* output,
                                       size_t output_stride) noexcept;

void neon_qnk_prefill_q8_0_gemm136(const uint8_t* weight_pack,
                                         size_t rows,
                                         uint32_t columns,
                                         const uint8_t* activation_pack,
                                         const float* activation_scales,
                                         size_t token_count,
                                         float* output,
                                         size_t output_stride) noexcept;

void neon_qnk_prefill_q8_0_gemm(const uint8_t* weight_pack,
                                      size_t rows,
                                      uint32_t columns,
                                      const uint8_t* activation_pack,
                                      const float* activation_scales,
                                      size_t token_count,
                                      float* output,
                                      size_t output_stride) noexcept;

void neon_qnk_q8_quant_pack4_group(const float* source,
                                    size_t group,
                                    uint32_t columns,
                                    int8_t* pack,
                                    float* scales) noexcept;

void neon_qnk_q8_quant_pack4(const float* source,
                              size_t tokens,
                              uint32_t columns,
                              int8_t* pack,
                              float* scales) noexcept;

void neon_qnk_q8_quant(const float* source,
                       int8_t* values,
                       float* scales,
                       int32_t* sums,
                       uint32_t columns) noexcept;

[[nodiscard]] float neon_qnk_q8_row(DType dtype,
                                    const uint8_t* weights,
                                    size_t block_stride,
                                    uint32_t block_count,
                                    const int8_t* input,
                                    const float* input_scales,
                                    const int32_t* input_sums) noexcept;

// Int8 prefill GEMM for Q8_0. Activations are token-major Q8_0 rows; the
// kernel processes four output rows and four tokens per block and reuses each
// weight block across all tokens.
void neon_qnk_prefill_q8_0_rows8(const uint8_t* __restrict weights,
                                 size_t rows,
                                 uint32_t columns,
                                 const int8_t* __restrict activation_codes,
                                 const float* __restrict activation_scales,
                                 size_t token_count,
                                 float* __restrict output,
                                 size_t output_stride) noexcept;

void neon_qnk_prefill_q8_0(const uint8_t* weights,
                           size_t rows,
                           uint32_t columns,
                           const int8_t* activation_codes,
                           const float* activation_scales,
                           size_t token_count,
                           float* output,
                           size_t output_stride) noexcept;

// Generic prefill GEMM for the remaining 32-value QnK formats.
void neon_qnk_prefill_q8_rows8(DType dtype,
                               const uint8_t* weights,
                               size_t rows,
                               uint32_t columns,
                               const int8_t* activation_codes,
                               const float* activation_scales,
                               const int32_t* activation_sums,
                               size_t token_count,
                               float* output,
                               size_t output_stride) noexcept;

void neon_qnk_prefill_q8(DType dtype,
                         const uint8_t* weights,
                         size_t rows,
                         uint32_t columns,
                         const int8_t* activation_codes,
                         const float* activation_scales,
                         const int32_t* activation_sums,
                         size_t token_count,
                         float* output,
                         size_t output_stride) noexcept;
#endif

[[nodiscard]] bool qnk_pack_weights(const uint8_t* raw,
                                    size_t raw_bytes,
                                    DType dtype,
                                    size_t rows,
                                    uint32_t columns,
                                    QnKPack& output) noexcept;

// use_q8_activations selects the int8 activation kernels for 32-value block
// weights. It trades activation precision for a quarter of the activation
// bytes and stays opt-in, because the exact-float parity contracts assume the
// float32 dot path.
[[nodiscard]] bool forward_linear_qnk(std::initializer_list<const TensorData*> matrices,
                                      const ActivationBuffer& input,
                                      std::initializer_list<ActivationBuffer*> outputs,
                                      bool use_packed_weights,
                                      std::span<std::shared_ptr<const QnKPack>* const> sidecars = {},
                                      bool use_q8_activations = false) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_QNK_H
