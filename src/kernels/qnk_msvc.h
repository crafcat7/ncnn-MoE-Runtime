#ifndef NCNN_MOE_QNK_MSVC_H
#define NCNN_MOE_QNK_MSVC_H

#include "qnk.h"

#include <cstddef>

namespace ncnn {
namespace moe {

#if defined(NCNN_MOE_MSVC_X86_SIMD)
float msvc_avx2_qnk_dot_block(DType dtype,
                              const uint8_t* block,
                              const float* input) noexcept;

// Decode one 32-value GGML block into 32 consecutive float32 weights.
void msvc_avx2_qnk_deq32(DType dtype,
                         const uint8_t* block,
                         float* output) noexcept;

// Dot product of one row of 32-value GGML blocks against float32 activations.
// block_stride covers both the contiguous row-major distance and the packed
// tile distance between consecutive blocks of one row.
float msvc_avx2_qnk_dot_row(DType dtype,
                            const uint8_t* weights,
                            size_t block_stride,
                            uint32_t block_count,
                            const float* input) noexcept;

float msvc_avx512_qnk_dot_block(DType dtype,
                                const uint8_t* block,
                                const float* input) noexcept;
void msvc_avx2_qnk_gemm(const QnKPack& weights,
                        const float* input,
                        size_t input_stride,
                        size_t token_count,
                        float* output,
                        size_t output_stride) noexcept;

void msvc_avx512_qnk_gemm(const QnKPack& weights,
                          const float* input,
                          size_t input_stride,
                          size_t token_count,
                          float* output,
                          size_t output_stride) noexcept;

void msvc_avx2_qnk_q8k_quantize(const float* source,
                                uint8_t* output,
                                uint32_t columns) noexcept;

// Quantize one float32 activation row into 32-value int8 blocks. Each block
// stores its scale and its integer code sum, which the affine formats use to
// fold the per-block minimum into a scalar correction.
void msvc_avx2_qnk_q8_quant(const float* source,
                            int8_t* values,
                            float* scales,
                            int32_t* sums,
                            uint32_t columns) noexcept;

// Dot product of one row of 32-value GGML blocks against int8 activations.
// Codes stay packed and pair with two activations per instruction, so a row
// reads a quarter of the activation bytes the float32 row kernel needs.
float msvc_avx2_qnk_q8_row(DType dtype,
                           const uint8_t* weights,
                           size_t block_stride,
                           uint32_t block_count,
                           const int8_t* input,
                           const float* input_scales,
                           const int32_t* input_sums) noexcept;
void msvc_avx512_qnk_q8k_quantize(const float* source,
                                  uint8_t* output,
                                  uint32_t columns) noexcept;
#endif

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_QNK_MSVC_H
