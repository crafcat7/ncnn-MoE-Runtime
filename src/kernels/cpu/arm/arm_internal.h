#ifndef NCNN_MOE_KERNELS_CPU_ARM_INTERNAL_H
#define NCNN_MOE_KERNELS_CPU_ARM_INTERNAL_H

#include "kernels/cpu/cpu_ops.h"

namespace ncnn {
namespace moe {

#if defined(__aarch64__) || defined(_M_ARM64)

// BFloat16
void neon_float_to_bfloat16_array(uint16_t* output, const float* input, uint32_t count) noexcept;
float neon_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept;
void neon_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept;

// MXFP4
float neon_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept;
void neon_mxfp4_gemm_row(const uint8_t* packed,
                         const uint8_t* scales,
                         uint32_t block_count,
                         const float* input,
                         size_t input_stride,
                         size_t token_count,
                         float* output,
                         size_t output_stride) noexcept;
void neon_mxfp4_matmul_rows2(const uint8_t* first_packed,
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
                             size_t second_output_stride) noexcept;
void neon_mxfp4_matmul_row_pairs(const uint8_t* packed,
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
                                 size_t second_token_stride) noexcept;

#if defined(NCNN_MOE_ARM_SVE2_KERNEL)
float sve2_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept;
void sve2_mxfp4_matmul_rows2(const uint8_t* first_packed,
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
                             size_t second_output_stride) noexcept;
void sve2_mxfp4_matmul_row_pairs(const uint8_t* packed,
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
                                 size_t second_token_stride) noexcept;
#endif

// QnK
void neon_qnk_q8k_quantize(const float* source, uint8_t* output, uint32_t columns) noexcept;

[[nodiscard]] bool neon_qnk_q8_supported() noexcept;
[[nodiscard]] bool neon_qnk_i8mm_supported() noexcept;

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
float neon_qnk_q8_row(DType dtype,
                      const uint8_t* weights,
                      size_t block_stride,
                      uint32_t block_count,
                      const int8_t* input,
                      const float* input_scales,
                      const int32_t* input_sums) noexcept;
void neon_qnk_prefill_q8_0(const uint8_t* weights,
                           size_t rows,
                           uint32_t columns,
                           const int8_t* codes,
                           const float* scales,
                           size_t token_count,
                           float* output,
                           size_t output_stride) noexcept;
void neon_qnk_prefill_q8_0_rows8(const uint8_t* __restrict weights,
                                 size_t rows,
                                 uint32_t columns,
                                 const int8_t* __restrict codes,
                                 const float* __restrict scales,
                                 size_t token_count,
                                 float* __restrict output,
                                 size_t output_stride) noexcept;
void neon_qnk_prefill_q8(DType dtype,
                         const uint8_t* weights,
                         size_t rows,
                         uint32_t columns,
                         const int8_t* codes,
                         const float* scales,
                         const int32_t* sums,
                         size_t token_count,
                         float* output,
                         size_t output_stride) noexcept;
void neon_qnk_prefill_q8_rows8(DType dtype,
                               const uint8_t* weights,
                               size_t rows,
                               uint32_t columns,
                               const int8_t* codes,
                               const float* scales,
                               const int32_t* sums,
                               size_t token_count,
                               float* output,
                               size_t output_stride) noexcept;

void neon_qnk_prefill_q8_0_gemm4(const uint8_t* weight_pack,
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
void neon_qnk_prefill_q8_0_gemm16(const uint8_t* weight_pack,
                                  size_t rows,
                                  uint32_t columns,
                                  const uint8_t* activation_pack,
                                  const float* activation_scales,
                                  size_t token_count,
                                  float* output,
                                  size_t output_stride) noexcept;

#endif

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_KERNELS_CPU_ARM_INTERNAL_H
