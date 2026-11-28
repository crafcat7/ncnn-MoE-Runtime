#ifndef NCNN_MOE_KERNELS_CPU_X86_INTERNAL_H
#define NCNN_MOE_KERNELS_CPU_X86_INTERNAL_H

#include "kernels/cpu/cpu_ops.h"

#include <vector>

namespace ncnn {
namespace moe {

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)

// BFloat16
float avx2_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept;
void avx2_float_to_bfloat16_array(uint16_t* output, const float* input, uint32_t count) noexcept;
void avx2_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept;

float avx512_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept;
void avx512_float_to_bfloat16_array(uint16_t* output, const float* input, uint32_t count) noexcept;
float avx512_bfloat16_pair_dot(const uint16_t* left, const uint16_t* right, uint32_t count) noexcept;
void avx512_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept;
void avx512_pv_bf16(float* output,
                    const uint16_t* values0,
                    size_t value_stride,
                    const float* weights0,
                    uint32_t count0,
                    const uint16_t* values1,
                    const float* weights1,
                    uint32_t count1,
                    uint32_t columns) noexcept;
void avx512_bfloat16_batched_linear(const uint16_t* weights,
                                    const float* input,
                                    size_t input_stride,
                                    size_t token_count,
                                    uint32_t output_columns,
                                    uint32_t input_columns,
                                    float* output,
                                    size_t output_stride,
                                    int thread_count,
                                    std::vector<uint16_t>& packed_input);
void avx512_bfloat16_single_token_linear(const uint16_t* weights,
                                         const float* input,
                                         uint32_t output_columns,
                                         uint32_t input_columns,
                                         float* output,
                                         int thread_count) noexcept;

// Float8
float avx2_float8_e4m3_block_dot(const uint8_t* weights,
                                 const float* scales,
                                 const float* input,
                                 uint32_t count,
                                 uint32_t block_size) noexcept;
void avx2_float8_e4m3_block_dot_rows4(const uint8_t* weights,
                                      uint32_t weight_row_stride,
                                      const float* scales,
                                      const float* input,
                                      uint32_t count,
                                      uint32_t block_size,
                                      uint32_t row_count,
                                      float* output) noexcept;

float avx512_float8_e4m3_block_dot(const uint8_t* weights,
                                   const float* scales,
                                   const float* input,
                                   uint32_t count,
                                   uint32_t block_size) noexcept;
void avx512_float8_e4m3_block_dot_rows4(const uint8_t* weights,
                                        uint32_t weight_row_stride,
                                        const float* scales,
                                        const float* input,
                                        uint32_t count,
                                        uint32_t block_size,
                                        uint32_t row_count,
                                        float* output) noexcept;
float avx512_bfloat16_float8_e4m3_block_dot(const uint8_t* weights,
                                            const float* scales,
                                            const float* input,
                                            uint32_t count,
                                            uint32_t block_size) noexcept;
void avx512_bfloat16_float8_e4m3_block_dot_rows(const uint8_t* weights,
                                                uint32_t weight_row_stride,
                                                const float* scales,
                                                const float* input,
                                                uint32_t count,
                                                uint32_t block_size,
                                                uint32_t row_count,
                                                float* output) noexcept;
void avx512_bfloat16_float8_e4m3_block_dot_rows_batch(const uint8_t* weights,
                                                      uint32_t weight_row_stride,
                                                      const float* scales,
                                                      const float* input,
                                                      size_t input_stride,
                                                      uint32_t count,
                                                      uint32_t block_size,
                                                      uint32_t row_count,
                                                      size_t output_stride,
                                                      size_t token_count,
                                                      float* output) noexcept;
void avx512_quantize_float8_e4m3(const float* source,
                                 float* values,
                                 uint32_t count,
                                 uint32_t block_size,
                                 bool power_of_two_scale) noexcept;

// MXFP4
void avx2_unpack_gguf_mxfp4_rows(const uint8_t* source,
                                 uint8_t* blocks,
                                 uint8_t* scales,
                                 size_t rows,
                                 size_t blocks_per_row,
                                 size_t block_row_stride,
                                 size_t scale_row_stride) noexcept;
float avx2_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept;
float avx2_mxfp4_q8_dot(const uint8_t* packed,
                        const uint8_t* scales,
                        uint32_t block_count,
                        const int8_t* input,
                        const float* input_scales) noexcept;
void avx2_mxfp4_q8_quantize(const float* source, int8_t* values, float* scales, uint32_t columns) noexcept;
void avx2_mxfp4_q8_gemm_row(const uint8_t* packed,
                            const uint8_t* scales,
                            uint32_t block_count,
                            const int8_t* input,
                            size_t input_stride,
                            const float* input_scales,
                            size_t scale_stride,
                            size_t token_count,
                            float* output,
                            size_t output_stride) noexcept;
void avx2_mxfp4_q8_matmul_rows2(const uint8_t* first_packed,
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
                                size_t second_output_stride) noexcept;
void avx2_mxfp4_q8_matmul_row_pairs(const uint8_t* packed,
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
                                    size_t second_token_stride) noexcept;
void avx2_mxfp4_q8_packed_gemm(const uint8_t* packed,
                               uint32_t row_count,
                               uint32_t block_count,
                               uint32_t tile_rows,
                               const int8_t* input,
                               size_t input_stride,
                               const float* input_scales,
                               size_t scale_stride,
                               size_t token_count,
                               float* output,
                               size_t output_stride) noexcept;
void avx2_mxfp4_gemm_row(const uint8_t* packed,
                         const uint8_t* scales,
                         uint32_t block_count,
                         const float* input,
                         size_t input_stride,
                         size_t token_count,
                         float* output,
                         size_t output_stride) noexcept;
void avx2_mxfp4_matmul_rows2(const uint8_t* first_packed,
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
void avx2_mxfp4_matmul_row_pairs(const uint8_t* packed,
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

void avx512_mxfp4_q8_quantize(const float* source, int8_t* values, float* scales, uint32_t columns) noexcept;
float avx512_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept;
void avx512_mxfp4_q8_packed_gemm(const uint8_t* packed,
                                 uint32_t row_count,
                                 uint32_t block_count,
                                 uint32_t tile_rows,
                                 const int8_t* input,
                                 size_t input_stride,
                                 const float* input_scales,
                                 size_t scale_stride,
                                 size_t token_count,
                                 float* output,
                                 size_t output_stride) noexcept;
void avx512_mxfp4_gemm_row(const uint8_t* packed,
                           const uint8_t* scales,
                           uint32_t block_count,
                           const float* input,
                           size_t input_stride,
                           size_t token_count,
                           float* output,
                           size_t output_stride) noexcept;
void avx512_mxfp4_matmul_rows2(const uint8_t* first_packed,
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
void avx512_mxfp4_matmul_row_pairs(const uint8_t* packed,
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

// QnK
void avx2_qnk_deq32(DType dtype, const uint8_t* block, float* output) noexcept;
float avx2_qnk_dot_block(DType dtype, const uint8_t* block, const float* input) noexcept;
float avx2_qnk_dot_row(DType dtype,
                       const uint8_t* weights,
                       size_t block_stride,
                       uint32_t block_count,
                       const float* input) noexcept;
void avx2_qnk_gemm(const QnKPack& weights,
                   const float* input,
                   size_t input_stride,
                   size_t token_count,
                   float* output,
                   size_t output_stride) noexcept;
void avx2_qnk_q8k_quantize(const float* source, uint8_t* output, uint32_t columns) noexcept;
void avx2_qnk_q8_quant(const float* source,
                       int8_t* values,
                       float* scales,
                       int32_t* sums,
                       uint32_t columns) noexcept;
float avx2_qnk_q8_row(DType dtype,
                      const uint8_t* weights,
                      size_t block_stride,
                      uint32_t block_count,
                      const int8_t* input,
                      const float* input_scales,
                      const int32_t* input_sums) noexcept;

float avx512_qnk_dot_block(DType dtype, const uint8_t* block, const float* input) noexcept;
void avx512_qnk_gemm(const QnKPack& weights,
                     const float* input,
                     size_t input_stride,
                     size_t token_count,
                     float* output,
                     size_t output_stride) noexcept;
void avx512_qnk_q8k_quantize(const float* source, uint8_t* output, uint32_t columns) noexcept;

#endif

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_KERNELS_CPU_X86_INTERNAL_H
