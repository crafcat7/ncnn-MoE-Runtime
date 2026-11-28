#ifndef NCNN_MOE_KERNELS_CPU_OPS_H
#define NCNN_MOE_KERNELS_CPU_OPS_H

#include "kernels/cpu/cpu_caps.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

struct QnKPack;

struct CpuKernelOps
{
    float (*float_dot)(const float* left, const float* right, uint32_t count) noexcept = nullptr;
    size_t (*float_argmax)(const float* values, size_t count) noexcept = nullptr;

    void (*float_gemm_4x4)(const float* weights,
                           size_t weight_stride,
                           const float* input,
                           size_t input_stride,
                           uint32_t input_columns,
                           uint32_t output_count,
                           uint32_t token_count,
                           float* output,
                           size_t output_stride) noexcept = nullptr;
    void (*float_gemm_4x8)(const float* weights,
                           size_t weight_stride,
                           const float* input,
                           size_t input_stride,
                           uint32_t input_columns,
                           uint32_t output_count,
                           uint32_t token_count,
                           float* output,
                           size_t output_stride) noexcept = nullptr;
    void (*bfloat16_gemm_4x8)(const uint16_t* weights,
                              size_t weight_stride,
                              const float* input,
                              size_t input_stride,
                              uint32_t input_columns,
                              uint32_t output_count,
                              uint32_t token_count,
                              float* output,
                              size_t output_stride) noexcept = nullptr;

    void (*f32_to_f16)(uint16_t* output, const float* input, size_t count) noexcept = nullptr;
    void (*bf16_to_f16)(uint16_t* output, const uint16_t* input, size_t count) noexcept = nullptr;
    void (*f16_gemm_4x8)(const uint16_t* weights,
                         size_t weight_stride,
                         const uint16_t* input,
                         size_t input_stride,
                         uint32_t input_columns,
                         uint32_t output_count,
                         uint32_t token_count,
                         float* output,
                         size_t output_stride) noexcept = nullptr;
    void (*f16_pv)(float* output,
                   const uint16_t* values,
                   size_t value_stride,
                   const uint16_t* weights,
                   uint32_t count,
                   uint32_t columns) noexcept = nullptr;
    void (*f16_pv4)(float* output,
                    const uint16_t* values,
                    size_t value_stride,
                    const float* weights0,
                    const float* weights1,
                    const float* weights2,
                    const float* weights3,
                    uint32_t count,
                    uint32_t columns) noexcept = nullptr;

    float (*f32_scale_max)(float* values, float scale, uint32_t count) noexcept = nullptr;
    float (*f32_softmax)(float* values, uint32_t count, float maximum) noexcept = nullptr;
    void (*float_exp_inplace)(float* values, uint32_t count) noexcept = nullptr;
    void (*float_silu_inplace)(float* values, uint32_t count) noexcept = nullptr;

    float (*int8_float_dot)(const int8_t* left, const float* right, uint32_t count) noexcept = nullptr;
    void (*float_l2_scale_inplace)(float* values, float epsilon, uint32_t count) noexcept = nullptr;
    void (*float_rms_scale_inplace)(float* values, float epsilon, uint32_t count) noexcept = nullptr;
    void (*float_rms_norm)(float* output,
                           const float* input,
                           const float* weight,
                           float epsilon,
                           float weight_offset,
                           uint32_t count) noexcept = nullptr;
    void (*bfloat16_rms_norm)(float* output,
                              const float* input,
                              const uint16_t* weight,
                              float epsilon,
                              float weight_offset,
                              uint32_t count) noexcept = nullptr;

    void (*float_rope_inplace)(float* values,
                               const float* cosine,
                               const float* sine,
                               uint32_t dimension) noexcept = nullptr;
    void (*float_scale_inplace)(float* values, float scale, uint32_t count) noexcept = nullptr;
    void (*float_scaled_add)(float* output, const float* input, float scale, uint32_t count) noexcept = nullptr;
    void (*float_scale_add)(float* output,
                            float output_scale,
                            const float* input,
                            float input_scale,
                            uint32_t count) noexcept = nullptr;
    void (*scale_add)(float* values,
                      float value_scale,
                      float* output,
                      float output_scale,
                      uint32_t count) noexcept = nullptr;
    void (*scaled_acc)(float* values,
                       const float* input,
                       float input_scale,
                       float* output,
                       float output_scale,
                       uint32_t count) noexcept = nullptr;

    void (*gdn_step)(float* state,
                     const float* key,
                     const float* query,
                     const float* value,
                     uint32_t key_dim,
                     uint32_t value_dim,
                     float decay,
                     float beta,
                     float scale,
                     float* memory,
                     float* delta,
                     float* output) noexcept = nullptr;
    void (*gdn_conv4)(float* state,
                      float* values,
                      size_t value_stride,
                      size_t first,
                      size_t row_count,
                      uint32_t begin,
                      uint32_t end,
                      uint32_t columns,
                      const float* taps) noexcept = nullptr;
    void (*pv_f32)(float* output,
                   const float* values0,
                   size_t value_stride,
                   const float* weights0,
                   uint32_t count0,
                   const float* values1,
                   const float* weights1,
                   uint32_t count1,
                   uint32_t columns) noexcept = nullptr;

    void (*float_weighted_scale)(float* output,
                                 const float* input,
                                 const float* weight,
                                 float scale,
                                 float weight_offset,
                                 uint32_t count) noexcept = nullptr;
    void (*bfloat16_weighted_scale)(float* output,
                                    const float* input,
                                    const uint16_t* weight,
                                    float scale,
                                    float weight_offset,
                                    uint32_t count) noexcept = nullptr;
    void (*float_sigmoid_mul)(float* output,
                              const float* gate,
                              const float* input,
                              uint32_t count) noexcept = nullptr;
    void (*float_silu_mul)(float* output,
                           const float* gate,
                           const float* up,
                           float sigmoid_scale,
                           float up_offset,
                           uint32_t count) noexcept = nullptr;
    void (*float_hc_pre_4)(float* output,
                           const float* input,
                           float scale0,
                           float scale1,
                           float scale2,
                           float scale3,
                           uint32_t hidden_size) noexcept = nullptr;
    void (*float_hc_post_4)(float* output,
                            const float* branch,
                            const float* residual,
                            const float* post,
                            const float* combine,
                            uint32_t hidden_size) noexcept = nullptr;

    // BFloat16 kernels
    float (*bfloat16_dot)(const uint16_t* weights, const float* input, uint32_t count) noexcept = nullptr;
    void (*float_to_bfloat16_array)(uint16_t* output, const float* input, uint32_t count) noexcept = nullptr;
    float (*bfloat16_pair_dot)(const uint16_t* left, const uint16_t* right, uint32_t count) noexcept = nullptr;
    void (*bfloat16_scaled_add)(float* output, const uint16_t* input, float scale, uint32_t count) noexcept = nullptr;
    void (*pv_bf16)(float* output,
                    const uint16_t* values0,
                    size_t value_stride,
                    const float* weights0,
                    uint32_t count0,
                    const uint16_t* values1,
                    const float* weights1,
                    uint32_t count1,
                    uint32_t columns) noexcept = nullptr;
    void (*bfloat16_single_token_linear)(const uint16_t* weights,
                                         const float* input,
                                         uint32_t output_columns,
                                         uint32_t input_columns,
                                         float* output,
                                         int thread_count) noexcept = nullptr;
    void (*bfloat16_batched_linear)(const uint16_t* weights,
                                    const float* input,
                                    size_t input_stride,
                                    size_t token_count,
                                    uint32_t output_columns,
                                    uint32_t input_columns,
                                    float* output,
                                    size_t output_stride,
                                    int thread_count,
                                    std::vector<uint16_t>& packed_input) = nullptr;

    // Float8 kernels
    float (*float8_e4m3_block_dot)(const uint8_t* weights,
                                   const float* scales,
                                   const float* input,
                                   uint32_t count,
                                   uint32_t block_size) noexcept = nullptr;
    void (*float8_e4m3_block_dot_rows4)(const uint8_t* weights,
                                        uint32_t weight_row_stride,
                                        const float* scales,
                                        const float* input,
                                        uint32_t count,
                                        uint32_t block_size,
                                        uint32_t row_count,
                                        float* output) noexcept = nullptr;
    float (*bfloat16_float8_e4m3_block_dot)(const uint8_t* weights,
                                            const float* scales,
                                            const float* input,
                                            uint32_t count,
                                            uint32_t block_size) noexcept = nullptr;
    void (*bfloat16_float8_e4m3_block_dot_rows)(const uint8_t* weights,
                                                uint32_t weight_row_stride,
                                                const float* scales,
                                                const float* input,
                                                uint32_t count,
                                                uint32_t block_size,
                                                uint32_t row_count,
                                                float* output) noexcept = nullptr;
    void (*bfloat16_float8_e4m3_block_dot_rows_batch)(const uint8_t* weights,
                                                      uint32_t weight_row_stride,
                                                      const float* scales,
                                                      const float* input,
                                                      size_t input_stride,
                                                      uint32_t count,
                                                      uint32_t block_size,
                                                      uint32_t row_count,
                                                      size_t output_stride,
                                                      size_t token_count,
                                                      float* output) noexcept = nullptr;
    void (*quantize_float8_e4m3)(const float* source,
                                 float* values,
                                 uint32_t count,
                                 uint32_t block_size,
                                 bool power_of_two_scale) noexcept = nullptr;

    // MXFP4 kernels
    void (*unpack_gguf_mxfp4_rows)(const uint8_t* source,
                                   uint8_t* blocks,
                                   uint8_t* scales,
                                   size_t rows,
                                   size_t blocks_per_row,
                                   size_t block_row_stride,
                                   size_t scale_row_stride) noexcept = nullptr;
    float (*mxfp4_dot)(const uint8_t* packed,
                       const uint8_t* scales,
                       uint32_t block_count,
                       const float* input) noexcept = nullptr;
    void (*mxfp4_gemm_row)(const uint8_t* packed,
                           const uint8_t* scales,
                           uint32_t block_count,
                           const float* input,
                           size_t input_stride,
                           size_t token_count,
                           float* output,
                           size_t output_stride) noexcept = nullptr;
    void (*mxfp4_matmul_rows2)(const uint8_t* first_packed,
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
                               size_t second_output_stride) noexcept = nullptr;
    void (*mxfp4_matmul_row_pairs)(const uint8_t* packed,
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
                                   size_t second_token_stride) noexcept = nullptr;
    float (*mxfp4_q8_dot)(const uint8_t* packed,
                          const uint8_t* scales,
                          uint32_t block_count,
                          const int8_t* input,
                          const float* input_scales) noexcept = nullptr;
    void (*mxfp4_q8_quantize)(const float* source,
                              int8_t* values,
                              float* scales,
                              uint32_t columns) noexcept = nullptr;
    void (*mxfp4_q8_gemm_row)(const uint8_t* packed,
                              const uint8_t* scales,
                              uint32_t block_count,
                              const int8_t* input,
                              size_t input_stride,
                              const float* input_scales,
                              size_t scale_stride,
                              size_t token_count,
                              float* output,
                              size_t output_stride) noexcept = nullptr;
    void (*mxfp4_q8_matmul_rows2)(const uint8_t* first_packed,
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
                                  size_t second_output_stride) noexcept = nullptr;
    void (*mxfp4_q8_matmul_row_pairs)(const uint8_t* packed,
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
                                      size_t second_token_stride) noexcept = nullptr;
    void (*mxfp4_q8_packed_gemm)(const uint8_t* packed,
                                 uint32_t row_count,
                                 uint32_t block_count,
                                 uint32_t tile_rows,
                                 const int8_t* input,
                                 size_t input_stride,
                                 const float* input_scales,
                                 size_t scale_stride,
                                 size_t token_count,
                                 float* output,
                                 size_t output_stride) noexcept = nullptr;

    // QnK kernels
    void (*qnk_dequantize_block_32)(DType dtype, const uint8_t* block, float* output) noexcept = nullptr;
    float (*qnk_dot_block)(DType dtype, const uint8_t* block, const float* input) noexcept = nullptr;
    float (*qnk_dot_row)(DType dtype,
                         const uint8_t* weights,
                         size_t block_stride,
                         uint32_t block_count,
                         const float* input) noexcept = nullptr;
    void (*qnk_gemm)(const QnKPack& weights,
                     const float* input,
                     size_t input_stride,
                     size_t token_count,
                     float* output,
                     size_t output_stride) noexcept = nullptr;
    void (*qnk_q8k_quantize)(const float* source, uint8_t* output, uint32_t columns) noexcept = nullptr;
    void (*qnk_q8_quant)(const float* source,
                         int8_t* values,
                         float* scales,
                         int32_t* sums,
                         uint32_t columns) noexcept = nullptr;
    float (*qnk_q8_row)(DType dtype,
                        const uint8_t* weights,
                        size_t block_stride,
                        uint32_t block_count,
                        const int8_t* input,
                        const float* input_scales,
                        const int32_t* input_sums) noexcept = nullptr;
    void (*qnk_q8_quant_pack4)(const float* source,
                               size_t tokens,
                               uint32_t columns,
                               int8_t* pack,
                               float* scales) noexcept = nullptr;
    void (*qnk_prefill_q8_0)(const uint8_t* weights,
                             size_t rows,
                             uint32_t columns,
                             const int8_t* codes,
                             const float* scales,
                             size_t token_count,
                             float* output,
                             size_t output_stride) noexcept = nullptr;
    void (*qnk_prefill_q8_0_rows8)(const uint8_t* weights,
                                   size_t rows,
                                   uint32_t columns,
                                   const int8_t* codes,
                                   const float* scales,
                                   size_t token_count,
                                   float* output,
                                   size_t output_stride) noexcept = nullptr;
    void (*qnk_prefill_q8)(DType dtype,
                           const uint8_t* weights,
                           size_t rows,
                           uint32_t columns,
                           const int8_t* codes,
                           const float* scales,
                           const int32_t* sums,
                           size_t token_count,
                           float* output,
                           size_t output_stride) noexcept = nullptr;
    void (*qnk_prefill_q8_rows8)(DType dtype,
                                 const uint8_t* weights,
                                 size_t rows,
                                 uint32_t columns,
                                 const int8_t* codes,
                                 const float* scales,
                                 const int32_t* sums,
                                 size_t token_count,
                                 float* output,
                                 size_t output_stride) noexcept = nullptr;
    void (*qnk_prefill_q8_0_gemm4)(const uint8_t* weight_pack,
                                   size_t rows,
                                   uint32_t columns,
                                   const uint8_t* activation_pack,
                                   const float* activation_scales,
                                   size_t token_count,
                                   float* output,
                                   size_t output_stride) noexcept = nullptr;
    void (*qnk_prefill_q8_0_gemm8)(const uint8_t* weight_pack,
                                   size_t rows,
                                   uint32_t columns,
                                   const uint8_t* activation_pack,
                                   const float* activation_scales,
                                   size_t token_count,
                                   float* output,
                                   size_t output_stride) noexcept = nullptr;
    void (*qnk_prefill_q8_0_gemm16)(const uint8_t* weight_pack,
                                    size_t rows,
                                    uint32_t columns,
                                    const uint8_t* activation_pack,
                                    const float* activation_scales,
                                    size_t token_count,
                                    float* output,
                                    size_t output_stride) noexcept = nullptr;
};

[[nodiscard]] const CpuKernelOps& cpu_ops() noexcept;

void register_scalar_cpu_ops(CpuKernelOps& ops, CpuBackendCaps& caps) noexcept;
void register_x86_cpu_ops(CpuKernelOps& ops, CpuBackendCaps& caps, uint64_t enabled_isa) noexcept;
void register_arm_cpu_ops(CpuKernelOps& ops, CpuBackendCaps& caps, uint64_t enabled_isa) noexcept;
void register_ncnn_cpu_ops(CpuKernelOps& ops, CpuBackendCaps& caps, uint64_t enabled_isa) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_KERNELS_CPU_OPS_H
