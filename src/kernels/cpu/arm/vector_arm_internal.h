#ifndef NCNN_MOE_KERNELS_CPU_ARM_VECTOR_ARM_INTERNAL_H
#define NCNN_MOE_KERNELS_CPU_ARM_VECTOR_ARM_INTERNAL_H

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

#if defined(__aarch64__) || defined(_M_ARM64)

float neon_float_dot(const float* left, const float* right, uint32_t count) noexcept;
void neon_float_gemm(const float* weights,
                     size_t weight_stride,
                     const float* input,
                     size_t input_stride,
                     uint32_t input_columns,
                     uint32_t output_count,
                     uint32_t token_count,
                     float* output,
                     size_t output_stride) noexcept;
void neon_bfloat16_gemm(const uint16_t* weights,
                        size_t weight_stride,
                        const float* input,
                        size_t input_stride,
                        uint32_t input_columns,
                        uint32_t output_count,
                        uint32_t token_count,
                        float* output,
                        size_t output_stride) noexcept;
void neon_float_rms_norm(float* output,
                         const float* input,
                         const float* weight,
                         float epsilon,
                         float weight_offset,
                         uint32_t count) noexcept;
void neon_bfloat16_rms_norm(float* output,
                            const float* input,
                            const uint16_t* weight,
                            float epsilon,
                            float weight_offset,
                            uint32_t count) noexcept;
void neon_float_rope_inplace(float* values,
                             const float* cosine,
                             const float* sine,
                             uint32_t dimension) noexcept;
float neon_f32_scale_max(float* values, float scale, uint32_t count) noexcept;
float neon_f32_softmax(float* values, uint32_t count, float maximum) noexcept;
void neon_float_exp_inplace(float* values, uint32_t count) noexcept;
void neon_float_sigmoid_mul(float* output,
                            const float* gate,
                            const float* input,
                            uint32_t count) noexcept;
void neon_float_silu_mul(float* output,
                         const float* gate,
                         const float* up,
                         float sigmoid_scale,
                         float up_offset,
                         uint32_t count) noexcept;
void neon_float_silu_inplace(float* values, uint32_t count) noexcept;

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
void neon_f32_to_f16(uint16_t* output, const float* input, size_t count) noexcept;
void neon_bf16_to_f16(uint16_t* output, const uint16_t* input, size_t count) noexcept;
void neon_f16_gemm_4x8(const uint16_t* weights,
                       size_t weight_stride,
                       const uint16_t* input,
                       size_t input_stride,
                       uint32_t input_columns,
                       uint32_t output_count,
                       uint32_t token_count,
                       float* output,
                       size_t output_stride) noexcept;
void neon_f16_pv(float* output,
                 const uint16_t* values,
                 size_t value_stride,
                 const uint16_t* weights,
                 uint32_t count,
                 uint32_t columns) noexcept;
void neon_f16_pv4(float* output,
                  const uint16_t* values,
                  size_t value_stride,
                  const float* weights0,
                  const float* weights1,
                  const float* weights2,
                  const float* weights3,
                  uint32_t count,
                  uint32_t columns) noexcept;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#endif // ARM64

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_KERNELS_CPU_ARM_VECTOR_ARM_INTERNAL_H
