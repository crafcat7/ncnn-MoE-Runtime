#ifndef NCNN_MOE_CPU_VECTOR_H
#define NCNN_MOE_CPU_VECTOR_H

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

float float_dot(const float* left, const float* right, uint32_t count) noexcept;
void float_gemm_4x4(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept;
// Computes up to four input rows by up to eight output rows.  The caller
// keeps the tile within those bounds and uses this for small-batch GEMM
// dispatch so the input tile and each weight vector are reused across rows.
void float_gemm_4x8(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept;
void bfloat16_gemm_4x8(
    const uint16_t* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept;
void float_exp_inplace(float* values, uint32_t count) noexcept;
[[nodiscard]] bool float_exp_simd_available() noexcept;
float int8_float_dot(const int8_t* left, const float* right, uint32_t count) noexcept;
void float_scale_inplace(float* values, float scale, uint32_t count) noexcept;
void float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept;
void float_scale_inplace_and_scaled_add(
    float* values,
    float value_scale,
    float* output,
    float output_scale,
    uint32_t count) noexcept;
void float_weighted_scale(float* output, const float* input, const float* weight,
                          float scale, float weight_offset, uint32_t count) noexcept;
void bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight,
                             float scale, float weight_offset, uint32_t count) noexcept;
void float_silu_mul(float* output, const float* gate, const float* up,
                    float sigmoid_scale, float up_offset, uint32_t count) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_VECTOR_H
