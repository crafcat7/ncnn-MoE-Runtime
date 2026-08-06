#ifndef NCNN_MOE_CPU_VECTOR_MSVC_H
#define NCNN_MOE_CPU_VECTOR_MSVC_H

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

float msvc_avx2_float_dot(const float* left, const float* right, uint32_t count) noexcept;
float msvc_avx512_float_dot(const float* left, const float* right, uint32_t count) noexcept;
void msvc_avx2_float_gemm_4x4(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept;
void msvc_avx512_float_gemm_4x4(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept;
void msvc_avx2_float_gemm_4x8(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept;
void msvc_avx512_float_gemm_4x8(
    const float* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept;
void msvc_avx2_bfloat16_gemm_4x8(
    const uint16_t* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept;
void msvc_avx512_bfloat16_gemm_4x8(
    const uint16_t* weights,
    size_t weight_stride,
    const float* input,
    size_t input_stride,
    uint32_t input_columns,
    uint32_t output_count,
    uint32_t token_count,
    float* output,
    size_t output_stride) noexcept;
void msvc_avx2_float_exp_inplace(float* values, uint32_t count) noexcept;
void msvc_avx512_float_exp_inplace(float* values, uint32_t count) noexcept;
float msvc_avx2_int8_float_dot(const int8_t* left, const float* right, uint32_t count) noexcept;
float msvc_avx512_int8_float_dot(const int8_t* left, const float* right, uint32_t count) noexcept;
void msvc_avx2_float_scale_inplace(float* values, float scale, uint32_t count) noexcept;
void msvc_avx512_float_scale_inplace(float* values, float scale, uint32_t count) noexcept;
void msvc_avx2_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept;
void msvc_avx512_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept;
void msvc_avx2_float_scale_inplace_and_scaled_add(
    float* values,
    float value_scale,
    float* output,
    float output_scale,
    uint32_t count) noexcept;
void msvc_avx512_float_scale_inplace_and_scaled_add(
    float* values,
    float value_scale,
    float* output,
    float output_scale,
    uint32_t count) noexcept;
void msvc_avx2_float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept;
void msvc_avx512_float_weighted_scale(float* output, const float* input, const float* weight, float scale, float weight_offset, uint32_t count) noexcept;
void msvc_avx2_bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept;
void msvc_avx512_bfloat16_weighted_scale(float* output, const float* input, const uint16_t* weight, float scale, float weight_offset, uint32_t count) noexcept;
void msvc_avx2_float_silu_mul(float* output, const float* gate, const float* up,
                              float sigmoid_scale, float up_offset, uint32_t count) noexcept;
void msvc_avx512_float_silu_mul(float* output, const float* gate, const float* up,
                                float sigmoid_scale, float up_offset, uint32_t count) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_VECTOR_MSVC_H
