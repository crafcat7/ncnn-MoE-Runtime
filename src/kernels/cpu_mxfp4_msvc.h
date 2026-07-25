#ifndef NCNN_MOE_CPU_MXFP4_MSVC_H
#define NCNN_MOE_CPU_MXFP4_MSVC_H

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

[[nodiscard]] float msvc_avx2_mxfp4_dot(
    const uint8_t* packed,
    const uint8_t* scales,
    uint32_t block_count,
    const float* input) noexcept;

void msvc_avx2_mxfp4_gemm_row(
    const uint8_t* packed,
    const uint8_t* scales,
    uint32_t block_count,
    const float* input,
    size_t input_stride,
    size_t token_count,
    float* output,
    size_t output_stride) noexcept;

void msvc_avx2_mxfp4_matmul_rows2(
    const uint8_t* first_packed,
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

[[nodiscard]] float msvc_avx512_mxfp4_dot(
    const uint8_t* packed,
    const uint8_t* scales,
    uint32_t block_count,
    const float* input) noexcept;

void msvc_avx512_mxfp4_gemm_row(
    const uint8_t* packed,
    const uint8_t* scales,
    uint32_t block_count,
    const float* input,
    size_t input_stride,
    size_t token_count,
    float* output,
    size_t output_stride) noexcept;

void msvc_avx512_mxfp4_matmul_rows2(
    const uint8_t* first_packed,
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

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_MXFP4_MSVC_H
