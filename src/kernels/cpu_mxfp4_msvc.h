#ifndef NCNN_MOE_CPU_MXFP4_MSVC_H
#define NCNN_MOE_CPU_MXFP4_MSVC_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

[[nodiscard]] float msvc_avx2_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept;

void msvc_avx2_float_to_bfloat16_array(
    uint16_t* output,
    const float* input,
    uint32_t count) noexcept;

void msvc_avx2_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept;

[[nodiscard]] float msvc_avx2_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept;

void msvc_avx2_mxfp4_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
                              float* output, size_t output_stride) noexcept;

void msvc_avx2_mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                                  uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output,
                                  size_t first_output_stride, float* second_output, size_t second_output_stride) noexcept;

void msvc_avx2_mxfp4_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,
                                      size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                                      float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept;

[[nodiscard]] float msvc_avx512_bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept;

void msvc_avx512_float_to_bfloat16_array(
    uint16_t* output,
    const float* input,
    uint32_t count) noexcept;

[[nodiscard]] float msvc_avx512_bfloat16_pair_dot(
    const uint16_t* left,
    const uint16_t* right,
    uint32_t count) noexcept;

void msvc_avx512_bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept;

void msvc_avx512_bfloat16_batched_linear(const uint16_t* weights, const float* input, size_t input_stride, size_t token_count, uint32_t output_columns,
                                         uint32_t input_columns, float* output, size_t output_stride, int thread_count,
                                         std::vector<uint16_t>& packed_input);

[[nodiscard]] float msvc_avx512_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept;

void msvc_avx512_mxfp4_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
                                float* output, size_t output_stride) noexcept;

void msvc_avx512_mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                                    uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output,
                                    size_t first_output_stride, float* second_output, size_t second_output_stride) noexcept;

void msvc_avx512_mxfp4_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,
                                        size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                                        float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_MXFP4_MSVC_H
