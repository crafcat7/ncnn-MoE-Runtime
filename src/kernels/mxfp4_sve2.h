#ifndef NCNN_MOE_MXFP4_SVE2_H
#define NCNN_MOE_MXFP4_SVE2_H

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

[[nodiscard]] float sve2_mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept;

void sve2_mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                             uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output, size_t first_output_stride,
                             float* second_output, size_t second_output_stride) noexcept;

void sve2_mxfp4_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,
                                 size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                                 float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MXFP4_SVE2_H
