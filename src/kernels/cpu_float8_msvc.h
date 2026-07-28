#ifndef NCNN_MOE_CPU_FLOAT8_MSVC_H
#define NCNN_MOE_CPU_FLOAT8_MSVC_H

#include <cstdint>

namespace ncnn {
namespace moe {

[[nodiscard]] float msvc_avx2_float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept;
void msvc_avx2_float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count,
                                           uint32_t block_size, uint32_t row_count, float* output) noexcept;

[[nodiscard]] float msvc_avx512_float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept;
void msvc_avx512_float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count,
                                             uint32_t block_size, uint32_t row_count, float* output) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_FLOAT8_MSVC_H
