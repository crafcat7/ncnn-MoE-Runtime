#ifndef NCNN_MOE_CPU_FLOAT8_H
#define NCNN_MOE_CPU_FLOAT8_H

#include <cstdint>

namespace ncnn {
namespace moe {

[[nodiscard]] float float8_e4m3_to_float(uint8_t value) noexcept;
[[nodiscard]] uint8_t float_to_float8_e4m3(float value) noexcept;
[[nodiscard]] float float8_e4m3_block_dot(const uint8_t* weights, const float* scales, const float* input, uint32_t count, uint32_t block_size) noexcept;
void float8_e4m3_block_dot_rows4(const uint8_t* weights, uint32_t weight_row_stride, const float* scales, const float* input, uint32_t count, uint32_t block_size,
                                 uint32_t row_count, float* output) noexcept;
[[nodiscard]] const char* float8_kernel_name() noexcept;
[[nodiscard]] uint32_t float8_linear_row_group_size() noexcept;
void quantize_float8_e4m3_inplace(float* values, uint32_t count, uint32_t block_size, bool power_of_two_scale) noexcept;
void quantize_float4_e2m1_inplace(float* values, uint32_t count, uint32_t block_size) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_FLOAT8_H
