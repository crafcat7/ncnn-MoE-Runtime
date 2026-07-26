#ifndef NCNN_MOE_CPU_BFLOAT16_H
#define NCNN_MOE_CPU_BFLOAT16_H

#include <cstdint>

namespace ncnn {
namespace moe {

// Runtime-dispatched BF16-by-FP32 dot product with scalar fallback.
[[nodiscard]] float bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept;

[[nodiscard]] const char* bfloat16_kernel_name() noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_BFLOAT16_H
