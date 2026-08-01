#ifndef NCNN_MOE_CPU_VECTOR_H
#define NCNN_MOE_CPU_VECTOR_H

#include <cstdint>

namespace ncnn {
namespace moe {

float float_dot(const float* left, const float* right, uint32_t count) noexcept;
void float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_VECTOR_H
