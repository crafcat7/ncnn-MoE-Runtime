#ifndef NCNN_MOE_CPU_VECTOR_H
#define NCNN_MOE_CPU_VECTOR_H

#include <cstdint>

namespace ncnn {
namespace moe {

void float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_VECTOR_H
