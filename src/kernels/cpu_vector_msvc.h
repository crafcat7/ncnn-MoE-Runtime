#ifndef NCNN_MOE_CPU_VECTOR_MSVC_H
#define NCNN_MOE_CPU_VECTOR_MSVC_H

#include <cstdint>

namespace ncnn {
namespace moe {

void msvc_avx2_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept;
void msvc_avx512_float_scaled_add(float* output, const float* input, float scale, uint32_t count) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_VECTOR_MSVC_H
