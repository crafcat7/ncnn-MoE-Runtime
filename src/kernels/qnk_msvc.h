#ifndef NCNN_MOE_QNK_MSVC_H
#define NCNN_MOE_QNK_MSVC_H

#include "qnk.h"

#include <cstddef>

namespace ncnn {
namespace moe {

#if defined(NCNN_MOE_MSVC_X86_SIMD)
float msvc_avx2_qnk_dot_block(
    DType dtype,
    const uint8_t* block,
    const float* input) noexcept;

float msvc_avx512_qnk_dot_block(
    DType dtype,
    const uint8_t* block,
    const float* input) noexcept;

void msvc_avx2_qnk_gemm(
    const QnKPack& weights,
    const float* input,
    size_t input_stride,
    size_t token_count,
    float* output,
    size_t output_stride) noexcept;

void msvc_avx512_qnk_gemm(
    const QnKPack& weights,
    const float* input,
    size_t input_stride,
    size_t token_count,
    float* output,
    size_t output_stride) noexcept;

void msvc_avx2_qnk_q8k_quantize(
    const float* source,
    uint8_t* output,
    uint32_t columns) noexcept;

void msvc_avx512_qnk_q8k_quantize(
    const float* source,
    uint8_t* output,
    uint32_t columns) noexcept;
#endif

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_QNK_MSVC_H
