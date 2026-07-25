#ifndef NCNN_MOE_CPU_MXFP4_H
#define NCNN_MOE_CPU_MXFP4_H

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

enum class MxFp4KernelKind
{
    Scalar,
    ArmNeon,
    X86Avx2,
    X86Avx512
};

[[nodiscard]] MxFp4KernelKind mxfp4_kernel_kind() noexcept;
[[nodiscard]] const char* mxfp4_kernel_name() noexcept;

[[nodiscard]] float mxfp4_dot(
    const uint8_t* packed,
    const uint8_t* scales,
    uint32_t block_count,
    const float* input) noexcept;

// Computes one matrix row for multiple input tokens. Unlike repeated GEMV,
// this path decodes each packed weight block once and reuses it across tokens.
void mxfp4_gemm_row(
    const uint8_t* packed,
    const uint8_t* scales,
    uint32_t block_count,
    const float* input,
    size_t input_stride,
    size_t token_count,
    float* output,
    size_t output_stride) noexcept;

// Computes two adjacent matrix rows while sharing activation loads. This is
// used by decode GEMV and prefill GEMM, including interleaved Gate/Up rows.
void mxfp4_matmul_rows2(
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

#endif // NCNN_MOE_CPU_MXFP4_H
