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
    ArmSve2,
    X86Avx2,
    X86Avx512
};

[[nodiscard]] MxFp4KernelKind mxfp4_kernel_kind() noexcept;
[[nodiscard]] const char* mxfp4_kernel_name() noexcept;
[[nodiscard]] uint32_t mxfp4_decode_row_pair_group_size() noexcept;

[[nodiscard]] float mxfp4_dot(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input) noexcept;

// Reuses each decoded block across input tokens.
void mxfp4_gemm_row(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, const float* input, size_t input_stride, size_t token_count,
                    float* output, size_t output_stride) noexcept;

// Computes adjacent rows with shared activation loads.
void mxfp4_matmul_rows2(const uint8_t* first_packed, const uint8_t* first_scales, const uint8_t* second_packed, const uint8_t* second_scales,
                        uint32_t block_count, const float* input, size_t input_stride, size_t token_count, float* first_output, size_t first_output_stride,
                        float* second_output, size_t second_output_stride) noexcept;

// Computes a range of adjacent row pairs with one dispatch.
void mxfp4_matmul_row_pairs(const uint8_t* packed, const uint8_t* scales, uint32_t block_count, uint32_t row_pair_count, const float* input,
                            size_t input_stride, size_t token_count, float* first_output, size_t first_pair_stride, size_t first_token_stride,
                            float* second_output, size_t second_pair_stride, size_t second_token_stride) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_MXFP4_H
