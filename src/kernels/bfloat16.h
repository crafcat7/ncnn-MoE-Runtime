#ifndef NCNN_MOE_BFLOAT16_H
#define NCNN_MOE_BFLOAT16_H

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

// Runtime-dispatched BF16-by-FP32 dot product with scalar fallback.
[[nodiscard]] float bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept;
[[nodiscard]] const char* bfloat16_dot_kernel_name() noexcept;

// Runtime-dispatched round-to-nearest-even conversion used by the KV cache.
void float_to_bfloat16_array(uint16_t* output,
                             const float* input,
                             uint32_t count) noexcept;

// BF16-by-BF16 dot product. AVX512-BF16 implementations use dpbf16 when available.
[[nodiscard]] float bfloat16_pair_dot(const uint16_t* left,
                                      const uint16_t* right,
                                      uint32_t count) noexcept;

// Runtime-dispatched accumulation of BF16 values into an FP32 vector.
void bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept;

// output[c] += sum(weights[k] * values[k][c]) over two contiguous key ranges.
// Used by the attention probability-value product.
void pv_bf16(float* output,
             const uint16_t* values0,
             size_t value_stride,
             const float* weights0,
             uint32_t count0,
             const uint16_t* values1,
             const float* weights1,
             uint32_t count1,
             uint32_t columns) noexcept;

// Uses a runtime-dispatched BF16 dot-product matrix kernel when a batch can
// amortize packing the FP32 activations to BF16.
// Returns false without modifying output when the kernel is unavailable or not
// admitted.
[[nodiscard]] bool forward_linear_bf16(const uint16_t* weights,
                                       const float* input,
                                       size_t input_stride,
                                       size_t token_count,
                                       uint32_t output_columns,
                                       uint32_t input_columns,
                                       float* output,
                                       size_t output_stride,
                                       int thread_count,
                                       uint64_t optimization_flags);
[[nodiscard]] const char* bfloat16_batched_linear_kernel_name(uint64_t optimization_flags) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_BFLOAT16_H
