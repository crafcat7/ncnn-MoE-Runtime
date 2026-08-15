#ifndef NCNN_MOE_CPU_BFLOAT16_H
#define NCNN_MOE_CPU_BFLOAT16_H

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

class Bfloat16BatchedLinearExecutionCounter
{
public:
    [[nodiscard]] uint64_t dispatch_count() const noexcept;

private:
    friend class ScopedBfloat16BatchedLinearExecutionCounter;
    friend void record_bfloat16_batched_linear_dispatch() noexcept;
    std::atomic<uint64_t> dispatch_count_{0};
};

class ScopedBfloat16BatchedLinearExecutionCounter
{
public:
    explicit ScopedBfloat16BatchedLinearExecutionCounter(
        Bfloat16BatchedLinearExecutionCounter* counter) noexcept;
    ~ScopedBfloat16BatchedLinearExecutionCounter();

    ScopedBfloat16BatchedLinearExecutionCounter(
        const ScopedBfloat16BatchedLinearExecutionCounter&) = delete;
    ScopedBfloat16BatchedLinearExecutionCounter& operator=(
        const ScopedBfloat16BatchedLinearExecutionCounter&) = delete;

private:
    Bfloat16BatchedLinearExecutionCounter* previous_ = nullptr;
};

[[nodiscard]] Bfloat16BatchedLinearExecutionCounter*
current_bfloat16_batched_linear_execution_counter() noexcept;

// Runtime-dispatched BF16-by-FP32 dot product with scalar fallback.
[[nodiscard]] float bfloat16_dot(const uint16_t* weights, const float* input, uint32_t count) noexcept;
[[nodiscard]] const char* bfloat16_dot_kernel_name() noexcept;

// Runtime-dispatched round-to-nearest-even conversion used by the KV cache
// and optional BF16 attention packing paths.
void float_to_bfloat16_array(
    uint16_t* output,
    const float* input,
    uint32_t count) noexcept;

// BF16-by-BF16 dot product. AVX512-BF16 implementations use dpbf16; callers
// can query availability before accepting the additional input quantization.
[[nodiscard]] float bfloat16_pair_dot(
    const uint16_t* left,
    const uint16_t* right,
    uint32_t count) noexcept;
[[nodiscard]] bool bfloat16_pair_dot_available() noexcept;

// Runtime-dispatched accumulation of BF16 values into an FP32 vector.
void bfloat16_scaled_add(float* output, const uint16_t* input, float scale, uint32_t count) noexcept;

// Uses a runtime-dispatched BF16 dot-product matrix kernel when a batch can
// amortize packing the FP32 activations to BF16.
// Returns false without modifying output when the kernel is unavailable or not
// admitted.
[[nodiscard]] bool bfloat16_batched_linear(const uint16_t* weights,
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

#endif // NCNN_MOE_CPU_BFLOAT16_H
