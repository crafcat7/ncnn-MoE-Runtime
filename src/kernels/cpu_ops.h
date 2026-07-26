#ifndef NCNN_MOE_CPU_OPS_H
#define NCNN_MOE_CPU_OPS_H

#include "cpu_batch.h"

#include "ncnn/moe/types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct Mxfp4Task
{
    const TensorData* gate_up = nullptr;
    const TensorData* gate_up_bias = nullptr;
    const TensorData* down = nullptr;
    const TensorData* down_bias = nullptr;
    const CpuBatch* input = nullptr;
    CpuBatch* output = nullptr;
    float activation_limit = 0.0f;
};

struct Mxfp4Scratch
{
    std::vector<CpuBatch> activated;
    std::vector<CpuBatch> linear;
};

[[nodiscard]] float bfloat16_to_float(uint16_t value) noexcept;
[[nodiscard]] uint16_t float_to_bfloat16(float value) noexcept;
[[nodiscard]] float scaled_silu(float value, float sigmoid_scale = 1.0f) noexcept;
[[nodiscard]] float approximate_scaled_silu(float value, float sigmoid_scale = 1.0f) noexcept;
[[nodiscard]] const char* scaled_silu_kernel_name() noexcept;
[[nodiscard]] CpuBatch embedding_batch(const TensorData& embedding, std::span<const int32_t> input_ids);
void embedding_batch_into(const TensorData& embedding, std::span<const int32_t> input_ids, CpuBatch& output);
[[nodiscard]] CpuBatch linear_batch(const TensorData& matrix, const CpuBatch& input);
void linear_batch_into(const TensorData& matrix, const CpuBatch& input, CpuBatch& output);
[[nodiscard]] CpuBatch linear_batch(const TensorData& matrix, const TensorData& bias, const CpuBatch& input);
void linear_batch_into(const TensorData& matrix, const TensorData& bias, const CpuBatch& input, CpuBatch& output);
[[nodiscard]] CpuBatch fused_mxfp4_gate_up_batch(const TensorData& matrix, const TensorData* bias, const CpuBatch& input, float activation_limit);
[[nodiscard]] bool mxfp4_expert_batch(std::span<const Mxfp4Task> tasks, Mxfp4Scratch* scratch = nullptr);
[[nodiscard]] CpuBatch rms_norm_batch(const CpuBatch& input, const TensorData& weight, float epsilon);
void rms_norm_batch_into(const CpuBatch& input, const TensorData& weight, float epsilon, CpuBatch& output);
void add_bias_inplace(CpuBatch& destination, const TensorData& bias);
void add_batch_inplace(CpuBatch& destination, const CpuBatch& source);
[[nodiscard]] std::vector<std::vector<float>> batch_to_vectors(const CpuBatch& batch);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_OPS_H
