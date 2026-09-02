#ifndef NCNN_MOE_CPU_OPS_H
#define NCNN_MOE_CPU_OPS_H

#include "cpu_batch.h"
#include "cpu_mxfp4.h"

#include "ncnn/moe/types.h"
#include "ncnn/moe/compiled_operator.h"
#include "ncnn/moe/execution_graph.h"

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
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
    float activation_limit = 0.0f;
};

struct Mxfp4Scratch
{
    std::vector<CpuBatch> activated;
    std::vector<CpuBatch> linear;
    // Temporary full gate/up output used when the immutable MXFP4 weights
    // have a persistent 4/8-row Q8 packed sidecar.  The sidecar itself lives
    // on TensorData; these buffers are reused by the caller's scratch.
    std::vector<CpuBatch> packed_gate_up;
    std::vector<Mxfp4Q8Batch> q8_activated;
    std::vector<CpuBatch> unique_input;
    std::vector<CpuBatch> unique_output;
    std::vector<std::vector<uint32_t>> unique_row_maps;
    std::vector<Mxfp4Task> effective_tasks;
    std::vector<uint32_t> physical_input_rows;
};

[[nodiscard]] float bfloat16_to_float(uint16_t value) noexcept;
[[nodiscard]] uint16_t float_to_bfloat16(float value) noexcept;
[[nodiscard]] float scaled_silu(
    float value,
    float sigmoid_scale,
    uint64_t optimization_flags) noexcept;
[[nodiscard]] float approximate_scaled_silu(float value, float sigmoid_scale = 1.0f) noexcept;
[[nodiscard]] const char* scaled_silu_kernel_name(
    uint64_t optimization_flags) noexcept;
[[nodiscard]] bool cpu_fast_silu_enabled(uint64_t optimization_flags) noexcept;
[[nodiscard]] bool simd_rms_norm_enabled(uint64_t optimization_flags) noexcept;
[[nodiscard]] uint32_t cpu_linear_num_threads() noexcept;
[[nodiscard]] uint32_t float8_linear_num_threads() noexcept;
void embedding_batch_into(const TensorData& embedding, std::span<const int32_t> input_ids, CpuBatch& output);
[[nodiscard]] CpuBatch linear_batch(const TensorData& matrix, const CpuBatch& input, uint64_t optimization_flags, const CompiledOperator* executable = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu);
void linear_batch_into(const TensorData& matrix, const CpuBatch& input, CpuBatch& output, uint64_t optimization_flags, const CompiledOperator* executable = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu);
[[nodiscard]] bool float8_linear_pair_batch_into(
    const TensorData& first,
    const TensorData& second,
    const CpuBatch& input,
    CpuBatch& first_output,
    CpuBatch& second_output,
    uint64_t optimization_flags,
    const CompiledOperator* first_executable = nullptr,
    const CompiledOperator* second_executable = nullptr);
[[nodiscard]] bool float8_linear_rms_norm_batch_into(
    const TensorData& matrix,
    const CpuBatch& input,
    const TensorData& norm_weight,
    float epsilon,
    CpuBatch& output,
    uint64_t optimization_flags,
    const CompiledOperator* executable = nullptr);
[[nodiscard]] CpuBatch linear_batch(const TensorData& matrix, const TensorData& bias, const CpuBatch& input, uint64_t optimization_flags, const CompiledOperator* executable = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu);
void linear_batch_into(const TensorData& matrix, const TensorData& bias, const CpuBatch& input, CpuBatch& output, uint64_t optimization_flags, const CompiledOperator* executable = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu);
[[nodiscard]] bool fused_float8_gate_up_batch(const TensorData& gate, const TensorData& up, const CpuBatch& input,
                                              ExpertActivation activation, float activation_limit, CpuBatch& output, uint64_t optimization_flags,
                                              const CompiledOperator* gate_executable = nullptr,
                                              const CompiledOperator* up_executable = nullptr);
[[nodiscard]] CpuBatch fused_mxfp4_gate_up_batch(const TensorData& matrix, const TensorData* bias, const CpuBatch& input, ExpertActivation activation, float activation_limit,
                                                 uint64_t optimization_flags);
[[nodiscard]] bool mxfp4_expert_batch(std::span<const Mxfp4Task> tasks, Mxfp4Scratch* scratch, uint64_t optimization_flags);
[[nodiscard]] CpuBatch rms_norm_batch(const CpuBatch& input, const TensorData& weight, float epsilon, float weight_offset, uint64_t optimization_flags);
void rms_norm_batch_into(const CpuBatch& input, const TensorData& weight, float epsilon, CpuBatch& output, float weight_offset, uint64_t optimization_flags);
void add_bias_inplace(CpuBatch& destination, const TensorData& bias);
void add_batch_inplace(CpuBatch& destination, const CpuBatch& source);
[[nodiscard]] std::vector<std::vector<float>> batch_to_vectors(const CpuBatch& batch);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_OPS_H
