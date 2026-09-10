#ifndef NCNN_MOE_OPS_H
#define NCNN_MOE_OPS_H

#include "activationbuffer.h"
#include "mxfp4.h"

#include "ncnn/moe/types.h"
#include "graph/compiledoperator.h"
#include "graph/graph.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct Mxfp4Task
{
    const TensorData* gate_up = nullptr;
    const TensorData* gate_up_bias = nullptr;
    const CompiledOperator* gate_up_operator = nullptr;
    const TensorData* down = nullptr;
    const TensorData* down_bias = nullptr;
    const CompiledOperator* down_operator = nullptr;
    const ActivationBuffer* input = nullptr;
    ActivationBuffer* output = nullptr;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
    float activation_limit = 0.0f;
};

struct Mxfp4Scratch
{
    std::vector<ActivationBuffer> activated;
    std::vector<ActivationBuffer> linear;
    // Temporary full gate/up output used when the immutable MXFP4 weights
    // have a persistent 4/8-row Q8 packed sidecar.  The sidecar itself lives
    // on the compiled operator owner; these buffers are reused by the
    // caller's scratch.
    std::vector<ActivationBuffer> packed_gate_up;
    std::vector<Mxfp4Q8Batch> q8_inputs;
    std::vector<size_t> q8_input_owner;
    std::vector<Mxfp4Q8Batch> q8_activated;
    std::vector<ActivationBuffer> unique_input;
    std::vector<ActivationBuffer> unique_output;
    std::vector<std::vector<uint32_t>> unique_row_maps;
    std::vector<Mxfp4Task> effective_tasks;
    std::vector<uint32_t> physical_input_rows;
};

[[nodiscard]] bool mxfp4_expert_decode(std::span<const Mxfp4Task> tasks,
                                       Mxfp4Scratch* scratch,
                                       uint64_t optimization_flags);

[[nodiscard]] float bfloat16_to_float(uint16_t value) noexcept;
[[nodiscard]] uint16_t float_to_bfloat16(float value) noexcept;
[[nodiscard]] float scaled_silu(float value,
                                float sigmoid_scale,
                                uint64_t optimization_flags) noexcept;
[[nodiscard]] float approximate_scaled_silu(float value, float sigmoid_scale = 1.0f) noexcept;
[[nodiscard]] const char* scaled_silu_kernel_name(uint64_t optimization_flags) noexcept;
[[nodiscard]] uint32_t cpu_linear_num_threads() noexcept;
void embedding_batch_into(const TensorData& embedding, std::span<const int32_t> input_ids, ActivationBuffer& output);
[[nodiscard]] ActivationBuffer linear_batch(const TensorData& matrix, const ActivationBuffer& input, uint64_t optimization_flags, const CompiledOperator* executable = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu);
// A quantized input scratch, when supplied, must not alias input or output.
void linear_batch_into(const TensorData& matrix, const ActivationBuffer& input, ActivationBuffer& output, uint64_t optimization_flags, const CompiledOperator* executable = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu, ActivationBuffer* quantized_input_scratch = nullptr);
[[nodiscard]] bool float8_linear_pair_batch_into(const TensorData& first,
                                                 const TensorData& second,
                                                 const ActivationBuffer& input,
                                                 ActivationBuffer& first_output,
                                                 ActivationBuffer& second_output,
                                                 uint64_t optimization_flags,
                                                 const CompiledOperator* first_executable = nullptr,
                                                 const CompiledOperator* second_executable = nullptr,
                                                 ActivationBuffer* quantized_input_scratch = nullptr);
[[nodiscard]] bool float8_linear_rms_norm_batch_into(const TensorData& matrix,
                                                     const ActivationBuffer& input,
                                                     const TensorData& norm_weight,
                                                     float epsilon,
                                                     ActivationBuffer& output,
                                                     uint64_t optimization_flags,
                                                     const CompiledOperator* executable = nullptr,
                                                     ActivationBuffer* quantized_input_scratch = nullptr);
[[nodiscard]] ActivationBuffer linear_batch(const TensorData& matrix, const TensorData& bias, const ActivationBuffer& input, uint64_t optimization_flags, const CompiledOperator* executable = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu);
void linear_batch_into(const TensorData& matrix, const TensorData& bias, const ActivationBuffer& input, ActivationBuffer& output, uint64_t optimization_flags, const CompiledOperator* executable = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu, ActivationBuffer* quantized_input_scratch = nullptr);
[[nodiscard]] bool fused_float8_gate_up_batch(const TensorData& gate, const TensorData& up, const ActivationBuffer& input,
                                              ExpertActivation activation, float activation_limit, ActivationBuffer& output, uint64_t optimization_flags,
                                              const CompiledOperator* gate_executable = nullptr,
                                              const CompiledOperator* up_executable = nullptr);
[[nodiscard]] ActivationBuffer fused_mxfp4_gate_up_batch(const TensorData& matrix, const TensorData* bias, const ActivationBuffer& input, ExpertActivation activation, float activation_limit,
                                                         uint64_t optimization_flags);
[[nodiscard]] bool mxfp4_expert_batch(std::span<const Mxfp4Task> tasks, Mxfp4Scratch* scratch, uint64_t optimization_flags);
[[nodiscard]] ActivationBuffer rms_norm_batch(const ActivationBuffer& input, const TensorData& weight, float epsilon, float weight_offset, uint64_t optimization_flags);
// Input and output may be the same buffer; each row's RMS is computed before writing it.
void rms_norm_batch_into(const ActivationBuffer& input, const TensorData& weight, float epsilon, ActivationBuffer& output, float weight_offset, uint64_t optimization_flags);
void add_bias_inplace(ActivationBuffer& destination, const TensorData& bias);
void add_batch_inplace(ActivationBuffer& destination, const ActivationBuffer& source);
[[nodiscard]] std::vector<std::vector<float>> batch_to_vectors(const ActivationBuffer& batch);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_OPS_H
