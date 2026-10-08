#ifndef NCNN_MOE_OPS_H
#define NCNN_MOE_OPS_H

#include "activationbuffer.h"
#include "mxfp4.h"

#include "ncnn/moe/types.h"
#include "graph/graph.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct CompiledOperator;

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
    std::vector<uint32_t> representatives;
    std::vector<uint64_t> representative_hashes;
    std::vector<uint8_t> q8_down_enabled;
    std::vector<uint8_t> q8_gate_packed;
    std::vector<std::shared_ptr<const Mxfp4Q8PackedMatrix>> q8_down_packed;
};

[[nodiscard]] inline float bfloat16_to_float(uint16_t value) noexcept
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

[[nodiscard]] inline uint16_t float_to_bfloat16(float value) noexcept
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}
[[nodiscard]] float scaled_silu(float value,
                                float sigmoid_scale,
                                uint64_t optimization_flags) noexcept;
[[nodiscard]] const char* scaled_silu_kernel_name(uint64_t optimization_flags) noexcept;
[[nodiscard]] uint32_t cpu_linear_num_threads() noexcept;
// Shares Linear's workload threshold and current thread budget with callers
// choosing between operator-level and outer task parallelism.
[[nodiscard]] int cpu_linear_team_size(uint64_t operation_count, DType dtype) noexcept;
void forward_embedding(const TensorData& embedding, std::span<const int32_t> input_ids, ActivationBuffer& output);
[[nodiscard]] ActivationBuffer forward_linear(const TensorData& matrix, const ActivationBuffer& input, uint64_t optimization_flags, const CompiledOperator* op = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu);
// A quantized input scratch, when supplied, must not alias input or output.
void forward_linear(const TensorData& matrix, const ActivationBuffer& input, ActivationBuffer& output, uint64_t optimization_flags, const CompiledOperator* op = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu, ActivationBuffer* quantized_input_scratch = nullptr);
[[nodiscard]] bool forward_linear_pair_float8(const TensorData& first,
                                              const TensorData& second,
                                              const ActivationBuffer& input,
                                              ActivationBuffer& first_output,
                                              ActivationBuffer& second_output,
                                              uint64_t optimization_flags,
                                              const CompiledOperator* first_executable = nullptr,
                                              const CompiledOperator* second_executable = nullptr,
                                              ActivationBuffer* quantized_input_scratch = nullptr);
[[nodiscard]] bool forward_linear_rms_norm_float8(const TensorData& matrix,
                                                  const ActivationBuffer& input,
                                                  const TensorData& norm_weight,
                                                  float epsilon,
                                                  ActivationBuffer& output,
                                                  uint64_t optimization_flags,
                                                  const CompiledOperator* op = nullptr,
                                                  ActivationBuffer* quantized_input_scratch = nullptr);
[[nodiscard]] ActivationBuffer forward_linear(const TensorData& matrix, const TensorData& bias, const ActivationBuffer& input, uint64_t optimization_flags, const CompiledOperator* op = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu);
void forward_linear(const TensorData& matrix, const TensorData& bias, const ActivationBuffer& input, ActivationBuffer& output, uint64_t optimization_flags, const CompiledOperator* op = nullptr, ExecutionBackend backend = ExecutionBackend::Cpu, ActivationBuffer* quantized_input_scratch = nullptr);
[[nodiscard]] bool forward_gate_up_float8(const TensorData& gate, const TensorData& up, const ActivationBuffer& input,
                                          ExpertActivation activation, float activation_limit, ActivationBuffer& output, uint64_t optimization_flags,
                                          const CompiledOperator* gate_executable = nullptr,
                                          const CompiledOperator* up_executable = nullptr,
                                          ActivationBuffer* quantized_input_scratch = nullptr);
// Input and output must be distinct Float32 buffers.
void forward_gate_up_mxfp4(const TensorData& matrix, const TensorData* bias, const ActivationBuffer& input, ExpertActivation activation, float activation_limit,
                           ActivationBuffer& output, uint64_t optimization_flags);
[[nodiscard]] bool forward_experts_mxfp4(std::span<const Mxfp4Task> tasks, Mxfp4Scratch* scratch, uint64_t optimization_flags);
[[nodiscard]] ActivationBuffer forward_rms_norm(const ActivationBuffer& input, const TensorData& weight, float epsilon, float weight_offset);
// Input and output may be the same buffer; each row's RMS is computed before writing it.
void forward_rms_norm(const ActivationBuffer& input, const TensorData& weight, float epsilon, ActivationBuffer& output, float weight_offset);
void add_bias_inplace(ActivationBuffer& destination, const TensorData& bias);
void add_batch_inplace(ActivationBuffer& destination, const ActivationBuffer& source);
[[nodiscard]] std::vector<std::vector<float>> batch_to_vectors(const ActivationBuffer& batch);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_OPS_H
