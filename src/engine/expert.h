#ifndef NCNN_MOE_EXPERT_H
#define NCNN_MOE_EXPERT_H

#include "kernels/activationbuffer.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

struct CompiledModel;
struct ExpertExecutionMetrics;
struct ExpertScratch;
struct ExpertPlan;
struct ExpertVictimExecutionMetadata;
struct LayerGraphState;
struct MoeBlockPlan;
struct SessionStatistics;
struct TensorData;
enum class ExecutionBackend;

void record_mxfp4(const TensorData& matrix, size_t input_rows, ExpertExecutionMetrics& metrics);

// Input and output must be distinct; existing output capacity is reused.
void forward_shared_expert(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    const ActivationBuffer& input,
    ActivationBuffer& output,
    ExpertExecutionMetrics& metrics,
    uint64_t optimization_flags);

bool can_run_vulkan_expert(
    const ExpertPlan& expert,
    const TensorData& gate_up,
    const TensorData& down,
    uint64_t optimization_flags);

ExpertVictimExecutionMetadata victim_metadata(
    const CompiledModel& model,
    const ExpertPlan& expert,
    size_t token_count);

[[nodiscard]] Result<void> forward_moe(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    LayerGraphState& layer_state,
    SessionStatistics& statistics,
    ExpertScratch& scratch,
    uint32_t residency_group,
    ExecutionBackend backend,
    bool prefetch);

// Consume a committed aggregate, or initialize a zeroed CPU accumulator.
bool initialize_backend_aggregated_output(
    ExpertScratch& scratch,
    size_t rows,
    uint32_t columns,
    ActivationBuffer& output);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_H
