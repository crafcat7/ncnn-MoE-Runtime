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
struct ExpertWorkspace;
struct ExpertPlan;
struct ExpertVictimExecutionMetadata;
struct LayerState;
struct MoeBlockPlan;
struct SessionStatistics;
struct TensorData;
enum class ExecutionBackend;

void record_mxfp4(const TensorData& matrix, size_t input_rows, ExpertExecutionMetrics& metrics);

// Input and output must be distinct; existing output capacity is reused.
void forward_shared_expert(const CompiledModel& model,
                           const MoeBlockPlan& moe,
                           const ActivationBuffer& input,
                           ActivationBuffer& output,
                           ExpertWorkspace& workspace,
                           ExpertExecutionMetrics& metrics);

bool support_vulkan_expert(const ExpertPlan& expert,
                           const TensorData& gate_up,
                           const TensorData& down,
                           uint64_t optimization_flags);

ExpertVictimExecutionMetadata victim_metadata(const CompiledModel& model,
                                              const ExpertPlan& expert,
                                              size_t token_count);

// Transfer dispatched routes into reusable execution slots and record demand.
void prepare_moe_experts(const MoeBlockPlan& moe,
                         LayerState& layer_state,
                         SessionStatistics& statistics);

// Submit exact reads without waiting or acquiring compute-time leases.
[[nodiscard]] Result<void> request_moe_experts(const CompiledModel& model,
                                               const MoeBlockPlan& moe,
                                               const LayerState& layer_state,
                                               ExpertScratch& scratch,
                                               uint32_t residency_group,
                                               SessionStatistics& statistics);

[[nodiscard]] Result<void> forward_moe(const CompiledModel& model,
                                       const MoeBlockPlan& moe,
                                       LayerState& layer_state,
                                       SessionStatistics& statistics,
                                       ExpertScratch& scratch,
                                       uint32_t residency_group,
                                       ExecutionBackend backend,
                                       bool prefetch);

// Consume a committed aggregate, or initialize a zeroed CPU accumulator.
bool initialize_backend_aggregated_output(ExpertScratch& scratch,
                                          size_t rows,
                                          uint32_t columns,
                                          ActivationBuffer& output);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_H
