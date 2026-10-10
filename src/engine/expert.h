#ifndef NCNN_MOE_EXPERT_H
#define NCNN_MOE_EXPERT_H

#include "kernels/activationbuffer.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace ncnn {
namespace moe {

class DeviceTensor_vulkan;
struct CompiledModel;
struct ExpertScratch;
struct ExpertWorkspace;
struct ExpertPlan;
struct ExpertVictimExecutionMetadata;
struct LayerState;
struct MoeBlockPlan;
struct SessionStatistics;
struct TensorData;
enum class ExecutionBackend;

// Input and output must be distinct; existing output capacity is reused.
void forward_shared_expert(const CompiledModel& model,
                           const MoeBlockPlan& moe,
                           const ActivationBuffer& input,
                           ActivationBuffer& output,
                           ExpertWorkspace& workspace,
                           const std::shared_ptr<const DeviceTensor_vulkan>& device_input = {},
                           std::shared_ptr<const DeviceTensor_vulkan>* device_output = nullptr);

bool support_vulkan_expert(const ExpertPlan& expert,
                           const TensorData& gate_up,
                           const TensorData& down,
                           uint64_t optimization_flags);

ExpertVictimExecutionMetadata victim_metadata(const CompiledModel& model,
                                              const ExpertPlan& expert,
                                              size_t token_count);

// Transfer dispatched routes into reusable execution slots and record demand.
void prepare_experts(const MoeBlockPlan& moe,
                     LayerState& layer_state,
                     SessionStatistics& stats);

// Submit exact reads without waiting or acquiring compute-time leases.
[[nodiscard]] Result<void> request_experts(const CompiledModel& model,
                                           const MoeBlockPlan& moe,
                                           const LayerState& layer_state,
                                           ExpertScratch& scratch,
                                           uint32_t residency_group);

// independent_work runs at most once, after a demand ticket begins and
// before its wait; callers retain the normal graph-node fallback if it never runs.
[[nodiscard]] Result<void> forward_moe(const CompiledModel& model,
                                       const MoeBlockPlan& moe,
                                       LayerState& layer_state,
                                       SessionStatistics& stats,
                                       ExpertScratch& scratch,
                                       uint32_t residency_group,
                                       ExecutionBackend backend,
                                       bool prefetch,
                                       const std::function<void()>& independent_work = {});

// Consume a committed aggregate, or initialize a zeroed CPU accumulator.
bool init_moe_output(ExpertScratch& scratch,
                     size_t rows,
                     uint32_t columns,
                     ActivationBuffer& output);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_H
