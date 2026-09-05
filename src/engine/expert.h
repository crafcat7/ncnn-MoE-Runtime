#ifndef NCNN_MOE_EXPERT_H
#define NCNN_MOE_EXPERT_H

#include "kernels/activation.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

struct CompiledModel;
struct ExpertExecutionMetrics;
struct CpuExpertExecutionScratch;
struct ExpertPlan;
struct ExpertVictimExecutionMetadata;
struct LayerGraphState;
struct MoeBlockPlan;
struct SessionStatistics;
struct TensorData;
enum class ExecutionBackend;

void record_mxfp4(const TensorData& matrix, size_t input_rows, ExpertExecutionMetrics& metrics);

CpuBatch forward_shared_expert(
    const CompiledModel& model,
    const MoeBlockPlan& moe,
    const CpuBatch& input,
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
    CpuExpertExecutionScratch& scratch,
    uint32_t residency_group,
    ExecutionBackend backend,
    bool prefetch);

bool initialize_backend_aggregated_output(
    const CpuExpertExecutionScratch& scratch,
    size_t rows,
    uint32_t columns,
    CpuBatch& output);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_H
