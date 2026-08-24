#ifndef NCNN_MOE_CPU_GATED_DELTA_NET_H
#define NCNN_MOE_CPU_GATED_DELTA_NET_H

#include "backends/ncnn/ncnn_linear.h"
#include "ncnn/moe/execution_plan.h"

#include <span>
#include <vector>

namespace ncnn {
namespace moe {

class CpuBatch;
struct CpuGatedDeltaExecutionScratch;
struct CpuLayerCache;

struct CpuGatedDeltaBatchEntry
{
    const CpuBatch* hidden = nullptr;
    CpuGatedDeltaExecutionScratch* scratch = nullptr;
    CpuLayerCache* cache = nullptr;
    CpuBatch* output = nullptr;
};

struct CpuGatedDeltaBatchScratch
{
    std::vector<NcnnVulkanGatedDeltaBatchEntry> device_entries;
};

[[nodiscard]] Result<void> execute_gated_delta_net_into(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    CpuLayerCache& cache,
    CpuGatedDeltaExecutionScratch& scratch,
    const CpuBatch& hidden,
    CpuBatch& output,
    uint64_t optimization_flags);

bool execute_gated_delta_net_batch_into(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    std::span<CpuGatedDeltaBatchEntry> entries,
    CpuGatedDeltaBatchScratch& scratch,
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_GATED_DELTA_NET_H
