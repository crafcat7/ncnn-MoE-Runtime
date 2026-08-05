#ifndef NCNN_MOE_CPU_GATED_DELTA_NET_H
#define NCNN_MOE_CPU_GATED_DELTA_NET_H

#include "ncnn/moe/execution_plan.h"

#include <span>

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
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_GATED_DELTA_NET_H
