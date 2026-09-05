#ifndef NCNN_MOE_GATEDDELTANET_H
#define NCNN_MOE_GATEDDELTANET_H

#include "backends/ncnn/linear.h"
#include "graph/layerplan.h"
#include "graph/graph.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"
#include "storage/weightstore.h"

#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct CpuGatedDeltaExecutionScratch
{
    CpuBatch normalized;
    CpuBatch fused_input;
    CpuBatch qkv;
    CpuBatch z;
    CpuBatch beta;
    CpuBatch alpha;
    CpuBatch recurrent_output;
    CpuBatch projected;
    CpuBatch output;
    std::vector<float> recurrent_memory;
    std::vector<float> recurrent_delta;
};

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
    std::vector<NcnnVulkanGatedDeltaBatchEntry>& device_entries,
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GATEDDELTANET_H
