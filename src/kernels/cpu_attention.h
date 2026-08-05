#ifndef NCNN_MOE_CPU_ATTENTION_H
#define NCNN_MOE_CPU_ATTENTION_H

#include "cpu_batch.h"
#include "engine/cpu_session_state.h"

#include "ncnn/moe/execution_plan.h"

#include <cstdint>
#include <span>

namespace ncnn {
namespace moe {

struct CpuAttentionBatchEntry
{
    uint64_t position_offset = 0;
    CpuLayerCache* cache = nullptr;
    CpuAttentionExecutionScratch* scratch = nullptr;
    const CpuBatch* hidden = nullptr;
    CpuBatch* output = nullptr;
};

[[nodiscard]] Result<bool> execute_attention_block_batch_into(
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    std::span<CpuAttentionBatchEntry> entries,
    uint64_t optimization_flags);

[[nodiscard]] Result<void> execute_attention_block_into(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    DType kv_cache_dtype,
    uint64_t position_offset,
    CpuLayerCache& cache,
    CpuAttentionExecutionScratch& scratch,
    const CpuBatch& hidden,
    CpuBatch& output,
    uint64_t optimization_flags);

[[nodiscard]] Result<void> append_attention_context_into(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    DType kv_cache_dtype,
    uint64_t position_offset,
    CpuLayerCache& cache,
    CpuAttentionExecutionScratch& scratch,
    const CpuBatch& hidden,
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_ATTENTION_H
