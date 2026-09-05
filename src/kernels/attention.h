#ifndef NCNN_MOE_ATTENTION_H
#define NCNN_MOE_ATTENTION_H

#include "activation.h"

#include "graph/layerplan.h"
#include "graph/graph.h"
#include "ncnn/moe/result.h"
#include "storage/weightstore.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct CpuLayerCache;

struct CpuAttentionExecutionScratch
{
    CpuBatch normalized;
    CpuBatch query;
    CpuBatch key;
    CpuBatch value;
    CpuBatch fused_qkv;
    CpuBatch attention;
    CpuBatch gate;
    CpuBatch projected;
    CpuBatch output;
    CpuBatch qsa_query_key;
    CpuBatch qsa_query;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    std::vector<float> logits;
    std::vector<size_t> qsa_selected_offsets;
    std::vector<uint32_t> qsa_selected_indices;
    std::vector<float> flash_partial_max;
    std::vector<float> flash_partial_sum;
    std::vector<float> flash_partial_output;
    std::vector<float> rope_cosine;
    std::vector<float> rope_sine;
};

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

#endif // NCNN_MOE_ATTENTION_H
