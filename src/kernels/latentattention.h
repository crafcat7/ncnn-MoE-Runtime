#ifndef NCNN_MOE_LATENTATTENTION_H
#define NCNN_MOE_LATENTATTENTION_H

#include "activation.h"

#include "graph/layerplan.h"
#include "graph/graph.h"
#include "ncnn/moe/result.h"
#include "storage/weightstore.h"

#include <cstdint>
#include <span>

namespace ncnn {
namespace moe {

struct CpuLayerCache;

void begin_latent_cache_transaction(
    std::span<CpuLayerCache> caches);

[[nodiscard]] Result<void> finish_latent_cache_transaction(
    std::span<CpuLayerCache> caches,
    size_t committed_rows);

[[nodiscard]] Result<CpuBatch> execute_latent_attention(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& input,
    uint64_t optimization_flags);

[[nodiscard]] Result<CpuBatch> execute_latent_attention_batch(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    std::span<const uint64_t> positions,
    std::span<CpuLayerCache* const> caches,
    const CpuBatch& input,
    uint64_t optimization_flags);

[[nodiscard]] Result<void> append_dspark_attention_context(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& input,
    uint64_t optimization_flags);

[[nodiscard]] Result<CpuBatch> execute_dspark_attention(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    uint64_t position_offset,
    const CpuLayerCache& cache,
    const CpuBatch& input,
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LATENTATTENTION_H
