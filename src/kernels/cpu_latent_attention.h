#ifndef NCNN_MOE_CPU_LATENT_ATTENTION_H
#define NCNN_MOE_CPU_LATENT_ATTENTION_H

#include "cpu_batch.h"

#include "ncnn/moe/execution_plan.h"
#include "ncnn/moe/result.h"

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
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& input);

[[nodiscard]] Result<CpuBatch> execute_latent_attention_batch(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    std::span<const uint64_t> positions,
    std::span<CpuLayerCache* const> caches,
    const CpuBatch& input);

[[nodiscard]] Result<void> append_dspark_attention_context(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& input);

[[nodiscard]] Result<CpuBatch> execute_dspark_attention(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    uint64_t position_offset,
    const CpuLayerCache& cache,
    const CpuBatch& input);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_LATENT_ATTENTION_H
