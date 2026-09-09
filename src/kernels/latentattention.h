#ifndef NCNN_MOE_LATENTATTENTION_H
#define NCNN_MOE_LATENTATTENTION_H

#include "activationbuffer.h"

#include "graph/layerplan.h"
#include "graph/graph.h"
#include "ncnn/moe/result.h"
#include "storage/weightstore.h"

#include <cstdint>
#include <span>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
struct LayerCache;
struct AttentionScratch;

void begin_latent_cache_transaction(
    std::span<LayerCache> caches);

[[nodiscard]] Result<void> finish_latent_cache_transaction(
    std::span<LayerCache> caches,
    size_t committed_rows);

// Outputs are caller-owned. They may be scratch.output, but must not alias
// the input or another scratch slot. Scratch is exclusive to one forward call.
[[nodiscard]] Result<void> forward_latent_attention(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    uint64_t position_offset,
    LayerCache& cache,
    AttentionScratch& scratch,
    const ActivationBuffer& input,
    ActivationBuffer& output,
    uint64_t optimization_flags);

[[nodiscard]] Result<void> forward_latent_attention_batch(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    std::span<const uint64_t> positions,
    std::span<LayerCache* const> caches,
    AttentionScratch& scratch,
    const ActivationBuffer& input,
    ActivationBuffer& output,
    uint64_t optimization_flags);

[[nodiscard]] Result<void> append_dspark_attention_context(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    uint64_t position_offset,
    LayerCache& cache,
    const ActivationBuffer& input,
    uint64_t optimization_flags);

[[nodiscard]] Result<void> forward_dspark_attention(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    uint64_t position_offset,
    const LayerCache& cache,
    AttentionScratch& scratch,
    const ActivationBuffer& input,
    ActivationBuffer& output,
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LATENTATTENTION_H
