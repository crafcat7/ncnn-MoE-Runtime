#ifndef NCNN_MOE_ATTENTION_H
#define NCNN_MOE_ATTENTION_H

#include "activationbuffer.h"

#include "graph/layerplan.h"
#include "graph/graph.h"
#include "ncnn/moe/result.h"
#include "storage/weightstore.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
struct LayerCache;

struct LatentAttentionRowContext
{
    uint64_t window_begin = 0;
    uint32_t window_count = 0;
    uint32_t compressed_count = 0;
    bool selected_compressed_indices = false;
    std::span<const uint32_t> compressed_indices;
    std::span<float> logits;
};

// Scratch storage is exclusive to one in-flight attention call.
struct AttentionScratch
{
    ActivationBuffer normalized;
    ActivationBuffer query;
    ActivationBuffer key;
    ActivationBuffer value;
    ActivationBuffer fused_qkv;
    ActivationBuffer attention;
    ActivationBuffer gate;
    ActivationBuffer projected;
    ActivationBuffer output;
    ActivationBuffer qsa_query_key;
    ActivationBuffer qsa_query;
    ActivationBuffer quantized_input;
    ActivationBuffer latent_compressor_values;
    ActivationBuffer latent_compressor_scores;
    ActivationBuffer latent_index_compressor_values;
    ActivationBuffer latent_index_compressor_scores;
    ActivationBuffer latent_token_input;
    ActivationBuffer latent_token_rank;
    ActivationBuffer latent_index_query;
    ActivationBuffer latent_index_projected_weights;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    std::vector<float> logits;
    std::vector<std::pair<float, uint32_t>> index_scores;
    std::vector<float> latent_compressor_pooled;
    std::vector<float> latent_compressor_exponentials;
    std::vector<size_t> qsa_selected_offsets;
    std::vector<uint32_t> qsa_selected_indices;
    std::vector<float> flash_partial_max;
    std::vector<float> flash_partial_sum;
    std::vector<float> flash_partial_output;
    std::vector<float> workspace;
    // FP16 flash: KV window converted once per forward, FP16 QK/PV dots.
    std::vector<uint16_t> f16_key;
    std::vector<uint16_t> f16_val;
    std::vector<uint16_t> f16_work;
    std::vector<float> rope_cosine;
    std::vector<float> rope_sine;
    // Reused by every layer of a forward: cos/sin depend only on position.
    std::vector<float> rope_cosine_cache;
    std::vector<float> rope_sine_cache;
    uint64_t rope_cache_position = ~uint64_t(0);
    uint32_t rope_cache_rows = 0;
    uint32_t rope_cache_half_dim = 0;
    uint32_t rope_cache_active = 0;
    uint32_t rope_cache_context = 0;
    float rope_cache_theta = 0.0f;
    float rope_cache_factor = 0.0f;
    float rope_cache_ntk_alpha = 0.0f;
    float rope_cache_ntk_beta = 0.0f;
    std::vector<LatentAttentionRowContext> latent_row_contexts;
    std::vector<std::pair<const LayerCache*, uint32_t>> latent_projected_compressed_counts;
    std::vector<uint64_t> latent_positions;
    std::vector<LayerCache*> latent_caches;
};

struct AttentionBatchEntry
{
    uint64_t position_offset = 0;
    LayerCache* cache = nullptr;
    AttentionScratch* scratch = nullptr;
    const ActivationBuffer* hidden = nullptr;
    ActivationBuffer* output = nullptr;
};

[[nodiscard]] Result<bool> forward_attention_batch(const CompiledOperatorTable& operators,
                                                   const AttentionBlockPlan& plan,
                                                   ExecutionBackend backend,
                                                   std::span<AttentionBatchEntry> entries,
                                                   uint64_t optimization_flags);

[[nodiscard]] Result<void> forward_attention(const WeightStore& weights,
                                             const CompiledOperatorTable& operators,
                                             const AttentionBlockPlan& plan,
                                             ExecutionBackend backend,
                                             uint64_t position_offset,
                                             LayerCache& cache,
                                             AttentionScratch& scratch,
                                             const ActivationBuffer& hidden,
                                             ActivationBuffer& output,
                                             uint64_t optimization_flags);

[[nodiscard]] Result<void> append_attention_context(const WeightStore& weights,
                                                    const CompiledOperatorTable& operators,
                                                    const AttentionBlockPlan& plan,
                                                    ExecutionBackend backend,
                                                    uint64_t position_offset,
                                                    LayerCache& cache,
                                                    AttentionScratch& scratch,
                                                    const ActivationBuffer& hidden,
                                                    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_ATTENTION_H
