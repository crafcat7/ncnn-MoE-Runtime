#ifndef NCNN_MOE_STATECACHE_H
#define NCNN_MOE_STATECACHE_H

#include "activation.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

class NcnnVulkanAttentionCache;
class NcnnVulkanGatedDeltaState;

struct CpuLatentVectorUndo
{
    size_t original_size = 0;
    size_t offset = 0;
    std::vector<float> values;
    bool captured = false;
};

struct CpuLatentCacheUndo
{
    uint64_t latent_token_count = 0;
    size_t latent_compressed_size = 0;
    size_t latent_index_compressed_size = 0;
    CpuLatentVectorUndo latent_window;
    CpuLatentVectorUndo compressor_pending_values;
    CpuLatentVectorUndo compressor_pending_scores;
    CpuLatentVectorUndo compressor_previous_values;
    CpuLatentVectorUndo compressor_previous_scores;
    CpuLatentVectorUndo index_compressor_pending_values;
    CpuLatentVectorUndo index_compressor_pending_scores;
    CpuLatentVectorUndo index_compressor_previous_values;
    CpuLatentVectorUndo index_compressor_previous_scores;

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        return static_cast<uint64_t>(
                   latent_window.values.capacity()
                   + compressor_pending_values.values.capacity()
                   + compressor_pending_scores.values.capacity()
                   + compressor_previous_values.values.capacity()
                   + compressor_previous_scores.values.capacity()
                   + index_compressor_pending_values.values.capacity()
                   + index_compressor_pending_scores.values.capacity()
                   + index_compressor_previous_values.values.capacity()
                   + index_compressor_previous_scores.values.capacity())
               * sizeof(float);
    }
};

struct CpuStateCacheSnapshot
{
    std::vector<float> gated_delta_convolution;
    std::vector<float> gated_delta_recurrent;

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        return static_cast<uint64_t>(
                   gated_delta_convolution.capacity()
                   + gated_delta_recurrent.capacity())
               * sizeof(float);
    }
};

// Session transactions record reversible KV, DeltaNet, and latent state.
struct CpuSessionStateTransaction
{
    CpuStateCacheSnapshot initial;
    std::vector<CpuStateCacheSnapshot> rows;
    uint64_t initial_start_position = 0;
    uint64_t initial_token_count = 0;
    uint64_t initial_first_slot = 0;
    uint64_t initial_gated_delta_token_count = 0;
    size_t expected_rows = 0;
    size_t recorded_rows = 0;
    bool active = false;
    std::vector<CpuLatentCacheUndo> latent_undo;
    bool latent_active = false;

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        uint64_t bytes = initial.allocated_bytes()
                         + static_cast<uint64_t>(rows.capacity())
                               * sizeof(CpuStateCacheSnapshot);
        for (const CpuStateCacheSnapshot& row : rows)
            bytes += row.allocated_bytes();
        bytes += static_cast<uint64_t>(latent_undo.capacity())
                 * sizeof(CpuLatentCacheUndo);
        for (const CpuLatentCacheUndo& undo : latent_undo)
            bytes += undo.allocated_bytes();
        return bytes;
    }
};

struct CpuLatentScoredIndex
{
    uint32_t index = 0;
    float score = 0.0f;
};

struct CpuLayerCache
{
    using LatentScoredIndex = CpuLatentScoredIndex;

    // KV cache.
    std::vector<float> keys;
    std::vector<float> values;
    std::vector<uint16_t> bfloat16_keys;
    std::vector<uint16_t> bfloat16_values;
    std::shared_ptr<NcnnVulkanAttentionCache> vulkan_attention_cache;
    uint64_t start_position = 0;
    uint64_t token_count = 0;
    uint64_t first_slot = 0;
    uint64_t capacity_tokens = 0;
    uint32_t columns = 0;
    DType dtype = DType::Float32;
    bool vulkan_attention_promotion_disabled = false;
    bool vulkan_attention_state_unknown = false;

    // Gated DeltaNet state.
    std::vector<float> gated_delta_convolution;
    std::vector<float> gated_delta_recurrent;
    std::shared_ptr<NcnnVulkanGatedDeltaState> gated_delta_device_state;
    uint64_t gated_delta_token_count = 0;
    uint64_t device_allocated_size = 0;

    // MLA compression and index state.
    std::vector<float> latent_window;
    std::vector<float> latent_compressed;
    std::vector<float> latent_index_compressed;
    std::vector<float> compressor_pending_values;
    std::vector<float> compressor_pending_scores;
    std::vector<float> compressor_previous_values;
    std::vector<float> compressor_previous_scores;
    std::vector<float> index_compressor_pending_values;
    std::vector<float> index_compressor_pending_scores;
    std::vector<float> index_compressor_previous_values;
    std::vector<float> index_compressor_previous_scores;
    std::vector<float> compressor_pooled;
    std::vector<float> compressor_exponentials;
    CpuBatch compressor_values;
    CpuBatch compressor_scores;
    CpuBatch latent_token_input;
    CpuBatch latent_token_rank;
    CpuBatch latent_index_query;
    CpuBatch latent_index_projected_weights;
    std::vector<float> latent_index_scores;
    std::vector<CpuLatentScoredIndex> latent_scored_indices;
    std::vector<uint32_t> latent_selected_indices;
    std::vector<float> latent_attention_logits;
    std::vector<float> latent_rope_cosines;
    std::vector<float> latent_rope_sines;
    uint64_t latent_token_count = 0;
    bool latent_cache = false;

    CpuSessionStateTransaction transaction;

    std::vector<uint32_t> predicted_expert_ids;
    uint32_t router_target_top_k = 0;
    uint32_t router_prefetch_width = 0;
    uint64_t router_decisions = 0;
    uint64_t router_last_adjustment = 0;
    std::vector<uint16_t> qsa_index_keys;
    std::vector<int32_t> ple_token_history;
    std::vector<float> ple_convolution_state;

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        return static_cast<uint64_t>(keys.capacity() + values.capacity()
                                     + gated_delta_convolution.capacity() + gated_delta_recurrent.capacity()
                                     + latent_window.capacity() + latent_compressed.capacity() + latent_index_compressed.capacity()
                                     + compressor_pending_values.capacity() + compressor_pending_scores.capacity()
                                     + compressor_previous_values.capacity() + compressor_previous_scores.capacity()
                                     + index_compressor_pending_values.capacity() + index_compressor_pending_scores.capacity()
                                     + index_compressor_previous_values.capacity() + index_compressor_previous_scores.capacity()
                                     + compressor_pooled.capacity() + compressor_exponentials.capacity()
                                     + latent_index_scores.capacity() + latent_attention_logits.capacity()
                                     + latent_rope_cosines.capacity() + latent_rope_sines.capacity())
                   * sizeof(float)
               + compressor_values.allocated_bytes() + compressor_scores.allocated_bytes()
               + latent_token_input.allocated_bytes() + latent_token_rank.allocated_bytes()
               + latent_index_query.allocated_bytes() + latent_index_projected_weights.allocated_bytes()
               + static_cast<uint64_t>(bfloat16_keys.capacity() + bfloat16_values.capacity()) * sizeof(uint16_t)
               + static_cast<uint64_t>(latent_scored_indices.capacity()) * sizeof(LatentScoredIndex)
               + static_cast<uint64_t>(latent_selected_indices.capacity()) * sizeof(uint32_t)
               + static_cast<uint64_t>(predicted_expert_ids.capacity()) * sizeof(uint32_t)
               + static_cast<uint64_t>(qsa_index_keys.capacity()) * sizeof(uint16_t)
               + static_cast<uint64_t>(ple_token_history.capacity()) * sizeof(int32_t)
               + static_cast<uint64_t>(ple_convolution_state.capacity()) * sizeof(float)
               + transaction.allocated_bytes()
               + device_allocated_size;
    }

    [[nodiscard]] uint64_t logical_bytes() const noexcept
    {
        const uint64_t auxiliary_size = static_cast<uint64_t>(qsa_index_keys.size()) * sizeof(uint16_t)
                                        + static_cast<uint64_t>(ple_token_history.size()) * sizeof(int32_t)
                                        + static_cast<uint64_t>(ple_convolution_state.size()) * sizeof(float);
        if (latent_cache)
        {
            const uint64_t window_tokens = std::min(latent_token_count, capacity_tokens);
            return (window_tokens * columns + latent_compressed.size() + latent_index_compressed.size()) * sizeof(float)
                   + auxiliary_size;
        }
        if (!gated_delta_convolution.empty() || !gated_delta_recurrent.empty())
        {
            return static_cast<uint64_t>(gated_delta_convolution.size() + gated_delta_recurrent.size()) * sizeof(float)
                   + auxiliary_size;
        }
        if (gated_delta_device_state)
        {
            return device_allocated_size + auxiliary_size;
        }
        const uint64_t element_size = dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
        return token_count * columns * element_size * 2 + auxiliary_size;
    }
};

[[nodiscard]] Result<void> begin_state_cache_transaction(
    std::span<CpuLayerCache> caches,
    size_t expected_rows);

void record_standard_cache_transaction_rows(
    CpuLayerCache& cache,
    size_t rows);

void record_gated_delta_cache_transaction_row(
    CpuLayerCache& cache);

[[nodiscard]] Result<void> finish_state_cache_transaction(
    std::span<CpuLayerCache> caches,
    size_t committed_rows);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_STATECACHE_H
