#ifndef NCNN_MOE_CPU_SESSION_STATE_H
#define NCNN_MOE_CPU_SESSION_STATE_H

#include "kernels/cpu_ops.h"
#include "kernels/cpu_hyper_connection.h"
#include "engine/cpu_task_worker.h"
#include "engine/expert_backend.h"
#include "storage/expert_cache.h"

#include "ncnn/moe/expert_dispatcher.h"
#include "ncnn/moe/memory_manager.h"
#include "ncnn/moe/types.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

class NcnnVulkanAttentionCache;
class NcnnVulkanGatedDeltaState;

struct ExpertExecutionMetrics
{
    uint64_t hinted_bytes = 0;
    uint64_t cache_wait_time_microseconds = 0;
    uint64_t regroup_time_microseconds = 0;
    uint64_t mxfp4_decode_gemv_rows = 0;
    uint64_t mxfp4_prefill_gemm_rows = 0;
    uint64_t mxfp4_paired_rows = 0;
    uint64_t mxfp4_fused_gate_up_rows = 0;
    uint64_t mxfp4_reused_input_rows = 0;
};

struct ActiveExpertExecution
{
    ExpertBatch batch;
    CpuBatch input;
    CpuBatch output;
    ExpertCacheLease lease;
    ExpertExecutionMetrics metrics;
    Error error;
    bool failed = false;

    void prepare(const ExpertBatch& next_batch)
    {
        batch.expert_id = next_batch.expert_id;
        batch.routes.assign(next_batch.routes.begin(), next_batch.routes.end());
        lease = {};
        metrics = {};
        error = {};
        failed = false;
    }

    void prepare(ExpertBatch&& next_batch)
    {
        batch = std::move(next_batch);
        lease = {};
        metrics = {};
        error = {};
        failed = false;
    }
};

struct LayerGraphState
{
    CpuBatch normalized;
    CpuBatch router_logits;
    CpuHyperConnectionMix ffn_hyper_mix;
    CpuBatch shared_expert_output;
    ExpertDispatchPlan dispatch_plan;
    std::vector<ActiveExpertExecution> active_experts;
    std::chrono::steady_clock::time_point router_start;
    std::chrono::steady_clock::time_point expert_start;
    bool experts_executed = false;

    void reset()
    {
        normalized.clear();
        router_logits.clear();
        shared_expert_output.clear();
        for (ActiveExpertExecution& active : active_experts)
            active.lease = {};
        experts_executed = false;
    }
};

struct CpuExpertExecutionScratch
{
    Mxfp4Scratch kernels;
    std::vector<Mxfp4Task> decode_tasks;
    std::vector<size_t> uncached_indices;
    std::vector<size_t> pending_indices;
    std::vector<size_t> ready_indices;
    std::vector<ExpertCachePairRequest> cache_requests;
    std::vector<ExpertCacheLease> cache_leases;
    std::vector<uint8_t> backend_executed;
    std::vector<uint8_t> backend_aggregated;
    std::vector<size_t> backend_indices;
    std::vector<ExpertBackendRequest> backend_requests;
    std::vector<size_t> failed_indices;
    CpuBatch backend_aggregated_output;
    CpuBatch staged_merged;
    CpuBatch staged_output;
    CpuBatch staged_router_logits;
    std::vector<int32_t> staged_input_ids;
};

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
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    std::vector<float> logits;
    std::vector<uint16_t> query_bfloat16;
    std::vector<float> rope_cosine;
    std::vector<float> rope_sine;
};

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

// Transactional ownership belongs to the Session state boundary.  It records
// the reversible portion of KV, Gated DeltaNet, and latent state without
// making any of those state stores own transaction policy.
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

// Persistent attention state is kept in an explicit component.  The
// inheritance is intentional here: existing kernel code can access the
// component fields without a compatibility mirror, while ownership and
// accounting remain separated by state domain.
struct CpuKvState
{
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
};

// Recurrent state and its transactional device mirror are independent from
// KV/MLA storage.  This is the unit that Gated DeltaNet snapshots and commits.
struct CpuGatedDeltaState
{
    std::vector<float> gated_delta_convolution;
    std::vector<float> gated_delta_recurrent;
    std::shared_ptr<NcnnVulkanGatedDeltaState> gated_delta_device_state;
    uint64_t gated_delta_token_count = 0;
    uint64_t device_allocated_bytes = 0;
};

struct CpuLatentScoredIndex
{
    uint32_t index = 0;
    float score = 0.0f;
};

// MLA compression/index state and its undo records have their own lifetime;
// they are not KV entries and must not participate in recurrent transactions.
struct CpuLatentState
{
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
};

struct RouterPrefetchState
{
    uint32_t target_top_k = 0;
    uint32_t prefetch_width = 0;
    uint64_t decisions = 0;
    uint64_t last_adjustment_decision = 0;
};

struct CpuLayerCache
    : CpuKvState,
      CpuGatedDeltaState,
      CpuLatentState
{
    using LatentScoredIndex = CpuLatentScoredIndex;

    CpuSessionStateTransaction transaction;

    std::vector<uint32_t> predicted_expert_ids;
    RouterPrefetchState next_router_prediction;

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
               + transaction.allocated_bytes()
               + device_allocated_bytes;
    }

    [[nodiscard]] uint64_t logical_bytes() const noexcept
    {
        if (latent_cache)
        {
            const uint64_t window_tokens = std::min(latent_token_count, capacity_tokens);
            return (window_tokens * columns + latent_compressed.size() + latent_index_compressed.size()) * sizeof(float);
        }
        if (!gated_delta_convolution.empty() || !gated_delta_recurrent.empty())
        {
            return static_cast<uint64_t>(gated_delta_convolution.size() + gated_delta_recurrent.size()) * sizeof(float);
        }
        if (gated_delta_device_state)
        {
            return device_allocated_bytes;
        }
        const uint64_t element_size = dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
        return token_count * columns * element_size * 2;
    }
};

class CpuSessionState
{
public:
    explicit CpuSessionState(const ExecutionGraph& graph)
        : memory_manager(graph)
    {
    }

    std::vector<CpuLayerCache> layers;
    std::vector<CpuLayerCache> speculative_layers;
    std::vector<LayerGraphState> execution_layers;
    MemoryManager memory_manager;
    CpuExpertExecutionScratch expert_scratch;
    CpuAttentionExecutionScratch attention_scratch;
    CpuGatedDeltaExecutionScratch gated_delta_scratch;
    std::unique_ptr<CpuTaskWorker> router_prediction_worker;
    CpuBatch hidden;
    CpuBatch speculative_main_hidden;
    CpuBatch mtp_pending_target_hidden;
    std::vector<int32_t> speculative_input_ids;
    std::vector<int32_t> speculative_direct_alignment_ids;
    uint64_t speculative_main_hidden_position = 0;
    uint64_t mtp_pending_target_position = 0;
    bool speculative_context_enabled = true;

    [[nodiscard]] uint64_t kv_cache_allocated_bytes() const noexcept
    {
        uint64_t bytes = 0;
        for (const CpuLayerCache& layer : layers)
            bytes += layer.allocated_bytes();
        return bytes;
    }

    [[nodiscard]] uint64_t kv_cache_logical_bytes() const noexcept
    {
        uint64_t bytes = 0;
        for (const CpuLayerCache& layer : layers)
            bytes += layer.logical_bytes();
        return bytes;
    }
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_SESSION_STATE_H
