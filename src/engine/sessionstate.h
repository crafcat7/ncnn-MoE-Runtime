#ifndef NCNN_MOE_SESSIONSTATE_H
#define NCNN_MOE_SESSIONSTATE_H

#include "kernels/attention.h"
#include "kernels/gateddeltanet.h"
#include "kernels/hyperconnection.h"
#include "kernels/ops.h"
#include "kernels/statecache.h"
#include "cpu.h"
#include "expertbackend.h"
#include "storage/expertcache.h"

#include "graph/router.h"
#include "ncnn/moe/types.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

struct SessionStatistics;

struct DecodeRouteOrigin
{
    size_t session_index = 0;
    size_t active_index = 0;
    size_t route_index = 0;
};

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
    ActivationBuffer input;
    ActivationBuffer output;
    ExpertCacheLease lease;
    ExpertExecutionMetrics metrics;

    void prepare(ExpertBatch& next_batch)
    {
        // Consume next_batch.routes and return the old buffer for dispatch reuse.
        batch.expert_id = next_batch.expert_id;
        batch.routes.swap(next_batch.routes);
        next_batch.routes.clear();
        lease = {};
        metrics = {};
    }
};

struct LayerGraphState
{
    ActivationBuffer normalized;
    ActivationBuffer router_logits;
    HyperConnectionMix ffn_hyper_mix;
    ActivationBuffer shared_expert_output;
    ExpertDispatchPlan dispatch_plan;
    // Keep inactive slots too, so changing route counts does not free buffers.
    std::vector<ActiveExpertExecution> expert_slots;
    size_t active_expert_count = 0;
    std::chrono::steady_clock::time_point router_start;
    std::chrono::steady_clock::time_point expert_start;
    bool experts_executed = false;

    void resize_experts(size_t count)
    {
        if (expert_slots.size() < count)
            expert_slots.resize(count);
        active_expert_count = count;
    }

    std::span<ActiveExpertExecution> active_experts() noexcept
    {
        return {expert_slots.data(), active_expert_count};
    }

    std::span<const ActiveExpertExecution> active_experts() const noexcept
    {
        return {expert_slots.data(), active_expert_count};
    }

    void reset()
    {
        // Router overwrites normalized/router_logits scratch before reuse.
        // Empty shared output means Shared Expert has not run for this pass.
        shared_expert_output.clear();
        for (ActiveExpertExecution& active : expert_slots)
            active.lease = {};
        experts_executed = false;
    }
};

struct ExpertScratch
{
    // Staged execution gathers several sessions while their FFN state stays live.
    LayerGraphState staged_state;
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
    bool backend_aggregated_output_valid = false;
    ActivationBuffer backend_aggregated_output;
    ActivationBuffer staged_merged;
    ActivationBuffer staged_output;
    ActivationBuffer staged_router_logits;
    std::vector<int32_t> staged_input_ids;
    std::vector<uint32_t> explicit_expert_ids;
    std::vector<GatedDeltaBatchEntry> gated_delta_entries;
    std::vector<GatedDeltaBatchEntry_vulkan> gated_delta_device_entries;
    std::vector<uint64_t> staged_attention_positions;
    std::vector<LayerCache*> staged_attention_caches;
    std::vector<AttentionBatchEntry> attention_batch_entries;
    std::vector<ActivationBuffer> staged_batches;
    std::vector<size_t> combined_by_expert;
    std::vector<std::vector<DecodeRouteOrigin>> staged_route_origins;
    std::vector<uint8_t> combined_backend_aggregated;
    ActivationBuffer combined_backend_aggregated_output;
};

class SessionState
{
public:
    std::vector<LayerCache> layers;
    std::vector<LayerCache> speculative_layers;
    // The schedule finishes each layer's Combine before starting the next layer.
    LayerGraphState execution_state;
    ExpertScratch expert_scratch;
    HyperConnectionScratch hyper_connection_scratch;
    AttentionScratch attention_scratch;
    GatedDeltaScratch gated_delta_scratch;
    std::unique_ptr<CpuTaskWorker> router_prediction_worker;
    ActivationBuffer hidden;
    ActivationBuffer final_norm;
    ActivationBuffer speculative_main_hidden;
    ActivationBuffer mtp_pending_target_hidden;
    std::vector<int32_t> speculative_input_ids;
    std::vector<int32_t> speculative_direct_alignment_ids;
    uint64_t speculative_main_hidden_position = 0;
    uint64_t mtp_pending_target_position = 0;
    bool use_speculative_context = true;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SESSIONSTATE_H
