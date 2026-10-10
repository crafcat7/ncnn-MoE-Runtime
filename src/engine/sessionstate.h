#ifndef NCNN_MOE_SESSIONSTATE_H
#define NCNN_MOE_SESSIONSTATE_H

#include "backends/ncnn/latentlayer_vulkan.h"
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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

struct SessionStatistics;

struct ExpertExecutionMetrics
{
    uint64_t cache_wait_time_microseconds = 0;
};

struct ExpertWorkspace
{
    ActivationBuffer projection;
    ActivationBuffer gate;
    ActivationBuffer quantized_input;
};

struct ExpertState
{
    ExpertBatch batch;
    ActivationBuffer input;
    ActivationBuffer output;
    std::shared_ptr<DeviceTensor_vulkan> device_output;
    ExpertCacheLease lease;
    ExpertExecutionMetrics metrics;

    void prepare(ExpertBatch& next_batch)
    {
        // Consume next_batch.routes and return the old buffer for dispatch reuse.
        batch.expert_id = next_batch.expert_id;
        batch.routes.swap(next_batch.routes);
        next_batch.routes.clear();
        lease = {};
        device_output.reset();
        metrics = {};
    }
};

struct LayerState
{
    ActivationBuffer normalized;
    std::shared_ptr<const DeviceTensor_vulkan> normalized_device;
    ActivationBuffer router_logits;
    HyperConnectionMix ffn_hyper_mix;
    ActivationBuffer shared_expert_output;
    std::shared_ptr<const DeviceTensor_vulkan> shared_expert_device_output;
    ExpertWorkspace shared_expert_workspace;
    ExpertDispatchPlan dispatch_plan;
    // Keep inactive slots too, so changing route counts does not free buffers.
    std::vector<ExpertState> expert_slots;
    size_t active_expert_count = 0;
    bool experts_executed = false;

    void resize_experts(size_t count)
    {
        if (expert_slots.size() < count)
            expert_slots.resize(count);
        active_expert_count = count;
    }

    std::span<ExpertState> active_experts() noexcept
    {
        return {expert_slots.data(), active_expert_count};
    }

    std::span<const ExpertState> active_experts() const noexcept
    {
        return {expert_slots.data(), active_expert_count};
    }

    void reset()
    {
        // Router overwrites normalized/router_logits scratch before reuse.
        // Empty shared output means Shared Expert has not run for this pass.
        shared_expert_output.clear();
        shared_expert_device_output.reset();
        normalized_device.reset();
        for (ExpertState& active : expert_slots)
        {
            active.lease = {};
            active.device_output.reset();
        }
        experts_executed = false;
    }
};

struct ExpertScratch
{
    std::vector<ExpertWorkspace> expert_workspaces;
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
    // Final arrays are consumed synchronously before the next wait. Pending
    // submissions keep their own reservations and never borrow this storage.
    std::vector<ExpertBackendExecutionResult> backend_results;
    std::vector<size_t> failed_indices;
    // Separate wave storage stays stable while asynchronous submissions own
    // request spans. Clear owners after execution, retain vector capacity.
    std::vector<size_t> prestaged_indices;
    std::vector<ExpertDemandRequest> prestaged_demands;
    std::vector<std::shared_ptr<const void>> prestaged_pins;
    std::vector<size_t> prestaged_candidates;
    std::vector<ExpertCachePairRequest> prestaged_cache_requests;
    std::vector<ExpertCacheLease> prestaged_cache_leases;
    std::vector<size_t> current_wave_indices;
    std::vector<ExpertBackendRequest> current_wave_requests;
    std::vector<size_t> next_wave_indices;
    std::vector<ExpertBackendRequest> next_wave_requests;
    std::vector<size_t> wave_failed_indices;
    std::vector<size_t> wave_cpu_indices;
    std::vector<ExpertDemandRequest> wave_demands;
    std::vector<size_t> wave_demand_indices;
    std::vector<std::shared_ptr<const void>> prepared_pins;
    bool backend_aggregated_output_valid = false;
    ActivationBuffer backend_aggregated_output;
    ActivationBuffer staged_merged;
    ActivationBuffer staged_output;
    std::vector<uint32_t> explicit_expert_ids;
};

// Storage owned by one scheduler worker for one in-flight staged batch.  Shared
// staged-batch scratch is reused here instead of borrowing a SessionState.
struct BatchWorkspace
{
    ExpertScratch expert;
    LatentLayerWorkspace_vulkan latent_layer;
    ExpertDispatchPlan latent_routes;
    // Scratch that belongs to the in-flight staged batch rather than a
    // SessionState or an individual Expert execution.
    LayerState staged_state;
    ActivationBuffer staged_router_logits;
    std::vector<int32_t> staged_input_ids;
    std::vector<GatedDeltaBatchEntry> gated_delta_entries;
    std::vector<GatedDeltaBatchEntry_vulkan> gated_delta_device_entries;
    std::vector<uint64_t> staged_attention_positions;
    std::vector<LayerCache*> staged_attention_caches;
    std::vector<AttentionBatchEntry> attention_batch_entries;
    std::vector<size_t> combined_by_expert;
    AttentionScratch attention;
    HyperConnectionScratch hyper_connection;
};

class SessionState
{
public:
    std::vector<LayerCache> layers;
    std::vector<LayerCache> speculative_layers;
    // The schedule finishes each layer's Combine before starting the next layer.
    LayerState execution_state;
    ExpertScratch expert_scratch;
    HyperConnectionScratch hyper_connection_scratch;
    AttentionScratch attention_scratch;
    LatentLayerWorkspace_vulkan latent_layer;
    std::vector<uint64_t> attention_positions;
    std::vector<LayerCache*> attention_caches;
    GatedDeltaScratch gated_delta_scratch;
    std::unique_ptr<CpuTaskWorker> router_prediction_worker;
    ActivationBuffer hidden;
    ActivationBuffer final_norm;
    // Reusable single-session LM-head storage.  Keep this separate from the
    // staged Expert scratch used by graph and batch execution so a logits row
    // remains valid until the next model execution.
    ActivationBuffer logits;
    ActivationBuffer lm_head_input;
    ActivationBuffer speculative_main_hidden;
    ActivationBuffer mtp_pending_target_hidden;
    std::vector<int32_t> speculative_input_ids;
    std::vector<int32_t> speculative_direct_alignment_ids;
    uint64_t speculative_main_hidden_position = 0;
    uint64_t mtp_pending_target_position = 0;
    bool use_speculative_context = true;
    // Conservative marker armed before execution and cleared after commit/reset.
    bool execution_failed = false;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SESSIONSTATE_H
