#ifndef NCNN_MOE_SESSIONSTATE_H
#define NCNN_MOE_SESSIONSTATE_H

#include "kernels/attention.h"
#include "kernels/gateddeltanet.h"
#include "kernels/hyperconnection.h"
#include "kernels/ops.h"
#include "kernels/statecache.h"
#include "expertbackend.h"
#include "storage/expertcache.h"

#include "graph/router.h"
#include "ncnn/moe/types.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

struct SessionStatistics;

class CpuTaskWorker
{
public:
    explicit CpuTaskWorker(size_t maximum_outstanding_tasks);
    ~CpuTaskWorker();

    CpuTaskWorker(const CpuTaskWorker&) = delete;
    CpuTaskWorker& operator=(const CpuTaskWorker&) = delete;

    [[nodiscard]] bool try_submit(std::function<void()> task);

private:
    void worker_loop();

    const size_t task_limit;
    std::mutex mutex;
    std::condition_variable task_ready;
    std::deque<std::function<void()>> tasks;
    std::thread worker;
    size_t outstanding_tasks = 0;
    bool stop = false;
};

struct CpuDecodeRouteOrigin
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
    CpuBatch input;
    CpuBatch output;
    ExpertCacheLease lease;
    ExpertExecutionMetrics metrics;

    void prepare(const ExpertBatch& next_batch)
    {
        batch.expert_id = next_batch.expert_id;
        batch.routes.assign(next_batch.routes.begin(), next_batch.routes.end());
        lease = {};
        metrics = {};
    }

    void prepare(ExpertBatch&& next_batch)
    {
        batch = std::move(next_batch);
        lease = {};
        metrics = {};
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
    bool backend_aggregated_output_valid = false;
    CpuBatch backend_aggregated_output;
    CpuBatch staged_merged;
    CpuBatch staged_output;
    CpuBatch staged_router_logits;
    std::vector<int32_t> staged_input_ids;
    std::vector<uint32_t> staged_expert_ids;
    std::vector<CpuGatedDeltaBatchEntry> gated_delta_entries;
    std::vector<NcnnVulkanGatedDeltaBatchEntry> gated_delta_device_entries;
    std::vector<uint64_t> staged_attention_positions;
    std::vector<CpuLayerCache*> staged_attention_caches;
    std::vector<CpuAttentionBatchEntry> attention_batch_entries;
    std::vector<CpuBatch> staged_batches;
    std::vector<CpuHyperConnectionMix> staged_hyper_mixes;
    std::vector<size_t> combined_by_expert;
    std::vector<std::vector<CpuDecodeRouteOrigin>> staged_route_origins;
    std::vector<uint8_t> combined_backend_aggregated;
    bool combined_backend_aggregated_output_valid = false;
    CpuBatch combined_backend_aggregated_output;
};

class CpuSessionState
{
public:
    std::vector<CpuLayerCache> layers;
    std::vector<CpuLayerCache> speculative_layers;
    std::vector<LayerGraphState> execution_layers;
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
    bool use_speculative_context = true;

    [[nodiscard]] uint64_t kv_cache_allocated_size() const noexcept
    {
        uint64_t bytes = 0;
        for (const CpuLayerCache& layer : layers)
            bytes += layer.allocated_bytes();
        return bytes;
    }

    [[nodiscard]] uint64_t kv_cache_logical_size() const noexcept
    {
        uint64_t bytes = 0;
        for (const CpuLayerCache& layer : layers)
            bytes += layer.logical_bytes();
        return bytes;
    }
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SESSIONSTATE_H
