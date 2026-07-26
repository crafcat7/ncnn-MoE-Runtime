#ifndef NCNN_MOE_CPU_SESSION_STATE_H
#define NCNN_MOE_CPU_SESSION_STATE_H

#include "kernels/cpu_ops.h"
#include "engine/expert_backend.h"
#include "storage/expert_cache.h"

#include "ncnn/moe/expert_dispatcher.h"
#include "ncnn/moe/memory_manager.h"
#include "ncnn/moe/types.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

class NcnnVulkanAttentionCache;

struct ExpertExecutionMetrics
{
    uint64_t hinted_bytes = 0;
    uint64_t cache_wait_time_microseconds = 0;
    uint64_t regroup_time_microseconds = 0;
    uint64_t mxfp4_decode_gemv_rows = 0;
    uint64_t mxfp4_prefill_gemm_rows = 0;
    uint64_t mxfp4_paired_rows = 0;
    uint64_t mxfp4_fused_gate_up_rows = 0;
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
    ExpertDispatchPlan dispatch_plan;
    std::vector<ActiveExpertExecution> active_experts;
    std::chrono::steady_clock::time_point router_start;
    std::chrono::steady_clock::time_point expert_start;
    bool experts_executed = false;

    void reset()
    {
        normalized.clear();
        router_logits.clear();
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
    std::vector<size_t> backend_indices;
    std::vector<ExpertBackendRequest> backend_requests;
    std::vector<size_t> failed_indices;
    std::vector<int8_t> prediction_states;
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
    CpuBatch projected;
    CpuBatch output;
    std::vector<float> logits;
};

struct CpuLayerCache
{
    std::vector<float> keys;
    std::vector<float> values;
    std::vector<uint16_t> bfloat16_keys;
    std::vector<uint16_t> bfloat16_values;
    uint64_t start_position = 0;
    uint64_t token_count = 0;
    uint64_t first_slot = 0;
    uint64_t capacity_tokens = 0;
    uint32_t columns = 0;
    DType dtype = DType::Float32;
    std::shared_ptr<NcnnVulkanAttentionCache> vulkan_attention_cache;
    std::vector<uint32_t> previous_expert_ids;
    uint64_t device_allocated_bytes = 0;

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        return static_cast<uint64_t>(keys.size() + values.size()) * sizeof(float) + static_cast<uint64_t>(bfloat16_keys.size() + bfloat16_values.size()) * sizeof(uint16_t) + device_allocated_bytes;
    }

    [[nodiscard]] uint64_t logical_bytes() const noexcept
    {
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
    std::vector<LayerGraphState> execution_layers;
    MemoryManager memory_manager;
    CpuExpertExecutionScratch expert_scratch;
    CpuAttentionExecutionScratch attention_scratch;
    CpuBatch hidden;

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
