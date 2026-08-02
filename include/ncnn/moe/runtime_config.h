#ifndef NCNN_MOE_RUNTIME_CONFIG_H
#define NCNN_MOE_RUNTIME_CONFIG_H

#include "ncnn/moe/memory_plan.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

// RuntimeOptionFlag belongs with RuntimeConfig so applications can include
// the reusable configuration contract without pulling in Runtime itself.
#define NCNN_MOE_RUNTIME_MMAP_EXPERT_BIT         0
#define NCNN_MOE_RUNTIME_DIRECT_IO_BIT           1
#define NCNN_MOE_RUNTIME_BUFFERED_IO_BIT         2
#define NCNN_MOE_RUNTIME_DISABLE_VICTIM_EXEC_BIT 3
#define NCNN_MOE_RUNTIME_ROUTER_PRED_BIT         4
#define NCNN_MOE_RUNTIME_FORWARD_ARC_BIT        5
#define NCNN_MOE_RUNTIME_RANK_ADAPT_BIT         6
#define NCNN_MOE_RUNTIME_READ_MERGE_BIT         7
#define NCNN_MOE_RUNTIME_ASYNC_ROUTER_PRED_BIT  8
#define NCNN_MOE_RUNTIME_RELEASE_DENSE_BIT      9

enum RuntimeOptionFlag : uint32_t
{
    RuntimeOptionMemoryMapExperts = UINT32_C(1) << NCNN_MOE_RUNTIME_MMAP_EXPERT_BIT,
    RuntimeOptionDirectExpertIo = UINT32_C(1) << NCNN_MOE_RUNTIME_DIRECT_IO_BIT,
    RuntimeOptionBufferedExpertIo = UINT32_C(1) << NCNN_MOE_RUNTIME_BUFFERED_IO_BIT,
    RuntimeOptionDisableGpuVictimExecution = UINT32_C(1) << NCNN_MOE_RUNTIME_DISABLE_VICTIM_EXEC_BIT,
    RuntimeOptionRouterPrediction = UINT32_C(1) << NCNN_MOE_RUNTIME_ROUTER_PRED_BIT,
    RuntimeOptionForwardAwareCache = UINT32_C(1) << NCNN_MOE_RUNTIME_FORWARD_ARC_BIT,
    RuntimeOptionRankAdaptivePrefetch = UINT32_C(1) << NCNN_MOE_RUNTIME_RANK_ADAPT_BIT,
    RuntimeOptionCrossExpertReadCoalescing = UINT32_C(1) << NCNN_MOE_RUNTIME_READ_MERGE_BIT,
    RuntimeOptionAsyncRouterPrediction = UINT32_C(1) << NCNN_MOE_RUNTIME_ASYNC_ROUTER_PRED_BIT,
    RuntimeOptionReleaseVulkanDenseHostStorage = UINT32_C(1) << NCNN_MOE_RUNTIME_RELEASE_DENSE_BIT
};

// User-supplied runtime configuration. Zero-valued memory settings and Auto
// enum values leave hardware- and model-specific decisions to Runtime.
// Tokenizer, prompt, chat, and sampling state deliberately do not belong here.
struct RuntimeConfig
{
    HybridMode hybrid_mode = HybridMode::Auto;
    ExpertMemoryMode expert_memory_mode = ExpertMemoryMode::Auto;
    uint64_t host_memory_budget_bytes = 0;
    uint64_t expert_cache_bytes = 0;
    uint64_t expert_gpu_cache_bytes = 0;
    uint64_t expert_gpu_victim_cache_bytes = 0;
    uint32_t expert_gpu_victim_reuse_probe_interval = 1;
    uint32_t expert_io_workers = 0;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    uint32_t expected_concurrency = 1;
    std::vector<uint32_t> vulkan_device_indices;
    uint32_t flags = 0;
};

// Source-compatible name for existing native runners and clients.
using RuntimeOptions = RuntimeConfig;

// The concrete settings selected after RuntimeConfig::Auto has been resolved
// and the model has been compiled.  This is intentionally separate from
// RuntimeConfig so applications can show the effective plan without having
// to duplicate the runtime's hardware and memory decisions.
struct EffectiveRuntimeConfig
{
    HybridMode hybrid_mode = HybridMode::CpuOnly;
    ExpertMemoryMode requested_expert_memory_mode = ExpertMemoryMode::Auto;
    ExpertMemoryMode selected_expert_memory_mode = ExpertMemoryMode::Eager;
    uint64_t host_memory_budget_bytes = 0;
    uint64_t expert_cache_bytes = 0;
    uint64_t expert_gpu_cache_bytes = 0;
    uint64_t expert_gpu_victim_cache_bytes = 0;
    uint32_t expert_gpu_victim_reuse_probe_interval = 1;
    uint32_t expert_io_workers = 0;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    std::vector<uint32_t> vulkan_device_indices;
    uint32_t flags = 0;
    uint32_t expected_concurrency = 1;
    bool file_backed_experts = false;
};

// Source-compatible name for clients using the original terminology.
using EffectiveRuntimeOptions = EffectiveRuntimeConfig;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RUNTIME_CONFIG_H
