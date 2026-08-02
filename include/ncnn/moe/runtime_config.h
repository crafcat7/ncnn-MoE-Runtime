#ifndef NCNN_MOE_RUNTIME_CONFIG_H
#define NCNN_MOE_RUNTIME_CONFIG_H

#include "ncnn/moe/memory_plan.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

// The concrete settings selected after RuntimeOptions::Auto has been resolved
// and the model has been compiled.  This is intentionally separate from
// RuntimeOptions so applications can show the effective plan without having
// to duplicate the runtime's hardware and memory decisions.
struct EffectiveRuntimeOptions
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

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RUNTIME_CONFIG_H
