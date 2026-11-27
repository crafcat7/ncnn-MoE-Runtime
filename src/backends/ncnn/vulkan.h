#ifndef NCNN_MOE_VULKAN_H
#define NCNN_MOE_VULKAN_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

inline uint64_t saturating_add_u64(uint64_t left, uint64_t right) noexcept
{
    return right > std::numeric_limits<uint64_t>::max() - left
               ? std::numeric_limits<uint64_t>::max()
               : left + right;
}

inline uint64_t saturating_multiply_u64(uint64_t left, uint64_t right) noexcept
{
    return left != 0 && right > std::numeric_limits<uint64_t>::max() / left
               ? std::numeric_limits<uint64_t>::max()
               : left * right;
}

// Opaque ownership token for one model/runtime Vulkan resource domain.
class VulkanRuntime;
using VulkanRuntimePtr = std::shared_ptr<VulkanRuntime>;

struct VulkanGraphArenaSize
{
    uint64_t preparation_physical_size = 0;
    uint64_t total_physical_size = 0;
};

// Keeps the model VulkanContext alive outside VulkanRuntime to avoid a cycle.
[[nodiscard]] std::shared_ptr<void> retain_vulkan_graph_context(const VulkanRuntimePtr& runtime,
                                                                uint32_t device_index,
                                                                uint64_t opt_flags) noexcept;

[[nodiscard]] Result<VulkanGraphArenaSize> reserve_vulkan_graph_compute(const VulkanRuntimePtr& runtime,
                                                                        uint32_t device_index,
                                                                        uint64_t opt_flags,
                                                                        uint64_t compute_bytes);

[[nodiscard]] VulkanGraphArenaSize query_vulkan_graph_compute(const VulkanRuntimePtr& runtime,
                                                              uint32_t device_index,
                                                              uint64_t opt_flags) noexcept;

struct VulkanStatistics
{
    uint64_t dispatches = 0;
    uint64_t int4_dispatches = 0;
    uint64_t attention_blocks = 0;
    uint64_t compute_submissions = 0;
    uint64_t batch_uploads = 0;
    uint64_t batch_downloads = 0;
    uint64_t weight_upload_bytes = 0;
    uint64_t weight_transfer_submissions = 0;
    uint64_t weight_visibility_submissions = 0;
    uint64_t weight_transfer_fence_waits = 0;
    uint64_t weight_transfer_wait_microseconds = 0;
    uint64_t resident_graph_submissions = 0;
    uint64_t resident_graph_activation_upload_bytes = 0;
    uint64_t resident_graph_activation_download_bytes = 0;
    uint64_t resident_graph_route_download_bytes = 0;
    uint64_t attention_qkv_rope_fusions = 0;
    uint64_t attention_device_rope_fusions = 0;
    uint64_t attention_qkv_ring_fusions = 0;
    uint64_t attention_decode_sdpa_fusions = 0;
    uint64_t attention_cpu_fallbacks = 0;
    uint64_t shared_expert_swiglu_fusions = 0;
    uint64_t gated_delta_fusions = 0;
    uint64_t gated_delta_submissions = 0;
    uint64_t rms_norm_linear_fusions = 0;
    uint64_t kv_ring_appends = 0;
    uint64_t kv_ring_resizes = 0;
    uint64_t kv_ring_wrapped_views = 0;
};

[[nodiscard]] uint32_t get_gpu_count() noexcept;
[[nodiscard]] uint32_t get_default_gpu_index() noexcept;
[[nodiscard]] std::vector<GpuInfo> get_gpu_infos();
[[nodiscard]] VulkanStatistics get_vulkan_statistics(const VulkanRuntimePtr& runtime) noexcept;
[[nodiscard]] VulkanRuntimePtr create_vulkan_runtime();

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_VULKAN_H
