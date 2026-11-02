#ifndef NCNN_MOE_VULKAN_H
#define NCNN_MOE_VULKAN_H

#include "ncnn/moe/types.h"
#include "ncnn/moe/result.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

// Opaque ownership token for one model/runtime Vulkan resource domain.
// Operators created with the same token share command resources; different
// tokens never share mutable backend state.
class VulkanRuntime;
using VulkanRuntimePtr = std::shared_ptr<VulkanRuntime>;

struct VulkanGraphArenaSize
{
    uint64_t preparation_physical_size = 0;
    uint64_t total_physical_size = 0;
};

// GraphResources retains this opaque Context owner before preparing operators.
// Keeping it outside VulkanRuntime avoids a runtime/Context ownership cycle.
[[nodiscard]] std::shared_ptr<void> retain_vulkan_graph_context(const VulkanRuntimePtr& vulkan_runtime,
                                                                uint32_t device_index,
                                                                uint64_t optimization_flags) noexcept;

// Reserve the already prepared model Context's stable blob/workspace allocator.
// Physical sizes include Vulkan memory requirements, rather than logical tensor
// sizes. Preparation is snapshotted by the first reservation; compute is the
// additional backing retained afterwards. A failed reservation never selects CPU.
[[nodiscard]] Result<VulkanGraphArenaSize> reserve_vulkan_graph_compute(const VulkanRuntimePtr& vulkan_runtime,
                                                                        uint32_t device_index,
                                                                        uint64_t optimization_flags,
                                                                        uint64_t compute_bytes);
[[nodiscard]] VulkanGraphArenaSize query_vulkan_graph_compute(const VulkanRuntimePtr& vulkan_runtime,
                                                              uint32_t device_index,
                                                              uint64_t optimization_flags) noexcept;

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
    uint64_t attention_cache_materializations = 0;
    uint64_t attention_cpu_fallbacks = 0;
    uint64_t shared_expert_swiglu_fusions = 0;
    uint64_t gated_delta_fusions = 0;
    uint64_t gated_delta_submissions = 0;
    uint64_t rms_norm_linear_fusions = 0;
    uint64_t kv_ring_appends = 0;
    uint64_t kv_ring_resizes = 0;
    uint64_t kv_ring_wrapped_views = 0;
    uint64_t bfloat16_cooperative_matrix_dispatches = 0;
};

[[nodiscard]] uint32_t get_gpu_count() noexcept;
[[nodiscard]] uint32_t get_default_gpu_index() noexcept;
[[nodiscard]] std::vector<GpuInfo> get_gpu_infos();
[[nodiscard]] VulkanStatistics get_vulkan_statistics(const VulkanRuntimePtr& vulkan_runtime) noexcept;

[[nodiscard]] VulkanRuntimePtr
create_vulkan_runtime();

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_VULKAN_H
