#ifndef NCNN_MOE_VULKAN_H
#define NCNN_MOE_VULKAN_H

#include "ncnn/moe/types.h"

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

struct VulkanStatistics
{
    uint64_t dispatches = 0;
    uint64_t attention_blocks = 0;
    uint64_t compute_submissions = 0;
    uint64_t batch_uploads = 0;
    uint64_t batch_downloads = 0;
    uint64_t command_buffer_reuses = 0;
    uint64_t command_graph_submissions = 0;
    uint64_t command_graph_operations = 0;
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
