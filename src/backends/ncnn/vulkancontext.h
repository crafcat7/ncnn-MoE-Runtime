#ifndef NCNN_MOE_NCNN_VULKANCONTEXT_H
#define NCNN_MOE_NCNN_VULKANCONTEXT_H

#include "ncnn/moe/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

// Opaque ownership token for one model/runtime Vulkan resource domain.
// Operators created with the same token share command resources; different
// tokens never share mutable backend state.
class NcnnVulkanContextInstance;
using NcnnVulkanContextInstancePtr = std::shared_ptr<NcnnVulkanContextInstance>;

struct NcnnVulkanRuntimeCounters
{
    uint64_t compute_submissions = 0;
    uint64_t submit_wait_time_microseconds = 0;
    uint64_t batch_uploads = 0;
    uint64_t batch_downloads = 0;
    uint64_t auxiliary_uploads = 0;
    uint64_t auxiliary_upload_bytes = 0;
    uint64_t staging_slot_resizes = 0;
    uint64_t staging_slot_reuses = 0;
    uint64_t staging_slot_acquisitions = 0;
    uint64_t staging_slot_contentions = 0;
    uint64_t command_buffer_reuses = 0;
    uint64_t command_graph_submissions = 0;
    uint64_t command_graph_operations = 0;
    uint64_t direct_host_input_bindings = 0;
    uint64_t direct_host_output_bindings = 0;
    uint64_t attention_qkv_rope_fusions = 0;
    uint64_t attention_device_rope_fusions = 0;
    uint64_t attention_qkv_ring_fusions = 0;
    uint64_t attention_qkv_rope_pipeline_failures = 0;
    uint64_t attention_qkv_rope_shape_failures = 0;
    uint64_t attention_qkv_rope_source_failures = 0;
    uint64_t attention_qkv_rope_norm_failures = 0;
    uint64_t attention_qkv_rope_ring_failures = 0;
    uint64_t attention_qkv_rope_allocation_failures = 0;
    uint64_t attention_precondition_failures = 0;
    uint64_t attention_staging_failures = 0;
    uint64_t attention_norm_failures = 0;
    uint64_t attention_qkv_failures = 0;
    uint64_t attention_cache_failures = 0;
    uint64_t attention_sdpa_failures = 0;
    uint64_t attention_projection_failures = 0;
    uint64_t attention_output_failures = 0;
    uint64_t attention_submit_failures = 0;
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
    uint64_t kv_cache_promotions = 0;
    uint64_t kv_cache_promotion_bytes = 0;
    uint64_t bfloat16_cooperative_matrix_dispatches = 0;
    uint64_t command_dispatches = 0;
    uint64_t command_pipeline_binds = 0;
    uint64_t command_redundant_pipeline_binds = 0;
    uint64_t command_descriptor_bindings = 0;
    uint64_t command_push_constant_updates = 0;
    uint64_t command_resource_barrier_calls = 0;
    uint64_t command_buffer_resource_barriers = 0;
    uint64_t command_image_resource_barriers = 0;
};

struct NcnnVulkanExecutionSnapshot
{
    uint64_t dispatches = 0;
    uint64_t attention_blocks = 0;
    NcnnVulkanRuntimeCounters counters;
};

[[nodiscard]] uint32_t get_gpu_count() noexcept;
[[nodiscard]] uint32_t get_default_gpu_index() noexcept;
[[nodiscard]] std::vector<GpuInfo> get_gpu_infos();
[[nodiscard]] NcnnVulkanExecutionSnapshot get_vulkan_execution_snapshot(
    const NcnnVulkanContextInstancePtr& context_instance) noexcept;

[[nodiscard]] NcnnVulkanContextInstancePtr
create_ncnn_vulkan_context_instance();

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_VULKANCONTEXT_H
