#ifndef NCNN_MOE_NET_MEMORY_H
#define NCNN_MOE_NET_MEMORY_H

#include "backends/ncnn/layer/attention_vulkan.h"
#include "graph/memoryplan.h"
#include "graph/resourceplan.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <string>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
struct AttentionBlockPlan;
struct CompiledLayerPlan;
struct CompiledModel;
struct CompiledOperator;
struct CompilerOption;
struct TensorData;

// One model-wide ledger shared by persistent preparation and Expert caching.
// Shape-specific operators supply requests; policy is independent of family.
class ResourcePlanner
{
public:
    ResourcePlanner(CompiledModel& model, const CompilerOption& opt);

    [[nodiscard]] bool can_admit(uint32_t device, GpuResourceRequest request) const;
    [[nodiscard]] bool allocation_fits(uint32_t device, GpuResourceRequest request) const;
    [[nodiscard]] bool explicit_cache_reserve(uint32_t device) const;
    void refresh_expert_reserve(uint32_t device);
    void commit(uint32_t device, GpuResourceRequest request);

private:
    [[nodiscard]] const GpuResourcePlan* find(uint32_t device) const;

    CompiledModel& model;
    std::map<uint32_t, uint64_t> usage_baselines;
    std::map<uint32_t, uint64_t> explicit_cache_reserves;
};

[[nodiscard]] uint64_t gpu_buffer_alignment(const CompiledModel& compiled,
                                            uint32_t device_index);

[[nodiscard]] bool bf16_projection_supported(const CompiledModel& compiled,
                                             const TensorData& matrix,
                                             uint64_t alignment,
                                             uint32_t device_index);

[[nodiscard]] uint64_t bf16_fused_storage_size(const CompiledModel& compiled,
                                               std::initializer_list<TensorHandle> handles,
                                               uint64_t alignment,
                                               uint32_t device_index) noexcept;

[[nodiscard]] uint64_t projection_storage(const CompiledModel& model,
                                          TensorHandle weight,
                                          uint64_t alignment,
                                          uint32_t device,
                                          TensorHandle bias = invalid_tensor_handle);

[[nodiscard]] uint64_t constant_storage(const CompiledModel& model,
                                        std::initializer_list<TensorHandle> handles,
                                        uint64_t alignment);

[[nodiscard]] bool uses_vulkan_dense_operator(const CompiledOperator& executable) noexcept;

[[nodiscard]] bool attention_vulkan_supported(const CompiledOperatorTable& operators,
                                              const AttentionBlockPlan& attention) noexcept;

[[nodiscard]] bool attn_bf16_fusion(const CompiledModel& compiled,
                                    const AttentionBlockPlan& attention,
                                    uint64_t alignment,
                                    uint32_t device_index);

[[nodiscard]] bool attn_needs_native(const CompiledModel& compiled,
                                     const AttentionBlockPlan& attention,
                                     uint32_t device_index);

[[nodiscard]] bool attn_native(const CompiledModel& compiled,
                               const AttentionBlockPlan& attention) noexcept;

[[nodiscard]] AttentionConfig_vulkan attn_vk_config(const CompiledModel& compiled,
                                                    const AttentionBlockPlan& plan,
                                                    uint32_t device_index);

[[nodiscard]] uint64_t minimum_gpu_expert_reserve(const CompiledModel& compiled,
                                                  uint32_t device_index);

[[nodiscard]] uint64_t expert_wave_workspace_size(const CompiledModel& model,
                                                  uint32_t device);

[[nodiscard]] GpuResourceRequest gated_residual_resources(const CompiledModel& compiled,
                                                          uint32_t device_index) noexcept;

[[nodiscard]] Result<GpuResourceRequest> attention_resource_request(const CompiledModel& model,
                                                                    const CompiledLayerPlan& layer);

[[nodiscard]] GpuResourceRequest ffn_resource_request(const CompiledModel& model,
                                                      const CompiledLayerPlan& layer,
                                                      bool shared);

// Backend products are published as a complete group. A failed creation or a
// driver-budget change must preserve both the CPU weights and earlier groups.
[[nodiscard]] Result<bool> prepare_resource_group(CompiledModel& model,
                                                  ResourcePlanner& resources,
                                                  uint32_t device,
                                                  GpuResourceRequest request,
                                                  CompiledLayerPlan* layer,
                                                  const std::string& operation,
                                                  const std::function<Result<void>()>& prepare,
                                                  const std::function<bool()>& supported,
                                                  bool allow_budget_fallback = false);

[[nodiscard]] uint64_t attn_vk_size(const CompiledModel& compiled,
                                    const CompiledLayerPlan& layer,
                                    uint64_t alignment,
                                    uint32_t device_index = automatic_vulkan_device_index) noexcept;

[[nodiscard]] uint64_t dense_weight_size(const CompiledModel& compiled,
                                         const TensorData& matrix,
                                         uint64_t alignment,
                                         uint32_t device_index = automatic_vulkan_device_index,
                                         bool has_bias = false) noexcept;

[[nodiscard]] uint64_t dense_pending_size(const CompiledModel& compiled,
                                          uint32_t device_index) noexcept;

// The common resource plan's unallocated per-Session state and peak execution
// workspace for this device. Shape estimates use the default prefill chunk;
// larger chunks/contexts remain checked when their buffers are allocated.
[[nodiscard]] uint64_t vulkan_pending_reserve(const CompiledModel& compiled,
                                              uint32_t device_index) noexcept;

// Measure the already scheduled placement for representative prefill and
// decode execution. Persistent KV/recurrent state is reported separately from
// reusable graph buffers. Zero context_tokens uses prefill_rows.
[[nodiscard]] Result<ModelGraphMemoryPlan> plan_model_graph_memory(const CompiledModel& compiled,
                                                                   uint32_t prefill_rows,
                                                                   uint32_t sequence_count = 1,
                                                                   uint32_t context_tokens = 0,
                                                                   uint32_t submission_rows_override = 0);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NET_MEMORY_H
