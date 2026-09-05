#ifndef NCNN_MOE_NCNN_MODELPIPELINE_H
#define NCNN_MOE_NCNN_MODELPIPELINE_H

#include "ncnn/moe/result.h"
#include "storage/weightstore.h"
#include "vulkancontext.h"

#include <cstdint>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
struct CompiledOperator;
struct MoeBlockPlan;
struct CompiledModel;
struct CompiledLayerPlan;
struct AttentionBlockPlan;
struct AttentionDescriptor;
struct MoeModelDescriptor;
enum class NcnnLinearDevice;

Result<void> prepare_linear_operator(
    WeightStore& weights,
    CompiledOperatorTable& operators,
    TensorHandle matrix_handle,
    TensorHandle bias_handle,
    NcnnLinearDevice device,
    bool retain_cpu_dense_copy,
    uint32_t vulkan_device_index,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags,
    uint32_t input_group_count = 1,
    bool prefer_bfloat16_vulkan = true);

Result<void> prepare_lm_head_operator(
    CompiledModel& compiled,
    NcnnLinearDevice dense_device,
    bool retain_cpu_dense_copies);

void release_vulkan_dense_host_copies(CompiledModel& compiled);

Result<void> prepare_shared_expert_operators(
    WeightStore& weights,
    CompiledOperatorTable& operators,
    MoeBlockPlan& moe,
    NcnnLinearDevice device,
    bool retain_cpu_dense_copy,
    uint32_t vulkan_device_index,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags);

[[nodiscard]] bool support_vulkan_experts(
    const WeightStore& weights,
    const MoeBlockPlan& moe,
    uint64_t optimization_flags) noexcept;

[[nodiscard]] bool support_vulkan_shared_experts(
    const CompiledOperatorTable& operators,
    const MoeBlockPlan& moe) noexcept;

bool uses_vulkan_dense_operator(const CompiledOperator& executable) noexcept;

[[nodiscard]] bool support_vulkan_attention(
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& attention) noexcept;

uint64_t gated_delta_vulkan_working_set_size(
    const AttentionDescriptor& attention,
    const MoeModelDescriptor& descriptor) noexcept;

Result<void> prepare_vulkan_qkv_operator(
    CompiledModel& compiled,
    CompiledLayerPlan& layer_plan,
    bool use_bfloat16_fusion,
    const char* failure_message);

Result<void> prepare_latent_attention_operators(
    CompiledModel& compiled,
    CompiledLayerPlan& layer_plan,
    const AttentionDescriptor& attention,
    NcnnLinearDevice attention_device,
    bool retain_cpu_dense_copies,
    const char* diagnostic_prefix = "");

Result<void> prepare_gated_delta_attention_operators(
    CompiledModel& compiled,
    CompiledLayerPlan& layer_plan,
    NcnnLinearDevice attention_device,
    bool retain_cpu_dense_copies);

Result<void> prepare_standard_attention_operators(
    CompiledModel& compiled,
    CompiledLayerPlan& layer_plan,
    NcnnLinearDevice attention_device,
    bool retain_cpu_dense_copies);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_MODELPIPELINE_H
