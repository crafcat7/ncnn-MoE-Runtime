#ifndef NCNN_MOE_LINEAR_QNK_H
#define NCNN_MOE_LINEAR_QNK_H

#include "linear_core.h"

namespace ncnn {
namespace moe {

class QnkLinear_vulkan
{
public:
    ~QnkLinear_vulkan();

    [[nodiscard]] static uint64_t storage_bytes(const TensorData& matrix, uint64_t alignment) noexcept;
    // Standalone projections may losslessly unpack low-bit codes at load time.
    // Expert storage retains its native block layout and uses storage_bytes().
    [[nodiscard]] static uint64_t projection_storage_bytes(const TensorData& matrix, uint64_t alignment) noexcept;
    [[nodiscard]] static bool can_fuse(std::span<const TensorData* const> matrices,
                                       uint64_t alignment = 256, uint64_t storage_limit = UINT64_C(0x7fffffff)) noexcept;
    [[nodiscard]] static std::shared_ptr<QnkLinear_vulkan> create_fused(std::span<const TensorData* const> matrices,
                                                                        std::span<const TensorData* const> biases,
                                                                        uint32_t vulkan_device_index,
                                                                        const VulkanRuntimePtr& vulkan_runtime,
                                                                        uint64_t optimization_flags);

    [[nodiscard]] static std::shared_ptr<QnkLinear_vulkan> create(const TensorData& matrix,
                                                                  const TensorData* bias,
                                                                  uint32_t vulkan_device_index,
                                                                  const VulkanRuntimePtr& vulkan_runtime,
                                                                  uint64_t optimization_flags,
                                                                  uint32_t input_group_count = 1);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;
    [[nodiscard]] uint64_t device_bytes() const noexcept;

private:
    friend class GpuSegment_vulkan;
    friend class GpuSegmentContinuation_vulkan;
    friend class QnkExpert_vulkan;
    friend class GatedDeltaNet_vulkan;
    friend class HyperConnection_vulkan;
    friend class LayerHead_vulkan;
    friend class Attention_vulkan;
    friend class QsaAttention_vulkan;
    friend class GatedResidualLayer_vulkan;
    friend class GatedResidual_vulkan;

    friend class LatentAttention_vulkan;
    [[nodiscard]] bool record_forward(const ncnn::VkMat& input, ncnn::VkMat& output,
                                      ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace) const;
    void record(const ncnn::VkMat& input, ncnn::VkMat& output, ncnn::VkCompute& command, uint32_t output_column_offset = 0) const;
    void record(const ncnn::VkMat& input,
                ncnn::VkMat& output,
                ncnn::VkCompute& command,
                uint32_t output_column_offset,
                std::vector<ncnn::VkMat>& bindings,
                std::vector<ncnn::vk_constant_type>& constants) const;
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;

    [[nodiscard]] static std::shared_ptr<QnkLinear_vulkan> create_with_allocator(const TensorData& matrix,
                                                                                 const TensorData* bias,
                                                                                 uint32_t vulkan_device_index,
                                                                                 ncnn::VkAllocator* weight_allocator,
                                                                                 ncnn::VkAllocator* staging_allocator,
                                                                                 const VulkanRuntimePtr& vulkan_runtime,
                                                                                 uint64_t optimization_flags,
                                                                                 uint32_t input_group_count = 1,
                                                                                 const TensorData* extra = nullptr,
                                                                                 VulkanWeightUploadBatch* upload_batch = nullptr);
    class Implementation;

    QnkLinear_vulkan();
    std::unique_ptr<Implementation> d;
};

// Vulkan native block-quantized/affine INT4 Expert chain.
class QnkExpert_vulkan
{
public:
    ~QnkExpert_vulkan();

    [[nodiscard]] static std::shared_ptr<QnkExpert_vulkan> create(const TensorData& gate_up,
                                                                  const TensorData* gate_up_bias,
                                                                  const TensorData& down,
                                                                  const TensorData* down_bias,
                                                                  float activation_limit,
                                                                  uint32_t vulkan_device_index,
                                                                  ExpertActivation activation,
                                                                  const VulkanRuntimePtr& vulkan_runtime,
                                                                  uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<QnkExpert_vulkan> create_separate(const TensorData& gate,
                                                                           const TensorData* gate_bias,
                                                                           const TensorData& up,
                                                                           const TensorData* up_bias,
                                                                           const TensorData& down,
                                                                           const TensorData* down_bias,
                                                                           float activation_limit,
                                                                           uint32_t vulkan_device_index,
                                                                           ExpertActivation activation,
                                                                           const VulkanRuntimePtr& vulkan_runtime,
                                                                           uint64_t optimization_flags);
    [[nodiscard]] static std::vector<uint64_t> scratch_elements(uint32_t intermediate_columns,
                                                                uint32_t output_columns,
                                                                bool separate_up,
                                                                bool router_gate,
                                                                uint64_t rows);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] bool forward_device(const DeviceTensor_vulkan& input, DeviceTensor_vulkan& output) const;
    [[nodiscard]] bool prepare_router_gate(const TensorData& weight);
    [[nodiscard]] bool has_router_gate() const noexcept;
    [[nodiscard]] bool materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output) const;
    [[nodiscard]] uint64_t device_bytes() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;
    [[nodiscard]] uint32_t int4_operations() const noexcept;
    [[nodiscard]] uint32_t record_dispatches() const noexcept;
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    // Caller holds the context lock and retains all operands until completion.
    [[nodiscard]] bool record(const ncnn::VkMat& input, ncnn::VkMat& output,
                              ncnn::VkCompute& command,
                              std::vector<ncnn::VkMat>& workspace,
                              std::vector<ncnn::VkMat>& bindings,
                              std::vector<ncnn::vk_constant_type>& constants) const;

private:
    friend class VulkanExpertBackend;

    [[nodiscard]] static std::shared_ptr<QnkExpert_vulkan> create_impl(const TensorData& gate,
                                                                       const TensorData* gate_bias,
                                                                       const TensorData* up,
                                                                       const TensorData* up_bias,
                                                                       const TensorData& down,
                                                                       const TensorData* down_bias,
                                                                       float activation_limit,
                                                                       uint32_t vulkan_device_index,
                                                                       ncnn::VkAllocator* weight_allocator,
                                                                       ncnn::VkAllocator* staging_allocator,
                                                                       ExpertActivation activation,
                                                                       const VulkanRuntimePtr& vulkan_runtime,
                                                                       uint64_t optimization_flags,
                                                                       VulkanWeightUploadBatch* upload_batch = nullptr);
    [[nodiscard]] static bool forward_batch(std::span<const QnkExpert_vulkan*> experts,
                                            std::span<const ActivationBuffer*> inputs,
                                            std::span<ActivationBuffer*> outputs);
    class Implementation;

    QnkExpert_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif
