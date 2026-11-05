#ifndef NCNN_MOE_LINEAR_MXFP4_H
#define NCNN_MOE_LINEAR_MXFP4_H

#include "linear_core.h"

namespace ncnn {
namespace moe {

class Mxfp4Linear_vulkan
{
public:
    ~Mxfp4Linear_vulkan();

    [[nodiscard]] static std::shared_ptr<Mxfp4Linear_vulkan> create(const TensorData& matrix, const TensorData* bias,
                                                                    uint32_t vulkan_device_index,
                                                                    const VulkanRuntimePtr& vulkan_runtime,
                                                                    uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;

private:
    friend class Mxfp4Expert_vulkan;
    friend class VulkanExpertBackend;

    [[nodiscard]] static std::shared_ptr<Mxfp4Linear_vulkan> create_with_allocator(const TensorData& matrix, const TensorData* bias,
                                                                                   uint32_t vulkan_device_index, ncnn::VkAllocator* weight_allocator,
                                                                                   const VulkanRuntimePtr& vulkan_runtime,
                                                                                   uint64_t optimization_flags,
                                                                                   VulkanWeightUploadBatch* upload_batch = nullptr);
    class Implementation;

    Mxfp4Linear_vulkan();
    std::unique_ptr<Implementation> d;
};

// Vulkan native block-quantized projection with CPU fallback.

struct Mxfp4DeviceMatrixView_vulkan
{
    uint32_t output_columns = 0;
    uint32_t input_columns = 0;
    uint64_t packed_size = 0;
    uint64_t scales_size = 0;
    size_t packed_offset = 0;
    size_t scales_offset = 0;
};

class Mxfp4Expert_vulkan
{
public:
    ~Mxfp4Expert_vulkan();

    [[nodiscard]] static std::shared_ptr<Mxfp4Expert_vulkan> create(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                    const TensorData& down, const TensorData* down_bias, float activation_limit,
                                                                    uint32_t vulkan_device_index,
                                                                    ExpertActivation activation,
                                                                    const VulkanRuntimePtr& vulkan_runtime,
                                                                    uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<Mxfp4Expert_vulkan> create_from_device_storage(const Mxfp4DeviceMatrixView_vulkan& gate_up, const TensorData* gate_up_bias, const Mxfp4DeviceMatrixView_vulkan& down,
                                                                                        const TensorData* down_bias, float activation_limit, uint32_t vulkan_device_index, const ncnn::VkMat& storage,
                                                                                        ExpertActivation activation, const VulkanRuntimePtr& vulkan_runtime,
                                                                                        uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;

private:
    friend class VulkanExpertBackend;

    // Caller owns the scratch vectors and keeps all bound VkMats alive until submit.
    void record(const ncnn::VkMat& input,
                ncnn::VkMat& intermediate,
                ncnn::VkMat& output,
                ncnn::VkCompute& cmd,
                std::vector<ncnn::VkMat>& bindings,
                std::vector<ncnn::vk_constant_type>& constants) const;

    [[nodiscard]] static std::shared_ptr<Mxfp4Expert_vulkan> create_with_allocator(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                                   const TensorData& down, const TensorData* down_bias,
                                                                                   float activation_limit, uint32_t vulkan_device_index,
                                                                                   ncnn::VkAllocator* weight_allocator,
                                                                                   ExpertActivation activation,
                                                                                   const VulkanRuntimePtr& vulkan_runtime,
                                                                                   uint64_t optimization_flags,
                                                                                   VulkanWeightUploadBatch* upload_batch = nullptr);

    class Implementation;

    Mxfp4Expert_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif
