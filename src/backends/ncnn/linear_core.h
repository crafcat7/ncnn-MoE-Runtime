#ifndef NCNN_MOE_LINEAR_CORE_H
#define NCNN_MOE_LINEAR_CORE_H

#include "kernels/activationbuffer.h"

#include "ncnn/moe/types.h"
#include "ncnn/moe/option.h"
#include "vulkan.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
class VkAllocator;
class VkCompute;
class VkMat;
class Option;
union vk_constant_type;

namespace moe {

class VulkanContext;
class VulkanWeightUploadBatch;
class GatedDeltaNet_vulkan;
class SharedFfn_vulkan;
struct FfnRecordStatistics_vulkan;

class Linear
{
public:
    ~Linear();

    [[nodiscard]] static std::shared_ptr<Linear> create(const TensorData& matrix, const TensorData* bias,
                                                        uint32_t vulkan_device_index,
                                                        const VulkanRuntimePtr& vulkan_runtime,
                                                        uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<Linear> create_fused(const std::vector<const TensorData*>& matrices,
                                                              const std::vector<const TensorData*>& biases,
                                                              uint32_t vulkan_device_index,
                                                              const VulkanRuntimePtr& vulkan_runtime,
                                                              uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    // Records a device projection without submitting or publishing counters.
    // The caller holds the context lock and retains inputs/outputs until completion.
    [[nodiscard]] int forward(const ncnn::VkMat& input, ncnn::VkMat& output, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

private:
    friend class CommandGraph_vulkan;
    friend class Attention_vulkan;
    friend class Router_vulkan;
    friend class LatentAttention_vulkan;
    friend class LayerHead_vulkan;
    friend class GpuSegment_vulkan;
    friend class GpuSegmentContinuation_vulkan;

    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;

    class Implementation;

    Linear();
    std::unique_ptr<Implementation> d;
};

// A small graph-level bridge for ncnn Vulkan Linear operators. Tensors stay
// in Vulkan storage until the caller explicitly requests a download.
class DeviceTensor_vulkan
{
public:
    DeviceTensor_vulkan();
    ~DeviceTensor_vulkan();

    DeviceTensor_vulkan(DeviceTensor_vulkan&&) noexcept;
    DeviceTensor_vulkan& operator=(DeviceTensor_vulkan&&) noexcept;
    DeviceTensor_vulkan(const DeviceTensor_vulkan&) = delete;
    DeviceTensor_vulkan& operator=(const DeviceTensor_vulkan&) = delete;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t rows() const noexcept;
    [[nodiscard]] uint32_t columns() const noexcept;

private:
    class Implementation;
    std::unique_ptr<Implementation> d;

    friend class CommandGraph_vulkan;
    friend class VulkanExpertBackend;
    friend class LatentLayer_vulkan;
    friend class MoeCombine_vulkan;
    friend class LatentLayerWorkspace_vulkan;
    friend class Float8Linear_vulkan;
    friend class Bfloat16Linear_vulkan;
    friend class QnkExpert_vulkan;
    friend class LatentAttention_vulkan;
    friend class LayerHead_vulkan;
    friend class QsaAttention_vulkan;
    friend class GatedResidualLayer_vulkan;
    friend class GatedResidual_vulkan;
    friend class GpuSegment_vulkan;
    friend class GpuSegmentContinuation_vulkan;
    friend class GpuSegmentWorkspace_vulkan;
    friend class ResidentFfnRegion_vulkan;

    // Wraps a completed unpacked FP32 device result without materializing it.
    [[nodiscard]] bool assign_completed(const ncnn::VkMat& value,
                                        std::shared_ptr<VulkanContext> context);
    [[nodiscard]] const ncnn::VkMat* value() const noexcept;
    // Re-arms a completed import; the caller holds the Vulkan context lock.
    void prepare_read() const noexcept;
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
};

class CommandGraph_vulkan
{
public:
    ~CommandGraph_vulkan();

    CommandGraph_vulkan(const CommandGraph_vulkan&) = delete;
    CommandGraph_vulkan& operator=(const CommandGraph_vulkan&) = delete;
    CommandGraph_vulkan(CommandGraph_vulkan&&) noexcept;
    CommandGraph_vulkan& operator=(CommandGraph_vulkan&&) noexcept;

    // Creates a graph using the Vulkan context and activation storage policy
    // of seed.  Returns null for CPU-only operators/devices.
    [[nodiscard]] static std::unique_ptr<CommandGraph_vulkan> create(const Linear& seed);

    [[nodiscard]] bool upload(const ActivationBuffer& input,
                              DeviceTensor_vulkan& output);
    [[nodiscard]] bool linear(const Linear& op,
                              const DeviceTensor_vulkan& input,
                              DeviceTensor_vulkan& output);
    [[nodiscard]] bool download(const DeviceTensor_vulkan& input,
                                ActivationBuffer& output);
    [[nodiscard]] bool submit();
    [[nodiscard]] bool wait();

private:
    class Implementation;
    std::unique_ptr<Implementation> d;

    explicit CommandGraph_vulkan(std::unique_ptr<Implementation> implementation);
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LINEAR_CORE_H
