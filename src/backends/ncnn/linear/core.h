#ifndef NCNN_MOE_LINEAR_CORE_H
#define NCNN_MOE_LINEAR_CORE_H

#include "backends/ncnn/core/tensor.h"
#include "backends/ncnn/vulkan.h"
#include "kernels/activationbuffer.h"
#include "ncnn/moe/option.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
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
class VkWeightBatch;
class GatedDeltaNet_vulkan;
class SharedFfn_vulkan;

// Recorded work is counted by the command owner only after successful completion.
struct FfnRecordStatistics_vulkan
{
    uint64_t dispatches = 0;
    uint64_t int4_dispatches = 0;
    uint64_t shared_expert_swiglu_fusions = 0;
};

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

    // Records a device projection; caller holds context lock and retains operands until completion.
    [[nodiscard]] int forward(const ncnn::VkMat& input, ncnn::VkMat& output,
                              ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

private:
    friend class CommandGraph_vulkan;
    friend class Attention_vulkan;
    friend class Router_vulkan;
    friend class LatentAttention_vulkan;
    friend class LayerHead_vulkan;
    friend class GpuSegment_vulkan;
    friend class GpuSegmentContinuation_vulkan;

    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;

    class Implementation;

    Linear();
    std::unique_ptr<Implementation> d;
};

class CommandGraph_vulkan
{
public:
    ~CommandGraph_vulkan();

    CommandGraph_vulkan(const CommandGraph_vulkan&) = delete;
    CommandGraph_vulkan& operator=(const CommandGraph_vulkan&) = delete;
    CommandGraph_vulkan(CommandGraph_vulkan&&) noexcept;
    CommandGraph_vulkan& operator=(CommandGraph_vulkan&&) noexcept;

    // Creates a graph using seed's Vulkan context; returns null on CPU-only operators.
    [[nodiscard]] static std::unique_ptr<CommandGraph_vulkan> create(const Linear& seed);

    [[nodiscard]] bool upload(const ActivationBuffer& input, DeviceTensor_vulkan& output);
    [[nodiscard]] bool linear(const Linear& op, const DeviceTensor_vulkan& input, DeviceTensor_vulkan& output);
    [[nodiscard]] bool download(const DeviceTensor_vulkan& input, ActivationBuffer& output);
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
