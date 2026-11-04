#ifndef NCNN_MOE_LAYERNORM_VULKAN_H
#define NCNN_MOE_LAYERNORM_VULKAN_H

#include "vulkan.h"

#include <memory>

namespace ncnn {
class Option;
class VkAllocator;
class VkMat;
class VkCompute;

namespace moe {
class VulkanContext;
struct TensorData;

// LayerNorm recorder backed by ncnn's Vulkan layer.
class LayerNorm_vulkan
{
public:
    ~LayerNorm_vulkan();
    // An optional weight allocator must outlive the operator. Output storage
    // continues to use the Context's graph allocator.
    [[nodiscard]] static std::shared_ptr<LayerNorm_vulkan> create(const TensorData& weight,
                                                                 const TensorData& bias,
                                                                 float epsilon,
                                                                 std::shared_ptr<VulkanContext> context,
                                                                 const ncnn::Option& opt,
                                                                 ncnn::VkAllocator* weight_allocator = nullptr);
    // The caller holds the context lock, which protects record scratch, and
    // retains outputs until submission completes.
    // Input and output variables must differ; an output view sharing input
    // storage receives a distinct allocation and never overwrites input.
    [[nodiscard]] bool record(const ncnn::VkMat& input, ncnn::VkMat& output,
                              ncnn::VkCompute& cmd) const;

private:
    class Implementation;
    LayerNorm_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LAYERNORM_VULKAN_H
