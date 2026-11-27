#ifndef NCNN_MOE_LAYERNORM_VULKAN_H
#define NCNN_MOE_LAYERNORM_VULKAN_H

#include "backends/ncnn/vulkan.h"

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

    // weight_allocator, when set, must outlive the operator.
    [[nodiscard]] static std::shared_ptr<LayerNorm_vulkan> create(const TensorData& weight,
                                                                  const TensorData& bias,
                                                                  float epsilon,
                                                                  std::shared_ptr<VulkanContext> context,
                                                                  const ncnn::Option& opt,
                                                                  ncnn::VkAllocator* weight_allocator = nullptr);
    // Input and output must be distinct mats.
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
