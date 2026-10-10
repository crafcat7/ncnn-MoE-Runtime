#ifndef NCNN_MOE_RMSNORM_VULKAN_H
#define NCNN_MOE_RMSNORM_VULKAN_H

#include "vulkan.h"

#include <memory>

namespace ncnn {
class Option;
class VkMat;
class VkCompute;

namespace moe {
class VulkanContext;
struct TensorData;

class RmsNorm_vulkan
{
public:
    ~RmsNorm_vulkan();
    [[nodiscard]] static std::shared_ptr<RmsNorm_vulkan> create(const TensorData& weight,
                                                                float epsilon,
                                                                float weight_offset,
                                                                std::shared_ptr<VulkanContext> context,
                                                                const ncnn::Option& opt);
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
#endif
    // The caller holds the context lock and retains outputs until submission completes.
    [[nodiscard]] bool record(const ncnn::VkMat& input, ncnn::VkMat& output,
                              ncnn::VkCompute& cmd) const;

private:
    class Implementation;
    RmsNorm_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RMSNORM_VULKAN_H
