#ifndef NCNN_MOE_GATEDRESIDUAL_VULKAN_H
#define NCNN_MOE_GATEDRESIDUAL_VULKAN_H

#include "vulkan.h"
#include "graph/layerplan.h"
#include "ncnn/moe/modeldescriptor.h"
#include <memory>
#include <vector>
#if NCNN_MOE_WITH_VULKAN
#include <mat.h>
#endif

namespace ncnn {
class VkCompute;
class Option;
namespace moe {
class WeightStore;
class VulkanContext;

class GatedResidual_vulkan
{
public:
    ~GatedResidual_vulkan();
    [[nodiscard]] static std::shared_ptr<GatedResidual_vulkan> create(const WeightStore& weights,
                                                                      const GatedResidualPlan& plan, const MoeModelDescriptor& descriptor, uint32_t device,
                                                                      const VulkanRuntimePtr& runtime, uint64_t flags);
    [[nodiscard]] uint64_t device_bytes() const noexcept;
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    [[nodiscard]] bool record_pre(const ncnn::VkMat& input, ncnn::VkMat& reduced,
                                  ncnn::VkMat& injection, ncnn::VkCompute& cmd, std::vector<ncnn::VkMat>& storage) const;
    [[nodiscard]] bool record_post(const ncnn::VkMat& branch, const ncnn::VkMat& residual,
                                   const ncnn::VkMat& injection, ncnn::VkMat& output, ncnn::VkCompute& cmd) const;
#endif
private:
    class Implementation;
    GatedResidual_vulkan();
    std::unique_ptr<Implementation> d;
};
} // namespace moe
} // namespace ncnn
#endif
