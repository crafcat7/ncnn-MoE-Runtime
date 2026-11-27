#ifndef NCNN_MOE_RMSNORM_VULKAN_H
#define NCNN_MOE_RMSNORM_VULKAN_H

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

class RmsNorm_vulkan
{
public:
    ~RmsNorm_vulkan();

    // weight_allocator, when set, must outlive the operator. Output storage
    // always comes from the context's graph allocator.
    [[nodiscard]] static std::shared_ptr<RmsNorm_vulkan> create(const TensorData& weight,
                                                                float epsilon,
                                                                float weight_offset,
                                                                std::shared_ptr<VulkanContext> context,
                                                                const ncnn::Option& opt,
                                                                ncnn::VkAllocator* weight_allocator = nullptr,
                                                                bool add_norm = false);
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
#endif
    // add_norm returns input+branch as output and its normalized form as
    // residual; input, output and residual must be distinct mats. Callers
    // retain every output until the submission completes.
    [[nodiscard]] bool record(const ncnn::VkMat& input, ncnn::VkMat& output,
                              ncnn::VkCompute& cmd,
                              const ncnn::VkMat* branch = nullptr,
                              ncnn::VkMat* residual = nullptr,
                              bool bf16_out = false) const;

private:
    class Implementation;
    RmsNorm_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RMSNORM_VULKAN_H
