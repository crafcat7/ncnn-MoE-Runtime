#ifndef NCNN_MOE_RMSNORM_VULKAN_H
#define NCNN_MOE_RMSNORM_VULKAN_H

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

class RmsNorm_vulkan
{
public:
    ~RmsNorm_vulkan();
    // An optional weight allocator must outlive the operator. Output storage
    // continues to use the Context's graph allocator.
    [[nodiscard]] static std::shared_ptr<RmsNorm_vulkan> create(const TensorData& weight,
                                                                float epsilon,
                                                                float weight_offset,
                                                                std::shared_ptr<VulkanContext> context,
                                                                const ncnn::Option& opt,
                                                                ncnn::VkAllocator* weight_allocator = nullptr,
                                                                bool add_norm = false);
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
#endif
    // The caller holds the context lock, which protects record scratch, and
    // retains outputs until submission completes.
    // Input and output variables must differ; an output view sharing input
    // storage receives a distinct allocation and never overwrites input.
    // Add+norm mode requires branch and residual; it returns input+branch and
    // its normalized form as separate outputs retained by the caller.
    [[nodiscard]] bool record(const ncnn::VkMat& input, ncnn::VkMat& output,
                              ncnn::VkCompute& cmd,
                              const ncnn::VkMat* branch = nullptr,
                              ncnn::VkMat* residual = nullptr) const;

private:
    class Implementation;
    RmsNorm_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RMSNORM_VULKAN_H
