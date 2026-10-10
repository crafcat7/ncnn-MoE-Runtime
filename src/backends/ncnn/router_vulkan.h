#ifndef NCNN_MOE_ROUTER_VULKAN_H
#define NCNN_MOE_ROUTER_VULKAN_H

#include "graph/router.h"
#include "kernels/activationbuffer.h"
#include "vulkancontext.h"

#include <memory>

namespace ncnn {
namespace moe {

struct RouterWorkspace_vulkan
{
#if NCNN_MOE_WITH_VULKAN
    // Keep the allocator owner before its buffers; retain through submission.
    std::shared_ptr<VulkanContext> context;
    ncnn::VkMat projected;
    ncnn::VkMat logits;
    ncnn::VkMat selection_bias;
    ncnn::VkMat selection_bias_staging;
    ncnn::VkMat explicit_ids;
    ncnn::VkMat explicit_ids_staging;
    ncnn::VkMat selected;
#endif
};

class Router_vulkan
{
public:
    ~Router_vulkan();

    [[nodiscard]] static std::shared_ptr<Router_vulkan>
    create(const TensorData& weight, const TensorData* bias,
           const TensorData* selection_bias, uint32_t device_index,
           const VulkanRuntimePtr& runtime, uint64_t optimization_flags);

    [[nodiscard]] static uint32_t selected_columns(uint32_t top_k) noexcept;
    // Rows contain K ids, K weights, then a validation status. Only this
    // compact result needs to cross to the host ExpertStore scheduler.
    [[nodiscard]] static Result<void>
    decode_selected(const ActivationBuffer& selected,
                    const ExpertDispatchOptions& options,
                    ExpertDispatchPlan& result);

#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    // Input is unpacked FP32 and already normalized. The caller owns the
    // context command lock and keeps workspace alive until submission ends.
    [[nodiscard]] bool record(const ncnn::VkMat& input,
                              const ExpertDispatchOptions& options,
                              RouterWorkspace_vulkan& workspace,
                              ncnn::VkCompute& cmd) const;
#endif

private:
    class Implementation;
    Router_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_ROUTER_VULKAN_H
