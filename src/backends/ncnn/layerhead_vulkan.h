#ifndef NCNN_MOE_LAYERHEAD_VULKAN_H
#define NCNN_MOE_LAYERHEAD_VULKAN_H

#include "vulkan.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ncnn {
class VkMat;
class VkCompute;
class Option;
namespace moe {

class VulkanContext;
struct TensorData;
struct CompiledOperator;
struct HyperConnectionWorkspace_vulkan;

class LayerHead_vulkan
{
public:
    ~LayerHead_vulkan();
    [[nodiscard]] static std::shared_ptr<LayerHead_vulkan> create(const TensorData& function,
                                                                  const TensorData& scale,
                                                                  const TensorData& base,
                                                                  const TensorData& norm_weight,
                                                                  const CompiledOperator& lm_head,
                                                                  uint32_t multiplier,
                                                                  float norm_epsilon,
                                                                  float hyper_epsilon,
                                                                  uint32_t vulkan_device_index,
                                                                  const VulkanRuntimePtr& vulkan_runtime,
                                                                  uint64_t optimization_flags);
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;

    // Caller holds the context lock and retains all tensors/workspace until
    // cmd completes. The resident LM projection is shared, never re-uploaded.
    [[nodiscard]] bool record(const ncnn::VkMat& input,
                              ncnn::VkMat& logits,
                              HyperConnectionWorkspace_vulkan& hc_workspace,
                              std::vector<ncnn::VkMat>& temporaries,
                              ncnn::VkCompute& cmd) const;
    // Reuses an already-normalized device view without repeating HC or RMS.
    [[nodiscard]] bool record_logits(const ncnn::VkMat& normalized,
                                     ncnn::VkMat& logits,
                                     std::vector<ncnn::VkMat>& temporaries,
                                     ncnn::VkCompute& cmd) const;
    [[nodiscard]] bool record_hidden(const ncnn::VkMat& input,
                                     ncnn::VkMat& normalized,
                                     HyperConnectionWorkspace_vulkan& hc_workspace,
                                     std::vector<ncnn::VkMat>& temporaries,
                                     ncnn::VkCompute& cmd) const;
#endif
private:
    class Implementation;
    LayerHead_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LAYERHEAD_VULKAN_H
