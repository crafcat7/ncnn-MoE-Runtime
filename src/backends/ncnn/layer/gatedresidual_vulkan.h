#ifndef NCNN_MOE_GATEDRESIDUAL_VULKAN_H
#define NCNN_MOE_GATEDRESIDUAL_VULKAN_H

#include "backends/ncnn/vulkan.h"
#include "graph/layerplan.h"
#include "ncnn/moe/modeldescriptor.h"

#include <initializer_list>
#include <memory>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <mat.h>
#endif

namespace ncnn {
class Pipeline;
class VkCompute;
class Option;

namespace moe {

class WeightStore;
class VulkanContext;
struct TensorData;

class GatedResidual_vulkan
{
public:
    ~GatedResidual_vulkan();

    [[nodiscard]] static std::shared_ptr<GatedResidual_vulkan> create(const WeightStore& weights,
                                                                      const GatedResidualPlan& plan,
                                                                      const MoeModelDescriptor& descriptor,
                                                                      uint32_t device,
                                                                      const VulkanRuntimePtr& runtime,
                                                                      uint64_t flags);
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    [[nodiscard]] bool record_pre(const ncnn::VkMat& input, ncnn::VkMat& reduced,
                                  ncnn::VkMat& injection, ncnn::VkCompute& cmd,
                                  std::vector<ncnn::VkMat>& storage) const;
    [[nodiscard]] bool record_post(const ncnn::VkMat& branch, const ncnn::VkMat& residual,
                                   const ncnn::VkMat& injection, ncnn::VkMat& output,
                                   ncnn::VkCompute& cmd) const;
#endif

private:
    class Implementation;

    GatedResidual_vulkan();
    std::unique_ptr<Implementation> d;
};

#if NCNN_MOE_WITH_VULKAN
std::shared_ptr<ncnn::Pipeline> residual_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                  const ncnn::Option& opt);
void residual_record(const ncnn::Pipeline* pipeline, uint32_t mode, uint32_t hidden, uint32_t copies,
                     uint32_t rows, uint32_t columns, float epsilon, float offset,
                     std::initializer_list<ncnn::VkMat> values, ncnn::VkCompute& cmd,
                     uint32_t kernel = 0, uint32_t ngram = 0, uint32_t length = 0, uint32_t slot = 0);
bool residual_allocate(ncnn::VkMat& value, uint32_t columns, int rows,
                       const std::shared_ptr<VulkanContext>& context);
bool residual_float_weight(const TensorData& tensor, ncnn::VkMat& value,
                           ncnn::VkCompute& command, const ncnn::Option& opt);
#endif

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GATEDRESIDUAL_VULKAN_H
