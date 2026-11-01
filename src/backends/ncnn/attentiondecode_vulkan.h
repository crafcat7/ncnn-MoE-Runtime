#ifndef NCNN_MOE_ATTENTIONDECODE_VULKAN_H
#define NCNN_MOE_ATTENTIONDECODE_VULKAN_H

#include "attentiondecodeplan.h"
#include "vulkancontext.h"

#if NCNN_MOE_WITH_VULKAN

namespace ncnn {
namespace moe {

[[nodiscard]] AttentionDecodeCapabilities query_attention_decode_capabilities(const ncnn::GpuInfo& info);

// Configure once for the selected matrix shape, then record varying KV lengths.
// Allocation/compilation/recording failures are terminal for this selected path.
class AttentionDecodeVulkan
{
public:
    [[nodiscard]] bool create(const std::shared_ptr<VulkanContext>& context,
                              const ncnn::Option& option,
                              const AttentionDecodePlan& plan);

    // Caller owns command submission and retains workspace through completion.
    [[nodiscard]] bool record(const AttentionDecodePlan& plan,
                              const ncnn::VkMat& query,
                              const ncnn::VkMat& key,
                              const ncnn::VkMat& value,
                              const ncnn::VkMat& sinks,
                              bool attention_sink,
                              ncnn::VkMat& output,
                              ncnn::VkCompute& command,
                              ncnn::VkAllocator* allocator,
                              std::vector<ncnn::VkMat>& workspace) const;

private:
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<ncnn::Pipeline> matrix_pipeline;
    std::shared_ptr<ncnn::Pipeline> reduce_pipeline;
    AttentionDecodePlan pipeline_plan;
};

} // namespace moe
} // namespace ncnn

#endif
#endif
