#ifndef NCNN_MOE_SHAREDFFN_VULKAN_H
#define NCNN_MOE_SHAREDFFN_VULKAN_H

#include "vulkan.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace ncnn {
class VkMat;
class VkCompute;
namespace moe {
struct CompiledModel;
struct MoeBlockPlan;
struct ExpertPlan;
class VulkanContext;

// Recorded work is counted by the command owner only after successful
// completion. Recording or an abandoned command changes no runtime counters.
struct FfnRecordStatistics_vulkan
{
    uint64_t dispatches = 0;
    uint64_t int4_dispatches = 0;
    uint64_t shared_expert_swiglu_fusions = 0;
    uint64_t bfloat16_cooperative_matrix_dispatches = 0;
};

// A view of already prepared Shared Expert products. The adapter owns the
// products it records; creating it never uploads or prepares model weights.
class SharedFfn_vulkan
{
public:
    ~SharedFfn_vulkan();
    [[nodiscard]] static std::shared_ptr<SharedFfn_vulkan> create(const CompiledModel& model,
                                                                  const MoeBlockPlan& moe);
    [[nodiscard]] static std::shared_ptr<SharedFfn_vulkan> create_dense(const CompiledModel& model, const ExpertPlan& ffn);
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
#if NCNN_MOE_WITH_VULKAN
    // Caller holds the context lock and retains input/output/workspace and
    // this adapter until completion or command reset. Never submits or waits.
    [[nodiscard]] bool record(const ncnn::VkMat& normalized, ncnn::VkMat& output,
                              ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace,
                              FfnRecordStatistics_vulkan& statistics) const;
#endif
private:
    class Implementation;
    SharedFfn_vulkan();
    std::unique_ptr<Implementation> d;
};
} // namespace moe
} // namespace ncnn
#endif // NCNN_MOE_SHAREDFFN_VULKAN_H
