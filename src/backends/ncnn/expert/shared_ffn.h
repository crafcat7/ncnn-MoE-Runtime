#ifndef NCNN_MOE_EXPERT_SHARED_FFN_H
#define NCNN_MOE_EXPERT_SHARED_FFN_H

#include "backends/ncnn/linear/core.h"
#include "backends/ncnn/vulkan.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ncnn {
class VkMat;
class VkCompute;

namespace moe {

struct CompiledModel;
class CompiledOperatorTable;
struct MoeBlockPlan;
struct ExpertPlan;
class VulkanContext;

[[nodiscard]] bool support_vulkan_shared_experts(const CompiledOperatorTable& operators,
                                                 const MoeBlockPlan& moe) noexcept;
[[nodiscard]] bool support_vulkan_ffn(const CompiledOperatorTable& operators,
                                      const ExpertPlan& ffn) noexcept;

// Adapter over already prepared shared/dense FFN operators; never uploads weights.
class SharedFfn_vulkan
{
public:
    ~SharedFfn_vulkan();

    [[nodiscard]] static std::shared_ptr<SharedFfn_vulkan> create(const CompiledModel& model,
                                                                  const MoeBlockPlan& moe);
    [[nodiscard]] static std::shared_ptr<SharedFfn_vulkan> create(const CompiledModel& model,
                                                                  const ExpertPlan& ffn);
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] bool use_cm2() const noexcept;
    // Caller holds the context lock and retains operands until completion. Never submits.
    [[nodiscard]] bool record(const ncnn::VkMat& normalized, ncnn::VkMat& output,
                              ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace,
                              FfnRecordStatistics_vulkan& statistics,
                              const ncnn::VkMat* res = nullptr,
                              bool* res_fused = nullptr) const;
#endif

private:
    class Implementation;

    SharedFfn_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_SHARED_FFN_H
