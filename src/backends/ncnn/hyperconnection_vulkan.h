#ifndef NCNN_MOE_HYPERCONNECTION_VULKAN_H
#define NCNN_MOE_HYPERCONNECTION_VULKAN_H

#include "vulkan.h"
#include "ncnn/moe/types.h"

#if NCNN_MOE_WITH_VULKAN
#include <mat.h>
#endif

#include <cstdint>
#include <memory>

namespace ncnn {
class VkCompute;
class VkMat;
class Option;
namespace moe {

class VulkanContext;

struct HyperConnectionMix_vulkan
{
#if NCNN_MOE_WITH_VULKAN
    // Allocator ownership precedes its buffers and survives operator teardown.
    std::shared_ptr<VulkanContext> context;
    ncnn::VkMat reduced;
    ncnn::VkMat post;
    ncnn::VkMat combine;
#endif
};

struct HyperConnectionWorkspace_vulkan
{
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::VkMat projection;
    ncnn::VkMat pre;
#endif
};

class HyperConnection_vulkan
{
public:
    ~HyperConnection_vulkan();

    // Function rows select either pre/post (M*(M+2)) or head (M).
    [[nodiscard]] static std::shared_ptr<HyperConnection_vulkan> create(const TensorData& function,
                                                                        const TensorData& scale,
                                                                        const TensorData& base,
                                                                        uint32_t multiplier,
                                                                        uint32_t sinkhorn_iterations,
                                                                        float norm_epsilon,
                                                                        float hyper_epsilon,
                                                                        uint32_t vulkan_device_index,
                                                                        const VulkanRuntimePtr& vulkan_runtime,
                                                                        uint64_t optimization_flags);

#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t multiplier() const noexcept;
    [[nodiscard]] bool is_head() const noexcept;

    // All record methods require the context lock, unpacked FP32 input, and
    // live inputs/results/workspace/operator until the caller completes cmd.
    // A workspace may be reused only after its previous command completes.
    // No record method submits, waits, uploads, or downloads.
    [[nodiscard]] bool record_pre(const ncnn::VkMat& input,
                                  HyperConnectionMix_vulkan& result,
                                  HyperConnectionWorkspace_vulkan& workspace,
                                  ncnn::VkCompute& cmd) const;
    [[nodiscard]] bool record_post(const ncnn::VkMat& branch,
                                   const ncnn::VkMat& residual,
                                   const HyperConnectionMix_vulkan& mix,
                                   ncnn::VkMat& output,
                                   ncnn::VkCompute& cmd) const;
    [[nodiscard]] bool record_head(const ncnn::VkMat& input,
                                   ncnn::VkMat& output,
                                   HyperConnectionWorkspace_vulkan& workspace,
                                   ncnn::VkCompute& cmd) const;
#endif

private:
    class Implementation;
    HyperConnection_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_HYPERCONNECTION_VULKAN_H
