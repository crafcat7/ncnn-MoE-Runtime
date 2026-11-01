#ifndef NCNN_MOE_RESIDENTFFN_VULKAN_H
#define NCNN_MOE_RESIDENTFFN_VULKAN_H

#include "vulkan.h"
#include "kernels/activationbuffer.h"
#include "graph/router.h"

#include <memory>
#include <span>

#if NCNN_MOE_WITH_VULKAN
#include <mat.h>
#endif

namespace ncnn {
class Option;
class VkCompute;
namespace moe {
class VulkanContext;
class MoeCombine_vulkan;
class DeviceTensor_vulkan;

class ResidentFfnContinuation_vulkan
{
public:
    virtual ~ResidentFfnContinuation_vulkan() = default;
    [[nodiscard]] virtual const std::shared_ptr<VulkanContext>& context() const noexcept = 0;
#if NCNN_MOE_WITH_VULKAN
    // The wave owns command submission and retains combined until its fence.
    // Preparation happens before recording; this method must not submit or lock
    // the context command mutex, which the caller already holds.
    [[nodiscard]] virtual bool record(const ncnn::VkMat& combined, ncnn::VkCompute& command,
                                      const ncnn::Option& option) = 0;
#endif
    // These hooks must not lock the command mutex or publish Session state.
    // abort is idempotent and leaves an already published transaction intact.
    virtual void mark_submit_attempted() noexcept = 0;
    virtual void mark_completed() noexcept = 0;
    virtual void abort() noexcept = 0;
};

enum class ResidentFfnOutput_vulkan
{
    Host,
    Device
};

// One routed FFN invocation, independent of Attention kind and model family.
// Expert waves share one normalized upload and retain their device results.
// The last wave reduces in canonical Expert order. Host output downloads only
// the token-major aggregate; Device output retains it without a host copy.
// No method submits or waits; recording belongs to the existing wave command.
class ResidentFfnRegion_vulkan
{
public:
    ~ResidentFfnRegion_vulkan();
    [[nodiscard]] static std::shared_ptr<ResidentFfnRegion_vulkan> create(const ActivationBuffer& normalized,
                                                                          size_t expert_count,
                                                                          uint32_t output_columns);
    [[nodiscard]] static std::shared_ptr<ResidentFfnRegion_vulkan> create(std::shared_ptr<const DeviceTensor_vulkan> normalized,
                                                                          size_t expert_count, uint32_t output_columns,
                                                                          ResidentFfnOutput_vulkan output = ResidentFfnOutput_vulkan::Device);
    [[nodiscard]] bool take_output(ActivationBuffer& output);
    [[nodiscard]] bool take_device_output(std::shared_ptr<const DeviceTensor_vulkan>& output);
    [[nodiscard]] bool complete() const noexcept;
    [[nodiscard]] size_t rows() const noexcept;
    [[nodiscard]] uint32_t columns() const noexcept;
    [[nodiscard]] size_t expert_count() const noexcept;
    // Recording does not publish counters. Only a completed final wave exposes
    // its actual Combine dispatch count to the submission owner.
    [[nodiscard]] uint64_t pending_combine_dispatches() const noexcept;
    [[nodiscard]] uint64_t completed_combine_dispatches() const noexcept;
    // Attach before any wave starts. Only a device-output region on the same
    // context can continue into another prepared device segment.
    [[nodiscard]] bool set_continuation(std::shared_ptr<ResidentFfnContinuation_vulkan> continuation);
    void mark_submit_attempted() noexcept;

#if NCNN_MOE_WITH_VULKAN
    struct Output
    {
        size_t index = 0;
        ncnn::VkMat value;
        std::span<const ExpertRoute> routes;
    };
    // Called outside the command lock because shader creation takes it.
    [[nodiscard]] bool bind_context(std::shared_ptr<VulkanContext> context,
                                    std::shared_ptr<MoeCombine_vulkan> combiner);
    // Caller holds the context command lock and retains this region through
    // the wave wait. Only complete_wave commits a recorded input/output set.
    [[nodiscard]] bool record_input(ncnn::VkMat& input, ncnn::VkCompute& command,
                                    const ncnn::Option& option, bool& uploaded);
    [[nodiscard]] bool record_outputs(std::span<const Output> outputs, ncnn::VkCompute& command,
                                      const ncnn::Option& option, bool& downloaded);
    [[nodiscard]] bool complete_wave();
    void abort_wave() noexcept;
#endif

private:
    ResidentFfnRegion_vulkan();
    class Implementation;
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn
#endif // NCNN_MOE_RESIDENTFFN_VULKAN_H
