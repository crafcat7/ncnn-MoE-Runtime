#ifndef NCNN_MOE_EXPERT_FFN_REGION_H
#define NCNN_MOE_EXPERT_FFN_REGION_H

#include "backends/ncnn/vulkan.h"
#include "graph/router.h"
#include "kernels/activationbuffer.h"

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
    // Caller holds the context command lock and retains combined until the fence settles.
    [[nodiscard]] virtual bool record(const ncnn::VkMat& combined, ncnn::VkCompute& command,
                                      const ncnn::Option& option) = 0;
#endif
    // Must not lock command_mutex or publish session state; abort is idempotent.
    virtual void mark_submit_attempted() noexcept = 0;
    virtual void mark_completed() noexcept = 0;
    virtual void abort() noexcept = 0;
};

enum class ResidentFfnOutput_vulkan
{
    Host,
    Device
};

// Coordinates one routed FFN wave group across expert batches, combining
// results in canonical expert order on completion.
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
    [[nodiscard]] uint64_t pending_combine_dispatches() const noexcept;
    [[nodiscard]] uint64_t completed_combine_dispatches() const noexcept;
    [[nodiscard]] bool set_continuation(std::shared_ptr<ResidentFfnContinuation_vulkan> continuation);
    void mark_submit_attempted() noexcept;

#if NCNN_MOE_WITH_VULKAN
    struct Output
    {
        size_t index = 0;
        ncnn::VkMat value;
        std::span<const ExpertRoute> routes;
    };

    // Call outside command_mutex because pipeline creation acquires it.
    [[nodiscard]] bool bind_context(std::shared_ptr<VulkanContext> context,
                                    std::shared_ptr<MoeCombine_vulkan> combiner);
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

#endif // NCNN_MOE_EXPERT_FFN_REGION_H
