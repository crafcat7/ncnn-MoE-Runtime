#ifndef NCNN_MOE_MOECOMBINE_VULKAN_H
#define NCNN_MOE_MOECOMBINE_VULKAN_H

#include "kernels/activationbuffer.h"
#include "graph/router.h"
#include <memory>
#include <span>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <mat.h>
#endif

namespace ncnn {
class Pipeline;
class VkCompute;
namespace moe {
class DeviceTensor_vulkan;
class VulkanContext;

struct MoeCombineInput_vulkan
{
    const ActivationBuffer* host = nullptr;
    std::shared_ptr<const DeviceTensor_vulkan> device;
    std::span<const ExpertRoute> routes;
};

// Owns every activation and route needed by an unsubmitted Combine. Device
// results are retained; host results are copied only when a device result is absent.
class MoeCombinePending_vulkan
{
public:
    ~MoeCombinePending_vulkan();
    [[nodiscard]] size_t rows() const noexcept;
    [[nodiscard]] uint32_t columns() const noexcept;

private:
    friend class MoeCombine_vulkan;
    MoeCombinePending_vulkan();
    class Implementation;
    std::unique_ptr<Implementation> d;
};

class MoeCombine_vulkan
{
public:
    ~MoeCombine_vulkan();
    MoeCombine_vulkan(const MoeCombine_vulkan&) = delete;
    MoeCombine_vulkan& operator=(const MoeCombine_vulkan&) = delete;
    [[nodiscard]] static std::shared_ptr<MoeCombine_vulkan> create(std::shared_ptr<VulkanContext> context);
    [[nodiscard]] bool combine(std::span<const MoeCombineInput_vulkan> inputs,
                               size_t rows, uint32_t columns,
                               const ActivationBuffer* shared_host,
                               std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                               std::shared_ptr<DeviceTensor_vulkan>& output) const;
    // Capture without recording or submitting; invalid inputs leave the caller unchanged.
    [[nodiscard]] std::shared_ptr<MoeCombinePending_vulkan> prepare(std::span<const MoeCombineInput_vulkan> inputs,
                                                                    size_t rows, uint32_t columns,
                                                                    const ActivationBuffer* shared_host,
                                                                    std::shared_ptr<const DeviceTensor_vulkan> shared_device) const;
#if NCNN_MOE_WITH_VULKAN
    // Caller holds the context command mutex and retains pending/storage until wait.
    // Clear storage under that same mutex after completion or command reset;
    // the shared allocator's fastFree is not independently synchronized.
    [[nodiscard]] bool record(const MoeCombinePending_vulkan& pending, ncnn::VkMat& output,
                              ncnn::VkCompute& command, std::vector<ncnn::VkMat>& storage,
                              uint64_t& uploads) const;
    [[nodiscard]] const std::shared_ptr<ncnn::Pipeline>& pipeline() const noexcept;
#endif
    // Inputs and their route spans must be in the CPU executor's canonical order.
    // Publication is atomic: false leaves output unchanged.
    [[nodiscard]] static bool forward(std::span<const MoeCombineInput_vulkan> inputs,
                                      size_t rows, uint32_t columns,
                                      const ActivationBuffer* shared_host,
                                      std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                                      std::shared_ptr<DeviceTensor_vulkan>& output);
    [[nodiscard]] static bool materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output);

private:
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] bool valid_inputs(std::span<const MoeCombineInput_vulkan> inputs,
                                    size_t rows, uint32_t columns, const ActivationBuffer* shared_host,
                                    const std::shared_ptr<const DeviceTensor_vulkan>& shared_device) const;
    [[nodiscard]] bool record_inputs(std::span<const MoeCombineInput_vulkan> inputs,
                                     size_t rows, uint32_t columns, const ActivationBuffer* shared_host,
                                     const std::shared_ptr<const DeviceTensor_vulkan>& shared_device,
                                     ncnn::VkMat& output, ncnn::VkCompute& command,
                                     std::vector<ncnn::VkMat>& storage, uint64_t& uploads) const;
#endif
    class Implementation;
    MoeCombine_vulkan();
    std::unique_ptr<Implementation> d;
};
} // namespace moe
} // namespace ncnn
#endif // NCNN_MOE_MOECOMBINE_VULKAN_H
