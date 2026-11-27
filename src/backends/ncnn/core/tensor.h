#ifndef NCNN_MOE_BACKENDS_NCNN_CORE_TENSOR_H
#define NCNN_MOE_BACKENDS_NCNN_CORE_TENSOR_H

#include "backends/ncnn/core/context.h"
#include "kernels/activationbuffer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
struct CommandGraphState_vulkan
{
    size_t storage_variant = 0;
    bool submitted = false;
    bool completed = false;
    bool failed = false;
};
#endif

// Device-resident 2D activation tensor handle. Stays in Vulkan storage until
// materialized to CPU or consumed by another GPU command.
class DeviceTensor_vulkan
{
public:
    DeviceTensor_vulkan();
    ~DeviceTensor_vulkan();

    DeviceTensor_vulkan(DeviceTensor_vulkan&&) noexcept;
    DeviceTensor_vulkan& operator=(DeviceTensor_vulkan&&) noexcept;
    DeviceTensor_vulkan(const DeviceTensor_vulkan&) = delete;
    DeviceTensor_vulkan& operator=(const DeviceTensor_vulkan&) = delete;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t rows() const noexcept;
    [[nodiscard]] uint32_t columns() const noexcept;

#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] bool assign_completed(const ncnn::VkMat& val, std::shared_ptr<VulkanContext> ctx);
    void assign_pending(const ncnn::VkMat& val, size_t rows, uint32_t cols,
                        std::shared_ptr<VulkanContext> ctx,
                        std::shared_ptr<CommandGraphState_vulkan> graph);

    [[nodiscard]] const ncnn::VkMat* value() const noexcept;
    [[nodiscard]] const ncnn::VkMat& raw_value() const noexcept;
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
    [[nodiscard]] const std::shared_ptr<CommandGraphState_vulkan>& graph_state() const noexcept;
    [[nodiscard]] bool readable_from(const std::shared_ptr<CommandGraphState_vulkan>& target,
                                     size_t storage_variant) const noexcept;

    // Re-arms a completed import; caller holds context()->command_mutex().
    void prepare_read() const noexcept;

    // Downloads a completed FP32 device tensor to CPU in one staging submission.
    [[nodiscard]] bool materialize(ActivationBuffer& output) const;
#endif

private:
    class Impl;
    std::unique_ptr<Impl> d;
};

#if NCNN_MOE_WITH_VULKAN

// Ordered asynchronous submissions on one leased compute queue.
// Caller holds ctx.command_mutex() and retains all operands until reset().
class VkComputeBatch
{
public:
    explicit VkComputeBatch(VulkanContext& ctx);
    ~VkComputeBatch();

    VkComputeBatch(const VkComputeBatch&) = delete;
    VkComputeBatch& operator=(const VkComputeBatch&) = delete;

    [[nodiscard]] ncnn::VkCompute* next();
    [[nodiscard]] int submit();
    [[nodiscard]] int reset() noexcept;
    [[nodiscard]] size_t submission_count() const noexcept
    { return submitted; }

private:
    VulkanContext& context;
    VkQueue queue = VK_NULL_HANDLE;
    size_t recorded = 0;
    size_t submitted = 0;
    int completion_error = 0;
};

// Two transactional host-visible output slots per workspace. Published
// ActivationBuffers lease their slot via shared_ptr; only unleased slots reuse.
class VkHostOutput
{
public:
    VkHostOutput();
    ~VkHostOutput();

    VkHostOutput(const VkHostOutput&) = delete;
    VkHostOutput& operator=(const VkHostOutput&) = delete;

    [[nodiscard]] bool prepare(const std::shared_ptr<VulkanContext>& ctx,
                               size_t rows, uint32_t cols, uint64_t slot_limit,
                               ncnn::VkMat& dst, ActivationBuffer& pending);
    [[nodiscard]] static bool complete(ncnn::VkMat& dst, const ActivationBuffer& pending);
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;
    [[nodiscard]] uint64_t planned_capacity() const noexcept
    { return slot_capacity; }

private:
    struct Storage;
    std::array<std::shared_ptr<Storage>, 2> slots;
    uint64_t slot_capacity = 0;
};

#endif // NCNN_MOE_WITH_VULKAN

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_BACKENDS_NCNN_CORE_TENSOR_H
