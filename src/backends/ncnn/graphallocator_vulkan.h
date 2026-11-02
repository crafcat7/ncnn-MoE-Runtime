#ifndef NCNN_MOE_GRAPHALLOCATOR_VULKAN_H
#define NCNN_MOE_GRAPHALLOCATOR_VULKAN_H

#include "vulkan.h"

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <memory>
#include <mutex>

namespace ncnn {
namespace moe {

class VulkanExpertWeightAllocator;

// One stable allocator is captured by all operators in a model Context. Its
// backing is initially allowed to grow for persistent preparation; explicit
// graph reservation then warms reusable storage and closes physical growth.
// Worker threads share this allocator, without caller-thread routing state.
class VulkanGraphAllocator final : public ncnn::VkAllocator
{
public:
    explicit VulkanGraphAllocator(const ncnn::VulkanDevice* device);
    ~VulkanGraphAllocator() override;

    ncnn::VkBufferMemory* fastMalloc(size_t size) override;
    void fastFree(ncnn::VkBufferMemory* buffer) override;
    ncnn::VkImageMemory* fastMalloc(int, int, int, size_t, int) override;
    void fastFree(ncnn::VkImageMemory*) override;
    void clear() override;

    // The caller reserves between executions. On the first call, or when the
    // requested size grows, a contiguous free range of at least this size is
    // physically backed before returning. Existing live tensors stay valid.
    // Repeating a covered size preserves live state without a second warm-up.
    [[nodiscard]] bool reserve_graph_compute(uint64_t bytes);
    [[nodiscard]] VulkanGraphArenaSize physical_size() const;

private:
    mutable std::mutex mutex;
    std::unique_ptr<VulkanExpertWeightAllocator> pool;
    uint64_t preparation_physical_size = 0;
    uint64_t reserved_compute_size = 0;
    bool reserved = false;
};

} // namespace moe
} // namespace ncnn
#endif // NCNN_MOE_WITH_VULKAN

#endif // NCNN_MOE_GRAPHALLOCATOR_VULKAN_H
