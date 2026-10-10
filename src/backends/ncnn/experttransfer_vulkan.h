#ifndef NCNN_MOE_EXPERTTRANSFER_VULKAN_H
#define NCNN_MOE_EXPERTTRANSFER_VULKAN_H

#include "vulkan.h"

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <mat.h>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <vector>
#include <span>

namespace ncnn {
namespace moe {

class VulkanContext;

// Immutable Expert buffers are shared by the transfer and compute families.
// Pool metadata has its own mutex; neither allocator calls nor GPU waits need
// the foreground command mutex. Images are deliberately unsupported.
class VulkanExpertWeightAllocator final : public ncnn::VkBlobAllocator
{
public:
    VulkanExpertWeightAllocator(const ncnn::VulkanDevice* device, size_t block_size);
    ~VulkanExpertWeightAllocator() override;
    using ncnn::VkBlobAllocator::fastFree;
    using ncnn::VkBlobAllocator::fastMalloc;
    ncnn::VkBufferMemory* fastMalloc(size_t size) override;
    void fastFree(ncnn::VkBufferMemory* buffer) override;
    ncnn::VkImageMemory* fastMalloc(int, int, int, size_t, int) override;
    void fastFree(ncnn::VkImageMemory*) override;
    void clear() override;

private:
    struct Range
    {
        size_t offset;
        size_t size;
    };
    struct Block
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        size_t capacity = 0;
        std::list<Range> free;
        std::list<Range> used;
    };
    const size_t block_size;
    const size_t alignment;
    mutable std::mutex mutex;
    std::vector<std::unique_ptr<Block>> blocks;
};

// Two isolated admission lanes retain pools, command buffers, fences and
// semaphores. Leases hold their Context alive; this pool owns no Context.
// Runtime admissions use at most two concurrent recording batches. A further
// caller waits for a submitted or cancelled lease to release its lane.
class VulkanExpertTransferPool
{
public:
    struct Lane;
    explicit VulkanExpertTransferPool(const ncnn::VulkanDevice* device);
    ~VulkanExpertTransferPool();
    VulkanExpertTransferPool(const VulkanExpertTransferPool&) = delete;
    VulkanExpertTransferPool& operator=(const VulkanExpertTransferPool&) = delete;
    [[nodiscard]] Lane* acquire();
    void release(Lane* lane, bool reusable) noexcept;

private:
    struct Implementation;
    std::unique_ptr<Implementation> d;
};

// Each in-flight batch exclusively leases a transfer lane. A completed copy
// semaphore is consumed by a short compute visibility command before weights
// are published. Success and unsubmitted cancellation permit pool reuse.
class VulkanIndependentWeightTransfer
{
public:
    VulkanIndependentWeightTransfer(std::shared_ptr<VulkanContext> context,
                                    ncnn::VkAllocator* staging_allocator);
    ~VulkanIndependentWeightTransfer();
    bool record(const ncnn::Mat& source, ncnn::VkMat& destination,
                VulkanExpertWeightAllocator* weight_allocator);
    // One reservation at a time, owned by this batch. The returned mapped
    // span remains valid until record_prepared_storage or batch destruction.
    [[nodiscard]] std::span<uint8_t> prepare_storage(size_t bytes, VulkanExpertWeightAllocator* allocator);
    bool record_prepared_storage(ncnn::VkMat& destination, VulkanExpertWeightAllocator* allocator);
    bool submit_and_wait();

private:
    struct Implementation;
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn
#endif

#endif
