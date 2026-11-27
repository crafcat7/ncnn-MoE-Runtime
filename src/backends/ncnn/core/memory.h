#ifndef NCNN_MOE_BACKENDS_NCNN_CORE_MEMORY_H
#define NCNN_MOE_BACKENDS_NCNN_CORE_MEMORY_H

#include "backends/ncnn/vulkan.h"

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <gpu.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

namespace ncnn {
namespace moe {

// Device-local buffer pool shared concurrently by transfer and compute queues.
// Uses its own mutex and never depends on the foreground command mutex.
class VkWeightAllocator final : public ncnn::VkBlobAllocator
{
public:
    VkWeightAllocator(const ncnn::VulkanDevice* device, size_t block_size,
                      uint64_t physical_limit_bytes = 0);
    ~VkWeightAllocator() override;

    using ncnn::VkBlobAllocator::fastFree;
    using ncnn::VkBlobAllocator::fastMalloc;

    ncnn::VkBufferMemory* fastMalloc(size_t size) override;
    void fastFree(ncnn::VkBufferMemory* buffer) override;
    ncnn::VkImageMemory* fastMalloc(int, int, int, size_t, int) override;
    void fastFree(ncnn::VkImageMemory*) override;
    void clear() override;

    [[nodiscard]] uint64_t physical_bytes() const;
    [[nodiscard]] uint64_t physical_limit_rejections() const;

    // Graph-arena hooks: retain free blocks, toggle physical growth, and drop obsolete blocks.
    void set_graph_mode(bool allow_growth);
    void trim_free_blocks(uint64_t min_keep_bytes);

private:
    struct Allocation;

    struct Range
    {
        size_t offset;
        size_t size;
        Allocation* allocation = nullptr;
    };

    struct Block
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        size_t capacity = 0;
        VkDeviceSize memory_size = 0;
        std::list<Range> free;
        std::list<Range> used;
    };

    struct Allocation : ncnn::VkBufferMemory
    {
        Block* block = nullptr;
        std::list<Range>::iterator range;
        Allocation* next = nullptr;
    };

    const size_t block_size;
    const size_t alignment;
    const uint64_t physical_limit;

    uint64_t allocated_bytes = 0;
    uint64_t limit_rejections = 0;
    bool retain_free_blocks = false;
    bool allow_physical_growth = true;

    mutable std::mutex mutex;
    std::vector<std::unique_ptr<Block>> blocks;

    // Intrusive freelist bounded by peak_live_allocations in graph mode.
    Allocation* spare_allocations = nullptr;
    std::list<Range> spare_ranges;
    size_t live_allocations = 0;
    size_t peak_live_allocations = 0;
};

// Stable per-context device-local arena. Grows during preparation, then seals
// after reserve() warms a single contiguous compute peak.
class VkGraphAllocator final : public ncnn::VkAllocator
{
public:
    explicit VkGraphAllocator(const ncnn::VulkanDevice* device);
    ~VkGraphAllocator() override;

    ncnn::VkBufferMemory* fastMalloc(size_t size) override;
    void fastFree(ncnn::VkBufferMemory* buffer) override;
    ncnn::VkImageMemory* fastMalloc(int, int, int, size_t, int) override;
    void fastFree(ncnn::VkImageMemory*) override;
    void clear() override;

    [[nodiscard]] bool reserve(uint64_t bytes);
    [[nodiscard]] VulkanGraphArenaSize physical_size() const;
    [[nodiscard]] bool sealed() const;

private:
    mutable std::mutex mutex;
    std::unique_ptr<VkWeightAllocator> pool;
    uint64_t preparation_physical_size = 0;
    uint64_t reserved_compute_size = 0;
    bool reserved = false;
};

// Bounded host-visible staging cache for serialized weight admissions.
// Throws std::bad_alloc on OOM so callers never dereference a null mapped_ptr.
class VkStagingAllocator final : public ncnn::VkStagingAllocator
{
public:
    VkStagingAllocator(const ncnn::VulkanDevice* device, uint64_t cache_limit);

    using ncnn::VkStagingAllocator::fastFree;
    using ncnn::VkStagingAllocator::fastMalloc;

    ncnn::VkBufferMemory* fastMalloc(size_t size) override;
    void fastFree(ncnn::VkBufferMemory* buffer) override;
    void clear() override;

private:
    struct CachedBuffer
    {
        ncnn::VkBufferMemory* buffer = nullptr;
        uint64_t allocation_size = 0;
    };

    const uint64_t cache_limit;
    std::array<CachedBuffer, 32> cached_buffers{};
    size_t cached_count = 0;
    uint64_t cached_size = 0;
};

} // namespace moe
} // namespace ncnn
#endif // NCNN_MOE_WITH_VULKAN

#endif // NCNN_MOE_BACKENDS_NCNN_CORE_MEMORY_H
