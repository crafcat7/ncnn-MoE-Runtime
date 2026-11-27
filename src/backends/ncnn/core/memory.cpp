#include "backends/ncnn/core/memory.h"

#include <algorithm>
#include <limits>
#include <new>

namespace ncnn {
namespace moe {

VkWeightAllocator::VkWeightAllocator(const ncnn::VulkanDevice* device, size_t _block_size,
                                     uint64_t physical_limit_bytes)
    : ncnn::VkBlobAllocator(device, _block_size),
      block_size(_block_size),
      alignment(std::max<size_t>(4, device->info.buffer_offset_alignment())),
      physical_limit(physical_limit_bytes)
{
    // Force explicit staging even on UMA so host writes never race compute reads.
    mappable = false;
    coherent = false;
}

VkWeightAllocator::~VkWeightAllocator()
{
    clear();
}

ncnn::VkBufferMemory* VkWeightAllocator::fastMalloc(size_t size)
{
    if (size == 0 || size > std::numeric_limits<size_t>::max() - alignment + 1)
        return nullptr;

    const size_t aligned = ((size + alignment - 1) / alignment) * alignment;
    const std::lock_guard<std::mutex> lock(mutex);

    const auto take_range = [&](Block& block, std::list<Range>::iterator range) {
        std::unique_ptr<Allocation> alloc;
        if (retain_free_blocks && spare_allocations)
        {
            alloc.reset(spare_allocations);
            spare_allocations = spare_allocations->next;
        }
        else
        {
            alloc = std::make_unique<Allocation>();
        }

        alloc->buffer = block.buffer;
        alloc->offset = range->offset;
        alloc->capacity = aligned;
        alloc->memory = block.memory;
        alloc->mapped_ptr = nullptr;
        alloc->memory_type_index = buffer_memory_type_index;
        alloc->access_flags = 0;
        alloc->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        alloc->refcount = 0;
        alloc->next = nullptr;

        if (range->size == aligned)
        {
            block.used.splice(block.used.end(), block.free, range);
            alloc->range = range;
        }
        else
        {
            // Acquire the list node before mutating free ranges for strong exception safety.
            if (retain_free_blocks && !spare_ranges.empty())
            {
                auto reused = spare_ranges.begin();
                *reused = {range->offset, aligned};
                block.used.splice(block.used.end(), spare_ranges, reused);
            }
            else
            {
                block.used.push_back({range->offset, aligned});
            }

            alloc->range = std::prev(block.used.end());
            range->offset += aligned;
            range->size -= aligned;
        }

        alloc->block = &block;
        alloc->range->allocation = alloc.get();
        ++live_allocations;
        peak_live_allocations = std::max(peak_live_allocations, live_allocations);

        return static_cast<ncnn::VkBufferMemory*>(alloc.release());
    };

    if (retain_free_blocks)
    {
        // Best-fit hole selection matches the offline graph memory planner.
        Block* best_block = nullptr;
        std::list<Range>::iterator best_range;

        for (const auto& block : blocks)
        {
            for (auto range = block->free.begin(); range != block->free.end(); ++range)
            {
                if (range->size >= aligned
                    && (!best_block
                        || range->size < best_range->size
                        || (range->size == best_range->size && range->offset < best_range->offset)))
                {
                    best_block = block.get();
                    best_range = range;
                }
            }
        }

        if (best_block)
            return take_range(*best_block, best_range);
    }
    else
    {
        for (const auto& block : blocks)
        {
            for (auto range = block->free.begin(); range != block->free.end(); ++range)
            {
                if (range->size >= aligned)
                    return take_range(*block, range);
            }
        }
    }

    if (!allow_physical_growth)
    {
        if (limit_rejections != std::numeric_limits<uint64_t>::max())
            ++limit_rejections;
        return nullptr;
    }

    auto block = std::make_unique<Block>();
    block->capacity = std::max(block_size, aligned);

    const uint32_t families[] = {
        vkdev->info.compute_queue_family_index(),
        vkdev->info.transfer_queue_family_index()};
    const bool concurrent = families[0] != families[1];

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = block->capacity;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = concurrent ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    buffer_info.queueFamilyIndexCount = concurrent ? 2 : 0;
    buffer_info.pQueueFamilyIndices = concurrent ? families : nullptr;

    if (vkCreateBuffer(vkdev->vkdevice(), &buffer_info, nullptr, &block->buffer) != VK_SUCCESS)
        return nullptr;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &req);

    if (physical_limit != 0 && (req.size > physical_limit || allocated_bytes > physical_limit - req.size))
    {
        // Reclaim wholly free warm blocks before rejecting against the physical limit.
        for (auto it = blocks.begin(); it != blocks.end();)
        {
            if (!(*it)->used.empty())
            {
                ++it;
                continue;
            }
            vkDestroyBuffer(vkdev->vkdevice(), (*it)->buffer, nullptr);
            vkFreeMemory(vkdev->vkdevice(), (*it)->memory, nullptr);
            allocated_bytes -= (*it)->memory_size;
            it = blocks.erase(it);
        }

        if (req.size > physical_limit || allocated_bytes > physical_limit - req.size)
        {
            vkDestroyBuffer(vkdev->vkdevice(), block->buffer, nullptr);
            if (limit_rejections != std::numeric_limits<uint64_t>::max())
                ++limit_rejections;
            return nullptr;
        }
    }

    if (req.size > std::numeric_limits<uint64_t>::max() - allocated_bytes)
    {
        vkDestroyBuffer(vkdev->vkdevice(), block->buffer, nullptr);
        return nullptr;
    }

    if (buffer_memory_type_index == UINT32_MAX)
        buffer_memory_type_index = vkdev->find_memory_index(req.memoryTypeBits,
                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                            0,
                                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    block->memory = allocate_memory(req.size, buffer_memory_type_index);
    if (block->memory == VK_NULL_HANDLE
        || vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0) != VK_SUCCESS)
    {
        vkDestroyBuffer(vkdev->vkdevice(), block->buffer, nullptr);
        if (block->memory)
            vkFreeMemory(vkdev->vkdevice(), block->memory, nullptr);
        return nullptr;
    }

    block->memory_size = req.size;

    try
    {
        block->free.push_back({0, block->capacity});
        blocks.push_back(std::move(block));
        allocated_bytes += req.size;
    }
    catch (...)
    {
        if (block)
        {
            vkDestroyBuffer(vkdev->vkdevice(), block->buffer, nullptr);
            vkFreeMemory(vkdev->vkdevice(), block->memory, nullptr);
        }
        throw;
    }

    return take_range(*blocks.back(), blocks.back()->free.begin());
}

void VkWeightAllocator::fastFree(ncnn::VkBufferMemory* buffer)
{
    if (!buffer)
        return;

    const std::lock_guard<std::mutex> lock(mutex);
    auto* alloc = static_cast<Allocation*>(buffer);
    Block* block = alloc->block;

    if (!block)
    {
        // Disarmed by clear(); destroy the detached header safely.
        delete alloc;
        return;
    }

    auto next = block->free.begin();
    while (next != block->free.end() && next->offset < buffer->offset)
        ++next;

    auto cur = alloc->range;
    block->free.splice(next, block->used, cur);
    cur->allocation = nullptr;
    alloc->block = nullptr;
    alloc->range = {};
    --live_allocations;

    if (cur != block->free.begin())
    {
        auto prev = std::prev(cur);
        if (prev->offset + prev->size == cur->offset)
        {
            prev->size += cur->size;
            if (retain_free_blocks && spare_ranges.size() < peak_live_allocations)
            {
                cur->allocation = nullptr;
                spare_ranges.splice(spare_ranges.end(), block->free, cur);
            }
            else
            {
                block->free.erase(cur);
            }
            cur = prev;
        }
    }

    next = std::next(cur);
    if (next != block->free.end() && cur->offset + cur->size == next->offset)
    {
        cur->size += next->size;
        if (retain_free_blocks && spare_ranges.size() < peak_live_allocations)
        {
            next->allocation = nullptr;
            spare_ranges.splice(spare_ranges.end(), block->free, next);
        }
        else
        {
            block->free.erase(next);
        }
    }

    if (retain_free_blocks)
    {
        alloc->next = spare_allocations;
        spare_allocations = alloc;
        return;
    }

    delete alloc;

    if (block->used.empty())
    {
        const bool another_free = std::any_of(blocks.begin(), blocks.end(), [&](const auto& other) {
            return other.get() != block && other->used.empty();
        });

        // Retain at most one ordinary warm block; drop oversized or redundant free blocks.
        if (block->capacity > block_size || another_free)
        {
            vkDestroyBuffer(vkdev->vkdevice(), block->buffer, nullptr);
            vkFreeMemory(vkdev->vkdevice(), block->memory, nullptr);
            allocated_bytes -= block->memory_size;
            blocks.erase(std::remove_if(blocks.begin(), blocks.end(),
                                        [block](const auto& item) { return item.get() == block; }),
                         blocks.end());
        }
    }
}

ncnn::VkImageMemory* VkWeightAllocator::fastMalloc(int, int, int, size_t, int)
{
    return nullptr;
}

void VkWeightAllocator::fastFree(ncnn::VkImageMemory*)
{
}

void VkWeightAllocator::clear()
{
    const std::lock_guard<std::mutex> lock(mutex);

    for (const auto& block : blocks)
    {
        for (const auto& range : block->used)
        {
            range.allocation->block = nullptr;
            range.allocation->range = {};
        }

        vkDestroyBuffer(vkdev->vkdevice(), block->buffer, nullptr);
        vkFreeMemory(vkdev->vkdevice(), block->memory, nullptr);
    }

    blocks.clear();

    while (spare_allocations)
    {
        Allocation* next = spare_allocations->next;
        delete spare_allocations;
        spare_allocations = next;
    }

    spare_ranges.clear();
    live_allocations = 0;
    peak_live_allocations = 0;
    allocated_bytes = 0;
}

uint64_t VkWeightAllocator::physical_bytes() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    return allocated_bytes;
}

uint64_t VkWeightAllocator::physical_limit_rejections() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    return limit_rejections;
}

void VkWeightAllocator::set_graph_mode(bool allow_growth)
{
    const std::lock_guard<std::mutex> lock(mutex);
    retain_free_blocks = true;
    allow_physical_growth = allow_growth;
}

void VkWeightAllocator::trim_free_blocks(uint64_t min_keep_bytes)
{
    const std::lock_guard<std::mutex> lock(mutex);

    const Block* kept = nullptr;
    for (const auto& block : blocks)
    {
        if (block->used.empty() && block->capacity >= min_keep_bytes
            && (!kept || block->capacity < kept->capacity))
        {
            kept = block.get();
        }
    }

    for (auto it = blocks.begin(); it != blocks.end();)
    {
        if (!(*it)->used.empty() || it->get() == kept)
        {
            ++it;
            continue;
        }

        vkDestroyBuffer(vkdev->vkdevice(), (*it)->buffer, nullptr);
        vkFreeMemory(vkdev->vkdevice(), (*it)->memory, nullptr);
        allocated_bytes -= (*it)->memory_size;
        it = blocks.erase(it);
    }
}

// --- VkGraphAllocator --------------------------------------------------------

VkGraphAllocator::VkGraphAllocator(const ncnn::VulkanDevice* device)
    : ncnn::VkAllocator(device),
      pool(std::make_unique<VkWeightAllocator>(device, 64 * 1024))
{
    mappable = false;
    coherent = false;
    cached = false;
    pool->set_graph_mode(true);
}

VkGraphAllocator::~VkGraphAllocator() = default;

ncnn::VkBufferMemory* VkGraphAllocator::fastMalloc(size_t size)
{
    const std::lock_guard<std::mutex> lock(mutex);
    ncnn::VkBufferMemory* buffer = pool->fastMalloc(size);
    buffer_memory_type_index = pool->buffer_memory_type_index;
    return buffer;
}

void VkGraphAllocator::fastFree(ncnn::VkBufferMemory* buffer)
{
    pool->fastFree(buffer);
}

ncnn::VkImageMemory* VkGraphAllocator::fastMalloc(int, int, int, size_t, int)
{
    return nullptr;
}

void VkGraphAllocator::fastFree(ncnn::VkImageMemory*)
{
}

void VkGraphAllocator::clear()
{
    const std::lock_guard<std::mutex> lock(mutex);
    pool->clear();
    preparation_physical_size = 0;
    reserved_compute_size = 0;
    // Keep a previously sealed arena sealed until an explicit re-reservation.
    pool->set_graph_mode(!reserved);
}

bool VkGraphAllocator::reserve(uint64_t bytes)
{
    const std::lock_guard<std::mutex> lock(mutex);

    if (reserved && bytes <= reserved_compute_size)
        return true;

    if (!reserved)
        preparation_physical_size = pool->physical_bytes();

    reserved_compute_size = 0;
    pool->trim_free_blocks(bytes);
    preparation_physical_size = std::min(preparation_physical_size, pool->physical_bytes());
    pool->set_graph_mode(true);

    ncnn::VkBufferMemory* warm = nullptr;
    try
    {
        if (bytes != 0)
            warm = pool->fastMalloc(static_cast<size_t>(bytes));
        if (warm)
            pool->fastFree(warm);
    }
    catch (...)
    {
        pool->set_graph_mode(false);
        return false;
    }

    pool->set_graph_mode(false);
    if (bytes != 0 && !warm)
        return false;

    reserved_compute_size = bytes;
    reserved = true;
    buffer_memory_type_index = pool->buffer_memory_type_index;
    return true;
}

VulkanGraphArenaSize VkGraphAllocator::physical_size() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    const uint64_t total = pool->physical_bytes();
    return {reserved ? preparation_physical_size : total, total};
}

bool VkGraphAllocator::sealed() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    return reserved;
}

// --- VkStagingAllocator ------------------------------------------------------

VkStagingAllocator::VkStagingAllocator(const ncnn::VulkanDevice* device, uint64_t _cache_limit)
    : ncnn::VkStagingAllocator(device),
      cache_limit(_cache_limit)
{
}

ncnn::VkBufferMemory* VkStagingAllocator::fastMalloc(size_t size)
{
    ncnn::VkBufferMemory* buffer = ncnn::VkStagingAllocator::fastMalloc(size);
    if (!buffer)
        throw std::bad_alloc();

    for (size_t i = 0; i < cached_count; ++i)
    {
        if (cached_buffers[i].buffer != buffer)
            continue;

        cached_size -= cached_buffers[i].allocation_size;
        cached_buffers[i] = cached_buffers[--cached_count];
        cached_buffers[cached_count] = CachedBuffer{};
        return buffer;
    }

    return buffer;
}

void VkStagingAllocator::fastFree(ncnn::VkBufferMemory* buffer)
{
    if (!buffer)
        return;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), buffer->buffer, &req);
    const uint64_t alloc_size = req.size;

    if (cached_count == cached_buffers.size() || alloc_size > cache_limit - cached_size)
        clear();

    try
    {
        ncnn::VkStagingAllocator::fastFree(buffer);
    }
    catch (...)
    {
        // Safe fallback when ncnn's std::list insertion throws during noexcept teardown.
        if (buffer->mapped_ptr)
            vkUnmapMemory(vkdev->vkdevice(), buffer->memory);
        vkDestroyBuffer(vkdev->vkdevice(), buffer->buffer, nullptr);
        vkFreeMemory(vkdev->vkdevice(), buffer->memory, nullptr);
        delete buffer;
        return;
    }

    if (alloc_size > cache_limit)
    {
        clear();
        return;
    }

    cached_buffers[cached_count++] = {buffer, alloc_size};
    cached_size += alloc_size;
}

void VkStagingAllocator::clear()
{
    ncnn::VkStagingAllocator::clear();
    cached_buffers.fill(CachedBuffer{});
    cached_count = 0;
    cached_size = 0;
}

} // namespace moe
} // namespace ncnn
