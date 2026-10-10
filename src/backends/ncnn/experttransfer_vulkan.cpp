#include "experttransfer_vulkan.h"
#include "vulkancontext.h"

#if NCNN_MOE_WITH_VULKAN
#include <algorithm>
#include <array>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <thread>

namespace ncnn {
namespace moe {

VulkanExpertWeightAllocator::VulkanExpertWeightAllocator(const ncnn::VulkanDevice* device, size_t _block_size)
    : ncnn::VkBlobAllocator(device, _block_size),
      block_size(_block_size),
      alignment(std::max<size_t>(4, device->info.buffer_offset_alignment()))
{
    // Always stage uploads, including on unified-memory devices. This keeps
    // host writes away from buffers concurrently read by a compute queue.
    mappable = false;
    coherent = false;
}

VulkanExpertWeightAllocator::~VulkanExpertWeightAllocator()
{
    clear();
}

ncnn::VkBufferMemory* VulkanExpertWeightAllocator::fastMalloc(size_t size)
{
    if (size == 0 || size > std::numeric_limits<size_t>::max() - alignment + 1)
        return nullptr;
    const size_t aligned = ((size + alignment - 1) / alignment) * alignment;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto allocate_range = [&](Block& block, std::list<Range>::iterator range) {
        auto result = std::make_unique<ncnn::VkBufferMemory>();
        result->buffer = block.buffer;
        result->offset = range->offset;
        result->capacity = aligned;
        result->memory = block.memory;
        result->mapped_ptr = nullptr;
        result->memory_type_index = buffer_memory_type_index;
        result->access_flags = 0;
        result->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        result->refcount = 0;
        // Allocate the range node before changing free-space accounting.
        // fastFree can then splice it without allocating from a destructor.
        block.used.push_back({range->offset, aligned});
        range->offset += aligned;
        range->size -= aligned;
        if (range->size == 0) block.free.erase(range);
        return result.release();
    };
    for (const auto& block : blocks)
        for (auto range = block->free.begin(); range != block->free.end(); ++range)
            if (range->size >= aligned) return allocate_range(*block, range);

    auto block = std::make_unique<Block>();
    block->capacity = std::max(block_size, aligned);
    const uint32_t families[] = {vkdev->info.compute_queue_family_index(), vkdev->info.transfer_queue_family_index()};
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = block->capacity;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = families[0] == families[1] ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT;
    buffer_info.queueFamilyIndexCount = families[0] == families[1] ? 0 : 2;
    buffer_info.pQueueFamilyIndices = families[0] == families[1] ? nullptr : families;
    if (vkCreateBuffer(vkdev->vkdevice(), &buffer_info, nullptr, &block->buffer) != VK_SUCCESS)
        return nullptr;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &requirements);
    if (buffer_memory_type_index == UINT32_MAX)
        buffer_memory_type_index = vkdev->find_memory_index(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    block->memory = allocate_memory(requirements.size, buffer_memory_type_index);
    if (block->memory == VK_NULL_HANDLE
        || vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0) != VK_SUCCESS)
    {
        vkDestroyBuffer(vkdev->vkdevice(), block->buffer, nullptr);
        if (block->memory) vkFreeMemory(vkdev->vkdevice(), block->memory, nullptr);
        return nullptr;
    }
    try
    {
        block->free.push_back({0, block->capacity});
        blocks.push_back(std::move(block));
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
    return allocate_range(*blocks.back(), blocks.back()->free.begin());
}

void VulkanExpertWeightAllocator::fastFree(ncnn::VkBufferMemory* buffer)
{
    if (!buffer) return;
    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto& block : blocks)
    {
        if (block->buffer != buffer->buffer) continue;
        auto next = block->free.begin();
        while (next != block->free.end() && next->offset < buffer->offset) ++next;
        auto used = block->used.begin();
        while (used != block->used.end() && used->offset != buffer->offset) ++used;
        if (used == block->used.end())
        {
            delete buffer;
            return;
        }
        block->free.splice(next, block->used, used);
        auto inserted = used;
        if (inserted != block->free.begin())
        {
            auto previous = std::prev(inserted);
            if (previous->offset + previous->size == inserted->offset)
            {
                previous->size += inserted->size;
                block->free.erase(inserted);
                inserted = previous;
            }
        }
        next = std::next(inserted);
        if (next != block->free.end() && inserted->offset + inserted->size == next->offset)
        {
            inserted->size += next->size;
            block->free.erase(next);
        }
        delete buffer;
        if (block->used.empty())
        {
            const bool another_free = std::any_of(blocks.begin(), blocks.end(), [&](const auto& other) {
                return other.get() != block.get() && other->used.empty();
            });
            // Retain at most one ordinary warm block. Oversized fully free
            // buffers are returned immediately, so shape churn is bounded.
            if (block->capacity > block_size || another_free)
            {
                const Block* released = block.get();
                vkDestroyBuffer(vkdev->vkdevice(), block->buffer, nullptr);
                vkFreeMemory(vkdev->vkdevice(), block->memory, nullptr);
                blocks.erase(std::remove_if(blocks.begin(), blocks.end(),
                                            [released](const auto& candidate) { return candidate.get() == released; }),
                             blocks.end());
            }
        }
        return;
    }
    delete buffer;
}

ncnn::VkImageMemory* VulkanExpertWeightAllocator::fastMalloc(int, int, int, size_t, int)
{ return nullptr; }
void VulkanExpertWeightAllocator::fastFree(ncnn::VkImageMemory*)
{
}

void VulkanExpertWeightAllocator::clear()
{
    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto& block : blocks)
    {
        vkDestroyBuffer(vkdev->vkdevice(), block->buffer, nullptr);
        vkFreeMemory(vkdev->vkdevice(), block->memory, nullptr);
    }
    blocks.clear();
}

struct VulkanExpertTransferPool::Lane
{
    const ncnn::VulkanDevice* device;
    const bool independent;
    VkCommandPool transfer_pool = VK_NULL_HANDLE;
    VkCommandPool compute_pool = VK_NULL_HANDLE;
    VkCommandBuffer transfer = VK_NULL_HANDLE;
    VkCommandBuffer compute = VK_NULL_HANDLE;
    VkFence transfer_fence = VK_NULL_HANDLE;
    VkFence compute_fence = VK_NULL_HANDLE;
    VkSemaphore ready = VK_NULL_HANDLE;

    explicit Lane(const ncnn::VulkanDevice* _device)
        : device(_device), independent(!_device->info.unified_compute_transfer_queue())
    {
    }

    bool make_command(uint32_t family, VkCommandPool& pool, VkCommandBuffer& command, VkFence& fence)
    {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = family;
        if (vkCreateCommandPool(device->vkdevice(), &pool_info, nullptr, &pool) != VK_SUCCESS)
        {
            pool = VK_NULL_HANDLE;
            return false;
        }
        VkCommandBufferAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device->vkdevice(), &allocate_info, &command) != VK_SUCCESS)
        {
            command = VK_NULL_HANDLE;
            return false;
        }
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device->vkdevice(), &fence_info, nullptr, &fence) != VK_SUCCESS)
        {
            fence = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }

    bool initialize()
    {
        const auto& info = device->info;
        if (!make_command(independent ? info.transfer_queue_family_index() : info.compute_queue_family_index(), transfer_pool, transfer, transfer_fence))
            return false;
        if (independent)
        {
            if (!make_command(info.compute_queue_family_index(), compute_pool, compute, compute_fence))
                return false;
            VkSemaphoreCreateInfo semaphore_info{};
            semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            if (vkCreateSemaphore(device->vkdevice(), &semaphore_info, nullptr, &ready) != VK_SUCCESS)
            {
                ready = VK_NULL_HANDLE;
                return false;
            }
        }
        return true;
    }

    bool begin()
    {
        if (vkResetCommandPool(device->vkdevice(), transfer_pool, 0) != VK_SUCCESS
            || vkResetFences(device->vkdevice(), 1, &transfer_fence) != VK_SUCCESS)
            return false;
        if (independent && (vkResetCommandPool(device->vkdevice(), compute_pool, 0) != VK_SUCCESS || vkResetFences(device->vkdevice(), 1, &compute_fence) != VK_SUCCESS))
            return false;
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        return vkBeginCommandBuffer(transfer, &begin_info) == VK_SUCCESS
               && (!independent || vkBeginCommandBuffer(compute, &begin_info) == VK_SUCCESS);
    }

    ~Lane()
    {
        const VkDevice native = device->vkdevice();
        if (transfer) vkFreeCommandBuffers(native, transfer_pool, 1, &transfer);
        if (compute) vkFreeCommandBuffers(native, compute_pool, 1, &compute);
        if (ready) vkDestroySemaphore(native, ready, nullptr);
        if (transfer_fence) vkDestroyFence(native, transfer_fence, nullptr);
        if (compute_fence) vkDestroyFence(native, compute_fence, nullptr);
        if (transfer_pool) vkDestroyCommandPool(native, transfer_pool, nullptr);
        if (compute_pool) vkDestroyCommandPool(native, compute_pool, nullptr);
    }
};

struct VulkanExpertTransferPool::Implementation
{
    const ncnn::VulkanDevice* device;
    struct Slot
    {
        std::unique_ptr<Lane> lane;
        bool in_use = false;
    };
    std::array<Slot, 2> slots;
    mutable std::mutex mutex;
    std::condition_variable available;
    explicit Implementation(const ncnn::VulkanDevice* _device) : device(_device)
    {
    }
};

VulkanExpertTransferPool::VulkanExpertTransferPool(const ncnn::VulkanDevice* device)
    : d(std::make_unique<Implementation>(device))
{
}
VulkanExpertTransferPool::~VulkanExpertTransferPool() = default;

VulkanExpertTransferPool::Lane* VulkanExpertTransferPool::acquire()
{
    std::unique_lock<std::mutex> lock(d->mutex);
    d->available.wait(lock, [&] {
        return std::any_of(d->slots.begin(), d->slots.end(), [](const auto& slot) { return !slot.in_use; });
    });
    auto slot = std::find_if(d->slots.begin(), d->slots.end(), [](const auto& candidate) { return !candidate.in_use; });
    try
    {
        if (!slot->lane)
        {
            slot->lane = std::make_unique<Lane>(d->device);
            if (!slot->lane->initialize())
            {
                slot->lane.reset();
                return nullptr;
            }
        }
        if (!slot->lane->begin())
        {
            slot->lane.reset();
            return nullptr;
        }
        slot->in_use = true;
        return slot->lane.get();
    }
    catch (...)
    {
        slot->lane.reset();
        return nullptr;
    }
}

void VulkanExpertTransferPool::release(Lane* lane, bool reusable) noexcept
{
    if (!lane) return;
    {
        const std::lock_guard<std::mutex> lock(d->mutex);
        for (auto& slot : d->slots)
        {
            if (slot.lane.get() != lane) continue;
            if (!reusable) slot.lane.reset();
            slot.in_use = false;
            break;
        }
    }
    d->available.notify_one();
}

struct VulkanIndependentWeightTransfer::Implementation
{
    std::shared_ptr<VulkanContext> context;
    ncnn::VkAllocator* staging_allocator;
    VulkanExpertWeightAllocator* weight_allocator = nullptr;
    VulkanExpertTransferPool::Lane* lane = nullptr;
    VkQueue transfer_queue = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    std::vector<ncnn::VkMat> staging;
    std::vector<ncnn::VkMat> destinations;
    ncnn::VkMat pending_storage;
    size_t pending_bytes = 0;
    bool independent_queue = false;
    bool attempted = false;
    bool transfer_submitted = false;
    bool compute_submitted = false;
    bool settled = false;
    bool failed = false;

    Implementation(std::shared_ptr<VulkanContext> _context, ncnn::VkAllocator* _staging)
        : context(std::move(_context)), staging_allocator(_staging)
    {
        independent_queue = context && !context->device()->info.unified_compute_transfer_queue();
    }

    bool valid_allocator(VulkanExpertWeightAllocator* allocator) const noexcept
    {
        return context && staging_allocator && staging_allocator->vkdev == context->device()
               && allocator && allocator->vkdev == context->device()
               && (!weight_allocator || weight_allocator == allocator);
    }

    bool initialize()
    {
        if (lane) return true;
        lane = context->expert_transfer_pool().acquire();
        if (!lane) return false;
        return true;
    }

    bool submit_command(uint32_t family, VkCommandBuffer command, VkFence fence, bool wait, bool signal, VkQueue& queue)
    {
        ncnn::VulkanDevice* device = context->device();
        queue = device->acquire_queue(family);
        if (!queue) return false;
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command;
        submit_info.waitSemaphoreCount = wait ? 1 : 0;
        submit_info.pWaitSemaphores = wait ? &lane->ready : nullptr;
        submit_info.pWaitDstStageMask = wait ? &wait_stage : nullptr;
        submit_info.signalSemaphoreCount = signal ? 1 : 0;
        submit_info.pSignalSemaphores = signal ? &lane->ready : nullptr;
        const VkResult result = vkQueueSubmit(queue, 1, &submit_info, fence);
        if (result != VK_SUCCESS)
        {
            device->reclaim_queue(family, queue);
            queue = VK_NULL_HANDLE;
        }
        // Keep exclusive submit ownership until completion. If a fence wait
        // reports OOM/UNKNOWN, cleanup can safely idle this exact queue without
        // racing a foreground vkQueueSubmit or idling the entire device.
        return result == VK_SUCCESS;
    }

    uint32_t transfer_family() const noexcept
    {
        const auto& info = context->device()->info;
        return independent_queue ? info.transfer_queue_family_index() : info.compute_queue_family_index();
    }

    void reclaim_queue(uint32_t family, VkQueue& queue) noexcept
    {
        if (!queue) return;
        context->device()->reclaim_queue(family, queue);
        queue = VK_NULL_HANDLE;
    }

    bool wait_fence(VkFence fence, uint32_t family, VkQueue& queue)
    {
        const VkResult result = vkWaitForFences(context->device()->vkdevice(), 1, &fence, VK_TRUE, UINT64_MAX);
        if (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST)
            reclaim_queue(family, queue);
        return result == VK_SUCCESS;
    }

    void settle_queue(VkFence fence, uint32_t family, VkQueue& queue) noexcept
    {
        if (!queue) return; // Its successful fence wait already settled it.
        VkResult result = vkWaitForFences(context->device()->vkdevice(), 1, &fence, VK_TRUE, UINT64_MAX);
        while (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST)
        {
            result = vkQueueWaitIdle(queue); // Queue lease supplies external synchronization.
            if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // An OOM/UNKNOWN return does not prove completion. Keep all tensors,
        // the command lane and queue lease alive until completion or loss.
        reclaim_queue(family, queue);
    }

    bool record_staging(const ncnn::VkMat& source, size_t bytes, ncnn::VkMat& destination,
                        VulkanExpertWeightAllocator* allocator)
    {
        if (failed || attempted || source.empty() || source.dims != 1 || source.elempack != 1
            || bytes == 0 || bytes % 4 != 0 || source.buffer_offset() % 4 != 0
            || bytes > source.total() * source.elemsize || !valid_allocator(allocator) || !initialize())
        {
            failed = true;
            return false;
        }
        weight_allocator = allocator;
        try
        {
            destination.create_like(source, allocator);
            if (destination.empty() || destination.buffer_offset() % 4 != 0
                || source.allocator->flush(source.data) != 0)
            {
                failed = true;
                destination.release();
                return false;
            }
            // Reserve owners before recording commands; they survive until all
            // submitted work settles, including factory failures and aborts.
            staging.push_back(source);
            destinations.push_back(destination);
            VkBufferMemoryBarrier host_barrier{};
            host_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            host_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_barrier.buffer = source.buffer();
            host_barrier.offset = source.buffer_offset();
            host_barrier.size = bytes;
            vkCmdPipelineBarrier(lane->transfer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &host_barrier, 0, nullptr);
            VkBufferCopy copy{source.buffer_offset(), destination.buffer_offset(), bytes};
            vkCmdCopyBuffer(lane->transfer, source.buffer(), destination.buffer(), 1, &copy);
            VkBufferMemoryBarrier visible{};
            visible.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            visible.srcAccessMask = independent_queue ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
            visible.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            visible.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            visible.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            visible.buffer = destination.buffer();
            visible.offset = destination.buffer_offset();
            visible.size = bytes;
            vkCmdPipelineBarrier(independent_queue ? lane->compute : lane->transfer,
                                 independent_queue ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &visible, 0, nullptr);
            return true;
        }
        catch (...)
        {
            failed = true;
            destination.release();
            return false;
        }
    }

    ~Implementation()
    {
        if (!context) return;
        if (lane && !settled && (transfer_submitted || compute_submitted))
        {
            if (transfer_submitted) settle_queue(lane->transfer_fence, transfer_family(), transfer_queue);
            if (compute_submitted) settle_queue(lane->compute_fence, context->device()->info.compute_queue_family_index(), compute_queue);
        }
        pending_storage.release();
        staging.clear();
        destinations.clear();
        // A partially submitted failed lane may contain an unconsumed binary
        // semaphore signal. Discard it instead of reusing ambiguous state.
        context->expert_transfer_pool().release(lane, !failed && (!attempted || settled));
    }
};

VulkanIndependentWeightTransfer::VulkanIndependentWeightTransfer(std::shared_ptr<VulkanContext> context, ncnn::VkAllocator* staging_allocator)
    : d(std::make_unique<Implementation>(std::move(context), staging_allocator))
{
}
VulkanIndependentWeightTransfer::~VulkanIndependentWeightTransfer() = default;

std::span<uint8_t> VulkanIndependentWeightTransfer::prepare_storage(size_t bytes, VulkanExpertWeightAllocator* allocator)
{
    if (d->failed || d->attempted || !d->pending_storage.empty() || bytes == 0 || bytes % 4 != 0
        || bytes > static_cast<size_t>(std::numeric_limits<int>::max()) || !d->valid_allocator(allocator))
    {
        d->failed = true;
        return {};
    }
    try
    {
        d->pending_storage.create(static_cast<int>(bytes), sizeof(uint8_t), d->staging_allocator);
        if (d->pending_storage.empty() || !d->pending_storage.mapped_ptr())
        {
            d->failed = true;
            return {};
        }
        d->weight_allocator = allocator;
        d->pending_bytes = bytes;
        return {static_cast<uint8_t*>(d->pending_storage.mapped_ptr()), bytes};
    }
    catch (...)
    {
        d->failed = true;
        return {};
    }
}

bool VulkanIndependentWeightTransfer::record_prepared_storage(ncnn::VkMat& destination, VulkanExpertWeightAllocator* allocator)
{
    if (d->pending_storage.empty())
    {
        d->failed = true;
        return false;
    }
    if (!d->record_staging(d->pending_storage, d->pending_bytes, destination, allocator))
        return false;
    d->pending_storage.release();
    d->pending_bytes = 0;
    return true;
}

bool VulkanIndependentWeightTransfer::record(const ncnn::Mat& source, ncnn::VkMat& destination,
                                             VulkanExpertWeightAllocator* allocator)
{
    if (d->failed || d->attempted || !d->pending_storage.empty() || source.empty() || source.dims != 1 || source.elempack != 1
        || (source.total() * source.elemsize) % 4 != 0 || !d->valid_allocator(allocator))
    {
        d->failed = true;
        return false;
    }
    try
    {
        ncnn::VkMat staging;
        staging.create_like(source, d->staging_allocator);
        if (staging.empty() || !staging.mapped_ptr())
        {
            d->failed = true;
            return false;
        }
        const size_t bytes = source.total() * source.elemsize;
        std::memcpy(staging.mapped_ptr(), source.data, bytes);
        if (!d->record_staging(staging, bytes, destination, allocator))
            return false;
        return true;
    }
    catch (...)
    {
        d->failed = true;
        destination.release();
        return false;
    }
}

bool VulkanIndependentWeightTransfer::submit_and_wait()
{
    if (d->failed || d->attempted || !d->pending_storage.empty() || d->destinations.empty()) return false;
    d->attempted = true;
    if (vkEndCommandBuffer(d->lane->transfer) != VK_SUCCESS
        || (d->independent_queue && vkEndCommandBuffer(d->lane->compute) != VK_SUCCESS))
    {
        d->failed = true;
        return false;
    }
    const auto& info = d->context->device()->info;
    d->transfer_submitted = d->submit_command(d->independent_queue ? info.transfer_queue_family_index() : info.compute_queue_family_index(),
                                              d->lane->transfer, d->lane->transfer_fence, false, d->independent_queue, d->transfer_queue);
    if (!d->transfer_submitted)
    {
        d->failed = true;
        return false;
    }
    const bool copied = d->wait_fence(d->lane->transfer_fence, d->transfer_family(), d->transfer_queue);
    if (!copied)
    {
        d->failed = true;
        return false;
    }
    if (d->independent_queue)
    {
        // Acquire visibility after copying completes, so unrelated foreground
        // kernels do not wait behind the upload on a compute queue.
        d->compute_submitted = d->submit_command(info.compute_queue_family_index(), d->lane->compute, d->lane->compute_fence, true, false, d->compute_queue);
        if (!d->compute_submitted)
        {
            d->failed = true;
            return false;
        }
        const bool visible = d->wait_fence(d->lane->compute_fence, info.compute_queue_family_index(), d->compute_queue);
        if (!visible)
        {
            d->failed = true;
            return false;
        }
    }
    // Fence writes availability -> host wait -> admission publication happens
    // before the consumer queueSubmit. The dependency chain covers later
    // accesses even if ncnn selects another compute queue.
    d->settled = true;
    for (const auto& destination : d->destinations)
    {
        destination.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        destination.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    // Completed commands can be reset for another admission even if a caller
    // retains this batch and its uploaded tensor owners.
    d->context->expert_transfer_pool().release(d->lane, true);
    d->lane = nullptr;
    return true;
}

} // namespace moe
} // namespace ncnn
#endif
