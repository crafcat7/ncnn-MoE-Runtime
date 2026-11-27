#include "backends/ncnn/core/transfer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>

namespace ncnn {
namespace moe {

// --- VkTransferPool ----------------------------------------------------------

struct VkTransferPool::Lane
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

    explicit Lane(const ncnn::VulkanDevice* dev)
        : device(dev),
          independent(!dev->info.unified_compute_transfer_queue())
    {
    }

    ~Lane()
    {
        const VkDevice vk = device->vkdevice();
        if (transfer) vkFreeCommandBuffers(vk, transfer_pool, 1, &transfer);
        if (compute) vkFreeCommandBuffers(vk, compute_pool, 1, &compute);
        if (ready) vkDestroySemaphore(vk, ready, nullptr);
        if (transfer_fence) vkDestroyFence(vk, transfer_fence, nullptr);
        if (compute_fence) vkDestroyFence(vk, compute_fence, nullptr);
        if (transfer_pool) vkDestroyCommandPool(vk, transfer_pool, nullptr);
        if (compute_pool) vkDestroyCommandPool(vk, compute_pool, nullptr);
    }

    bool make_cmd(uint32_t family, VkCommandPool& pool, VkCommandBuffer& cb, VkFence& fence)
    {
        const VkDevice vk = device->vkdevice();

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = family;
        if (vkCreateCommandPool(vk, &pool_info, nullptr, &pool) != VK_SUCCESS)
        {
            pool = VK_NULL_HANDLE;
            return false;
        }

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(vk, &alloc_info, &cb) != VK_SUCCESS)
        {
            cb = VK_NULL_HANDLE;
            return false;
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(vk, &fence_info, nullptr, &fence) != VK_SUCCESS)
        {
            fence = VK_NULL_HANDLE;
            return false;
        }

        return true;
    }

    bool init()
    {
        const auto& info = device->info;
        const uint32_t xfer_family = independent ? info.transfer_queue_family_index()
                                                 : info.compute_queue_family_index();
        if (!make_cmd(xfer_family, transfer_pool, transfer, transfer_fence))
            return false;

        if (independent)
        {
            if (!make_cmd(info.compute_queue_family_index(), compute_pool, compute, compute_fence))
                return false;

            VkSemaphoreCreateInfo sem_info{};
            sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            if (vkCreateSemaphore(device->vkdevice(), &sem_info, nullptr, &ready) != VK_SUCCESS)
            {
                ready = VK_NULL_HANDLE;
                return false;
            }
        }

        return true;
    }

    bool begin()
    {
        const VkDevice vk = device->vkdevice();
        if (vkResetCommandPool(vk, transfer_pool, 0) != VK_SUCCESS
            || vkResetFences(vk, 1, &transfer_fence) != VK_SUCCESS)
        {
            return false;
        }

        if (independent
            && (vkResetCommandPool(vk, compute_pool, 0) != VK_SUCCESS
                || vkResetFences(vk, 1, &compute_fence) != VK_SUCCESS))
        {
            return false;
        }

        VkCommandBufferBeginInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        return vkBeginCommandBuffer(transfer, &info) == VK_SUCCESS
               && (!independent || vkBeginCommandBuffer(compute, &info) == VK_SUCCESS);
    }
};

struct VkTransferPool::Impl
{
    struct Slot
    {
        std::unique_ptr<Lane> lane;
        bool in_use = false;
    };

    const ncnn::VulkanDevice* device;
    std::array<Slot, 2> slots;
    mutable std::mutex mutex;
    std::condition_variable available;

    explicit Impl(const ncnn::VulkanDevice* dev)
        : device(dev)
    {
    }
};

VkTransferPool::VkTransferPool(const ncnn::VulkanDevice* device)
    : d(std::make_unique<Impl>(device))
{
}

VkTransferPool::~VkTransferPool() = default;

VkTransferPool::Lane* VkTransferPool::acquire()
{
    std::unique_lock<std::mutex> lock(d->mutex);
    d->available.wait(lock, [&] {
        return !d->slots[0].in_use || !d->slots[1].in_use;
    });

    auto& slot = !d->slots[0].in_use ? d->slots[0] : d->slots[1];
    try
    {
        if (!slot.lane)
        {
            slot.lane = std::make_unique<Lane>(d->device);
            if (!slot.lane->init())
            {
                slot.lane.reset();
                return nullptr;
            }
        }

        if (!slot.lane->begin())
        {
            slot.lane.reset();
            return nullptr;
        }

        slot.in_use = true;
        return slot.lane.get();
    }
    catch (...)
    {
        slot.lane.reset();
        return nullptr;
    }
}

void VkTransferPool::release(Lane* lane, bool reusable) noexcept
{
    if (!lane)
        return;

    {
        const std::lock_guard<std::mutex> lock(d->mutex);
        for (auto& slot : d->slots)
        {
            if (slot.lane.get() != lane)
                continue;
            if (!reusable)
                slot.lane.reset();
            slot.in_use = false;
            break;
        }
    }

    d->available.notify_one();
}

// --- VkWeightTransfer --------------------------------------------------------

struct VkWeightTransfer::Impl
{
    std::shared_ptr<VulkanContext> context;
    ncnn::VkAllocator* staging_alloc;
    VkWeightAllocator* weight_alloc = nullptr;
    VkTransferPool::Lane* lane = nullptr;
    VkQueue transfer_queue = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;

    std::vector<ncnn::VkMat> staging;
    std::vector<ncnn::VkMat> destinations;
    ncnn::VkMat pending_storage;
    size_t pending_bytes = 0;
    uint64_t copy_bytes = 0;

    bool independent_queue = false;
    bool attempted = false;
    bool transfer_submitted = false;
    bool compute_submitted = false;
    bool visibility_only_wait = false;
    bool copy_accounted = false;
    bool settled = false;
    bool failed = false;

    Impl(std::shared_ptr<VulkanContext> ctx, ncnn::VkAllocator* staging_allocator)
        : context(std::move(ctx)),
          staging_alloc(staging_allocator),
          independent_queue(!context->device()->info.unified_compute_transfer_queue())
    {
    }

    bool submit_cb(uint32_t family, VkCommandBuffer cb, VkFence fence,
                   bool wait_sem, bool signal_sem, VkQueue& queue)
    {
        ncnn::VulkanDevice* dev = context->device();
        if (!queue)
            queue = dev->acquire_queue(family);
        if (!queue)
            return false;

        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &cb;
        info.waitSemaphoreCount = wait_sem ? 1 : 0;
        info.pWaitSemaphores = wait_sem ? &lane->ready : nullptr;
        info.pWaitDstStageMask = wait_sem ? &wait_stage : nullptr;
        info.signalSemaphoreCount = signal_sem ? 1 : 0;
        info.pSignalSemaphores = signal_sem ? &lane->ready : nullptr;

        // Retain exclusive queue lease on success so cleanup can idle it safely.
        const VkResult res = vkQueueSubmit(queue, 1, &info, fence);
        if (res != VK_SUCCESS)
        {
            dev->reclaim_queue(family, queue);
            queue = VK_NULL_HANDLE;
        }
        return res == VK_SUCCESS;
    }

    VkResult wait_fence_timed(VkFence fence)
    {
        const auto started = std::chrono::steady_clock::now();
        const VkResult res = vkWaitForFences(context->device()->vkdevice(), 1, &fence, VK_TRUE, UINT64_MAX);
        VulkanRuntimeState& stats = context->runtime_state();
        ++stats.weight_transfer_fence_waits;
        const auto delta = std::chrono::steady_clock::now() - started;
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(delta);
        stats.weight_transfer_wait_microseconds += static_cast<uint64_t>(elapsed.count());
        return res;
    }

    bool settle_queue(VkFence fence, uint32_t family, VkQueue& queue) noexcept
    {
        if (!queue)
            return false;

        VkResult res = fence != VK_NULL_HANDLE ? wait_fence_timed(fence) : VK_NOT_READY;
        while (res != VK_SUCCESS && res != VK_ERROR_DEVICE_LOST)
        {
            res = vkQueueWaitIdle(queue);
            if (res != VK_SUCCESS && res != VK_ERROR_DEVICE_LOST)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        context->device()->reclaim_queue(family, queue);
        queue = VK_NULL_HANDLE;
        return res == VK_SUCCESS;
    }

    bool record_copy(const ncnn::VkMat& src, size_t bytes, ncnn::VkMat& dst, VkWeightAllocator* alloc)
    {
        if (!lane)
        {
            lane = context->transfer_pool().acquire();
            if (!lane)
            {
                failed = true;
                return false;
            }
        }

        weight_alloc = alloc;
        try
        {
            dst.create_like(src, alloc);
            if (dst.empty() || src.allocator->flush(src.data) != 0)
            {
                failed = true;
                dst.release();
                return false;
            }

            staging.push_back(src);
            destinations.push_back(dst);

            VkBufferMemoryBarrier host_barrier{};
            host_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            host_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            host_barrier.buffer = src.buffer();
            host_barrier.offset = src.buffer_offset();
            host_barrier.size = bytes;
            vkCmdPipelineBarrier(lane->transfer, VK_PIPELINE_STAGE_HOST_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &host_barrier, 0, nullptr);

            VkBufferCopy copy{src.buffer_offset(), dst.buffer_offset(), bytes};
            vkCmdCopyBuffer(lane->transfer, src.buffer(), dst.buffer(), 1, &copy);

            VkBufferMemoryBarrier vis_barrier{};
            vis_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            vis_barrier.srcAccessMask = independent_queue ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
            vis_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vis_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vis_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vis_barrier.buffer = dst.buffer();
            vis_barrier.offset = dst.buffer_offset();
            vis_barrier.size = bytes;
            vkCmdPipelineBarrier(independent_queue ? lane->compute : lane->transfer,
                                 independent_queue ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &vis_barrier, 0, nullptr);

            copy_bytes += bytes;
            return true;
        }
        catch (...)
        {
            failed = true;
            dst.release();
            return false;
        }
    }

    ~Impl()
    {
        const auto& info = context->device()->info;
        const uint32_t xfer_family = independent_queue ? info.transfer_queue_family_index()
                                                       : info.compute_queue_family_index();
        const uint32_t comp_family = info.compute_queue_family_index();

        if (lane && !settled && (transfer_submitted || compute_submitted))
        {
            if (transfer_submitted)
            {
                const VkFence fence = visibility_only_wait ? VK_NULL_HANDLE : lane->transfer_fence;
                if (settle_queue(fence, xfer_family, transfer_queue) && !copy_accounted)
                {
                    context->runtime_state().weight_upload_bytes += copy_bytes;
                    copy_accounted = true;
                }
            }
            if (compute_submitted)
                settle_queue(lane->compute_fence, comp_family, compute_queue);
        }

        if (transfer_queue)
            context->device()->reclaim_queue(xfer_family, transfer_queue);
        if (compute_queue)
            context->device()->reclaim_queue(comp_family, compute_queue);

        pending_storage.release();
        staging.clear();
        destinations.clear();

        // Discard partially submitted failed lanes so an unconsumed semaphore is never reused.
        context->transfer_pool().release(lane, !failed && (!attempted || settled));
    }
};

VkWeightTransfer::VkWeightTransfer(std::shared_ptr<VulkanContext> ctx, ncnn::VkAllocator* staging_alloc)
    : d(std::make_unique<Impl>(std::move(ctx), staging_alloc))
{
}

VkWeightTransfer::~VkWeightTransfer() = default;

std::span<uint8_t> VkWeightTransfer::prepare_storage(size_t bytes, VkWeightAllocator* weight_alloc)
{
    if (d->failed || d->attempted || !d->pending_storage.empty()
        || bytes == 0 || bytes % 4 != 0
        || bytes > static_cast<size_t>(std::numeric_limits<int>::max())
        || (d->weight_alloc && d->weight_alloc != weight_alloc))
    {
        d->failed = true;
        return {};
    }

    try
    {
        d->pending_storage.create(static_cast<int>(bytes), sizeof(uint8_t), d->staging_alloc);
        if (d->pending_storage.empty() || !d->pending_storage.mapped_ptr())
        {
            d->failed = true;
            return {};
        }

        d->weight_alloc = weight_alloc;
        d->pending_bytes = bytes;
        return {static_cast<uint8_t*>(d->pending_storage.mapped_ptr()), bytes};
    }
    catch (...)
    {
        d->failed = true;
        return {};
    }
}

bool VkWeightTransfer::record_prepared_storage(ncnn::VkMat& dst, VkWeightAllocator* weight_alloc)
{
    if (d->failed || d->attempted || d->pending_storage.empty()
        || (d->weight_alloc && d->weight_alloc != weight_alloc))
    {
        d->failed = true;
        return false;
    }

    if (!d->record_copy(d->pending_storage, d->pending_bytes, dst, weight_alloc))
        return false;

    d->pending_storage.release();
    d->pending_bytes = 0;
    return true;
}

bool VkWeightTransfer::record(const ncnn::Mat& src, ncnn::VkMat& dst, VkWeightAllocator* weight_alloc)
{
    const size_t bytes = src.total() * src.elemsize;
    if (d->failed || d->attempted || !d->pending_storage.empty()
        || src.empty() || src.dims != 1 || src.elempack != 1 || bytes % 4 != 0
        || (d->weight_alloc && d->weight_alloc != weight_alloc))
    {
        d->failed = true;
        return false;
    }

    try
    {
        ncnn::VkMat staging;
        staging.create_like(src, d->staging_alloc);
        if (staging.empty() || !staging.mapped_ptr())
        {
            d->failed = true;
            return false;
        }

        std::memcpy(staging.mapped_ptr(), src.data, bytes);
        return d->record_copy(staging, bytes, dst, weight_alloc);
    }
    catch (...)
    {
        d->failed = true;
        dst.release();
        return false;
    }
}

bool VkWeightTransfer::submit_and_wait()
{
    if (d->failed || d->attempted || !d->pending_storage.empty() || d->destinations.empty())
        return false;

    d->attempted = true;
    if (vkEndCommandBuffer(d->lane->transfer) != VK_SUCCESS
        || (d->independent_queue && vkEndCommandBuffer(d->lane->compute) != VK_SUCCESS))
    {
        d->failed = true;
        return false;
    }

    const auto& info = d->context->device()->info;
    const uint32_t xfer_family = d->independent_queue ? info.transfer_queue_family_index()
                                                      : info.compute_queue_family_index();
    const uint32_t comp_family = info.compute_queue_family_index();

    d->visibility_only_wait = d->independent_queue && info.compute_queue_count() > 1;
    if (d->visibility_only_wait)
    {
        // Acquire compute before transfer (matching ncnn::VkTransfer) to avoid ABBA deadlock.
        d->compute_queue = d->context->device()->acquire_queue(comp_family);
        if (!d->compute_queue)
        {
            d->failed = true;
            return false;
        }
    }

    d->transfer_submitted = d->submit_cb(xfer_family, d->lane->transfer,
                                         d->visibility_only_wait ? VK_NULL_HANDLE : d->lane->transfer_fence,
                                         false, d->independent_queue, d->transfer_queue);
    if (!d->transfer_submitted)
    {
        d->failed = true;
        return false;
    }

    ++d->context->runtime_state().weight_transfer_submissions;

    if (!d->visibility_only_wait)
    {
        const VkResult res = d->wait_fence_timed(d->lane->transfer_fence);
        if (res == VK_SUCCESS || res == VK_ERROR_DEVICE_LOST)
        {
            d->context->device()->reclaim_queue(xfer_family, d->transfer_queue);
            d->transfer_queue = VK_NULL_HANDLE;
        }
        if (res != VK_SUCCESS)
        {
            d->failed = true;
            return false;
        }

        d->context->runtime_state().weight_upload_bytes += d->copy_bytes;
        d->copy_accounted = true;
    }

    if (d->independent_queue)
    {
        d->compute_submitted = d->submit_cb(comp_family, d->lane->compute,
                                            d->lane->compute_fence, true, false, d->compute_queue);
        if (!d->compute_submitted)
        {
            d->failed = true;
            return false;
        }

        ++d->context->runtime_state().weight_visibility_submissions;

        const VkResult res = d->wait_fence_timed(d->lane->compute_fence);
        if (res == VK_SUCCESS || res == VK_ERROR_DEVICE_LOST)
        {
            d->context->device()->reclaim_queue(comp_family, d->compute_queue);
            d->compute_queue = VK_NULL_HANDLE;
        }
        if (res != VK_SUCCESS)
        {
            d->failed = true;
            return false;
        }

        if (!d->copy_accounted)
        {
            d->context->runtime_state().weight_upload_bytes += d->copy_bytes;
            d->copy_accounted = true;
        }
        if (d->transfer_queue)
        {
            d->context->device()->reclaim_queue(xfer_family, d->transfer_queue);
            d->transfer_queue = VK_NULL_HANDLE;
        }
    }

    d->settled = true;
    for (const auto& dst : d->destinations)
    {
        dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    d->context->transfer_pool().release(d->lane, true);
    d->lane = nullptr;
    return true;
}

// --- VkWeightBatch -----------------------------------------------------------

static std::mutex legacy_queue_drain_mutex;

VkWeightBatch::VkWeightBatch(const std::shared_ptr<VulkanContext>& ctx,
                             ncnn::VkAllocator* borrowed_staging,
                             ncnn::Mat* borrowed_scratch)
    : context(ctx),
      borrowed_staging_allocator(borrowed_staging),
      borrowed_upload_scratch(borrowed_scratch)
{
}

VkWeightBatch::~VkWeightBatch() = default;

ncnn::Mat VkWeightBatch::host_storage(size_t bytes)
{
    if (failed || submitted || bytes == 0 || bytes > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {};

    constexpr size_t max_cached = 16 * 1024 * 1024;
    if (bytes > max_cached)
        return ncnn::Mat(static_cast<int>(bytes), sizeof(uint8_t));

    ncnn::Mat& scratch = borrowed_upload_scratch ? *borrowed_upload_scratch : upload_scratch;
    if (scratch.empty() || scratch.total() * scratch.elemsize < bytes)
        scratch.create(static_cast<int>(bytes), sizeof(uint8_t));

    if (scratch.empty())
        return {};
    return scratch.range(0, static_cast<int>(bytes));
}

std::span<uint8_t> VkWeightBatch::prepare_storage(size_t bytes, ncnn::VkAllocator* weight_alloc)
{
    auto* alloc = dynamic_cast<VkWeightAllocator*>(weight_alloc);
    if (failed || submitted || !alloc || cmd)
    {
        failed = true;
        return {};
    }

    try
    {
        if (!borrowed_staging_allocator && !staging_allocator)
            staging_allocator = std::make_unique<ncnn::VkWeightStagingAllocator>(context->device());

        ncnn::VkAllocator* staging = borrowed_staging_allocator ? borrowed_staging_allocator : staging_allocator.get();
        if (!independent_cmd)
            independent_cmd = std::make_unique<VkWeightTransfer>(context, staging);

        const auto span = independent_cmd->prepare_storage(bytes, alloc);
        if (!span.empty())
            return span;
    }
    catch (...)
    {
    }

    failed = true;
    return {};
}

bool VkWeightBatch::record_prepared_storage(ncnn::VkMat& dst, ncnn::VkAllocator* weight_alloc)
{
    auto* alloc = dynamic_cast<VkWeightAllocator*>(weight_alloc);
    if (failed || submitted || !independent_cmd || !alloc
        || !independent_cmd->record_prepared_storage(dst, alloc))
    {
        failed = true;
        return false;
    }
    return true;
}

bool VkWeightBatch::record(const ncnn::Mat& src, ncnn::VkMat& dst,
                           const ncnn::Option& opt, ncnn::VkAllocator* weight_alloc)
{
    if (failed || submitted || src.empty())
    {
        failed = true;
        return false;
    }

    try
    {
        if (!borrowed_staging_allocator && !staging_allocator)
            staging_allocator = std::make_unique<ncnn::VkWeightStagingAllocator>(context->device());

        ncnn::VkAllocator* staging = borrowed_staging_allocator ? borrowed_staging_allocator : staging_allocator.get();
        if (auto* independent_alloc = dynamic_cast<VkWeightAllocator*>(weight_alloc))
        {
            if (cmd)
            {
                failed = true;
                return false;
            }
            if (!independent_cmd)
                independent_cmd = std::make_unique<VkWeightTransfer>(context, staging);
            if (!independent_cmd->record(src, dst, independent_alloc))
            {
                failed = true;
                return false;
            }
        }
        else
        {
            if (independent_cmd)
            {
                failed = true;
                return false;
            }

            if (!cmd)
            {
                const auto& info = context->device()->info;
                const size_t qcount = std::max(info.compute_queue_count(), info.transfer_queue_count());
                if (qcount == 0)
                {
                    failed = true;
                    return false;
                }

                // Preallocate before submit so error recovery never allocates.
                legacy_queue_scratch.resize(qcount, VK_NULL_HANDLE);
                command_lock = std::unique_lock<std::mutex>(context->command_mutex());
                cmd = std::make_unique<ncnn::VkTransfer>(context->device());
            }

            ncnn::Option upload_opt = opt;
            upload_opt.blob_vkallocator = weight_alloc;
            upload_opt.workspace_vkallocator = weight_alloc;
            upload_opt.staging_vkallocator = staging;
            cmd->record_upload(src, dst, upload_opt);
        }

        if (!dst.empty())
            return true;
    }
    catch (...)
    {
        dst.release();
    }

    failed = true;
    return false;
}

void VkWeightBatch::settle_legacy_upload() noexcept
{
    const std::lock_guard<std::mutex> lock(legacy_queue_drain_mutex);
    const ncnn::VulkanDevice* dev = context->device();
    const auto& info = dev->info;

    // Drain and release transfer family before acquiring compute to avoid cross-family deadlock.
    if (!info.unified_compute_transfer_queue())
        drain_queue_family(dev, info.transfer_queue_family_index(),
                           std::span<VkQueue>(legacy_queue_scratch.data(), info.transfer_queue_count()));

    drain_queue_family(dev, info.compute_queue_family_index(),
                       std::span<VkQueue>(legacy_queue_scratch.data(), info.compute_queue_count()));
}

bool VkWeightBatch::submit()
{
    if (failed || submitted)
        return false;

    submitted = true;
    const bool ok = independent_cmd ? independent_cmd->submit_and_wait()
                                    : (cmd && cmd->submit_and_wait() == 0);
    if (!ok && cmd && !independent_cmd)
        settle_legacy_upload();

    failed = !ok;
    return ok;
}

void VkWeightBatch::retain_owner(std::shared_ptr<const void> owner)
{
    if (owner)
        retained_owners.push_back(std::move(owner));
}

} // namespace moe
} // namespace ncnn
