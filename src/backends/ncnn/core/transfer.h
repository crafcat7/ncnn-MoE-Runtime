#ifndef NCNN_MOE_BACKENDS_NCNN_CORE_TRANSFER_H
#define NCNN_MOE_BACKENDS_NCNN_CORE_TRANSFER_H

#include "backends/ncnn/vulkan.h"

#if NCNN_MOE_WITH_VULKAN
#include "backends/ncnn/core/context.h"
#include "backends/ncnn/core/memory.h"

#include <mat.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

// Two isolated DMA lanes retaining pools, command buffers, fences, and
// semaphores. Leases keep their VulkanContext alive; this pool owns none.
class VkTransferPool
{
public:
    struct Lane;

    explicit VkTransferPool(const ncnn::VulkanDevice* device);
    ~VkTransferPool();

    VkTransferPool(const VkTransferPool&) = delete;
    VkTransferPool& operator=(const VkTransferPool&) = delete;

    [[nodiscard]] Lane* acquire();
    void release(Lane* lane, bool reusable) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

// Async weight DMA batch leasing one VkTransferPool lane, followed by a short
// compute-queue visibility barrier before weights are published.
class VkWeightTransfer
{
public:
    VkWeightTransfer(std::shared_ptr<VulkanContext> ctx, ncnn::VkAllocator* staging_alloc);
    ~VkWeightTransfer();

    bool record(const ncnn::Mat& src, ncnn::VkMat& dst, VkWeightAllocator* weight_alloc);

    // One mapped reservation at a time; valid until record_prepared_storage() or destruction.
    [[nodiscard]] std::span<uint8_t> prepare_storage(size_t bytes, VkWeightAllocator* weight_alloc);
    bool record_prepared_storage(ncnn::VkMat& dst, VkWeightAllocator* weight_alloc);
    bool submit_and_wait();

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

// Bounds one group of immutable weight uploads. Uses VkWeightTransfer for
// VkWeightAllocator targets and falls back to ncnn::VkTransfer for legacy pools.
class VkWeightBatch
{
public:
    explicit VkWeightBatch(const std::shared_ptr<VulkanContext>& ctx,
                           ncnn::VkAllocator* borrowed_staging = nullptr,
                           ncnn::Mat* borrowed_scratch = nullptr);
    ~VkWeightBatch();

    VkWeightBatch(const VkWeightBatch&) = delete;
    VkWeightBatch& operator=(const VkWeightBatch&) = delete;

    [[nodiscard]] bool record(const ncnn::Mat& src, ncnn::VkMat& dst,
                              const ncnn::Option& opt, ncnn::VkAllocator* weight_alloc);
    [[nodiscard]] std::span<uint8_t> prepare_storage(size_t bytes, ncnn::VkAllocator* weight_alloc);
    [[nodiscard]] bool record_prepared_storage(ncnn::VkMat& dst, ncnn::VkAllocator* weight_alloc);
    [[nodiscard]] bool submit();

    // Reuses up to 16 MiB of host scratch across uploads; larger requests are transient.
    [[nodiscard]] ncnn::Mat host_storage(size_t bytes);

    // Keeps failed-factory pipeline owners alive until commands settle and locks release.
    void retain_owner(std::shared_ptr<const void> owner);

private:
    void settle_legacy_upload() noexcept;

    std::shared_ptr<VulkanContext> context;
    std::vector<VkQueue> legacy_queue_scratch;
    // Declared before cmd so allocator outlives VkTransfer's retained staging buffers.
    std::unique_ptr<ncnn::VkWeightStagingAllocator> staging_allocator;
    ncnn::VkAllocator* borrowed_staging_allocator = nullptr;
    ncnn::Mat* borrowed_upload_scratch = nullptr;
    ncnn::Mat upload_scratch;
    std::vector<std::shared_ptr<const void>> retained_owners;
    // Destroy cmd before unlocking command_lock.
    std::unique_lock<std::mutex> command_lock;
    std::unique_ptr<ncnn::VkTransfer> cmd;
    std::unique_ptr<VkWeightTransfer> independent_cmd;
    bool failed = false;
    bool submitted = false;
};

} // namespace moe
} // namespace ncnn
#endif // NCNN_MOE_WITH_VULKAN

#endif // NCNN_MOE_BACKENDS_NCNN_CORE_TRANSFER_H
