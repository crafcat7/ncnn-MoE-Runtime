#include "graphallocator_vulkan.h"

#if NCNN_MOE_WITH_VULKAN
#include "experttransfer_vulkan.h"

#include <limits>

namespace ncnn {
namespace moe {

VulkanGraphAllocator::VulkanGraphAllocator(const ncnn::VulkanDevice* device)
    : ncnn::VkAllocator(device),
      pool(std::make_unique<VulkanExpertWeightAllocator>(device, 64 * 1024))
{
    // Device-only buffers use explicit staging even on unified-memory devices.
    mappable = false;
    coherent = false;
    cached = false;
    pool->configure_graph_pool(true);
}

VulkanGraphAllocator::~VulkanGraphAllocator() = default;

ncnn::VkBufferMemory* VulkanGraphAllocator::fastMalloc(size_t size)
{
    const std::lock_guard<std::mutex> lock(mutex);
    ncnn::VkBufferMemory* buffer = pool->fastMalloc(size);
    buffer_memory_type_index = pool->buffer_memory_type_index;
    return buffer;
}

void VulkanGraphAllocator::fastFree(ncnn::VkBufferMemory* buffer)
{
    // The pool serializes frees; graph reservation state is untouched.
    pool->fastFree(buffer);
}

ncnn::VkImageMemory* VulkanGraphAllocator::fastMalloc(int, int, int, size_t, int)
{
    return nullptr;
}

void VulkanGraphAllocator::fastFree(ncnn::VkImageMemory*)
{
}

void VulkanGraphAllocator::clear()
{
    const std::lock_guard<std::mutex> lock(mutex);
    pool->clear();
    preparation_physical_size = 0;
    reserved_compute_size = 0;
    // Clearing a previously sealed arena must not silently reopen runtime
    // physical growth. A later explicit reservation can populate it again.
    pool->configure_graph_pool(!reserved);
}

bool VulkanGraphAllocator::reserve_graph_compute(uint64_t bytes)
{
    const std::lock_guard<std::mutex> lock(mutex);
    if (bytes > std::numeric_limits<size_t>::max())
        return false;
    if (reserved && bytes <= reserved_compute_size)
        return true;
    if (!reserved)
        preparation_physical_size = pool->physical_bytes();

    pool->configure_graph_pool(true);
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
        pool->configure_graph_pool(false);
        return false;
    }
    pool->configure_graph_pool(false);
    if (bytes != 0 && !warm)
        return false;
    reserved_compute_size = bytes;
    reserved = true;
    buffer_memory_type_index = pool->buffer_memory_type_index;
    return true;
}

VulkanGraphArenaSize VulkanGraphAllocator::physical_size() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    const uint64_t total = pool->physical_bytes();
    const uint64_t preparation = reserved ? preparation_physical_size : total;
    return {preparation, total};
}

} // namespace moe
} // namespace ncnn
#endif // NCNN_MOE_WITH_VULKAN
