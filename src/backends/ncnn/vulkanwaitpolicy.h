#ifndef NCNN_MOE_VULKANWAITPOLICY_H
#define NCNN_MOE_VULKANWAITPOLICY_H

#include <cstddef>

namespace ncnn {
namespace moe {

enum class VulkanCompletionProbeResult
{
    Retry,
    DeviceLost
};

// A probe uses an exclusively leased queue, which can differ from the original
// submission's queue. Only device loss permits abandoning the original fence.
// Success or a transient error on the probe says nothing about that fence.
constexpr VulkanCompletionProbeResult classify_vulkan_completion_probe(int result, int device_lost) noexcept
{
    return result == device_lost ? VulkanCompletionProbeResult::DeviceLost : VulkanCompletionProbeResult::Retry;
}

// Native VkCompute::wait collapses all fence errors to -1. Keep the command and
// all caller-owned resources alive through retries; ordinary failure is safe to
// return only after a separate probe confirms that the device has been lost.
struct VulkanCompletionCallbacks
{
    void* context;
    int (*wait)(void*);
    VulkanCompletionProbeResult (*probe)(void*);
    void (*pause)(void*);
};

inline int wait_for_vulkan_submission_completion(const VulkanCompletionCallbacks& callbacks)
{
    for (;;)
    {
        const int result = callbacks.wait(callbacks.context);
        if (result == 0)
            return 0;
        if (callbacks.probe(callbacks.context) == VulkanCompletionProbeResult::DeviceLost)
            return result;
        callbacks.pause(callbacks.context);
    }
}

// Native VkTransfer returns its queue leases before reporting a submit/wait
// error. Preallocated slots acquire an entire family to cover its unknown
// original queue. Keep every lease until all queues in this group have settled.
struct VulkanQueueFamilyDrainCallbacks
{
    void* context;
    size_t queue_count;
    void (*acquire)(void*, size_t);
    int (*wait_idle)(void*, size_t);
    void (*reclaim)(void*, size_t);
    void (*pause)(void*);
};

inline void drain_vulkan_queue_family(const VulkanQueueFamilyDrainCallbacks& callbacks,
                                      int success_result, int device_lost_result)
{
    for (size_t index = 0; index < callbacks.queue_count; ++index)
        callbacks.acquire(callbacks.context, index);
    for (size_t index = 0; index < callbacks.queue_count; ++index)
    {
        for (;;)
        {
            const int result = callbacks.wait_idle(callbacks.context, index);
            if (result == success_result || result == device_lost_result)
                break;
            callbacks.pause(callbacks.context);
        }
    }
    for (size_t index = 0; index < callbacks.queue_count; ++index)
        callbacks.reclaim(callbacks.context, index);
}

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_VULKANWAITPOLICY_H
