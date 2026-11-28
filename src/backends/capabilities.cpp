#include "backends/capabilities.h"

#include "backends/ncnn/vulkan.h"

#include <vector>

namespace ncnn {
namespace moe {

BackendCapabilities query_backend_capabilities(uint32_t vulkan_device_index) noexcept
{
    BackendCapabilities result;
    result.cpu = cpu_backend_caps();
#if defined(NCNN_MOE_WITH_VULKAN) && NCNN_MOE_WITH_VULKAN
    try
    {
        static const std::vector<GpuInfo> cached_gpus = get_gpu_infos();
        if (!cached_gpus.empty())
        {
            uint32_t chosen = vulkan_device_index;
            if (chosen >= cached_gpus.size())
            {
                const uint32_t default_gpu = get_default_gpu_index();
                chosen = default_gpu < cached_gpus.size() ? default_gpu : 0;
            }
            const GpuInfo& gpu = cached_gpus[chosen];
            result.vulkan.available = true;
            result.vulkan.device_index = gpu.device_index;
            result.vulkan.device_type = gpu.type;
            result.vulkan.unified_queue = has_flag(gpu.flags, VulkanDeviceUnifiedComputeTransfer);
            result.vulkan.resizable_bar = has_flag(gpu.flags, VulkanDeviceResizableBar);
            result.vulkan.fp16_storage = has_flag(gpu.flags, VulkanDeviceFp16Storage);
            result.vulkan.fp16_arithmetic = has_flag(gpu.flags, VulkanDeviceFp16Arithmetic);
            result.vulkan.bf16_storage = has_flag(gpu.flags, VulkanDeviceBf16Storage);
            result.vulkan.bf16_arithmetic = has_flag(gpu.flags, VulkanDeviceBf16Arithmetic);
            result.vulkan.int8_storage = has_flag(gpu.flags, VulkanDeviceInt8Storage);
            result.vulkan.int8_arithmetic = has_flag(gpu.flags, VulkanDeviceInt8Arithmetic);
            result.vulkan.integer_dot = has_flag(gpu.flags, VulkanDeviceIntegerDotProduct);
            result.vulkan.cooperative_matrix = has_flag(gpu.flags, VulkanDeviceCooperativeMatrix);
            result.vulkan.int8_cooperative_matrix = has_flag(gpu.flags, VulkanDeviceInt8CooperativeMatrix);
            result.vulkan.heap_budget_bytes = gpu.heap_budget;
            result.vulkan.heap_available_bytes = gpu.heap_available;
        }
    }
    catch (...)
    {
    }
#else
    (void)vulkan_device_index;
#endif
    return result;
}

size_t resolve_vulkan_expert_min_rows(const BackendCapabilities& caps, HybridMode hybrid_mode) noexcept
{
    if (hybrid_mode == HybridMode::HybridExperts)
        return 1;
    if (caps.vulkan.available && caps.vulkan.unified_queue)
    {
        return 1;
    }
    return 2;
}

} // namespace moe
} // namespace ncnn
