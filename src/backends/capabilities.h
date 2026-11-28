#ifndef NCNN_MOE_BACKENDS_CAPABILITIES_H
#define NCNN_MOE_BACKENDS_CAPABILITIES_H

#include "kernels/cpu/cpu_caps.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>

namespace ncnn {
namespace moe {

struct VulkanBackendCaps
{
    bool available = false;
    uint32_t device_index = 0;
    VulkanDeviceType device_type = VulkanDeviceType::Unknown;
    bool unified_queue = false;
    bool resizable_bar = false;
    bool fp16_storage = false;
    bool fp16_arithmetic = false;
    bool bf16_storage = false;
    bool bf16_arithmetic = false;
    bool int8_storage = false;
    bool int8_arithmetic = false;
    bool integer_dot = false;
    bool cooperative_matrix = false;
    bool int8_cooperative_matrix = false;
    uint64_t heap_budget_bytes = 0;
    uint64_t heap_available_bytes = 0;
};

struct BackendCapabilities
{
    CpuBackendCaps cpu;
    VulkanBackendCaps vulkan;
};

[[nodiscard]] BackendCapabilities query_backend_capabilities(uint32_t vulkan_device_index = automatic_vulkan_device_index) noexcept;

[[nodiscard]] size_t resolve_vulkan_expert_min_rows(const BackendCapabilities& caps,
                                                    HybridMode hybrid_mode) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_BACKENDS_CAPABILITIES_H
