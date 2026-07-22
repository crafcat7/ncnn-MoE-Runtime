#ifndef NCNN_MOE_CPU_SESSION_STATE_H
#define NCNN_MOE_CPU_SESSION_STATE_H

#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

class NcnnVulkanAttentionCache;

struct CpuLayerCache
{
    std::vector<float> keys;
    std::vector<float> values;
    std::vector<uint16_t> bfloat16_keys;
    std::vector<uint16_t> bfloat16_values;
    uint64_t start_position = 0;
    uint64_t token_count = 0;
    uint64_t first_slot = 0;
    uint64_t capacity_tokens = 0;
    uint32_t columns = 0;
    DType dtype = DType::Float32;
    std::shared_ptr<NcnnVulkanAttentionCache> vulkan_attention_cache;
    uint64_t device_allocated_bytes = 0;

    [[nodiscard]] uint64_t allocated_bytes() const noexcept
    {
        return static_cast<uint64_t>(keys.size() + values.size()) * sizeof(float)
               + static_cast<uint64_t>(bfloat16_keys.size() + bfloat16_values.size()) * sizeof(uint16_t)
               + device_allocated_bytes;
    }

    [[nodiscard]] uint64_t logical_bytes() const noexcept
    {
        const uint64_t element_size = dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
        return token_count * columns * element_size * 2;
    }
};

class CpuSessionState
{
public:
    std::vector<CpuLayerCache> layers;

    [[nodiscard]] uint64_t kv_cache_allocated_bytes() const noexcept
    {
        uint64_t bytes = 0;
        for (const CpuLayerCache& layer : layers)
            bytes += layer.allocated_bytes();
        return bytes;
    }

    [[nodiscard]] uint64_t kv_cache_logical_bytes() const noexcept
    {
        uint64_t bytes = 0;
        for (const CpuLayerCache& layer : layers)
            bytes += layer.logical_bytes();
        return bytes;
    }
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_SESSION_STATE_H
