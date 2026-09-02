#ifndef NCNN_MOE_EXPERT_VICTIM_CACHE_H
#define NCNN_MOE_EXPERT_VICTIM_CACHE_H

#include "ncnn/moe/types.h"
#include "ncnn/moe/vulkan_context.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

struct ExpertVictimPair
{
    std::shared_ptr<TensorData> gate_up;
    std::shared_ptr<TensorData> down;
};

struct ExpertVictimExecutionMetadata
{
    const TensorData* gate_up_bias = nullptr;
    const TensorData* down_bias = nullptr;
    float activation_limit = 0.0f;
    bool enabled = false;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
};

struct ExpertVictimCacheStatistics
{
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t filtered_admissions = 0;
    uint64_t reused_admissions = 0;
    uint64_t probe_admissions = 0;
    uint64_t stores = 0;
    uint64_t evictions = 0;
    uint64_t dropped_admissions = 0;
    uint64_t restore_failures = 0;
    uint64_t bytes_uploaded = 0;
    uint64_t bytes_downloaded = 0;
    uint64_t restore_time_microseconds = 0;
    uint64_t mapped_stores = 0;
    uint64_t mapped_restores = 0;
    uint64_t resident_size = 0;
    uint64_t pending_size = 0;
};

class IExpertVictimCache
{
public:
    virtual ~IExpertVictimCache() = default;

    // Admission is asynchronous and best effort.
    virtual void admit(std::string key, std::shared_ptr<const TensorData> gate_up, std::shared_ptr<const TensorData> down, uint32_t residency_group = std::numeric_limits<uint32_t>::max(), ExpertVictimExecutionMetadata execution = {}) = 0;
    [[nodiscard]] virtual std::optional<ExpertVictimPair> restore(const std::string& key, const TensorData& gate_up_source, const TensorData& down_source) = 0;
    virtual void wait_for_background_work() = 0;
    [[nodiscard]] virtual ExpertVictimCacheStatistics statistics() const = 0;
    [[nodiscard]] virtual uint64_t capacity() const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<IExpertVictimCache> create_vulkan_victim_cache(uint64_t cache_size, uint32_t device_index,
                                                                             const NcnnVulkanContextInstancePtr& context_instance,
                                                                             uint64_t optimization_flags);

// Payload-bounded reuse filter for upper-cache evictions.
[[nodiscard]] std::shared_ptr<IExpertVictimCache> create_reuse_victim_cache(std::shared_ptr<IExpertVictimCache> inner, uint32_t reuse_probe_interval);

// Deterministic sharding across independent cache devices.
[[nodiscard]] std::shared_ptr<IExpertVictimCache> create_sharded_victim_cache(std::vector<std::shared_ptr<IExpertVictimCache>> shards);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_VICTIM_CACHE_H
