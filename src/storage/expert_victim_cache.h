#ifndef NCNN_MOE_EXPERT_VICTIM_CACHE_H
#define NCNN_MOE_EXPERT_VICTIM_CACHE_H

#include "ncnn/moe/types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ncnn {
namespace moe {

struct ExpertVictimPair
{
    std::shared_ptr<TensorData> gate_up;
    std::shared_ptr<TensorData> down;
};

struct ExpertVictimCacheStatistics
{
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t stores = 0;
    uint64_t evictions = 0;
    uint64_t dropped_admissions = 0;
    uint64_t restore_failures = 0;
    uint64_t bytes_uploaded = 0;
    uint64_t bytes_downloaded = 0;
    uint64_t restore_time_microseconds = 0;
    uint64_t mapped_stores = 0;
    uint64_t mapped_restores = 0;
    uint64_t resident_bytes = 0;
    uint64_t pending_bytes = 0;
};

class IExpertVictimCache
{
public:
    virtual ~IExpertVictimCache() = default;

    // Admission is best effort and may complete asynchronously. Implementations
    // must bound any retained host memory independently from device capacity.
    virtual void admit(
        std::string key,
        std::shared_ptr<const TensorData> gate_up,
        std::shared_ptr<const TensorData> down) = 0;
    [[nodiscard]] virtual std::optional<ExpertVictimPair> restore(
        const std::string& key,
        const TensorData& gate_up_source,
        const TensorData& down_source) = 0;
    [[nodiscard]] virtual ExpertVictimCacheStatistics statistics() const = 0;
    [[nodiscard]] virtual uint64_t capacity_bytes() const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<IExpertVictimCache>
create_vulkan_expert_victim_cache(uint64_t capacity_bytes);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_VICTIM_CACHE_H
