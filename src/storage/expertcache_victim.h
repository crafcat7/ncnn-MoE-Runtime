#ifndef NCNN_MOE_EXPERTCACHE_VICTIM_H
#define NCNN_MOE_EXPERTCACHE_VICTIM_H

#include "ncnn/moe/types.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
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

struct ExpertKeySize
{
    std::string key;
    uint64_t size = 0;
};

class ExpertVictimCache
{
public:
    virtual ~ExpertVictimCache() = default;

    // Admission is asynchronous and best effort.
    virtual void admit(std::string key, std::shared_ptr<const TensorData> gate_up, std::shared_ptr<const TensorData> down, ExpertVictimExecutionMetadata execution = {}) = 0;
    [[nodiscard]] virtual std::optional<ExpertVictimPair> restore(const std::string& key, const TensorData& gate_up_source, const TensorData& down_source) = 0;
    virtual void wait_for_background_work() = 0;
    [[nodiscard]] virtual ExpertVictimCacheStatistics statistics() const = 0;
    [[nodiscard]] virtual uint64_t capacity() const noexcept = 0;
};

class ReuseFilteredExpertVictimCache final : public ExpertVictimCache
{
public:
    ReuseFilteredExpertVictimCache(std::shared_ptr<ExpertVictimCache> _inner, uint32_t _reuse_probe_interval);

    void admit(
        std::string key,
        std::shared_ptr<const TensorData> gate_up,
        std::shared_ptr<const TensorData> down,
        ExpertVictimExecutionMetadata execution) override;

    std::optional<ExpertVictimPair> restore(
        const std::string& key,
        const TensorData& gate_up_source,
        const TensorData& down_source) override;

    void wait_for_background_work() override;

    ExpertVictimCacheStatistics statistics() const override;

    uint64_t capacity() const noexcept override;

private:
    static bool add_size(uint64_t& size, size_t count);

    static uint64_t payload_size(const std::shared_ptr<const TensorData>& gate_up, const std::shared_ptr<const TensorData>& down);

    void trim_history_locked();

    std::shared_ptr<ExpertVictimCache> inner;
    uint32_t reuse_probe_interval = 1;
    std::mutex history_mutex;
    std::list<ExpertKeySize> history;
    std::unordered_map<std::string, std::list<ExpertKeySize>::iterator> history_index;
    uint64_t history_size = 0;
    std::atomic<uint64_t> reuse_probe_ticket{0};
    std::atomic<uint64_t> filtered_admissions{0};
    std::atomic<uint64_t> reused_admissions{0};
    std::atomic<uint64_t> probe_admissions{0};
};

class ShardedExpertVictimCache final : public ExpertVictimCache
{
public:
    explicit ShardedExpertVictimCache(std::vector<std::shared_ptr<ExpertVictimCache>> _shards);

    void admit(
        std::string key,
        std::shared_ptr<const TensorData> gate_up,
        std::shared_ptr<const TensorData> down,
        ExpertVictimExecutionMetadata execution) override;

    std::optional<ExpertVictimPair> restore(
        const std::string& key,
        const TensorData& gate_up_source,
        const TensorData& down_source) override;

    void wait_for_background_work() override;

    ExpertVictimCacheStatistics statistics() const override;

    uint64_t capacity() const noexcept override;

private:
    ExpertVictimCache& shard(const std::string& key) const;

    std::vector<std::shared_ptr<ExpertVictimCache>> shards;
    uint64_t total_size = 0;
};

// Payload-bounded reuse filter for upper-cache evictions.
[[nodiscard]] std::shared_ptr<ExpertVictimCache> create_reuse_victim_cache(std::shared_ptr<ExpertVictimCache> inner, uint32_t reuse_probe_interval);

// Deterministic sharding across independent cache devices.
[[nodiscard]] std::shared_ptr<ExpertVictimCache> create_sharded_victim_cache(std::vector<std::shared_ptr<ExpertVictimCache>> shards);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERTCACHE_VICTIM_H
