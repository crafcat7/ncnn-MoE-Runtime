#include "expert_victim_cache.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <list>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace ncnn {
namespace moe {

static void add_statistics(ExpertVictimCacheStatistics& destination, const ExpertVictimCacheStatistics& source)
{
    destination.hits += source.hits;
    destination.misses += source.misses;
    destination.admissions += source.admissions;
    destination.filtered_admissions += source.filtered_admissions;
    destination.reused_admissions += source.reused_admissions;
    destination.probe_admissions += source.probe_admissions;
    destination.stores += source.stores;
    destination.evictions += source.evictions;
    destination.dropped_admissions += source.dropped_admissions;
    destination.restore_failures += source.restore_failures;
    destination.bytes_uploaded += source.bytes_uploaded;
    destination.bytes_downloaded += source.bytes_downloaded;
    destination.restore_time_microseconds += source.restore_time_microseconds;
    destination.mapped_stores += source.mapped_stores;
    destination.mapped_restores += source.mapped_restores;
    destination.resident_size += source.resident_size;
    destination.pending_size += source.pending_size;
}

class ReuseFilteredExpertVictimCache final : public IExpertVictimCache
{
public:
    ReuseFilteredExpertVictimCache(std::shared_ptr<IExpertVictimCache> _inner, uint32_t _reuse_probe_interval)
        : inner(std::move(_inner)),
          reuse_probe_interval(std::max(1u, _reuse_probe_interval))
    {
    }

    void admit(
        std::string key,
        std::shared_ptr<const TensorData> gate_up,
        std::shared_ptr<const TensorData> down,
        uint32_t residency_group,
        ExpertVictimExecutionMetadata execution) override
    {
        bool reused = false;
        if (reuse_probe_interval > 1)
        {
            const uint64_t size = payload_size(gate_up, down);
            const std::lock_guard<std::mutex> lock(history_mutex);
            const auto existing = history_index.find(key);
            if (existing != history_index.end())
            {
                reused = true;
                history.splice(history.end(), history, existing->second);
                existing->second = std::prev(history.end());
            }
            else if (size != 0 && size <= inner->capacity())
            {
                history.push_back({key, size});
                const auto position = std::prev(history.end());
                history_index[position->key] = position;
                history_size += size;
                trim_history_locked();
            }
        }
        if (!reused && reuse_probe_interval > 1)
        {
            const uint64_t ticket = reuse_probe_ticket.fetch_add(1, std::memory_order_relaxed);
            if (ticket % reuse_probe_interval != 0)
            {
                filtered_admissions.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        if (reused)
        {
            reused_admissions.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            probe_admissions.fetch_add(1, std::memory_order_relaxed);
        }
        inner->admit(std::move(key), std::move(gate_up), std::move(down), residency_group, execution);
    }

    std::optional<ExpertVictimPair> restore(
        const std::string& key,
        const TensorData& gate_up_source,
        const TensorData& down_source) override
    {
        return inner->restore(key, gate_up_source, down_source);
    }

    void wait_for_background_work() override
    {
        inner->wait_for_background_work();
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        ExpertVictimCacheStatistics result = inner->statistics();
        result.filtered_admissions += filtered_admissions.load(std::memory_order_relaxed);
        result.reused_admissions += reused_admissions.load(std::memory_order_relaxed);
        result.probe_admissions += probe_admissions.load(std::memory_order_relaxed);
        return result;
    }

    uint64_t capacity() const noexcept override
    {
        return inner->capacity();
    }

private:
    struct HistoryRecord
    {
        std::string key;
        uint64_t size = 0;
    };

    static bool add_size(uint64_t& size, size_t count)
    {
        if (count > std::numeric_limits<uint64_t>::max() - size)
            return false;
        size += static_cast<uint64_t>(count);
        return true;
    }

    static uint64_t payload_size(const std::shared_ptr<const TensorData>& gate_up, const std::shared_ptr<const TensorData>& down)
    {
        if (!gate_up || !down)
            return 0;
        uint64_t size = 0;
        if (!add_size(size, gate_up->mxfp4_blocks.size())
            || !add_size(size, gate_up->mxfp4_scales.size())
            || !add_size(size, down->mxfp4_blocks.size())
            || !add_size(size, down->mxfp4_scales.size()))
            return 0;
        return size;
    }

    void trim_history_locked()
    {
        const uint64_t cache_size = inner->capacity();
        while (history_size > cache_size && !history.empty())
        {
            history_size -= history.front().size;
            history_index.erase(history.front().key);
            history.pop_front();
        }
    }

    std::shared_ptr<IExpertVictimCache> inner;
    uint32_t reuse_probe_interval = 1;
    std::mutex history_mutex;
    std::list<HistoryRecord> history;
    std::unordered_map<std::string, std::list<HistoryRecord>::iterator> history_index;
    uint64_t history_size = 0;
    std::atomic<uint64_t> reuse_probe_ticket{0};
    std::atomic<uint64_t> filtered_admissions{0};
    std::atomic<uint64_t> reused_admissions{0};
    std::atomic<uint64_t> probe_admissions{0};
};

class ShardedExpertVictimCache final : public IExpertVictimCache
{
public:
    explicit ShardedExpertVictimCache(std::vector<std::shared_ptr<IExpertVictimCache>> _shards)
        : shards(std::move(_shards))
    {
        for (const auto& item : shards)
            total_size += item->capacity();
    }

    void admit(
        std::string key,
        std::shared_ptr<const TensorData> gate_up,
        std::shared_ptr<const TensorData> down,
        uint32_t residency_group,
        ExpertVictimExecutionMetadata execution) override
    {
        shard(key).admit(std::move(key), std::move(gate_up), std::move(down), residency_group, execution);
    }

    std::optional<ExpertVictimPair> restore(
        const std::string& key,
        const TensorData& gate_up_source,
        const TensorData& down_source) override
    {
        return shard(key).restore(key, gate_up_source, down_source);
    }

    void wait_for_background_work() override
    {
        for (const auto& item : shards)
            item->wait_for_background_work();
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        ExpertVictimCacheStatistics aggregate;
        for (const auto& item : shards)
        {
            add_statistics(aggregate, item->statistics());
        }
        return aggregate;
    }

    uint64_t capacity() const noexcept override
    {
        return total_size;
    }

private:
    IExpertVictimCache& shard(const std::string& key) const
    {
        const size_t index = std::hash<std::string_view>{}(key) % shards.size();
        return *shards[index];
    }

    std::vector<std::shared_ptr<IExpertVictimCache>> shards;
    uint64_t total_size = 0;
};

std::shared_ptr<IExpertVictimCache> create_reuse_victim_cache(std::shared_ptr<IExpertVictimCache> inner, uint32_t reuse_probe_interval)
{
    if (!inner)
        return {};
    return std::make_shared<ReuseFilteredExpertVictimCache>(std::move(inner), reuse_probe_interval);
}

std::shared_ptr<IExpertVictimCache> create_sharded_victim_cache(std::vector<std::shared_ptr<IExpertVictimCache>> shards)
{
    shards.erase(std::remove(shards.begin(), shards.end(), nullptr), shards.end());
    if (shards.empty())
        return {};
    if (shards.size() == 1)
        return std::move(shards.front());
    return std::make_shared<ShardedExpertVictimCache>(std::move(shards));
}

} // namespace moe
} // namespace ncnn
