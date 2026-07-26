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
    destination.resident_bytes += source.resident_bytes;
    destination.pending_bytes += source.pending_bytes;
}

class ReuseFilteredExpertVictimCache final : public IExpertVictimCache
{
public:
    ReuseFilteredExpertVictimCache(std::shared_ptr<IExpertVictimCache> inner, uint32_t reuse_probe_interval)
        : inner_(std::move(inner)),
          reuse_probe_interval_(std::max(1u, reuse_probe_interval))
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
        if (reuse_probe_interval_ > 1)
        {
            const uint64_t bytes = payload_bytes(gate_up, down);
            const std::lock_guard<std::mutex> lock(history_mutex_);
            const auto existing = history_index_.find(key);
            if (existing != history_index_.end())
            {
                reused = true;
                history_.splice(history_.end(), history_, existing->second);
                existing->second = std::prev(history_.end());
            }
            else if (bytes != 0 && bytes <= inner_->capacity_bytes())
            {
                history_.push_back({key, bytes});
                const auto position = std::prev(history_.end());
                history_index_[position->key] = position;
                history_bytes_ += bytes;
                trim_history_locked();
            }
        }
        if (!reused && reuse_probe_interval_ > 1)
        {
            const uint64_t ticket = reuse_probe_ticket_.fetch_add(1, std::memory_order_relaxed);
            if (ticket % reuse_probe_interval_ != 0)
            {
                filtered_admissions_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        if (reused)
        {
            reused_admissions_.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            probe_admissions_.fetch_add(1, std::memory_order_relaxed);
        }
        inner_->admit(std::move(key), std::move(gate_up), std::move(down), residency_group, execution);
    }

    std::optional<ExpertVictimPair> restore(
        const std::string& key,
        const TensorData& gate_up_source,
        const TensorData& down_source) override
    {
        return inner_->restore(key, gate_up_source, down_source);
    }

    void wait_for_background_work() override
    {
        inner_->wait_for_background_work();
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        ExpertVictimCacheStatistics result = inner_->statistics();
        result.filtered_admissions += filtered_admissions_.load(std::memory_order_relaxed);
        result.reused_admissions += reused_admissions_.load(std::memory_order_relaxed);
        result.probe_admissions += probe_admissions_.load(std::memory_order_relaxed);
        return result;
    }

    uint64_t capacity_bytes() const noexcept override
    {
        return inner_->capacity_bytes();
    }

private:
    struct HistoryRecord
    {
        std::string key;
        uint64_t bytes = 0;
    };

    static bool add_bytes(uint64_t& bytes, size_t count)
    {
        if (count > std::numeric_limits<uint64_t>::max() - bytes)
            return false;
        bytes += static_cast<uint64_t>(count);
        return true;
    }

    static uint64_t payload_bytes(const std::shared_ptr<const TensorData>& gate_up, const std::shared_ptr<const TensorData>& down)
    {
        if (!gate_up || !down)
            return 0;
        uint64_t bytes = 0;
        if (!add_bytes(bytes, gate_up->mxfp4_blocks.size())
            || !add_bytes(bytes, gate_up->mxfp4_scales.size())
            || !add_bytes(bytes, down->mxfp4_blocks.size())
            || !add_bytes(bytes, down->mxfp4_scales.size()))
            return 0;
        return bytes;
    }

    void trim_history_locked()
    {
        const uint64_t capacity = inner_->capacity_bytes();
        while (history_bytes_ > capacity && !history_.empty())
        {
            history_bytes_ -= history_.front().bytes;
            history_index_.erase(history_.front().key);
            history_.pop_front();
        }
    }

    std::shared_ptr<IExpertVictimCache> inner_;
    uint32_t reuse_probe_interval_ = 1;
    std::mutex history_mutex_;
    std::list<HistoryRecord> history_;
    std::unordered_map<std::string, std::list<HistoryRecord>::iterator> history_index_;
    uint64_t history_bytes_ = 0;
    std::atomic<uint64_t> reuse_probe_ticket_{0};
    std::atomic<uint64_t> filtered_admissions_{0};
    std::atomic<uint64_t> reused_admissions_{0};
    std::atomic<uint64_t> probe_admissions_{0};
};

class ShardedExpertVictimCache final : public IExpertVictimCache
{
public:
    explicit ShardedExpertVictimCache(std::vector<std::shared_ptr<IExpertVictimCache>> shards)
        : shards_(std::move(shards))
    {
        for (const auto& shard : shards_)
            capacity_bytes_ += shard->capacity_bytes();
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
        for (const auto& shard : shards_)
            shard->wait_for_background_work();
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        ExpertVictimCacheStatistics aggregate;
        for (const auto& shard : shards_)
        {
            add_statistics(aggregate, shard->statistics());
        }
        return aggregate;
    }

    uint64_t capacity_bytes() const noexcept override
    {
        return capacity_bytes_;
    }

private:
    IExpertVictimCache& shard(const std::string& key) const
    {
        const size_t index = std::hash<std::string_view>{}(key) % shards_.size();
        return *shards_[index];
    }

    std::vector<std::shared_ptr<IExpertVictimCache>> shards_;
    uint64_t capacity_bytes_ = 0;
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
