#ifndef NCNN_MOE_EXPERT_CACHE_H
#define NCNN_MOE_EXPERT_CACHE_H

#include "expert_victim_cache.h"

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

struct ExpertCacheLease
{
    std::shared_ptr<const TensorData> gate_up;
    std::shared_ptr<const TensorData> down;
    bool cache_hit = false;
    uint64_t bytes_read = 0;

private:
    std::shared_ptr<const void> pin;

    friend class Mxfp4ExpertCache;
};

struct ExpertCacheStatistics
{
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    uint64_t resident_bytes = 0;
    uint64_t queued_reads = 0;
    uint64_t speculative_reads = 0;
    uint64_t mapped_ranges = 0;
    uint64_t mapped_bytes = 0;
    ExpertVictimCacheStatistics victim;
};

enum ExpertCacheOptionFlag : uint32_t
{
    ExpertCacheMemoryMapRanges = 1u << 0
};

class Mxfp4ExpertCache
{
public:
    explicit Mxfp4ExpertCache(
        uint64_t capacity_bytes,
        uint32_t io_worker_count = 0,
        std::shared_ptr<IExpertVictimCache> victim_cache = {},
        uint32_t flags = 0);
    ~Mxfp4ExpertCache();

    Mxfp4ExpertCache(const Mxfp4ExpertCache&) = delete;
    Mxfp4ExpertCache& operator=(const Mxfp4ExpertCache&) = delete;

    // Schedules an exact route without waiting for storage. Calling this for
    // every selected expert before acquiring any lease lets cache hits compute
    // while misses are read by the bounded worker pool.
    [[nodiscard]] Result<void> request_pair(
        const TensorData& gate_up,
        const TensorData& down);
    // Best-effort, low-priority admission for predicted routes. Capacity
    // pressure is intentionally not an error and exact requests always win.
    [[nodiscard]] Result<void> prefetch_pair(
        const TensorData& gate_up,
        const TensorData& down);
    [[nodiscard]] Result<ExpertCacheLease> acquire_pair(
        const TensorData& gate_up,
        const TensorData& down);
    [[nodiscard]] bool is_ready(
        const TensorData& gate_up,
        const TensorData& down) const;
    // Keeps only the newest prediction. The predictor is deliberately
    // independent from the I/O pool so router work cannot consume read slots.
    void submit_prediction(std::function<void(uint64_t)> prediction);
    void cancel_prediction();
    [[nodiscard]] bool prediction_is_current(uint64_t generation) const;
    [[nodiscard]] ExpertCacheStatistics statistics() const;
    [[nodiscard]] uint64_t capacity_bytes() const noexcept
    {
        return capacity_bytes_;
    }

private:
    struct Entry;
    struct FileRangeReader;

    [[nodiscard]] Result<std::shared_ptr<TensorData> > load_tensor(
        const TensorData& source,
        uint64_t& mapped_ranges,
        uint64_t& mapped_bytes);
    [[nodiscard]] static Result<uint64_t> stored_bytes(const TensorData& tensor);
    [[nodiscard]] static std::string key_for(
        const TensorData& gate_up,
        const TensorData& down);
    [[nodiscard]] Result<std::shared_ptr<Entry> > enqueue_pair(
        const TensorData& gate_up,
        const TensorData& down,
        bool speculative);
    [[nodiscard]] bool evict_one_locked();
    void worker_loop();
    void predictor_loop();

    uint64_t capacity_bytes_ = 0;
    uint64_t resident_bytes_ = 0;
    uint64_t clock_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    uint64_t bytes_read_ = 0;
    uint64_t queued_reads_ = 0;
    uint64_t speculative_reads_ = 0;
    uint64_t mapped_ranges_ = 0;
    uint64_t mapped_bytes_ = 0;
    bool stopping_ = false;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable work_available_;
    std::condition_variable prediction_available_;
    std::unordered_map<std::string, std::shared_ptr<Entry> > entries_;
    std::deque<std::shared_ptr<Entry> > high_priority_;
    std::deque<std::shared_ptr<Entry> > low_priority_;
    std::vector<std::thread> workers_;
    std::thread predictor_;
    std::function<void(uint64_t)> pending_prediction_;
    uint64_t pending_prediction_generation_ = 0;
    uint64_t prediction_generation_ = 0;
    std::unique_ptr<FileRangeReader> reader_;
    std::shared_ptr<IExpertVictimCache> victim_cache_;
    uint32_t flags_ = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_CACHE_H
