#ifndef NCNN_MOE_EXPERT_CACHE_H
#define NCNN_MOE_EXPERT_CACHE_H

#include "expert_victim_cache.h"

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
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
    uint64_t cancelled_speculative_reads = 0;
    uint64_t dropped_speculative_admissions = 0;
    uint64_t arc_recent_bytes = 0;
    uint64_t arc_frequent_bytes = 0;
    uint64_t arc_recent_target_bytes = 0;
    uint64_t arc_recent_ghost_bytes = 0;
    uint64_t arc_frequent_ghost_bytes = 0;
    uint64_t arc_recent_ghost_hits = 0;
    uint64_t arc_frequent_ghost_hits = 0;
    uint64_t mapped_ranges = 0;
    uint64_t mapped_bytes = 0;
    uint64_t direct_read_ranges = 0;
    uint64_t direct_read_bytes = 0;
    uint64_t direct_read_fallbacks = 0;
    uint64_t buffered_read_ranges = 0;
    uint64_t buffered_read_bytes = 0;
    uint32_t adaptive_read_policy = 0;
    ExpertVictimCacheStatistics victim;
};

struct ExpertCachePairRequest
{
    const TensorData* gate_up = nullptr;
    const TensorData* down = nullptr;
    uint32_t residency_group = std::numeric_limits<uint32_t>::max();
    std::string_view prepared_key;
    ExpertVictimExecutionMetadata victim_execution;
};

#define NCNN_MOE_EXPERT_CACHE_MMAP_BIT        0
#define NCNN_MOE_EXPERT_CACHE_DIRECT_IO_BIT   1
#define NCNN_MOE_EXPERT_CACHE_BUFFERED_IO_BIT 2

enum ExpertCacheOptionFlag : uint32_t
{
    ExpertCacheMemoryMapRanges = UINT32_C(1) << NCNN_MOE_EXPERT_CACHE_MMAP_BIT,
    ExpertCacheDirectReads = UINT32_C(1) << NCNN_MOE_EXPERT_CACHE_DIRECT_IO_BIT,
    ExpertCacheBufferedReads = UINT32_C(1) << NCNN_MOE_EXPERT_CACHE_BUFFERED_IO_BIT
};

class Mxfp4ExpertCache
{
private:
    static constexpr uint32_t invalid_residency_group = std::numeric_limits<uint32_t>::max();

    struct Entry;
    struct FileRangeReader;
    struct GhostRecord
    {
        std::string key;
        uint64_t bytes = 0;
    };
    using GhostList = std::list<GhostRecord>;
    struct TransparentStringHash
    {
        using is_transparent = void;

        [[nodiscard]] size_t operator()(std::string_view value) const noexcept
        {
            return std::hash<std::string_view>{}(value);
        }
    };
    using EntryMap = std::unordered_map<std::string, std::shared_ptr<Entry>, TransparentStringHash, std::equal_to<>>;
    using GhostIndex = std::unordered_map<std::string, GhostList::iterator, TransparentStringHash, std::equal_to<>>;

    [[nodiscard]] Result<std::shared_ptr<TensorData>> load_tensor(const TensorData& source, uint64_t& mapped_ranges, uint64_t& mapped_bytes);
    [[nodiscard]] Result<ExpertVictimPair> load_pair(const TensorData& gate_up, const TensorData& down, uint64_t& mapped_ranges, uint64_t& mapped_bytes);
    [[nodiscard]] static Result<uint64_t> stored_bytes(const TensorData& tensor);
    [[nodiscard]] Result<std::shared_ptr<Entry>> enqueue_pair(
        const TensorData& gate_up,
        const TensorData& down,
        bool speculative,
        uint32_t residency_group,
        std::string_view prepared_key,
        ExpertVictimExecutionMetadata victim_execution = {},
        bool* already_ready = nullptr);
    [[nodiscard]] bool evict_one_locked(bool incoming_from_frequent_ghost, bool speculative_admission, uint32_t incoming_group, uint64_t required);
    void insert_resident_locked(Entry& entry, bool frequent);
    void touch_resident_locked(Entry& entry, bool repeated);
    void remove_resident_locked(Entry& entry, bool add_ghost);
    void add_ghost_locked(const Entry& entry);
    void erase_ghost_locked(GhostIndex& index, GhostList& list, uint64_t& bytes, std::string_view key);
    void trim_ghost_front_locked(GhostIndex& index, GhostList& list, uint64_t& bytes);
    void trim_ghosts_locked();
    [[nodiscard]] uint64_t arc_delta(uint64_t required, uint64_t numerator, uint64_t denominator) const;
    [[nodiscard]] bool consume_ghost_locked(std::string_view key, uint64_t required, bool& frequent, bool& from_frequent_ghost);
    [[nodiscard]] Entry* find_victim_locked(const std::list<Entry*>& list, bool speculative, uint32_t residency_group);
    void worker_loop();

    uint64_t capacity_bytes_ = 0;
    uint64_t resident_bytes_ = 0;
    uint64_t arc_recent_bytes_ = 0;
    uint64_t arc_frequent_bytes_ = 0;
    uint64_t arc_recent_target_bytes_ = 0;
    uint64_t arc_recent_ghost_bytes_ = 0;
    uint64_t arc_frequent_ghost_bytes_ = 0;
    uint64_t arc_recent_ghost_hits_ = 0;
    uint64_t arc_frequent_ghost_hits_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t evictions_ = 0;
    uint64_t bytes_read_ = 0;
    uint64_t queued_reads_ = 0;
    uint64_t speculative_reads_ = 0;
    uint64_t cancelled_speculative_reads_ = 0;
    uint64_t dropped_speculative_admissions_ = 0;
    uint64_t mapped_ranges_ = 0;
    uint64_t mapped_bytes_ = 0;
    bool stopping_ = false;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable work_available_;
    std::condition_variable idle_;
    uint32_t active_jobs_ = 0;
    EntryMap entries_;
    std::list<Entry*> arc_recent_;
    std::list<Entry*> arc_frequent_;
    std::vector<uint64_t> residency_group_bytes_;
    GhostList arc_recent_ghost_;
    GhostList arc_frequent_ghost_;
    GhostIndex arc_recent_ghost_index_;
    GhostIndex arc_frequent_ghost_index_;
    std::deque<std::shared_ptr<Entry>> high_priority_;
    std::deque<std::shared_ptr<Entry>> low_priority_;
    std::vector<std::thread> workers_;
    std::unique_ptr<FileRangeReader> reader_;
    std::shared_ptr<IExpertVictimCache> victim_cache_;
    uint32_t flags_ = 0;

public:
    explicit Mxfp4ExpertCache(
        uint64_t capacity_bytes,
        uint32_t io_worker_count = 0,
        std::shared_ptr<IExpertVictimCache> victim_cache = {},
        uint32_t flags = 0,
        uint32_t residency_group_count = 0);
    ~Mxfp4ExpertCache();

    Mxfp4ExpertCache(const Mxfp4ExpertCache&) = delete;
    Mxfp4ExpertCache& operator=(const Mxfp4ExpertCache&) = delete;

    // Queues an exact read and reports readiness at the same lock point.
    [[nodiscard]] Result<bool> request_pair(
        const TensorData& gate_up,
        const TensorData& down,
        uint32_t residency_group = std::numeric_limits<uint32_t>::max(),
        std::string_view prepared_key = {},
        ExpertVictimExecutionMetadata victim_execution = {});
    // Best-effort admission; exact reads retain priority.
    [[nodiscard]] Result<bool> prefetch_pair(
        const TensorData& gate_up,
        const TensorData& down,
        uint32_t residency_group = std::numeric_limits<uint32_t>::max(),
        std::string_view prepared_key = {});
    [[nodiscard]] Result<ExpertCacheLease> acquire_pair(
        const TensorData& gate_up,
        const TensorData& down,
        uint32_t residency_group = std::numeric_limits<uint32_t>::max(),
        std::string_view prepared_key = {},
        ExpertVictimExecutionMetadata victim_execution = {});
    // Acquires a ready group under one cache lock.
    [[nodiscard]] Result<bool> try_acquire_ready_pairs(std::span<const ExpertCachePairRequest> requests, std::span<ExpertCacheLease> leases);
    // Waits for one completion and acquires all pairs ready at that point.
    [[nodiscard]] Result<size_t> wait_acquire_ready_pairs(std::span<const ExpertCachePairRequest> requests, std::span<ExpertCacheLease> leases, bool wait_for_any = true);
    [[nodiscard]] bool is_ready(const TensorData& gate_up, const TensorData& down, std::string_view prepared_key = {}) const;
    [[nodiscard]] static std::string make_pair_key(const TensorData& gate_up, const TensorData& down);
    void cancel_prediction();
    void wait_for_background_work();
    [[nodiscard]] ExpertCacheStatistics statistics() const;
    [[nodiscard]] uint64_t capacity_bytes() const noexcept
    {
        return capacity_bytes_;
    }
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_CACHE_H
