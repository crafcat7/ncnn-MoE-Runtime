#ifndef NCNN_MOE_EXPERTCACHE_H
#define NCNN_MOE_EXPERTCACHE_H

#include "expertcache_victim.h"

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

struct CompiledOperator;

struct ExpertCacheLease
{
    std::shared_ptr<const TensorData> gate_up;
    std::shared_ptr<const TensorData> down;
    // These pointers refer to the matching Entry fields and are valid while
    // this lease's private pin keeps that Entry alive.
    const CompiledOperator* gate_up_operator = nullptr;
    const CompiledOperator* down_operator = nullptr;
    bool cache_hit = false;
    uint64_t bytes_read = 0;

private:
    std::shared_ptr<const void> pin;

    friend class ExpertCache;
};

struct ExpertCacheStatistics
{
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    uint64_t resident_size = 0;
    uint64_t queued_reads = 0;
    uint64_t speculative_reads = 0;
    uint64_t cancelled_speculative_reads = 0;
    uint64_t dropped_speculative_admissions = 0;
    uint64_t unused_speculative_reads = 0;
    uint64_t short_term_reloads = 0;
    uint64_t arc_recent_size = 0;
    uint64_t arc_frequent_size = 0;
    uint64_t arc_recent_target_size = 0;
    uint64_t arc_recent_ghost_size = 0;
    uint64_t arc_frequent_ghost_size = 0;
    uint64_t arc_recent_ghost_hits = 0;
    uint64_t arc_frequent_ghost_hits = 0;
    uint64_t mapped_ranges = 0;
    uint64_t mapped_bytes = 0;
    uint64_t direct_read_ranges = 0;
    uint64_t direct_read_bytes = 0;
    uint64_t direct_read_fallbacks = 0;
    uint64_t buffered_read_ranges = 0;
    uint64_t buffered_read_bytes = 0;
    uint64_t coalesced_read_batches = 0;
    uint64_t coalesced_experts = 0;
    uint64_t coalesced_read_ranges_saved = 0;
    uint32_t adaptive_read_policy = 0;
    uint32_t num_io_threads = 0;
    uint32_t num_active_io_threads = 0;
    uint64_t io_read_samples = 0;
    uint64_t io_read_time_microseconds = 0;
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

#define NCNN_MOE_EXPERT_CACHE_MMAP_BIT              0
#define NCNN_MOE_EXPERT_CACHE_DIRECT_IO_BIT         1
#define NCNN_MOE_EXPERT_CACHE_BUFFERED_IO_BIT       2
#define NCNN_MOE_EXPERT_CACHE_FORWARD_ARC_BIT       3
#define NCNN_MOE_EXPERT_CACHE_READ_MERGE_BIT        4
#define NCNN_MOE_EXPERT_CACHE_SPECULATIVE_EVICT_BIT 5

enum ExpertCacheOptionFlag : uint32_t
{
    ExpertCacheMemoryMapRanges = UINT32_C(1) << NCNN_MOE_EXPERT_CACHE_MMAP_BIT,
    ExpertCacheDirectReads = UINT32_C(1) << NCNN_MOE_EXPERT_CACHE_DIRECT_IO_BIT,
    ExpertCacheBufferedReads = UINT32_C(1) << NCNN_MOE_EXPERT_CACHE_BUFFERED_IO_BIT,
    ExpertCacheForwardAwareEviction = UINT32_C(1) << NCNN_MOE_EXPERT_CACHE_FORWARD_ARC_BIT,
    ExpertCacheCrossExpertReadCoalescing = UINT32_C(1) << NCNN_MOE_EXPERT_CACHE_READ_MERGE_BIT,
    ExpertCacheAllowSpeculativeEviction = UINT32_C(1) << NCNN_MOE_EXPERT_CACHE_SPECULATIVE_EVICT_BIT
};

class ExpertCache
{
private:
    static constexpr uint32_t invalid_residency_group = std::numeric_limits<uint32_t>::max();

    struct Entry;
    struct FileRangeReader;
    using GhostList = std::list<ExpertKeySize>;
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

    [[nodiscard]] Result<std::shared_ptr<TensorData>> load_tensor(const TensorData& source, uint64_t& mapped_range_count, uint64_t& mapped_size);
    [[nodiscard]] Result<ExpertVictimPair> load_independent_pair(const TensorData& gate_up, const TensorData& down, uint64_t& mapped_range_count, uint64_t& mapped_size);
    [[nodiscard]] Result<ExpertVictimPair> load_interleaved_pair(const TensorData& gate_up, const TensorData& down, uint64_t& mapped_range_count, uint64_t& mapped_size);
    [[nodiscard]] Result<ExpertVictimPair> load_pair(const TensorData& gate_up, const TensorData& down, uint64_t& mapped_range_count, uint64_t& mapped_size);
    [[nodiscard]] Result<std::vector<ExpertVictimPair>> load_coalesced_pairs(
        std::span<const std::shared_ptr<Entry>> batch,
        uint64_t& mapped_range_count,
        uint64_t& mapped_size,
        uint64_t& saved_range_count,
        bool& coalesced);
    [[nodiscard]] static Result<uint64_t> stored_size(const TensorData& tensor);
    [[nodiscard]] static Result<uint64_t> packed_weight_size(const TensorData& tensor);
    [[nodiscard]] Result<std::shared_ptr<Entry>> enqueue_pair(
        const TensorData& gate_up,
        const TensorData& down,
        bool speculative,
        uint32_t residency_group,
        std::string_view prepared_key,
        ExpertVictimExecutionMetadata victim_execution = {},
        bool* already_ready = nullptr,
        bool* temporarily_exhausted = nullptr);
    [[nodiscard]] bool evict_one_locked(bool incoming_from_frequent_ghost, bool speculative_admission, uint32_t incoming_group, uint64_t required_size);
    void insert_resident_locked(Entry& entry, bool frequent);
    void touch_resident_locked(Entry& entry, bool repeated);
    void remove_resident_locked(Entry& entry, bool add_ghost);
    void add_ghost_locked(const Entry& entry);
    void erase_ghost_locked(GhostIndex& index, GhostList& list, uint64_t& size, std::string_view key);
    void trim_ghost_front_locked(GhostIndex& index, GhostList& list, uint64_t& size);
    void trim_ghosts_locked();
    [[nodiscard]] uint64_t arc_delta(uint64_t required_size, uint64_t numerator, uint64_t denominator) const;
    [[nodiscard]] bool consume_ghost_locked(std::string_view key, uint64_t required_size, bool& frequent, bool& from_frequent_ghost);
    [[nodiscard]] Entry* find_victim_locked(
        const std::list<Entry*>& list,
        bool speculative,
        uint32_t residency_group,
        uint32_t forward_anchor = invalid_residency_group,
        bool allow_predicted_victim = false);
    void stop_workers();
    void worker_loop();

    uint64_t cache_size = 0;
    uint64_t resident_size = 0;
    uint64_t arc_recent_size = 0;
    uint64_t arc_frequent_size = 0;
    uint64_t arc_recent_target_size = 0;
    uint64_t arc_recent_ghost_size = 0;
    uint64_t arc_frequent_ghost_size = 0;
    uint64_t arc_recent_ghost_hits = 0;
    uint64_t arc_frequent_ghost_hits = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    uint64_t queued_reads = 0;
    uint64_t speculative_reads = 0;
    uint64_t cancelled_speculative_reads = 0;
    uint64_t dropped_speculative_admissions = 0;
    uint64_t unused_speculative_reads = 0;
    uint64_t short_term_reloads = 0;
    uint64_t coalesced_read_batches = 0;
    uint64_t coalesced_experts = 0;
    uint64_t coalesced_read_ranges_saved = 0;
    uint64_t mapped_ranges = 0;
    uint64_t mapped_bytes = 0;
    bool stopping = false;
    mutable std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable work_available;
    std::condition_variable idle;
    uint32_t active_jobs = 0;
    EntryMap entries;
    std::list<Entry*> arc_recent;
    std::list<Entry*> arc_frequent;
    std::vector<uint64_t> residency_group_sizes;
    GhostList arc_recent_ghost;
    GhostList arc_frequent_ghost;
    GhostIndex arc_recent_ghost_index;
    GhostIndex arc_frequent_ghost_index;
    std::deque<std::shared_ptr<Entry>> high_priority;
    std::deque<std::shared_ptr<Entry>> low_priority;
    std::vector<std::thread> workers;
    uint64_t io_read_samples = 0;
    uint64_t io_read_time_nanoseconds = 0;
    std::unique_ptr<FileRangeReader> reader;
    std::shared_ptr<ExpertVictimCache> victim_cache;
    uint32_t flags = 0;
    bool reserve_cpu_packed_weights = false;

public:
    explicit ExpertCache(
        uint64_t _cache_size,
        uint32_t num_io_threads = 0,
        std::shared_ptr<ExpertVictimCache> _victim_cache = {},
        uint32_t _flags = 0,
        uint32_t num_residency_groups = 0,
        bool _reserve_cpu_packed_weights = false);
    ~ExpertCache();

    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

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
    // Enqueues as many pairs as the current cache capacity permits, waits for
    // one completion, and acquires all enqueued pairs ready at that point.
    [[nodiscard]] Result<size_t> wait_acquire_ready_pairs(std::span<const ExpertCachePairRequest> requests, std::span<ExpertCacheLease> leases, bool wait_for_any = true);
    [[nodiscard]] bool is_ready(const TensorData& gate_up, const TensorData& down, std::string_view prepared_key = {}) const;
    [[nodiscard]] static std::string make_pair_key(const TensorData& gate_up, const TensorData& down);
    void cancel_prediction();
    void resolve_predictions(uint32_t residency_group, std::span<const std::string_view> demanded_keys);
    void wait_for_background_work();
    [[nodiscard]] ExpertCacheStatistics statistics() const;
    [[nodiscard]] uint64_t capacity() const noexcept
    {
        return cache_size;
    }
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERTCACHE_H
