#include "expert_cache.h"
#include "kernels/cpu_mxfp4.h"
#include "storage/mapped_file.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <utility>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ncnn {
namespace moe {

#if defined(_WIN32)
struct ThreadReadEvent
{
    ThreadReadEvent()
        : handle(CreateEventW(nullptr, TRUE, FALSE, nullptr))
    {
    }

    ~ThreadReadEvent()
    {
        if (handle != nullptr)
            CloseHandle(handle);
    }

    HANDLE handle = nullptr;
};

// Every Expert I/O worker issues reads serially.  Keep one overlapped event
// per worker instead of creating and closing a kernel object for every range;
// a cold routed wave can otherwise create tens of thousands of events.
static ThreadReadEvent& thread_read_event() noexcept
{
    static thread_local ThreadReadEvent event;
    return event;
}

#endif

#define NCNN_MOE_CACHE_ENTRY_SPECULATIVE_BIT 0
#define NCNN_MOE_CACHE_ENTRY_JOB_STARTED_BIT 1
#define NCNN_MOE_CACHE_ENTRY_EXACT_MISS_BIT  2
#define NCNN_MOE_CACHE_ENTRY_DISCARD_BIT     3

struct Mxfp4ExpertCache::Entry
{
    enum class State
    {
        Loading,
        Ready,
        Failed
    };

    enum class ArcList
    {
        None,
        Recent,
        Frequent
    };

    enum Flag : uint32_t
    {
        Speculative = UINT32_C(1) << NCNN_MOE_CACHE_ENTRY_SPECULATIVE_BIT,
        JobStarted = UINT32_C(1) << NCNN_MOE_CACHE_ENTRY_JOB_STARTED_BIT,
        FirstExactMiss = UINT32_C(1) << NCNN_MOE_CACHE_ENTRY_EXACT_MISS_BIT,
        DiscardWhenReady = UINT32_C(1) << NCNN_MOE_CACHE_ENTRY_DISCARD_BIT
    };

    State state = State::Loading;
    std::string key;
    uint64_t resident_size = 0;
    uint64_t stored_size = 0;
    uint64_t exact_accesses = 0;
    uint32_t residency_group = Mxfp4ExpertCache::invalid_residency_group;
    ArcList arc_list = ArcList::None;
    std::list<Entry*>::iterator arc_position;
    uint32_t flags = 0;
    Error error;
    TensorData gate_up_source;
    TensorData down_source;
    ExpertVictimExecutionMetadata victim_execution;
    std::shared_ptr<TensorData> gate_up;
    std::shared_ptr<TensorData> down;
};

struct Mxfp4ExpertCache::FileRangeReader
{
    struct LoadedRange
    {
        MxFp4ByteBuffer data;
        bool mapped = false;
    };

    static constexpr uint32_t ReadPolicySampling = 0;
    static constexpr uint32_t ReadPolicyBuffered = 1;
    static constexpr uint32_t ReadPolicyDirect = 2;

#if defined(_WIN32)
    using Handle = HANDLE;
    static Handle invalid_handle() noexcept
    {
        return INVALID_HANDLE_VALUE;
    }
#else
    using Handle = int;
    static Handle invalid_handle() noexcept
    {
        return -1;
    }
#endif

    ~FileRangeReader()
    {
        std::unique_lock<std::shared_mutex> lock(handle_mutex);
        for (const auto& item : handles)
            close_handle(item.second);
#if defined(_WIN32)
        for (const auto& item : direct_handles)
            close_handle(item.second);
#endif
    }

    Result<LoadedRange> load(const std::string& path, uint64_t offset, uint64_t size, uint32_t flags)
    {
        if (size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        {
            return Error{
                ErrorCode::InvalidModel,
                "expert shard range is too large: " + path};
        }
        if (has_flag(flags, ExpertCacheMemoryMapRanges))
        {
            auto mapping = MappedFileRange::open(path, offset, size);
            if (mapping)
            {
                mapping.value()->prefault();
                LoadedRange loaded;
                loaded.data = mapping.value()->share_bytes();
                loaded.mapped = true;
                return loaded;
            }
        }
#if defined(_WIN32)
        const bool force_direct = has_flag(flags, ExpertCacheDirectReads);
        const bool force_buffered = has_flag(flags, ExpertCacheBufferedReads);
        const bool adaptive = !force_direct && !force_buffered;
        const uint32_t policy = adaptive_read_policy.load(std::memory_order_relaxed);
        const bool sample = adaptive && policy == ReadPolicySampling && size >= 1024 * 1024;
        const bool try_direct = force_direct
                                || (adaptive
                                    && (policy == ReadPolicyDirect
                                        || (sample && adaptive_sample_ticket.fetch_add(1, std::memory_order_relaxed) % 2 == 1)));
        if (try_direct)
        {
            const auto started = std::chrono::steady_clock::now();
            auto direct = read_direct(path, offset, size);
            const uint64_t elapsed = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
            if (direct)
            {
                direct_read_ranges.fetch_add(1, std::memory_order_relaxed);
                direct_read_bytes.fetch_add(size, std::memory_order_relaxed);
                if (sample)
                {
                    record_adaptive_sample(true, size, elapsed);
                }
                LoadedRange loaded;
                loaded.data = std::move(direct).value();
                return loaded;
            }
            direct_read_fallbacks.fetch_add(1, std::memory_order_relaxed);
            if (adaptive)
            {
                adaptive_read_policy.store(ReadPolicyBuffered, std::memory_order_relaxed);
            }
        }
#endif

        const auto buffered_started = std::chrono::steady_clock::now();
        LoadedRange loaded;
        loaded.data.resize(static_cast<size_t>(size));
        auto status = read(path, offset, loaded.data);
        if (!status)
            return status.error();
        const uint64_t buffered_elapsed = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - buffered_started).count());
        buffered_read_ranges.fetch_add(1, std::memory_order_relaxed);
        buffered_read_bytes.fetch_add(size, std::memory_order_relaxed);
#if defined(_WIN32)
        if (sample)
        {
            record_adaptive_sample(false, size, buffered_elapsed);
        }
#endif
        return loaded;
    }

    void record_adaptive_sample(bool use_direct, uint64_t size, uint64_t nanoseconds)
    {
        if (use_direct)
        {
            adaptive_direct_bytes.fetch_add(size, std::memory_order_relaxed);
            adaptive_direct_nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
            adaptive_direct_samples.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            adaptive_buffered_bytes.fetch_add(size, std::memory_order_relaxed);
            adaptive_buffered_nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
            adaptive_buffered_samples.fetch_add(1, std::memory_order_relaxed);
        }
        if (adaptive_direct_samples.load(std::memory_order_relaxed) < 16 || adaptive_buffered_samples.load(std::memory_order_relaxed) < 16)
        {
            return;
        }

        const long double direct_rate = static_cast<long double>(adaptive_direct_bytes.load(std::memory_order_relaxed))
                                        / static_cast<long double>(adaptive_direct_nanoseconds.load(std::memory_order_relaxed));
        const long double buffered_rate = static_cast<long double>(adaptive_buffered_bytes.load(std::memory_order_relaxed))
                                          / static_cast<long double>(adaptive_buffered_nanoseconds.load(std::memory_order_relaxed));
        adaptive_read_policy.store(direct_rate > buffered_rate * 1.05L ? ReadPolicyDirect : ReadPolicyBuffered, std::memory_order_relaxed);
    }

    [[nodiscard]] uint32_t selected_read_policy() const noexcept
    {
        return adaptive_read_policy.load(std::memory_order_relaxed);
    }

    void populate_statistics(ExpertCacheStatistics& result) const noexcept
    {
        result.direct_read_ranges = direct_read_ranges.load(std::memory_order_relaxed);
        result.direct_read_bytes = direct_read_bytes.load(std::memory_order_relaxed);
        result.direct_read_fallbacks = direct_read_fallbacks.load(std::memory_order_relaxed);
        result.buffered_read_ranges = buffered_read_ranges.load(std::memory_order_relaxed);
        result.buffered_read_bytes = buffered_read_bytes.load(std::memory_order_relaxed);
        result.adaptive_read_policy = selected_read_policy();
    }

#if defined(_WIN32)
    Result<MxFp4ByteBuffer> read_direct(const std::string& path, uint64_t offset, uint64_t size)
    {
        const HANDLE read_event = thread_read_event().handle;
        if (read_event == nullptr)
        {
            return Error{
                ErrorCode::IoError,
                "cannot create direct Expert read event: " + path};
        }
        static constexpr uint64_t direct_alignment = 64 * 1024;
        const uint64_t aligned_offset = offset - offset % direct_alignment;
        const uint64_t prefix = offset - aligned_offset;
        if (size > std::numeric_limits<uint64_t>::max() - prefix)
        {
            return Error{
                ErrorCode::InvalidModel,
                "direct Expert range overflows: " + path};
        }
        const uint64_t logical_size = prefix + size;
        if (logical_size > std::numeric_limits<uint64_t>::max() - (direct_alignment - 1))
        {
            return Error{
                ErrorCode::InvalidModel,
                "aligned direct Expert range overflows: " + path};
        }
        const uint64_t request_size = (logical_size + direct_alignment - 1) / direct_alignment * direct_alignment;
        if (request_size == 0 || request_size > std::numeric_limits<DWORD>::max())
        {
            return Error{
                ErrorCode::InvalidArgument,
                "direct Expert range is outside the supported size"};
        }

        auto handle_result = direct_handle_for(path);
        if (!handle_result)
            return handle_result.error();
        const Handle handle = handle_result.value();
        uint8_t* allocation = static_cast<uint8_t*>(VirtualAlloc(nullptr, static_cast<SIZE_T>(request_size), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!allocation)
        {
            return Error{
                ErrorCode::IoError,
                "cannot allocate aligned direct Expert buffer"};
        }
        std::shared_ptr<uint8_t> owner(
            allocation,
            [](uint8_t* pointer) {
                if (pointer)
                    VirtualFree(pointer, 0, MEM_RELEASE);
            });

        OVERLAPPED operation{};
        operation.Offset = static_cast<DWORD>(aligned_offset);
        operation.OffsetHigh = static_cast<DWORD>(aligned_offset >> 32);
        operation.hEvent = read_event;
        if (operation.hEvent == nullptr || !ResetEvent(operation.hEvent))
        {
            return Error{
                ErrorCode::IoError,
                "cannot prepare direct Expert read event: " + path};
        }
        DWORD read_bytes = 0;
        const BOOL started = ReadFile(handle, allocation, static_cast<DWORD>(request_size), nullptr, &operation);
        const DWORD start_error = started ? ERROR_SUCCESS : GetLastError();
        if (!started && start_error != ERROR_IO_PENDING)
        {
            return Error{
                ErrorCode::IoError,
                "cannot start direct Expert read: " + path};
        }
        if (!GetOverlappedResult(handle, &operation, &read_bytes, TRUE) || static_cast<uint64_t>(read_bytes) < logical_size)
        {
            return Error{
                ErrorCode::IoError,
                "cannot complete direct Expert read: " + path};
        }

        std::shared_ptr<uint8_t> view(owner, allocation + static_cast<size_t>(prefix));
        return MxFp4ByteBuffer(std::move(view), static_cast<size_t>(size));
    }
#endif

    Result<void> read(const std::string& path, uint64_t offset, MxFp4ByteBuffer& destination)
    {
#if defined(_WIN32)
        const HANDLE read_event = thread_read_event().handle;
        if (read_event == nullptr)
        {
            return Error{
                ErrorCode::IoError,
                "cannot create expert shard read event: " + path};
        }
#endif
        auto handle_result = handle_for(path);
        if (!handle_result)
            return handle_result.error();
        const Handle handle = handle_result.value();

        size_t completed = 0;
        while (completed < destination.size())
        {
#if defined(_WIN32)
            const size_t remaining = destination.size() - completed;
            const DWORD request = static_cast<DWORD>(std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<DWORD>::max())));
            const uint64_t current_offset = offset + completed;
            OVERLAPPED operation{};
            operation.Offset = static_cast<DWORD>(current_offset);
            operation.OffsetHigh = static_cast<DWORD>(current_offset >> 32);
            operation.hEvent = read_event;
            if (!ResetEvent(operation.hEvent))
            {
                return Error{
                    ErrorCode::IoError,
                    "cannot reset expert shard read event: " + path};
            }

            DWORD read_bytes = 0;
            const BOOL started = ReadFile(handle, destination.data() + completed, request, nullptr, &operation);
            const DWORD start_error = started ? ERROR_SUCCESS : GetLastError();
            if (!started && start_error != ERROR_IO_PENDING)
            {
                return Error{
                    ErrorCode::IoError,
                    "cannot read expert shard range: " + path};
            }
            if (!GetOverlappedResult(handle, &operation, &read_bytes, TRUE))
            {
                return Error{
                    ErrorCode::IoError,
                    "cannot complete expert shard range: " + path};
            }
            if (read_bytes == 0)
                return Error{ErrorCode::IoError, "expert shard range is truncated: " + path};
            completed += read_bytes;
#else
            const size_t remaining = destination.size() - completed;
            const size_t request = std::min<size_t>(remaining, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
            const uint64_t current_offset = offset + completed;
            if (current_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
            {
                return Error{ErrorCode::InvalidModel, "expert file offset is too large: " + path};
            }
            const ssize_t read_bytes = pread(handle, destination.data() + completed, request, static_cast<off_t>(current_offset));
            if (read_bytes < 0)
            {
                if (errno == EINTR)
                    continue;
                return Error{
                    ErrorCode::IoError,
                    "cannot read expert shard range: " + path + ": " + std::strerror(errno)};
            }
            if (read_bytes == 0)
                return Error{ErrorCode::IoError, "expert shard range is truncated: " + path};
            completed += static_cast<size_t>(read_bytes);
#endif
        }

#if defined(__linux__) && defined(POSIX_FADV_DONTNEED)
        if (!destination.empty())
        {
            posix_fadvise(handle, static_cast<off_t>(offset), static_cast<off_t>(destination.size()), POSIX_FADV_DONTNEED);
        }
#endif
        return {};
    }

private:
    static void close_handle(Handle handle) noexcept
    {
#if defined(_WIN32)
        CloseHandle(handle);
#else
        close(handle);
#endif
    }

#if defined(_WIN32)
    Result<Handle> direct_handle_for(const std::string& path)
    {
        {
            std::shared_lock<std::shared_mutex> lock(handle_mutex);
            const auto existing = direct_handles.find(path);
            if (existing != direct_handles.end())
                return existing->second;
        }

        const std::wstring native_path = std::filesystem::path(path).wstring();
        const Handle handle = CreateFileW(
            native_path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING,
            nullptr);
        if (handle == invalid_handle())
        {
            return Error{
                ErrorCode::IoError,
                "cannot open direct Expert shard: " + path};
        }
        std::unique_lock<std::shared_mutex> lock(handle_mutex);
        const auto inserted = direct_handles.emplace(path, handle);
        if (!inserted.second)
        {
            close_handle(handle);
            return inserted.first->second;
        }
        return handle;
    }
#endif

    Result<Handle> handle_for(const std::string& path)
    {
        {
            std::shared_lock<std::shared_mutex> lock(handle_mutex);
            const auto existing = handles.find(path);
            if (existing != handles.end())
                return existing->second;
        }

#if defined(_WIN32)
        const std::wstring native_path = std::filesystem::path(path).wstring();
        const Handle handle = CreateFileW(
            native_path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED,
            nullptr);
#else
        const Handle handle = open(path.c_str(), O_RDONLY);
#endif
        if (handle == invalid_handle())
            return Error{ErrorCode::IoError, "cannot open expert shard: " + path};

#if defined(__APPLE__) && defined(F_NOCACHE)
        (void)fcntl(handle, F_NOCACHE, 1);
#endif
        std::unique_lock<std::shared_mutex> lock(handle_mutex);
        const auto inserted = handles.emplace(path, handle);
        if (!inserted.second)
        {
            close_handle(handle);
            return inserted.first->second;
        }
        return handle;
    }

    std::shared_mutex handle_mutex;
    std::unordered_map<std::string, Handle> handles;
#if defined(_WIN32)
    std::unordered_map<std::string, Handle> direct_handles;
#endif
#if defined(_WIN32)
    std::atomic<uint32_t> adaptive_read_policy{ReadPolicyDirect};
#else
    std::atomic<uint32_t> adaptive_read_policy{ReadPolicyBuffered};
#endif
    std::atomic<uint64_t> adaptive_sample_ticket{0};
    std::atomic<uint64_t> adaptive_direct_samples{0};
    std::atomic<uint64_t> adaptive_buffered_samples{0};
    std::atomic<uint64_t> adaptive_direct_bytes{0};
    std::atomic<uint64_t> adaptive_buffered_bytes{0};
    std::atomic<uint64_t> adaptive_direct_nanoseconds{0};
    std::atomic<uint64_t> adaptive_buffered_nanoseconds{0};
    std::atomic<uint64_t> direct_read_ranges{0};
    std::atomic<uint64_t> direct_read_bytes{0};
    std::atomic<uint64_t> direct_read_fallbacks{0};
    std::atomic<uint64_t> buffered_read_ranges{0};
    std::atomic<uint64_t> buffered_read_bytes{0};
};

Mxfp4ExpertCache::Mxfp4ExpertCache(
    uint64_t _cache_size,
    uint32_t _num_io_threads,
    std::shared_ptr<IExpertVictimCache> _victim_cache,
    uint32_t _flags,
    uint32_t num_residency_groups,
    bool _reserve_cpu_packed_weights)
    : cache_size(_cache_size),
      residency_group_sizes(num_residency_groups, 0),
      reader(std::make_unique<FileRangeReader>()),
      victim_cache(std::move(_victim_cache)),
      flags(_flags),
      reserve_cpu_packed_weights(_reserve_cpu_packed_weights)
{
    if (_num_io_threads == 0)
    {
        const uint32_t hardware = std::max(1u, std::thread::hardware_concurrency());
        _num_io_threads = std::min(4u, hardware);
    }
    num_io_threads = std::max(1u, _num_io_threads);
    num_active_io_threads = num_io_threads;
    workers.reserve(num_io_threads);
    for (uint32_t worker = 0; worker < num_io_threads; ++worker)
        workers.emplace_back(&Mxfp4ExpertCache::worker_loop, this, worker);
}

Mxfp4ExpertCache::~Mxfp4ExpertCache()
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        low_priority.clear();
    }
    work_available.notify_all();
    for (std::thread& worker : workers)
    {
        if (worker.joinable())
            worker.join();
    }
}

void Mxfp4ExpertCache::cancel_prediction()
{
    std::lock_guard<std::mutex> lock(mutex);
    for (const std::shared_ptr<Entry>& entry : low_priority)
    {
        if (!entry
            || has_flag(entry->flags, Entry::JobStarted)
            || entry->state != Entry::State::Loading
            || !has_flag(entry->flags, Entry::Speculative))
        {
            continue;
        }
        const auto existing = entries.find(entry->key);
        if (existing != entries.end() && existing->second == entry)
        {
            remove_resident_locked(*entry, false);
            entries.erase(existing);
            ++cancelled_speculative_reads;
        }
    }
    low_priority.clear();
    if (high_priority.empty() && active_jobs == 0)
    {
        idle.notify_all();
    }
}

void Mxfp4ExpertCache::resolve_predictions(uint32_t residency_group, std::span<const std::string_view> demanded_keys)
{
    std::lock_guard<std::mutex> lock(mutex);
    for (auto iterator = entries.begin(); iterator != entries.end();)
    {
        const std::shared_ptr<Entry>& entry = iterator->second;
        if (!entry
            || entry->residency_group != residency_group
            || !has_flag(entry->flags, Entry::Speculative))
        {
            ++iterator;
            continue;
        }
        const bool demanded = std::find(demanded_keys.begin(), demanded_keys.end(), std::string_view(entry->key)) != demanded_keys.end();
        if (demanded)
        {
            ++iterator;
            continue;
        }
        if (entry->state == Entry::State::Loading && has_flag(entry->flags, Entry::JobStarted))
        {
            entry->flags |= Entry::DiscardWhenReady;
            ++iterator;
            continue;
        }
        if (entry->state == Entry::State::Loading)
        {
            entry->state = Entry::State::Failed;
            entry->error = Error{ErrorCode::InternalError, "speculative Expert read was cancelled"};
            ++cancelled_speculative_reads;
        }
        else if (entry->state == Entry::State::Ready)
        {
            ++unused_speculative_reads;
        }
        remove_resident_locked(*entry, false);
        iterator = entries.erase(iterator);
    }
    if (high_priority.empty() && active_jobs == 0)
        idle.notify_all();
}

void Mxfp4ExpertCache::wait_for_background_work()
{
    {
        std::unique_lock<std::mutex> lock(mutex);
        idle.wait(lock, [this] {
            return high_priority.empty()
                   && low_priority.empty()
                   && active_jobs == 0;
        });
    }
    if (victim_cache)
        victim_cache->wait_for_background_work();
}

static bool is_file_backed_bfloat16_tensor(const TensorData& tensor) noexcept
{
    if (tensor.dtype != DType::BFloat16
        || tensor.shape.size() != 2
        || !tensor.mapped_data)
    {
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(tensor.shape[0])
                              * tensor.shape[1];
    if (elements == 0
        || elements > std::numeric_limits<uint64_t>::max() / sizeof(uint16_t)
        || elements > std::numeric_limits<size_t>::max())
    {
        return false;
    }
    return tensor.mapped_size == elements * sizeof(uint16_t)
           && tensor.bfloat16_values().size() == static_cast<size_t>(elements);
}

static bool is_supported_file_backed_pair(
    const TensorData& gate_up,
    const TensorData& down) noexcept
{
    return (gate_up.mxfp4_file_storage && down.mxfp4_file_storage)
           || (is_file_backed_bfloat16_tensor(gate_up)
               && is_file_backed_bfloat16_tensor(down));
}

Result<uint64_t> Mxfp4ExpertCache::stored_size(const TensorData& tensor)
{
    if (is_file_backed_bfloat16_tensor(tensor))
        return tensor.mapped_size;
    if (tensor.dtype != DType::MxFp4 || !tensor.mxfp4_file_storage)
        return Error{ErrorCode::InvalidArgument, "expert cache requires file-backed MXFP4 or BF16 tensors"};
    const MxFp4FileStorage& source = *tensor.mxfp4_file_storage;
    if (source.scales_size > std::numeric_limits<uint64_t>::max() - source.blocks_size)
        return Error{ErrorCode::InvalidModel, "file-backed MXFP4 tensor byte count overflows"};
    uint64_t size = source.blocks_size + source.scales_size;
    if (!source.interleave_rows)
    {
        if (source.secondary_blocks_size != 0
            || source.secondary_scales_size != 0
            || !source.secondary_blocks_path.empty()
            || !source.secondary_scales_path.empty())
        {
            return Error{ErrorCode::InvalidModel, "non-interleaved file-backed MXFP4 tensor has secondary ranges"};
        }
        return size;
    }
    if (source.secondary_blocks_size == 0
        || source.secondary_scales_size == 0
        || source.secondary_blocks_path.empty()
        || source.secondary_scales_path.empty())
    {
        return Error{ErrorCode::InvalidModel, "interleaved file-backed MXFP4 tensor is missing secondary ranges"};
    }
    if (source.secondary_blocks_size > std::numeric_limits<uint64_t>::max() - size)
        return Error{ErrorCode::InvalidModel, "file-backed MXFP4 tensor byte count overflows"};
    size += source.secondary_blocks_size;
    if (source.secondary_scales_size > std::numeric_limits<uint64_t>::max() - size)
        return Error{ErrorCode::InvalidModel, "file-backed MXFP4 tensor byte count overflows"};
    return size + source.secondary_scales_size;
}

Result<uint64_t> Mxfp4ExpertCache::packed_weight_size(const TensorData& tensor)
{
    if (tensor.dtype != DType::MxFp4 || tensor.shape.size() != 2)
        return Error{ErrorCode::InvalidArgument, "packed Expert reservation requires an MXFP4 matrix"};
    const uint64_t rows = tensor.shape[0];
    const uint64_t columns = tensor.shape[1];
    if (rows < 4)
        return uint64_t{0};
    if (rows > std::numeric_limits<size_t>::max()
        || columns == 0
        || columns > std::numeric_limits<uint32_t>::max()
        || columns % 32 != 0)
    {
        return Error{ErrorCode::InvalidModel, "packed Expert reservation requires MXFP4 columns divisible by 32"};
    }
    const uint64_t size = mxfp4_q8_packed_storage_bytes(
        static_cast<size_t>(rows),
        static_cast<uint32_t>(columns / 32));
    if (size == 0)
        return Error{ErrorCode::InvalidModel, "packed Expert byte count overflows"};
    return size;
}

std::string Mxfp4ExpertCache::make_pair_key(const TensorData& gate_up, const TensorData& down)
{
    if (is_file_backed_bfloat16_tensor(gate_up)
        && is_file_backed_bfloat16_tensor(down))
    {
        return "bf16:"
               + std::to_string(reinterpret_cast<uintptr_t>(gate_up.mapped_data.get()))
               + ":" + std::to_string(gate_up.mapped_size)
               + "|"
               + std::to_string(reinterpret_cast<uintptr_t>(down.mapped_data.get()))
               + ":" + std::to_string(down.mapped_size);
    }
    const MxFp4FileStorage& gate = *gate_up.mxfp4_file_storage;
    const MxFp4FileStorage& projection = *down.mxfp4_file_storage;
    return gate.blocks_path + ":" + std::to_string(gate.blocks_offset)
           + ":" + std::to_string(gate.blocks_size)
           + ":" + gate.scales_path + ":" + std::to_string(gate.scales_offset)
           + ":" + std::to_string(gate.scales_size)
           + ":" + gate.secondary_blocks_path + ":" + std::to_string(gate.secondary_blocks_offset)
           + ":" + std::to_string(gate.secondary_blocks_size)
           + ":" + gate.secondary_scales_path + ":" + std::to_string(gate.secondary_scales_offset)
           + ":" + std::to_string(gate.secondary_scales_size)
           + "|" + projection.blocks_path + ":" + std::to_string(projection.blocks_offset)
           + ":" + std::to_string(projection.blocks_size)
           + ":" + projection.scales_path + ":" + std::to_string(projection.scales_offset)
           + ":" + std::to_string(projection.scales_size);
}

static Result<void> copy_interleaved_mxfp4_rows(
    const TensorData& source,
    const MxFp4ByteBuffer& first_blocks,
    const MxFp4ByteBuffer& first_scales,
    const MxFp4ByteBuffer& second_blocks,
    const MxFp4ByteBuffer& second_scales,
    MxFp4ByteBuffer& destination_blocks,
    MxFp4ByteBuffer& destination_scales)
{
    if (first_blocks.size() != second_blocks.size()
        || first_scales.size() != second_scales.size()
        || source.shape.size() != 2
        || source.shape[0] == 0
        || source.shape[0] % 2 != 0
        || source.shape[1] == 0
        || source.shape[1] % 32 != 0)
    {
        return Error{ErrorCode::InvalidModel, "invalid interleaved file-backed MXFP4 tensor"};
    }
    const size_t source_rows = source.shape[0] / 2;
    const uint64_t expected_block_size = static_cast<uint64_t>(source_rows) * source.shape[1] / 2;
    const uint64_t expected_scale_size = static_cast<uint64_t>(source_rows) * source.shape[1] / 32;
    if (first_blocks.size() % source_rows != 0
        || first_scales.size() % source_rows != 0
        || first_blocks.size() != expected_block_size
        || first_scales.size() != expected_scale_size
        || first_blocks.size() > std::numeric_limits<size_t>::max() - second_blocks.size()
        || first_scales.size() > std::numeric_limits<size_t>::max() - second_scales.size())
    {
        return Error{ErrorCode::InvalidModel, "invalid interleaved file-backed MXFP4 tensor"};
    }
    const size_t block_row_size = first_blocks.size() / source_rows;
    const size_t scale_row_size = first_scales.size() / source_rows;
    destination_blocks.resize(first_blocks.size() + second_blocks.size());
    destination_scales.resize(first_scales.size() + second_scales.size());
    for (size_t row = 0; row < source_rows; ++row)
    {
        std::memcpy(destination_blocks.data() + row * block_row_size * 2, first_blocks.data() + row * block_row_size, block_row_size);
        std::memcpy(destination_blocks.data() + (row * 2 + 1) * block_row_size, second_blocks.data() + row * block_row_size, block_row_size);
        std::memcpy(destination_scales.data() + row * scale_row_size * 2, first_scales.data() + row * scale_row_size, scale_row_size);
        std::memcpy(destination_scales.data() + (row * 2 + 1) * scale_row_size, second_scales.data() + row * scale_row_size, scale_row_size);
    }
    return {};
}

Result<std::shared_ptr<TensorData>> Mxfp4ExpertCache::load_tensor(const TensorData& source, uint64_t& mapped_range_count, uint64_t& mapped_size)
{
    if (is_file_backed_bfloat16_tensor(source))
    {
        auto loaded = std::make_shared<TensorData>();
        loaded->dtype = DType::BFloat16;
        loaded->shape = source.shape;
        if (has_flag(flags, ExpertCacheMemoryMapRanges))
        {
            loaded->mapped_data = source.mapped_data;
            loaded->mapped_size = source.mapped_size;
            // The source is a slice of a mapped Expert bank.  Give the OS the
            // exact slice range while this cache worker is waiting for it so
            // CPU execution can overlap page staging with other admissions.
            prefetch_mapped_memory(
                loaded->mapped_data.get(),
                loaded->mapped_size);
            ++mapped_range_count;
            mapped_size += source.mapped_size;
        }
        else
        {
            const std::span<const uint16_t> values = source.bfloat16_values();
            loaded->bfloat16_data.assign(values.begin(), values.end());
        }
        return loaded;
    }
    if (!source.mxfp4_file_storage)
        return Error{ErrorCode::InvalidArgument, "Expert tensor is not file-backed"};
    const MxFp4FileStorage& file = *source.mxfp4_file_storage;
    if (file.blocks_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
        || file.scales_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
        || file.secondary_blocks_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
        || file.secondary_scales_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        return Error{ErrorCode::InvalidModel, "file-backed MXFP4 tensor is too large"};
    }

    auto loaded = std::make_shared<TensorData>();
    loaded->dtype = DType::MxFp4;
    loaded->shape = source.shape;
    // A pair loader may overlap Gate/Up with Down, while each tensor overlaps
    // its independent blocks and scales.  That keeps the physical read fanout
    // fixed at four per cache worker instead of tying storage latency to four
    // serial ranges.
    auto blocks_future = std::async(
        std::launch::async,
        [this, &file] {
            return reader->load(file.blocks_path, file.blocks_offset, file.blocks_size, flags);
        });
    auto scales = reader->load(file.scales_path, file.scales_offset, file.scales_size, flags);
    auto blocks = blocks_future.get();
    if (!blocks)
        return blocks.error();
    if (!scales)
        return scales.error();
    if (blocks.value().mapped)
    {
        ++mapped_range_count;
        mapped_size += file.blocks_size;
    }
    if (scales.value().mapped)
    {
        ++mapped_range_count;
        mapped_size += file.scales_size;
    }
    if (!file.interleave_rows)
    {
        loaded->mxfp4_blocks = std::move(blocks).value().data;
        loaded->mxfp4_scales = std::move(scales).value().data;
        return loaded;
    }

    auto secondary_blocks = reader->load(file.secondary_blocks_path, file.secondary_blocks_offset, file.secondary_blocks_size, flags);
    if (!secondary_blocks)
        return secondary_blocks.error();
    auto secondary_scales = reader->load(file.secondary_scales_path, file.secondary_scales_offset, file.secondary_scales_size, flags);
    if (!secondary_scales)
        return secondary_scales.error();
    if (secondary_blocks.value().mapped)
    {
        ++mapped_range_count;
        mapped_size += file.secondary_blocks_size;
    }
    if (secondary_scales.value().mapped)
    {
        ++mapped_range_count;
        mapped_size += file.secondary_scales_size;
    }
    const MxFp4ByteBuffer& first_blocks = blocks.value().data;
    const MxFp4ByteBuffer& first_scales = scales.value().data;
    const MxFp4ByteBuffer& second_blocks = secondary_blocks.value().data;
    const MxFp4ByteBuffer& second_scales = secondary_scales.value().data;
    auto copied = copy_interleaved_mxfp4_rows(source, first_blocks, first_scales, second_blocks, second_scales, loaded->mxfp4_blocks, loaded->mxfp4_scales);
    if (!copied)
        return copied.error();
    return loaded;
}

Result<ExpertVictimPair> Mxfp4ExpertCache::load_interleaved_pair(const TensorData& gate_up, const TensorData& down, uint64_t& mapped_range_count, uint64_t& mapped_size)
{
    const MxFp4FileStorage& gate = *gate_up.mxfp4_file_storage;
    const MxFp4FileStorage& projection = *down.mxfp4_file_storage;
    const bool one_file = !projection.interleave_rows
                          && gate.blocks_path == gate.scales_path
                          && gate.blocks_path == gate.secondary_blocks_path
                          && gate.blocks_path == gate.secondary_scales_path
                          && gate.blocks_path == projection.blocks_path
                          && gate.blocks_path == projection.scales_path;
    if (!one_file)
    {
        uint64_t down_mapped_range_count = 0;
        uint64_t down_mapped_size = 0;
        auto gate_future = std::async(
            std::launch::async,
            [this, &gate_up, &mapped_range_count, &mapped_size] {
                return load_tensor(gate_up, mapped_range_count, mapped_size);
            });
        auto loaded_down = load_tensor(down, down_mapped_range_count, down_mapped_size);
        auto loaded_gate = gate_future.get();
        if (!loaded_gate)
            return loaded_gate.error();
        if (!loaded_down)
            return loaded_down.error();
        mapped_range_count += down_mapped_range_count;
        mapped_size += down_mapped_size;
        return ExpertVictimPair{std::move(loaded_gate).value(), std::move(loaded_down).value()};
    }

    struct Range
    {
        uint64_t offset = 0;
        uint64_t size = 0;
        uint32_t destination = 0;
        uint32_t cluster = 0;
    };
    std::array<Range, 6> ranges{{
        {gate.blocks_offset, gate.blocks_size, 0, 0},
        {gate.scales_offset, gate.scales_size, 1, 0},
        {gate.secondary_blocks_offset, gate.secondary_blocks_size, 2, 0},
        {gate.secondary_scales_offset, gate.secondary_scales_size, 3, 0},
        {projection.blocks_offset, projection.blocks_size, 4, 0},
        {projection.scales_offset, projection.scales_size, 5, 0},
    }};
    std::sort(ranges.begin(), ranges.end(), [](const Range& first, const Range& second) {
        return first.offset < second.offset;
    });

    struct Cluster
    {
        uint64_t offset = 0;
        uint64_t size = 0;
        FileRangeReader::LoadedRange loaded;
    };
    std::array<Cluster, 6> clusters;
    size_t cluster_count = 0;
    for (Range& range : ranges)
    {
        if (range.size == 0
            || range.size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
            || range.offset > std::numeric_limits<uint64_t>::max() - range.size)
        {
            return Error{ErrorCode::InvalidModel, "invalid interleaved file-backed MXFP4 tensor range"};
        }
        if (cluster_count == 0
            || clusters[cluster_count - 1].offset + clusters[cluster_count - 1].size != range.offset)
        {
            if (cluster_count != 0
                && range.offset < clusters[cluster_count - 1].offset + clusters[cluster_count - 1].size)
            {
                return Error{ErrorCode::InvalidModel, "overlapping interleaved file-backed MXFP4 tensor ranges"};
            }
            clusters[cluster_count].offset = range.offset;
            clusters[cluster_count].size = range.size;
            ++cluster_count;
        }
        else
        {
            clusters[cluster_count - 1].size += range.size;
        }
        range.cluster = static_cast<uint32_t>(cluster_count - 1);
    }
    if (cluster_count == ranges.size())
    {
        auto loaded_gate = load_tensor(gate_up, mapped_range_count, mapped_size);
        if (!loaded_gate)
            return loaded_gate.error();
        auto loaded_down = load_tensor(down, mapped_range_count, mapped_size);
        if (!loaded_down)
            return loaded_down.error();
        return ExpertVictimPair{std::move(loaded_gate).value(), std::move(loaded_down).value()};
    }

    for (size_t cluster_index = 0; cluster_index < cluster_count; ++cluster_index)
    {
        Cluster& cluster = clusters[cluster_index];
        auto loaded = reader->load(gate.blocks_path, cluster.offset, cluster.size, flags);
        if (!loaded)
            return loaded.error();
        cluster.loaded = std::move(loaded).value();
        if (cluster.loaded.mapped)
        {
            ++mapped_range_count;
            mapped_size += cluster.size;
        }
    }

    std::array<MxFp4ByteBuffer, 6> sources;
    for (const Range& range : ranges)
    {
        const Cluster& cluster = clusters[range.cluster];
        const std::shared_ptr<uint8_t> owner = cluster.loaded.data.storage;
        std::shared_ptr<uint8_t> view(owner, owner.get() + static_cast<size_t>(range.offset - cluster.offset));
        sources[range.destination] = MxFp4ByteBuffer(std::move(view), static_cast<size_t>(range.size));
    }

    auto gate_size = stored_size(gate_up);
    if (!gate_size)
        return gate_size.error();
    auto down_size = stored_size(down);
    if (!down_size)
        return down_size.error();
    if (gate_size.value() > std::numeric_limits<uint64_t>::max() - down_size.value())
        return Error{ErrorCode::InvalidModel, "interleaved file-backed MXFP4 Expert pair byte count overflows"};
    const uint64_t total_size = gate_size.value() + down_size.value();
    if (total_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return Error{ErrorCode::InvalidModel, "interleaved file-backed MXFP4 Expert pair is too large"};
    MxFp4ByteBuffer resident;
    resident.resize(static_cast<size_t>(total_size));
    const std::shared_ptr<uint8_t> resident_owner = resident.storage;
    size_t resident_offset = 0;
    const auto make_resident_view = [&](size_t size) {
        std::shared_ptr<uint8_t> view(resident_owner, resident_owner.get() + resident_offset);
        resident_offset += size;
        return MxFp4ByteBuffer(std::move(view), size);
    };

    auto loaded_gate = std::make_shared<TensorData>();
    loaded_gate->dtype = DType::MxFp4;
    loaded_gate->shape = gate_up.shape;
    loaded_gate->mxfp4_blocks = make_resident_view(static_cast<size_t>(gate.blocks_size + gate.secondary_blocks_size));
    loaded_gate->mxfp4_scales = make_resident_view(static_cast<size_t>(gate.scales_size + gate.secondary_scales_size));
    auto copied = copy_interleaved_mxfp4_rows(gate_up, sources[0], sources[1], sources[2], sources[3], loaded_gate->mxfp4_blocks, loaded_gate->mxfp4_scales);
    if (!copied)
        return copied.error();

    auto loaded_down = std::make_shared<TensorData>();
    loaded_down->dtype = DType::MxFp4;
    loaded_down->shape = down.shape;
    loaded_down->mxfp4_blocks = make_resident_view(static_cast<size_t>(projection.blocks_size));
    loaded_down->mxfp4_scales = make_resident_view(static_cast<size_t>(projection.scales_size));
    std::memcpy(loaded_down->mxfp4_blocks.data(), sources[4].data(), sources[4].size());
    std::memcpy(loaded_down->mxfp4_scales.data(), sources[5].data(), sources[5].size());
    return ExpertVictimPair{std::move(loaded_gate), std::move(loaded_down)};
}

Result<ExpertVictimPair> Mxfp4ExpertCache::load_pair(const TensorData& gate_up, const TensorData& down, uint64_t& mapped_range_count, uint64_t& mapped_size)
{
    if (is_file_backed_bfloat16_tensor(gate_up)
        && is_file_backed_bfloat16_tensor(down))
    {
        auto loaded_gate = load_tensor(gate_up, mapped_range_count, mapped_size);
        if (!loaded_gate)
            return loaded_gate.error();
        auto loaded_down = load_tensor(down, mapped_range_count, mapped_size);
        if (!loaded_down)
            return loaded_down.error();
        return ExpertVictimPair{
            std::move(loaded_gate).value(),
            std::move(loaded_down).value()};
    }
    if (!gate_up.mxfp4_file_storage || !down.mxfp4_file_storage)
    {
        return Error{
            ErrorCode::InvalidArgument,
            "Expert pair does not use a supported file-backed encoding"};
    }
    const MxFp4FileStorage& gate = *gate_up.mxfp4_file_storage;
    const MxFp4FileStorage& projection = *down.mxfp4_file_storage;
    if (gate.interleave_rows)
        return load_interleaved_pair(gate_up, down, mapped_range_count, mapped_size);
    struct Range
    {
        uint64_t offset = 0;
        uint64_t size = 0;
        uint32_t destination = 0;
    };
    std::array<Range, 4> ranges{{
        {gate.blocks_offset, gate.blocks_size, 0},
        {gate.scales_offset, gate.scales_size, 1},
        {projection.blocks_offset, projection.blocks_size, 2},
        {projection.scales_offset, projection.scales_size, 3},
    }};
    const bool one_file = gate.blocks_path == gate.scales_path
                          && gate.blocks_path == projection.blocks_path
                          && gate.blocks_path == projection.scales_path;
    bool contiguous = one_file && !gate.interleave_rows;
    if (contiguous)
    {
        std::sort(ranges.begin(), ranges.end(), [](const Range& first, const Range& second) {
            return first.offset < second.offset;
        });
        for (size_t index = 0; index < ranges.size(); ++index)
        {
            const Range& range = ranges[index];
            if (range.size == 0 || range.offset > std::numeric_limits<uint64_t>::max() - range.size)
            {
                contiguous = false;
                break;
            }
            if (index != 0)
            {
                const Range& previous = ranges[index - 1];
                if (previous.offset + previous.size != range.offset)
                {
                    contiguous = false;
                    break;
                }
            }
        }
    }

    if (!contiguous)
    {
        uint64_t down_mapped_range_count = 0;
        uint64_t down_mapped_size = 0;
        auto gate_future = std::async(
            std::launch::async,
            [this, &gate_up, &mapped_range_count, &mapped_size] {
                return load_tensor(gate_up, mapped_range_count, mapped_size);
            });
        auto loaded_down = load_tensor(down, down_mapped_range_count, down_mapped_size);
        auto loaded_gate = gate_future.get();
        if (!loaded_gate)
            return loaded_gate.error();
        if (!loaded_down)
            return loaded_down.error();
        mapped_range_count += down_mapped_range_count;
        mapped_size += down_mapped_size;
        return ExpertVictimPair{std::move(loaded_gate).value(), std::move(loaded_down).value()};
    }

    const uint64_t first_offset = ranges.front().offset;
    const Range& last = ranges.back();
    const uint64_t span_size = last.offset + last.size - first_offset;
    auto combined = reader->load(gate.blocks_path, first_offset, span_size, flags);
    if (!combined)
        return combined.error();
    if (combined.value().mapped)
    {
        ++mapped_range_count;
        mapped_size += span_size;
    }

    auto loaded_gate = std::make_shared<TensorData>();
    loaded_gate->dtype = DType::MxFp4;
    loaded_gate->shape = gate_up.shape;
    auto loaded_down = std::make_shared<TensorData>();
    loaded_down->dtype = DType::MxFp4;
    loaded_down->shape = down.shape;
    const std::shared_ptr<uint8_t> owner = combined.value().data.storage;
    for (const Range& range : ranges)
    {
        std::shared_ptr<uint8_t> view(owner, owner.get() + static_cast<size_t>(range.offset - first_offset));
        MxFp4ByteBuffer data(std::move(view), static_cast<size_t>(range.size));
        switch (range.destination)
        {
        case 0:
            loaded_gate->mxfp4_blocks = std::move(data);
            break;
        case 1:
            loaded_gate->mxfp4_scales = std::move(data);
            break;
        case 2:
            loaded_down->mxfp4_blocks = std::move(data);
            break;
        case 3:
            loaded_down->mxfp4_scales = std::move(data);
            break;
        }
    }
    return ExpertVictimPair{std::move(loaded_gate), std::move(loaded_down)};
}

Result<std::vector<ExpertVictimPair>> Mxfp4ExpertCache::load_coalesced_pairs(
    std::span<const std::shared_ptr<Entry>> batch,
    uint64_t& mapped_range_count,
    uint64_t& mapped_size,
    uint64_t& saved_range_count,
    bool& coalesced)
{
    static constexpr uint64_t maximum_cluster_size = UINT64_C(64) * 1024 * 1024;
    struct Range
    {
        const std::string* path = nullptr;
        uint64_t offset = 0;
        uint64_t size = 0;
        uint32_t entry = 0;
        uint32_t destination = 0;
        uint32_t cluster = 0;
    };
    struct Cluster
    {
        const std::string* path = nullptr;
        uint64_t offset = 0;
        uint64_t size = 0;
        FileRangeReader::LoadedRange loaded;
    };

    coalesced = false;
    saved_range_count = 0;
    mapped_range_count = 0;
    mapped_size = 0;
    if (batch.size() < 2 || has_flag(flags, ExpertCacheMemoryMapRanges))
        return std::vector<ExpertVictimPair>();
    for (const std::shared_ptr<Entry>& entry : batch)
    {
        if (is_file_backed_bfloat16_tensor(entry->gate_up_source)
            || is_file_backed_bfloat16_tensor(entry->down_source))
        {
            return std::vector<ExpertVictimPair>();
        }
    }

    std::vector<Range> ranges;
    ranges.reserve(batch.size() * 6);
    for (size_t entry_index = 0; entry_index < batch.size(); ++entry_index)
    {
        const Entry& entry = *batch[entry_index];
        if (!entry.gate_up_source.mxfp4_file_storage
            || !entry.down_source.mxfp4_file_storage)
        {
            return Error{ErrorCode::InvalidArgument, "coalesced Expert read requires file-backed MXFP4 pairs"};
        }
        const MxFp4FileStorage& gate = *entry.gate_up_source.mxfp4_file_storage;
        const MxFp4FileStorage& down = *entry.down_source.mxfp4_file_storage;
        if (down.interleave_rows)
            return Error{ErrorCode::InvalidModel, "coalesced Expert read does not support an interleaved Down tensor"};
        ranges.push_back({&gate.blocks_path, gate.blocks_offset, gate.blocks_size, static_cast<uint32_t>(entry_index), 0, 0});
        ranges.push_back({&gate.scales_path, gate.scales_offset, gate.scales_size, static_cast<uint32_t>(entry_index), 1, 0});
        if (gate.interleave_rows)
        {
            ranges.push_back({&gate.secondary_blocks_path, gate.secondary_blocks_offset, gate.secondary_blocks_size, static_cast<uint32_t>(entry_index), 2, 0});
            ranges.push_back({&gate.secondary_scales_path, gate.secondary_scales_offset, gate.secondary_scales_size, static_cast<uint32_t>(entry_index), 3, 0});
        }
        ranges.push_back({&down.blocks_path, down.blocks_offset, down.blocks_size, static_cast<uint32_t>(entry_index), 4, 0});
        ranges.push_back({&down.scales_path, down.scales_offset, down.scales_size, static_cast<uint32_t>(entry_index), 5, 0});
    }
    for (const Range& range : ranges)
    {
        if (!range.path
            || range.path->empty()
            || range.size == 0
            || range.size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
            || range.offset > std::numeric_limits<uint64_t>::max() - range.size)
        {
            return Error{ErrorCode::InvalidModel, "invalid coalesced Expert file range"};
        }
    }
    uint64_t independent_range_count = 0;
    for (uint32_t entry_index = 0; entry_index < batch.size(); ++entry_index)
    {
        std::vector<const Range*> entry_ranges;
        for (const Range& range : ranges)
        {
            if (range.entry == entry_index)
                entry_ranges.push_back(&range);
        }
        std::sort(entry_ranges.begin(), entry_ranges.end(), [](const Range* first, const Range* second) {
            if (*first->path != *second->path)
                return *first->path < *second->path;
            return first->offset < second->offset;
        });
        const Range* previous = nullptr;
        uint64_t cluster_size = 0;
        for (const Range* range : entry_ranges)
        {
            const bool adjacent = previous
                                  && *previous->path == *range->path
                                  && previous->offset + previous->size == range->offset
                                  && cluster_size <= maximum_cluster_size
                                  && range->size <= maximum_cluster_size - cluster_size;
            if (!adjacent)
            {
                ++independent_range_count;
                cluster_size = range->size;
            }
            else
            {
                cluster_size += range->size;
            }
            previous = range;
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range& first, const Range& second) {
        if (*first.path != *second.path)
            return *first.path < *second.path;
        return first.offset < second.offset;
    });

    std::vector<Cluster> clusters;
    clusters.reserve(ranges.size());
    for (Range& range : ranges)
    {
        const bool adjacent = !clusters.empty()
                              && *clusters.back().path == *range.path
                              && clusters.back().offset + clusters.back().size == range.offset
                              && clusters.back().size <= maximum_cluster_size
                              && range.size <= maximum_cluster_size - clusters.back().size;
        if (!clusters.empty()
            && *clusters.back().path == *range.path
            && range.offset < clusters.back().offset + clusters.back().size)
        {
            return Error{ErrorCode::InvalidModel, "overlapping coalesced Expert file ranges"};
        }
        if (!adjacent)
        {
            clusters.push_back({range.path, range.offset, range.size, {}});
        }
        else
        {
            clusters.back().size += range.size;
        }
        range.cluster = static_cast<uint32_t>(clusters.size() - 1);
    }
    if (clusters.size() >= independent_range_count)
        return std::vector<ExpertVictimPair>();

    coalesced = true;
    saved_range_count = independent_range_count - clusters.size();
    for (Cluster& cluster : clusters)
    {
        auto loaded = reader->load(*cluster.path, cluster.offset, cluster.size, flags);
        if (!loaded)
            return loaded.error();
        cluster.loaded = std::move(loaded).value();
        if (cluster.loaded.mapped)
        {
            ++mapped_range_count;
            mapped_size += cluster.size;
        }
    }

    std::vector<std::array<MxFp4ByteBuffer, 6>> sources(batch.size());
    for (const Range& range : ranges)
    {
        const Cluster& cluster = clusters[range.cluster];
        const std::shared_ptr<uint8_t> owner = cluster.loaded.data.storage;
        std::shared_ptr<uint8_t> view(owner, owner.get() + static_cast<size_t>(range.offset - cluster.offset));
        sources[range.entry][range.destination] = MxFp4ByteBuffer(std::move(view), static_cast<size_t>(range.size));
    }

    std::vector<ExpertVictimPair> loaded_pairs(batch.size());
    for (size_t entry_index = 0; entry_index < batch.size(); ++entry_index)
    {
        const TensorData& gate_source = batch[entry_index]->gate_up_source;
        const TensorData& down_source = batch[entry_index]->down_source;
        const MxFp4FileStorage& gate = *gate_source.mxfp4_file_storage;
        const std::array<MxFp4ByteBuffer, 6>& entry_sources = sources[entry_index];

        auto loaded_gate = std::make_shared<TensorData>();
        loaded_gate->dtype = DType::MxFp4;
        loaded_gate->shape = gate_source.shape;
        if (gate.interleave_rows)
        {
            auto copied = copy_interleaved_mxfp4_rows(
                gate_source,
                entry_sources[0],
                entry_sources[1],
                entry_sources[2],
                entry_sources[3],
                loaded_gate->mxfp4_blocks,
                loaded_gate->mxfp4_scales);
            if (!copied)
                return copied.error();
        }
        else
        {
            loaded_gate->mxfp4_blocks.assign(entry_sources[0].data(), entry_sources[0].size());
            loaded_gate->mxfp4_scales.assign(entry_sources[1].data(), entry_sources[1].size());
        }

        auto loaded_down = std::make_shared<TensorData>();
        loaded_down->dtype = DType::MxFp4;
        loaded_down->shape = down_source.shape;
        loaded_down->mxfp4_blocks.assign(entry_sources[4].data(), entry_sources[4].size());
        loaded_down->mxfp4_scales.assign(entry_sources[5].data(), entry_sources[5].size());
        loaded_pairs[entry_index] = {std::move(loaded_gate), std::move(loaded_down)};
    }
    return loaded_pairs;
}

void Mxfp4ExpertCache::insert_resident_locked(Entry& entry, bool frequent)
{
    if (frequent)
    {
        arc_frequent.push_back(&entry);
        entry.arc_position = std::prev(arc_frequent.end());
        entry.arc_list = Entry::ArcList::Frequent;
        arc_frequent_size += entry.resident_size;
    }
    else
    {
        arc_recent.push_back(&entry);
        entry.arc_position = std::prev(arc_recent.end());
        entry.arc_list = Entry::ArcList::Recent;
        arc_recent_size += entry.resident_size;
    }
    resident_size += entry.resident_size;
    if (entry.residency_group < residency_group_sizes.size())
    {
        residency_group_sizes[entry.residency_group] += entry.resident_size;
    }
}

void Mxfp4ExpertCache::touch_resident_locked(Entry& entry, bool repeated)
{
    if (entry.arc_list == Entry::ArcList::Recent && repeated)
    {
        arc_recent.erase(entry.arc_position);
        arc_recent_size -= entry.resident_size;
        arc_frequent.push_back(&entry);
        entry.arc_position = std::prev(arc_frequent.end());
        entry.arc_list = Entry::ArcList::Frequent;
        arc_frequent_size += entry.resident_size;
        return;
    }
    if (entry.arc_list == Entry::ArcList::Recent)
    {
        arc_recent.splice(arc_recent.end(), arc_recent, entry.arc_position);
        entry.arc_position = std::prev(arc_recent.end());
    }
    else if (entry.arc_list == Entry::ArcList::Frequent)
    {
        arc_frequent.splice(arc_frequent.end(), arc_frequent, entry.arc_position);
        entry.arc_position = std::prev(arc_frequent.end());
    }
}

void Mxfp4ExpertCache::erase_ghost_locked(GhostIndex& index, GhostList& list, uint64_t& size, std::string_view key)
{
    const auto existing = index.find(key);
    if (existing == index.end())
        return;
    size -= existing->second->size;
    list.erase(existing->second);
    index.erase(existing);
}

void Mxfp4ExpertCache::add_ghost_locked(const Entry& entry)
{
    erase_ghost_locked(arc_recent_ghost_index, arc_recent_ghost, arc_recent_ghost_size, entry.key);
    erase_ghost_locked(arc_frequent_ghost_index, arc_frequent_ghost, arc_frequent_ghost_size, entry.key);

    GhostRecord record{entry.key, entry.resident_size};
    if (entry.arc_list == Entry::ArcList::Recent)
    {
        arc_recent_ghost.push_back(std::move(record));
        const auto position = std::prev(arc_recent_ghost.end());
        arc_recent_ghost_index[position->key] = position;
        arc_recent_ghost_size += entry.resident_size;
    }
    else if (entry.arc_list == Entry::ArcList::Frequent)
    {
        arc_frequent_ghost.push_back(std::move(record));
        const auto position = std::prev(arc_frequent_ghost.end());
        arc_frequent_ghost_index[position->key] = position;
        arc_frequent_ghost_size += entry.resident_size;
    }
    trim_ghosts_locked();
}

void Mxfp4ExpertCache::trim_ghost_front_locked(GhostIndex& index, GhostList& list, uint64_t& size)
{
    if (list.empty())
        return;
    size -= list.front().size;
    index.erase(list.front().key);
    list.pop_front();
}

void Mxfp4ExpertCache::trim_ghosts_locked()
{
    while (arc_recent_ghost_size + arc_frequent_ghost_size > cache_size)
    {
        if (arc_recent_ghost_size >= arc_frequent_ghost_size && !arc_recent_ghost.empty())
        {
            trim_ghost_front_locked(arc_recent_ghost_index, arc_recent_ghost, arc_recent_ghost_size);
        }
        else
        {
            trim_ghost_front_locked(arc_frequent_ghost_index, arc_frequent_ghost, arc_frequent_ghost_size);
        }
    }
}

void Mxfp4ExpertCache::remove_resident_locked(Entry& entry, bool add_ghost)
{
    if (entry.arc_list == Entry::ArcList::None)
        return;
    if (add_ghost)
        add_ghost_locked(entry);
    if (entry.arc_list == Entry::ArcList::Recent)
    {
        arc_recent.erase(entry.arc_position);
        arc_recent_size -= entry.resident_size;
    }
    else
    {
        arc_frequent.erase(entry.arc_position);
        arc_frequent_size -= entry.resident_size;
    }
    resident_size -= entry.resident_size;
    if (entry.residency_group < residency_group_sizes.size())
    {
        residency_group_sizes[entry.residency_group] -= entry.resident_size;
    }
    entry.arc_list = Entry::ArcList::None;
}

uint64_t Mxfp4ExpertCache::arc_delta(uint64_t required_size, uint64_t numerator, uint64_t denominator) const
{
    if (denominator == 0 || numerator <= denominator)
        return required_size;
    const uint64_t ratio = numerator / denominator;
    if (required_size != 0 && ratio > cache_size / required_size)
        return cache_size;
    return std::min(cache_size, std::max(required_size, required_size * ratio));
}

bool Mxfp4ExpertCache::consume_ghost_locked(std::string_view key, uint64_t required_size, bool& frequent, bool& from_frequent_ghost)
{
    frequent = false;
    from_frequent_ghost = false;
    const auto recent = arc_recent_ghost_index.find(key);
    if (recent != arc_recent_ghost_index.end())
    {
        const uint64_t delta = arc_delta(required_size, arc_frequent_ghost_size, arc_recent_ghost_size);
        arc_recent_ghost_size -= recent->second->size;
        arc_recent_ghost.erase(recent->second);
        arc_recent_ghost_index.erase(recent);
        arc_recent_target_size = std::min(cache_size, arc_recent_target_size + std::min(delta, cache_size - arc_recent_target_size));
        ++arc_recent_ghost_hits;
        frequent = true;
        return true;
    }

    const auto frequent_ghost = arc_frequent_ghost_index.find(key);
    if (frequent_ghost == arc_frequent_ghost_index.end())
    {
        return false;
    }
    const uint64_t delta = arc_delta(required_size, arc_recent_ghost_size, arc_frequent_ghost_size);
    arc_frequent_ghost_size -= frequent_ghost->second->size;
    arc_frequent_ghost.erase(frequent_ghost->second);
    arc_frequent_ghost_index.erase(frequent_ghost);
    arc_recent_target_size -= std::min(delta, arc_recent_target_size);
    ++arc_frequent_ghost_hits;
    frequent = true;
    from_frequent_ghost = true;
    return true;
}

Mxfp4ExpertCache::Entry* Mxfp4ExpertCache::find_victim_locked(
    const std::list<Entry*>& list,
    bool speculative,
    uint32_t residency_group,
    uint32_t forward_anchor,
    bool allow_predicted_victim)
{
    Entry* selected = nullptr;
    uint32_t selected_distance = 0;
    for (Entry* entry : list)
    {
        const bool speculative_requires_speculative_victim = speculative
                                                             && !has_flag(flags, ExpertCacheForwardAwareEviction)
                                                             && !has_flag(flags, ExpertCacheAllowSpeculativeEviction);
        if (!entry
            || entry->state != Entry::State::Ready
            || (speculative_requires_speculative_victim
                && !has_flag(entry->flags, Entry::Speculative))
            || (has_flag(flags, ExpertCacheForwardAwareEviction)
                && !speculative
                && !allow_predicted_victim
                && has_flag(entry->flags, Entry::Speculative))
            || (residency_group != invalid_residency_group
                && entry->residency_group != residency_group))
        {
            continue;
        }
        const auto existing = entries.find(entry->key);
        if (existing != entries.end() && existing->second.get() == entry && existing->second.use_count() == 1)
        {
            if (forward_anchor == invalid_residency_group
                || forward_anchor >= residency_group_sizes.size()
                || entry->residency_group >= residency_group_sizes.size())
            {
                return entry;
            }
            const uint32_t group_count = static_cast<uint32_t>(residency_group_sizes.size());
            const uint32_t distance = entry->residency_group > forward_anchor
                                          ? entry->residency_group - forward_anchor
                                          : group_count - forward_anchor + entry->residency_group;
            if (!selected || distance > selected_distance)
            {
                selected = entry;
                selected_distance = distance;
            }
        }
    }
    return selected;
}

bool Mxfp4ExpertCache::evict_one_locked(bool incoming_from_frequent_ghost, bool speculative_admission, uint32_t incoming_group, uint64_t required_size)
{
    const bool prefer_recent = arc_recent_size > arc_recent_target_size
                               || (incoming_from_frequent_ghost && arc_recent_size == arc_recent_target_size);
    uint32_t preferred_group = invalid_residency_group;
    if (!residency_group_sizes.empty())
    {
        const uint64_t fair_share = cache_size / residency_group_sizes.size();
        if (incoming_group < residency_group_sizes.size() && residency_group_sizes[incoming_group] + required_size > fair_share)
        {
            preferred_group = incoming_group;
        }
        else
        {
            uint64_t maximum_excess = 0;
            for (uint32_t group = 0; group < residency_group_sizes.size(); ++group)
            {
                const uint64_t size = residency_group_sizes[group];
                const uint64_t excess = size > fair_share ? size - fair_share : 0;
                if (excess > maximum_excess)
                {
                    maximum_excess = excess;
                    preferred_group = group;
                }
            }
        }
    }

    const uint32_t forward_anchor = has_flag(flags, ExpertCacheForwardAwareEviction)
                                        ? incoming_group
                                        : invalid_residency_group;
    if (forward_anchor != invalid_residency_group)
        preferred_group = invalid_residency_group;
    Entry* victim = prefer_recent
                        ? find_victim_locked(arc_recent, speculative_admission, preferred_group, forward_anchor)
                        : find_victim_locked(arc_frequent, speculative_admission, preferred_group, forward_anchor);
    if (!victim)
    {
        victim = prefer_recent
                     ? find_victim_locked(arc_frequent, speculative_admission, preferred_group, forward_anchor)
                     : find_victim_locked(arc_recent, speculative_admission, preferred_group, forward_anchor);
    }
    if (!victim && preferred_group != invalid_residency_group)
    {
        victim = prefer_recent
                     ? find_victim_locked(arc_recent, speculative_admission, invalid_residency_group)
                     : find_victim_locked(arc_frequent, speculative_admission, invalid_residency_group);
        if (!victim)
        {
            victim = prefer_recent
                         ? find_victim_locked(arc_frequent, speculative_admission, invalid_residency_group)
                         : find_victim_locked(arc_recent, speculative_admission, invalid_residency_group);
        }
    }
    if (!victim
        && !speculative_admission
        && forward_anchor != invalid_residency_group)
    {
        victim = prefer_recent
                     ? find_victim_locked(
                           arc_recent,
                           false,
                           invalid_residency_group,
                           forward_anchor,
                           true)
                     : find_victim_locked(
                           arc_frequent,
                           false,
                           invalid_residency_group,
                           forward_anchor,
                           true);
        if (!victim)
        {
            victim = prefer_recent
                         ? find_victim_locked(
                               arc_frequent,
                               false,
                               invalid_residency_group,
                               forward_anchor,
                               true)
                         : find_victim_locked(
                               arc_recent,
                               false,
                               invalid_residency_group,
                               forward_anchor,
                               true);
        }
    }
    if (!victim)
        return false;

    const auto existing = entries.find(victim->key);
    if (existing == entries.end())
        return false;
    if (victim_cache)
    {
        victim_cache->admit(victim->key, victim->gate_up, victim->down, victim->residency_group, victim->victim_execution);
    }
    if (has_flag(victim->flags, Entry::Speculative))
        ++unused_speculative_reads;
    remove_resident_locked(*victim, !has_flag(victim->flags, Entry::Speculative));
    entries.erase(existing);
    ++evictions;
    return true;
}

Result<std::shared_ptr<Mxfp4ExpertCache::Entry>> Mxfp4ExpertCache::enqueue_pair(
    const TensorData& gate_up,
    const TensorData& down,
    bool speculative,
    uint32_t residency_group,
    std::string_view prepared_key,
    ExpertVictimExecutionMetadata victim_execution,
    bool* already_ready,
    bool* temporarily_exhausted)
{
    if (already_ready)
        *already_ready = false;
    if (temporarily_exhausted)
        *temporarily_exhausted = false;
    auto gate_size = stored_size(gate_up);
    if (!gate_size)
        return gate_size.error();
    auto down_size = stored_size(down);
    if (!down_size)
        return down_size.error();
    if (down_size.value() > std::numeric_limits<uint64_t>::max() - gate_size.value())
        return Error{ErrorCode::InvalidModel, "expert pair byte count overflows"};
    const uint64_t stored_size = gate_size.value() + down_size.value();
    uint64_t required_size = stored_size;
    if (reserve_cpu_packed_weights)
    {
        auto packed_gate_size = packed_weight_size(gate_up);
        if (!packed_gate_size)
            return packed_gate_size.error();
        auto packed_down_size = packed_weight_size(down);
        if (!packed_down_size)
            return packed_down_size.error();
        if (packed_gate_size.value() > std::numeric_limits<uint64_t>::max() - required_size)
            return Error{ErrorCode::InvalidModel, "packed expert pair byte count overflows"};
        required_size += packed_gate_size.value();
        if (packed_down_size.value() > std::numeric_limits<uint64_t>::max() - required_size)
            return Error{ErrorCode::InvalidModel, "packed expert pair byte count overflows"};
        required_size += packed_down_size.value();
    }
    if (required_size > cache_size)
    {
        if (speculative)
        {
            const std::lock_guard<std::mutex> lock(mutex);
            ++dropped_speculative_admissions;
            return std::shared_ptr<Entry>();
        }
        return Error{
            ErrorCode::InvalidArgument,
            "expert cache is smaller than one resident expert pair"};
    }

    std::string generated_key;
    if (prepared_key.empty())
        generated_key = make_pair_key(gate_up, down);
    const std::string_view key = prepared_key.empty() ? std::string_view(generated_key) : prepared_key;
    std::lock_guard<std::mutex> lock(mutex);
    const auto existing = entries.find(key);
    if (existing != entries.end())
    {
        const std::shared_ptr<Entry>& entry = existing->second;
        if (already_ready)
        {
            *already_ready = entry->state == Entry::State::Ready;
        }
        if (victim_execution.enabled)
            entry->victim_execution = victim_execution;
        bool frequent = false;
        bool from_frequent_ghost = false;
        if (!speculative
            && entry->exact_accesses == 0
            && consume_ghost_locked(key, required_size, frequent, from_frequent_ghost)
            && frequent
            && entry->arc_list == Entry::ArcList::Recent)
        {
            touch_resident_locked(*entry, true);
        }
        if (!speculative && has_flag(entry->flags, Entry::Speculative) && entry->state == Entry::State::Loading)
        {
            entry->flags &= ~Entry::Speculative;
            entry->flags &= ~Entry::DiscardWhenReady;
            entry->flags |= Entry::FirstExactMiss;
            ++misses;
            if (!has_flag(entry->flags, Entry::JobStarted))
                high_priority.push_back(entry);
            // Adaptive worker throttling is keyed by worker index.  Wake all
            // workers so an inactive worker cannot consume the only signal
            // while every eligible worker remains asleep.
            work_available.notify_all();
        }
        return entry;
    }

    bool insert_frequent = false;
    bool incoming_from_frequent_ghost = false;
    if (!speculative)
    {
        if (consume_ghost_locked(key, required_size, insert_frequent, incoming_from_frequent_ghost))
            ++short_term_reloads;
    }
    while (resident_size > cache_size - required_size)
    {
        if (!evict_one_locked(incoming_from_frequent_ghost, speculative, residency_group, required_size))
        {
            if (speculative)
            {
                ++dropped_speculative_admissions;
                return std::shared_ptr<Entry>();
            }
            if (temporarily_exhausted)
                *temporarily_exhausted = true;
            return Error{
                ErrorCode::InvalidArgument,
                "expert cache capacity is exhausted by active expert leases or reads"};
        }
    }

    auto entry = std::make_shared<Entry>();
    entry->key.assign(key);
    entry->resident_size = required_size;
    entry->stored_size = stored_size;
    entry->residency_group = residency_group;
    if (speculative)
        entry->flags |= Entry::Speculative;
    else
        entry->flags |= Entry::FirstExactMiss;
    entry->gate_up_source = gate_up;
    entry->down_source = down;
    entry->victim_execution = victim_execution;
    insert_resident_locked(*entry, insert_frequent);
    entries.emplace(entry->key, entry);
    ++queued_reads;
    if (speculative)
    {
        ++speculative_reads;
        low_priority.push_back(entry);
    }
    else
    {
        ++misses;
        high_priority.push_back(entry);
    }
    // See the worker predicate in worker_loop(): after adaptive shrinkage a
    // notify_one() may select an ineligible worker and strand this request.
    work_available.notify_all();
    return entry;
}

Result<bool> Mxfp4ExpertCache::request_pair(
    const TensorData& gate_up,
    const TensorData& down,
    uint32_t residency_group,
    std::string_view prepared_key,
    ExpertVictimExecutionMetadata victim_execution)
{
    bool already_ready = false;
    auto entry = enqueue_pair(gate_up, down, false, residency_group, prepared_key, victim_execution, &already_ready);
    if (!entry)
        return entry.error();
    return already_ready;
}

Result<bool> Mxfp4ExpertCache::prefetch_pair(
    const TensorData& gate_up,
    const TensorData& down,
    uint32_t residency_group,
    std::string_view prepared_key)
{
    bool already_ready = false;
    auto entry = enqueue_pair(gate_up, down, true, residency_group, prepared_key, {}, &already_ready);
    if (!entry)
        return entry.error();
    return already_ready;
}

void Mxfp4ExpertCache::worker_loop(uint32_t worker_index)
{
    static constexpr size_t maximum_batch_size = 8;
    struct PhysicalRange
    {
        const std::string* path = nullptr;
        uint64_t offset = 0;
        uint64_t size = 0;
    };
    const auto collect_ranges = [](const Entry& entry, std::array<PhysicalRange, 6>& ranges) {
        const MxFp4FileStorage& gate = *entry.gate_up_source.mxfp4_file_storage;
        const MxFp4FileStorage& down = *entry.down_source.mxfp4_file_storage;
        size_t count = 0;
        ranges[count++] = {&gate.blocks_path, gate.blocks_offset, gate.blocks_size};
        ranges[count++] = {&gate.scales_path, gate.scales_offset, gate.scales_size};
        if (gate.interleave_rows)
        {
            ranges[count++] = {&gate.secondary_blocks_path, gate.secondary_blocks_offset, gate.secondary_blocks_size};
            ranges[count++] = {&gate.secondary_scales_path, gate.secondary_scales_offset, gate.secondary_scales_size};
        }
        ranges[count++] = {&down.blocks_path, down.blocks_offset, down.blocks_size};
        ranges[count++] = {&down.scales_path, down.scales_offset, down.scales_size};
        return count;
    };
    const auto entries_are_adjacent = [&collect_ranges](const Entry& first, const Entry& second) {
        if (!first.gate_up_source.mxfp4_file_storage
            || !first.down_source.mxfp4_file_storage
            || !second.gate_up_source.mxfp4_file_storage
            || !second.down_source.mxfp4_file_storage)
        {
            return false;
        }
        std::array<PhysicalRange, 6> first_ranges;
        std::array<PhysicalRange, 6> second_ranges;
        const size_t first_count = collect_ranges(first, first_ranges);
        const size_t second_count = collect_ranges(second, second_ranges);
        for (size_t first_index = 0; first_index < first_count; ++first_index)
        {
            const PhysicalRange& left = first_ranges[first_index];
            for (size_t second_index = 0; second_index < second_count; ++second_index)
            {
                const PhysicalRange& right = second_ranges[second_index];
                if (*left.path != *right.path)
                    continue;
                if (left.offset + left.size == right.offset
                    || right.offset + right.size == left.offset)
                {
                    return true;
                }
            }
        }
        return false;
    };

    std::vector<std::shared_ptr<Entry>> batch;
    std::vector<std::optional<ExpertVictimPair>> loaded_pairs;
    std::vector<std::optional<Error>> errors;
    std::vector<uint8_t> restored;
    std::vector<uint64_t> entry_mapped_range_counts;
    std::vector<uint64_t> entry_mapped_sizes;
    std::vector<std::shared_ptr<Entry>> disk_entries;
    std::vector<size_t> disk_slots;
    batch.reserve(maximum_batch_size);
    loaded_pairs.reserve(maximum_batch_size);
    errors.reserve(maximum_batch_size);
    restored.reserve(maximum_batch_size);
    entry_mapped_range_counts.reserve(maximum_batch_size);
    entry_mapped_sizes.reserve(maximum_batch_size);
    disk_entries.reserve(maximum_batch_size);
    disk_slots.reserve(maximum_batch_size);

    for (;;)
    {
        batch.clear();
        {
            std::unique_lock<std::mutex> lock(mutex);
            work_available.wait(lock, [this, worker_index] {
                return stopping
                       || (worker_index < num_active_io_threads
                           && (!high_priority.empty()
                               || !low_priority.empty()));
            });
            if (stopping && high_priority.empty() && low_priority.empty())
                return;
            const bool high_priority_batch = !high_priority.empty();
            std::deque<std::shared_ptr<Entry>>& queue = high_priority_batch ? high_priority : low_priority;
            while (!queue.empty())
            {
                std::shared_ptr<Entry> entry = std::move(queue.front());
                queue.pop_front();
                if (entry && !has_flag(entry->flags, Entry::JobStarted) && entry->state == Entry::State::Loading)
                {
                    entry->flags |= Entry::JobStarted;
                    batch.push_back(std::move(entry));
                    break;
                }
            }
            if (batch.size() == 1
                && has_flag(flags, ExpertCacheCrossExpertReadCoalescing)
                && !has_flag(flags, ExpertCacheMemoryMapRanges))
            {
                if (queue.empty())
                {
                    work_available.wait_for(lock, std::chrono::microseconds(50), [&] {
                        return stopping || !queue.empty();
                    });
                }
                for (auto iterator = queue.begin(); iterator != queue.end() && batch.size() < maximum_batch_size;)
                {
                    const std::shared_ptr<Entry>& candidate = *iterator;
                    if (!candidate
                        || has_flag(candidate->flags, Entry::JobStarted)
                        || candidate->state != Entry::State::Loading)
                    {
                        iterator = queue.erase(iterator);
                        continue;
                    }
                    bool adjacent = false;
                    for (const std::shared_ptr<Entry>& selected : batch)
                        adjacent = adjacent || entries_are_adjacent(*selected, *candidate);
                    if (!adjacent)
                    {
                        ++iterator;
                        continue;
                    }
                    candidate->flags |= Entry::JobStarted;
                    batch.push_back(std::move(*iterator));
                    iterator = queue.erase(iterator);
                }
            }
            if (batch.empty())
            {
                if (high_priority.empty() && low_priority.empty() && active_jobs == 0)
                {
                    idle.notify_all();
                }
                continue;
            }
            active_jobs += static_cast<uint32_t>(batch.size());
        }

        const auto io_started = std::chrono::steady_clock::now();
        loaded_pairs.clear();
        loaded_pairs.resize(batch.size());
        errors.clear();
        errors.resize(batch.size());
        restored.assign(batch.size(), 0);
        entry_mapped_range_counts.assign(batch.size(), 0);
        entry_mapped_sizes.assign(batch.size(), 0);
        disk_entries.clear();
        disk_slots.clear();
        for (size_t index = 0; index < batch.size(); ++index)
        {
            if (victim_cache)
            {
                std::optional<ExpertVictimPair> victim = victim_cache->restore(
                    batch[index]->key,
                    batch[index]->gate_up_source,
                    batch[index]->down_source);
                if (victim)
                {
                    loaded_pairs[index] = std::move(*victim);
                    restored[index] = 1;
                    continue;
                }
            }
            disk_entries.push_back(batch[index]);
            disk_slots.push_back(index);
        }

        bool coalesced = false;
        uint64_t coalesced_mapped_range_count = 0;
        uint64_t coalesced_mapped_size = 0;
        uint64_t coalesced_saved_range_count = 0;
        if (disk_entries.size() > 1
            && has_flag(flags, ExpertCacheCrossExpertReadCoalescing))
        {
            auto loaded = load_coalesced_pairs(
                disk_entries,
                coalesced_mapped_range_count,
                coalesced_mapped_size,
                coalesced_saved_range_count,
                coalesced);
            if (!loaded)
            {
                for (size_t slot : disk_slots)
                    errors[slot] = loaded.error();
            }
            else if (coalesced)
            {
                std::vector<ExpertVictimPair> pairs = std::move(loaded).value();
                for (size_t index = 0; index < disk_slots.size(); ++index)
                    loaded_pairs[disk_slots[index]] = std::move(pairs[index]);
                entry_mapped_range_counts[disk_slots.front()] = coalesced_mapped_range_count;
                entry_mapped_sizes[disk_slots.front()] = coalesced_mapped_size;
            }
        }
        if (!coalesced)
        {
            for (size_t index = 0; index < disk_entries.size(); ++index)
            {
                const size_t slot = disk_slots[index];
                if (errors[slot])
                    continue;
                auto loaded = load_pair(
                    disk_entries[index]->gate_up_source,
                    disk_entries[index]->down_source,
                    entry_mapped_range_counts[slot],
                    entry_mapped_sizes[slot]);
                if (!loaded)
                {
                    errors[slot] = loaded.error();
                    continue;
                }
                loaded_pairs[slot] = std::move(loaded).value();
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!disk_entries.empty())
            {
                const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - io_started);
                const uint64_t elapsed_nanoseconds = std::max<int64_t>(1, elapsed.count());
                ++io_read_samples;
                io_read_time_nanoseconds += elapsed_nanoseconds;
                // The runtime thread budget is the single ownership boundary
                // for I/O concurrency.  Keep an explicit worker count stable;
                // changing it from sampled read latency is a calibration loop
                // and can starve the CPU Expert path exactly when a prompt is
                // already waiting on cold weights.
            }
            if (coalesced)
            {
                ++coalesced_read_batches;
                coalesced_experts += disk_entries.size();
                coalesced_read_ranges_saved += coalesced_saved_range_count;
            }
            for (size_t index = 0; index < batch.size(); ++index)
            {
                const std::shared_ptr<Entry>& entry = batch[index];
                if (errors[index])
                {
                    entry->state = Entry::State::Failed;
                    entry->error = *errors[index];
                    const auto existing = entries.find(entry->key);
                    if (existing != entries.end() && existing->second == entry)
                    {
                        remove_resident_locked(*entry, false);
                        entries.erase(existing);
                    }
                    continue;
                }
                if (has_flag(entry->flags, Entry::DiscardWhenReady)
                    && has_flag(entry->flags, Entry::Speculative))
                {
                    if (!restored[index])
                    {
                        bytes_read += entry->stored_size;
                        mapped_ranges += entry_mapped_range_counts[index];
                        mapped_bytes += entry_mapped_sizes[index];
                    }
                    entry->state = Entry::State::Failed;
                    entry->error = Error{ErrorCode::InternalError, "unused speculative Expert read was discarded"};
                    const auto existing = entries.find(entry->key);
                    if (existing != entries.end() && existing->second == entry)
                    {
                        remove_resident_locked(*entry, false);
                        entries.erase(existing);
                    }
                    ++unused_speculative_reads;
                    continue;
                }
                if (!loaded_pairs[index])
                {
                    entry->state = Entry::State::Failed;
                    entry->error = Error{ErrorCode::InternalError, "Expert read completed without a payload"};
                    const auto existing = entries.find(entry->key);
                    if (existing != entries.end() && existing->second == entry)
                    {
                        remove_resident_locked(*entry, false);
                        entries.erase(existing);
                    }
                    continue;
                }
                ExpertVictimPair pair = std::move(*loaded_pairs[index]);
                entry->gate_up = std::move(pair.gate_up);
                entry->down = std::move(pair.down);
                entry->state = Entry::State::Ready;
                if (!restored[index])
                {
                    bytes_read += entry->stored_size;
                    mapped_ranges += entry_mapped_range_counts[index];
                    mapped_bytes += entry_mapped_sizes[index];
                }
            }
            active_jobs -= static_cast<uint32_t>(batch.size());
            if (high_priority.empty() && low_priority.empty() && active_jobs == 0)
                idle.notify_all();
        }
        batch.clear();
        loaded_pairs.clear();
        errors.clear();
        restored.clear();
        entry_mapped_range_counts.clear();
        entry_mapped_sizes.clear();
        disk_entries.clear();
        disk_slots.clear();
        ready.notify_all();
    }
}

Result<ExpertCacheLease> Mxfp4ExpertCache::acquire_pair(
    const TensorData& gate_up,
    const TensorData& down,
    uint32_t residency_group,
    std::string_view prepared_key,
    ExpertVictimExecutionMetadata victim_execution)
{
    auto queued = enqueue_pair(
        gate_up,
        down,
        false,
        residency_group,
        prepared_key,
        victim_execution);
    if (!queued)
        return queued.error();
    std::shared_ptr<Entry> entry = std::move(queued).value();

    std::unique_lock<std::mutex> lock(mutex);
    ready.wait(lock, [&entry] {
        return entry->state != Entry::State::Loading;
    });
    if (entry->state == Entry::State::Failed)
        return entry->error;

    ++entry->exact_accesses;
    touch_resident_locked(*entry, entry->exact_accesses > 1);
    entry->flags &= ~Entry::Speculative;
    const bool cache_hit = !has_flag(entry->flags, Entry::FirstExactMiss);
    if (cache_hit)
        ++hits;
    entry->flags &= ~Entry::FirstExactMiss;

    ExpertCacheLease lease;
    lease.gate_up = entry->gate_up;
    lease.down = entry->down;
    lease.cache_hit = cache_hit;
    lease.bytes_read = cache_hit ? 0 : entry->stored_size;
    lease.pin = entry;
    return lease;
}

Result<bool> Mxfp4ExpertCache::try_acquire_ready_pairs(
    std::span<const ExpertCachePairRequest> requests,
    std::span<ExpertCacheLease> leases)
{
    if (requests.size() != leases.size())
    {
        return Error{
            ErrorCode::InvalidArgument,
            "Expert cache batch request and lease counts differ"};
    }
    for (const ExpertCachePairRequest& request : requests)
    {
        if (!request.gate_up
            || !request.down
            || !is_supported_file_backed_pair(
                *request.gate_up, *request.down)
            || request.prepared_key.empty())
        {
            return Error{
                ErrorCode::InvalidArgument,
                "Expert cache batch acquisition requires prepared file-backed pairs"};
        }
    }

    static constexpr size_t inline_entry_count = 16;
    std::array<std::shared_ptr<Entry>, inline_entry_count> inline_entries;
    std::vector<std::shared_ptr<Entry>> overflow_entries;
    std::shared_ptr<Entry>* ready_entries = inline_entries.data();
    if (requests.size() > inline_entry_count)
    {
        overflow_entries.resize(requests.size());
        ready_entries = overflow_entries.data();
    }

    std::lock_guard<std::mutex> lock(mutex);
    for (size_t index = 0; index < requests.size(); ++index)
    {
        const ExpertCachePairRequest& request = requests[index];
        const auto existing = entries.find(request.prepared_key);
        if (existing == entries.end() || existing->second->state != Entry::State::Ready)
        {
            return false;
        }
        ready_entries[index] = existing->second;
    }

    for (size_t index = 0; index < requests.size(); ++index)
    {
        const std::shared_ptr<Entry>& entry = ready_entries[index];
        if (requests[index].victim_execution.enabled)
        {
            entry->victim_execution = requests[index].victim_execution;
        }
        ++entry->exact_accesses;
        touch_resident_locked(*entry, entry->exact_accesses > 1);
        entry->flags &= ~Entry::Speculative;
        const bool cache_hit = !has_flag(entry->flags, Entry::FirstExactMiss);
        if (cache_hit)
            ++hits;
        entry->flags &= ~Entry::FirstExactMiss;

        ExpertCacheLease& lease = leases[index];
        lease.gate_up = entry->gate_up;
        lease.down = entry->down;
        lease.cache_hit = cache_hit;
        lease.bytes_read = cache_hit ? 0 : entry->stored_size;
        lease.pin = entry;
    }
    return true;
}

Result<size_t> Mxfp4ExpertCache::wait_acquire_ready_pairs(
    std::span<const ExpertCachePairRequest> requests,
    std::span<ExpertCacheLease> leases,
    bool wait_for_any)
{
    if (requests.size() != leases.size())
    {
        return Error{
            ErrorCode::InvalidArgument,
            "Expert cache batch request and lease counts differ"};
    }
    if (requests.empty())
    {
        return Error{
            ErrorCode::InvalidArgument,
            "Expert cache ready wait requires at least one pair"};
    }

    static constexpr size_t inline_entry_count = 16;
    std::array<std::shared_ptr<Entry>, inline_entry_count> inline_entries;
    std::vector<std::shared_ptr<Entry>> overflow_entries;
    std::shared_ptr<Entry>* ready_entries = inline_entries.data();
    if (requests.size() > inline_entry_count)
    {
        overflow_entries.resize(requests.size());
        ready_entries = overflow_entries.data();
    }
    size_t enqueued_count = 0;
    bool capacity_exhausted = false;
    for (size_t index = 0; index < requests.size(); ++index)
    {
        leases[index] = {};
        const ExpertCachePairRequest& request = requests[index];
        if (!request.gate_up
            || !request.down
            || !is_supported_file_backed_pair(
                *request.gate_up, *request.down)
            || request.prepared_key.empty())
        {
            return Error{
                ErrorCode::InvalidArgument,
                "Expert cache ready wait requires prepared file-backed pairs"};
        }
        for (;;)
        {
            bool temporarily_exhausted = false;
            auto queued = enqueue_pair(
                *request.gate_up,
                *request.down,
                false,
                request.residency_group,
                request.prepared_key,
                request.victim_execution,
                nullptr,
                &temporarily_exhausted);
            if (queued)
            {
                ready_entries[index] = std::move(queued).value();
                ++enqueued_count;
                break;
            }
            if (!temporarily_exhausted)
                return queued.error();
            if (enqueued_count != 0)
            {
                capacity_exhausted = true;
                break;
            }

            // A small host cache can have every resident entry pinned by an
            // in-flight speculative read or another foreground request.  Do
            // not turn that transient state into a hard failure when the
            // first exact request arrives; wait briefly for a read or lease
            // to become reclaimable, then retry the admission.
            std::unique_lock<std::mutex> wait_lock(mutex);
            if (stopping)
                return Error{ErrorCode::InternalError, "expert cache is stopping"};
            ready.wait_for(wait_lock, std::chrono::milliseconds(1));
        }
        if (capacity_exhausted)
        {
            // ready_entries is consumed as a prefix below.  Stop the batch at
            // the first capacity miss instead of leaving a null slot between
            // requests admitted by concurrent releases.
            break;
        }
    }

    std::unique_lock<std::mutex> lock(mutex);
    ready.wait(lock, [ready_entries,
                      request_count = enqueued_count,
                      wait_for_any] {
        if (!wait_for_any)
        {
            return ready_entries[0]->state != Entry::State::Loading;
        }
        for (size_t index = 0; index < request_count; ++index)
        {
            if (ready_entries[index]->state != Entry::State::Loading)
            {
                return true;
            }
        }
        return false;
    });

    for (size_t index = 0; index < enqueued_count; ++index)
    {
        const std::shared_ptr<Entry>& entry = ready_entries[index];
        if (entry->state == Entry::State::Failed)
            return entry->error;
    }

    size_t acquired = 0;
    for (size_t index = 0; index < enqueued_count; ++index)
    {
        const std::shared_ptr<Entry>& entry = ready_entries[index];
        if (entry->state != Entry::State::Ready)
            continue;
        ++entry->exact_accesses;
        touch_resident_locked(*entry, entry->exact_accesses > 1);
        entry->flags &= ~Entry::Speculative;
        const bool cache_hit = !has_flag(entry->flags, Entry::FirstExactMiss);
        if (cache_hit)
            ++hits;
        entry->flags &= ~Entry::FirstExactMiss;

        ExpertCacheLease& lease = leases[index];
        lease.gate_up = entry->gate_up;
        lease.down = entry->down;
        lease.cache_hit = cache_hit;
        lease.bytes_read = cache_hit ? 0 : entry->stored_size;
        lease.pin = entry;
        ++acquired;
    }
    return acquired;
}

bool Mxfp4ExpertCache::is_ready(const TensorData& gate_up, const TensorData& down, std::string_view prepared_key) const
{
    if (!is_supported_file_backed_pair(gate_up, down))
        return false;
    std::string generated_key;
    if (prepared_key.empty())
        generated_key = make_pair_key(gate_up, down);
    const std::string_view key = prepared_key.empty() ? std::string_view(generated_key) : prepared_key;
    std::lock_guard<std::mutex> lock(mutex);
    const auto existing = entries.find(key);
    return existing != entries.end() && existing->second->state == Entry::State::Ready;
}

ExpertCacheStatistics Mxfp4ExpertCache::statistics() const
{
    ExpertCacheStatistics result;
    std::shared_ptr<IExpertVictimCache> victim;
    {
        std::lock_guard<std::mutex> lock(mutex);
        result.hits = hits;
        result.misses = misses;
        result.evictions = evictions;
        result.bytes_read = bytes_read;
        result.resident_size = resident_size;
        result.queued_reads = queued_reads;
        result.speculative_reads = speculative_reads;
        result.cancelled_speculative_reads = cancelled_speculative_reads;
        result.dropped_speculative_admissions = dropped_speculative_admissions;
        result.unused_speculative_reads = unused_speculative_reads;
        result.short_term_reloads = short_term_reloads;
        result.coalesced_read_batches = coalesced_read_batches;
        result.coalesced_experts = coalesced_experts;
        result.coalesced_read_ranges_saved = coalesced_read_ranges_saved;
        result.arc_recent_size = arc_recent_size;
        result.arc_frequent_size = arc_frequent_size;
        result.arc_recent_target_size = arc_recent_target_size;
        result.arc_recent_ghost_size = arc_recent_ghost_size;
        result.arc_frequent_ghost_size = arc_frequent_ghost_size;
        result.arc_recent_ghost_hits = arc_recent_ghost_hits;
        result.arc_frequent_ghost_hits = arc_frequent_ghost_hits;
        result.mapped_ranges = mapped_ranges;
        result.mapped_bytes = mapped_bytes;
        result.num_io_threads = num_io_threads;
        result.num_active_io_threads = num_active_io_threads;
        result.io_read_samples = io_read_samples;
        result.io_read_time_microseconds = (io_read_time_nanoseconds + 999) / 1000;
        victim = victim_cache;
    }
    reader->populate_statistics(result);
    if (has_flag(flags, ExpertCacheMemoryMapRanges))
        result.adaptive_read_policy = 3;
    else if (has_flag(flags, ExpertCacheDirectReads))
    {
#if defined(_WIN32)
        result.adaptive_read_policy = FileRangeReader::ReadPolicyDirect;
#else
        result.adaptive_read_policy = FileRangeReader::ReadPolicyBuffered;
#endif
    }
    else if (has_flag(flags, ExpertCacheBufferedReads))
        result.adaptive_read_policy = FileRangeReader::ReadPolicyBuffered;
    if (victim)
        result.victim = victim->statistics();
    return result;
}

} // namespace moe
} // namespace ncnn
