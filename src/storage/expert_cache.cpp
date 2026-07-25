#include "expert_cache.h"
#include "storage/mapped_file.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <limits>
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

static HANDLE thread_read_event()
{
    static thread_local ThreadReadEvent event;
    return event.handle;
}
#endif

struct Mxfp4ExpertCache::Entry
{
    enum class State
    {
        Loading,
        Ready,
        Failed
    };

    State state = State::Loading;
    std::string key;
    uint64_t bytes = 0;
    uint64_t used_at = 0;
    bool speculative = false;
    bool job_started = false;
    bool first_exact_acquire_is_miss = false;
    Error error;
    TensorData gate_up_source;
    TensorData down_source;
    std::shared_ptr<TensorData> gate_up;
    std::shared_ptr<TensorData> down;
};

struct Mxfp4ExpertCache::FileRangeReader
{
    struct LoadedRange
    {
        MxFp4ByteBuffer bytes;
        bool mapped = false;
    };

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
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto& item : handles) {
#if defined(_WIN32)
            CloseHandle(item.second);
#else
            close(item.second);
#endif
        }
    }

    Result<LoadedRange> load(
        const std::string& path,
        uint64_t offset,
        uint64_t byte_count,
        uint32_t flags)
    {
        if (byte_count
            > static_cast<uint64_t>(
                std::numeric_limits<size_t>::max())) {
            return Error{
                ErrorCode::InvalidModel,
                "expert shard range is too large: " + path};
        }
        if (has_flag(flags, ExpertCacheMemoryMapRanges)) {
            auto mapping = MappedFileRange::open(
                path,
                offset,
                byte_count);
            if (mapping) {
                mapping.value()->prefault();
                LoadedRange loaded;
                loaded.bytes = mapping.value()->share_bytes();
                loaded.mapped = true;
                return loaded;
            }
        }

        LoadedRange loaded;
        loaded.bytes.resize(static_cast<size_t>(byte_count));
        auto status = read(path, offset, loaded.bytes);
        if (!status)
            return status.error();
        return loaded;
    }

    Result<void> read(
        const std::string& path,
        uint64_t offset,
        MxFp4ByteBuffer& destination)
    {
        auto handle_result = handle_for(path);
        if (!handle_result)
            return handle_result.error();
        const Handle handle = handle_result.value();

        size_t completed = 0;
        while (completed < destination.size()) {
#if defined(_WIN32)
            const size_t remaining = destination.size() - completed;
            const DWORD request = static_cast<DWORD>(std::min<size_t>(
                remaining,
                static_cast<size_t>(std::numeric_limits<DWORD>::max())));
            const uint64_t current_offset = offset + completed;
            OVERLAPPED operation{};
            operation.Offset = static_cast<DWORD>(current_offset);
            operation.OffsetHigh = static_cast<DWORD>(current_offset >> 32);
            operation.hEvent = thread_read_event();
            if (operation.hEvent == nullptr) {
                return Error{
                    ErrorCode::IoError,
                    "cannot create expert shard read event: " + path};
            }
            if (!ResetEvent(operation.hEvent)) {
                return Error{
                    ErrorCode::IoError,
                    "cannot reset expert shard read event: " + path};
            }

            DWORD read_bytes = 0;
            const BOOL started = ReadFile(
                handle,
                destination.data() + completed,
                request,
                nullptr,
                &operation);
            const DWORD start_error = started ? ERROR_SUCCESS : GetLastError();
            if (!started && start_error != ERROR_IO_PENDING) {
                return Error{
                    ErrorCode::IoError,
                    "cannot read expert shard range: " + path};
            }
            if (!GetOverlappedResult(handle, &operation, &read_bytes, TRUE)) {
                return Error{
                    ErrorCode::IoError,
                    "cannot complete expert shard range: " + path};
            }
            if (read_bytes == 0)
                return Error{ErrorCode::IoError, "expert shard range is truncated: " + path};
            completed += read_bytes;
#else
            const size_t remaining = destination.size() - completed;
            const size_t request = std::min<size_t>(
                remaining,
                static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
            const uint64_t current_offset = offset + completed;
            if (current_offset
                > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
                return Error{ErrorCode::InvalidModel, "expert file offset is too large: " + path};
            }
            const ssize_t read_bytes = pread(
                handle,
                destination.data() + completed,
                request,
                static_cast<off_t>(current_offset));
            if (read_bytes < 0) {
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
        if (!destination.empty()) {
            posix_fadvise(
                handle,
                static_cast<off_t>(offset),
                static_cast<off_t>(destination.size()),
                POSIX_FADV_DONTNEED);
        }
#endif
        return {};
    }

private:
    Result<Handle> handle_for(const std::string& path)
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto existing = handles.find(path);
        if (existing != handles.end())
            return existing->second;

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
        // Dedicated expert handles bypass the unified page cache so streamed
        // weights do not evict dense weights and KV state on constrained Macs.
        (void)fcntl(handle, F_NOCACHE, 1);
#endif
        handles.emplace(path, handle);
        return handle;
    }

    std::mutex mutex;
    std::unordered_map<std::string, Handle> handles;
};

Mxfp4ExpertCache::Mxfp4ExpertCache(
    uint64_t capacity_bytes,
    uint32_t io_worker_count,
    std::shared_ptr<IExpertVictimCache> victim_cache,
    uint32_t flags)
    : capacity_bytes_(capacity_bytes),
      reader_(std::make_unique<FileRangeReader>()),
      victim_cache_(std::move(victim_cache)),
      flags_(flags)
{
    if (io_worker_count == 0) {
        const uint32_t hardware = std::max(1u, std::thread::hardware_concurrency());
        io_worker_count = std::min(4u, hardware);
    }
    workers_.reserve(io_worker_count);
    for (uint32_t worker = 0; worker < io_worker_count; ++worker)
        workers_.emplace_back(&Mxfp4ExpertCache::worker_loop, this);
    predictor_ = std::thread(&Mxfp4ExpertCache::predictor_loop, this);
}

Mxfp4ExpertCache::~Mxfp4ExpertCache()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        low_priority_.clear();
        pending_prediction_ = {};
    }
    work_available_.notify_all();
    prediction_available_.notify_all();
    if (predictor_.joinable())
        predictor_.join();
    for (std::thread& worker : workers_) {
        if (worker.joinable())
            worker.join();
    }
}

void Mxfp4ExpertCache::submit_prediction(
    std::function<void(uint64_t)> prediction)
{
    if (!prediction)
        return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_)
            return;
        pending_prediction_generation_ = ++prediction_generation_;
        pending_prediction_ = std::move(prediction);
    }
    prediction_available_.notify_one();
}

void Mxfp4ExpertCache::cancel_prediction()
{
    std::lock_guard<std::mutex> lock(mutex_);
    ++prediction_generation_;
    pending_prediction_ = {};
}

bool Mxfp4ExpertCache::prediction_is_current(uint64_t generation) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return !stopping_ && generation == prediction_generation_;
}

void Mxfp4ExpertCache::predictor_loop()
{
    for (;;) {
        std::function<void(uint64_t)> prediction;
        uint64_t generation = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            prediction_available_.wait(lock, [this] {
                return stopping_ || static_cast<bool>(pending_prediction_);
            });
            if (stopping_)
                return;
            prediction = std::move(pending_prediction_);
            generation = pending_prediction_generation_;
            pending_prediction_ = {};
        }
        if (prediction_is_current(generation))
            prediction(generation);
    }
}

Result<uint64_t> Mxfp4ExpertCache::stored_bytes(const TensorData& tensor)
{
    if (tensor.dtype != DType::MxFp4 || !tensor.mxfp4_file_storage)
        return Error{ErrorCode::InvalidArgument, "expert cache requires file-backed MXFP4 tensors"};
    const MxFp4FileStorage& source = *tensor.mxfp4_file_storage;
    if (source.scales_bytes > std::numeric_limits<uint64_t>::max() - source.blocks_bytes)
        return Error{ErrorCode::InvalidModel, "file-backed MXFP4 tensor byte count overflows"};
    return source.blocks_bytes + source.scales_bytes;
}

std::string Mxfp4ExpertCache::key_for(
    const TensorData& gate_up,
    const TensorData& down)
{
    const MxFp4FileStorage& gate = *gate_up.mxfp4_file_storage;
    const MxFp4FileStorage& projection = *down.mxfp4_file_storage;
    return gate.blocks_path + ":" + std::to_string(gate.blocks_offset)
           + ":" + std::to_string(gate.blocks_bytes)
           + ":" + gate.scales_path + ":" + std::to_string(gate.scales_offset)
           + ":" + std::to_string(gate.scales_bytes)
           + "|" + projection.blocks_path + ":" + std::to_string(projection.blocks_offset)
           + ":" + std::to_string(projection.blocks_bytes)
           + ":" + projection.scales_path + ":" + std::to_string(projection.scales_offset)
           + ":" + std::to_string(projection.scales_bytes);
}

Result<std::shared_ptr<TensorData> > Mxfp4ExpertCache::load_tensor(
    const TensorData& source,
    uint64_t& mapped_ranges,
    uint64_t& mapped_bytes)
{
    if (!source.mxfp4_file_storage)
        return Error{ErrorCode::InvalidArgument, "MXFP4 tensor is not file-backed"};
    const MxFp4FileStorage& file = *source.mxfp4_file_storage;
    if (file.blocks_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())
        || file.scales_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return Error{ErrorCode::InvalidModel, "file-backed MXFP4 tensor is too large"};
    }

    auto loaded = std::make_shared<TensorData>();
    loaded->dtype = DType::MxFp4;
    loaded->shape = source.shape;
    auto blocks = reader_->load(
        file.blocks_path,
        file.blocks_offset,
        file.blocks_bytes,
        flags_);
    if (!blocks)
        return blocks.error();
    auto scales = reader_->load(
        file.scales_path,
        file.scales_offset,
        file.scales_bytes,
        flags_);
    if (!scales)
        return scales.error();
    if (blocks.value().mapped) {
        ++mapped_ranges;
        mapped_bytes += file.blocks_bytes;
    }
    if (scales.value().mapped) {
        ++mapped_ranges;
        mapped_bytes += file.scales_bytes;
    }
    loaded->mxfp4_blocks
        = std::move(blocks).value().bytes;
    loaded->mxfp4_scales
        = std::move(scales).value().bytes;
    return loaded;
}

bool Mxfp4ExpertCache::evict_one_locked()
{
    auto victim = entries_.end();
    for (auto iterator = entries_.begin(); iterator != entries_.end(); ++iterator) {
        const std::shared_ptr<Entry>& entry = iterator->second;
        if (entry->state != Entry::State::Ready || entry.use_count() != 1)
            continue;
        if (victim == entries_.end()
            || entry->used_at < victim->second->used_at) {
            victim = iterator;
        }
    }
    if (victim == entries_.end())
        return false;
    if (victim_cache_) {
        victim_cache_->admit(
            victim->second->key,
            victim->second->gate_up,
            victim->second->down);
    }
    resident_bytes_ -= victim->second->bytes;
    entries_.erase(victim);
    ++evictions_;
    return true;
}

Result<std::shared_ptr<Mxfp4ExpertCache::Entry> > Mxfp4ExpertCache::enqueue_pair(
    const TensorData& gate_up,
    const TensorData& down,
    bool speculative)
{
    auto gate_bytes = stored_bytes(gate_up);
    if (!gate_bytes)
        return gate_bytes.error();
    auto down_bytes = stored_bytes(down);
    if (!down_bytes)
        return down_bytes.error();
    if (down_bytes.value() > std::numeric_limits<uint64_t>::max() - gate_bytes.value())
        return Error{ErrorCode::InvalidModel, "expert pair byte count overflows"};
    const uint64_t required = gate_bytes.value() + down_bytes.value();
    if (required > capacity_bytes_) {
        if (speculative)
            return std::shared_ptr<Entry>();
        return Error{
            ErrorCode::InvalidArgument,
            "expert cache is smaller than one MXFP4 expert pair"};
    }

    const std::string key = key_for(gate_up, down);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = entries_.find(key);
    if (existing != entries_.end()) {
        const std::shared_ptr<Entry>& entry = existing->second;
        if (!speculative && entry->speculative && entry->state == Entry::State::Loading) {
            entry->speculative = false;
            entry->first_exact_acquire_is_miss = true;
            ++misses_;
            if (!entry->job_started)
                high_priority_.push_back(entry);
            work_available_.notify_one();
        }
        return entry;
    }

    while (resident_bytes_ > capacity_bytes_ - required) {
        if (!evict_one_locked()) {
            if (speculative)
                return std::shared_ptr<Entry>();
            return Error{
                ErrorCode::InvalidArgument,
                "expert cache capacity is exhausted by active expert leases or reads"};
        }
    }

    auto entry = std::make_shared<Entry>();
    entry->key = key;
    entry->bytes = required;
    entry->used_at = ++clock_;
    entry->speculative = speculative;
    entry->first_exact_acquire_is_miss = !speculative;
    entry->gate_up_source = gate_up;
    entry->down_source = down;
    entries_.emplace(key, entry);
    resident_bytes_ += required;
    ++queued_reads_;
    if (speculative) {
        ++speculative_reads_;
        low_priority_.push_back(entry);
    }
    else {
        ++misses_;
        high_priority_.push_back(entry);
    }
    work_available_.notify_one();
    return entry;
}

Result<void> Mxfp4ExpertCache::request_pair(
    const TensorData& gate_up,
    const TensorData& down)
{
    auto entry = enqueue_pair(gate_up, down, false);
    if (!entry)
        return entry.error();
    return {};
}

Result<void> Mxfp4ExpertCache::prefetch_pair(
    const TensorData& gate_up,
    const TensorData& down)
{
    auto entry = enqueue_pair(gate_up, down, true);
    if (!entry)
        return entry.error();
    return {};
}

void Mxfp4ExpertCache::worker_loop()
{
    for (;;) {
        std::shared_ptr<Entry> entry;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_available_.wait(lock, [this] {
                return stopping_ || !high_priority_.empty() || !low_priority_.empty();
            });
            if (stopping_ && high_priority_.empty() && low_priority_.empty())
                return;
            for (;;) {
                if (!high_priority_.empty()) {
                    entry = std::move(high_priority_.front());
                    high_priority_.pop_front();
                }
                else if (!low_priority_.empty()) {
                    entry = std::move(low_priority_.front());
                    low_priority_.pop_front();
                }
                else {
                    break;
                }
                if (entry && !entry->job_started
                    && entry->state == Entry::State::Loading) {
                    entry->job_started = true;
                    break;
                }
                entry.reset();
            }
            if (!entry)
                continue;
        }

        std::optional<ExpertVictimPair> restored;
        if (victim_cache_) {
            restored = victim_cache_->restore(
                entry->key,
                entry->gate_up_source,
                entry->down_source);
        }
        uint64_t mapped_ranges = 0;
        uint64_t mapped_bytes = 0;
        auto loaded_gate = restored
                               ? Result<std::shared_ptr<TensorData> >(
                                     restored->gate_up)
                               : load_tensor(
                                     entry->gate_up_source,
                                     mapped_ranges,
                                     mapped_bytes);
        auto loaded_down = restored
                               ? Result<std::shared_ptr<TensorData> >(
                                     restored->down)
                           : loaded_gate
                               ? load_tensor(
                                     entry->down_source,
                                     mapped_ranges,
                                     mapped_bytes)
                               : Result<std::shared_ptr<TensorData> >(
                                     loaded_gate.error());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!loaded_gate || !loaded_down) {
                entry->state = Entry::State::Failed;
                entry->error = !loaded_gate ? loaded_gate.error() : loaded_down.error();
                const auto existing = entries_.find(entry->key);
                if (existing != entries_.end() && existing->second == entry) {
                    resident_bytes_ -= entry->bytes;
                    entries_.erase(existing);
                }
            }
            else {
                entry->gate_up = std::move(loaded_gate).value();
                entry->down = std::move(loaded_down).value();
                entry->state = Entry::State::Ready;
                entry->used_at = ++clock_;
                if (!restored) {
                    bytes_read_ += entry->bytes;
                    mapped_ranges_ += mapped_ranges;
                    mapped_bytes_ += mapped_bytes;
                }
            }
        }
        entry.reset();
        ready_.notify_all();
    }
}

Result<ExpertCacheLease> Mxfp4ExpertCache::acquire_pair(
    const TensorData& gate_up,
    const TensorData& down)
{
    auto queued = enqueue_pair(gate_up, down, false);
    if (!queued)
        return queued.error();
    const std::shared_ptr<Entry> entry = queued.value();

    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [&entry] {
        return entry->state != Entry::State::Loading;
    });
    if (entry->state == Entry::State::Failed)
        return entry->error;

    entry->used_at = ++clock_;
    const bool cache_hit = !entry->first_exact_acquire_is_miss;
    if (cache_hit)
        ++hits_;
    entry->first_exact_acquire_is_miss = false;

    ExpertCacheLease lease;
    lease.gate_up = entry->gate_up;
    lease.down = entry->down;
    lease.cache_hit = cache_hit;
    lease.bytes_read = cache_hit ? 0 : entry->bytes;
    lease.pin = entry;
    return lease;
}

bool Mxfp4ExpertCache::is_ready(
    const TensorData& gate_up,
    const TensorData& down) const
{
    if (!gate_up.mxfp4_file_storage || !down.mxfp4_file_storage)
        return false;
    const std::string key = key_for(gate_up, down);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = entries_.find(key);
    return existing != entries_.end()
           && existing->second->state == Entry::State::Ready;
}

ExpertCacheStatistics Mxfp4ExpertCache::statistics() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    ExpertCacheStatistics result;
    result.hits = hits_;
    result.misses = misses_;
    result.evictions = evictions_;
    result.bytes_read = bytes_read_;
    result.resident_bytes = resident_bytes_;
    result.queued_reads = queued_reads_;
    result.speculative_reads = speculative_reads_;
    result.mapped_ranges = mapped_ranges_;
    result.mapped_bytes = mapped_bytes_;
    if (victim_cache_)
        result.victim = victim_cache_->statistics();
    return result;
}

} // namespace moe
} // namespace ncnn
