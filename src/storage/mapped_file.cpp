#include "mapped_file.h"

#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ncnn {
namespace moe {

void prefetch_mapped_memory(const void* data, size_t byte_count) noexcept
{
    if (!data || byte_count == 0)
        return;
#if defined(_WIN32)
    // Resolve the API dynamically so the runtime still works on Windows
    // versions whose SDK does not expose PrefetchVirtualMemory.
    struct MemoryRange
    {
        void* virtual_address;
        SIZE_T number_of_bytes;
    };
    using PrefetchVirtualMemoryFunction = BOOL(WINAPI*)(HANDLE, ULONG_PTR, MemoryRange*, ULONG);
    static const PrefetchVirtualMemoryFunction prefetch = [] {
        HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
        if (!kernel32)
            return static_cast<PrefetchVirtualMemoryFunction>(nullptr);
        return reinterpret_cast<PrefetchVirtualMemoryFunction>(
            GetProcAddress(kernel32, "PrefetchVirtualMemory"));
    }();
    if (!prefetch)
        return;
    MemoryRange range{const_cast<void*>(data), byte_count};
    (void)prefetch(GetCurrentProcess(), 1, &range, 0);
#elif defined(MADV_WILLNEED)
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        return;
    const uintptr_t address = reinterpret_cast<uintptr_t>(data);
    const uintptr_t aligned = address & ~static_cast<uintptr_t>(page_size - 1);
    const size_t prefix = static_cast<size_t>(address - aligned);
    if (byte_count > std::numeric_limits<size_t>::max() - prefix)
        return;
    (void)madvise(reinterpret_cast<void*>(aligned), prefix + byte_count, MADV_WILLNEED);
#else
    (void)data;
    (void)byte_count;
#endif
}

class MappedFile
{
public:
    ~MappedFile()
    {
#if defined(_WIN32)
        if (mapping_ != nullptr)
            CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE)
            CloseHandle(file_);
#else
        if (file_ >= 0)
            close(file_);
#endif
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    [[nodiscard]] static Result<std::shared_ptr<MappedFile>> open(const std::filesystem::path& path)
    {
        auto file = std::shared_ptr<MappedFile>(new MappedFile());
#if defined(_WIN32)
        file->file_ = CreateFileW(
            path.wstring().c_str(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr,
            OPEN_EXISTING,
            // Expert routing touches slices in a demand-driven order across
            // layers; retain the random-access hint for the bounded working
            // set used by on-demand mappings.
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
            nullptr);
        if (file->file_ == INVALID_HANDLE_VALUE)
        {
            return Error{
                ErrorCode::IoError,
                "cannot open memory-mapped model shard: " + path.string()};
        }
        LARGE_INTEGER file_size{};
        if (!GetFileSizeEx(file->file_, &file_size) || file_size.QuadPart < 0)
        {
            return Error{
                ErrorCode::IoError,
                "cannot query memory-mapped model shard: " + path.string()};
        }
        file->size_ = static_cast<uint64_t>(file_size.QuadPart);
        // Model weights are immutable.  PAGE_WRITECOPY charges commit for the
        // full private view on Windows, which makes checkpoints larger than
        // physical memory impossible to map even when only a small working set
        // will be resident.
        file->mapping_ = CreateFileMappingW(file->file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (file->mapping_ == nullptr && file->size_ != 0)
        {
            return Error{
                ErrorCode::IoError,
                "cannot create model shard mapping: " + path.string()};
        }
        SYSTEM_INFO information{};
        GetSystemInfo(&information);
        file->granularity_ = static_cast<uint64_t>(information.dwAllocationGranularity);
#else
        file->file_ = ::open(path.c_str(), O_RDONLY);
        if (file->file_ < 0)
        {
            return Error{ErrorCode::IoError, "cannot open memory-mapped model shard: " + path.string() + ": " + std::strerror(errno)};
        }
        struct stat information
        {
        };
        if (fstat(file->file_, &information) != 0 || information.st_size < 0)
        {
            return Error{ErrorCode::IoError, "cannot query memory-mapped model shard: " + path.string() + ": " + std::strerror(errno)};
        }
        file->size_ = static_cast<uint64_t>(information.st_size);
        const long page_size = sysconf(_SC_PAGESIZE);
        file->granularity_ = page_size > 0 ? static_cast<uint64_t>(page_size) : UINT64_C(4096);
#endif
        return file;
    }

    [[nodiscard]] uint64_t size() const noexcept
    {
        return size_;
    }

    [[nodiscard]] uint64_t granularity() const noexcept
    {
        return granularity_;
    }

#if defined(_WIN32)
    [[nodiscard]] HANDLE mapping() const noexcept
    {
        return mapping_;
    }
#else
    [[nodiscard]] int descriptor() const noexcept
    {
        return file_;
    }
#endif

private:
    MappedFile() = default;

    uint64_t size_ = 0;
    uint64_t granularity_ = 0;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_ = -1;
#endif
};

static Result<std::shared_ptr<MappedFile>> acquire_mapped_file(const std::filesystem::path& path)
{
    // Mapping ownership is deliberately scoped to each MappedFileRange.  A
    // process-global weak cache made independent Runtime instances share file
    // handles and lifetime indirectly, which was incompatible with instance
    // isolation and made teardown depend on unrelated models.
    return MappedFile::open(path);
}

Result<std::shared_ptr<MappedFileRange>> MappedFileRange::open(const std::filesystem::path& path, uint64_t offset, uint64_t byte_count)
{
    if (byte_count == 0)
        return Error{ErrorCode::InvalidArgument, "cannot map an empty model shard range"};
    auto file_result = acquire_mapped_file(path);
    if (!file_result)
        return file_result.error();
    std::shared_ptr<MappedFile> file = std::move(file_result).value();
    if (offset > file->size() || byte_count > file->size() - offset)
    {
        return Error{
            ErrorCode::InvalidModel,
            "memory-mapped model shard range is truncated: " + path.string()};
    }
    const uint64_t granularity = file->granularity();
    if (granularity == 0)
        return Error{ErrorCode::IoError, "invalid model mapping granularity"};
    const uint64_t aligned_offset = offset - offset % granularity;
    const uint64_t prefix = offset - aligned_offset;
    if (byte_count > std::numeric_limits<uint64_t>::max() - prefix)
        return Error{ErrorCode::InvalidModel, "model shard mapping range overflows"};
    const uint64_t view_size = prefix + byte_count;
    if (view_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        return Error{ErrorCode::InvalidModel, "model shard mapping is too large"};
    }

    auto range = std::shared_ptr<MappedFileRange>(new MappedFileRange());
    range->file_ = std::move(file);
    range->view_size_ = static_cast<size_t>(view_size);
    range->size_ = static_cast<size_t>(byte_count);
#if defined(_WIN32)
    range->view_ = MapViewOfFile(
        range->file_->mapping(),
        FILE_MAP_READ,
        static_cast<DWORD>(aligned_offset >> 32),
        static_cast<DWORD>(aligned_offset),
        range->view_size_);
    if (range->view_ == nullptr)
    {
        return Error{
            ErrorCode::IoError,
            "cannot map model shard range: " + path.string()};
    }
#else
    if (aligned_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
    {
        return Error{ErrorCode::InvalidModel, "model shard mapping offset is too large"};
    }
    range->view_ = mmap(
        nullptr,
        range->view_size_,
        PROT_READ,
        MAP_PRIVATE,
        range->file_->descriptor(),
        static_cast<off_t>(aligned_offset));
    if (range->view_ == MAP_FAILED)
    {
        range->view_ = nullptr;
        return Error{ErrorCode::IoError, "cannot map model shard range: " + path.string() + ": " + std::strerror(errno)};
    }
#endif
    range->data_ = static_cast<uint8_t*>(range->view_) + static_cast<size_t>(prefix);
    return range;
}

MappedFileRange::~MappedFileRange()
{
    if (view_ == nullptr)
        return;
#if defined(_WIN32)
    UnmapViewOfFile(view_);
#else
#if defined(MADV_DONTNEED)
    (void)madvise(view_, view_size_, MADV_DONTNEED);
#endif
    munmap(view_, view_size_);
#endif
}

MxFp4ByteBuffer MappedFileRange::share_bytes()
{
    std::shared_ptr<uint8_t> shared(shared_from_this(), data_);
    return MxFp4ByteBuffer(std::move(shared), size_);
}

std::shared_ptr<const uint8_t> MappedFileRange::share_data()
{
    return std::shared_ptr<const uint8_t>(shared_from_this(), data_);
}

void MappedFileRange::prefault() const noexcept
{
#if !defined(_WIN32) && defined(MADV_WILLNEED)
    (void)madvise(view_, view_size_, MADV_WILLNEED);
#endif
    uint8_t checksum = 0;
    constexpr size_t page_size = 4096;
    for (size_t offset = 0; offset < size_; offset += page_size)
        checksum ^= data_[offset];
    checksum ^= data_[size_ - 1];
    volatile uint8_t sink = checksum;
    (void)sink;
}

} // namespace moe
} // namespace ncnn
