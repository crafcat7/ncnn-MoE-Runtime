#include "mappedfile.h"

#include <cerrno>
#include <cstring>
#include <limits>
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

MappedFileRange::MappedFileRange()
#if defined(_WIN32)
    : file_handle(INVALID_HANDLE_VALUE)
#endif
{
}

Result<std::shared_ptr<MappedFileRange>> MappedFileRange::open(const std::filesystem::path& path, uint64_t offset, uint64_t byte_count)
{
    if (byte_count == 0)
        return Error{ErrorCode::InvalidArgument, "cannot map an empty model shard range"};
    // Each range owns its handles; shared data keeps this range alive.
    auto range = std::shared_ptr<MappedFileRange>(new MappedFileRange());
    uint64_t file_size = 0;
    uint64_t granularity = 0;
#if defined(_WIN32)
    range->file_handle = CreateFileW(
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
    if (range->file_handle == INVALID_HANDLE_VALUE)
    {
        return Error{
            ErrorCode::IoError,
            "cannot open memory-mapped model shard: " + path.string()};
    }
    LARGE_INTEGER length{};
    if (!GetFileSizeEx(range->file_handle, &length) || length.QuadPart < 0)
    {
        return Error{
            ErrorCode::IoError,
            "cannot query memory-mapped model shard: " + path.string()};
    }
    file_size = static_cast<uint64_t>(length.QuadPart);
    // Model weights are immutable.  PAGE_WRITECOPY charges commit for the
    // full private view on Windows, which makes checkpoints larger than
    // physical memory impossible to map even when only a small working set
    // will be resident.
    range->mapping_handle = CreateFileMappingW(range->file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (range->mapping_handle == nullptr && file_size != 0)
    {
        return Error{
            ErrorCode::IoError,
            "cannot create model shard mapping: " + path.string()};
    }
    SYSTEM_INFO information{};
    GetSystemInfo(&information);
    granularity = static_cast<uint64_t>(information.dwAllocationGranularity);
#else
    range->file_handle = ::open(path.c_str(), O_RDONLY);
    if (range->file_handle < 0)
    {
        return Error{ErrorCode::IoError, "cannot open memory-mapped model shard: " + path.string() + ": " + std::strerror(errno)};
    }
    struct stat information{};
    if (fstat(range->file_handle, &information) != 0 || information.st_size < 0)
    {
        return Error{ErrorCode::IoError, "cannot query memory-mapped model shard: " + path.string() + ": " + std::strerror(errno)};
    }
    file_size = static_cast<uint64_t>(information.st_size);
    const long page_size = sysconf(_SC_PAGESIZE);
    granularity = page_size > 0 ? static_cast<uint64_t>(page_size) : UINT64_C(4096);
#endif

    if (offset > file_size || byte_count > file_size - offset)
    {
        return Error{
            ErrorCode::InvalidModel,
            "memory-mapped model shard range is truncated: " + path.string()};
    }
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

    range->view_size = static_cast<size_t>(view_size);
    range->data_size = static_cast<size_t>(byte_count);
#if defined(_WIN32)
    range->view = MapViewOfFile(
        range->mapping_handle,
        FILE_MAP_READ,
        static_cast<DWORD>(aligned_offset >> 32),
        static_cast<DWORD>(aligned_offset),
        range->view_size);
    if (range->view == nullptr)
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
    range->view = mmap(
        nullptr,
        range->view_size,
        PROT_READ,
        MAP_PRIVATE,
        range->file_handle,
        static_cast<off_t>(aligned_offset));
    if (range->view == MAP_FAILED)
    {
        range->view = nullptr;
        return Error{ErrorCode::IoError, "cannot map model shard range: " + path.string() + ": " + std::strerror(errno)};
    }
#endif
    range->data_ptr = static_cast<uint8_t*>(range->view) + static_cast<size_t>(prefix);
    return range;
}

MappedFileRange::~MappedFileRange()
{
#if defined(_WIN32)
    if (view != nullptr)
        UnmapViewOfFile(view);
    if (mapping_handle != nullptr)
        CloseHandle(mapping_handle);
    if (file_handle != INVALID_HANDLE_VALUE)
        CloseHandle(file_handle);
#else
    if (view != nullptr)
    {
#if defined(MADV_DONTNEED)
        (void)madvise(view, view_size, MADV_DONTNEED);
#endif
        munmap(view, view_size);
    }
    if (file_handle >= 0)
        close(file_handle);
#endif
}

MxFp4ByteBuffer MappedFileRange::share_bytes()
{
    std::shared_ptr<uint8_t> shared(shared_from_this(), data_ptr);
    return MxFp4ByteBuffer(std::move(shared), data_size);
}

std::shared_ptr<const uint8_t> MappedFileRange::share_data()
{
    return std::shared_ptr<const uint8_t>(shared_from_this(), data_ptr);
}

void MappedFileRange::prefault() const noexcept
{
#if !defined(_WIN32) && defined(MADV_WILLNEED)
    (void)madvise(view, view_size, MADV_WILLNEED);
#endif
    uint8_t checksum = 0;
    constexpr size_t page_size = 4096;
    for (size_t offset = 0; offset < data_size; offset += page_size)
        checksum ^= data_ptr[offset];
    checksum ^= data_ptr[data_size - 1];
    volatile uint8_t sink = checksum;
    (void)sink;
}

} // namespace moe
} // namespace ncnn
