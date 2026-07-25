#include "system_memory.h"

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

namespace ncnn {
namespace moe {

uint64_t physical_memory_bytes() noexcept
{
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status)
               ? static_cast<uint64_t>(status.ullTotalPhys)
               : 0;
#elif defined(__APPLE__)
    uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    return sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0
               ? bytes
               : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0)
        return 0;
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
#endif
}

} // namespace moe
} // namespace ncnn
