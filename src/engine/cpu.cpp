#include "cpu.h"

#include "ncnn/moe/runtime.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

static void append_name(std::string& names, const char* name)
{
    if (!names.empty())
        names += ',';
    names += name;
}

#if defined(__linux__)
static std::string read_first_line(const std::string& path)
{
    std::ifstream stream(path);
    std::string line;
    if (stream)
        std::getline(stream, line);
    return line;
}
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
static uint32_t maximum_cpuid_leaf() noexcept
{
    int registers[4] = {};
    __cpuid(registers, 0);
    return static_cast<uint32_t>(registers[0]);
}

static std::array<uint32_t, 4> cpuid(uint32_t leaf, uint32_t subleaf) noexcept
{
    int registers[4] = {};
    __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
    return {static_cast<uint32_t>(registers[0]), static_cast<uint32_t>(registers[1]), static_cast<uint32_t>(registers[2]), static_cast<uint32_t>(registers[3])};
}

static uint64_t enabled_xstate() noexcept
{
    return _xgetbv(0);
}
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
static uint32_t maximum_cpuid_leaf() noexcept
{
    return __get_cpuid_max(0, nullptr);
}

static std::array<uint32_t, 4> cpuid(uint32_t leaf, uint32_t subleaf) noexcept
{
    std::array<uint32_t, 4> registers = {};
    __cpuid_count(leaf, subleaf, registers[0], registers[1], registers[2], registers[3]);
    return registers;
}

static uint64_t enabled_xstate() noexcept
{
    uint32_t low = 0;
    uint32_t high = 0;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return static_cast<uint64_t>(high) << 32 | low;
}
#endif

static uint64_t detect_cpu_isa_flags() noexcept
{
    uint64_t flags = 0;

#if defined(__aarch64__) || defined(_M_ARM64)
    flags |= CpuIsaArmNeon;
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_SVE)
    const unsigned long hardware = getauxval(AT_HWCAP);
    if ((hardware & HWCAP_SVE) != 0)
    {
        flags |= CpuIsaArmSve;
    }
#elif defined(__ARM_FEATURE_SVE)
    flags |= CpuIsaArmSve;
#endif
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_SVE2)
    const unsigned long hardware2 = getauxval(AT_HWCAP2);
    if ((hardware2 & HWCAP2_SVE2) != 0)
    {
        flags |= CpuIsaArmSve2;
    }
#elif defined(__ARM_FEATURE_SVE2)
    flags |= CpuIsaArmSve2;
#endif
#elif defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    const uint32_t maximum_leaf = maximum_cpuid_leaf();
    if (maximum_leaf >= 1)
    {
        const std::array<uint32_t, 4> leaf1 = cpuid(1, 0);
        const uint32_t ecx = leaf1[2];
        const bool osxsave = (ecx & (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_OSXSAVE_BIT)) != 0;
        const bool avx = (ecx & (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_AVX_BIT)) != 0;
        const bool fma = (ecx & (UINT32_C(1) << NCNN_MOE_CPUID_1_ECX_FMA_BIT)) != 0;
        const uint64_t xstate = osxsave ? enabled_xstate() : 0;
        const uint64_t avx_state_mask = (UINT64_C(1) << NCNN_MOE_XSTATE_XMM_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_YMM_BIT);
        const bool avx_state = avx && (xstate & avx_state_mask) == avx_state_mask;
        if (maximum_leaf >= 7 && avx_state)
        {
            const std::array<uint32_t, 4> leaf7 = cpuid(7, 0);
            const uint32_t ebx = leaf7[1];
            const uint32_t ecx7 = leaf7[2];
            const uint32_t edx = leaf7[3];
            const bool avx2 = (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX2_BIT)) != 0;
            if (avx2 && fma)
            {
                flags |= CpuIsaX86Avx2Fma;
            }

            const uint64_t avx512_state_mask = avx_state_mask | (UINT64_C(1) << NCNN_MOE_XSTATE_OPMASK_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_ZMM_HI256_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_HI16_ZMM_BIT);
            const bool avx512_state = (xstate & avx512_state_mask) == avx512_state_mask;
            const bool avx512 = avx512_state
                                && (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512F_BIT)) != 0
                                && (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512BW_BIT)) != 0
                                && (ebx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EBX_AVX512VL_BIT)) != 0;
            if (avx512)
            {
                flags |= CpuIsaX86Avx512;
            }
            if (avx512 && (ecx7 & (UINT32_C(1) << NCNN_MOE_CPUID_7_ECX_AVX512VNNI_BIT)) != 0)
            {
                flags |= CpuIsaX86Avx512Vnni;
            }

            if (maximum_leaf >= 7)
            {
                const std::array<uint32_t, 4> leaf71 = cpuid(7, 1);
                if (avx2 && (leaf71[0] & (UINT32_C(1) << NCNN_MOE_CPUID_7_1_EAX_AVXVNNI_BIT)) != 0)
                {
                    flags |= CpuIsaX86AvxVnni;
                }
                if (avx512 && (leaf71[0] & (UINT32_C(1) << NCNN_MOE_CPUID_7_1_EAX_AVX512BF16_BIT)) != 0)
                {
                    flags |= CpuIsaX86Avx512Bf16;
                }
            }

            const uint64_t tile_state_mask = (UINT64_C(1) << NCNN_MOE_XSTATE_TILECFG_BIT) | (UINT64_C(1) << NCNN_MOE_XSTATE_TILEDATA_BIT);
            const bool tile_state = (xstate & tile_state_mask) == tile_state_mask;
            const bool amx_tile = tile_state && (edx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EDX_AMX_TILE_BIT)) != 0;
            if (amx_tile)
            {
                flags |= CpuIsaX86AmxTile;
            }
            if (amx_tile && (edx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EDX_AMX_INT8_BIT)) != 0)
            {
                flags |= CpuIsaX86AmxInt8;
            }
            if (amx_tile && (edx & (UINT32_C(1) << NCNN_MOE_CPUID_7_EDX_AMX_BF16_BIT)) != 0)
            {
                flags |= CpuIsaX86AmxBf16;
            }
        }
    }
#endif

    return flags;
}

uint32_t get_physical_cpu_count()
{
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        return 0;

    std::set<std::pair<std::string, std::string>> physical_cores;
    for (uint32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    {
        if (!CPU_ISSET(cpu, &allowed))
            continue;
        const std::string prefix = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        const std::string package_id = read_first_line(prefix + "physical_package_id");
        const std::string core_id = read_first_line(prefix + "core_id");
        if (!package_id.empty() && !core_id.empty())
            physical_cores.emplace(package_id, core_id);
    }
    return static_cast<uint32_t>(physical_cores.size());
#elif defined(_WIN32)
    DWORD byte_count = 0;
    (void)GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &byte_count);
    if (byte_count == 0)
        return 0;

    std::vector<uint8_t> storage(byte_count);
    auto* information = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, information, &byte_count))
        return 0;

    uint32_t physical_cpu_count = 0;
    DWORD offset = 0;
    while (offset < byte_count)
    {
        auto* entry = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data() + offset);
        if (entry->Relationship == RelationProcessorCore)
            ++physical_cpu_count;
        if (entry->Size == 0)
            break;
        offset += entry->Size;
    }
    return physical_cpu_count;
#else
    return 0;
#endif
}

uint64_t cpu_isa_flags() noexcept
{
    static const uint64_t flags = detect_cpu_isa_flags();
    return flags;
}

std::string cpu_isa_names(uint64_t flags)
{
    std::string names;
    if (has_flag(flags, CpuIsaArmNeon))
        append_name(names, "neon");
    if (has_flag(flags, CpuIsaArmSve))
        append_name(names, "sve");
    if (has_flag(flags, CpuIsaArmSve2))
        append_name(names, "sve2");
    if (has_flag(flags, CpuIsaX86Avx2Fma))
        append_name(names, "avx2-fma");
    if (has_flag(flags, CpuIsaX86Avx512))
        append_name(names, "avx512");
    if (has_flag(flags, CpuIsaX86Avx512Vnni))
        append_name(names, "avx512-vnni");
    if (has_flag(flags, CpuIsaX86AvxVnni))
        append_name(names, "avx-vnni");
    if (has_flag(flags, CpuIsaX86Avx512Bf16))
        append_name(names, "avx512-bf16");
    if (has_flag(flags, CpuIsaX86AmxTile))
        append_name(names, "amx-tile");
    if (has_flag(flags, CpuIsaX86AmxInt8))
        append_name(names, "amx-int8");
    if (has_flag(flags, CpuIsaX86AmxBf16))
        append_name(names, "amx-bf16");
    return names.empty() ? "scalar" : names;
}

uint64_t physical_memory_size() noexcept
{
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? static_cast<uint64_t>(status.ullTotalPhys) : 0;
#elif defined(__APPLE__)
    uint64_t memory_size = 0;
    size_t size = sizeof(memory_size);
    return sysctlbyname("hw.memsize", &memory_size, &size, nullptr, 0) == 0 ? memory_size : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0)
        return 0;
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
#endif
}

uint64_t available_memory_size() noexcept
{
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? static_cast<uint64_t>(status.ullAvailPhys) : 0;
#elif defined(__APPLE__)
    // macOS does not expose a stable cross-version equivalent of
    // ullAvailPhys through sysctl.  The planner falls back to total memory.
    return 0;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0)
        return 0;
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
#endif
}

#if defined(_OPENMP)
static thread_local uint32_t cpu_openmp_thread_limit_override = 0;
#endif

CpuThreadBudget resolve_cpu_thread_budget(uint32_t num_io_threads) noexcept
{
    const uint32_t max_threads = std::max(1u, get_physical_cpu_count());
    if (num_io_threads == 0)
        num_io_threads = std::min(6u, std::max(2u, max_threads / 2u));
    num_io_threads = std::min(num_io_threads, max_threads);

    // Leave one core for service work in addition to the I/O reservation.
    const uint32_t reserved = static_cast<uint32_t>(std::min<uint64_t>(
        max_threads, static_cast<uint64_t>(num_io_threads) + 1));
    return {std::max(1u, max_threads - reserved), num_io_threads, max_threads};
}

CpuOpenMpThreadLimitScope::CpuOpenMpThreadLimitScope() noexcept
#if defined(_OPENMP)
    : previous(cpu_openmp_thread_limit_override)
#endif
{
}

CpuOpenMpThreadLimitScope::~CpuOpenMpThreadLimitScope() noexcept
{
#if defined(_OPENMP)
    cpu_openmp_thread_limit_override = previous;
#endif
}

void CpuOpenMpThreadLimitScope::set(uint32_t limit) noexcept
{
#if defined(_OPENMP)
    cpu_openmp_thread_limit_override = std::max(1u, limit);
#else
    (void)limit;
#endif
}

uint32_t cpu_openmp_thread_limit() noexcept
{
#if defined(_OPENMP)
    if (cpu_openmp_thread_limit_override != 0)
        return cpu_openmp_thread_limit_override;
    return static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
#else
    return 1;
#endif
}

CpuThreadBudgetController::Lease::Lease(
    CpuThreadBudgetController* _owner, uint32_t _count, uint32_t _base_count) noexcept
    : owner(_owner), count(_count), base_count(_base_count)
{
}

CpuThreadBudgetController::Lease::~Lease()
{
    release();
}

CpuThreadBudgetController::Lease::Lease(Lease&& other) noexcept
    : owner(std::exchange(other.owner, nullptr)),
      count(std::exchange(other.count, 0)),
      base_count(std::exchange(other.base_count, 0))
{
}

CpuThreadBudgetController::Lease& CpuThreadBudgetController::Lease::operator=(Lease&& other) noexcept
{
    if (this == &other)
        return *this;
    release();
    owner = std::exchange(other.owner, nullptr);
    count = std::exchange(other.count, 0);
    base_count = std::exchange(other.base_count, 0);
    return *this;
}

void CpuThreadBudgetController::Lease::release() noexcept
{
    if (owner)
        owner->release(*this);
    owner = nullptr;
    count = 0;
    base_count = 0;
}

CpuThreadBudgetController::CpuThreadBudgetController(CpuThreadBudget _thread_budget) noexcept
    : thread_budget(_thread_budget)
{
    thread_budget.max_threads = std::max(1u, thread_budget.max_threads);
    thread_budget.num_threads = std::max(1u, std::min(thread_budget.num_threads, thread_budget.max_threads));
}

uint32_t CpuThreadBudgetController::available_locked(bool use_extra_threads) const noexcept
{
    const uint32_t remaining = thread_budget.max_threads - active_threads;
    if (use_extra_threads)
        return remaining;
    return std::min(remaining, thread_budget.num_threads - active_base_threads);
}

CpuThreadBudgetController::Lease CpuThreadBudgetController::acquire_locked(
    uint32_t requested, bool use_extra_threads) noexcept
{
    const uint32_t count = std::min(requested, available_locked(use_extra_threads));
    if (count == 0)
        return {};
    const uint32_t base_count = std::min(count, thread_budget.num_threads - active_base_threads);
    active_threads += count;
    active_base_threads += base_count;
    return Lease(this, count, base_count);
}

CpuThreadBudgetController::Lease CpuThreadBudgetController::acquire_compute(
    uint32_t requested, bool use_extra_threads)
{
    requested = std::max(1u, requested);
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [this, use_extra_threads] {
        return available_locked(use_extra_threads) != 0;
    });
    return acquire_locked(requested, use_extra_threads);
}

CpuThreadBudgetController::Lease CpuThreadBudgetController::try_acquire_compute(
    uint32_t requested, bool use_extra_threads) noexcept
{
    const std::lock_guard<std::mutex> lock(mutex);
    return acquire_locked(requested, use_extra_threads);
}

uint32_t CpuThreadBudgetController::available(bool use_extra_threads) const noexcept
{
    const std::lock_guard<std::mutex> lock(mutex);
    return available_locked(use_extra_threads);
}

void CpuThreadBudgetController::release(const Lease& lease) noexcept
{
    {
        const std::lock_guard<std::mutex> lock(mutex);
        active_threads -= lease.count;
        active_base_threads -= lease.base_count;
    }
    changed.notify_all();
}

uint32_t choose_cpu_team_size(
    uint64_t work_units,
    uint32_t independent_work_items,
    uint32_t threads_per_work_item,
    uint32_t available_threads) noexcept
{
    available_threads = std::max(1u, available_threads);
    independent_work_items = std::max(1u, independent_work_items);
    threads_per_work_item = std::max(1u, threads_per_work_item);
    if (work_units == 0)
        return 1;

    const uint64_t average_work = work_units / independent_work_items + (work_units % independent_work_items != 0);
    const uint32_t useful_threads_per_item = static_cast<uint32_t>(std::min<uint64_t>(
        threads_per_work_item,
        std::max<uint64_t>(1, average_work)));
    const uint64_t requested = static_cast<uint64_t>(independent_work_items) * useful_threads_per_item;
    const uint32_t shape_limit = static_cast<uint32_t>(std::min<uint64_t>(
        available_threads,
        std::max<uint64_t>(1, requested)));
    return std::max(1u, shape_limit);
}

} // namespace moe
} // namespace ncnn
