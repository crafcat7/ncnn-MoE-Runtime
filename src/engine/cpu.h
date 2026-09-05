#ifndef NCNN_MOE_CPU_H
#define NCNN_MOE_CPU_H

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>

#define NCNN_MOE_CPUID_1_ECX_SSSE3_BIT        9
#define NCNN_MOE_CPUID_1_ECX_FMA_BIT          12
#define NCNN_MOE_CPUID_1_ECX_OSXSAVE_BIT      27
#define NCNN_MOE_CPUID_1_ECX_AVX_BIT          28
#define NCNN_MOE_CPUID_7_EBX_AVX2_BIT         5
#define NCNN_MOE_CPUID_7_EBX_AVX512F_BIT      16
#define NCNN_MOE_CPUID_7_EBX_AVX512BW_BIT     30
#define NCNN_MOE_CPUID_7_EBX_AVX512VL_BIT     31
#define NCNN_MOE_CPUID_7_ECX_AVX512VNNI_BIT   11
#define NCNN_MOE_CPUID_7_1_EAX_AVXVNNI_BIT    4
#define NCNN_MOE_CPUID_7_1_EAX_AVX512BF16_BIT 5
#define NCNN_MOE_CPUID_7_EDX_AMX_BF16_BIT     22
#define NCNN_MOE_CPUID_7_EDX_AMX_TILE_BIT     24
#define NCNN_MOE_CPUID_7_EDX_AMX_INT8_BIT     25
#define NCNN_MOE_XSTATE_XMM_BIT               1
#define NCNN_MOE_XSTATE_YMM_BIT               2
#define NCNN_MOE_XSTATE_OPMASK_BIT            5
#define NCNN_MOE_XSTATE_ZMM_HI256_BIT         6
#define NCNN_MOE_XSTATE_HI16_ZMM_BIT          7
#define NCNN_MOE_XSTATE_TILECFG_BIT           17
#define NCNN_MOE_XSTATE_TILEDATA_BIT          18

namespace ncnn {
namespace moe {

[[nodiscard]] uint64_t cpu_isa_flags() noexcept;
[[nodiscard]] std::string cpu_isa_names(uint64_t flags);
[[nodiscard]] uint32_t get_physical_cpu_count();

[[nodiscard]] uint64_t physical_memory_size() noexcept;
[[nodiscard]] uint64_t available_memory_size() noexcept;

struct CpuThreadBudget
{
    uint32_t num_threads = 1;
    uint32_t num_io_threads = 1;
    // Staged teams can expand into the static I/O and service reservation.
    uint32_t max_threads = 1;
};

[[nodiscard]] CpuThreadBudget resolve_cpu_thread_budget(uint32_t num_io_threads = 0) noexcept;

// Per-thread OpenMP limit used by scheduler workers.
class CpuOpenMpThreadLimitScope
{
public:
    CpuOpenMpThreadLimitScope() noexcept;
    ~CpuOpenMpThreadLimitScope() noexcept;

    CpuOpenMpThreadLimitScope(const CpuOpenMpThreadLimitScope&) = delete;
    CpuOpenMpThreadLimitScope& operator=(const CpuOpenMpThreadLimitScope&) = delete;

    void set(uint32_t limit) noexcept;

private:
    uint32_t previous = 0;
};

// Returns the current thread's OpenMP limit.
[[nodiscard]] uint32_t cpu_openmp_thread_limit() noexcept;

// Leases compute capacity without creating threads or managing I/O workers.
class CpuThreadBudgetController
{
public:
    class Lease
    {
    public:
        Lease() = default;
        ~Lease();

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;

        [[nodiscard]] uint32_t size() const noexcept
        { return count; }
        [[nodiscard]] bool empty() const noexcept
        { return count == 0; }

    private:
        friend class CpuThreadBudgetController;
        Lease(CpuThreadBudgetController* _owner, uint32_t _count, uint32_t _base_count) noexcept;
        void release() noexcept;

        CpuThreadBudgetController* owner = nullptr;
        uint32_t count = 0;
        uint32_t base_count = 0;
    };

    explicit CpuThreadBudgetController(CpuThreadBudget _thread_budget) noexcept;

    [[nodiscard]] const CpuThreadBudget& budget() const noexcept
    { return thread_budget; }

    // Workers use the base capacity; staged teams can also use extra capacity.
    [[nodiscard]] Lease acquire_compute(uint32_t requested, bool use_extra_threads = true);
    [[nodiscard]] Lease try_acquire_compute(uint32_t requested, bool use_extra_threads = true) noexcept;
    [[nodiscard]] uint32_t available(bool use_extra_threads = true) const noexcept;

private:
    [[nodiscard]] uint32_t available_locked(bool use_extra_threads) const noexcept;
    [[nodiscard]] Lease acquire_locked(uint32_t requested, bool use_extra_threads) noexcept;
    void release(const Lease& lease) noexcept;

    CpuThreadBudget thread_budget;
    mutable std::mutex mutex;
    std::condition_variable changed;
    uint32_t active_threads = 0;
    uint32_t active_base_threads = 0;
};

// Select a team size for the given work shape.
[[nodiscard]] uint32_t choose_cpu_team_size(
    uint64_t work_units,
    uint32_t independent_work_items,
    uint32_t threads_per_work_item,
    uint32_t available_threads) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_H
