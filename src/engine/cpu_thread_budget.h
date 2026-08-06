#ifndef NCNN_MOE_CPU_THREAD_BUDGET_H
#define NCNN_MOE_CPU_THREAD_BUDGET_H

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace ncnn {
namespace moe {

struct CpuThreadBudget
{
    uint32_t logical_threads = 1;
    uint32_t physical_cores = 1;
    uint32_t reserved_io_threads = 1;
    uint32_t reserved_service_threads = 1;
    uint32_t compute_threads = 1;
    // The hard compute cap includes reservations that can be borrowed while
    // their owning pool is idle. An explicit compute request also acts as a
    // cap, so callers can keep the previous fixed-budget behavior.
    uint32_t max_compute_threads = 1;
};

[[nodiscard]] CpuThreadBudget resolve_cpu_thread_budget(
    uint32_t requested_io_threads = 0,
    uint32_t requested_service_threads = 1,
    uint32_t requested_compute_threads = 0) noexcept;

// OpenMP's process-level thread setting is not isolated reliably by every
// Windows runtime. Keep the scheduler's execution cap in a thread-local
// override so a worker cannot lower the limit seen by an unrelated session.
class CpuOpenMpThreadLimitScope
{
public:
    CpuOpenMpThreadLimitScope() noexcept;
    explicit CpuOpenMpThreadLimitScope(uint32_t limit) noexcept;
    ~CpuOpenMpThreadLimitScope() noexcept;

    CpuOpenMpThreadLimitScope(const CpuOpenMpThreadLimitScope&) = delete;
    CpuOpenMpThreadLimitScope& operator=(const CpuOpenMpThreadLimitScope&) = delete;

    void set(uint32_t limit) noexcept;

private:
    uint32_t previous_ = 0;
};

// Returns the current thread's OpenMP team limit before topology clipping.
// A zero scope override falls back to the runtime's configured limit.
[[nodiscard]] uint32_t cpu_openmp_thread_limit() noexcept;

struct CpuThreadBudgetSnapshot
{
    uint32_t compute_threads = 0;
    uint32_t max_compute_threads = 0;
    uint32_t active_compute_threads = 0;
    uint32_t available_compute_threads = 0;
    uint32_t active_io_threads = 0;
    uint32_t active_service_threads = 0;
    uint32_t borrowed_compute_threads = 0;
    uint32_t idle_io_threads = 0;
    uint32_t idle_service_threads = 0;
    uint64_t compute_acquisitions = 0;
    uint64_t compute_returns = 0;
};

// A small in-process coordinator for scheduler work. It does not create
// threads; it only leases the already configured CPU budget. Compute leases
// may borrow idle I/O/service reservations and return them on destruction.
class CpuThreadBudgetController
{
public:
    enum class Pool : uint8_t
    {
        Compute,
        Io,
        Service
    };

    class Lease
    {
    public:
        Lease() = default;
        ~Lease();

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;

        [[nodiscard]] uint32_t size() const noexcept { return count_; }
        [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    private:
        friend class CpuThreadBudgetController;
        Lease(
            CpuThreadBudgetController* owner,
            Pool pool,
            uint32_t count,
            uint32_t compute_pool_threads,
            uint32_t borrowed_io_threads,
            uint32_t borrowed_service_threads) noexcept;

        void release() noexcept;

        CpuThreadBudgetController* owner_ = nullptr;
        Pool pool_ = Pool::Compute;
        uint32_t count_ = 0;
        uint32_t compute_pool_threads_ = 0;
        uint32_t borrowed_io_threads_ = 0;
        uint32_t borrowed_service_threads_ = 0;
    };

    explicit CpuThreadBudgetController(CpuThreadBudget budget) noexcept;

    [[nodiscard]] const CpuThreadBudget& budget() const noexcept { return budget_; }

    // The blocking form waits only for the minimum requested progress. It may
    // return fewer than requested when another pool is temporarily active.
    [[nodiscard]] Lease acquire_compute(
        uint32_t requested,
        bool allow_idle_reservations = true,
        uint32_t minimum = 1);

    // The non-blocking form is used by a running team to reclaim currently
    // idle capacity without making scheduler workers wait on one another.
    [[nodiscard]] Lease try_acquire_compute(
        uint32_t requested,
        bool allow_idle_reservations = true) noexcept;

    [[nodiscard]] Lease acquire_io(uint32_t requested = 1);
    [[nodiscard]] Lease acquire_service(uint32_t requested = 1);

    [[nodiscard]] CpuThreadBudgetSnapshot snapshot() const noexcept;

private:
    [[nodiscard]] uint32_t total_active_threads_locked() const noexcept;
    [[nodiscard]] uint32_t available_compute_threads_locked(bool allow_idle_reservations) const noexcept;
    [[nodiscard]] uint32_t available_pool_threads_locked(Pool pool) const noexcept;
    [[nodiscard]] Lease acquire_compute_locked(uint32_t requested, bool allow_idle_reservations) noexcept;
    [[nodiscard]] Lease acquire_pool_locked(Pool pool, uint32_t requested) noexcept;
    void release(const Lease& lease) noexcept;

    CpuThreadBudget budget_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    uint32_t active_compute_pool_threads_ = 0;
    uint32_t active_borrowed_io_threads_ = 0;
    uint32_t active_borrowed_service_threads_ = 0;
    uint32_t active_io_threads_ = 0;
    uint32_t active_service_threads_ = 0;
    uint64_t compute_acquisitions_ = 0;
    uint64_t compute_returns_ = 0;
};

// Select a useful team from a normalized work shape. independent_work_items
// describes parallel request/row groups; work_units lets short shapes use a
// smaller team while larger shapes can reach threads_per_work_item.
[[nodiscard]] uint32_t choose_cpu_team_size(
    uint64_t work_units,
    uint32_t independent_work_items,
    uint32_t threads_per_work_item,
    uint32_t available_threads,
    uint32_t configured_max_threads = 0) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_THREAD_BUDGET_H
