#include "cpu_thread_budget.h"

#include "cpu_topology.h"

#include <algorithm>
#include <limits>
#include <thread>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

#if defined(_OPENMP)
static thread_local uint32_t cpu_openmp_thread_limit_override = 0;
#endif

CpuThreadBudget resolve_cpu_thread_budget(
    uint32_t requested_io_threads,
    uint32_t requested_service_threads,
    uint32_t requested_compute_threads) noexcept
{
    const uint32_t cpu_count = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t physical_cpu_count = std::max(1u, discover_cpu_topology().physical_cpu_count);
    // Keep I/O parallelism bounded so it does not consume the compute pool.
    const uint32_t automatic_io = std::min(
        physical_cpu_count,
        std::min(6u, std::max(2u, physical_cpu_count / 2u)));
    const uint32_t reserved_io = requested_io_threads == 0
                                     ? automatic_io
                                     : std::min(requested_io_threads, physical_cpu_count);
    const uint32_t reserved_service = std::min(std::max(1u, requested_service_threads), physical_cpu_count);
    const uint64_t unavailable = std::min<uint64_t>(
        physical_cpu_count,
        static_cast<uint64_t>(reserved_io) + reserved_service);
    const uint32_t available_compute = std::max(
        1u,
        physical_cpu_count - static_cast<uint32_t>(unavailable));
    const uint32_t compute = requested_compute_threads == 0
                                 ? available_compute
                                 : std::max(1u, std::min(requested_compute_threads, available_compute));
    const uint32_t max_compute = requested_compute_threads == 0
                                     ? physical_cpu_count
                                     : std::max(1u, std::min(requested_compute_threads, physical_cpu_count));
    return {cpu_count, physical_cpu_count, reserved_io, reserved_service, compute, max_compute};
}

CpuOpenMpThreadLimitScope::CpuOpenMpThreadLimitScope() noexcept
#if defined(_OPENMP)
    : previous_(cpu_openmp_thread_limit_override)
#endif
{
}

CpuOpenMpThreadLimitScope::CpuOpenMpThreadLimitScope(uint32_t limit) noexcept
#if defined(_OPENMP)
    : previous_(cpu_openmp_thread_limit_override)
#endif
{
    set(limit);
}

CpuOpenMpThreadLimitScope::~CpuOpenMpThreadLimitScope() noexcept
{
#if defined(_OPENMP)
    cpu_openmp_thread_limit_override = previous_;
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
    CpuThreadBudgetController* owner,
    Pool pool,
    uint32_t count,
    uint32_t compute_pool_threads,
    uint32_t borrowed_io_threads,
    uint32_t borrowed_service_threads) noexcept
    : owner_(owner),
      pool_(pool),
      count_(count),
      compute_pool_threads_(compute_pool_threads),
      borrowed_io_threads_(borrowed_io_threads),
      borrowed_service_threads_(borrowed_service_threads)
{
}

CpuThreadBudgetController::Lease::~Lease()
{
    release();
}

CpuThreadBudgetController::Lease::Lease(Lease&& other) noexcept
    : owner_(other.owner_),
      pool_(other.pool_),
      count_(other.count_),
      compute_pool_threads_(other.compute_pool_threads_),
      borrowed_io_threads_(other.borrowed_io_threads_),
      borrowed_service_threads_(other.borrowed_service_threads_)
{
    other.owner_ = nullptr;
    other.count_ = 0;
    other.compute_pool_threads_ = 0;
    other.borrowed_io_threads_ = 0;
    other.borrowed_service_threads_ = 0;
}

CpuThreadBudgetController::Lease& CpuThreadBudgetController::Lease::operator=(Lease&& other) noexcept
{
    if (this == &other)
        return *this;
    release();
    owner_ = other.owner_;
    pool_ = other.pool_;
    count_ = other.count_;
    compute_pool_threads_ = other.compute_pool_threads_;
    borrowed_io_threads_ = other.borrowed_io_threads_;
    borrowed_service_threads_ = other.borrowed_service_threads_;
    other.owner_ = nullptr;
    other.count_ = 0;
    other.compute_pool_threads_ = 0;
    other.borrowed_io_threads_ = 0;
    other.borrowed_service_threads_ = 0;
    return *this;
}

void CpuThreadBudgetController::Lease::release() noexcept
{
    if (owner_ == nullptr || count_ == 0)
        return;
    owner_->release(*this);
    owner_ = nullptr;
    count_ = 0;
    compute_pool_threads_ = 0;
    borrowed_io_threads_ = 0;
    borrowed_service_threads_ = 0;
}

CpuThreadBudgetController::CpuThreadBudgetController(CpuThreadBudget budget) noexcept
    : budget_(budget)
{
    budget_.physical_cores = std::max(1u, budget_.physical_cores);
    budget_.compute_threads = std::max(1u, std::min(budget_.compute_threads, budget_.physical_cores));
    budget_.max_compute_threads = std::max(
        budget_.compute_threads,
        std::min(budget_.max_compute_threads, budget_.physical_cores));
}

uint32_t CpuThreadBudgetController::total_active_threads_locked() const noexcept
{
    return active_compute_pool_threads_
           + active_borrowed_io_threads_
           + active_borrowed_service_threads_
           + active_io_threads_
           + active_service_threads_;
}

uint32_t CpuThreadBudgetController::available_compute_threads_locked(bool allow_idle_reservations) const noexcept
{
    const uint32_t active = total_active_threads_locked();
    const uint32_t total_free = active < budget_.physical_cores
                                    ? budget_.physical_cores - active
                                    : 0;
    const uint32_t active_compute = active_compute_pool_threads_
                                    + active_borrowed_io_threads_
                                    + active_borrowed_service_threads_;
    const uint32_t compute_cap_free = active_compute < budget_.max_compute_threads
                                          ? budget_.max_compute_threads - active_compute
                                          : 0;
    const uint32_t compute_pool_free = active_compute_pool_threads_ < budget_.compute_threads
                                           ? budget_.compute_threads - active_compute_pool_threads_
                                           : 0;
    if (!allow_idle_reservations)
        return std::min(total_free, std::min(compute_cap_free, compute_pool_free));

    const uint32_t io_used = active_io_threads_ + active_borrowed_io_threads_;
    const uint32_t service_used = active_service_threads_ + active_borrowed_service_threads_;
    const uint32_t idle_io = io_used < budget_.reserved_io_threads
                                 ? budget_.reserved_io_threads - io_used
                                 : 0;
    const uint32_t idle_service = service_used < budget_.reserved_service_threads
                                      ? budget_.reserved_service_threads - service_used
                                      : 0;
    return std::min(
        total_free,
        std::min(compute_cap_free, compute_pool_free + idle_io + idle_service));
}

uint32_t CpuThreadBudgetController::available_pool_threads_locked(Pool pool) const noexcept
{
    const uint32_t active = total_active_threads_locked();
    const uint32_t total_free = active < budget_.physical_cores
                                    ? budget_.physical_cores - active
                                    : 0;
    if (pool == Pool::Io)
    {
        const uint32_t used = active_io_threads_ + active_borrowed_io_threads_;
        return used < budget_.reserved_io_threads
                   ? std::min(total_free, budget_.reserved_io_threads - used)
                   : 0;
    }
    if (pool == Pool::Service)
    {
        const uint32_t used = active_service_threads_ + active_borrowed_service_threads_;
        return used < budget_.reserved_service_threads
                   ? std::min(total_free, budget_.reserved_service_threads - used)
                   : 0;
    }
    return available_compute_threads_locked(false);
}

CpuThreadBudgetController::Lease CpuThreadBudgetController::acquire_compute_locked(
    uint32_t requested,
    bool allow_idle_reservations) noexcept
{
    if (requested == 0)
        return {};

    const uint32_t available = available_compute_threads_locked(allow_idle_reservations);
    const uint32_t count = std::min(requested, available);
    if (count == 0)
        return {};

    const uint32_t compute_pool_threads = std::min(
        count,
        active_compute_pool_threads_ < budget_.compute_threads
            ? budget_.compute_threads - active_compute_pool_threads_
            : 0);
    uint32_t remaining = count - compute_pool_threads;
    const uint32_t io_used = active_io_threads_ + active_borrowed_io_threads_;
    const uint32_t idle_io = io_used < budget_.reserved_io_threads
                                 ? budget_.reserved_io_threads - io_used
                                 : 0;
    const uint32_t borrowed_io_threads = std::min(remaining, idle_io);
    remaining -= borrowed_io_threads;
    const uint32_t service_used = active_service_threads_ + active_borrowed_service_threads_;
    const uint32_t idle_service = service_used < budget_.reserved_service_threads
                                      ? budget_.reserved_service_threads - service_used
                                      : 0;
    const uint32_t borrowed_service_threads = std::min(remaining, idle_service);

    active_compute_pool_threads_ += compute_pool_threads;
    active_borrowed_io_threads_ += borrowed_io_threads;
    active_borrowed_service_threads_ += borrowed_service_threads;
    ++compute_acquisitions_;
    return Lease(
        this,
        Pool::Compute,
        count,
        compute_pool_threads,
        borrowed_io_threads,
        borrowed_service_threads);
}

CpuThreadBudgetController::Lease CpuThreadBudgetController::acquire_pool_locked(
    Pool pool,
    uint32_t requested) noexcept
{
    if (requested == 0 || (pool != Pool::Io && pool != Pool::Service))
        return {};
    const uint32_t count = std::min(requested, available_pool_threads_locked(pool));
    if (count == 0)
        return {};
    if (pool == Pool::Io)
        active_io_threads_ += count;
    else
        active_service_threads_ += count;
    return Lease(this, pool, count, 0, 0, 0);
}

CpuThreadBudgetController::Lease CpuThreadBudgetController::acquire_compute(
    uint32_t requested,
    bool allow_idle_reservations,
    uint32_t minimum)
{
    requested = std::max(requested, minimum);
    if (requested == 0)
        return {};
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this, allow_idle_reservations, minimum] {
        return available_compute_threads_locked(allow_idle_reservations) >= minimum;
    });
    return acquire_compute_locked(requested, allow_idle_reservations);
}

CpuThreadBudgetController::Lease CpuThreadBudgetController::try_acquire_compute(
    uint32_t requested,
    bool allow_idle_reservations) noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return acquire_compute_locked(requested, allow_idle_reservations);
}

CpuThreadBudgetController::Lease CpuThreadBudgetController::acquire_io(uint32_t requested)
{
    requested = std::max(1u, requested);
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return available_pool_threads_locked(Pool::Io) != 0; });
    return acquire_pool_locked(Pool::Io, requested);
}

CpuThreadBudgetController::Lease CpuThreadBudgetController::acquire_service(uint32_t requested)
{
    requested = std::max(1u, requested);
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return available_pool_threads_locked(Pool::Service) != 0; });
    return acquire_pool_locked(Pool::Service, requested);
}

CpuThreadBudgetSnapshot CpuThreadBudgetController::snapshot() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    CpuThreadBudgetSnapshot result;
    result.compute_threads = budget_.compute_threads;
    result.max_compute_threads = budget_.max_compute_threads;
    result.active_compute_threads = active_compute_pool_threads_
                                    + active_borrowed_io_threads_
                                    + active_borrowed_service_threads_;
    result.available_compute_threads = available_compute_threads_locked(true);
    result.active_io_threads = active_io_threads_;
    result.active_service_threads = active_service_threads_;
    result.borrowed_compute_threads = active_borrowed_io_threads_ + active_borrowed_service_threads_;
    const uint32_t io_used = active_io_threads_ + active_borrowed_io_threads_;
    const uint32_t service_used = active_service_threads_ + active_borrowed_service_threads_;
    result.idle_io_threads = io_used < budget_.reserved_io_threads
                                 ? budget_.reserved_io_threads - io_used
                                 : 0;
    result.idle_service_threads = service_used < budget_.reserved_service_threads
                                      ? budget_.reserved_service_threads - service_used
                                      : 0;
    result.compute_acquisitions = compute_acquisitions_;
    result.compute_returns = compute_returns_;
    return result;
}

void CpuThreadBudgetController::release(const Lease& lease) noexcept
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lease.pool_ == Pool::Compute)
        {
            active_compute_pool_threads_ = active_compute_pool_threads_ >= lease.compute_pool_threads_
                                               ? active_compute_pool_threads_ - lease.compute_pool_threads_
                                               : 0;
            active_borrowed_io_threads_ = active_borrowed_io_threads_ >= lease.borrowed_io_threads_
                                              ? active_borrowed_io_threads_ - lease.borrowed_io_threads_
                                              : 0;
            active_borrowed_service_threads_ = active_borrowed_service_threads_ >= lease.borrowed_service_threads_
                                                   ? active_borrowed_service_threads_ - lease.borrowed_service_threads_
                                                   : 0;
            ++compute_returns_;
        }
        else if (lease.pool_ == Pool::Io)
        {
            active_io_threads_ = active_io_threads_ >= lease.count_ ? active_io_threads_ - lease.count_ : 0;
        }
        else
        {
            active_service_threads_ = active_service_threads_ >= lease.count_ ? active_service_threads_ - lease.count_ : 0;
        }
    }
    changed_.notify_all();
}

uint32_t choose_cpu_team_size(
    uint64_t work_units,
    uint32_t independent_work_items,
    uint32_t threads_per_work_item,
    uint32_t available_threads,
    uint32_t configured_max_threads) noexcept
{
    available_threads = std::max(1u, available_threads);
    independent_work_items = std::max(1u, independent_work_items);
    threads_per_work_item = std::max(1u, threads_per_work_item);
    const uint32_t configured_limit = configured_max_threads == 0
                                          ? available_threads
                                          : std::max(1u, std::min(configured_max_threads, available_threads));
    if (work_units == 0)
        return 1;

    const uint64_t average_work = (work_units + independent_work_items - 1) / independent_work_items;
    const uint32_t useful_threads_per_item = static_cast<uint32_t>(std::min<uint64_t>(
        threads_per_work_item,
        std::max<uint64_t>(1, average_work)));
    const uint64_t requested = static_cast<uint64_t>(independent_work_items) * useful_threads_per_item;
    const uint32_t shape_limit = static_cast<uint32_t>(std::min<uint64_t>(
        configured_limit,
        std::max<uint64_t>(1, requested)));
    return std::max(1u, shape_limit);
}

} // namespace moe
} // namespace ncnn
