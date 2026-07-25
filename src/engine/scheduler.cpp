#include "ncnn/moe/scheduler.h"

#include "engine/cpu_topology.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace ncnn {
namespace moe {

class BatchScheduler::Implementation
{
public:
    explicit Implementation(const SchedulerOptions& options)
        : pin_workers_(has_flag(options.flags, SchedulerOptionPinWorkers)
                       || !options.worker_cpu_sets.empty()),
          worker_cpu_sets_(options.worker_cpu_sets)
    {
        const uint32_t hardware_threads = std::max(1u, std::thread::hardware_concurrency());
#if defined(__linux__)
        CpuTopology automatic_topology;
        if (pin_workers_ && worker_cpu_sets_.empty()) {
            automatic_topology = discover_cpu_topology();
            numa_nodes_detected_
                = static_cast<uint32_t>(automatic_topology.numa_nodes.size());
        }
#endif
        uint32_t default_worker_count = std::min(4u, hardware_threads);
#if defined(__linux__)
        if (!automatic_topology.allowed_cpus.empty()) {
            const uint32_t allowed_cpu_count
                = static_cast<uint32_t>(automatic_topology.allowed_cpus.size());
            default_worker_count = std::min(4u, allowed_cpu_count);
            if (!automatic_topology.numa_nodes.empty()) {
                default_worker_count = std::max(
                    default_worker_count,
                    std::min(
                        allowed_cpu_count,
                        static_cast<uint32_t>(automatic_topology.numa_nodes.size())));
            }
        }
#endif
        worker_count_ = !worker_cpu_sets_.empty()
                            ? static_cast<uint32_t>(worker_cpu_sets_.size())
                        : options.worker_count == 0
                            ? default_worker_count
                            : options.worker_count;
        worker_count_ = std::max(1u, worker_count_);
#if defined(__linux__)
        if (pin_workers_ && worker_cpu_sets_.empty()) {
            worker_cpu_sets_
                = partition_cpu_topology(automatic_topology, worker_count_);
            automatic_topology_affinity_ = !worker_cpu_sets_.empty();
        }
#endif
        std::vector<uint32_t> affinity_cpus;
        for (const std::vector<uint32_t>& cpu_set : worker_cpu_sets_)
            affinity_cpus.insert(
                affinity_cpus.end(), cpu_set.begin(), cpu_set.end());
        std::sort(affinity_cpus.begin(), affinity_cpus.end());
        affinity_cpus.erase(
            std::unique(affinity_cpus.begin(), affinity_cpus.end()),
            affinity_cpus.end());
        affinity_cpu_count_ = static_cast<uint32_t>(affinity_cpus.size());
#if defined(_OPENMP)
        const uint32_t affinity_threads
            = affinity_cpu_count_ == 0 ? hardware_threads : affinity_cpu_count_;
        expert_threads_per_worker_
            = options.expert_threads_per_worker == 0
                  ? std::max(1u, affinity_threads / worker_count_)
                  : options.expert_threads_per_worker;
#else
        (void)options.expert_threads_per_worker;
        expert_threads_per_worker_ = 1;
#endif
        workers_.reserve(worker_count_);
        for (uint32_t worker_index = 0; worker_index < worker_count_; ++worker_index) {
            workers_.emplace_back([this, worker_index] {
                configure_worker(worker_index);
                worker_loop();
            });
        }
    }

    ~Implementation()
    {
        {
            const std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        queue_ready_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable())
                worker.join();
        }
    }

    std::future<std::vector<Result<DecodeResult> > > submit_decode(
        std::vector<DecodeBatchRequest> requests)
    {
        struct BatchState
        {
            explicit BatchState(size_t size)
                : results(size),
                  remaining(size)
            {
            }

            std::vector<std::optional<Result<DecodeResult> > > results;
            std::atomic<size_t> remaining;
            std::promise<std::vector<Result<DecodeResult> > > promise;
        };

        auto state = std::make_shared<BatchState>(requests.size());
        std::future<std::vector<Result<DecodeResult> > > future = state->promise.get_future();
        submitted_batches_.fetch_add(1, std::memory_order_relaxed);
        submitted_requests_.fetch_add(requests.size(), std::memory_order_relaxed);
        update_max(max_batch_size_, requests.size());

        if (requests.empty()) {
            state->promise.set_value({});
            return future;
        }

        std::unordered_map<Session*, size_t> session_counts;
        for (const DecodeBatchRequest& request : requests) {
            if (request.session)
                ++session_counts[request.session.get()];
        }

        auto complete = [this, state](size_t index, Result<DecodeResult> result) {
            state->results[index].emplace(std::move(result));
            completed_requests_.fetch_add(1, std::memory_order_relaxed);
            const uint64_t in_flight = in_flight_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            (void)in_flight;
            if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) != 1)
                return;
            std::vector<Result<DecodeResult> > ordered;
            ordered.reserve(state->results.size());
            for (std::optional<Result<DecodeResult> >& item : state->results)
                ordered.push_back(std::move(*item));
            state->promise.set_value(std::move(ordered));
        };

        const uint64_t current_in_flight
            = in_flight_.fetch_add(requests.size(), std::memory_order_acq_rel) + requests.size();
        update_max(max_in_flight_, current_in_flight);
        for (size_t index = 0; index < requests.size(); ++index) {
            DecodeBatchRequest request = std::move(requests[index]);
            if (!request.session) {
                rejected_requests_.fetch_add(1, std::memory_order_relaxed);
                complete(
                    index,
                    Error{ErrorCode::InvalidArgument, "batch decode session cannot be null"});
                continue;
            }
            if (session_counts[request.session.get()] != 1) {
                rejected_requests_.fetch_add(1, std::memory_order_relaxed);
                complete(
                    index,
                    Error{
                        ErrorCode::InvalidArgument,
                        "a decode batch may contain each session at most once"});
                continue;
            }
            Session* session_key = request.session.get();
            enqueue_session(
                session_key,
                [request = std::move(request), index, complete]() mutable {
                    try {
                        complete(index, request.session->decode(request.input_id));
                    }
                    catch (const std::exception& error) {
                        complete(
                            index,
                            Error{
                                ErrorCode::InternalError,
                                std::string("decode worker failed: ") + error.what()});
                    }
                    catch (...) {
                        complete(
                            index,
                            Error{ErrorCode::InternalError, "decode worker failed"});
                    }
                });
        }
        return future;
    }

    SchedulerStatistics statistics() const noexcept
    {
        SchedulerStatistics result;
        result.submitted_batches = submitted_batches_.load(std::memory_order_relaxed);
        result.submitted_requests = submitted_requests_.load(std::memory_order_relaxed);
        result.completed_requests = completed_requests_.load(std::memory_order_relaxed);
        result.rejected_requests = rejected_requests_.load(std::memory_order_relaxed);
        result.max_batch_size = max_batch_size_.load(std::memory_order_relaxed);
        result.max_in_flight = max_in_flight_.load(std::memory_order_relaxed);
        result.serialized_session_requests
            = serialized_session_requests_.load(std::memory_order_relaxed);
        result.affinity_workers_configured
            = affinity_workers_configured_.load(std::memory_order_relaxed);
        result.affinity_failures
            = affinity_failures_.load(std::memory_order_relaxed);
        result.affinity_cpu_count = affinity_cpu_count_;
        result.numa_nodes_detected = numa_nodes_detected_;
        result.automatic_topology_affinity = automatic_topology_affinity_;
        result.worker_count = worker_count_;
        result.expert_threads_per_worker = expert_threads_per_worker_;
        return result;
    }

private:
    static void update_max(std::atomic<uint64_t>& destination, uint64_t value)
    {
        uint64_t previous = destination.load(std::memory_order_relaxed);
        while (previous < value
               && !destination.compare_exchange_weak(
                   previous,
                   value,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
    }

    struct SessionQueue
    {
        std::deque<std::function<void()> > work;
        bool active = false;
    };

    void enqueue_session(Session* session, std::function<void()> work)
    {
        {
            const std::lock_guard<std::mutex> lock(queue_mutex_);
            SessionQueue& pending = session_queues_[session];
            if (pending.active || !pending.work.empty())
                serialized_session_requests_.fetch_add(1, std::memory_order_relaxed);
            pending.work.push_back(std::move(work));
            if (!pending.active) {
                pending.active = true;
                queue_.push_back([this, session] {
                    execute_session(session);
                });
            }
        }
        queue_ready_.notify_one();
    }

    void execute_session(Session* session)
    {
        std::function<void()> work;
        {
            const std::lock_guard<std::mutex> lock(queue_mutex_);
            auto pending = session_queues_.find(session);
            if (pending == session_queues_.end() || pending->second.work.empty())
                return;
            work = std::move(pending->second.work.front());
            pending->second.work.pop_front();
        }
        work();
        {
            const std::lock_guard<std::mutex> lock(queue_mutex_);
            auto pending = session_queues_.find(session);
            if (pending == session_queues_.end())
                return;
            if (pending->second.work.empty()) {
                session_queues_.erase(pending);
            }
            else {
                queue_.push_back([this, session] {
                    execute_session(session);
                });
                queue_ready_.notify_one();
            }
        }
    }

    void worker_loop()
    {
        for (;;) {
            std::function<void()> work;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_ready_.wait(lock, [this] {
                    return stopping_ || !queue_.empty();
                });
                if (stopping_ && queue_.empty())
                    return;
                work = std::move(queue_.front());
                queue_.pop_front();
            }
            work();
        }
    }

    void configure_worker(uint32_t worker_index)
    {
#if defined(_OPENMP)
        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(expert_threads_per_worker_));
#endif
#if defined(__linux__)
        if (!pin_workers_)
            return;
        const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (online_cpus <= 0)
            return;
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        if (!worker_cpu_sets_.empty()) {
            for (uint32_t cpu : worker_cpu_sets_[worker_index]) {
                if (cpu >= CPU_SETSIZE) {
                    affinity_failures_.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                CPU_SET(cpu, &cpu_set);
            }
        }
        else {
            const uint32_t online = static_cast<uint32_t>(online_cpus);
            const uint32_t partition_size
                = worker_count_ <= online
                      ? online / worker_count_
                            + (worker_index < online % worker_count_ ? 1u : 0u)
                      : 1u;
            const uint32_t first_cpu
                = worker_count_ <= online
                      ? worker_index * (online / worker_count_)
                            + std::min(worker_index, online % worker_count_)
                      : worker_index % online;
            for (uint32_t offset = 0; offset < partition_size; ++offset)
                CPU_SET((first_cpu + offset) % online, &cpu_set);
        }
        const int affinity_result = pthread_setaffinity_np(
            pthread_self(), sizeof(cpu_set), &cpu_set);
        if (affinity_result == 0)
            affinity_workers_configured_.fetch_add(1, std::memory_order_relaxed);
        else
            affinity_failures_.fetch_add(1, std::memory_order_relaxed);
#else
        (void)pin_workers_;
        (void)worker_index;
#endif
    }

    bool pin_workers_ = false;
    bool stopping_ = false;
    uint32_t worker_count_ = 0;
    uint32_t expert_threads_per_worker_ = 1;
    uint32_t affinity_cpu_count_ = 0;
    uint32_t numa_nodes_detected_ = 0;
    bool automatic_topology_affinity_ = false;
    std::vector<std::vector<uint32_t> > worker_cpu_sets_;
    std::vector<std::thread> workers_;
    std::deque<std::function<void()> > queue_;
    std::unordered_map<Session*, SessionQueue> session_queues_;
    std::mutex queue_mutex_;
    std::condition_variable queue_ready_;
    std::atomic<uint64_t> submitted_batches_{0};
    std::atomic<uint64_t> submitted_requests_{0};
    std::atomic<uint64_t> completed_requests_{0};
    std::atomic<uint64_t> rejected_requests_{0};
    std::atomic<uint64_t> max_batch_size_{0};
    std::atomic<uint64_t> in_flight_{0};
    std::atomic<uint64_t> max_in_flight_{0};
    std::atomic<uint64_t> serialized_session_requests_{0};
    std::atomic<uint64_t> affinity_workers_configured_{0};
    std::atomic<uint64_t> affinity_failures_{0};
};

BatchScheduler::BatchScheduler(const SchedulerOptions& options)
    : implementation_(new Implementation(options))
{
}

BatchScheduler::~BatchScheduler() = default;

std::future<std::vector<Result<DecodeResult> > > BatchScheduler::submit_decode(
    std::vector<DecodeBatchRequest> requests)
{
    return implementation_->submit_decode(std::move(requests));
}

SchedulerStatistics BatchScheduler::statistics() const noexcept
{
    return implementation_->statistics();
}

} // namespace moe
} // namespace ncnn
