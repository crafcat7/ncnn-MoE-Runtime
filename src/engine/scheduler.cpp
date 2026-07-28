#include "ncnn/moe/scheduler.h"

#include "engine/session_batch.h"
#include "engine/cpu_topology.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <iterator>
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

#define NCNN_MOE_SCHEDULER_AUTO_AFFINITY_BIT 31
#define NCNN_MOE_ADAPTIVE_PREFER_STAGED_BIT  0
#define NCNN_MOE_ADAPTIVE_CONFIRM_BIT        1
#define NCNN_MOE_ADAPTIVE_INITIALIZED_BIT    2

static double adaptive_switch_margin(uint64_t samples)
{
    if (samples < 4)
        return 1.10;
    if (samples < 8)
        return 1.075;
    return 1.05;
}

class BatchScheduler::Implementation
{
    enum class AdaptivePhase : uint8_t
    {
        Resident = 0,
        Mixed = 1,
        Storage = 2
    };

    static constexpr size_t adaptive_base_bucket_count = 16;
    static constexpr size_t adaptive_phase_count = 3;
    static constexpr uint32_t scheduler_internal_automatic_topology_affinity = UINT32_C(1) << NCNN_MOE_SCHEDULER_AUTO_AFFINITY_BIT;

    struct PendingCrossCall
    {
        DecodeBatchRequest request;
        std::promise<std::vector<Result<DecodeResult>>> promise;
        std::chrono::steady_clock::time_point submitted_at;
    };

public:
    explicit Implementation(const SchedulerOptions& options)
        : option_flags_(options.flags | (!options.worker_cpu_sets.empty() ? static_cast<uint32_t>(SchedulerOptionPinWorkers) : 0u)),
          adaptive_probe_interval_(options.adaptive_probe_interval == 0 ? 32 : options.adaptive_probe_interval),
          cross_call_window_microseconds_(options.cross_call_window_microseconds),
          cross_call_max_batch_size_(options.cross_call_max_batch_size),
          worker_cpu_sets_(options.worker_cpu_sets)
    {
        const uint32_t hardware_threads = std::max(1u, std::thread::hardware_concurrency());
#if defined(__linux__)
        CpuTopology automatic_topology;
        if (has_flag(option_flags_, SchedulerOptionPinWorkers) && worker_cpu_sets_.empty())
        {
            automatic_topology = discover_cpu_topology();
            numa_nodes_detected_ = static_cast<uint32_t>(automatic_topology.numa_nodes.size());
        }
#endif
        uint32_t default_worker_count = std::min(4u, hardware_threads);
#if defined(__linux__)
        if (!automatic_topology.allowed_cpus.empty())
        {
            const uint32_t allowed_cpu_count = static_cast<uint32_t>(automatic_topology.allowed_cpus.size());
            default_worker_count = std::min(4u, allowed_cpu_count);
            if (!automatic_topology.numa_nodes.empty())
            {
                default_worker_count = std::max(default_worker_count, std::min(allowed_cpu_count, static_cast<uint32_t>(automatic_topology.numa_nodes.size())));
            }
        }
#endif
        worker_count_ = !worker_cpu_sets_.empty()
                            ? static_cast<uint32_t>(worker_cpu_sets_.size())
                        : options.worker_count == 0
                            ? default_worker_count
                            : options.worker_count;
        worker_count_ = std::max(1u, worker_count_);
        if (cross_call_max_batch_size_ == 0)
            cross_call_max_batch_size_ = worker_count_;
#if defined(__linux__)
        if (has_flag(option_flags_, SchedulerOptionPinWorkers) && worker_cpu_sets_.empty())
        {
            worker_cpu_sets_ = partition_cpu_topology(automatic_topology, worker_count_);
            if (!worker_cpu_sets_.empty())
            {
                option_flags_ |= scheduler_internal_automatic_topology_affinity;
            }
        }
#endif
        std::vector<uint32_t> affinity_cpus;
        for (const std::vector<uint32_t>& cpu_set : worker_cpu_sets_)
            affinity_cpus.insert(affinity_cpus.end(), cpu_set.begin(), cpu_set.end());
        std::sort(affinity_cpus.begin(), affinity_cpus.end());
        affinity_cpus.erase(std::unique(affinity_cpus.begin(), affinity_cpus.end()), affinity_cpus.end());
        affinity_cpu_count_ = static_cast<uint32_t>(affinity_cpus.size());
#if defined(_OPENMP)
        const uint32_t affinity_threads = affinity_cpu_count_ == 0 ? hardware_threads : affinity_cpu_count_;
        expert_threads_per_worker_ = options.expert_threads_per_worker == 0
                                         ? std::max(1u, affinity_threads / worker_count_)
                                         : options.expert_threads_per_worker;
#else
        (void)options.expert_threads_per_worker;
        expert_threads_per_worker_ = 1;
#endif
        workers_.reserve(worker_count_);
        for (uint32_t worker_index = 0; worker_index < worker_count_; ++worker_index)
        {
            workers_.emplace_back([this, worker_index] {
                configure_worker(worker_index);
                worker_loop();
            });
        }
        if (!has_flag(option_flags_, SchedulerOptionDisableCrossCallBatching) && cross_call_window_microseconds_ != 0 && cross_call_max_batch_size_ > 1)
        {
            cross_call_collector_ = std::thread([this] { cross_call_loop(); });
        }
    }

    ~Implementation()
    {
        {
            const std::lock_guard<std::mutex> lock(cross_call_mutex_);
            cross_call_stopping_ = true;
        }
        cross_call_ready_.notify_all();
        if (cross_call_collector_.joinable())
            cross_call_collector_.join();
        {
            const std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        queue_ready_.notify_all();
        for (std::thread& worker : workers_)
        {
            if (worker.joinable())
                worker.join();
        }
    }

    std::future<std::vector<Result<PrefillResult>>> submit_prefill(
        std::vector<PrefillBatchRequest> requests)
    {
        struct PrefillState
        {
            std::promise<std::vector<Result<PrefillResult>>> promise;
        };

        submitted_prefill_batches_.fetch_add(
            1,
            std::memory_order_relaxed);
        submitted_prefill_requests_.fetch_add(
            requests.size(),
            std::memory_order_relaxed);
        auto state = std::make_shared<PrefillState>();
        std::future<std::vector<Result<PrefillResult>>> future = state->promise.get_future();
        if (requests.empty())
        {
            state->promise.set_value({});
            return future;
        }

        std::unordered_map<Session*, size_t> session_counts;
        std::vector<Session*> sessions;
        sessions.reserve(requests.size());
        bool valid = true;
        for (const PrefillBatchRequest& request : requests)
        {
            if (!request.session || request.input_ids.empty())
            {
                valid = false;
                continue;
            }
            sessions.push_back(request.session.get());
            ++session_counts[request.session.get()];
        }
        if (sessions.size() != requests.size()
            || std::any_of(
                session_counts.begin(),
                session_counts.end(),
                [](const auto& item) {
                    return item.second != 1;
                }))
        {
            valid = false;
        }

        auto reject = [this,
                       state,
                       request_count = requests.size()](
                          const std::string& message) {
            std::vector<Result<PrefillResult>> results;
            results.reserve(request_count);
            for (size_t index = 0;
                 index < request_count;
                 ++index)
            {
                results.emplace_back(Error{
                    ErrorCode::InvalidArgument,
                    message});
            }
            completed_prefill_requests_.fetch_add(
                request_count,
                std::memory_order_relaxed);
            state->promise.set_value(std::move(results));
        };
        if (!valid)
        {
            reject(
                "a prefill batch requires unique sessions and "
                "non-empty inputs");
            return future;
        }

        if (requests.size() == 1)
        {
            PrefillBatchRequest request = std::move(requests.front());
            Session* session = request.session.get();
            enqueue_session(
                session,
                [this,
                 state,
                 request = std::move(request)]() mutable {
                    std::vector<Result<PrefillResult>> results;
                    try
                    {
                        results.emplace_back(
                            request.session->prefill(
                                request.input_ids));
                    }
                    catch (const std::exception& error)
                    {
                        results.emplace_back(Error{
                            ErrorCode::InternalError,
                            std::string(
                                "prefill worker failed: ")
                                + error.what()});
                    }
                    catch (...)
                    {
                        results.emplace_back(Error{
                            ErrorCode::InternalError,
                            "prefill worker failed"});
                    }
                    completed_prefill_requests_.fetch_add(
                        1,
                        std::memory_order_relaxed);
                    state->promise.set_value(
                        std::move(results));
                });
            return future;
        }

        if (!SessionBatchAccess::compatible(sessions))
        {
            reject(
                "staged prefill sessions must share one loaded model");
            return future;
        }

        std::function<void()> staged_work = [this,
                                             state,
                                             sessions,
                                             requests = std::move(requests)]() mutable {
#if defined(_OPENMP)
            struct OpenMpTeamRestore
            {
                int team_size = 1;
                ~OpenMpTeamRestore()
                {
                    omp_set_num_threads(team_size);
                }
            };
            const OpenMpTeamRestore restore{
                omp_get_max_threads()};
            const uint32_t staged_team_size = std::max(
                1u,
                std::min(
                    static_cast<uint32_t>(
                        std::thread::hardware_concurrency()),
                    expert_threads_per_worker_
                        * static_cast<uint32_t>(
                            requests.size())));
            omp_set_num_threads(
                static_cast<int>(staged_team_size));
#endif
            std::vector<Result<PrefillResult>> results;
            try
            {
                std::vector<std::vector<int32_t>> input_ids;
                input_ids.reserve(requests.size());
                for (PrefillBatchRequest& request : requests)
                {
                    input_ids.push_back(
                        std::move(request.input_ids));
                }
                StagedDecodeBatchMetrics metrics;
                auto prefilled = SessionBatchAccess::prefill(
                    sessions,
                    input_ids,
                    metrics);
                staged_prefill_batches_.fetch_add(
                    1,
                    std::memory_order_relaxed);
                staged_prefill_requests_.fetch_add(
                    requests.size(),
                    std::memory_order_relaxed);
                logical_expert_batches_.fetch_add(
                    metrics.logical_expert_batches,
                    std::memory_order_relaxed);
                physical_expert_batches_.fetch_add(
                    metrics.physical_expert_batches,
                    std::memory_order_relaxed);
                if (metrics.logical_expert_batches
                    > metrics.physical_expert_batches)
                {
                    coalesced_expert_batches_.fetch_add(
                        metrics.logical_expert_batches
                            - metrics.physical_expert_batches,
                        std::memory_order_relaxed);
                }
                coalesced_expert_routes_.fetch_add(
                    metrics.coalesced_expert_routes,
                    std::memory_order_relaxed);
                update_max(
                    max_coalesced_expert_batch_size_,
                    metrics.max_expert_batch_size);
                if (prefilled)
                {
                    results.reserve(prefilled.value().size());
                    for (PrefillResult& result :
                         prefilled.value())
                    {
                        results.emplace_back(
                            std::move(result));
                    }
                }
                else
                {
                    results.reserve(requests.size());
                    for (size_t index = 0;
                         index < requests.size();
                         ++index)
                    {
                        results.emplace_back(
                            prefilled.error());
                    }
                }
            }
            catch (const std::exception& error)
            {
                results.reserve(requests.size());
                for (size_t index = 0;
                     index < requests.size();
                     ++index)
                {
                    results.emplace_back(Error{
                        ErrorCode::InternalError,
                        std::string(
                            "staged prefill worker failed: ")
                            + error.what()});
                }
            }
            catch (...)
            {
                results.reserve(requests.size());
                for (size_t index = 0;
                     index < requests.size();
                     ++index)
                {
                    results.emplace_back(Error{
                        ErrorCode::InternalError,
                        "staged prefill worker failed"});
                }
            }
            release_batch_sessions(sessions);
            completed_prefill_requests_.fetch_add(
                requests.size(),
                std::memory_order_relaxed);
            state->promise.set_value(std::move(results));
        };
        if (!try_enqueue_batch(sessions, std::move(staged_work)))
        {
            reject(
                "staged prefill sessions have pending scheduler work");
        }
        return future;
    }

    std::future<std::vector<Result<DecodeResult>>> submit_decode(std::vector<DecodeBatchRequest> requests)
    {
        submitted_batches_.fetch_add(1, std::memory_order_relaxed);
        submitted_requests_.fetch_add(requests.size(), std::memory_order_relaxed);
        update_max(max_batch_size_, requests.size());
        if (requests.size() != 1 || !requests.front().session || !cross_call_collector_.joinable())
        {
            return schedule_decode(std::move(requests));
        }

        PendingCrossCall pending;
        pending.request = std::move(requests.front());
        pending.submitted_at = std::chrono::steady_clock::now();
        std::future<std::vector<Result<DecodeResult>>> future = pending.promise.get_future();
        {
            const std::lock_guard<std::mutex> lock(cross_call_mutex_);
            cross_call_pending_.push_back(std::move(pending));
            update_max(max_cross_call_pending_, cross_call_pending_.size());
        }
        cross_call_ready_.notify_one();
        return future;
    }

    std::future<std::vector<Result<DecodeResult>>> schedule_decode(std::vector<DecodeBatchRequest> requests)
    {
        struct BatchState
        {
            explicit BatchState(size_t size)
                : results(size), remaining(size), submitted_at(std::chrono::steady_clock::now())
            {
            }

            std::vector<std::optional<Result<DecodeResult>>> results;
            std::atomic<size_t> remaining;
            std::promise<std::vector<Result<DecodeResult>>> promise;
            std::chrono::steady_clock::time_point submitted_at;
            std::atomic<uint64_t> expert_time_microseconds{0};
            std::atomic<uint64_t> expert_cache_wait_time_microseconds{0};
            size_t adaptive_base_bucket = 0;
            bool adaptive_eligible = false;
            bool adaptive_initial_preference = false;
            bool used_staged_batching = false;
        };

        auto state = std::make_shared<BatchState>(requests.size());
        std::future<std::vector<Result<DecodeResult>>> future = state->promise.get_future();

        if (requests.empty())
        {
            state->promise.set_value({});
            return future;
        }

        std::unordered_map<Session*, size_t> session_counts;
        for (const DecodeBatchRequest& request : requests)
        {
            if (request.session)
                ++session_counts[request.session.get()];
        }

        auto complete = [this, state](size_t index, Result<DecodeResult> result, SessionDecodePhaseSnapshot phase) {
            state->results[index].emplace(std::move(result));
            state->expert_time_microseconds.fetch_add(phase.expert_time_microseconds, std::memory_order_relaxed);
            state->expert_cache_wait_time_microseconds.fetch_add(phase.expert_cache_wait_time_microseconds, std::memory_order_relaxed);
            completed_requests_.fetch_add(1, std::memory_order_relaxed);
            const uint64_t in_flight = in_flight_.fetch_sub(1, std::memory_order_acq_rel) - 1;
            (void)in_flight;
            if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) != 1)
                return;
            if (state->adaptive_eligible)
            {
                observe_adaptive_staging(state->adaptive_base_bucket, state->used_staged_batching, state->adaptive_initial_preference,
                                         static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - state->submitted_at).count()), state->results.size(),
                                         state->expert_time_microseconds.load(std::memory_order_relaxed), state->expert_cache_wait_time_microseconds.load(std::memory_order_relaxed));
            }
            std::vector<Result<DecodeResult>> ordered;
            ordered.reserve(state->results.size());
            for (std::optional<Result<DecodeResult>>& item : state->results)
                ordered.push_back(std::move(*item));
            state->promise.set_value(std::move(ordered));
        };

        const uint64_t current_in_flight = in_flight_.fetch_add(requests.size(), std::memory_order_acq_rel) + requests.size();
        update_max(max_in_flight_, current_in_flight);
        bool staged_candidate = requests.size() > 1 && !has_flag(option_flags_, SchedulerOptionDisableStagedBatching);
        std::vector<Session*> staged_sessions;
        bool matching_inputs = false;
        if (staged_candidate)
        {
            staged_sessions.reserve(requests.size());
            for (const DecodeBatchRequest& request : requests)
            {
                if (!request.session || session_counts[request.session.get()] != 1)
                {
                    staged_candidate = false;
                    break;
                }
                staged_sessions.push_back(request.session.get());
            }
            staged_candidate = staged_candidate && SessionBatchAccess::compatible(staged_sessions);
            if (staged_candidate)
            {
                const int32_t first_input = requests.front().input_id;
                matching_inputs = std::all_of(requests.begin(), requests.end(), [first_input](const DecodeBatchRequest& request) { return request.input_id == first_input; });
            }
            if (staged_candidate && !has_flag(option_flags_, SchedulerOptionForceStagedBatching))
            {
                uint64_t maximum_sequence_length = 0;
                for (const DecodeBatchRequest& request : requests)
                {
                    maximum_sequence_length = std::max<uint64_t>(maximum_sequence_length, request.session->sequence_length());
                }
                state->adaptive_base_bucket = adaptive_bucket(requests.size(), maximum_sequence_length);
                state->adaptive_eligible = true;
                state->adaptive_initial_preference = requests.size() >= 3 || (matching_inputs && worker_count_ < requests.size());
                staged_candidate = choose_staged_batching(state->adaptive_base_bucket, state->adaptive_initial_preference);
            }
        }
        if (requests.size() > 1 && !has_flag(option_flags_, SchedulerOptionDisableStagedBatching) && !staged_candidate)
        {
            staging_bypassed_batches_.fetch_add(1, std::memory_order_relaxed);
        }
        if (staged_candidate)
        {
            std::function<void()> staged_work = [this, requests, complete]() mutable {
                std::vector<Session*> sessions;
                std::vector<int32_t> input_ids;
                sessions.reserve(requests.size());
                input_ids.reserve(requests.size());
                for (const DecodeBatchRequest& request : requests)
                {
                    sessions.push_back(request.session.get());
                    input_ids.push_back(request.input_id);
                }
#if defined(_OPENMP)
                struct OpenMpTeamRestore
                {
                    int team_size = 1;
                    ~OpenMpTeamRestore()
                    {
                        omp_set_num_threads(team_size);
                    }
                };
                const OpenMpTeamRestore restore{omp_get_max_threads()};
                const uint32_t staged_team_size = std::max(1u, std::min(static_cast<uint32_t>(std::thread::hardware_concurrency()), expert_threads_per_worker_ * static_cast<uint32_t>(requests.size())));
                omp_set_num_threads(static_cast<int>(staged_team_size));
#endif
                try
                {
                    StagedDecodeBatchMetrics metrics;
                    auto decoded = SessionBatchAccess::decode(sessions, input_ids, metrics);
                    staged_batches_.fetch_add(1, std::memory_order_relaxed);
                    staged_requests_.fetch_add(requests.size(), std::memory_order_relaxed);
                    logical_expert_batches_.fetch_add(metrics.logical_expert_batches, std::memory_order_relaxed);
                    physical_expert_batches_.fetch_add(metrics.physical_expert_batches, std::memory_order_relaxed);
                    if (metrics.logical_expert_batches > metrics.physical_expert_batches)
                    {
                        coalesced_expert_batches_.fetch_add(metrics.logical_expert_batches - metrics.physical_expert_batches, std::memory_order_relaxed);
                    }
                    coalesced_expert_routes_.fetch_add(metrics.coalesced_expert_routes, std::memory_order_relaxed);
                    update_max(max_coalesced_expert_batch_size_, metrics.max_expert_batch_size);
                    if (!decoded)
                    {
                        release_batch_sessions(sessions);
                        for (size_t index = 0; index < requests.size(); ++index)
                        {
                            complete(index, decoded.error(), {});
                        }
                        return;
                    }
                    release_batch_sessions(sessions);
                    for (size_t index = 0; index < requests.size(); ++index)
                    {
                        complete(
                            index,
                            std::move(decoded.value()[index]),
                            index == 0 ? SessionDecodePhaseSnapshot{metrics.expert_time_microseconds, metrics.expert_cache_wait_time_microseconds} : SessionDecodePhaseSnapshot{});
                    }
                }
                catch (const std::exception& error)
                {
                    release_batch_sessions(sessions);
                    for (size_t index = 0; index < requests.size(); ++index)
                    {
                        complete(index, Error{ErrorCode::InternalError, std::string("staged decode worker failed: ") + error.what()}, {});
                    }
                }
                catch (...)
                {
                    release_batch_sessions(sessions);
                    for (size_t index = 0; index < requests.size(); ++index)
                    {
                        complete(index, Error{ErrorCode::InternalError, "staged decode worker failed"}, {});
                    }
                }
            };
            state->used_staged_batching = true;
            if (try_enqueue_batch(staged_sessions, std::move(staged_work)))
            {
                return future;
            }
            state->used_staged_batching = false;
        }
        for (size_t index = 0; index < requests.size(); ++index)
        {
            DecodeBatchRequest request = std::move(requests[index]);
            if (!request.session)
            {
                rejected_requests_.fetch_add(1, std::memory_order_relaxed);
                complete(index, Error{ErrorCode::InvalidArgument, "batch decode session cannot be null"}, {});
                continue;
            }
            if (session_counts[request.session.get()] != 1)
            {
                rejected_requests_.fetch_add(1, std::memory_order_relaxed);
                complete(index, Error{ErrorCode::InvalidArgument, "a decode batch may contain each session at most once"}, {});
                continue;
            }
            Session* session_key = request.session.get();
            enqueue_session(session_key, [request = std::move(request), index, complete]() mutable {
                try
                {
                    const SessionDecodePhaseSnapshot before = SessionBatchAccess::phase_snapshot(*request.session);
                    auto decoded = request.session->decode(request.input_id);
                    const SessionDecodePhaseSnapshot after = SessionBatchAccess::phase_snapshot(*request.session);
                    complete(index, std::move(decoded),
                             {
                                 after.expert_time_microseconds - before.expert_time_microseconds,
                                 after.expert_cache_wait_time_microseconds - before.expert_cache_wait_time_microseconds,
                             });
                }
                catch (const std::exception& error)
                {
                    complete(index, Error{ErrorCode::InternalError, std::string("decode worker failed: ") + error.what()}, {});
                }
                catch (...)
                {
                    complete(index, Error{ErrorCode::InternalError, "decode worker failed"}, {});
                }
            });
        }
        return future;
    }

    SchedulerStatistics statistics() const noexcept
    {
        SchedulerStatistics result;
        result.submitted_prefill_batches = submitted_prefill_batches_.load(
            std::memory_order_relaxed);
        result.submitted_prefill_requests = submitted_prefill_requests_.load(
            std::memory_order_relaxed);
        result.completed_prefill_requests = completed_prefill_requests_.load(
            std::memory_order_relaxed);
        result.staged_prefill_batches = staged_prefill_batches_.load(
            std::memory_order_relaxed);
        result.staged_prefill_requests = staged_prefill_requests_.load(
            std::memory_order_relaxed);
        result.submitted_batches = submitted_batches_.load(std::memory_order_relaxed);
        result.submitted_requests = submitted_requests_.load(std::memory_order_relaxed);
        result.completed_requests = completed_requests_.load(std::memory_order_relaxed);
        result.rejected_requests = rejected_requests_.load(std::memory_order_relaxed);
        result.max_batch_size = max_batch_size_.load(std::memory_order_relaxed);
        result.max_in_flight = max_in_flight_.load(std::memory_order_relaxed);
        result.serialized_session_requests = serialized_session_requests_.load(std::memory_order_relaxed);
        result.staged_batches = staged_batches_.load(std::memory_order_relaxed);
        result.staged_requests = staged_requests_.load(std::memory_order_relaxed);
        result.staging_bypassed_batches = staging_bypassed_batches_.load(std::memory_order_relaxed);
        result.logical_expert_batches = logical_expert_batches_.load(std::memory_order_relaxed);
        result.physical_expert_batches = physical_expert_batches_.load(std::memory_order_relaxed);
        result.coalesced_expert_batches = coalesced_expert_batches_.load(std::memory_order_relaxed);
        result.coalesced_expert_routes = coalesced_expert_routes_.load(std::memory_order_relaxed);
        result.max_coalesced_expert_batch_size = max_coalesced_expert_batch_size_.load(std::memory_order_relaxed);
        result.adaptive_staged_decisions = adaptive_staged_decisions_.load(std::memory_order_relaxed);
        result.adaptive_independent_decisions = adaptive_independent_decisions_.load(std::memory_order_relaxed);
        result.adaptive_probe_decisions = adaptive_probe_decisions_.load(std::memory_order_relaxed);
        result.adaptive_policy_switches = adaptive_policy_switches_.load(std::memory_order_relaxed);
        result.adaptive_staged_observations = adaptive_staged_observations_.load(std::memory_order_relaxed);
        result.adaptive_independent_observations = adaptive_independent_observations_.load(std::memory_order_relaxed);
        result.adaptive_staged_time_microseconds = adaptive_staged_time_microseconds_.load(std::memory_order_relaxed);
        result.adaptive_independent_time_microseconds = adaptive_independent_time_microseconds_.load(std::memory_order_relaxed);
        result.adaptive_resident_decisions = adaptive_resident_decisions_.load(std::memory_order_relaxed);
        result.adaptive_mixed_decisions = adaptive_mixed_decisions_.load(std::memory_order_relaxed);
        result.adaptive_storage_decisions = adaptive_storage_decisions_.load(std::memory_order_relaxed);
        result.adaptive_resident_observations = adaptive_resident_observations_.load(std::memory_order_relaxed);
        result.adaptive_mixed_observations = adaptive_mixed_observations_.load(std::memory_order_relaxed);
        result.adaptive_storage_observations = adaptive_storage_observations_.load(std::memory_order_relaxed);
        result.adaptive_phase_changes = adaptive_phase_changes_.load(std::memory_order_relaxed);
        result.adaptive_noisy_switch_rejections = adaptive_noisy_switch_rejections_.load(std::memory_order_relaxed);
        result.cross_call_collected_batches = cross_call_collected_batches_.load(std::memory_order_relaxed);
        result.cross_call_collected_requests = cross_call_collected_requests_.load(std::memory_order_relaxed);
        result.cross_call_collection_probes = cross_call_collection_probes_.load(std::memory_order_relaxed);
        result.cross_call_collection_timeouts = cross_call_collection_timeouts_.load(std::memory_order_relaxed);
        result.cross_call_collection_bypasses = cross_call_collection_bypasses_.load(std::memory_order_relaxed);
        result.cross_call_collection_wait_microseconds = cross_call_collection_wait_microseconds_.load(std::memory_order_relaxed);
        result.max_cross_call_batch_size = max_cross_call_batch_size_.load(std::memory_order_relaxed);
        result.max_cross_call_pending = max_cross_call_pending_.load(std::memory_order_relaxed);
        result.affinity_workers_configured = affinity_workers_configured_.load(std::memory_order_relaxed);
        result.affinity_failures = affinity_failures_.load(std::memory_order_relaxed);
        result.affinity_cpu_count = affinity_cpu_count_;
        result.numa_nodes_detected = numa_nodes_detected_;
        result.automatic_topology_affinity = has_flag(option_flags_, scheduler_internal_automatic_topology_affinity);
        result.worker_count = worker_count_;
        result.expert_threads_per_worker = expert_threads_per_worker_;
        return result;
    }

private:
    static void update_max(std::atomic<uint64_t>& destination, uint64_t value)
    {
        uint64_t previous = destination.load(std::memory_order_relaxed);
        while (previous < value && !destination.compare_exchange_weak(previous, value, std::memory_order_relaxed, std::memory_order_relaxed))
        {
        }
    }

    enum AdaptiveStagingFlag : uint32_t
    {
        AdaptivePreferredStaged = UINT32_C(1) << NCNN_MOE_ADAPTIVE_PREFER_STAGED_BIT,
        AdaptiveConfirmationPending = UINT32_C(1) << NCNN_MOE_ADAPTIVE_CONFIRM_BIT,
        AdaptiveInitialized = UINT32_C(1) << NCNN_MOE_ADAPTIVE_INITIALIZED_BIT
    };

    struct AdaptiveStagingPolicy
    {
        double staged_microseconds_per_request = 0.0;
        double independent_microseconds_per_request = 0.0;
        double staged_deviation_microseconds_per_request = 0.0;
        double independent_deviation_microseconds_per_request = 0.0;
        uint64_t staged_samples = 0;
        uint64_t independent_samples = 0;
        uint64_t decisions = 0;
        uint64_t last_probe_decision = 0;
        uint32_t flags = 0;
    };

    static AdaptivePhase classify_adaptive_phase(uint64_t expert_time_microseconds, uint64_t expert_cache_wait_time_microseconds) noexcept
    {
        if (expert_time_microseconds == 0 || expert_cache_wait_time_microseconds <= expert_time_microseconds / 20)
        {
            return AdaptivePhase::Resident;
        }
        if (expert_cache_wait_time_microseconds < expert_time_microseconds / 2)
        {
            return AdaptivePhase::Mixed;
        }
        return AdaptivePhase::Storage;
    }

    static size_t adaptive_policy_bucket(size_t base_bucket, AdaptivePhase phase) noexcept
    {
        return base_bucket * adaptive_phase_count + static_cast<size_t>(phase);
    }

    static size_t adaptive_bucket(size_t request_count, uint64_t sequence_length)
    {
        const size_t batch_bucket = request_count <= 2 ? 0 : request_count <= 4 ? 1
                                                         : request_count <= 8   ? 2
                                                                                : 3;
        const size_t context_bucket = sequence_length <= 512 ? 0 : sequence_length <= 2048 ? 1
                                                               : sequence_length <= 8192   ? 2
                                                                                           : 3;
        return context_bucket * 4 + batch_bucket;
    }

    uint64_t effective_probe_interval(const AdaptiveStagingPolicy& policy) const noexcept
    {
        uint64_t multiplier = 1;
        if (policy.staged_samples != 0 && policy.independent_samples != 0)
        {
            const double faster = std::min(policy.staged_microseconds_per_request, policy.independent_microseconds_per_request);
            const double slower = std::max(policy.staged_microseconds_per_request, policy.independent_microseconds_per_request);
            // Back off probes when the alternative is clearly slower.
            if (faster > 0.0)
            {
                if (slower > faster * 1.50)
                    multiplier = 8;
                else if (slower > faster * 1.25)
                    multiplier = 4;
                else if (slower > faster * 1.10)
                    multiplier = 2;
            }
        }
        return static_cast<uint64_t>(adaptive_probe_interval_) * multiplier;
    }

    bool choose_staged_batching(size_t base_bucket, bool initial_preference)
    {
        bool staged = initial_preference;
        bool probe = false;
        AdaptivePhase phase = AdaptivePhase::Resident;
        {
            const std::lock_guard<std::mutex> lock(adaptive_mutex_);
            phase = adaptive_phases_[base_bucket];
            AdaptiveStagingPolicy& policy = adaptive_policies_[adaptive_policy_bucket(base_bucket, phase)];
            if (!has_flag(policy.flags, AdaptiveInitialized))
            {
                policy.flags |= AdaptiveInitialized;
                if (initial_preference)
                    policy.flags |= AdaptivePreferredStaged;
            }
            const bool preferred_staged = has_flag(policy.flags, AdaptivePreferredStaged);
            staged = preferred_staged;
            const uint64_t probe_interval = effective_probe_interval(policy);
            const bool probe_due = policy.decisions != 0 && policy.decisions - policy.last_probe_decision >= probe_interval;
            const uint64_t preferred_samples = preferred_staged ? policy.staged_samples : policy.independent_samples;
            if (preferred_samples == 0)
            {
                staged = preferred_staged;
            }
            else if (has_flag(policy.flags, AdaptiveConfirmationPending))
            {
                staged = !preferred_staged;
                probe = true;
                policy.flags &= ~AdaptiveConfirmationPending;
                policy.last_probe_decision = policy.decisions;
            }
            else if (probe_due)
            {
                staged = !preferred_staged;
                probe = true;
                policy.last_probe_decision = policy.decisions;
            }
            ++policy.decisions;
        }
        if (staged)
        {
            adaptive_staged_decisions_.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            adaptive_independent_decisions_.fetch_add(1, std::memory_order_relaxed);
        }
        if (probe)
        {
            adaptive_probe_decisions_.fetch_add(1, std::memory_order_relaxed);
        }
        switch (phase)
        {
        case AdaptivePhase::Resident: adaptive_resident_decisions_.fetch_add(1, std::memory_order_relaxed); break;
        case AdaptivePhase::Mixed: adaptive_mixed_decisions_.fetch_add(1, std::memory_order_relaxed); break;
        case AdaptivePhase::Storage: adaptive_storage_decisions_.fetch_add(1, std::memory_order_relaxed); break;
        }
        return staged;
    }

    void observe_adaptive_staging(size_t base_bucket, bool staged, bool initial_preference, uint64_t elapsed_microseconds, size_t request_count, uint64_t expert_time_microseconds, uint64_t expert_cache_wait_time_microseconds)
    {
        if (request_count == 0 || base_bucket >= adaptive_base_bucket_count)
            return;
        const double observation = static_cast<double>(elapsed_microseconds) / static_cast<double>(request_count);
        const AdaptivePhase phase = classify_adaptive_phase(expert_time_microseconds, expert_cache_wait_time_microseconds);
        bool switched = false;
        bool phase_changed = false;
        bool noisy_switch_rejected = false;
        {
            const std::lock_guard<std::mutex> lock(adaptive_mutex_);
            phase_changed = adaptive_phases_[base_bucket] != phase;
            adaptive_phases_[base_bucket] = phase;
            AdaptiveStagingPolicy& policy = adaptive_policies_[adaptive_policy_bucket(base_bucket, phase)];
            if (!has_flag(policy.flags, AdaptiveInitialized))
            {
                policy.flags |= AdaptiveInitialized;
                if (initial_preference)
                    policy.flags |= AdaptivePreferredStaged;
            }
            double& estimate = staged ? policy.staged_microseconds_per_request : policy.independent_microseconds_per_request;
            double& deviation = staged ? policy.staged_deviation_microseconds_per_request : policy.independent_deviation_microseconds_per_request;
            uint64_t& samples = staged ? policy.staged_samples : policy.independent_samples;
            if (samples == 0)
            {
                estimate = observation;
                deviation = 0.0;
            }
            else
            {
                const double previous_estimate = estimate;
                estimate = previous_estimate * 0.75 + observation * 0.25;
                deviation = deviation * 0.75 + std::abs(observation - previous_estimate) * 0.25;
            }
            ++samples;
            if (policy.staged_samples >= 1 && policy.independent_samples >= 1)
            {
                bool preferred = has_flag(policy.flags, AdaptivePreferredStaged);
                const double candidate_estimate = preferred ? policy.independent_microseconds_per_request : policy.staged_microseconds_per_request;
                const double preferred_estimate = preferred ? policy.staged_microseconds_per_request : policy.independent_microseconds_per_request;
                const double candidate_deviation = preferred
                                                       ? policy.independent_deviation_microseconds_per_request
                                                       : policy.staged_deviation_microseconds_per_request;
                const double preferred_deviation = preferred
                                                       ? policy.staged_deviation_microseconds_per_request
                                                       : policy.independent_deviation_microseconds_per_request;
                const uint64_t candidate_samples = preferred ? policy.independent_samples : policy.staged_samples;
                const double margin = adaptive_switch_margin(candidate_samples);
                const bool relative_candidate = candidate_estimate * margin < preferred_estimate;
                // Reject gains smaller than half the combined deviation.
                const double noise_guard = (candidate_deviation + preferred_deviation) * 0.5;
                const bool confident_candidate = candidate_estimate * margin + noise_guard < preferred_estimate;
                if (relative_candidate && !confident_candidate)
                {
                    noisy_switch_rejected = true;
                }
                if (confident_candidate)
                {
                    preferred = !preferred;
                }
                if (preferred != has_flag(policy.flags, AdaptivePreferredStaged))
                {
                    const uint64_t confirmed_candidate_samples = preferred ? policy.staged_samples : policy.independent_samples;
                    const bool observed_candidate = staged == preferred;
                    if (observed_candidate && confirmed_candidate_samples >= 2)
                    {
                        switched = true;
                        if (preferred)
                            policy.flags |= AdaptivePreferredStaged;
                        else
                            policy.flags &= ~AdaptivePreferredStaged;
                        policy.flags &= ~AdaptiveConfirmationPending;
                    }
                    else
                    {
                        policy.flags |= AdaptiveConfirmationPending;
                    }
                }
                else
                {
                    policy.flags &= ~AdaptiveConfirmationPending;
                }
            }
        }
        if (phase_changed)
        {
            adaptive_phase_changes_.fetch_add(1, std::memory_order_relaxed);
        }
        if (noisy_switch_rejected)
        {
            adaptive_noisy_switch_rejections_.fetch_add(1, std::memory_order_relaxed);
        }
        switch (phase)
        {
        case AdaptivePhase::Resident: adaptive_resident_observations_.fetch_add(1, std::memory_order_relaxed); break;
        case AdaptivePhase::Mixed: adaptive_mixed_observations_.fetch_add(1, std::memory_order_relaxed); break;
        case AdaptivePhase::Storage: adaptive_storage_observations_.fetch_add(1, std::memory_order_relaxed); break;
        }
        if (staged)
        {
            adaptive_staged_observations_.fetch_add(1, std::memory_order_relaxed);
            adaptive_staged_time_microseconds_.fetch_add(static_cast<uint64_t>(observation), std::memory_order_relaxed);
        }
        else
        {
            adaptive_independent_observations_.fetch_add(1, std::memory_order_relaxed);
            adaptive_independent_time_microseconds_.fetch_add(static_cast<uint64_t>(observation), std::memory_order_relaxed);
        }
        if (switched)
        {
            adaptive_policy_switches_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void cross_call_loop()
    {
        uint32_t consecutive_timeouts = 0;
        uint64_t decisions = 0;
        uint64_t last_probe_decision = 0;
        for (;;)
        {
            std::vector<PendingCrossCall> pending;
            bool probed = false;
            {
                std::unique_lock<std::mutex> lock(cross_call_mutex_);
                cross_call_ready_.wait(lock, [this] { return cross_call_stopping_ || !cross_call_pending_.empty(); });
                if (cross_call_stopping_ && cross_call_pending_.empty())
                {
                    return;
                }

                ++decisions;
                const auto has_distinct_peer = [this] {
                    if (cross_call_pending_.size() < 2)
                    {
                        return false;
                    }
                    Session* first = cross_call_pending_.front().request.session.get();
                    return std::any_of(std::next(cross_call_pending_.begin()), cross_call_pending_.end(), [first](const PendingCrossCall& submission) { return submission.request.session.get() != first; });
                };
                const bool already_collected = has_distinct_peer();
                const bool probe_due = consecutive_timeouts < 4 || decisions - last_probe_decision >= adaptive_probe_interval_;
                if (!already_collected && !cross_call_stopping_ && probe_due)
                {
                    probed = true;
                    last_probe_decision = decisions;
                    cross_call_collection_probes_.fetch_add(1, std::memory_order_relaxed);
                    const auto wait_started = std::chrono::steady_clock::now();
                    const auto deadline = cross_call_pending_.front().submitted_at + std::chrono::microseconds(cross_call_window_microseconds_);
                    if (deadline > wait_started)
                    {
                        cross_call_ready_.wait_until(lock, deadline, [this, &has_distinct_peer] { return cross_call_stopping_ || has_distinct_peer(); });
                    }
                    cross_call_collection_wait_microseconds_.fetch_add(static_cast<uint64_t>(std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_started).count())),
                                                                       std::memory_order_relaxed);
                }
                else if (!already_collected && !cross_call_stopping_)
                {
                    cross_call_collection_bypasses_.fetch_add(1, std::memory_order_relaxed);
                }

                pending.reserve(std::min<size_t>(cross_call_pending_.size(), cross_call_max_batch_size_));
                std::vector<Session*> selected_sessions;
                auto submission = cross_call_pending_.begin();
                while (submission != cross_call_pending_.end() && pending.size() < cross_call_max_batch_size_)
                {
                    Session* session = submission->request.session.get();
                    if (std::find(selected_sessions.begin(), selected_sessions.end(), session) != selected_sessions.end())
                    {
                        ++submission;
                        continue;
                    }
                    selected_sessions.push_back(session);
                    pending.push_back(std::move(*submission));
                    submission = cross_call_pending_.erase(submission);
                }
            }

            if (pending.size() > 1)
            {
                consecutive_timeouts = 0;
                cross_call_collected_batches_.fetch_add(1, std::memory_order_relaxed);
                cross_call_collected_requests_.fetch_add(pending.size(), std::memory_order_relaxed);
                update_max(max_cross_call_batch_size_, pending.size());
                update_max(max_batch_size_, pending.size());
            }
            else if (probed)
            {
                ++consecutive_timeouts;
                cross_call_collection_timeouts_.fetch_add(1, std::memory_order_relaxed);
            }

            std::vector<DecodeBatchRequest> requests;
            requests.reserve(pending.size());
            for (PendingCrossCall& submission : pending)
            {
                requests.push_back(std::move(submission.request));
            }

            try
            {
                std::vector<Result<DecodeResult>> results = schedule_decode(std::move(requests)).get();
                if (results.size() != pending.size())
                {
                    const Error error{ErrorCode::InternalError, "cross-call decode returned an invalid result count"};
                    for (PendingCrossCall& submission : pending)
                    {
                        submission.promise.set_value({error});
                    }
                    continue;
                }
                for (size_t index = 0; index < pending.size(); ++index)
                {
                    std::vector<Result<DecodeResult>> single;
                    single.push_back(std::move(results[index]));
                    pending[index].promise.set_value(std::move(single));
                }
            }
            catch (const std::exception& error)
            {
                const Error result{ErrorCode::InternalError, std::string("cross-call decode failed: ") + error.what()};
                for (PendingCrossCall& submission : pending)
                {
                    submission.promise.set_value({result});
                }
            }
            catch (...)
            {
                const Error result{ErrorCode::InternalError, "cross-call decode failed"};
                for (PendingCrossCall& submission : pending)
                {
                    submission.promise.set_value({result});
                }
            }
        }
    }

    struct SessionQueue
    {
        std::deque<std::function<void()>> work;
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
            if (!pending.active)
            {
                pending.active = true;
                queue_.push_back([this, session] { execute_session(session); });
            }
        }
        queue_ready_.notify_one();
    }

    bool try_enqueue_batch(const std::vector<Session*>& sessions, std::function<void()> work)
    {
        {
            const std::lock_guard<std::mutex> lock(queue_mutex_);
            for (Session* session : sessions)
            {
                if (session_queues_.find(session) != session_queues_.end())
                {
                    return false;
                }
            }
            for (Session* session : sessions)
                session_queues_[session].active = true;
            queue_.push_back(std::move(work));
        }
        queue_ready_.notify_one();
        return true;
    }

    void release_batch_sessions(const std::vector<Session*>& sessions)
    {
        const std::lock_guard<std::mutex> lock(queue_mutex_);
        for (Session* session : sessions)
        {
            auto pending = session_queues_.find(session);
            if (pending == session_queues_.end())
                continue;
            if (pending->second.work.empty())
            {
                session_queues_.erase(pending);
                continue;
            }
            queue_.push_back([this, session] { execute_session(session); });
        }
        queue_ready_.notify_all();
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
            if (pending->second.work.empty())
            {
                session_queues_.erase(pending);
            }
            else
            {
                queue_.push_back([this, session] { execute_session(session); });
                queue_ready_.notify_one();
            }
        }
    }

    void worker_loop()
    {
        for (;;)
        {
            std::function<void()> work;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
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
        if (!has_flag(option_flags_, SchedulerOptionPinWorkers))
            return;
        const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (online_cpus <= 0)
            return;
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        if (!worker_cpu_sets_.empty())
        {
            for (uint32_t cpu : worker_cpu_sets_[worker_index])
            {
                if (cpu >= CPU_SETSIZE)
                {
                    affinity_failures_.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                CPU_SET(cpu, &cpu_set);
            }
        }
        else
        {
            const uint32_t online = static_cast<uint32_t>(online_cpus);
            const uint32_t partition_size = worker_count_ <= online
                                                ? online / worker_count_ + (worker_index < online % worker_count_ ? 1u : 0u)
                                                : 1u;
            const uint32_t first_cpu = worker_count_ <= online
                                           ? worker_index * (online / worker_count_) + std::min(worker_index, online % worker_count_)
                                           : worker_index % online;
            for (uint32_t offset = 0; offset < partition_size; ++offset)
                CPU_SET((first_cpu + offset) % online, &cpu_set);
        }
        const int affinity_result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
        if (affinity_result == 0)
            affinity_workers_configured_.fetch_add(1, std::memory_order_relaxed);
        else
            affinity_failures_.fetch_add(1, std::memory_order_relaxed);
#else
        (void)option_flags_;
        (void)worker_index;
#endif
    }

    uint32_t option_flags_ = 0;
    bool stopping_ = false;
    bool cross_call_stopping_ = false;
    uint32_t adaptive_probe_interval_ = 32;
    uint32_t cross_call_window_microseconds_ = 200;
    uint32_t cross_call_max_batch_size_ = 0;
    uint32_t worker_count_ = 0;
    uint32_t expert_threads_per_worker_ = 1;
    uint32_t affinity_cpu_count_ = 0;
    uint32_t numa_nodes_detected_ = 0;
    std::vector<std::vector<uint32_t>> worker_cpu_sets_;
    std::vector<std::thread> workers_;
    std::thread cross_call_collector_;
    std::deque<std::function<void()>> queue_;
    std::unordered_map<Session*, SessionQueue> session_queues_;
    std::mutex queue_mutex_;
    std::condition_variable queue_ready_;
    std::deque<PendingCrossCall> cross_call_pending_;
    std::mutex cross_call_mutex_;
    std::condition_variable cross_call_ready_;
    std::array<AdaptiveStagingPolicy, adaptive_base_bucket_count * adaptive_phase_count> adaptive_policies_;
    std::array<AdaptivePhase, adaptive_base_bucket_count> adaptive_phases_{};
    std::mutex adaptive_mutex_;
    std::atomic<uint64_t> submitted_batches_{0};
    std::atomic<uint64_t> submitted_requests_{0};
    std::atomic<uint64_t> completed_requests_{0};
    std::atomic<uint64_t> rejected_requests_{0};
    std::atomic<uint64_t> max_batch_size_{0};
    std::atomic<uint64_t> in_flight_{0};
    std::atomic<uint64_t> max_in_flight_{0};
    std::atomic<uint64_t> serialized_session_requests_{0};
    std::atomic<uint64_t> staged_batches_{0};
    std::atomic<uint64_t> staged_requests_{0};
    std::atomic<uint64_t> staging_bypassed_batches_{0};
    std::atomic<uint64_t> logical_expert_batches_{0};
    std::atomic<uint64_t> physical_expert_batches_{0};
    std::atomic<uint64_t> coalesced_expert_batches_{0};
    std::atomic<uint64_t> coalesced_expert_routes_{0};
    std::atomic<uint64_t> max_coalesced_expert_batch_size_{0};
    std::atomic<uint64_t> adaptive_staged_decisions_{0};
    std::atomic<uint64_t> adaptive_independent_decisions_{0};
    std::atomic<uint64_t> adaptive_probe_decisions_{0};
    std::atomic<uint64_t> adaptive_policy_switches_{0};
    std::atomic<uint64_t> adaptive_staged_observations_{0};
    std::atomic<uint64_t> adaptive_independent_observations_{0};
    std::atomic<uint64_t> adaptive_staged_time_microseconds_{0};
    std::atomic<uint64_t> adaptive_independent_time_microseconds_{0};
    std::atomic<uint64_t> adaptive_resident_decisions_{0};
    std::atomic<uint64_t> adaptive_mixed_decisions_{0};
    std::atomic<uint64_t> adaptive_storage_decisions_{0};
    std::atomic<uint64_t> adaptive_resident_observations_{0};
    std::atomic<uint64_t> adaptive_mixed_observations_{0};
    std::atomic<uint64_t> adaptive_storage_observations_{0};
    std::atomic<uint64_t> adaptive_phase_changes_{0};
    std::atomic<uint64_t> adaptive_noisy_switch_rejections_{0};
    std::atomic<uint64_t> cross_call_collected_batches_{0};
    std::atomic<uint64_t> cross_call_collected_requests_{0};
    std::atomic<uint64_t> cross_call_collection_probes_{0};
    std::atomic<uint64_t> cross_call_collection_timeouts_{0};
    std::atomic<uint64_t> cross_call_collection_bypasses_{0};
    std::atomic<uint64_t> cross_call_collection_wait_microseconds_{0};
    std::atomic<uint64_t> max_cross_call_batch_size_{0};
    std::atomic<uint64_t> max_cross_call_pending_{0};
    std::atomic<uint64_t> affinity_workers_configured_{0};
    std::atomic<uint64_t> affinity_failures_{0};
    std::atomic<uint64_t> submitted_prefill_batches_{0};
    std::atomic<uint64_t> submitted_prefill_requests_{0};
    std::atomic<uint64_t> completed_prefill_requests_{0};
    std::atomic<uint64_t> staged_prefill_batches_{0};
    std::atomic<uint64_t> staged_prefill_requests_{0};
};

BatchScheduler::BatchScheduler(const SchedulerOptions& options)
    : implementation_(new Implementation(options))
{
}

BatchScheduler::~BatchScheduler() = default;

std::future<std::vector<Result<PrefillResult>>> BatchScheduler::submit_prefill(
    std::vector<PrefillBatchRequest> requests)
{
    return implementation_->submit_prefill(std::move(requests));
}

std::future<std::vector<Result<DecodeResult>>> BatchScheduler::submit_decode(std::vector<DecodeBatchRequest> requests)
{
    return implementation_->submit_decode(std::move(requests));
}

SchedulerStatistics BatchScheduler::statistics() const noexcept
{
    return implementation_->statistics();
}

} // namespace moe
} // namespace ncnn
