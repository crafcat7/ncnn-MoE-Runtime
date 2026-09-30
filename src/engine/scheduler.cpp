#include "ncnn/moe/scheduler.h"

#include "executor.h"
#include "cpu.h"
#include "sessionstate.h"
#include "graph/compiledmodel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <list>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace ncnn {
namespace moe {

static constexpr const char* token_id_error_message = "token id is outside the model vocabulary";
static constexpr const char* execution_failure_error_message = "session is unavailable after an execution failure; reset is required";

[[nodiscard]] static bool valid_token_id(const MoeModelDescriptor& descriptor, int32_t token_id) noexcept
{
    return token_id >= 0 && static_cast<uint32_t>(token_id) < descriptor.vocabulary_size;
}

class BatchSchedulerPrivate
{
public:
    explicit BatchSchedulerPrivate(const SchedulerOptions& opt);
    ~BatchSchedulerPrivate();

    std::future<std::vector<Result<PrefillResult>>> submit_prefill(std::vector<PrefillBatchRequest> requests);
    std::future<std::vector<Result<DecodeResult>>> submit_decode(std::vector<DecodeBatchRequest> requests);
    SchedulerStatistics statistics() const noexcept;

private:
    using Work = std::function<void(BatchWorkspace&)>;

    [[nodiscard]] static Result<void> get_session_lock_order(std::vector<Session*>& sessions,
                                                             const char* null_message,
                                                             const char* duplicate_message);
    [[nodiscard]] static Result<std::vector<PrefillResult>> prefill(std::span<Session* const> lock_order,
                                                                    std::span<const PrefillBatchRequest> requests,
                                                                    BatchWorkspace& workspace);
    [[nodiscard]] static Result<std::vector<DecodeResult>> decode(std::span<Session* const> lock_order,
                                                                  std::span<const DecodeBatchRequest> requests,
                                                                  BatchWorkspace& workspace);

    void stop_workers();
    void worker_loop();
    void enqueue_session(Session* session, Work work);
    bool try_enqueue_batch(const std::vector<Session*>& sessions, Work work);
    void release_sessions(std::span<Session* const> sessions);

#if defined(_OPENMP)
    [[nodiscard]] uint32_t prepare_staging_team(uint64_t work_units,
                                                uint32_t independent_work_items,
                                                CpuThreadBudgetController::Lease& extra_compute);
#endif // defined(_OPENMP)

    CpuThreadBudgetController cpu_budget;
    bool use_staged_decode = true;
    bool stopping = false;
    uint32_t team_size = 1;
    std::vector<std::thread> workers;
    std::list<Work> queue;
    // Presence reserves a Session, even while its pending queue is empty.
    std::unordered_map<Session*, std::list<Work>> session_queues;
    std::mutex queue_mutex;
    std::condition_variable queue_ready;
    std::atomic<uint64_t> prefill_batches{0};
    std::atomic<uint64_t> decode_batches{0};
    std::atomic<uint64_t> staged_prefill_batches{0};
    std::atomic<uint64_t> staged_decode_batches{0};
};

Result<void> BatchSchedulerPrivate::get_session_lock_order(std::vector<Session*>& sessions,
                                                           const char* null_message,
                                                           const char* duplicate_message)
{
    for (const Session* session : sessions)
    {
        if (!session)
            return Error{ErrorCode::InvalidArgument, null_message};
    }
    std::sort(sessions.begin(), sessions.end(), std::less<Session*>());
    if (std::adjacent_find(sessions.begin(), sessions.end()) != sessions.end())
        return Error{ErrorCode::InvalidArgument, duplicate_message};
    return {};
}

Result<std::vector<PrefillResult>> BatchSchedulerPrivate::prefill(std::span<Session* const> lock_order,
                                                                  std::span<const PrefillBatchRequest> requests,
                                                                  BatchWorkspace& workspace)
{
    if (lock_order.empty() || lock_order.size() != requests.size())
    {
        return Error{
            ErrorCode::InvalidArgument,
            "staged prefill requires one input sequence per session"};
    }

    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(lock_order.size());
    for (Session* session : lock_order)
        locks.emplace_back(session->mutex);

    const ModelPtr& model = requests.front().session->model;
    const CompiledModel& compiled = model_compiled(*model);
    const uint32_t max_context_length = Session::get_max_context_length(model->descriptor());
    size_t maximum_tokens = 0;
    for (size_t index = 0; index < requests.size(); ++index)
    {
        Session& session = *requests[index].session;
        if (session.state->execution_failed)
            return Error{ErrorCode::InternalError, execution_failure_error_message};
        if (requests[index].input_ids.empty())
        {
            return Error{
                ErrorCode::InvalidArgument,
                "staged prefill requires non-empty input sequences"};
        }
        if (requests[index].input_ids.size()
            > std::numeric_limits<uint32_t>::max())
        {
            return Error{
                ErrorCode::InvalidArgument,
                "prefill token count exceeds uint32 range"};
        }
        if (max_context_length > 0
            && requests[index].input_ids.size()
                   > max_context_length
                         - std::min<uint64_t>(session.token_count,
                                              max_context_length))
        {
            return Error{
                ErrorCode::InvalidArgument,
                "prefill exceeds the model context length"};
        }
        for (int32_t token_id : requests[index].input_ids)
        {
            if (!valid_token_id(model->descriptor(), token_id))
                return Error{ErrorCode::InvalidArgument, token_id_error_message};
        }
        maximum_tokens = std::max(maximum_tokens, requests[index].input_ids.size());
    }

    for (Session* session : lock_order)
    {
        session->state->execution_failed = true;
        session->stats_scratch = session->stats;
    }

    std::vector<PrefillResult> results(requests.size());
    std::vector<DecodeBatchEntry> entries;
    entries.reserve(requests.size());

    for (size_t token_index = 0;
         token_index < maximum_tokens;
         ++token_index)
    {
        entries.clear();
        for (size_t session_index = 0;
             session_index < requests.size();
             ++session_index)
        {
            if (token_index >= requests[session_index].input_ids.size())
                continue;
            Session& session = *requests[session_index].session;
            entries.push_back({
                requests[session_index].input_ids[token_index],
                &session.stats_scratch,
                session.state.get(),
                session.token_count + token_index,
                token_index + 1 == requests[session_index].input_ids.size(),
            });
        }

        auto logits = forward_decode_batch(compiled,
                                           entries,
                                           workspace);
        if (!logits)
            return logits.error();
        if (logits.value().size() != entries.size())
        {
            return Error{
                ErrorCode::InternalError,
                "staged prefill returned an invalid result count"};
        }
        // Active batch results retain the input session order.
        size_t active_index = 0;
        for (size_t session_index = 0;
             session_index < requests.size();
             ++session_index)
        {
            if (token_index >= requests[session_index].input_ids.size())
                continue;
            Session& session = *requests[session_index].session;
            auto speculative_context = update_speculative_context(compiled,
                                                                  session.stats_scratch,
                                                                  *session.state);
            if (!speculative_context)
                return speculative_context.error();
            results[session_index].logits = std::move(logits.value()[active_index++]);
        }
    }

    for (size_t index = 0; index < requests.size(); ++index)
    {
        Session& session = *requests[index].session;
        session.commit_execution(requests[index].input_ids.size(), 0);
        results[index].processed_tokens = static_cast<uint32_t>(requests[index].input_ids.size());
    }
    return results;
}

Result<std::vector<DecodeResult>> BatchSchedulerPrivate::decode(std::span<Session* const> lock_order,
                                                                std::span<const DecodeBatchRequest> requests,
                                                                BatchWorkspace& workspace)
{
    if (lock_order.empty() || lock_order.size() != requests.size())
    {
        return Error{ErrorCode::InvalidArgument, "staged decode requires one input id per session"};
    }

    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(lock_order.size());
    for (Session* session : lock_order)
        locks.emplace_back(session->mutex);

    const ModelPtr& model = requests.front().session->model;
    const CompiledModel& compiled = model_compiled(*model);
    const uint32_t max_context_length = Session::get_max_context_length(model->descriptor());
    for (size_t index = 0; index < requests.size(); ++index)
    {
        Session& session = *requests[index].session;
        if (session.state->execution_failed)
            return Error{ErrorCode::InternalError, execution_failure_error_message};
        if (max_context_length > 0 && session.token_count >= max_context_length)
        {
            return Error{ErrorCode::InvalidArgument, "decode exceeds the model context length"};
        }
        if (!valid_token_id(model->descriptor(), requests[index].input_id))
            return Error{ErrorCode::InvalidArgument, token_id_error_message};
    }

    std::vector<DecodeBatchEntry> entries(requests.size());
    for (size_t index = 0; index < requests.size(); ++index)
    {
        Session& session = *requests[index].session;
        session.state->execution_failed = true;
        session.stats_scratch = session.stats;
        entries[index] = {
            requests[index].input_id,
            &session.stats_scratch,
            session.state.get(),
            session.token_count,
        };
    }

    auto logits = forward_decode_batch(compiled, entries, workspace);
    if (!logits)
        return logits.error();
    if (logits.value().size() != requests.size())
    {
        return Error{ErrorCode::InternalError, "staged decode returned an invalid result count"};
    }
    for (size_t index = 0; index < requests.size(); ++index)
    {
        auto speculative_context = update_speculative_context(compiled,
                                                              requests[index].session->stats_scratch,
                                                              *requests[index].session->state);
        if (!speculative_context)
            return speculative_context.error();
    }

    std::vector<DecodeResult> results(requests.size());
    for (size_t index = 0; index < requests.size(); ++index)
    {
        Session& session = *requests[index].session;
        session.commit_execution(0, 1);
        results[index].logits = std::move(logits.value()[index]);
        results[index].sequence_length = session.token_count;
    }
    return results;
}

BatchSchedulerPrivate::BatchSchedulerPrivate(const SchedulerOptions& opt)
    : cpu_budget(resolve_cpu_thread_budget()),
      use_staged_decode(opt.use_staged_decode)
{
    const CpuThreadBudget& thread_budget = cpu_budget.budget();
    uint32_t num_threads = opt.num_threads == 0 ? std::min(4u, thread_budget.num_threads) : opt.num_threads;
    num_threads = std::max(1u, std::min(num_threads, thread_budget.num_threads));
#if defined(_OPENMP)
    team_size = std::max(1u, thread_budget.num_threads / num_threads);
#else
    team_size = 1;
#endif
    try
    {
        workers.reserve(num_threads);
        for (uint32_t i = 0; i < num_threads; i++)
            workers.emplace_back([this] { worker_loop(); });
    }
    catch (...)
    {
        stop_workers();
        throw;
    }
}

BatchSchedulerPrivate::~BatchSchedulerPrivate()
{
    stop_workers();
}

void BatchSchedulerPrivate::stop_workers()
{
    {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        stopping = true;
    }
    queue_ready.notify_all();
    for (std::thread& worker : workers)
    {
        if (worker.joinable())
            worker.join();
    }
}

std::future<std::vector<Result<PrefillResult>>> BatchSchedulerPrivate::submit_prefill(std::vector<PrefillBatchRequest> requests)
{
    prefill_batches.fetch_add(1, std::memory_order_relaxed);
    auto promise = std::make_shared<std::promise<std::vector<Result<PrefillResult>>>>();
    auto future = promise->get_future();
    auto reject = [promise, count = requests.size()](const char* message) {
        promise->set_value(std::vector<Result<PrefillResult>>(count, Error{ErrorCode::InvalidArgument, message}));
    };
    if (requests.empty())
    {
        promise->set_value({});
        return future;
    }

    if (requests.size() == 1)
    {
        if (!requests.front().session || requests.front().input_ids.empty())
        {
            reject("a prefill batch requires unique sessions and non-empty inputs");
            return future;
        }
        PrefillBatchRequest request = std::move(requests.front());
        Session* session = request.session.get();
        enqueue_session(session, [this, promise, session, request = std::move(request)](BatchWorkspace&) mutable {
            std::vector<Result<PrefillResult>> results;
            try
            {
                results.emplace_back(request.session->prefill(request.input_ids));
            }
            catch (const std::exception& error)
            {
                results.assign(1, Error{ErrorCode::InternalError, std::string("prefill worker failed: ") + error.what()});
            }
            catch (...)
            {
                results.assign(1, Error{ErrorCode::InternalError, "prefill worker failed"});
            }
            release_sessions({&session, 1});
            promise->set_value(std::move(results));
        });
        return future;
    }

    std::vector<Session*> lock_order;
    lock_order.reserve(requests.size());
    const Model* model = nullptr;
    for (const PrefillBatchRequest& request : requests)
    {
        Session* session = request.session.get();
        if (!session || request.input_ids.empty())
        {
            reject("a prefill batch requires unique sessions and non-empty inputs");
            return future;
        }
        if (!model)
            model = session->model.get();
        else if (session->model.get() != model)
        {
            reject("staged prefill sessions must share one loaded model");
            return future;
        }
        lock_order.push_back(session);
    }

    auto lock_result = get_session_lock_order(lock_order,
                                              "staged prefill session cannot be null",
                                              "staged prefill requires unique sessions");
    if (!lock_result)
    {
        reject("a prefill batch requires unique sessions and non-empty inputs");
        return future;
    }

    Work work = [this, promise, lock_order, requests = std::move(requests)](BatchWorkspace& workspace) mutable {
        std::vector<Result<PrefillResult>> results;
        try
        {
#if defined(_OPENMP)
            CpuOpenMpThreadLimitScope thread_limit;
            CpuThreadBudgetController::Lease extra_compute;
            uint64_t work_units = 0;
            for (const PrefillBatchRequest& request : requests)
                work_units += static_cast<uint64_t>(request.input_ids.size());
            thread_limit.set(prepare_staging_team(work_units, static_cast<uint32_t>(requests.size()), extra_compute));
#endif
            auto ret = prefill(lock_order, requests, workspace);
            staged_prefill_batches.fetch_add(1, std::memory_order_relaxed);
            if (ret)
            {
                results.reserve(ret.value().size());
                for (PrefillResult& result : ret.value())
                    results.emplace_back(std::move(result));
            }
            else
            {
                results.assign(requests.size(), ret.error());
            }
        }
        catch (const std::exception& error)
        {
            results.assign(requests.size(), Error{ErrorCode::InternalError, std::string("staged prefill worker failed: ") + error.what()});
        }
        catch (...)
        {
            results.assign(requests.size(), Error{ErrorCode::InternalError, "staged prefill worker failed"});
        }
        release_sessions(lock_order);
        promise->set_value(std::move(results));
    };
    if (!try_enqueue_batch(lock_order, std::move(work)))
        reject("staged prefill sessions have pending scheduler work");
    return future;
}

std::future<std::vector<Result<DecodeResult>>> BatchSchedulerPrivate::submit_decode(std::vector<DecodeBatchRequest> requests)
{
    decode_batches.fetch_add(1, std::memory_order_relaxed);
    struct BatchState
    {
        std::vector<Result<DecodeResult>> results;
        std::atomic<size_t> remaining{0};
        std::promise<std::vector<Result<DecodeResult>>> promise;
    };

    auto state = std::make_shared<BatchState>();
    auto future = state->promise.get_future();
    if (requests.empty())
    {
        state->promise.set_value({});
        return future;
    }

    const bool attempt_stage = use_staged_decode && requests.size() > 1;
    std::vector<Session*> lock_order;
    const Model* model = nullptr;
    bool same_model = true;
    lock_order.reserve(requests.size());
    for (const DecodeBatchRequest& request : requests)
    {
        Session* session = request.session.get();
        if (!session)
        {
            state->promise.set_value(std::vector<Result<DecodeResult>>(requests.size(), Error{ErrorCode::InvalidArgument, "a decode batch requires unique non-null sessions"}));
            return future;
        }
        lock_order.push_back(session);
        if (attempt_stage)
        {
            if (!model)
                model = session->model.get();
            else if (session->model.get() != model)
                same_model = false;
        }
    }

    auto lock_result = get_session_lock_order(lock_order,
                                              "staged decode session cannot be null",
                                              "a decode batch requires unique non-null sessions");
    if (!lock_result)
    {
        state->promise.set_value(std::vector<Result<DecodeResult>>(requests.size(), lock_result.error()));
        return future;
    }

    const bool can_stage = attempt_stage && same_model;
    if (can_stage)
    {
        Work work = [this, state, requests, lock_order](BatchWorkspace& workspace) mutable {
            std::vector<Result<DecodeResult>>& results = state->results;
            try
            {
#if defined(_OPENMP)
                CpuOpenMpThreadLimitScope thread_limit;
                CpuThreadBudgetController::Lease extra_compute;
                const uint64_t work_units = static_cast<uint64_t>(requests.size()) * team_size;
                thread_limit.set(prepare_staging_team(work_units, static_cast<uint32_t>(requests.size()), extra_compute));
#endif
                auto ret = decode(lock_order, requests, workspace);
                staged_decode_batches.fetch_add(1, std::memory_order_relaxed);
                if (ret)
                {
                    results.reserve(ret.value().size());
                    for (DecodeResult& result : ret.value())
                        results.emplace_back(std::move(result));
                }
                else
                {
                    results.assign(requests.size(), ret.error());
                }
            }
            catch (const std::exception& error)
            {
                results.assign(requests.size(), Error{ErrorCode::InternalError, std::string("staged decode worker failed: ") + error.what()});
            }
            catch (...)
            {
                results.assign(requests.size(), Error{ErrorCode::InternalError, "staged decode worker failed"});
            }
            release_sessions(lock_order);
            state->promise.set_value(std::move(results));
        };
        if (try_enqueue_batch(lock_order, std::move(work)))
            return future;
    }

    // Only independent workers need a result slot for each request.
    state->results.assign(requests.size(), Error{});
    state->remaining.store(requests.size(), std::memory_order_relaxed);
    auto complete = [state](size_t index, Result<DecodeResult> result) {
        state->results[index] = std::move(result);
        if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
            state->promise.set_value(std::move(state->results));
    };
    for (size_t i = 0; i < requests.size(); i++)
    {
        DecodeBatchRequest request = std::move(requests[i]);
        Session* session = request.session.get();
        enqueue_session(session, [this, session, request = std::move(request), i, complete](BatchWorkspace&) mutable {
            Result<DecodeResult> result = Error{};
            try
            {
                result = request.session->decode(request.input_id);
            }
            catch (const std::exception& error)
            {
                result = Error{ErrorCode::InternalError, std::string("decode worker failed: ") + error.what()};
            }
            catch (...)
            {
                result = Error{ErrorCode::InternalError, "decode worker failed"};
            }
            release_sessions({&session, 1});
            complete(i, std::move(result));
        });
    }
    return future;
}

SchedulerStatistics BatchSchedulerPrivate::statistics() const noexcept
{
    SchedulerStatistics result;
    result.prefill_batches = prefill_batches.load(std::memory_order_relaxed);
    result.decode_batches = decode_batches.load(std::memory_order_relaxed);
    result.staged_prefill_batches = staged_prefill_batches.load(std::memory_order_relaxed);
    result.staged_decode_batches = staged_decode_batches.load(std::memory_order_relaxed);
    result.num_threads = static_cast<uint32_t>(workers.size());
    return result;
}

#if defined(_OPENMP)
uint32_t BatchSchedulerPrivate::prepare_staging_team(uint64_t work_units,
                                                     uint32_t independent_work_items,
                                                     CpuThreadBudgetController::Lease& extra_compute)
{
    const uint32_t current_team_size = cpu_openmp_thread_limit();
    const uint32_t available_for_team = std::max(1u,
                                                 std::min(cpu_budget.budget().max_threads,
                                                          current_team_size + cpu_budget.available()));
    const uint32_t desired_team_size = choose_cpu_team_size(work_units,
                                                            independent_work_items,
                                                            team_size,
                                                            available_for_team);
    if (desired_team_size > current_team_size)
    {
        extra_compute = cpu_budget.try_acquire_compute(desired_team_size - current_team_size,
                                                       true);
    }
    return std::max(1u,
                    std::min(desired_team_size,
                             current_team_size + extra_compute.size()));
}
#endif // defined(_OPENMP)

void BatchSchedulerPrivate::enqueue_session(Session* session, Work work)
{
    {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        auto pending = session_queues.try_emplace(session);
        try
        {
            if (pending.second)
                queue.push_back(std::move(work));
            else
                pending.first->second.push_back(std::move(work));
        }
        catch (...)
        {
            if (pending.second)
                session_queues.erase(pending.first);
            throw;
        }
    }
    queue_ready.notify_one();
}

bool BatchSchedulerPrivate::try_enqueue_batch(const std::vector<Session*>& sessions, Work work)
{
    {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        for (Session* session : sessions)
        {
            if (session_queues.find(session) != session_queues.end())
            {
                return false;
            }
        }
        size_t reserved = 0;
        try
        {
            for (Session* session : sessions)
            {
                session_queues.try_emplace(session);
                ++reserved;
            }
            queue.push_back(std::move(work));
        }
        catch (...)
        {
            for (size_t i = 0; i < reserved; i++)
                session_queues.erase(sessions[i]);
            throw;
        }
    }
    queue_ready.notify_one();
    return true;
}

void BatchSchedulerPrivate::release_sessions(std::span<Session* const> sessions)
{
    {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        for (Session* session : sessions)
        {
            auto pending = session_queues.find(session);
            if (pending == session_queues.end())
                continue;
            if (pending->second.empty())
            {
                session_queues.erase(pending);
            }
            else
            {
                // Reuse the pending node so releasing a batch cannot allocate.
                queue.splice(queue.end(), pending->second, pending->second.begin());
            }
        }
    }
    queue_ready.notify_all();
}

void BatchSchedulerPrivate::worker_loop()
{
    CpuOpenMpThreadLimitScope thread_limit;
    BatchWorkspace workspace;
    for (;;)
    {
        Work work;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_ready.wait(lock, [this] { return stopping || !queue.empty(); });
            if (stopping && queue.empty())
                return;
            work = std::move(queue.front());
            queue.pop_front();
        }
        CpuThreadBudgetController::Lease compute_lease = cpu_budget.acquire_compute(team_size,
                                                                                    false);
        thread_limit.set(std::max(1u, compute_lease.size()));
        work(workspace);
    }
}

BatchScheduler::BatchScheduler(const SchedulerOptions& opt)
    : d(new BatchSchedulerPrivate(opt))
{
}

BatchScheduler::~BatchScheduler() = default;

std::future<std::vector<Result<PrefillResult>>> BatchScheduler::submit_prefill(std::vector<PrefillBatchRequest> requests)
{
    return d->submit_prefill(std::move(requests));
}

std::future<std::vector<Result<DecodeResult>>> BatchScheduler::submit_decode(std::vector<DecodeBatchRequest> requests)
{
    return d->submit_decode(std::move(requests));
}

SchedulerStatistics BatchScheduler::statistics() const noexcept
{
    return d->statistics();
}

} // namespace moe
} // namespace ncnn
