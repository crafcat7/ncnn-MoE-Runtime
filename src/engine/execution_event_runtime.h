#ifndef NCNN_MOE_EXECUTION_EVENT_RUNTIME_H
#define NCNN_MOE_EXECUTION_EVENT_RUNTIME_H

#include "ncnn/moe/execution_graph.h"

#include <condition_variable>
#include <mutex>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

// Per-execution event state.  The graph contains the static dependency
// metadata; this object owns the runtime completion state for one execution
// and therefore cannot leak signals between sessions or model instances.
class ExecutionEventRuntime
{
public:
    explicit ExecutionEventRuntime(std::span<const ExecutionEvent> events)
        : signaled_(events.size(), false)
    {
    }

    [[nodiscard]] Result<void> wait(std::span<const ExecutionEventId> events)
    {
        if (events.empty())
            return {};

        std::unique_lock<std::mutex> lock(mutex_);
        for (ExecutionEventId event : events)
        {
            if (event >= signaled_.size())
                return Error{ErrorCode::InternalError, "execution node waits on an invalid runtime event"};
        }
        condition_.wait(lock, [this, events] {
            for (ExecutionEventId event : events)
            {
                if (!signaled_[event])
                    return false;
            }
            return true;
        });
        return {};
    }

    void signal(ExecutionEventId event) noexcept
    {
        if (event == invalid_execution_event_id)
            return;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (event >= signaled_.size())
                return;
            signaled_[event] = true;
        }
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<bool> signaled_;
};

class ExecutionNodeEventGuard
{
public:
    ExecutionNodeEventGuard(ExecutionEventRuntime& runtime, ExecutionEventId event) noexcept
        : runtime_(&runtime), event_(event)
    {
    }

    ~ExecutionNodeEventGuard()
    {
        runtime_->signal(event_);
    }

    ExecutionNodeEventGuard(const ExecutionNodeEventGuard&) = delete;
    ExecutionNodeEventGuard& operator=(const ExecutionNodeEventGuard&) = delete;

private:
    ExecutionEventRuntime* runtime_;
    ExecutionEventId event_;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXECUTION_EVENT_RUNTIME_H
