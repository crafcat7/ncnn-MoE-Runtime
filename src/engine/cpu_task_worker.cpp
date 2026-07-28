#include "cpu_task_worker.h"

#include <algorithm>
#include <utility>

namespace ncnn {
namespace moe {

CpuTaskWorker::CpuTaskWorker(size_t maximum_outstanding_tasks)
    : maximum_outstanding_tasks_(
          std::max<size_t>(1, maximum_outstanding_tasks)),
      worker_(&CpuTaskWorker::worker_loop, this)
{
}

CpuTaskWorker::~CpuTaskWorker()
{
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    task_ready_.notify_all();
    if (worker_.joinable())
        worker_.join();
}

bool CpuTaskWorker::try_submit(std::function<void()> task)
{
    if (!task)
        return false;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (stop_
            || outstanding_tasks_ >= maximum_outstanding_tasks_)
        {
            return false;
        }
        tasks_.push_back(std::move(task));
        ++outstanding_tasks_;
    }
    task_ready_.notify_one();
    return true;
}

void CpuTaskWorker::wait_idle()
{
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this] {
        return outstanding_tasks_ == 0;
    });
}

uint64_t CpuTaskWorker::completed_tasks() const
{
    const std::lock_guard<std::mutex> lock(mutex_);
    return completed_tasks_;
}

void CpuTaskWorker::worker_loop()
{
    for (;;)
    {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_ready_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });
            if (stop_ && tasks_.empty())
                return;
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        try
        {
            task();
        }
        catch (...)
        {
        }
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            --outstanding_tasks_;
            ++completed_tasks_;
            if (outstanding_tasks_ == 0)
                idle_.notify_all();
        }
    }
}

} // namespace moe
} // namespace ncnn
