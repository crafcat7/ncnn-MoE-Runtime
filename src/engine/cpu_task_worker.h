#ifndef NCNN_MOE_CPU_TASK_WORKER_H
#define NCNN_MOE_CPU_TASK_WORKER_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace ncnn {
namespace moe {

class CpuTaskWorker
{
public:
    explicit CpuTaskWorker(size_t maximum_outstanding_tasks);
    ~CpuTaskWorker();

    CpuTaskWorker(const CpuTaskWorker&) = delete;
    CpuTaskWorker& operator=(const CpuTaskWorker&) = delete;

    [[nodiscard]] bool try_submit(std::function<void()> task);
    void wait_idle();

    [[nodiscard]] uint64_t completed_tasks() const;

private:
    void worker_loop();

    const size_t maximum_outstanding_tasks_;
    mutable std::mutex mutex_;
    std::condition_variable task_ready_;
    std::condition_variable idle_;
    std::deque<std::function<void()>> tasks_;
    std::thread worker_;
    size_t outstanding_tasks_ = 0;
    uint64_t completed_tasks_ = 0;
    bool stop_ = false;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_TASK_WORKER_H
