#ifndef NCNN_MOE_SCHEDULER_H
#define NCNN_MOE_SCHEDULER_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/session.h"

#include <cstdint>
#include <future>
#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

struct DecodeBatchRequest
{
    SessionPtr session;
    int32_t input_id = -1;
};

struct SchedulerOptions
{
    // Zero selects a small allowed-CPU-derived pool intended to overlap GPU
    // and CPU phases without multiplying every OpenMP team. On Linux with
    // automatic NUMA affinity, it also creates at least one worker per node.
    uint32_t worker_count = 0;
    // Zero divides the allowed CPUs across scheduler workers. This bounds
    // OpenMP expert teams so cross-session concurrency does not oversubscribe
    // the machine.
    uint32_t expert_threads_per_worker = 0;
    // Linux workers can be pinned to disjoint allowed-CPU partitions. When
    // sysfs exposes NUMA nodes, automatic partitions stay within a node.
    bool pin_workers = false;
    // Optional Linux CPU sets, one set per scheduler worker. This permits
    // callers to align workers with known NUMA-node CPU lists.
    std::vector<std::vector<uint32_t> > worker_cpu_sets;
};

struct SchedulerStatistics
{
    uint64_t submitted_batches = 0;
    uint64_t submitted_requests = 0;
    uint64_t completed_requests = 0;
    uint64_t rejected_requests = 0;
    uint64_t max_batch_size = 0;
    uint64_t max_in_flight = 0;
    uint64_t serialized_session_requests = 0;
    uint64_t affinity_workers_configured = 0;
    uint64_t affinity_failures = 0;
    uint32_t affinity_cpu_count = 0;
    uint32_t numa_nodes_detected = 0;
    bool automatic_topology_affinity = false;
    uint32_t worker_count = 0;
    uint32_t expert_threads_per_worker = 0;
};

class BatchScheduler
{
public:
    explicit BatchScheduler(const SchedulerOptions& options = {});
    ~BatchScheduler();

    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;

    [[nodiscard]] std::future<std::vector<Result<DecodeResult> > > submit_decode(
        std::vector<DecodeBatchRequest> requests);
    [[nodiscard]] SchedulerStatistics statistics() const noexcept;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

using BatchSchedulerPtr = std::shared_ptr<BatchScheduler>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SCHEDULER_H
