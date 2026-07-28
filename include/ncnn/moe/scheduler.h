#ifndef NCNN_MOE_SCHEDULER_H
#define NCNN_MOE_SCHEDULER_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/session.h"
#include "ncnn/moe/types.h"

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

struct PrefillBatchRequest
{
    SessionPtr session;
    std::vector<int32_t> input_ids;
};

#define NCNN_MOE_SCHEDULER_PIN_WORKER_BIT         0
#define NCNN_MOE_SCHEDULER_DISABLE_STAGED_BIT     1
#define NCNN_MOE_SCHEDULER_FORCE_STAGED_BIT       2
#define NCNN_MOE_SCHEDULER_DISABLE_CROSS_CALL_BIT 3

enum SchedulerOptionFlag : uint32_t
{
    SchedulerOptionPinWorkers = UINT32_C(1) << NCNN_MOE_SCHEDULER_PIN_WORKER_BIT,
    SchedulerOptionDisableStagedBatching = UINT32_C(1) << NCNN_MOE_SCHEDULER_DISABLE_STAGED_BIT,
    SchedulerOptionForceStagedBatching = UINT32_C(1) << NCNN_MOE_SCHEDULER_FORCE_STAGED_BIT,
    SchedulerOptionDisableCrossCallBatching = UINT32_C(1) << NCNN_MOE_SCHEDULER_DISABLE_CROSS_CALL_BIT
};

struct SchedulerOptions
{
    uint32_t worker_count = 0;
    uint32_t expert_threads_per_worker = 0;
    // Probe interval for each request-count and context bucket.
    uint32_t adaptive_probe_interval = 32;
    // Zero disables cross-call collection.
    uint32_t cross_call_window_microseconds = 200;
    uint32_t cross_call_max_batch_size = 0;
    uint32_t flags = 0;
    std::vector<std::vector<uint32_t>> worker_cpu_sets;
};

struct SchedulerStatistics
{
    uint64_t submitted_prefill_batches = 0;
    uint64_t submitted_prefill_requests = 0;
    uint64_t completed_prefill_requests = 0;
    uint64_t staged_prefill_batches = 0;
    uint64_t staged_prefill_requests = 0;
    uint64_t submitted_batches = 0;
    uint64_t submitted_requests = 0;
    uint64_t completed_requests = 0;
    uint64_t rejected_requests = 0;
    uint64_t max_batch_size = 0;
    uint64_t max_in_flight = 0;
    uint64_t serialized_session_requests = 0;
    uint64_t staged_batches = 0;
    uint64_t staged_requests = 0;
    uint64_t staging_bypassed_batches = 0;
    uint64_t logical_expert_batches = 0;
    uint64_t physical_expert_batches = 0;
    uint64_t coalesced_expert_batches = 0;
    uint64_t coalesced_expert_routes = 0;
    uint64_t max_coalesced_expert_batch_size = 0;
    uint64_t adaptive_staged_decisions = 0;
    uint64_t adaptive_independent_decisions = 0;
    uint64_t adaptive_probe_decisions = 0;
    uint64_t adaptive_policy_switches = 0;
    uint64_t adaptive_staged_observations = 0;
    uint64_t adaptive_independent_observations = 0;
    uint64_t adaptive_staged_time_microseconds = 0;
    uint64_t adaptive_independent_time_microseconds = 0;
    uint64_t adaptive_resident_decisions = 0;
    uint64_t adaptive_mixed_decisions = 0;
    uint64_t adaptive_storage_decisions = 0;
    uint64_t adaptive_resident_observations = 0;
    uint64_t adaptive_mixed_observations = 0;
    uint64_t adaptive_storage_observations = 0;
    uint64_t adaptive_phase_changes = 0;
    uint64_t adaptive_noisy_switch_rejections = 0;
    uint64_t cross_call_collected_batches = 0;
    uint64_t cross_call_collected_requests = 0;
    uint64_t cross_call_collection_probes = 0;
    uint64_t cross_call_collection_timeouts = 0;
    uint64_t cross_call_collection_bypasses = 0;
    uint64_t cross_call_collection_wait_microseconds = 0;
    uint64_t max_cross_call_batch_size = 0;
    uint64_t max_cross_call_pending = 0;
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
private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;

public:
    explicit BatchScheduler(const SchedulerOptions& options = {});
    ~BatchScheduler();

    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;

    [[nodiscard]] std::future<std::vector<Result<PrefillResult>>> submit_prefill(std::vector<PrefillBatchRequest> requests);
    [[nodiscard]] std::future<std::vector<Result<DecodeResult>>> submit_decode(std::vector<DecodeBatchRequest> requests);
    [[nodiscard]] SchedulerStatistics statistics() const noexcept;
};

using BatchSchedulerPtr = std::shared_ptr<BatchScheduler>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SCHEDULER_H
