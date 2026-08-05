#ifndef NCNN_MOE_EXPERT_BACKEND_H
#define NCNN_MOE_EXPERT_BACKEND_H

#include "kernels/cpu_batch.h"

#include "ncnn/moe/expert_dispatcher.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"
#include "ncnn/moe/runtime_config.h"
#include "ncnn/moe/vulkan_context.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ncnn {
namespace moe {

class IExpertVictimCache;

enum class ExpertBackendExecutionResult
{
    NotResident,
    PreferCpu,
    Executed,
    Failed
};

struct ExpertBackendStatistics
{
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t stores = 0;
    uint64_t evictions = 0;
    uint64_t dropped_admissions = 0;
    uint64_t executions = 0;
    uint64_t execution_failures = 0;
    uint64_t cpu_preferred = 0;
    uint64_t bytes_uploaded = 0;
    uint64_t resident_bytes = 0;
    uint64_t pending_bytes = 0;
    uint64_t execution_time_microseconds = 0;
    uint64_t arc_recent_bytes = 0;
    uint64_t arc_frequent_bytes = 0;
    uint64_t arc_recent_target_bytes = 0;
    uint64_t arc_recent_ghost_bytes = 0;
    uint64_t arc_frequent_ghost_bytes = 0;
    uint64_t device_source_hits = 0;
    uint64_t device_source_misses = 0;
    uint64_t device_source_executions = 0;
    uint64_t device_source_execution_failures = 0;
    uint64_t route_aggregation_batches = 0;
    uint64_t route_aggregation_routes = 0;
    uint64_t route_aggregation_bytes_saved = 0;
};

struct ExpertBackendDeviceStatistics
{
    uint32_t device_index = automatic_vulkan_device_index;
    uint64_t capacity_bytes = 0;
    ExpertBackendStatistics statistics;
};

struct ExpertBackendRequest
{
    struct RouteAggregation
    {
        // The backend may fill output only when completed is set to 1 after
        // the device-side reduction has completed successfully.  A wrapper
        // backend must clear this field unless it can preserve single-writer
        // semantics for the shared output.
        ActivationBuffer* output = nullptr;
        std::span<const ExpertRoute> routes;
        uint32_t token_count = 0;
        uint8_t* completed = nullptr;
    };

    std::string_view key;
    const ActivationBuffer* input = nullptr;
    ActivationBuffer* output = nullptr;
    uint64_t weight_bytes = 0;
    RouteAggregation route_aggregation;
};

class IExpertBackendBatchSubmission
{
public:
    virtual ~IExpertBackendBatchSubmission() = default;

    // The reservation span has exactly one entry per request and remains
    // stable until wait() returns. It is only a scheduling decision: backend
    // output is private to the submission until commit() succeeds.
    [[nodiscard]] virtual std::span<const ExpertBackendExecutionResult> reservations() const noexcept = 0;

    // wait() returns exactly one final result per reservation. A final
    // Executed result is valid only for a request reserved as Executed.
    [[nodiscard]] virtual std::vector<ExpertBackendExecutionResult> wait() = 0;

    // commit() is the sole publication point. It must publish all successful
    // outputs atomically from the caller's perspective; false leaves every
    // reserved request eligible for CPU fallback.
    [[nodiscard]] virtual bool commit() = 0;

    // abort() makes the submission non-publishable and is idempotent.
    virtual void abort() noexcept = 0;
};

class IExpertExecutionBackend
{
public:
    virtual ~IExpertExecutionBackend() = default;

    // Weight ownership is retained until asynchronous admission completes.
    virtual void admit(
        std::string key,
        std::shared_ptr<const TensorData> gate_up,
        const TensorData* gate_up_bias,
        std::shared_ptr<const TensorData> down,
        const TensorData* down_bias,
        uint32_t residency_group,
        uint32_t token_count,
        float activation_limit,
        ExpertActivation activation = ExpertActivation::GptOssSwiGlu) = 0;

    [[nodiscard]] virtual ExpertBackendExecutionResult try_execute(const std::string& key, const ActivationBuffer& input, ActivationBuffer& output) = 0;

    [[nodiscard]] virtual std::vector<ExpertBackendExecutionResult> try_execute_batch(std::span<const ExpertBackendRequest> requests)
    {
        std::vector<ExpertBackendExecutionResult> results;
        results.reserve(requests.size());
        for (const ExpertBackendRequest& request : requests)
        {
            if (!request.input || !request.output)
            {
                results.push_back(ExpertBackendExecutionResult ::Failed);
                continue;
            }
            results.push_back(try_execute(std::string(request.key), *request.input, *request.output));
        }
        return results;
    }

    [[nodiscard]] virtual std::unique_ptr<IExpertBackendBatchSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) = 0;

    virtual void observe_cpu(uint32_t token_count, uint64_t weight_bytes, uint64_t elapsed_microseconds) = 0;

    // A zero accelerated byte count records the CPU counterfactual.
    virtual void observe_phase(uint32_t token_count, uint64_t total_weight_bytes, uint64_t accelerated_weight_bytes, uint64_t elapsed_microseconds) = 0;

    // Suspend background device-weight admission while the foreground
    // executor owns the Vulkan submission path. The cache may continue to
    // reserve requests; concrete backends decide when those uploads resume.
    virtual void set_foreground_active(bool active) noexcept
    {
        (void)active;
    }

    virtual void wait_for_background_work() = 0;

    [[nodiscard]] virtual ExpertBackendStatistics statistics() const = 0;
    [[nodiscard]] virtual std::vector<ExpertBackendDeviceStatistics> device_statistics() const
    {
        return {};
    }
    [[nodiscard]] virtual uint64_t capacity_bytes() const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<IExpertExecutionBackend> create_vulkan_mxfp4_expert_backend(
    uint64_t capacity_bytes,
    uint32_t vulkan_device_index,
    std::shared_ptr<IExpertVictimCache> device_weight_source,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags);

[[nodiscard]] std::shared_ptr<IExpertExecutionBackend> create_multi_device_expert_backend(std::vector<std::shared_ptr<IExpertExecutionBackend>> backends, std::vector<uint32_t> device_indices, std::vector<uint32_t> residency_group_devices);

[[nodiscard]] std::shared_ptr<IExpertExecutionBackend> create_key_sharded_expert_backend(std::vector<std::shared_ptr<IExpertExecutionBackend>> backends, std::vector<uint32_t> device_indices);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_BACKEND_H
