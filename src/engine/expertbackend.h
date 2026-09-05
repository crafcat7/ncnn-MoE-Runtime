#ifndef NCNN_MOE_EXPERTBACKEND_H
#define NCNN_MOE_EXPERTBACKEND_H

#include "kernels/activation.h"

#include "graph/router.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"
#include "ncnn/moe/option.h"

#include <cstdint>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

class ExpertVictimCache;

struct ExpertKeyHash
{
    using is_transparent = void;

    [[nodiscard]] size_t operator()(std::string_view value) const noexcept
    {
        return std::hash<std::string_view>{}(value);
    }
};

enum class ExpertBackendExecutionResult
{
    NotResident,
    Executed,
    Failed
};

inline constexpr size_t vulkan_expert_gpu_min_rows = 2;

// Admission is asynchronous; CPU remains available while weights upload.
inline constexpr size_t vulkan_expert_gpu_admission_min_rows = 2;

inline constexpr size_t vulkan_expert_gpu_victim_min_rows = 32;

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
    uint64_t bytes_uploaded = 0;
    uint64_t resident_size = 0;
    uint64_t pending_size = 0;
    uint64_t execution_time_microseconds = 0;
    uint64_t arc_recent_size = 0;
    uint64_t arc_frequent_size = 0;
    uint64_t arc_recent_target_size = 0;
    uint64_t arc_recent_ghost_size = 0;
    uint64_t arc_frequent_ghost_size = 0;
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
    uint64_t cache_size = 0;
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
        // Require a complete batch before publishing an aggregate result.
        bool require_all_requests = false;
    };

    std::string_view key;
    const ActivationBuffer* input = nullptr;
    ActivationBuffer* output = nullptr;
    uint64_t weight_size = 0;
    RouteAggregation route_aggregation;
};

class ExpertSubmission
{
public:
    virtual ~ExpertSubmission() = default;

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

class ExpertBackend
{
public:
    virtual ~ExpertBackend() = default;

    // Weight ownership is retained until asynchronous admission completes.
    virtual void admit(
        std::string key,
        std::shared_ptr<const TensorData> gate_up,
        const TensorData* gate_up_bias,
        std::shared_ptr<const TensorData> down,
        const TensorData* down_bias,
        uint32_t residency_group,
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

    [[nodiscard]] virtual std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) = 0;

    virtual void observe_cpu(uint32_t token_count, uint64_t weight_size, uint64_t elapsed_microseconds) = 0;

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
    [[nodiscard]] virtual uint64_t capacity() const noexcept = 0;
};

class MultiDeviceExpertBackend final : public ExpertBackend
{
public:
    MultiDeviceExpertBackend(std::vector<std::shared_ptr<ExpertBackend>> _backends, std::vector<uint32_t> _device_indices, std::vector<uint32_t> _residency_group_devices, bool _key_sharded);

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down, const TensorData* down_bias, uint32_t residency_group,
               float activation_limit, ExpertActivation activation) override;

    ExpertBackendExecutionResult try_execute(const std::string& key, const ActivationBuffer& input, ActivationBuffer& output) override;

    std::vector<ExpertBackendExecutionResult> try_execute_batch(std::span<const ExpertBackendRequest> requests) override;

    std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override;

    void observe_cpu(uint32_t token_count, uint64_t weight_size, uint64_t elapsed_microseconds) override;

    void observe_phase(uint32_t token_count, uint64_t total_weight_bytes, uint64_t accelerated_weight_bytes, uint64_t elapsed_microseconds) override;

    void set_foreground_active(bool active) noexcept override;

    void wait_for_background_work() override;

    ExpertBackendStatistics statistics() const override;

    std::vector<ExpertBackendDeviceStatistics> device_statistics() const override;

    uint64_t capacity() const noexcept override;

private:
    size_t backend_for_key(std::string_view key) const;

    void publish_accelerated_bytes(std::vector<uint64_t> values);

    struct ChildSubmission
    {
        size_t backend_index = 0;
        std::vector<size_t> request_indices;
        std::vector<ExpertBackendRequest> requests;
        std::unique_ptr<ExpertSubmission> submission;
        bool reservation_shape_valid = true;
    };

    class Submission final : public ExpertSubmission
    {
    public:
        Submission(MultiDeviceExpertBackend* _owner, std::span<const ExpertBackendRequest> requests, std::vector<std::vector<size_t>> request_indices);

        ~Submission() override;

        std::span<const ExpertBackendExecutionResult> reservations() const noexcept override;

        std::vector<ExpertBackendExecutionResult> wait() override;

        bool commit() override;

        void abort() noexcept override;

    private:
        MultiDeviceExpertBackend* owner;
        std::vector<ExpertBackendRequest> client_requests;
        std::vector<ActivationBuffer> private_outputs;
        std::vector<ChildSubmission> children;
        std::vector<ExpertBackendExecutionResult> planned;
        std::vector<ExpertBackendExecutionResult> final;
        std::vector<uint64_t> accelerated_bytes;
        bool waited = false;
        bool committed = false;
        bool aborted = false;
    };

    size_t backend_for_group(uint32_t residency_group) const;

    size_t fallback_backend(std::string_view key) const;

    std::vector<std::shared_ptr<ExpertBackend>> backends;
    std::vector<uint32_t> device_indices;
    std::vector<uint32_t> residency_group_devices;
    std::unordered_map<uint32_t, size_t> device_to_backend;
    mutable std::mutex placement_mutex;
    std::unordered_map<std::string, size_t, ExpertKeyHash, std::equal_to<>> key_placements;
    bool key_sharded = false;
    mutable std::mutex phase_mutex;
    std::deque<std::vector<uint64_t>> pending_accelerated_bytes;
};

class ScopedExpertBackendForeground
{
public:
    explicit ScopedExpertBackendForeground(
        const std::shared_ptr<ExpertBackend>& _backend) noexcept;
    ~ScopedExpertBackendForeground();

    ScopedExpertBackendForeground(const ScopedExpertBackendForeground&) = delete;
    ScopedExpertBackendForeground& operator=(const ScopedExpertBackendForeground&) = delete;

private:
    std::shared_ptr<ExpertBackend> backend;
};

[[nodiscard]] std::shared_ptr<ExpertBackend> create_multi_device_expert_backend(std::vector<std::shared_ptr<ExpertBackend>> backends, std::vector<uint32_t> device_indices, std::vector<uint32_t> residency_group_devices);

[[nodiscard]] std::shared_ptr<ExpertBackend> create_key_sharded_expert_backend(std::vector<std::shared_ptr<ExpertBackend>> backends, std::vector<uint32_t> device_indices);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERTBACKEND_H
