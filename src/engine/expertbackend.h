#ifndef NCNN_MOE_EXPERTBACKEND_H
#define NCNN_MOE_EXPERTBACKEND_H

#include "kernels/activationbuffer.h"

#include "graph/router.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"
#include "ncnn/moe/option.h"

#include <cstdint>
#include <cstddef>
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
class ExpertResidencyCoordinator;
class DeviceTensor_vulkan;

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
    // Optional completed normalized batch. Host input may be absent until
    // the caller actually needs CPU fallback.
    std::shared_ptr<const DeviceTensor_vulkan> device_input;
    std::span<const ExpertRoute> device_routes;
    // Optional completed FP32 output, published only by successful commit().
    // Unsupported backends return a host output and publish an empty pointer.
    std::shared_ptr<DeviceTensor_vulkan>* device_output = nullptr;
    // Explicit logical shape permits device execution without allocating a
    // gathered host tensor. Legacy callers can continue providing input only.
    size_t input_rows = 0;
    uint32_t input_columns = 0;

    [[nodiscard]] size_t rows() const noexcept
    {
        return input_rows != 0 ? input_rows : input ? input->rows()
                                                    : 0;
    }

    [[nodiscard]] uint32_t columns() const noexcept
    {
        return input_columns != 0 ? input_columns : input ? input->columns()
                                                          : 0;
    }

    [[nodiscard]] bool has_host_input() const noexcept
    {
        return input && input->dtype() == DType::Float32
               && input->rows() == rows() && input->columns() == columns()
               && rows() != 0 && columns() != 0
               && input->bytes().size() == input->rows() * static_cast<size_t>(input->columns()) * sizeof(float);
    }
};

struct ExpertDemandRequest
{
    std::string_view key;
    std::shared_ptr<const TensorData> gate_up;
    const TensorData* gate_up_bias = nullptr;
    std::shared_ptr<const TensorData> down;
    const TensorData* down_bias = nullptr;
    uint32_t residency_group = 0;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
};

class ExpertDemandSubmission
{
public:
    virtual ~ExpertDemandSubmission() = default;

    // The pin span must match the submitted request count. Return a prepared
    // contiguous prefix; every successful pin is nonempty and the tail empty.
    // wait() is the sole publication point and is safe to repeat. Async
    // implementations retain request weight/bias ownership until completion.
    [[nodiscard]] virtual size_t wait(std::span<std::shared_ptr<const void>> pins) = 0;

    // Idempotent cancellation prevents later pin publication. The submission
    // must drain any in-flight upload before releasing its request ownership.
    virtual void abort() noexcept = 0;
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

    // Replace the caller's result array. Backends can preserve its capacity;
    // the default keeps existing by-value implementations compatible.
    virtual void wait(std::vector<ExpertBackendExecutionResult>& results)
    {
        results = wait();
    }

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
    virtual void admit(std::string key,
                       std::shared_ptr<const TensorData> gate_up,
                       const TensorData* gate_up_bias,
                       std::shared_ptr<const TensorData> down,
                       const TensorData* down_bias,
                       uint32_t residency_group,
                       float activation_limit,
                       ExpertActivation activation = ExpertActivation::GptOssSwiGlu)
        = 0;

    // Complete a bounded current-wave upload. On success, pin keeps the
    // resident pair alive until submit_batch captures its selected entries.
    // Unsupported or exhausted backends leave pin empty for CPU fallback.
    [[nodiscard]] virtual bool prepare_demand(std::string key,
                                              std::shared_ptr<const TensorData> gate_up,
                                              const TensorData* gate_up_bias,
                                              std::shared_ptr<const TensorData> down,
                                              const TensorData* down_bias,
                                              uint32_t residency_group,
                                              float activation_limit,
                                              ExpertActivation activation,
                                              std::shared_ptr<const void>& pin)
    {
        (void)key;
        (void)gate_up;
        (void)gate_up_bias;
        (void)down;
        (void)down_bias;
        (void)residency_group;
        (void)activation_limit;
        (void)activation;
        pin.reset();
        return false;
    }

    // Return a prepared contiguous prefix. Each successful request owns a
    // nonempty pin; every unprepared pin is empty. Capacity-limited callers
    // execute and release their current wave before retrying the remainder.
    // A zero prefix leaves the first request eligible for CPU fallback.
    [[nodiscard]] virtual size_t prepare_demand_batch(std::span<const ExpertDemandRequest> requests,
                                                      std::span<std::shared_ptr<const void>> pins);

    // Begin one bounded upload wave without requiring the caller to wait for
    // it before submitting independent resident work. The default preserves
    // synchronous backends; asynchronous backends copy the request ownership.
    [[nodiscard]] virtual std::unique_ptr<ExpertDemandSubmission> begin_demand_batch(std::span<const ExpertDemandRequest> requests);

    virtual void set_residency_coordinator(std::shared_ptr<ExpertResidencyCoordinator> coordinator)
    {
        (void)coordinator;
    }

    [[nodiscard]] virtual std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) = 0;

    // Suspend background device-weight admission while the foreground
    // executor owns the Vulkan submission path. The cache may continue to
    // reserve requests; concrete backends decide when those uploads resume.
    virtual void set_foreground_active(bool active) noexcept
    {
        (void)active;
    }

    virtual void wait_for_background_work() = 0;

    [[nodiscard]] virtual ExpertBackendStatistics statistics() const = 0;
};

class MultiDeviceExpertBackend final : public ExpertBackend
{
public:
    MultiDeviceExpertBackend(std::vector<std::shared_ptr<ExpertBackend>> _backends, std::vector<uint32_t> device_indices, std::vector<uint32_t> _residency_group_devices, bool _key_sharded);

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down, const TensorData* down_bias, uint32_t residency_group,
               float activation_limit, ExpertActivation activation) override;

    bool prepare_demand(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias,
                        std::shared_ptr<const TensorData> down, const TensorData* down_bias, uint32_t residency_group,
                        float activation_limit, ExpertActivation activation, std::shared_ptr<const void>& pin) override;

    size_t prepare_demand_batch(std::span<const ExpertDemandRequest> requests,
                                std::span<std::shared_ptr<const void>> pins) override;

    std::unique_ptr<ExpertDemandSubmission> begin_demand_batch(std::span<const ExpertDemandRequest> requests) override;

    void set_residency_coordinator(std::shared_ptr<ExpertResidencyCoordinator> coordinator) override;

    std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override;

    void set_foreground_active(bool active) noexcept override;

    void wait_for_background_work() override;

    ExpertBackendStatistics statistics() const override;

private:
    struct ChildSubmission
    {
        std::vector<size_t> request_indices;
        std::vector<ExpertBackendRequest> requests;
        std::unique_ptr<ExpertSubmission> submission;
        bool reservation_shape_valid = true;
    };

    class Submission final : public ExpertSubmission
    {
    public:
        Submission(MultiDeviceExpertBackend* owner, std::span<const ExpertBackendRequest> requests, std::vector<std::vector<size_t>> request_indices);

        ~Submission() override;

        std::span<const ExpertBackendExecutionResult> reservations() const noexcept override;

        std::vector<ExpertBackendExecutionResult> wait() override;

        void wait(std::vector<ExpertBackendExecutionResult>& results) override;

        bool commit() override;

        void abort() noexcept override;

    private:
        std::vector<ExpertBackendRequest> client_requests;
        std::vector<ActivationBuffer> private_outputs;
        std::vector<ChildSubmission> children;
        std::vector<ExpertBackendExecutionResult> planned;
        std::vector<ExpertBackendExecutionResult> final;
        bool waited = false;
        bool committed = false;
        bool aborted = false;
    };

    size_t backend_for_group(uint32_t residency_group) const;

    size_t fallback_backend(std::string_view key) const;

    std::vector<std::shared_ptr<ExpertBackend>> backends;
    std::vector<uint32_t> residency_group_devices;
    std::unordered_map<uint32_t, size_t> device_to_backend;
    mutable std::mutex placement_mutex;
    std::unordered_map<std::string, size_t, ExpertKeyHash, std::equal_to<>> key_placements;
    bool key_sharded = false;
};

class ScopedExpertBackendForeground
{
public:
    explicit ScopedExpertBackendForeground(const std::shared_ptr<ExpertBackend>& _backend) noexcept;
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
