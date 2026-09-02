#include "expert_backend.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace ncnn {
namespace moe {

struct ExpertBackendTransparentStringHash
{
    using is_transparent = void;

    [[nodiscard]] size_t operator()(std::string_view value) const noexcept
    {
        return std::hash<std::string_view>{}(value);
    }
};

static void add_statistics(ExpertBackendStatistics& destination, const ExpertBackendStatistics& source)
{
#define NCNN_MOE_ADD_EXPERT_STATISTIC(field) destination.field += source.field
    NCNN_MOE_ADD_EXPERT_STATISTIC(hits);
    NCNN_MOE_ADD_EXPERT_STATISTIC(misses);
    NCNN_MOE_ADD_EXPERT_STATISTIC(admissions);
    NCNN_MOE_ADD_EXPERT_STATISTIC(stores);
    NCNN_MOE_ADD_EXPERT_STATISTIC(evictions);
    NCNN_MOE_ADD_EXPERT_STATISTIC(dropped_admissions);
    NCNN_MOE_ADD_EXPERT_STATISTIC(executions);
    NCNN_MOE_ADD_EXPERT_STATISTIC(execution_failures);
    NCNN_MOE_ADD_EXPERT_STATISTIC(cpu_preferred);
    NCNN_MOE_ADD_EXPERT_STATISTIC(bytes_uploaded);
    NCNN_MOE_ADD_EXPERT_STATISTIC(resident_size);
    NCNN_MOE_ADD_EXPERT_STATISTIC(pending_size);
    NCNN_MOE_ADD_EXPERT_STATISTIC(execution_time_microseconds);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_recent_size);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_frequent_size);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_recent_target_size);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_recent_ghost_size);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_frequent_ghost_size);
    NCNN_MOE_ADD_EXPERT_STATISTIC(device_source_hits);
    NCNN_MOE_ADD_EXPERT_STATISTIC(device_source_misses);
    NCNN_MOE_ADD_EXPERT_STATISTIC(device_source_executions);
    NCNN_MOE_ADD_EXPERT_STATISTIC(device_source_execution_failures);
    NCNN_MOE_ADD_EXPERT_STATISTIC(route_aggregation_batches);
    NCNN_MOE_ADD_EXPERT_STATISTIC(route_aggregation_routes);
    NCNN_MOE_ADD_EXPERT_STATISTIC(route_aggregation_bytes_saved);
#undef NCNN_MOE_ADD_EXPERT_STATISTIC
}

class MultiDeviceExpertBackend final : public IExpertExecutionBackend
{
public:
    MultiDeviceExpertBackend(std::vector<std::shared_ptr<IExpertExecutionBackend>> _backends, std::vector<uint32_t> _device_indices, std::vector<uint32_t> _residency_group_devices, bool _key_sharded)
        : backends(std::move(_backends)),
          device_indices(std::move(_device_indices)),
          residency_group_devices(std::move(_residency_group_devices)),
          key_sharded(_key_sharded)
    {
        for (size_t index = 0; index < device_indices.size(); ++index)
        {
            device_to_backend.emplace(device_indices[index], index);
        }
    }

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down, const TensorData* down_bias, uint32_t residency_group, uint32_t token_count,
               float activation_limit, ExpertActivation activation) override
    {
        if (backends.empty() || key.empty())
            return;
        const size_t backend_index = key_sharded ? fallback_backend(key) : backend_for_group(residency_group);
        {
            const std::lock_guard<std::mutex> lock(placement_mutex);
            key_placements.insert_or_assign(key, backend_index);
        }
        backends[backend_index]->admit(
            std::move(key),
            std::move(gate_up),
            gate_up_bias,
            std::move(down),
            down_bias,
            residency_group,
            token_count,
            activation_limit,
            activation);
    }

    ExpertBackendExecutionResult try_execute(const std::string& key, const ActivationBuffer& input, ActivationBuffer& output) override
    {
        if (backends.empty())
            return ExpertBackendExecutionResult::Failed;
        return backends[backend_for_key(key)]->try_execute(key, input, output);
    }

    std::vector<ExpertBackendExecutionResult> try_execute_batch(std::span<const ExpertBackendRequest> requests) override
    {
        if (backends.size() == 1)
            return backends.front()->try_execute_batch(requests);
        auto submission = submit_batch(requests);
        if (!submission)
            return std::vector<ExpertBackendExecutionResult>(requests.size(), ExpertBackendExecutionResult::Failed);
        const std::span<const ExpertBackendExecutionResult> planned = submission->reservations();
        std::vector<ExpertBackendExecutionResult> results = submission->wait();
        if (planned.size() != requests.size() || results.size() != requests.size())
        {
            submission->abort();
            return std::vector<ExpertBackendExecutionResult>(requests.size(), ExpertBackendExecutionResult::Failed);
        }
        for (size_t index = 0; index < results.size(); ++index)
        {
            if (results[index] == ExpertBackendExecutionResult::Executed
                && planned[index] != ExpertBackendExecutionResult::Executed)
            {
                submission->abort();
                return std::vector<ExpertBackendExecutionResult>(requests.size(), ExpertBackendExecutionResult::Failed);
            }
        }
        bool has_executed = false;
        for (ExpertBackendExecutionResult result : results)
            has_executed = has_executed || result == ExpertBackendExecutionResult::Executed;
        if (has_executed)
        {
            if (!submission->commit())
            {
                for (ExpertBackendExecutionResult& result : results)
                {
                    if (result == ExpertBackendExecutionResult::Executed)
                        result = ExpertBackendExecutionResult::Failed;
                }
                submission->abort();
            }
        }
        else
            submission->abort();
        return results;
    }

    std::unique_ptr<IExpertBackendBatchSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override
    {
        std::vector<std::vector<size_t>> request_indices(backends.size());
        {
            const std::lock_guard<std::mutex> lock(placement_mutex);
            for (size_t request_index = 0; request_index < requests.size(); ++request_index)
            {
                const auto placed = key_placements.find(requests[request_index].key);
                const size_t backend_index = placed == key_placements.end() ? fallback_backend(requests[request_index].key) : placed->second;
                request_indices[backend_index].push_back(request_index);
            }
        }
        return std::make_unique<Submission>(this, requests, std::move(request_indices));
    }

    void observe_cpu(uint32_t token_count, uint64_t weight_size, uint64_t elapsed_microseconds) override
    {
        for (const auto& backend : backends)
        {
            backend->observe_cpu(token_count, weight_size, elapsed_microseconds);
        }
    }

    void observe_phase(uint32_t token_count, uint64_t total_weight_bytes, uint64_t accelerated_weight_bytes, uint64_t elapsed_microseconds) override
    {
        if (accelerated_weight_bytes == 0)
        {
            for (const auto& backend : backends)
            {
                backend->observe_phase(token_count, total_weight_bytes, 0, elapsed_microseconds);
            }
            return;
        }
        std::vector<uint64_t> observation;
        {
            const std::lock_guard<std::mutex> lock(phase_mutex);
            if (pending_accelerated_bytes.empty())
                return;
            observation = std::move(pending_accelerated_bytes.front());
            pending_accelerated_bytes.pop_front();
        }
        for (size_t backend_index = 0; backend_index < backends.size(); ++backend_index)
        {
            const uint64_t bytes = backend_index < observation.size() ? observation[backend_index] : 0;
            if (bytes == 0)
                continue;
            backends[backend_index]->observe_phase(token_count, total_weight_bytes, bytes, elapsed_microseconds);
        }
    }

    void set_foreground_active(bool active) noexcept override
    {
        for (const auto& backend : backends)
            backend->set_foreground_active(active);
    }

    void wait_for_background_work() override
    {
        for (const auto& backend : backends)
            backend->wait_for_background_work();
    }

    ExpertBackendStatistics statistics() const override
    {
        ExpertBackendStatistics aggregate;
        for (const auto& backend : backends)
            add_statistics(aggregate, backend->statistics());
        return aggregate;
    }

    std::vector<ExpertBackendDeviceStatistics> device_statistics() const override
    {
        std::vector<ExpertBackendDeviceStatistics> result;
        result.reserve(backends.size());
        for (size_t backend_index = 0; backend_index < backends.size(); ++backend_index)
        {
            std::vector<ExpertBackendDeviceStatistics> child = backends[backend_index]->device_statistics();
            if (child.empty())
            {
                result.push_back({device_indices[backend_index], backends[backend_index]->capacity(), backends[backend_index]->statistics()});
            }
            else
            {
                result.insert(result.end(), child.begin(), child.end());
            }
        }
        return result;
    }

    uint64_t capacity() const noexcept override
    {
        uint64_t total_size = 0;
        for (const auto& backend : backends)
            total_size += backend->capacity();
        return total_size;
    }

private:
    size_t backend_for_key(std::string_view key) const
    {
        const std::lock_guard<std::mutex> lock(placement_mutex);
        const auto placed = key_placements.find(key);
        return placed == key_placements.end() ? fallback_backend(key) : placed->second;
    }

    void publish_accelerated_bytes(std::vector<uint64_t> values)
    {
        const std::lock_guard<std::mutex> lock(phase_mutex);
        pending_accelerated_bytes.push_back(std::move(values));
    }

    struct ChildSubmission
    {
        std::vector<size_t> request_indices;
        std::vector<ExpertBackendRequest> requests;
        std::unique_ptr<IExpertBackendBatchSubmission> submission;
        bool reservation_shape_valid = true;
    };

    class Submission final : public IExpertBackendBatchSubmission
    {
    public:
        Submission(MultiDeviceExpertBackend* _owner, std::span<const ExpertBackendRequest> requests, std::vector<std::vector<size_t>> request_indices)
            : owner(_owner),
              client_requests(requests.begin(), requests.end()),
              private_outputs(requests.size()),
              planned(requests.size(), ExpertBackendExecutionResult ::NotResident),
              final(planned),
              accelerated_bytes(owner->backends.size(), 0)
        {
            children.reserve(owner->backends.size());
            for (size_t backend_index = 0; backend_index < owner->backends.size(); ++backend_index)
            {
                if (request_indices[backend_index].empty())
                {
                    continue;
                }
                ChildSubmission child;
                child.request_indices = std::move(request_indices[backend_index]);
                child.requests.reserve(child.request_indices.size());
                for (size_t request_index : child.request_indices)
                {
                    ExpertBackendRequest child_request = requests[request_index];
                    child_request.output = &private_outputs[request_index];
                    // Let the framework combine multi-device outputs on CPU.
                    child_request.route_aggregation = {};
                    child.requests.push_back(child_request);
                }
                child.submission = owner->backends[backend_index]->submit_batch(child.requests);
                if (child.submission)
                {
                    const auto child_planned = child.submission->reservations();
                    child.reservation_shape_valid = child_planned.size() == child.request_indices.size();
                    if (!child.reservation_shape_valid)
                    {
                        child.submission->abort();
                        for (const size_t request_index : child.request_indices)
                            planned[request_index] = ExpertBackendExecutionResult ::Failed;
                    }
                    else
                    {
                        for (size_t index = 0; index < child_planned.size(); ++index)
                        {
                            planned[child.request_indices[index]] = child_planned[index];
                        }
                    }
                }
                children.push_back(std::move(child));
                child_backend_indices.push_back(backend_index);
            }
            final = planned;
        }

        ~Submission() override
        {
            if (!waited)
                (void)wait();
            if (!committed && !aborted)
                abort();
        }

        std::span<const ExpertBackendExecutionResult> reservations() const noexcept override
        {
            return planned;
        }

        std::vector<ExpertBackendExecutionResult> wait() override
        {
            if (waited)
                return final;
            for (size_t child_index = 0; child_index < children.size(); ++child_index)
            {
                ChildSubmission& child = children[child_index];
                if (!child.submission)
                    continue;
                const auto child_final = child.submission->wait();
                const size_t backend_index = child_backend_indices[child_index];
                bool result_shape_valid = child.reservation_shape_valid && child_final.size() == child.request_indices.size();
                if (result_shape_valid)
                {
                    for (size_t index = 0; index < child_final.size(); ++index)
                    {
                        const size_t request_index = child.request_indices[index];
                        if (child_final[index] == ExpertBackendExecutionResult ::Executed
                            && planned[request_index] != ExpertBackendExecutionResult::Executed)
                        {
                            result_shape_valid = false;
                            break;
                        }
                    }
                }
                if (!result_shape_valid)
                {
                    child.submission->abort();
                    for (const size_t request_index : child.request_indices)
                        final[request_index] = ExpertBackendExecutionResult ::Failed;
                    continue;
                }
                for (size_t index = 0; index < child_final.size(); ++index)
                {
                    const size_t request_index = child.request_indices[index];
                    final[request_index] = child_final[index];
                    if (child_final[index] == ExpertBackendExecutionResult ::Executed)
                    {
                        accelerated_bytes[backend_index] += child.requests[index].weight_size;
                    }
                }
            }
            waited = true;
            return final;
        }

        bool commit() override
        {
            if (!waited)
                (void)wait();
            if (committed || aborted)
                return committed;
            for (size_t index = 0; index < final.size(); ++index)
            {
                if (final[index] != ExpertBackendExecutionResult::Executed)
                    continue;
                if (!client_requests[index].output)
                {
                    abort();
                    return false;
                }
            }
            for (ChildSubmission& child : children)
            {
                if (child.submission)
                {
                    bool child_has_executed = false;
                    for (const size_t request_index : child.request_indices)
                    {
                        child_has_executed = child_has_executed || final[request_index] == ExpertBackendExecutionResult::Executed;
                    }
                    if (child_has_executed && !child.submission->commit())
                    {
                        abort();
                        return false;
                    }
                    if (!child_has_executed)
                        child.submission->abort();
                }
            }
            for (size_t index = 0; index < final.size(); ++index)
            {
                if (final[index] == ExpertBackendExecutionResult::Executed)
                    client_requests[index].output->swap(private_outputs[index]);
            }
            committed = true;
            owner->publish_accelerated_bytes(std::move(accelerated_bytes));
            return true;
        }

        void abort() noexcept override
        {
            if (committed || aborted)
                return;
            aborted = true;
            for (ChildSubmission& child : children)
            {
                if (child.submission)
                    child.submission->abort();
            }
        }

    private:
        MultiDeviceExpertBackend* owner;
        std::vector<ExpertBackendRequest> client_requests;
        std::vector<ActivationBuffer> private_outputs;
        std::vector<ChildSubmission> children;
        std::vector<size_t> child_backend_indices;
        std::vector<ExpertBackendExecutionResult> planned;
        std::vector<ExpertBackendExecutionResult> final;
        std::vector<uint64_t> accelerated_bytes;
        bool waited = false;
        bool committed = false;
        bool aborted = false;
    };

    size_t backend_for_group(uint32_t residency_group) const
    {
        if (residency_group < residency_group_devices.size())
        {
            const auto backend = device_to_backend.find(residency_group_devices[residency_group]);
            if (backend != device_to_backend.end())
            {
                return backend->second;
            }
        }
        return static_cast<size_t>(residency_group) % backends.size();
    }

    size_t fallback_backend(std::string_view key) const
    {
        return std::hash<std::string_view>{}(key) % backends.size();
    }

    std::vector<std::shared_ptr<IExpertExecutionBackend>> backends;
    std::vector<uint32_t> device_indices;
    std::vector<uint32_t> residency_group_devices;
    std::unordered_map<uint32_t, size_t> device_to_backend;
    mutable std::mutex placement_mutex;
    std::unordered_map<std::string, size_t, ExpertBackendTransparentStringHash, std::equal_to<>> key_placements;
    bool key_sharded = false;
    mutable std::mutex phase_mutex;
    std::deque<std::vector<uint64_t>> pending_accelerated_bytes;
};

std::shared_ptr<IExpertExecutionBackend> create_multi_device_expert_backend(std::vector<std::shared_ptr<IExpertExecutionBackend>> backends, std::vector<uint32_t> device_indices, std::vector<uint32_t> residency_group_devices)
{
    if (backends.empty() || backends.size() != device_indices.size())
    {
        return {};
    }
    for (const auto& backend : backends)
    {
        if (!backend)
            return {};
    }
    if (backends.size() == 1)
        return backends.front();
    return std::make_shared<MultiDeviceExpertBackend>(std::move(backends), std::move(device_indices), std::move(residency_group_devices), false);
}

std::shared_ptr<IExpertExecutionBackend> create_key_sharded_expert_backend(std::vector<std::shared_ptr<IExpertExecutionBackend>> backends, std::vector<uint32_t> device_indices)
{
    if (backends.empty() || backends.size() != device_indices.size())
    {
        return {};
    }
    for (const auto& backend : backends)
    {
        if (!backend)
            return {};
    }
    if (backends.size() == 1)
        return backends.front();
    return std::make_shared<MultiDeviceExpertBackend>(std::move(backends), std::move(device_indices), std::vector<uint32_t>(), true);
}

} // namespace moe
} // namespace ncnn
