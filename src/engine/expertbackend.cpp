#include "expertbackend.h"

#include <functional>
#include <utility>

namespace ncnn {
namespace moe {

ScopedExpertBackendForeground::ScopedExpertBackendForeground(
    const std::shared_ptr<ExpertBackend>& _backend) noexcept
    : backend(_backend)
{
    if (backend)
        backend->set_foreground_active(true);
}

ScopedExpertBackendForeground::~ScopedExpertBackendForeground()
{
    if (backend)
        backend->set_foreground_active(false);
}

static void add_statistics(ExpertBackendStatistics& destination, const ExpertBackendStatistics& source)
{
    destination.hits += source.hits;
    destination.misses += source.misses;
    destination.admissions += source.admissions;
    destination.stores += source.stores;
    destination.evictions += source.evictions;
    destination.dropped_admissions += source.dropped_admissions;
    destination.executions += source.executions;
    destination.execution_failures += source.execution_failures;
    destination.bytes_uploaded += source.bytes_uploaded;
    destination.resident_size += source.resident_size;
    destination.pending_size += source.pending_size;
    destination.execution_time_microseconds += source.execution_time_microseconds;
    destination.arc_recent_size += source.arc_recent_size;
    destination.arc_frequent_size += source.arc_frequent_size;
    destination.arc_recent_target_size += source.arc_recent_target_size;
    destination.arc_recent_ghost_size += source.arc_recent_ghost_size;
    destination.arc_frequent_ghost_size += source.arc_frequent_ghost_size;
    destination.device_source_hits += source.device_source_hits;
    destination.device_source_misses += source.device_source_misses;
    destination.device_source_executions += source.device_source_executions;
    destination.device_source_execution_failures += source.device_source_execution_failures;
    destination.route_aggregation_batches += source.route_aggregation_batches;
    destination.route_aggregation_routes += source.route_aggregation_routes;
    destination.route_aggregation_bytes_saved += source.route_aggregation_bytes_saved;
}

MultiDeviceExpertBackend::MultiDeviceExpertBackend(std::vector<std::shared_ptr<ExpertBackend>> _backends, std::vector<uint32_t> _device_indices, std::vector<uint32_t> _residency_group_devices, bool _key_sharded)
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

void MultiDeviceExpertBackend::admit(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down, const TensorData* down_bias, uint32_t residency_group,
                                     float activation_limit, ExpertActivation activation)
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
        activation_limit,
        activation);
}

ExpertBackendExecutionResult MultiDeviceExpertBackend::try_execute(const std::string& key, const ActivationBuffer& input, ActivationBuffer& output)
{
    if (backends.empty())
        return ExpertBackendExecutionResult::Failed;
    return backends[backend_for_key(key)]->try_execute(key, input, output);
}

std::vector<ExpertBackendExecutionResult> MultiDeviceExpertBackend::try_execute_batch(std::span<const ExpertBackendRequest> requests)
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

std::unique_ptr<ExpertSubmission> MultiDeviceExpertBackend::submit_batch(std::span<const ExpertBackendRequest> requests)
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

void MultiDeviceExpertBackend::observe_cpu(uint32_t token_count, uint64_t weight_size, uint64_t elapsed_microseconds)
{
    for (const auto& backend : backends)
    {
        backend->observe_cpu(token_count, weight_size, elapsed_microseconds);
    }
}

void MultiDeviceExpertBackend::observe_phase(uint32_t token_count, uint64_t total_weight_bytes, uint64_t accelerated_weight_bytes, uint64_t elapsed_microseconds)
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

void MultiDeviceExpertBackend::set_foreground_active(bool active) noexcept
{
    for (const auto& backend : backends)
        backend->set_foreground_active(active);
}

void MultiDeviceExpertBackend::wait_for_background_work()
{
    for (const auto& backend : backends)
        backend->wait_for_background_work();
}

ExpertBackendStatistics MultiDeviceExpertBackend::statistics() const
{
    ExpertBackendStatistics aggregate;
    for (const auto& backend : backends)
        add_statistics(aggregate, backend->statistics());
    return aggregate;
}

std::vector<ExpertBackendDeviceStatistics> MultiDeviceExpertBackend::device_statistics() const
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

uint64_t MultiDeviceExpertBackend::capacity() const noexcept
{
    uint64_t total_size = 0;
    for (const auto& backend : backends)
        total_size += backend->capacity();
    return total_size;
}

size_t MultiDeviceExpertBackend::backend_for_key(std::string_view key) const
{
    const std::lock_guard<std::mutex> lock(placement_mutex);
    const auto placed = key_placements.find(key);
    return placed == key_placements.end() ? fallback_backend(key) : placed->second;
}

void MultiDeviceExpertBackend::publish_accelerated_bytes(std::vector<uint64_t> values)
{
    const std::lock_guard<std::mutex> lock(phase_mutex);
    pending_accelerated_bytes.push_back(std::move(values));
}

MultiDeviceExpertBackend::Submission::Submission(MultiDeviceExpertBackend* _owner, std::span<const ExpertBackendRequest> requests, std::vector<std::vector<size_t>> request_indices)
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
        child.backend_index = backend_index;
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
    }
    final = planned;
}

MultiDeviceExpertBackend::Submission::~Submission()
{
    if (!waited)
        (void)wait();
    if (!committed && !aborted)
        abort();
}

std::span<const ExpertBackendExecutionResult> MultiDeviceExpertBackend::Submission::reservations() const noexcept
{
    return planned;
}

std::vector<ExpertBackendExecutionResult> MultiDeviceExpertBackend::Submission::wait()
{
    if (waited)
        return final;
    for (ChildSubmission& child : children)
    {
        if (!child.submission)
            continue;
        const auto child_final = child.submission->wait();
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
                accelerated_bytes[child.backend_index] += child.requests[index].weight_size;
            }
        }
    }
    waited = true;
    return final;
}

bool MultiDeviceExpertBackend::Submission::commit()
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

void MultiDeviceExpertBackend::Submission::abort() noexcept
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

size_t MultiDeviceExpertBackend::backend_for_group(uint32_t residency_group) const
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

size_t MultiDeviceExpertBackend::fallback_backend(std::string_view key) const
{
    return std::hash<std::string_view>{}(key) % backends.size();
}

std::shared_ptr<ExpertBackend> create_multi_device_expert_backend(std::vector<std::shared_ptr<ExpertBackend>> backends, std::vector<uint32_t> device_indices, std::vector<uint32_t> residency_group_devices)
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

std::shared_ptr<ExpertBackend> create_key_sharded_expert_backend(std::vector<std::shared_ptr<ExpertBackend>> backends, std::vector<uint32_t> device_indices)
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
