#include "expertbackend.h"

#include <functional>
#include <utility>

namespace ncnn {
namespace moe {

ScopedExpertBackendForeground::ScopedExpertBackendForeground(const std::shared_ptr<ExpertBackend>& _backend) noexcept
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

size_t ExpertBackend::prepare_demand_batch(std::span<const ExpertDemandRequest> requests,
                                           std::span<std::shared_ptr<const void>> pins)
{
    for (auto& pin : pins) pin.reset();
    if (requests.size() != pins.size()) return 0;
    size_t prepared = 0;
    for (const auto& request : requests)
    {
        if (!prepare_demand(std::string(request.key), request.gate_up, request.gate_up_bias,
                            request.down, request.down_bias, request.residency_group,
                            request.activation_limit, request.activation, pins[prepared]))
            break;
        ++prepared;
    }
    return prepared;
}

class CompletedExpertDemandSubmission final : public ExpertDemandSubmission
{
public:
    CompletedExpertDemandSubmission(std::vector<std::shared_ptr<const void>> _pins, size_t _prepared)
        : pins(std::move(_pins)), prepared(_prepared)
    {
    }

    size_t wait(std::span<std::shared_ptr<const void>> outputs) override
    {
        for (auto& output : outputs) output.reset();
        if (aborted || outputs.size() != pins.size()) return 0;
        for (size_t index = 0; index < prepared; ++index) outputs[index] = pins[index];
        return prepared;
    }

    void abort() noexcept override
    {
        aborted = true;
        pins.clear();
    }

private:
    std::vector<std::shared_ptr<const void>> pins;
    size_t prepared = 0;
    bool aborted = false;
};

class MultiDeviceExpertDemandSubmission final : public ExpertDemandSubmission
{
public:
    struct Child
    {
        size_t count = 0;
        std::unique_ptr<ExpertDemandSubmission> submission;
    };

    explicit MultiDeviceExpertDemandSubmission(size_t request_count)
        : pins(request_count)
    {
    }

    ~MultiDeviceExpertDemandSubmission() override
    {
        if (!waited && !aborted) abort();
    }

    size_t wait(std::span<std::shared_ptr<const void>> outputs) override
    {
        for (auto& output : outputs) output.reset();
        if (aborted || outputs.size() != pins.size()) return 0;
        if (!waited)
        {
            size_t offset = 0;
            bool prefix_open = true;
            for (auto& child : children)
            {
                if (prefix_open && child.submission)
                {
                    auto child_pins = std::span<std::shared_ptr<const void>>(pins).subspan(offset, child.count);
                    const size_t child_prepared = child.submission->wait(child_pins);
                    bool valid = child_prepared <= child.count;
                    if (valid)
                        for (size_t index = 0; index < child_prepared; ++index) valid = valid && static_cast<bool>(child_pins[index]);
                    if (!valid)
                    {
                        child.submission->abort();
                        for (auto& pin : child_pins) pin.reset();
                    }
                    else
                    {
                        prepared += child_prepared;
                    }
                    prefix_open = valid && child_prepared == child.count;
                }
                else
                {
                    prefix_open = false;
                    if (child.submission) child.submission->abort();
                }
                offset += child.count;
            }
            for (size_t index = prepared; index < pins.size(); ++index) pins[index].reset();
            waited = true;
        }
        for (size_t index = 0; index < prepared; ++index) outputs[index] = pins[index];
        return prepared;
    }

    void abort() noexcept override
    {
        if (aborted) return;
        aborted = true;
        for (auto& child : children)
            if (child.submission) child.submission->abort();
        pins.clear();
    }

    std::vector<Child> children;

private:
    std::vector<std::shared_ptr<const void>> pins;
    size_t prepared = 0;
    bool waited = false;
    bool aborted = false;
};

std::unique_ptr<ExpertDemandSubmission> ExpertBackend::begin_demand_batch(std::span<const ExpertDemandRequest> requests)
{
    std::vector<std::shared_ptr<const void>> pins(requests.size());
    size_t prepared = prepare_demand_batch(requests, pins);
    bool valid = prepared <= pins.size();
    if (valid)
        for (size_t index = 0; index < prepared; ++index) valid = valid && static_cast<bool>(pins[index]);
    if (!valid) prepared = 0;
    for (size_t index = prepared; index < pins.size(); ++index) pins[index].reset();
    return std::make_unique<CompletedExpertDemandSubmission>(std::move(pins), prepared);
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

MultiDeviceExpertBackend::MultiDeviceExpertBackend(std::vector<std::shared_ptr<ExpertBackend>> _backends, std::vector<uint32_t> device_indices, std::vector<uint32_t> _residency_group_devices, bool _key_sharded)
    : backends(std::move(_backends)),
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
    backends[backend_index]->admit(std::move(key),
                                   std::move(gate_up),
                                   gate_up_bias,
                                   std::move(down),
                                   down_bias,
                                   residency_group,
                                   activation_limit,
                                   activation);
}

bool MultiDeviceExpertBackend::prepare_demand(std::string key, std::shared_ptr<const TensorData> gate_up,
                                              const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down,
                                              const TensorData* down_bias, uint32_t residency_group,
                                              float activation_limit, ExpertActivation activation,
                                              std::shared_ptr<const void>& pin)
{
    pin.reset();
    if (backends.empty() || key.empty())
        return false;
    const size_t backend_index = key_sharded ? fallback_backend(key) : backend_for_group(residency_group);
    {
        const std::lock_guard<std::mutex> lock(placement_mutex);
        key_placements.insert_or_assign(key, backend_index);
    }
    return backends[backend_index]->prepare_demand(std::move(key), std::move(gate_up), gate_up_bias,
                                                   std::move(down), down_bias, residency_group,
                                                   activation_limit, activation, pin);
}

size_t MultiDeviceExpertBackend::prepare_demand_batch(std::span<const ExpertDemandRequest> requests,
                                                      std::span<std::shared_ptr<const void>> pins)
{
    for (auto& pin : pins) pin.reset();
    if (requests.size() != pins.size() || backends.empty()) return 0;
    size_t prepared = 0;
    while (prepared < requests.size())
    {
        if (requests[prepared].key.empty()) break;
        const auto select = [this](const ExpertDemandRequest& request) {
            return key_sharded ? fallback_backend(request.key) : backend_for_group(request.residency_group);
        };
        const size_t backend_index = select(requests[prepared]);
        size_t count = 1;
        while (prepared + count < requests.size() && !requests[prepared + count].key.empty()
               && select(requests[prepared + count]) == backend_index) ++count;
        {
            const std::lock_guard<std::mutex> lock(placement_mutex);
            for (size_t index = 0; index < count; ++index)
                key_placements.insert_or_assign(std::string(requests[prepared + index].key), backend_index);
        }
        const size_t child_prepared = backends[backend_index]->prepare_demand_batch(requests.subspan(prepared, count), pins.subspan(prepared, count));
        if (child_prepared > count)
        {
            for (size_t index = prepared; index < pins.size(); ++index) pins[index].reset();
            return prepared;
        }
        prepared += child_prepared;
        if (child_prepared != count) break;
    }
    return prepared;
}

std::unique_ptr<ExpertDemandSubmission> MultiDeviceExpertBackend::begin_demand_batch(std::span<const ExpertDemandRequest> requests)
{
    auto work = std::make_unique<MultiDeviceExpertDemandSubmission>(requests.size());
    if (backends.empty()) return work;
    size_t cursor = 0;
    while (cursor < requests.size())
    {
        if (requests[cursor].key.empty()) break;
        const auto select = [this](const ExpertDemandRequest& request) {
            return key_sharded ? fallback_backend(request.key) : backend_for_group(request.residency_group);
        };
        const size_t backend_index = select(requests[cursor]);
        size_t count = 1;
        while (cursor + count < requests.size() && !requests[cursor + count].key.empty()
               && select(requests[cursor + count]) == backend_index) ++count;
        {
            const std::lock_guard<std::mutex> lock(placement_mutex);
            for (size_t index = 0; index < count; ++index)
                key_placements.insert_or_assign(std::string(requests[cursor + index].key), backend_index);
        }
        auto child = backends[backend_index]->begin_demand_batch(requests.subspan(cursor, count));
        work->children.push_back({count, std::move(child)});
        cursor += count;
    }
    return work;
}

void MultiDeviceExpertBackend::set_residency_coordinator(std::shared_ptr<ExpertResidencyCoordinator> coordinator)
{
    for (const auto& backend : backends)
        backend->set_residency_coordinator(coordinator);
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

MultiDeviceExpertBackend::Submission::Submission(MultiDeviceExpertBackend* owner, std::span<const ExpertBackendRequest> requests, std::vector<std::vector<size_t>> request_indices)
    : client_requests(requests.begin(), requests.end()),
      private_outputs(requests.size()),
      planned(requests.size(), ExpertBackendExecutionResult ::NotResident),
      final(planned)
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
            // Cross-device aggregation uses completed host outputs.
            child_request.device_output = nullptr;
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
    std::vector<ExpertBackendExecutionResult> results;
    wait(results);
    return results;
}

void MultiDeviceExpertBackend::Submission::wait(std::vector<ExpertBackendExecutionResult>& results)
{
    if (waited)
    {
        results.assign(final.begin(), final.end());
        return;
    }
    for (ChildSubmission& child : children)
    {
        if (!child.submission)
            continue;
        results.clear();
        child.submission->wait(results);
        const std::vector<ExpertBackendExecutionResult>& child_final = results;
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
        }
    }
    waited = true;
    results.assign(final.begin(), final.end());
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
        if (client_requests[index].device_output)
            client_requests[index].device_output->reset();
    }
    committed = true;
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
