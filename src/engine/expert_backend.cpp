#include "expert_backend.h"

#include <algorithm>
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
    NCNN_MOE_ADD_EXPERT_STATISTIC(resident_bytes);
    NCNN_MOE_ADD_EXPERT_STATISTIC(pending_bytes);
    NCNN_MOE_ADD_EXPERT_STATISTIC(execution_time_microseconds);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_recent_bytes);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_frequent_bytes);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_recent_target_bytes);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_recent_ghost_bytes);
    NCNN_MOE_ADD_EXPERT_STATISTIC(arc_frequent_ghost_bytes);
    NCNN_MOE_ADD_EXPERT_STATISTIC(device_source_hits);
    NCNN_MOE_ADD_EXPERT_STATISTIC(device_source_misses);
    NCNN_MOE_ADD_EXPERT_STATISTIC(device_source_executions);
    NCNN_MOE_ADD_EXPERT_STATISTIC(device_source_execution_failures);
#undef NCNN_MOE_ADD_EXPERT_STATISTIC
}

class MultiDeviceExpertBackend final : public IExpertExecutionBackend
{
public:
    MultiDeviceExpertBackend(std::vector<std::shared_ptr<IExpertExecutionBackend>> backends, std::vector<uint32_t> device_indices, std::vector<uint32_t> residency_group_devices, bool key_sharded)
        : backends_(std::move(backends)),
          device_indices_(std::move(device_indices)),
          residency_group_devices_(std::move(residency_group_devices)),
          key_sharded_(key_sharded)
    {
        for (size_t index = 0; index < device_indices_.size(); ++index)
        {
            device_to_backend_.emplace(device_indices_[index], index);
        }
    }

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, const TensorData* gate_up_bias, std::shared_ptr<const TensorData> down, const TensorData* down_bias, uint32_t residency_group, uint32_t token_count,
               float activation_limit) override
    {
        if (backends_.empty() || key.empty())
            return;
        const size_t backend_index = key_sharded_ ? fallback_backend(key) : backend_for_group(residency_group);
        {
            const std::lock_guard<std::mutex> lock(placement_mutex_);
            key_placements_.insert_or_assign(key, backend_index);
        }
        backends_[backend_index]->admit(std::move(key), std::move(gate_up), gate_up_bias, std::move(down), down_bias, residency_group, token_count, activation_limit);
    }

    ExpertBackendExecutionResult try_execute(const std::string& key, const CpuBatch& input, CpuBatch& output) override
    {
        const ExpertBackendRequest request{
            key,
            &input,
            &output,
            0,
        };
        std::vector<ExpertBackendExecutionResult> results = try_execute_batch(std::span<const ExpertBackendRequest>(&request, 1));
        return results.empty() ? ExpertBackendExecutionResult::Failed : results.front();
    }

    std::vector<ExpertBackendExecutionResult> try_execute_batch(std::span<const ExpertBackendRequest> requests) override
    {
        auto submission = submit_batch(requests);
        return submission ? submission->wait() : std::vector<ExpertBackendExecutionResult>(requests.size(), ExpertBackendExecutionResult::Failed);
    }

    std::unique_ptr<IExpertBackendBatchSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override
    {
        std::vector<std::vector<size_t>> request_indices(backends_.size());
        {
            const std::lock_guard<std::mutex> lock(placement_mutex_);
            for (size_t request_index = 0; request_index < requests.size(); ++request_index)
            {
                const auto placed = key_placements_.find(requests[request_index].key);
                const size_t backend_index = placed == key_placements_.end() ? fallback_backend(requests[request_index].key) : placed->second;
                request_indices[backend_index].push_back(request_index);
            }
        }
        return std::unique_ptr<IExpertBackendBatchSubmission>(new Submission(this, requests, std::move(request_indices)));
    }

    void observe_cpu(uint32_t token_count, uint64_t weight_bytes, uint64_t elapsed_microseconds) override
    {
        for (const auto& backend : backends_)
        {
            backend->observe_cpu(token_count, weight_bytes, elapsed_microseconds);
        }
    }

    void observe_phase(uint32_t token_count, uint64_t total_weight_bytes, uint64_t accelerated_weight_bytes, uint64_t elapsed_microseconds) override
    {
        if (accelerated_weight_bytes == 0)
        {
            for (const auto& backend : backends_)
            {
                backend->observe_phase(token_count, total_weight_bytes, 0, elapsed_microseconds);
            }
            return;
        }
        auto observation = thread_accelerated_bytes_.find(this);
        if (observation == thread_accelerated_bytes_.end())
        {
            return;
        }
        for (size_t backend_index = 0; backend_index < backends_.size(); ++backend_index)
        {
            const uint64_t bytes = backend_index < observation->second.size() ? observation->second[backend_index] : 0;
            if (bytes == 0)
                continue;
            backends_[backend_index]->observe_phase(token_count, total_weight_bytes, bytes, elapsed_microseconds);
        }
        thread_accelerated_bytes_.erase(observation);
    }

    void wait_for_background_work() override
    {
        for (const auto& backend : backends_)
            backend->wait_for_background_work();
    }

    ExpertBackendStatistics statistics() const override
    {
        ExpertBackendStatistics aggregate;
        for (const auto& backend : backends_)
            add_statistics(aggregate, backend->statistics());
        return aggregate;
    }

    std::vector<ExpertBackendDeviceStatistics> device_statistics() const override
    {
        std::vector<ExpertBackendDeviceStatistics> result;
        result.reserve(backends_.size());
        for (size_t backend_index = 0; backend_index < backends_.size(); ++backend_index)
        {
            std::vector<ExpertBackendDeviceStatistics> child = backends_[backend_index]->device_statistics();
            if (child.empty())
            {
                result.push_back({device_indices_[backend_index], backends_[backend_index]->capacity_bytes(), backends_[backend_index]->statistics()});
            }
            else
            {
                result.insert(result.end(), child.begin(), child.end());
            }
        }
        return result;
    }

    uint64_t capacity_bytes() const noexcept override
    {
        uint64_t capacity = 0;
        for (const auto& backend : backends_)
            capacity += backend->capacity_bytes();
        return capacity;
    }

private:
    struct ChildSubmission
    {
        std::vector<size_t> request_indices;
        std::vector<ExpertBackendRequest> requests;
        std::unique_ptr<IExpertBackendBatchSubmission> submission;
    };

    class Submission final : public IExpertBackendBatchSubmission
    {
    public:
        Submission(MultiDeviceExpertBackend* owner, std::span<const ExpertBackendRequest> requests, std::vector<std::vector<size_t>> request_indices)
            : owner_(owner),
              planned_(requests.size(), ExpertBackendExecutionResult ::NotResident),
              final_(planned_),
              accelerated_bytes_(owner->backends_.size(), 0)
        {
            children_.reserve(owner->backends_.size());
            for (size_t backend_index = 0; backend_index < owner->backends_.size(); ++backend_index)
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
                    child.requests.push_back(requests[request_index]);
                }
                child.submission = owner->backends_[backend_index]->submit_batch(child.requests);
                if (child.submission)
                {
                    const auto child_planned = child.submission->planned_results();
                    const size_t result_count = std::min(child_planned.size(), child.request_indices.size());
                    for (size_t index = 0; index < result_count; ++index)
                    {
                        planned_[child.request_indices[index]] = child_planned[index];
                    }
                }
                children_.push_back(std::move(child));
                child_backend_indices_.push_back(backend_index);
            }
            final_ = planned_;
        }

        ~Submission() override
        {
            if (!waited_)
                (void)wait();
        }

        std::span<const ExpertBackendExecutionResult> planned_results() const noexcept override
        {
            return planned_;
        }

        std::vector<ExpertBackendExecutionResult> wait() override
        {
            if (waited_)
                return final_;
            for (size_t child_index = 0; child_index < children_.size(); ++child_index)
            {
                ChildSubmission& child = children_[child_index];
                if (!child.submission)
                    continue;
                const auto child_final = child.submission->wait();
                const size_t result_count = std::min(child_final.size(), child.request_indices.size());
                const size_t backend_index = child_backend_indices_[child_index];
                for (size_t index = 0; index < result_count; ++index)
                {
                    const size_t request_index = child.request_indices[index];
                    final_[request_index] = child_final[index];
                    if (child_final[index] == ExpertBackendExecutionResult ::Executed)
                    {
                        accelerated_bytes_[backend_index] += child.requests[index].weight_bytes;
                    }
                }
            }
            thread_accelerated_bytes_.insert_or_assign(owner_, accelerated_bytes_);
            waited_ = true;
            return final_;
        }

    private:
        MultiDeviceExpertBackend* owner_;
        std::vector<ChildSubmission> children_;
        std::vector<size_t> child_backend_indices_;
        std::vector<ExpertBackendExecutionResult> planned_;
        std::vector<ExpertBackendExecutionResult> final_;
        std::vector<uint64_t> accelerated_bytes_;
        bool waited_ = false;
    };

    size_t backend_for_group(uint32_t residency_group) const
    {
        if (residency_group < residency_group_devices_.size())
        {
            const auto backend = device_to_backend_.find(residency_group_devices_[residency_group]);
            if (backend != device_to_backend_.end())
            {
                return backend->second;
            }
        }
        return static_cast<size_t>(residency_group) % backends_.size();
    }

    size_t fallback_backend(std::string_view key) const
    {
        return std::hash<std::string_view>{}(key) % backends_.size();
    }

    std::vector<std::shared_ptr<IExpertExecutionBackend>> backends_;
    std::vector<uint32_t> device_indices_;
    std::vector<uint32_t> residency_group_devices_;
    std::unordered_map<uint32_t, size_t> device_to_backend_;
    mutable std::mutex placement_mutex_;
    std::unordered_map<std::string, size_t, ExpertBackendTransparentStringHash, std::equal_to<>> key_placements_;
    bool key_sharded_ = false;
    static thread_local std::unordered_map<const MultiDeviceExpertBackend*, std::vector<uint64_t>> thread_accelerated_bytes_;
};

thread_local std::unordered_map<const MultiDeviceExpertBackend*, std::vector<uint64_t>> MultiDeviceExpertBackend::thread_accelerated_bytes_;

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
