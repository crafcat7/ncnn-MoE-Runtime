#include "backends/ncnn/expertbackend_vulkan.h"
#include "engine/expertbackend.h"
#include "graph/resourceplan.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace moe {

static void budget_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void test_resource_exact_fit_and_cache_reserve()
{
    const GpuResourceSnapshot initial{2, 1024, 128};
    auto plan = make_gpu_resource_plan(initial, 1, 128);
    const uint64_t persistent = initial.heap_budget - initial.heap_usage
                                - 64 - plan.expert_cache_reserve;
    const GpuResourceRequest exact{persistent, 0, 64};
    budget_check(can_admit_gpu_resources(plan, initial, exact),
                 "persistent admission accepts an exact fit after explicit cache and workspace reservations");
    budget_check(!can_admit_gpu_resources(plan, initial, {persistent + 1, 0, 64}),
                 "persistent admission rejects a one-byte overcommit");
    budget_check(plan.admitted.persistent_size == 0,
                 "admission queries do not consume resources before creation succeeds");
    commit_gpu_resources(plan, exact);
    const GpuResourceSnapshot allocated{2, 1024, initial.heap_usage + persistent};
    budget_check(!can_admit_gpu_resources(plan, allocated, {1, 0, 0}),
                 "committed persistent operators preserve the explicitly requested cache capacity");
    budget_check(gpu_resource_cache_capacity(plan, allocated, 16) == 128,
                 "cache admission releases its explicit reservation instead of subtracting it twice");
    budget_check(gpu_resource_cache_capacity(plan, allocated, 16, 65) == 64,
                 "cache capacity is bounded by eligible payload and whole entries");

    plan.allocated_execution_size = 64;
    const GpuResourceSnapshot graph_allocated{2, 1024, allocated.heap_usage + 64};
    budget_check(gpu_resource_cache_capacity(plan, graph_allocated, 16) == 128,
                 "physically allocated graph memory is not charged twice to the live heap");
    budget_check(can_admit_gpu_resources(plan, graph_allocated, {}),
                 "an exact fit remains valid after graph allocation");
    plan.allocated_execution_size = 80;
    budget_check(gpu_resource_cache_capacity(plan, {2, 1024, allocated.heap_usage + 80}, 16) == 112,
                 "actual allocation alignment is charged to the baseline and live heap once");
}

static void test_resource_live_usage_and_device_bounds()
{
    const GpuResourceSnapshot initial{2, 1024, 128};
    auto plan = make_gpu_resource_plan(initial, 1);
    budget_check(gpu_resource_cache_capacity(plan, initial, 1) == initial.heap_budget - initial.heap_usage,
                 "unused heap remains available to the cache without an arbitrary safety fraction");
    const uint64_t capacity = initial.heap_budget - initial.heap_usage;
    const GpuResourceRequest first{capacity / 2, 0, 0};
    budget_check(can_admit_gpu_resources(plan, initial, first), "first operator fits the initial device allowance");
    commit_gpu_resources(plan, first);
    const GpuResourceRequest second{capacity - first.persistent_size, 0, 0};
    const GpuResourceSnapshot allocated{2, 1024, initial.heap_usage + first.persistent_size};
    budget_check(can_admit_gpu_resources(plan, allocated, second),
                 "live allocation of committed weights is not subtracted a second time");
    budget_check(can_admit_gpu_resources(plan, initial, second),
                 "the committed baseline remains authoritative when live usage has not caught up");
    budget_check(!can_admit_gpu_resources(plan, {2, 1024, allocated.heap_usage + 1}, second),
                 "external live heap growth prevents an otherwise exact-fit allocation");
    budget_check(!can_admit_gpu_resources(plan, {2, 1022, allocated.heap_usage}, second),
                 "a smaller live heap budget constrains existing plans");
    budget_check(!can_admit_gpu_resources(plan, {7, 1024, 0}, {1, 0, 0})
                     && gpu_resource_cache_capacity(plan, {7, 1024, 0}, 1) == 0,
                 "a plan cannot spend another device's available memory");
    auto other = make_gpu_resource_plan({7, 1024, 0}, 1);
    budget_check(can_admit_gpu_resources(other, {7, 1024, 0}, {capacity, 0, 0}),
                 "independent devices retain their own resource allowances");
    budget_check(!can_admit_gpu_resources(plan, {2, 0, 0}, {1, 0, 0})
                     && gpu_resource_cache_capacity(plan, {2, 0, 0}, 1) == 0,
                 "an unknown live heap cannot grant GPU resources");
    auto unknown = make_gpu_resource_plan({2, 0, 0}, 1);
    budget_check(!can_admit_gpu_resources(unknown, initial, {1, 0, 0})
                     && gpu_resource_cache_capacity(unknown, initial, 1) == 0,
                 "an unknown initial heap cannot borrow the live snapshot as an implicit budget");
}

static void test_resource_concurrency_and_peak_workspace()
{
    const GpuResourceSnapshot initial{3, 4096, 0};
    budget_check(can_admit_gpu_resources(make_gpu_resource_plan(initial, 4), initial, {3072, 0, 128}),
                 "shared persistent weights must not be multiplied by session concurrency");
    const GpuResourceRequest stateful{2048, 256, 384};
    budget_check(can_admit_gpu_resources(make_gpu_resource_plan(initial, 2), initial, stateful)
                     && !can_admit_gpu_resources(make_gpu_resource_plan(initial, 4), initial, stateful),
                 "per-session state and workspace must constrain concurrent execution");
    auto plan = make_gpu_resource_plan(initial, 3);
    commit_gpu_resources(plan, {256, 64, 100});
    commit_gpu_resources(plan, {256, 64, 200});
    budget_check(gpu_resource_execution_reserve(plan) == (128 + 200) * 3,
                 "sequential workspaces must contribute their peak, while session state accumulates");
}

static void test_resource_overflow_and_state_only_requests()
{
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    const GpuResourceSnapshot initial{0, 1024, 0};
    auto plan = make_gpu_resource_plan(initial, 1);
    const uint64_t state_capacity = initial.heap_budget;
    budget_check(can_admit_gpu_resources(plan, initial, {0, state_capacity, 0}),
                 "state-only admission accepts an exact fit");
    budget_check(!can_admit_gpu_resources(plan, initial, {0, state_capacity + 1, 0})
                     && !can_admit_gpu_resources(plan, initial, {0, 0, initial.heap_budget + 1}),
                 "zero persistent bytes do not bypass execution reservation limits");
    auto concurrent = make_gpu_resource_plan(initial, 2);
    budget_check(!can_admit_gpu_resources(concurrent, initial, {0, maximum, 0}),
                 "session-state and workspace multiplication cannot wrap into admission");
    auto overflow = make_gpu_resource_plan({0, maximum, 0}, 1);
    budget_check(!can_admit_gpu_resources(overflow, {0, maximum, 0}, {0, maximum, 1}),
                 "state plus workspace cannot overflow even when the device budget is UINT64_MAX");
    auto reserve_overflow = make_gpu_resource_plan({0, maximum, 0}, 1, 1);
    budget_check(!can_admit_gpu_resources(reserve_overflow, {0, maximum, 0}, {0, maximum, 0}),
                 "state plus explicit cache cannot overflow at the maximum device budget");
    commit_gpu_resources(overflow, {maximum, 1, 0});
    budget_check(!can_admit_gpu_resources(overflow, {0, maximum, 0}, {1, 0, 0})
                     && !can_admit_gpu_resources(overflow, {0, maximum, 0}, {0, maximum, 0}),
                 "accumulated persistent and state sizes cannot overflow into a smaller request");
    budget_check(gpu_resource_cache_capacity(overflow, {0, maximum, 0}, 1) == 0,
                 "saturated committed usage cannot leave a cache allowance");
}

static void test_live_budget()
{
    // Three layers each select two 5-byte Experts. Their stable routes need
    // 30 bytes across the decode, while the previous one-layer wave needs 10.
    constexpr uint64_t all_experts = 120;
    constexpr uint64_t largest_expert = 5;
    budget_check(native_quantized_gpu_cache_size(200, all_experts, largest_expert) == all_experts,
                 "cache budget is capped by exact eligible device payload");
    budget_check(native_quantized_gpu_cache_size(20, all_experts, largest_expert) == 20,
                 "partial live heap remains available to ARC rather than shrinking to one layer");
    budget_check(native_quantized_gpu_cache_size(7, all_experts, largest_expert) == 7,
                 "bounded fallback never exceeds the live heap allowance");
    budget_check(native_quantized_gpu_cache_size(4, all_experts, largest_expert) == 0,
                 "a device that cannot hold one Expert uses CPU fallback");
    budget_check(native_quantized_gpu_cache_size(80, 12, 5) == 12,
                 "heterogeneous Expert upper bounds cannot exceed total payload");
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    budget_check(native_quantized_gpu_cache_size(maximum, maximum, 5) == maximum,
                 "budget selection does not overflow at uint64 limits");
    budget_check(native_quantized_gpu_cache_size(0, all_experts, largest_expert) == 0,
                 "missing live allowance does not reserve GPU memory");
    budget_check(native_quantized_gpu_cache_size(100, 0, 0) == 0,
                 "a device without eligible Experts has no cache");
}

class CountingBudgetBackend final : public ExpertBackend
{
public:
    void admit(const ExpertDemandRequest&) override
    {
        ++admissions;
    }

    bool prepare_demand(const ExpertDemandRequest&, std::shared_ptr<const void>& pin) override
    {
        ++demands;
        pin = std::make_shared<int>(1);
        return true;
    }

    std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertRequest> requests) override
    {
        ++submissions;
        if (!requests.empty()) last_key = requests.front().key;
        return {};
    }

    void wait_for_background_work() override
    {
    }
    ExpertBackendStatistics statistics() const override
    { return {}; }

    uint32_t admissions = 0;
    uint32_t demands = 0;
    uint32_t submissions = 0;
    std::string last_key;
};

static void test_key_sharded_routing()
{
    const std::array<std::shared_ptr<CountingBudgetBackend>, 2> children = {
        std::make_shared<CountingBudgetBackend>(), std::make_shared<CountingBudgetBackend>()};
    auto backend = create_key_sharded_expert_backend({children[0], children[1]}, {2, 7});
    std::array<std::string, 2> keys = {"first", "second"};
    for (size_t index = 0; index < keys.size(); ++index)
    {
        // An odd suffix changes the low FNV hash bit on MSVC too.
        while (std::hash<std::string_view>{}(keys[index]) % children.size() != index)
            keys[index] += 'a';
    }
    auto weight = std::make_shared<TensorData>();
    for (size_t index = 0; index < keys.size(); ++index)
    {
        // A conflicting group must not override deterministic key placement.
        const uint32_t group = static_cast<uint32_t>(1 - index);
        const ExpertDemandRequest packed{keys[index], weight, nullptr, weight, nullptr,
                                         group, 0.f, ExpertActivation::Silu};
        const ExpertDemandRequest separate{keys[index], weight, nullptr, weight, nullptr,
                                           group, 0.f, ExpertActivation::Silu, weight};
        backend->admit(packed);
        backend->admit(separate);
        std::shared_ptr<const void> pin;
        budget_check(backend->prepare_demand(packed, pin)
                         && pin,
                     "key-sharded demand prepares its hash-selected child");
        budget_check(backend->prepare_demand(separate, pin)
                         && pin,
                     "key-sharded separate demand prepares its hash-selected child");
        budget_check(children[index]->admissions == 2 && children[index]->demands == 2,
                     "key-sharded single requests ignore conflicting residency groups");
    }
    const std::array<ExpertDemandRequest, 2> demands = {
        ExpertDemandRequest{keys[0], weight, nullptr, weight, nullptr, 1, 0.f, ExpertActivation::Silu},
        ExpertDemandRequest{keys[1], weight, nullptr, weight, nullptr, 0, 0.f, ExpertActivation::Silu},
    };
    std::array<std::shared_ptr<const void>, 2> pins;
    budget_check(backend->prepare_demand_batch(demands, pins) == 2 && pins[0] && pins[1],
                 "key-sharded batch demand preserves the prepared prefix");
    auto upload = backend->begin_demand_batch(demands);
    budget_check(upload && upload->wait(pins) == 2 && pins[0] && pins[1],
                 "key-sharded asynchronous demand preserves the prepared prefix");
    ActivationBuffer input(1, 1);
    std::array<ActivationBuffer, 2> outputs;
    const std::array<ExpertRequest, 2> requests = {
        ExpertRequest{keys[0], &input, &outputs[0]},
        ExpertRequest{keys[1], &input, &outputs[1]},
    };
    auto submission = backend->submit_batch(requests);
    budget_check(static_cast<bool>(submission), "key-sharded batch submission exists");
    for (size_t index = 0; index < children.size(); ++index)
    {
        budget_check(children[index]->demands == 4 && children[index]->submissions == 1
                         && children[index]->last_key == keys[index],
                     "key-sharded admission and execution select the same child");
    }
}

static void test_unbudgeted_device_fallback()
{
    auto child = std::make_shared<CountingBudgetBackend>();
    // Device 7 has no live cache allowance. Its known layer must not use
    // device 2 merely because device 2 is the only remaining backend.
    auto backend = create_multi_device_expert_backend({child}, {2}, {2, 7});
    auto weight = std::make_shared<TensorData>();
    const ExpertDemandRequest missing{"missing", weight, nullptr, weight, nullptr, 1, 0.f, ExpertActivation::Silu};
    const ExpertDemandRequest missing_separate{"missing-separate", weight, nullptr, weight, nullptr,
                                               1, 0.f, ExpertActivation::Silu, weight};
    backend->admit(missing);
    backend->admit(missing_separate);
    budget_check(child->admissions == 0, "unbudgeted device must not upload to a different device");
    const ExpertDemandRequest resident{"resident", weight, nullptr, weight, nullptr, 0, 0.f, ExpertActivation::Silu};
    backend->admit(resident);
    budget_check(child->admissions == 1, "budgeted device keeps normal admission");

    std::shared_ptr<const void> pin;
    const ExpertDemandRequest missing_demand{"missing", weight, nullptr, weight, nullptr,
                                             1, 0.f, ExpertActivation::Silu, weight};
    budget_check(!backend->prepare_demand(missing_demand, pin)
                     && !pin && child->demands == 0,
                 "unbudgeted demand stays available for CPU fallback");
    const std::array<ExpertDemandRequest, 2> demands = {
        ExpertDemandRequest{"resident", weight, nullptr, weight, nullptr, 0, 0.f, ExpertActivation::Silu},
        ExpertDemandRequest{"missing", weight, nullptr, weight, nullptr, 1, 0.f, ExpertActivation::Silu},
    };
    std::array<std::shared_ptr<const void>, 2> pins;
    budget_check(backend->prepare_demand_batch(demands, pins) == 1 && pins[0] && !pins[1]
                     && child->demands == 1,
                 "partial demand returns only the prepared device prefix");
    auto async = backend->begin_demand_batch(std::span<const ExpertDemandRequest>(demands).subspan(1));
    std::array<std::shared_ptr<const void>, 1> async_pins;
    budget_check(async && async->wait(async_pins) == 0 && !async_pins[0] && child->demands == 1,
                 "asynchronous demand does not reroute an unbudgeted device");

    ActivationBuffer input(1, 1);
    ActivationBuffer output(1, 1);
    output.row(0)[0] = 123.f;
    const std::array<ExpertRequest, 1> requests = {ExpertRequest{"missing", &input, &output}};
    auto submission = backend->submit_batch(requests);
    budget_check(submission && submission->reservations().size() == 1
                     && submission->reservations()[0] == ExpertStatus::Failed,
                 "unbudgeted submission selects CPU fallback without a demand retry");
    const auto results = submission->wait();
    budget_check(results.size() == 1 && results[0] == ExpertStatus::Failed
                     && output.row(0)[0] == 123.f && child->submissions == 0,
                 "unbudgeted submission neither executes nor publishes output");
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::test_resource_exact_fit_and_cache_reserve();
        ncnn::moe::test_resource_live_usage_and_device_bounds();
        ncnn::moe::test_resource_concurrency_and_peak_workspace();
        ncnn::moe::test_resource_overflow_and_state_only_requests();
        ncnn::moe::test_live_budget();
        ncnn::moe::test_unbudgeted_device_fallback();
        ncnn::moe::test_key_sharded_routing();
        std::cout << "Native quantized GPU budget tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Native quantized GPU budget test failed: " << error.what() << '\n';
        return 1;
    }
}
