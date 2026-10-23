#include "backends/ncnn/expertbackend_vulkan.h"
#include "engine/expertbackend.h"

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

static void test_live_budget()
{
    // Three layers each select two 5-byte Experts. Their stable routes need
    // 30 bytes across the decode, while the previous one-layer wave needs 10.
    constexpr uint64_t all_experts = 120;
    constexpr uint64_t largest_expert = 5;
    budget_check(native_quantized_gpu_cache_size(80, all_experts, largest_expert) == 80,
                 "live budget retains all layers' active sets and additional routes");
    budget_check(native_quantized_gpu_cache_size(30, all_experts, largest_expert) == 30,
                 "exact active-set capacity does not shrink to one layer");
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
    void admit(std::string, std::shared_ptr<const TensorData>, const TensorData*,
               std::shared_ptr<const TensorData>, const TensorData*, uint32_t,
               float, ExpertActivation) override
    {
        ++admissions;
    }

    void admit_separate(std::string, std::shared_ptr<const TensorData>, const TensorData*,
                        std::shared_ptr<const TensorData>, const TensorData*,
                        std::shared_ptr<const TensorData>, const TensorData*, uint32_t,
                        float, ExpertActivation) override
    {
        ++admissions;
    }

    bool prepare_demand(std::string, std::shared_ptr<const TensorData>, const TensorData*,
                        std::shared_ptr<const TensorData>, const TensorData*, uint32_t,
                        float, ExpertActivation, std::shared_ptr<const void>& pin) override
    {
        ++demands;
        pin = std::make_shared<int>(1);
        return true;
    }

    bool prepare_demand_separate(std::string, std::shared_ptr<const TensorData>, const TensorData*,
                                 std::shared_ptr<const TensorData>, const TensorData*,
                                 std::shared_ptr<const TensorData>, const TensorData*, uint32_t,
                                 float, ExpertActivation, std::shared_ptr<const void>& pin) override
    {
        ++demands;
        pin = std::make_shared<int>(1);
        return true;
    }

    std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override
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
        backend->admit(keys[index], weight, nullptr, weight, nullptr, group, 0.f, ExpertActivation::Silu);
        backend->admit_separate(keys[index], weight, nullptr, weight, nullptr, weight, nullptr,
                                group, 0.f, ExpertActivation::Silu);
        std::shared_ptr<const void> pin;
        budget_check(backend->prepare_demand(keys[index], weight, nullptr, weight, nullptr,
                                             group, 0.f, ExpertActivation::Silu, pin)
                         && pin,
                     "key-sharded demand prepares its hash-selected child");
        budget_check(backend->prepare_demand_separate(keys[index], weight, nullptr, weight, nullptr, weight, nullptr,
                                                      group, 0.f, ExpertActivation::Silu, pin)
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
    const std::array<ExpertBackendRequest, 2> requests = {
        ExpertBackendRequest{keys[0], &input, &outputs[0]},
        ExpertBackendRequest{keys[1], &input, &outputs[1]},
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
    backend->admit("missing", weight, nullptr, weight, nullptr, 1, 0.f, ExpertActivation::Silu);
    backend->admit_separate("missing-separate", weight, nullptr, weight, nullptr, weight, nullptr, 1, 0.f, ExpertActivation::Silu);
    budget_check(child->admissions == 0, "unbudgeted device must not upload to a different device");
    backend->admit("resident", weight, nullptr, weight, nullptr, 0, 0.f, ExpertActivation::Silu);
    budget_check(child->admissions == 1, "budgeted device keeps normal admission");

    std::shared_ptr<const void> pin;
    budget_check(!backend->prepare_demand_separate("missing", weight, nullptr, weight, nullptr, weight, nullptr,
                                                   1, 0.f, ExpertActivation::Silu, pin)
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
    const std::array<ExpertBackendRequest, 1> requests = {ExpertBackendRequest{"missing", &input, &output}};
    auto submission = backend->submit_batch(requests);
    budget_check(submission && submission->reservations().size() == 1
                     && submission->reservations()[0] == ExpertBackendExecutionResult::NotResident,
                 "unbudgeted submission reserves CPU fallback");
    const auto results = submission->wait();
    budget_check(results.size() == 1 && results[0] == ExpertBackendExecutionResult::NotResident
                     && output.row(0)[0] == 123.f && child->submissions == 0,
                 "unbudgeted submission neither executes nor publishes output");
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
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
