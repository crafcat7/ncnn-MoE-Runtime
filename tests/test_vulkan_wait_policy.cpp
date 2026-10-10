#include "backends/ncnn/vulkanwaitpolicy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace moe {

enum class TestProbeResult
{
    Success,
    OutOfHostMemory,
    OutOfDeviceMemory,
    Unknown,
    DeviceLost,
    NoQueue
};

static void wait_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct VulkanWaitTestSequence
{
    const std::string& name;
    std::initializer_list<int> waits;
    std::initializer_list<TestProbeResult> probes;
    bool probe_device_present;
    const std::weak_ptr<int>& owner;
    const size_t& releases;
    const bool& recording_scope_retained;
    size_t wait_calls = 0;
    size_t probe_calls = 0;
    size_t pause_calls = 0;
    size_t queue_waits = 0;

    void check_retained() const
    {
        wait_check(!owner.expired() && releases == 0, name + ": caller resource released while submission can be pending");
        wait_check(recording_scope_retained, name + ": caller recording scope escaped during wait retry");
    }
};

static void check_wait_sequence(const char* label,
                                std::initializer_list<int> waits,
                                std::initializer_list<TestProbeResult> probes,
                                int expected_result,
                                size_t expected_waits,
                                size_t expected_probes,
                                size_t expected_pauses,
                                size_t expected_queue_waits,
                                bool probe_device_present = true)
{
    const std::string name(label);
    size_t releases = 0;
    bool recording_scope_retained = true;
    auto owner = std::shared_ptr<int>(new int(17), [&releases](int* value) {
        ++releases;
        delete value;
    });
    const std::weak_ptr<int> observed_owner = owner;
    VulkanWaitTestSequence sequence{name, waits, probes, probe_device_present, observed_owner, releases, recording_scope_retained};
    const VulkanCompletionCallbacks callbacks{
        &sequence,
        [](void* context) {
            auto& sequence = *static_cast<VulkanWaitTestSequence*>(context);
            sequence.check_retained();
            wait_check(sequence.wait_calls < sequence.waits.size(), sequence.name + ": unexpected original-fence wait");
            return sequence.waits.begin()[sequence.wait_calls++];
        },
        [](void* context) {
            auto& sequence = *static_cast<VulkanWaitTestSequence*>(context);
            sequence.check_retained();
            ++sequence.probe_calls;
            if (!sequence.probe_device_present)
                return VulkanCompletionProbeResult::Retry;
            wait_check(sequence.probe_calls <= sequence.probes.size(), sequence.name + ": unexpected loss probe");
            const TestProbeResult probe = sequence.probes.begin()[sequence.probe_calls - 1];
            if (probe == TestProbeResult::NoQueue)
                return VulkanCompletionProbeResult::Retry;
            ++sequence.queue_waits;
            return classify_vulkan_completion_probe(static_cast<int>(probe), static_cast<int>(TestProbeResult::DeviceLost));
        },
        [](void* context) {
            auto& sequence = *static_cast<VulkanWaitTestSequence*>(context);
            sequence.check_retained();
            ++sequence.pause_calls;
        }};
    const int result = wait_for_vulkan_submission_completion(callbacks);

    sequence.check_retained();
    wait_check(result == expected_result, name + ": completion result");
    wait_check(sequence.wait_calls == expected_waits, name + ": original-fence waits");
    wait_check(sequence.probe_calls == expected_probes, name + ": loss probes");
    wait_check(sequence.pause_calls == expected_pauses, name + ": transient-error pauses");
    wait_check(sequence.queue_waits == expected_queue_waits, name + ": exclusively leased queue probes");
    recording_scope_retained = false;
    owner.reset();
    wait_check(observed_owner.expired() && releases == 1, name + ": resources released after safe completion only");
}

struct VulkanQueueDrainTestSequence
{
    const char* label;
    std::array<std::array<TestProbeResult, 3>, 2> results;
    std::array<size_t, 2> result_counts;
    std::array<size_t, 2> wait_calls{};
    std::array<bool, 2> leased{};
    std::array<bool, 2> completed{};
    size_t acquisitions = 0;
    size_t reclaims = 0;
    size_t pauses = 0;
    bool owners_retained = true;
};

static void check_queue_family_drain(const char* label,
                                     std::array<std::array<TestProbeResult, 3>, 2> results,
                                     std::array<size_t, 2> counts,
                                     size_t expected_pauses)
{
    VulkanQueueDrainTestSequence sequence{label, results, counts};
    const VulkanQueueFamilyDrainCallbacks callbacks{
        &sequence, sequence.leased.size(),
        [](void* context, size_t index) {
            auto& sequence = *static_cast<VulkanQueueDrainTestSequence*>(context);
            wait_check(sequence.owners_retained && sequence.reclaims == 0, std::string(sequence.label) + ": owners retained while acquiring unknown original queues");
            wait_check(index == sequence.acquisitions && !sequence.leased[index], std::string(sequence.label) + ": all distinct queues acquired once");
            sequence.leased[index] = true;
            ++sequence.acquisitions;
        },
        [](void* context, size_t index) {
            auto& sequence = *static_cast<VulkanQueueDrainTestSequence*>(context);
            wait_check(sequence.owners_retained && sequence.reclaims == 0, std::string(sequence.label) + ": idle failure retains command and allocator owners");
            wait_check(sequence.acquisitions == sequence.leased.size(), std::string(sequence.label) + ": acquire whole family before any idle wait");
            for (bool leased : sequence.leased)
                wait_check(leased, std::string(sequence.label) + ": queue host access remains exclusive throughout idle waits");
            wait_check(sequence.wait_calls[index] < sequence.result_counts[index], std::string(sequence.label) + ": expected idle retry count");
            const TestProbeResult result = sequence.results[index][sequence.wait_calls[index]++];
            sequence.completed[index] = result == TestProbeResult::Success || result == TestProbeResult::DeviceLost;
            return static_cast<int>(result);
        },
        [](void* context, size_t index) {
            auto& sequence = *static_cast<VulkanQueueDrainTestSequence*>(context);
            wait_check(sequence.owners_retained && sequence.leased[index], std::string(sequence.label) + ": queue reclaim retains resources");
            for (bool completed : sequence.completed)
                wait_check(completed, std::string(sequence.label) + ": reclaim only after entire family is safely settled");
            wait_check(index == sequence.reclaims, std::string(sequence.label) + ": reclaim each queue exactly once");
            sequence.leased[index] = false;
            ++sequence.reclaims;
        },
        [](void* context) {
            auto& sequence = *static_cast<VulkanQueueDrainTestSequence*>(context);
            wait_check(sequence.owners_retained && sequence.reclaims == 0, std::string(sequence.label) + ": transient errors retain all leases and resources");
            ++sequence.pauses;
        }};
    drain_vulkan_queue_family(callbacks, static_cast<int>(TestProbeResult::Success), static_cast<int>(TestProbeResult::DeviceLost));
    wait_check(sequence.wait_calls == counts, std::string(label) + ": every original-queue possibility waited safely");
    wait_check(sequence.acquisitions == sequence.leased.size() && sequence.reclaims == sequence.leased.size(), std::string(label) + ": complete family lease lifecycle");
    wait_check(sequence.pauses == expected_pauses, std::string(label) + ": expected OOM/UNKNOWN retry pauses");
    sequence.owners_retained = false;
}

static void check_queue_family_drain_policy()
{
    check_queue_family_drain("legacy family idle success", {{{TestProbeResult::Success}, {TestProbeResult::Success}}}, {1, 1}, 0);
    check_queue_family_drain("legacy family OOM/UNKNOWN retries", {{{TestProbeResult::OutOfHostMemory, TestProbeResult::Unknown, TestProbeResult::Success}, {TestProbeResult::OutOfDeviceMemory, TestProbeResult::Success}}}, {3, 2}, 3);
    check_queue_family_drain("legacy family device loss permits safe destruction", {{{TestProbeResult::OutOfDeviceMemory, TestProbeResult::DeviceLost}, {TestProbeResult::Unknown, TestProbeResult::DeviceLost}}}, {2, 2}, 2);
}

static int run_vulkan_wait_policy_tests()
{
    try
    {
        check_wait_sequence("first wait succeeds", {0}, {}, 0, 1, 0, 0, 0);
        check_wait_sequence("OOM then idle success must retry original fence", {-1, 0}, {TestProbeResult::Success}, 0, 2, 1, 1, 1);
        check_wait_sequence("probe OOM and unknown remain pending", {-1, -1, -1, 0},
                            {TestProbeResult::OutOfHostMemory, TestProbeResult::OutOfDeviceMemory, TestProbeResult::Unknown}, 0, 4, 3, 3, 3);
        check_wait_sequence("only confirmed device loss terminates failed wait", {-1, -7, 0},
                            {TestProbeResult::Success, TestProbeResult::DeviceLost}, -7, 2, 2, 1, 2);
        check_wait_sequence("first probe confirms device loss", {-1, 0}, {TestProbeResult::DeviceLost}, -1, 1, 1, 0, 1);
        check_wait_sequence("missing device probe keeps original wait retrying", {-1, -1, 0}, {}, 0, 3, 2, 2, 0, false);
        check_wait_sequence("no probe queue keeps original wait retrying", {-1, -1, 0},
                            {TestProbeResult::NoQueue, TestProbeResult::NoQueue}, 0, 3, 2, 2, 0);
        check_queue_family_drain_policy();
        std::cout << "Vulkan completion retry and queue drain policies passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

} // namespace moe
} // namespace ncnn

int main()
{
    return ncnn::moe::run_vulkan_wait_policy_tests();
}
