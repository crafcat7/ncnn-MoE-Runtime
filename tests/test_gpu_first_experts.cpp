#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/moecombine_vulkan.h"
#include "backends/ncnn/vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "backends/ncnn/experttransfer_vulkan.h"
#include "storage/expertresidency.h"
#include "engine/expertbackend.h"
#include "engine/expert.h"
#include "engine/sessionstate.h"
#include "graph/compiledmodel.h"
#include "ncnn/moe/session.h"
#include "kernels/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <future>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ncnn {
namespace moe {

static void gpu_first_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

#if NCNN_MOE_WITH_VULKAN
static std::shared_ptr<const TensorData> gpu_first_weight(uint32_t rows, uint32_t columns, uint32_t seed)
{
    auto weight = std::make_shared<TensorData>();
    weight->dtype = DType::MxFp4;
    weight->shape = {rows, columns};
    weight->mxfp4_blocks.resize(static_cast<size_t>(rows) * columns / 2);
    weight->mxfp4_scales.resize(static_cast<size_t>(rows) * columns / 32);
    std::fill_n(weight->mxfp4_scales.data(), weight->mxfp4_scales.size(), uint8_t(123));
    for (size_t index = 0; index < weight->mxfp4_blocks.size(); ++index)
    {
        const uint8_t low = static_cast<uint8_t>((index * 7 + seed * 3 + index / 13) % 16);
        const uint8_t high = static_cast<uint8_t>((index * 11 + seed * 5 + index / 17) % 16);
        weight->mxfp4_blocks[index] = static_cast<uint8_t>(low | high << 4);
    }
    return weight;
}

struct GpuFirstFixture
{
    std::shared_ptr<const TensorData> gate = gpu_first_weight(64, 32, 1);
    std::shared_ptr<const TensorData> down = gpu_first_weight(32, 32, 3);
    ActivationBuffer input{1, 32};
    ActivationBuffer expected;
    uint64_t size = 0;
    const uint64_t flags = OptimizationDefaultFlags;
    VulkanRuntimePtr runtime = create_vulkan_runtime();

    GpuFirstFixture()
    {
        size = gate->mxfp4_blocks.size() + gate->mxfp4_scales.size()
               + down->mxfp4_blocks.size() + down->mxfp4_scales.size();
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(0)[column] = static_cast<float>(static_cast<int>(column % 11) - 5) * 0.015625f;
        ActivationBuffer activated;
        forward_gate_up_mxfp4(*gate, nullptr, input, ExpertActivation::DeepSeekSwiGlu, 4.0f, activated, flags);
        expected = forward_linear(*down, activated, flags);
    }

    bool demand(ExpertBackend& backend, const std::string& key, std::shared_ptr<const void>& pin) const
    {
        return backend.prepare_demand(key, gate, nullptr, down, nullptr, 0, 4.0f,
                                      ExpertActivation::DeepSeekSwiGlu, pin);
    }

    void check_output(const ActivationBuffer& actual) const
    {
        gpu_first_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), "GPU demand output shape");
        float maximum = 0.0f;
        for (size_t index = 0; index < expected.values().size(); ++index)
        {
            maximum = std::max(maximum, std::abs(expected.values()[index]));
            gpu_first_check(std::isfinite(actual.values()[index])
                                && std::abs(actual.values()[index] - expected.values()[index]) < 0.0002f,
                            "GPU demand output differs from CPU MXFP4 oracle");
        }
        gpu_first_check(maximum > 0.00001f, "GPU demand CPU oracle must be nonzero");
    }
};

static void gpu_first_execute(const GpuFirstFixture& fixture, ExpertBackend& backend,
                              const std::string& key, std::shared_ptr<const void>& pin)
{
    ActivationBuffer output(1, 32);
    std::fill_n(output.row(0), output.columns(), -123.0f);
    ExpertBackendRequest request{key, &fixture.input, &output, fixture.size, {}};
    auto submission = backend.submit_batch(std::span<const ExpertBackendRequest>(&request, 1));
    gpu_first_check(submission && submission->reservations().size() == 1
                        && submission->reservations()[0] == ExpertBackendExecutionResult::Executed,
                    "cold current-wave request must reserve GPU after demand upload");
    pin.reset();
    gpu_first_check(output.values()[0] == -123.0f, "GPU output is private before commit");
    const auto results = submission->wait();
    gpu_first_check(results.size() == 1 && results[0] == ExpertBackendExecutionResult::Executed,
                    "cold current-wave request must execute on GPU");
    gpu_first_check(output.values()[0] == -123.0f, "wait cannot publish GPU output");
    gpu_first_check(submission->commit(), "GPU demand commit");
    fixture.check_output(output);
}

static void gpu_first_cold_and_capacity()
{
    GpuFirstFixture fixture;
    auto backend = create_vulkan_expert_backend(fixture.size, 0, {}, fixture.runtime, fixture.flags);
    gpu_first_check(static_cast<bool>(backend), "GPU demand backend creation");
    std::shared_ptr<const void> pin;
    {
        ScopedExpertBackendForeground foreground(backend);
        gpu_first_check(fixture.demand(*backend, "cold-A", pin) && pin,
                        "foreground cold demand must complete without waiting for background queue");
    }
    auto stats = backend->statistics();
    gpu_first_check(stats.bytes_uploaded == fixture.size
                        && stats.resident_size == fixture.size
                        && stats.pending_size == 0,
                    "cold GPU demand bounded upload accounting");

    std::shared_ptr<const void> blocked_pin;
    gpu_first_check(!fixture.demand(*backend, "cold-B", blocked_pin) && !blocked_pin,
                    "pinned incumbent cannot be evicted to overfill one-pair cache");
    stats = backend->statistics();
    gpu_first_check(stats.bytes_uploaded == fixture.size
                        && stats.evictions == 0,
                    "capacity failure cannot upload or evict pinned GPU work");
    pin.reset();
    gpu_first_check(fixture.demand(*backend, "cold-B", pin) && pin, "released cache capacity must admit next cold demand");
    stats = backend->statistics();
    gpu_first_check(stats.evictions == 1 && stats.resident_size == fixture.size
                        && stats.pending_size == 0 && stats.bytes_uploaded == fixture.size * 2,
                    "demand eviction stays within cache budget");
    gpu_first_execute(fixture, *backend, "cold-B", pin);
    gpu_first_check(backend->statistics().executions == 1, "cold demand actually executes on GPU");

    auto too_small = create_vulkan_expert_backend(fixture.size - 1, 0, {}, fixture.runtime, fixture.flags);
    gpu_first_check(too_small && !fixture.demand(*too_small, "oversize", pin) && !pin,
                    "Expert pair larger than cache must permit CPU fallback");
    gpu_first_check(too_small->statistics().bytes_uploaded == 0 && too_small->statistics().pending_size == 0,
                    "oversize demand does not allocate GPU weights or retain pending host data");

    auto unsupported = std::make_shared<TensorData>();
    unsupported->dtype = DType::Float32;
    unsupported->shape = {64, 32};
    unsupported->float32_data.assign(64 * 32, 1.0f);
    gpu_first_check(!backend->prepare_demand("unsupported", unsupported, nullptr, fixture.down, nullptr,
                                             0, 4.0f, ExpertActivation::DeepSeekSwiGlu, pin)
                        && !pin,
                    "unsupported demand must permit CPU fallback");

    const auto before_invalid = backend->statistics();
    auto malformed = std::make_shared<TensorData>();
    malformed->dtype = DType::MxFp4;
    malformed->shape = {64, 32};
    malformed->mxfp4_blocks.resize(64 * 16);
    malformed->mxfp4_scales.resize(63); // A valid gate needs 64 scales.
    gpu_first_check(!backend->prepare_demand("invalid-payload", malformed, nullptr, fixture.down, nullptr,
                                             0, 4.0f, ExpertActivation::DeepSeekSwiGlu, pin)
                        && !pin,
                    "malformed payload is rejected before demand eviction");
    TensorData malformed_bias;
    malformed_bias.dtype = DType::Float32;
    malformed_bias.shape = {64};
    malformed_bias.float32_data.assign(63, 0.0f);
    gpu_first_check(!backend->prepare_demand("invalid-bias", fixture.gate, &malformed_bias, fixture.down, nullptr,
                                             0, 4.0f, ExpertActivation::DeepSeekSwiGlu, pin)
                        && !pin,
                    "malformed bias is rejected before demand eviction");
    const auto after_invalid = backend->statistics();
    gpu_first_check(after_invalid.evictions == before_invalid.evictions
                        && after_invalid.bytes_uploaded == before_invalid.bytes_uploaded
                        && after_invalid.resident_size == before_invalid.resident_size
                        && after_invalid.pending_size == 0,
                    "invalid demand preserves previously resident GPU weights and accounting");
}

static void gpu_first_duplicate_and_shard()
{
    GpuFirstFixture fixture;
    auto backend = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    gpu_first_check(static_cast<bool>(backend), "parallel GPU demand backend");
    std::array<std::shared_ptr<const void>, 2> pins;
    std::array<uint8_t, 2> success{};
    std::array<std::thread, 2> threads;
    for (size_t index = 0; index < threads.size(); ++index)
        threads[index] = std::thread([&, index] { success[index] = fixture.demand(*backend, "same-key", pins[index]); });
    for (auto& thread : threads)
        thread.join();
    gpu_first_check(success[0] && success[1] && pins[0] && pins[1], "concurrent demands share completed GPU pair");
    auto stats = backend->statistics();
    gpu_first_check(stats.bytes_uploaded == fixture.size
                        && stats.pending_size == 0,
                    "concurrent demand uploads exactly once");
    pins = {};
    backend->admit("queued", fixture.gate, nullptr, fixture.down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu);
    std::shared_ptr<const void> pin;
    {
        ScopedExpertBackendForeground foreground(backend);
        gpu_first_check(fixture.demand(*backend, "queued", pin) && pin, "queued admission can be completed by foreground demand");
    }
    backend->wait_for_background_work();
    stats = backend->statistics();
    gpu_first_check(stats.bytes_uploaded == fixture.size * 2 && stats.pending_size == 0
                        && stats.resident_size <= fixture.size * 2,
                    "background and demand admission do not duplicate weight uploads");
    gpu_first_execute(fixture, *backend, "queued", pin);
    {
        ScopedExpertBackendForeground foreground(backend);
        backend->admit("explicit-drain", fixture.gate, nullptr, fixture.down, nullptr,
                       0, 4.0f, ExpertActivation::DeepSeekSwiGlu);
        backend->wait_for_background_work();
        gpu_first_check(backend->statistics().pending_size == 0,
                        "explicit background drain must complete inside foreground scope");
    }

    auto child = create_vulkan_expert_backend(fixture.size, 0, {}, fixture.runtime, fixture.flags);
    auto shard = std::make_shared<MultiDeviceExpertBackend>(std::vector<std::shared_ptr<ExpertBackend>>{child},
                                                            std::vector<uint32_t>{0}, std::vector<uint32_t>{0}, false);
    gpu_first_check(shard && fixture.demand(*shard, "shard", pin) && pin, "multi-device demand routes to owning shard");
    gpu_first_execute(fixture, *shard, "shard", pin);
    stats = shard->statistics();
    gpu_first_check(stats.bytes_uploaded == fixture.size && stats.executions == 1,
                    "multi-device demand uploads one resident pair and executes it");
}
static void gpu_first_execute_batch(const GpuFirstFixture& fixture, ExpertBackend& backend,
                                    std::span<const std::string> keys,
                                    std::span<std::shared_ptr<const void>> pins)
{
    std::vector<ActivationBuffer> outputs(keys.size());
    std::vector<ExpertBackendRequest> requests;
    requests.reserve(keys.size());
    for (size_t index = 0; index < keys.size(); ++index)
    {
        outputs[index].reset(1, 32, false);
        std::fill_n(outputs[index].row(0), 32, -123.0f);
        requests.push_back({keys[index], &fixture.input, &outputs[index], fixture.size, {}});
    }
    auto submission = backend.submit_batch(requests);
    gpu_first_check(submission && submission->reservations().size() == keys.size(), "prepared batch reserves GPU requests");
    for (auto result : submission->reservations())
        gpu_first_check(result == ExpertBackendExecutionResult::Executed, "every demand prefix request is resident");
    for (auto& pin : pins) pin.reset();
    const auto results = submission->wait();
    gpu_first_check(results.size() == keys.size(), "prepared batch execution result count");
    for (size_t index = 0; index < keys.size(); ++index)
    {
        gpu_first_check(results[index] == ExpertBackendExecutionResult::Executed, "prepared batch actually executes on GPU");
        gpu_first_check(outputs[index].values()[0] == -123.0f, "demand batch output remains private before commit");
    }
    gpu_first_check(submission->commit(), "demand batch output commit");
    for (const auto& output : outputs) fixture.check_output(output);
}

static void gpu_first_foreground_batch()
{
    GpuFirstFixture fixture;
    std::array<std::string, 9> keys;
    std::array<ExpertDemandRequest, 9> requests;
    for (size_t index = 0; index < keys.size(); ++index)
    {
        keys[index] = "batch-" + std::to_string(index);
        requests[index] = {keys[index], fixture.gate, nullptr, fixture.down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu};
    }
    std::array<std::shared_ptr<const void>, 9> pins;
    auto backend = create_vulkan_expert_backend(fixture.size * 9, 0, {}, fixture.runtime, fixture.flags);
    ScopedExpertBackendForeground foreground(backend);
    gpu_first_check(backend->prepare_demand_batch(requests, pins) == 8, "foreground upload group has an eight-request bound");
    for (size_t index = 0; index < 8; ++index) gpu_first_check(static_cast<bool>(pins[index]), "prepared prefix pins resident weights");
    gpu_first_check(!pins.back(), "batch bound leaves unprepared tail pin empty");
    auto stats = backend->statistics();
    gpu_first_check(stats.pending_size == 0
                        && stats.resident_size == fixture.size * 8,
                    "eight cold experts become resident within the cache budget");
    gpu_first_execute_batch(fixture, *backend, std::span(keys).first(8), std::span(pins).first(8));
    gpu_first_check(backend->prepare_demand_batch(std::span(requests).last(1), std::span(pins).last(1)) == 1,
                    "bounded upload tail can be prepared in another wave");
    gpu_first_execute_batch(fixture, *backend, std::span(keys).last(1), std::span(pins).last(1));
    stats = backend->statistics();
    gpu_first_check(stats.executions == 9,
                    "bounded batches execute every prepared expert");

    auto small = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    ScopedExpertBackendForeground small_foreground(small);
    gpu_first_check(small->prepare_demand_batch(std::span(requests).first(3), std::span(pins).first(3)) == 2,
                    "small GPU cache returns a resident capacity-limited prefix");
    gpu_first_check(pins[0] && pins[1] && !pins[2], "capacity prefix leaves its tail unpublished");
    gpu_first_check(small->prepare_demand_batch(std::span(requests).subspan(2, 1), std::span(pins).subspan(2, 1)) == 0,
                    "held current-wave pins prevent oversubscribing a small cache");
    gpu_first_check(small->statistics().resident_size == fixture.size * 2,
                    "capacity retry cannot overfill the resident cache");
    gpu_first_execute_batch(fixture, *small, std::span(keys).first(2), std::span(pins).first(2));
    gpu_first_check(small->prepare_demand_batch(std::span(requests).subspan(2, 1), std::span(pins).subspan(2, 1)) == 1,
                    "flushed GPU wave releases capacity for the next prefix");
    gpu_first_execute_batch(fixture, *small, std::span(keys).subspan(2, 1), std::span(pins).subspan(2, 1));
    gpu_first_check(small->statistics().resident_size <= fixture.size * 2
                        && small->statistics().pending_size == 0
                        && small->statistics().executions == 3,
                    "capacity-wave retry remains GPU-first and bounded");

    // Releasing tail hit pins must not make an un-evicted hit look cold.
    // Evicting one double-size pair leaves room for the new small pair and
    // the remaining small hit, while a third large request needs another wave.
    auto varied = create_vulkan_expert_backend(fixture.size * 3, 0, {}, fixture.runtime, fixture.flags);
    auto big_gate = gpu_first_weight(128, 32, 17);
    auto big_down = gpu_first_weight(32, 64, 19);
    std::shared_ptr<const void> varied_pin;
    gpu_first_check(varied->prepare_demand("varied-large", big_gate, nullptr, big_down, nullptr, 0, 4.0f,
                                           ExpertActivation::DeepSeekSwiGlu, varied_pin),
                    "unequal-size cache fixture large pair");
    varied_pin.reset();
    gpu_first_check(fixture.demand(*varied, keys[1], varied_pin), "unequal-size cache fixture small hit");
    varied_pin.reset();
    std::array<ExpertDemandRequest, 3> varied_requests = {requests[0], requests[1],
                                                          ExpertDemandRequest{"varied-large", big_gate, nullptr, big_down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu}};
    const auto varied_before = varied->statistics();
    gpu_first_check(varied->prepare_demand_batch(varied_requests, std::span(pins).first(3)) == 2
                        && pins[0] && pins[1] && !pins[2],
                    "released surviving cache hit still belongs to prepared prefix");
    const auto varied_after = varied->statistics();
    gpu_first_check(varied_after.bytes_uploaded == varied_before.bytes_uploaded + fixture.size,
                    "released surviving hit is not uploaded or installed twice");
    gpu_first_execute_batch(fixture, *varied, std::span(keys).first(2), std::span(pins).first(2));

    auto duplicate = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    std::array<ExpertDemandRequest, 3> duplicate_requests = {requests[0], requests[0], requests[1]};
    std::array<std::shared_ptr<const void>, 3> duplicate_pins;
    gpu_first_check(duplicate->prepare_demand_batch(duplicate_requests, duplicate_pins) == 3
                        && duplicate_pins[0] == duplicate_pins[1],
                    "duplicate keys inside one demand batch share a single uploaded pair");
    stats = duplicate->statistics();
    gpu_first_check(stats.bytes_uploaded == fixture.size * 2,
                    "duplicate demand uploads each weight pair once");
    duplicate_pins = {};
    std::array<std::array<std::shared_ptr<const void>, 2>, 2> concurrent_pins;
    std::array<size_t, 2> concurrent_counts{};
    std::array<std::thread, 2> threads;
    auto concurrent = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    for (size_t index = 0; index < threads.size(); ++index)
        threads[index] = std::thread([&, index] {
            concurrent_counts[index] = concurrent->prepare_demand_batch(std::span(requests).first(2), concurrent_pins[index]);
        });
    for (auto& thread : threads) thread.join();
    stats = concurrent->statistics();
    gpu_first_check(concurrent_counts[0] == 2
                        && concurrent_counts[1] == 2
                        && stats.pending_size == 0,
                    "concurrent demand batches reuse completed resident pairs");

    auto queued = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    ScopedExpertBackendForeground queued_foreground(queued);
    for (size_t index = 0; index < 2; ++index)
        queued->admit(keys[index], fixture.gate, nullptr, fixture.down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu);
    gpu_first_check(queued->prepare_demand_batch(std::span(requests).first(2), std::span(pins).first(2)) == 2,
                    "foreground batch takes over same-key suspended background admissions");
    stats = queued->statistics();
    gpu_first_check(stats.admissions == 2
                        && stats.pending_size == 0
                        && stats.dropped_admissions == 0,
                    "queue takeover does not duplicate admissions or retain host references");
    gpu_first_execute_batch(fixture, *queued, std::span(keys).first(2), std::span(pins).first(2));

    auto malformed = std::make_shared<TensorData>(*fixture.gate);
    malformed->mxfp4_scales.resize(63);
    std::array<ExpertDemandRequest, 2> invalid_requests = {requests[0], requests[1]};
    invalid_requests[1].gate_up = malformed;
    auto validated = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    gpu_first_check(validated->prepare_demand_batch(invalid_requests, std::span(pins).first(2)) == 1
                        && pins[0] && !pins[1],
                    "malformed batch tail is rejected before publishing that pair");
    stats = validated->statistics();
    gpu_first_check(stats.pending_size == 0,
                    "invalid batch tail preserves its valid contiguous prefix");
    pins = {};
    gpu_first_check(validated->prepare_demand_batch(invalid_requests, std::span(pins).first(1)) == 0 && !pins[0],
                    "mismatched descriptor and pin spans are rejected safely");
    auto child = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    MultiDeviceExpertBackend shard({child}, {0}, {0}, false);
    gpu_first_check(shard.prepare_demand_batch(std::span(requests).first(2), std::span(pins).first(2)) == 2,
                    "multi-device wrapper preserves contiguous same-device upload batches");
    gpu_first_execute_batch(fixture, shard, std::span(keys).first(2), std::span(pins).first(2));
    stats = shard.statistics();
    gpu_first_check(stats.executions == 2,
                    "multi-device batch executes both prepared experts");

    // This layout is about 12.75 MiB per pair. An eight-pair payload budget
    // fits the requests but the complete aligned staging layouts exceed 64 MiB.
    auto large_gate = gpu_first_weight(2048, 8192, 7);
    auto large_down = gpu_first_weight(8192, 1024, 11);
    const uint64_t large_size = large_gate->mxfp4_blocks.size() + large_gate->mxfp4_scales.size()
                                + large_down->mxfp4_blocks.size() + large_down->mxfp4_scales.size();
    auto bounded = create_vulkan_expert_backend(large_size * 8, 0, {}, fixture.runtime, fixture.flags);
    for (size_t index = 0; index < 8; ++index)
    {
        requests[index].gate_up = large_gate;
        requests[index].down = large_down;
    }
    const size_t bounded_prefix = bounded->prepare_demand_batch(std::span(requests).first(8), std::span(pins).first(8));
    auto context = VulkanContext::acquire(0, fixture.runtime, fixture.flags);
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    const auto align = [alignment](uint64_t bytes) { return ((bytes + alignment - 1) / alignment) * alignment; };
    const uint64_t complete_size = align(large_gate->mxfp4_blocks.size()) + align(large_gate->mxfp4_scales.size())
                                   + align(static_cast<uint64_t>(large_gate->shape[0]) * sizeof(float))
                                   + align(large_down->mxfp4_blocks.size()) + align(large_down->mxfp4_scales.size())
                                   + align(static_cast<uint64_t>(large_down->shape[0]) * sizeof(float));
    const size_t expected_prefix = std::min<size_t>(8, (UINT64_C(64) * 1024 * 1024) / complete_size);
    gpu_first_check(bounded_prefix == expected_prefix
                        && bounded_prefix > 0
                        && bounded_prefix < 8
                        && bounded->statistics().resident_size == bounded_prefix * large_size,
                    "foreground batch staging cap includes padding and synthesized FP32 bias");
    for (size_t index = bounded_prefix; index < pins.size(); ++index)
        gpu_first_check(!pins[index], "byte cap leaves unprepared tail empty");
}

static void gpu_first_joint_eviction_window()
{
    GpuFirstFixture fixture;
    static constexpr size_t cache_entries = 6;
    // Exercise both ARC classes. Turning off the device preference leaves
    // the host registry active, so this is an independent GPU policy ablation.
    for (bool frequent : {false, true})
    {
        for (bool use_joint_eviction : {false, true})
        {
            auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
            const uint64_t flags = use_joint_eviction
                                       ? fixture.flags | OptimizationVulkanJointCacheEviction
                                       : fixture.flags & ~OptimizationVulkanJointCacheEviction;
            auto backend = create_vulkan_expert_backend(fixture.size * cache_entries, 0, {}, fixture.runtime, flags);
            gpu_first_check(static_cast<bool>(backend), "joint window backend");
            backend->set_residency_coordinator(coordinator);
            std::array<std::string, cache_entries> keys;
            for (size_t index = 0; index < keys.size(); ++index)
                keys[index] = "joint-window-" + std::to_string(index);
            auto newest_host = coordinator->register_host(keys.back());
            std::shared_ptr<const void> pin;
            for (const auto& key : keys)
            {
                gpu_first_check(fixture.demand(*backend, key, pin) && pin, "joint window fixture admission");
                if (frequent)
                    gpu_first_execute(fixture, *backend, key, pin);
                else
                    pin.reset();
            }
            auto stats = backend->statistics();
            gpu_first_check((frequent ? stats.arc_frequent_size : stats.arc_recent_size) == fixture.size * cache_entries,
                            "joint window fixture uses the requested ARC class");
            gpu_first_check(fixture.demand(*backend, "joint-window-incoming", pin) && pin,
                            "joint window pressure admission");
            gpu_first_check(!coordinator->device_resident(keys.front()) && coordinator->device_resident(keys.back()),
                            "newest host-backed GPU weight outside the four-candidate window stays resident");
            stats = backend->statistics();
            gpu_first_check(stats.evictions == 1
                                && stats.resident_size == fixture.size * cache_entries,
                            "recency-limited eviction drops the oldest unique GPU weight within budget");
            if (frequent)
                gpu_first_execute(fixture, *backend, "joint-window-incoming", pin);
            else
                pin.reset();

            auto nearby_host = coordinator->register_host(keys[2]);
            gpu_first_check(fixture.demand(*backend, "joint-window-next", pin) && pin,
                            "joint window nearby pressure admission");
            const size_t evicted_index = use_joint_eviction ? 2 : 1;
            const size_t retained_index = use_joint_eviction ? 1 : 2;
            gpu_first_check(!coordinator->device_resident(keys[evicted_index])
                                && coordinator->device_resident(keys[retained_index])
                                && coordinator->device_resident(keys.back()),
                            "nearby host-backed preference is bounded and can be disabled independently");
            stats = backend->statistics();
            gpu_first_check(stats.evictions == 2
                                && stats.resident_size == fixture.size * cache_entries
                                && stats.pending_size == 0,
                            "joint window eviction preserves cache capacity");
            pin.reset();
            backend.reset();
            for (const auto& key : keys)
                gpu_first_check(!coordinator->device_resident(key), "joint window destruction removes device registrations");
            gpu_first_check(!coordinator->device_resident("joint-window-incoming")
                                && !coordinator->device_resident("joint-window-next")
                                && coordinator->host_resident(keys.back()) && coordinator->host_resident(keys[2]),
                            "joint window destruction preserves host registrations");
            newest_host.reset();
            nearby_host.reset();
            for (const auto& key : keys)
                gpu_first_check(!coordinator->host_resident(key), "joint window registration lifetimes settle");
        }
    }

    // Pinned entries are ineligible and do not consume a candidate slot.
    auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
    auto backend = create_vulkan_expert_backend(fixture.size * cache_entries, 0, {}, fixture.runtime, fixture.flags);
    backend->set_residency_coordinator(coordinator);
    std::array<std::string, cache_entries> keys;
    for (size_t index = 0; index < keys.size(); ++index)
        keys[index] = "joint-pinned-" + std::to_string(index);
    auto newest_host = coordinator->register_host(keys.back());
    auto pinned_host = coordinator->register_host(keys.front());
    std::shared_ptr<const void> pinned;
    std::shared_ptr<const void> pin;
    gpu_first_check(fixture.demand(*backend, keys.front(), pinned) && pinned, "joint window pinned oldest admission");
    for (size_t index = 1; index < keys.size(); ++index)
    {
        gpu_first_check(fixture.demand(*backend, keys[index], pin) && pin, "joint window unpinned fixture admission");
        pin.reset();
    }
    gpu_first_check(fixture.demand(*backend, "joint-pinned-incoming", pin) && pin, "joint window pinned pressure admission");
    gpu_first_check(coordinator->device_resident(keys.front())
                        && !coordinator->device_resident(keys[1])
                        && coordinator->device_resident(keys.back()),
                    "joint window preserves a pinned duplicate and a newer unpinned duplicate");
    pin.reset();
    pinned.reset();
    backend.reset();

    // Fair-share remains ahead of the duplicate tie-break within a class.
    auto grouped = create_vulkan_expert_backend(fixture.size * 4, 0, {}, fixture.runtime, fixture.flags);
    grouped->set_residency_coordinator(coordinator);
    auto other_group_host = coordinator->register_host("joint-group-other");
    auto incoming_group_host = coordinator->register_host("joint-group-nearby");
    const auto demand_group = [&](const std::string& key, uint32_t group) {
        const bool ready = grouped->prepare_demand(key, fixture.gate, nullptr, fixture.down, nullptr, group,
                                                   4.0f, ExpertActivation::DeepSeekSwiGlu, pin);
        gpu_first_check(ready && pin, "joint fair-share fixture admission");
        pin.reset();
    };
    demand_group("joint-group-oldest", 0);
    demand_group("joint-group-other", 1);
    demand_group("joint-group-unique", 1);
    demand_group("joint-group-nearby", 0);
    demand_group("joint-group-incoming", 0);
    gpu_first_check(coordinator->device_resident("joint-group-oldest")
                        && coordinator->device_resident("joint-group-other")
                        && !coordinator->device_resident("joint-group-nearby"),
                    "joint window selects a nearby duplicate from the incoming over-budget group");

    // A tiny cache can still prefer a duplicate, but never a pinned entry.
    auto tiny = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    tiny->set_residency_coordinator(coordinator);
    auto tiny_host = coordinator->register_host("joint-tiny-pinned");
    std::array<std::shared_ptr<const void>, 2> tiny_pins;
    gpu_first_check(fixture.demand(*tiny, "joint-tiny-pinned", tiny_pins[0]) && tiny_pins[0]
                        && fixture.demand(*tiny, "joint-tiny-unique", tiny_pins[1]) && tiny_pins[1],
                    "joint tiny cache fixture pins both entries");
    gpu_first_check(!fixture.demand(*tiny, "joint-tiny-incoming", pin)
                        && !pin
                        && tiny->statistics().evictions == 0,
                    "joint preference cannot evict either pinned entry in a full tiny cache");
    tiny_pins[1].reset();
    gpu_first_check(fixture.demand(*tiny, "joint-tiny-incoming", pin) && pin
                        && coordinator->device_resident("joint-tiny-pinned")
                        && !coordinator->device_resident("joint-tiny-unique")
                        && tiny->statistics().resident_size == fixture.size * 2,
                    "joint tiny cache reclaims only released capacity");
}

static void gpu_first_async_transfer_and_joint_eviction()
{
    GpuFirstFixture fixture;
    auto context = VulkanContext::acquire(0, fixture.runtime, fixture.flags);
    auto backend = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    ScopedExpertBackendForeground foreground(backend);
    std::string mutable_key = "async-original-key";
    TensorData caller_bias;
    caller_bias.dtype = DType::Float32;
    caller_bias.shape = {64};
    caller_bias.float32_data.assign(64, 0.0f);
    ExpertDemandRequest request{mutable_key, fixture.gate, &caller_bias, fixture.down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu};
    std::array<std::shared_ptr<const void>, 1> pins;
    std::unique_lock<std::mutex> foreground_lock(context->command_mutex());
    auto transfer = backend->begin_demand_batch(std::span<const ExpertDemandRequest>(&request, 1));
    mutable_key.assign("mutated-caller-key");
    std::fill(caller_bias.float32_data.begin(), caller_bias.float32_data.end(), std::numeric_limits<float>::quiet_NaN());
    request.gate_up.reset();
    request.down.reset();
    auto settled = std::async(std::launch::async, [&] { return transfer->wait(pins); });
    const bool independent = settled.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    // Unlock even on failure so the regression reports an assertion instead
    // of trapping the draining submission destructor behind its own lock.
    foreground_lock.unlock();
    gpu_first_check(independent && settled.get() == 1 && pins[0],
                    "async cold upload completes while foreground command mutex is held");
    transfer.reset();
    gpu_first_execute(fixture, *backend, "async-original-key", pins[0]);
    auto stats = backend->statistics();
    gpu_first_check(stats.executions == 1
                        && stats.pending_size == 0,
                    "async demand owns caller keys and shared weight references");

    request = {"async-cancelled", fixture.gate, nullptr, fixture.down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu};
    auto cancelled = backend->begin_demand_batch(std::span<const ExpertDemandRequest>(&request, 1));
    cancelled->abort();
    gpu_first_check(cancelled->wait(pins) == 0 && !pins[0], "aborted upload publishes no request pins");
    cancelled.reset();
    backend->wait_for_background_work();
    gpu_first_check(backend->statistics().pending_size == 0, "cancelled uploads settle all reservations");

    auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
    auto joint = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    joint->set_residency_coordinator(coordinator);
    auto host_copy = coordinator->register_host("joint-host-backed");
    std::shared_ptr<const void> pin;
    gpu_first_check(fixture.demand(*joint, "joint-device-only", pin), "joint older device-only entry");
    pin.reset();
    gpu_first_check(fixture.demand(*joint, "joint-host-backed", pin), "joint newer host-backed entry");
    pin.reset();
    gpu_first_check(fixture.demand(*joint, "joint-incoming", pin), "joint pressure admission");
    gpu_first_check(coordinator->device_resident("joint-device-only") && !coordinator->device_resident("joint-host-backed")
                        && coordinator->device_resident("joint-incoming"),
                    "GPU ARC evicts duplicate within same heat/group before unique device copy");
    gpu_first_execute(fixture, *joint, "joint-incoming", pin);
    joint->set_residency_coordinator({});
    gpu_first_check(!coordinator->device_resident("joint-device-only") && !coordinator->device_resident("joint-incoming"),
                    "coordinator detach unregisters completed device cache entries");

    // A source-backed operator may outlive both the victim cache and its
    // explicit lease pin. Its aliasing owner must retain the weight allocator.
    auto victim = std::make_shared<VulkanExpertVictimCache>(context,
                                                            fixture.size + 4 * std::max<size_t>(4, context->device()->info.buffer_offset_alignment()));
    victim->set_residency_coordinator(coordinator);
    ExpertVictimExecutionMetadata execution;
    execution.enabled = true;
    execution.activation_limit = 4.0f;
    execution.activation = ExpertActivation::DeepSeekSwiGlu;
    victim->admit("victim-operator-owner", fixture.gate, fixture.down, execution);
    victim->wait_for_background_work();
    auto lease = victim->find_device_operation("victim-operator-owner");
    gpu_first_check(lease && lease->operation && lease->pin, "victim device operation owns ready GPU weights");
    std::weak_ptr<const void> owner = lease->pin;
    auto retained_operation = lease->operation;
    lease.reset();
    victim.reset();
    gpu_first_check(!owner.expired() && !coordinator->device_resident("victim-operator-owner"),
                    "operator retains allocator after victim lookup residency ends");
    ActivationBuffer retained_output;
    gpu_first_check(retained_operation->forward(fixture.input, retained_output),
                    "source-backed operator executes after victim destruction");
    fixture.check_output(retained_output);
    retained_operation.reset();
    gpu_first_check(owner.expired(), "source-backed operator releases the last allocator owner");

    VulkanExpertWeightAllocator allocator(context->device(), 4096);
    VulkanUploadStagingAllocator staging(context->device(), 4096);
    ncnn::Option option;
    ncnn::Mat unaligned_storage(16, sizeof(uint8_t));
    ncnn::Mat unaligned = unaligned_storage.range(0, 7);
    ncnn::VkMat rejected;
    VulkanWeightUploadBatch invalid(context, &staging);
    gpu_first_check(!invalid.record(unaligned, rejected, option, &allocator) && rejected.empty() && !invalid.submit(),
                    "raw transfer rejects illegal VkBufferCopy byte length");
    std::array<std::thread, 4> allocating;
    for (size_t lane = 0; lane < allocating.size(); ++lane)
        allocating[lane] = std::thread([&, lane] {
            for (size_t round = 0; round < 128; ++round)
            {
                ncnn::VkMat tensor(static_cast<int>(64 + ((round + lane) % 8) * 256), sizeof(uint8_t), &allocator);
                if (tensor.empty()) std::terminate();
            }
        });
    for (auto& lane : allocating) lane.join();
    // The allocator remains usable across concurrent allocations and oversized shapes.
    for (int count : {8192, 16384, 32768, 65536})
    {
        ncnn::VkMat tensor(count, sizeof(uint8_t), &allocator);
        gpu_first_check(!tensor.empty(), "pooled concurrent allocator shape churn");
    }
}

static void gpu_first_context_lifetime()
{
    GpuFirstFixture fixture;
    auto context = VulkanContext::acquire(0, fixture.runtime, fixture.flags);
    gpu_first_check(static_cast<bool>(context), "lifetime context creation");
    std::weak_ptr<VulkanContext> retired = context;
    ncnn::VkAllocator* previous_blob = context->blob_allocator();
    std::array<ncnn::VkAllocator*, 3> previous_staging{};
    previous_staging[0] = context->staging_allocator();
    auto projection = Mxfp4Linear_vulkan::create(*fixture.down, nullptr, 0, fixture.runtime, fixture.flags);
    gpu_first_check(static_cast<bool>(projection), "lifetime cached pipeline creation");
    ActivationBuffer output;
    gpu_first_check(projection->forward(fixture.input, output), "lifetime cached pipeline executes");
    // Default CPU Q8 input quantization is approximate; this shader consumes
    // FP32 inputs, so compare the matching exact CPU projection path.
    const ActivationBuffer expected = forward_linear(*fixture.down, fixture.input,
                                                     fixture.flags & ~OptimizationCpuMxfp4Q8);
    for (size_t column = 0; column < expected.columns(); ++column)
        gpu_first_check(std::abs(output.row(0)[column] - expected.row(0)[column]) < 0.0002f,
                        "lifetime cached pipeline preserves the CPU oracle");
    // Populate every slot tensor, including staging-backed buffers. Their last
    // reference must be released before a subsequent context borrows the pool.
    for (size_t index = 0; index < 2; ++index)
    {
        auto lease = context->acquire_transfer_slot();
        auto& slot = lease.slot();
        previous_staging[index + 1] = slot.staging_allocator;
        slot.upload.create(32, sizeof(float), slot.staging_allocator);
        slot.download.create(32, sizeof(float), slot.staging_allocator);
        std::array<ncnn::VkMat*, 10> tensors = {
            &slot.expert_slots, &slot.expert_row_offsets, &slot.route_offsets,
            &slot.route_rows, &slot.route_weights, &slot.rope_cosine,
            &slot.rope_sine, &slot.attention_mask, &slot.attention_cache_key,
            &slot.attention_cache_value};
        for (ncnn::VkMat* tensor : tensors)
        {
            tensor->create(32, sizeof(float), context->blob_allocator());
            gpu_first_check(!tensor->empty(), "lifetime slot allocation");
        }
        gpu_first_check(!slot.upload.empty() && !slot.download.empty(), "lifetime staging slot allocation");
    }
    context.reset();
    gpu_first_check(!retired.expired(), "live pipeline operation retains its context");
    projection.reset();
    gpu_first_check(retired.expired(), "pipeline weak cache cannot retain a disposed context");
    context = VulkanContext::acquire(0, fixture.runtime, fixture.flags);
    gpu_first_check(context && context->blob_allocator() == previous_blob,
                    "disposed context returns its borrowed blob allocator");
    std::array<ncnn::VkAllocator*, 3> reused_staging{};
    reused_staging[0] = context->staging_allocator();
    for (size_t index = 0; index < 2; ++index)
    {
        auto lease = context->acquire_transfer_slot();
        auto& slot = lease.slot();
        reused_staging[index + 1] = slot.staging_allocator;
        slot.upload.create(32, sizeof(float), slot.staging_allocator);
        gpu_first_check(!slot.upload.empty(), "returned staging allocator is reusable");
    }
    gpu_first_check(std::is_permutation(previous_staging.begin(), previous_staging.end(), reused_staging.begin()),
                    "disposed context returns all borrowed staging allocators");
    retired = context;
    context.reset();
    gpu_first_check(retired.expired(), "reused context releases normally");
}

static void gpu_first_aligned_batch_outputs()
{
    GpuFirstFixture fixture;
    auto down = gpu_first_weight(7, 32, 37);
    ActivationBuffer activated;
    forward_gate_up_mxfp4(*fixture.gate, nullptr, fixture.input, ExpertActivation::DeepSeekSwiGlu,
                          4.0f, activated, fixture.flags);
    const ActivationBuffer expected = forward_linear(*down, activated, fixture.flags);
    const uint64_t pair_bytes = fixture.gate->mxfp4_blocks.size() + fixture.gate->mxfp4_scales.size()
                                + down->mxfp4_blocks.size() + down->mxfp4_scales.size();
    auto backend = create_vulkan_expert_backend(pair_bytes * 2, 0, {}, fixture.runtime, fixture.flags);
    std::array<std::string, 2> keys = {"aligned-output-A", "aligned-output-B"};
    std::array<ExpertDemandRequest, 2> demand;
    for (size_t index = 0; index < 2; ++index)
        demand[index] = {keys[index], fixture.gate, nullptr, down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu};
    std::array<std::shared_ptr<const void>, 2> pins;
    gpu_first_check(backend->prepare_demand_batch(demand, pins) == 2, "odd-width GPU batch demand");

    TensorData identity;
    identity.dtype = DType::Float32;
    identity.shape = {7, 7};
    identity.float32_data.assign(49, 0.0f);
    for (size_t column = 0; column < 7; ++column) identity.float32_data[column * 7 + column] = 1.0f;
    auto identity_op = Linear::create(identity, nullptr, LinearDevice::Vulkan, 0, fixture.runtime, fixture.flags);
    gpu_first_check(static_cast<bool>(identity_op), "odd-width device output consumer");
    const auto foreign_op = Linear::create(identity, nullptr, LinearDevice::Vulkan, 0, create_vulkan_runtime(), fixture.flags);
    gpu_first_check(static_cast<bool>(foreign_op), "completed output foreign consumer");
    for (bool retain : {false, true})
    {
        std::array<ActivationBuffer, 2> outputs;
        std::array<std::shared_ptr<DeviceTensor_vulkan>, 2> device_outputs;
        std::array<ExpertBackendRequest, 2> requests;
        for (size_t index = 0; index < 2; ++index)
        {
            requests[index].key = keys[index];
            requests[index].input = &fixture.input;
            requests[index].output = &outputs[index];
            requests[index].weight_size = pair_bytes;
            if (retain) requests[index].device_output = &device_outputs[index];
        }
        auto submission = backend->submit_batch(requests);
        gpu_first_check(submission && submission->reservations().size() == 2, "odd-width output GPU reservations");
        for (auto result : submission->wait())
            gpu_first_check(result == ExpertBackendExecutionResult::Executed, "odd-width outputs execute on GPU");
        gpu_first_check(submission->commit(), "odd-width output commit");
        submission.reset();
        if (retain)
        {
            for (size_t index = 0; index < 2; ++index)
            {
                auto consumer = CommandGraph_vulkan::create(*identity_op);
                DeviceTensor_vulkan copied;
                ActivationBuffer directly_downloaded;
                gpu_first_check(device_outputs[index] && consumer
                                    && consumer->linear(*identity_op, *device_outputs[index], copied)
                                    && consumer->download(*device_outputs[index], directly_downloaded)
                                    && consumer->download(copied, outputs[index]) && consumer->submit() && consumer->wait(),
                                "intrinsically completed output supports projection and direct download");
                gpu_first_check(directly_downloaded.rows() == outputs[index].rows()
                                    && directly_downloaded.columns() == outputs[index].columns(),
                                "completed direct download shape");
                for (size_t value = 0; value < outputs[index].values().size(); ++value)
                    gpu_first_check(directly_downloaded.values()[value] == outputs[index].values()[value],
                                    "completed direct download equals identity projection");
                auto foreign_consumer = CommandGraph_vulkan::create(*foreign_op);
                gpu_first_check(foreign_consumer && !foreign_consumer->linear(*foreign_op, *device_outputs[index], copied)
                                    && !foreign_consumer->download(*device_outputs[index], directly_downloaded),
                                "intrinsically completed tensor still rejects foreign context");
            }
        }
        for (const auto& output : outputs)
        {
            gpu_first_check(output.rows() == 1 && output.columns() == 7, "odd-width output shape");
            for (size_t column = 0; column < 7; ++column)
                gpu_first_check(std::isfinite(output.row(0)[column])
                                    && std::abs(output.row(0)[column] - expected.row(0)[column]) < 0.0002f,
                                "aligned row output preserves CPU oracle values");
        }
    }
}

static void gpu_first_upload_host_scratch()
{
    GpuFirstFixture fixture;
    auto context = VulkanContext::acquire(0, fixture.runtime, fixture.flags);
    VulkanUploadStagingAllocator pool(context->device(), 4096);
    ncnn::VkWeightAllocator weights(context->device());
    ncnn::Mat scratch;
    ncnn::VkMat first_destination;
    ncnn::VkMat second_destination;
    ncnn::Option option;
    option.use_packing_layout = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.blob_vkallocator = context->blob_allocator();
    option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    void* retained_host_buffer = nullptr;
    {
        VulkanWeightUploadBatch batch(context, &pool, &scratch);
        ncnn::Mat first = batch.host_storage(1024);
        gpu_first_check(!first.empty() && first.total() * first.elemsize == 1024, "bounded host upload scratch allocation");
        retained_host_buffer = first.data;
        std::fill_n(static_cast<uint8_t*>(first.data), 1024, uint8_t(0x31));
        gpu_first_check(batch.record(first, first_destination, option, &weights), "first host scratch source records synchronously");
        ncnn::Mat second = batch.host_storage(512);
        gpu_first_check(second.data == first.data && second.total() * second.elemsize == 512,
                        "smaller matrix reuses scratch with an exact source view");
        std::fill_n(static_cast<uint8_t*>(second.data), 512, uint8_t(0xa7));
        gpu_first_check(batch.record(second, second_destination, option, &weights) && batch.submit(), "two scratch sources share an upload batch");
        gpu_first_check(batch.host_storage(256).empty(), "submitted batch cannot return writable scratch");
    }
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute download(context->device(), context->command_optimization_flags());
        ncnn::Mat first;
        ncnn::Mat second;
        download.record_download(first_destination, first, option);
        download.record_download(second_destination, second, option);
        gpu_first_check(download.submit_and_wait() == 0 && !first.empty() && !second.empty(), "host scratch upload payload download");
        for (size_t index = 0; index < 1024; ++index)
            gpu_first_check(static_cast<const uint8_t*>(first.data)[index] == uint8_t(0x31),
                            "later scratch writes cannot corrupt an earlier recorded upload");
        for (size_t index = 0; index < 512; ++index)
            gpu_first_check(static_cast<const uint8_t*>(second.data)[index] == uint8_t(0xa7), "second scratch upload payload is exact");
    }
    {
        VulkanWeightUploadBatch next(context, &pool, &scratch);
        gpu_first_check(next.host_storage(1024).data == retained_host_buffer, "host scratch persists across successive demand batches");
        ncnn::Mat transient = next.host_storage(17 * 1024 * 1024);
        gpu_first_check(!transient.empty() && transient.data != scratch.data && scratch.total() * scratch.elemsize == 1024,
                        "oversize host upload scratch remains transient instead of growing the retained cache");
    }
}

static double gpu_first_mxfp4_value(const TensorData& weight, uint32_t row, uint32_t column)
{
    const uint32_t blocks = weight.shape[1] / 32;
    const uint8_t packed = weight.mxfp4_blocks[(static_cast<size_t>(row) * blocks + column / 32) * 16 + (column % 32) / 2];
    const uint8_t nibble = (column & 1) == 0 ? packed & 15 : packed >> 4;
    constexpr std::array<double, 8> magnitudes = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const uint8_t exponent = weight.mxfp4_scales[static_cast<size_t>(row) * blocks + column / 32];
    return ((nibble & 8) == 0 ? magnitudes[nibble & 7] : -magnitudes[nibble & 7])
           * std::ldexp(1.0, static_cast<int>(exponent) - 127);
}

static void gpu_first_gptoss_prefill()
{
    // Reproduce the existing CPU-block fixture, but explicitly require cold
    // admission and every routed request to execute on GPU in this fixture.
    auto gate = std::make_shared<TensorData>();
    gate->dtype = DType::MxFp4;
    gate->shape = {64, 32};
    gate->mxfp4_blocks.resize(64 * 16);
    gate->mxfp4_scales.resize(64);
    TensorData gate_bias;
    gate_bias.dtype = DType::Float32;
    gate_bias.shape = {64};
    gate_bias.float32_data.resize(64);
    for (uint32_t row = 0; row < 64; ++row)
    {
        gate->mxfp4_scales[row] = static_cast<uint8_t>(125 + row % 5);
        gate_bias.float32_data[row] = static_cast<float>(static_cast<int>(row % 9) - 4) * 0.03125f;
        for (uint32_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t low = static_cast<uint8_t>((row * 3 + byte * 5 + 1) % 16);
            const uint8_t high = static_cast<uint8_t>((row * 7 + byte * 2 + 4) % 16);
            gate->mxfp4_blocks[row * 16 + byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    auto down = std::make_shared<TensorData>();
    down->dtype = DType::MxFp4;
    down->shape = {7, 32};
    down->mxfp4_blocks.resize(7 * 16);
    down->mxfp4_scales.resize(7);
    TensorData down_bias;
    down_bias.dtype = DType::Float32;
    down_bias.shape = {7};
    down_bias.float32_data.resize(7);
    for (uint32_t row = 0; row < 7; ++row)
    {
        down->mxfp4_scales[row] = static_cast<uint8_t>(126 + row % 3);
        down_bias.float32_data[row] = static_cast<float>(static_cast<int>(row) - 3) * 0.0625f;
        for (uint32_t byte = 0; byte < 16; ++byte)
        {
            const uint8_t low = static_cast<uint8_t>((row * 11 + byte * 3 + 2) % 16);
            const uint8_t high = static_cast<uint8_t>((row * 5 + byte * 7 + 8) % 16);
            down->mxfp4_blocks[row * 16 + byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    const uint64_t flags = OptimizationDefaultFlags & ~OptimizationCpuMxfp4Q8 & ~OptimizationCpuFastSilu;
    auto runtime = create_vulkan_runtime();
    auto backend = create_vulkan_expert_backend(4096, 0, {}, runtime, flags);
    gpu_first_check(static_cast<bool>(backend), "GptOss cold-prefill backend");
    CompiledModel model;
    model.descriptor.hidden_size = 7;
    model.opt.hybrid_mode = HybridMode::HybridExperts;
    model.opt.optimization_flags = flags;
    model.expert_backend = backend;
    auto gate_handle = model.weights.add("gate", *gate);
    auto down_handle = model.weights.add("down", *down);
    auto gate_bias_handle = model.weights.add("gate_bias", gate_bias);
    auto down_bias_handle = model.weights.add("down_bias", down_bias);
    gpu_first_check(gate_handle && down_handle && gate_bias_handle && down_bias_handle, "GptOss fixture weights");
    model.operators.bind_weight_count(model.weights.size());
    MoeBlockPlan moe;
    moe.experts.resize(2);
    for (size_t index = 0; index < 2; ++index)
    {
        auto& expert = moe.experts[index];
        expert.gate_up_weight = gate_handle.value();
        expert.down_weight = down_handle.value();
        expert.gate_up_bias = gate_bias_handle.value();
        expert.down_bias = down_bias_handle.value();
        expert.activation = ExpertActivation::GptOssSwiGlu;
        expert.activation_limit = 5.25f;
        expert.cache_key = "gptoss-cold-" + std::to_string(index);
    }
    LayerState state;
    state.normalized.reset(96, 32, false);
    state.resize_experts(2);
    ExpertScratch scratch;
    SessionStatistics statistics;
    const double unit_roundoff = std::numeric_limits<float>::epsilon() * 0.5;
    const auto gamma = [unit_roundoff](uint32_t operations) {
        return operations * unit_roundoff / (1.0 - operations * unit_roundoff);
    };
    double maximum_gpu_error = 0.0;
    double maximum_cpu_error = 0.0;
    double maximum_budget_fraction = 0.0;
    ScopedExpertBackendForeground foreground(backend);
    for (size_t pass = 0; pass < 3; ++pass)
    {
        for (size_t row = 0; row < state.normalized.rows(); ++row)
            for (uint32_t column = 0; column < 32; ++column)
                state.normalized.row(row)[column] = static_cast<float>(static_cast<int>((row + column + pass) % 17) - 8) * 0.015625f;
        for (size_t index = 0; index < 2; ++index)
        {
            auto& active = state.active_experts()[index];
            active.batch.expert_id = static_cast<uint32_t>(index);
            const size_t count = index == 0 ? (pass == 1 ? 32 : 65) : (pass == 1 ? 8 : 16);
            active.batch.routes.resize(count);
            for (size_t row = 0; row < count; ++row)
                active.batch.routes[row] = {static_cast<uint32_t>(index == 0 ? row : row * (pass == 1 ? 3 : 5)), 0, 1.0f};
        }
        gpu_first_check(static_cast<bool>(forward_moe(model, moe, state, statistics, scratch, 0, ExecutionBackend::Vulkan, false)), "GptOss cold-prefill forward");
        for (const auto& active : state.active_experts())
        {
            ActivationBuffer activated;
            forward_gate_up_mxfp4(*gate, &gate_bias, active.input, ExpertActivation::GptOssSwiGlu, 5.25f, activated, flags);
            const auto cpu = forward_linear(*down, down_bias, activated, flags);
            gpu_first_check(active.output.rows() == active.input.rows() && active.output.columns() == 7, "GptOss GPU output shape");
            for (size_t row = 0; row < active.input.rows(); ++row)
            {
                std::array<double, 32> reference_activation{};
                for (uint32_t column = 0; column < 32; ++column)
                {
                    double gate_value = gate_bias.float32_data[column * 2];
                    double up_value = gate_bias.float32_data[column * 2 + 1];
                    for (uint32_t k = 0; k < 32; ++k)
                    {
                        gate_value += gpu_first_mxfp4_value(*gate, column * 2, k) * active.input.row(row)[k];
                        up_value += gpu_first_mxfp4_value(*gate, column * 2 + 1, k) * active.input.row(row)[k];
                    }
                    // Dyadic weights, inputs and biases make both projections
                    // exact in FP32 here; isolate activation and down reduction.
                    gpu_first_check(static_cast<double>(static_cast<float>(gate_value)) == gate_value
                                        && static_cast<double>(static_cast<float>(up_value)) == up_value,
                                    "GptOss oracle projections must be exactly representable");
                    gate_value = std::min(gate_value, 5.25);
                    up_value = std::clamp(up_value, -5.25, 5.25);
                    const float exponent = -1.702f * static_cast<float>(gate_value);
                    reference_activation[column] = gate_value / (1.0 + std::exp(static_cast<double>(exponent))) * (up_value + 1.0);
                }
                for (uint32_t column = 0; column < 7; ++column)
                {
                    double reference = down_bias.float32_data[column];
                    double absolute_sum = std::abs(reference);
                    for (uint32_t k = 0; k < 32; ++k)
                    {
                        const double term = gpu_first_mxfp4_value(*down, column, k) * reference_activation[k];
                        reference += term;
                        absolute_sum += std::abs(term);
                    }
                    // Eight FP32 rounding units cover the activation's exp,
                    // addition, division and multiplication; gamma(33) is the
                    // conservative dot-product plus bias accumulation bound.
                    // The budget scales with actual summands, including cancellation.
                    const double bound = (gamma(8) + gamma(33) * (1.0 + gamma(8))) * absolute_sum;
                    const double gpu_error = std::abs(static_cast<double>(active.output.row(row)[column]) - reference);
                    const double cpu_error = std::abs(static_cast<double>(cpu.row(row)[column]) - reference);
                    gpu_first_check(std::isfinite(gpu_error) && gpu_error <= bound && cpu_error <= bound,
                                    "GptOss prefill exceeds FP32 activation/reduction error budget");
                    maximum_gpu_error = std::max(maximum_gpu_error, gpu_error);
                    maximum_cpu_error = std::max(maximum_cpu_error, cpu_error);
                    maximum_budget_fraction = std::max(maximum_budget_fraction, gpu_error / bound);
                }
            }
        }
        state.reset();
    }
    const auto stats = backend->statistics();
    gpu_first_check(stats.executions == 6
                        && stats.execution_failures == 0,
                    "GptOss cold and warm prefill must execute every request on GPU");
    std::cout << "GptOss FP64 oracle max GPU error=" << maximum_gpu_error
              << " CPU error=" << maximum_cpu_error << " FP32 budget fraction=" << maximum_budget_fraction << '\n';
}

static std::shared_ptr<const DeviceTensor_vulkan> gpu_first_upload_normalized(const ActivationBuffer& input, const VulkanRuntimePtr& runtime, uint64_t flags)
{
    TensorData identity;
    identity.dtype = DType::Float32;
    identity.shape = {input.columns(), input.columns()};
    identity.float32_data.assign(static_cast<size_t>(input.columns()) * input.columns(), 0.0f);
    for (uint32_t column = 0; column < input.columns(); ++column)
        identity.float32_data[static_cast<size_t>(column) * input.columns() + column] = 1.0f;
    const auto seed = Linear::create(identity, nullptr, LinearDevice::Vulkan, 0, runtime, flags);
    gpu_first_check(static_cast<bool>(seed), "GPU-first normalized input seed");
    auto command = CommandGraph_vulkan::create(*seed);
    auto device = std::make_shared<DeviceTensor_vulkan>();
    gpu_first_check(command && command->upload(input, *device) && command->submit() && command->wait(),
                    "GPU-first normalized input upload");
    return device;
}

static void gpu_first_lazy_host_input()
{
    GpuFirstFixture fixture;
    constexpr size_t expert_count = 10;
    CompiledModel model;
    model.descriptor.hidden_size = 32;
    model.opt.hybrid_mode = HybridMode::HybridExperts;
    model.opt.optimization_flags = fixture.flags;
    const auto gate = model.weights.add("lazy-gate", *fixture.gate);
    const auto down = model.weights.add("lazy-down", *fixture.down);
    gpu_first_check(gate && down, "lazy Host fixture weights");
    model.operators.bind_weight_count(model.weights.size());
    MoeBlockPlan moe;
    moe.experts.resize(expert_count);
    for (size_t index = 0; index < expert_count; ++index)
    {
        auto& expert = moe.experts[index];
        expert.gate_up_weight = gate.value();
        expert.down_weight = down.value();
        expert.weight_size = fixture.size;
        expert.activation = ExpertActivation::DeepSeekSwiGlu;
        expert.activation_limit = 4.0f;
        // The last Expert has no demand key and deliberately consumes CPU
        // input alongside nine GPU Experts sharing a two-pair cache.
        if (index + 1 < expert_count) expert.cache_key = "lazy-" + std::to_string(index);
    }
    ExpertScratch scratch;
    SessionStatistics statistics;
    LayerState state;
    state.resize_experts(expert_count);
    state.normalized.reset(4, 32, false);
    for (size_t row = 0; row < state.normalized.rows(); ++row)
        for (uint32_t column = 0; column < state.normalized.columns(); ++column)
            state.normalized.row(row)[column] = static_cast<float>(static_cast<int>((row * 7 + column * 3) % 19) - 9) * 0.015625f;
    for (size_t index = 0; index < expert_count; ++index)
    {
        auto& active = state.active_experts()[index];
        active.batch.expert_id = static_cast<uint32_t>(index);
        active.batch.routes = {{static_cast<uint32_t>(index % 4), 0, 1.0f},
                               {static_cast<uint32_t>((index + 3) % 4), 1, 0.5f}};
    }
    const auto device = gpu_first_upload_normalized(state.normalized, fixture.runtime, fixture.flags);
    const auto foreign_device = gpu_first_upload_normalized(state.normalized, create_vulkan_runtime(), fixture.flags);
    model.expert_backend = create_vulkan_expert_backend(fixture.size * 2, 0, {}, fixture.runtime, fixture.flags);
    ScopedExpertBackendForeground foreground(model.expert_backend);
    const void* wave_requests[2] = {};
    for (size_t pass = 0; pass < 4; ++pass)
    {
        // Pass 1 retains poisoned Host capacity from a previous execution;
        // pass 2 injects a foreign context to force GPU execution failure;
        // pass 3 disables GPU execution. Both fallback paths must gather.
        state.normalized_device = pass == 2 ? foreign_device : device;
        model.opt.flags = pass == 3 ? OptionDisableGpuExpertExecution : uint64_t(0);
        if (pass == 1)
            for (auto& active : state.active_experts())
            {
                active.input.reset(active.batch.routes.size(), 32, false);
                std::fill_n(active.input.row(0), active.input.values().size(), 123.0f);
            }
        gpu_first_check(static_cast<bool>(forward_moe(model, moe, state, statistics, scratch, 0, ExecutionBackend::Vulkan, false)),
                        "lazy Host mixed GPU/CPU forward completed");
        for (size_t index = 0; index < expert_count; ++index)
        {
            const auto& active = state.active_experts()[index];
            ActivationBuffer gathered(active.batch.routes.size(), 32);
            for (size_t row = 0; row < active.batch.routes.size(); ++row)
                std::copy_n(state.normalized.row(active.batch.routes[row].token_index), 32, gathered.row(row));
            const bool gpu = pass < 2 && index + 1 < expert_count;
            // GPU Experts retain FP32 activations. Compare those outputs to
            // an FP32 CPU oracle, while actual CPU fallback keeps its default
            // Q8 activation precision policy. This is a data-lifetime test,
            // so quantization error must not mask a stale gathered tensor.
            const uint64_t oracle_flags = gpu ? fixture.flags & ~OptimizationCpuMxfp4Q8 : fixture.flags;
            ActivationBuffer activated;
            forward_gate_up_mxfp4(*fixture.gate, nullptr, gathered, ExpertActivation::DeepSeekSwiGlu, 4.0f, activated, oracle_flags);
            const auto expected = forward_linear(*fixture.down, activated, oracle_flags);
            ActivationBuffer materialized;
            const ActivationBuffer* actual = &active.output;
            if (gpu)
            {
                gpu_first_check(active.input.rows() == 0 && active.device_output,
                                "successful GPU Expert has no gathered Host input");
                if (pass == 0) gpu_first_check(active.input.allocated_bytes() == 0, "GPU Expert allocates no Host input storage");
                gpu_first_check(MoeCombine_vulkan::materialize(*active.device_output, materialized), "lazy Host device output materialization");
                actual = &materialized;
            }
            else
            {
                gpu_first_check(active.input.rows() == gathered.rows() && !active.device_output,
                                "CPU consumer or failed GPU execution gathers Host input");
                for (size_t value = 0; value < gathered.values().size(); ++value)
                    gpu_first_check(active.input.values()[value] == gathered.values()[value], "fallback replaces stale Host values");
            }
            gpu_first_check(actual->rows() == expected.rows() && actual->columns() == expected.columns(), "lazy Host output shape");
            for (size_t value = 0; value < expected.values().size(); ++value)
            {
                if (!std::isfinite(actual->values()[value]) || std::abs(actual->values()[value] - expected.values()[value]) >= 0.000001f)
                {
                    const size_t row = value / expected.columns();
                    const uint32_t column = static_cast<uint32_t>(value % expected.columns());
                    double reference = 0.0;
                    for (uint32_t intermediate = 0; intermediate < 32; ++intermediate)
                    {
                        double gate_value = 0.0, up_value = 0.0;
                        for (uint32_t k = 0; k < 32; ++k)
                        {
                            gate_value += gpu_first_mxfp4_value(*fixture.gate, intermediate * 2, k) * gathered.row(row)[k];
                            up_value += gpu_first_mxfp4_value(*fixture.gate, intermediate * 2 + 1, k) * gathered.row(row)[k];
                        }
                        gate_value = std::min(gate_value, 4.0);
                        up_value = std::clamp(up_value, -4.0, 4.0);
                        reference += gpu_first_mxfp4_value(*fixture.down, column, intermediate)
                                     * gate_value / (1.0 + std::exp(-gate_value)) * up_value;
                    }
                    std::cerr.precision(10);
                    std::cerr << "lazy Host mismatch pass=" << pass << " expert=" << index << " gpu=" << gpu
                              << " row=" << row << " column=" << column << " actual=" << actual->values()[value]
                              << " CPU=" << expected.values()[value] << " FP64=" << reference
                              << " actual_error=" << std::abs(static_cast<double>(actual->values()[value]) - reference)
                              << " CPU_error=" << std::abs(static_cast<double>(expected.values()[value]) - reference) << '\n';
                    gpu_first_check(false, "lazy Host output matches independent CPU oracle");
                }
            }
        }
        gpu_first_check(scratch.prestaged_demands.empty() && scratch.prestaged_pins.empty() && scratch.prestaged_cache_leases.empty()
                            && scratch.current_wave_requests.empty() && scratch.next_wave_requests.empty()
                            && scratch.wave_demands.empty() && scratch.prepared_pins.empty(),
                        "completed waves release retained owners while preserving storage");
        if (pass == 0)
        {
            wave_requests[0] = scratch.current_wave_requests.data();
            wave_requests[1] = scratch.next_wave_requests.data();
            gpu_first_check(wave_requests[0] && wave_requests[1], "bounded wave buffers retain both capacities");
        }
        if (pass == 1)
        {
            const auto current = scratch.current_wave_requests.data();
            const auto next = scratch.next_wave_requests.data();
            gpu_first_check((current == wave_requests[0] && next == wave_requests[1])
                                || (current == wave_requests[1] && next == wave_requests[0]),
                            "warm wave submission reuses both request buffers");
        }
        state.reset();
    }
    gpu_first_check(model.expert_backend->statistics().execution_failures != 0, "foreign context exercises failed GPU fallback");
}

// Force the transfer staging path even on unified or host-visible heaps while
// letting the real allocator keep its own mapping and memory lifetime flags.
class GpuFirstWeightAllocator final : public ncnn::VkAllocator
{
public:
    explicit GpuFirstWeightAllocator(const ncnn::VulkanDevice* device, size_t allowed_allocations)
        : ncnn::VkAllocator(device), weights(device, size_t(4096)), allowed(allowed_allocations)
    {
        mappable = false;
    }

    ncnn::VkBufferMemory* fastMalloc(size_t size) override
    {
        if (calls++ >= allowed)
            return nullptr;
        return weights.fastMalloc(size);
    }

    void fastFree(ncnn::VkBufferMemory* buffer) override
    {
        weights.fastFree(buffer);
    }

    ncnn::VkImageMemory* fastMalloc(int, int, int, size_t, int) override
    {
        return nullptr;
    }

    void fastFree(ncnn::VkImageMemory* image) override
    {
        weights.fastFree(image);
    }

    size_t calls = 0;

private:
    ncnn::VkWeightAllocator weights;
    size_t allowed;
};

class GpuFirstThrowingStagingAllocator final : public ncnn::VkStagingAllocator
{
public:
    explicit GpuFirstThrowingStagingAllocator(const ncnn::VulkanDevice* device)
        : ncnn::VkStagingAllocator(device)
    {
    }

    using ncnn::VkStagingAllocator::fastMalloc;
    ncnn::VkBufferMemory* fastMalloc(size_t) override
    {
        ++failures;
        throw std::bad_alloc();
    }

    size_t failures = 0;
};

static void gpu_first_upload_staging_pool()
{
    GpuFirstFixture fixture;
    auto context = VulkanContext::acquire(0, fixture.runtime, fixture.flags);
    gpu_first_check(static_cast<bool>(context), "upload staging test context");
    VulkanUploadStagingAllocator pool(context->device(), 4096);
    ncnn::VkBufferMemory* initial_buffer = nullptr;
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkMat staging(256, sizeof(float), &pool);
        gpu_first_check(!staging.empty() && staging.mapped_ptr(), "pooled staging buffer allocation");
        initial_buffer = staging.data;
    }
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkMat staging(256, sizeof(float), &pool);
        gpu_first_check(staging.data == initial_buffer, "matching staging allocation reuses the same mapped buffer");
    }
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkMat oversize(2048, sizeof(float), &pool);
        gpu_first_check(!oversize.empty(), "upload larger than cache remains legal in flight");
    }

    ncnn::Mat source(256, sizeof(float));
    for (int index = 0; index < source.w; ++index)
        static_cast<float*>(source.data)[index] = static_cast<float>(index - 128) * 0.03125f;
    ncnn::Option option;
    option.blob_vkallocator = context->blob_allocator();
    option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    option.use_packing_layout = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    ncnn::VkWeightAllocator weights(context->device());
    ncnn::VkMat destination;
    const auto upload = [&] {
        VulkanWeightUploadBatch batch(context, &pool);
        gpu_first_check(batch.record(source, destination, option, &weights) && batch.submit(), "pooled weight upload");
        gpu_first_check(!batch.submit(), "completed weight batch cannot submit twice without reset");
    };
    upload();
    {
        VulkanWeightUploadBatch cancelled(context, &pool);
        gpu_first_check(cancelled.record(source, destination, option, &weights), "cancelled upload records staging");
    }
    {
        VulkanWeightUploadBatch failed(context, &pool);
        gpu_first_check(failed.record(source, destination, option, &weights), "failed batch has retained staging");
        gpu_first_check(!failed.record(ncnn::Mat(), destination, option, &weights) && !failed.submit(),
                        "invalid record cancels submission and preserves safe staging lifetime");
    }
    for (size_t accepted_allocations : {size_t(0), size_t(1)})
    {
        GpuFirstWeightAllocator rejected_weights(context->device(), accepted_allocations);
        ncnn::VkMat accepted_destination;
        ncnn::VkMat rejected_destination;
        bool owner_destroyed = false;
        // Pipeline deleters in the MXFP4 factories take this same context
        // mutex. Exercise that last-owner contract without a test-only factory
        // API or any attempt to exhaust actual device memory.
        std::shared_ptr<const void> owner(new int(1), [context, &owner_destroyed](const void* value) {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            owner_destroyed = true;
            delete static_cast<const int*>(value);
        });
        const std::weak_ptr<const void> last_owner = owner;
        {
            VulkanWeightUploadBatch rejected(context, &pool);
            rejected.retain_owner(owner);
            owner.reset();
            if (accepted_allocations != 0)
                gpu_first_check(rejected.record(source, accepted_destination, option, &rejected_weights),
                                "allocation rejection fixture first records a partial batch");
            gpu_first_check(!rejected.record(source, rejected_destination, option, &rejected_weights)
                                && rejected_destination.empty() && !rejected.submit(),
                            "weight allocation rejection cannot publish an incomplete upload");
            gpu_first_check(rejected_weights.calls == accepted_allocations + 1,
                            "weight rejection reached the configured allocation boundary");
            gpu_first_check(!last_owner.expired() && !owner_destroyed,
                            "failed factory's last owner survives while batch owns context lock");
        }
        gpu_first_check(last_owner.expired() && owner_destroyed,
                        "failed batch destroys retained owners after releasing context lock");
    }
    {
        GpuFirstWeightAllocator device_weights(context->device(), std::numeric_limits<size_t>::max());
        GpuFirstThrowingStagingAllocator throwing_staging(context->device());
        ncnn::VkMat incomplete_destination;
        VulkanWeightUploadBatch rejected(context, &throwing_staging);
        gpu_first_check(!rejected.record(source, incomplete_destination, option, &device_weights)
                            && incomplete_destination.empty() && !rejected.submit()
                            && throwing_staging.failures == 1,
                        "staging allocation exception cancels upload without publishing a destination");
    }
    upload();
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device(), context->command_optimization_flags());
        ncnn::Mat downloaded;
        option.staging_vkallocator = context->staging_allocator();
        command.record_download(destination, downloaded, option);
        gpu_first_check(command.submit_and_wait() == 0 && !downloaded.empty(), "retry upload download");
        for (size_t index = 0; index < source.total() * source.elemsize; ++index)
            gpu_first_check(static_cast<const uint8_t*>(downloaded.data)[index] == static_cast<const uint8_t*>(source.data)[index],
                            "reused staging keeps retry payload intact");
    }
    {
        VulkanWeightUploadBatch compatible(context);
        gpu_first_check(compatible.record(source, destination, option, &weights) && compatible.submit(), "default owned staging upload remains compatible");
    }

    auto backend = create_vulkan_expert_backend(4096, 0, {}, fixture.runtime, fixture.flags);
    ScopedExpertBackendForeground foreground(backend);
    std::shared_ptr<const void> pin;
    gpu_first_check(fixture.demand(*backend, "pool-cold-A", pin), "first cold upload using backend pool");
    gpu_first_execute(fixture, *backend, "pool-cold-A", pin);
    for (const char* key : {"pool-cold-B", "pool-cold-C"})
    {
        gpu_first_check(fixture.demand(*backend, key, pin), "next cold upload using backend pool");
        gpu_first_execute(fixture, *backend, key, pin);
    }
    const auto reused = backend->statistics();
    gpu_first_check(reused.executions == 3 && reused.evictions == 1 && reused.resident_size <= 4096,
                    "pooled cold uploads execute and evict within bounds");
}

#endif

} // namespace moe
} // namespace ncnn

int main()
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::moe::get_gpu_count() == 0)
    {
        std::cout << "SKIP: Vulkan GPU unavailable\n";
        return 77;
    }
    try
    {
        ncnn::moe::gpu_first_cold_and_capacity();
        ncnn::moe::gpu_first_duplicate_and_shard();
        ncnn::moe::gpu_first_gptoss_prefill();
        ncnn::moe::gpu_first_lazy_host_input();
        ncnn::moe::gpu_first_upload_staging_pool();
        ncnn::moe::gpu_first_foreground_batch();
        ncnn::moe::gpu_first_upload_host_scratch();
        ncnn::moe::gpu_first_async_transfer_and_joint_eviction();
        ncnn::moe::gpu_first_joint_eviction_window();
        ncnn::moe::gpu_first_aligned_batch_outputs();
        ncnn::moe::gpu_first_context_lifetime();
        std::cout << "GPU-first Expert policy tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "GPU-first Expert policy test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
