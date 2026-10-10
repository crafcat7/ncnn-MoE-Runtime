#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/vulkan.h"
#include "engine/expertbackend.h"
#include "kernels/activationbuffer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void expert_device_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void expert_device_check_output(const ActivationBuffer& actual,
                                       const ActivationBuffer& expected)
{
    expert_device_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), "device expert shape");
    float maximum = 0.0f;
    for (size_t index = 0; index < actual.values().size(); ++index)
    {
        maximum = std::max(maximum, std::abs(expected.values()[index]));
        expert_device_check(std::abs(actual.values()[index] - expected.values()[index]) < 0.000001f, "device expert output parity");
    }
    expert_device_check(maximum > 0.00000001f, "device expert nonzero oracle");
}

static TensorData expert_device_mxfp4(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData weight;
    weight.dtype = DType::MxFp4;
    weight.shape = {rows, columns};
    weight.mxfp4_blocks.resize(static_cast<size_t>(rows) * columns / 2);
    weight.mxfp4_scales.resize(static_cast<size_t>(rows) * columns / 32);
    std::fill_n(weight.mxfp4_scales.data(), weight.mxfp4_scales.size(), uint8_t(123));
    for (size_t index = 0; index < weight.mxfp4_blocks.size(); ++index)
    {
        const uint8_t low = static_cast<uint8_t>((index * 7 + seed * 3 + index / 13) % 16);
        const uint8_t high = static_cast<uint8_t>((index * 11 + seed * 5 + index / 17) % 16);
        weight.mxfp4_blocks[index] = static_cast<uint8_t>(low | (high << 4));
    }
    return weight;
}

#if NCNN_MOE_WITH_VULKAN
static std::shared_ptr<const DeviceTensor_vulkan> expert_device_upload(const ActivationBuffer& input,
                                                                       const VulkanRuntimePtr& runtime,
                                                                       uint64_t flags)
{
    TensorData seed_weight;
    seed_weight.dtype = DType::Float32;
    seed_weight.shape = {input.columns(), input.columns()};
    seed_weight.float32_data.assign(static_cast<size_t>(input.columns()) * input.columns(), 0.0f);
    for (uint32_t column = 0; column < input.columns(); ++column)
        seed_weight.float32_data[static_cast<size_t>(column) * input.columns() + column] = 1.0f;
    auto seed = Linear::create(seed_weight, nullptr, LinearDevice::Vulkan, 0, runtime, flags);
    expert_device_check(static_cast<bool>(seed), "device input graph seed");
    auto graph = CommandGraph_vulkan::create(*seed);
    auto tensor = std::make_shared<DeviceTensor_vulkan>();
    expert_device_check(graph && graph->upload(input, *tensor) && graph->submit() && graph->wait(), "device normalized upload");
    graph.reset();
    seed.reset();
    expert_device_check(!tensor->empty(), "device tensor outlives producer");
    return tensor;
}

static void expert_device_execute(ExpertBackend& backend, std::span<const ExpertBackendRequest> requests)
{
    auto submission = backend.submit_batch(requests);
    expert_device_check(static_cast<bool>(submission), "device expert submission");
    for (auto reservation : submission->reservations())
        expert_device_check(reservation == ExpertBackendExecutionResult::Executed, "device expert resident reservation");
    const auto results = submission->wait();
    expert_device_check(results.size() == requests.size(), "device expert results");
    for (auto result : results)
        expert_device_check(result == ExpertBackendExecutionResult::Executed, "device expert execution");
    expert_device_check(submission->commit(), "device expert commit");
}

static void expert_device_test()
{
    const uint64_t flags = OptimizationDefaultFlags & ~OptimizationVulkanRouteAggregation;
    const auto runtime = create_vulkan_runtime();
    auto backend = create_vulkan_expert_backend(1024 * 1024, 0, nullptr, runtime, flags);
    expert_device_check(static_cast<bool>(backend), "device expert backend");
    auto gate = std::make_shared<TensorData>(expert_device_mxfp4(64, 32, 1));
    auto down = std::make_shared<TensorData>(expert_device_mxfp4(32, 32, 2));
    backend->admit("device-expert-a", gate, nullptr, down, nullptr, 0, 0.0f, ExpertActivation::Silu);
    backend->admit("device-expert-b", gate, nullptr, down, nullptr, 0, 0.0f, ExpertActivation::Silu);
    backend->wait_for_background_work();
    expert_device_check(backend->statistics().stores == 2, "device experts admitted");
    ActivationBuffer normalized(4, 32);
    for (size_t row = 0; row < normalized.rows(); ++row)
        for (uint32_t column = 0; column < normalized.columns(); ++column)
            normalized.row(row)[column] = static_cast<float>(static_cast<int>((column * 7 + row * 5) % 23) - 8) * 0.015625f;
    const std::array<std::vector<ExpertRoute>, 2> routes = {
        std::vector<ExpertRoute>{{3, 0, 0.7f}, {1, 1, 0.4f}, {3, 1, 0.3f}},
        std::vector<ExpertRoute>{{0, 0, 0.9f}, {2, 0, 0.8f}}};
    std::array<ActivationBuffer, 2> inputs;
    std::array<ActivationBuffer, 2> expected;
    std::array<ActivationBuffer, 2> actual;
    std::array<ExpertBackendRequest, 2> requests;
    for (size_t expert = 0; expert < 2; ++expert)
    {
        inputs[expert].reset(routes[expert].size(), 32, false);
        for (size_t row = 0; row < routes[expert].size(); ++row)
            std::copy_n(normalized.row(routes[expert][row].token_index), 32, inputs[expert].row(row));
        requests[expert].key = expert == 0 ? "device-expert-a" : "device-expert-b";
        requests[expert].input = &inputs[expert];
        requests[expert].output = &expected[expert];
    }
    expert_device_execute(*backend, requests);
    // Four upload rows use ncnn's pack4 layout; routing still indexes
    // logical tokens and must gather every lane without a host transfer.
    auto device = expert_device_upload(normalized, runtime, flags);
    expert_device_check(device->rows() == 4 && device->columns() == 32, "device normalized logical shape");
    for (size_t expert = 0; expert < 2; ++expert)
    {
        requests[expert].output = &actual[expert];
        requests[expert].device_input = device;
        requests[expert].device_routes = routes[expert];
        requests[expert].input_rows = routes[expert].size();
        requests[expert].input_columns = normalized.columns();
        requests[expert].input = nullptr;
        // The CPU oracle no longer has to stay allocated for GPU execution.
        inputs[expert].clear();
    }
    const auto before = get_vulkan_statistics(runtime);
    expert_device_execute(*backend, requests);
    const auto after = get_vulkan_statistics(runtime);
    expert_device_check(after.batch_uploads == before.batch_uploads, "resident expert inputs avoid activation upload");
    for (size_t expert = 0; expert < 2; ++expert)
        expert_device_check_output(actual[expert], expected[expert]);

    // Five upload rows retain pack1. The same non-monotonic route ids
    // must also reuse this resident tensor with the canonical row layout.
    ActivationBuffer unpacked_normalized(5, 32);
    for (size_t row = 0; row < normalized.rows(); ++row)
        std::copy_n(normalized.row(row), 32, unpacked_normalized.row(row));
    std::fill_n(unpacked_normalized.row(4), 32, 0.0f);
    auto unpacked_device = expert_device_upload(unpacked_normalized, runtime, flags);
    for (auto& request : requests)
        request.device_input = unpacked_device;
    const auto unpacked_before = get_vulkan_statistics(runtime);
    expert_device_execute(*backend, requests);
    const auto unpacked_after = get_vulkan_statistics(runtime);
    expert_device_check(unpacked_after.batch_uploads == unpacked_before.batch_uploads, "pack1 resident expert inputs avoid activation upload");
    for (size_t expert = 0; expert < 2; ++expert)
        expert_device_check_output(actual[expert], expected[expert]);

    // Device-only shape/gather rejection cannot publish partial output.
    const auto rejected_output = actual;
    const auto reject_device_only = [&](const char* message) {
        auto rejected = backend->submit_batch(requests);
        expert_device_check(static_cast<bool>(rejected), message);
        const auto results = rejected->wait();
        expert_device_check(results.size() == requests.size(), "device-only rejection result shape");
        for (auto result : results)
            expert_device_check(result == ExpertBackendExecutionResult::Failed, message);
        rejected->abort();
        expert_device_check(!rejected->commit(), "rejected batch cannot publish");
        for (size_t expert = 0; expert < 2; ++expert)
            expert_device_check_output(actual[expert], rejected_output[expert]);
    };
    std::vector<ExpertRoute> invalid_device_routes = routes[0];
    invalid_device_routes[0].token_index = 5;
    requests[0].device_routes = invalid_device_routes;
    reject_device_only("invalid device-only row is rejected safely");
    requests[0].device_routes = routes[0];
    requests[0].input_rows += 1;
    reject_device_only("device-only logical shape mismatch is rejected safely");
    requests[0].input_rows -= 1;
    auto foreign = expert_device_upload(normalized, create_vulkan_runtime(), flags);
    for (auto& request : requests) request.device_input = foreign;
    reject_device_only("foreign device-only context is rejected safely");

    // A mismatched context must use the canonical host input when available.
    for (size_t expert = 0; expert < 2; ++expert)
    {
        inputs[expert].reset(routes[expert].size(), 32, false);
        for (size_t row = 0; row < routes[expert].size(); ++row)
            std::copy_n(normalized.row(routes[expert][row].token_index), 32, inputs[expert].row(row));
        requests[expert].input = &inputs[expert];
    }
    for (auto& request : requests)
        request.device_input = foreign;
    const auto foreign_before = get_vulkan_statistics(runtime);
    expert_device_execute(*backend, requests);
    expert_device_check(get_vulkan_statistics(runtime).batch_uploads == foreign_before.batch_uploads + 1, "foreign device context falls back to host transfer");
    for (size_t expert = 0; expert < 2; ++expert)
        expert_device_check_output(actual[expert], expected[expert]);

    // Invalid device gather metadata also preserves the valid host path.
    std::vector<ExpertRoute> invalid_routes = routes[0];
    invalid_routes[0].token_index = 4;
    for (auto& request : requests)
        request.device_input = device;
    requests[0].device_routes = invalid_routes;
    const auto invalid_before = get_vulkan_statistics(runtime);
    expert_device_execute(*backend, requests);
    expert_device_check(get_vulkan_statistics(runtime).batch_uploads == invalid_before.batch_uploads + 1, "invalid device row falls back to host transfer");
    for (size_t expert = 0; expert < 2; ++expert)
        expert_device_check_output(actual[expert], expected[expert]);
    requests[0].device_routes = routes[0];

    // The combined gather crosses Vulkan's minimum Y limit; each Expert
    // projection still has a legal row count. This exercises the Z slice.
    {
        constexpr size_t rows_per_expert = 40000;
        std::array<std::vector<ExpertRoute>, 2> large_routes;
        std::array<ActivationBuffer, 2> large_outputs;
        std::array<ExpertBackendRequest, 2> large_requests = requests;
        for (size_t expert = 0; expert < 2; ++expert)
        {
            const uint32_t input_row = expert == 0 ? 3u : 0u;
            large_routes[expert].assign(rows_per_expert, {input_row, 0, 1.0f});
            large_requests[expert].input = nullptr;
            large_requests[expert].input_rows = rows_per_expert;
            large_requests[expert].output = &large_outputs[expert];
            large_requests[expert].device_routes = large_routes[expert];
        }
        const auto large_before = get_vulkan_statistics(runtime);
        expert_device_execute(*backend, large_requests);
        const auto large_after = get_vulkan_statistics(runtime);
        expert_device_check(large_after.batch_uploads == large_before.batch_uploads,
                            "large resident gather crosses Y limit without activation upload");
        for (size_t expert = 0; expert < 2; ++expert)
        {
            expert_device_check(large_outputs[expert].rows() == rows_per_expert, "large gather output rows");
            for (size_t row : {size_t(0), size_t(32767), rows_per_expert - 1})
                for (uint32_t column = 0; column < 32; ++column)
                    expert_device_check(std::abs(large_outputs[expert].row(row)[column] - expected[expert].row(0)[column]) < 0.000001f,
                                        "large gather slice output parity");
        }
    }

    // WorkItem retains the completed tensor after the executor drops its copy.
    for (auto& request : requests) request.input = nullptr;
    const std::weak_ptr<const DeviceTensor_vulkan> weak_device = device;
    auto submission = backend->submit_batch(requests);
    device.reset();
    for (auto& request : requests)
        request.device_input.reset();
    expert_device_check(!weak_device.expired(), "submission pins resident input");
    const auto results = submission->wait();
    expert_device_check(results.size() == 2 && results[0] == ExpertBackendExecutionResult::Executed && results[1] == ExpertBackendExecutionResult::Executed, "pinned device expert completed");
    expert_device_check(!weak_device.expired(), "input pin survives until commit");
    expert_device_check(submission->commit(), "pinned device commit");
    for (size_t expert = 0; expert < 2; ++expert)
        expert_device_check_output(actual[expert], expected[expert]);
    submission.reset();
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
        ncnn::moe::expert_device_test();
        std::cout << "Vulkan resident expert input tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Vulkan resident expert input test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
