#include "backends/ncnn/moecombine_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "ncnn/moe/option.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {
#if NCNN_MOE_WITH_VULKAN
static constexpr uint64_t combine_flags = OptimizationVulkanExpertGpuPriority;
static void combine_check(bool condition, const std::string& label)
{
    if (!condition) throw std::runtime_error(label);
}
static std::shared_ptr<Linear> combine_seed(const VulkanRuntimePtr& runtime, uint32_t columns)
{
    TensorData matrix;
    matrix.dtype = DType::Float32;
    matrix.shape = {columns, columns};
    matrix.float32_data.assign(static_cast<size_t>(columns) * columns, 0.0f);
    for (uint32_t i = 0; i < columns; ++i) matrix.float32_data[static_cast<size_t>(i) * columns + i] = 1.0f;
    auto result = Linear::create(matrix, nullptr, LinearDevice::Vulkan, get_default_gpu_index(), runtime, combine_flags);
    combine_check(static_cast<bool>(result), "GPU seed creation");
    return result;
}
static std::shared_ptr<DeviceTensor_vulkan> combine_upload(const Linear& seed, const ActivationBuffer& input)
{
    auto graph = CommandGraph_vulkan::create(seed);
    auto output = std::make_shared<DeviceTensor_vulkan>();
    combine_check(graph && graph->upload(input, *output) && graph->submit() && graph->wait(), "completed GPU fixture upload");
    return output;
}
static ActivationBuffer combine_reference(std::span<const MoeCombineInput_vulkan> inputs, size_t rows,
                                          uint32_t columns, const ActivationBuffer* shared)
{
    ActivationBuffer result(rows, columns);
    for (const auto& input : inputs)
        for (size_t index = 0; index < input.routes.size(); ++index)
        {
            const auto& route = input.routes[index];
            for (uint32_t column = 0; column < columns; ++column)
            {
                // Test oracle explicitly reproduces the scalar CPU accumulation.
                volatile float product = route.weight * input.host->row(index)[column];
                result.row(route.token_index)[column] += product;
            }
        }
    if (shared)
        for (size_t row = 0; row < rows; ++row)
            for (uint32_t column = 0; column < columns; ++column) result.row(row)[column] += shared->row(row)[column];
    return result;
}
static void combine_exact(const ActivationBuffer& actual, const ActivationBuffer& expected, const std::string& label)
{
    combine_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + ": shape");
    for (size_t index = 0; index < actual.values().size(); ++index)
        combine_check(std::isfinite(actual.values()[index])
                          && std::bit_cast<uint32_t>(actual.values()[index]) == std::bit_cast<uint32_t>(expected.values()[index]),
                      label + ": index=" + std::to_string(index) + " actual=" + std::to_string(actual.values()[index])
                          + " expected=" + std::to_string(expected.values()[index]));
    for (size_t row = 0; row < actual.rows(); ++row)
    {
        const auto token = [](const ActivationBuffer& buffer, size_t row) {
            return std::max_element(buffer.row(row), buffer.row(row) + buffer.columns()) - buffer.row(row);
        };
        combine_check(token(actual, row) == token(expected, row), label + ": greedy token parity");
    }
}
static void combine_test_pipeline_lifetime()
{
    const auto runtime = create_vulkan_runtime();
    auto context = VulkanContext::acquire(get_default_gpu_index(), runtime, combine_flags);
    combine_check(static_cast<bool>(context), "persistent Combine context");
    const std::weak_ptr<VulkanContext> weak_context = context;
    std::vector<std::shared_ptr<MoeCombine_vulkan>> model_owners;
    for (size_t layer = 0; layer < 43; ++layer)
    {
        auto prepared = MoeCombine_vulkan::create(context);
        combine_check(prepared && prepared->pipeline(), "prepare per-layer Combine owner");
        if (!model_owners.empty())
            combine_check(prepared->pipeline() == model_owners.front()->pipeline(), "43 layers share the same weak-cache pipeline");
        model_owners.push_back(std::move(prepared));
    }
    const std::weak_ptr<ncnn::Pipeline> weak_pipeline = model_owners.front()->pipeline();
    const auto* pipeline_identity = model_owners.front()->pipeline().get();
    auto workspace_owner = model_owners.back();
    model_owners.clear();
    context.reset();
    combine_check(weak_pipeline.lock().get() == pipeline_identity && !weak_context.expired(), "pending workspace retains pipeline after model owners are released");
    ActivationBuffer source(1, 32);
    for (uint32_t column = 0; column < source.columns(); ++column) source.row(0)[column] = static_cast<float>(column + 1) / 8.0f;
    const std::array<ExpertRoute, 1> routes = {{{0, 0, 0.5f}}};
    const std::array<MoeCombineInput_vulkan, 1> inputs = {{{&source, {}, routes}}};
    const auto expected = combine_reference(inputs, 1, source.columns(), nullptr);
    std::shared_ptr<DeviceTensor_vulkan> output;
    combine_check(workspace_owner->combine(inputs, 1, source.columns(), nullptr, {}, output), "workspace can complete after model destruction");
    ActivationBuffer actual;
    combine_check(MoeCombine_vulkan::materialize(*output, actual), "completed activation survives its model owners");
    combine_exact(actual, expected, "persistent pipeline canonical math");
    combine_check(weak_pipeline.lock().get() == pipeline_identity, "execution preserves prepared pipeline identity");
    {
        const auto context_owner = weak_context.lock();
        const auto previous_combiner_owner = workspace_owner;
        std::unique_lock<std::mutex> lock(context_owner->command_mutex());
        // A layer transition replaces its workspace owner under the command
        // lock. Its previous owner must survive until that lock is released.
        workspace_owner.reset();
        combine_check(!weak_pipeline.expired(), "previous Combine owner survives replacement under command lock");
        lock.unlock();
    }
    combine_check(weak_pipeline.expired(), "destroying previous workspace owner after unlock releases pipeline");
    combine_check(!weak_context.expired(), "completed activation independently retains context");
    output.reset();
    combine_check(weak_context.expired(), "runtime weak cache introduces no context ownership cycle");
    combine_check(!MoeCombine_vulkan::create({}), "invalid factory has safe empty result");
}

static void combine_test_order(size_t rows, bool device_shared)
{
    constexpr uint32_t columns = 33;
    const auto runtime = create_vulkan_runtime();
    auto seed = combine_seed(runtime, columns);
    std::vector<ExpertRoute> routes;
    for (size_t row = 0; row < rows; ++row) routes.push_back({static_cast<uint32_t>(rows - row - 1), 0, 1.0f});
    ActivationBuffer huge(rows, columns), small(rows, columns), negative(rows, columns), tail(rows, columns), shared(rows, columns);
    for (size_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
        {
            huge.row(row)[column] = 1.0e20f;
            negative.row(row)[column] = -1.0e20f;
            small.row(row)[column] = 3.25f + static_cast<float>(column) / 8.0f;
            tail.row(row)[column] = static_cast<float>((column * 7 + row * 3) % 17) / 4.0f;
            shared.row(row)[column] = static_cast<float>(column % 5) / 8.0f;
        }
    const auto huge_gpu = combine_upload(*seed, huge);
    const auto negative_gpu = combine_upload(*seed, negative);
    const auto shared_gpu = device_shared ? combine_upload(*seed, shared) : nullptr;
    std::array<MoeCombineInput_vulkan, 4> inputs = {{{&huge, huge_gpu, routes}, {&small, {}, routes}, {&negative, negative_gpu, routes}, {&tail, {}, routes}}};
    const auto expected = combine_reference(inputs, rows, columns, &shared);
    std::shared_ptr<DeviceTensor_vulkan> output;
    const auto before = get_vulkan_statistics(runtime);
    combine_check(MoeCombine_vulkan::forward(inputs, rows, columns, &shared, shared_gpu, output), "mixed canonical device combine");
    auto after = get_vulkan_statistics(runtime);
    combine_check(output && !output->empty() && after.compute_submissions == before.compute_submissions + 1
                      && after.batch_downloads == before.batch_downloads,
                  "one GPU combine submit and zero host download");
    ActivationBuffer actual;
    combine_check(MoeCombine_vulkan::materialize(*output, actual), "combined result materializes");
    combine_exact(actual, expected, "canonical mixed CPU/GPU sum");
    // A device aggregate can be reused after staging slots and the producer are released.
    std::array<MoeCombineInput_vulkan, 1> chained = {{{&expected, output, routes}}};
    const auto expected_chain = combine_reference(chained, rows, columns, &shared);
    std::shared_ptr<DeviceTensor_vulkan> chained_output;
    const auto chain_before = get_vulkan_statistics(runtime);
    combine_check(MoeCombine_vulkan::forward(chained, rows, columns, &shared, shared_gpu, chained_output), "resident aggregate reuse");
    combine_check(get_vulkan_statistics(runtime).batch_downloads == chain_before.batch_downloads, "resident reuse avoids roundtrip");
    combine_check(MoeCombine_vulkan::materialize(*chained_output, actual), "chained result materializes");
    combine_exact(actual, expected_chain, "chained aggregate token parity");
    const auto preserved = output;
    const auto invalid_before = get_vulkan_statistics(runtime);
    auto bad_routes = routes;
    bad_routes[0].token_index = static_cast<uint32_t>(rows);
    inputs[0].routes = bad_routes;
    combine_check(!MoeCombine_vulkan::forward(inputs, rows, columns, &shared, shared_gpu, output) && output == preserved,
                  "invalid route leaves resident publication unchanged");
    combine_check(get_vulkan_statistics(runtime).compute_submissions == invalid_before.compute_submissions, "invalid route records no submit");
    inputs[0].routes = routes;
    const auto foreign_runtime = create_vulkan_runtime();
    auto foreign_seed = combine_seed(foreign_runtime, columns);
    inputs[2].device = combine_upload(*foreign_seed, negative);
    combine_check(!MoeCombine_vulkan::forward(inputs, rows, columns, &shared, shared_gpu, output) && output == preserved,
                  "foreign context rejects without modifying output");
    ActivationBuffer foreign_host;
    combine_check(MoeCombine_vulkan::materialize(*inputs[2].device, foreign_host), "foreign source has safe owning-context fallback");
    combine_exact(foreign_host, negative, "foreign materialization parity");
}
static void combine_test_deferred_ownership()
{
    constexpr uint32_t columns = 33;
    constexpr size_t rows = 3;
    const auto runtime = create_vulkan_runtime();
    auto seed = combine_seed(runtime, columns);
    auto context = VulkanContext::acquire(get_default_gpu_index(), runtime, combine_flags);
    auto combiner = MoeCombine_vulkan::create(context);
    combine_check(static_cast<bool>(combiner), "prepare deferred combiner");
    std::vector<ExpertRoute> routes = {{2, 0, 0.75f}, {0, 1, 0.5f}, {1, 0, 0.25f}};
    ActivationBuffer large(rows, columns), small(rows, columns), negative(rows, columns), shared(rows, columns);
    for (size_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
        {
            large.row(row)[column] = 1.0e20f;
            small.row(row)[column] = static_cast<float>(column + row + 1) / 8.0f;
            negative.row(row)[column] = -1.0e20f;
            shared.row(row)[column] = static_cast<float>(column % 7) / 4.0f;
        }
    auto large_gpu = combine_upload(*seed, large);
    auto negative_gpu = combine_upload(*seed, negative);
    std::array<MoeCombineInput_vulkan, 3> inputs = {{{&large, large_gpu, routes}, {&small, {}, routes}, {&negative, negative_gpu, routes}}};
    const auto expected = combine_reference(inputs, rows, columns, &shared);
    const auto before = get_vulkan_statistics(runtime);
    auto pending = combiner->prepare(inputs, rows, columns, &shared, {});
    combine_check(pending && pending->rows() == rows && pending->columns() == columns, "capture deferred shape");
    auto bad_routes = routes;
    bad_routes[0].token_index = rows;
    inputs[0].routes = bad_routes;
    combine_check(!combiner->prepare(inputs, rows, columns, &shared, {}), "invalid capture rejects before recording");
    inputs[0].routes = routes;
    const auto foreign_runtime = create_vulkan_runtime();
    auto foreign_seed = combine_seed(foreign_runtime, columns);
    auto foreign_combiner = MoeCombine_vulkan::create(VulkanContext::acquire(get_default_gpu_index(), foreign_runtime, combine_flags));
    combine_check(!foreign_combiner->prepare(inputs, rows, columns, &shared, {}), "foreign capture rejects");
    const std::weak_ptr<DeviceTensor_vulkan> weak_large = large_gpu;
    const std::weak_ptr<DeviceTensor_vulkan> weak_negative = negative_gpu;
    inputs = {};
    large_gpu.reset();
    negative_gpu.reset();
    large.clear();
    negative.clear();
    small.clear();
    shared.clear();
    routes.assign(17, {99, 3, -123.0f});
    combine_check(!weak_large.expired() && !weak_negative.expired(), "pending owns completed device sources");
    combine_check(get_vulkan_statistics(runtime).compute_submissions == before.compute_submissions, "capture performs no submit");
    // Abandon a populated command, then record again from the owned canonical
    // inputs. Re-arming imports must not depend on barriers in the abandoned command.
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        std::vector<ncnn::VkMat> storage;
        uint64_t uploads = 0;
        ncnn::VkCompute command(context->device(), context->command_optimization_flags());
        ncnn::VkMat output;
        combine_check(combiner->record(*pending, output, command, storage, uploads), "record deferred sum without submitting");
        combine_check(uploads == 2 && !output.empty(), "only owned host Expert and Shared activations upload");
        combine_check(command.reset() == 0, "cancel unsubmitted Combine command");
    }
    ActivationBuffer actual(rows, columns);
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        std::vector<ncnn::VkMat> storage;
        uint64_t uploads = 0;
        ncnn::VkCompute command(context->device(), context->command_optimization_flags());
        ncnn::VkMat output;
        combine_check(!foreign_combiner->record(*pending, output, command, storage, uploads) && output.empty() && uploads == 0,
                      "foreign recorder rejects without recording");
        combine_check(combiner->record(*pending, output, command, storage, uploads), "retry deferred recorder");
        ncnn::VkMat staging;
        ncnn::Option option;
        option.use_vulkan_compute = true;
        option.use_fp16_packed = false;
        option.use_fp16_storage = false;
        option.use_fp16_arithmetic = false;
        option.use_bf16_storage = false;
        option.use_packing_layout = false;
        option.blob_vkallocator = context->blob_allocator();
        option.workspace_vkallocator = context->blob_allocator();
        option.staging_vkallocator = context->staging_allocator();
        combine_check(prepare_staging_batch(staging, rows, columns, context->staging_allocator())
                          && record_prepared_activation_staging_download(output, rows, columns, staging, command, context->device(), option),
                      "record consumer download in the same command");
        combine_check(submit_compute_and_wait(command, context->device()) == 0 && copy_staging_to_cpu_batch(staging, actual),
                      "one completed command for deferred sum and consumer");
    }
    combine_exact(actual, expected, "deferred owned canonical result after caller destruction and retry");
    pending.reset();
    combine_check(weak_large.expired() && weak_negative.expired(), "pending releases retained device sources");
}

static std::shared_ptr<TensorData> combine_mxfp4(uint32_t rows, uint32_t columns)
{
    auto result = std::make_shared<TensorData>();
    result->dtype = DType::MxFp4;
    result->shape = {rows, columns};
    std::vector<uint8_t> scales(static_cast<size_t>(rows) * columns / 32, 125);
    result->mxfp4_scales.assign(scales.data(), scales.size());
    std::vector<uint8_t> blocks(static_cast<size_t>(rows) * columns / 2, 0x21);
    result->mxfp4_blocks.assign(blocks.data(), blocks.size());
    return result;
}
static void combine_test_expert_publication()
{
    constexpr uint32_t columns = 32;
    const auto runtime = create_vulkan_runtime();
    auto seed = combine_seed(runtime, columns);
    VulkanExpertBackend backend(65536, get_default_gpu_index(), {}, runtime, combine_flags);
    auto gate = combine_mxfp4(64, columns), down = combine_mxfp4(columns, columns);
    backend.admit("retained", gate, nullptr, down, nullptr, 0, 0.0f, ExpertActivation::DeepSeekSwiGlu);
    backend.wait_for_background_work();
    ActivationBuffer normalized(4, columns);
    for (size_t row = 0; row < normalized.rows(); ++row)
        for (uint32_t column = 0; column < columns; ++column) normalized.row(row)[column] = static_cast<float>((column * 5 + row * 3) % 17 + 1) / 128.0f;
    auto normalized_gpu = combine_upload(*seed, normalized);
    std::array<std::vector<ExpertRoute>, 2> routes;
    routes[0] = {{2, 0, 0.5f}};
    routes[1] = {{3, 0, 0.25f}, {0, 0, 0.75f}, {1, 0, 0.5f}};
    std::array<ActivationBuffer, 2> input, host_output;
    std::array<ExpertBackendRequest, 2> requests;
    for (size_t i = 0; i < requests.size(); ++i)
    {
        input[i].reset(routes[i].size(), columns, false);
        for (size_t row = 0; row < routes[i].size(); ++row) std::copy_n(normalized.row(routes[i][row].token_index), columns, input[i].row(row));
        requests[i] = {"retained", &input[i], &host_output[i], 0, {}};
    }
    auto baseline = backend.submit_batch(requests);
    combine_check(baseline && baseline->wait() == std::vector<ExpertBackendExecutionResult>(2, ExpertBackendExecutionResult::Executed)
                      && baseline->commit(),
                  "GPU host baseline expert output");
    baseline.reset();
    const auto reference = host_output;
    std::array<std::shared_ptr<DeviceTensor_vulkan>, 2> resident_output = {normalized_gpu, normalized_gpu};
    for (size_t i = 0; i < requests.size(); ++i)
    {
        requests[i].device_input = normalized_gpu;
        requests[i].device_routes = routes[i];
        requests[i].device_output = &resident_output[i];
    }
    auto canceled = backend.submit_batch(requests);
    combine_check(canceled && canceled->wait() == std::vector<ExpertBackendExecutionResult>(2, ExpertBackendExecutionResult::Executed), "resident expert private execution");
    canceled->abort();
    combine_check(!canceled->commit() && resident_output[0] == normalized_gpu && resident_output[1] == normalized_gpu, "abort publishes no resident output");
    combine_exact(host_output[0], reference[0], "abort preserves first host output");
    combine_exact(host_output[1], reference[1], "abort preserves second host output");
    canceled.reset();
    const auto before = get_vulkan_statistics(runtime);
    auto completed = backend.submit_batch(requests);
    combine_check(completed && completed->wait() == std::vector<ExpertBackendExecutionResult>(2, ExpertBackendExecutionResult::Executed), "resident expert completion");
    combine_check(resident_output[0] == normalized_gpu && resident_output[1] == normalized_gpu, "wait does not publish private output");
    combine_check(completed->commit(), "resident expert atomic commit");
    completed.reset();
    const auto after = get_vulkan_statistics(runtime);
    combine_check(after.batch_downloads == before.batch_downloads, "GPU Expert outputs avoid host download");
    backend.wait_for_background_work();
    for (size_t i = 0; i < requests.size(); ++i)
    {
        combine_check(resident_output[i] && resident_output[i] != normalized_gpu && host_output[i].rows() == 0, "commit exposes device output only");
        ActivationBuffer actual;
        combine_check(MoeCombine_vulkan::materialize(*resident_output[i], actual), "resident expert view stays alive after submission");
        combine_exact(actual, reference[i], "gathered resident expert output equals GPU host path");
    }
    std::array<MoeCombineInput_vulkan, 2> inputs = {{{&reference[0], resident_output[0], routes[0]}, {&reference[1], resident_output[1], routes[1]}}};
    const auto expected = combine_reference(inputs, 4, columns, nullptr);
    std::shared_ptr<DeviceTensor_vulkan> combined;
    combine_check(MoeCombine_vulkan::forward(inputs, 4, columns, nullptr, {}, combined), "retained Expert views feed canonical Combine");
    ActivationBuffer actual;
    combine_check(MoeCombine_vulkan::materialize(*combined, actual), "retained Expert aggregate downloads for verification");
    combine_exact(actual, expected, "retained Expert route/row parity");
}
#endif
} // namespace moe
} // namespace ncnn

int main()
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::moe::get_gpu_count() == 0)
    {
        std::cout << "SKIP: no Vulkan device\n";
        return 77;
    }
    try
    {
        ncnn::moe::combine_test_pipeline_lifetime();
        ncnn::moe::combine_test_deferred_ownership();
        ncnn::moe::combine_test_order(1, false);
        ncnn::moe::combine_test_order(3, false);
        ncnn::moe::combine_test_order(3, true);
        ncnn::moe::combine_test_expert_publication();
        std::cout << "Vulkan MoE Combine tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Vulkan MoE Combine failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan disabled\n";
    return 77;
#endif
}
