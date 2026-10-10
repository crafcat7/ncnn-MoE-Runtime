#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/experttransfer_vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "kernels/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace ncnn {
namespace moe {

static void upload_check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

#if NCNN_MOE_WITH_VULKAN
static ncnn::Option upload_option(const std::shared_ptr<VulkanContext>& context)
{
    ncnn::Option option;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.blob_vkallocator = context->blob_allocator();
    option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    return option;
}

static void verify_payload(const std::shared_ptr<VulkanContext>& context, const ncnn::VkMat& device,
                           size_t bytes, uint8_t seed)
{
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    ncnn::VkCompute command(context->device(), context->command_optimization_flags());
    ncnn::Mat downloaded;
    command.record_download(device, downloaded, upload_option(context));
    upload_check(command.submit_and_wait() == 0 && !downloaded.empty(), "download uploaded byte storage");
    for (size_t index = 0; index < bytes; ++index)
        upload_check(static_cast<const uint8_t*>(downloaded.data)[index] == static_cast<uint8_t>(index * 13 + seed),
                     "mapped staging upload preserves every byte");
}

static void direct_upload_pool()
{
    auto runtime = create_vulkan_runtime();
    auto context = VulkanContext::acquire(0, runtime, OptimizationDefaultFlags);
    upload_check(static_cast<bool>(context), "upload context");
    VulkanExpertWeightAllocator weights(context->device(), 16384);
    VulkanUploadStagingAllocator first_staging(context->device(), 8192);
    VulkanUploadStagingAllocator second_staging(context->device(), 8192);
    const auto option = upload_option(context);
    constexpr size_t bytes = 4096;
    ncnn::VkMat first, second;
    {
        // Recording uses two isolated command pools and independent staging
        // allocators, including while the foreground command mutex is held.
        const std::lock_guard<std::mutex> foreground(context->command_mutex());
        VulkanWeightUploadBatch first_batch(context, &first_staging);
        VulkanWeightUploadBatch second_batch(context, &second_staging);
        auto a = first_batch.prepare_storage(bytes, &weights);
        auto b = second_batch.prepare_storage(bytes, &weights);
        upload_check(a.size() == bytes && b.size() == bytes && a.data() != b.data(), "isolated mapped staging reservations");
        for (size_t index = 0; index < bytes; ++index)
        {
            a[index] = static_cast<uint8_t>(index * 13 + 1);
            b[index] = static_cast<uint8_t>(index * 13 + 5);
        }
        upload_check(first_batch.record_prepared_storage(first, option, &weights)
                         && second_batch.record_prepared_storage(second, option, &weights),
                     "record both mapped reservations");
        upload_check(first_batch.submit() && second_batch.submit(), "submit both isolated lanes");
    }
    verify_payload(context, first, bytes, 1);
    verify_payload(context, second, bytes, 5);
    first.release();
    second.release();
    for (uint8_t seed = 8; seed < 40; ++seed)
    {
        ncnn::VkMat destination;
        {
            VulkanWeightUploadBatch batch(context, &first_staging);
            auto mapped = batch.prepare_storage(bytes, &weights);
            upload_check(mapped.size() == bytes, "reused mapped staging size");
            for (size_t index = 0; index < bytes; ++index)
                mapped[index] = static_cast<uint8_t>(index * 13 + seed);
            upload_check(batch.record_prepared_storage(destination, option, &weights) && batch.submit(), "reuse command pool, CB, fences and consumed semaphore");
        }
        verify_payload(context, destination, bytes, seed);
    }

    // Abandon a reservation before recording, then a recorded command before
    // submission. Neither case can leave a busy pool lane or submitted copy.
    {
        VulkanWeightUploadBatch cancelled(context, &first_staging);
        upload_check(cancelled.prepare_storage(bytes, &weights).size() == bytes, "cancel unrecorded reservation");
        upload_check(!cancelled.submit(), "unrecorded staging cannot submit");
    }
    {
        VulkanWeightUploadBatch cancelled(context, &first_staging);
        auto mapped = cancelled.prepare_storage(bytes, &weights);
        std::fill(mapped.begin(), mapped.end(), uint8_t(77));
        ncnn::VkMat destination;
        upload_check(cancelled.record_prepared_storage(destination, option, &weights), "cancel recorded unsubmitted batch");
    }
    {
        VulkanWeightUploadBatch retry(context, &first_staging);
        auto mapped = retry.prepare_storage(bytes, &weights);
        for (size_t index = 0; index < bytes; ++index)
            mapped[index] = static_cast<uint8_t>(index * 13 + 3);
        ncnn::VkMat destination;
        upload_check(retry.record_prepared_storage(destination, option, &weights) && retry.submit(), "retry resets cancelled command recording");
        verify_payload(context, destination, bytes, 3);
    }
    {
        VulkanWeightUploadBatch invalid(context, &first_staging);
        upload_check(invalid.prepare_storage(7, &weights).empty() && !invalid.submit(), "reject illegal VkBufferCopy byte count");
    }
    {
        VulkanWeightUploadBatch poisoned(context, &first_staging);
        auto mapped = poisoned.prepare_storage(bytes, &weights);
        upload_check(mapped.size() == bytes, "poison fixture mapped reservation");
        std::fill(mapped.begin(), mapped.end(), uint8_t(33));
        ncnn::VkMat destination;
        upload_check(poisoned.record_prepared_storage(destination, option, &weights), "poison fixture first records a valid copy");
        upload_check(poisoned.prepare_storage(7, &weights).empty() && !poisoned.submit(), "illegal reservation poisons existing recorded lane");
    }
    {
        VulkanWeightUploadBatch retry(context, &first_staging);
        auto mapped = retry.prepare_storage(bytes, &weights);
        upload_check(mapped.size() == bytes, "poison retry mapped reservation");
        for (size_t index = 0; index < bytes; ++index)
            mapped[index] = static_cast<uint8_t>(index * 13 + 17);
        ncnn::VkMat destination;
        upload_check(retry.record_prepared_storage(destination, option, &weights) && retry.submit(), "fresh command resources replace poisoned lane");
        verify_payload(context, destination, bytes, 17);
    }
    {
        ncnn::VkWeightAllocator legacy(context->device(), size_t(16384));
        ncnn::Mat source(static_cast<int>(bytes), sizeof(uint8_t));
        for (size_t index = 0; index < bytes; ++index)
            static_cast<uint8_t*>(source.data)[index] = static_cast<uint8_t>(index * 13 + 9);
        ncnn::VkMat destination;
        {
            VulkanWeightUploadBatch batch(context);
            upload_check(batch.record(source, destination, option, &legacy) && batch.submit(), "legacy exclusive allocator keeps native upload compatibility");
        }
        verify_payload(context, destination, bytes, 9);
    }
}

static std::shared_ptr<TensorData> upload_weight(uint32_t rows, uint32_t seed)
{
    auto weight = std::make_shared<TensorData>();
    weight->dtype = DType::MxFp4;
    weight->shape = {rows, 32};
    weight->mxfp4_blocks.resize(rows * 16);
    weight->mxfp4_scales.resize(rows);
    std::fill_n(weight->mxfp4_scales.data(), rows, uint8_t(123));
    for (size_t index = 0; index < weight->mxfp4_blocks.size(); ++index)
        weight->mxfp4_blocks[index] = static_cast<uint8_t>(((index * 3 + seed) % 16) | (((index * 5 + seed * 7) % 16) << 4));
    return weight;
}

static void factory_direct_staging_ab()
{
    const auto gate = upload_weight(64, 2);
    const auto down = upload_weight(32, 4);
    const uint64_t flags = OptimizationDefaultFlags & ~OptimizationCpuMxfp4Q8 & ~OptimizationCpuFastSilu;
    auto runtime = create_vulkan_runtime();
    ActivationBuffer input(1, 32);
    for (uint32_t column = 0; column < 32; ++column)
        input.row(0)[column] = (static_cast<int>(column % 9) - 4) * 0.03125f;
    for (DType bias_type : {DType::Float32, DType::BFloat16})
    {
        TensorData gate_bias, down_bias;
        gate_bias.dtype = down_bias.dtype = bias_type;
        gate_bias.shape = {64};
        down_bias.shape = {32};
        if (bias_type == DType::Float32)
        {
            gate_bias.float32_data.assign(64, 0.03125f);
            down_bias.float32_data.assign(32, -0.0625f);
        }
        else
        {
            gate_bias.bfloat16_data.assign(64, float_to_bfloat16(0.03125f));
            down_bias.bfloat16_data.assign(32, float_to_bfloat16(-0.0625f));
        }
        ActivationBuffer activated;
        forward_gate_up_mxfp4(*gate, &gate_bias, input, ExpertActivation::DeepSeekSwiGlu, 4.0f, activated, flags);
        const auto expected = forward_linear(*down, down_bias, activated, flags);
        for (bool direct : {false, true})
        {
            const auto selected_flags = direct ? flags | OptimizationVulkanExpertDirectStaging : flags & ~OptimizationVulkanExpertDirectStaging;
            auto backend = create_vulkan_expert_backend(8192, 0, {}, runtime, selected_flags);
            ScopedExpertBackendForeground foreground(backend);
            std::shared_ptr<const void> pin;
            upload_check(backend && backend->prepare_demand("packing-ab", gate, &gate_bias, down, &down_bias, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu, pin), "MXFP4 factory packs gate/down and both bias types");
            ActivationBuffer actual(1, 32);
            ExpertBackendRequest request{"packing-ab", &input, &actual, 0, {}};
            auto submission = backend->submit_batch(std::span<const ExpertBackendRequest>(&request, 1));
            upload_check(submission && submission->reservations().size() == 1
                             && submission->reservations()[0] == ExpertBackendExecutionResult::Executed,
                         "factory A/B retains GPU execution");
            const auto results = submission->wait();
            upload_check(results.size() == 1 && results[0] == ExpertBackendExecutionResult::Executed && submission->commit(), "factory A/B GPU submission completes");
            for (size_t index = 0; index < actual.values().size(); ++index)
                upload_check(std::isfinite(actual.values()[index]) && std::abs(actual.values()[index] - expected.values()[index]) < 0.0002f,
                             "direct and copied factory layouts match exact CPU oracle");
        }
    }
}

static std::shared_ptr<TensorData> submission_pool_weight(uint32_t rows, uint32_t seed)
{
    auto weight = upload_weight(rows, seed);
    // Break the upload-only fixture's row-periodic pattern. Distinct gate,
    // up and down rows keep the bias-free CPU oracle input-dependent/nonzero.
    for (size_t index = 0; index < weight->mxfp4_blocks.size(); ++index)
    {
        const uint8_t low = static_cast<uint8_t>((index * 7 + seed * 3 + index / 13) % 16);
        const uint8_t high = static_cast<uint8_t>((index * 11 + seed * 5 + index / 17) % 16);
        weight->mxfp4_blocks[index] = static_cast<uint8_t>(low | (high << 4));
    }
    return weight;
}

static ActivationBuffer submission_pool_input(size_t rows, uint32_t seed)
{
    ActivationBuffer input(rows, 32);
    for (size_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < 32; ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((column * 7 + row * 3 + seed) % 17) - 8) * 0.015625f;
    return input;
}

static ActivationBuffer submission_pool_oracle(const TensorData& gate, const TensorData& down,
                                               const ActivationBuffer& input, uint64_t flags)
{
    ActivationBuffer activated;
    forward_gate_up_mxfp4(gate, nullptr, input, ExpertActivation::DeepSeekSwiGlu, 4.0f, activated, flags);
    return forward_linear(down, activated, flags);
}

static void submission_pool_compare(const ActivationBuffer& actual, const ActivationBuffer& expected,
                                    const char* message, float tolerance = 0.0002f)
{
    upload_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), message);
    float maximum = 0.0f;
    for (size_t index = 0; index < expected.values().size(); ++index)
    {
        maximum = std::max(maximum, std::abs(expected.values()[index]));
        upload_check(std::isfinite(actual.values()[index])
                         && std::abs(actual.values()[index] - expected.values()[index]) <= tolerance,
                     message);
    }
    upload_check(maximum > 0.00000001f, "submission pool oracle is nonzero");
}

static void submission_pool_wait(ExpertSubmission& submission)
{
    const auto results = submission.wait();
    upload_check(results.size() == 1 && results[0] == ExpertBackendExecutionResult::Executed,
                 "submission pool completed GPU Expert result");
}

static void expert_submission_pool()
{
    const auto gate = submission_pool_weight(64, 9);
    const auto down = submission_pool_weight(32, 11);
    const uint64_t flags = OptimizationDefaultFlags & ~OptimizationCpuMxfp4Q8 & ~OptimizationCpuFastSilu
                           & ~OptimizationVulkanRouteAggregation;
    const auto runtime = create_vulkan_runtime();
    auto backend = create_vulkan_expert_backend(8192, 0, {}, runtime, flags);
    upload_check(static_cast<bool>(backend), "submission pool backend");
    std::shared_ptr<const void> pin;
    upload_check(backend->prepare_demand("submission-pool", gate, nullptr, down, nullptr, 0, 4.0f,
                                         ExpertActivation::DeepSeekSwiGlu, pin),
                 "submission pool resident Expert");
    const auto context = VulkanContext::acquire(0, runtime, flags);
    upload_check(static_cast<bool>(context), "submission pool context");

    std::array<ActivationBuffer, 3> held_inputs;
    std::array<ActivationBuffer, 3> held_outputs;
    std::array<ActivationBuffer, 3> held_expected;
    std::array<ActivationBuffer, 3> held_preserved;
    std::array<ExpertBackendRequest, 3> held_requests;
    std::array<std::unique_ptr<ExpertSubmission>, 3> held_submissions;
    for (size_t index = 0; index < held_requests.size(); ++index)
    {
        held_inputs[index] = submission_pool_input(2, static_cast<uint32_t>(index + 1));
        held_expected[index] = submission_pool_oracle(*gate, *down, held_inputs[index], flags);
        held_outputs[index].reset(2, 32, false);
        std::fill_n(held_outputs[index].row(0), held_outputs[index].values().size(), -7.0f - static_cast<float>(index));
        held_preserved[index] = held_outputs[index];
        held_requests[index] = {"submission-pool", &held_inputs[index], &held_outputs[index], 0, {}};
    }
    {
        // Prevent worker GPU completion while three independent live handles
        // reserve their private storage. Aborting cannot publish any output.
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        for (size_t index = 0; index < held_requests.size(); ++index)
        {
            held_submissions[index] = backend->submit_batch(std::span<const ExpertBackendRequest>(&held_requests[index], 1));
            upload_check(held_submissions[index] && held_submissions[index]->reservations().size() == 1
                             && held_submissions[index]->reservations()[0] == ExpertBackendExecutionResult::Executed,
                         "distinct in-flight submission reservations");
            submission_pool_compare(held_outputs[index], held_preserved[index], "in-flight output stays private", 0.0f);
        }
        held_submissions[0]->abort();
        held_submissions[0]->abort();
    }
    for (auto& submission : held_submissions)
        submission_pool_wait(*submission);
    upload_check(!held_submissions[0]->commit(), "aborted handle cannot publish after GPU completion");
    upload_check(held_submissions[1]->commit() && held_submissions[1]->commit(), "committed handle is idempotent");
    submission_pool_compare(held_outputs[0], held_preserved[0], "aborted output remains unchanged", 0.0f);
    submission_pool_compare(held_outputs[1], held_expected[1], "committed handle GPU output oracle");
    submission_pool_compare(held_outputs[2], held_preserved[2], "wait alone does not publish", 0.0f);

    ActivationBuffer output(4, 32);
    std::fill_n(output.row(0), output.values().size(), -13.0f);
    const ActivationBuffer barrier_input = submission_pool_input(1, 71);
    const ActivationBuffer barrier_expected = submission_pool_oracle(*gate, *down, barrier_input, flags);
    std::array<const std::byte*, 2> warmed_storage = {};
    std::array<uint64_t, 2> warmed_capacities = {};
    const std::array<size_t, 8> row_counts = {4, 4, 4, 4, 1, 1, 4, 4};
    for (size_t iteration = 0; iteration < row_counts.size(); ++iteration)
    {
        const ActivationBuffer input = submission_pool_input(row_counts[iteration], static_cast<uint32_t>(iteration + 17));
        const ActivationBuffer expected = submission_pool_oracle(*gate, *down, input, flags);
        const ActivationBuffer preserved = output;
        ExpertBackendRequest request{"submission-pool", &input, &output, 0, {}};
        auto submission = backend->submit_batch(std::span<const ExpertBackendRequest>(&request, 1));
        upload_check(submission && submission->reservations().size() == 1
                         && submission->reservations()[0] == ExpertBackendExecutionResult::Executed,
                     "pooled resident execution reservation");
        submission_pool_wait(*submission);
        submission_pool_compare(output, preserved, "pooled completion remains private before commit", 0.0f);
        upload_check(submission->commit(), "pooled completion publishes atomically");
        submission_pool_compare(output, expected, "reused private output matches CPU oracle");
        if (iteration == 2 || iteration == 3)
        {
            warmed_storage[iteration - 2] = output.bytes().data();
            warmed_capacities[iteration - 2] = output.allocated_bytes();
        }
        else if (iteration > 3)
        {
            upload_check(output.bytes().data() == warmed_storage[iteration % warmed_storage.size()]
                             && output.allocated_bytes() == warmed_capacities[iteration % warmed_capacities.size()],
                         "warm output storage survives shrink and regrow");
        }

        // The worker may briefly retain a completed WorkItem after notifying
        // waiters. A subsequent resident execution is a deterministic barrier
        // before releasing the fourth handle. All four pool slots are still
        // owned, so this fifth handle must use temporary, unretained storage.
        ActivationBuffer barrier_output;
        ExpertBackendRequest barrier_request{"submission-pool", &barrier_input, &barrier_output, 0, {}};
        auto barrier = backend->submit_batch(std::span<const ExpertBackendRequest>(&barrier_request, 1));
        upload_check(static_cast<bool>(barrier), "temporary overflow submission");
        submission_pool_wait(*barrier);
        upload_check(barrier->commit(), "temporary overflow publication");
        submission_pool_compare(barrier_output, barrier_expected, "overflow output CPU oracle");
        submission.reset();
        barrier.reset();
        submission_pool_compare(held_outputs[0], held_preserved[0], "pool reuse preserves aborted handle output", 0.0f);
        submission_pool_compare(held_outputs[1], held_expected[1], "pool reuse preserves committed handle output");
        submission_pool_compare(held_outputs[2], held_preserved[2], "pool reuse cannot publish an older uncommitted handle", 0.0f);
    }
    // Reuse the same warmed private Host output for a device-only request.
    // A retained tensor must publish an empty Host result, not old Host rows.
    TensorData identity;
    identity.dtype = DType::Float32;
    identity.shape = {32, 32};
    identity.float32_data.assign(32 * 32, 0.0f);
    for (uint32_t column = 0; column < 32; ++column)
        identity.float32_data[column * 32 + column] = 1.0f;
    const auto graph_seed = Linear::create(identity, nullptr, LinearDevice::Vulkan, 0, runtime, flags);
    upload_check(static_cast<bool>(graph_seed), "device-only pool graph seed");
    const ActivationBuffer normalized = submission_pool_input(4, 83);
    const ActivationBuffer resident_expected = submission_pool_oracle(*gate, *down, normalized, flags);
    auto device_input = std::make_shared<DeviceTensor_vulkan>();
    {
        auto graph = CommandGraph_vulkan::create(*graph_seed);
        upload_check(graph && graph->upload(normalized, *device_input) && graph->submit() && graph->wait(),
                     "device-only pool input upload");
    }
    std::array<ExpertRoute, 4> device_routes;
    for (uint32_t row = 0; row < device_routes.size(); ++row)
        device_routes[row] = {row, 0, 1.0f};
    std::shared_ptr<DeviceTensor_vulkan> device_output;
    ExpertBackendRequest resident_request{"submission-pool", nullptr, &output, 0, {}};
    resident_request.device_input = device_input;
    resident_request.device_routes = device_routes;
    resident_request.device_output = &device_output;
    resident_request.input_rows = normalized.rows();
    resident_request.input_columns = normalized.columns();
    const ActivationBuffer host_before_resident = output;
    auto resident_submission = backend->submit_batch(std::span<const ExpertBackendRequest>(&resident_request, 1));
    upload_check(static_cast<bool>(resident_submission), "device-only pooled submission");
    submission_pool_wait(*resident_submission);
    submission_pool_compare(output, host_before_resident, "device-only completion waits for publication", 0.0f);
    upload_check(!device_output && resident_submission->commit(), "device-only result publishes at commit");
    upload_check(output.rows() == 0 && output.columns() == 0 && output.bytes().empty()
                     && device_output && !device_output->empty(),
                 "reused WorkItem never publishes stale Host output for device-only result");
    {
        auto graph = CommandGraph_vulkan::create(*graph_seed);
        ActivationBuffer downloaded;
        upload_check(graph && graph->download(*device_output, downloaded) && graph->submit() && graph->wait(),
                     "device-only pooled output download");
        submission_pool_compare(downloaded, resident_expected, "device-only pooled output CPU oracle");
    }
    const std::weak_ptr<const DeviceTensor_vulkan> weak_input = device_input;
    device_input.reset();
    resident_request.device_input.reset();
    // A resident barrier ends the worker's extra WorkItem ownership before
    // destructor reference cleanup is checked.
    {
        ActivationBuffer barrier_output;
        ExpertBackendRequest barrier_request{"submission-pool", &barrier_input, &barrier_output, 0, {}};
        auto barrier = backend->submit_batch(std::span<const ExpertBackendRequest>(&barrier_request, 1));
        upload_check(static_cast<bool>(barrier), "device-only lifetime barrier");
        submission_pool_wait(*barrier);
        upload_check(barrier->commit(), "device-only lifetime barrier commit");
        submission_pool_compare(barrier_output, barrier_expected, "device-only lifetime barrier oracle");
    }
    resident_submission.reset();
    upload_check(weak_input.expired(), "released pooled submission drops device input references");
    upload_check(!held_submissions[0]->commit(), "aborted handle remains non-publishable after pool reuse");
    submission_pool_wait(*held_submissions[2]);
    upload_check(held_submissions[2]->commit(), "old uncommitted handle publishes its own completion");
    submission_pool_compare(held_outputs[2], held_expected[2], "old handle output independent of all pooled executions");
    held_submissions = {};
    pin.reset();
    backend.reset();
    {
        auto graph = CommandGraph_vulkan::create(*graph_seed);
        ActivationBuffer downloaded;
        upload_check(graph && graph->download(*device_output, downloaded) && graph->submit() && graph->wait(),
                     "completed pooled DeviceTensor survives backend destruction");
        submission_pool_compare(downloaded, resident_expected, "retained device result remains valid after backend destruction");
    }

    // Submission owns completion resources independently of the backend.
    // Backend destruction drains pending GPU work, then an external handle
    // still controls whether the completed result reaches its client.
    auto detached_backend = create_vulkan_expert_backend(8192, 0, {}, runtime, flags);
    upload_check(detached_backend && detached_backend->prepare_demand("detached-pool", gate, nullptr, down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu, pin),
                 "detached submission resident Expert");
    const ActivationBuffer detached_input = submission_pool_input(3, 97);
    const ActivationBuffer detached_expected = submission_pool_oracle(*gate, *down, detached_input, flags);
    ActivationBuffer detached_output(1, 32);
    std::fill_n(detached_output.row(0), detached_output.values().size(), -19.0f);
    const ActivationBuffer detached_preserved = detached_output;
    ExpertBackendRequest detached_request{"detached-pool", &detached_input, &detached_output, 0, {}};
    auto detached = detached_backend->submit_batch(std::span<const ExpertBackendRequest>(&detached_request, 1));
    upload_check(detached && detached->reservations().size() == 1
                     && detached->reservations()[0] == ExpertBackendExecutionResult::Executed,
                 "external submission before backend destruction");
    const std::weak_ptr<ExpertBackend> weak_backend = detached_backend;
    pin.reset();
    detached_backend.reset();
    upload_check(weak_backend.expired(), "external submission does not keep backend alive");
    submission_pool_wait(*detached);
    submission_pool_compare(detached_output, detached_preserved, "backend destruction cannot publish external output", 0.0f);
    upload_check(detached->commit(), "external submission commits after backend destruction");
    submission_pool_compare(detached_output, detached_expected, "detached completion CPU oracle");
    detached.reset();
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
        ncnn::moe::direct_upload_pool();
        ncnn::moe::factory_direct_staging_ab();
        ncnn::moe::expert_submission_pool();
        std::cout << "Expert direct staging, transfer and submission resource reuse passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Expert upload Vulkan test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
