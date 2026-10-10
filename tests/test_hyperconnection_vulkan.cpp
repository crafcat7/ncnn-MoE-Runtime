#include "backends/ncnn/hyperconnection_vulkan.h"
#include "backends/ncnn/layerhead_vulkan.h"
#include "backends/ncnn/linear.h"
#include "graph/compiledoperator.h"
#include "kernels/float8.h"
#include "kernels/fastmath.h"
#include "kernels/ops.h"
#include "backends/ncnn/vulkancontext.h"
#include "kernels/hyperconnection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static void hyper_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static int hyper_submit(ncnn::VkCompute& cmd, VulkanRuntimeState& state, const ncnn::VulkanDevice* device)
{
    const int ret = submit_compute_and_wait(cmd, device);
    if (ret == 0)
        ++state.compute_submissions;
    return ret;
}

static TensorData hyper_weight(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData result;
    result.dtype = DType::Float32;
    result.shape = {rows, columns};
    result.float32_data.resize(static_cast<size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const int sample = static_cast<int>((row * 19 + column * 7 + column / 11 + seed * 5) % 37) - 18;
            result.float32_data[static_cast<size_t>(row) * columns + column] = static_cast<float>(sample) * 0.0078125f;
        }
    }
    return result;
}

static TensorData hyper_vector(std::span<const float> values)
{
    TensorData result;
    result.dtype = DType::Float32;
    result.shape = {static_cast<uint32_t>(values.size())};
    result.float32_data.assign(values.begin(), values.end());
    return result;
}

static TensorData hyper_bases(uint32_t copies, bool head, uint32_t seed)
{
    TensorData result;
    result.dtype = DType::Float32;
    result.shape = {head ? copies : copies * (copies + 2)};
    result.float32_data.resize(result.shape[0]);
    for (uint32_t column = 0; column < result.shape[0]; ++column)
    {
        const int sample = static_cast<int>((column * 11 + column / 3 + seed * 7) % 23) - 11;
        result.float32_data[column] = static_cast<float>(sample) * 0.375f;
    }
    return result;
}

static ActivationBuffer hyper_input(uint32_t rows, uint32_t columns, uint32_t seed)
{
    ActivationBuffer result(rows, columns);
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const int sample = static_cast<int>((row * 17 + column * 13 + column / 5 + seed * 3) % 43) - 21;
            result.row(row)[column] = static_cast<float>(sample) * 0.0625f + static_cast<float>(column / 37) * 0.001f;
        }
    }
    return result;
}

static ncnn::Mat hyper_host(const ActivationBuffer& input)
{
    ncnn::Mat result(static_cast<int>(input.columns()), static_cast<int>(input.rows()), sizeof(float));
    for (size_t row = 0; row < input.rows(); ++row)
        std::memcpy(result.row(static_cast<int>(row)), input.row(row), static_cast<size_t>(input.columns()) * sizeof(float));
    return result;
}

static void hyper_compare(const ncnn::Mat& actual, std::span<const float> expected, const std::string& label,
                          float absolute_tolerance = 0.0003f, float relative_tolerance = 0.0003f)
{
    hyper_check(actual.elemsize == sizeof(float) && actual.elempack == 1 && static_cast<size_t>(actual.w) * actual.h == expected.size(), label + ": shape/storage");
    const float* values = reinterpret_cast<const float*>(actual.data);
    for (size_t index = 0; index < expected.size(); ++index)
    {
        const float tolerance = absolute_tolerance + relative_tolerance * std::abs(expected[index]);
        hyper_check(std::isfinite(values[index]) && std::abs(values[index] - expected[index]) <= tolerance,
                    label + ": index " + std::to_string(index) + " actual=" + std::to_string(values[index]) + " expected=" + std::to_string(expected[index]));
    }
}

static void hyper_test_numerical(uint32_t copies, uint32_t hidden_size, uint32_t rows,
                                 uint32_t iterations, float norm_epsilon, float hyper_epsilon, bool cancel_first = false)
{
    const std::string label = "M=" + std::to_string(copies) + " H=" + std::to_string(hidden_size) + " rows=" + std::to_string(rows) + " iterations=" + std::to_string(iterations);
    const uint32_t mix_columns = copies * (copies + 2);
    const TensorData function = hyper_weight(mix_columns, copies * hidden_size, 3);
    const TensorData head_function = hyper_weight(copies, copies * hidden_size, 7);
    const std::vector<float> scales = {0.4f, -0.75f, 1.25f};
    const std::vector<float> head_scales = {-0.55f};
    const TensorData scale = hyper_vector(scales);
    const TensorData head_scale = hyper_vector(head_scales);
    const TensorData base = hyper_bases(copies, false, 2);
    const TensorData head_base = hyper_bases(copies, true, 5);
    ActivationBuffer input = hyper_input(rows, copies * hidden_size, 3);
    ActivationBuffer branch = hyper_input(rows, hidden_size, 9);
    if (hidden_size == 1)
        input.reset(rows, copies * hidden_size, true);
    HyperConnectionScratch scratch;
    HyperConnectionMix expected_mix;
    ActivationBuffer expected_post;
    ActivationBuffer expected_head;
    hyper_check(static_cast<bool>(forward_hyper_connection_pre(input, function, scale, base, copies, iterations,
                                                               norm_epsilon, hyper_epsilon, expected_mix, scratch, 0)),
                "CPU pre oracle");
    const ActivationBuffer expected_projection = scratch.projection;
    std::vector<float> expected_pre(rows * copies);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t copy = 0; copy < copies; ++copy)
            expected_pre[row * copies + copy] = 1.0f / (1.0f + float_approximate_exp(-expected_projection.row(row)[copy] * scales[0] - base.float32_data[copy])) + hyper_epsilon;
    hyper_check(static_cast<bool>(forward_hyper_connection_post(branch, input, expected_mix, copies, expected_post)), "CPU post oracle");
    hyper_check(static_cast<bool>(forward_hyper_connection_head(input, head_function, head_scale, head_base, copies,
                                                                norm_epsilon, hyper_epsilon, expected_head, scratch, 0)),
                "CPU head oracle");
    if (copies > 1 && iterations == 1)
    {
        bool asymmetric = false;
        for (uint32_t source = 0; source < copies; ++source)
            for (uint32_t destination = 0; destination < copies; ++destination)
                asymmetric |= std::abs(expected_mix.combine[source * copies + destination] - expected_mix.combine[destination * copies + source]) > 0.001f;
        hyper_check(asymmetric, "transpose oracle requires asymmetric combine");
    }
    const VulkanRuntimePtr runtime = create_vulkan_runtime();
    auto op = HyperConnection_vulkan::create(function, scale, base, copies, iterations, norm_epsilon, hyper_epsilon, 0, runtime, OptimizationDefaultFlags);
    auto head = HyperConnection_vulkan::create(head_function, head_scale, head_base, copies, iterations, norm_epsilon, hyper_epsilon, 0, runtime, OptimizationDefaultFlags);
    hyper_check(op && head && !op->is_head() && head->is_head(), "create HC operators");
    hyper_check(op->input_columns() == copies * hidden_size && op->multiplier() == copies && op->vulkan_context() == head->vulkan_context(), "HC operator shape/context");
    const std::shared_ptr<VulkanContext> context = op->vulkan_context();
    const ncnn::Mat input_host = hyper_host(input);
    const ncnn::Mat branch_host = hyper_host(branch);
    ncnn::VkMat input_device;
    ncnn::VkMat branch_device;
    ncnn::VkMat post_device;
    ncnn::VkMat head_device;
    HyperConnectionMix_vulkan mix;
    HyperConnectionWorkspace_vulkan workspace;
    HyperConnectionWorkspace_vulkan head_workspace;
    ncnn::Mat observed_input;
    ncnn::Mat projection_host;
    ncnn::Mat pre_host;
    ncnn::Mat reduced_host;
    ncnn::Mat post_coefficient_host;
    ncnn::Mat combine_host;
    ncnn::Mat post_host;
    ncnn::Mat head_host;
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const uint64_t submissions = get_vulkan_statistics(runtime).compute_submissions;
    if (cancel_first)
    {
        ncnn::VkMat abandoned_input;
        ncnn::VkMat abandoned_branch;
        ncnn::VkMat abandoned_post;
        ncnn::VkMat abandoned_head;
        HyperConnectionMix_vulkan abandoned_mix;
        HyperConnectionWorkspace_vulkan abandoned_work;
        HyperConnectionWorkspace_vulkan abandoned_head_work;
        // Keep tensors alive until the unsubmitted recorder is destroyed.
        ncnn::VkCompute abandoned(context->device(), context->command_optimization_flags());
        abandoned.record_clone(input_host, abandoned_input, op->option());
        abandoned.record_clone(branch_host, abandoned_branch, op->option());
        hyper_check(op->record_pre(abandoned_input, abandoned_mix, abandoned_work, abandoned), label + " abandoned pre");
        hyper_check(op->record_post(abandoned_branch, abandoned_input, abandoned_mix, abandoned_post, abandoned), label + " abandoned post");
        hyper_check(head->record_head(abandoned_input, abandoned_head, abandoned_head_work, abandoned), label + " abandoned head");
    }
    hyper_check(get_vulkan_statistics(runtime).compute_submissions == submissions, label + " abandoned command never submits");
    // Reuse exactly the same operators in a new recorder and the existing CPU parity assertions.
    ncnn::VkCompute cmd(context->device(), context->command_optimization_flags());
    cmd.record_clone(input_host, input_device, op->option());
    cmd.record_clone(branch_host, branch_device, op->option());
    hyper_check(op->record_pre(input_device, mix, workspace, cmd), "record pre");
    hyper_check(op->record_post(branch_device, input_device, mix, post_device, cmd), "record post");
    hyper_check(head->record_head(input_device, head_device, head_workspace, cmd), "record head");
    hyper_check(get_vulkan_statistics(runtime).compute_submissions == submissions, "HC record must not submit");
    cmd.record_download(input_device, observed_input, op->option());
    cmd.record_download(workspace.projection, projection_host, op->option());
    cmd.record_download(workspace.pre, pre_host, op->option());
    cmd.record_download(mix.reduced, reduced_host, op->option());
    cmd.record_download(mix.post, post_coefficient_host, op->option());
    cmd.record_download(mix.combine, combine_host, op->option());
    cmd.record_download(post_device, post_host, op->option());
    cmd.record_download(head_device, head_host, op->option());
    hyper_check(hyper_submit(cmd, context->runtime_state(), context->device()) == 0, "HC submit");
    hyper_check(get_vulkan_statistics(runtime).compute_submissions == submissions + 1, "HC stages share one submission");
    hyper_compare(observed_input, input.values(), label + " untouched GPU input", 0.0f, 0.0f);
    hyper_compare(projection_host, expected_projection.values(), label + " projection");
    hyper_compare(pre_host, expected_pre, label + " pre coefficients");
    hyper_compare(reduced_host, expected_mix.reduced.values(), label + " pre reduced");
    hyper_compare(post_coefficient_host, expected_mix.post, "post coefficients");
    hyper_compare(combine_host, expected_mix.combine, "Sinkhorn combine");
    hyper_compare(post_host, expected_post.values(), "post residual orientation");
    hyper_compare(head_host, expected_head.values(), "head output");
}

static void hyper_test_device_chain()
{
    constexpr uint32_t copies = 4;
    constexpr uint32_t hidden_size = 257;
    constexpr uint32_t rows = 3;
    constexpr uint32_t iterations = 20;
    constexpr float norm_epsilon = 1e-6f;
    constexpr float hyper_epsilon = 1e-5f;
    const TensorData first_function = hyper_weight(24, copies * hidden_size, 2);
    const TensorData second_function = hyper_weight(24, copies * hidden_size, 11);
    const TensorData head_function = hyper_weight(copies, copies * hidden_size, 5);
    const TensorData first_base = hyper_bases(copies, false, 3);
    const TensorData second_base = hyper_bases(copies, false, 7);
    const TensorData head_base = hyper_bases(copies, true, 2);
    const std::vector<float> scales = {0.9f, -0.7f, 1.1f};
    const std::vector<float> head_scales = {0.6f};
    const TensorData scale = hyper_vector(scales);
    const TensorData head_scale = hyper_vector(head_scales);
    ActivationBuffer input = hyper_input(rows, copies * hidden_size, 8);
    HyperConnectionScratch scratch;
    HyperConnectionMix first_expected;
    HyperConnectionMix second_expected;
    ActivationBuffer first_post_expected;
    ActivationBuffer second_post_expected;
    ActivationBuffer expected;
    hyper_check(static_cast<bool>(forward_hyper_connection_pre(input, first_function, scale, first_base, copies, iterations,
                                                               norm_epsilon, hyper_epsilon, first_expected, scratch, 0)),
                "chain CPU first pre");
    hyper_check(static_cast<bool>(forward_hyper_connection_post(first_expected.reduced, input, first_expected, copies, first_post_expected)), "chain CPU first post");
    hyper_check(static_cast<bool>(forward_hyper_connection_pre(first_post_expected, second_function, scale, second_base, copies, iterations,
                                                               norm_epsilon, hyper_epsilon, second_expected, scratch, 0)),
                "chain CPU second pre");
    hyper_check(static_cast<bool>(forward_hyper_connection_post(second_expected.reduced, first_post_expected, second_expected, copies, second_post_expected)), "chain CPU second post");
    hyper_check(static_cast<bool>(forward_hyper_connection_head(second_post_expected, head_function, head_scale, head_base, copies,
                                                                norm_epsilon, hyper_epsilon, expected, scratch, 0)),
                "chain CPU head");
    const VulkanRuntimePtr runtime = create_vulkan_runtime();
    auto first = HyperConnection_vulkan::create(first_function, scale, first_base, copies, iterations, norm_epsilon, hyper_epsilon, 0, runtime, OptimizationDefaultFlags);
    auto second = HyperConnection_vulkan::create(second_function, scale, second_base, copies, iterations, norm_epsilon, hyper_epsilon, 0, runtime, OptimizationDefaultFlags);
    auto head = HyperConnection_vulkan::create(head_function, head_scale, head_base, copies, iterations, norm_epsilon, hyper_epsilon, 0, runtime, OptimizationDefaultFlags);
    hyper_check(first && second && head, "create chain operators");
    const std::shared_ptr<VulkanContext> context = first->vulkan_context();
    HyperConnectionMix_vulkan first_mix;
    HyperConnectionMix_vulkan second_mix;
    HyperConnectionWorkspace_vulkan first_workspace;
    HyperConnectionWorkspace_vulkan second_workspace;
    HyperConnectionWorkspace_vulkan head_workspace;
    ncnn::VkMat input_device;
    ncnn::VkMat first_post;
    ncnn::VkMat second_post;
    ncnn::VkMat final_device;
    const ncnn::Mat input_host = hyper_host(input);
    ncnn::Mat output_host;
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    ncnn::VkCompute cmd(context->device(), context->command_optimization_flags());
    const uint64_t submissions = get_vulkan_statistics(runtime).compute_submissions;
    cmd.record_clone(input_host, input_device, first->option());
    hyper_check(first->record_pre(input_device, first_mix, first_workspace, cmd), "chain first pre");
    hyper_check(first->record_post(first_mix.reduced, input_device, first_mix, first_post, cmd), "chain first post");
    hyper_check(second->record_pre(first_post, second_mix, second_workspace, cmd), "chain second pre");
    hyper_check(second->record_post(second_mix.reduced, first_post, second_mix, second_post, cmd), "chain second post");
    hyper_check(head->record_head(second_post, final_device, head_workspace, cmd), "chain head");
    hyper_check(get_vulkan_statistics(runtime).compute_submissions == submissions, "chain records keep activation on device");
    cmd.record_download(final_device, output_host, first->option());
    hyper_check(hyper_submit(cmd, context->runtime_state(), context->device()) == 0, "chain submit");
    hyper_check(get_vulkan_statistics(runtime).compute_submissions == submissions + 1, "chain one submission/one final download");
    hyper_compare(output_host, expected.values(), "device-resident HC chain");
}

static TensorData hyper_lm_weight(DType dtype)
{
    TensorData result = hyper_weight(128, 128, 17);
    if (dtype == DType::BFloat16)
    {
        result.dtype = dtype;
        result.bfloat16_data.resize(result.float32_data.size());
        for (size_t index = 0; index < result.float32_data.size(); ++index)
            result.bfloat16_data[index] = float_to_bfloat16(result.float32_data[index]);
        result.float32_data.clear();
    }
    else if (dtype == DType::Float8E4M3)
    {
        result.dtype = dtype;
        std::shared_ptr<uint8_t[]> storage(new uint8_t[result.float32_data.size()], std::default_delete<uint8_t[]>());
        for (size_t index = 0; index < result.float32_data.size(); ++index)
            storage[index] = float_to_float8_e4m3(result.float32_data[index]);
        result.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
        result.mapped_size = result.float32_data.size();
        result.float32_data.clear();
        result.quantization_scales = {1.0f};
    }
    return result;
}

static void hyper_test_layer_head(DType dtype, uint32_t rows)
{
    constexpr uint32_t copies = 4;
    constexpr uint32_t hidden_size = 128;
    constexpr float norm_epsilon = 1e-6f;
    constexpr float hyper_epsilon = 1e-5f;
    const TensorData function = hyper_weight(copies, copies * hidden_size, 3);
    const TensorData base = hyper_bases(copies, true, 7);
    const std::vector<float> scales = {0.45f};
    const TensorData scale = hyper_vector(scales);
    std::vector<float> norm_values(hidden_size);
    for (uint32_t column = 0; column < hidden_size; ++column)
        norm_values[column] = 0.7f + static_cast<float>(column % 13) * 0.025f;
    const TensorData norm_weight = hyper_vector(norm_values);
    const TensorData lm_weight = hyper_lm_weight(dtype);
    const ActivationBuffer input = hyper_input(rows, copies * hidden_size, 5);
    HyperConnectionScratch scratch;
    ActivationBuffer headed;
    hyper_check(static_cast<bool>(forward_hyper_connection_head(input, function, scale, base, copies,
                                                                norm_epsilon, hyper_epsilon, headed, scratch, 0)),
                "layer head CPU HC");
    const ActivationBuffer expected_hidden = forward_rms_norm(headed, norm_weight, norm_epsilon, 0.0f);
    const VulkanRuntimePtr runtime = create_vulkan_runtime();
    CompiledOperator compiled;
    if (dtype == DType::Float8E4M3)
        compiled.float8 = Float8Linear_vulkan::create(lm_weight, nullptr, 1, 0, runtime, OptimizationDefaultFlags);
    else if (dtype == DType::BFloat16)
        compiled.bfloat16 = Bfloat16Linear_vulkan::create(lm_weight, nullptr, 0, runtime, OptimizationDefaultFlags);
    else
        compiled.linear = Linear::create(lm_weight, nullptr, LinearDevice::Vulkan, 0, runtime, OptimizationDefaultFlags);
    ActivationBuffer expected;
    bool projected = false;
    if (compiled.float8)
        projected = compiled.float8->forward(expected_hidden, expected);
    else if (compiled.bfloat16)
        projected = compiled.bfloat16->forward(expected_hidden, expected);
    else if (compiled.linear)
        projected = compiled.linear->forward(expected_hidden, expected);
    hyper_check(projected, "existing resident LM projection oracle");
    const long previous_count = compiled.float8     ? compiled.float8.use_count()
                                : compiled.bfloat16 ? compiled.bfloat16.use_count()
                                                    : compiled.linear.use_count();
    auto op = LayerHead_vulkan::create(function, scale, base, norm_weight, compiled, copies,
                                       norm_epsilon, hyper_epsilon, 0, runtime, OptimizationDefaultFlags);
    hyper_check(static_cast<bool>(op), "create LayerHead");
    const long shared_count = compiled.float8     ? compiled.float8.use_count()
                              : compiled.bfloat16 ? compiled.bfloat16.use_count()
                                                  : compiled.linear.use_count();
    hyper_check(shared_count == previous_count + 1, "LayerHead shares existing LM projection owner");
    // The head keeps the already-prepared projection alive independently.
    compiled = {};
    const std::shared_ptr<VulkanContext> context = op->vulkan_context();
    const ncnn::Mat input_host = hyper_host(input);
    ncnn::VkMat input_device;
    ncnn::VkMat logits;
    ncnn::VkMat normalized;
    ncnn::VkMat reused_logits;
    HyperConnectionWorkspace_vulkan hc_workspace;
    HyperConnectionWorkspace_vulkan hidden_workspace;
    std::vector<ncnn::VkMat> temporaries;
    std::vector<ncnn::VkMat> hidden_temporaries;
    ncnn::Mat output_host;
    ncnn::Mat normalized_host;
    ncnn::Mat reused_host;
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const uint64_t submissions = get_vulkan_statistics(runtime).compute_submissions;
    ncnn::VkCompute cmd(context->device(), context->command_optimization_flags());
    cmd.record_clone(input_host, input_device, op->option());
    hyper_check(op->record(input_device, logits, hc_workspace, temporaries, cmd), "record LayerHead");
    hyper_check(temporaries.size() == (dtype == DType::Float8E4M3 ? 3u : 2u), "HC/RMS/LM device stages recorded");
    hyper_check(op->record_hidden(input_device, normalized, hidden_workspace, hidden_temporaries, cmd), "record LayerHead normalized hidden");
    hyper_check(op->record_logits(hidden_temporaries.back(), reused_logits, hidden_temporaries, cmd), "reuse normalized device logits");
    hyper_check(get_vulkan_statistics(runtime).compute_submissions == submissions, "LayerHead never submits internally");
    cmd.record_download(logits, output_host, op->option());
    cmd.record_download(normalized, normalized_host, op->option());
    cmd.record_download(reused_logits, reused_host, op->option());
    hyper_check(hyper_submit(cmd, context->runtime_state(), context->device()) == 0, "LayerHead caller submission");
    hyper_check(get_vulkan_statistics(runtime).compute_submissions == submissions + 1, "LayerHead stages one submission");
    hyper_compare(normalized_host, expected_hidden.values(), "LayerHead normalized parity");
    hyper_compare(output_host, expected.values(), "LayerHead resident projection parity", 0.003f, 0.003f);
    hyper_compare(reused_host, expected.values(), "LayerHead reuses normalized projection", 0.003f, 0.003f);
}

static void hyper_test_rejection()
{
    constexpr uint32_t copies = 4;
    const TensorData function = hyper_weight(24, 148, 3);
    const TensorData base = hyper_bases(copies, false, 2);
    const std::vector<float> scales = {0.4f, 0.8f, 0.6f};
    const TensorData scale = hyper_vector(scales);
    const VulkanRuntimePtr runtime = create_vulkan_runtime();
    hyper_check(!HyperConnection_vulkan::create(function, scale, base, 0, 20, 1e-6f, 1e-5f, 0, runtime, OptimizationDefaultFlags), "reject multiplier zero");
    hyper_check(!HyperConnection_vulkan::create(function, scale, base, 4, 0, 1e-6f, 1e-5f, 0, runtime, OptimizationDefaultFlags), "reject iterations zero");
    hyper_check(!HyperConnection_vulkan::create(function, scale, base, 4, 20, std::numeric_limits<float>::infinity(), 1e-5f, 0, runtime, OptimizationDefaultFlags), "reject infinite norm epsilon");
    auto op = HyperConnection_vulkan::create(function, scale, base, copies, 20, 1e-6f, 1e-5f, 0, runtime, OptimizationDefaultFlags);
    hyper_check(static_cast<bool>(op), "create rejection operator");
    const std::shared_ptr<VulkanContext> context = op->vulkan_context();
    const ActivationBuffer input = hyper_input(2, 148, 3);
    const ncnn::Mat input_host = hyper_host(input);
    ncnn::VkMat input_device;
    ncnn::VkMat wrong_columns;
    ncnn::VkMat output;
    HyperConnectionMix_vulkan mix;
    HyperConnectionWorkspace_vulkan workspace;
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    ncnn::VkCompute cmd(context->device(), context->command_optimization_flags());
    cmd.record_clone(input_host, input_device, op->option());
    hyper_check(op->record_pre(input_device, mix, workspace, cmd), "valid pre before rejections");
    const ncnn::VkBufferMemory* preserved_reduced = mix.reduced.data;
    const ncnn::VkBufferMemory* preserved_projection = workspace.projection.data;
    wrong_columns.create(36, 2, sizeof(float), context->blob_allocator());
    hyper_check(!op->record_pre(wrong_columns, mix, workspace, cmd), "reject pre shape");
    hyper_check(!op->record_head(input_device, output, workspace, cmd), "reject head on pre operator");
    hyper_check(!op->record_post(wrong_columns, input_device, mix, output, cmd), "reject branch shape");
    hyper_check(!op->record_post(mix.reduced, input_device, mix, input_device, cmd), "reject output alias");
    HyperConnectionMix_vulkan other_context = mix;
    other_context.context.reset();
    hyper_check(!op->record_post(mix.reduced, input_device, other_context, output, cmd), "reject mix context mismatch");
    hyper_check(mix.reduced.data == preserved_reduced && workspace.projection.data == preserved_projection && output.empty(), "invalid calls preserve outputs/workspace");
    hyper_check(hyper_submit(cmd, context->runtime_state(), context->device()) == 0, "rejection valid command completes");
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
        ncnn::moe::hyper_test_numerical(1, 37, 1, 1, 1e-6f, 1e-5f);
        ncnn::moe::hyper_test_numerical(3, 37, 3, 1, 1e-6f, 1e-5f);
        ncnn::moe::hyper_test_numerical(4, 4096, 1, 20, 1e-6f, 1e-5f, true);
        ncnn::moe::hyper_test_numerical(4, 129, 3, 2, 0.01f, 0.005f);
        ncnn::moe::hyper_test_numerical(8, 33, 2, 1, 1e-5f, 0.01f);
        ncnn::moe::hyper_test_numerical(4, 1, 2, 20, 1e-6f, 1e-5f);
        ncnn::moe::hyper_test_device_chain();
        ncnn::moe::hyper_test_layer_head(ncnn::moe::DType::Float32, 3);
        ncnn::moe::hyper_test_layer_head(ncnn::moe::DType::BFloat16, 1);
        ncnn::moe::hyper_test_layer_head(ncnn::moe::DType::BFloat16, 3);
        ncnn::moe::hyper_test_layer_head(ncnn::moe::DType::Float8E4M3, 1);
        ncnn::moe::hyper_test_layer_head(ncnn::moe::DType::Float8E4M3, 3);
        ncnn::moe::hyper_test_rejection();
        std::cout << "Vulkan HyperConnection tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Vulkan HyperConnection test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
