#include "backends/ncnn/linear.h"
#include "backends/ncnn/modelpipeline.h"
#include "backends/ncnn/vulkan.h"
#include "engine/expert.h"
#include "engine/sessionstate.h"
#include "graph/compiledmodel.h"
#include "graph/compiler.h"
#include "kernels/float8.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static void shared_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void shared_check_output(const ActivationBuffer& actual, const ActivationBuffer& expected,
                                const std::string& label, float absolute = 0.003f, float relative = 0.01f)
{
    shared_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + ": shape");
    float maximum = 0.0f;
    for (size_t index = 0; index < expected.values().size(); ++index)
    {
        const float reference = expected.values()[index];
        const float observed = actual.values()[index];
        maximum = std::max(maximum, std::abs(reference));
        shared_check(std::isfinite(observed) && std::abs(observed - reference) <= absolute + relative * std::abs(reference),
                     label + ": element " + std::to_string(index) + " actual=" + std::to_string(observed)
                         + " expected=" + std::to_string(reference));
    }
    shared_check(maximum > 0.001f, label + ": nonzero oracle");
}

static TensorData shared_fp8(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData weight;
    weight.dtype = DType::Float8E4M3;
    weight.shape = {rows, columns};
    const size_t count = static_cast<size_t>(rows) * columns;
    std::shared_ptr<uint8_t[]> storage(new uint8_t[count], std::default_delete<uint8_t[]>());
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
        {
            const int sample = static_cast<int>((row * 17 + column * 13 + column / 7 + seed * 11) % 31) - 15;
            storage[static_cast<size_t>(row) * columns + column] = float_to_float8_e4m3(static_cast<float>(sample) * 0.0625f);
        }
    weight.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
    weight.mapped_size = count;
    weight.quantization_scales.resize(static_cast<size_t>((rows + 127) / 128) * ((columns + 127) / 128));
    for (size_t index = 0; index < weight.quantization_scales.size(); ++index)
        weight.quantization_scales[index] = index % 2 == 0 ? 0.125f : 0.0625f;
    return weight;
}

static void shared_model(CompiledModel& model, uint32_t hidden, uint32_t intermediate,
                         float limit, const VulkanRuntimePtr& runtime, bool gpu, bool expect_gpu = true)
{
    model.descriptor.hidden_size = hidden;
    model.descriptor.layers.resize(1);
    model.descriptor.layers[0].attention.kind = AttentionKind::None;
    model.graph.layer_plans.resize(1);
    auto& plan = model.graph.layer_plans[0];
    plan.attention.kind = AttentionKind::None;
    plan.vulkan_device_index = 0;
    plan.moe.has_shared_expert = true;
    plan.moe.shared_expert.activation = ExpertActivation::DeepSeekSwiGlu;
    plan.moe.shared_expert.activation_limit = limit;
    auto add = [&](const char* name, TensorData weight) {
        auto result = model.weights.add(name, std::move(weight));
        shared_check(static_cast<bool>(result), "shared weight binding");
        return result.value();
    };
    model.lm_head_weight = add("lm_head.weight", shared_fp8(128, hidden, 4));
    auto& shared = plan.moe.shared_expert;
    shared.gate_weight = add("shared.gate.weight", shared_fp8(intermediate, hidden, 1));
    shared.up_weight = add("shared.up.weight", shared_fp8(intermediate, hidden, 2));
    shared.down_weight = add("shared.down.weight", shared_fp8(hidden, intermediate, 3));
    model.opt.optimization_flags = OptimizationDefaultFlags;
    model.opt.vulkan_device_index = 0;
    model.opt.hybrid_mode = gpu ? HybridMode::HybridExperts : HybridMode::CpuOnly;
    model.vulkan_runtime = runtime;
    if (gpu)
    {
        CompilerOption option;
        option.flags |= BackendVulkanDense;
        option.device_index = 0;
        option.vulkan_runtime = runtime;
        auto result = prepare_model_pipeline(model, option);
        shared_check(static_cast<bool>(result), "FP8 Shared Expert preparation");
        shared_check(support_vulkan_shared_experts(model.operators, plan.moe) == expect_gpu, "FP8 Shared Expert GPU capability");
        shared_check(plan.moe.fused_shared_input_bfloat16_operator == invalid_compiled_operator_handle, "FP8 uses separate resident weights");
        for (auto handle : {shared.gate_weight, shared.up_weight, shared.down_weight})
            shared_check(static_cast<bool>(model.operators.at_weight(handle).float8) == expect_gpu, "Shared chain preparation is atomic");
    }
    else
        model.operators.bind_weight_count(model.weights.size());
}

static void shared_test_optional_preparation()
{
    const auto runtime = create_vulkan_runtime();
    CompiledModel model;
    CompiledModel cpu;
    // Gate and Up support 129 output columns. Down's 129 input columns are a
    // valid CPU tail, but fail GPU creation after those first two operators.
    shared_model(model, 128, 129, 0.0f, runtime, true, false);
    shared_model(cpu, 128, 129, 0.0f, {}, false);
    shared_check(static_cast<bool>(model.operators.at_weight(model.lm_head_weight).float8),
                 "optional Shared failure preserves prepared non-Shared operator");
    release_vulkan_dense_host_copies(model);
    const auto& moe = model.graph.layer_plans[0].moe;
    const auto& shared = moe.shared_expert;
    for (auto handle : {shared.gate_weight, shared.up_weight, shared.down_weight})
        shared_check(!model.operators.at_weight(handle).float8 && !model.weights.at(handle).float8_values().empty()
                         && !model.weights.at(handle).quantization_scales.empty(),
                     "partial Shared GPU preparation never publishes or releases CPU weights");
    ActivationBuffer input(1, 128);
    for (uint32_t column = 0; column < input.columns(); ++column)
        input.row(0)[column] = static_cast<float>(static_cast<int>((column * 7 + column / 11) % 29) - 14) * 0.0625f;
    ExpertWorkspace workspace;
    ActivationBuffer expected;
    ActivationBuffer actual;
    forward_shared_expert(cpu, cpu.graph.layer_plans[0].moe, input, expected, workspace);
    const auto before = get_vulkan_statistics(runtime);
    forward_shared_expert(model, moe, input, actual, workspace);
    shared_check(get_vulkan_statistics(runtime).compute_submissions == before.compute_submissions,
                 "optional Shared preparation failure uses CPU");
    shared_check_output(actual, expected, "optional Shared CPU tail parity", 0.0f, 0.0f);
}

static std::shared_ptr<DeviceTensor_vulkan> shared_upload(const ActivationBuffer& input,
                                                          const VulkanRuntimePtr& runtime)
{
    TensorData identity;
    identity.dtype = DType::Float32;
    identity.shape = {input.columns(), input.columns()};
    identity.float32_data.assign(static_cast<size_t>(input.columns()) * input.columns(), 0.0f);
    for (uint32_t column = 0; column < input.columns(); ++column)
        identity.float32_data[static_cast<size_t>(column) * input.columns() + column] = 1.0f;
    auto seed = Linear::create(identity, nullptr, LinearDevice::Vulkan, 0, runtime, OptimizationDefaultFlags);
    shared_check(static_cast<bool>(seed), "resident Shared input producer");
    auto graph = CommandGraph_vulkan::create(*seed);
    auto result = std::make_shared<DeviceTensor_vulkan>();
    shared_check(graph && graph->upload(input, *result) && graph->submit() && graph->wait(), "resident Shared input upload");
    graph.reset();
    seed.reset();
    shared_check(!result->empty(), "resident Shared input outlives producer");
    return result;
}

static void shared_test(size_t rows, uint32_t hidden, float limit)
{
    const auto runtime = create_vulkan_runtime();
    CompiledModel model;
    CompiledModel cpu;
    shared_model(model, hidden, 256, limit, runtime, true);
    shared_model(cpu, hidden, 256, limit, {}, false);
    const auto& moe = model.graph.layer_plans[0].moe;
    const auto& shared = moe.shared_expert;
    auto gate = model.operators.at_weight(shared.gate_weight).float8;
    const auto up = model.operators.at_weight(shared.up_weight).float8;
    const auto down = model.operators.at_weight(shared.down_weight).float8;
    ActivationBuffer input(rows, hidden);
    for (size_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < hidden; ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((column * 7 + row * 5 + column / 11) % 29) - 14) * 0.0625f;
    ExpertWorkspace workspace;
    ActivationBuffer expected;
    forward_shared_expert(cpu, cpu.graph.layer_plans[0].moe, input, expected, workspace);
    if (limit > 0.0f)
    {
        cpu.graph.layer_plans[0].moe.shared_expert.activation_limit = 0.0f;
        ActivationBuffer unclipped;
        forward_shared_expert(cpu, cpu.graph.layer_plans[0].moe, input, unclipped, workspace);
        float difference = 0.0f;
        for (size_t index = 0; index < expected.values().size(); ++index)
            difference = std::max(difference, std::abs(expected.values()[index] - unclipped.values()[index]));
        shared_check(difference > 0.001f, "clip limit exercises actual DeepSeek SwiGLU clipping");
        cpu.graph.layer_plans[0].moe.shared_expert.activation_limit = limit;
    }
    ActivationBuffer host_output;
    const auto host_before = get_vulkan_statistics(runtime);
    forward_shared_expert(model, moe, input, host_output, workspace);
    const auto host_after = get_vulkan_statistics(runtime);
    shared_check(host_after.compute_submissions == host_before.compute_submissions + 1, "host Shared GPU chain uses one submit");
    shared_check(host_after.shared_expert_swiglu_fusions == host_before.shared_expert_swiglu_fusions + 1, "host Shared GPU completed batch counted");
    shared_check_output(host_output, expected, "FP8 Shared host GPU CPU parity");

    auto resident = shared_upload(input, runtime);
    std::shared_ptr<const DeviceTensor_vulkan> output;
    const auto before = get_vulkan_statistics(runtime);
    forward_shared_expert(model, moe, input, host_output, workspace, resident, &output);
    const auto after = get_vulkan_statistics(runtime);
    shared_check(output && !output->empty() && host_output.rows() == 0, "resident Shared output publication");
    shared_check(output->rows() == rows && output->columns() == hidden, "resident Shared output logical shape");
    shared_check(after.compute_submissions == before.compute_submissions + 1, "resident Shared chain uses one submit");
    shared_check(after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads,
                 "resident Shared chain avoids activation upload and download");
    ActivationBuffer actual;
    shared_check(gate->materialize(*output, actual), "resident Shared materialization");
    shared_check_output(actual, expected, "FP8 Shared device GPU CPU parity");
    ActivationBuffer unchanged;
    shared_check(gate->materialize(*resident, unchanged), "resident Shared input re-read");
    shared_check_output(unchanged, input, "resident Shared input immutable", 0.0f, 0.0f);

    // Later projections may reuse scratch allocations, but an earlier completed
    // result and its input remain independently owned by their device tensors.
    const auto retained_output = output;
    ActivationBuffer next = input;
    for (size_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < hidden; ++column)
            next.row(row)[column] = -next.row(row)[column] * 0.5f;
    auto next_resident = shared_upload(next, runtime);
    ActivationBuffer next_expected;
    forward_shared_expert(cpu, cpu.graph.layer_plans[0].moe, next, next_expected, workspace);
    forward_shared_expert(model, moe, next, host_output, workspace, next_resident, &output);
    shared_check(output && output != retained_output, "later Shared batch replaces output owner");
    shared_check(gate->materialize(*output, actual), "later Shared output materialization");
    shared_check_output(actual, next_expected, "later FP8 Shared device parity");
    shared_check(gate->materialize(*retained_output, actual), "retained Shared output materialization");
    shared_check_output(actual, expected, "retained Shared output ownership");
    const auto reject_before = get_vulkan_statistics(runtime);
    shared_check(!gate->forward_swiglu_chain_device(*resident, *up, *down, ExpertActivation::DeepSeekSwiGlu, limit, *resident),
                 "Shared input output alias rejected");
    DeviceTensor_vulkan rejected;
    shared_check(!gate->forward_swiglu_chain_device(*resident, *up, *down, ExpertActivation::GptOssSwiGlu, limit, rejected)
                     && rejected.empty(),
                 "unsupported activation preserves output state");
    shared_check(get_vulkan_statistics(runtime).compute_submissions == reject_before.compute_submissions,
                 "rejected Shared device requests do not submit");

    // A foreign context cannot be imported; the canonical host input still
    // executes the supported GPU chain and publishes a host fallback result.
    auto foreign = shared_upload(input, create_vulkan_runtime());
    forward_shared_expert(model, moe, input, host_output, workspace, foreign, &output);
    shared_check(!output && host_output.rows() == rows, "foreign context uses host Shared chain");
    shared_check_output(host_output, expected, "foreign context Shared fallback parity");

    release_vulkan_dense_host_copies(model);
    shared_check(model.weights.at(model.lm_head_weight).float8_values().empty(), "optional release still releases other dense host weights");
    for (auto handle : {shared.gate_weight, shared.up_weight, shared.down_weight})
    {
        shared_check(!model.weights.at(handle).float8_values().empty() && !model.weights.at(handle).quantization_scales.empty(),
                     "FP8 Shared retains canonical CPU fallback weights");
        model.operators.at_weight_mutable(handle).float8.reset();
    }
    shared_check(!support_vulkan_shared_experts(model.operators, moe), "removed Shared GPU operators remove capability");
    const auto cpu_before = get_vulkan_statistics(runtime);
    forward_shared_expert(model, moe, input, host_output, workspace, resident, &output);
    shared_check(!output && get_vulkan_statistics(runtime).compute_submissions == cpu_before.compute_submissions,
                 "Shared CPU fallback publishes only host result");
    shared_check_output(host_output, expected, "Shared CPU fallback after host release", 0.0f, 0.0f);
    resident.reset();
    next_resident.reset();
    shared_check(gate->materialize(*retained_output, actual), "completed Shared output outlives producer inputs and operator table");
    shared_check_output(actual, expected, "Shared retained lifetime parity");
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
        ncnn::moe::shared_test_optional_preparation();
        ncnn::moe::shared_test(1, 128, 0.0f);
        ncnn::moe::shared_test(4, 128, 0.125f);
        ncnn::moe::shared_test(5, 256, 0.0f);
        std::cout << "FP8 Shared Expert Vulkan tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FP8 Shared Expert Vulkan test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
