#ifndef NCNN_MOE_GPU_SEGMENT_TEST_FIXTURE_H
#define NCNN_MOE_GPU_SEGMENT_TEST_FIXTURE_H

#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/gateddeltanet_vulkan.h"
#include "backends/ncnn/gpusegment_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/moecombine_vulkan.h"
#include "backends/ncnn/router_vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "engine/executor.h"
#include "engine/expert.h"
#include "engine/graphresources.h"
#include "engine/sessionstate.h"
#include "graph/compiler.h"
#include "kernels/attention.h"
#include "kernels/ops.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void segment_check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

#if NCNN_MOE_WITH_VULKAN
static constexpr uint32_t segment_hidden = 128;
static constexpr uint32_t segment_intermediate = 128;
static constexpr uint32_t segment_experts = 2;
static constexpr uint32_t segment_top_k = 2;
static constexpr uint32_t segment_layers = 2;
static constexpr uint64_t segment_flags = OptimizationDefaultFlags & ~OptimizationCpuFastSilu;

static TensorData segment_weight(std::vector<uint32_t> shape, uint32_t seed, bool norm = false)
{
    TensorData tensor;
    tensor.dtype = DType::BFloat16;
    tensor.shape = std::move(shape);
    tensor.bfloat16_data.resize(tensor.element_count());
    for (size_t index = 0; index < tensor.bfloat16_data.size(); ++index)
    {
        const int pattern = static_cast<int>((index * 7 + seed * 13 + index / segment_hidden) % 29) - 14;
        const float value = norm ? 1.0f + pattern * 0.00390625f : pattern * 0.00390625f;
        tensor.bfloat16_data[index] = float_to_bfloat16(value);
    }
    return tensor;
}

static MoeModelDescriptor segment_descriptor(bool native_delta)
{
    // A synthetic, model-neutral graph deliberately mixes two stateful kernels.
    MoeModelDescriptor descriptor;
    descriptor.model_type = "generic_resident_segment_test";
    descriptor.vocabulary_size = 32;
    descriptor.hidden_size = segment_hidden;
    descriptor.intermediate_size = segment_intermediate;
    descriptor.attention_head_count = 4;
    descriptor.kv_head_count = 2;
    descriptor.head_dimension = 32;
    descriptor.expert_count = segment_experts;
    descriptor.experts_per_token = segment_top_k;
    descriptor.activation_dtype = DType::BFloat16;
    descriptor.kv_cache_dtype = DType::BFloat16;
    descriptor.norm_epsilon = 1e-6f;
    descriptor.layers.resize(segment_layers);
    for (uint32_t index = 0; index < segment_layers; ++index)
    {
        auto& layer = descriptor.layers[index];
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention.kind = index == 0 ? AttentionKind::Standard : AttentionKind::GatedDeltaNet;
        layer.attention.head_count = 4;
        layer.attention.kv_head_count = 2;
        layer.attention.head_dimension = 32;
        layer.attention.value_head_dimension = 32;
        layer.attention.qk_rope_head_dimension = index == 0 ? 32 : 0;
        layer.attention.initial_context_length = 16;
        layer.attention.convolution_kernel_size = index == 0 ? 0 : 2;
        layer.attention.projection_weight_dtype = native_delta && index == 1 ? DType::Float32 : DType::BFloat16;
        layer.moe.expert_count = segment_experts;
        layer.moe.top_k = segment_top_k;
        layer.moe.intermediate_size = segment_intermediate;
        layer.moe.shared_expert_count = 1;
        layer.moe.layout = ExpertLayout::PackedGateUpDown;
        layer.moe.expert_weight_dtype = DType::BFloat16;
        layer.moe.shared_expert_weight_dtype = DType::BFloat16;
        layer.moe.flags = MoeDescriptorSharedExpertGate;
    }
    return descriptor;
}

static WeightMapping segment_mapping(bool native_delta)
{
    WeightMapping mapping;
    uint32_t seed = 0;
    const auto add = [&](const std::string& name, std::vector<uint32_t> shape, bool norm = false) {
        mapping.emplace(name, segment_weight(std::move(shape), ++seed, norm));
    };
    add("token_embedding.weight", {32, segment_hidden});
    add("final_norm.weight", {segment_hidden}, true);
    add("lm_head.weight", {32, segment_hidden});
    for (uint32_t index = 0; index < segment_layers; ++index)
    {
        const std::string prefix = "layers." + std::to_string(index) + ".";
        add(prefix + "pre_attention_norm.weight", {segment_hidden}, true);
        add(prefix + "pre_ffn_norm.weight", {segment_hidden}, true);
        add(prefix + "router.weight", {segment_experts, segment_hidden});
        add(prefix + "shared_expert.gate.weight", {segment_intermediate, segment_hidden});
        add(prefix + "shared_expert.up.weight", {segment_intermediate, segment_hidden});
        add(prefix + "shared_expert.down.weight", {segment_hidden, segment_intermediate});
        add(prefix + "shared_expert.router_gate.weight", {1, segment_hidden});
        for (uint32_t expert = 0; expert < segment_experts; ++expert)
        {
            const std::string expert_prefix = prefix + "experts." + std::to_string(expert) + ".";
            add(expert_prefix + "gate_up.weight", {segment_intermediate * 2, segment_hidden});
            add(expert_prefix + "down.weight", {segment_hidden, segment_intermediate});
        }
        if (index == 0)
        {
            add(prefix + "attention.query.weight", {128, segment_hidden});
            add(prefix + "attention.key.weight", {64, segment_hidden});
            add(prefix + "attention.value.weight", {64, segment_hidden});
            add(prefix + "attention.output.weight", {segment_hidden, 128});
        }
        else
        {
            add(prefix + "attention.delta.qkv.weight", {256, segment_hidden});
            add(prefix + "attention.delta.z.weight", {128, segment_hidden});
            add(prefix + "attention.delta.beta.weight", {4, segment_hidden});
            add(prefix + "attention.delta.alpha.weight", {4, segment_hidden});
            add(prefix + "attention.delta.convolution.weight", {256, 1, 2});
            add(prefix + "attention.delta.time_bias", {4});
            add(prefix + "attention.delta.decay_log", {4});
            add(prefix + "attention.delta.norm.weight", {32}, true);
            add(prefix + "attention.output.weight", {segment_hidden, 128});
        }
    }
    if (native_delta)
    {
        // Preserve every weight value exactly while selecting the native Qnk
        // record path, whose input RMSNorm is a separate recorded operation.
        for (const char* suffix : {"attention.delta.qkv.weight", "attention.delta.z.weight", "attention.delta.beta.weight",
                                   "attention.delta.alpha.weight", "attention.output.weight"})
        {
            auto& tensor = mapping.at(std::string("layers.1.") + suffix);
            tensor.float32_data.reserve(tensor.bfloat16_data.size());
            for (const uint16_t value : tensor.bfloat16_data)
                tensor.float32_data.push_back(bfloat16_to_float(value));
            tensor.bfloat16_data.clear();
            tensor.dtype = DType::Float32;
        }
    }
    return mapping;
}

static CompiledModel segment_compile(bool gpu, bool native_delta)
{
    CompilerOption options;
    options.optimization_flags = segment_flags;
    options.num_concurrent_sessions = 2;
    if (gpu)
    {
        options.flags |= BackendVulkanDense | BackendVulkanAttention | BackendVulkanExperts;
        options.device_index = get_default_gpu_index();
        options.vulkan_runtime = create_vulkan_runtime();
    }
    CompiledModel model;
    const auto result = compile_model(segment_descriptor(native_delta), segment_mapping(native_delta), model,
                                      gpu ? HybridMode::HybridExperts : HybridMode::CpuOnly, options);
    segment_check(static_cast<bool>(result), result ? "" : result.error().message);
    if (gpu)
    {
        model.expert_backend = create_vulkan_expert_backend(UINT64_C(16) * 1024 * 1024, options.device_index,
                                                            {}, model.vulkan_runtime, options.optimization_flags);
        segment_check(static_cast<bool>(model.expert_backend), "cannot create resident segment Expert backend");
        segment_check(model.gpu_segments.size() == segment_layers && model.gpu_head,
                      "generic graph did not prepare its resident segments and head");
        for (size_t index = 0; index < segment_layers; ++index)
            segment_check(model.gpu_segments[index] && GpuSegment_vulkan::supports(model, model.graph.layer_plans[index]),
                          "a supported generic Standard/GDN segment was not prepared at layer " + std::to_string(index));
        for (const auto& node : model.graph.nodes)
            if (node.type == ExecutionNodeType::Attention || node.type == ExecutionNodeType::Router
                || node.type == ExecutionNodeType::ExpertGroup || node.type == ExecutionNodeType::Combine
                || node.type == ExecutionNodeType::FinalNorm || node.type == ExecutionNodeType::LmHead)
                segment_check(node.backend == ExecutionBackend::Vulkan, "resident graph node was silently placed on CPU: " + node.name);
    }
    else
        segment_check(model.gpu_segments.empty() && !model.gpu_head, "CPU model unexpectedly owns a resident GPU segment");
    return model;
}

static void segment_compare(std::span<const float> actual, std::span<const float> expected, const std::string& context)
{
    segment_check(actual.size() == expected.size(), context + " shape mismatch");
    float scale = 0.0f;
    for (const float value : expected) scale = std::max(scale, std::abs(value));
    // Both use FP32 activations and BF16 KV; weight kernels differ in reduction order.
    const float tolerance = 2e-4f + 2e-3f * scale;
    for (size_t index = 0; index < actual.size(); ++index)
        segment_check(std::isfinite(actual[index]) && std::abs(actual[index] - expected[index]) <= tolerance,
                      context + " numerical mismatch at " + std::to_string(index) + ": " + std::to_string(actual[index])
                          + " vs " + std::to_string(expected[index]) + ", tolerance " + std::to_string(tolerance));
}

static void segment_gpu_routes(const SessionStatistics& statistics)
{
    segment_check(statistics.expert_assignments > 0 && statistics.expert_gpu_routes == statistics.expert_assignments
                      && statistics.expert_cpu_routes == 0 && statistics.expert_cpu_batches == 0 && statistics.expert_cpu_fallbacks == 0,
                  "selected resident Experts did not execute entirely on GPU");
}

static std::shared_ptr<const DeviceTensor_vulkan> segment_upload(const ActivationBuffer& input, const CompiledModel& model)
{
    TensorData identity;
    identity.dtype = DType::Float32;
    identity.shape = {segment_hidden, segment_hidden};
    identity.float32_data.assign(segment_hidden * segment_hidden, 0.0f);
    for (uint32_t index = 0; index < segment_hidden; ++index)
        identity.float32_data[static_cast<size_t>(index) * segment_hidden + index] = 1.0f;
    const auto seed = Linear::create(identity, nullptr, model.opt.vulkan_device_index, model.vulkan_runtime, model.opt.optimization_flags);
    segment_check(static_cast<bool>(seed), "cannot create device aggregate upload seed");
    auto command = CommandGraph_vulkan::create(*seed);
    auto result = std::make_shared<DeviceTensor_vulkan>();
    segment_check(command && command->upload(input, *result) && command->submit() && command->wait(),
                  "cannot publish device aggregate fixture");
    return result;
}

#endif

} // namespace moe
} // namespace ncnn

#endif
