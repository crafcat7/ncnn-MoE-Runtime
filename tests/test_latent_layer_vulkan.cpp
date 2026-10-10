#include "backends/ncnn/hyperconnection_vulkan.h"
#include "backends/ncnn/latentattention_vulkan.h"
#include "backends/ncnn/latentlayer_vulkan.h"
#include "backends/ncnn/moecombine_vulkan.h"
#include "backends/ncnn/layerhead_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/rmsnorm_vulkan.h"
#include "backends/ncnn/router_vulkan.h"
#include "backends/ncnn/vulkan.h"
#include "graph/compiledoperator.h"
#include "kernels/attention.h"
#include "kernels/float8.h"
#include "kernels/hyperconnection.h"
#include "kernels/latentattention.h"
#include "kernels/ops.h"
#include "kernels/statecache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static constexpr uint32_t layer_hidden = 128;
static constexpr uint32_t layer_copies = 4;
static constexpr float layer_norm_epsilon = 1e-6f;
static constexpr float layer_hyper_epsilon = 1e-5f;
static constexpr uint64_t layer_flags = static_cast<uint64_t>(OptimizationVulkanAttention)
                                        | static_cast<uint64_t>(OptimizationVulkanLatentInputRmsNorm)
                                        | static_cast<uint64_t>(OptimizationVulkanLatentCompressor);

static void layer_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void layer_compare(const ActivationBuffer& actual, const ActivationBuffer& expected,
                          const std::string& label, float absolute = 0.003f, float relative = 0.003f)
{
    layer_check(actual.rows() == expected.rows() && actual.columns() == expected.columns()
                    && actual.dtype() == DType::Float32 && expected.dtype() == DType::Float32,
                label + ": FP32 shape");
    layer_check(std::any_of(expected.values().begin(), expected.values().end(),
                            [](float value) { return std::abs(value) > 0.0001f; }),
                label + ": nonzero oracle");
    for (size_t index = 0; index < actual.values().size(); ++index)
    {
        const float value = actual.values()[index];
        const float reference = expected.values()[index];
        layer_check(std::isfinite(value) && std::abs(value - reference) <= absolute + relative * std::abs(reference),
                    label + ": index " + std::to_string(index) + " actual=" + std::to_string(value)
                        + " expected=" + std::to_string(reference));
    }
}

static TensorData layer_constant(std::vector<uint32_t> shape, float value)
{
    TensorData result;
    result.dtype = DType::Float32;
    result.shape = std::move(shape);
    result.float32_data.assign(static_cast<size_t>(result.element_count()), value);
    return result;
}

static TensorData layer_matrix(uint32_t rows, uint32_t columns, uint32_t seed, float scale)
{
    TensorData result = layer_constant({rows, columns}, 0.0f);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
            result.float32_data[static_cast<size_t>(row) * columns + column] = static_cast<float>(static_cast<int>((row * 19 + column * 7 + column / 11 + seed * 5) % 37) - 18) * scale;
    return result;
}

// Same block-scaled FP8 fixture family as test_latent_vulkan, with separate seeds per layer.
static TensorData layer_fp8(uint32_t rows, uint32_t columns, uint32_t seed, float scale)
{
    TensorData result;
    result.dtype = DType::Float8E4M3;
    result.shape = {rows, columns};
    const size_t count = static_cast<size_t>(result.element_count());
    std::shared_ptr<uint8_t[]> storage(new uint8_t[count], std::default_delete<uint8_t[]>());
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
        {
            const int sample = static_cast<int>((row * 17 + column * 13 + column / 7 + seed * 11) % 31) - 15;
            storage[static_cast<size_t>(row) * columns + column] = float_to_float8_e4m3(static_cast<float>(sample) * 0.0625f);
        }
    result.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
    result.mapped_size = count;
    result.quantization_scales.resize(static_cast<size_t>((rows + 127) / 128) * ((columns + 127) / 128));
    for (size_t index = 0; index < result.quantization_scales.size(); ++index)
        result.quantization_scales[index] = scale * (1.0f + static_cast<float>(index % 3) * 0.5f);
    return result;
}

static TensorHandle layer_add(WeightStore& weights, TensorData tensor)
{
    auto result = weights.add("latent_layer_" + std::to_string(weights.size()), std::move(tensor));
    layer_check(static_cast<bool>(result), "add layer fixture weight");
    return result.value();
}

static TensorData layer_base(uint32_t rows, uint32_t seed)
{
    TensorData result = layer_constant({rows}, 0.0f);
    for (uint32_t index = 0; index < rows; ++index)
        result.float32_data[index] = static_cast<float>(static_cast<int>((index * 11 + index / 3 + seed * 7) % 23) - 11) * 0.125f;
    return result;
}

struct LatentLayerFixture
{
    WeightStore weights;
    AttentionBlockPlan plan;
    CompiledOperatorTable operators;
    TensorData attention_function;
    TensorData ffn_function;
    TensorData attention_base;
    TensorData ffn_base;
    TensorData scale;
    TensorData ffn_norm;
    TensorData router_weight;
    TensorData router_bias;
    std::shared_ptr<LatentLayer_vulkan> layer;
    LayerCache cpu_cache;
    LayerCache staged_cache;
    LayerCache gpu_cache;

    LatentLayerFixture(const VulkanRuntimePtr& runtime, uint32_t seed)
        : attention_function(layer_matrix(24, layer_copies * layer_hidden, seed, 0.0005f)),
          ffn_function(layer_matrix(24, layer_copies * layer_hidden, seed + 3, 0.0005f)),
          attention_base(layer_base(24, seed)),
          ffn_base(layer_base(24, seed + 2)),
          scale(layer_constant({3}, 0.0f)),
          ffn_norm(layer_constant({layer_hidden}, 1.0f)),
          router_weight(layer_matrix(12, layer_hidden, seed + 11, 0.0001f)),
          router_bias(layer_constant({12}, 0.0f))
    {
        scale.float32_data = {0.4f, -0.75f, 1.25f};
        for (uint32_t column = 0; column < layer_hidden; ++column)
            ffn_norm.float32_data[column] = 0.75f + static_cast<float>(column % 13) * 0.02f;
        // Distinct selected scores keep rank stable across existing FP8 activation quantization.
        router_bias.float32_data = {-1.0f, 0.5f, -2.0f, 2.5f, -0.5f, 1.5f, -1.5f, 0.0f, -2.5f, 3.5f, -3.0f, 1.0f};
        plan.kind = AttentionKind::MultiHeadLatent;
        plan.head_count = 4;
        plan.kv_head_count = 1;
        plan.head_dimension = 512;
        plan.rope_head_dimension = 64;
        plan.sliding_window = 5;
        plan.query_lora_rank = 128;
        plan.output_lora_rank = 64;
        plan.output_group_count = 2;
        plan.initial_context_length = 4096;
        plan.compression_ratio = 0;
        plan.rope_theta = 10000.0f;
        plan.compressed_rope_theta = 1000000.0f;
        plan.rope_scaling_factor = 8.0f;
        plan.rope_ntk_alpha = 1.0f;
        plan.rope_ntk_beta = 32.0f;
        plan.norm_epsilon = layer_norm_epsilon;
        plan.pre_attention_norm_weight = layer_add(weights, layer_constant({128}, 1.0f));
        plan.query_a_weight = layer_add(weights, layer_fp8(128, 128, seed + 1, 0.25f));
        plan.query_norm_weight = layer_add(weights, layer_constant({128}, 1.0f));
        plan.query_b_weight = layer_add(weights, layer_fp8(2048, 128, seed + 5, 0.125f));
        plan.key_value_weight = layer_add(weights, layer_fp8(512, 128, seed + 7, 0.125f));
        plan.key_value_norm_weight = layer_add(weights, layer_constant({512}, 1.0f));
        TensorData sinks = layer_constant({4}, 0.0f);
        sinks.float32_data = {-1.5f, 0.0f, 3.0f, 12.0f};
        plan.sinks = layer_add(weights, std::move(sinks));
        // Small output scales retain a nonzero MLA contribution without amplifying quantization.
        plan.output_a_weight = layer_add(weights, layer_fp8(128, 1024, seed + 3, 0.015625f));
        plan.output_b_weight = layer_add(weights, layer_fp8(128, 128, seed + 11, 0.0625f));
        operators.bind_weight_count(weights.size());
        const std::array<TensorHandle, 5> projections = {plan.query_a_weight, plan.query_b_weight, plan.key_value_weight, plan.output_a_weight, plan.output_b_weight};
        for (const TensorHandle handle : projections)
        {
            const uint32_t groups = handle == plan.output_a_weight ? plan.output_group_count : 1;
            operators.at_weight_mutable(handle).float8 = Float8Linear_vulkan::create(weights.at(handle), nullptr, groups, get_default_gpu_index(), runtime, layer_flags);
            layer_check(static_cast<bool>(operators.at_weight(handle).float8), "create layer FP8 projection");
        }
        const auto& query = operators.at_weight(plan.query_a_weight).float8;
        layer_check(query->prepare_rms_norm(weights.at(plan.query_norm_weight), layer_norm_epsilon), "prepare query norm");
        layer_check(query->prepare_input_rms_norm(weights.at(plan.pre_attention_norm_weight), layer_norm_epsilon), "prepare MLA input norm");
        auto attention = LatentAttention_vulkan::create(plan, weights.at(plan.sinks), operators.at_weight(plan.output_a_weight).float8, operators.at_weight(plan.output_b_weight).float8);
        layer_check(attention && attention->prepare(weights, operators, layer_flags) && attention->can_record(), "prepare layer MLA recorder");
        auto attention_hyper = HyperConnection_vulkan::create(attention_function, scale, attention_base, layer_copies, 20, layer_norm_epsilon, layer_hyper_epsilon, get_default_gpu_index(), runtime, layer_flags);
        auto ffn_hyper = HyperConnection_vulkan::create(ffn_function, scale, ffn_base, layer_copies, 20, layer_norm_epsilon, layer_hyper_epsilon, get_default_gpu_index(), runtime, layer_flags);
        auto norm = RmsNorm_vulkan::create(ffn_norm, layer_norm_epsilon, 0.0f, attention->vulkan_context(), attention->option());
        auto router = Router_vulkan::create(router_weight, &router_bias, nullptr, get_default_gpu_index(), runtime, layer_flags);
        layer = LatentLayer_vulkan::create(attention, attention_hyper, ffn_hyper, norm, router);
        layer_check(static_cast<bool>(layer), "create complete layer GPU segment");
    }
};

static ActivationBuffer layer_input(uint32_t rows, uint32_t seed)
{
    ActivationBuffer result(rows, layer_copies * layer_hidden);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < result.columns(); ++column)
            result.row(row)[column] = std::sin(static_cast<float>(column * 7 + row * 17 + seed * 23) * 0.037f)
                                      + 0.3f * std::cos(static_cast<float>(column * 3 + row * 11 + seed * 5) * 0.019f);
    return result;
}

// Simulates the host ExpertStore result crossing back to the GPU residual combine.
static void layer_experts(const ActivationBuffer& normalized, uint32_t seed, ActivationBuffer& result)
{
    result.reset(normalized.rows(), normalized.columns(), false);
    for (size_t row = 0; row < result.rows(); ++row)
        for (uint32_t column = 0; column < result.columns(); ++column)
            result.row(row)[column] = 0.125f * normalized.row(row)[column]
                                      + static_cast<float>(static_cast<int>((column * 7 + seed * 11) % 19) - 9) * 0.01f;
}

static ActivationBuffer layer_experts(const ActivationBuffer& normalized, uint32_t seed)
{
    ActivationBuffer result;
    layer_experts(normalized, seed, result);
    return result;
}

static ExpertDispatchOptions layer_router_options()
{
    ExpertDispatchOptions result;
    result.expert_count = 12;
    result.top_k = 3;
    result.score_function = RouterScoreFunction::Softmax;
    result.normalization = RouterNormalization::SelectedExperts;
    result.routed_scaling_factor = 1.0f;
    return result;
}

static ActivationBuffer layer_cpu(LatentLayerFixture& fixture, const ActivationBuffer& input,
                                  std::span<const uint64_t> positions, const ExpertDispatchOptions& options,
                                  ActivationBuffer& normalized, ExpertDispatchPlan& routes, uint32_t seed,
                                  ExecutionBackend backend = ExecutionBackend::Cpu,
                                  ActivationBuffer* attention_result = nullptr)
{
    HyperConnectionScratch hc_scratch;
    HyperConnectionMix attention_mix;
    HyperConnectionMix ffn_mix;
    ActivationBuffer attention_output;
    ActivationBuffer residual;
    AttentionScratch attention_scratch;
    layer_check(static_cast<bool>(forward_hyper_connection_pre(input, fixture.attention_function, fixture.scale, fixture.attention_base, layer_copies, 20, layer_norm_epsilon, layer_hyper_epsilon, attention_mix, hc_scratch, 0)), "CPU attention HC pre");
    std::vector<LayerCache*> caches(positions.size(), backend == ExecutionBackend::Cpu ? &fixture.cpu_cache : &fixture.staged_cache);
    layer_check(static_cast<bool>(forward_latent_attention_batch(fixture.weights, fixture.operators, fixture.plan, backend, positions, caches, attention_scratch, attention_mix.reduced, attention_output, layer_flags)), "CPU MLA reference");
    if (attention_result)
        *attention_result = attention_output;
    layer_check(static_cast<bool>(forward_hyper_connection_post(attention_output, input, attention_mix, layer_copies, residual)), "CPU attention HC post");
    layer_check(static_cast<bool>(forward_hyper_connection_pre(residual, fixture.ffn_function, fixture.scale, fixture.ffn_base, layer_copies, 20, layer_norm_epsilon, layer_hyper_epsilon, ffn_mix, hc_scratch, 0)), "CPU FFN HC pre");
    normalized = forward_rms_norm(ffn_mix.reduced, fixture.ffn_norm, layer_norm_epsilon, 0.0f);
    const ActivationBuffer logits = forward_linear(fixture.router_weight, fixture.router_bias, normalized, 0);
    layer_check(static_cast<bool>(forward_router(logits.values(), static_cast<uint32_t>(normalized.rows()), options, routes)), "CPU router reference");
    const ActivationBuffer combined = layer_experts(normalized, seed);
    ActivationBuffer expanded;
    layer_check(static_cast<bool>(forward_hyper_connection_post(combined, residual, ffn_mix, layer_copies, expanded)), "CPU FFN HC post");
    return expanded;
}

static void layer_compare_routes(const ExpertDispatchPlan& actual, const ExpertDispatchPlan& expected,
                                 const std::string& label)
{
    layer_check(actual.assignment_count == expected.assignment_count && actual.batches.size() == expected.batches.size(), label + ": assignments");
    for (size_t batch = 0; batch < expected.batches.size(); ++batch)
    {
        const auto& observed = actual.batches[batch];
        const auto& reference = expected.batches[batch];
        layer_check(observed.expert_id == reference.expert_id && observed.routes.size() == reference.routes.size(), label + ": selected experts");
        for (size_t index = 0; index < reference.routes.size(); ++index)
        {
            const auto& a = observed.routes[index];
            const auto& b = reference.routes[index];
            layer_check(a.token_index == b.token_index && a.rank == b.rank && std::abs(a.weight - b.weight) <= 0.0001f, label + ": token/rank/weight");
        }
    }
}

static void layer_compare_device(const std::shared_ptr<const DeviceTensor_vulkan>& input,
                                 const Linear& seed, const ActivationBuffer& expected)
{
    layer_check(input && !input->empty() && input->rows() == expected.rows() && input->columns() == expected.columns(), "resident normalized shape");
    auto graph = CommandGraph_vulkan::create(seed);
    ActivationBuffer downloaded;
    layer_check(graph && graph->download(*input, downloaded) && graph->submit() && graph->wait(), "download resident normalized for verification");
    layer_compare(downloaded, expected, "host normalized equals ExpertStore device input", 0.0f, 0.0f);
}

enum class LayerEndpoint
{
    Materialize,
    AllLogits,
    LastLogits,
    HiddenOnly
};

static void layer_test_two_layers(uint32_t rows, LayerEndpoint endpoint, uint32_t seed, bool resident_combine = false, bool deferred_combine = false)
{
    const std::string label = "rows=" + std::to_string(rows) + " endpoint=" + std::to_string(static_cast<uint32_t>(endpoint))
                              + " seed=" + std::to_string(seed) + " combine=" + (deferred_combine ? "deferred" : (resident_combine ? "resident" : "host"));
    const VulkanRuntimePtr runtime = create_vulkan_runtime();
    LatentLayerFixture first(runtime, seed);
    LatentLayerFixture second(runtime, seed + 17);
    std::unique_ptr<LatentLayerFixture> first_host;
    std::unique_ptr<LatentLayerFixture> second_host;
    LatentLayerWorkspace_vulkan host_workspace;
    if (resident_combine || deferred_combine)
    {
        first_host = std::make_unique<LatentLayerFixture>(runtime, seed);
        second_host = std::make_unique<LatentLayerFixture>(runtime, seed + 17);
    }
    const TensorData head_function = layer_matrix(layer_copies, layer_copies * layer_hidden, seed + 29, 0.0005f);
    const TensorData head_base = layer_base(layer_copies, seed + 31);
    const TensorData head_scale = layer_constant({1}, 0.4f);
    const TensorData head_norm = layer_constant({layer_hidden}, 1.0f);
    const TensorData lm_weight = layer_matrix(128, layer_hidden, seed + 37, 0.001f);
    CompiledOperator lm;
    lm.linear = Linear::create(lm_weight, nullptr, LinearDevice::Vulkan, get_default_gpu_index(), runtime, layer_flags);
    layer_check(static_cast<bool>(lm.linear), "create resident LM projection");
    auto head = LayerHead_vulkan::create(head_function, head_scale, head_base, head_norm, lm, layer_copies, layer_norm_epsilon, layer_hyper_epsilon, get_default_gpu_index(), runtime, layer_flags);
    layer_check(static_cast<bool>(head), "create fused final head");
    const VulkanRuntimePtr foreign_runtime = create_vulkan_runtime();
    CompiledOperator foreign_lm;
    foreign_lm.linear = Linear::create(lm_weight, nullptr, LinearDevice::Vulkan, get_default_gpu_index(), foreign_runtime, layer_flags);
    auto foreign_head = LayerHead_vulkan::create(head_function, head_scale, head_base, head_norm, foreign_lm, layer_copies, layer_norm_epsilon, layer_hyper_epsilon, get_default_gpu_index(), foreign_runtime, layer_flags);
    layer_check(static_cast<bool>(foreign_head), "create foreign runtime head");
    const ActivationBuffer input = layer_input(rows, seed);
    std::vector<uint64_t> positions(rows);
    for (uint32_t row = 0; row < rows; ++row)
        positions[row] = row;
    const ExpertDispatchOptions options = layer_router_options();
    LatentLayerWorkspace_vulkan workspace;
    ActivationBuffer expected_expanded = input;
    ActivationBuffer staged_expanded = input;
    const std::array<LatentLayerFixture*, 2> fixtures = {&first, &second};
    for (size_t index = 0; index < fixtures.size(); ++index)
    {
        auto& fixture = *fixtures[index];
        ActivationBuffer expected_normalized;
        ActivationBuffer cpu_attention;
        ExpertDispatchPlan expected_routes;
        expected_expanded = layer_cpu(fixture, expected_expanded, positions, options, expected_normalized, expected_routes, seed + static_cast<uint32_t>(index), ExecutionBackend::Cpu, &cpu_attention);
        ActivationBuffer staged_normalized;
        ActivationBuffer staged_attention;
        ExpertDispatchPlan staged_routes;
        staged_expanded = layer_cpu(fixture, staged_expanded, positions, options, staged_normalized, staged_routes,
                                    seed + static_cast<uint32_t>(index), ExecutionBackend::Vulkan, &staged_attention);
        ActivationBuffer actual_normalized;
        ExpertDispatchPlan actual_routes;
        std::vector<LayerCache*> caches(rows, &fixture.gpu_cache);
        const VulkanStatistics before = get_vulkan_statistics(runtime);
        const ActivationBuffer empty;
        layer_check(fixture.layer->forward(index == 0 ? input : empty, positions, caches, options, actual_normalized, actual_routes, workspace), label + ": GPU layer forward");
        const VulkanStatistics after = get_vulkan_statistics(runtime);
        layer_check(after.compute_submissions == before.compute_submissions + 1, label + ": one submission per complete layer");
        if (resident_combine || deferred_combine)
        {
            // Compare identical GPU attention/cache work. Its RoPE metadata
            // uploads remain necessary; only the pending FFN activation differs.
            auto& host_fixture = index == 0 ? *first_host : *second_host;
            std::vector<LayerCache*> host_caches(rows, &host_fixture.gpu_cache);
            ActivationBuffer host_normalized;
            ExpertDispatchPlan host_routes;
            const auto host_before = get_vulkan_statistics(runtime);
            layer_check(host_fixture.layer->forward(index == 0 ? input : empty, positions, host_caches, options, host_normalized, host_routes, host_workspace), label + ": matched host-pending control");
            const auto host_after = get_vulkan_statistics(runtime);
            layer_check((after.batch_uploads - before.batch_uploads) + (index == 0 ? 0 : 1)
                            == host_after.batch_uploads - host_before.batch_uploads,
                        label + ": resident Combine handoff saves exactly one activation upload");
            layer_compare(host_normalized, actual_normalized, label + ": resident/control normalized parity", 0.00001f, 0.00001f);
            ActivationBuffer host_combined = layer_experts(host_normalized, seed + static_cast<uint32_t>(index));
            layer_check(LatentLayer_vulkan::defer_combine(host_combined, host_workspace), label + ": control host defer");
        }
        layer_check(after.attention_blocks == before.attention_blocks + rows && after.attention_cpu_fallbacks == before.attention_cpu_fallbacks, label + ": GPU MLA executes every row");
        layer_check(workspace.active() && fixture.gpu_cache.latent_token_count == rows && fixture.gpu_cache.latent_device_state, label + ": persistent GPU activation/cache");
        // CPU grouped output-A uses original activations; the established Hybrid chain quantizes to FP8.
        // Check that difference at the MLA branch, before RMS can amplify it, then check fusion tightly.
        layer_compare(staged_attention, cpu_attention, label + ": CPU MLA branch including existing FP8 projection quantization", 0.05f, 0.08f);
        layer_compare(actual_normalized, staged_normalized, label + ": CPU HC/norm with staged Hybrid MLA parity", 0.003f, 0.003f);
        const ActivationBuffer cpu_router_logits = forward_linear(fixture.router_weight, fixture.router_bias, actual_normalized, 0);
        ExpertDispatchPlan cpu_routes;
        layer_check(static_cast<bool>(forward_router(cpu_router_logits.values(), rows, options, cpu_routes)), "independent CPU router at published normalized input");
        layer_compare_routes(actual_routes, cpu_routes, label + ": CPU router projection/selection parity");
        layer_compare_routes(actual_routes, staged_routes, label + ": staged Hybrid router parity");
        layer_compare_device(workspace.device_input(), *lm.linear, actual_normalized);
        ActivationBuffer combined = layer_experts(actual_normalized, seed + static_cast<uint32_t>(index));
        std::shared_ptr<DeviceTensor_vulkan> device_combined;
        if (resident_combine || deferred_combine)
        {
            auto graph = CommandGraph_vulkan::create(*lm.linear);
            device_combined = std::make_shared<DeviceTensor_vulkan>();
            layer_check(graph && graph->upload(combined, *device_combined) && graph->submit() && graph->wait(), label + ": completed resident Expert fixture");
            std::vector<ExpertRoute> identity;
            for (uint32_t row = 0; row < rows; ++row) identity.push_back({row, 0, 1.0f});
            MoeCombineInput_vulkan source{&combined, device_combined, identity};
            if (deferred_combine)
            {
                layer_check(LatentLayer_vulkan::defer_combine(std::span<const MoeCombineInput_vulkan>(&source, 1), rows, layer_hidden,
                                                              nullptr, {}, workspace),
                            label + ": capture canonical pending Combine");
                // Source slots and route buffers are reused immediately by the
                // executor. The pending workspace must already own their content.
                identity.assign(rows + 1, {rows + 3, 9, -123.0f});
                combined.clear();
                device_combined.reset();
                source = {};
                auto bad_routes = std::vector<ExpertRoute>{{rows, 0, 1.0f}};
                MoeCombineInput_vulkan bad{nullptr, workspace.device_input(), bad_routes};
                layer_check(!LatentLayer_vulkan::defer_combine(std::span<const MoeCombineInput_vulkan>(&bad, 1), rows, layer_hidden,
                                                               nullptr, {}, workspace),
                            label + ": invalid replacement preserves pending Combine");
            }
            else
            {
                std::shared_ptr<DeviceTensor_vulkan> aggregate;
                layer_check(LatentLayer_vulkan::combine_device(std::span<const MoeCombineInput_vulkan>(&source, 1), rows, layer_hidden, nullptr, {}, aggregate, workspace), label + ": prepared GPU aggregate");
                device_combined = std::move(aggregate);
            }
        }
        const uint64_t submissions = get_vulkan_statistics(runtime).compute_submissions;
        if (!deferred_combine)
            layer_check(resident_combine ? LatentLayer_vulkan::defer_combine(device_combined, workspace) : LatentLayer_vulkan::defer_combine(combined, workspace), label + ": defer expert residual combine");
        layer_check(get_vulkan_statistics(runtime).compute_submissions == submissions && workspace.active(), label + ": deferred combine does not synchronize");
    }
    if (resident_combine || deferred_combine)
    {
        // A pending workspace must own everything needed for its post/head,
        // even after the compiled layer operator owners are destroyed.
        first.layer.reset();
        second.layer.reset();
        first_host.reset();
        second_host.reset();
    }
    const auto completed_input = workspace.device_input();
    layer_check(!LatentLayer_vulkan::can_finish(workspace, *foreign_head), label + ": reject foreign runtime head");
    ActivationBuffer rejected_hidden = layer_input(1, 91);
    ActivationBuffer rejected_logits = layer_input(1, 93);
    const ActivationBuffer preserved_hidden = rejected_hidden;
    const ActivationBuffer preserved_logits = rejected_logits;
    const uint64_t submissions = get_vulkan_statistics(runtime).compute_submissions;
    layer_check(!LatentLayer_vulkan::finish(workspace, *foreign_head, true, false, rejected_hidden, rejected_logits), label + ": foreign finish rejects before recording");
    layer_check(workspace.active() && workspace.device_input() == completed_input && get_vulkan_statistics(runtime).compute_submissions == submissions, label + ": foreign rejection preserves workspace");
    layer_compare(rejected_hidden, preserved_hidden, "foreign rejection preserves hidden", 0.0f, 0.0f);
    layer_compare(rejected_logits, preserved_logits, "foreign rejection preserves logits", 0.0f, 0.0f);
    ActivationBuffer hidden;
    ActivationBuffer logits;
    if (endpoint == LayerEndpoint::Materialize)
    {
        layer_check(LatentLayer_vulkan::materialize_hidden(workspace, hidden), label + ": materialize deferred post");
        layer_compare(hidden, staged_expanded, label + ": staged Hybrid expanded residual parity", 0.003f, 0.003f);
        layer_compare(hidden, expected_expanded, label + ": CPU expanded residual parity including existing quantization", 0.05f, 0.08f);
    }
    else
    {
        HyperConnectionScratch scratch;
        ActivationBuffer headed;
        layer_check(static_cast<bool>(forward_hyper_connection_head(staged_expanded, head_function, head_scale, head_base, layer_copies, layer_norm_epsilon, layer_hyper_epsilon, headed, scratch, 0)), "CPU final HC head");
        const ActivationBuffer expected_hidden = forward_rms_norm(headed, head_norm, layer_norm_epsilon, 0.0f);
        const bool produce_logits = endpoint != LayerEndpoint::HiddenOnly;
        const bool last_row_only = endpoint == LayerEndpoint::LastLogits;
        layer_check(LatentLayer_vulkan::can_finish(workspace, *head), label + ": same runtime can finish");
        layer_check(LatentLayer_vulkan::finish(workspace, *head, produce_logits, last_row_only, hidden, logits), label + ": finish deferred post/head/norm/LM");
        layer_compare(hidden, expected_hidden, label + ": staged Hybrid final normalized hidden", 0.003f, 0.003f);
        if (produce_logits)
        {
            ActivationBuffer lm_input = expected_hidden;
            if (last_row_only)
            {
                lm_input.reset(1, layer_hidden, true);
                std::copy_n(expected_hidden.row(rows - 1), layer_hidden, lm_input.row(0));
            }
            const ActivationBuffer expected_logits = forward_linear(lm_weight, lm_input, 0);
            layer_compare(logits, expected_logits, label + ": CPU logits including last-row view", 0.003f, 0.02f);
            ActivationBuffer published_input = hidden;
            if (last_row_only)
            {
                published_input.reset(1, layer_hidden, true);
                std::copy_n(hidden.row(rows - 1), layer_hidden, published_input.row(0));
            }
            const ActivationBuffer projected_published = forward_linear(lm_weight, published_input, 0);
            layer_compare(logits, projected_published, label + ": projection reads the published normalized rows", 0.0003f, 0.0003f);
        }
        else
            layer_check(logits.rows() == 0 && logits.columns() == 0, label + ": hidden-only endpoint omits logits");
    }
    layer_check(get_vulkan_statistics(runtime).compute_submissions == submissions + 1, label + ": endpoint completes in one submission");
    layer_check(!workspace.active() && !workspace.device_input(), label + ": completed workspace releases pending state");
    layer_check(!LatentLayer_vulkan::can_finish(workspace, *head), label + ": inactive workspace cannot finish");
}

struct LayerPublishedStorage
{
    const std::byte* normalized = nullptr;
    const std::byte* hidden = nullptr;
    const std::byte* logits = nullptr;
    const ExpertBatch* batches = nullptr;
    std::array<const ExpertRoute*, 3> routes = {};
};

static LayerPublishedStorage layer_storage(const ActivationBuffer& normalized,
                                           const ActivationBuffer& hidden,
                                           const ActivationBuffer& logits,
                                           const ExpertDispatchPlan& routes)
{
    layer_check(routes.batches.size() == 3, "stable fixture selects three Expert batches");
    LayerPublishedStorage result;
    result.normalized = normalized.bytes().data();
    result.hidden = hidden.bytes().data();
    result.logits = logits.bytes().data();
    result.batches = routes.batches.data();
    for (size_t index = 0; index < result.routes.size(); ++index)
        result.routes[index] = routes.batches[index].routes.data();
    return result;
}

static uint64_t layer_published_bytes(const ActivationBuffer& normalized,
                                      const ActivationBuffer& hidden,
                                      const ActivationBuffer& logits,
                                      const ExpertDispatchPlan& routes)
{
    uint64_t bytes = normalized.allocated_bytes() + hidden.allocated_bytes() + logits.allocated_bytes();
    bytes += static_cast<uint64_t>(routes.batches.capacity()) * sizeof(ExpertBatch)
             + static_cast<uint64_t>(routes.route_scratch.capacity()) * sizeof(std::vector<ExpertRoute>)
             + static_cast<uint64_t>(routes.scores.capacity()) * sizeof(float)
             + static_cast<uint64_t>(routes.selected.capacity()) * sizeof(RouteCandidate);
    for (const auto& batch : routes.batches)
        bytes += static_cast<uint64_t>(batch.routes.capacity()) * sizeof(ExpertRoute);
    for (const auto& scratch : routes.route_scratch)
        bytes += static_cast<uint64_t>(scratch.capacity()) * sizeof(ExpertRoute);
    return bytes;
}

static void layer_test_workspace_reuse()
{
    const uint32_t seed = 41;
    const VulkanRuntimePtr runtime = create_vulkan_runtime();
    LatentLayerFixture fixture(runtime, seed);
    const TensorData head_function = layer_matrix(layer_copies, layer_copies * layer_hidden, seed + 29, 0.0005f);
    const TensorData head_base = layer_base(layer_copies, seed + 31);
    const TensorData head_scale = layer_constant({1}, 0.4f);
    const TensorData head_norm = layer_constant({layer_hidden}, 1.0f);
    const TensorData lm_weight = layer_matrix(128, layer_hidden, seed + 37, 0.001f);
    CompiledOperator lm;
    lm.linear = Linear::create(lm_weight, nullptr, LinearDevice::Vulkan, get_default_gpu_index(), runtime, layer_flags);
    layer_check(static_cast<bool>(lm.linear), "reuse fixture LM projection");
    const auto head = LayerHead_vulkan::create(head_function, head_scale, head_base, head_norm, lm, layer_copies,
                                               layer_norm_epsilon, layer_hyper_epsilon, get_default_gpu_index(), runtime, layer_flags);
    layer_check(static_cast<bool>(head), "reuse fixture final head");
    const ExpertDispatchOptions options = layer_router_options();
    LatentLayerWorkspace_vulkan workspace;
    ActivationBuffer normalized;
    ActivationBuffer hidden;
    ActivationBuffer logits;
    ActivationBuffer combined;
    ExpertDispatchPlan routes;
    std::array<LayerPublishedStorage, 4> warmed;
    uint64_t retained_bytes = 0;
    // Each route plan alternates its published batches with route scratch,
    // so two pending/published plans need four warm route-storage sets.
    // Then shrink and regrow below the established high-water mark.
    const std::array<uint32_t, 12> row_counts = {4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4};
    for (size_t iteration = 0; iteration < row_counts.size(); ++iteration)
    {
        const uint32_t rows = row_counts[iteration];
        const std::string label = "workspace reuse iteration=" + std::to_string(iteration);
        workspace.reset();
        fixture.cpu_cache = {};
        fixture.staged_cache = {};
        fixture.gpu_cache = {};
        const ActivationBuffer input = layer_input(rows, seed);
        std::vector<uint64_t> positions(rows);
        for (uint32_t row = 0; row < rows; ++row)
            positions[row] = row;
        std::vector<LayerCache*> caches(rows, &fixture.gpu_cache);
        ActivationBuffer expected_normalized;
        ExpertDispatchPlan expected_routes;
        const ActivationBuffer expanded = layer_cpu(fixture, input, positions, options, expected_normalized,
                                                    expected_routes, seed, ExecutionBackend::Vulkan);
        layer_check(fixture.layer->forward(input, positions, caches, options, normalized, routes, workspace), label + ": forward");
        layer_compare(normalized, expected_normalized, label + ": normalized parity");
        layer_compare_routes(routes, expected_routes, label + ": router parity");
        layer_experts(normalized, seed, combined);
        layer_check(LatentLayer_vulkan::defer_combine(combined, workspace), label + ": pending Expert output");
        layer_check(LatentLayer_vulkan::finish(workspace, *head, true, false, hidden, logits), label + ": finish");
        HyperConnectionScratch scratch;
        ActivationBuffer headed;
        layer_check(static_cast<bool>(forward_hyper_connection_head(expanded, head_function, head_scale, head_base,
                                                                    layer_copies, layer_norm_epsilon, layer_hyper_epsilon,
                                                                    headed, scratch, 0)),
                    label + ": staged head oracle");
        const ActivationBuffer expected_hidden = forward_rms_norm(headed, head_norm, layer_norm_epsilon, 0.0f);
        const ActivationBuffer expected_logits = forward_linear(lm_weight, expected_hidden, 0);
        layer_compare(hidden, expected_hidden, label + ": hidden parity");
        layer_compare(logits, expected_logits, label + ": logits parity", 0.003f, 0.02f);
        layer_check(!workspace.active() && !workspace.device_input(), label + ": ended GPU segment");
        const auto storage = layer_storage(normalized, hidden, logits, routes);
        // Capacity moves between the pending and published slots at commit.
        // Include both owners and the reusable deferred-Combine source.
        const uint64_t owned_bytes = workspace.allocated_bytes()
                                     + layer_published_bytes(normalized, hidden, logits, routes)
                                     + combined.allocated_bytes();
        if (iteration >= 4 && iteration <= 7)
        {
            warmed[iteration - 4] = storage;
            if (iteration == 4)
                retained_bytes = owned_bytes;
            else
                layer_check(owned_bytes == retained_bytes, "pending and published sets reach stable total capacity");
        }
        else if (iteration > 7)
        {
            const auto& expected = warmed[iteration % warmed.size()];
            layer_check(storage.normalized == expected.normalized && storage.hidden == expected.hidden
                            && storage.logits == expected.logits && storage.batches == expected.batches
                            && storage.routes == expected.routes,
                        label + ": warm publication storage survives shrink/regrow");
            layer_check(owned_bytes == retained_bytes, label + ": total retained capacity survives shrink/regrow");
        }
    }
    const uint64_t pending_bytes = workspace.allocated_bytes();
    layer_check(pending_bytes > 0 && retained_bytes >= pending_bytes, "workspace exposes retained Host scratch accounting");
    workspace.reset();
    layer_check(workspace.allocated_bytes() == pending_bytes, "segment reset retains Host capacities");

    // NaN reaches the GPU router validation result after the command finishes.
    // Reused pending storage may change, but none of the published outputs do.
    fixture.gpu_cache = {};
    ActivationBuffer invalid_input = layer_input(1, seed);
    invalid_input.row(0)[0] = std::numeric_limits<float>::quiet_NaN();
    const std::array<uint64_t, 1> positions = {0};
    const std::array<LayerCache*, 1> caches = {&fixture.gpu_cache};
    const ActivationBuffer preserved_normalized = normalized;
    const ActivationBuffer preserved_hidden = hidden;
    const ActivationBuffer preserved_logits = logits;
    const ExpertDispatchPlan preserved_routes = routes;
    const auto published_storage = layer_storage(normalized, hidden, logits, routes);
    const uint64_t submissions = get_vulkan_statistics(runtime).compute_submissions;
    layer_check(!fixture.layer->forward(invalid_input, positions, caches, options, normalized, routes, workspace),
                "nonfinite router rejects completed pending publication");
    layer_check(get_vulkan_statistics(runtime).compute_submissions == submissions + 1,
                "failed publication test reaches completed GPU command");
    layer_compare(normalized, preserved_normalized, "failed publication preserves normalized", 0.0f, 0.0f);
    layer_compare(hidden, preserved_hidden, "failed publication preserves hidden", 0.0f, 0.0f);
    layer_compare(logits, preserved_logits, "failed publication preserves logits", 0.0f, 0.0f);
    layer_compare_routes(routes, preserved_routes, "failed publication preserves routes");
    const auto rejected_storage = layer_storage(normalized, hidden, logits, routes);
    layer_check(rejected_storage.normalized == published_storage.normalized
                    && rejected_storage.hidden == published_storage.hidden
                    && rejected_storage.logits == published_storage.logits
                    && rejected_storage.batches == published_storage.batches
                    && rejected_storage.routes == published_storage.routes,
                "failed publication preserves output backing storage");
    layer_check(!workspace.active() && !workspace.device_input() && fixture.gpu_cache.latent_token_count == 0,
                "failed publication leaves cache/workspace uncommitted");

    // Aborting an active segment releases its owners and keeps completed
    // tensors held by consumers valid until those consumers release them.
    fixture.gpu_cache = {};
    const ActivationBuffer input = layer_input(1, seed);
    layer_check(fixture.layer->forward(input, positions, caches, options, normalized, routes, workspace), "restart after failed publication");
    const auto completed_device = workspace.device_input();
    const uint64_t active_retained_bytes = workspace.allocated_bytes();
    workspace.reset();
    layer_check(!workspace.active() && !workspace.device_input()
                    && workspace.allocated_bytes() == active_retained_bytes,
                "active segment reset retains Host scratch and clears pending GPU state");
    layer_compare_device(completed_device, *lm.linear, normalized);
    workspace.release();
    layer_check(workspace.allocated_bytes() == 0 && !workspace.active() && !workspace.device_input(),
                "explicit release drops all retained Host scratch");
    layer_compare_device(completed_device, *lm.linear, normalized);
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
        ncnn::moe::layer_test_two_layers(1, ncnn::moe::LayerEndpoint::Materialize, 1);
        ncnn::moe::layer_test_two_layers(1, ncnn::moe::LayerEndpoint::AllLogits, 2);
        ncnn::moe::layer_test_two_layers(3, ncnn::moe::LayerEndpoint::LastLogits, 3);
        ncnn::moe::layer_test_two_layers(3, ncnn::moe::LayerEndpoint::HiddenOnly, 4);
        ncnn::moe::layer_test_two_layers(4, ncnn::moe::LayerEndpoint::AllLogits, 5);
        ncnn::moe::layer_test_two_layers(3, ncnn::moe::LayerEndpoint::LastLogits, 7, true);
        ncnn::moe::layer_test_two_layers(3, ncnn::moe::LayerEndpoint::Materialize, 13, true);
        ncnn::moe::layer_test_two_layers(1, ncnn::moe::LayerEndpoint::HiddenOnly, 17, true);
        ncnn::moe::layer_test_two_layers(3, ncnn::moe::LayerEndpoint::LastLogits, 19, false, true);
        ncnn::moe::layer_test_two_layers(3, ncnn::moe::LayerEndpoint::Materialize, 13, false, true);
        ncnn::moe::layer_test_two_layers(1, ncnn::moe::LayerEndpoint::HiddenOnly, 29, false, true);
        ncnn::moe::layer_test_workspace_reuse();
    }
    catch (const std::exception& error)
    {
        std::cerr << "Vulkan latent layer test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Vulkan latent layer fusion tests passed\n";
    return 0;
#else
    std::cout << "SKIP: Vulkan support disabled\n";
    return 77;
#endif
}
