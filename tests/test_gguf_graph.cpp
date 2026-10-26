#include "ncnn/moe/runtime.h"
#include "graph/compiledmodel.h"
#include "engine/executor.h"
#include "engine/sessionstate.h"
#include "backends/ncnn/vulkan.h"
#include "kernels/qnk.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void check(bool condition, const std::string& context)
{
    if (!condition)
        throw std::runtime_error(context);
}

static std::vector<float> logits_values(const SessionState& state)
{
    std::vector<float> result;
    result.reserve(state.logits.rows() * state.logits.columns());
    for (size_t row = 0; row < state.logits.rows(); ++row)
        result.insert(result.end(), state.logits.row(row), state.logits.row(row) + state.logits.columns());
    return result;
}

static void compare_logits(const std::vector<float>& expected, const std::vector<float>& actual,
                           const std::string& context, bool hybrid)
{
    check(!expected.empty() && expected.size() == actual.size(), context + ": logits size mismatch");
    float scale = 0.0f;
    for (float value : expected)
    {
        check(std::isfinite(value), context + ": reference logits are non-finite");
        scale = std::max(scale, std::abs(value));
    }
    const auto [minimum, maximum] = std::minmax_element(expected.begin(), expected.end());
    check(scale > 1e-7f && *maximum - *minimum > 1e-7f, context + ": fixture logits are constant or zero");
    // Quantized reference matrices may be rounded to BF16. Hybrid also keeps
    // FP32 KV where CPU uses BF16; use the established vector-scale bound.
    const float tolerance = hybrid ? 1e-3f + 5e-3f * scale : 2e-6f + 2e-6f * scale;
    float maximum_error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const float error = std::abs(actual[index] - expected[index]);
        if (!(error <= tolerance))
        {
            std::ostringstream message;
            message << context << ": index=" << index << " actual=" << actual[index]
                    << " expected=" << expected[index] << " error=" << error << " tolerance=" << tolerance;
            throw std::runtime_error(message.str());
        }
        maximum_error = std::max(maximum_error, error);
    }
    std::cout << context << ": values=" << actual.size() << " maximum_error=" << maximum_error
              << " tolerance=" << tolerance << '\n';
}

static void compare_graphs(const CompiledModel& reference, const CompiledModel& candidate, bool hybrid)
{
    std::array<SessionState, 2> expected_states;
    std::array<SessionState, 2> actual_states;
    std::array<SessionStatistics, 2> expected_statistics;
    std::array<SessionStatistics, 2> actual_statistics;
    const std::array<std::vector<int32_t>, 2> prompts = {{{1, 2, 3}, {3, 2}}};
    const std::string family = reference.descriptor.model_type;
    std::vector<size_t> standard_gqa_layers;
    for (size_t layer_index = 0; layer_index < candidate.descriptor.layers.size(); ++layer_index)
    {
        const AttentionDescriptor& attention = candidate.descriptor.layers[layer_index].attention;
        if (attention.kind == AttentionKind::Standard && attention.head_count > attention.kv_head_count
            && !has_flag(attention.flags, AttentionDescriptorQsa))
            standard_gqa_layers.push_back(layer_index);
    }
    if (family == "qwen3_5" || family == "qwen3_5_moe")
    {
        check(!standard_gqa_layers.empty(), family + ": fixture has no standard GQA layer");
        for (size_t layer_index : standard_gqa_layers)
        {
            const AttentionDescriptor& attention = candidate.descriptor.layers[layer_index].attention;
            check(has_flag(attention.flags, AttentionDescriptorQueryKeyNorm),
                  family + ": GQA descriptor lost Q/K normalization");
            check(has_flag(attention.flags, AttentionDescriptorOutputGate),
                  family + ": GQA descriptor lost the output gate");
            check(attention.qk_rope_head_dimension != 0
                      && attention.qk_rope_head_dimension * 2 == attention.head_dimension,
                  family + ": fixture does not exercise half-dimension partial RoPE");
        }
    }
    if (family == "gpt_oss")
    {
        check(!standard_gqa_layers.empty(), "GPT-OSS fixture has no standard GQA layer");
        bool has_sliding_window_layer = false;
        for (size_t layer_index : standard_gqa_layers)
        {
            const AttentionDescriptor& attention = candidate.descriptor.layers[layer_index].attention;
            check(has_flag(attention.flags, AttentionDescriptorSinks), "GPT-OSS GQA lost attention sinks");
            const bool expected_sliding_layer = layer_index % 2 == 0;
            check((attention.sliding_window != 0) == expected_sliding_layer,
                  "GPT-OSS GQA layer has the wrong full/sliding-window mode");
            has_sliding_window_layer = has_sliding_window_layer || attention.sliding_window != 0;
            check(!has_flag(attention.flags, AttentionDescriptorQueryKeyNorm)
                      && !has_flag(attention.flags, AttentionDescriptorOutputGate),
                  "GPT-OSS GQA unexpectedly enabled Q/K norm or output gate");
        }
        check(has_sliding_window_layer, "GPT-OSS GQA fixture has no sliding-window layer");
    }
    for (size_t session = 0; session < prompts.size(); ++session)
    {
        const auto expected = forward_model(reference, prompts[session], expected_statistics[session], expected_states[session], 0, LogitsOutput::All);
        const auto actual = forward_model(candidate, prompts[session], actual_statistics[session], actual_states[session], 0, LogitsOutput::All);
        check(static_cast<bool>(expected), expected ? "" : expected.error().message);
        check(static_cast<bool>(actual), actual ? "" : actual.error().message);
        compare_logits(logits_values(expected_states[session]), logits_values(actual_states[session]),
                       family + " prefill session=" + std::to_string(session), hybrid);
    }
    for (size_t session = 0; session < prompts.size(); ++session)
    {
        const std::array<int32_t, 1> token = {static_cast<int32_t>(session + 4)};
        const auto expected = forward_model(reference, token, expected_statistics[session], expected_states[session], prompts[session].size(), LogitsOutput::Last);
        const auto actual = forward_model(candidate, token, actual_statistics[session], actual_states[session], prompts[session].size(), LogitsOutput::Last);
        check(static_cast<bool>(expected), expected ? "" : expected.error().message);
        check(static_cast<bool>(actual), actual ? "" : actual.error().message);
        compare_logits(logits_values(expected_states[session]), logits_values(actual_states[session]),
                       family + " decode session=" + std::to_string(session), hybrid);
    }
    std::array<std::vector<float>, 2> expected_logits;
    std::array<std::vector<float>, 2> actual_logits;
    std::array<DecodeBatchEntry, 2> entries;
    for (size_t session = 0; session < prompts.size(); ++session)
    {
        const std::array<int32_t, 1> token = {static_cast<int32_t>(session + 5)};
        const uint64_t position = prompts[session].size() + 1;
        auto expected = forward_model(reference, token, expected_statistics[session], expected_states[session], position, LogitsOutput::Last);
        check(static_cast<bool>(expected), expected ? "" : expected.error().message);
        expected_logits[session] = logits_values(expected_states[session]);
        entries[session] = {token.front(), &actual_statistics[session], &actual_states[session], position, &actual_logits[session]};
    }
    BatchWorkspace workspace;
    auto batch = forward_decode_batch(candidate, entries, workspace);
    check(static_cast<bool>(batch), batch ? "" : batch.error().message);
    for (size_t session = 0; session < prompts.size(); ++session)
        compare_logits(expected_logits[session], actual_logits[session], family + " two-session batch session=" + std::to_string(session), hybrid);
    for (const auto& statistics : actual_statistics)
        check(statistics.expert_cpu_routes + statistics.expert_gpu_routes == statistics.expert_assignments,
              family + ": per-session CPU/GPU routes do not cover all Expert assignments after batched decode");
    if (candidate.qwen_device_chain)
    {
        for (const auto& statistics : actual_statistics)
        {
            check(statistics.vulkan_qwen_layers == 3 * candidate.graph.layer_plans.size(), "Qwen prefill/decode did not execute every device layer");
            check(statistics.expert_assignments > 0 && statistics.expert_gpu_routes == statistics.expert_assignments, "Qwen GPU routes do not cover every assignment");
            check(statistics.expert_cpu_routes == 0 && statistics.expert_cpu_fallbacks == 0
                      && statistics.model_compute_cpu_fallbacks == 0,
                  "Qwen device chain performed CPU model execution");
            check(statistics.vulkan_ple_blocks > 0, "Qwen device PLE was not executed");
        }
    }
    const auto counters = get_vulkan_statistics(candidate.vulkan_runtime);
    if (hybrid)
    {
        bool native_blocks = false;
        for (TensorHandle handle = 0; handle < candidate.weights.size(); ++handle)
        {
            native_blocks = native_blocks || is_qnk_dtype(candidate.weights.at(handle).dtype);
        }
        if (native_blocks)
        {
            check(counters.dispatches > 0, family + ": native GGUF weights executed no Vulkan projections");
            if (!standard_gqa_layers.empty())
            {
                check(counters.attention_blocks > 0,
                      family + ": native GQA executed no Vulkan Attention blocks");
                check(counters.attention_cpu_fallbacks == 0,
                      family + ": native GQA Attention fell back to CPU");
                for (size_t layer_index : standard_gqa_layers)
                {
                    const AttentionBlockPlan& attention = candidate.graph.layer_plans[layer_index].attention;
                    bool has_native_projection = false;
                    for (TensorHandle handle : {attention.query_weight, attention.key_weight,
                                                attention.value_weight, attention.output_weight})
                    {
                        check(handle != invalid_tensor_handle,
                              family + ": GQA projection is missing from the compiled graph");
                        const DType dtype = candidate.weights.at(handle).dtype;
                        const bool native = dtype == DType::Int4 || is_qnk_dtype(dtype);
                        const bool dense = dtype == DType::BFloat16 || dtype == DType::Float32;
                        check(native || dense,
                              family + ": GQA projection has an unsupported GGUF dtype");
                        has_native_projection = has_native_projection || native;
                    }
                    check(has_native_projection,
                          family + ": GQA layer has no native quantized projection");
                }
                for (const auto& statistics : actual_statistics)
                {
                    check(statistics.vulkan_attention_blocks > 0,
                          family + ": session executed no Vulkan GQA Attention block");
                    check(statistics.vulkan_attention_device_rope_fusions > 0,
                          family + ": GQA executed no Vulkan device RoPE");
                    check(statistics.vulkan_kv_ring_appends > 0,
                          family + ": GQA decode did not append to the Vulkan KV ring");
                    check(statistics.vulkan_attention_cpu_fallbacks == 0
                              && statistics.model_compute_cpu_fallbacks == 0,
                          family + ": native GQA had a CPU model-compute fallback");
                }
            }
            if (family == "deepseek_v4")
                check(counters.attention_blocks > 0, "native DeepSeek GGUF weights did not execute MLA on Vulkan");
            if (std::any_of(candidate.graph.layer_plans.begin(), candidate.graph.layer_plans.end(), [](const CompiledLayerPlan& layer) {
                    return layer.attention.kind == AttentionKind::GatedDeltaNet;
                }))
                check(counters.gated_delta_fusions > 0, family + ": native GGUF weights did not execute GDN on Vulkan");
            check(counters.attention_cpu_fallbacks == 0, family + ": native GGUF attention fell back to CPU");
            if (!standard_gqa_layers.empty())
            {
                for (const auto& statistics : actual_statistics)
                {
                    if (candidate.descriptor.expert_count != 0)
                        check(statistics.expert_assignments > 0 && statistics.expert_gpu_routes == statistics.expert_assignments,
                              family + ": GPU-priority routes do not cover every native Expert assignment");
                    check(statistics.expert_cpu_routes == 0 && statistics.expert_cpu_fallbacks == 0
                              && statistics.model_compute_cpu_fallbacks == 0,
                          family + ": native Expert execution fell back to CPU");
                }
            }
        }
        const bool gpu_planned = std::any_of(candidate.graph.nodes.begin(), candidate.graph.nodes.end(), [](const ExecutionNode& node) {
            return node.backend == ExecutionBackend::Vulkan;
        });
        if (gpu_planned)
            check(counters.dispatches > 0, family + ": planned Vulkan nodes executed no dispatches");
        else
            std::cout << "SKIP " << family << " Vulkan execution: plan admitted no Vulkan nodes\n";
        const bool gdn_planned = std::any_of(candidate.graph.nodes.begin(), candidate.graph.nodes.end(), [&candidate](const ExecutionNode& node) {
            return node.type == ExecutionNodeType::Attention && node.backend == ExecutionBackend::Vulkan
                   && candidate.graph.layer_plans[node.layer_plan_index].attention.kind == AttentionKind::GatedDeltaNet;
        });
        if (gdn_planned)
            check(counters.gated_delta_fusions > 0, family + ": Hybrid executed no GDN Vulkan operations");
        else if (std::any_of(candidate.graph.layer_plans.begin(), candidate.graph.layer_plans.end(), [](const CompiledLayerPlan& layer) {
                     return layer.attention.kind == AttentionKind::GatedDeltaNet;
                 }))
            std::cout << "SKIP " << family << " Vulkan GDN execution: the plan selected CPU recurrence\n";
    }
    else if (!standard_gqa_layers.empty())
    {
        check(counters.attention_blocks == 0 && counters.attention_cpu_fallbacks == 0,
              family + ": explicit CPU GQA profile executed or fell back from Vulkan Attention");
        for (const auto& statistics : actual_statistics)
            check(statistics.vulkan_attention_blocks == 0 && statistics.vulkan_attention_cpu_fallbacks == 0,
                  family + ": explicit CPU GQA profile reported Vulkan Attention work or fallback");
    }
    const auto experts = candidate.expert_backend ? candidate.expert_backend->statistics() : ExpertBackendStatistics{};
    const auto host_cache = candidate.expert_cache ? candidate.expert_cache->statistics() : ExpertCacheStatistics{};
    std::cout << family << " graph comparison passed; Vulkan dispatches=" << counters.dispatches
              << " int4=" << counters.int4_dispatches
              << " attention=" << counters.attention_blocks << " gated_delta=" << counters.gated_delta_fusions
              << " expert_gpu=" << experts.executions << " expert_gpu_failures=" << experts.execution_failures
              << " host_cache_reads=" << host_cache.bytes_read << " host_cache_misses=" << host_cache.misses << '\n';
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    try
    {
        if (argc < 3 || argc > 5)
            throw std::runtime_error("usage: test_gguf_graph HF_DIRECTORY GGUF_SOURCE [--hybrid] [--on-demand]");
        bool hybrid = false;
        bool on_demand = false;
        for (int index = 3; index < argc; ++index)
        {
            const std::string flag = argv[index];
            if (flag == "--hybrid" && !hybrid)
                hybrid = true;
            else if (flag == "--on-demand" && !on_demand)
                on_demand = true;
            else
                throw std::runtime_error("unknown or duplicate graph comparison option: " + flag);
        }
        ncnn::moe::Runtime runtime;
        const auto gpu = std::find_if(runtime.info().gpu_infos.begin(), runtime.info().gpu_infos.end(), [](const ncnn::moe::GpuInfo& gpu) {
            return gpu.type != ncnn::moe::VulkanDeviceType::Cpu && gpu.compute_queue_count != 0;
        });
        if (hybrid && gpu == runtime.info().gpu_infos.end())
        {
            std::cout << "SKIP Hybrid GGUF graph comparison: no Vulkan GPU\n";
            return 77;
        }
        ncnn::moe::Option cpu;
        cpu.hybrid_mode = ncnn::moe::HybridMode::CpuOnly;
        cpu.expert_memory_mode = ncnn::moe::ExpertMemoryMode::Eager;
        // Keep graph storage parity on FP32 activations consistently across ISA.
        cpu.optimization_flags &= ~(ncnn::moe::OptimizationCpuMxfp4Q8
                                    | ncnn::moe::OptimizationCpuBfloat16Batched);
        cpu.num_concurrent_sessions = 2;
        auto expected = runtime.load_model(argv[1], cpu);
        ncnn::moe::check(static_cast<bool>(expected), expected ? "" : expected.error().message);
        auto option = cpu;
        option.hybrid_mode = hybrid ? ncnn::moe::HybridMode::HybridExperts : ncnn::moe::HybridMode::CpuOnly;
        if (hybrid)
        {
            option.vulkan_device_index = gpu->device_index;
            option.vulkan_device_indices = {gpu->device_index};
        }
        if (on_demand)
        {
            option.expert_memory_mode = ncnn::moe::ExpertMemoryMode::OnDemand;
            option.expert_cache_size = UINT64_C(2) * 1024 * 1024;
        }
        auto actual = runtime.load_model(argv[2], option);
        ncnn::moe::check(static_cast<bool>(actual), actual ? "" : actual.error().message);
        const auto& reference = ncnn::moe::model_compiled(*expected.value());
        const auto& candidate = ncnn::moe::model_compiled(*actual.value());
        bool qnk_experts = false;
        if (on_demand)
        {
            ncnn::moe::check(candidate.memory_plan.selected_mode == ncnn::moe::ExpertMemoryMode::OnDemand
                                 && candidate.expert_cache,
                             "on-demand graph comparison did not construct the host Expert cache");
            ncnn::moe::check(candidate.expert_cache->statistics().bytes_read == 0,
                             "on-demand model read Expert payload before execution");
            for (const auto& layer : candidate.graph.layer_plans)
            {
                for (const auto& expert : layer.moe.experts)
                {
                    for (auto handle : {expert.gate_up_weight, expert.down_weight})
                    {
                        ncnn::moe::check(handle != ncnn::moe::invalid_tensor_handle,
                                         "on-demand Expert is missing its cached projection");
                        const auto& tensor = candidate.weights.at(handle);
                        if (!ncnn::moe::is_qnk_dtype(tensor.dtype))
                            continue;
                        qnk_experts = true;
                        ncnn::moe::check(tensor.mxfp4_file_storage && !tensor.mapped_data
                                             && tensor.quantized_data.empty() && tensor.qnk_values().empty(),
                                         "on-demand Qn_K projection retained resident payload");
                    }
                }
            }
        }
        ncnn::moe::check(reference.descriptor.model_type == candidate.descriptor.model_type
                             && reference.descriptor.layers.size() == candidate.descriptor.layers.size(),
                         "HF/GGUF graph descriptor mismatch");
        ncnn::moe::compare_graphs(reference, candidate, hybrid);
        if (on_demand)
        {
            const auto cache = candidate.expert_cache->statistics();
            ncnn::moe::check(cache.resident_size <= candidate.expert_cache->capacity(),
                             "on-demand graph exceeded the Expert cache payload budget");
            ncnn::moe::check(cache.bytes_read > 0,
                             qnk_experts ? "on-demand Qn_K comparison did not read any Expert ranges"
                                         : "on-demand graph comparison did not read any Expert ranges");
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
