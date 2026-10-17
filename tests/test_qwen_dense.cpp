#include "models/modeladapter_qwen3_5.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"
#include "graph/compiledmodel.h"
#include "graph/compiler.h"
#include "engine/executor.h"
#include "engine/expert.h"
#include "engine/expertbackend.h"
#include "engine/sessionstate.h"
#include "backends/ncnn/expertbackend_vulkan.h"
#include "backends/ncnn/modelpipeline.h"
#include "backends/ncnn/vulkan.h"
#include "ncnn/moe/runtime.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

namespace ncnn {
namespace moe {

static void check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void check_logits(const std::vector<float>& actual, const std::vector<float>& expected,
                         const std::string& context, bool vulkan = true)
{
    check(actual.size() == expected.size(), context + " logits size mismatch");
    float maximum = 0.0f;
    for (float value : expected)
        maximum = std::max(maximum, std::abs(value));
    // CPU stores KV in BF16 while Vulkan retains FP32 values. Bound error by
    // the logit vector scale to allow rounding without magnifying cancellation.
    const float tolerance = vulkan ? 1e-3f + 5e-3f * maximum : 1e-6f + 1e-5f * maximum;
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const float error = std::abs(actual[index] - expected[index]);
        if (!(error < tolerance))
        {
            std::ostringstream message;
            message << context << " index=" << index << " actual=" << actual[index]
                    << " expected=" << expected[index] << " absolute_error=" << error
                    << " tolerance=" << tolerance;
            throw std::runtime_error(message.str());
        }
    }
}

static std::string dense_config(bool tied)
{
    return std::string(R"({"model_type":"qwen3_5","text_config":{
        "model_type":"qwen3_5_text","vocab_size":8,"hidden_size":4,
        "intermediate_size":8,"num_hidden_layers":2,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,
        "linear_num_key_heads":1,"linear_num_value_heads":1,
        "linear_key_head_dim":2,"linear_value_head_dim":2,"linear_conv_kernel_dim":2,
        "max_position_embeddings":32,"rms_norm_eps":0.000001,
        "rope_parameters":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
        "layer_types":["linear_attention","full_attention"],"hidden_act":"silu",
        "dtype":"bfloat16","mamba_ssm_dtype":"float32",
        "attention_bias":false,"attn_output_gate":true,"tie_word_embeddings":)")
           + (tied ? "true" : "false") + "}}";
}

static void check_budget_fallback(const MoeModelDescriptor& descriptor, const WeightMapping& mapping,
                                  const std::vector<float>& prefill_logits, const std::vector<float>& decode_logits,
                                  uint32_t device_index)
{
    for (uint64_t budget : {UINT64_C(0), UINT64_C(1)})
    {
        CompilerOption opt;
        opt.flags = BackendCpuExecution | BackendVulkanDense | BackendVulkanAttention | BackendReleaseVulkanDenseHostStorage;
        opt.device_index = device_index;
        opt.gpu_heap_budget = budget;
        opt.vulkan_runtime = create_vulkan_runtime();
        CompiledModel model;
        auto compiled = compile_model(descriptor, mapping, model, HybridMode::HybridExperts, opt);
        check(static_cast<bool>(compiled), compiled ? "" : compiled.error().message);
        for (const ExecutionNode& node : model.graph.nodes)
            check(node.backend == ExecutionBackend::Cpu, "unbudgeted dense node must execute on CPU");
        check(!model.weights.at(model.lm_head_weight).bfloat16_values().empty(), "fallback LM head host weights were released");
        SessionState state;
        SessionStatistics statistics;
        const std::vector<int32_t> ids = {1, 2};
        auto prefill = forward_model(model, ids, statistics, state, 0, LogitsOutput::Last);
        check(static_cast<bool>(prefill), prefill ? "" : prefill.error().message);
        check_logits(std::vector<float>(state.logits.row(0), state.logits.row(0) + state.logits.columns()),
                     prefill_logits, "unbudgeted dense prefill", false);
        const std::vector<int32_t> next = {3};
        auto decode = forward_model(model, next, statistics, state, ids.size(), LogitsOutput::Last);
        check(static_cast<bool>(decode), decode ? "" : decode.error().message);
        check_logits(std::vector<float>(state.logits.row(0), state.logits.row(0) + state.logits.columns()),
                     decode_logits, "unbudgeted dense decode", false);
    }
}

struct DenseTestDimensions
{
    uint32_t hidden = 4;
    uint32_t intermediate = 8;
    uint32_t vocabulary = 8;
    uint32_t full_head_dimension = 2;
    uint32_t linear_head_dimension = 2;
    uint32_t linear_value_heads = 1;
};

static void write_weights(const std::filesystem::path& root, bool tied, bool missing_down, bool quantized = false,
                          const DenseTestDimensions* dimensions = nullptr)
{
    DenseTestDimensions defaults;
    if (quantized)
        defaults.hidden = defaults.intermediate = 256;
    const DenseTestDimensions& dims = dimensions ? *dimensions : defaults;
    const uint32_t hidden = dims.hidden;
    const uint32_t intermediate = dims.intermediate;
    const uint32_t linear_key_size = dims.linear_head_dimension;
    const uint32_t linear_value_size = dims.linear_head_dimension * dims.linear_value_heads;
    using Tensor = std::pair<std::string, std::vector<uint32_t>>;
    std::vector<Tensor> tensors = {
        {"model.language_model.embed_tokens.weight", {dims.vocabulary, hidden}},
        {"model.language_model.norm.weight", {hidden}},
    };
    if (!tied)
        tensors.push_back({"lm_head.weight", {dims.vocabulary, hidden}});
    for (uint32_t layer_id = 0; layer_id < 2; ++layer_id)
    {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer_id) + ".";
        const std::vector<Tensor> common = {
            {"input_layernorm.weight", {hidden}},
            {"post_attention_layernorm.weight", {hidden}},
            {"mlp.gate_proj.weight", {intermediate, hidden}},
            {"mlp.up_proj.weight", {intermediate, hidden}},
        };
        for (const auto& item : common)
            tensors.push_back({prefix + item.first, item.second});
        if (!missing_down || layer_id != 0)
            tensors.push_back({prefix + "mlp.down_proj.weight", {hidden, intermediate}});
        const std::vector<Tensor> attention = layer_id == 0
                                                  ? std::vector<Tensor>{
                                                        {"linear_attn.in_proj_qkv.weight", {linear_key_size * 2 + linear_value_size, hidden}},
                                                        {"linear_attn.in_proj_z.weight", {linear_value_size, hidden}},
                                                        {"linear_attn.in_proj_b.weight", {dims.linear_value_heads, hidden}},
                                                        {"linear_attn.in_proj_a.weight", {dims.linear_value_heads, hidden}},
                                                        {"linear_attn.conv1d.weight", {linear_key_size * 2 + linear_value_size, 1, 2}},
                                                        {"linear_attn.dt_bias", {dims.linear_value_heads}},
                                                        {"linear_attn.A_log", {dims.linear_value_heads}},
                                                        {"linear_attn.norm.weight", {dims.linear_head_dimension}},
                                                        {"linear_attn.out_proj.weight", {hidden, linear_value_size}},
                                                    }
                                                  : std::vector<Tensor>{
                                                        {"self_attn.q_proj.weight", {dims.full_head_dimension * 4, hidden}},
                                                        {"self_attn.k_proj.weight", {dims.full_head_dimension, hidden}},
                                                        {"self_attn.v_proj.weight", {dims.full_head_dimension, hidden}},
                                                        {"self_attn.o_proj.weight", {hidden, dims.full_head_dimension * 2}},
                                                        {"self_attn.q_norm.weight", {dims.full_head_dimension}},
                                                        {"self_attn.k_norm.weight", {dims.full_head_dimension}},
                                                    };
        for (const auto& item : attention)
            tensors.push_back({prefix + item.first, item.second});
    }
    std::ostringstream header;
    std::vector<uint8_t> data;
    header << '{';
    for (size_t index = 0; index < tensors.size(); ++index)
    {
        const auto& item = tensors[index];
        size_t elements = 1;
        if (index != 0)
            header << ',';
        const bool packed = quantized && item.first.find(".mlp.") != std::string::npos;
        if (packed)
        {
            const DType dtype = item.first.find("down_proj") != std::string::npos ? DType::Q6K : DType::Q4K;
            const size_t bytes = qnk_storage_bytes(dtype, item.second[0], item.second[1]);
            const size_t begin = data.size();
            data.insert(data.end(), bytes, 0);
            header << '"' << item.first << R"(":{"dtype":"U8","shape":[)" << bytes
                   << "],\"data_offsets\":[" << begin << ',' << data.size() << "]}";
            continue;
        }
        header << '"' << item.first << R"(":{"dtype":"BF16","shape":[)";
        for (size_t axis = 0; axis < item.second.size(); ++axis)
        {
            if (axis != 0)
                header << ',';
            header << item.second[axis];
            elements *= item.second[axis];
        }
        const size_t begin = data.size();
        // Distinct rows expose projection permutations and stale activations.
        // DeltaNet's gated norm uses weight directly; the other norms add one.
        const bool norm = item.first.find("norm.weight") != std::string::npos;
        const bool delta_norm = item.first.find("linear_attn.norm.weight") != std::string::npos;
        for (size_t element = 0; element < elements; ++element)
        {
            const int pattern = static_cast<int>((element * 7 + index * 3 + element / hidden) % 19) - 9;
            const float matrix_scale = dimensions ? 0.00390625f : 0.015625f;
            const float value = (delta_norm ? 1.0f : 0.0f) + static_cast<float>(pattern) * (norm ? 0.0078125f : matrix_scale);
            const uint16_t bits = float_to_bfloat16(value);
            data.push_back(static_cast<uint8_t>(bits));
            data.push_back(static_cast<uint8_t>(bits >> 8));
        }
        header << "],\"data_offsets\":[" << begin << ',' << data.size() << "]}";
    }
    header << '}';
    std::string json = header.str();
    json.append((8 - json.size() % 8) % 8, ' ');
    const uint64_t header_size = json.size();
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(json.data(), static_cast<std::streamsize>(json.size()));
    output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    check(static_cast<bool>(output), "cannot write dense test weights");
}

struct DenseBudgetTrace
{
    std::vector<float> prefill;
    std::vector<float> decode;
    VulkanStatistics counters;
    uint64_t gated_delta_state_bytes = 0;
};

static DenseBudgetTrace execute_budget_model(const CompiledModel& model)
{
    SessionState state;
    SessionStatistics statistics;
    const VulkanStatistics before = get_vulkan_statistics(model.vulkan_runtime);
    const std::vector<int32_t> ids = {1, 2};
    auto prefill = forward_model(model, ids, statistics, state, 0, LogitsOutput::Last);
    check(static_cast<bool>(prefill), prefill ? "" : prefill.error().message);
    DenseBudgetTrace trace;
    trace.prefill.assign(state.logits.row(0), state.logits.row(0) + state.logits.columns());
    const std::vector<int32_t> next = {3};
    auto decode = forward_model(model, next, statistics, state, ids.size(), LogitsOutput::Last);
    check(static_cast<bool>(decode), decode ? "" : decode.error().message);
    trace.decode.assign(state.logits.row(0), state.logits.row(0) + state.logits.columns());
    const VulkanStatistics after = get_vulkan_statistics(model.vulkan_runtime);
    trace.counters.dispatches = after.dispatches - before.dispatches;
    trace.counters.compute_submissions = after.compute_submissions - before.compute_submissions;
    trace.counters.attention_blocks = after.attention_blocks - before.attention_blocks;
    trace.counters.gated_delta_fusions = after.gated_delta_fusions - before.gated_delta_fusions;
    trace.counters.attention_cpu_fallbacks = after.attention_cpu_fallbacks - before.attention_cpu_fallbacks;
    for (size_t index = 0; index < state.layers.size(); ++index)
        if (model.graph.layer_plans[index].attention.kind == AttentionKind::GatedDeltaNet)
            trace.gated_delta_state_bytes += state.layers[index].device_allocated_size;
    check(statistics.expert_assignments == 0 && statistics.expert_batches == 0,
          "budgeted dense execution recorded expert routes");
    return trace;
}

static void check_budget_trace(const DenseBudgetTrace& actual, const DenseBudgetTrace& expected,
                               const std::string& context)
{
    check_logits(actual.prefill, expected.prefill, context + " prefill");
    check_logits(actual.decode, expected.decode, context + " decode");
}

static size_t resident_ffn_projections(const CompiledModel& model)
{
    size_t count = 0;
    for (const CompiledLayerPlan& layer : model.graph.layer_plans)
        for (TensorHandle handle : {layer.dense_ffn.gate_weight, layer.dense_ffn.up_weight, layer.dense_ffn.down_weight})
            if (model.operators.at_weight(handle).bfloat16 || model.operators.at_weight(handle).linear)
                ++count;
    return count;
}

static ModelPackage write_dense_fixture(const std::filesystem::path& root, const DenseTestDimensions& dims)
{
    std::filesystem::create_directories(root);
    ModelPackage package;
    package.root = root;
    package.manifest.model_type = "qwen3_5";
    std::ostringstream config;
    config << R"({"model_type":"qwen3_5","text_config":{
        "model_type":"qwen3_5_text","vocab_size":)"
           << dims.vocabulary << R"(,"hidden_size":)" << dims.hidden
           << R"(,"intermediate_size":)" << dims.intermediate << R"(,"num_hidden_layers":2,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":)"
           << dims.full_head_dimension
           << R"(,"linear_num_key_heads":1,"linear_num_value_heads":)" << dims.linear_value_heads
           << R"(,"linear_key_head_dim":)" << dims.linear_head_dimension
           << R"(,"linear_value_head_dim":)" << dims.linear_head_dimension << R"(,"linear_conv_kernel_dim":2,
        "max_position_embeddings":32,"rms_norm_eps":0.000001,
        "rope_parameters":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
        "layer_types":["linear_attention","full_attention"],"hidden_act":"silu",
        "dtype":"bfloat16","mamba_ssm_dtype":"float32",
        "attention_bias":false,"attn_output_gate":true,"tie_word_embeddings":false}})";
    package.manifest.raw_json = config.str();
    std::ofstream(root / "config.json") << package.manifest.raw_json;
    write_weights(root, false, false, false, &dims);
    return package;
}

static bool test_partial_gpu_budget(const std::filesystem::path& root, const GpuInfo& gpu)
{
    const uint32_t device_index = gpu.device_index;
    const std::string device = "device=" + std::to_string(device_index) + " (" + gpu.device_name + ")";
    if (gpu.compute_queue_count == 0 || gpu.heap_available == 0)
    {
        std::cout << "SKIP dense GPU budget " << device
                  << (gpu.compute_queue_count == 0 ? ": no Vulkan compute queue\n" : ": device reports no available Vulkan heap\n");
        return false;
    }
    std::cout << "Dense GPU budget testing " << device << '\n';
    DenseTestDimensions dims;
    dims.hidden = 256;
    dims.intermediate = 1024;
    // The head must exceed both attention groups, including their dedicated
    // constant allocators, so admission can skip it and still use attention.
    dims.vocabulary = 65536;
    dims.full_head_dimension = 4;
    dims.linear_head_dimension = 4;
    dims.linear_value_heads = 3;
    const ModelPackage package = write_dense_fixture(root, dims);
    Qwen3_5MoeModelAdapter adapter;
    auto descriptor = adapter.parse_model(package);
    check(static_cast<bool>(descriptor), descriptor ? "" : descriptor.error().message);
    auto mapping = adapter.map_weights(package, descriptor.value());
    check(static_cast<bool>(mapping), mapping ? "" : mapping.error().message);
    CompilerOption cpu_opt;
    cpu_opt.flags = BackendCpuExecution;
    CompiledModel cpu;
    auto status = compile_model(descriptor.value(), mapping.value(), cpu, HybridMode::CpuOnly, cpu_opt);
    check(static_cast<bool>(status), status ? "" : status.error().message);
    const DenseBudgetTrace expected = execute_budget_model(cpu);
    cpu.vulkan_runtime = create_vulkan_runtime();
    // Every large fixture buffer is a multiple of 64 KiB; 256 also bounds
    // Vulkan's required storage-buffer alignment for the smaller segments.
    constexpr uint64_t alignment = 256;
    TensorData oversized_head;
    oversized_head.dtype = DType::Float32;
    oversized_head.shape = {248320, 5120};
    check(dense_vulkan_weight_storage_size(cpu, oversized_head, alignment, device_index) == 0,
          "FP32 head exceeding Vulkan's single-buffer range was admitted");
    TensorData biased_projection;
    biased_projection.dtype = DType::BFloat16;
    biased_projection.shape = {4, 6};
    const uint64_t bias_free_storage = dense_vulkan_weight_storage_size(cpu, biased_projection, alignment, device_index);
    check(bias_free_storage > 0
              && dense_vulkan_weight_storage_size(cpu, biased_projection, alignment, device_index, true) > bias_free_storage,
          "generic FP32 projection budget omitted its bias allocation");
    const auto compile_budget = [&](CompiledModel& model, uint64_t budget, bool attention, bool release_host) {
        CompilerOption opt;
        opt.flags = BackendCpuExecution | BackendVulkanDense;
        if (attention)
            opt.flags |= BackendVulkanAttention;
        if (release_host)
            opt.flags |= BackendReleaseVulkanDenseHostStorage;
        opt.device_index = device_index;
        opt.gpu_heap_budget = budget;
        opt.vulkan_runtime = create_vulkan_runtime();
        auto compiled = compile_model(descriptor.value(), mapping.value(), model, HybridMode::HybridExperts, opt);
        check(static_cast<bool>(compiled), compiled ? "" : compiled.error().message);
        const CompiledOperator& head = model.operators.at_weight(model.lm_head_weight);
        check(!head.bfloat16 && !head.linear,
              "oversize LM head consumed the limited GPU budget");
        check(!model.weights.at(model.lm_head_weight).bfloat16_values().empty(),
              "CPU LM head lost host weights under partial residency");
    };

    uint64_t attention_storage = 0;
    bool attention_supported = true;
    for (const CompiledLayerPlan& layer : cpu.graph.layer_plans)
    {
        const uint64_t size = dense_vulkan_attention_storage_size(cpu, layer, alignment, device_index);
        if (size == 0)
            attention_supported = false;
        attention_storage += size;
    }
    // Invert the small-budget workspace reserve, leaving only enough capacity
    // for the two attention groups, ahead of any dense FFN projection.
    const uint64_t attention_budget = (attention_storage * 4 + 2) / 3;
    const uint64_t projection_storage = dense_vulkan_weight_storage_size(cpu, cpu.weights.at(cpu.graph.layer_plans.front().dense_ffn.gate_weight),
                                                                         alignment, device_index);
    const uint64_t head_storage = dense_vulkan_weight_storage_size(cpu, cpu.weights.at(cpu.lm_head_weight), alignment, device_index);
    if (projection_storage == 0)
    {
        std::cout << "SKIP dense GPU budget partial FFN " << device
                  << ": device cannot represent the FFN projection fixture\n";
        return false;
    }
    const uint64_t partial_budget = projection_storage + projection_storage / 2;
    check(head_storage == 0 || head_storage > partial_budget, "budget fixture head does not exceed the partial FFN budget");
    bool tested_attention = false;
    if (attention_supported && gpu.heap_available >= attention_budget)
    {
        check(attention_storage > 0 && (head_storage == 0 || head_storage > attention_budget),
              "budget fixture head does not exceed the attention budget");
        for (bool release_host : {false, true})
        {
            CompiledModel model;
            compile_budget(model, attention_budget, true, release_host);
            size_t attention_count = 0;
            for (const CompiledLayerPlan& layer : model.graph.layer_plans)
            {
                check(support_vulkan_attention(model.operators, layer.attention),
                      "limited budget did not prioritize every attention layer");
                ++attention_count;
            }
            check(attention_count == 2 && resident_ffn_projections(model) == 0,
                  "attention-only budget admitted a dense FFN projection");
            const DenseBudgetTrace actual = execute_budget_model(model);
            check_budget_trace(actual, expected, device + " attention-priority budget");
            check(actual.counters.attention_blocks > 0 && actual.counters.gated_delta_fusions > 0,
                  "resident attention groups did not execute both Vulkan attention kinds");
            check(actual.counters.attention_cpu_fallbacks == 0,
                  "budgeted attention unexpectedly fell back during execution");
            check(actual.gated_delta_state_bytes > 0
                      && dense_vulkan_pending_state_size(model, device_index) >= actual.gated_delta_state_bytes,
                  "pending DeltaNet reservation does not cover the real decode state allocation");
            std::cout << "Dense budget attention " << device << ": groups=" << attention_count << "/2 FFN projections=0/6 head_gpu=0 release_host=" << release_host
                      << " attention=" << actual.counters.attention_blocks
                      << " DeltaNet=" << actual.counters.gated_delta_fusions << '\n';
        }
        tested_attention = true;
    }
    else
    {
        std::cout << "SKIP dense GPU budget attention " << device
                  << (attention_supported ? ": available heap is below the attention fixture budget\n"
                                          : ": device does not support every attention kind in the fixture\n");
    }

    if (gpu.heap_available < partial_budget)
    {
        std::cout << "SKIP dense GPU budget partial FFN " << device
                  << ": available heap is below the partial FFN fixture budget\n";
        return tested_attention;
    }

    for (bool release_host : {false, true})
    {
        CompiledModel model;
        compile_budget(model, partial_budget, false, release_host);
        check(resident_ffn_projections(model) == 1,
              "partial FFN budget did not retain exactly one projection");
        size_t gpu_ffn_nodes = 0;
        for (const ExecutionNode& node : model.graph.nodes)
            if (node.type == ExecutionNodeType::DenseFfn && node.backend == ExecutionBackend::Vulkan)
                ++gpu_ffn_nodes;
        check(gpu_ffn_nodes == 1, "partially resident FFN was not scheduled on Vulkan");
        for (const CompiledLayerPlan& layer : model.graph.layer_plans)
        {
            check(!support_vulkan_attention(model.operators, layer.attention),
                  "disabled attention consumed the partial FFN budget");
            for (TensorHandle handle : {layer.dense_ffn.gate_weight, layer.dense_ffn.up_weight, layer.dense_ffn.down_weight})
            {
                const CompiledOperator& projection = model.operators.at_weight(handle);
                const bool resident = projection.bfloat16 || projection.linear;
                const bool host_present = !model.weights.at(handle).bfloat16_values().empty();
                check(host_present == (!release_host || !resident),
                      "partial host release removed a fallback or retained resident-only FFN weights");
            }
        }
        const DenseBudgetTrace actual = execute_budget_model(model);
        check_budget_trace(actual, expected, device + (release_host ? " released partial FFN budget" : " partial FFN budget"));
        // Head and attention are CPU-only here, so these dispatches can only
        // come from the one admitted FFN projection, across prefill and decode.
        check(actual.counters.dispatches >= 2 && actual.counters.compute_submissions >= 2,
              "partially resident FFN projection was uploaded but not executed");
        check(actual.counters.attention_blocks == 0 && actual.counters.gated_delta_fusions == 0,
              "disabled attention unexpectedly executed on Vulkan");
        check(actual.gated_delta_state_bytes == 0 && dense_vulkan_pending_state_size(model, device_index) == 0,
              "unprepared attention contributed a pending GPU state reservation");
        const CompiledOperator& projection = model.operators.at_weight(model.graph.layer_plans.front().dense_ffn.gate_weight);
        check(projection.bfloat16 || projection.linear, "first admitted FFN projection has no Vulkan operator");
        std::cout << "Dense budget partial FFN " << device
                  << ": path=" << (projection.bfloat16 ? "BF16" : "FP32")
                  << " projections=1/6 head_gpu=0 release_host=" << release_host
                  << " dispatches=" << actual.counters.dispatches
                  << " submissions=" << actual.counters.compute_submissions << '\n';
    }
    return true;
}

enum class DenseCacheFault
{
    Demand,
    Execution,
    Commit
};

class FailingDenseBackend final : public ExpertBackend
{
public:
    explicit FailingDenseBackend(DenseCacheFault fault)
        : fault(fault)
    {
    }

    void admit(std::string, std::shared_ptr<const TensorData>, const TensorData*,
               std::shared_ptr<const TensorData>, const TensorData*, uint32_t, float, ExpertActivation) override
    {
    }

    bool prepare_demand(std::string key, std::shared_ptr<const TensorData>, const TensorData*,
                        std::shared_ptr<const TensorData>, const TensorData*, uint32_t, float,
                        ExpertActivation, std::shared_ptr<const void>& pin) override
    {
        ++demands;
        pin.reset();
        if (fault == DenseCacheFault::Demand)
            return false;
        prepared.insert(std::move(key));
        pin = std::make_shared<int>(1);
        return true;
    }

    bool prepare_demand_separate(std::string key, std::shared_ptr<const TensorData> gate, const TensorData* gate_bias,
                                 std::shared_ptr<const TensorData>, const TensorData*,
                                 std::shared_ptr<const TensorData> down, const TensorData* down_bias,
                                 uint32_t group, float limit, ExpertActivation activation,
                                 std::shared_ptr<const void>& pin) override
    {
        return prepare_demand(std::move(key), std::move(gate), gate_bias, std::move(down), down_bias,
                              group, limit, activation, pin);
    }

    class Submission final : public ExpertSubmission
    {
    public:
        Submission(FailingDenseBackend& owner, std::span<const ExpertBackendRequest> requests)
            : owner(owner)
        {
            for (const auto& request : requests)
            {
                const bool resident = owner.prepared.contains(std::string(request.key));
                reserved.push_back(resident ? ExpertBackendExecutionResult::Executed : ExpertBackendExecutionResult::NotResident);
                if (resident)
                    ++owner.resident_reservations;
            }
        }

        std::span<const ExpertBackendExecutionResult> reservations() const noexcept override
        {
            return reserved;
        }

        std::vector<ExpertBackendExecutionResult> wait() override
        {
            ++owner.waits;
            auto result = reserved;
            if (owner.fault == DenseCacheFault::Execution)
                for (auto& item : result)
                    if (item == ExpertBackendExecutionResult::Executed)
                        item = ExpertBackendExecutionResult::Failed;
            return result;
        }

        bool commit() override
        {
            ++owner.commits;
            // No submission publishes output. A successful reservation or wait
            // must not prevent the dense executor from running CPU fallback.
            return false;
        }

        void abort() noexcept override
        {
            if (!aborted)
                ++owner.aborts;
            aborted = true;
        }

    private:
        FailingDenseBackend& owner;
        std::vector<ExpertBackendExecutionResult> reserved;
        bool aborted = false;
    };

    std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override
    {
        ++submissions;
        return std::make_unique<Submission>(*this, requests);
    }

    void wait_for_background_work() override
    {
    }
    ExpertBackendStatistics statistics() const override
    { return {}; }

    uint64_t demands = 0;
    uint64_t submissions = 0;
    uint64_t resident_reservations = 0;
    uint64_t waits = 0;
    uint64_t commits = 0;
    uint64_t aborts = 0;

private:
    DenseCacheFault fault;
    std::unordered_set<std::string> prepared;
};

static void check_dynamic_dense_plan(const CompiledModel& model, uint64_t entry_size, bool require_backend)
{
    check(!require_backend || model.expert_backend, "runtime did not create the dense FFN GPU cache");
    check(resident_ffn_projections(model) == 0, "dynamic FFN retained static Vulkan projections");
    std::unordered_set<std::string> keys;
    for (const CompiledLayerPlan& layer : model.graph.layer_plans)
    {
        const ExpertPlan& ffn = layer.dense_ffn;
        check(ffn.use_gpu_cache && !ffn.cache_key.empty() && ffn.weight_size == entry_size,
              "dense FFN was not compiled as a budgeted cache entry");
        check(ffn.residency_group == layer.layer_id && keys.insert(ffn.cache_key).second,
              "dense cache key or residency group is not unique to its layer");
        for (TensorHandle handle : {ffn.gate_weight, ffn.up_weight, ffn.down_weight})
            check(!model.weights.at(handle).bfloat16_values().empty(),
                  "dynamic FFN lost the host weights needed after GPU eviction");
    }
    check(keys.size() == 2, "dynamic fixture does not contain two distinct dense cache entries");
}

static void check_dynamic_cache_batch(const CompiledModel& cpu, const CompiledModel& model, uint64_t capacity,
                                      const std::string& context)
{
    std::array<SessionState, 2> cpu_states;
    std::array<SessionState, 2> gpu_states;
    std::array<SessionStatistics, 2> cpu_statistics;
    std::array<SessionStatistics, 2> gpu_statistics;
    const std::array<std::array<int32_t, 2>, 2> prefixes = {{{1, 2}, {2, 1}}};
    std::array<std::vector<float>, 2> expected;
    std::array<std::vector<float>, 2> actual;
    std::array<DecodeBatchEntry, 2> entries;
    for (size_t index = 0; index < entries.size(); ++index)
    {
        auto cpu_prefill = forward_model(cpu, prefixes[index], cpu_statistics[index], cpu_states[index], 0, LogitsOutput::Last);
        auto gpu_prefill = forward_model(model, prefixes[index], gpu_statistics[index], gpu_states[index], 0, LogitsOutput::Last);
        check(cpu_prefill && gpu_prefill, "dynamic batch fixture prefill failed");
        check_logits(std::vector<float>(gpu_states[index].logits.row(0), gpu_states[index].logits.row(0) + gpu_states[index].logits.columns()),
                     std::vector<float>(cpu_states[index].logits.row(0), cpu_states[index].logits.row(0) + cpu_states[index].logits.columns()),
                     context + " batch prefill session=" + std::to_string(index));
        const std::array<int32_t, 1> next = {static_cast<int32_t>(index + 3)};
        auto cpu_decode = forward_model(cpu, next, cpu_statistics[index], cpu_states[index], 2, LogitsOutput::Last);
        check(static_cast<bool>(cpu_decode), cpu_decode ? "" : cpu_decode.error().message);
        expected[index].assign(cpu_states[index].logits.row(0), cpu_states[index].logits.row(0) + cpu_states[index].logits.columns());
        entries[index] = {next.front(), &gpu_statistics[index], &gpu_states[index], 2, &actual[index]};
    }
    const auto before = model.expert_backend->statistics();
    BatchWorkspace workspace;
    auto batch = forward_decode_batch(model, entries, workspace);
    check(static_cast<bool>(batch), batch ? "" : batch.error().message);
    const auto after = model.expert_backend->statistics();
    check(after.executions - before.executions == 2,
          "two-session dense batch did not execute one GPU cache request per layer");
    check(after.evictions >= before.evictions + 2 && after.resident_size <= capacity && after.pending_size == 0,
          "batched dense cache execution exceeded capacity or failed to rotate layers");
    for (size_t index = 0; index < entries.size(); ++index)
    {
        check_logits(actual[index], expected[index], context + " batch session=" + std::to_string(index));
        check(gpu_statistics[index].expert_assignments == 0 && gpu_statistics[index].expert_batches == 0,
              "dense batch execution recorded MoE routes");
    }
}

static bool test_dynamic_gpu_cache(const std::filesystem::path& root, const GpuInfo& gpu)
{
    const std::string context = "dynamic dense device=" + std::to_string(gpu.device_index) + " (" + gpu.device_name + ")";
    if (gpu.compute_queue_count == 0 || gpu.heap_available == 0)
    {
        std::cout << "SKIP " << context << ": no compute queue or reported available heap\n";
        return false;
    }
    DenseTestDimensions dims;
    dims.hidden = 256;
    dims.intermediate = 1024;
    dims.full_head_dimension = dims.linear_head_dimension = 4;
    dims.linear_value_heads = 3;
    const ModelPackage package = write_dense_fixture(root, dims);
    Qwen3_5MoeModelAdapter adapter;
    auto descriptor = adapter.parse_model(package);
    check(static_cast<bool>(descriptor), descriptor ? "" : descriptor.error().message);
    auto mapping = adapter.map_weights(package, descriptor.value());
    check(static_cast<bool>(mapping), mapping ? "" : mapping.error().message);
    CompilerOption cpu_opt;
    cpu_opt.flags = BackendCpuExecution;
    CompiledModel cpu;
    auto status = compile_model(descriptor.value(), mapping.value(), cpu, HybridMode::CpuOnly, cpu_opt);
    check(static_cast<bool>(status), status ? "" : status.error().message);
    const DenseBudgetTrace expected = execute_budget_model(cpu);
    const ExpertPlan& first = cpu.graph.layer_plans.front().dense_ffn;
    const auto sizing_runtime = create_vulkan_runtime();
    const uint64_t entry_size = native_quantized_expert_storage_bytes(cpu.weights.at(first.gate_weight),
                                                                      &cpu.weights.at(first.up_weight), cpu.weights.at(first.down_weight), gpu.device_index,
                                                                      sizing_runtime, OptimizationDefaultFlags);
    if (entry_size == 0 || gpu.heap_available < entry_size * 4)
    {
        std::cout << "SKIP " << context << ": dense cache entry is unsupported or available heap is too small\n";
        return false;
    }
    Runtime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::HybridExperts;
    opt.vulkan_device_index = gpu.device_index;
    opt.vulkan_device_indices = {gpu.device_index};
    opt.num_concurrent_sessions = 2;
    opt.flags = OptionReleaseVulkanDenseHostStorage;
    opt.optimization_flags &= ~OptimizationVulkanAttention;
    opt.expert_gpu_cache_size = entry_size;
    {
        auto loaded = runtime.load_model(root, opt);
        check(static_cast<bool>(loaded), loaded ? "" : loaded.error().message);
        const CompiledModel& model = model_compiled(*loaded.value());
        check_dynamic_dense_plan(model, entry_size, true);
        check(model.opt.expert_gpu_cache_size == entry_size, "explicit dense GPU cache budget was not preserved");
        const auto cold = model.expert_backend->statistics();
        check(cold.resident_size == 0 && cold.bytes_uploaded == 0, "dense FFNs uploaded before their first demand");
        auto session = runtime.create_session(loaded.value());
        check(static_cast<bool>(session), "cannot create dynamically cached dense session");
        const std::array<int32_t, 2> ids = {1, 2};
        auto prefill = session.value()->prefill(ids);
        check(static_cast<bool>(prefill), prefill ? "" : prefill.error().message);
        check_logits(prefill.value().logits, expected.prefill, context + " cold prefill");
        const auto first_pass = model.expert_backend->statistics();
        check(first_pass.executions >= 2 && first_pass.stores >= 2 && first_pass.evictions >= 1
                  && first_pass.resident_size <= entry_size && first_pass.pending_size == 0,
              "one-entry dense cache did not execute and evict both cold layers");
        auto decode = session.value()->decode(3);
        check(static_cast<bool>(decode), decode ? "" : decode.error().message);
        check_logits(decode.value().logits, expected.decode, context + " evicted decode");
        const auto after = model.expert_backend->statistics();
        check(after.executions >= first_pass.executions + 2 && after.evictions >= first_pass.evictions + 2
                  && after.stores >= first_pass.stores + 2 && after.bytes_uploaded >= entry_size * 4
                  && after.resident_size <= entry_size && after.pending_size == 0,
              "dense decode did not reload evicted FFNs within the one-entry cache");
        const auto statistics = session.value()->statistics();
        check(statistics.expert_assignments == 0 && statistics.expert_batches == 0,
              "dynamically cached dense FFNs recorded expert routes");
        check_dynamic_dense_plan(model, entry_size, true);
        check_dynamic_cache_batch(cpu, model, entry_size, context);
        std::cout << context << ": one-entry executions=" << after.executions << " evictions=" << after.evictions
                  << " uploaded=" << after.bytes_uploaded << " resident=" << after.resident_size << '/' << entry_size << '\n';
    }
    {
        Option automatic = opt;
        automatic.expert_gpu_cache_size = 0;
        auto loaded = runtime.load_model(root, automatic);
        check(static_cast<bool>(loaded), loaded ? "" : loaded.error().message);
        const CompiledModel& model = model_compiled(*loaded.value());
        check_dynamic_dense_plan(model, entry_size, true);
        check(model.opt.expert_gpu_cache_size >= entry_size && model.opt.expert_gpu_cache_size <= entry_size * 2,
              "automatic dense cache did not resolve to the eligible FFN payload");
        check_budget_trace(execute_budget_model(model), expected, context + " automatic cache");
        check(model.expert_backend->statistics().executions >= 4, "automatic dense cache was created but not executed");
    }
    {
        Option insufficient = opt;
        insufficient.expert_gpu_cache_size = entry_size - 1;
        auto loaded = runtime.load_model(root, insufficient);
        check(static_cast<bool>(loaded), loaded ? "" : loaded.error().message);
        const CompiledModel& model = model_compiled(*loaded.value());
        check_dynamic_dense_plan(model, entry_size, false);
        check(model.opt.expert_gpu_cache_size == 0,
              "undersized dense cache reported a GPU capacity without constructing a backend");
        check_budget_trace(execute_budget_model(model), expected, context + " insufficient cache");
        if (model.expert_backend)
        {
            const auto statistics = model.expert_backend->statistics();
            check(statistics.executions == 0 && statistics.bytes_uploaded == 0 && statistics.resident_size == 0 && statistics.pending_size == 0,
                  "undersized dense GPU cache uploaded or executed an entry");
        }
    }
    {
        // The direct compiler flag exercises host release independently of
        // Runtime's existing Eager/OnDemand policy for a pure dense model.
        CompilerOption release_opt;
        release_opt.flags = BackendCpuExecution | BackendVulkanDense | BackendVulkanExperts | BackendReleaseVulkanDenseHostStorage;
        release_opt.device_index = gpu.device_index;
        release_opt.gpu_heap_budget = gpu.heap_available;
        release_opt.vulkan_runtime = create_vulkan_runtime();
        CompiledModel model;
        auto compiled = compile_model(descriptor.value(), mapping.value(), model, HybridMode::HybridExperts, release_opt);
        check(static_cast<bool>(compiled), compiled ? "" : compiled.error().message);
        check_dynamic_dense_plan(model, entry_size, false);
        const CompiledOperator& head = model.operators.at_weight(model.lm_head_weight);
        check((head.bfloat16 || head.linear) && model.weights.at(model.lm_head_weight).bfloat16_values().empty(),
              "host-release fixture did not release a persistent Vulkan projection");
    }
    for (DenseCacheFault fault : {DenseCacheFault::Demand, DenseCacheFault::Execution, DenseCacheFault::Commit})
    {
        CompilerOption failing_opt;
        failing_opt.flags = BackendCpuExecution | BackendVulkanDense | BackendVulkanExperts | BackendReleaseVulkanDenseHostStorage;
        failing_opt.device_index = gpu.device_index;
        failing_opt.vulkan_runtime = create_vulkan_runtime();
        CompiledModel model;
        auto compiled = compile_model(descriptor.value(), mapping.value(), model, HybridMode::HybridExperts, failing_opt);
        check(static_cast<bool>(compiled), compiled ? "" : compiled.error().message);
        auto backend = std::make_shared<FailingDenseBackend>(fault);
        model.expert_backend = backend;
        check_dynamic_dense_plan(model, entry_size, true);
        check_budget_trace(execute_budget_model(model), expected, context + " failed cache stage=" + std::to_string(static_cast<int>(fault)));
        ActivationBuffer input(2, dims.hidden);
        for (size_t row = 0; row < input.rows(); ++row)
            for (uint32_t column = 0; column < input.columns(); ++column)
                input.row(row)[column] = static_cast<float>(static_cast<int>((row * 7 + column * 3) % 19) - 9) * 0.03125f;
        ActivationBuffer expected_ffn;
        ActivationBuffer actual_ffn(input.rows(), dims.hidden);
        std::fill_n(actual_ffn.row(0), actual_ffn.values().size(), 123.0f);
        ExpertWorkspace cpu_workspace;
        ExpertWorkspace failing_workspace;
        forward_dense_ffn(cpu, first, input, expected_ffn, cpu_workspace, ExecutionBackend::Cpu);
        forward_dense_ffn(model, model.graph.layer_plans.front().dense_ffn, input, actual_ffn,
                          failing_workspace, ExecutionBackend::Vulkan);
        for (size_t row = 0; row < input.rows(); ++row)
            check_logits(std::vector<float>(actual_ffn.row(row), actual_ffn.row(row) + actual_ffn.columns()),
                         std::vector<float>(expected_ffn.row(row), expected_ffn.row(row) + expected_ffn.columns()),
                         context + " unpublished failure output row=" + std::to_string(row));
        check(backend->demands > 0 && backend->submissions > 0 && backend->aborts > 0,
              "dense cache fault fixture did not exercise demand and failed submission cleanup");
        if (fault != DenseCacheFault::Demand)
            check(backend->resident_reservations > 0 && backend->waits > 0,
                  "dense cache execution fault did not reach a resident submission");
        if (fault == DenseCacheFault::Commit)
            check(backend->commits > 0, "dense cache commit fault was not reached");
    }
    return true;
}

static void run_dense_package(const std::filesystem::path& root, bool tied, bool quantized = false)
{
    std::filesystem::create_directories(root);
    ModelPackage package;
    package.root = root;
    package.manifest.model_type = "qwen3_5";
    package.manifest.raw_json = dense_config(tied);
    if (quantized)
    {
        const size_t hidden = package.manifest.raw_json.find("\"hidden_size\":4");
        package.manifest.raw_json.replace(hidden, 15, "\"hidden_size\":256");
        const size_t intermediate = package.manifest.raw_json.find("\"intermediate_size\":8");
        package.manifest.raw_json.replace(intermediate, 21, "\"intermediate_size\":256");
    }
    std::ofstream(root / "config.json") << package.manifest.raw_json;
    write_weights(root, tied, false, quantized);
    Qwen3_5MoeModelAdapter adapter;
    check(adapter.can_load(package.manifest), "dense model not recognized");
    auto descriptor = adapter.parse_model(package);
    check(static_cast<bool>(descriptor), descriptor ? "" : descriptor.error().message);
    check(descriptor.value().expert_count == 0 && descriptor.value().experts_per_token == 0, "dense model has routed experts");
    check(descriptor.value().layers.front().dense.intermediate_size == (quantized ? 256 : 8), "dense FFN size lost");
    if (quantized)
    {
        check(descriptor.value().layers.front().dense.gate_weight_dtype == DType::Q4K, "dense Q4_K gate dtype lost");
        check(descriptor.value().layers.front().dense.down_weight_dtype == DType::Q6K, "dense Q6_K down dtype lost");
    }
    auto mapping = adapter.map_weights(package, descriptor.value());
    check(static_cast<bool>(mapping), mapping ? "" : mapping.error().message);
    check(mapping.value().contains("layers.0.ffn.gate.weight"), "dense gate not mapped");
    check(!mapping.value().contains("layers.0.router.weight"), "dense model requires router");
    if (tied)
    {
        const auto head = mapping.value().at("lm_head.weight").bfloat16_values();
        const auto embedding = mapping.value().at("token_embedding.weight").bfloat16_values();
        check(head.size() == embedding.size(), "tied LM head size differs");
        for (size_t index = 0; index < head.size(); ++index)
            check(head[index] == embedding[index], "tied LM head differs");
    }

    Runtime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(root, opt);
    check(static_cast<bool>(model), model ? "" : model.error().message);
    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session), "cannot create dense session");
    const std::vector<int32_t> ids = {1, 2};
    auto prefill = session.value()->prefill(ids);
    check(static_cast<bool>(prefill), prefill ? "" : prefill.error().message);
    check(prefill.value().logits.size() == 8, "invalid dense prefill logits");
    auto decode = session.value()->decode(3);
    check(static_cast<bool>(decode), decode ? "" : decode.error().message);
    for (float logit : decode.value().logits)
        check(std::isfinite(logit), "dense decode returned non-finite logits");
    const auto statistics = session.value()->statistics();
    check(statistics.expert_assignments == 0 && statistics.expert_batches == 0, "dense FFN recorded routing work");
    const auto& gpu_infos = runtime.info().gpu_infos;
    const auto gpu = std::find_if(gpu_infos.begin(), gpu_infos.end(), [](const GpuInfo& info) {
        return info.type != VulkanDeviceType::Cpu && info.compute_queue_count > 0;
    });
    if (gpu != gpu_infos.end())
    {
        if (!quantized)
            check_budget_fallback(descriptor.value(), mapping.value(), prefill.value().logits,
                                  decode.value().logits, gpu->device_index);
        Option hybrid = opt;
        hybrid.hybrid_mode = HybridMode::HybridExperts;
        // Isolate FFN parity from CPU BF16 and Vulkan FP32 KV differences.
        hybrid.optimization_flags &= ~OptimizationVulkanAttention;
        hybrid.vulkan_device_index = gpu->device_index;
        hybrid.vulkan_device_indices = {gpu->device_index};
        auto hybrid_model = runtime.load_model(root, hybrid);
        check(static_cast<bool>(hybrid_model), hybrid_model ? "" : hybrid_model.error().message);
        const CompiledModel& compiled = model_compiled(*hybrid_model.value());
        size_t vulkan_ffn_count = 0;
        size_t cached_ffn_count = 0;
        size_t static_ffn_count = 0;
        size_t dense_ffn_nodes = 0;
        for (const ExecutionNode& node : compiled.graph.nodes)
        {
            if (node.type != ExecutionNodeType::DenseFfn)
                continue;
            ++dense_ffn_nodes;
            if (node.backend == ExecutionBackend::Cpu)
                continue;
            check(node.backend == ExecutionBackend::Vulkan, "dense FFN selected an unsupported backend");
            const CompiledLayerPlan& layer = compiled.graph.layer_plans[node.layer_plan_index];
            if (layer.dense_ffn.use_gpu_cache)
            {
                check(static_cast<bool>(compiled.expert_backend), "cached dense Vulkan FFN lacks an Expert backend");
                ++cached_ffn_count;
            }
            else
            {
                check(support_vulkan_ffn(compiled.operators, layer.dense_ffn),
                      "dense Vulkan plan lacks an executable FFN");
                ++static_ffn_count;
            }
            check(layer.vulkan_device_index == compiled.opt.vulkan_device_index, "dense FFN has the wrong Vulkan device");
            ++vulkan_ffn_count;
        }
        check(dense_ffn_nodes == compiled.descriptor.layers.size(), "dense FFN missing from schedule");
        auto hybrid_session = runtime.create_session(hybrid_model.value());
        check(static_cast<bool>(hybrid_session), "cannot create hybrid dense session");
        auto hybrid_prefill = hybrid_session.value()->prefill(ids);
        check(static_cast<bool>(hybrid_prefill), hybrid_prefill ? "" : hybrid_prefill.error().message);
        const std::string context = std::string("hybrid dense tied=") + (tied ? "true" : "false")
                                    + " quantized=" + (quantized ? "true" : "false");
        check_logits(hybrid_prefill.value().logits, prefill.value().logits, context + " prefill");
        auto hybrid_decode = hybrid_session.value()->decode(3);
        check(static_cast<bool>(hybrid_decode), hybrid_decode ? "" : hybrid_decode.error().message);
        check(hybrid_decode.value().logits.size() == decode.value().logits.size(), context + " decode logits size mismatch");
        for (size_t index = 0; index < decode.value().logits.size(); ++index)
            check(std::abs(hybrid_decode.value().logits[index] - decode.value().logits[index]) < 1e-3f,
                  context + " decode differs from CPU");
        const SessionStatistics hybrid_statistics = hybrid_session.value()->statistics();
        check(hybrid_statistics.expert_assignments == 0, "hybrid dense FFN recorded routes");
        if (cached_ffn_count != 0)
            check(compiled.expert_backend->statistics().executions > 0,
                  "admitted cached dense FFNs did not execute on Vulkan");
        if (static_ffn_count != 0)
            check(hybrid_statistics.vulkan_linear_dispatches > 0,
                  "admitted static dense FFNs produced no Vulkan linear dispatches");
        if (vulkan_ffn_count == 0)
            std::cout << "SKIP hybrid dense Vulkan FFN execution: the compiled plan admitted no FFNs\n";
    }
    else if (!gpu_infos.empty())
    {
        std::cout << "SKIP hybrid dense Vulkan: no non-CPU device with a compute queue is available\n";
    }
    if (quantized)
    {
        Option packed = opt;
        packed.cpu_packed_weight_mode = CpuPackedWeightMode::Enabled;
        auto unsupported = runtime.load_model(root, packed);
        check(!unsupported && unsupported.error().code == ErrorCode::UnsupportedModel
                  && unsupported.error().message.find("CpuPackedWeightMode::Disabled") != std::string::npos,
              "dense packed weights must be rejected before loading");
    }

    const std::filesystem::path missing_root = root / "missing-down";
    std::filesystem::create_directories(missing_root);
    std::ofstream(missing_root / "config.json") << package.manifest.raw_json;
    write_weights(missing_root, tied, true, quantized);
    auto missing = runtime.load_model(missing_root, opt);
    check(!missing && missing.error().message.find("down_proj") != std::string::npos, "missing dense FFN tensor was accepted");
}

} // namespace moe
} // namespace ncnn

int main()
{
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() / ("ncnn-moe-qwen-dense-" + std::to_string(unique));
    try
    {
        std::filesystem::create_directories(root);
        ncnn::moe::run_dense_package(root / "tied", true);
        ncnn::moe::run_dense_package(root / "untied", false);
        ncnn::moe::run_dense_package(root / "quantized", false, true);
        const auto gpus = ncnn::moe::get_gpu_infos();
        size_t tested_devices = 0;
        size_t cached_devices = 0;
        for (const auto& gpu : gpus)
        {
            if (ncnn::moe::test_partial_gpu_budget(root / ("budget-device-" + std::to_string(gpu.device_index)), gpu))
                ++tested_devices;
            if (ncnn::moe::test_dynamic_gpu_cache(root / ("cache-device-" + std::to_string(gpu.device_index)), gpu))
                ++cached_devices;
        }
        std::cout << "Dense GPU budget validation: tested=" << tested_devices
                  << " skipped=" << gpus.size() - tested_devices << '\n';
        std::cout << "Dynamic dense GPU cache validation: tested=" << cached_devices
                  << " skipped=" << gpus.size() - cached_devices << '\n';
        if (gpus.empty())
            std::cout << "SKIP dense GPU budget: no Vulkan device is available\n";
        std::filesystem::remove_all(root);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
