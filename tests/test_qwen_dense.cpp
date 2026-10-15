#include "models/modeladapter_qwen3_5.h"
#include "kernels/qnk.h"
#include "graph/compiledmodel.h"
#include "backends/ncnn/modelpipeline.h"
#include "ncnn/moe/runtime.h"
#include "ncnn/moe/session.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static void check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
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

static void write_weights(const std::filesystem::path& root, bool tied, bool missing_down, bool quantized = false)
{
    const uint32_t hidden = quantized ? 256 : 4;
    const uint32_t intermediate = quantized ? 256 : 8;
    using Tensor = std::pair<std::string, std::vector<uint32_t>>;
    std::vector<Tensor> tensors = {
        {"model.language_model.embed_tokens.weight", {8, hidden}},
        {"model.language_model.norm.weight", {hidden}},
    };
    if (!tied)
        tensors.push_back({"lm_head.weight", {8, hidden}});
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
                                                        {"linear_attn.in_proj_qkv.weight", {6, hidden}},
                                                        {"linear_attn.in_proj_z.weight", {2, hidden}},
                                                        {"linear_attn.in_proj_b.weight", {1, hidden}},
                                                        {"linear_attn.in_proj_a.weight", {1, hidden}},
                                                        {"linear_attn.conv1d.weight", {6, 1, 2}},
                                                        {"linear_attn.dt_bias", {1}},
                                                        {"linear_attn.A_log", {1}},
                                                        {"linear_attn.norm.weight", {2}},
                                                        {"linear_attn.out_proj.weight", {hidden, 2}},
                                                    }
                                                  : std::vector<Tensor>{
                                                        {"self_attn.q_proj.weight", {8, hidden}},
                                                        {"self_attn.k_proj.weight", {2, hidden}},
                                                        {"self_attn.v_proj.weight", {2, hidden}},
                                                        {"self_attn.o_proj.weight", {hidden, 4}},
                                                        {"self_attn.q_norm.weight", {2}},
                                                        {"self_attn.k_norm.weight", {2}},
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
        // Nonzero projections exercise the whole attention and MLP path.
        const bool norm = item.first.find("norm.weight") != std::string::npos;
        for (size_t element = 0; element < elements; ++element)
        {
            data.push_back(norm ? 0 : 0x80);
            data.push_back(norm ? 0 : 0x3d);
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

static void run_dense_package(const std::filesystem::path& root, bool tied, bool quantized = false)
{
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
        check(mapping.value().at("lm_head.weight").bfloat16_values()[0] == mapping.value().at("token_embedding.weight").bfloat16_values()[0], "tied LM head differs");

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
    if (!runtime.info().gpu_infos.empty())
    {
        Option hybrid = opt;
        hybrid.hybrid_mode = HybridMode::HybridExperts;
        auto hybrid_model = runtime.load_model(root, hybrid);
        check(static_cast<bool>(hybrid_model), hybrid_model ? "" : hybrid_model.error().message);
        const CompiledModel& compiled = model_compiled(*hybrid_model.value());
        size_t vulkan_ffn_count = 0;
        for (const ExecutionNode& node : compiled.graph.nodes)
        {
            if (node.type != ExecutionNodeType::DenseFfn)
                continue;
            check(node.backend == ExecutionBackend::Vulkan, "hybrid dense FFN was pinned to CPU");
            const CompiledLayerPlan& layer = compiled.graph.layer_plans[node.layer_plan_index];
            check(support_vulkan_ffn(compiled.operators, layer.dense_ffn), "dense Vulkan plan lacks an executable FFN");
            check(layer.vulkan_device_index == compiled.opt.vulkan_device_index, "dense FFN has the wrong Vulkan device");
            ++vulkan_ffn_count;
        }
        check(vulkan_ffn_count == compiled.descriptor.layers.size(), "dense FFN missing from Vulkan schedule");
        auto hybrid_session = runtime.create_session(hybrid_model.value());
        check(static_cast<bool>(hybrid_session), "cannot create hybrid dense session");
        auto hybrid_prefill = hybrid_session.value()->prefill(ids);
        check(static_cast<bool>(hybrid_prefill), hybrid_prefill ? "" : hybrid_prefill.error().message);
        auto hybrid_decode = hybrid_session.value()->decode(3);
        check(static_cast<bool>(hybrid_decode), hybrid_decode ? "" : hybrid_decode.error().message);
        check(hybrid_decode.value().logits.size() == decode.value().logits.size(), "hybrid dense logits size mismatch");
        for (size_t index = 0; index < decode.value().logits.size(); ++index)
            check(std::abs(hybrid_decode.value().logits[index] - decode.value().logits[index]) < 1e-3f, "hybrid dense decode differs from CPU");
        check(hybrid_session.value()->statistics().expert_assignments == 0, "hybrid dense FFN recorded routes");
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

    write_weights(root, tied, true, quantized);
    auto missing = runtime.load_model(root, opt);
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
        ncnn::moe::run_dense_package(root, true);
        ncnn::moe::run_dense_package(root, false);
        ncnn::moe::run_dense_package(root, false, true);
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
