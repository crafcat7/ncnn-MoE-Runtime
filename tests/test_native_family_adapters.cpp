#include "models/modeladapter_gptoss.h"
#include "models/modeladapter_deepseekv4.h"
#include "kernels/qnk.h"

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void family_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

class FamilyArchive
{
public:
    FamilyArchive()
    {
        static uint32_t sequence = 0;
        root = std::filesystem::temp_directory_path()
               / ("ncnn_native_family_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                  + "_" + std::to_string(sequence++));
        family_check(std::filesystem::create_directory(root), "create family archive");
    }

    ~FamilyArchive()
    {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
    }

    void add(const std::string& name, const char* dtype, const std::vector<uint32_t>& shape, const void* values, size_t size)
    {
        if (count++)
            header << ',';
        header << '"' << name << "\":{\"dtype\":\"" << dtype << "\",\"shape\":[";
        for (size_t dimension = 0; dimension < shape.size(); ++dimension)
            header << (dimension ? "," : "") << shape[dimension];
        header << "],\"data_offsets\":[" << data.size() << ',' << data.size() + size << "]}";
        const size_t offset = data.size();
        data.resize(offset + size);
        std::memcpy(data.data() + offset, values, size);
    }

    void plain(const std::string& name, bool float32 = false, uint32_t elements = 1)
    {
        if (float32)
        {
            std::vector<float> values(elements, 1.f / 7.f);
            add(name, "F32", {elements}, values.data(), values.size() * sizeof(float));
        }
        else
        {
            std::vector<uint16_t> values(elements);
            for (uint32_t index = 0; index < elements; ++index)
                values[index] = static_cast<uint16_t>(0x3e00 + index % 127);
            add(name, "BF16", {elements}, values.data(), values.size() * sizeof(uint16_t));
        }
    }

    void matrix(const std::string& name, const std::string& encoding)
    {
        constexpr uint32_t rows = 8;
        constexpr uint32_t columns = 256;
        if (encoding == "qnk")
        {
            std::vector<uint8_t> blocks(static_cast<size_t>(qnk_storage_bytes(DType::Q8K, rows, columns)), 0);
            for (uint32_t row = 0; row < rows; ++row)
            {
                const float scale = 0.03125f;
                std::memcpy(blocks.data() + row * 292, &scale, sizeof(scale));
                for (uint32_t column = 0; column < columns; ++column)
                    blocks[row * 292 + 4 + column] = static_cast<uint8_t>(static_cast<int>(column % 7) - 3);
            }
            const int64_t shape[3] = {8, rows, columns};
            add(name, "U8", {rows, 292}, blocks.data(), blocks.size());
            add(name + ".__qnk_shape", "I64", {3}, shape, sizeof(shape));
            return;
        }
        if (encoding == "bf16")
        {
            std::vector<uint16_t> values(rows * columns, 0x3d00);
            add(name, "BF16", {rows, columns}, values.data(), values.size() * sizeof(uint16_t));
            return;
        }
        std::vector<float> scales(rows * 2, 0.03125f);
        if (encoding == "int4")
        {
            std::vector<uint8_t> values(rows * columns / 2, 0x98);
            std::vector<uint8_t> zeros(rows * 2, 8);
            std::vector<int64_t> indices(columns);
            for (uint32_t column = 0; column < columns; ++column)
                indices[column] = (column / 16) % 2;
            const int64_t shape[3] = {rows, columns, 128};
            add(name, "U8", {rows, columns / 2}, values.data(), values.size());
            add(name + ".__int4_shape", "I64", {3}, shape, sizeof(shape));
            add(name + ".__int4_scales", "F32", {rows, 2}, scales.data(), scales.size() * sizeof(float));
            add(name + ".__int4_zeros", "U8", {rows, 2}, zeros.data(), zeros.size());
            add(name + ".__int4_g_idx", "I64", {columns}, indices.data(), indices.size() * sizeof(int64_t));
            return;
        }
        const bool awq = encoding == "awq";
        constexpr std::array<uint32_t, 8> order = {0, 4, 1, 5, 2, 6, 3, 7};
        std::vector<uint32_t> weights(rows * columns / 8, 0);
        std::vector<uint32_t> zeros(2, 0);
        for (uint32_t row = 0; row < rows; ++row)
        {
            for (uint32_t column = 0; column < columns; ++column)
            {
                const uint32_t index = awq ? column : column / 8 * rows + row;
                const uint32_t shift = (awq ? order[row] : column % 8) * 4;
                weights[index] |= (8 + column % 2) << shift;
            }
            for (uint32_t group = 0; group < 2; ++group)
                zeros[group] |= (awq ? 8u : 7u) << ((awq ? order[row] : row) * 4);
        }
        const std::string prefix = name.substr(0, name.size() - 6);
        add(prefix + "qweight", "I32", awq ? std::vector<uint32_t>{columns, 1} : std::vector<uint32_t>{columns / 8, rows},
            weights.data(), weights.size() * sizeof(uint32_t));
        add(prefix + "qzeros", "I32", {2, 1}, zeros.data(), zeros.size() * sizeof(uint32_t));
        add(prefix + "scales", "F32", {2, rows}, scales.data(), scales.size() * sizeof(float));
        if (!awq)
        {
            std::vector<uint32_t> indices(columns);
            for (uint32_t column = 0; column < columns; ++column)
                indices[column] = (column / 16) % 2;
            add(prefix + "g_idx", "I32", {columns}, indices.data(), indices.size() * sizeof(uint32_t));
        }
    }

    ModelPackage finish(const std::string& type, const std::string& config)
    {
        std::string encoded = '{' + header.str() + '}';
        encoded.append((8 - encoded.size() % 8) % 8, ' ');
        const uint64_t size = encoded.size();
        std::ofstream stream(root / "model.safetensors", std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        family_check(static_cast<bool>(stream), "write family archive");
        std::ofstream(root / "config.json") << config;
        ModelPackage package;
        package.root = root;
        package.manifest.model_type = type;
        package.manifest.raw_json = config;
        return package;
    }

    std::filesystem::path root;

private:
    uint32_t count = 0;
    std::ostringstream header;
    std::vector<uint8_t> data;
};

static std::string quantization_config(const std::string& encoding)
{
    return encoding == "awq" || encoding == "gptq"
               ? ",\"quantization_config\":{\"quant_method\":\"" + encoding + "\",\"bits\":4,\"group_size\":128,\"desc_act\":true}"
               : "";
}

static void check_native_payload(const TensorData& tensor, const std::string& encoding)
{
    family_check(tensor.bfloat16_data.empty() && tensor.float32_data.empty(), "native projection expanded to floating storage");
    if (encoding == "qnk")
    {
        family_check(tensor.dtype == DType::Q8K && tensor.qnk_values().size() == 8 * 292, "Qn_K matrix encoding changed");
        family_check(tensor.qnk_values()[5] == static_cast<uint8_t>(-2), "Qn_K packed payload changed");
    }
    else
    {
        family_check(tensor.dtype == DType::Int4 && tensor.int4_values().size() == 8 * 128, "INT4 matrix encoding changed");
        family_check(tensor.int4_values()[0] == 0x98 && tensor.quantization_scales[0] == 0.03125f
                         && tensor.quantization_zero_points[0] == 8,
                     "INT4 code, scale or zero point changed");
        if (encoding != "awq")
            family_check(tensor.quantization_group_indices.size() == 256
                             && tensor.quantization_group_indices[16] == 1
                             && tensor.quantization_group_indices[32] == 0,
                         "INT4 group-index permutation changed");
    }
}

static void test_gpt_oss(const std::string& encoding, bool fused)
{
    FamilyArchive file;
    for (const char* name : {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"})
        file.plain(name);
    const std::string source = "model.layers.0.";
    for (const char* suffix : {"input_layernorm.weight", "post_attention_layernorm.weight", "mlp.router.weight", "mlp.router.bias", "self_attn.sinks"})
        file.plain(source + suffix);
    for (const char* projection : {"q_proj", "k_proj", "v_proj", "o_proj"})
        file.matrix(source + "self_attn." + projection + ".weight", encoding);
    const std::string experts = source + "mlp.experts.";
    std::vector<uint16_t> bias(256);
    for (uint32_t index = 0; index < bias.size(); ++index)
        bias[index] = static_cast<uint16_t>(0x3e00 + index % 127);
    file.add(experts + "gate_up_proj_bias", "BF16", {1, 16}, bias.data(), 16 * sizeof(uint16_t));
    file.add(experts + "down_proj_bias", "BF16", {1, 256}, bias.data(), bias.size() * sizeof(uint16_t));
    for (const std::string& projection : fused ? std::vector<std::string>{"gate_up_proj", "down_proj"}
                                               : std::vector<std::string>{"gate_proj", "up_proj", "down_proj"})
        file.matrix(experts + "0." + projection + ".weight", encoding);
    const std::string config = R"({"model_type":"gpt_oss","vocab_size":2,"hidden_size":256,"intermediate_size":8,"num_hidden_layers":1,"num_local_experts":1,"experts_per_token":1,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":64,"sliding_window":4,"initial_context_length":16,"max_position_embeddings":32,"attention_bias":false)"
                               + quantization_config(encoding) + "}";
    auto package = file.finish("gpt_oss", config);
    GptOssModelAdapter adapter;
    auto descriptor = adapter.parse_model(package);
    family_check(static_cast<bool>(descriptor), descriptor ? "" : descriptor.error().message);
    family_check(descriptor.value().layers[0].moe.layout == (fused ? ExpertLayout::InterleavedGateUpDown : ExpertLayout::GateUpDown), "GPT-OSS layout remained hardcoded");
    auto mapped = adapter.map_weights(package, descriptor.value());
    family_check(static_cast<bool>(mapped), mapped ? "" : mapped.error().message);
    const auto& mapping = mapped.value();
    check_native_payload(mapping.at("layers.0.attention.query.weight"), encoding);
    check_native_payload(mapping.at(fused ? "layers.0.experts.0.gate_up.weight" : "layers.0.experts.0.gate.weight"), encoding);
    family_check(mapping.at("layers.0.experts.0.down.bias").bfloat16_values().size() == 256, "GPT-OSS down bias lost");
    if (!fused)
    {
        const auto gate = mapping.at("layers.0.experts.0.gate.bias").bfloat16_values();
        const auto up = mapping.at("layers.0.experts.0.up.bias").bfloat16_values();
        family_check(gate.size() == 8 && up.size() == 8 && gate[3] == 0x3e06 && up[3] == 0x3e07,
                     "GPT-OSS interleaved bias did not split losslessly");
        family_check(!mapping.contains("layers.0.experts.0.gate_up.weight"), "GPT-OSS separate weights copied into fused storage");
    }
}

static void deepseek_layer(FamilyArchive& file, const std::string& source, const std::string& encoding, bool fused)
{
    for (const char* suffix : {"hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale", "attn.attn_sink", "ffn.gate.bias"})
        file.plain(source + suffix, true);
    for (const char* suffix : {"attn_norm.weight", "attn.q_norm.weight", "attn.kv_norm.weight", "ffn_norm.weight", "ffn.gate.weight"})
        file.plain(source + suffix);
    for (const char* suffix : {"attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b", "ffn.shared_experts.w1", "ffn.shared_experts.w3", "ffn.shared_experts.w2"})
        file.matrix(source + suffix + ".weight", encoding);
    for (const std::string& projection : fused ? std::vector<std::string>{"gate_up_proj", "w2"} : std::vector<std::string>{"w1", "w3", "w2"})
        file.matrix(source + "ffn.experts.0." + projection + ".weight", encoding);
}

static void test_deepseek(const std::string& encoding, bool fused, bool dspark)
{
    FamilyArchive file;
    for (const char* name : {"embed.weight", "norm.weight", "head.weight"})
        file.plain(name);
    for (const char* name : {"hc_head_fn", "hc_head_base", "hc_head_scale"})
        file.plain(name, true);
    deepseek_layer(file, "layers.0.", encoding, fused);
    if (dspark)
    {
        deepseek_layer(file, "mtp.0.", encoding, fused);
        file.matrix("mtp.0.main_proj.weight", encoding);
        for (const char* suffix : {"main_norm.weight", "norm.weight", "markov_head.markov_w1.weight", "markov_head.markov_w2.weight", "confidence_head.proj.weight"})
            file.plain(std::string("mtp.0.") + suffix);
        for (const char* suffix : {"hc_head_fn", "hc_head_base", "hc_head_scale"})
            file.plain(std::string("mtp.0.") + suffix, true);
    }
    const std::string config = R"({"model_type":"deepseek_v4","vocab_size":2,"hidden_size":256,"moe_intermediate_size":8,"num_hidden_layers":1,"n_routed_experts":1,"num_experts_per_tok":1,"n_shared_experts":1,"num_attention_heads":4,"num_key_value_heads":1,"head_dim":64,"q_lora_rank":256,"qk_rope_head_dim":16,"o_groups":1,"o_lora_rank":256,"sliding_window":4,"max_position_embeddings":32,"original_max_position_embeddings":16,"num_hash_layers":0,"index_n_heads":1,"index_head_dim":64,"index_topk":1,"hc_mult":2,"hc_sinkhorn_iters":4,"compress_ratios":[0],"scoring_func":"sqrtsoftplus")"
                               + quantization_config(encoding)
                               + (dspark ? ",\"dspark_target_layer_ids\":[0],\"dspark_block_size\":2,\"dspark_noise_token_id\":0,\"dspark_markov_rank\":8" : "") + "}";
    auto package = file.finish("deepseek_v4", config);
    DeepSeekV4ModelAdapter adapter;
    auto descriptor = adapter.parse_model(package);
    family_check(static_cast<bool>(descriptor), descriptor ? "" : descriptor.error().message);
    family_check(descriptor.value().layers[0].moe.layout == (fused ? ExpertLayout::InterleavedGateUpDown : ExpertLayout::GateUpDown), "DeepSeek layout remained hardcoded");
    family_check(descriptor.value().speculative_layer_count == (dspark ? 1u : 0u), "DSpark descriptor lost");
    auto mapped = adapter.map_weights(package, descriptor.value());
    family_check(static_cast<bool>(mapped), mapped ? "" : mapped.error().message);
    const auto& mapping = mapped.value();
    check_native_payload(mapping.at("layers.0.attention.query_a.weight"), encoding);
    check_native_payload(mapping.at("layers.0.shared_expert.gate.weight"), encoding);
    check_native_payload(mapping.at(fused ? "layers.0.experts.0.gate_up.weight" : "layers.0.experts.0.gate.weight"), encoding);
    for (const char* name : {"layers.0.hyper.attention.function", "layers.0.hyper.ffn.scale", "layers.0.attention.sinks", "layers.0.router.selection_bias", "hyper.head.function"})
    {
        const TensorData& parameter = mapping.at(name);
        family_check(parameter.dtype == DType::Float32 && parameter.float32_values()[0] == 1.f / 7.f,
                     std::string("DeepSeek FP32 parameter was rounded in INT4 package: ") + name);
    }
    if (dspark)
    {
        check_native_payload(mapping.at("speculative.layers.0.attention.query_a.weight"), encoding);
        check_native_payload(mapping.at("speculative.main_projection.weight"), encoding);
    }
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        for (const std::string encoding : {"int4", "qnk", "awq", "gptq"})
        {
            for (bool fused : {false, true})
            {
                ncnn::moe::test_gpt_oss(encoding, fused);
                ncnn::moe::test_deepseek(encoding, fused, false);
            }
            ncnn::moe::test_deepseek(encoding, false, true);
        }
        std::cout << "Native family adapter tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Native family adapter test failed: " << error.what() << '\n';
        return 1;
    }
}
