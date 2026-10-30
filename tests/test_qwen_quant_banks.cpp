#include "models/modeladapter_qwen4exp.h"
#include "graph/memoryplan.h"
#include "kernels/qnk.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static constexpr uint32_t hidden_size = 256;
static constexpr uint32_t intermediate_size = 256;
static constexpr uint32_t expert_count = 2;
static constexpr std::array<DType, 7> expert_dtypes = {
    DType::Q2K,
    DType::Q3K,
    DType::Q4K,
    DType::Q5K,
    DType::Q6K,
    DType::Q8K,
    DType::BFloat16,
};

static void bank_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

class BankTestDirectory
{
public:
    BankTestDirectory()
    {
        static uint32_t sequence = 0;
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path()
               / ("ncnn_moe_qwen_quant_banks_" + std::to_string(timestamp) + "_" + std::to_string(sequence++));
        bank_check(std::filesystem::create_directory(root), "create bank test directory");
    }

    ~BankTestDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    std::filesystem::path root;
};

class BankTestArchive
{
public:
    void add(const std::string& name, const char* dtype, const std::vector<uint32_t>& shape,
             uint64_t size, uint8_t value = 0)
    {
        if (data.size() != 0)
            header << ',';
        header << '"' << name << "\":{\"dtype\":\"" << dtype << "\",\"shape\":[";
        for (size_t dimension = 0; dimension < shape.size(); ++dimension)
        {
            if (dimension != 0)
                header << ',';
            header << shape[dimension];
        }
        header << "],\"data_offsets\":[" << data.size() << ',' << data.size() + size << "]}";
        data.insert(data.end(), static_cast<size_t>(size), value);
    }

    void add_bfloat16(const std::string& name, std::vector<uint32_t> shape)
    {
        uint64_t elements = 1;
        for (uint32_t dimension : shape)
            elements *= dimension;
        add(name, "BF16", shape, elements * 2);
    }

    void add_bank(const std::string& name, DType dtype, uint32_t rows, uint32_t columns)
    {
        if (dtype == DType::BFloat16)
        {
            add_bfloat16(name, {expert_count, rows, columns});
            return;
        }
        const uint64_t expert_size = qnk_storage_bytes(dtype, rows, columns);
        add(name, "U8", {expert_count, rows, columns / qnk_block_elements, static_cast<uint32_t>(qnk_block_bytes(dtype))},
            expert_size * expert_count);
        for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
        {
            std::fill_n(data.end() - static_cast<ptrdiff_t>((expert_count - expert_id) * expert_size),
                        static_cast<size_t>(expert_size), static_cast<uint8_t>(expert_id + 1));
        }
    }

    void write(const std::filesystem::path& root, const char* name = "model.safetensors") const
    {
        std::string encoded = '{' + header.str() + '}';
        encoded.append((8 - encoded.size() % 8) % 8, ' ');
        std::ofstream stream(root / name, std::ios::binary);
        const uint64_t header_size = encoded.size();
        stream.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        stream.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        bank_check(static_cast<bool>(stream), "write bank test archive");
    }

private:
    std::ostringstream header;
    std::vector<uint8_t> data;
};

static ModelPackage bank_test_package(const std::filesystem::path& root)
{
    ModelPackage package;
    package.root = root;
    package.manifest.model_type = "qwen4_exp";
    package.manifest.raw_json = R"({"text_config":{
        "attention_bias":false,"dtype":"bfloat16","eos_token_id":7,
        "hc_count":4,"hc_lowrank":2,"head_dim":8,"heads_per_ngram":1,
        "hidden_act":"silu","hidden_size":256,"indexer_budget":16,
        "indexer_compress_ratio":4,"indexer_head_dim":4,"indexer_kv_heads":1,
        "indexer_n_heads":2,"layer_types":["linear_attention","linear_attention",
        "linear_attention","linear_attention","linear_attention","linear_attention","full_attention"],
        "linear_conv_kernel_dim":2,"linear_key_head_dim":4,"linear_num_key_heads":1,
        "linear_num_value_heads":2,"linear_value_head_dim":4,"mamba_ssm_dtype":"float32",
        "max_position_embeddings":128,"make_ngram_vocab_size_divisible_by":4,
        "moe_intermediate_size":256,"ngram_size":3,"ngram_vocab_size_base":2,
        "num_attention_heads":2,"num_experts":2,"num_experts_per_tok":1,
        "num_hidden_layers":7,"num_key_value_heads":1,"output_gate_type":"sigmoid",
        "ple_conv_kernel_size":2,"ple_embed_dim":256,"ple_layer_ids":[1],
        "rms_norm_eps":0.000001,"rope_parameters":{"partial_rotary_factor":0.5,"rope_theta":10000.0},
        "shared_expert_intermediate_size":256,"split_ngram_parts":2,"vocab_size":8
    }})";
    return package;
}

static void add_residual(BankTestArchive& archive, const std::string& prefix, bool injection)
{
    archive.add_bfloat16(prefix + "hc_norm.weight", {hidden_size * 4});
    archive.add_bfloat16(prefix + "input_mix_weight_down.weight", {2, hidden_size * 4});
    archive.add_bfloat16(prefix + "input_mix_weight_up.weight", {hidden_size * 4, 2});
    if (injection)
        archive.add_bfloat16(prefix + "block_inject_weight.weight", {4, hidden_size * 4});
}

static void write_mixed_banks(const std::filesystem::path& root)
{
    BankTestArchive archive;
    archive.add_bfloat16("model.language_model.embed_tokens.weight", {8, hidden_size});
    archive.add_bfloat16("lm_head.weight", {8, hidden_size});
    add_residual(archive, "model.language_model.hyper_connection_mixer.", false);
    for (uint32_t layer = 0; layer < expert_dtypes.size(); ++layer)
    {
        const std::string source = "model.language_model.layers." + std::to_string(layer) + '.';
        add_residual(archive, source + "attn_hyper_connection.", true);
        add_residual(archive, source + "mlp_hyper_connection.", true);
        archive.add_bfloat16(source + "mlp.gate.weight", {expert_count, hidden_size});
        archive.add_bfloat16(source + "mlp.shared_expert.gate_proj.weight", {intermediate_size, hidden_size});
        archive.add_bfloat16(source + "mlp.shared_expert.up_proj.weight", {intermediate_size, hidden_size});
        archive.add_bfloat16(source + "mlp.shared_expert.down_proj.weight", {hidden_size, intermediate_size});
        archive.add_bfloat16(source + "mlp.shared_expert_gate.weight", {1, hidden_size});
        archive.add_bank(source + "mlp.experts.gate_up_proj", expert_dtypes[layer], intermediate_size * 2, hidden_size);
        archive.add_bank(source + "mlp.experts.down_proj", expert_dtypes[layer], hidden_size, intermediate_size);
        if (layer + 1 < expert_dtypes.size())
        {
            archive.add_bfloat16(source + "linear_attn.in_proj_qkv.weight", {16, hidden_size});
            archive.add_bfloat16(source + "linear_attn.in_proj_z.weight", {8, hidden_size});
            archive.add_bfloat16(source + "linear_attn.in_proj_b.weight", {2, hidden_size});
            archive.add_bfloat16(source + "linear_attn.in_proj_a.weight", {2, hidden_size});
            archive.add_bfloat16(source + "linear_attn.conv1d.weight", {16, 1, 2});
            archive.add_bfloat16(source + "linear_attn.dt_bias", {2});
            archive.add_bfloat16(source + "linear_attn.A_log", {2});
            archive.add_bfloat16(source + "linear_attn.norm.weight", {4});
            archive.add_bfloat16(source + "linear_attn.out_proj.weight", {hidden_size, 8});
        }
        else
        {
            archive.add_bfloat16(source + "self_attn.q_proj.weight", {32, hidden_size});
            archive.add_bfloat16(source + "self_attn.k_proj.weight", {8, hidden_size});
            archive.add_bfloat16(source + "self_attn.v_proj.weight", {8, hidden_size});
            archive.add_bfloat16(source + "self_attn.o_proj.weight", {hidden_size, 16});
            archive.add_bfloat16(source + "self_attn.q_norm.weight", {8});
            archive.add_bfloat16(source + "self_attn.k_norm.weight", {8});
            archive.add_bfloat16(source + "self_attn.indexer.index_qk_proj.weight", {12, hidden_size});
            archive.add_bfloat16(source + "self_attn.indexer.q_layernorm.weight", {4});
            archive.add_bfloat16(source + "self_attn.indexer.k_layernorm.weight", {4});
        }
        if (layer == 0)
        {
            const std::string ple = source + "ple.";
            archive.add_bfloat16(ple + "key_proj.weight", {hidden_size * 4, hidden_size});
            archive.add_bfloat16(ple + "value_proj.weight", {hidden_size, hidden_size});
            for (const char* name : {"norm_key.weight", "norm_query.weight", "norm_conv.weight"})
                archive.add_bfloat16(ple + name, {hidden_size * 4});
            archive.add_bfloat16(ple + "conv1d.weight", {hidden_size * 4, 1, 2});
            archive.add(ple + "ple_embedding.layer_multipliers", "I64", {3}, 24);
            archive.add(ple + "ple_embedding.ngram_heads_vocab_sizes", "I64", {2}, 16);
            archive.add(ple + "ple_embedding.ngram_heads_offsets", "I64", {2}, 16);
            for (uint32_t shard = 0; shard < 2; ++shard)
                archive.add_bfloat16(ple + "ple_embedding.ngram_embedding.shard_" + std::to_string(shard) + ".weight", {4, hidden_size / 2});
        }
    }
    archive.write(root);
}

static void test_mixed_banks()
{
    BankTestDirectory directory;
    write_mixed_banks(directory.root);
    const ModelPackage package = bank_test_package(directory.root);
    Qwen4ExpModelAdapter adapter;
    auto parsed = adapter.parse_model(package);
    bank_check(static_cast<bool>(parsed), parsed ? "" : parsed.error().message);
    uint64_t expected_bytes = 0;
    for (size_t layer = 0; layer < expert_dtypes.size(); ++layer)
    {
        bank_check(parsed.value().layers[layer].moe.expert_weight_dtype == expert_dtypes[layer], "detect per-layer Expert dtype");
        expected_bytes += expert_count * (is_qnk_dtype(expert_dtypes[layer]) ? qnk_storage_bytes(expert_dtypes[layer], intermediate_size * 2, hidden_size) + qnk_storage_bytes(expert_dtypes[layer], hidden_size, intermediate_size) : static_cast<uint64_t>(hidden_size) * intermediate_size * 3 * 2);
    }
    auto planned = plan_model_memory(parsed.value(), Option{}, UINT64_C(8) * 1024 * 1024 * 1024);
    bank_check(static_cast<bool>(planned), planned ? "" : planned.error().message);
    bank_check(planned.value().estimated_expert_size == expected_bytes, "sum mixed Expert storage bytes");
    auto mapping = adapter.map_weights(package, parsed.value());
    bank_check(static_cast<bool>(mapping), mapping ? "" : mapping.error().message);
    for (uint32_t layer = 0; layer < expert_dtypes.size(); ++layer)
    {
        for (uint32_t expert = 0; expert < expert_count; ++expert)
        {
            const std::string prefix = "layers." + std::to_string(layer) + ".experts." + std::to_string(expert) + '.';
            for (const char* projection : {"gate_up.weight", "down.weight"})
            {
                const TensorData& tensor = mapping.value().at(prefix + projection);
                bank_check(tensor.dtype == expert_dtypes[layer], "mapped Expert dtype");
                if (is_qnk_dtype(tensor.dtype))
                {
                    const auto bytes = tensor.qnk_values();
                    bank_check(bytes.size() == qnk_storage_bytes(tensor.dtype, tensor.shape[0], tensor.shape[1]), "mapped Expert byte count");
                    bank_check(std::all_of(bytes.begin(), bytes.end(), [expert](uint8_t value) { return value == expert + 1; }), "lossless Expert slice");
                }
            }
        }
        const TensorData& shared = mapping.value().at("layers." + std::to_string(layer) + ".shared_expert.gate.weight");
        bank_check(shared.dtype == DType::BFloat16, "preserve dense shared Expert dtype");
    }
}

static void test_invalid_banks()
{
    const std::string source = "model.language_model.layers.0.mlp.experts.";
    for (uint32_t invalid_case = 0; invalid_case < 4; ++invalid_case)
    {
        BankTestDirectory directory;
        BankTestArchive archive;
        if (invalid_case == 0)
            archive.add(source + "gate_up_proj", "U8", {1}, 1);
        else
            archive.add_bank(source + "gate_up_proj", DType::Q4K, intermediate_size * 2, hidden_size);
        if (invalid_case == 0 || invalid_case == 1)
            archive.add_bank(source + "down_proj", DType::Q4K, hidden_size, intermediate_size);
        else if (invalid_case == 2)
            archive.add_bank(source + "down_proj", DType::Q6K, hidden_size, intermediate_size);
        else
            archive.add_bank(source + "down_proj", DType::BFloat16, hidden_size, intermediate_size);
        if (invalid_case == 1)
        {
            BankTestArchive missing_down;
            missing_down.add_bank(source + "gate_up_proj", DType::Q4K, intermediate_size * 2, hidden_size);
            missing_down.write(directory.root);
        }
        else
        {
            archive.write(directory.root);
        }
        auto parsed = Qwen4ExpModelAdapter{}.parse_model(bank_test_package(directory.root));
        bank_check(!parsed, "reject malformed or mismatched Qn_K bank pair");
        bank_check(parsed.error().code == ErrorCode::InvalidModel, "invalid Qn_K bank error code");
    }
}

static void test_mixed_dense_moe_memory()
{
    MoeModelDescriptor descriptor;
    descriptor.model_type = "mixed-dense-moe";
    descriptor.vocabulary_size = 8;
    descriptor.hidden_size = hidden_size;
    descriptor.intermediate_size = intermediate_size;
    descriptor.expert_count = expert_count;
    descriptor.experts_per_token = 1;
    descriptor.layers.resize(2);
    descriptor.layers[0].dense.intermediate_size = intermediate_size;
    MoeDescriptor& moe = descriptor.layers[1].moe;
    moe.expert_count = expert_count;
    moe.top_k = 1;
    moe.intermediate_size = intermediate_size;
    moe.expert_weight_dtype = DType::BFloat16;
    moe.flags = MoeDescriptorFileBackedExperts;
    Option options;
    options.expert_memory_mode = ExpertMemoryMode::OnDemand;
    options.expert_cache_size = 1024 * 1024;
    auto plan = plan_model_memory(descriptor, options, UINT64_C(8) * 1024 * 1024 * 1024);
    bank_check(static_cast<bool>(plan), plan ? "" : plan.error().message);
    bank_check(plan.value().selected_mode == ExpertMemoryMode::OnDemand, "mixed dense/MoE keeps on-demand storage");
    bank_check(plan.value().estimated_expert_size == static_cast<uint64_t>(expert_count) * hidden_size * intermediate_size * 3 * 2,
               "mixed dense/MoE counts only routed Expert weights");
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::test_mixed_banks();
        ncnn::moe::test_invalid_banks();
        ncnn::moe::test_mixed_dense_moe_memory();
        std::cout << "Qwen quantized bank tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
