#include "ggufmodel.h"
#include "ggufmodel_gptoss.h"
#include "ggufmodel_qwen.h"
#include "json.h"

#include <cmath>
#include <charconv>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>

namespace ncnn {
namespace moe {

static bool gguf_extension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".gguf";
}

bool has_gguf_source(const std::filesystem::path& path)
{
    if (gguf_extension(path))
        return true;
    std::error_code ec;
    if (!std::filesystem::is_directory(path, ec))
        return false;
    bool found = false;
    for (const auto& entry : std::filesystem::directory_iterator(path, ec))
    {
        if (ec)
            return false;
        if (entry.is_regular_file(ec) && entry.path().extension() == ".safetensors")
            return false;
        if (entry.is_regular_file(ec) && gguf_extension(entry.path()))
            found = true;
    }
    return found;
}

static Result<std::string> metadata_json(const GgufMetadataValue& value)
{
    if (const auto* string = value.string())
        return json_escape(*string);
    if (const auto* array = value.array())
    {
        std::string result = "[";
        for (const auto& item : *array)
        {
            auto encoded = metadata_json(item);
            if (!encoded)
                return encoded.error();
            if (result.size() != 1)
                result += ',';
            result += encoded.value();
        }
        return result + ']';
    }
    if (const auto number = value.number())
    {
        if (!std::isfinite(*number))
            return Error{ErrorCode::InvalidModel, "non-finite GGUF model metadata"};
        std::ostringstream stream;
        stream << std::setprecision(17) << *number;
        return stream.str();
    }
    return Error{ErrorCode::InvalidModel, "invalid GGUF model metadata type"};
}

static Result<std::string> deepseek_gguf_manifest(const GgufArchive& archive)
{
    const auto* architecture = archive.find_metadata("general.architecture");
    if (!architecture || !architecture->string() || *architecture->string() != "deepseek4")
        return Error{ErrorCode::UnsupportedModel, "direct GGUF loading currently requires deepseek4 architecture"};
    const auto* tokens = archive.find_metadata("tokenizer.ggml.tokens");
    if (!tokens || !tokens->array() || tokens->array()->empty())
        return Error{ErrorCode::InvalidModel, "GGUF tokenizer vocabulary is missing"};
    const auto* blocks = archive.find_metadata("deepseek4.block_count");
    const auto total_layers = blocks ? blocks->unsigned_integer() : std::nullopt;
    const auto* mtp_layers = archive.find_metadata("deepseek4.nextn_predict_layers");
    const auto draft_layers = mtp_layers ? mtp_layers->unsigned_integer() : std::optional<uint64_t>(0);
    if (!total_layers || !draft_layers || *draft_layers >= *total_layers || *total_layers > archive.tensor_infos().size())
        return Error{ErrorCode::InvalidModel, "invalid DeepSeek GGUF target/MTP layer counts"};
    const uint64_t target_layers = *total_layers - *draft_layers;
    std::string result = "{\"model_type\":\"deepseek_v4\",\"expert_dtype\":\"fp4\",\"scoring_func\":\"sqrtsoftplus\",\"vocab_size\":"
                         + std::to_string(tokens->array()->size());
    const std::pair<const char*, const char*> fields[] = {
        {"block_count", "num_hidden_layers"},
        {"context_length", "max_position_embeddings"},
        {"embedding_length", "hidden_size"},
        {"expert_count", "n_routed_experts"},
        {"expert_used_count", "num_experts_per_tok"},
        {"expert_shared_count", "n_shared_experts"},
        {"expert_feed_forward_length", "moe_intermediate_size"},
        {"expert_weights_scale", "routed_scaling_factor"},
        {"attention.head_count", "num_attention_heads"},
        {"attention.head_count_kv", "num_key_value_heads"},
        {"attention.key_length", "head_dim"},
        {"attention.q_lora_rank", "q_lora_rank"},
        {"attention.output_group_count", "o_groups"},
        {"attention.output_lora_rank", "o_lora_rank"},
        {"attention.sliding_window", "sliding_window"},
        {"attention.indexer.head_count", "index_n_heads"},
        {"attention.indexer.key_length", "index_head_dim"},
        {"attention.indexer.top_k", "index_topk"},
        {"rope.dimension_count", "qk_rope_head_dim"},
        {"rope.freq_base", "rope_theta"},
        {"attention.compress_rope_freq_base", "compress_rope_theta"},
        {"attention.compress_ratios", "compress_ratios"},
        {"attention.layer_norm_rms_epsilon", "rms_norm_eps"},
        {"hyper_connection.count", "hc_mult"},
        {"hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters"},
        {"hyper_connection.epsilon", "hc_eps"},
        {"hash_layer_count", "num_hash_layers"}};
    for (const auto& [source, target] : fields)
    {
        const auto* value = archive.find_metadata(std::string("deepseek4.") + source);
        if (!value)
            return Error{ErrorCode::InvalidModel, std::string("missing GGUF model metadata: deepseek4.") + source};
        auto encoded = metadata_json(*value);
        if (!encoded)
            return encoded.error();
        if (std::string_view(source) == "block_count")
        {
            result += ",\"num_hidden_layers\":" + std::to_string(target_layers);
        }
        else
            result += ",\"" + std::string(target) + "\":" + encoded.value();
    }
    const auto* gating = archive.find_metadata("deepseek4.expert_gating_func");
    if (!gating || gating->unsigned_integer() != 4)
        return Error{ErrorCode::UnsupportedModel, "unsupported DeepSeek GGUF routing score function"};
    const auto* normalized = archive.find_metadata("deepseek4.expert_weights_norm");
    if (!normalized || normalized->boolean() != true)
        return Error{ErrorCode::UnsupportedModel, "DeepSeek GGUF requires normalized selected Expert weights"};
    const auto* clamp = archive.find_metadata("deepseek4.swiglu_clamp_exp");
    if (!clamp)
        return Error{ErrorCode::InvalidModel, "missing DeepSeek GGUF SwiGLU clamp"};
    std::optional<double> limit = clamp->number();
    if (const auto* values = clamp->array())
    {
        if (values->size() < target_layers)
            return Error{ErrorCode::InvalidModel, "DeepSeek GGUF SwiGLU clamp array is shorter than its target layers"};
        for (size_t i = 0; i < target_layers; ++i)
        {
            const auto number = (*values)[i].number();
            if (!number || (limit && *number != *limit))
                return Error{ErrorCode::UnsupportedModel, "nonuniform DeepSeek GGUF SwiGLU clamps are unsupported"};
            limit = number;
        }
    }
    if (!limit || !std::isfinite(*limit) || *limit <= 0)
        return Error{ErrorCode::InvalidModel, "invalid DeepSeek GGUF SwiGLU clamp"};
    if (const auto* shared_clamp = archive.find_metadata("deepseek4.swiglu_clamp_shexp"))
    {
        if (const auto* values = shared_clamp->array())
        {
            if (values->size() < target_layers)
                return Error{ErrorCode::InvalidModel, "DeepSeek GGUF shared clamp array is shorter than its target layers"};
            for (size_t i = 0; i < target_layers; ++i)
                if ((*values)[i].number() != limit)
                    return Error{ErrorCode::UnsupportedModel, "DeepSeek GGUF requires the same routed and shared Expert clamps"};
        }
        else if (shared_clamp->number() != limit)
            return Error{ErrorCode::UnsupportedModel, "DeepSeek GGUF requires the same routed and shared Expert clamps"};
    }
    const auto* scaling = archive.find_metadata("deepseek4.rope.scaling.type");
    if (!scaling || !scaling->string() || *scaling->string() != "yarn")
        return Error{ErrorCode::UnsupportedModel, "DeepSeek GGUF requires explicit YaRN scaling metadata"};
    result += ",\"swiglu_limit\":" + std::to_string(*limit) + ",\"rope_scaling\":{\"type\":\"yarn\"";
    for (const auto& [source, target] : std::initializer_list<std::pair<const char*, const char*>>{
             {"factor", "factor"},
             {"original_context_length", "original_max_position_embeddings"},
             {"yarn_beta_fast", "beta_fast"},
             {"yarn_beta_slow", "beta_slow"}})
    {
        const auto* value = archive.find_metadata(std::string("deepseek4.rope.scaling.") + source);
        if (!value)
            return Error{ErrorCode::InvalidModel, std::string("missing GGUF RoPE metadata: ") + source};
        auto encoded = metadata_json(*value);
        if (!encoded)
            return encoded.error();
        result += ",\"" + std::string(target) + "\":" + encoded.value();
    }
    return result + "}}";
}

static Result<std::string> deepseek_gguf_tensor_name(const std::string& name)
{
    static const std::unordered_map<std::string, std::string> roots = {
        {"token_embd.weight", "embed.weight"},
        {"output_norm.weight", "norm.weight"},
        {"output.weight", "head.weight"},
        {"output_hc_fn.weight", "hc_head_fn"},
        {"output_hc_base.weight", "hc_head_base"},
        {"output_hc_scale.weight", "hc_head_scale"}};
    if (const auto found = roots.find(name); found != roots.end())
        return found->second;
    static const std::unordered_map<std::string, std::string> fields = {
        {"hc_attn_fn.weight", "hc_attn_fn"},
        {"hc_attn_base.weight", "hc_attn_base"},
        {"hc_attn_scale.weight", "hc_attn_scale"},
        {"hc_ffn_fn.weight", "hc_ffn_fn"},
        {"hc_ffn_base.weight", "hc_ffn_base"},
        {"hc_ffn_scale.weight", "hc_ffn_scale"},
        {"attn_sinks.weight", "attn.attn_sink"},
        {"attn_q_a.weight", "attn.wq_a.weight"},
        {"attn_q_b.weight", "attn.wq_b.weight"},
        {"attn_q_a_norm.weight", "attn.q_norm.weight"},
        {"attn_kv.weight", "attn.wkv.weight"},
        {"attn_kv_norm.weight", "attn.kv_norm.weight"},
        {"attn_kv_a_norm.weight", "attn.kv_norm.weight"},
        {"attn_output_a.weight", "attn.wo_a.weight"},
        {"attn_output_b.weight", "attn.wo_b.weight"},
        {"attn_compressor_ape.weight", "attn.compressor.ape"},
        {"attn_compressor_kv.weight", "attn.compressor.wkv.weight"},
        {"attn_compressor_gate.weight", "attn.compressor.wgate.weight"},
        {"attn_compressor_norm.weight", "attn.compressor.norm.weight"},
        {"indexer.attn_q_b.weight", "attn.indexer.wq_b.weight"},
        {"indexer.proj.weight", "attn.indexer.weights_proj.weight"},
        {"indexer_compressor_ape.weight", "attn.indexer.compressor.ape"},
        {"indexer_compressor_kv.weight", "attn.indexer.compressor.wkv.weight"},
        {"indexer_compressor_gate.weight", "attn.indexer.compressor.wgate.weight"},
        {"indexer_compressor_norm.weight", "attn.indexer.compressor.norm.weight"},
        {"attn_norm.weight", "attn_norm.weight"},
        {"ffn_norm.weight", "ffn_norm.weight"},
        {"ffn_gate_inp.weight", "ffn.gate.weight"},
        {"exp_probs_b.bias", "ffn.gate.bias"},
        {"exp_probs_b_vl.bias", "ffn.gate.bias_vl"},
        {"ffn_gate_tid2eid.weight", "ffn.gate.tid2eid"},
        {"ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight"},
        {"ffn_up_shexp.weight", "ffn.shared_experts.w3.weight"},
        {"ffn_down_shexp.weight", "ffn.shared_experts.w2.weight"}};
    static const std::regex block("blk\\.([0-9]+)\\.(.+)");
    std::smatch match;
    if (std::regex_match(name, match, block))
    {
        if (const auto found = fields.find(match[2].str()); found != fields.end())
            return "layers." + match[1].str() + "." + found->second;
    }
    return Error{ErrorCode::UnsupportedModel, "unsupported DeepSeek GGUF tensor: " + name};
}

static std::string gguf_storage_dtype(uint32_t type)
{
    switch (type)
    {
    case 0: return "F32";
    case 1: return "F16";
    case 26: return "I32";
    case 30: return "BF16";
    case 39: return "MXFP4";
    default: return "U8";
    }
}

static Result<std::unordered_map<std::string, SafetensorInfo>> deepseek_gguf_tensors(const GgufArchive& archive)
{
    constexpr size_t maximum_tensor_aliases = 1024 * 1024;
    auto manifest = deepseek_gguf_manifest(archive);
    if (!manifest)
        return manifest.error();
    const auto* layers = archive.find_metadata("deepseek4.block_count");
    const auto* experts = archive.find_metadata("deepseek4.expert_count");
    const auto layer_count = layers ? layers->unsigned_integer() : std::nullopt;
    const auto expert_count = experts ? experts->unsigned_integer() : std::nullopt;
    if (!layer_count || !expert_count || *expert_count == 0 || *expert_count > UINT32_MAX)
        return Error{ErrorCode::InvalidModel, "invalid GGUF layer or Expert count"};
    const auto* mtp = archive.find_metadata("deepseek4.nextn_predict_layers");
    const uint64_t target_layers = *layer_count - (mtp ? mtp->unsigned_integer().value_or(0) : 0);
    std::unordered_map<std::string, SafetensorInfo> result;
    static const std::regex expert_bank("blk\\.([0-9]+)\\.ffn_(gate|up|down)_exps\\.weight");
    static const std::regex block("blk\\.([0-9]+)\\..+");
    for (const auto& [name, info] : archive.tensor_infos())
    {
        std::smatch match;
        if (std::regex_match(name, match, block))
        {
            const std::string number = match[1].str();
            uint64_t layer = 0;
            const auto parsed = std::from_chars(number.data(), number.data() + number.size(), layer);
            if (parsed.ec != std::errc() || layer >= *layer_count)
                return Error{ErrorCode::InvalidModel, "GGUF tensor exceeds target layer count: " + name};
            if (layer >= target_layers)
                continue;
        }
        if (std::regex_match(name, match, expert_bank))
        {
            if (info.shape.size() != 3 || info.shape[0] != *expert_count || info.size % *expert_count != 0)
                return Error{ErrorCode::InvalidModel, "invalid GGUF Expert bank: " + name};
            if (*expert_count > maximum_tensor_aliases - std::min(result.size(), maximum_tensor_aliases))
                return Error{ErrorCode::InvalidModel, "GGUF Expert aliases exceed the supported catalog size"};
            const std::string role = match[2].str() == "gate" ? "w1" : match[2].str() == "up" ? "w3"
                                                                                              : "w2";
            const uint64_t slice_size = info.size / *expert_count;
            for (uint32_t expert = 0; expert < *expert_count; ++expert)
            {
                const std::string target = "layers." + match[1].str() + ".ffn.experts." + std::to_string(expert) + "." + role + ".weight";
                SafetensorInfo slice{info.path, gguf_storage_dtype(info.ggml_type), {info.shape[1], info.shape[2]}, info.offset + expert * slice_size, slice_size, info.ggml_type};
                if (!result.emplace(target, std::move(slice)).second)
                    return Error{ErrorCode::InvalidModel, "duplicate mapped GGUF tensor: " + target};
            }
            continue;
        }
        if (info.dtype == DType::MxFp4)
            return Error{ErrorCode::UnsupportedModel, "GGUF MXFP4 is currently supported for routed Experts: " + name};
        auto target = deepseek_gguf_tensor_name(name);
        if (!target)
            return target.error();
        if (result.size() >= maximum_tensor_aliases)
            return Error{ErrorCode::InvalidModel, "GGUF tensor aliases exceed the supported catalog size"};
        SafetensorInfo tensor{info.path, gguf_storage_dtype(info.ggml_type), info.shape, info.offset, info.size, info.ggml_type};
        if (!result.emplace(target.value(), std::move(tensor)).second)
            return Error{ErrorCode::InvalidModel, "duplicate mapped GGUF tensor: " + target.value()};
    }
    return result;
}

struct GgufModelEntry
{
    const char* architecture;
    Result<std::string> (*manifest)(const GgufArchive&);
    Result<std::unordered_map<std::string, SafetensorInfo>> (*tensors)(const GgufArchive&);
};

static Result<const GgufModelEntry*> gguf_model_entry(const GgufArchive& archive)
{
    static const GgufModelEntry entries[] = {
        {"deepseek4", deepseek_gguf_manifest, deepseek_gguf_tensors},
        {"gpt-oss", gguf_gpt_oss_manifest, gguf_gpt_oss_tensors},
        {"qwen35", gguf_qwen_manifest, gguf_qwen_tensors},
        {"qwen35moe", gguf_qwen_manifest, gguf_qwen_tensors},
        {"qwen4exp", gguf_qwen_manifest, gguf_qwen_tensors}};
    const auto* architecture = archive.find_metadata("general.architecture");
    if (!architecture || !architecture->string())
        return Error{ErrorCode::InvalidModel, "GGUF model architecture is missing"};
    for (const auto& [name, value] : archive.metadata_values())
    {
        (void)value;
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (lower.find("hadamard") != std::string::npos)
            return Error{ErrorCode::UnsupportedModel, "Hadamard-transformed GGUF weights are unsupported"};
    }
    for (const auto& entry : entries)
        if (*architecture->string() == entry.architecture)
            return &entry;
    return Error{ErrorCode::UnsupportedModel, "unsupported GGUF model architecture: " + *architecture->string()};
}

Result<std::string> gguf_model_manifest(const GgufArchive& archive)
{
    auto entry = gguf_model_entry(archive);
    if (!entry)
        return entry.error();
    return entry.value()->manifest(archive);
}

Result<std::unordered_map<std::string, SafetensorInfo>> gguf_model_tensors(const GgufArchive& archive)
{
    auto entry = gguf_model_entry(archive);
    if (!entry)
        return entry.error();
    return entry.value()->tensors(archive);
}

} // namespace moe
} // namespace ncnn
