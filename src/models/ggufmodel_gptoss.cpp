#include "ggufmodel_gptoss.h"
#include "json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <regex>

namespace ncnn {
namespace moe {

static std::optional<uint64_t> gpt_oss_integer(const GgufArchive& archive, const std::string& key)
{
    const auto* value = archive.find_metadata("gpt-oss." + key);
    if (!value)
        return std::nullopt;
    if (const auto scalar = value->unsigned_integer())
        return scalar;
    const auto* values = value->array();
    if (!values || values->empty())
        return std::nullopt;
    const auto first = values->front().unsigned_integer();
    for (const auto& item : *values)
        if (!first || item.unsigned_integer() != first)
            return std::nullopt;
    return first;
}

static std::optional<double> gpt_oss_number(const GgufArchive& archive, const std::string& key)
{
    const auto* value = archive.find_metadata("gpt-oss." + key);
    return value ? value->number() : std::nullopt;
}

Result<std::string> gguf_gpt_oss_manifest(const GgufArchive& archive)
{
    for (const char* key : {"rope.freq_base", "rope.freq_base_swa", "rope.scaling.factor", "rope.scaling.yarn_beta_slow", "rope.scaling.yarn_beta_fast", "attention.layer_norm_rms_epsilon"})
    {
        const auto* value = archive.find_metadata(std::string("gpt-oss.") + key);
        if (value && (!value->number() || !std::isfinite(*value->number()) || *value->number() <= 0))
            return Error{ErrorCode::InvalidModel, std::string("invalid GPT-OSS GGUF numeric metadata: ") + key};
    }
    const auto* tokens = archive.find_metadata("tokenizer.ggml.tokens");
    if (!tokens || !tokens->array() || tokens->array()->empty())
        return Error{ErrorCode::InvalidModel, "GPT-OSS GGUF vocabulary is missing"};
    JsonObject config;
    config.add_string("model_type", "gpt_oss");
    config.add_uint("vocab_size", tokens->array()->size());
    const std::pair<const char*, const char*> fields[] = {
        {"block_count", "num_hidden_layers"},
        {"embedding_length", "hidden_size"},
        {"expert_feed_forward_length", "intermediate_size"},
        {"expert_count", "num_local_experts"},
        {"expert_used_count", "experts_per_token"},
        {"attention.head_count", "num_attention_heads"},
        {"attention.head_count_kv", "num_key_value_heads"},
        {"attention.sliding_window", "sliding_window"},
        {"context_length", "max_position_embeddings"}};
    for (const auto& [source, target] : fields)
    {
        const auto value = gpt_oss_integer(archive, source);
        if (!value || *value == 0 || *value > UINT32_MAX)
            return Error{ErrorCode::InvalidModel, std::string("invalid GPT-OSS GGUF metadata: ") + source};
        config.add_uint(target, *value);
    }
    const auto layers = *gpt_oss_integer(archive, "block_count");
    if (layers > archive.tensor_infos().size())
        return Error{ErrorCode::InvalidModel, "GPT-OSS GGUF layer count exceeds its tensor catalog"};
    if (const auto* pattern = archive.find_metadata("gpt-oss.attention.sliding_window_pattern"))
    {
        if (const auto* values = pattern->array())
        {
            if (values->size() != layers)
                return Error{ErrorCode::InvalidModel, "GPT-OSS GGUF sliding-window pattern length mismatch"};
            for (size_t i = 0; i < values->size(); ++i)
            {
                const bool expected = i % 2 == 0;
                if ((*values)[i].boolean() != expected && (*values)[i].unsigned_integer() != static_cast<uint64_t>(expected))
                    return Error{ErrorCode::UnsupportedModel, "unsupported GPT-OSS GGUF sliding-window pattern"};
            }
        }
        else if (pattern->unsigned_integer() != 2)
            return Error{ErrorCode::UnsupportedModel, "unsupported GPT-OSS GGUF sliding-window pattern"};
    }
    const auto heads = *gpt_oss_integer(archive, "attention.head_count");
    auto dimension = gpt_oss_integer(archive, "attention.key_length");
    if (!dimension)
    {
        const auto* query = archive.find("blk.0.attn_q.weight");
        if (!query || query->shape.size() != 2 || query->shape[0] % heads != 0)
            return Error{ErrorCode::InvalidModel, "GPT-OSS GGUF query shape cannot determine head dimension"};
        dimension = query->shape[0] / heads;
    }
    if (*dimension == 0 || *dimension > UINT32_MAX)
        return Error{ErrorCode::InvalidModel, "invalid GPT-OSS GGUF head dimension"};
    for (const char* key : {"attention.value_length", "attention.key_length_swa", "attention.value_length_swa", "rope.dimension_count_swa"})
        if (const auto* value = archive.find_metadata(std::string("gpt-oss.") + key); value && value->unsigned_integer() != dimension)
            return Error{ErrorCode::UnsupportedModel, "GPT-OSS GGUF requires matching key, value and rotary dimensions"};
    const auto rotary = gpt_oss_integer(archive, "rope.dimension_count");
    if (rotary && *rotary != *dimension)
        return Error{ErrorCode::UnsupportedModel, "GPT-OSS requires full-head rotary dimensions"};
    config.add_uint("head_dim", *dimension);
    const double factor = gpt_oss_number(archive, "rope.scaling.factor").value_or(1.0);
    const auto* scaling = archive.find_metadata("gpt-oss.rope.scaling.type");
    if (scaling && (!scaling->string() || (*scaling->string() != "yarn" && *scaling->string() != "none")))
        return Error{ErrorCode::UnsupportedModel, "unsupported GPT-OSS GGUF RoPE scaling type"};
    if (factor != 1 && (!scaling || *scaling->string() != "yarn"))
        return Error{ErrorCode::UnsupportedModel, "GPT-OSS GGUF scaled RoPE requires explicit YaRN metadata"};
    if (const auto* attention_factor = archive.find_metadata("gpt-oss.rope.scaling.attn_factor"); attention_factor && attention_factor->number() != 1.0)
        return Error{ErrorCode::UnsupportedModel, "unsupported GPT-OSS GGUF RoPE attention factor"};
    const auto original = gpt_oss_integer(archive, "rope.scaling.original_context_length");
    if (!std::isfinite(factor) || factor < 1 || (factor != 1 && (!original || *original == 0)))
        return Error{ErrorCode::InvalidModel, "GPT-OSS GGUF scaled RoPE requires its original context length"};
    config.add_uint("initial_context_length", original.value_or(*gpt_oss_integer(archive, "context_length")));
    const double theta = gpt_oss_number(archive, "rope.freq_base").value_or(10000);
    if (const auto swa_theta = gpt_oss_number(archive, "rope.freq_base_swa"); swa_theta && *swa_theta != theta)
        return Error{ErrorCode::UnsupportedModel, "GPT-OSS GGUF requires one RoPE frequency base across attention layers"};
    config.add_double("rope_theta", theta);
    config.add_double("rms_norm_eps", gpt_oss_number(archive, "attention.layer_norm_rms_epsilon").value_or(1e-5));
    JsonObject rope;
    rope.add_double("factor", factor);
    rope.add_double("beta_slow", gpt_oss_number(archive, "rope.scaling.yarn_beta_slow").value_or(1));
    rope.add_double("beta_fast", gpt_oss_number(archive, "rope.scaling.yarn_beta_fast").value_or(32));
    config.add_raw("rope_scaling", rope.finish());
    double clamp = 7;
    if (const auto* value = archive.find_metadata("gpt-oss.swiglu_clamp_exp"))
    {
        if (const auto number = value->number())
            clamp = *number;
        else if (const auto* values = value->array(); values && values->size() >= layers)
        {
            const auto first = values->front().number();
            if (!first)
                return Error{ErrorCode::InvalidModel, "invalid GPT-OSS GGUF activation clamp"};
            clamp = *first;
            for (size_t i = 0; i < layers; ++i)
                if ((*values)[i].number() != first)
                    return Error{ErrorCode::UnsupportedModel, "nonuniform GPT-OSS GGUF activation clamps are unsupported"};
        }
        else
            return Error{ErrorCode::InvalidModel, "invalid GPT-OSS GGUF activation clamp"};
    }
    if (!std::isfinite(clamp) || clamp <= 0)
        return Error{ErrorCode::InvalidModel, "invalid GPT-OSS GGUF activation clamp"};
    config.add_double("swiglu_limit", clamp);
    const bool bias = archive.find("blk.0.attn_q.bias") != nullptr;
    for (uint64_t layer = 0; layer < layers; ++layer)
        for (const char* role : {"q", "k", "v", "output"})
            if ((archive.find("blk." + std::to_string(layer) + ".attn_" + role + ".bias") != nullptr) != bias)
                return Error{ErrorCode::InvalidModel, "inconsistent GPT-OSS GGUF attention biases"};
    config.add_bool("attention_bias", bias);
    return config.finish();
}

static std::string gpt_oss_dtype(uint32_t type)
{
    switch (type)
    {
    case 0: return "F32";
    case 1: return "F16";
    case 30: return "BF16";
    case 39: return "MXFP4";
    default: return "U8";
    }
}

Result<std::unordered_map<std::string, SafetensorInfo>> gguf_gpt_oss_tensors(const GgufArchive& archive)
{
    auto manifest = gguf_gpt_oss_manifest(archive);
    if (!manifest)
        return manifest.error();
    const uint64_t layers = *gpt_oss_integer(archive, "block_count");
    const uint64_t experts = *gpt_oss_integer(archive, "expert_count");
    const uint64_t hidden = *gpt_oss_integer(archive, "embedding_length");
    const uint64_t intermediate = *gpt_oss_integer(archive, "expert_feed_forward_length");
    static const std::unordered_map<std::string, std::string> roots = {
        {"token_embd.weight", "model.embed_tokens.weight"},
        {"output_norm.weight", "model.norm.weight"},
        {"output.weight", "lm_head.weight"}};
    static const std::unordered_map<std::string, std::string> roles = {
        {"attn_norm.weight", "input_layernorm.weight"},
        {"ffn_norm.weight", "post_attention_layernorm.weight"},
        {"post_attention_norm.weight", "post_attention_layernorm.weight"},
        {"attn_q.weight", "self_attn.q_proj.weight"},
        {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"},
        {"attn_output.weight", "self_attn.o_proj.weight"},
        {"attn_q.bias", "self_attn.q_proj.bias"},
        {"attn_k.bias", "self_attn.k_proj.bias"},
        {"attn_v.bias", "self_attn.v_proj.bias"},
        {"attn_output.bias", "self_attn.o_proj.bias"},
        {"attn_sinks.weight", "self_attn.sinks"},
        {"ffn_gate_inp.weight", "mlp.router.weight"},
        {"ffn_gate_inp.bias", "mlp.router.bias"}};
    static const std::regex block("blk\\.([0-9]+)\\.(.+)");
    static const std::regex bank("ffn_(gate|up|down)_exps\\.(weight|bias)");
    std::unordered_map<std::string, SafetensorInfo> result;
    const auto add = [&](const std::string& name, const GgufTensorInfo& info, std::vector<uint32_t> shape, uint64_t offset, uint64_t size) -> bool {
        return result.size() < 1024 * 1024
               && result.emplace(name, SafetensorInfo{info.path, gpt_oss_dtype(info.ggml_type), std::move(shape), offset, size, info.ggml_type}).second;
    };
    for (const auto& [name, info] : archive.tensor_infos())
    {
        if (const auto root = roots.find(name); root != roots.end())
        {
            if (!add(root->second, info, info.shape, info.offset, info.size))
                return Error{ErrorCode::InvalidModel, "duplicate or excessive GPT-OSS GGUF tensors"};
            continue;
        }
        std::smatch match;
        if (!std::regex_match(name, match, block))
            return Error{ErrorCode::UnsupportedModel, "unknown GPT-OSS GGUF tensor: " + name};
        const std::string number = match[1].str();
        uint64_t layer = 0;
        if (std::from_chars(number.data(), number.data() + number.size(), layer).ec != std::errc() || layer >= layers)
            return Error{ErrorCode::InvalidModel, "GPT-OSS GGUF tensor layer is out of range"};
        const std::string prefix = "model.layers." + number + ".";
        const std::string role = match[2].str();
        std::smatch expert;
        if (std::regex_match(role, expert, bank))
        {
            const bool down = expert[1].str() == "down";
            const bool bias = expert[2].str() == "bias";
            const std::vector<uint32_t> expected = bias ? std::vector<uint32_t>{static_cast<uint32_t>(experts), static_cast<uint32_t>(down ? hidden : intermediate)}
                                                        : std::vector<uint32_t>{static_cast<uint32_t>(experts), static_cast<uint32_t>(down ? hidden : intermediate), static_cast<uint32_t>(down ? intermediate : hidden)};
            if (info.shape != expected || info.size % experts != 0 || experts > 1024 * 1024 - result.size())
                return Error{ErrorCode::InvalidModel, "invalid GPT-OSS GGUF Expert bank: " + name};
            const uint64_t size = info.size / experts;
            const std::string projection = down ? "down_proj" : expert[1].str() == "gate" ? "gate_proj"
                                                                                          : "up_proj";
            const std::vector<uint32_t> shape(expected.begin() + 1, expected.end());
            for (uint64_t index = 0; index < experts; ++index)
                if (!add(prefix + "mlp.experts." + std::to_string(index) + "." + projection + (bias ? ".bias" : ".weight"), info, shape, info.offset + index * size, size))
                    return Error{ErrorCode::InvalidModel, "duplicate GPT-OSS GGUF Expert tensors"};
            continue;
        }
        const auto mapped = roles.find(role);
        if (mapped == roles.end() || info.ggml_type == 39)
            return Error{ErrorCode::UnsupportedModel, "unsupported GPT-OSS GGUF tensor: " + name};
        if (!add(prefix + mapped->second, info, info.shape, info.offset, info.size))
            return Error{ErrorCode::InvalidModel, "duplicate GPT-OSS GGUF tensor alias"};
    }
    return result;
}

} // namespace moe
} // namespace ncnn
