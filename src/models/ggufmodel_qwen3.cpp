#include "ggufmodel_qwen3.h"
#include "ggufmodel.h"
#include "json.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ncnn {
namespace moe {

static Result<uint32_t> qwen3_integer(const GgufArchive& archive, const char* key, bool allow_zero = false)
{
    const std::string name = std::string("qwen3.") + key;
    const auto* field = archive.find_metadata(name);
    const auto value = field ? field->unsigned_integer() : std::nullopt;
    if (!value || *value > UINT32_MAX || (!allow_zero && *value == 0))
        return Error{ErrorCode::InvalidModel, "missing or invalid Qwen3 GGUF integer: " + name};
    return static_cast<uint32_t>(*value);
}

static Result<double> qwen3_number(const GgufArchive& archive, const char* key)
{
    const std::string name = std::string("qwen3.") + key;
    const auto* field = archive.find_metadata(name);
    const auto value = field ? field->number() : std::nullopt;
    if (!value || !std::isfinite(*value) || *value <= 0.0)
        return Error{ErrorCode::InvalidModel, "missing or invalid Qwen3 GGUF number: " + name};
    return *value;
}

Result<std::string> gguf_qwen3_manifest(const GgufArchive& archive)
{
    const auto* architecture = archive.find_metadata("general.architecture");
    if (!architecture || !architecture->string() || *architecture->string() != "qwen3")
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen3 GGUF architecture"};
    if (const auto* type = archive.find_metadata("qwen3.rope.scaling.type"))
    {
        if (!type->string() || (!type->string()->empty() && *type->string() != "none"))
            return Error{ErrorCode::UnsupportedModel, "Qwen3 GGUF RoPE scaling is unsupported"};
    }
    for (const char* key : {"qwen3.rope.scaling.factor", "qwen3.rope.scaling.attn_factor", "qwen3.rope.scaling.attention_factor"})
    {
        const auto* value = archive.find_metadata(key);
        if (value && value->number() != 1.0)
            return Error{ErrorCode::UnsupportedModel, "Qwen3 GGUF RoPE scaling factors must equal one"};
    }

    const auto layers = qwen3_integer(archive, "block_count");
    if (!layers)
        return layers.error();
    const auto context = qwen3_integer(archive, "context_length");
    if (!context)
        return context.error();
    const auto hidden = qwen3_integer(archive, "embedding_length");
    if (!hidden)
        return hidden.error();
    const auto heads = qwen3_integer(archive, "attention.head_count");
    if (!heads)
        return heads.error();
    const auto kv_heads = qwen3_integer(archive, "attention.head_count_kv");
    if (!kv_heads)
        return kv_heads.error();
    const auto intermediate = qwen3_integer(archive, "feed_forward_length");
    if (!intermediate)
        return intermediate.error();
    const auto epsilon = qwen3_number(archive, "attention.layer_norm_rms_epsilon");
    if (!epsilon)
        return epsilon.error();
    const auto rope_theta = qwen3_number(archive, "rope.freq_base");
    if (!rope_theta)
        return rope_theta.error();
    if (heads.value() % kv_heads.value() != 0)
        return Error{ErrorCode::InvalidModel, "Qwen3 GGUF attention head counts disagree"};

    uint32_t head_dimension = 0;
    if (archive.find_metadata("qwen3.attention.key_length"))
    {
        auto value = qwen3_integer(archive, "attention.key_length");
        if (!value)
            return value.error();
        head_dimension = value.value();
    }
    else
    {
        const auto* norm = archive.find("blk.0.attn_q_norm.weight");
        if (norm && norm->shape.size() == 1)
            head_dimension = norm->shape[0];
        else if (hidden.value() % heads.value() == 0)
            head_dimension = hidden.value() / heads.value();
    }
    if (head_dimension == 0)
        return Error{ErrorCode::InvalidModel, "Qwen3 GGUF head dimension is missing"};
    if (const auto* value = archive.find_metadata("qwen3.attention.value_length");
        value && value->unsigned_integer() != head_dimension)
        return Error{ErrorCode::UnsupportedModel, "Qwen3 GGUF key/value head dimensions must match"};

    const auto* embedding = archive.find("token_embd.weight");
    const auto* tokens = archive.find_metadata("tokenizer.ggml.tokens");
    if (!embedding || embedding->shape.size() != 2 || embedding->shape[1] != hidden.value()
        || !tokens || !tokens->array() || tokens->array()->size() != embedding->shape[0])
        return Error{ErrorCode::InvalidModel, "Qwen3 GGUF embedding/vocabulary metadata disagree"};
    if (const auto* output = archive.find("output.weight");
        output && (output->shape.size() != 2 || output->shape != embedding->shape))
        return Error{ErrorCode::InvalidModel, "Qwen3 GGUF output projection shape disagrees"};

    const uint32_t vocabulary = embedding->shape[0];
    const bool tied = archive.find("output.weight") == nullptr;
    JsonObject manifest;
    manifest.add_string("model_type", "qwen3");
    manifest.add_uint("vocab_size", vocabulary);
    manifest.add_uint("hidden_size", hidden.value());
    manifest.add_uint("intermediate_size", intermediate.value());
    manifest.add_uint("num_hidden_layers", layers.value());
    manifest.add_uint("num_attention_heads", heads.value());
    manifest.add_uint("num_key_value_heads", kv_heads.value());
    manifest.add_uint("head_dim", head_dimension);
    manifest.add_uint("max_position_embeddings", context.value());
    manifest.add_double("rms_norm_eps", epsilon.value());
    manifest.add_double("rope_theta", rope_theta.value());
    manifest.add_string("hidden_act", "silu");
    manifest.add_bool("attention_bias", false);
    manifest.add_bool("use_sliding_window", false);
    manifest.add_bool("tie_word_embeddings", tied);
    manifest.add_string("dtype", "bfloat16");
    return manifest.finish();
}

static Result<std::string> qwen3_tensor_name(std::string_view number, std::string_view suffix)
{
    static const std::unordered_map<std::string, std::string> fields = {
        {"attn_norm.weight", "input_layernorm.weight"},
        {"ffn_norm.weight", "post_attention_layernorm.weight"},
        {"attn_q.weight", "self_attn.q_proj.weight"},
        {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"},
        {"attn_output.weight", "self_attn.o_proj.weight"},
        {"attn_q_norm.weight", "self_attn.q_norm.weight"},
        {"attn_k_norm.weight", "self_attn.k_norm.weight"},
        {"ffn_gate.weight", "mlp.gate_proj.weight"},
        {"ffn_up.weight", "mlp.up_proj.weight"},
        {"ffn_down.weight", "mlp.down_proj.weight"}};
    const auto found = fields.find(std::string(suffix));
    if (found == fields.end())
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen3 GGUF layer tensor: blk." + std::string(number) + "." + std::string(suffix)};
    return "model.layers." + std::string(number) + "." + found->second;
}

Result<std::unordered_map<std::string, SafetensorInfo>> gguf_qwen3_tensors(const GgufArchive& archive)
{
    const auto* architecture = archive.find_metadata("general.architecture");
    if (!architecture || !architecture->string() || *architecture->string() != "qwen3")
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen3 GGUF architecture"};
    std::unordered_map<std::string, SafetensorInfo> result;
    const auto add = [&](const std::string& name, const GgufTensorInfo& tensor) -> Result<void> {
        auto inserted = result.emplace(name, SafetensorInfo{tensor.path, gguf_storage_dtype(tensor.ggml_type), tensor.shape,
                                                            tensor.offset, tensor.size, tensor.ggml_type});
        if (!inserted.second)
            return Error{ErrorCode::InvalidModel, "duplicate Qwen3 GGUF tensor alias: " + name};
        return {};
    };
    for (const auto& [name, tensor] : archive.tensor_infos())
    {
        std::string target;
        if (name == "token_embd.weight")
            target = "model.embed_tokens.weight";
        else if (name == "output_norm.weight")
            target = "model.norm.weight";
        else if (name == "output.weight")
            target = "lm_head.weight";
        else
        {
            std::string_view number;
            std::string_view suffix;
            if (!gguf_layer_name(name, number, suffix))
                return Error{ErrorCode::InvalidModel, "unsupported Qwen3 GGUF tensor: " + name};
            auto mapped = qwen3_tensor_name(number, suffix);
            if (!mapped)
                return mapped.error();
            target = std::move(mapped).value();
        }
        auto status = add(target, tensor);
        if (!status)
            return status.error();
    }
    return result;
}

} // namespace moe
} // namespace ncnn
