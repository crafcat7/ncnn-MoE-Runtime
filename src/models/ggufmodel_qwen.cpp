#include "ggufmodel_qwen.h"
#include "ggufmodel.h"
#include "json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string_view>
#include <unordered_set>

namespace ncnn {
namespace moe {

static Result<uint32_t> qwen_gguf_integer(const GgufArchive& archive, const std::string& key, bool allow_zero = false)
{
    const auto* field = archive.find_metadata(key);
    const auto value = field ? field->unsigned_integer() : std::nullopt;
    if (!value || *value > UINT32_MAX || (!allow_zero && *value == 0))
        return Error{ErrorCode::InvalidModel, "missing or invalid Qwen GGUF integer: " + key};
    return static_cast<uint32_t>(*value);
}

static Result<std::vector<int64_t>> qwen_gguf_integers(const GgufArchive& archive, const std::string& key)
{
    const auto* field = archive.find_metadata(key);
    if (!field || !field->array() || field->array()->empty())
        return Error{ErrorCode::InvalidModel, "missing Qwen GGUF integer array: " + key};
    std::vector<int64_t> result;
    result.reserve(field->array()->size());
    for (const GgufMetadataValue& item : *field->array())
    {
        const auto value = item.integer();
        if (!value)
            return Error{ErrorCode::InvalidModel, "Qwen GGUF array requires exact signed integers: " + key};
        result.push_back(*value);
    }
    return result;
}

struct QwenGgufConfig
{
    std::string architecture;
    std::string model_type;
    uint32_t source_layers = 0;
    uint32_t layers = 0;
    uint32_t hidden = 0;
    uint32_t vocabulary = 0;
    uint32_t heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dimension = 0;
    uint32_t rope_dimension = 0;
    uint32_t linear_keys = 0;
    uint32_t linear_values = 0;
    uint32_t linear_key_dimension = 0;
    uint32_t linear_value_dimension = 0;
    uint32_t linear_inner = 0;
    uint32_t convolution = 0;
    uint32_t intermediate = 0;
    uint32_t shared_intermediate = 0;
    uint32_t experts = 0;
    uint32_t top_k = 0;
    uint32_t context = 0;
    std::vector<bool> recurrent;
};

static Result<QwenGgufConfig> qwen_gguf_config(const GgufArchive& archive)
{
    const auto* architecture = archive.find_metadata("general.architecture");
    if (!architecture || !architecture->string())
        return Error{ErrorCode::InvalidModel, "GGUF architecture is missing"};
    QwenGgufConfig config;
    config.architecture = *architecture->string();
    if (config.architecture != "qwen35" && config.architecture != "qwen35moe" && config.architecture != "qwen4exp")
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen GGUF architecture: " + config.architecture};
    config.model_type = config.architecture == "qwen35"      ? "qwen3_5"
                        : config.architecture == "qwen35moe" ? "qwen3_5_moe"
                                                             : "qwen4_exp";
    const std::string prefix = config.architecture + ".";
    if (const auto* type = archive.find_metadata(prefix + "rope.scaling.type"))
    {
        if (!type->string() || (*type->string() != "none" && !type->string()->empty()))
            return Error{ErrorCode::UnsupportedModel, "Qwen GGUF RoPE scaling is unsupported"};
    }
    for (const char* key : {"rope.scaling.factor", "rope.scaling.attn_factor", "rope.scaling.attention_factor"})
    {
        if (const auto* value = archive.find_metadata(prefix + key); value && value->number() != 1.0)
            return Error{ErrorCode::UnsupportedModel, "Qwen GGUF RoPE scaling factors must equal one"};
    }
    const std::pair<const char*, uint32_t*> fields[] = {
        {"block_count", &config.source_layers},
        {"context_length", &config.context},
        {"embedding_length", &config.hidden},
        {"attention.head_count", &config.heads},
        {"attention.head_count_kv", &config.kv_heads},
        {"rope.dimension_count", &config.rope_dimension},
        {"ssm.group_count", &config.linear_keys},
        {"ssm.state_size", &config.linear_key_dimension},
        {"ssm.time_step_rank", &config.linear_values},
        {"ssm.inner_size", &config.linear_inner},
        {"ssm.conv_kernel", &config.convolution}};
    for (const auto& [name, destination] : fields)
    {
        auto value = qwen_gguf_integer(archive, prefix + name);
        if (!value)
            return value.error();
        *destination = value.value();
    }
    uint32_t mtp = 0;
    if (archive.find_metadata(prefix + "nextn_predict_layers"))
    {
        auto value = qwen_gguf_integer(archive, prefix + "nextn_predict_layers", true);
        if (!value)
            return value.error();
        mtp = value.value();
    }
    if (mtp >= config.source_layers || config.source_layers > archive.tensor_infos().size()
        || config.linear_values % config.linear_keys != 0 || config.linear_inner % config.linear_values != 0)
        return Error{ErrorCode::InvalidModel, "invalid Qwen GGUF layer or GDN head dimensions"};
    config.layers = config.source_layers - mtp;
    config.linear_value_dimension = config.linear_inner / config.linear_values;
    const uint64_t key_rows = static_cast<uint64_t>(config.linear_keys) * config.linear_key_dimension;
    if (key_rows > (static_cast<uint64_t>(UINT32_MAX) - config.linear_inner) / 2)
        return Error{ErrorCode::InvalidModel, "Qwen GGUF GDN projection dimensions overflow"};
    const uint64_t qkv_rows = key_rows * 2 + config.linear_inner;
    const auto* embedding = archive.find("token_embd.weight");
    const auto* tokens = archive.find_metadata("tokenizer.ggml.tokens");
    if (!embedding || embedding->shape.size() != 2 || embedding->shape[1] != config.hidden
        || !tokens || !tokens->array() || tokens->array()->size() != embedding->shape[0])
        return Error{ErrorCode::InvalidModel, "Qwen GGUF embedding/vocabulary metadata disagree"};
    config.vocabulary = embedding->shape[0];
    for (uint32_t layer = 0; layer < config.layers; ++layer)
    {
        const std::string block = "blk." + std::to_string(layer) + ".";
        const bool linear = archive.find(block + "ssm_conv1d.weight") != nullptr;
        if (linear == (archive.find(block + "attn_q.weight") != nullptr))
            return Error{ErrorCode::InvalidModel, "Qwen GGUF layer must contain exactly one attention kind: " + block};
        if (linear)
        {
            const std::pair<const char*, std::vector<uint32_t>> shapes[] = {
                {"attn_qkv.weight", {static_cast<uint32_t>(qkv_rows), config.hidden}},
                {"attn_gate.weight", {config.linear_inner, config.hidden}},
                {"ssm_alpha.weight", {config.linear_values, config.hidden}},
                {"ssm_beta.weight", {config.linear_values, config.hidden}},
                {"ssm_out.weight", {config.hidden, config.linear_inner}},
                {"ssm_conv1d.weight", {static_cast<uint32_t>(qkv_rows), config.convolution}}};
            for (const auto& [name, shape] : shapes)
            {
                const auto* tensor = archive.find(block + name);
                if (!tensor || tensor->shape != shape)
                    return Error{ErrorCode::InvalidModel, "Qwen GGUF GDN metadata disagree with tensor shape: " + block + name};
            }
        }
        config.recurrent.push_back(linear);
        if (!linear && config.head_dimension == 0)
        {
            const auto* norm = archive.find(block + "attn_q_norm.weight");
            if (norm && norm->shape.size() == 1)
                config.head_dimension = norm->shape[0];
        }
    }
    if (archive.find_metadata(prefix + "attention.key_length"))
    {
        auto dimension = qwen_gguf_integer(archive, prefix + "attention.key_length");
        if (!dimension)
            return dimension.error();
        if (config.head_dimension != 0 && config.head_dimension != dimension.value())
            return Error{ErrorCode::InvalidModel, "Qwen GGUF attention head metadata disagree with query norm"};
        config.head_dimension = dimension.value();
    }
    if (config.head_dimension == 0 || config.rope_dimension > config.head_dimension
        || config.heads % config.kv_heads != 0)
        return Error{ErrorCode::InvalidModel, "invalid Qwen GGUF full attention dimensions"};
    if (const auto* value = archive.find_metadata(prefix + "attention.value_length");
        value && value->unsigned_integer() != config.head_dimension)
        return Error{ErrorCode::UnsupportedModel, "Qwen GGUF key/value head dimensions must match"};
    const std::pair<const char*, uint32_t*> ffn_fields[] = {
        {config.experts == 0 && config.architecture == "qwen35" ? "feed_forward_length" : "expert_feed_forward_length", &config.intermediate},
        {"expert_shared_feed_forward_length", &config.shared_intermediate},
        {"expert_count", &config.experts},
        {"expert_used_count", &config.top_k}};
    for (size_t i = 0; i < (config.architecture == "qwen35" ? 1u : 4u); ++i)
    {
        auto value = qwen_gguf_integer(archive, prefix + ffn_fields[i].first);
        if (!value)
            return value.error();
        *ffn_fields[i].second = value.value();
    }
    return config;
}

static Result<void> qwen_add_metadata_number(JsonObject& object, const GgufArchive& archive,
                                             const std::string& source, const char* destination)
{
    const auto* field = archive.find_metadata(source);
    const auto number = field ? field->number() : std::nullopt;
    if (!number || !std::isfinite(*number) || *number <= 0)
        return Error{ErrorCode::InvalidModel, "missing or invalid Qwen GGUF number: " + source};
    object.add_double(destination, *number);
    return {};
}

Result<std::string> gguf_qwen_manifest(const GgufArchive& archive)
{
    auto parsed = qwen_gguf_config(archive);
    if (!parsed)
        return parsed.error();
    const QwenGgufConfig& config = parsed.value();
    const std::string prefix = config.architecture + ".";
    JsonObject text;
    text.add_string("model_type", config.model_type + "_text");
    const std::pair<const char*, uint32_t> fields[] = {
        {"vocab_size", config.vocabulary},
        {"hidden_size", config.hidden},
        {"num_hidden_layers", config.layers},
        {"num_attention_heads", config.heads},
        {"num_key_value_heads", config.kv_heads},
        {"head_dim", config.head_dimension},
        {"linear_num_key_heads", config.linear_keys},
        {"linear_num_value_heads", config.linear_values},
        {"linear_key_head_dim", config.linear_key_dimension},
        {"linear_value_head_dim", config.linear_value_dimension},
        {"linear_conv_kernel_dim", config.convolution},
        {"max_position_embeddings", config.context}};
    for (const auto& [name, value] : fields)
        text.add_uint(name, value);
    text.add_uint(config.experts == 0 ? "intermediate_size" : "moe_intermediate_size", config.intermediate);
    if (config.experts != 0)
    {
        text.add_uint("num_experts", config.experts);
        text.add_uint("num_experts_per_tok", config.top_k);
        text.add_uint("shared_expert_intermediate_size", config.shared_intermediate);
    }
    text.add_uint("mtp_num_hidden_layers", 0);
    text.add_bool("mtp_use_dedicated_embeddings", false);
    text.add_string("hidden_act", "silu");
    text.add_string("dtype", "bfloat16");
    text.add_string("mamba_ssm_dtype", "float32");
    text.add_bool("attention_bias", false);
    text.add_bool("attn_output_gate", true);
    text.add_bool("tie_word_embeddings", archive.find("output.weight") == nullptr);
    auto status = qwen_add_metadata_number(text, archive, prefix + "attention.layer_norm_rms_epsilon", "rms_norm_eps");
    if (!status)
        return status.error();
    JsonObject rope;
    status = qwen_add_metadata_number(rope, archive, prefix + "rope.freq_base", "rope_theta");
    if (!status)
        return status.error();
    rope.add_double("partial_rotary_factor", static_cast<double>(config.rope_dimension) / config.head_dimension);
    text.add_raw("rope_parameters", rope.finish());
    std::string layers = "[";
    for (bool recurrent : config.recurrent)
    {
        if (layers.size() != 1)
            layers += ',';
        layers += recurrent ? "\"linear_attention\"" : "\"full_attention\"";
    }
    text.add_raw("layer_types", layers + ']');
    if (config.architecture == "qwen4exp")
    {
        const std::pair<const char*, const char*> extra_fields[] = {
            {"hyper_connection.count", "hc_count"},
            {"hyper_connection.low_rank", "hc_lowrank"},
            {"attention.indexer.head_count", "indexer_n_heads"},
            {"attention.indexer.key_length", "indexer_head_dim"},
            {"attention.indexer.top_k", "indexer_budget"},
            {"ple.ngram_size", "ngram_size"},
            {"ple.heads_per_ngram", "heads_per_ngram"},
            {"ple.conv_kernel", "ple_conv_kernel_size"}};
        for (const auto& [source, destination] : extra_fields)
        {
            auto value = qwen_gguf_integer(archive, prefix + source);
            if (!value)
                return value.error();
            text.add_uint(destination, value.value());
        }
        auto ratios = qwen_gguf_integers(archive, prefix + "attention.compress_ratios");
        if (!ratios || ratios.value().size() < config.layers)
            return Error{ErrorCode::InvalidModel, "Qwen4 GGUF compress ratios do not cover the target layers"};
        uint32_t compression = 0;
        for (uint32_t layer = 0; layer < config.layers; ++layer)
        {
            const int64_t ratio = ratios.value()[layer];
            if (ratio < 0 || ratio > UINT32_MAX || (config.recurrent[layer] && ratio != 0)
                || (!config.recurrent[layer] && (ratio == 0 || (compression != 0 && ratio != compression))))
                return Error{ErrorCode::UnsupportedModel, "Qwen4 GGUF requires uniform full-attention QSA compression"};
            if (ratio != 0)
                compression = static_cast<uint32_t>(ratio);
        }
        text.add_uint("indexer_compress_ratio", compression);
        text.add_uint("indexer_kv_heads", 1);
        text.add_string("output_gate_type", "sigmoid");
        auto ple_layers = qwen_gguf_integers(archive, prefix + "ple.layers");
        auto vocabulary = qwen_gguf_integers(archive, prefix + "ple.head_vocab_sizes");
        const auto* table = archive.find("per_layer_token_embd.weight");
        if (!ple_layers || ple_layers.value().size() != 1 || ple_layers.value()[0] < 0
            || ple_layers.value()[0] >= config.layers || !vocabulary || vocabulary.value()[0] <= 0
            || vocabulary.value()[0] > UINT32_MAX || !table || table->shape.size() != 2)
            return Error{ErrorCode::InvalidModel, "invalid Qwen4 GGUF PLE layer/table metadata"};
        text.add_raw("ple_layer_ids", "[" + std::to_string(ple_layers.value()[0] + 1) + "]");
        text.add_uint("ple_embed_dim", config.hidden);
        // GGUF combines the original HF storage shards. One mapped shard and
        // an equivalent row alignment preserve every hash-table address.
        text.add_uint("split_ngram_parts", 1);
        text.add_uint("ngram_vocab_size_base", static_cast<uint64_t>(vocabulary.value()[0]));
        text.add_uint("make_ngram_vocab_size_divisible_by", table->shape[0]);
        auto eos = qwen_gguf_integer(archive, prefix + "ple.eos_token_id", true);
        if (!eos)
            return eos.error();
        text.add_uint("eos_token_id", eos.value());
    }
    JsonObject manifest;
    manifest.add_string("model_type", config.model_type);
    manifest.add_raw("text_config", text.finish());
    return manifest.finish();
}

static SafetensorInfo qwen_gguf_info(const GgufTensorInfo& tensor)
{
    return SafetensorInfo{tensor.path, gguf_storage_dtype(tensor.ggml_type), tensor.shape, tensor.offset, tensor.size, tensor.ggml_type};
}

static std::vector<uint32_t> qwen_grouped_rows(uint32_t keys, uint32_t heads, uint32_t dimension, uint32_t prefix = 0)
{
    const uint32_t repetitions = heads / keys;
    std::vector<uint32_t> rows(static_cast<size_t>(prefix) + static_cast<size_t>(heads) * dimension);
    std::iota(rows.begin(), rows.begin() + prefix, 0u);
    for (uint32_t key = 0; key < keys; ++key)
    {
        for (uint32_t repetition = 0; repetition < repetitions; ++repetition)
        {
            for (uint32_t column = 0; column < dimension; ++column)
                rows[prefix + (key * repetitions + repetition) * dimension + column] = prefix + (repetition * keys + key) * dimension + column;
        }
    }
    return rows;
}

static Result<std::string> qwen_tensor_name(const std::string& name, bool extended,
                                            std::string_view number, std::string_view suffix)
{
    const std::string prefix = "model.language_model.";
    static const std::unordered_map<std::string, std::string> roots = {
        {"token_embd.weight", "embed_tokens.weight"},
        {"output_norm.weight", "norm.weight"},
        {"output_hc_norm.weight", "hyper_connection_mixer.hc_norm.weight"},
        {"output_hc_down.weight", "hyper_connection_mixer.input_mix_weight_down.weight"},
        {"output_hc_up.weight", "hyper_connection_mixer.input_mix_weight_up.weight"}};
    if (name == "output.weight")
        return std::string("lm_head.weight");
    if (const auto found = roots.find(name); found != roots.end())
        return prefix + found->second;
    if (number.empty() || suffix.empty())
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen GGUF tensor: " + name};
    static const std::unordered_map<std::string, std::string> fields = {
        {"attn_norm.weight", "input_layernorm.weight"},
        {"post_attention_norm.weight", "post_attention_layernorm.weight"},
        {"ffn_norm.weight", "post_attention_layernorm.weight"},
        {"attn_q.weight", "self_attn.q_proj.weight"},
        {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"},
        {"attn_output.weight", "self_attn.o_proj.weight"},
        {"attn_q_norm.weight", "self_attn.q_norm.weight"},
        {"attn_k_norm.weight", "self_attn.k_norm.weight"},
        {"attn_qkv.weight", "linear_attn.in_proj_qkv.weight"},
        {"attn_gate.weight", "linear_attn.in_proj_z.weight"},
        {"ssm_conv1d.weight", "linear_attn.conv1d.weight"},
        {"ssm_dt.bias", "linear_attn.dt_bias"},
        {"ssm_a", "linear_attn.A_log"},
        {"ssm_norm.weight", "linear_attn.norm.weight"},
        {"ssm_out.weight", "linear_attn.out_proj.weight"},
        {"ssm_alpha.weight", "linear_attn.in_proj_a.weight"},
        {"ssm_beta.weight", "linear_attn.in_proj_b.weight"},
        {"ffn_gate.weight", "mlp.gate_proj.weight"},
        {"ffn_up.weight", "mlp.up_proj.weight"},
        {"ffn_down.weight", "mlp.down_proj.weight"},
        {"ffn_gate_inp.weight", "mlp.gate.weight"},
        {"ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight"},
        {"ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight"},
        {"ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight"},
        {"ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight"},
        {"indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight"},
        {"indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight"},
        {"ple_key.weight", "ple.key_proj.weight"},
        {"ple_value.weight", "ple.value_proj.weight"},
        {"ple_norm_key.weight", "ple.norm_key.weight"},
        {"ple_norm_query.weight", "ple.norm_query.weight"},
        {"ple_norm_conv.weight", "ple.norm_conv.weight"},
        {"ple_conv1d.weight", "ple.conv1d.weight"}};
    const auto found = fields.find(std::string(suffix));
    if (found != fields.end())
        return prefix + "layers." + std::string(number) + "." + found->second;
    if (extended)
    {
        for (const auto& [gguf_role, hf_role] : {std::pair{"attn", "attn"}, std::pair{"ffn", "mlp"}})
        {
            for (const auto& [gguf_part, hf_part] : {std::pair{"norm", "hc_norm"}, std::pair{"down", "input_mix_weight_down"},
                                                     std::pair{"up", "input_mix_weight_up"}, std::pair{"inject", "block_inject_weight"}})
            {
                if (suffix == std::string("hc_") + gguf_role + "_" + gguf_part + ".weight")
                    return prefix + "layers." + std::string(number) + "." + hf_role + "_hyper_connection." + hf_part + ".weight";
            }
        }
    }
    return Error{ErrorCode::UnsupportedModel, "unsupported Qwen GGUF tensor: " + name};
}

Result<std::unordered_map<std::string, SafetensorInfo>> gguf_qwen_tensors(const GgufArchive& archive)
{
    auto parsed = qwen_gguf_config(archive);
    if (!parsed)
        return parsed.error();
    const QwenGgufConfig& config = parsed.value();
    const bool extended = config.architecture == "qwen4exp";
    std::unordered_map<std::string, SafetensorInfo> result;
    std::unordered_set<std::string> consumed;
    const auto add = [&](const std::string& name, SafetensorInfo info) -> Result<void> {
        if (result.size() >= 1024 * 1024 || !result.emplace(name, std::move(info)).second)
            return Error{ErrorCode::InvalidModel, "duplicate or excessive Qwen GGUF tensor aliases: " + name};
        return {};
    };
    for (const auto& [name, tensor] : archive.tensor_infos())
    {
        if (consumed.contains(name) || name == "per_layer_token_embd.weight")
            continue;
        std::string_view number;
        std::string_view suffix;
        const bool block_tensor = gguf_layer_name(name, number, suffix);
        if (block_tensor)
        {
            uint32_t layer = 0;
            const auto conversion = std::from_chars(number.data(), number.data() + number.size(), layer);
            if (conversion.ec != std::errc() || conversion.ptr != number.data() + number.size() || layer >= config.source_layers)
                return Error{ErrorCode::InvalidModel, "Qwen GGUF tensor exceeds source layer count: " + name};
            if (layer >= config.layers)
                continue;
            std::string_view role;
            if (suffix == "ffn_gate_exps.weight")
                role = "gate";
            else if (suffix == "ffn_up_exps.weight")
                role = "up";
            else if (suffix == "ffn_down_exps.weight")
                role = "down";
            else if (suffix == "ffn_gate_up_exps.weight")
                role = "gate_up";
            if (!role.empty())
            {
                if (config.experts == 0 || tensor.shape.size() != 3 || tensor.shape[0] != config.experts
                    || tensor.size % config.experts != 0 || config.experts > 1024 * 1024 - result.size())
                    return Error{ErrorCode::InvalidModel, "invalid Qwen GGUF Expert bank: " + name};
                const uint64_t expert_size = tensor.size / config.experts;
                for (uint32_t index = 0; index < config.experts; ++index)
                {
                    SafetensorInfo info = qwen_gguf_info(tensor);
                    info.offset += static_cast<uint64_t>(index) * expert_size;
                    info.size = expert_size;
                    info.shape.erase(info.shape.begin());
                    const std::string target = "model.language_model.layers." + std::string(number) + ".mlp.experts." + std::to_string(index) + ".";
                    if (role == "gate_up")
                    {
                        if (info.shape[0] != config.intermediate * 2 || info.size % 2 != 0)
                            return Error{ErrorCode::InvalidModel, "invalid Qwen fused Expert gate/up bank: " + name};
                        info.shape[0] /= 2;
                        info.size /= 2;
                        auto status = add(target + "gate_proj.weight", info);
                        if (!status)
                            return status.error();
                        info.offset += info.size;
                        status = add(target + "up_proj.weight", std::move(info));
                        if (!status)
                            return status.error();
                    }
                    else
                    {
                        auto status = add(target + std::string(role) + "_proj.weight", std::move(info));
                        if (!status)
                            return status.error();
                    }
                }
                continue;
            }
            if (extended && (suffix == "indexer.q_proj.weight" || suffix == "indexer.k_proj.weight"))
            {
                const std::string base = "blk." + std::string(number) + ".indexer.";
                const auto* query = archive.find(base + "q_proj.weight");
                const auto* key = archive.find(base + "k_proj.weight");
                if (!query || !key || query->shape.size() != 2 || key->shape.size() != 2
                    || query->shape[1] != key->shape[1] || query->ggml_type != key->ggml_type
                    || query->shape[0] > UINT32_MAX - key->shape[0])
                    return Error{ErrorCode::InvalidModel, "invalid Qwen4 GGUF indexer projection pair"};
                SafetensorInfo info = qwen_gguf_info(*query);
                auto transform = std::make_shared<GgufTensorTransform>();
                transform->sources = {info, qwen_gguf_info(*key)};
                info.shape[0] += key->shape[0];
                info.size += key->size;
                info.transform = std::move(transform);
                auto status = add("model.language_model.layers." + std::string(number) + ".self_attn.indexer.index_qk_proj.weight", std::move(info));
                if (!status)
                    return status.error();
                consumed.insert(base + "q_proj.weight");
                consumed.insert(base + "k_proj.weight");
                continue;
            }
        }
        auto target = qwen_tensor_name(name, extended, number, suffix);
        if (!target)
            return target.error();
        SafetensorInfo info = qwen_gguf_info(tensor);
        GgufTensorTransform transform;
        const std::string& canonical = target.value();
        if (canonical.ends_with(".mlp.shared_expert_gate.weight"))
        {
            // GGML writes this one-output Linear as an n_embd vector. Restore
            // its logical row without transforming or copying the file range.
            if (info.shape == std::vector<uint32_t>{config.hidden})
                info.shape.insert(info.shape.begin(), 1);
            else if (info.shape != std::vector<uint32_t>{1, config.hidden})
                return Error{ErrorCode::InvalidModel, "invalid Qwen GGUF shared Expert gate shape: " + name};
        }
        if ((canonical.ends_with("norm.weight") || canonical.ends_with(".ple.norm_key.weight")
             || canonical.ends_with(".ple.norm_query.weight") || canonical.ends_with(".ple.norm_conv.weight"))
            && !canonical.ends_with("linear_attn.norm.weight"))
            transform.value_transform = GgufValueTransform::SubtractOne;
        if (canonical.ends_with("linear_attn.A_log"))
            transform.value_transform = GgufValueTransform::LogNegative;
        if (canonical.find(".linear_attn.") != std::string::npos && config.linear_keys != config.linear_values)
        {
            if (canonical.ends_with("in_proj_qkv.weight") || canonical.ends_with("conv1d.weight"))
                transform.row_indices = qwen_grouped_rows(config.linear_keys, config.linear_values, config.linear_value_dimension,
                                                          config.linear_keys * config.linear_key_dimension * 2);
            else if (canonical.ends_with("in_proj_z.weight"))
                transform.row_indices = qwen_grouped_rows(config.linear_keys, config.linear_values, config.linear_value_dimension);
            else if (canonical.ends_with("in_proj_a.weight") || canonical.ends_with("in_proj_b.weight")
                     || canonical.ends_with(".A_log") || canonical.ends_with(".dt_bias"))
                transform.row_indices = qwen_grouped_rows(config.linear_keys, config.linear_values, 1);
            else if (canonical.ends_with("out_proj.weight"))
            {
                auto columns = qwen_grouped_rows(config.linear_keys, config.linear_values, config.linear_value_dimension);
                if (is_qnk_dtype(tensor.dtype) || tensor.dtype == DType::MxFp4)
                {
                    transform.input_column_indices.resize(columns.size());
                    for (uint32_t column = 0; column < columns.size(); ++column)
                        transform.input_column_indices[columns[column]] = column;
                }
                else
                    transform.column_indices = std::move(columns);
            }
        }
        if (canonical.ends_with("conv1d.weight"))
        {
            if (info.shape.size() != 2)
                return Error{ErrorCode::InvalidModel, "Qwen GGUF convolution must be a matrix"};
            info.shape.insert(info.shape.begin() + 1, 1);
        }
        if (transform.value_transform != GgufValueTransform::None || !transform.row_indices.empty()
            || !transform.column_indices.empty() || !transform.input_column_indices.empty())
        {
            transform.sources.push_back(qwen_gguf_info(tensor));
            info.transform = std::make_shared<GgufTensorTransform>(std::move(transform));
        }
        auto status = add(canonical, std::move(info));
        if (!status)
            return status.error();
    }
    if (!archive.find("output.weight"))
    {
        auto status = add("lm_head.weight", qwen_gguf_info(*archive.find("token_embd.weight")));
        if (!status)
            return status.error();
    }
    if (extended)
    {
        auto ple_layers = qwen_gguf_integers(archive, "qwen4exp.ple.layers");
        const auto* table = archive.find("per_layer_token_embd.weight");
        if (!ple_layers || ple_layers.value().size() != 1 || ple_layers.value()[0] < 0
            || ple_layers.value()[0] >= config.layers || !table || table->shape.size() != 2)
            return Error{ErrorCode::InvalidModel, "invalid Qwen4 GGUF PLE embedding metadata"};
        const std::string prefix = "model.language_model.layers." + std::to_string(ple_layers.value()[0]) + ".ple.ple_embedding.";
        for (const auto& [source, target] : {std::pair{"layer_multipliers", "layer_multipliers"},
                                             std::pair{"head_offsets", "ngram_heads_offsets"},
                                             std::pair{"head_vocab_sizes", "ngram_heads_vocab_sizes"}})
        {
            auto values = qwen_gguf_integers(archive, std::string("qwen4exp.ple.") + source);
            if (!values)
                return values.error();
            auto transform = std::make_shared<GgufTensorTransform>();
            transform->generated_i64 = std::move(values).value();
            SafetensorInfo info;
            info.dtype = "I64";
            info.shape = {static_cast<uint32_t>(transform->generated_i64.size())};
            info.size = static_cast<uint64_t>(info.shape[0]) * sizeof(int64_t);
            info.transform = std::move(transform);
            auto status = add(prefix + target, std::move(info));
            if (!status)
                return status.error();
        }
        auto status = add(prefix + "ngram_embedding.shard_0.weight", qwen_gguf_info(*table));
        if (!status)
            return status.error();
    }
    return result;
}

} // namespace moe
} // namespace ncnn
