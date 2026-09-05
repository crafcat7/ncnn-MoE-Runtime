#include "modeladapter_qwen4exp.h"

#include "tensornames.h"
#include "modeladapter.h"
#include "safetensors.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <limits>
#include <optional>
#include <regex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static constexpr const char* q4_mxfp4_artifact_name = "ncnn-moe-qwen3.8-mxfp4.safetensors";

static std::string q4_mxfp4_expert_prefix(uint32_t layer_id)
{
    return "__ncnn_moe_qwen3_8_mxfp4__.layers." + std::to_string(layer_id) + ".experts.";
}

static Result<void> q4_validate_mxfp4_artifact(
    const std::filesystem::path& model_root,
    uint32_t layer_count,
    uint32_t mtp_layer_count,
    uint32_t expert_count,
    uint32_t hidden_size,
    uint32_t intermediate_size)
{
    if (hidden_size % 32 != 0 || intermediate_size % 32 != 0)
        return Error{ErrorCode::InvalidModel, "Qwen4 Exp MXFP4 artifact dimensions must be divisible by 32"};
    auto opened = SafetensorsArchive::open_file(model_root / q4_mxfp4_artifact_name);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    auto status = validate_mxfp4_artifact_identity(
        archive,
        model_root,
        "__ncnn_moe_qwen3_8_mxfp4__.identity.v1.",
        layer_count,
        mtp_layer_count,
        expert_count,
        hidden_size,
        intermediate_size,
        "Qwen4 Exp artifact identity source",
        "Qwen4 Exp MXFP4 artifact");
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id)
    {
        status = validate_mxfp4_artifact_expert_bank(
            archive,
            q4_mxfp4_expert_prefix(layer_id),
            expert_count,
            hidden_size,
            intermediate_size,
            "Qwen4 Exp MXFP4 artifact");
        if (!status)
            return status.error();
    }
    return {};
}

static void q4_skip_whitespace(const std::string& json, size_t& position) noexcept
{
    while (position < json.size()
           && std::isspace(static_cast<unsigned char>(json[position])))
    {
        ++position;
    }
}

static bool q4_parse_member_name(
    const std::string& json,
    size_t& position,
    std::string& name)
{
    if (position >= json.size() || json[position] != '"')
        return false;
    ++position;
    name.clear();
    bool escaped = false;
    while (position < json.size())
    {
        const char value = json[position++];
        if (escaped)
        {
            name.push_back(value);
            escaped = false;
        }
        else if (value == '\\')
        {
            escaped = true;
        }
        else if (value == '"')
        {
            return true;
        }
        else
        {
            name.push_back(value);
        }
    }
    return false;
}

static std::optional<std::string> q4_object_member(
    const std::string& json,
    const std::string& requested_name)
{
    size_t position = 0;
    q4_skip_whitespace(json, position);
    if (position >= json.size() || json[position++] != '{')
        return std::nullopt;
    while (position < json.size())
    {
        q4_skip_whitespace(json, position);
        if (position < json.size() && json[position] == ',')
        {
            ++position;
            q4_skip_whitespace(json, position);
        }
        if (position >= json.size() || json[position] == '}')
            break;
        std::string name;
        if (!q4_parse_member_name(json, position, name))
            return std::nullopt;
        q4_skip_whitespace(json, position);
        if (position >= json.size() || json[position++] != ':')
            return std::nullopt;
        q4_skip_whitespace(json, position);
        const size_t value_start = position;
        uint32_t object_depth = 0;
        uint32_t array_depth = 0;
        bool in_string = false;
        bool escaped = false;
        while (position < json.size())
        {
            const char value = json[position];
            if (in_string)
            {
                if (escaped)
                    escaped = false;
                else if (value == '\\')
                    escaped = true;
                else if (value == '"')
                    in_string = false;
            }
            else if (value == '"')
            {
                in_string = true;
            }
            else if (value == '{')
            {
                ++object_depth;
            }
            else if (value == '[')
            {
                ++array_depth;
            }
            else if (value == '}')
            {
                if (object_depth == 0 && array_depth == 0)
                    break;
                if (object_depth == 0)
                    return std::nullopt;
                --object_depth;
            }
            else if (value == ']')
            {
                if (array_depth == 0)
                    return std::nullopt;
                --array_depth;
            }
            else if (value == ',' && object_depth == 0 && array_depth == 0)
            {
                break;
            }
            ++position;
        }
        size_t value_end = position;
        while (value_end > value_start
               && std::isspace(static_cast<unsigned char>(json[value_end - 1])))
        {
            --value_end;
        }
        if (name == requested_name)
            return json.substr(value_start, value_end - value_start);
    }
    return std::nullopt;
}

static Result<uint32_t> q4_uint(const std::string& json, const std::string& key)
{
    const std::optional<std::string> value = q4_object_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, "Qwen4 Exp manifest is missing integer field: " + key};
    const std::regex expression("^\\s*([0-9]+)\\s*$");
    std::smatch match;
    if (!std::regex_match(*value, match, expression))
        return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp integer field: " + key};
    try
    {
        const unsigned long long parsed_value = std::stoull(match[1].str());
        if (parsed_value > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, "Qwen4 Exp integer is out of range: " + key};
        return static_cast<uint32_t>(parsed_value);
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp integer field: " + key};
    }
}

static Result<float> q4_float(const std::string& json, const std::string& key)
{
    const std::optional<std::string> value = q4_object_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, "Qwen4 Exp manifest is missing numeric field: " + key};
    const std::regex expression("^\\s*([-+]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?)\\s*$");
    std::smatch match;
    if (!std::regex_match(*value, match, expression))
        return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp numeric field: " + key};
    try
    {
        return std::stof(match[1].str());
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp numeric field: " + key};
    }
}

static Result<std::string> q4_string(const std::string& json, const std::string& key)
{
    const std::optional<std::string> value = q4_object_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, "Qwen4 Exp manifest is missing string field: " + key};
    const std::regex expression("^\\s*\\\"([^\\\"]+)\\\"\\s*$");
    std::smatch match;
    if (!std::regex_match(*value, match, expression))
        return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp string field: " + key};
    return match[1].str();
}

static Result<bool> q4_bool(const std::string& json, const std::string& key)
{
    const std::optional<std::string> value = q4_object_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, "Qwen4 Exp manifest is missing boolean field: " + key};
    const std::regex expression("^\\s*(true|false)\\s*$");
    std::smatch match;
    if (!std::regex_match(*value, match, expression))
        return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp boolean field: " + key};
    return match[1].str() == "true";
}

static Result<std::vector<std::string>> q4_strings(const std::string& json, const std::string& key)
{
    const std::optional<std::string> value = q4_object_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, "Qwen4 Exp manifest is missing string array: " + key};
    const std::regex expression("^\\s*\\[([^\\]]*)\\]\\s*$");
    std::smatch match;
    if (!std::regex_match(*value, match, expression))
        return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp string array: " + key};
    const std::regex value_expression("\\\"([^\\\"]+)\\\"");
    std::vector<std::string> values;
    for (std::sregex_iterator iterator(match[1].first, match[1].second, value_expression), end; iterator != end; ++iterator)
        values.push_back((*iterator)[1].str());
    return values;
}

static Result<std::vector<uint32_t>> q4_uints(const std::string& json, const std::string& key)
{
    const std::optional<std::string> value = q4_object_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, "Qwen4 Exp manifest is missing integer array: " + key};
    const std::regex expression("^\\s*\\[([^\\]]*)\\]\\s*$");
    std::smatch match;
    if (!std::regex_match(*value, match, expression))
        return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp integer array: " + key};
    const std::regex value_expression("([0-9]+)");
    std::vector<uint32_t> values;
    for (std::sregex_iterator iterator(match[1].first, match[1].second, value_expression), end; iterator != end; ++iterator)
    {
        try
        {
            const unsigned long long parsed_value = std::stoull((*iterator)[1].str());
            if (parsed_value > std::numeric_limits<uint32_t>::max())
                return Error{ErrorCode::InvalidModel, "Qwen4 Exp integer array value is out of range: " + key};
            values.push_back(static_cast<uint32_t>(parsed_value));
        }
        catch (const std::exception&)
        {
            return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp integer array: " + key};
        }
    }
    return values;
}

static Result<void> q4_add_expert_bank(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& target_suffix,
    const std::string& source,
    uint32_t layer_id,
    uint32_t expert_count,
    uint32_t rows,
    uint32_t columns)
{
    auto loaded = archive.load_tensor(source);
    if (!loaded)
        return loaded.error();
    TensorData bank = std::move(loaded).value();
    const uint64_t slice_elements = static_cast<uint64_t>(rows) * columns;
    const uint64_t slice_size = slice_elements * sizeof(uint16_t);
    if (bank.dtype != DType::BFloat16
        || bank.shape != std::vector<uint32_t>{expert_count, rows, columns}
        || bank.bfloat16_values().size() != bank.element_count()
        || slice_size > std::numeric_limits<size_t>::max())
    {
        return Error{
            ErrorCode::InvalidModel,
            "invalid Qwen4 Exp Expert bank tensor: " + source};
    }

    const std::span<const uint16_t> values = bank.bfloat16_values();
    for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
    {
        TensorData slice;
        slice.dtype = DType::BFloat16;
        slice.shape = {rows, columns};
        const size_t element_offset = static_cast<size_t>(expert_id * slice_elements);
        if (bank.mapped_data)
        {
            const size_t byte_offset = element_offset * sizeof(uint16_t);
            slice.mapped_data = std::shared_ptr<const uint8_t>(
                bank.mapped_data, bank.mapped_data.get() + byte_offset);
            slice.mapped_size = slice_size;
        }
        else
        {
            slice.bfloat16_data.assign(
                values.begin() + element_offset,
                values.begin() + element_offset + static_cast<size_t>(slice_elements));
        }
        mapping.emplace(
            expert_prefix(layer_id, expert_id) + target_suffix,
            std::move(slice));
    }
    return {};
}

static bool q4_is_prime(uint64_t value) noexcept
{
    if (value < 2)
        return false;
    if (value % 2 == 0)
        return value == 2;
    for (uint64_t divisor = 3; divisor <= value / divisor; divisor += 2)
    {
        if (value % divisor == 0)
            return false;
    }
    return true;
}

static Result<uint64_t> q4_ple_embedding_rows(
    uint32_t ngram_size,
    uint32_t heads_per_ngram,
    uint32_t vocabulary_base,
    uint32_t alignment)
{
    if (ngram_size < 2 || heads_per_ngram == 0
        || vocabulary_base == 0 || alignment == 0)
    {
        return Error{ErrorCode::InvalidModel, "invalid Qwen4 Exp PLE vocabulary"};
    }
    const uint64_t head_count = static_cast<uint64_t>(ngram_size - 1)
                                * heads_per_ngram;
    uint64_t candidate = vocabulary_base - 1;
    uint64_t total = 0;
    for (uint64_t head = 0; head < head_count; ++head)
    {
        do
        {
            if (candidate == std::numeric_limits<uint64_t>::max())
                return Error{ErrorCode::InvalidModel, "Qwen4 Exp PLE vocabulary overflows"};
            ++candidate;
        } while (!q4_is_prime(candidate));
        if (candidate > std::numeric_limits<uint64_t>::max() - total)
            return Error{ErrorCode::InvalidModel, "Qwen4 Exp PLE vocabulary overflows"};
        total += candidate;
    }
    const uint64_t remainder = total % alignment;
    if (remainder == 0)
        return total;
    const uint64_t padding = alignment - remainder;
    if (padding > std::numeric_limits<uint64_t>::max() - total)
        return Error{ErrorCode::InvalidModel, "Qwen4 Exp PLE vocabulary overflows"};
    return total + padding;
}

static Result<void> q4_add_gated_residual(WeightMapping& mapping, const SafetensorsArchive& archive,
                                          const std::string& source, const std::string& target,
                                          bool has_injection)
{
    const std::pair<const char*, const char*> common[] = {
        {"norm.weight", "hc_norm.weight"},
        {"mix_down.weight", "input_mix_weight_down.weight"},
        {"mix_up.weight", "input_mix_weight_up.weight"},
    };
    for (const auto& item : common)
    {
        auto status = add_tensor(mapping, archive, target + item.first, source + item.second);
        if (!status)
            return status.error();
    }
    if (has_injection)
        return add_tensor(mapping, archive, target + "inject.weight", source + "block_inject_weight.weight");
    return {};
}

bool Qwen4ExpModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "qwen4_exp";
}

Result<MoeModelDescriptor> Qwen4ExpModelAdapter::parse_model(const ModelPackage& package) const
{
    if (package.manifest.model_type != "qwen4_exp")
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen4 Exp model_type: " + package.manifest.model_type};

    const std::optional<std::string> text_config = q4_object_member(
        package.manifest.raw_json, "text_config");
    if (!text_config)
        return Error{ErrorCode::InvalidModel, "Qwen4 Exp manifest is missing text_config"};
    const std::string& json = *text_config;
    const std::optional<std::string> rope_parameters = q4_object_member(
        json, "rope_parameters");
    auto vocabulary_size = q4_uint(json, "vocab_size");
    if (!vocabulary_size)
        return vocabulary_size.error();
    auto hidden_size = q4_uint(json, "hidden_size");
    if (!hidden_size)
        return hidden_size.error();
    auto intermediate_size = q4_uint(json, "moe_intermediate_size");
    if (!intermediate_size)
        return intermediate_size.error();
    auto shared_intermediate_size = q4_uint(json, "shared_expert_intermediate_size");
    if (!shared_intermediate_size)
        return shared_intermediate_size.error();
    auto layer_count = q4_uint(json, "num_hidden_layers");
    if (!layer_count)
        return layer_count.error();
    auto expert_count = q4_uint(json, "num_experts");
    if (!expert_count)
        return expert_count.error();
    auto top_k = q4_uint(json, "num_experts_per_tok");
    if (!top_k)
        return top_k.error();
    auto attention_head_count = q4_uint(json, "num_attention_heads");
    if (!attention_head_count)
        return attention_head_count.error();
    auto kv_head_count = q4_uint(json, "num_key_value_heads");
    if (!kv_head_count)
        return kv_head_count.error();
    auto head_dimension = q4_uint(json, "head_dim");
    if (!head_dimension)
        return head_dimension.error();
    auto linear_key_head_count = q4_uint(json, "linear_num_key_heads");
    if (!linear_key_head_count)
        return linear_key_head_count.error();
    auto linear_value_head_count = q4_uint(json, "linear_num_value_heads");
    if (!linear_value_head_count)
        return linear_value_head_count.error();
    auto linear_key_head_dimension = q4_uint(json, "linear_key_head_dim");
    if (!linear_key_head_dimension)
        return linear_key_head_dimension.error();
    auto linear_value_head_dimension = q4_uint(json, "linear_value_head_dim");
    if (!linear_value_head_dimension)
        return linear_value_head_dimension.error();
    auto linear_convolution = q4_uint(json, "linear_conv_kernel_dim");
    if (!linear_convolution)
        return linear_convolution.error();
    auto maximum_context = q4_uint(json, "max_position_embeddings");
    if (!maximum_context)
        return maximum_context.error();
    auto norm_epsilon = q4_float(json, "rms_norm_eps");
    if (!norm_epsilon)
        return norm_epsilon.error();
    auto rope_theta = q4_float(
        rope_parameters ? *rope_parameters : json, "rope_theta");
    if (!rope_theta)
        return rope_theta.error();
    auto partial_rotary_factor = q4_float(
        rope_parameters ? *rope_parameters : json, "partial_rotary_factor");
    if (!partial_rotary_factor)
        return partial_rotary_factor.error();
    auto index_head_count = q4_uint(json, "indexer_n_heads");
    if (!index_head_count)
        return index_head_count.error();
    auto index_kv_head_count = q4_uint(json, "indexer_kv_heads");
    if (!index_kv_head_count)
        return index_kv_head_count.error();
    auto index_head_dimension = q4_uint(json, "indexer_head_dim");
    if (!index_head_dimension)
        return index_head_dimension.error();
    auto index_budget = q4_uint(json, "indexer_budget");
    if (!index_budget)
        return index_budget.error();
    auto index_compression = q4_uint(json, "indexer_compress_ratio");
    if (!index_compression)
        return index_compression.error();
    auto hyper_count = q4_uint(json, "hc_count");
    if (!hyper_count)
        return hyper_count.error();
    auto hyper_low_rank = q4_uint(json, "hc_lowrank");
    if (!hyper_low_rank)
        return hyper_low_rank.error();
    auto ple_embedding_dimension = q4_uint(json, "ple_embed_dim");
    if (!ple_embedding_dimension)
        return ple_embedding_dimension.error();
    auto ple_convolution = q4_uint(json, "ple_conv_kernel_size");
    if (!ple_convolution)
        return ple_convolution.error();
    auto ngram_size = q4_uint(json, "ngram_size");
    if (!ngram_size)
        return ngram_size.error();
    auto heads_per_ngram = q4_uint(json, "heads_per_ngram");
    if (!heads_per_ngram)
        return heads_per_ngram.error();
    auto ngram_vocabulary_base = q4_uint(json, "ngram_vocab_size_base");
    if (!ngram_vocabulary_base)
        return ngram_vocabulary_base.error();
    auto ngram_vocabulary_alignment = q4_uint(
        json, "make_ngram_vocab_size_divisible_by");
    if (!ngram_vocabulary_alignment)
        return ngram_vocabulary_alignment.error();
    auto embedding_shards = q4_uint(json, "split_ngram_parts");
    if (!embedding_shards)
        return embedding_shards.error();
    auto eos_token_id = q4_uint(json, "eos_token_id");
    if (!eos_token_id)
        return eos_token_id.error();
    auto layer_types = q4_strings(json, "layer_types");
    if (!layer_types)
        return layer_types.error();
    auto ple_layer_ids = q4_uints(json, "ple_layer_ids");
    if (!ple_layer_ids)
        return ple_layer_ids.error();
    auto activation = q4_string(json, "hidden_act");
    if (!activation)
        return activation.error();
    auto activation_dtype = q4_string(json, "dtype");
    if (!activation_dtype)
        return activation_dtype.error();
    auto state_dtype = q4_string(json, "mamba_ssm_dtype");
    if (!state_dtype)
        return state_dtype.error();
    auto output_gate_type = q4_string(json, "output_gate_type");
    if (!output_gate_type)
        return output_gate_type.error();
    auto attention_bias = q4_bool(json, "attention_bias");
    if (!attention_bias)
        return attention_bias.error();

    auto rotary_dimension_result = get_rotary_dimension(
        head_dimension.value(), partial_rotary_factor.value(),
        "unsupported Qwen4 Exp architectural dimensions");
    if (!rotary_dimension_result)
        return rotary_dimension_result.error();
    const uint32_t rotary_dimension = rotary_dimension_result.value();
    if (hidden_size.value() == 0 || intermediate_size.value() == 0
        || layer_count.value() == 0 || expert_count.value() == 0
        || top_k.value() == 0 || top_k.value() > expert_count.value()
        || attention_head_count.value() == 0 || kv_head_count.value() == 0
        || head_dimension.value() == 0
        || linear_key_head_count.value() == 0
        || linear_value_head_count.value() == 0
        || linear_key_head_dimension.value() == 0
        || linear_value_head_dimension.value() == 0
        || layer_types.value().size() != layer_count.value()
        || shared_intermediate_size.value() != intermediate_size.value()
        || attention_head_count.value() % kv_head_count.value() != 0
        || linear_value_head_count.value() % linear_key_head_count.value() != 0
        || linear_convolution.value() == 0
        || index_head_count.value() == 0
        || index_head_dimension.value() == 0
        || index_kv_head_count.value() != 1 || index_compression.value() == 0
        || index_budget.value() % index_compression.value() != 0
        || hyper_low_rank.value() == 0
        || rotary_dimension > index_head_dimension.value()
        || activation.value() != "silu" || activation_dtype.value() != "bfloat16"
        || state_dtype.value() != "float32" || output_gate_type.value() != "sigmoid"
        || attention_bias.value() || hyper_count.value() != 4
        || ple_embedding_dimension.value() != hidden_size.value()
        || ple_convolution.value() == 0 || embedding_shards.value() == 0
        || ple_layer_ids.value().size() != 1 || ple_layer_ids.value()[0] == 0
        || ple_layer_ids.value()[0] > layer_count.value())
    {
        return Error{ErrorCode::InvalidModel, "unsupported Qwen4 Exp architectural dimensions"};
    }
    auto ple_embedding_rows = q4_ple_embedding_rows(
        ngram_size.value(), heads_per_ngram.value(),
        ngram_vocabulary_base.value(), ngram_vocabulary_alignment.value());
    if (!ple_embedding_rows)
        return ple_embedding_rows.error();

    AttentionDescriptor full_attention;
    full_attention.kind = AttentionKind::Standard;
    full_attention.head_count = attention_head_count.value();
    full_attention.kv_head_count = kv_head_count.value();
    full_attention.head_dimension = head_dimension.value();
    full_attention.value_head_dimension = head_dimension.value();
    full_attention.qk_rope_head_dimension = rotary_dimension;
    full_attention.initial_context_length = maximum_context.value();
    full_attention.max_context_length = maximum_context.value();
    full_attention.rope_theta = rope_theta.value();
    full_attention.flags = AttentionDescriptorQueryKeyNorm | AttentionDescriptorOutputGate | AttentionDescriptorQsa;
    full_attention.index_head_count = index_head_count.value();
    full_attention.index_head_dimension = index_head_dimension.value();
    full_attention.index_token_budget = index_budget.value();
    full_attention.index_top_k = index_budget.value() / index_compression.value();
    full_attention.compression_ratio = index_compression.value();

    AttentionDescriptor linear_attention;
    linear_attention.kind = AttentionKind::GatedDeltaNet;
    linear_attention.head_count = linear_value_head_count.value();
    linear_attention.kv_head_count = linear_key_head_count.value();
    linear_attention.head_dimension = linear_key_head_dimension.value();
    linear_attention.value_head_dimension = linear_value_head_dimension.value();
    linear_attention.convolution_kernel_size = linear_convolution.value();
    linear_attention.max_context_length = maximum_context.value();
    linear_attention.flags = AttentionDescriptorSigmoidGate;

    MoeDescriptor moe;
    moe.expert_count = expert_count.value();
    moe.top_k = top_k.value();
    moe.intermediate_size = intermediate_size.value();
    moe.shared_expert_count = 1;
    moe.score_function = RouterScoreFunction::Softmax;
    moe.normalization = RouterNormalization::SelectedExperts;
    moe.activation = ExpertActivation::Silu;
    moe.layout = ExpertLayout::PackedGateUpDown;
    uint32_t mtp_layer_count = 0;
    const std::optional<std::string> mtp_layer_count_text = q4_object_member(json, "mtp_num_hidden_layers");
    if (mtp_layer_count_text)
    {
        auto parsed_mtp_layer_count = q4_uint(json, "mtp_num_hidden_layers");
        if (!parsed_mtp_layer_count)
            return parsed_mtp_layer_count.error();
        mtp_layer_count = parsed_mtp_layer_count.value();
    }
    auto artifact_status = optional_artifact_exists(
        package.root / q4_mxfp4_artifact_name, "Qwen4 Exp MXFP4 artifact");
    if (!artifact_status)
        return artifact_status.error();
    const bool artifact_exists = artifact_status.value();
    if (artifact_exists)
    {
        auto artifact_status = q4_validate_mxfp4_artifact(
            package.root,
            layer_count.value(),
            mtp_layer_count,
            expert_count.value(),
            hidden_size.value(),
            intermediate_size.value());
        if (!artifact_status)
            return artifact_status.error();
    }
    moe.expert_weight_dtype = artifact_exists ? DType::MxFp4 : DType::BFloat16;
    moe.shared_expert_weight_dtype = DType::BFloat16;
    moe.flags = MoeDescriptorSharedExpertGate
                | MoeDescriptorFileBackedExperts;

    MoeModelDescriptor descriptor;
    descriptor.model_type = "qwen4_exp";
    descriptor.vocabulary_size = vocabulary_size.value();
    descriptor.hidden_size = hidden_size.value();
    descriptor.intermediate_size = intermediate_size.value();
    descriptor.attention_head_count = attention_head_count.value();
    descriptor.kv_head_count = kv_head_count.value();
    descriptor.head_dimension = head_dimension.value();
    descriptor.expert_count = expert_count.value();
    descriptor.experts_per_token = top_k.value();
    descriptor.activation_dtype = DType::BFloat16;
    descriptor.kv_cache_dtype = DType::BFloat16;
    descriptor.norm_epsilon = norm_epsilon.value();
    descriptor.norm_weight_offset = 1.0f;
    descriptor.final_norm = NormType::None;
    descriptor.hyper_connection_kind = HyperConnectionKind::GatedResidual;
    descriptor.hyper_connection_multiplier = hyper_count.value();
    descriptor.hyper_connection_low_rank = hyper_low_rank.value();
    descriptor.layers.resize(layer_count.value());
    for (uint32_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
    {
        const std::string& type = layer_types.value()[layer_id];
        if (type != "linear_attention" && type != "full_attention")
            return Error{ErrorCode::InvalidModel, "unsupported Qwen4 Exp layer type: " + type};
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.pre_attention_norm = NormType::None;
        layer.pre_ffn_norm = NormType::None;
        layer.attention = type == "linear_attention" ? linear_attention : full_attention;
        layer.ffn.moe = moe;
        if (layer_id + 1 == ple_layer_ids.value()[0])
        {
            layer.ple.embedding_dimension = ple_embedding_dimension.value();
            layer.ple.convolution_kernel_size = ple_convolution.value();
            layer.ple.ngram_size = ngram_size.value();
            layer.ple.heads_per_ngram = heads_per_ngram.value();
            layer.ple.embedding_shard_count = embedding_shards.value();
            layer.ple.embedding_row_count = ple_embedding_rows.value();
            layer.ple.eos_token_id = eos_token_id.value();
        }
    }
    return descriptor;
}

Result<WeightMapping> Qwen4ExpModelAdapter::map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const
{
    if (descriptor.model_type != "qwen4_exp")
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen4 Exp model_type: " + descriptor.model_type};
    auto opened = SafetensorsArchive::open(package.root);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    WeightMapping mapping;
    uint32_t expert_load_flags = 0;
    if (has_flag(package.flags, ModelPackageDeferMxfp4Experts))
        expert_load_flags |= SafetensorLoadDeferMxfp4Data;
    auto status = add_tensor(mapping, archive, "token_embedding.weight", "model.language_model.embed_tokens.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "lm_head.weight", "lm_head.weight");
    if (!status)
        return status.error();
    status = q4_add_gated_residual(mapping, archive, "model.language_model.hyper_connection_mixer.", "gated_residual.head.", false);
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
    {
        const std::string source = "model.language_model.layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        status = q4_add_gated_residual(mapping, archive, source + "attn_hyper_connection.", target + "gated_residual.attention.", true);
        if (!status)
            return status.error();
        status = q4_add_gated_residual(mapping, archive, source + "mlp_hyper_connection.", target + "gated_residual.ffn.", true);
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "router.weight", source + "mlp.gate.weight");
        if (!status)
            return status.error();
        status = add_qwen_shared_expert(mapping, archive, source, target);
        if (!status)
            return status.error();
        const MoeDescriptor& moe = descriptor.layers[layer_id].ffn.moe;
        const std::string artifact_experts = q4_mxfp4_expert_prefix(layer_id);
        if (moe.expert_weight_dtype == DType::MxFp4)
        {
            for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
            {
                const std::string expert = expert_prefix(layer_id, expert_id);
                status = add_mxfp4_expert(
                    mapping,
                    archive,
                    expert + "gate_up.weight",
                    artifact_experts + "gate_up.blocks",
                    artifact_experts + "gate_up.scales",
                    expert_id,
                    moe.intermediate_size * 2,
                    descriptor.hidden_size,
                    expert_load_flags);
                if (!status)
                    return status.error();
                status = add_mxfp4_expert(
                    mapping,
                    archive,
                    expert + "down.weight",
                    artifact_experts + "down.blocks",
                    artifact_experts + "down.scales",
                    expert_id,
                    descriptor.hidden_size,
                    moe.intermediate_size,
                    expert_load_flags);
                if (!status)
                    return status.error();
            }
        }
        else
        {
            status = q4_add_expert_bank(
                mapping, archive, "gate_up.weight",
                source + "mlp.experts.gate_up_proj", layer_id,
                moe.expert_count, moe.intermediate_size * 2,
                descriptor.hidden_size);
            if (!status)
                return status.error();
            status = q4_add_expert_bank(
                mapping, archive, "down.weight",
                source + "mlp.experts.down_proj", layer_id,
                moe.expert_count, descriptor.hidden_size,
                moe.intermediate_size);
            if (!status)
                return status.error();
        }

        const AttentionDescriptor& attention = descriptor.layers[layer_id].attention;
        if (attention.kind == AttentionKind::GatedDeltaNet)
        {
            status = add_qwen_gated_delta_net(mapping, archive, source, target);
            if (!status)
                return status.error();
        }
        else
        {
            status = add_qwen_attention(
                mapping, archive, source, target,
                attention.head_count, attention.head_dimension,
                descriptor.hidden_size, "Qwen4 Exp");
            if (!status)
                return status.error();
            const std::pair<const char*, const char*> qsa_tensors[] = {
                {"attention.qsa.query_key.weight", "self_attn.indexer.index_qk_proj.weight"},
                {"attention.qsa.query_norm.weight", "self_attn.indexer.q_layernorm.weight"},
                {"attention.qsa.key_norm.weight", "self_attn.indexer.k_layernorm.weight"},
            };
            for (const auto& item : qsa_tensors)
            {
                status = add_tensor(mapping, archive, target + item.first, source + item.second);
                if (!status)
                    return status.error();
            }
        }

        const PleDescriptor& ple = descriptor.layers[layer_id].ple;
        if (!ple.enabled())
            continue;
        const std::string ple_source = source + "ple.";
        const std::pair<const char*, const char*> ple_tensors[] = {
            {"ple.key.weight", "key_proj.weight"},
            {"ple.value.weight", "value_proj.weight"},
            {"ple.key_norm.weight", "norm_key.weight"},
            {"ple.query_norm.weight", "norm_query.weight"},
            {"ple.convolution_norm.weight", "norm_conv.weight"},
            {"ple.convolution.weight", "conv1d.weight"},
            {"ple.hash_multipliers", "ple_embedding.layer_multipliers"},
            {"ple.head_vocabulary_sizes", "ple_embedding.ngram_heads_vocab_sizes"},
            {"ple.head_offsets", "ple_embedding.ngram_heads_offsets"},
        };
        for (const auto& item : ple_tensors)
        {
            status = add_tensor(mapping, archive, target + item.first, ple_source + item.second);
            if (!status)
                return status.error();
        }
        for (uint32_t shard = 0; shard < ple.embedding_shard_count; ++shard)
        {
            status = add_tensor(mapping, archive, target + "ple.embedding_shard." + std::to_string(shard),
                                ple_source + "ple_embedding.ngram_embedding.shard_" + std::to_string(shard) + ".weight");
            if (!status)
                return status.error();
        }
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
