#include "qwen3_5_moe_model_adapter.h"

#include "internal/tensor_names.h"
#include "safetensors.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <span>
#include <utility>

namespace ncnn {
namespace moe {

static constexpr const char* qwen_mxfp4_artifact_name = "ncnn-moe-qwen3.6-mxfp4.safetensors";

static std::string qwen_mxfp4_expert_prefix(uint32_t layer_id)
{
    return "__ncnn_moe_qwen3_6_mxfp4__.layers." + std::to_string(layer_id) + ".experts.";
}

static std::string qwen_mxfp4_mtp_expert_prefix(uint32_t layer_id)
{
    return "__ncnn_moe_qwen3_6_mxfp4__.mtp.layers."
           + std::to_string(layer_id) + ".experts.";
}

static Result<uint64_t> qwen_fnv1a64_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open Qwen artifact identity source: " + path.string()};
    uint64_t hash = UINT64_C(14695981039346656037);
    std::array<char, 64 * 1024> buffer;
    while (stream)
    {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        for (std::streamsize index = 0; index < count; ++index)
        {
            hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(index)]);
            hash *= UINT64_C(1099511628211);
        }
    }
    if (!stream.eof())
        return Error{ErrorCode::IoError, "cannot read Qwen artifact identity source: " + path.string()};
    return hash;
}

static std::string qwen_mxfp4_identity_name(
    uint32_t layer_count,
    uint32_t mtp_layer_count,
    uint32_t expert_count,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    uint64_t config_hash,
    uint64_t index_hash)
{
    std::ostringstream name;
    name << "__ncnn_moe_qwen3_6_mxfp4__.identity.v3."
         << layer_count << '.' << mtp_layer_count << '.'
         << expert_count << '.'
         << hidden_size << '.'
         << intermediate_size << '.'
         << std::hex << std::setfill('0')
         << std::setw(16) << config_hash << '.'
         << std::setw(16) << index_hash;
    return name.str();
}

static Result<void> qwen_validate_artifact_tensor(
    const SafetensorsArchive& archive,
    const std::string& name,
    std::vector<uint32_t> shape)
{
    const SafetensorInfo* info = archive.find(name);
    if (!info)
        return Error{ErrorCode::InvalidModel, "Qwen MXFP4 artifact is missing tensor: " + name};
    uint64_t expected_size = 1;
    for (uint32_t dimension : shape)
    {
        if (dimension != 0
            && expected_size > std::numeric_limits<uint64_t>::max() / dimension)
        {
            return Error{ErrorCode::InvalidModel, "Qwen MXFP4 artifact tensor is too large: " + name};
        }
        expected_size *= dimension;
    }
    if (info->dtype != "U8"
        || info->shape != shape
        || info->size != expected_size)
    {
        return Error{ErrorCode::InvalidModel, "invalid Qwen MXFP4 artifact tensor: " + name};
    }
    return {};
}

static Result<void> qwen_validate_artifact_expert_bank(
    const SafetensorsArchive& archive,
    const std::string& prefix,
    uint32_t expert_count,
    uint32_t hidden_size,
    uint32_t intermediate_size)
{
    auto status = qwen_validate_artifact_tensor(
        archive,
        prefix + "gate_up.blocks",
        {expert_count, intermediate_size * 2, hidden_size / 32, 16});
    if (!status)
        return status.error();
    status = qwen_validate_artifact_tensor(
        archive,
        prefix + "gate_up.scales",
        {expert_count, intermediate_size * 2, hidden_size / 32});
    if (!status)
        return status.error();
    status = qwen_validate_artifact_tensor(
        archive,
        prefix + "down.blocks",
        {expert_count, hidden_size, intermediate_size / 32, 16});
    if (!status)
        return status.error();
    return qwen_validate_artifact_tensor(
        archive,
        prefix + "down.scales",
        {expert_count, hidden_size, intermediate_size / 32});
}

static Result<void> qwen_validate_mxfp4_artifact(
    const std::filesystem::path& model_root,
    uint32_t layer_count,
    uint32_t mtp_layer_count,
    uint32_t expert_count,
    uint32_t hidden_size,
    uint32_t intermediate_size)
{
    if (hidden_size % 32 != 0 || intermediate_size % 32 != 0)
        return Error{ErrorCode::InvalidModel, "Qwen MXFP4 artifact dimensions must be divisible by 32"};
    auto opened = SafetensorsArchive::open_file(model_root / qwen_mxfp4_artifact_name);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    auto config_hash = qwen_fnv1a64_file(model_root / "config.json");
    if (!config_hash)
        return config_hash.error();
    auto index_hash = qwen_fnv1a64_file(model_root / "model.safetensors.index.json");
    if (!index_hash)
        return index_hash.error();
    const std::string identity = qwen_mxfp4_identity_name(
        layer_count,
        mtp_layer_count,
        expert_count,
        hidden_size,
        intermediate_size,
        config_hash.value(),
        index_hash.value());
    auto status = qwen_validate_artifact_tensor(archive, identity, {0});
    if (!status)
    {
        return Error{
            ErrorCode::InvalidModel,
            status.error().message
                + "; rebuild the artifact for this exact checkpoint or remove it to use BF16 Experts"};
    }

    for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id)
    {
        status = qwen_validate_artifact_expert_bank(
            archive,
            qwen_mxfp4_expert_prefix(layer_id),
            expert_count,
            hidden_size,
            intermediate_size);
        if (!status)
            return status.error();
    }
    for (uint32_t layer_id = 0;
         layer_id < mtp_layer_count;
         ++layer_id)
    {
        status = qwen_validate_artifact_expert_bank(
            archive,
            qwen_mxfp4_mtp_expert_prefix(layer_id),
            expert_count,
            hidden_size,
            intermediate_size);
        if (!status)
            return status.error();
    }
    return {};
}

static Result<uint32_t> qwen_required_uint32(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "Qwen3 MoE manifest is missing integer field: " + key};
    try
    {
        const unsigned long long value = std::stoull(match[1].str());
        if (value > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, "Qwen3 MoE manifest integer is out of range: " + key};
        return static_cast<uint32_t>(value);
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid Qwen3 MoE integer field: " + key};
    }
}

static Result<float> qwen_required_float(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*([-+]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "Qwen3 MoE manifest is missing numeric field: " + key};
    try
    {
        return std::stof(match[1].str());
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid Qwen3 MoE numeric field: " + key};
    }
}

static Result<std::string> qwen_required_string(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "Qwen3 MoE manifest is missing string field: " + key};
    return match[1].str();
}

static Result<bool> qwen_required_bool(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "Qwen3 MoE manifest is missing boolean field: " + key};
    return match[1].str() == "true";
}

static Result<std::vector<std::string>> qwen_required_string_array(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "Qwen3 MoE manifest is missing string array: " + key};
    const std::regex value_expression("\\\"([^\\\"]+)\\\"");
    std::vector<std::string> values;
    for (std::sregex_iterator iterator(match[1].first, match[1].second, value_expression), end; iterator != end; ++iterator)
        values.push_back((*iterator)[1].str());
    if (values.empty())
        return Error{ErrorCode::InvalidModel, "Qwen3 MoE string array is empty: " + key};
    return values;
}

bool Qwen3_5MoeModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "qwen3_5_moe";
}

Result<MoeIR> Qwen3_5MoeModelAdapter::parse_model(const ModelPackage& package) const
{
    if (package.manifest.model_type != "qwen3_5_moe")
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen model_type: " + package.manifest.model_type};

    const std::string& json = package.manifest.raw_json;
    auto vocabulary_size = qwen_required_uint32(json, "vocab_size");
    auto hidden_size = qwen_required_uint32(json, "hidden_size");
    auto intermediate_size = qwen_required_uint32(json, "moe_intermediate_size");
    auto shared_intermediate_size = qwen_required_uint32(json, "shared_expert_intermediate_size");
    auto layer_count = qwen_required_uint32(json, "num_hidden_layers");
    auto mtp_layer_count = qwen_required_uint32(json, "mtp_num_hidden_layers");
    auto expert_count = qwen_required_uint32(json, "num_experts");
    auto top_k = qwen_required_uint32(json, "num_experts_per_tok");
    auto attention_head_count = qwen_required_uint32(json, "num_attention_heads");
    auto kv_head_count = qwen_required_uint32(json, "num_key_value_heads");
    auto head_dimension = qwen_required_uint32(json, "head_dim");
    auto linear_key_head_count = qwen_required_uint32(json, "linear_num_key_heads");
    auto linear_value_head_count = qwen_required_uint32(json, "linear_num_value_heads");
    auto linear_key_head_dimension = qwen_required_uint32(json, "linear_key_head_dim");
    auto linear_value_head_dimension = qwen_required_uint32(json, "linear_value_head_dim");
    auto convolution_kernel_size = qwen_required_uint32(json, "linear_conv_kernel_dim");
    auto maximum_context = qwen_required_uint32(json, "max_position_embeddings");
    auto norm_epsilon = qwen_required_float(json, "rms_norm_eps");
    auto rope_theta = qwen_required_float(json, "rope_theta");
    auto partial_rotary_factor = qwen_required_float(json, "partial_rotary_factor");
    auto layer_types = qwen_required_string_array(json, "layer_types");
    auto activation = qwen_required_string(json, "hidden_act");
    auto activation_dtype = qwen_required_string(json, "dtype");
    auto state_dtype = qwen_required_string(json, "mamba_ssm_dtype");
    auto attention_bias = qwen_required_bool(json, "attention_bias");
    auto attention_output_gate = qwen_required_bool(json, "attn_output_gate");
    auto mtp_dedicated_embeddings = qwen_required_bool(json, "mtp_use_dedicated_embeddings");

    if (!vocabulary_size || !hidden_size || !intermediate_size || !shared_intermediate_size
        || !layer_count || !mtp_layer_count || !expert_count || !top_k
        || !attention_head_count || !kv_head_count || !head_dimension
        || !linear_key_head_count || !linear_value_head_count
        || !linear_key_head_dimension || !linear_value_head_dimension
        || !convolution_kernel_size || !maximum_context
        || !norm_epsilon || !rope_theta || !partial_rotary_factor
        || !layer_types || !activation || !activation_dtype || !state_dtype
        || !attention_bias || !attention_output_gate
        || !mtp_dedicated_embeddings)
    {
        const Error* error = !vocabulary_size               ? &vocabulary_size.error()
                             : !hidden_size                 ? &hidden_size.error()
                             : !intermediate_size           ? &intermediate_size.error()
                             : !shared_intermediate_size    ? &shared_intermediate_size.error()
                             : !layer_count                 ? &layer_count.error()
                             : !mtp_layer_count             ? &mtp_layer_count.error()
                             : !expert_count                ? &expert_count.error()
                             : !top_k                       ? &top_k.error()
                             : !attention_head_count        ? &attention_head_count.error()
                             : !kv_head_count               ? &kv_head_count.error()
                             : !head_dimension              ? &head_dimension.error()
                             : !linear_key_head_count       ? &linear_key_head_count.error()
                             : !linear_value_head_count     ? &linear_value_head_count.error()
                             : !linear_key_head_dimension   ? &linear_key_head_dimension.error()
                             : !linear_value_head_dimension ? &linear_value_head_dimension.error()
                             : !convolution_kernel_size     ? &convolution_kernel_size.error()
                             : !maximum_context             ? &maximum_context.error()
                             : !norm_epsilon                ? &norm_epsilon.error()
                             : !rope_theta                  ? &rope_theta.error()
                             : !partial_rotary_factor       ? &partial_rotary_factor.error()
                             : !layer_types                 ? &layer_types.error()
                             : !activation                  ? &activation.error()
                             : !activation_dtype            ? &activation_dtype.error()
                             : !state_dtype                 ? &state_dtype.error()
                             : !attention_bias              ? &attention_bias.error()
                             : !attention_output_gate       ? &attention_output_gate.error()
                                                            : &mtp_dedicated_embeddings.error();
        return *error;
    }

    const float rotary_dimension_value = static_cast<float>(head_dimension.value()) * partial_rotary_factor.value();
    const uint32_t rotary_dimension = static_cast<uint32_t>(std::round(rotary_dimension_value));
    if (layer_types.value().size() != layer_count.value()
        || shared_intermediate_size.value() != intermediate_size.value()
        || attention_head_count.value() % kv_head_count.value() != 0
        || linear_value_head_count.value() % linear_key_head_count.value() != 0
        || rotary_dimension == 0
        || rotary_dimension > head_dimension.value()
        || rotary_dimension % 2 != 0
        || std::fabs(rotary_dimension_value - static_cast<float>(rotary_dimension)) > 1e-4f
        || activation.value() != "silu"
        || activation_dtype.value() != "bfloat16"
        || state_dtype.value() != "float32"
        || attention_bias.value()
        || !attention_output_gate.value()
        || mtp_layer_count.value() != 1
        || mtp_dedicated_embeddings.value())
    {
        return Error{ErrorCode::InvalidModel, "unsupported Qwen3 MoE architectural dimensions"};
    }

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
    full_attention.flags = AttentionDescriptorQueryKeyNorm | AttentionDescriptorOutputGate;

    AttentionDescriptor linear_attention;
    linear_attention.kind = AttentionKind::GatedDeltaNet;
    linear_attention.head_count = linear_value_head_count.value();
    linear_attention.kv_head_count = linear_key_head_count.value();
    linear_attention.head_dimension = linear_key_head_dimension.value();
    linear_attention.value_head_dimension = linear_value_head_dimension.value();
    linear_attention.convolution_kernel_size = convolution_kernel_size.value();
    linear_attention.max_context_length = maximum_context.value();

    MoeDescriptor moe;
    moe.expert_count = expert_count.value();
    moe.top_k = top_k.value();
    moe.intermediate_size = intermediate_size.value();
    moe.shared_expert_count = 1;
    moe.score_function = RouterScoreFunction::Softmax;
    moe.normalization = RouterNormalization::SelectedExperts;
    moe.activation = ExpertActivation::Silu;
    moe.layout = ExpertLayout::PackedGateUpDown;
    const std::filesystem::path artifact_path = package.root / qwen_mxfp4_artifact_name;
    std::error_code artifact_error;
    const bool artifact_exists = std::filesystem::exists(artifact_path, artifact_error);
    if (artifact_error)
        return Error{ErrorCode::IoError, "cannot inspect the optional Qwen MXFP4 artifact"};
    if (artifact_exists && !std::filesystem::is_regular_file(artifact_path, artifact_error))
        return Error{ErrorCode::InvalidModel, "the optional Qwen MXFP4 artifact path is not a regular file"};
    if (artifact_error)
        return Error{ErrorCode::IoError, "cannot inspect the optional Qwen MXFP4 artifact"};
    if (artifact_exists)
    {
        auto artifact_status = qwen_validate_mxfp4_artifact(
            package.root,
            layer_count.value(),
            mtp_layer_count.value(),
            expert_count.value(),
            hidden_size.value(),
            intermediate_size.value());
        if (!artifact_status)
            return artifact_status.error();
    }
    const bool compiled_mxfp4_experts = artifact_exists;
    std::optional<DType> detected_qnk_expert_dtype;
    if (!compiled_mxfp4_experts && !package.root.empty())
    {
        std::error_code archive_error;
        if (std::filesystem::is_directory(package.root, archive_error) && !archive_error)
        {
            auto opened = SafetensorsArchive::open(package.root);
            if (opened)
            {
                SafetensorsArchive& archive = opened.value();
                for (uint32_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
                {
                    const std::string source = "model.language_model.layers." + std::to_string(layer_id) + ".mlp.experts.";
                    const auto gate_up_dtype = archive.find_qnk_expert_dtype(
                        source + "gate_up_proj",
                        expert_count.value(),
                        intermediate_size.value() * 2,
                        hidden_size.value());
                    const auto down_dtype = archive.find_qnk_expert_dtype(
                        source + "down_proj",
                        expert_count.value(),
                        hidden_size.value(),
                        intermediate_size.value());
                    if (!gate_up_dtype || !down_dtype || gate_up_dtype != down_dtype)
                    {
                        detected_qnk_expert_dtype.reset();
                        break;
                    }
                    if (!detected_qnk_expert_dtype)
                        detected_qnk_expert_dtype = gate_up_dtype;
                    else if (detected_qnk_expert_dtype != gate_up_dtype)
                    {
                        detected_qnk_expert_dtype.reset();
                        break;
                    }
                }
            }
        }
    }
    moe.expert_weight_dtype = compiled_mxfp4_experts
                                  ? DType::MxFp4
                                  : detected_qnk_expert_dtype.value_or(DType::BFloat16);
    moe.shared_expert_weight_dtype = DType::BFloat16;
    moe.flags = MoeDescriptorNormalizeTopKWeights | MoeDescriptorSharedExpert | MoeDescriptorSharedExpertGate;

    MoeIR descriptor;
    descriptor.model_type = "qwen3_5_moe";
    descriptor.vocabulary_size = vocabulary_size.value();
    descriptor.hidden_size = hidden_size.value();
    descriptor.intermediate_size = intermediate_size.value();
    descriptor.layer_count = layer_count.value();
    descriptor.attention_head_count = attention_head_count.value();
    descriptor.kv_head_count = kv_head_count.value();
    descriptor.head_dimension = head_dimension.value();
    descriptor.expert_count = expert_count.value();
    descriptor.experts_per_token = top_k.value();
    descriptor.activation_dtype = DType::BFloat16;
    descriptor.kv_cache_dtype = DType::BFloat16;
    descriptor.norm_epsilon = norm_epsilon.value();
    descriptor.norm_weight_offset = 1.0f;
    if (compiled_mxfp4_experts)
    {
        descriptor.speculative_kind = SpeculativeModelKind::Mtp;
        descriptor.speculative_layer_count = mtp_layer_count.value();
        descriptor.speculative_block_size = 2;
    }
    descriptor.layers.resize(descriptor.layer_count);
    for (uint32_t layer_id = 0; layer_id < descriptor.layer_count; ++layer_id)
    {
        const std::string& layer_type = layer_types.value()[layer_id];
        if (layer_type != "linear_attention" && layer_type != "full_attention")
            return Error{ErrorCode::InvalidModel, "unsupported Qwen3 MoE layer type: " + layer_type};
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.flags = LayerDescriptorAttention | LayerDescriptorMoe;
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention = layer_type == "linear_attention" ? linear_attention : full_attention;
        layer.ffn.moe = moe;
    }
    return descriptor;
}

static Result<void> qwen_add_tensor(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& target,
    const std::string& source)
{
    auto tensor = archive.load_tensor(source);
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(target, std::move(tensor).value());
    return {};
}

static Result<void> qwen_add_expert_slice(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& target,
    const std::string& source,
    uint32_t expert_id,
    std::vector<uint32_t> shape)
{
    auto tensor = archive.load_bfloat16_slice(source, expert_id, std::move(shape));
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(target, std::move(tensor).value());
    return {};
}

static Result<void> qwen_add_qnk_expert(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& target,
    const std::string& source,
    DType dtype,
    uint32_t expert_id,
    uint32_t expert_count,
    uint32_t rows,
    uint32_t columns)
{
    auto tensor = archive.load_qnk_expert(
        source,
        dtype,
        expert_id,
        expert_count,
        rows,
        columns);
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(target, std::move(tensor).value());
    return {};
}

static Result<void> qwen_add_mxfp4_expert(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& target,
    const std::string& source,
    uint32_t expert_id,
    uint32_t rows,
    uint32_t columns,
    uint32_t flags)
{
    auto tensor = archive.load_mxfp4_expert(
        source + "blocks",
        source + "scales",
        expert_id,
        rows,
        columns,
        flags);
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(target, std::move(tensor).value());
    return {};
}

static Result<void> qwen_add_query_gate(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& source,
    const std::string& target_prefix,
    uint32_t head_count,
    uint32_t head_dimension,
    uint32_t hidden_size)
{
    auto loaded = archive.load_tensor(source);
    if (!loaded)
        return loaded.error();
    const TensorData& combined = loaded.value();
    const uint32_t combined_head_dimension = head_dimension * 2;
    if (combined.dtype != DType::BFloat16
        || combined.shape != std::vector<uint32_t>{head_count * combined_head_dimension, hidden_size}
        || combined.bfloat16_values().size() != combined.element_count())
    {
        return Error{ErrorCode::InvalidModel, "invalid interleaved Qwen query/gate tensor: " + source};
    }

    TensorData query;
    query.dtype = DType::BFloat16;
    query.shape = {head_count * head_dimension, hidden_size};
    query.bfloat16_data.resize(static_cast<size_t>(head_count) * head_dimension * hidden_size);
    TensorData gate;
    gate.dtype = DType::BFloat16;
    gate.shape = query.shape;
    gate.bfloat16_data.resize(query.bfloat16_data.size());
    const std::span<const uint16_t> values = combined.bfloat16_values();
    const size_t head_elements = static_cast<size_t>(head_dimension) * hidden_size;
    const size_t combined_head_elements = head_elements * 2;
    for (uint32_t head = 0; head < head_count; ++head)
    {
        const uint16_t* source_head = values.data() + static_cast<size_t>(head) * combined_head_elements;
        std::copy_n(source_head, head_elements, query.bfloat16_data.data() + static_cast<size_t>(head) * head_elements);
        std::copy_n(source_head + head_elements, head_elements, gate.bfloat16_data.data() + static_cast<size_t>(head) * head_elements);
    }
    mapping.tensors.emplace(target_prefix + "attention.query.weight", std::move(query));
    mapping.tensors.emplace(target_prefix + "attention.output_gate.weight", std::move(gate));
    return {};
}

Result<WeightMapping> Qwen3_5MoeModelAdapter::map_weights(const ModelPackage& package, const MoeIR& descriptor) const
{
    if (descriptor.model_type != "qwen3_5_moe")
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen model_type: " + descriptor.model_type};
    auto opened = SafetensorsArchive::open(package.root);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    WeightMapping mapping;
    uint32_t expert_load_flags = 0;
    if (has_flag(package.flags, ModelPackageDeferMxfp4Experts))
        expert_load_flags |= SafetensorLoadDeferMxfp4Data;
    auto status = qwen_add_tensor(
        mapping,
        archive,
        "token_embedding.weight",
        "model.language_model.embed_tokens.weight");
    if (!status)
        return status.error();
    status = qwen_add_tensor(
        mapping,
        archive,
        "final_norm.weight",
        "model.language_model.norm.weight");
    if (!status)
        return status.error();
    status = qwen_add_tensor(mapping, archive, "lm_head.weight", "lm_head.weight");
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < descriptor.layer_count; ++layer_id)
    {
        const std::string source = "model.language_model.layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        status = qwen_add_tensor(
            mapping,
            archive,
            target + "pre_attention_norm.weight",
            source + "input_layernorm.weight");
        if (!status)
            return status.error();
        status = qwen_add_tensor(
            mapping,
            archive,
            target + "pre_ffn_norm.weight",
            source + "post_attention_layernorm.weight");
        if (!status)
            return status.error();
        status = qwen_add_tensor(
            mapping,
            archive,
            target + "router.weight",
            source + "mlp.gate.weight");
        if (!status)
            return status.error();
        const std::pair<const char*, const char*> shared_tensors[] = {
            {"shared_expert.gate.weight", "mlp.shared_expert.gate_proj.weight"},
            {"shared_expert.up.weight", "mlp.shared_expert.up_proj.weight"},
            {"shared_expert.down.weight", "mlp.shared_expert.down_proj.weight"},
            {"shared_expert.router_gate.weight", "mlp.shared_expert_gate.weight"},
        };
        for (const auto& item : shared_tensors)
        {
            status = qwen_add_tensor(mapping, archive, target + item.first, source + item.second);
            if (!status)
                return status.error();
        }

        const MoeDescriptor& moe = descriptor.layers[layer_id].ffn.moe;
        const bool compiled_mxfp4_experts = moe.expert_weight_dtype == DType::MxFp4;
        const bool compiled_qnk_experts = is_qnk_dtype(moe.expert_weight_dtype);
        const std::string artifact_experts = qwen_mxfp4_expert_prefix(layer_id);
        for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
        {
            const std::string expert = expert_prefix(layer_id, expert_id);
            if (compiled_qnk_experts)
            {
                status = qwen_add_qnk_expert(
                    mapping,
                    archive,
                    expert + "gate_up.weight",
                    source + "mlp.experts.gate_up_proj",
                    moe.expert_weight_dtype,
                    expert_id,
                    moe.expert_count,
                    moe.intermediate_size * 2,
                    descriptor.hidden_size);
            }
            else if (compiled_mxfp4_experts)
            {
                status = qwen_add_mxfp4_expert(
                    mapping,
                    archive,
                    expert + "gate_up.weight",
                    artifact_experts + "gate_up.",
                    expert_id,
                    moe.intermediate_size * 2,
                    descriptor.hidden_size,
                    expert_load_flags);
            }
            else
            {
                status = qwen_add_expert_slice(
                    mapping,
                    archive,
                    expert + "gate_up.weight",
                    source + "mlp.experts.gate_up_proj",
                    expert_id,
                    {moe.intermediate_size * 2, descriptor.hidden_size});
            }
            if (!status)
                return status.error();
            if (compiled_qnk_experts)
            {
                status = qwen_add_qnk_expert(
                    mapping,
                    archive,
                    expert + "down.weight",
                    source + "mlp.experts.down_proj",
                    moe.expert_weight_dtype,
                    expert_id,
                    moe.expert_count,
                    descriptor.hidden_size,
                    moe.intermediate_size);
            }
            else if (compiled_mxfp4_experts)
            {
                status = qwen_add_mxfp4_expert(
                    mapping,
                    archive,
                    expert + "down.weight",
                    artifact_experts + "down.",
                    expert_id,
                    descriptor.hidden_size,
                    moe.intermediate_size,
                    expert_load_flags);
            }
            else
            {
                status = qwen_add_expert_slice(
                    mapping,
                    archive,
                    expert + "down.weight",
                    source + "mlp.experts.down_proj",
                    expert_id,
                    {descriptor.hidden_size, moe.intermediate_size});
            }
            if (!status)
                return status.error();
        }

        const AttentionDescriptor& attention = descriptor.layers[layer_id].attention;
        if (attention.kind == AttentionKind::GatedDeltaNet)
        {
            const std::pair<const char*, const char*> delta_tensors[] = {
                {"attention.delta.qkv.weight", "linear_attn.in_proj_qkv.weight"},
                {"attention.delta.z.weight", "linear_attn.in_proj_z.weight"},
                {"attention.delta.beta.weight", "linear_attn.in_proj_b.weight"},
                {"attention.delta.alpha.weight", "linear_attn.in_proj_a.weight"},
                {"attention.delta.convolution.weight", "linear_attn.conv1d.weight"},
                {"attention.delta.time_bias", "linear_attn.dt_bias"},
                {"attention.delta.decay_log", "linear_attn.A_log"},
                {"attention.delta.norm.weight", "linear_attn.norm.weight"},
                {"attention.output.weight", "linear_attn.out_proj.weight"},
            };
            for (const auto& item : delta_tensors)
            {
                status = qwen_add_tensor(mapping, archive, target + item.first, source + item.second);
                if (!status)
                    return status.error();
            }
            continue;
        }

        status = qwen_add_query_gate(
            mapping,
            archive,
            source + "self_attn.q_proj.weight",
            target,
            attention.head_count,
            attention.head_dimension,
            descriptor.hidden_size);
        if (!status)
            return status.error();
        const std::pair<const char*, const char*> attention_tensors[] = {
            {"attention.key.weight", "self_attn.k_proj.weight"},
            {"attention.value.weight", "self_attn.v_proj.weight"},
            {"attention.output.weight", "self_attn.o_proj.weight"},
            {"attention.query_norm.weight", "self_attn.q_norm.weight"},
            {"attention.key_norm.weight", "self_attn.k_norm.weight"},
        };
        for (const auto& item : attention_tensors)
        {
            status = qwen_add_tensor(mapping, archive, target + item.first, source + item.second);
            if (!status)
                return status.error();
        }
    }

    if (descriptor.speculative_kind != SpeculativeModelKind::Mtp)
        return mapping;

    const std::pair<const char*, const char*> mtp_stem_tensors[] = {
        {"speculative.mtp.embedding_norm.weight", "mtp.pre_fc_norm_embedding.weight"},
        {"speculative.mtp.hidden_norm.weight", "mtp.pre_fc_norm_hidden.weight"},
        {"speculative.mtp.input_projection.weight", "mtp.fc.weight"},
        {"speculative.final_norm.weight", "mtp.norm.weight"},
    };
    for (const auto& item : mtp_stem_tensors)
    {
        status = qwen_add_tensor(
            mapping,
            archive,
            item.first,
            item.second);
        if (!status)
            return status.error();
    }

    for (uint32_t layer_id = 0;
         layer_id < descriptor.speculative_layer_count;
         ++layer_id)
    {
        const std::string source = "mtp.layers." + std::to_string(layer_id) + ".";
        const std::string target = speculative_layer_prefix(layer_id);
        status = qwen_add_tensor(
            mapping,
            archive,
            target + "pre_attention_norm.weight",
            source + "input_layernorm.weight");
        if (!status)
            return status.error();
        status = qwen_add_tensor(
            mapping,
            archive,
            target + "pre_ffn_norm.weight",
            source + "post_attention_layernorm.weight");
        if (!status)
            return status.error();
        status = qwen_add_tensor(
            mapping,
            archive,
            target + "router.weight",
            source + "mlp.gate.weight");
        if (!status)
            return status.error();

        const std::pair<const char*, const char*> shared_tensors[] = {
            {"shared_expert.gate.weight", "mlp.shared_expert.gate_proj.weight"},
            {"shared_expert.up.weight", "mlp.shared_expert.up_proj.weight"},
            {"shared_expert.down.weight", "mlp.shared_expert.down_proj.weight"},
            {"shared_expert.router_gate.weight", "mlp.shared_expert_gate.weight"},
        };
        for (const auto& item : shared_tensors)
        {
            status = qwen_add_tensor(
                mapping,
                archive,
                target + item.first,
                source + item.second);
            if (!status)
                return status.error();
        }

        const MoeDescriptor& moe = descriptor.layers.back().ffn.moe;
        const bool compiled_mxfp4_experts = moe.expert_weight_dtype == DType::MxFp4;
        const std::string artifact_experts = qwen_mxfp4_mtp_expert_prefix(layer_id);
        for (uint32_t expert_id = 0;
             expert_id < moe.expert_count;
             ++expert_id)
        {
            const std::string expert = speculative_expert_prefix(layer_id, expert_id);
            status = compiled_mxfp4_experts
                         ? qwen_add_mxfp4_expert(
                               mapping,
                               archive,
                               expert + "gate_up.weight",
                               artifact_experts + "gate_up.",
                               expert_id,
                               moe.intermediate_size * 2,
                               descriptor.hidden_size,
                               expert_load_flags)
                         : qwen_add_expert_slice(
                               mapping,
                               archive,
                               expert + "gate_up.weight",
                               source + "mlp.experts.gate_up_proj",
                               expert_id,
                               {moe.intermediate_size * 2,
                                descriptor.hidden_size});
            if (!status)
                return status.error();
            status = compiled_mxfp4_experts
                         ? qwen_add_mxfp4_expert(
                               mapping,
                               archive,
                               expert + "down.weight",
                               artifact_experts + "down.",
                               expert_id,
                               descriptor.hidden_size,
                               moe.intermediate_size,
                               expert_load_flags)
                         : qwen_add_expert_slice(
                               mapping,
                               archive,
                               expert + "down.weight",
                               source + "mlp.experts.down_proj",
                               expert_id,
                               {descriptor.hidden_size,
                                moe.intermediate_size});
            if (!status)
                return status.error();
        }

        const AttentionDescriptor& attention = descriptor.layers.back().attention;
        status = qwen_add_query_gate(
            mapping,
            archive,
            source + "self_attn.q_proj.weight",
            target,
            attention.head_count,
            attention.head_dimension,
            descriptor.hidden_size);
        if (!status)
            return status.error();
        const std::pair<const char*, const char*> attention_tensors[] = {
            {"attention.key.weight", "self_attn.k_proj.weight"},
            {"attention.value.weight", "self_attn.v_proj.weight"},
            {"attention.output.weight", "self_attn.o_proj.weight"},
            {"attention.query_norm.weight", "self_attn.q_norm.weight"},
            {"attention.key_norm.weight", "self_attn.k_norm.weight"},
        };
        for (const auto& item : attention_tensors)
        {
            status = qwen_add_tensor(
                mapping,
                archive,
                target + item.first,
                source + item.second);
            if (!status)
                return status.error();
        }
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
