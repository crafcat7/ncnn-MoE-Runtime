#include "modeladapter_qwen3_5.h"

#include "tensornames.h"
#include "modeladapter.h"
#include "safetensors.h"

#include <regex>
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
    auto status = validate_mxfp4_artifact_identity(
        archive,
        model_root,
        "__ncnn_moe_qwen3_6_mxfp4__.identity.v3.",
        layer_count,
        mtp_layer_count,
        expert_count,
        hidden_size,
        intermediate_size,
        "Qwen artifact identity source",
        "Qwen MXFP4 artifact");
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id)
    {
        status = validate_mxfp4_artifact_expert_bank(
            archive,
            qwen_mxfp4_expert_prefix(layer_id),
            expert_count,
            hidden_size,
            intermediate_size,
            "Qwen MXFP4 artifact");
        if (!status)
            return status.error();
    }
    for (uint32_t layer_id = 0;
         layer_id < mtp_layer_count;
         ++layer_id)
    {
        status = validate_mxfp4_artifact_expert_bank(
            archive,
            qwen_mxfp4_mtp_expert_prefix(layer_id),
            expert_count,
            hidden_size,
            intermediate_size,
            "Qwen MXFP4 artifact");
        if (!status)
            return status.error();
    }
    return {};
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

Result<MoeModelDescriptor> Qwen3_5MoeModelAdapter::parse_model(const ModelPackage& package) const
{
    if (package.manifest.model_type != "qwen3_5_moe")
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen model_type: " + package.manifest.model_type};

    const std::string& json = package.manifest.raw_json;
    auto vocabulary_size = read_manifest_uint32(json, "vocab_size", "Qwen3 MoE ");
    if (!vocabulary_size)
        return vocabulary_size.error();
    auto hidden_size = read_manifest_uint32(json, "hidden_size", "Qwen3 MoE ");
    if (!hidden_size)
        return hidden_size.error();
    auto intermediate_size = read_manifest_uint32(json, "moe_intermediate_size", "Qwen3 MoE ");
    if (!intermediate_size)
        return intermediate_size.error();
    auto shared_intermediate_size = read_manifest_uint32(json, "shared_expert_intermediate_size", "Qwen3 MoE ");
    if (!shared_intermediate_size)
        return shared_intermediate_size.error();
    auto layer_count = read_manifest_uint32(json, "num_hidden_layers", "Qwen3 MoE ");
    if (!layer_count)
        return layer_count.error();
    auto mtp_layer_count = read_manifest_uint32(json, "mtp_num_hidden_layers", "Qwen3 MoE ");
    if (!mtp_layer_count)
        return mtp_layer_count.error();
    auto expert_count = read_manifest_uint32(json, "num_experts", "Qwen3 MoE ");
    if (!expert_count)
        return expert_count.error();
    auto top_k = read_manifest_uint32(json, "num_experts_per_tok", "Qwen3 MoE ");
    if (!top_k)
        return top_k.error();
    auto attention_head_count = read_manifest_uint32(json, "num_attention_heads", "Qwen3 MoE ");
    if (!attention_head_count)
        return attention_head_count.error();
    auto kv_head_count = read_manifest_uint32(json, "num_key_value_heads", "Qwen3 MoE ");
    if (!kv_head_count)
        return kv_head_count.error();
    auto head_dimension = read_manifest_uint32(json, "head_dim", "Qwen3 MoE ");
    if (!head_dimension)
        return head_dimension.error();
    auto linear_key_head_count = read_manifest_uint32(json, "linear_num_key_heads", "Qwen3 MoE ");
    if (!linear_key_head_count)
        return linear_key_head_count.error();
    auto linear_value_head_count = read_manifest_uint32(json, "linear_num_value_heads", "Qwen3 MoE ");
    if (!linear_value_head_count)
        return linear_value_head_count.error();
    auto linear_key_head_dimension = read_manifest_uint32(json, "linear_key_head_dim", "Qwen3 MoE ");
    if (!linear_key_head_dimension)
        return linear_key_head_dimension.error();
    auto linear_value_head_dimension = read_manifest_uint32(json, "linear_value_head_dim", "Qwen3 MoE ");
    if (!linear_value_head_dimension)
        return linear_value_head_dimension.error();
    auto convolution_kernel_size = read_manifest_uint32(json, "linear_conv_kernel_dim", "Qwen3 MoE ");
    if (!convolution_kernel_size)
        return convolution_kernel_size.error();
    auto maximum_context = read_manifest_uint32(json, "max_position_embeddings", "Qwen3 MoE ");
    if (!maximum_context)
        return maximum_context.error();
    auto norm_epsilon = read_manifest_float(json, "rms_norm_eps", "Qwen3 MoE ");
    if (!norm_epsilon)
        return norm_epsilon.error();
    auto rope_theta = read_manifest_float(json, "rope_theta", "Qwen3 MoE ");
    if (!rope_theta)
        return rope_theta.error();
    auto partial_rotary_factor = read_manifest_float(json, "partial_rotary_factor", "Qwen3 MoE ");
    if (!partial_rotary_factor)
        return partial_rotary_factor.error();
    auto layer_types = qwen_required_string_array(json, "layer_types");
    if (!layer_types)
        return layer_types.error();
    auto activation = read_manifest_string(json, "hidden_act", "Qwen3 MoE ");
    if (!activation)
        return activation.error();
    auto activation_dtype = read_manifest_string(json, "dtype", "Qwen3 MoE ");
    if (!activation_dtype)
        return activation_dtype.error();
    auto state_dtype = read_manifest_string(json, "mamba_ssm_dtype", "Qwen3 MoE ");
    if (!state_dtype)
        return state_dtype.error();
    auto attention_bias = read_manifest_bool(json, "attention_bias", "Qwen3 MoE ");
    if (!attention_bias)
        return attention_bias.error();
    auto attention_output_gate = read_manifest_bool(json, "attn_output_gate", "Qwen3 MoE ");
    if (!attention_output_gate)
        return attention_output_gate.error();
    auto mtp_dedicated_embeddings = read_manifest_bool(json, "mtp_use_dedicated_embeddings", "Qwen3 MoE ");
    if (!mtp_dedicated_embeddings)
        return mtp_dedicated_embeddings.error();

    if (kv_head_count.value() == 0 || linear_key_head_count.value() == 0)
        return Error{ErrorCode::InvalidModel, "unsupported Qwen3 MoE architectural dimensions"};

    auto rotary_dimension_result = get_rotary_dimension(
        head_dimension.value(), partial_rotary_factor.value(),
        "unsupported Qwen3 MoE architectural dimensions");
    if (!rotary_dimension_result)
        return rotary_dimension_result.error();
    const uint32_t rotary_dimension = rotary_dimension_result.value();
    if (layer_types.value().size() != layer_count.value()
        || shared_intermediate_size.value() != intermediate_size.value()
        || attention_head_count.value() % kv_head_count.value() != 0
        || linear_value_head_count.value() % linear_key_head_count.value() != 0
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
    auto artifact_status = optional_artifact_exists(
        package.root / qwen_mxfp4_artifact_name, "Qwen MXFP4 artifact");
    if (!artifact_status)
        return artifact_status.error();
    const bool artifact_exists = artifact_status.value();
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
    std::optional<DType> detected_qnk_expert_dtype;
    if (!artifact_exists && !package.root.empty())
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
    moe.expert_weight_dtype = artifact_exists
                                  ? DType::MxFp4
                                  : detected_qnk_expert_dtype.value_or(DType::BFloat16);
    moe.shared_expert_weight_dtype = DType::BFloat16;
    moe.flags = MoeDescriptorSharedExpertGate;

    MoeModelDescriptor descriptor;
    descriptor.model_type = "qwen3_5_moe";
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
    if (artifact_exists)
    {
        descriptor.speculative_kind = SpeculativeModelKind::Mtp;
        descriptor.speculative_layer_count = mtp_layer_count.value();
        descriptor.speculative_block_size = 2;
    }
    descriptor.layers.resize(layer_count.value());
    for (uint32_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
    {
        const std::string& layer_type = layer_types.value()[layer_id];
        if (layer_type != "linear_attention" && layer_type != "full_attention")
            return Error{ErrorCode::InvalidModel, "unsupported Qwen3 MoE layer type: " + layer_type};
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention = layer_type == "linear_attention" ? linear_attention : full_attention;
        layer.ffn.moe = moe;
    }
    return descriptor;
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
    mapping.emplace(target, std::move(tensor).value());
    return {};
}

Result<WeightMapping> Qwen3_5MoeModelAdapter::map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const
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
    auto status = add_tensor(
        mapping,
        archive,
        "token_embedding.weight",
        "model.language_model.embed_tokens.weight");
    if (!status)
        return status.error();
    status = add_tensor(
        mapping,
        archive,
        "final_norm.weight",
        "model.language_model.norm.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "lm_head.weight", "lm_head.weight");
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
    {
        const std::string source = "model.language_model.layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        status = add_tensor(
            mapping,
            archive,
            target + "pre_attention_norm.weight",
            source + "input_layernorm.weight");
        if (!status)
            return status.error();
        status = add_tensor(
            mapping,
            archive,
            target + "pre_ffn_norm.weight",
            source + "post_attention_layernorm.weight");
        if (!status)
            return status.error();
        status = add_tensor(
            mapping,
            archive,
            target + "router.weight",
            source + "mlp.gate.weight");
        if (!status)
            return status.error();
        status = add_qwen_shared_expert(mapping, archive, source, target);
        if (!status)
            return status.error();

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
            }
            else
            {
                status = add_bfloat16_slice(
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
            }
            else
            {
                status = add_bfloat16_slice(
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
            status = add_qwen_gated_delta_net(mapping, archive, source, target);
            if (!status)
                return status.error();
            continue;
        }

        status = add_qwen_attention(
            mapping, archive, source, target,
            attention.head_count, attention.head_dimension,
            descriptor.hidden_size, "Qwen");
        if (!status)
            return status.error();
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
        status = add_tensor(
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
        status = add_tensor(
            mapping,
            archive,
            target + "pre_attention_norm.weight",
            source + "input_layernorm.weight");
        if (!status)
            return status.error();
        status = add_tensor(
            mapping,
            archive,
            target + "pre_ffn_norm.weight",
            source + "post_attention_layernorm.weight");
        if (!status)
            return status.error();
        status = add_tensor(
            mapping,
            archive,
            target + "router.weight",
            source + "mlp.gate.weight");
        if (!status)
            return status.error();

        status = add_qwen_shared_expert(mapping, archive, source, target);
        if (!status)
            return status.error();

        const MoeDescriptor& moe = descriptor.layers.back().ffn.moe;
        const bool compiled_mxfp4_experts = moe.expert_weight_dtype == DType::MxFp4;
        const std::string artifact_experts = qwen_mxfp4_mtp_expert_prefix(layer_id);
        for (uint32_t expert_id = 0;
             expert_id < moe.expert_count;
             ++expert_id)
        {
            const std::string expert = speculative_expert_prefix(layer_id, expert_id);
            status = compiled_mxfp4_experts
                         ? add_mxfp4_expert(
                               mapping,
                               archive,
                               expert + "gate_up.weight",
                               artifact_experts + "gate_up.blocks",
                               artifact_experts + "gate_up.scales",
                               expert_id,
                               moe.intermediate_size * 2,
                               descriptor.hidden_size,
                               expert_load_flags)
                         : add_bfloat16_slice(
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
                         ? add_mxfp4_expert(
                               mapping,
                               archive,
                               expert + "down.weight",
                               artifact_experts + "down.blocks",
                               artifact_experts + "down.scales",
                               expert_id,
                               descriptor.hidden_size,
                               moe.intermediate_size,
                               expert_load_flags)
                         : add_bfloat16_slice(
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
        status = add_qwen_attention(
            mapping, archive, source, target,
            attention.head_count, attention.head_dimension,
            descriptor.hidden_size, "Qwen");
        if (!status)
            return status.error();
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
