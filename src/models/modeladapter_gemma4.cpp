#include "modeladapter_gemma4.h"

#include "json.h"
#include "kernels/bfloat16.h"
#include "modeladapter.h"
#include "safetensors.h"
#include "tensornames.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

// Reference constants from transformers' Gemma4 implementation:
// - Gemma4TextAttention hardcodes self.scaling = 1.0 (not 1/sqrt(head_dim)).
// - Gemma4TextScaledWordEmbedding multiplies embeddings by sqrt(hidden_size).
// - final_logit_softcapping defaults to None, i.e. no soft cap when absent.
static constexpr float attn_scale = 1.0f;
static constexpr float softcap_default = 0.0f;

// The Vulkan gated-down kernel tiles the intermediate dimension by 128.
// TODO: this is a backend tiling detail; move the padding into the Vulkan
// weight preparation once the backend can pad internally.
static constexpr uint32_t ffn_tile = 128;

static bool opt_bool(const std::string& json, const char* key, bool fallback)
{
    if (!find_manifest_member(json, key))
        return fallback;
    auto value = read_manifest_bool(json, key, "Gemma4 ");
    return value ? value.value() : fallback;
}

static float scalar_at(const TensorData& tensor, size_t index)
{
    if (tensor.dtype == DType::Float32)
        return tensor.float32_values()[index];
    if (tensor.dtype == DType::BFloat16)
        return bfloat16_to_float(tensor.bfloat16_values()[index]);
    return 0.0f;
}

static Result<void> add_scaled(WeightMapping& mapping,
                               const SafetensorsArchive& archive,
                               const std::string& target,
                               const std::string& source,
                               uint32_t count,
                               float multiplier)
{
    auto loaded = archive.load_tensor(source);
    if (!loaded)
        return loaded.error();
    const TensorData& tensor = loaded.value();
    if ((tensor.dtype != DType::Float32 && tensor.dtype != DType::BFloat16)
        || tensor.element_count() != count)
        return Error{ErrorCode::InvalidModel, "invalid Gemma4 vector: " + source};

    TensorData scaled;
    scaled.dtype = DType::Float32;
    scaled.shape = {count};
    scaled.float32_data.resize(count);
    for (uint32_t index = 0; index < count; ++index)
        scaled.float32_data[index] = scalar_at(tensor, index) * multiplier;
    mapping.emplace(target, std::move(scaled));
    return {};
}

static Result<void> add_ones(WeightMapping& mapping,
                             const std::string& target,
                             uint32_t count,
                             DType dtype)
{
    TensorData ones;
    ones.dtype = dtype;
    ones.shape = {count};
    if (dtype == DType::Float32)
    {
        ones.float32_data.assign(count, 1.0f);
    }
    else if (dtype == DType::BFloat16)
    {
        ones.bfloat16_data.assign(count, 0x3f80);
    }
    else
    {
        return Error{ErrorCode::UnsupportedModel, "Gemma4 value norm requires BF16 or F32 storage"};
    }
    mapping.emplace(target, std::move(ones));
    return {};
}

struct RopeConfig
{
    float theta = 10000.0f;
    float partial = 1.0f;
    std::string type = "default";
};

static Result<RopeConfig> read_rope(const std::string& json, const char* name)
{
    const auto raw = find_manifest_member(json, name);
    if (!raw)
        return Error{ErrorCode::InvalidModel, std::string("Gemma4 rope_parameters is missing ") + name};
    const std::string object(*raw);
    RopeConfig result;
    auto theta = read_manifest_float(object, "rope_theta", "Gemma4 RoPE ");
    if (!theta)
        return theta.error();
    result.theta = theta.value();
    result.partial = optional_manifest_float(object, "partial", 1.0f);
    auto type = read_manifest_string(object, "rope_type", "Gemma4 RoPE ");
    if (!type)
        return type.error();
    result.type = type.value();
    return result;
}

static constexpr const char* artifact_name = "ncnn-moe-gemma4-26b-a4b-mxfp4.safetensors";
static constexpr const char* artifact_prefix = "__ncnn_moe_gemma4_mxfp4__";
static constexpr const char* artifact_version = "v1";

static std::string expert_prefix(uint32_t layer_id)
{
    return std::string(artifact_prefix) + ".layers." + std::to_string(layer_id) + ".experts.";
}

static Result<bool> artifact_exists(const ModelPackage& package)
{
    if (package.root.empty() || !package.weights_path.empty())
        return false;
    return optional_artifact_exists(package.root / artifact_name, "Gemma4 MXFP4 artifact");
}

static Result<void> validate_artifact(const ModelPackage& package,
                                      uint32_t layer_count,
                                      uint32_t expert_count,
                                      uint32_t hidden_size,
                                      uint32_t intermediate_size)
{
    if (hidden_size % 32 != 0 || intermediate_size % 32 != 0)
        return Error{ErrorCode::InvalidModel, "Gemma4 MXFP4 artifact dimensions must be divisible by 32"};
    auto opened = SafetensorsArchive::open_file(package.root / artifact_name);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    const std::string identity_prefix = std::string(artifact_prefix) + ".identity."
                                        + artifact_version + ".";
    auto status = validate_artifact_id(archive,
                                       package,
                                       identity_prefix.c_str(),
                                       layer_count,
                                       0,
                                       expert_count,
                                       hidden_size,
                                       intermediate_size,
                                       "Gemma4 artifact identity source",
                                       "Gemma4 MXFP4 artifact");
    if (!status)
        return status.error();
    for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id)
    {
        status = validate_artifact_bank(archive,
                                        expert_prefix(layer_id),
                                        expert_count,
                                        hidden_size,
                                        intermediate_size,
                                        "Gemma4 MXFP4 artifact");
        if (!status)
            return status.error();
    }
    return {};
}

static Result<void> add_padded_bf16(WeightMapping& mapping,
                                    const SafetensorsArchive& archive,
                                    const std::string& target,
                                    const std::string& source,
                                    uint32_t rows,
                                    uint32_t columns)
{
    auto loaded = archive.load_tensor(source);
    if (!loaded)
        return loaded.error();
    const TensorData& tensor = loaded.value();
    if (tensor.dtype != DType::BFloat16 || tensor.shape.size() != 2
        || tensor.shape[0] == 0 || tensor.shape[1] == 0
        || tensor.shape[0] > rows || tensor.shape[1] > columns)
        return Error{ErrorCode::InvalidModel, "invalid Gemma4 padded dense FFN tensor: " + source};
    const std::span<const uint16_t> values = tensor.bfloat16_values();
    if (values.size() != static_cast<size_t>(tensor.shape[0]) * tensor.shape[1])
        return Error{ErrorCode::InvalidModel, "invalid Gemma4 padded dense FFN tensor: " + source};
    TensorData padded;
    padded.dtype = DType::BFloat16;
    padded.shape = {rows, columns};
    padded.bfloat16_data.assign(static_cast<size_t>(rows) * columns, 0);
    for (uint32_t row = 0; row < tensor.shape[0]; ++row)
    {
        std::copy_n(values.begin() + static_cast<size_t>(row) * tensor.shape[1],
                    tensor.shape[1],
                    padded.bfloat16_data.begin() + static_cast<size_t>(row) * columns);
    }
    mapping.emplace(target, std::move(padded));
    return {};
}

bool Gemma4ModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "gemma4" || manifest.model_type == "gemma4_text";
}

Result<MoeModelDescriptor> Gemma4ModelAdapter::parse_model(const ModelPackage& package) const
{
    if (!can_load(package.manifest))
        return Error{ErrorCode::UnsupportedModel, "unsupported Gemma4 model_type: " + package.manifest.model_type};

    const std::string& root_json = package.manifest.raw_json;
    auto text_member = find_manifest_member(root_json, "text_config");
    const std::string text_json = text_member ? std::string(*text_member) : root_json;
    auto text_type = read_manifest_string(text_json, "model_type", "Gemma4 ");
    if (!text_type)
        return text_type.error();
    if (text_type.value() != "gemma4_text" && text_type.value() != "gemma4")
        return Error{ErrorCode::InvalidModel, "unexpected Gemma4 text model_type: " + text_type.value()};

    auto vocabulary_size = read_manifest_uint32(text_json, "vocab_size", "Gemma4 ");
    if (!vocabulary_size)
        return vocabulary_size.error();
    auto hidden_size = read_manifest_uint32(text_json, "hidden_size", "Gemma4 ");
    if (!hidden_size)
        return hidden_size.error();
    auto dense_intermediate_size = read_manifest_uint32(text_json, "intermediate_size", "Gemma4 ");
    if (!dense_intermediate_size)
        return dense_intermediate_size.error();
    auto layer_count = read_manifest_uint32(text_json, "num_hidden_layers", "Gemma4 ");
    if (!layer_count)
        return layer_count.error();
    auto attention_head_count = read_manifest_uint32(text_json, "num_attention_heads", "Gemma4 ");
    if (!attention_head_count)
        return attention_head_count.error();
    auto head_dimension = read_manifest_uint32(text_json, "head_dim", "Gemma4 ");
    if (!head_dimension)
        return head_dimension.error();
    auto kv_head_count = read_manifest_uint32(text_json, "num_key_value_heads", "Gemma4 ");
    if (!kv_head_count)
        return kv_head_count.error();
    const uint32_t global_head_dimension = find_manifest_member(text_json, "global_head_dim")
                                               ? read_manifest_uint32(text_json, "global_head_dim", "Gemma4 ").value()
                                               : head_dimension.value();
    const uint32_t global_kv_head_count = find_manifest_member(text_json, "num_global_key_value_heads")
                                              ? read_manifest_uint32(text_json, "num_global_key_value_heads", "Gemma4 ").value()
                                              : kv_head_count.value();
    auto maximum_context = read_manifest_uint32(text_json, "max_position_embeddings", "Gemma4 ");
    if (!maximum_context)
        return maximum_context.error();
    auto norm_epsilon = read_manifest_float(text_json, "rms_norm_eps", "Gemma4 ");
    if (!norm_epsilon)
        return norm_epsilon.error();
    auto expert_count = read_manifest_uint32(text_json, "num_experts", "Gemma4 ");
    if (!expert_count)
        return expert_count.error();
    auto top_k = read_manifest_uint32(text_json, "top_k_experts", "Gemma4 ");
    if (!top_k)
        return top_k.error();
    auto moe_intermediate_size = read_manifest_uint32(text_json, "moe_intermediate_size", "Gemma4 ");
    if (!moe_intermediate_size)
        return moe_intermediate_size.error();
    auto sliding_window = read_manifest_uint32(text_json, "sliding_window", "Gemma4 ");
    if (!sliding_window)
        return sliding_window.error();
    auto layer_types = read_manifest_string_array(text_json, "layer_types", "Gemma4 ");
    if (!layer_types)
        return layer_types.error();
    auto activation = read_manifest_string(text_json, "hidden_activation", "Gemma4 ");
    if (!activation)
        return activation.error();
    auto attention_bias = read_manifest_bool(text_json, "attention_bias", "Gemma4 ");
    if (!attention_bias)
        return attention_bias.error();
    auto attention_k_eq_v = read_manifest_bool(text_json, "attention_k_eq_v", "Gemma4 ");
    if (!attention_k_eq_v)
        return attention_k_eq_v.error();
    auto enable_moe_block = read_manifest_bool(text_json, "enable_moe_block", "Gemma4 ");
    if (!enable_moe_block)
        return enable_moe_block.error();
    auto rope_parameters = find_manifest_member(text_json, "rope_parameters");
    if (!rope_parameters)
        return Error{ErrorCode::InvalidModel, "Gemma4 rope_parameters is missing"};
    const std::string rope_json(*rope_parameters);
    auto sliding_rope = read_rope(rope_json, "sliding_attention");
    if (!sliding_rope)
        return sliding_rope.error();
    auto full_rope = read_rope(rope_json, "full_attention");
    if (!full_rope)
        return full_rope.error();

    const bool use_double_wide_mlp = opt_bool(text_json, "use_double_wide_mlp", false);
    const uint32_t kv_shared_layers = find_manifest_member(text_json, "num_kv_shared_layers")
                                          ? read_manifest_uint32(text_json, "num_kv_shared_layers", "Gemma4 ").value()
                                          : 0;
    const uint32_t per_layer_input_size = find_manifest_member(text_json, "hidden_size_per_layer_input")
                                              ? read_manifest_uint32(text_json, "hidden_size_per_layer_input", "Gemma4 ").value()
                                              : 0;
    const bool tie_word_embeddings = find_manifest_member(root_json, "tie_word_embeddings")
                                         ? opt_bool(root_json, "tie_word_embeddings", true)
                                         : opt_bool(text_json, "tie_word_embeddings", true);
    const float logit_softcap = find_manifest_member(text_json, "final_logit_softcapping")
                                    ? read_manifest_float(text_json, "final_logit_softcapping", "Gemma4 ").value()
                                    : softcap_default;

    if (attention_bias.value() || !attention_k_eq_v.value() || !enable_moe_block.value()
        || use_double_wide_mlp || kv_shared_layers != 0 || per_layer_input_size != 0
        || activation.value() != "gelu_pytorch_tanh"
        || !tie_word_embeddings
        || layer_types.value().size() != layer_count.value())
    {
        return Error{ErrorCode::UnsupportedModel, "unsupported Gemma4 text configuration"};
    }
    if (attention_head_count.value() == 0 || kv_head_count.value() == 0
        || global_kv_head_count == 0
        || head_dimension.value() == 0 || global_head_dimension == 0
        || head_dimension.value() % 2 != 0 || global_head_dimension % 2 != 0
        || attention_head_count.value() % kv_head_count.value() != 0
        || attention_head_count.value() % global_kv_head_count != 0
        || top_k.value() == 0 || top_k.value() > expert_count.value()
        || dense_intermediate_size.value() == 0 || moe_intermediate_size.value() == 0
        || !std::isfinite(norm_epsilon.value()) || norm_epsilon.value() <= 0.0f
        || !std::isfinite(logit_softcap) || logit_softcap < 0.0f)
    {
        return Error{ErrorCode::InvalidModel, "unsupported Gemma4 architectural dimensions"};
    }
    if (sliding_rope.value().type != "default"
        || (full_rope.value().type != "default" && full_rope.value().type != "proportional"))
        return Error{ErrorCode::UnsupportedModel, "unsupported Gemma4 RoPE type"};

    auto activation_dtype_name = read_manifest_string(text_json,
                                                      find_manifest_member(text_json, "dtype") ? "dtype" : "torch_dtype",
                                                      "Gemma4 ");
    if (!activation_dtype_name)
        return activation_dtype_name.error();
    DType weight_dtype = DType::BFloat16;
    if (activation_dtype_name.value() == "bfloat16" || activation_dtype_name.value() == "float16")
        weight_dtype = DType::BFloat16;
    else if (activation_dtype_name.value() == "float32")
        weight_dtype = DType::Float32;
    else
        return Error{ErrorCode::UnsupportedModel, "unsupported Gemma4 dtype: " + activation_dtype_name.value()};

    auto artifact_status = artifact_exists(package);
    if (!artifact_status)
        return artifact_status.error();
    const bool mxfp4 = artifact_status.value();
    if (mxfp4)
    {
        auto valid_artifact = validate_artifact(package,
                                                layer_count.value(),
                                                expert_count.value(),
                                                hidden_size.value(),
                                                moe_intermediate_size.value());
        if (!valid_artifact)
            return valid_artifact.error();
    }

    // Pad the dense FFN width so the same weights serve CPU and GPU; the zero
    // rows leave the result bit-identical on the CPU path.
    const uint32_t aligned_dense_intermediate = ((dense_intermediate_size.value() + ffn_tile - 1) / ffn_tile)
                                                * ffn_tile;

    MoeModelDescriptor descriptor;
    descriptor.model_type = "gemma4";
    descriptor.vocabulary_size = vocabulary_size.value();
    descriptor.hidden_size = hidden_size.value();
    descriptor.intermediate_size = moe_intermediate_size.value();
    descriptor.attention_head_count = attention_head_count.value();
    descriptor.kv_head_count = std::max(kv_head_count.value(), global_kv_head_count);
    descriptor.head_dimension = std::max(head_dimension.value(), global_head_dimension);
    descriptor.expert_count = expert_count.value();
    descriptor.experts_per_token = top_k.value();
    descriptor.activation_dtype = weight_dtype;
    descriptor.kv_cache_dtype = weight_dtype;
    descriptor.norm_epsilon = norm_epsilon.value();
    descriptor.norm_weight_offset = 0.0f;
    descriptor.final_norm = NormType::RmsNorm;
    descriptor.varied_attn = true;
    // Gemma4TextScaledWordEmbedding: embeddings * sqrt(hidden_size).
    descriptor.embed_scale = std::sqrt(static_cast<float>(hidden_size.value()));
    descriptor.logit_softcap = logit_softcap;
    descriptor.layers.resize(layer_count.value());

    for (uint32_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
    {
        const std::string& layer_type = layer_types.value()[layer_id];
        const bool full_attention = layer_type == "full_attention";
        const bool sliding_attention = layer_type == "sliding_attention";
        if (!full_attention && !sliding_attention)
            return Error{ErrorCode::UnsupportedModel, "unsupported Gemma4 layer type: " + layer_type};
        const bool use_global_kv = full_attention && attention_k_eq_v.value();
        const uint32_t layer_head_dimension = full_attention ? global_head_dimension : head_dimension.value();
        const uint32_t layer_kv_head_count = use_global_kv ? global_kv_head_count : kv_head_count.value();
        const RopeConfig& rope = full_attention ? full_rope.value() : sliding_rope.value();
        uint32_t rope_dimension = layer_head_dimension;
        if (full_attention && rope.type == "proportional")
        {
            auto rotary = get_rotary_dimension(layer_head_dimension,
                                               rope.partial,
                                               "Gemma4 partial rotary dimension");
            if (!rotary)
                return rotary.error();
            rope_dimension = rotary.value();
        }

        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention.kind = AttentionKind::Standard;
        layer.attention.head_count = attention_head_count.value();
        layer.attention.kv_head_count = layer_kv_head_count;
        layer.attention.head_dimension = layer_head_dimension;
        layer.attention.value_head_dimension = layer_head_dimension;
        layer.attention.qk_rope_head_dimension = layer_head_dimension;
        layer.attention.rope_active = rope_dimension == layer_head_dimension ? 0 : rope_dimension;
        layer.attention.sliding_window = full_attention ? 0 : sliding_window.value();
        layer.attention.initial_context_length = maximum_context.value();
        layer.attention.max_context_length = maximum_context.value();
        layer.attention.rope_theta = rope.theta;
        layer.attention.attn_scale = attn_scale;
        layer.attention.flags = AttentionDescriptorQueryKeyNorm
                                | AttentionDescriptorValueNorm
                                | AttentionDescriptorPostNorm;

        layer.dense.intermediate_size = aligned_dense_intermediate;
        layer.dense.activation = ExpertActivation::GeluTanh;
        layer.dense.gate_weight_dtype = weight_dtype;
        layer.dense.up_weight_dtype = weight_dtype;
        layer.dense.down_weight_dtype = weight_dtype;

        layer.moe.expert_count = expert_count.value();
        layer.moe.top_k = top_k.value();
        layer.moe.intermediate_size = moe_intermediate_size.value();
        layer.moe.score_function = RouterScoreFunction::Softmax;
        layer.moe.normalization = RouterNormalization::SelectedExperts;
        layer.moe.activation = ExpertActivation::GeluTanh;
        layer.moe.layout = ExpertLayout::PackedGateUpDown;
        layer.moe.expert_weight_dtype = mxfp4 ? DType::MxFp4 : weight_dtype;
        layer.moe.routed_scaling_factor = 1.0f;
        layer.moe.flags = MoeDescriptorParallelFfn
                          | MoeDescriptorRouterNorm
                          | MoeDescriptorPerExpertScale;
    }
    return descriptor;
}

Result<WeightMapping> Gemma4ModelAdapter::map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const
{
    if (descriptor.model_type != "gemma4")
        return Error{ErrorCode::UnsupportedModel, "unsupported Gemma4 model_type: " + descriptor.model_type};
    auto opened = SafetensorsArchive::open(package.weights_path.empty() ? package.root : package.weights_path,
                                           package.manifest.raw_json);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    WeightMapping mapping;

    std::optional<SafetensorsArchive> mxfp4_archive;
    if (!descriptor.layers.empty() && descriptor.layers.front().moe.expert_weight_dtype == DType::MxFp4)
    {
        auto artifact = SafetensorsArchive::open_file(package.root / artifact_name);
        if (!artifact)
            return artifact.error();
        mxfp4_archive = std::move(artifact).value();
    }
    uint32_t expert_load_flags = 0;
    if (has_flag(package.flags, ModelPackageDeferMxfp4Experts))
        expert_load_flags |= SafetensorLoadDeferMxfp4Data;

    const std::string source_root = "model.language_model.";
    auto status = add_tensor(mapping, archive, "token_embedding.weight", source_root + "embed_tokens.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "final_norm.weight", source_root + "norm.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "lm_head.weight", source_root + "embed_tokens.weight");
    if (!status)
        return status.error();

    const float hidden_root_scale = 1.0f / std::sqrt(static_cast<float>(descriptor.hidden_size));
    for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
    {
        const LayerDescriptor& layer = descriptor.layers[layer_id];
        const std::string source = source_root + "layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        const std::pair<const char*, const char*> layer_tensors[] = {
            {"pre_attention_norm.weight", "input_layernorm.weight"},
            {"post_attention_norm.weight", "post_attention_layernorm.weight"},
            {"attention.query.weight", "self_attn.q_proj.weight"},
            {"attention.key.weight", "self_attn.k_proj.weight"},
            {"attention.output.weight", "self_attn.o_proj.weight"},
            {"attention.query_norm.weight", "self_attn.q_norm.weight"},
            {"attention.key_norm.weight", "self_attn.k_norm.weight"},
            {"dense_pre_ffn_norm.weight", "pre_feedforward_layernorm.weight"},
            {"pre_ffn_norm.weight", "pre_feedforward_layernorm_2.weight"},
            {"dense_post_ffn_norm.weight", "post_feedforward_layernorm_1.weight"},
            {"moe_post_ffn_norm.weight", "post_feedforward_layernorm_2.weight"},
            {"post_ffn_norm.weight", "post_feedforward_layernorm.weight"},
            {"router.weight", "router.proj.weight"},
            {"output_scale", "layer_scalar"},
        };
        for (const auto& item : layer_tensors)
        {
            status = add_tensor(mapping, archive, target + item.first, source + item.second);
            if (!status)
                return status.error();
        }
        const SafetensorInfo* gate_info = archive.find(source + "mlp.gate_proj.weight");
        const bool pad_dense = !gate_info || static_cast<uint32_t>(gate_info->shape.empty() ? 0 : gate_info->shape[0]) != layer.dense.intermediate_size;
        if (pad_dense)
        {
            status = add_padded_bf16(mapping,
                                     archive,
                                     target + "ffn.gate.weight",
                                     source + "mlp.gate_proj.weight",
                                     layer.dense.intermediate_size,
                                     descriptor.hidden_size);
            if (!status)
                return status.error();
            status = add_padded_bf16(mapping,
                                     archive,
                                     target + "ffn.up.weight",
                                     source + "mlp.up_proj.weight",
                                     layer.dense.intermediate_size,
                                     descriptor.hidden_size);
            if (!status)
                return status.error();
            status = add_padded_bf16(mapping,
                                     archive,
                                     target + "ffn.down.weight",
                                     source + "mlp.down_proj.weight",
                                     descriptor.hidden_size,
                                     layer.dense.intermediate_size);
            if (!status)
                return status.error();
        }
        else
        {
            status = add_tensor(mapping, archive, target + "ffn.gate.weight", source + "mlp.gate_proj.weight");
            if (!status)
                return status.error();
            status = add_tensor(mapping, archive, target + "ffn.up.weight", source + "mlp.up_proj.weight");
            if (!status)
                return status.error();
            status = add_tensor(mapping, archive, target + "ffn.down.weight", source + "mlp.down_proj.weight");
            if (!status)
                return status.error();
        }
        const bool has_value_projection = archive.contains_tensor(source + "self_attn.v_proj.weight");
        const std::string value_source = has_value_projection
                                             ? source + "self_attn.v_proj.weight"
                                             : source + "self_attn.k_proj.weight";
        status = add_tensor(mapping, archive, target + "attention.value.weight", value_source);
        if (!status)
            return status.error();
        status = add_ones(mapping,
                          target + "attention.value_norm.weight",
                          layer.attention.head_dimension,
                          descriptor.activation_dtype);
        if (!status)
            return status.error();
        status = add_scaled(mapping, archive,
                            target + "router.norm.weight",
                            source + "router.scale",
                            descriptor.hidden_size,
                            hidden_root_scale);
        if (!status)
            return status.error();
        status = add_scaled(mapping, archive,
                            target + "router.expert_scale",
                            source + "router.per_expert_scale",
                            descriptor.expert_count,
                            1.0f);
        if (!status)
            return status.error();

        const uint32_t expert_intermediate = layer.moe.intermediate_size;
        if (mxfp4_archive)
        {
            const std::string artifact_experts = expert_prefix(layer_id);
            for (uint32_t expert_id = 0; expert_id < descriptor.expert_count; ++expert_id)
            {
                const std::string expert = target + "experts." + std::to_string(expert_id) + ".";
                status = add_mxfp4_expert(mapping,
                                          *mxfp4_archive,
                                          expert + "gate_up.weight",
                                          artifact_experts + "gate_up.blocks",
                                          artifact_experts + "gate_up.scales",
                                          expert_id,
                                          expert_intermediate * 2,
                                          descriptor.hidden_size,
                                          expert_load_flags);
                if (!status)
                    return status.error();
                status = add_mxfp4_expert(mapping,
                                          *mxfp4_archive,
                                          expert + "down.weight",
                                          artifact_experts + "down.blocks",
                                          artifact_experts + "down.scales",
                                          expert_id,
                                          descriptor.hidden_size,
                                          expert_intermediate,
                                          expert_load_flags);
                if (!status)
                    return status.error();
            }
        }
        else
        {
            status = add_bfloat16_expert_bank(mapping,
                                              archive,
                                              target,
                                              "gate_up.weight",
                                              source + "experts.gate_up_proj",
                                              descriptor.expert_count,
                                              {expert_intermediate * 2, descriptor.hidden_size});
            if (!status)
                return status.error();
            status = add_bfloat16_expert_bank(mapping,
                                              archive,
                                              target,
                                              "down.weight",
                                              source + "experts.down_proj",
                                              descriptor.expert_count,
                                              {descriptor.hidden_size, expert_intermediate});
            if (!status)
                return status.error();
        }
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
