#include "modeladapter_qwen3.h"

#include "json.h"
#include "modeladapter.h"
#include "safetensors.h"
#include "tensornames.h"

#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace ncnn {
namespace moe {

bool Qwen3Adapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "qwen3";
}

Result<MoeModelDescriptor> Qwen3Adapter::parse_model(const ModelPackage& package) const
{
    if (!can_load(package.manifest))
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen3 model_type: " + package.manifest.model_type};

    const std::string& json = package.manifest.raw_json;
    MoeModelDescriptor descriptor;
    descriptor.model_type = package.manifest.model_type;
    uint32_t layer_count = 0;
    uint32_t maximum_context = 0;
    const std::pair<const char*, uint32_t*> dimensions[] = {
        {"vocab_size", &descriptor.vocabulary_size},
        {"hidden_size", &descriptor.hidden_size},
        {"intermediate_size", &descriptor.intermediate_size},
        {"num_hidden_layers", &layer_count},
        {"num_attention_heads", &descriptor.attention_head_count},
        {"max_position_embeddings", &maximum_context},
    };
    for (const auto& dimension : dimensions)
    {
        auto value = read_manifest_uint32(json, dimension.first, "Qwen3 ");
        if (!value)
            return value.error();
        if (value.value() == 0)
            return Error{ErrorCode::InvalidModel, std::string("Qwen3 dimension must be non-zero: ") + dimension.first};
        *dimension.second = value.value();
    }
    auto head_dimension = find_manifest_member(json, "head_dim")
                              ? read_manifest_uint32(json, "head_dim", "Qwen3 ")
                              : Result<uint32_t>(descriptor.hidden_size / descriptor.attention_head_count);
    if (!head_dimension)
        return head_dimension.error();
    auto kv_head_count = find_manifest_member(json, "num_key_value_heads")
                             ? read_manifest_uint32(json, "num_key_value_heads", "Qwen3 ")
                             : Result<uint32_t>(descriptor.attention_head_count);
    if (!kv_head_count)
        return kv_head_count.error();
    descriptor.head_dimension = head_dimension.value();
    descriptor.kv_head_count = kv_head_count.value();
    if (descriptor.head_dimension == 0 || descriptor.head_dimension % 2 != 0
        || descriptor.kv_head_count == 0 || descriptor.attention_head_count % descriptor.kv_head_count != 0
        || descriptor.attention_head_count > std::numeric_limits<uint32_t>::max() / descriptor.head_dimension
        || (!find_manifest_member(json, "head_dim") && descriptor.hidden_size % descriptor.attention_head_count != 0))
        return Error{ErrorCode::InvalidModel, "unsupported Qwen3 attention dimensions"};

    auto norm_epsilon = read_manifest_float(json, "rms_norm_eps", "Qwen3 ");
    if (!norm_epsilon)
        return norm_epsilon.error();
    auto rope_theta = read_manifest_float(json, "rope_theta", "Qwen3 ");
    if (!rope_theta)
        return rope_theta.error();
    if (!std::isfinite(norm_epsilon.value()) || norm_epsilon.value() <= 0.0f
        || !std::isfinite(rope_theta.value()) || rope_theta.value() <= 0.0f)
        return Error{ErrorCode::InvalidModel, "Qwen3 rms_norm_eps and rope_theta must be finite and positive"};
    descriptor.norm_epsilon = norm_epsilon.value();
    auto activation = read_manifest_string(json, "hidden_act", "Qwen3 ");
    if (!activation)
        return activation.error();
    if (activation.value() != "silu")
        return Error{ErrorCode::UnsupportedModel, "Qwen3 requires silu activation"};
    for (const char* name : {"attention_bias", "use_sliding_window"})
    {
        if (!find_manifest_member(json, name))
            continue;
        auto value = read_manifest_bool(json, name, "Qwen3 ");
        if (!value)
            return value.error();
        if (value.value())
            return Error{ErrorCode::UnsupportedModel, std::string("unsupported Qwen3 option: ") + name};
    }
    const auto rope_scaling = find_manifest_member(json, "rope_scaling");
    if (rope_scaling && *rope_scaling != "null")
        return Error{ErrorCode::UnsupportedModel, "Qwen3 rope_scaling is not supported"};
    // Newer serialized configs can move RoPE parameters into this object.
    // Reject them until their semantics are explicitly mapped.
    if (find_manifest_member(json, "rope_parameters"))
        return Error{ErrorCode::UnsupportedModel, "Qwen3 rope_parameters is not supported; use the original checkpoint config"};
    if (find_manifest_member(json, "layer_types"))
    {
        auto types = read_manifest_string_array(json, "layer_types", "Qwen3 ");
        if (!types)
            return types.error();
        if (types.value().size() != layer_count)
            return Error{ErrorCode::InvalidModel, "Qwen3 layer_types must match num_hidden_layers"};
        for (const auto& type : types.value())
            if (type != "full_attention")
                return Error{ErrorCode::UnsupportedModel, "unsupported Qwen3 layer type: " + type};
    }

    DType weight_dtype = DType::Float32;
    const char* dtype_key = find_manifest_member(json, "dtype") ? "dtype" : "torch_dtype";
    if (find_manifest_member(json, dtype_key))
    {
        auto dtype = read_manifest_string(json, dtype_key, "Qwen3 ");
        if (!dtype)
            return dtype.error();
        if (dtype.value() == "bfloat16" || dtype.value() == "float16")
            weight_dtype = DType::BFloat16;
        else if (dtype.value() != "float32")
            return Error{ErrorCode::UnsupportedModel, "unsupported Qwen3 dtype: " + dtype.value()};
    }
    descriptor.activation_dtype = weight_dtype;
    descriptor.kv_cache_dtype = weight_dtype;

    std::optional<SafetensorsArchive> archive;
    if (!package.root.empty() || !package.weights_path.empty())
    {
        auto opened = SafetensorsArchive::open(package.weights_path.empty() ? package.root : package.weights_path, json);
        if (!opened)
            return opened.error();
        archive = std::move(opened).value();
    }
    descriptor.layers.resize(layer_count);
    for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id)
    {
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention.kind = AttentionKind::Standard;
        layer.attention.head_count = descriptor.attention_head_count;
        layer.attention.kv_head_count = descriptor.kv_head_count;
        layer.attention.head_dimension = descriptor.head_dimension;
        layer.attention.value_head_dimension = descriptor.head_dimension;
        layer.attention.qk_rope_head_dimension = descriptor.head_dimension;
        layer.attention.initial_context_length = maximum_context;
        layer.attention.max_context_length = maximum_context;
        layer.attention.rope_theta = rope_theta.value();
        layer.attention.flags = AttentionDescriptorQueryKeyNorm;
        layer.dense.intermediate_size = descriptor.intermediate_size;
        const std::pair<const char*, DType*> projections[] = {
            {"gate_proj.weight", &layer.dense.gate_weight_dtype},
            {"up_proj.weight", &layer.dense.up_weight_dtype},
            {"down_proj.weight", &layer.dense.down_weight_dtype},
        };
        for (const auto& projection : projections)
        {
            *projection.second = weight_dtype;
            if (archive)
            {
                auto dtype = archive->tensor_dtype("model.layers." + std::to_string(layer_id) + ".mlp." + projection.first);
                if (!dtype)
                    return dtype.error();
                *projection.second = dtype.value();
            }
        }
    }
    return descriptor;
}

Result<WeightMapping> Qwen3Adapter::map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const
{
    if (!can_load(package.manifest) || descriptor.model_type != "qwen3")
        return Error{ErrorCode::UnsupportedModel, "unsupported Qwen3 model_type: " + descriptor.model_type};
    auto opened = SafetensorsArchive::open(package.weights_path.empty() ? package.root : package.weights_path, package.manifest.raw_json);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    auto tied = find_manifest_member(package.manifest.raw_json, "tie_word_embeddings")
                    ? read_manifest_bool(package.manifest.raw_json, "tie_word_embeddings", "Qwen3 ")
                    : Result<bool>(false);
    if (!tied)
        return tied.error();
    WeightMapping mapping;
    const std::pair<const char*, const char*> global_tensors[] = {
        {"token_embedding.weight", "model.embed_tokens.weight"},
        {"final_norm.weight", "model.norm.weight"},
        {"lm_head.weight", tied.value() ? "model.embed_tokens.weight" : "lm_head.weight"},
    };
    for (const auto& tensor : global_tensors)
    {
        auto status = add_tensor(mapping, archive, tensor.first, tensor.second);
        if (!status)
            return status.error();
    }
    const std::pair<const char*, const char*> layer_tensors[] = {
        {"pre_attention_norm.weight", "input_layernorm.weight"},
        {"pre_ffn_norm.weight", "post_attention_layernorm.weight"},
        {"attention.query.weight", "self_attn.q_proj.weight"},
        {"attention.key.weight", "self_attn.k_proj.weight"},
        {"attention.value.weight", "self_attn.v_proj.weight"},
        {"attention.output.weight", "self_attn.o_proj.weight"},
        {"attention.query_norm.weight", "self_attn.q_norm.weight"},
        {"attention.key_norm.weight", "self_attn.k_norm.weight"},
        {"ffn.gate.weight", "mlp.gate_proj.weight"},
        {"ffn.up.weight", "mlp.up_proj.weight"},
        {"ffn.down.weight", "mlp.down_proj.weight"},
    };
    for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
    {
        const std::string source = "model.layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        for (const auto& tensor : layer_tensors)
        {
            auto status = add_tensor(mapping, archive, target + tensor.first, source + tensor.second);
            if (!status)
                return status.error();
        }
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
