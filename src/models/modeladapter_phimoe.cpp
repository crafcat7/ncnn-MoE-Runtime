#include "modeladapter_phimoe.h"

#include "json.h"
#include "modeladapter.h"
#include "safetensors.h"
#include "tensornames.h"

#include <utility>

namespace ncnn {
namespace moe {

bool PhiMoeModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "phimoe";
}

Result<MoeModelDescriptor> PhiMoeModelAdapter::parse_model(const ModelPackage& package) const
{
    if (!can_load(package.manifest))
        return Error{ErrorCode::UnsupportedModel, "unsupported Phi model_type: " + package.manifest.model_type};

    const std::string& json = package.manifest.raw_json;
    auto vocabulary_size = read_manifest_uint32(json, "vocab_size", "PhiMoE ");
    if (!vocabulary_size)
        return vocabulary_size.error();
    auto hidden_size = read_manifest_uint32(json, "hidden_size", "PhiMoE ");
    if (!hidden_size)
        return hidden_size.error();
    auto intermediate_size = read_manifest_uint32(json, "intermediate_size", "PhiMoE ");
    if (!intermediate_size)
        return intermediate_size.error();
    auto layer_count = read_manifest_uint32(json, "num_hidden_layers", "PhiMoE ");
    if (!layer_count)
        return layer_count.error();
    auto expert_count = read_manifest_uint32(json, "num_local_experts", "PhiMoE ");
    if (!expert_count)
        return expert_count.error();
    auto top_k = read_manifest_uint32(json, "num_experts_per_tok", "PhiMoE ");
    if (!top_k)
        return top_k.error();
    auto attention_head_count = read_manifest_uint32(json, "num_attention_heads", "PhiMoE ");
    if (!attention_head_count)
        return attention_head_count.error();
    auto kv_head_count = read_manifest_uint32(json, "num_key_value_heads", "PhiMoE ");
    if (!kv_head_count)
        return kv_head_count.error();
    auto head_dimension = read_manifest_uint32(json, "head_dim", "PhiMoE ");
    if (!head_dimension)
        return head_dimension.error();
    auto maximum_context = read_manifest_uint32(json, "max_position_embeddings", "PhiMoE ");
    if (!maximum_context)
        return maximum_context.error();
    auto norm_epsilon = read_manifest_float(json, "rms_norm_eps", "PhiMoE ");
    if (!norm_epsilon)
        return norm_epsilon.error();
    auto rope_theta = read_manifest_float(json, "rope_theta", "PhiMoE ");
    if (!rope_theta)
        return rope_theta.error();
    auto activation = read_manifest_string(json, "hidden_act", "PhiMoE ");
    if (!activation)
        return activation.error();
    auto activation_dtype = read_manifest_string(json, find_manifest_member(json, "dtype") ? "dtype" : "torch_dtype", "PhiMoE ");
    if (!activation_dtype)
        return activation_dtype.error();
    auto attention_bias = read_manifest_bool(json, "attention_bias", "PhiMoE ");
    if (!attention_bias)
        return attention_bias.error();

    if (kv_head_count.value() == 0
        || attention_head_count.value() % kv_head_count.value() != 0
        || head_dimension.value() == 0
        || head_dimension.value() % 2 != 0
        || activation.value() != "silu"
        || (activation_dtype.value() != "bfloat16" && activation_dtype.value() != "float16")
        || !attention_bias.value())
    {
        return Error{ErrorCode::InvalidModel, "unsupported PhiMoE architectural dimensions"};
    }

    AttentionDescriptor attention;
    attention.kind = AttentionKind::Standard;
    attention.head_count = attention_head_count.value();
    attention.kv_head_count = kv_head_count.value();
    attention.head_dimension = head_dimension.value();
    attention.qk_rope_head_dimension = head_dimension.value();
    attention.value_head_dimension = head_dimension.value();
    // The official PhiMoE eager/SDPA attention is full causal attention.
    // Sliding window is only active in the FlashAttention path, so keep the
    // runtime cache full length.
    attention.sliding_window = 0;
    attention.initial_context_length = maximum_context.value();
    attention.max_context_length = maximum_context.value();
    attention.rope_theta = rope_theta.value();
    attention.flags = AttentionDescriptorBias;

    MoeDescriptor moe;
    moe.expert_count = expert_count.value();
    moe.top_k = top_k.value();
    moe.intermediate_size = intermediate_size.value();
    moe.score_function = RouterScoreFunction::SparseMixer;
    moe.jitter = optional_manifest_float(json, "router_jitter_noise", 0.01f);
    moe.normalization = RouterNormalization::SelectedExperts;
    moe.activation = ExpertActivation::Silu;
    moe.layout = ExpertLayout::GateUpDown;
    moe.expert_weight_dtype = DType::BFloat16;

    MoeModelDescriptor descriptor;
    descriptor.model_type = package.manifest.model_type;
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
    descriptor.norm_weight_offset = 0.0f;
    descriptor.final_norm = NormType::LayerNorm;
    descriptor.layers.resize(layer_count.value());
    for (uint32_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
    {
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.pre_attention_norm = NormType::LayerNorm;
        layer.pre_ffn_norm = NormType::LayerNorm;
        layer.attention = attention;
        layer.moe = moe;
    }
    return descriptor;
}

Result<WeightMapping> PhiMoeModelAdapter::map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const
{
    if (descriptor.model_type != "phimoe")
        return Error{ErrorCode::UnsupportedModel, "unsupported Phi model_type: " + descriptor.model_type};
    auto opened = SafetensorsArchive::open(package.weights_path.empty() ? package.root : package.weights_path, package.manifest.raw_json);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    WeightMapping mapping;
    uint32_t expert_load_flags = 0;
    if (has_flag(package.flags, ModelPackageDeferMxfp4Experts))
        expert_load_flags |= SafetensorLoadDeferMxfp4Data;

    auto status = add_tensor(mapping,
                             archive,
                             "token_embedding.weight",
                             "model.embed_tokens.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping,
                        archive,
                        "final_norm.weight",
                        "model.norm.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping,
                        archive,
                        "final_norm.bias",
                        "model.norm.bias");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "lm_head.weight", "lm_head.weight");
    if (!status)
        return status.error();
    if (archive.contains_tensor("lm_head.bias"))
    {
        status = add_tensor(mapping, archive, "lm_head.bias", "lm_head.bias");
        if (!status)
            return status.error();
    }

    for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
    {
        const std::string source = "model.layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        const std::pair<const char*, const char*> layer_tensors[] = {
            {"pre_attention_norm.weight", "input_layernorm.weight"},
            {"pre_attention_norm.bias", "input_layernorm.bias"},
            {"attention.query.weight", "self_attn.q_proj.weight"},
            {"attention.query.bias", "self_attn.q_proj.bias"},
            {"attention.key.weight", "self_attn.k_proj.weight"},
            {"attention.key.bias", "self_attn.k_proj.bias"},
            {"attention.value.weight", "self_attn.v_proj.weight"},
            {"attention.value.bias", "self_attn.v_proj.bias"},
            {"attention.output.weight", "self_attn.o_proj.weight"},
            {"attention.output.bias", "self_attn.o_proj.bias"},
            {"pre_ffn_norm.weight", "post_attention_layernorm.weight"},
            {"pre_ffn_norm.bias", "post_attention_layernorm.bias"},
            {"router.weight", "block_sparse_moe.gate.weight"},
        };
        for (const auto& item : layer_tensors)
        {
            status = add_tensor(mapping, archive, target + item.first, source + item.second);
            if (!status)
                return status.error();
        }

        const MoeDescriptor& moe = descriptor.layers[layer_id].moe;
        for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
        {
            const std::string expert_source = source + "block_sparse_moe.experts." + std::to_string(expert_id) + ".";
            const std::string expert_target = expert_prefix(layer_id, expert_id);
            const std::pair<const char*, const char*> expert_tensors[] = {
                {"gate.weight", "w1.weight"},
                {"up.weight", "w3.weight"},
                {"down.weight", "w2.weight"},
            };
            for (const auto& item : expert_tensors)
            {
                status = add_tensor(mapping, archive, expert_target + item.first, expert_source + item.second, expert_load_flags);
                if (!status)
                    return status.error();
            }
        }
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
