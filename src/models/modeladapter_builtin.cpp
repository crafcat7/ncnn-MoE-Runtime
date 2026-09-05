#include "modeladapter_builtin.h"

#include "tensornames.h"
#include "modeladapter.h"
#include "safetensors.h"

#include <utility>

namespace ncnn {
namespace moe {

static Result<MoeModelDescriptor> parse_gpt_oss_model(const ModelPackage& package)
{
    const std::string& json = package.manifest.raw_json;
    auto vocabulary_size = read_manifest_uint32(json, "vocab_size");
    if (!vocabulary_size)
        return vocabulary_size.error();
    auto hidden_size = read_manifest_uint32(json, "hidden_size");
    if (!hidden_size)
        return hidden_size.error();
    auto intermediate_size = read_manifest_uint32(json, "intermediate_size");
    if (!intermediate_size)
        return intermediate_size.error();
    auto layer_count = read_manifest_uint32(json, "num_hidden_layers");
    if (!layer_count)
        return layer_count.error();
    auto expert_count = read_manifest_uint32(json, "num_local_experts");
    if (!expert_count)
        return expert_count.error();
    auto top_k = read_manifest_uint32(json, "experts_per_token");
    if (!top_k)
        return top_k.error();
    auto attention_head_count = read_manifest_uint32(json, "num_attention_heads");
    if (!attention_head_count)
        return attention_head_count.error();
    auto kv_head_count = read_manifest_uint32(json, "num_key_value_heads");
    if (!kv_head_count)
        return kv_head_count.error();
    auto head_dimension = read_manifest_uint32(json, "head_dim");
    if (!head_dimension)
        return head_dimension.error();
    auto sliding_window = read_manifest_uint32(json, "sliding_window");
    if (!sliding_window)
        return sliding_window.error();
    auto initial_context_length = read_manifest_uint32(json, "initial_context_length");
    if (!initial_context_length)
        return initial_context_length.error();
    auto max_context_length = read_manifest_uint32(json, "max_position_embeddings");
    if (!max_context_length)
        return max_context_length.error();

    MoeModelDescriptor descriptor;
    descriptor.model_type = "gpt_oss";
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
    descriptor.norm_epsilon = optional_manifest_float(json, "rms_norm_eps", 1e-5f);

    MoeDescriptor moe;
    moe.expert_count = descriptor.expert_count;
    moe.top_k = descriptor.experts_per_token;
    moe.intermediate_size = descriptor.intermediate_size;
    moe.score_function = RouterScoreFunction::Softmax;
    moe.normalization = RouterNormalization::SelectedExperts;
    moe.activation = ExpertActivation::GptOssSwiGlu;
    moe.layout = ExpertLayout::InterleavedGateUpDown;
    moe.expert_weight_dtype = DType::MxFp4;
    moe.flags = MoeDescriptorRouterBias | MoeDescriptorProjectionBias;
    moe.activation_limit = optional_manifest_float(json, "swiglu_limit", 7.0f);

    AttentionDescriptor attention;
    attention.kind = AttentionKind::Standard;
    attention.head_count = descriptor.attention_head_count;
    attention.kv_head_count = descriptor.kv_head_count;
    attention.head_dimension = descriptor.head_dimension;
    attention.initial_context_length = initial_context_length.value();
    attention.max_context_length = max_context_length.value();
    attention.rope_theta = optional_manifest_float(json, "rope_theta", 150000.0f);
    attention.rope_scaling_factor = optional_manifest_float(json, "factor", 32.0f);
    attention.rope_ntk_alpha = optional_manifest_float(json, "beta_slow", 1.0f);
    attention.rope_ntk_beta = optional_manifest_float(json, "beta_fast", 32.0f);
    attention.flags = AttentionDescriptorSinks;
    auto attention_bias = read_manifest_bool(json, "attention_bias");
    const bool use_attention_bias = !attention_bias || attention_bias.value();
    if (use_attention_bias)
        attention.flags |= AttentionDescriptorBias;

    descriptor.layers.resize(layer_count.value());
    for (uint32_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
    {
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention = attention;
        layer.attention.sliding_window = layer_id % 2 == 0 ? sliding_window.value() : 0;
        layer.ffn.moe = moe;
    }
    return descriptor;
}

static Result<WeightMapping> map_gpt_oss_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor)
{
    auto opened = SafetensorsArchive::open(package.root);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    uint32_t expert_load_flags = 0;
    if (has_flag(package.flags, ModelPackageDeferMxfp4Experts))
        expert_load_flags |= SafetensorLoadDeferMxfp4Data;
    WeightMapping mapping;

    auto status = add_tensor(mapping, archive, "token_embedding.weight", "model.embed_tokens.weight");
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
    {
        const std::string source = "model.layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        status = add_tensor(mapping, archive, target + "pre_attention_norm.weight", source + "input_layernorm.weight");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "attention.query.weight", source + "self_attn.q_proj.weight");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "attention.query.bias", source + "self_attn.q_proj.bias");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "attention.key.weight", source + "self_attn.k_proj.weight");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "attention.key.bias", source + "self_attn.k_proj.bias");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "attention.value.weight", source + "self_attn.v_proj.weight");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "attention.value.bias", source + "self_attn.v_proj.bias");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "attention.output.weight", source + "self_attn.o_proj.weight");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "attention.output.bias", source + "self_attn.o_proj.bias");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "attention.sinks", source + "self_attn.sinks");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "pre_ffn_norm.weight", source + "post_attention_layernorm.weight");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "router.weight", source + "mlp.router.weight");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "router.bias", source + "mlp.router.bias");
        if (!status)
            return status.error();

        const std::string gate_up_bias = source + "mlp.experts.gate_up_proj_bias";
        const std::string down_bias = source + "mlp.experts.down_proj_bias";
        const std::string gate_up_blocks = source + "mlp.experts.gate_up_proj_blocks";
        const std::string gate_up_scales = source + "mlp.experts.gate_up_proj_scales";
        const std::string down_blocks = source + "mlp.experts.down_proj_blocks";
        const std::string down_scales = source + "mlp.experts.down_proj_scales";
        for (uint32_t expert_id = 0; expert_id < descriptor.expert_count; ++expert_id)
        {
            const std::string expert = expert_prefix(layer_id, expert_id);
            status = add_mxfp4_expert(
                mapping, archive, expert + "gate_up.weight", gate_up_blocks, gate_up_scales,
                expert_id, descriptor.intermediate_size * 2, descriptor.hidden_size,
                expert_load_flags);
            if (!status)
                return status.error();
            status = add_bfloat16_slice(
                mapping, archive, expert + "gate_up.bias", gate_up_bias,
                expert_id, {descriptor.intermediate_size * 2});
            if (!status)
                return status.error();
            status = add_mxfp4_expert(
                mapping, archive, expert + "down.weight", down_blocks, down_scales,
                expert_id, descriptor.hidden_size, descriptor.intermediate_size,
                expert_load_flags);
            if (!status)
                return status.error();
            status = add_bfloat16_slice(mapping, archive, expert + "down.bias", down_bias, expert_id, {descriptor.hidden_size});
            if (!status)
                return status.error();
        }
    }

    status = add_tensor(mapping, archive, "final_norm.weight", "model.norm.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "lm_head.weight", "lm_head.weight");
    if (!status)
        return status.error();
    return mapping;
}

bool BuiltinModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "gpt_oss";
}

Result<MoeModelDescriptor> BuiltinModelAdapter::parse_model(const ModelPackage& package) const
{
    if (package.manifest.model_type != "gpt_oss")
        return Error{ErrorCode::UnsupportedModel, "unsupported built-in model_type: " + package.manifest.model_type};
    return parse_gpt_oss_model(package);
}

Result<WeightMapping> BuiltinModelAdapter::map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const
{
    if (descriptor.model_type != "gpt_oss")
        return Error{ErrorCode::UnsupportedModel, "unsupported built-in model_type: " + descriptor.model_type};
    return map_gpt_oss_weights(package, descriptor);
}

} // namespace moe
} // namespace ncnn
