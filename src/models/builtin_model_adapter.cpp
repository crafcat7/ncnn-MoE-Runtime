#include "builtin_model_adapter.h"

#include "internal/tensor_names.h"
#include "safetensors.h"

#include <limits>
#include <regex>
#include <utility>

namespace ncnn {
namespace moe {

static Result<uint32_t> required_uint32(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "manifest is missing integer field: " + key};

    try
    {
        const unsigned long long value = std::stoull(match[1].str());
        if (value > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, "manifest integer is out of range: " + key};
        return static_cast<uint32_t>(value);
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid integer field: " + key};
    }
}

static float optional_float(const std::string& json, const std::string& key, float fallback)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*([-+]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return fallback;
    try
    {
        return std::stof(match[1].str());
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

static bool optional_bool(const std::string& json, const std::string& key, bool fallback)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return fallback;
    return match[1].str() == "true";
}

static Result<MoeIR> parse_gpt_oss_model(const ModelPackage& package)
{
    const std::string& json = package.manifest.raw_json;
    auto vocabulary_size = required_uint32(json, "vocab_size");
    auto hidden_size = required_uint32(json, "hidden_size");
    auto intermediate_size = required_uint32(json, "intermediate_size");
    auto layer_count = required_uint32(json, "num_hidden_layers");
    auto expert_count = required_uint32(json, "num_local_experts");
    auto top_k = required_uint32(json, "experts_per_token");
    auto attention_head_count = required_uint32(json, "num_attention_heads");
    auto kv_head_count = required_uint32(json, "num_key_value_heads");
    auto head_dimension = required_uint32(json, "head_dim");
    auto sliding_window = required_uint32(json, "sliding_window");
    auto initial_context_length = required_uint32(json, "initial_context_length");
    auto max_context_length = required_uint32(json, "max_position_embeddings");
    if (!vocabulary_size || !hidden_size || !intermediate_size || !layer_count || !expert_count || !top_k
        || !attention_head_count || !kv_head_count || !head_dimension || !sliding_window
        || !initial_context_length || !max_context_length)
    {
        const Error* error = !vocabulary_size          ? &vocabulary_size.error()
                             : !hidden_size            ? &hidden_size.error()
                             : !intermediate_size      ? &intermediate_size.error()
                             : !layer_count            ? &layer_count.error()
                             : !expert_count           ? &expert_count.error()
                             : !top_k                  ? &top_k.error()
                             : !attention_head_count   ? &attention_head_count.error()
                             : !kv_head_count          ? &kv_head_count.error()
                             : !head_dimension         ? &head_dimension.error()
                             : !sliding_window         ? &sliding_window.error()
                             : !initial_context_length ? &initial_context_length.error()
                                                       : &max_context_length.error();
        return *error;
    }

    MoeIR descriptor;
    descriptor.model_type = "gpt_oss";
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
    descriptor.norm_epsilon = optional_float(json, "rms_norm_eps", 1e-5f);

    MoeDescriptor moe;
    moe.expert_count = descriptor.expert_count;
    moe.top_k = descriptor.experts_per_token;
    moe.intermediate_size = descriptor.intermediate_size;
    moe.score_function = RouterScoreFunction::Softmax;
    moe.normalization = RouterNormalization::SelectedExperts;
    moe.activation = ExpertActivation::GptOssSwiGlu;
    moe.layout = ExpertLayout::InterleavedGateUpDown;
    moe.expert_weight_dtype = DType::MxFp4;
    moe.flags = MoeDescriptorNormalizeTopKWeights | MoeDescriptorRouterBias | MoeDescriptorProjectionBias;
    moe.activation_limit = optional_float(json, "swiglu_limit", 7.0f);

    AttentionDescriptor attention;
    attention.head_count = descriptor.attention_head_count;
    attention.kv_head_count = descriptor.kv_head_count;
    attention.head_dimension = descriptor.head_dimension;
    attention.initial_context_length = initial_context_length.value();
    attention.max_context_length = max_context_length.value();
    attention.rope_theta = optional_float(json, "rope_theta", 150000.0f);
    attention.rope_scaling_factor = optional_float(json, "factor", 32.0f);
    attention.rope_ntk_alpha = optional_float(json, "beta_slow", 1.0f);
    attention.rope_ntk_beta = optional_float(json, "beta_fast", 32.0f);
    attention.flags = AttentionDescriptorSinks;
    if (optional_bool(json, "attention_bias", true))
        attention.flags |= AttentionDescriptorBias;

    descriptor.layers.resize(descriptor.layer_count);
    for (uint32_t layer_id = 0; layer_id < descriptor.layer_count; ++layer_id)
    {
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.flags = LayerDescriptorAttention | LayerDescriptorMoe;
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention = attention;
        layer.attention.sliding_window = layer_id % 2 == 0 ? sliding_window.value() : 0;
        layer.ffn.moe = moe;
        layer.nodes = {
            {ModelNodeType::RmsNorm},
            {ModelNodeType::FusedQkv},
            {ModelNodeType::Rope},
            {ModelNodeType::AttentionSink},
            {ModelNodeType::Sdpa},
            {ModelNodeType::Projection},
            {ModelNodeType::RmsNorm},
            {ModelNodeType::Router},
            {ModelNodeType::TopK},
            {ModelNodeType::ExpertGroup},
            {ModelNodeType::Combine},
        };
    }
    return descriptor;
}

static Result<void> add_safetensor(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& target_name,
    const std::string& source_name)
{
    auto tensor = archive.load_tensor(source_name);
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(target_name, std::move(tensor).value());
    return {};
}

static Result<void> add_safetensor_slice(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& target_name,
    const std::string& source_name,
    uint32_t index,
    std::vector<uint32_t> shape)
{
    auto tensor = archive.load_bfloat16_slice(source_name, index, std::move(shape));
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(target_name, std::move(tensor).value());
    return {};
}

static Result<void> add_mxfp4_expert(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& target_name,
    const std::string& blocks_name,
    const std::string& scales_name,
    uint32_t expert_id,
    uint32_t rows,
    uint32_t columns,
    uint32_t flags)
{
    auto tensor = archive.load_mxfp4_expert(blocks_name, scales_name, expert_id, rows, columns, flags);
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(target_name, std::move(tensor).value());
    return {};
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

    auto status = add_safetensor(mapping, archive, "token_embedding.weight", "model.embed_tokens.weight");
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < descriptor.layer_count; ++layer_id)
    {
        const std::string source = "model.layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        status = add_safetensor(mapping, archive, target + "pre_attention_norm.weight", source + "input_layernorm.weight");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "attention.query.weight", source + "self_attn.q_proj.weight");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "attention.query.bias", source + "self_attn.q_proj.bias");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "attention.key.weight", source + "self_attn.k_proj.weight");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "attention.key.bias", source + "self_attn.k_proj.bias");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "attention.value.weight", source + "self_attn.v_proj.weight");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "attention.value.bias", source + "self_attn.v_proj.bias");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "attention.output.weight", source + "self_attn.o_proj.weight");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "attention.output.bias", source + "self_attn.o_proj.bias");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "attention.sinks", source + "self_attn.sinks");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "pre_ffn_norm.weight", source + "post_attention_layernorm.weight");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "router.weight", source + "mlp.router.weight");
        if (!status)
            return status.error();
        status = add_safetensor(mapping, archive, target + "router.bias", source + "mlp.router.bias");
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
            status = add_safetensor_slice(
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
            status = add_safetensor_slice(mapping, archive, expert + "down.bias", down_bias, expert_id, {descriptor.hidden_size});
            if (!status)
                return status.error();
        }
    }

    status = add_safetensor(mapping, archive, "final_norm.weight", "model.norm.weight");
    if (!status)
        return status.error();
    status = add_safetensor(mapping, archive, "lm_head.weight", "lm_head.weight");
    if (!status)
        return status.error();
    return mapping;
}

bool BuiltinModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "gpt_oss";
}

Result<MoeIR> BuiltinModelAdapter::parse_model(const ModelPackage& package) const
{
    if (package.manifest.model_type != "gpt_oss")
        return Error{ErrorCode::UnsupportedModel, "unsupported built-in model_type: " + package.manifest.model_type};
    return parse_gpt_oss_model(package);
}

Result<WeightMapping> BuiltinModelAdapter::map_weights(const ModelPackage& package, const MoeIR& descriptor) const
{
    if (descriptor.model_type != "gpt_oss")
        return Error{ErrorCode::UnsupportedModel, "unsupported built-in model_type: " + descriptor.model_type};
    return map_gpt_oss_weights(package, descriptor);
}

} // namespace moe
} // namespace ncnn
