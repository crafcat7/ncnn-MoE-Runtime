#include "deepseek_v4_model_adapter.h"

#include "internal/tensor_names.h"
#include "safetensors.h"

#include <limits>
#include <regex>
#include <utility>

namespace ncnn {
namespace moe {

static Result<uint32_t> deepseek_required_uint32(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "DeepSeek-V4 manifest is missing integer field: " + key};
    try
    {
        const unsigned long long value = std::stoull(match[1].str());
        if (value > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, "DeepSeek-V4 manifest integer is out of range: " + key};
        return static_cast<uint32_t>(value);
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid DeepSeek-V4 integer field: " + key};
    }
}

static Result<std::string> deepseek_required_string(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "DeepSeek-V4 manifest is missing string field: " + key};
    return match[1].str();
}

static float deepseek_optional_float(const std::string& json, const std::string& key, float fallback)
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

static Result<std::vector<uint32_t>> deepseek_required_uint32_array(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "DeepSeek-V4 manifest is missing integer array: " + key};
    const std::regex number_expression("[0-9]+");
    std::vector<uint32_t> values;
    for (std::sregex_iterator iterator(match[1].first, match[1].second, number_expression), end; iterator != end; ++iterator)
    {
        try
        {
            const unsigned long long value = std::stoull(iterator->str());
            if (value > std::numeric_limits<uint32_t>::max())
                return Error{ErrorCode::InvalidModel, "DeepSeek-V4 array value is out of range: " + key};
            values.push_back(static_cast<uint32_t>(value));
        }
        catch (const std::exception&)
        {
            return Error{ErrorCode::InvalidModel, "invalid DeepSeek-V4 integer array: " + key};
        }
    }
    if (values.empty())
        return Error{ErrorCode::InvalidModel, "DeepSeek-V4 integer array is empty: " + key};
    return values;
}

static bool deepseek_has_key(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:");
    return std::regex_search(json, expression);
}

bool DeepSeekV4ModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "deepseek_v4";
}

Result<MoeIR> DeepSeekV4ModelAdapter::parse_model(const ModelPackage& package) const
{
    if (package.manifest.model_type != "deepseek_v4")
        return Error{ErrorCode::UnsupportedModel, "unsupported DeepSeek model_type: " + package.manifest.model_type};

    const std::string& json = package.manifest.raw_json;
    auto vocabulary_size = deepseek_required_uint32(json, "vocab_size");
    auto hidden_size = deepseek_required_uint32(json, "hidden_size");
    auto intermediate_size = deepseek_required_uint32(json, "moe_intermediate_size");
    auto layer_count = deepseek_required_uint32(json, "num_hidden_layers");
    auto expert_count = deepseek_required_uint32(json, "n_routed_experts");
    auto top_k = deepseek_required_uint32(json, "num_experts_per_tok");
    auto shared_expert_count = deepseek_required_uint32(json, "n_shared_experts");
    auto attention_head_count = deepseek_required_uint32(json, "num_attention_heads");
    auto kv_head_count = deepseek_required_uint32(json, "num_key_value_heads");
    auto head_dimension = deepseek_required_uint32(json, "head_dim");
    auto query_lora_rank = deepseek_required_uint32(json, "q_lora_rank");
    auto rope_head_dimension = deepseek_required_uint32(json, "qk_rope_head_dim");
    auto output_group_count = deepseek_required_uint32(json, "o_groups");
    auto output_lora_rank = deepseek_required_uint32(json, "o_lora_rank");
    auto sliding_window = deepseek_required_uint32(json, "sliding_window");
    auto maximum_context = deepseek_required_uint32(json, "max_position_embeddings");
    auto initial_context = deepseek_required_uint32(json, "original_max_position_embeddings");
    auto hash_layer_count = deepseek_required_uint32(json, "num_hash_layers");
    auto index_head_count = deepseek_required_uint32(json, "index_n_heads");
    auto index_head_dimension = deepseek_required_uint32(json, "index_head_dim");
    auto index_top_k = deepseek_required_uint32(json, "index_topk");
    auto hyper_multiplier = deepseek_required_uint32(json, "hc_mult");
    auto hyper_iterations = deepseek_required_uint32(json, "hc_sinkhorn_iters");
    auto compress_ratios = deepseek_required_uint32_array(json, "compress_ratios");
    auto expert_dtype = deepseek_required_string(json, "expert_dtype");
    auto scoring_function = deepseek_required_string(json, "scoring_func");
    auto quantization_method = deepseek_required_string(json, "quant_method");
    auto quantization_format = deepseek_required_string(json, "fmt");
    auto scale_format = deepseek_required_string(json, "scale_fmt");
    auto weight_block_size = deepseek_required_uint32_array(json, "weight_block_size");

    if (!vocabulary_size || !hidden_size || !intermediate_size || !layer_count || !expert_count || !top_k || !shared_expert_count
        || !attention_head_count || !kv_head_count || !head_dimension || !query_lora_rank || !rope_head_dimension
        || !output_group_count || !output_lora_rank || !sliding_window || !maximum_context || !initial_context
        || !hash_layer_count || !index_head_count || !index_head_dimension || !index_top_k
        || !hyper_multiplier || !hyper_iterations || !compress_ratios
        || !expert_dtype || !scoring_function || !quantization_method || !quantization_format || !scale_format || !weight_block_size)
    {
        const Error* error = !vocabulary_size        ? &vocabulary_size.error()
                             : !hidden_size          ? &hidden_size.error()
                             : !intermediate_size    ? &intermediate_size.error()
                             : !layer_count          ? &layer_count.error()
                             : !expert_count         ? &expert_count.error()
                             : !top_k                ? &top_k.error()
                             : !shared_expert_count  ? &shared_expert_count.error()
                             : !attention_head_count ? &attention_head_count.error()
                             : !kv_head_count        ? &kv_head_count.error()
                             : !head_dimension       ? &head_dimension.error()
                             : !query_lora_rank      ? &query_lora_rank.error()
                             : !rope_head_dimension  ? &rope_head_dimension.error()
                             : !output_group_count   ? &output_group_count.error()
                             : !output_lora_rank     ? &output_lora_rank.error()
                             : !sliding_window       ? &sliding_window.error()
                             : !maximum_context      ? &maximum_context.error()
                             : !initial_context      ? &initial_context.error()
                             : !hash_layer_count     ? &hash_layer_count.error()
                             : !index_head_count     ? &index_head_count.error()
                             : !index_head_dimension ? &index_head_dimension.error()
                             : !index_top_k          ? &index_top_k.error()
                             : !hyper_multiplier     ? &hyper_multiplier.error()
                             : !hyper_iterations     ? &hyper_iterations.error()
                             : !compress_ratios      ? &compress_ratios.error()
                             : !expert_dtype         ? &expert_dtype.error()
                             : !scoring_function     ? &scoring_function.error()
                             : !quantization_method  ? &quantization_method.error()
                             : !quantization_format  ? &quantization_format.error()
                             : !scale_format         ? &scale_format.error()
                                                     : &weight_block_size.error();
        return *error;
    }

    std::vector<uint32_t> speculative_targets;
    uint32_t speculative_block_size = 0;
    uint32_t speculative_noise_token_id = 0;
    uint32_t speculative_markov_rank = 0;
    const bool has_dspark = deepseek_has_key(json, "dspark_target_layer_ids")
                            || deepseek_has_key(json, "dspark_block_size")
                            || deepseek_has_key(json, "dspark_noise_token_id")
                            || deepseek_has_key(json, "dspark_markov_rank");
    if (has_dspark)
    {
        auto targets = deepseek_required_uint32_array(json, "dspark_target_layer_ids");
        auto block_size = deepseek_required_uint32(json, "dspark_block_size");
        auto noise_token_id = deepseek_required_uint32(json, "dspark_noise_token_id");
        auto markov_rank = deepseek_required_uint32(json, "dspark_markov_rank");
        if (!targets)
            return targets.error();
        if (!block_size)
            return block_size.error();
        if (!noise_token_id)
            return noise_token_id.error();
        if (!markov_rank)
            return markov_rank.error();
        speculative_targets = std::move(targets).value();
        speculative_block_size = block_size.value();
        speculative_noise_token_id = noise_token_id.value();
        speculative_markov_rank = markov_rank.value();
    }

    if (compress_ratios.value().size() < layer_count.value())
        return Error{ErrorCode::InvalidModel, "DeepSeek-V4 compress_ratios is shorter than num_hidden_layers"};
    for (uint32_t target_layer_id : speculative_targets)
    {
        if (target_layer_id >= layer_count.value())
            return Error{ErrorCode::InvalidModel, "DeepSeek-V4 DSpark target layer is out of range"};
    }
    if (rope_head_dimension.value() >= head_dimension.value()
        || attention_head_count.value() % output_group_count.value() != 0
        || hash_layer_count.value() > layer_count.value()
        || shared_expert_count.value() != 1
        || expert_dtype.value() != "fp4"
        || scoring_function.value() != "sqrtsoftplus"
        || quantization_method.value() != "fp8"
        || quantization_format.value() != "e4m3"
        || scale_format.value() != "ue8m0"
        || weight_block_size.value() != std::vector<uint32_t>{128, 128})
    {
        return Error{ErrorCode::InvalidModel, "unsupported DeepSeek-V4 architectural dimensions"};
    }

    MoeIR descriptor;
    descriptor.model_type = "deepseek_v4";
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
    descriptor.norm_epsilon = deepseek_optional_float(json, "rms_norm_eps", 1e-6f);
    descriptor.hyper_connection_kind = HyperConnectionKind::Sinkhorn;
    descriptor.hyper_connection_multiplier = hyper_multiplier.value();
    descriptor.hyper_connection_iterations = hyper_iterations.value();
    descriptor.hyper_connection_epsilon = deepseek_optional_float(json, "hc_eps", 1e-6f);
    descriptor.hash_routing_layer_count = hash_layer_count.value();
    descriptor.speculative_kind = speculative_targets.empty()
                                      ? SpeculativeModelKind::None
                                      : SpeculativeModelKind::DSpark;
    descriptor.speculative_layer_count = static_cast<uint32_t>(speculative_targets.size());
    descriptor.speculative_block_size = speculative_block_size;
    descriptor.speculative_noise_token_id = speculative_noise_token_id;
    descriptor.speculative_markov_rank = speculative_markov_rank;
    descriptor.speculative_target_layer_ids = std::move(speculative_targets);

    AttentionDescriptor attention;
    attention.kind = AttentionKind::MultiHeadLatent;
    attention.head_count = attention_head_count.value();
    attention.kv_head_count = kv_head_count.value();
    attention.head_dimension = head_dimension.value();
    attention.sliding_window = sliding_window.value();
    attention.initial_context_length = initial_context.value();
    attention.max_context_length = maximum_context.value();
    attention.query_lora_rank = query_lora_rank.value();
    attention.kv_lora_rank = head_dimension.value();
    attention.qk_nope_head_dimension = head_dimension.value() - rope_head_dimension.value();
    attention.qk_rope_head_dimension = rope_head_dimension.value();
    attention.value_head_dimension = head_dimension.value();
    attention.output_lora_rank = output_lora_rank.value();
    attention.output_group_count = output_group_count.value();
    attention.index_head_count = index_head_count.value();
    attention.index_head_dimension = index_head_dimension.value();
    attention.index_top_k = index_top_k.value();
    attention.rope_theta = deepseek_optional_float(json, "rope_theta", 10000.0f);
    attention.compressed_rope_theta = deepseek_optional_float(json, "compress_rope_theta", 160000.0f);
    attention.rope_scaling_factor = deepseek_optional_float(json, "factor", 16.0f);
    attention.rope_ntk_alpha = deepseek_optional_float(json, "beta_slow", 1.0f);
    attention.rope_ntk_beta = deepseek_optional_float(json, "beta_fast", 32.0f);
    attention.projection_weight_dtype = DType::Float8E4M3;
    attention.flags = AttentionDescriptorSinks | AttentionDescriptorQueryKeyNorm | AttentionDescriptorRopeInterleaved;

    MoeDescriptor moe;
    moe.expert_count = expert_count.value();
    moe.top_k = top_k.value();
    moe.intermediate_size = intermediate_size.value();
    moe.shared_expert_count = shared_expert_count.value();
    moe.score_function = RouterScoreFunction::SqrtSoftplus;
    moe.normalization = RouterNormalization::SelectedExperts;
    moe.activation = ExpertActivation::DeepSeekSwiGlu;
    moe.layout = ExpertLayout::InterleavedGateUpDown;
    moe.expert_weight_dtype = DType::MxFp4;
    moe.shared_expert_weight_dtype = DType::Float8E4M3;
    moe.activation_limit = deepseek_optional_float(json, "swiglu_limit", 10.0f);
    moe.routed_scaling_factor = deepseek_optional_float(json, "routed_scaling_factor", 1.5f);
    moe.flags = MoeDescriptorNormalizeTopKWeights | MoeDescriptorSharedExpert;

    descriptor.layers.resize(descriptor.layer_count);
    for (uint32_t layer_id = 0; layer_id < descriptor.layer_count; ++layer_id)
    {
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.flags = LayerDescriptorAttention | LayerDescriptorMoe;
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention = attention;
        layer.attention.compression_ratio = compress_ratios.value()[layer_id];
        layer.ffn.moe = moe;
        if (layer_id >= descriptor.hash_routing_layer_count)
            layer.ffn.moe.flags |= MoeDescriptorRouterBias;
    }
    return descriptor;
}

static Result<void> deepseek_add_tensor(WeightMapping& mapping, const SafetensorsArchive& archive, const std::string& target, const std::string& source)
{
    auto tensor = archive.load_tensor(source);
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(target, std::move(tensor).value());
    return {};
}

static Result<void> deepseek_add_float8(WeightMapping& mapping, const SafetensorsArchive& archive, const std::string& target, const std::string& source)
{
    auto tensor = archive.load_float8_tensor(source + ".weight", source + ".scale");
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(target, std::move(tensor).value());
    return {};
}

static Result<void> deepseek_validate_dspark(const SafetensorsArchive& archive, const MoeIR& descriptor)
{
    if (descriptor.speculative_layer_count == 0)
        return {};
    for (uint32_t layer_id = 0; layer_id < descriptor.speculative_layer_count; ++layer_id)
    {
        const std::string prefix = "mtp." + std::to_string(layer_id) + ".";
        if (!archive.find(prefix + "attn.wq_a.weight")
            || !archive.find(prefix + "ffn.gate.weight")
            || !archive.find(prefix + "ffn.experts.0.w1.weight"))
        {
            return Error{ErrorCode::InvalidModel, "incomplete DSpark draft layer: " + std::to_string(layer_id)};
        }
    }
    const std::string last = "mtp." + std::to_string(descriptor.speculative_layer_count - 1) + ".";
    if (!archive.find(last + "markov_head.markov_w1.weight")
        || !archive.find(last + "markov_head.markov_w2.weight")
        || !archive.find(last + "confidence_head.proj.weight"))
    {
        return Error{ErrorCode::InvalidModel, "incomplete DSpark prediction heads"};
    }
    return {};
}

Result<WeightMapping> DeepSeekV4ModelAdapter::map_weights(const ModelPackage& package, const MoeIR& descriptor) const
{
    if (descriptor.model_type != "deepseek_v4")
        return Error{ErrorCode::UnsupportedModel, "unsupported DeepSeek model_type: " + descriptor.model_type};

    auto opened = SafetensorsArchive::open(package.root);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    auto dspark_status = deepseek_validate_dspark(archive, descriptor);
    if (!dspark_status)
        return dspark_status.error();

    uint32_t expert_flags = 0;
    if (has_flag(package.flags, ModelPackageDeferMxfp4Experts))
        expert_flags |= SafetensorLoadDeferMxfp4Data;

    WeightMapping mapping;
    auto status = deepseek_add_tensor(mapping, archive, "token_embedding.weight", "embed.weight");
    if (!status)
        return status.error();
    status = deepseek_add_tensor(mapping, archive, "final_norm.weight", "norm.weight");
    if (!status)
        return status.error();
    status = deepseek_add_tensor(mapping, archive, "lm_head.weight", "head.weight");
    if (!status)
        return status.error();
    status = deepseek_add_tensor(mapping, archive, "hyper.head.function", "hc_head_fn");
    if (!status)
        return status.error();
    status = deepseek_add_tensor(mapping, archive, "hyper.head.base", "hc_head_base");
    if (!status)
        return status.error();
    status = deepseek_add_tensor(mapping, archive, "hyper.head.scale", "hc_head_scale");
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < descriptor.layer_count; ++layer_id)
    {
        const std::string source = "layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        const std::pair<const char*, const char*> direct_tensors[] = {
            {"hyper.attention.function", "hc_attn_fn"},
            {"hyper.attention.base", "hc_attn_base"},
            {"hyper.attention.scale", "hc_attn_scale"},
            {"hyper.ffn.function", "hc_ffn_fn"},
            {"hyper.ffn.base", "hc_ffn_base"},
            {"hyper.ffn.scale", "hc_ffn_scale"},
            {"pre_attention_norm.weight", "attn_norm.weight"},
            {"attention.query_norm.weight", "attn.q_norm.weight"},
            {"attention.key_value_norm.weight", "attn.kv_norm.weight"},
            {"attention.sinks", "attn.attn_sink"},
            {"pre_ffn_norm.weight", "ffn_norm.weight"},
            {"router.weight", "ffn.gate.weight"},
        };
        for (const auto& item : direct_tensors)
        {
            status = deepseek_add_tensor(mapping, archive, target + item.first, source + item.second);
            if (!status)
                return status.error();
        }
        const std::pair<const char*, const char*> float8_tensors[] = {
            {"attention.query_a.weight", "attn.wq_a"},
            {"attention.query_b.weight", "attn.wq_b"},
            {"attention.key_value.weight", "attn.wkv"},
            {"attention.output_a.weight", "attn.wo_a"},
            {"attention.output_b.weight", "attn.wo_b"},
            {"shared_expert.gate.weight", "ffn.shared_experts.w1"},
            {"shared_expert.up.weight", "ffn.shared_experts.w3"},
            {"shared_expert.down.weight", "ffn.shared_experts.w2"},
        };
        for (const auto& item : float8_tensors)
        {
            status = deepseek_add_float8(mapping, archive, target + item.first, source + item.second);
            if (!status)
                return status.error();
        }

        if (layer_id < descriptor.hash_routing_layer_count)
        {
            status = deepseek_add_tensor(mapping, archive, target + "router.token_experts", source + "ffn.gate.tid2eid");
        }
        else
        {
            status = deepseek_add_tensor(mapping, archive, target + "router.selection_bias", source + "ffn.gate.bias");
        }
        if (!status)
            return status.error();

        const AttentionDescriptor& attention = descriptor.layers[layer_id].attention;
        if (attention.compression_ratio != 0)
        {
            const std::pair<const char*, const char*> compressor_tensors[] = {
                {"attention.compressor.position", "attn.compressor.ape"},
                {"attention.compressor.norm.weight", "attn.compressor.norm.weight"},
                {"attention.compressor.key_value.weight", "attn.compressor.wkv.weight"},
                {"attention.compressor.gate.weight", "attn.compressor.wgate.weight"},
            };
            for (const auto& item : compressor_tensors)
            {
                status = deepseek_add_tensor(mapping, archive, target + item.first, source + item.second);
                if (!status)
                    return status.error();
            }
            if (attention.compression_ratio == 4)
            {
                const std::pair<const char*, const char*> index_tensors[] = {
                    {"attention.indexer.compressor.position", "attn.indexer.compressor.ape"},
                    {"attention.indexer.compressor.norm.weight", "attn.indexer.compressor.norm.weight"},
                    {"attention.indexer.compressor.key_value.weight", "attn.indexer.compressor.wkv.weight"},
                    {"attention.indexer.compressor.gate.weight", "attn.indexer.compressor.wgate.weight"},
                    {"attention.indexer.weights.weight", "attn.indexer.weights_proj.weight"},
                };
                for (const auto& item : index_tensors)
                {
                    status = deepseek_add_tensor(mapping, archive, target + item.first, source + item.second);
                    if (!status)
                        return status.error();
                }
                status = deepseek_add_float8(mapping, archive, target + "attention.indexer.query.weight", source + "attn.indexer.wq_b");
                if (!status)
                    return status.error();
            }
        }

        const MoeDescriptor& moe = descriptor.layers[layer_id].ffn.moe;
        for (uint32_t expert_id = 0; expert_id < descriptor.expert_count; ++expert_id)
        {
            const std::string expert_source = source + "ffn.experts." + std::to_string(expert_id) + ".";
            const std::string expert_target = expert_prefix(layer_id, expert_id);
            auto gate_up = archive.load_interleaved_mxfp4_tensor(
                expert_source + "w1.weight", expert_source + "w1.scale",
                expert_source + "w3.weight", expert_source + "w3.scale",
                moe.intermediate_size, descriptor.hidden_size, expert_flags);
            if (!gate_up)
                return gate_up.error();
            mapping.tensors.emplace(expert_target + "gate_up.weight", std::move(gate_up).value());
            auto down = archive.load_mxfp4_tensor(expert_source + "w2.weight", expert_source + "w2.scale", descriptor.hidden_size, moe.intermediate_size, expert_flags);
            if (!down)
                return down.error();
            mapping.tensors.emplace(expert_target + "down.weight", std::move(down).value());
        }
    }

    if (descriptor.speculative_layer_count == 0)
        return mapping;

    for (uint32_t layer_id = 0; layer_id < descriptor.speculative_layer_count; ++layer_id)
    {
        const std::string source = "mtp." + std::to_string(layer_id) + ".";
        const std::string target = speculative_layer_prefix(layer_id);
        const std::pair<const char*, const char*> direct_tensors[] = {
            {"hyper.attention.function", "hc_attn_fn"},
            {"hyper.attention.base", "hc_attn_base"},
            {"hyper.attention.scale", "hc_attn_scale"},
            {"hyper.ffn.function", "hc_ffn_fn"},
            {"hyper.ffn.base", "hc_ffn_base"},
            {"hyper.ffn.scale", "hc_ffn_scale"},
            {"pre_attention_norm.weight", "attn_norm.weight"},
            {"attention.query_norm.weight", "attn.q_norm.weight"},
            {"attention.key_value_norm.weight", "attn.kv_norm.weight"},
            {"attention.sinks", "attn.attn_sink"},
            {"pre_ffn_norm.weight", "ffn_norm.weight"},
            {"router.weight", "ffn.gate.weight"},
            {"router.selection_bias", "ffn.gate.bias"},
        };
        for (const auto& item : direct_tensors)
        {
            status = deepseek_add_tensor(mapping, archive, target + item.first, source + item.second);
            if (!status)
                return status.error();
        }
        const std::pair<const char*, const char*> float8_tensors[] = {
            {"attention.query_a.weight", "attn.wq_a"},
            {"attention.query_b.weight", "attn.wq_b"},
            {"attention.key_value.weight", "attn.wkv"},
            {"attention.output_a.weight", "attn.wo_a"},
            {"attention.output_b.weight", "attn.wo_b"},
            {"shared_expert.gate.weight", "ffn.shared_experts.w1"},
            {"shared_expert.up.weight", "ffn.shared_experts.w3"},
            {"shared_expert.down.weight", "ffn.shared_experts.w2"},
        };
        for (const auto& item : float8_tensors)
        {
            status = deepseek_add_float8(mapping, archive, target + item.first, source + item.second);
            if (!status)
                return status.error();
        }
        for (uint32_t expert_id = 0; expert_id < descriptor.expert_count; ++expert_id)
        {
            const std::string expert_source = source + "ffn.experts." + std::to_string(expert_id) + ".";
            const std::string expert_target = speculative_expert_prefix(layer_id, expert_id);
            auto gate_up = archive.load_interleaved_mxfp4_tensor(
                expert_source + "w1.weight", expert_source + "w1.scale",
                expert_source + "w3.weight", expert_source + "w3.scale",
                descriptor.intermediate_size, descriptor.hidden_size, expert_flags);
            if (!gate_up)
                return gate_up.error();
            mapping.tensors.emplace(expert_target + "gate_up.weight", std::move(gate_up).value());
            auto down = archive.load_mxfp4_tensor(
                expert_source + "w2.weight", expert_source + "w2.scale",
                descriptor.hidden_size, descriptor.intermediate_size, expert_flags);
            if (!down)
                return down.error();
            mapping.tensors.emplace(expert_target + "down.weight", std::move(down).value());
        }
    }

    status = deepseek_add_float8(mapping, archive, "speculative.main_projection.weight", "mtp.0.main_proj");
    if (!status)
        return status.error();
    status = deepseek_add_tensor(mapping, archive, "speculative.main_norm.weight", "mtp.0.main_norm.weight");
    if (!status)
        return status.error();
    const std::string final_source = "mtp." + std::to_string(descriptor.speculative_layer_count - 1) + ".";
    const std::pair<const char*, const char*> final_tensors[] = {
        {"speculative.final_norm.weight", "norm.weight"},
        {"speculative.hyper.head.function", "hc_head_fn"},
        {"speculative.hyper.head.base", "hc_head_base"},
        {"speculative.hyper.head.scale", "hc_head_scale"},
        {"speculative.markov.embedding.weight", "markov_head.markov_w1.weight"},
        {"speculative.markov.head.weight", "markov_head.markov_w2.weight"},
        {"speculative.confidence.weight", "confidence_head.proj.weight"},
    };
    for (const auto& item : final_tensors)
    {
        status = deepseek_add_tensor(mapping, archive, item.first, final_source + item.second);
        if (!status)
            return status.error();
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
