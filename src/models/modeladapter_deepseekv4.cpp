#include "modeladapter_deepseekv4.h"

#include "tensornames.h"
#include "modeladapter.h"
#include "safetensors.h"

#include <limits>
#include <regex>
#include <utility>

namespace ncnn {
namespace moe {

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

static Result<void> deepseek_add_experts(
    WeightMapping& mapping, const SafetensorsArchive& archive,
    const std::string& source, const std::string& target,
    uint32_t expert_count, uint32_t hidden_size, uint32_t intermediate_size, uint32_t flags)
{
    for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
    {
        const std::string expert_source = source + "ffn.experts." + std::to_string(expert_id) + ".";
        const std::string expert_target = target + "experts." + std::to_string(expert_id) + ".";
        auto gate_up = archive.load_interleaved_mxfp4_tensor(
            expert_source + "w1.weight", expert_source + "w1.scale",
            expert_source + "w3.weight", expert_source + "w3.scale",
            intermediate_size, hidden_size, flags);
        if (!gate_up)
            return gate_up.error();
        mapping.emplace(expert_target + "gate_up.weight", std::move(gate_up).value());
        auto down = archive.load_mxfp4_tensor(
            expert_source + "w2.weight", expert_source + "w2.scale",
            hidden_size, intermediate_size, flags);
        if (!down)
            return down.error();
        mapping.emplace(expert_target + "down.weight", std::move(down).value());
    }
    return {};
}

bool DeepSeekV4ModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "deepseek_v4";
}

Result<MoeModelDescriptor> DeepSeekV4ModelAdapter::parse_model(const ModelPackage& package) const
{
    if (package.manifest.model_type != "deepseek_v4")
        return Error{ErrorCode::UnsupportedModel, "unsupported DeepSeek model_type: " + package.manifest.model_type};

    const std::string& json = package.manifest.raw_json;
    auto vocabulary_size = read_manifest_uint32(json, "vocab_size", "DeepSeek-V4 ");
    if (!vocabulary_size)
        return vocabulary_size.error();
    auto hidden_size = read_manifest_uint32(json, "hidden_size", "DeepSeek-V4 ");
    if (!hidden_size)
        return hidden_size.error();
    auto intermediate_size = read_manifest_uint32(json, "moe_intermediate_size", "DeepSeek-V4 ");
    if (!intermediate_size)
        return intermediate_size.error();
    auto layer_count = read_manifest_uint32(json, "num_hidden_layers", "DeepSeek-V4 ");
    if (!layer_count)
        return layer_count.error();
    auto expert_count = read_manifest_uint32(json, "n_routed_experts", "DeepSeek-V4 ");
    if (!expert_count)
        return expert_count.error();
    auto top_k = read_manifest_uint32(json, "num_experts_per_tok", "DeepSeek-V4 ");
    if (!top_k)
        return top_k.error();
    auto shared_expert_count = read_manifest_uint32(json, "n_shared_experts", "DeepSeek-V4 ");
    if (!shared_expert_count)
        return shared_expert_count.error();
    auto attention_head_count = read_manifest_uint32(json, "num_attention_heads", "DeepSeek-V4 ");
    if (!attention_head_count)
        return attention_head_count.error();
    auto kv_head_count = read_manifest_uint32(json, "num_key_value_heads", "DeepSeek-V4 ");
    if (!kv_head_count)
        return kv_head_count.error();
    auto head_dimension = read_manifest_uint32(json, "head_dim", "DeepSeek-V4 ");
    if (!head_dimension)
        return head_dimension.error();
    auto query_lora_rank = read_manifest_uint32(json, "q_lora_rank", "DeepSeek-V4 ");
    if (!query_lora_rank)
        return query_lora_rank.error();
    auto rope_head_dimension = read_manifest_uint32(json, "qk_rope_head_dim", "DeepSeek-V4 ");
    if (!rope_head_dimension)
        return rope_head_dimension.error();
    auto output_group_count = read_manifest_uint32(json, "o_groups", "DeepSeek-V4 ");
    if (!output_group_count)
        return output_group_count.error();
    auto output_lora_rank = read_manifest_uint32(json, "o_lora_rank", "DeepSeek-V4 ");
    if (!output_lora_rank)
        return output_lora_rank.error();
    auto sliding_window = read_manifest_uint32(json, "sliding_window", "DeepSeek-V4 ");
    if (!sliding_window)
        return sliding_window.error();
    auto maximum_context = read_manifest_uint32(json, "max_position_embeddings", "DeepSeek-V4 ");
    if (!maximum_context)
        return maximum_context.error();
    auto initial_context = read_manifest_uint32(json, "original_max_position_embeddings", "DeepSeek-V4 ");
    if (!initial_context)
        return initial_context.error();
    auto hash_layer_count = read_manifest_uint32(json, "num_hash_layers", "DeepSeek-V4 ");
    if (!hash_layer_count)
        return hash_layer_count.error();
    auto index_head_count = read_manifest_uint32(json, "index_n_heads", "DeepSeek-V4 ");
    if (!index_head_count)
        return index_head_count.error();
    auto index_head_dimension = read_manifest_uint32(json, "index_head_dim", "DeepSeek-V4 ");
    if (!index_head_dimension)
        return index_head_dimension.error();
    auto index_top_k = read_manifest_uint32(json, "index_topk", "DeepSeek-V4 ");
    if (!index_top_k)
        return index_top_k.error();
    auto hyper_multiplier = read_manifest_uint32(json, "hc_mult", "DeepSeek-V4 ");
    if (!hyper_multiplier)
        return hyper_multiplier.error();
    auto hyper_iterations = read_manifest_uint32(json, "hc_sinkhorn_iters", "DeepSeek-V4 ");
    if (!hyper_iterations)
        return hyper_iterations.error();
    auto compress_ratios = deepseek_required_uint32_array(json, "compress_ratios");
    if (!compress_ratios)
        return compress_ratios.error();
    auto expert_dtype = read_manifest_string(json, "expert_dtype", "DeepSeek-V4 ");
    if (!expert_dtype)
        return expert_dtype.error();
    auto scoring_function = read_manifest_string(json, "scoring_func", "DeepSeek-V4 ");
    if (!scoring_function)
        return scoring_function.error();
    auto quantization_method = read_manifest_string(json, "quant_method", "DeepSeek-V4 ");
    if (!quantization_method)
        return quantization_method.error();
    auto quantization_format = read_manifest_string(json, "fmt", "DeepSeek-V4 ");
    if (!quantization_format)
        return quantization_format.error();
    auto scale_format = read_manifest_string(json, "scale_fmt", "DeepSeek-V4 ");
    if (!scale_format)
        return scale_format.error();
    auto weight_block_size = deepseek_required_uint32_array(json, "weight_block_size");
    if (!weight_block_size)
        return weight_block_size.error();

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
        if (!targets)
            return targets.error();
        auto block_size = read_manifest_uint32(json, "dspark_block_size", "DeepSeek-V4 ");
        if (!block_size)
            return block_size.error();
        auto noise_token_id = read_manifest_uint32(json, "dspark_noise_token_id", "DeepSeek-V4 ");
        if (!noise_token_id)
            return noise_token_id.error();
        auto markov_rank = read_manifest_uint32(json, "dspark_markov_rank", "DeepSeek-V4 ");
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
    if (output_group_count.value() == 0)
        return Error{ErrorCode::InvalidModel, "unsupported DeepSeek-V4 architectural dimensions"};
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

    MoeModelDescriptor descriptor;
    descriptor.model_type = "deepseek_v4";
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
    descriptor.norm_epsilon = optional_manifest_float(json, "rms_norm_eps", 1e-6f);
    descriptor.hyper_connection_kind = HyperConnectionKind::Sinkhorn;
    descriptor.hyper_connection_multiplier = hyper_multiplier.value();
    descriptor.hyper_connection_iterations = hyper_iterations.value();
    descriptor.hyper_connection_epsilon = optional_manifest_float(json, "hc_eps", 1e-6f);
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
    attention.rope_theta = optional_manifest_float(json, "rope_theta", 10000.0f);
    attention.compressed_rope_theta = optional_manifest_float(json, "compress_rope_theta", 160000.0f);
    attention.rope_scaling_factor = optional_manifest_float(json, "factor", 16.0f);
    attention.rope_ntk_alpha = optional_manifest_float(json, "beta_slow", 1.0f);
    attention.rope_ntk_beta = optional_manifest_float(json, "beta_fast", 32.0f);
    attention.projection_weight_dtype = DType::Float8E4M3;
    attention.flags = AttentionDescriptorSinks | AttentionDescriptorQueryKeyNorm;

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
    moe.activation_limit = optional_manifest_float(json, "swiglu_limit", 10.0f);
    moe.routed_scaling_factor = optional_manifest_float(json, "routed_scaling_factor", 1.5f);

    descriptor.layers.resize(layer_count.value());
    for (uint32_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
    {
        LayerDescriptor& layer = descriptor.layers[layer_id];
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

static Result<void> deepseek_add_common_layer_tensors(
    WeightMapping& mapping, const SafetensorsArchive& archive,
    const std::string& source, const std::string& target)
{
    const std::pair<const char*, const char*> tensors[] = {
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
    for (const auto& item : tensors)
    {
        auto ret = add_tensor(mapping, archive, target + item.first, source + item.second);
        if (!ret)
            return ret.error();
    }
    return {};
}

static Result<void> deepseek_add_float8(WeightMapping& mapping, const SafetensorsArchive& archive, const std::string& target, const std::string& source)
{
    auto tensor = archive.load_float8_tensor(source + ".weight", source + ".scale");
    if (!tensor)
        return tensor.error();
    mapping.emplace(target, std::move(tensor).value());
    return {};
}

static Result<void> deepseek_add_float8_layer_tensors(
    WeightMapping& mapping, const SafetensorsArchive& archive,
    const std::string& source, const std::string& target)
{
    const std::pair<const char*, const char*> tensors[] = {
        {"attention.query_a.weight", "attn.wq_a"},
        {"attention.query_b.weight", "attn.wq_b"},
        {"attention.key_value.weight", "attn.wkv"},
        {"attention.output_a.weight", "attn.wo_a"},
        {"attention.output_b.weight", "attn.wo_b"},
        {"shared_expert.gate.weight", "ffn.shared_experts.w1"},
        {"shared_expert.up.weight", "ffn.shared_experts.w3"},
        {"shared_expert.down.weight", "ffn.shared_experts.w2"},
    };
    for (const auto& item : tensors)
    {
        auto ret = deepseek_add_float8(mapping, archive, target + item.first, source + item.second);
        if (!ret)
            return ret.error();
    }
    return {};
}

static Result<void> deepseek_validate_dspark(const SafetensorsArchive& archive, const MoeModelDescriptor& descriptor)
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

Result<WeightMapping> DeepSeekV4ModelAdapter::map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const
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
    auto status = add_tensor(mapping, archive, "token_embedding.weight", "embed.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "final_norm.weight", "norm.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "lm_head.weight", "head.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "hyper.head.function", "hc_head_fn");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "hyper.head.base", "hc_head_base");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "hyper.head.scale", "hc_head_scale");
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
    {
        const std::string source = "layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        status = deepseek_add_common_layer_tensors(mapping, archive, source, target);
        if (!status)
            return status.error();
        status = deepseek_add_float8_layer_tensors(mapping, archive, source, target);
        if (!status)
            return status.error();

        if (layer_id < descriptor.hash_routing_layer_count)
        {
            status = add_tensor(mapping, archive, target + "router.token_experts", source + "ffn.gate.tid2eid");
        }
        else
        {
            status = add_tensor(mapping, archive, target + "router.selection_bias", source + "ffn.gate.bias");
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
                status = add_tensor(mapping, archive, target + item.first, source + item.second);
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
                    status = add_tensor(mapping, archive, target + item.first, source + item.second);
                    if (!status)
                        return status.error();
                }
                status = deepseek_add_float8(mapping, archive, target + "attention.indexer.query.weight", source + "attn.indexer.wq_b");
                if (!status)
                    return status.error();
            }
        }

        const MoeDescriptor& moe = descriptor.layers[layer_id].ffn.moe;
        status = deepseek_add_experts(
            mapping, archive, source, target,
            descriptor.expert_count, descriptor.hidden_size,
            moe.intermediate_size, expert_flags);
        if (!status)
            return status.error();
    }

    if (descriptor.speculative_layer_count == 0)
        return mapping;

    for (uint32_t layer_id = 0; layer_id < descriptor.speculative_layer_count; ++layer_id)
    {
        const std::string source = "mtp." + std::to_string(layer_id) + ".";
        const std::string target = speculative_layer_prefix(layer_id);
        status = deepseek_add_common_layer_tensors(mapping, archive, source, target);
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "router.selection_bias", source + "ffn.gate.bias");
        if (!status)
            return status.error();
        status = deepseek_add_float8_layer_tensors(mapping, archive, source, target);
        if (!status)
            return status.error();
        status = deepseek_add_experts(
            mapping, archive, source, target,
            descriptor.expert_count, descriptor.hidden_size,
            descriptor.intermediate_size, expert_flags);
        if (!status)
            return status.error();
    }

    status = deepseek_add_float8(mapping, archive, "speculative.main_projection.weight", "mtp.0.main_proj");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "speculative.main_norm.weight", "mtp.0.main_norm.weight");
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
        status = add_tensor(mapping, archive, item.first, final_source + item.second);
        if (!status)
            return status.error();
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
