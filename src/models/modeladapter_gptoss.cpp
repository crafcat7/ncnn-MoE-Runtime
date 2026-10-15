#include "modeladapter_gptoss.h"
#include "json.h"

#include "tensornames.h"
#include "modeladapter.h"
#include "safetensors.h"

#include <optional>
#include <utility>

namespace ncnn {
namespace moe {

static Result<void> inspect_gpt_oss_experts(const SafetensorsArchive& archive,
                                            const std::string& source, MoeDescriptor& moe)
{
    const bool official = archive.find(source + "gate_up_proj_blocks") != nullptr
                          || archive.find(source + "0.gate_up_proj_blocks") != nullptr;
    if (official)
    {
        moe.layout = ExpertLayout::InterleavedGateUpDown;
        moe.expert_weight_dtype = DType::MxFp4;
        for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
        {
            const std::string expert = source + std::to_string(expert_id) + ".";
            if (archive.contains_tensor(expert + "gate_up_proj.weight")
                || archive.contains_tensor(expert + "gate_proj.weight")
                || archive.contains_tensor(expert + "up_proj.weight"))
                return Error{ErrorCode::InvalidModel, "duplicate GPT-OSS Expert projection encodings: " + expert};
        }
        return {};
    }
    const bool fused = archive.contains_tensor(source + "0.gate_up_proj.weight");
    moe.layout = fused ? ExpertLayout::InterleavedGateUpDown : ExpertLayout::GateUpDown;
    moe.expert_weight_dtype = DType::BFloat16;
    constexpr const char* fused_projections[] = {"gate_up_proj", "down_proj"};
    constexpr const char* separate_projections[] = {"gate_proj", "up_proj", "down_proj"};
    const char* const* projections = fused ? fused_projections : separate_projections;
    const size_t projection_count = fused ? 2 : 3;
    for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
    {
        const std::string expert = source + std::to_string(expert_id) + ".";
        if (archive.contains_tensor(expert + "gate_up_proj.weight") != fused
            || (fused && (archive.contains_tensor(expert + "gate_proj.weight") || archive.contains_tensor(expert + "up_proj.weight"))))
            return Error{ErrorCode::InvalidModel, "inconsistent GPT-OSS Expert projection layout: " + expert};
        for (size_t projection_index = 0; projection_index < projection_count; ++projection_index)
        {
            auto dtype = archive.tensor_dtype(expert + projections[projection_index] + ".weight");
            if (!dtype)
                return dtype.error();
            // The compiler validates each projection independently. This is
            // only the descriptor's memory-planning estimate for mixed banks.
            if (expert_id == 0 && projection_index == 0)
                moe.expert_weight_dtype = dtype.value();
        }
    }
    return {};
}

static Result<void> add_gpt_oss_expert_bias(WeightMapping& mapping, const SafetensorsArchive& archive,
                                            const std::string& source, const std::string& target,
                                            uint32_t expert_id, uint32_t intermediate_size, uint32_t hidden_size,
                                            ExpertLayout layout)
{
    const std::string expert = source + std::to_string(expert_id) + ".";
    const bool separate = layout == ExpertLayout::GateUpDown;
    if (separate && (archive.contains_tensor(expert + "gate_proj.bias") || archive.contains_tensor(expert + "up_proj.bias")))
    {
        auto status = add_tensor(mapping, archive, target + "gate.bias", expert + "gate_proj.bias");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "up.bias", expert + "up_proj.bias");
        if (!status)
            return status.error();
    }
    else
    {
        auto bias = archive.contains_tensor(expert + "gate_up_proj.bias")
                        ? archive.load_tensor(expert + "gate_up_proj.bias")
                        : archive.load_bfloat16_slice(source + "gate_up_proj_bias", expert_id, {intermediate_size * 2});
        if (!bias)
            return bias.error();
        if (!separate)
            mapping.emplace(target + "gate_up.bias", std::move(bias).value());
        else
        {
            const TensorData& packed = bias.value();
            if (packed.dtype != DType::BFloat16 || packed.shape != std::vector<uint32_t>{intermediate_size * 2}
                || packed.bfloat16_values().size() != static_cast<uint64_t>(intermediate_size) * 2)
                return Error{ErrorCode::InvalidModel, "invalid GPT-OSS interleaved Expert bias: " + expert};
            TensorData gate;
            gate.dtype = DType::BFloat16;
            gate.shape = {intermediate_size};
            gate.bfloat16_data.resize(intermediate_size);
            TensorData up = gate;
            for (uint32_t row = 0; row < intermediate_size; ++row)
            {
                gate.bfloat16_data[row] = packed.bfloat16_values()[row * 2];
                up.bfloat16_data[row] = packed.bfloat16_values()[row * 2 + 1];
            }
            mapping.emplace(target + "gate.bias", std::move(gate));
            mapping.emplace(target + "up.bias", std::move(up));
        }
    }
    if (archive.contains_tensor(expert + "down_proj.bias"))
        return add_tensor(mapping, archive, target + "down.bias", expert + "down_proj.bias");
    return add_bfloat16_slice(mapping, archive, target + "down.bias", source + "down_proj_bias", expert_id, {hidden_size});
}

bool GptOssModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "gpt_oss";
}

Result<MoeModelDescriptor> GptOssModelAdapter::parse_model(const ModelPackage& package) const
{
    if (package.manifest.model_type != "gpt_oss")
        return Error{ErrorCode::UnsupportedModel, "unsupported built-in model_type: " + package.manifest.model_type};
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

    std::string rope_scaling_json;
    const std::string* rope_json = &json;
    if (find_manifest_member(json, "rope_scaling"))
    {
        auto parsed_rope_scaling = read_manifest_object(json, "rope_scaling", "GPT-OSS ");
        if (!parsed_rope_scaling)
            return parsed_rope_scaling.error();
        rope_scaling_json = std::move(parsed_rope_scaling).value();
        rope_json = &rope_scaling_json;
    }

    AttentionDescriptor attention;
    attention.kind = AttentionKind::Standard;
    attention.head_count = descriptor.attention_head_count;
    attention.kv_head_count = descriptor.kv_head_count;
    attention.head_dimension = descriptor.head_dimension;
    attention.initial_context_length = initial_context_length.value();
    attention.max_context_length = max_context_length.value();
    attention.rope_theta = optional_manifest_float(json, "rope_theta", 150000.0f);
    attention.rope_scaling_factor = optional_manifest_float(*rope_json, "factor", 32.0f);
    attention.rope_ntk_alpha = optional_manifest_float(*rope_json, "beta_slow", 1.0f);
    attention.rope_ntk_beta = optional_manifest_float(*rope_json, "beta_fast", 32.0f);
    attention.flags = AttentionDescriptorSinks;
    auto attention_bias = read_manifest_bool(json, "attention_bias");
    const bool use_attention_bias = !attention_bias || attention_bias.value();
    if (use_attention_bias)
        attention.flags |= AttentionDescriptorBias;

    std::optional<SafetensorsArchive> archive;
    if (!package.root.empty())
    {
        auto opened = SafetensorsArchive::open(package.root, package.manifest.raw_json);
        if (opened)
            archive = std::move(opened).value();
    }

    descriptor.layers.resize(layer_count.value());
    for (uint32_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
    {
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention = attention;
        layer.attention.sliding_window = layer_id % 2 == 0 ? sliding_window.value() : 0;
        layer.moe = moe;
        if (archive)
        {
            auto status = inspect_gpt_oss_experts(*archive, "model.layers." + std::to_string(layer_id) + ".mlp.experts.", layer.moe);
            if (!status)
                return status.error();
        }
    }
    return descriptor;
}

Result<WeightMapping> GptOssModelAdapter::map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const
{
    if (descriptor.model_type != "gpt_oss")
        return Error{ErrorCode::UnsupportedModel, "unsupported built-in model_type: " + descriptor.model_type};
    auto opened = SafetensorsArchive::open(package.root, package.manifest.raw_json);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    uint32_t expert_load_flags = 0;
    if (has_flag(package.flags, ModelPackageDeferMxfp4Experts))
        expert_load_flags |= SafetensorLoadDeferMxfp4Data;
    constexpr std::pair<const char*, const char*> fused_projections[] = {
        {"gate_up", "gate_up_proj"},
        {"down", "down_proj"}};
    constexpr std::pair<const char*, const char*> separate_projections[] = {
        {"gate", "gate_proj"},
        {"up", "up_proj"},
        {"down", "down_proj"}};
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
        if (has_flag(descriptor.layers[layer_id].attention.flags, AttentionDescriptorBias))
        {
            status = add_tensor(mapping, archive, target + "attention.query.bias", source + "self_attn.q_proj.bias");
            if (!status)
                return status.error();
        }
        status = add_tensor(mapping, archive, target + "attention.key.weight", source + "self_attn.k_proj.weight");
        if (!status)
            return status.error();
        if (has_flag(descriptor.layers[layer_id].attention.flags, AttentionDescriptorBias))
        {
            status = add_tensor(mapping, archive, target + "attention.key.bias", source + "self_attn.k_proj.bias");
            if (!status)
                return status.error();
        }
        status = add_tensor(mapping, archive, target + "attention.value.weight", source + "self_attn.v_proj.weight");
        if (!status)
            return status.error();
        if (has_flag(descriptor.layers[layer_id].attention.flags, AttentionDescriptorBias))
        {
            status = add_tensor(mapping, archive, target + "attention.value.bias", source + "self_attn.v_proj.bias");
            if (!status)
                return status.error();
        }
        status = add_tensor(mapping, archive, target + "attention.output.weight", source + "self_attn.o_proj.weight");
        if (!status)
            return status.error();
        if (has_flag(descriptor.layers[layer_id].attention.flags, AttentionDescriptorBias))
        {
            status = add_tensor(mapping, archive, target + "attention.output.bias", source + "self_attn.o_proj.bias");
            if (!status)
                return status.error();
        }
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

        const std::string experts_source = source + "mlp.experts.";
        const std::string gate_up_blocks = source + "mlp.experts.gate_up_proj_blocks";
        const std::string gate_up_scales = source + "mlp.experts.gate_up_proj_scales";
        const std::string down_blocks = source + "mlp.experts.down_proj_blocks";
        const std::string down_scales = source + "mlp.experts.down_proj_scales";
        MoeDescriptor moe = descriptor.layers[layer_id].moe;
        status = inspect_gpt_oss_experts(archive, experts_source, moe);
        if (!status)
            return status.error();
        if (moe.layout != descriptor.layers[layer_id].moe.layout)
            return Error{ErrorCode::InvalidModel, "GPT-OSS Expert layout changed after model parsing"};
        for (uint32_t expert_id = 0; expert_id < descriptor.expert_count; ++expert_id)
        {
            const std::string expert = expert_prefix(layer_id, expert_id);
            status = add_gpt_oss_expert_bias(mapping, archive, experts_source, expert, expert_id,
                                             descriptor.intermediate_size, descriptor.hidden_size, moe.layout);
            if (!status)
                return status.error();
            if (moe.expert_weight_dtype != DType::MxFp4)
            {
                const std::string expert_source = experts_source + std::to_string(expert_id) + ".";
                const bool fused = moe.layout == ExpertLayout::InterleavedGateUpDown;
                const auto* projections = fused ? fused_projections : separate_projections;
                const size_t projection_count = fused ? 2 : 3;
                for (size_t projection_index = 0; projection_index < projection_count; ++projection_index)
                {
                    const auto& projection = projections[projection_index];
                    status = add_tensor(mapping, archive, expert + projection.first + ".weight", expert_source + projection.second + ".weight");
                    if (!status)
                        return status.error();
                }
                continue;
            }
            const std::string native_expert = experts_source + std::to_string(expert_id) + ".";
            if (archive.find(native_expert + "gate_up_proj_blocks"))
            {
                auto gate_up = archive.load_mxfp4_tensor(native_expert + "gate_up_proj_blocks", native_expert + "gate_up_proj_scales",
                                                         descriptor.intermediate_size * 2, descriptor.hidden_size, expert_load_flags);
                if (!gate_up)
                    return gate_up.error();
                mapping.emplace(expert + "gate_up.weight", std::move(gate_up).value());
                auto down = archive.load_mxfp4_tensor(native_expert + "down_proj_blocks", native_expert + "down_proj_scales",
                                                      descriptor.hidden_size, descriptor.intermediate_size, expert_load_flags);
                if (!down)
                    return down.error();
                mapping.emplace(expert + "down.weight", std::move(down).value());
                continue;
            }
            status = add_mxfp4_expert(mapping, archive, expert + "gate_up.weight", gate_up_blocks, gate_up_scales,
                                      expert_id, descriptor.intermediate_size * 2, descriptor.hidden_size,
                                      expert_load_flags);
            if (!status)
                return status.error();
            status = add_mxfp4_expert(mapping, archive, expert + "down.weight", down_blocks, down_scales,
                                      expert_id, descriptor.hidden_size, descriptor.intermediate_size,
                                      expert_load_flags);
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

} // namespace moe
} // namespace ncnn
