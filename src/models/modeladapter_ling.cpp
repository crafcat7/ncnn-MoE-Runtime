#include "modeladapter_ling.h"

#include "json.h"
#include "kernels/ops.h"
#include "modeladapter.h"
#include "safetensors.h"
#include "tensornames.h"

#include <cmath>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

bool LingModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "bailing_hybrid";
}

Result<MoeModelDescriptor> LingModelAdapter::parse_model(const ModelPackage& package) const
{
    if (!can_load(package.manifest))
        return Error{ErrorCode::UnsupportedModel, "unsupported Ling model_type: " + package.manifest.model_type};

    const std::string& json = package.manifest.raw_json;
    auto vocabulary_size = read_manifest_uint32(json, "vocab_size", "Ling ");
    if (!vocabulary_size)
        return vocabulary_size.error();
    auto hidden_size = read_manifest_uint32(json, "hidden_size", "Ling ");
    if (!hidden_size)
        return hidden_size.error();
    auto intermediate_size = read_manifest_uint32(json, "intermediate_size", "Ling ");
    if (!intermediate_size)
        return intermediate_size.error();
    auto moe_intermediate_size = read_manifest_uint32(json, "moe_intermediate_size", "Ling ");
    if (!moe_intermediate_size)
        return moe_intermediate_size.error();
    auto layer_count = read_manifest_uint32(json, "num_hidden_layers", "Ling ");
    if (!layer_count)
        return layer_count.error();
    auto dense_layer_count = read_manifest_uint32(json, "first_k_dense_replace", "Ling ");
    if (!dense_layer_count)
        return dense_layer_count.error();
    auto expert_count = read_manifest_uint32(json, "num_experts", "Ling ");
    if (!expert_count)
        return expert_count.error();
    auto top_k = read_manifest_uint32(json, "num_experts_per_tok", "Ling ");
    if (!top_k)
        return top_k.error();
    auto shared_expert_count = read_manifest_uint32(json, "num_shared_experts", "Ling ");
    if (!shared_expert_count)
        return shared_expert_count.error();
    auto group_count = read_manifest_uint32(json, "n_group", "Ling ");
    if (!group_count)
        return group_count.error();
    auto top_k_groups = read_manifest_uint32(json, "topk_group", "Ling ");
    if (!top_k_groups)
        return top_k_groups.error();
    auto layer_group_size = read_manifest_uint32(json, "layer_group_size", "Ling ");
    if (!layer_group_size)
        return layer_group_size.error();
    auto attention_head_count = read_manifest_uint32(json, "num_attention_heads", "Ling ");
    if (!attention_head_count)
        return attention_head_count.error();
    auto kv_head_count = read_manifest_uint32(json, "num_key_value_heads", "Ling ");
    if (!kv_head_count)
        return kv_head_count.error();
    auto head_dimension = read_manifest_uint32(json, "head_dim", "Ling ");
    if (!head_dimension)
        return head_dimension.error();
    auto q_lora_rank = read_manifest_uint32(json, "q_lora_rank", "Ling ");
    if (!q_lora_rank)
        return q_lora_rank.error();
    auto kv_lora_rank = read_manifest_uint32(json, "kv_lora_rank", "Ling ");
    if (!kv_lora_rank)
        return kv_lora_rank.error();
    auto qk_nope_head_dimension = read_manifest_uint32(json, "qk_nope_head_dim", "Ling ");
    if (!qk_nope_head_dimension)
        return qk_nope_head_dimension.error();
    auto qk_rope_head_dimension = read_manifest_uint32(json, "qk_rope_head_dim", "Ling ");
    if (!qk_rope_head_dimension)
        return qk_rope_head_dimension.error();
    auto value_head_dimension = read_manifest_uint32(json, "v_head_dim", "Ling ");
    if (!value_head_dimension)
        return value_head_dimension.error();
    auto convolution_kernel_size = read_manifest_uint32(json, "short_conv_kernel_size", "Ling ");
    if (!convolution_kernel_size)
        return convolution_kernel_size.error();
    auto maximum_context = read_manifest_uint32(json, "max_position_embeddings", "Ling ");
    if (!maximum_context)
        return maximum_context.error();
    auto norm_epsilon = read_manifest_float(json, "rms_norm_eps", "Ling ");
    if (!norm_epsilon)
        return norm_epsilon.error();
    auto rope_theta = read_manifest_float(json, "rope_theta", "Ling ");
    if (!rope_theta)
        return rope_theta.error();
    auto delta_lower_bound = read_manifest_float(json, "kda_lower_bound", "Ling ");
    if (!delta_lower_bound)
        return delta_lower_bound.error();
    auto activation = read_manifest_string(json, "hidden_act", "Ling ");
    if (!activation)
        return activation.error();
    auto score_function = read_manifest_string(json, "score_function", "Ling ");
    if (!score_function)
        return score_function.error();
    auto gated_granularity = read_manifest_string(json, "gated_attention_proj_granularity_type", "Ling ");
    if (!gated_granularity)
        return gated_granularity.error();
    auto activation_dtype = read_manifest_string(json, find_manifest_member(json, "dtype") ? "dtype" : "torch_dtype", "Ling ");
    if (!activation_dtype)
        return activation_dtype.error();
    auto safe_gate = read_manifest_bool(json, "kda_safe_gate", "Ling ");
    if (!safe_gate)
        return safe_gate.error();
    auto no_delta_lora = read_manifest_bool(json, "no_kda_lora", "Ling ");
    if (!no_delta_lora)
        return no_delta_lora.error();
    auto interleaved_rope = read_manifest_bool(json, "rope_interleave", "Ling ");
    if (!interleaved_rope)
        return interleaved_rope.error();
    auto qkv_bias = read_manifest_bool(json, "use_qkv_bias", "Ling ");
    if (!qkv_bias)
        return qkv_bias.error();
    auto bias = read_manifest_bool(json, "use_bias", "Ling ");
    if (!bias)
        return bias.error();
    auto expert_bias = read_manifest_bool(json, "moe_router_enable_expert_bias", "Ling ");
    if (!expert_bias)
        return expert_bias.error();
    auto nextn_layer_count = read_manifest_uint32(json, "num_nextn_predict_layers", "Ling ");
    if (!nextn_layer_count)
        return nextn_layer_count.error();
    auto mtp_loss_scaling = optional_manifest_float(json, "mtp_loss_scaling_factor", 0.0f);
    const uint32_t qk_head_dimension = qk_nope_head_dimension.value() + qk_rope_head_dimension.value();

    if (layer_count.value() == 0
        || layer_group_size.value() == 0
        || layer_count.value() % layer_group_size.value() != 0
        || dense_layer_count.value() > layer_count.value()
        || shared_expert_count.value() != 1
        || group_count.value() == 0
        || top_k_groups.value() == 0
        || top_k_groups.value() > group_count.value()
        || expert_count.value() % group_count.value() != 0
        || kv_head_count.value() != attention_head_count.value()
        || value_head_dimension.value() != head_dimension.value()
        || qk_nope_head_dimension.value() == 0
        || qk_rope_head_dimension.value() == 0
        || value_head_dimension.value() == 0
        || q_lora_rank.value() == 0
        || kv_lora_rank.value() == 0
        || convolution_kernel_size.value() == 0
        || activation.value() != "silu"
        || score_function.value() != "sigmoid"
        || gated_granularity.value() != "head_wise"
        || activation_dtype.value() != "bfloat16"
        || !safe_gate.value()
        || !no_delta_lora.value()
        || !interleaved_rope.value()
        || qkv_bias.value()
        || bias.value()
        || !expert_bias.value()
        || nextn_layer_count.value() != 0
        || mtp_loss_scaling != 0.0f)
    {
        return Error{ErrorCode::InvalidModel, "unsupported Ling architectural dimensions"};
    }

    AttentionDescriptor latent_attention;
    latent_attention.kind = AttentionKind::MultiHeadLatent;
    latent_attention.head_count = attention_head_count.value();
    latent_attention.kv_head_count = kv_head_count.value();
    latent_attention.head_dimension = qk_head_dimension;
    latent_attention.value_head_dimension = value_head_dimension.value();
    latent_attention.qk_nope_head_dimension = qk_nope_head_dimension.value();
    latent_attention.qk_rope_head_dimension = qk_rope_head_dimension.value();
    latent_attention.query_lora_rank = q_lora_rank.value();
    latent_attention.kv_lora_rank = kv_lora_rank.value();
    latent_attention.initial_context_length = maximum_context.value();
    latent_attention.max_context_length = maximum_context.value();
    latent_attention.rope_theta = rope_theta.value();
    latent_attention.projection_weight_dtype = DType::BFloat16;
    latent_attention.output_lora_rank = value_head_dimension.value();
    latent_attention.output_group_count = attention_head_count.value();
    latent_attention.flags = AttentionDescriptorLatentValue | AttentionDescriptorOutputGate;

    AttentionDescriptor delta_attention;
    delta_attention.kind = AttentionKind::GatedDeltaNet;
    delta_attention.head_count = attention_head_count.value();
    delta_attention.kv_head_count = kv_head_count.value();
    delta_attention.head_dimension = head_dimension.value();
    delta_attention.value_head_dimension = value_head_dimension.value();
    delta_attention.convolution_kernel_size = convolution_kernel_size.value();
    delta_attention.max_context_length = maximum_context.value();
    delta_attention.delta_lower_bound = delta_lower_bound.value();
    delta_attention.flags = AttentionDescriptorSigmoidGate | AttentionDescriptorDeltaGate;

    MoeDescriptor moe;
    moe.expert_count = expert_count.value();
    moe.top_k = top_k.value();
    moe.intermediate_size = moe_intermediate_size.value();
    moe.shared_expert_count = shared_expert_count.value();
    moe.router_group_count = group_count.value();
    moe.router_top_k_groups = top_k_groups.value();
    moe.score_function = RouterScoreFunction::Sigmoid;
    moe.normalization = RouterNormalization::SelectedExperts;
    moe.activation = ExpertActivation::Silu;
    moe.layout = ExpertLayout::GateUpDown;
    moe.expert_weight_dtype = DType::BFloat16;
    moe.shared_expert_weight_dtype = DType::BFloat16;
    moe.routed_scaling_factor = optional_manifest_float(json, "routed_scaling_factor", 1.0f);
    moe.flags = MoeDescriptorRouterBias;

    MoeModelDescriptor descriptor;
    descriptor.model_type = package.manifest.model_type;
    descriptor.vocabulary_size = vocabulary_size.value();
    descriptor.hidden_size = hidden_size.value();
    descriptor.intermediate_size = moe_intermediate_size.value();
    descriptor.attention_head_count = attention_head_count.value();
    descriptor.kv_head_count = kv_head_count.value();
    descriptor.head_dimension = qk_head_dimension;
    descriptor.expert_count = expert_count.value();
    descriptor.experts_per_token = top_k.value();
    descriptor.activation_dtype = DType::BFloat16;
    descriptor.kv_cache_dtype = DType::Float32;
    descriptor.norm_epsilon = norm_epsilon.value();
    descriptor.norm_weight_offset = 0.0f;
    descriptor.final_norm = NormType::RmsNorm;
    descriptor.layers.resize(layer_count.value());
    for (uint32_t layer_id = 0; layer_id < layer_count.value(); ++layer_id)
    {
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.pre_attention_norm = NormType::RmsNorm;
        layer.pre_ffn_norm = NormType::RmsNorm;
        layer.attention = (layer_id + 1) % layer_group_size.value() == 0
                              ? latent_attention
                              : delta_attention;
        if (layer_id < dense_layer_count.value())
        {
            layer.dense.intermediate_size = intermediate_size.value();
            layer.dense.gate_weight_dtype = DType::BFloat16;
            layer.dense.up_weight_dtype = DType::BFloat16;
            layer.dense.down_weight_dtype = DType::BFloat16;
            layer.dense.activation = ExpertActivation::Silu;
        }
        else
            layer.moe = moe;
    }
    return descriptor;
}

static Result<void> add_row_concat(WeightMapping& mapping, const SafetensorsArchive& archive,
                                   const std::string& target, const std::vector<std::string>& sources,
                                   const std::vector<uint32_t>& shape)
{
    uint64_t elements = 1;
    for (uint32_t dimension : shape)
        elements *= dimension;
    TensorData combined;
    combined.dtype = DType::BFloat16;
    combined.shape = shape;
    combined.bfloat16_data.reserve(static_cast<size_t>(elements));
    for (const std::string& source : sources)
    {
        auto loaded = archive.load_tensor(source);
        if (!loaded)
            return loaded.error();
        const TensorData& tensor = loaded.value();
        if (tensor.dtype != DType::BFloat16)
            return Error{ErrorCode::InvalidModel, "Ling fused projection is not BF16: " + source};
        const std::span<const uint16_t> values = tensor.bfloat16_values();
        combined.bfloat16_data.insert(combined.bfloat16_data.end(), values.begin(), values.end());
    }
    if (combined.bfloat16_data.size() != elements)
        return Error{ErrorCode::InvalidModel, "Ling fused projection shape mismatch: " + target};
    mapping.emplace(target, std::move(combined));
    return {};
}

static Result<void> add_ling_delta(WeightMapping& mapping, const SafetensorsArchive& archive,
                                   const std::string& source, const std::string& target,
                                   const AttentionDescriptor& attention, uint32_t hidden_size)
{
    const uint32_t key_size = attention.kv_head_count * attention.head_dimension;
    const uint32_t value_size = attention.head_count * attention.value_head_dimension;
    const uint32_t convolution_size = key_size * 2 + value_size;
    std::vector<std::string> projections;
    std::vector<std::string> convolutions;
    for (const char* role : {"q", "k", "v"})
    {
        projections.push_back(source + "attention." + role + "_proj.weight");
        convolutions.push_back(source + "attention." + role + "_conv1d.weight");
    }
    auto status = add_row_concat(mapping, archive, target + "attention.delta.qkv.weight",
                                 projections, {convolution_size, hidden_size});
    if (!status)
        return status.error();
    status = add_row_concat(mapping, archive, target + "attention.delta.convolution.weight",
                            convolutions, {convolution_size, 1, attention.convolution_kernel_size});
    if (!status)
        return status.error();
    const std::pair<const char*, const char*> tensors[] = {
        {"attention.delta.z.weight", "attention.g_proj.weight"},
        {"attention.delta.beta.weight", "attention.b_proj.weight"},
        {"attention.delta.alpha.weight", "attention.f_proj.weight"},
        {"attention.delta.time_bias", "attention.dt_bias"},
        {"attention.delta.decay_log", "attention.A_log"},
        {"attention.delta.norm.weight", "attention.o_norm.weight"},
        {"attention.output.weight", "attention.o_proj.weight"},
    };
    for (const auto& item : tensors)
    {
        status = add_tensor(mapping, archive, target + item.first, source + item.second);
        if (!status)
            return status.error();
    }
    return {};
}

// Fold the key-value expansion into the query projection and the output
// projection so the Vulkan latent core sees one compressed latent per token.
static Result<void> add_ling_absorbed(WeightMapping& mapping, const SafetensorsArchive& archive,
                                      const std::string& source, const std::string& target,
                                      const AttentionDescriptor& attention)
{
    const uint32_t heads = attention.head_count;
    const uint32_t nope = attention.qk_nope_head_dimension;
    const uint32_t rope = attention.qk_rope_head_dimension;
    const uint32_t vdim = attention.value_head_dimension;
    const uint32_t lora = attention.kv_lora_rank;
    const uint32_t width = lora + rope;
    const uint32_t qdim = nope + rope;
    const uint32_t q_lora = attention.query_lora_rank;
    auto query_b = archive.load_tensor(source + "attention.q_b_proj.weight");
    if (!query_b)
        return query_b.error();
    auto key_value_b = archive.load_tensor(source + "attention.kv_b_proj.weight");
    if (!key_value_b)
        return key_value_b.error();
    if (query_b.value().dtype != DType::BFloat16 || key_value_b.value().dtype != DType::BFloat16)
        return Error{ErrorCode::InvalidModel, "Ling MLA absorption requires BF16 projections"};
    const std::span<const uint16_t> q = query_b.value().bfloat16_values();
    const std::span<const uint16_t> kv = key_value_b.value().bfloat16_values();

    const float scale = std::sqrt(static_cast<float>(width) / static_cast<float>(qdim));
    TensorData absorbed;
    absorbed.dtype = DType::BFloat16;
    absorbed.shape = {heads * width, q_lora};
    absorbed.bfloat16_data.resize(static_cast<size_t>(heads) * width * q_lora);
    std::vector<float> accumulated(static_cast<size_t>(lora) * q_lora);
    for (uint32_t head = 0; head < heads; ++head)
    {
        std::fill(accumulated.begin(), accumulated.end(), 0.0f);
        const uint16_t* nope_rows = q.data() + static_cast<size_t>(head) * qdim * q_lora;
        const uint16_t* key_rows = kv.data() + static_cast<size_t>(head) * (nope + vdim) * lora;
        for (uint32_t row = 0; row < nope; ++row)
        {
            const uint16_t* key = key_rows + static_cast<size_t>(row) * lora;
            const uint16_t* query = nope_rows + static_cast<size_t>(row) * q_lora;
            for (uint32_t column = 0; column < lora; ++column)
            {
                const float key_value = bfloat16_to_float(key[column]);
                float* destination = accumulated.data() + static_cast<size_t>(column) * q_lora;
                for (uint32_t rank = 0; rank < q_lora; ++rank)
                    destination[rank] += key_value * bfloat16_to_float(query[rank]);
            }
        }
        for (uint32_t column = 0; column < lora; ++column)
        {
            const float* values = accumulated.data() + static_cast<size_t>(column) * q_lora;
            uint16_t* destination = absorbed.bfloat16_data.data()
                                    + (static_cast<size_t>(head) * width + column) * q_lora;
            for (uint32_t rank = 0; rank < q_lora; ++rank)
                destination[rank] = float_to_bfloat16(values[rank] * scale);
        }
        const uint16_t* rope_rows = nope_rows + static_cast<size_t>(nope) * q_lora;
        std::memcpy(absorbed.bfloat16_data.data()
                        + (static_cast<size_t>(head) * width + lora) * q_lora,
                    rope_rows,
                    static_cast<size_t>(rope) * q_lora * sizeof(uint16_t));
        uint16_t* rope_destination = absorbed.bfloat16_data.data()
                                     + (static_cast<size_t>(head) * width + lora) * q_lora;
        for (size_t index = 0; index < static_cast<size_t>(rope) * q_lora; ++index)
            rope_destination[index] = float_to_bfloat16(bfloat16_to_float(rope_destination[index]) * scale);
    }
    mapping.emplace(target + "attention.query_b_absorbed.weight", std::move(absorbed));

    TensorData output_a;
    output_a.dtype = DType::BFloat16;
    output_a.shape = {heads * vdim, width};
    output_a.bfloat16_data.assign(static_cast<size_t>(heads) * vdim * width, 0);
    for (uint32_t head = 0; head < heads; ++head)
    {
        const uint16_t* value_rows = kv.data()
                                     + (static_cast<size_t>(head) * (nope + vdim) + nope) * lora;
        for (uint32_t row = 0; row < vdim; ++row)
        {
            const uint16_t* source_row = value_rows + static_cast<size_t>(row) * lora;
            uint16_t* destination = output_a.bfloat16_data.data()
                                    + (static_cast<size_t>(head) * vdim + row) * width;
            std::memcpy(destination, source_row, static_cast<size_t>(lora) * sizeof(uint16_t));
        }
    }
    mapping.emplace(target + "attention.output_a.weight", std::move(output_a));
    return {};
}

static Result<void> add_ling_latent(WeightMapping& mapping, const SafetensorsArchive& archive,
                                    const std::string& source, const std::string& target,
                                    const AttentionDescriptor& attention)
{
    auto status = add_ling_absorbed(mapping, archive, source, target, attention);
    if (!status)
        return status.error();
    const std::pair<const char*, const char*> tensors[] = {
        {"attention.query_a.weight", "attention.q_a_proj.weight"},
        {"attention.query_norm.weight", "attention.q_a_layernorm.weight"},
        {"attention.query_b.weight", "attention.q_b_proj.weight"},
        {"attention.key_value.weight", "attention.kv_a_proj_with_mqa.weight"},
        {"attention.key_value_norm.weight", "attention.kv_a_layernorm.weight"},
        {"attention.key_value_b.weight", "attention.kv_b_proj.weight"},
        {"attention.output_gate.weight", "attention.g_proj.weight"},
        {"attention.output.weight", "attention.dense.weight"},
    };
    for (const auto& item : tensors)
    {
        status = add_tensor(mapping, archive, target + item.first, source + item.second);
        if (!status)
            return status.error();
    }
    return {};
}

Result<WeightMapping> LingModelAdapter::map_weights(const ModelPackage& package, const MoeModelDescriptor& descriptor) const
{
    if (descriptor.model_type != "bailing_hybrid")
        return Error{ErrorCode::UnsupportedModel, "unsupported Ling model_type: " + descriptor.model_type};
    auto opened = SafetensorsArchive::open(package.weights_path.empty() ? package.root : package.weights_path, package.manifest.raw_json);
    if (!opened)
        return opened.error();
    SafetensorsArchive archive = std::move(opened).value();
    WeightMapping mapping;
    auto status = add_tensor(mapping, archive, "token_embedding.weight", "model.word_embeddings.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "final_norm.weight", "model.norm.weight");
    if (!status)
        return status.error();
    status = add_tensor(mapping, archive, "lm_head.weight", "lm_head.weight");
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < descriptor.layers.size(); ++layer_id)
    {
        const std::string source = "model.layers." + std::to_string(layer_id) + ".";
        const std::string target = layer_prefix(layer_id);
        status = add_tensor(mapping, archive, target + "pre_attention_norm.weight", source + "input_layernorm.weight");
        if (!status)
            return status.error();
        status = add_tensor(mapping, archive, target + "pre_ffn_norm.weight", source + "post_attention_layernorm.weight");
        if (!status)
            return status.error();

        const LayerDescriptor& layer = descriptor.layers[layer_id];
        if (layer.dense.enabled())
        {
            for (const auto& projection : {std::pair{"gate", "gate_proj"}, std::pair{"up", "up_proj"}, std::pair{"down", "down_proj"}})
            {
                status = add_tensor(mapping, archive, target + "ffn." + projection.first + ".weight",
                                    source + "mlp." + projection.second + ".weight");
                if (!status)
                    return status.error();
            }
        }
        else
        {
            status = add_tensor(mapping, archive, target + "router.weight", source + "mlp.gate.weight");
            if (!status)
                return status.error();
            status = add_tensor(mapping, archive, target + "router.selection_bias", source + "mlp.gate.expert_bias");
            if (!status)
                return status.error();
            for (const auto& projection : {std::pair{"gate", "gate_proj"}, std::pair{"up", "up_proj"}, std::pair{"down", "down_proj"}})
            {
                status = add_tensor(mapping, archive, target + "shared_expert." + projection.first + ".weight",
                                    source + "mlp.shared_experts." + projection.second + ".weight");
                if (!status)
                    return status.error();
            }
            const MoeDescriptor& moe = layer.moe;
            for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
            {
                const std::string expert_source = source + "mlp.experts." + std::to_string(expert_id) + ".";
                const std::string expert_target = expert_prefix(layer_id, expert_id);
                for (const auto& projection : {std::pair{"gate", "gate_proj"}, std::pair{"up", "up_proj"}, std::pair{"down", "down_proj"}})
                {
                    status = add_tensor(mapping, archive, expert_target + projection.first + ".weight",
                                        expert_source + projection.second + ".weight");
                    if (!status)
                        return status.error();
                }
            }
        }

        const AttentionDescriptor& attention = layer.attention;
        status = attention.kind == AttentionKind::MultiHeadLatent
                     ? add_ling_latent(mapping, archive, source, target, attention)
                     : add_ling_delta(mapping, archive, source, target, attention, descriptor.hidden_size);
        if (!status)
            return status.error();
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
