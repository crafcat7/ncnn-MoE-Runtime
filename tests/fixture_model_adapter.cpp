#include "fixture_model_adapter.h"

#include "models/internal/tensor_names.h"

#include <cstring>
#include <fstream>
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

    try {
        const unsigned long long value = std::stoull(match[1].str());
        if (value > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, "manifest integer is out of range: " + key};
        return static_cast<uint32_t>(value);
    }
    catch (const std::exception&) {
        return Error{ErrorCode::InvalidModel, "invalid integer field: " + key};
    }
}

static float optional_float(const std::string& json, const std::string& key, float fallback)
{
    const std::regex expression(
        "\\\"" + key + "\\\"\\s*:\\s*([-+]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return fallback;
    try {
        return std::stof(match[1].str());
    }
    catch (const std::exception&) {
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

static std::string optional_string(const std::string& json, const std::string& key, std::string fallback)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return fallback;
    return match[1].str();
}

static Result<ExpertActivation> parse_activation(const std::string& value)
{
    if (value == "relu")
        return ExpertActivation::Relu;
    if (value == "silu")
        return ExpertActivation::Silu;
    if (value == "gelu")
        return ExpertActivation::Gelu;
    if (value == "clamped_silu")
        return ExpertActivation::ClampedSilu;
    return Error{ErrorCode::InvalidModel, "unsupported expert_activation: " + value};
}

static Result<ExpertLayout> parse_layout(const std::string& value)
{
    if (value == "up_down")
        return ExpertLayout::UpDown;
    if (value == "gate_up_down")
        return ExpertLayout::GateUpDown;
    return Error{ErrorCode::InvalidModel, "unsupported expert_layout: " + value};
}

static Result<DType> parse_dtype(const std::string& value)
{
    if (value == "float32")
        return DType::Float32;
    if (value == "int8")
        return DType::Int8;
    return Error{ErrorCode::InvalidModel, "unsupported expert_weight_dtype: " + value};
}

static Result<DType> parse_kv_cache_dtype(const std::string& value)
{
    if (value == "float32")
        return DType::Float32;
    if (value == "bfloat16")
        return DType::BFloat16;
    return Error{ErrorCode::InvalidModel, "unsupported kv_cache_dtype: " + value};
}

static Result<std::vector<uint8_t> > read_binary_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open weights: " + path.string()};

    const std::streamsize byte_count = stream.tellg();
    if (byte_count < 0)
        return Error{ErrorCode::IoError, "cannot determine weight file size: " + path.string()};
    stream.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(static_cast<size_t>(byte_count));
    if (byte_count > 0 && !stream.read(reinterpret_cast<char*>(bytes.data()), byte_count))
        return Error{ErrorCode::IoError, "cannot read weights: " + path.string()};
    return bytes;
}

class SequentialWeightReader
{
public:
    explicit SequentialWeightReader(std::vector<uint8_t> bytes) : bytes_(std::move(bytes))
    {
    }

    Result<TensorData> take(const std::string& name, std::vector<uint32_t> shape, DType dtype)
    {
        uint64_t element_count = 1;
        for (uint32_t dimension : shape)
            element_count *= dimension;
        if (element_count > std::numeric_limits<size_t>::max())
            return Error{ErrorCode::InvalidModel, "tensor is too large: " + name};

        TensorData tensor;
        tensor.dtype = dtype;
        tensor.shape = std::move(shape);
        const size_t count = static_cast<size_t>(element_count);

        if (dtype == DType::Float32) {
            if (count > remaining() / sizeof(float))
                return Error{ErrorCode::InvalidModel, "weight file ends before float32 tensor: " + name};
            tensor.float32_data.resize(count);
            std::memcpy(tensor.float32_data.data(), bytes_.data() + offset_, count * sizeof(float));
            offset_ += count * sizeof(float);
        }
        else if (dtype == DType::Int8) {
            if (tensor.shape.size() != 2)
                return Error{ErrorCode::InvalidModel, "int8 tensor must be a matrix: " + name};
            if (count > remaining())
                return Error{ErrorCode::InvalidModel, "weight file ends before int8 tensor: " + name};
            tensor.int8_data.resize(count);
            std::memcpy(tensor.int8_data.data(), bytes_.data() + offset_, count);
            offset_ += count;

            const size_t scale_count = tensor.shape[0];
            if (scale_count > remaining() / sizeof(float))
                return Error{ErrorCode::InvalidModel, "weight file ends before int8 scales: " + name};
            tensor.quantization_scales.resize(scale_count);
            std::memcpy(tensor.quantization_scales.data(), bytes_.data() + offset_, scale_count * sizeof(float));
            offset_ += scale_count * sizeof(float);
        }
        else {
            return Error{ErrorCode::UnsupportedModel, "unsupported tensor dtype: " + name};
        }
        return tensor;
    }

    [[nodiscard]] bool exhausted() const noexcept
    {
        return offset_ == bytes_.size();
    }

    [[nodiscard]] size_t remaining() const noexcept
    {
        return bytes_.size() - offset_;
    }

private:
    std::vector<uint8_t> bytes_;
    size_t offset_ = 0;
};

static Result<void> add_tensor(
    WeightMapping& mapping,
    SequentialWeightReader& reader,
    const std::string& name,
    std::vector<uint32_t> shape,
    DType dtype)
{
    auto tensor = reader.take(name, std::move(shape), dtype);
    if (!tensor)
        return tensor.error();
    mapping.tensors.emplace(name, std::move(tensor).value());
    return {};
}

bool FixtureModelAdapter::can_load(const ModelManifest& manifest) const
{
    return manifest.model_type == "test_moe";
}

Result<MoeIR> FixtureModelAdapter::parse_model(const ModelPackage& package) const
{
    if (package.manifest.model_type != "test_moe")
        return Error{ErrorCode::UnsupportedModel, "unsupported fixture model_type: " + package.manifest.model_type};

    const std::string& json = package.manifest.raw_json;
    auto vocabulary_size = required_uint32(json, "vocabulary_size");
    auto hidden_size = required_uint32(json, "hidden_size");
    auto intermediate_size = required_uint32(json, "intermediate_size");
    auto layer_count = required_uint32(json, "layer_count");
    auto expert_count = required_uint32(json, "expert_count");
    auto top_k = required_uint32(json, "experts_per_token");
    if (!vocabulary_size || !hidden_size || !intermediate_size || !layer_count || !expert_count || !top_k) {
        const Error* error = !vocabulary_size     ? &vocabulary_size.error()
                             : !hidden_size       ? &hidden_size.error()
                             : !intermediate_size ? &intermediate_size.error()
                             : !layer_count       ? &layer_count.error()
                             : !expert_count      ? &expert_count.error()
                                                  : &top_k.error();
        return *error;
    }

    auto activation = parse_activation(optional_string(json, "expert_activation", "silu"));
    if (!activation)
        return activation.error();
    auto layout = parse_layout(optional_string(json, "expert_layout", "gate_up_down"));
    if (!layout)
        return layout.error();
    auto expert_weight_dtype = parse_dtype(optional_string(json, "expert_weight_dtype", "float32"));
    if (!expert_weight_dtype)
        return expert_weight_dtype.error();
    auto kv_cache_dtype = parse_kv_cache_dtype(optional_string(json, "kv_cache_dtype", "float32"));
    if (!kv_cache_dtype)
        return kv_cache_dtype.error();

    MoeIR descriptor;
    descriptor.model_type = "test_moe";
    descriptor.vocabulary_size = vocabulary_size.value();
    descriptor.hidden_size = hidden_size.value();
    descriptor.intermediate_size = intermediate_size.value();
    descriptor.layer_count = layer_count.value();
    descriptor.expert_count = expert_count.value();
    descriptor.experts_per_token = top_k.value();
    descriptor.kv_cache_dtype = kv_cache_dtype.value();
    descriptor.norm_epsilon = optional_float(json, "norm_epsilon", 1e-5f);

    const uint32_t layer_flags
        = optional_bool(json, "use_attention", false)
              ? LayerDescriptorAttention | LayerDescriptorMoe
              : LayerDescriptorMoe;
    uint32_t sliding_window = 0;
    uint32_t initial_context_length = 0;
    uint32_t max_context_length = 0;
    if (has_flag(layer_flags, LayerDescriptorAttention)) {
        auto attention_head_count = required_uint32(json, "attention_head_count");
        auto kv_head_count = required_uint32(json, "kv_head_count");
        auto head_dimension = required_uint32(json, "head_dimension");
        auto parsed_sliding_window = required_uint32(json, "sliding_window");
        auto parsed_initial_context_length = required_uint32(json, "initial_context_length");
        auto parsed_max_context_length = required_uint32(json, "max_context_length");
        if (!attention_head_count || !kv_head_count || !head_dimension || !parsed_sliding_window
            || !parsed_initial_context_length || !parsed_max_context_length) {
            const Error* error = !attention_head_count            ? &attention_head_count.error()
                                 : !kv_head_count                 ? &kv_head_count.error()
                                 : !head_dimension                ? &head_dimension.error()
                                 : !parsed_sliding_window         ? &parsed_sliding_window.error()
                                 : !parsed_initial_context_length ? &parsed_initial_context_length.error()
                                                                  : &parsed_max_context_length.error();
            return *error;
        }
        descriptor.attention_head_count = attention_head_count.value();
        descriptor.kv_head_count = kv_head_count.value();
        descriptor.head_dimension = head_dimension.value();
        sliding_window = parsed_sliding_window.value();
        initial_context_length = parsed_initial_context_length.value();
        max_context_length = parsed_max_context_length.value();
    }

    MoeDescriptor moe;
    moe.expert_count = descriptor.expert_count;
    moe.top_k = descriptor.experts_per_token;
    moe.intermediate_size = descriptor.intermediate_size;
    moe.activation = activation.value();
    moe.layout = layout.value();
    moe.expert_weight_dtype = expert_weight_dtype.value();
    moe.flags = 0;
    if (optional_bool(json, "normalize_topk_weights", true))
        moe.flags |= MoeDescriptorNormalizeTopKWeights;
    moe.normalization = has_flag(moe.flags, MoeDescriptorNormalizeTopKWeights)
                            ? RouterNormalization::SelectedExperts
                            : RouterNormalization::None;
    if (optional_bool(json, "use_expert_bias", false))
        moe.flags |= MoeDescriptorExpertBias | MoeDescriptorRouterBias;
    moe.activation_limit = optional_float(json, "activation_limit", 0.0f);

    descriptor.layers.resize(descriptor.layer_count);
    for (uint32_t layer_id = 0; layer_id < descriptor.layer_count; ++layer_id) {
        LayerDescriptor& layer = descriptor.layers[layer_id];
        layer.ffn.moe = moe;
        layer.flags = layer_flags;
        layer.nodes.push_back({ModelNodeType::RmsNorm});
        if (has_flag(layer_flags, LayerDescriptorAttention)) {
            layer.pre_attention_norm = NormType::RmsNorm;
            layer.attention.head_count = descriptor.attention_head_count;
            layer.attention.kv_head_count = descriptor.kv_head_count;
            layer.attention.head_dimension = descriptor.head_dimension;
            layer.attention.sliding_window = layer_id % 2 == 0 ? sliding_window : 0;
            layer.attention.initial_context_length = initial_context_length;
            layer.attention.max_context_length = max_context_length;
            layer.attention.rope_theta = optional_float(json, "rope_theta", 10000.0f);
            layer.attention.rope_scaling_factor = optional_float(json, "rope_scaling_factor", 1.0f);
            layer.attention.rope_ntk_alpha = optional_float(json, "rope_ntk_alpha", 1.0f);
            layer.attention.rope_ntk_beta = optional_float(json, "rope_ntk_beta", 32.0f);
            if (optional_bool(json, "attention_bias", true))
                layer.attention.flags |= AttentionDescriptorBias;
            if (optional_bool(json, "attention_sinks", true))
                layer.attention.flags |= AttentionDescriptorSinks;
            layer.nodes = {
                {ModelNodeType::RmsNorm},
                {ModelNodeType::FusedQkv},
                {ModelNodeType::Rope},
            };
            if (has_flag(layer.attention.flags, AttentionDescriptorSinks))
                layer.nodes.push_back({ModelNodeType::AttentionSink});
            layer.nodes.push_back({ModelNodeType::Sdpa});
            layer.nodes.push_back({ModelNodeType::Projection});
            layer.nodes.push_back({ModelNodeType::RmsNorm});
        }
        layer.nodes.push_back({ModelNodeType::Router});
        layer.nodes.push_back({ModelNodeType::TopK});
        layer.nodes.push_back({ModelNodeType::ExpertGroup});
        layer.nodes.push_back({ModelNodeType::Combine});
    }

    return descriptor;
}

Result<WeightMapping> FixtureModelAdapter::map_weights(
    const ModelPackage& package,
    const MoeIR& descriptor) const
{
    if (descriptor.model_type != "test_moe")
        return Error{ErrorCode::UnsupportedModel, "unsupported fixture model_type: " + descriptor.model_type};

    const std::string weights_name = optional_string(
        package.manifest.raw_json, "weights_file", "model.test.bin");
    auto bytes = read_binary_file(package.root / weights_name);
    if (!bytes)
        return bytes.error();

    SequentialWeightReader reader(std::move(bytes).value());
    WeightMapping mapping;
    auto add = [&](const std::string& name, std::vector<uint32_t> shape, DType dtype) -> Result<void> {
        return add_tensor(mapping, reader, name, std::move(shape), dtype);
    };

    auto status = add(
        "token_embedding.weight", {descriptor.vocabulary_size, descriptor.hidden_size}, DType::Float32);
    if (!status)
        return status.error();

    for (uint32_t layer_id = 0; layer_id < descriptor.layer_count; ++layer_id) {
        const MoeDescriptor& moe = descriptor.layers[layer_id].ffn.moe;
        const std::string layer = layer_prefix(layer_id);
        if (has_flag(descriptor.layers[layer_id].flags, LayerDescriptorAttention)) {
            const AttentionDescriptor& attention = descriptor.layers[layer_id].attention;
            const uint32_t query_size = attention.head_count * attention.head_dimension;
            const uint32_t key_value_size = attention.kv_head_count * attention.head_dimension;
            status = add(layer + "pre_attention_norm.weight", {descriptor.hidden_size}, DType::Float32);
            if (!status)
                return status.error();
            status = add(layer + "attention.query.weight", {query_size, descriptor.hidden_size}, DType::Float32);
            if (!status)
                return status.error();
            if (has_flag(attention.flags, AttentionDescriptorBias)) {
                status = add(layer + "attention.query.bias", {query_size}, DType::Float32);
                if (!status)
                    return status.error();
            }
            status = add(layer + "attention.key.weight", {key_value_size, descriptor.hidden_size}, DType::Float32);
            if (!status)
                return status.error();
            if (has_flag(attention.flags, AttentionDescriptorBias)) {
                status = add(layer + "attention.key.bias", {key_value_size}, DType::Float32);
                if (!status)
                    return status.error();
            }
            status = add(layer + "attention.value.weight", {key_value_size, descriptor.hidden_size}, DType::Float32);
            if (!status)
                return status.error();
            if (has_flag(attention.flags, AttentionDescriptorBias)) {
                status = add(layer + "attention.value.bias", {key_value_size}, DType::Float32);
                if (!status)
                    return status.error();
            }
            status = add(layer + "attention.output.weight", {descriptor.hidden_size, query_size}, DType::Float32);
            if (!status)
                return status.error();
            if (has_flag(attention.flags, AttentionDescriptorBias)) {
                status = add(layer + "attention.output.bias", {descriptor.hidden_size}, DType::Float32);
                if (!status)
                    return status.error();
            }
            if (has_flag(attention.flags, AttentionDescriptorSinks)) {
                status = add(layer + "attention.sinks", {attention.head_count}, DType::Float32);
                if (!status)
                    return status.error();
            }
        }
        status = add(layer + "pre_ffn_norm.weight", {descriptor.hidden_size}, DType::Float32);
        if (!status)
            return status.error();
        status = add(layer + "router.weight", {moe.expert_count, descriptor.hidden_size}, DType::Float32);
        if (!status)
            return status.error();
        if (has_flag(moe.flags, MoeDescriptorRouterBias)) {
            status = add(layer + "router.bias", {moe.expert_count}, DType::Float32);
            if (!status)
                return status.error();
        }

        for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id) {
            const std::string expert = expert_prefix(layer_id, expert_id);
            if (moe.layout == ExpertLayout::GateUpDown) {
                status = add(expert + "gate.weight", {moe.intermediate_size, descriptor.hidden_size}, moe.expert_weight_dtype);
                if (!status)
                    return status.error();
            }
            status = add(expert + "up.weight", {moe.intermediate_size, descriptor.hidden_size}, moe.expert_weight_dtype);
            if (!status)
                return status.error();
            status = add(expert + "down.weight", {descriptor.hidden_size, moe.intermediate_size}, moe.expert_weight_dtype);
            if (!status)
                return status.error();
        }
    }

    status = add("final_norm.weight", {descriptor.hidden_size}, DType::Float32);
    if (!status)
        return status.error();
    status = add("lm_head.weight", {descriptor.vocabulary_size, descriptor.hidden_size}, DType::Float32);
    if (!status)
        return status.error();

    if (!reader.exhausted()) {
        return Error{
            ErrorCode::InvalidModel,
            "weight file contains " + std::to_string(reader.remaining()) + " trailing bytes"};
    }
    return mapping;
}

} // namespace moe
} // namespace ncnn
