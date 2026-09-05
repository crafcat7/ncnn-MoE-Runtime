#include "modeladapter.h"

#include "safetensors.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <span>
#include <sstream>
#include <utility>

namespace ncnn {
namespace moe {

Result<uint32_t> read_manifest_uint32(const std::string& json, const std::string& key, const char* prefix)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing integer field: " + key};

    try
    {
        const unsigned long long value = std::stoull(match[1].str());
        if (value > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest integer is out of range: " + key};
        return static_cast<uint32_t>(value);
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "integer field: " + key};
    }
}

Result<std::string> read_manifest_string(const std::string& json, const std::string& key, const char* prefix)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing string field: " + key};
    return match[1].str();
}

Result<float> read_manifest_float(const std::string& json, const std::string& key, const char* prefix)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*([-+]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing numeric field: " + key};
    try
    {
        return std::stof(match[1].str());
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "numeric field: " + key};
    }
}

Result<bool> read_manifest_bool(const std::string& json, const std::string& key, const char* prefix)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing boolean field: " + key};
    return match[1].str() == "true";
}

float optional_manifest_float(const std::string& json, const std::string& key, float fallback)
{
    auto value = read_manifest_float(json, key);
    return value ? value.value() : fallback;
}

Result<uint32_t> get_rotary_dimension(
    uint32_t head_dimension,
    float partial_rotary_factor,
    const char* description)
{
    const float value = static_cast<float>(head_dimension) * partial_rotary_factor;
    const float rounded = std::round(value);
    if (!std::isfinite(value) || value < 0.0f
        || static_cast<double>(rounded) > std::numeric_limits<uint32_t>::max())
    {
        return Error{ErrorCode::InvalidModel, description};
    }

    const uint32_t dimension = static_cast<uint32_t>(rounded);
    if (dimension == 0 || dimension > head_dimension || dimension % 2 != 0
        || std::fabs(value - static_cast<float>(dimension)) > 1e-4f)
    {
        return Error{ErrorCode::InvalidModel, description};
    }
    return dimension;
}

Result<uint64_t> fnv1a64_file(
    const std::filesystem::path& path,
    const char* description)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open " + std::string(description) + ": " + path.string()};

    uint64_t hash = UINT64_C(14695981039346656037);
    std::array<char, 64 * 1024> buffer;
    while (stream)
    {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        for (std::streamsize i = 0; i < count; ++i)
        {
            hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(i)]);
            hash *= UINT64_C(1099511628211);
        }
    }

    if (!stream.eof())
        return Error{ErrorCode::IoError, "cannot read " + std::string(description) + ": " + path.string()};

    return hash;
}

Result<bool> optional_artifact_exists(
    const std::filesystem::path& path,
    const char* description)
{
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error)
        return Error{ErrorCode::IoError, "cannot inspect the optional " + std::string(description)};
    if (exists && !std::filesystem::is_regular_file(path, error))
        return Error{ErrorCode::InvalidModel, "the optional " + std::string(description) + " path is not a regular file"};
    if (error)
        return Error{ErrorCode::IoError, "cannot inspect the optional " + std::string(description)};
    return exists;
}

std::string mxfp4_artifact_identity_name(
    const char* prefix,
    uint32_t layer_count,
    uint32_t mtp_layer_count,
    uint32_t expert_count,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    uint64_t config_hash,
    uint64_t index_hash)
{
    std::ostringstream name;
    name << prefix
         << layer_count << '.' << mtp_layer_count << '.'
         << expert_count << '.'
         << hidden_size << '.'
         << intermediate_size << '.'
         << std::hex << std::setfill('0')
         << std::setw(16) << config_hash << '.'
         << std::setw(16) << index_hash;
    return name.str();
}

Result<void> validate_mxfp4_artifact_identity(
    const SafetensorsArchive& archive,
    const std::filesystem::path& model_root,
    const char* identity_prefix,
    uint32_t layer_count,
    uint32_t mtp_layer_count,
    uint32_t expert_count,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    const char* identity_description,
    const char* artifact_description)
{
    auto config_hash = fnv1a64_file(model_root / "config.json", identity_description);
    if (!config_hash)
        return config_hash.error();
    auto index_hash = fnv1a64_file(model_root / "model.safetensors.index.json", identity_description);
    if (!index_hash)
        return index_hash.error();
    const std::string identity = mxfp4_artifact_identity_name(
        identity_prefix,
        layer_count,
        mtp_layer_count,
        expert_count,
        hidden_size,
        intermediate_size,
        config_hash.value(),
        index_hash.value());
    auto status = validate_u8_artifact_tensor(archive, identity, {0}, artifact_description);
    if (!status)
    {
        return Error{
            ErrorCode::InvalidModel,
            status.error().message
                + "; rebuild the artifact for this exact checkpoint or remove it to use BF16 Experts"};
    }
    return {};
}

Result<void> validate_u8_artifact_tensor(
    const SafetensorsArchive& archive,
    const std::string& name,
    const std::vector<uint32_t>& shape,
    const char* description)
{
    const SafetensorInfo* info = archive.find(name);
    if (!info)
        return Error{ErrorCode::InvalidModel, std::string(description) + " is missing tensor: " + name};

    uint64_t expected_size = 1;
    for (uint32_t dimension : shape)
    {
        if (dimension != 0 && expected_size > std::numeric_limits<uint64_t>::max() / dimension)
            return Error{ErrorCode::InvalidModel, std::string(description) + " tensor is too large: " + name};

        expected_size *= dimension;
    }

    if (info->dtype != "U8" || info->shape != shape || info->size != expected_size)
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(description) + " tensor: " + name};

    return {};
}

Result<void> validate_mxfp4_artifact_expert_bank(
    const SafetensorsArchive& archive,
    const std::string& prefix,
    uint32_t expert_count,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    const char* description)
{
    auto status = validate_u8_artifact_tensor(
        archive,
        prefix + "gate_up.blocks",
        {expert_count, intermediate_size * 2, hidden_size / 32, 16},
        description);
    if (!status)
        return status.error();

    status = validate_u8_artifact_tensor(
        archive,
        prefix + "gate_up.scales",
        {expert_count, intermediate_size * 2, hidden_size / 32},
        description);
    if (!status)
        return status.error();

    status = validate_u8_artifact_tensor(
        archive,
        prefix + "down.blocks",
        {expert_count, hidden_size, intermediate_size / 32, 16},
        description);
    if (!status)
        return status.error();

    return validate_u8_artifact_tensor(
        archive,
        prefix + "down.scales",
        {expert_count, hidden_size, intermediate_size / 32},
        description);
}

Result<void> add_tensor(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& target,
    const std::string& source)
{
    auto tensor = archive.load_tensor(source);
    if (!tensor)
        return tensor.error();

    mapping.emplace(target, std::move(tensor).value());
    return {};
}

Result<void> add_bfloat16_slice(
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
    mapping.emplace(target_name, std::move(tensor).value());
    return {};
}

static Result<void> add_query_gate(
    WeightMapping& mapping,
    const SafetensorsArchive& archive,
    const std::string& source,
    const std::string& target_prefix,
    uint32_t head_count,
    uint32_t head_dimension,
    uint32_t hidden_size,
    const char* description)
{
    const uint64_t query_rows = static_cast<uint64_t>(head_count) * head_dimension;
    if (query_rows == 0 || query_rows > std::numeric_limits<uint32_t>::max() / 2 || hidden_size == 0)
    {
        return Error{ErrorCode::InvalidModel, "invalid interleaved " + std::string(description) + " query/gate tensor: " + source};
    }
    const uint32_t combined_rows = static_cast<uint32_t>(query_rows * 2);
    if (query_rows * hidden_size > std::numeric_limits<size_t>::max() / (2 * sizeof(uint16_t)))
    {
        return Error{ErrorCode::InvalidModel, "invalid interleaved " + std::string(description) + " query/gate tensor: " + source};
    }
    const size_t head_elements = static_cast<size_t>(static_cast<uint64_t>(head_dimension) * hidden_size);
    const size_t combined_head_elements = head_elements * 2;
    auto loaded = archive.load_tensor(source);
    if (!loaded)
        return loaded.error();
    const TensorData& combined = loaded.value();
    if (combined.dtype != DType::BFloat16
        || combined.shape != std::vector<uint32_t>{combined_rows, hidden_size}
        || combined.bfloat16_values().size() != combined.element_count())
    {
        return Error{ErrorCode::InvalidModel, "invalid interleaved " + std::string(description) + " query/gate tensor: " + source};
    }

    TensorData query;
    query.dtype = DType::BFloat16;
    query.shape = {static_cast<uint32_t>(query_rows), hidden_size};
    query.bfloat16_data.resize(static_cast<size_t>(query_rows * hidden_size));
    TensorData gate = query;
    const std::span<const uint16_t> values = combined.bfloat16_values();
    for (uint32_t head = 0; head < head_count; ++head)
    {
        const uint16_t* source_head = values.data() + static_cast<size_t>(head) * combined_head_elements;
        std::copy_n(source_head, head_elements, query.bfloat16_data.data() + static_cast<size_t>(head) * head_elements);
        std::copy_n(source_head + head_elements, head_elements, gate.bfloat16_data.data() + static_cast<size_t>(head) * head_elements);
    }
    mapping.emplace(target_prefix + "attention.query.weight", std::move(query));
    mapping.emplace(target_prefix + "attention.output_gate.weight", std::move(gate));
    return {};
}

Result<void> add_qwen_attention(
    WeightMapping& mapping, const SafetensorsArchive& archive,
    const std::string& source_prefix, const std::string& target_prefix,
    uint32_t head_count, uint32_t head_dimension, uint32_t hidden_size, const char* description)
{
    auto ret = add_query_gate(
        mapping, archive, source_prefix + "self_attn.q_proj.weight", target_prefix,
        head_count, head_dimension, hidden_size, description);
    if (!ret)
        return ret.error();

    const std::pair<const char*, const char*> attention_tensors[] = {
        {"attention.key.weight", "self_attn.k_proj.weight"},
        {"attention.value.weight", "self_attn.v_proj.weight"},
        {"attention.output.weight", "self_attn.o_proj.weight"},
        {"attention.query_norm.weight", "self_attn.q_norm.weight"},
        {"attention.key_norm.weight", "self_attn.k_norm.weight"},
    };
    for (const auto& item : attention_tensors)
    {
        ret = add_tensor(mapping, archive, target_prefix + item.first, source_prefix + item.second);
        if (!ret)
            return ret.error();
    }
    return {};
}

Result<void> add_qwen_gated_delta_net(
    WeightMapping& mapping, const SafetensorsArchive& archive,
    const std::string& source_prefix, const std::string& target_prefix)
{
    const std::pair<const char*, const char*> delta_tensors[] = {
        {"attention.delta.qkv.weight", "linear_attn.in_proj_qkv.weight"},
        {"attention.delta.z.weight", "linear_attn.in_proj_z.weight"},
        {"attention.delta.beta.weight", "linear_attn.in_proj_b.weight"},
        {"attention.delta.alpha.weight", "linear_attn.in_proj_a.weight"},
        {"attention.delta.convolution.weight", "linear_attn.conv1d.weight"},
        {"attention.delta.time_bias", "linear_attn.dt_bias"},
        {"attention.delta.decay_log", "linear_attn.A_log"},
        {"attention.delta.norm.weight", "linear_attn.norm.weight"},
        {"attention.output.weight", "linear_attn.out_proj.weight"},
    };
    for (const auto& item : delta_tensors)
    {
        auto ret = add_tensor(mapping, archive, target_prefix + item.first, source_prefix + item.second);
        if (!ret)
            return ret.error();
    }
    return {};
}

Result<void> add_qwen_shared_expert(
    WeightMapping& mapping, const SafetensorsArchive& archive,
    const std::string& source_prefix, const std::string& target_prefix)
{
    const std::pair<const char*, const char*> shared_expert_tensors[] = {
        {"shared_expert.gate.weight", "mlp.shared_expert.gate_proj.weight"},
        {"shared_expert.up.weight", "mlp.shared_expert.up_proj.weight"},
        {"shared_expert.down.weight", "mlp.shared_expert.down_proj.weight"},
        {"shared_expert.router_gate.weight", "mlp.shared_expert_gate.weight"},
    };
    for (const auto& item : shared_expert_tensors)
    {
        auto ret = add_tensor(mapping, archive, target_prefix + item.first, source_prefix + item.second);
        if (!ret)
            return ret.error();
    }
    return {};
}

Result<void> add_mxfp4_expert(
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
    mapping.emplace(target_name, std::move(tensor).value());
    return {};
}

} // namespace moe
} // namespace ncnn
