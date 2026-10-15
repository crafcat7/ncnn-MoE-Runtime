#include "safetensors.h"
#include "json.h"
#include "kernels/int4.h"
#include "kernels/qnk.h"
#include "storage/mappedfile.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstring>
#include <fstream>
#include <cmath>
#include <limits>
#include <regex>
#include <sstream>
#include <utility>

namespace ncnn {
namespace moe {

struct SourceQuantizationConfig
{
    std::string method;
    int64_t group_size = 0;
    bool desc_act = false;
    bool symmetric = true;
};

static std::string lowercase_quantization_string(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

static Result<SourceQuantizationConfig> parse_source_quantization_config(std::string_view json)
{
    if (json.empty() || json.front() != '{')
        return Error{ErrorCode::InvalidModel, "quantization_config must be an object"};
    SourceQuantizationConfig config;
    const auto method = json_string(json, "quant_method");
    if (!method)
        return config;
    config.method = lowercase_quantization_string(*method);
    if (config.method != "awq" && config.method != "gptq")
    {
        config.method.clear();
        return config;
    }
    const auto bits = json_integer(json, "bits");
    const auto w_bit = json_integer(json, "w_bit");
    const auto group = json_integer(json, "group_size");
    const auto q_group = json_integer(json, "q_group_size");
    if ((find_manifest_member(json, "bits") && !bits) || (find_manifest_member(json, "w_bit") && !w_bit)
        || (bits && w_bit && *bits != *w_bit) || bits.value_or(w_bit.value_or(0)) != 4
        || (find_manifest_member(json, "group_size") && !group) || (find_manifest_member(json, "q_group_size") && !q_group)
        || (group && q_group && *group != *q_group))
    {
        return Error{ErrorCode::UnsupportedModel, "AWQ/GPTQ require consistent 4-bit and group-size metadata"};
    }
    config.group_size = group.value_or(q_group.value_or(0));
    if (config.group_size == 0 || config.group_size < -1
        || (config.group_size > 0 && static_cast<uint64_t>(config.group_size) > std::numeric_limits<uint32_t>::max()))
    {
        return Error{ErrorCode::UnsupportedModel, "AWQ/GPTQ group_size must be positive or -1"};
    }
    if (const auto backend = find_manifest_member(json, "backend"))
    {
        if (*backend != "null" && json_string(json, "backend") != "auto"
            && json_string(json, "backend") != "auto_trainable")
        {
            return Error{ErrorCode::UnsupportedModel, "repacked AWQ/GPTQ backends are unsupported"};
        }
    }
    for (const char* key : {"modules_in_block_to_quantize", "dynamic"})
    {
        const auto value = find_manifest_member(json, key);
        if (value && *value != "null" && *value != "false" && *value != "{}" && *value != "[]" && *value != "0" && *value != "\"\"")
            return Error{ErrorCode::UnsupportedModel, "AWQ/GPTQ per-module quantization overrides are unsupported"};
    }
    if (config.method == "awq")
    {
        if ((find_manifest_member(json, "version") && (!json_string(json, "version") || lowercase_quantization_string(*json_string(json, "version")) != "gemm"))
            || (find_manifest_member(json, "zero_point") && json_boolean(json, "zero_point") != true))
        {
            return Error{ErrorCode::UnsupportedModel, "only asymmetric AWQ GEMM checkpoints are supported"};
        }
    }
    else
    {
        if (find_manifest_member(json, "checkpoint_format")
            && (!json_string(json, "checkpoint_format") || lowercase_quantization_string(*json_string(json, "checkpoint_format")) != "gptq"))
        {
            return Error{ErrorCode::UnsupportedModel, "only legacy GPTQ checkpoints are supported; GPTQ v2/Marlin are unsupported"};
        }
        for (const char* key : {"desc_act", "sym"})
        {
            if (find_manifest_member(json, key) && !json_boolean(json, key))
                return Error{ErrorCode::InvalidModel, "GPTQ desc_act/sym must be boolean"};
        }
        config.desc_act = json_boolean(json, "desc_act").value_or(false);
        config.symmetric = json_boolean(json, "sym").value_or(true);
        for (const char* key : {"is_marlin_format", "use_marlin"})
        {
            if (const auto value = find_manifest_member(json, key); value && *value != "false" && *value != "null")
                return Error{ErrorCode::UnsupportedModel, "Marlin GPTQ checkpoints are unsupported"};
        }
    }
    return config;
}

static Result<SourceQuantizationConfig> read_source_quantization_config(const std::filesystem::path& root, std::string_view manifest_json)
{
    const std::filesystem::path path = root / "config.json";
    std::string owned_json;
    if (manifest_json.empty())
    {
        std::error_code error;
        if (!std::filesystem::exists(path, error))
        {
            if (error)
                return Error{ErrorCode::IoError, "cannot inspect model config: " + path.string()};
            return SourceQuantizationConfig{};
        }
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
            return Error{ErrorCode::IoError, "cannot read model config: " + path.string()};
        std::ostringstream buffer;
        buffer << stream.rdbuf();
        owned_json = buffer.str();
        manifest_json = owned_json;
    }
    const std::string_view json = manifest_json;
    size_t position = skip_space(json, 0);
    if (position >= json.size() || json[position] != '{' || !parse_json_value(json, position) || skip_space(json, position) != json.size())
        return Error{ErrorCode::InvalidModel, "invalid model config JSON: " + path.string()};
    const std::string_view object(json);
    const auto outer = find_manifest_member(object, "quantization_config");
    const auto text = find_manifest_member(object, "text_config");
    if (text && (text->empty() || text->front() != '{'))
        return Error{ErrorCode::InvalidModel, "model text_config must be an object"};
    const auto inner = text ? find_manifest_member(*text, "quantization_config") : std::nullopt;
    const auto selected = outer && *outer != "null" ? outer : inner;
    if (!selected || *selected == "null")
        return SourceQuantizationConfig{};
    auto config = parse_source_quantization_config(*selected);
    if (!config)
        return config.error();
    if (outer && *outer != "null" && inner && *inner != "null")
    {
        auto nested = parse_source_quantization_config(*inner);
        if (!nested)
            return nested.error();
        if (config.value().method != nested.value().method || config.value().group_size != nested.value().group_size
            || config.value().desc_act != nested.value().desc_act || config.value().symmetric != nested.value().symmetric)
        {
            return Error{ErrorCode::InvalidModel, "conflicting nested quantization_config"};
        }
    }
    return config;
}

static Result<std::vector<uint8_t>> read_range(const std::filesystem::path& path, uint64_t offset, uint64_t size)
{
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) || size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return Error{ErrorCode::InvalidModel, "safetensors range is too large: " + path.string()};

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open safetensors shard: " + path.string()};
    stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot seek safetensors shard: " + path.string()};

    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size > 0 && !stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size)))
        return Error{ErrorCode::InvalidModel, "safetensors shard is truncated: " + path.string()};
    return data;
}

static std::string packed_tensor_name(const std::string& name)
{
    return "__ncnn_moe_packed__." + name;
}

static bool parse_json_string(const std::string& json, size_t& position, std::string& value)
{
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position])))
        ++position;
    if (position >= json.size() || json[position] != '"')
        return false;
    ++position;
    value.clear();
    while (position < json.size())
    {
        const char character = json[position++];
        if (character == '"')
            return true;
        if (character == '\\' && position < json.size())
        {
            value.push_back(json[position++]);
        }
        else
        {
            value.push_back(character);
        }
    }
    return false;
}

static bool find_object_end(const std::string& json, size_t begin, size_t& end)
{
    if (begin >= json.size() || json[begin] != '{')
        return false;
    uint32_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t position = begin; position < json.size(); ++position)
    {
        const char character = json[position];
        if (in_string)
        {
            if (escaped)
            {
                escaped = false;
            }
            else if (character == '\\')
            {
                escaped = true;
            }
            else if (character == '"')
            {
                in_string = false;
            }
            continue;
        }
        if (character == '"')
        {
            in_string = true;
        }
        else if (character == '{')
        {
            ++depth;
        }
        else if (character == '}')
        {
            if (--depth == 0)
            {
                end = position + 1;
                return true;
            }
        }
    }
    return false;
}

static Result<std::vector<uint32_t>> parse_shape(const std::string& object)
{
    const auto dimensions = json_integer_array(object, "shape");
    if (!dimensions)
        return Error{ErrorCode::InvalidModel, "invalid safetensors shape array"};
    std::vector<uint32_t> shape;
    shape.reserve(dimensions->size());
    for (const int64_t dimension : *dimensions)
    {
        if (dimension < 0 || static_cast<uint64_t>(dimension) > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, "safetensors dimension is out of range"};
        shape.push_back(static_cast<uint32_t>(dimension));
    }
    return shape;
}

static Result<void> parse_header(const std::filesystem::path& path,
                                 const std::string& json,
                                 uint64_t data_start,
                                 uint64_t file_size,
                                 std::unordered_map<std::string, SafetensorInfo>& tensors)
{
    const std::regex dtype_expression("\"dtype\"\\s*:\\s*\"([^\"]+)\"");
    const std::regex offsets_expression("\"data_offsets\"\\s*:\\s*\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
    size_t position = 1;
    while (position < json.size())
    {
        while (position < json.size() && (std::isspace(static_cast<unsigned char>(json[position])) || json[position] == ','))
            ++position;
        if (position >= json.size() || json[position] == '}')
            break;

        std::string name;
        if (!parse_json_string(json, position, name))
            return Error{ErrorCode::InvalidModel, "invalid safetensors header key: " + path.string()};
        while (position < json.size() && (std::isspace(static_cast<unsigned char>(json[position])) || json[position] == ':'))
            ++position;
        size_t object_end = 0;
        if (!find_object_end(json, position, object_end))
            return Error{ErrorCode::InvalidModel, "invalid safetensors tensor object: " + path.string()};
        const std::string object = json.substr(position, object_end - position);
        position = object_end;
        if (name == "__metadata__")
            continue;

        std::smatch dtype_match;
        std::smatch offsets_match;
        if (!std::regex_search(object, dtype_match, dtype_expression)
            || !std::regex_search(object, offsets_match, offsets_expression))
            return Error{ErrorCode::InvalidModel, "incomplete safetensors tensor metadata: " + name};

        auto shape = parse_shape(object);
        if (!shape)
            return shape.error();
        uint64_t begin = 0;
        uint64_t end = 0;
        try
        {
            begin = std::stoull(offsets_match[1].str());
            end = std::stoull(offsets_match[2].str());
        }
        catch (const std::exception&)
        {
            return Error{ErrorCode::InvalidModel, "invalid safetensors data offsets: " + name};
        }
        if (end < begin
            || data_start > std::numeric_limits<uint64_t>::max() - end
            || data_start + end > file_size)
            return Error{ErrorCode::InvalidModel, "invalid safetensors data range: " + name};
        if (tensors.contains(name))
            return Error{ErrorCode::InvalidModel, "duplicate safetensors tensor: " + name};
        tensors.emplace(std::move(name), SafetensorInfo{path, dtype_match[1].str(), std::move(shape).value(), data_start + begin, end - begin});
    }
    return {};
}

static Result<void> open_safetensors_file(const std::filesystem::path& path,
                                          std::unordered_map<std::string, SafetensorInfo>& tensors)
{
    std::error_code size_error;
    const uint64_t file_size = std::filesystem::file_size(path, size_error);
    if (size_error)
        return Error{ErrorCode::IoError, "cannot inspect safetensors shard: " + path.string()};

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open safetensors shard: " + path.string()};
    uint64_t header_length = 0;
    stream.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
    if (!stream || header_length == 0 || header_length > 128 * 1024 * 1024)
        return Error{ErrorCode::InvalidModel, "invalid safetensors header length: " + path.string()};
    if (header_length > file_size - std::min<uint64_t>(file_size, sizeof(header_length)))
        return Error{ErrorCode::InvalidModel, "truncated safetensors header: " + path.string()};
    std::string header(static_cast<size_t>(header_length), '\0');
    stream.read(header.data(), static_cast<std::streamsize>(header_length));
    if (!stream)
        return Error{ErrorCode::InvalidModel, "truncated safetensors header: " + path.string()};
    return parse_header(path,
                        header,
                        sizeof(header_length) + header_length,
                        file_size,
                        tensors);
}

static std::string source_quantized_prefix(const std::string& name)
{
    if (name.ends_with(".weight"))
        return name.substr(0, name.size() - 6);
    if (name.ends_with(".qweight"))
        return name.substr(0, name.size() - 7);
    return {};
}

struct SourceInt4TensorInfo
{
    const SafetensorInfo* weights = nullptr;
    const SafetensorInfo* zeros = nullptr;
    const SafetensorInfo* scales = nullptr;
    const SafetensorInfo* indices = nullptr;
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint32_t group_size = 0;
    uint32_t groups = 0;
};

static Result<SourceInt4TensorInfo> source_int4_tensor_info(const SafetensorsArchive& archive,
                                                            const std::string& name,
                                                            const std::string& method,
                                                            int64_t group_size,
                                                            bool desc_act)
{
    if (method != "awq" && method != "gptq")
        return Error{ErrorCode::UnsupportedModel, "source qweight requires AWQ/GPTQ quantization_config: " + name};
    const std::string prefix = source_quantized_prefix(name);
    SourceInt4TensorInfo info;
    info.weights = archive.find(prefix + "qweight");
    info.zeros = archive.find(prefix + "qzeros");
    info.scales = archive.find(prefix + "scales");
    info.indices = archive.find(prefix + "g_idx");
    if (prefix.empty() || !info.weights || !info.zeros || !info.scales || archive.find(prefix + "weight"))
        return Error{ErrorCode::InvalidModel, "incomplete or duplicate AWQ/GPTQ projection: " + name};
    if (info.weights->dtype != "I32" || info.weights->shape.size() != 2
        || info.weights->shape[0] == 0 || info.weights->shape[1] == 0)
    {
        return Error{ErrorCode::InvalidModel, "AWQ/GPTQ qweight must be a nonempty I32 matrix: " + name};
    }
    const uint64_t rows = method == "awq" ? static_cast<uint64_t>(info.weights->shape[1]) * 8 : info.weights->shape[1];
    const uint64_t columns = method == "gptq" ? static_cast<uint64_t>(info.weights->shape[0]) * 8 : info.weights->shape[0];
    if (rows > std::numeric_limits<uint32_t>::max() || columns > std::numeric_limits<uint32_t>::max()
        || rows % 8 != 0 || info.weights->size != rows * columns / 2)
    {
        return Error{ErrorCode::InvalidModel, "AWQ/GPTQ packed weight shape or byte count is invalid: " + name};
    }
    info.rows = static_cast<uint32_t>(rows);
    info.columns = static_cast<uint32_t>(columns);
    info.group_size = group_size == -1 ? info.columns : static_cast<uint32_t>(group_size);
    if (info.group_size == 0)
        return Error{ErrorCode::InvalidModel, "AWQ/GPTQ group size is invalid: " + name};
    info.groups = info.columns / info.group_size + (info.columns % info.group_size != 0);
    const uint64_t metadata_elements = rows * info.groups;
    const uint32_t scale_bytes = info.scales->dtype == "F32" ? 4 : info.scales->dtype == "F16" || info.scales->dtype == "BF16" ? 2
                                                                                                                               : 0;
    if (info.zeros->dtype != "I32" || info.zeros->shape != std::vector<uint32_t>{info.groups, info.rows / 8}
        || info.zeros->size != metadata_elements / 2 || scale_bytes == 0
        || info.scales->shape != std::vector<uint32_t>{info.groups, info.rows}
        || metadata_elements > std::numeric_limits<size_t>::max() / sizeof(float)
        || info.scales->size != metadata_elements * scale_bytes
        || rows * ((columns + 1) / 2) > std::numeric_limits<size_t>::max())
    {
        return Error{ErrorCode::InvalidModel, "AWQ/GPTQ qzeros/scales shapes or byte counts disagree with groups: " + name};
    }
    if ((method == "awq" && info.indices) || (desc_act && !info.indices)
        || (info.indices && (info.indices->dtype != "I32" || info.indices->shape != std::vector<uint32_t>{info.columns} || info.indices->size != columns * sizeof(int32_t))))
    {
        return Error{ErrorCode::InvalidModel, "invalid AWQ/GPTQ g_idx metadata: " + name};
    }
    return info;
}

Result<SafetensorsArchive> SafetensorsArchive::open(const std::filesystem::path& root, std::string_view manifest_json)
{
    SafetensorsArchive archive;
    std::error_code filesystem_error;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(root, filesystem_error))
    {
        if (filesystem_error)
            return Error{ErrorCode::IoError, "cannot enumerate model directory: " + root.string()};
        if (!entry.is_regular_file() || entry.path().extension() != ".safetensors")
            continue;

        auto status = open_safetensors_file(entry.path(), archive.tensors);
        if (!status)
            return status.error();
    }
    if (archive.tensors.empty())
        return Error{ErrorCode::InvalidModel, "model directory contains no safetensors tensors"};
    auto config = read_source_quantization_config(root, manifest_json);
    if (!config)
        return config.error();
    archive.source_quantization_method = config.value().method;
    archive.source_quantization_group_size = config.value().group_size;
    archive.source_quantization_desc_act = config.value().desc_act;
    for (const auto& item : archive.tensors)
    {
        if (item.first.ends_with(".qweight"))
        {
            auto info = source_int4_tensor_info(archive, item.first, archive.source_quantization_method,
                                                archive.source_quantization_group_size, archive.source_quantization_desc_act);
            if (!info)
                return info.error();
        }
        else if (!archive.source_quantization_method.empty()
                 && (item.first.ends_with(".qzeros") || item.first.ends_with(".g_idx")))
        {
            const std::string prefix = item.first.substr(0, item.first.size() - (item.first.ends_with(".g_idx") ? 5 : 6));
            if (!archive.find(prefix + "qweight"))
                return Error{ErrorCode::InvalidModel, "orphan AWQ/GPTQ quantization metadata: " + item.first};
        }
    }
    return archive;
}

Result<SafetensorsArchive> SafetensorsArchive::open_file(const std::filesystem::path& path)
{
    SafetensorsArchive archive;
    auto status = open_safetensors_file(path, archive.tensors);
    if (!status)
        return status.error();
    if (archive.tensors.empty())
        return Error{ErrorCode::InvalidModel, "safetensors file contains no tensors: " + path.string()};
    return archive;
}

const SafetensorInfo* SafetensorsArchive::find(const std::string& name) const noexcept
{
    const auto iterator = tensors.find(name);
    if (iterator != tensors.end())
        return &iterator->second;
    constexpr std::string_view prefix = "model.language_model.";
    if (name.starts_with(prefix))
    {
        const auto alias = tensors.find("model." + name.substr(prefix.size()));
        if (alias != tensors.end())
            return &alias->second;
    }
    return nullptr;
}

bool SafetensorsArchive::contains_tensor(const std::string& name) const noexcept
{
    if (find(name))
        return true;
    const std::string prefix = source_quantized_prefix(name);
    return !prefix.empty() && find(prefix + "qweight");
}

bool SafetensorsArchive::has_native_quantized_tensors() const noexcept
{
    for (const auto& item : tensors)
    {
        if (item.first.ends_with(".__int4_shape") || item.first.ends_with(".__qnk_shape"))
            return true;
        if (!source_quantization_method.empty() && item.first.ends_with(".qweight"))
            return true;
    }
    return false;
}

bool SafetensorsArchive::has_float8_tensors() const noexcept
{
    return std::any_of(tensors.begin(), tensors.end(), [](const auto& item) {
        return item.second.dtype == "F8_E4M3";
    });
}

struct QuantizedTensorInfo
{
    DType dtype = DType::Float32;
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint32_t group_size = 0;
};

static Result<std::array<int64_t, 3>> read_quantized_shape(const SafetensorsArchive& archive,
                                                           const std::string& name)
{
    const SafetensorInfo* shape = archive.find(name);
    if (!shape || shape->dtype != "I64" || shape->shape != std::vector<uint32_t>{3}
        || shape->size != 3 * sizeof(int64_t))
    {
        return Error{ErrorCode::InvalidModel, "invalid quantized tensor shape metadata: " + name};
    }
    auto data = read_range(shape->path, shape->offset, shape->size);
    if (!data)
        return data.error();
    std::array<int64_t, 3> values;
    std::memcpy(values.data(), data.value().data(), sizeof(values));
    for (const int64_t value : values)
    {
        if (value <= 0 || static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, "quantized tensor dimension is out of range: " + name};
    }
    return values;
}

static Result<QuantizedTensorInfo> tensor_encoding(const SafetensorsArchive& archive,
                                                   const std::string& name)
{
    const SafetensorInfo* source = archive.find(name);
    if (!source)
        return Error{ErrorCode::InvalidModel, "missing safetensors tensor: " + name};
    const bool int4 = archive.find(name + ".__int4_shape")
                      || archive.find(name + ".__int4_scales")
                      || archive.find(name + ".__int4_zeros")
                      || archive.find(name + ".__int4_g_idx");
    const bool qnk = archive.find(name + ".__qnk_shape") || archive.find(name + ".__qnk_columns");
    if (int4 && qnk)
        return Error{ErrorCode::InvalidModel, "conflicting quantized tensor metadata: " + name};
    if (int4)
    {
        auto shape = read_quantized_shape(archive, name + ".__int4_shape");
        if (!shape)
            return shape.error();
        QuantizedTensorInfo result{DType::Int4,
                                   static_cast<uint32_t>(shape.value()[0]),
                                   static_cast<uint32_t>(shape.value()[1]),
                                   static_cast<uint32_t>(shape.value()[2])};
        const uint32_t packed_columns = result.columns / 2 + result.columns % 2;
        const uint32_t groups = result.columns / result.group_size + (result.columns % result.group_size != 0);
        const uint64_t group_count = static_cast<uint64_t>(result.rows) * groups;
        const SafetensorInfo* scales = archive.find(name + ".__int4_scales");
        const SafetensorInfo* zeros = archive.find(name + ".__int4_zeros");
        const SafetensorInfo* indices = archive.find(name + ".__int4_g_idx");
        if (source->dtype != "U8" || source->shape != std::vector<uint32_t>{result.rows, packed_columns}
            || source->size != static_cast<uint64_t>(result.rows) * packed_columns
            || group_count > std::numeric_limits<size_t>::max() / sizeof(float)
            || !scales || scales->dtype != "F32" || scales->shape != std::vector<uint32_t>{result.rows, groups}
            || scales->size != group_count * sizeof(float)
            || !zeros || zeros->dtype != "U8" || zeros->shape != scales->shape || zeros->size != group_count
            || (indices && (indices->dtype != "I64" || indices->shape != std::vector<uint32_t>{result.columns} || indices->size != static_cast<uint64_t>(result.columns) * sizeof(int64_t))))
        {
            return Error{ErrorCode::InvalidModel, "invalid native INT4 safetensors tensor: " + name};
        }
        return result;
    }
    if (qnk)
    {
        auto shape = read_quantized_shape(archive, name + ".__qnk_shape");
        if (!shape)
            return shape.error();
        DType dtype;
        switch (shape.value()[0])
        {
        case 2: dtype = DType::Q2K; break;
        case 3: dtype = DType::Q3K; break;
        case 4: dtype = DType::Q4K; break;
        case 5: dtype = DType::Q5K; break;
        case 6: dtype = DType::Q6K; break;
        case 8: dtype = DType::Q8K; break;
        default: return Error{ErrorCode::InvalidModel, "invalid Qn_K bit width: " + name};
        }
        const uint32_t rows = static_cast<uint32_t>(shape.value()[1]);
        const uint32_t columns = static_cast<uint32_t>(shape.value()[2]);
        const SafetensorInfo* column_indices = archive.find(name + ".__qnk_columns");
        uint64_t byte_count = 1;
        for (const uint32_t dimension : source->shape)
        {
            if (dimension == 0 || byte_count > std::numeric_limits<uint64_t>::max() / dimension)
                return Error{ErrorCode::InvalidModel, "invalid tagged Qn_K storage shape: " + name};
            byte_count *= dimension;
        }
        if (source->dtype != "U8" || !qnk_shape_supported(dtype, rows, columns)
            || source->size != byte_count || source->size != qnk_storage_bytes(dtype, rows, columns)
            || (column_indices && (column_indices->dtype != "I64" || column_indices->shape != std::vector<uint32_t>{columns} || column_indices->size != static_cast<uint64_t>(columns) * sizeof(int64_t))))
        {
            return Error{ErrorCode::InvalidModel, "invalid tagged Qn_K safetensors tensor: " + name};
        }
        return QuantizedTensorInfo{dtype, rows, columns, 0};
    }
    if (source->dtype == "BF16" || source->dtype == "F16")
        return QuantizedTensorInfo{DType::BFloat16};
    if (source->dtype == "F32")
        return QuantizedTensorInfo{DType::Float32};
    if (source->dtype == "I64")
        return QuantizedTensorInfo{DType::Int64};
    if (source->dtype == "F8_E4M3")
        return QuantizedTensorInfo{DType::Float8E4M3};
    return Error{ErrorCode::UnsupportedModel, "unsupported safetensors dtype for tensor: " + name};
}

Result<DType> SafetensorsArchive::tensor_dtype(const std::string& name) const
{
    const std::string prefix = source_quantized_prefix(name);
    if (!prefix.empty() && find(prefix + "qweight"))
    {
        auto info = source_int4_tensor_info(*this, name, source_quantization_method, source_quantization_group_size, source_quantization_desc_act);
        if (!info)
            return info.error();
        return DType::Int4;
    }
    auto encoding = tensor_encoding(*this, name);
    if (!encoding)
        return encoding.error();
    if (!source_quantization_method.empty() && encoding.value().dtype == DType::Float32)
        return DType::BFloat16;
    return encoding.value().dtype;
}

std::optional<DType> SafetensorsArchive::find_qnk_expert_dtype(const std::string& name,
                                                               uint32_t expert_count,
                                                               uint32_t rows,
                                                               uint32_t columns) const noexcept
{
    if (expert_count == 0 || !qnk_shape_supported(DType::Q2K, rows, columns))
        return {};

    const SafetensorInfo* source = find(name);
    const std::array<DType, 6> dtypes = {
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Q8K,
    };
    for (const DType dtype : dtypes)
    {
        const uint64_t expert_size = qnk_storage_bytes(dtype, rows, columns);
        if (expert_size == 0
            || expert_size > std::numeric_limits<uint64_t>::max() / expert_count)
        {
            continue;
        }
        const uint64_t bank_size = expert_size * expert_count;
        if (source
            && source->dtype == "U8"
            && source->size == bank_size)
        {
            return dtype;
        }
    }
    return {};
}

static uint32_t read_packed_u32(const uint8_t* data, size_t index) noexcept
{
    uint32_t value;
    std::memcpy(&value, data + index * sizeof(value), sizeof(value));
    return value;
}

static float safetensor_half_to_float(uint16_t value) noexcept
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 31u;
    uint32_t mantissa = value & 1023u;
    uint32_t bits = sign;
    if (exponent == 0)
    {
        if (mantissa != 0)
        {
            int32_t shift = 0;
            while ((mantissa & 1024u) == 0)
            {
                mantissa <<= 1;
                --shift;
            }
            bits |= static_cast<uint32_t>(113 + shift) << 23;
            bits |= (mantissa & 1023u) << 13;
        }
    }
    else if (exponent == 31)
        bits |= 0x7f800000u | (mantissa << 13);
    else
        bits |= (exponent + 112) << 23 | (mantissa << 13);
    return std::bit_cast<float>(bits);
}

static float read_safetensor_float(const uint8_t* data, size_t index, const std::string& dtype) noexcept
{
    if (dtype == "F32")
        return std::bit_cast<float>(read_packed_u32(data, index));
    uint16_t value;
    std::memcpy(&value, data + index * sizeof(value), sizeof(value));
    if (dtype == "BF16")
        return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
    return safetensor_half_to_float(value);
}

Result<TensorData> SafetensorsArchive::load_source_int4_tensor(const std::string& name) const
{
    auto parsed = source_int4_tensor_info(*this, name, source_quantization_method, source_quantization_group_size, source_quantization_desc_act);
    if (!parsed)
        return parsed.error();
    const SourceInt4TensorInfo& info = parsed.value();
    TensorData tensor;
    tensor.dtype = DType::Int4;
    tensor.shape = {info.rows, info.columns};
    tensor.quantization_group_size = info.group_size;
    if (info.indices)
    {
        auto bytes = read_range(info.indices->path, info.indices->offset, info.indices->size);
        if (!bytes)
            return bytes.error();
        tensor.quantization_group_indices.resize(info.columns);
        bool canonical = true;
        std::vector<uint32_t> counts(info.groups);
        for (uint32_t column = 0; column < info.columns; ++column)
        {
            const uint32_t group = read_packed_u32(bytes.value().data(), column);
            if (group >= info.groups || (!source_quantization_desc_act && group != column / info.group_size))
                return Error{ErrorCode::InvalidModel, "invalid or noncanonical GPTQ g_idx: " + name};
            tensor.quantization_group_indices[column] = group;
            canonical = canonical && group == column / info.group_size;
            ++counts[group];
        }
        for (uint32_t group = 0; group < info.groups; ++group)
        {
            if (counts[group] != std::min(info.group_size, info.columns - group * info.group_size))
                return Error{ErrorCode::InvalidModel, "GPTQ g_idx group counts disagree with group_size: " + name};
        }
        if (canonical)
            std::vector<uint32_t>().swap(tensor.quantization_group_indices);
    }
    auto scales = read_range(info.scales->path, info.scales->offset, info.scales->size);
    if (!scales)
        return scales.error();
    auto zeros = read_range(info.zeros->path, info.zeros->offset, info.zeros->size);
    if (!zeros)
        return zeros.error();
    const bool awq = source_quantization_method == "awq";
    constexpr std::array<uint32_t, 8> awq_reverse_order = {0, 4, 1, 5, 2, 6, 3, 7};
    const size_t metadata_count = static_cast<size_t>(info.rows) * info.groups;
    tensor.quantization_scales.resize(metadata_count);
    tensor.quantization_zero_points.resize(metadata_count);
    for (uint32_t row = 0; row < info.rows; ++row)
    {
        for (uint32_t group = 0; group < info.groups; ++group)
        {
            const size_t source_index = static_cast<size_t>(group) * info.rows + row;
            const float scale = read_safetensor_float(scales.value().data(), source_index, info.scales->dtype);
            if (!std::isfinite(scale) || scale <= 0.0f)
                return Error{ErrorCode::InvalidModel, "AWQ/GPTQ scales must be finite and positive: " + name};
            const uint32_t packed_zero = read_packed_u32(zeros.value().data(), source_index / 8);
            const uint32_t shift = (awq ? awq_reverse_order[row % 8] : row % 8) * 4;
            const uint8_t zero = static_cast<uint8_t>(((packed_zero >> shift) & 15u) + (awq ? 0 : 1));
            const size_t target_index = static_cast<size_t>(row) * info.groups + group;
            tensor.quantization_scales[target_index] = scale;
            tensor.quantization_zero_points[target_index] = zero;
        }
    }
    std::shared_ptr<MappedFileRange> mapping;
    std::vector<uint8_t> weight_bytes;
    const uint8_t* weights;
    auto mapped = MappedFileRange::open(info.weights->path, info.weights->offset, info.weights->size);
    if (mapped)
    {
        mapping = std::move(mapped).value();
        weights = mapping->data();
    }
    else
    {
        auto bytes = read_range(info.weights->path, info.weights->offset, info.weights->size);
        if (!bytes)
            return bytes.error();
        weight_bytes = std::move(bytes).value();
        weights = weight_bytes.data();
    }
    const size_t row_bytes = info.columns / 2 + info.columns % 2;
    tensor.quantized_data.resize(static_cast<size_t>(info.rows) * row_bytes, 0);
    for (uint32_t row = 0; row < info.rows; ++row)
    {
        uint8_t* output = tensor.quantized_data.data() + static_cast<size_t>(row) * row_bytes;
        for (uint32_t column = 0; column < info.columns; ++column)
        {
            const size_t source_index = awq ? static_cast<size_t>(column) * (info.rows / 8) + row / 8
                                            : static_cast<size_t>(column / 8) * info.rows + row;
            const uint32_t shift = (awq ? awq_reverse_order[row % 8] : column % 8) * 4;
            const uint8_t code = static_cast<uint8_t>((read_packed_u32(weights, source_index) >> shift) & 15u);
            output[column / 2] |= static_cast<uint8_t>(code << ((column & 1u) * 4));
        }
    }
    auto status = validate_int4_tensor(tensor);
    if (!status)
        return Error{ErrorCode::InvalidModel, status.error().message + ": " + name};
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_float32_tensor(const std::string& name) const
{
    const SafetensorInfo* info = find(name);
    if (!info || info->dtype != "F32")
        return Error{ErrorCode::InvalidModel, "expected F32 tensor: " + name};
    TensorData tensor;
    tensor.dtype = DType::Float32;
    tensor.shape = info->shape;
    if (tensor.element_count() > std::numeric_limits<uint64_t>::max() / sizeof(float)
        || info->size != tensor.element_count() * sizeof(float))
        return Error{ErrorCode::InvalidModel, "invalid F32 byte count: " + name};
    auto mapped = MappedFileRange::open(info->path, info->offset, info->size);
    if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(float) == 0)
    {
        tensor.mapped_data = mapped.value()->share_data();
        tensor.mapped_size = info->size;
        return tensor;
    }
    auto data = read_range(info->path, info->offset, info->size);
    if (!data)
        return data.error();
    tensor.float32_data.resize(static_cast<size_t>(info->size / sizeof(float)));
    std::memcpy(tensor.float32_data.data(), data.value().data(), static_cast<size_t>(info->size));
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_tensor(const std::string& name) const
{
    const std::string prefix = source_quantized_prefix(name);
    if (!prefix.empty() && find(prefix + "qweight"))
        return load_source_int4_tensor(name);
    const SafetensorInfo* source_info = find(name);
    if (source_info && source_info->dtype == "F8_E4M3")
    {
        const std::string base = name.ends_with(".weight") ? name.substr(0, name.size() - 7) : name;
        for (const std::string& scale_name : {base + ".scale", base + ".weight_scale_inv", name + "_scale_inv"})
        {
            if (find(scale_name))
                return load_float8_tensor(name, scale_name);
        }
        return Error{ErrorCode::InvalidModel, "missing blockwise FP8 scale: " + name};
    }
    auto encoding = tensor_encoding(*this, name);
    if (!encoding)
        return encoding.error();
    if (is_qnk_dtype(encoding.value().dtype))
        return load_qnk_tensor(name, encoding.value().dtype, encoding.value().rows, encoding.value().columns);
    const SafetensorInfo* info = find(name);

    TensorData tensor;
    tensor.shape = info->shape;
    if (encoding.value().dtype == DType::Int4)
    {
        tensor.dtype = DType::Int4;
        tensor.shape = {encoding.value().rows, encoding.value().columns};
        tensor.quantization_group_size = encoding.value().group_size;
        const SafetensorInfo* scales = find(name + ".__int4_scales");
        const SafetensorInfo* zeros = find(name + ".__int4_zeros");
        auto scale_data = read_range(scales->path, scales->offset, scales->size);
        if (!scale_data)
            return scale_data.error();
        tensor.quantization_scales.resize(static_cast<size_t>(scales->size / sizeof(float)));
        std::memcpy(tensor.quantization_scales.data(), scale_data.value().data(), static_cast<size_t>(scales->size));
        auto zero_data = read_range(zeros->path, zeros->offset, zeros->size);
        if (!zero_data)
            return zero_data.error();
        tensor.quantization_zero_points = std::move(zero_data).value();
        const SafetensorInfo* indices = find(name + ".__int4_g_idx");
        if (indices)
        {
            auto index_data = read_range(indices->path, indices->offset, indices->size);
            if (!index_data)
                return index_data.error();
            const uint32_t groups = encoding.value().columns / encoding.value().group_size
                                    + (encoding.value().columns % encoding.value().group_size != 0);
            tensor.quantization_group_indices.resize(encoding.value().columns);
            bool canonical = true;
            for (uint32_t column = 0; column < encoding.value().columns; ++column)
            {
                int64_t index;
                std::memcpy(&index, index_data.value().data() + static_cast<size_t>(column) * sizeof(index), sizeof(index));
                if (index < 0 || static_cast<uint64_t>(index) >= groups)
                    return Error{ErrorCode::InvalidModel, "native INT4 group index is out of range: " + name};
                tensor.quantization_group_indices[column] = static_cast<uint32_t>(index);
                canonical = canonical && static_cast<uint64_t>(index) == column / encoding.value().group_size;
            }
            if (canonical)
                std::vector<uint32_t>().swap(tensor.quantization_group_indices);
        }
        auto mapped = MappedFileRange::open(info->path, info->offset, info->size);
        if (mapped)
        {
            tensor.mapped_data = mapped.value()->share_data();
            tensor.mapped_size = info->size;
        }
        else
        {
            auto data = read_range(info->path, info->offset, info->size);
            if (!data)
                return data.error();
            tensor.quantized_data = std::move(data).value();
        }
        auto status = validate_int4_tensor(tensor);
        if (!status)
            return Error{ErrorCode::InvalidModel, status.error().message + ": " + name};
    }
    else if (info->dtype == "F16" || (info->dtype == "F32" && !source_quantization_method.empty()))
    {
        const size_t element_size = info->dtype == "F16" ? sizeof(uint16_t) : sizeof(float);
        if (tensor.element_count() > std::numeric_limits<uint64_t>::max() / element_size
            || info->size != tensor.element_count() * element_size)
        {
            return Error{ErrorCode::InvalidModel, "invalid floating-point byte count: " + name};
        }
        auto bytes = read_range(info->path, info->offset, info->size);
        if (!bytes)
            return bytes.error();
        tensor.dtype = DType::BFloat16;
        tensor.bfloat16_data.resize(static_cast<size_t>(tensor.element_count()));
        for (size_t index = 0; index < tensor.bfloat16_data.size(); ++index)
        {
            const float value = read_safetensor_float(bytes.value().data(), index, info->dtype);
            if (!std::isfinite(value))
                return Error{ErrorCode::InvalidModel, "non-finite floating-point tensor: " + name};
            const uint32_t bits = std::bit_cast<uint32_t>(value);
            const uint16_t rounded = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
            if ((rounded & 0x7f80u) == 0x7f80u)
                return Error{ErrorCode::InvalidModel, "floating-point tensor overflows BF16: " + name};
            tensor.bfloat16_data[index] = rounded;
        }
    }
    else if (info->dtype == "BF16")
    {
        if (info->size % sizeof(uint16_t) != 0)
            return Error{ErrorCode::InvalidModel, "invalid BF16 byte count: " + name};
        tensor.dtype = DType::BFloat16;
        auto mapped = MappedFileRange::open(info->path, info->offset, info->size);
        if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(uint16_t) == 0)
        {
            tensor.mapped_data = mapped.value()->share_data();
            tensor.mapped_size = info->size;
            return tensor;
        }
        auto data = read_range(info->path, info->offset, info->size);
        if (!data)
            return data.error();
        tensor.bfloat16_data.resize(static_cast<size_t>(info->size / sizeof(uint16_t)));
        std::memcpy(tensor.bfloat16_data.data(), data.value().data(), static_cast<size_t>(info->size));
    }
    else if (info->dtype == "F32")
    {
        if (info->size % sizeof(float) != 0)
            return Error{ErrorCode::InvalidModel, "invalid F32 byte count: " + name};
        tensor.dtype = DType::Float32;
        auto mapped = MappedFileRange::open(info->path, info->offset, info->size);
        if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(float) == 0)
        {
            tensor.mapped_data = mapped.value()->share_data();
            tensor.mapped_size = info->size;
            return tensor;
        }
        auto data = read_range(info->path, info->offset, info->size);
        if (!data)
            return data.error();
        tensor.float32_data.resize(static_cast<size_t>(info->size / sizeof(float)));
        std::memcpy(tensor.float32_data.data(), data.value().data(), static_cast<size_t>(info->size));
    }
    else if (info->dtype == "I64")
    {
        if (info->size % sizeof(int64_t) != 0)
            return Error{ErrorCode::InvalidModel, "invalid I64 byte count: " + name};
        tensor.dtype = DType::Int64;
        auto mapped = MappedFileRange::open(info->path, info->offset, info->size);
        if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(int64_t) == 0)
        {
            tensor.mapped_data = mapped.value()->share_data();
            tensor.mapped_size = info->size;
            return tensor;
        }
        auto data = read_range(info->path, info->offset, info->size);
        if (!data)
            return data.error();
        tensor.int64_data.resize(static_cast<size_t>(info->size / sizeof(int64_t)));
        std::memcpy(tensor.int64_data.data(), data.value().data(), static_cast<size_t>(info->size));
    }
    else
    {
        return Error{ErrorCode::UnsupportedModel, "unsupported safetensors dtype for tensor: " + name};
    }
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_qnk_tensor(const std::string& name,
                                                       DType dtype,
                                                       uint32_t rows,
                                                       uint32_t columns) const
{
    if (!qnk_shape_supported(dtype, rows, columns))
        return Error{ErrorCode::InvalidArgument, "invalid Qn_K tensor shape: " + name};
    if (find(name + ".__qnk_shape") || find(name + ".__qnk_columns"))
    {
        auto encoding = tensor_encoding(*this, name);
        if (!encoding)
            return encoding.error();
        if (encoding.value().dtype != dtype || encoding.value().rows != rows || encoding.value().columns != columns)
            return Error{ErrorCode::InvalidModel, "Qn_K tensor metadata disagrees with requested matrix shape: " + name};
    }
    const uint64_t expected_size = qnk_storage_bytes(dtype, rows, columns);
    const SafetensorInfo* source = find(name);
    if (!source || source->dtype != "U8" || source->size != expected_size)
        return Error{ErrorCode::InvalidModel, "invalid Qn_K safetensors tensor: " + name};

    TensorData tensor;
    tensor.dtype = dtype;
    tensor.shape = {rows, columns};
    const SafetensorInfo* column_indices = find(name + ".__qnk_columns");
    if (column_indices)
    {
        if (!find(name + ".__qnk_shape") || column_indices->dtype != "I64"
            || column_indices->shape != std::vector<uint32_t>{columns}
            || column_indices->size != static_cast<uint64_t>(columns) * sizeof(int64_t))
        {
            return Error{ErrorCode::InvalidModel, "invalid Qn_K input-column metadata: " + name};
        }
        auto index_data = read_range(column_indices->path, column_indices->offset, column_indices->size);
        if (!index_data)
            return index_data.error();
        tensor.input_column_indices.resize(columns);
        for (uint32_t column = 0; column < columns; ++column)
        {
            int64_t index;
            std::memcpy(&index, index_data.value().data() + static_cast<size_t>(column) * sizeof(index), sizeof(index));
            if (index < 0 || static_cast<uint64_t>(index) >= columns)
                return Error{ErrorCode::InvalidModel, "Qn_K input-column index is out of range: " + name};
            tensor.input_column_indices[column] = static_cast<uint32_t>(index);
        }
        auto status = validate_input_column_indices(tensor);
        if (!status)
            return Error{ErrorCode::InvalidModel, status.error().message + ": " + name};
    }
    auto mapped = MappedFileRange::open(source->path, source->offset, source->size);
    if (mapped)
    {
        mapped.value()->prefault();
        tensor.mapped_data = mapped.value()->share_data();
        tensor.mapped_size = source->size;
        return tensor;
    }
    auto data = read_range(source->path, source->offset, source->size);
    if (!data)
        return data.error();
    tensor.quantized_data = std::move(data).value();
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_qnk_expert(const std::string& name,
                                                       DType dtype,
                                                       uint32_t expert_id,
                                                       uint32_t expert_count,
                                                       uint32_t rows,
                                                       uint32_t columns) const
{
    if (expert_count == 0 || expert_id >= expert_count || !qnk_shape_supported(dtype, rows, columns))
        return Error{ErrorCode::InvalidArgument, "invalid Qn_K Expert shape: " + name};
    const uint64_t expert_size = qnk_storage_bytes(dtype, rows, columns);
    const SafetensorInfo* source = find(name);
    if (!source || source->dtype != "U8"
        || expert_size > std::numeric_limits<uint64_t>::max() / expert_count
        || source->size != expert_size * expert_count)
    {
        return Error{ErrorCode::InvalidModel, "invalid Qn_K Expert tensor: " + name};
    }
    const uint64_t offset = source->offset + expert_id * expert_size;

    TensorData tensor;
    tensor.dtype = dtype;
    tensor.shape = {rows, columns};
    auto mapped = MappedFileRange::open(source->path, offset, expert_size);
    if (mapped)
    {
        mapped.value()->prefault();
        tensor.mapped_data = mapped.value()->share_data();
        tensor.mapped_size = expert_size;
        return tensor;
    }
    auto data = read_range(source->path, offset, expert_size);
    if (!data)
        return data.error();
    tensor.quantized_data = std::move(data).value();
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_float8_tensor(const std::string& weight_name, const std::string& scale_name) const
{
    const SafetensorInfo* weight = find(weight_name);
    const SafetensorInfo* scale = find(scale_name);
    if (!weight || !scale || weight->dtype != "F8_E4M3"
        || (scale->dtype != "F8_E8M0" && scale->dtype != "F32" && scale->dtype != "F16" && scale->dtype != "BF16")
        || weight->shape.size() != 2 || scale->shape.size() != 2)
        return Error{ErrorCode::InvalidModel, "invalid blockwise FP8 tensor: " + weight_name};
    const uint32_t output_blocks = (weight->shape[0] + 127) / 128;
    const uint32_t input_blocks = (weight->shape[1] + 127) / 128;
    const size_t scale_bytes = scale->dtype == "F32" ? 4 : scale->dtype == "F8_E8M0" ? 1
                                                                                     : 2;
    if (scale->shape != std::vector<uint32_t>{output_blocks, input_blocks}
        || weight->size != static_cast<uint64_t>(weight->shape[0]) * weight->shape[1]
        || scale->size != static_cast<uint64_t>(output_blocks) * input_blocks * scale_bytes)
    {
        return Error{ErrorCode::InvalidModel, "invalid blockwise FP8 tensor shape: " + weight_name};
    }

    TensorData tensor;
    tensor.dtype = DType::Float8E4M3;
    tensor.shape = weight->shape;
    auto mapped = MappedFileRange::open(weight->path, weight->offset, weight->size);
    if (mapped)
    {
        tensor.mapped_data = mapped.value()->share_data();
        tensor.mapped_size = weight->size;
    }
    else
    {
        return Error{ErrorCode::IoError, "cannot map blockwise FP8 tensor: " + weight_name};
    }

    auto scale_data = read_range(scale->path, scale->offset, scale->size);
    if (!scale_data)
        return scale_data.error();
    tensor.quantization_scales.resize(static_cast<size_t>(output_blocks) * input_blocks);
    for (size_t index = 0; index < tensor.quantization_scales.size(); ++index)
    {
        const float value = scale->dtype == "F8_E8M0"
                                ? std::ldexp(1.0f, static_cast<int>(scale_data.value()[index]) - 127)
                                : read_safetensor_float(scale_data.value().data(), index, scale->dtype);
        if (!std::isfinite(value) || value <= 0.0f)
            return Error{ErrorCode::InvalidModel, "invalid blockwise FP8 scale: " + scale_name};
        tensor.quantization_scales[index] = value;
    }
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_mxfp4_tensor(const std::string& blocks_name, const std::string& scales_name, uint32_t rows, uint32_t columns, uint32_t flags) const
{
    const SafetensorInfo* blocks = find(blocks_name);
    const SafetensorInfo* scales = find(scales_name);
    const SafetensorInfo* packed_blocks = find(packed_tensor_name(blocks_name));
    const SafetensorInfo* packed_scales = find(packed_tensor_name(scales_name));
    if ((packed_blocks == nullptr) != (packed_scales == nullptr))
    {
        return Error{ErrorCode::InvalidModel, "incomplete packed MXFP4 tensor pair: " + blocks_name};
    }
    if (packed_blocks)
    {
        blocks = packed_blocks;
        scales = packed_scales;
    }
    if (!blocks || !scales || blocks->dtype != "I8" || scales->dtype != "F8_E8M0"
        || rows == 0 || columns == 0 || columns % 32 != 0
        || blocks->size != static_cast<uint64_t>(rows) * columns / 2
        || scales->size != static_cast<uint64_t>(rows) * columns / 32
        || blocks->shape != std::vector<uint32_t>{rows, columns / 2}
        || scales->shape != std::vector<uint32_t>{rows, columns / 32})
    {
        return Error{ErrorCode::InvalidModel, "invalid packed MXFP4 tensor: " + blocks_name};
    }

    TensorData tensor;
    tensor.dtype = DType::MxFp4;
    tensor.shape = {rows, columns};
    if (has_flag(flags, SafetensorLoadDeferMxfp4Data))
    {
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = blocks->path.string();
        storage->blocks_offset = blocks->offset;
        storage->blocks_size = blocks->size;
        storage->scales_path = scales->path.string();
        storage->scales_offset = scales->offset;
        storage->scales_size = scales->size;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    }

    auto block_mapping = MappedFileRange::open(blocks->path, blocks->offset, blocks->size);
    auto scale_mapping = MappedFileRange::open(scales->path, scales->offset, scales->size);
    if (block_mapping && scale_mapping)
    {
        block_mapping.value()->prefault();
        scale_mapping.value()->prefault();
        tensor.mxfp4_blocks = block_mapping.value()->share_bytes();
        tensor.mxfp4_scales = scale_mapping.value()->share_bytes();
        return tensor;
    }
    auto block_data = read_range(blocks->path, blocks->offset, blocks->size);
    auto scale_data = read_range(scales->path, scales->offset, scales->size);
    if (!block_data)
        return block_data.error();
    if (!scale_data)
        return scale_data.error();
    tensor.mxfp4_blocks.assign(block_data.value().data(), block_data.value().size());
    tensor.mxfp4_scales.assign(scale_data.value().data(), scale_data.value().size());
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_interleaved_mxfp4_tensor(const std::string& gate_blocks_name, const std::string& gate_scales_name,
                                                                     const std::string& up_blocks_name, const std::string& up_scales_name,
                                                                     uint32_t rows, uint32_t columns, uint32_t flags) const
{
    if (rows == 0 || columns == 0 || rows > std::numeric_limits<uint32_t>::max() / 2
        || columns % 32 != 0)
    {
        return Error{ErrorCode::InvalidModel, "invalid interleaved MXFP4 tensor: " + gate_blocks_name};
    }
    const uint64_t block_size = static_cast<uint64_t>(rows) * columns / 2;
    const uint64_t scale_size = static_cast<uint64_t>(rows) * columns / 32;
    if (has_flag(flags, SafetensorLoadDeferMxfp4Data))
    {
        const SafetensorInfo* gate_blocks = find(gate_blocks_name);
        const SafetensorInfo* gate_scales = find(gate_scales_name);
        const SafetensorInfo* up_blocks = find(up_blocks_name);
        const SafetensorInfo* up_scales = find(up_scales_name);
        const SafetensorInfo* packed_gate_blocks = find(packed_tensor_name(gate_blocks_name));
        const SafetensorInfo* packed_gate_scales = find(packed_tensor_name(gate_scales_name));
        const SafetensorInfo* packed_up_blocks = find(packed_tensor_name(up_blocks_name));
        const SafetensorInfo* packed_up_scales = find(packed_tensor_name(up_scales_name));
        const bool any_packed = packed_gate_blocks || packed_gate_scales
                                || packed_up_blocks || packed_up_scales;
        const bool all_packed = packed_gate_blocks && packed_gate_scales
                                && packed_up_blocks && packed_up_scales;
        if (any_packed && !all_packed)
        {
            return Error{ErrorCode::InvalidModel, "incomplete packed interleaved MXFP4 tensor: " + gate_blocks_name};
        }
        if (all_packed)
        {
            gate_blocks = packed_gate_blocks;
            gate_scales = packed_gate_scales;
            up_blocks = packed_up_blocks;
            up_scales = packed_up_scales;
        }
        if (!gate_blocks || !gate_scales || !up_blocks || !up_scales
            || gate_blocks->dtype != "I8" || up_blocks->dtype != "I8"
            || gate_scales->dtype != "F8_E8M0" || up_scales->dtype != "F8_E8M0"
            || gate_blocks->size != block_size || up_blocks->size != block_size
            || gate_scales->size != scale_size || up_scales->size != scale_size
            || gate_blocks->shape != std::vector<uint32_t>{rows, columns / 2}
            || up_blocks->shape != gate_blocks->shape
            || gate_scales->shape != std::vector<uint32_t>{rows, columns / 32}
            || up_scales->shape != gate_scales->shape)
        {
            return Error{ErrorCode::InvalidModel, "invalid interleaved MXFP4 tensor: " + gate_blocks_name};
        }
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {rows * 2, columns};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = gate_blocks->path.string();
        storage->blocks_offset = gate_blocks->offset;
        storage->blocks_size = gate_blocks->size;
        storage->scales_path = gate_scales->path.string();
        storage->scales_offset = gate_scales->offset;
        storage->scales_size = gate_scales->size;
        storage->secondary_blocks_path = up_blocks->path.string();
        storage->secondary_blocks_offset = up_blocks->offset;
        storage->secondary_blocks_size = up_blocks->size;
        storage->secondary_scales_path = up_scales->path.string();
        storage->secondary_scales_offset = up_scales->offset;
        storage->secondary_scales_size = up_scales->size;
        storage->interleave_rows = true;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    }

    if (block_size > std::numeric_limits<size_t>::max() / 2)
        return Error{ErrorCode::InvalidModel, "invalid interleaved MXFP4 tensor: " + gate_blocks_name};

    auto gate = load_mxfp4_tensor(gate_blocks_name, gate_scales_name, rows, columns);
    auto up = load_mxfp4_tensor(up_blocks_name, up_scales_name, rows, columns);
    if (!gate)
        return gate.error();
    if (!up)
        return up.error();
    TensorData tensor;
    tensor.dtype = DType::MxFp4;
    tensor.shape = {rows * 2, columns};
    const size_t block_row_size = columns / 2;
    const size_t scale_row_size = columns / 32;
    tensor.mxfp4_blocks.resize(static_cast<size_t>(block_size * 2));
    tensor.mxfp4_scales.resize(static_cast<size_t>(scale_size * 2));
    for (uint32_t row = 0; row < rows; ++row)
    {
        std::memcpy(tensor.mxfp4_blocks.data() + static_cast<size_t>(row * 2) * block_row_size, gate.value().mxfp4_blocks.data() + static_cast<size_t>(row) * block_row_size, block_row_size);
        std::memcpy(tensor.mxfp4_blocks.data() + static_cast<size_t>(row * 2 + 1) * block_row_size, up.value().mxfp4_blocks.data() + static_cast<size_t>(row) * block_row_size, block_row_size);
        std::memcpy(tensor.mxfp4_scales.data() + static_cast<size_t>(row * 2) * scale_row_size, gate.value().mxfp4_scales.data() + static_cast<size_t>(row) * scale_row_size, scale_row_size);
        std::memcpy(tensor.mxfp4_scales.data() + static_cast<size_t>(row * 2 + 1) * scale_row_size, up.value().mxfp4_scales.data() + static_cast<size_t>(row) * scale_row_size, scale_row_size);
    }
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_bfloat16_slice(const std::string& name, uint32_t index, std::vector<uint32_t> shape) const
{
    const SafetensorInfo* info = find(name);
    if (!info || info->dtype != "BF16" || info->shape.empty() || index >= info->shape[0])
        return Error{ErrorCode::InvalidModel, "invalid BF16 tensor slice: " + name};
    uint64_t element_count = 1;
    for (uint32_t dimension : shape)
        element_count *= dimension;
    const uint64_t slice_size = element_count * sizeof(uint16_t);
    if (slice_size * info->shape[0] != info->size)
        return Error{ErrorCode::InvalidModel, "BF16 tensor slice shape mismatch: " + name};

    TensorData tensor;
    tensor.dtype = DType::BFloat16;
    tensor.shape = std::move(shape);
    auto mapped = MappedFileRange::open(info->path, info->offset + index * slice_size, slice_size);
    if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(uint16_t) == 0)
    {
        tensor.mapped_data = mapped.value()->share_data();
        tensor.mapped_size = slice_size;
        return tensor;
    }
    auto data = read_range(info->path, info->offset + index * slice_size, slice_size);
    if (!data)
        return data.error();
    tensor.bfloat16_data.resize(static_cast<size_t>(element_count));
    std::memcpy(tensor.bfloat16_data.data(), data.value().data(), static_cast<size_t>(slice_size));
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_mxfp4_expert(const std::string& blocks_name,
                                                         const std::string& scales_name,
                                                         uint32_t expert_id,
                                                         uint32_t rows,
                                                         uint32_t columns,
                                                         uint32_t flags) const
{
    const SafetensorInfo* blocks = find(blocks_name);
    const SafetensorInfo* scales = find(scales_name);
    if (!blocks || !scales || blocks->dtype != "U8" || scales->dtype != "U8"
        || rows == 0 || columns == 0 || columns % 32 != 0
        || blocks->shape.size() != 4 || scales->shape.size() != 3)
        return Error{ErrorCode::InvalidModel, "invalid MXFP4 expert tensors: " + blocks_name};
    const std::vector<uint32_t> expected_blocks = {blocks->shape[0], rows, columns / 32, 16};
    const std::vector<uint32_t> expected_scales = {scales->shape[0], rows, columns / 32};
    if (blocks->shape != expected_blocks || scales->shape != expected_scales
        || blocks->shape[0] != scales->shape[0] || expert_id >= blocks->shape[0])
        return Error{ErrorCode::InvalidModel, "invalid MXFP4 expert tensors: " + blocks_name};

    const uint64_t block_size = static_cast<uint64_t>(rows) * columns / 2;
    const uint64_t scale_size = static_cast<uint64_t>(rows) * columns / 32;
    const uint64_t expert_count = blocks->shape[0];
    const std::string packed_prefix = "__ncnn_moe_packed__." + std::to_string(expert_id) + ".";
    const SafetensorInfo* packed_blocks = find(packed_prefix + blocks_name);
    const SafetensorInfo* packed_scales = find(packed_prefix + scales_name);
    if ((packed_blocks == nullptr) != (packed_scales == nullptr))
    {
        return Error{ErrorCode::InvalidModel, "incomplete packed MXFP4 Expert pair: " + blocks_name};
    }
    if (packed_blocks
        && (packed_blocks->dtype != "U8"
            || packed_scales->dtype != "U8"
            || packed_blocks->size != block_size
            || packed_scales->size != scale_size
            || packed_blocks->shape != std::vector<uint32_t>{rows, columns / 32, 16}
            || packed_scales->shape != std::vector<uint32_t>{rows, columns / 32}))
    {
        return Error{ErrorCode::InvalidModel, "invalid packed MXFP4 Expert tensors: " + blocks_name};
    }
    if (!packed_blocks
        && (block_size > std::numeric_limits<uint64_t>::max() / expert_count
            || blocks->size != block_size * expert_count
            || scales->size != scale_size * expert_count))
    {
        return Error{ErrorCode::InvalidModel, "invalid MXFP4 Expert tensors: " + blocks_name};
    }
    const SafetensorInfo* selected_blocks = packed_blocks ? packed_blocks : blocks;
    const SafetensorInfo* selected_scales = packed_scales ? packed_scales : scales;
    const uint64_t block_offset = selected_blocks->offset + (packed_blocks ? 0 : expert_id * block_size);
    const uint64_t scale_offset = selected_scales->offset + (packed_scales ? 0 : expert_id * scale_size);
    TensorData tensor;
    tensor.dtype = DType::MxFp4;
    tensor.shape = {rows, columns};
    if (has_flag(flags, SafetensorLoadDeferMxfp4Data))
    {
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = selected_blocks->path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_size = block_size;
        storage->scales_path = selected_scales->path.string();
        storage->scales_offset = scale_offset;
        storage->scales_size = scale_size;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    }

    auto block_mapping = MappedFileRange::open(selected_blocks->path, block_offset, block_size);
    auto scale_mapping = MappedFileRange::open(selected_scales->path, scale_offset, scale_size);
    if (block_mapping && scale_mapping)
    {
        block_mapping.value()->prefault();
        scale_mapping.value()->prefault();
        tensor.mxfp4_blocks = block_mapping.value()->share_bytes();
        tensor.mxfp4_scales = scale_mapping.value()->share_bytes();
        return tensor;
    }

    auto block_data = read_range(selected_blocks->path, block_offset, block_size);
    if (!block_data)
        return block_data.error();
    auto scale_data = read_range(selected_scales->path, scale_offset, scale_size);
    if (!scale_data)
        return scale_data.error();
    tensor.mxfp4_blocks.assign(block_data.value().data(), block_data.value().size());
    tensor.mxfp4_scales.assign(scale_data.value().data(), scale_data.value().size());
    return tensor;
}

} // namespace moe
} // namespace ncnn
