#include "safetensors.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <utility>

namespace ncnn {
namespace moe {

static Result<std::vector<uint8_t> > read_range(
    const std::filesystem::path& path,
    uint64_t offset,
    uint64_t byte_count)
{
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())
        || byte_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return Error{ErrorCode::InvalidModel, "safetensors range is too large: " + path.string()};

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open safetensors shard: " + path.string()};
    stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot seek safetensors shard: " + path.string()};

    std::vector<uint8_t> bytes(static_cast<size_t>(byte_count));
    if (byte_count > 0
        && !stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(byte_count)))
        return Error{ErrorCode::InvalidModel, "safetensors shard is truncated: " + path.string()};
    return bytes;
}

static bool parse_json_string(const std::string& json, size_t& position, std::string& value)
{
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position])))
        ++position;
    if (position >= json.size() || json[position] != '"')
        return false;
    ++position;
    value.clear();
    while (position < json.size()) {
        const char character = json[position++];
        if (character == '"')
            return true;
        if (character == '\\' && position < json.size()) {
            value.push_back(json[position++]);
        }
        else {
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
    for (size_t position = begin; position < json.size(); ++position) {
        const char character = json[position];
        if (in_string) {
            if (escaped) {
                escaped = false;
            }
            else if (character == '\\') {
                escaped = true;
            }
            else if (character == '"') {
                in_string = false;
            }
            continue;
        }
        if (character == '"') {
            in_string = true;
        }
        else if (character == '{') {
            ++depth;
        }
        else if (character == '}') {
            if (--depth == 0) {
                end = position + 1;
                return true;
            }
        }
    }
    return false;
}

static Result<std::vector<uint32_t> > parse_shape(const std::string& text)
{
    const std::regex number_expression("[0-9]+");
    std::vector<uint32_t> shape;
    for (std::sregex_iterator iterator(text.begin(), text.end(), number_expression), end; iterator != end; ++iterator) {
        try {
            const unsigned long long value = std::stoull(iterator->str());
            if (value > std::numeric_limits<uint32_t>::max())
                return Error{ErrorCode::InvalidModel, "safetensors dimension is out of range"};
            shape.push_back(static_cast<uint32_t>(value));
        }
        catch (const std::exception&) {
            return Error{ErrorCode::InvalidModel, "invalid safetensors dimension"};
        }
    }
    return shape;
}

static Result<void> parse_header(
    const std::filesystem::path& path,
    const std::string& json,
    uint64_t data_start,
    std::unordered_map<std::string, SafetensorInfo>& tensors)
{
    const std::regex dtype_expression("\"dtype\"\\s*:\\s*\"([^\"]+)\"");
    const std::regex shape_expression("\"shape\"\\s*:\\s*\\[([^\\]]*)\\]");
    const std::regex offsets_expression("\"data_offsets\"\\s*:\\s*\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
    size_t position = 1;
    while (position < json.size()) {
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
        std::smatch shape_match;
        std::smatch offsets_match;
        if (!std::regex_search(object, dtype_match, dtype_expression)
            || !std::regex_search(object, shape_match, shape_expression)
            || !std::regex_search(object, offsets_match, offsets_expression))
            return Error{ErrorCode::InvalidModel, "incomplete safetensors tensor metadata: " + name};

        auto shape = parse_shape(shape_match[1].str());
        if (!shape)
            return shape.error();
        uint64_t begin = 0;
        uint64_t end = 0;
        try {
            begin = std::stoull(offsets_match[1].str());
            end = std::stoull(offsets_match[2].str());
        }
        catch (const std::exception&) {
            return Error{ErrorCode::InvalidModel, "invalid safetensors data offsets: " + name};
        }
        if (end < begin || data_start > std::numeric_limits<uint64_t>::max() - begin)
            return Error{ErrorCode::InvalidModel, "invalid safetensors data range: " + name};
        if (tensors.contains(name))
            return Error{ErrorCode::InvalidModel, "duplicate safetensors tensor: " + name};
        tensors.emplace(
            std::move(name),
            SafetensorInfo{path, dtype_match[1].str(), std::move(shape).value(), data_start + begin, end - begin});
    }
    return {};
}

Result<SafetensorsArchive> SafetensorsArchive::open(const std::filesystem::path& root)
{
    SafetensorsArchive archive;
    std::error_code filesystem_error;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(root, filesystem_error)) {
        if (filesystem_error)
            return Error{ErrorCode::IoError, "cannot enumerate model directory: " + root.string()};
        if (!entry.is_regular_file() || entry.path().extension() != ".safetensors")
            continue;

        std::ifstream stream(entry.path(), std::ios::binary);
        if (!stream)
            return Error{ErrorCode::IoError, "cannot open safetensors shard: " + entry.path().string()};
        uint64_t header_length = 0;
        stream.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
        if (!stream || header_length == 0 || header_length > 128 * 1024 * 1024)
            return Error{ErrorCode::InvalidModel, "invalid safetensors header length: " + entry.path().string()};
        std::string header(static_cast<size_t>(header_length), '\0');
        stream.read(header.data(), static_cast<std::streamsize>(header_length));
        if (!stream)
            return Error{ErrorCode::InvalidModel, "truncated safetensors header: " + entry.path().string()};
        auto status = parse_header(entry.path(), header, sizeof(header_length) + header_length, archive.tensors_);
        if (!status)
            return status.error();
    }
    if (archive.tensors_.empty())
        return Error{ErrorCode::InvalidModel, "model directory contains no safetensors tensors"};
    return archive;
}

const SafetensorInfo* SafetensorsArchive::find(const std::string& name) const noexcept
{
    const auto iterator = tensors_.find(name);
    return iterator == tensors_.end() ? nullptr : &iterator->second;
}

Result<TensorData> SafetensorsArchive::load_tensor(const std::string& name) const
{
    const SafetensorInfo* info = find(name);
    if (!info)
        return Error{ErrorCode::InvalidModel, "missing safetensors tensor: " + name};
    auto bytes = read_range(info->path, info->offset, info->byte_count);
    if (!bytes)
        return bytes.error();

    TensorData tensor;
    tensor.shape = info->shape;
    if (info->dtype == "BF16") {
        if (info->byte_count % sizeof(uint16_t) != 0)
            return Error{ErrorCode::InvalidModel, "invalid BF16 byte count: " + name};
        tensor.dtype = DType::BFloat16;
        tensor.bfloat16_data.resize(static_cast<size_t>(info->byte_count / sizeof(uint16_t)));
        std::memcpy(tensor.bfloat16_data.data(), bytes.value().data(), static_cast<size_t>(info->byte_count));
    }
    else if (info->dtype == "F32") {
        if (info->byte_count % sizeof(float) != 0)
            return Error{ErrorCode::InvalidModel, "invalid F32 byte count: " + name};
        tensor.dtype = DType::Float32;
        tensor.float32_data.resize(static_cast<size_t>(info->byte_count / sizeof(float)));
        std::memcpy(tensor.float32_data.data(), bytes.value().data(), static_cast<size_t>(info->byte_count));
    }
    else {
        return Error{ErrorCode::UnsupportedModel, "unsupported safetensors dtype for tensor: " + name};
    }
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_bfloat16_slice(
    const std::string& name,
    uint32_t index,
    std::vector<uint32_t> shape) const
{
    const SafetensorInfo* info = find(name);
    if (!info || info->dtype != "BF16" || info->shape.empty() || index >= info->shape[0])
        return Error{ErrorCode::InvalidModel, "invalid BF16 tensor slice: " + name};
    uint64_t element_count = 1;
    for (uint32_t dimension : shape)
        element_count *= dimension;
    const uint64_t byte_count = element_count * sizeof(uint16_t);
    if (byte_count * info->shape[0] != info->byte_count)
        return Error{ErrorCode::InvalidModel, "BF16 tensor slice shape mismatch: " + name};
    auto bytes = read_range(info->path, info->offset + index * byte_count, byte_count);
    if (!bytes)
        return bytes.error();

    TensorData tensor;
    tensor.dtype = DType::BFloat16;
    tensor.shape = std::move(shape);
    tensor.bfloat16_data.resize(static_cast<size_t>(element_count));
    std::memcpy(tensor.bfloat16_data.data(), bytes.value().data(), static_cast<size_t>(byte_count));
    return tensor;
}

Result<TensorData> SafetensorsArchive::load_mxfp4_expert(
    const std::string& blocks_name,
    const std::string& scales_name,
    uint32_t expert_id,
    uint32_t rows,
    uint32_t columns) const
{
    const SafetensorInfo* blocks = find(blocks_name);
    const SafetensorInfo* scales = find(scales_name);
    if (!blocks || !scales || blocks->dtype != "U8" || scales->dtype != "U8"
        || columns % 32 != 0 || blocks->shape.size() != 4 || scales->shape.size() != 3)
        return Error{ErrorCode::InvalidModel, "invalid MXFP4 expert tensors: " + blocks_name};
    const std::vector<uint32_t> expected_blocks = {blocks->shape[0], rows, columns / 32, 16};
    const std::vector<uint32_t> expected_scales = {scales->shape[0], rows, columns / 32};
    if (blocks->shape != expected_blocks || scales->shape != expected_scales
        || blocks->shape[0] != scales->shape[0] || expert_id >= blocks->shape[0])
        return Error{ErrorCode::InvalidModel, "invalid MXFP4 expert tensors: " + blocks_name};

    const uint64_t block_bytes = static_cast<uint64_t>(rows) * columns / 2;
    const uint64_t scale_bytes = static_cast<uint64_t>(rows) * columns / 32;
    auto block_data = read_range(blocks->path, blocks->offset + expert_id * block_bytes, block_bytes);
    if (!block_data)
        return block_data.error();
    auto scale_data = read_range(scales->path, scales->offset + expert_id * scale_bytes, scale_bytes);
    if (!scale_data)
        return scale_data.error();

    TensorData tensor;
    tensor.dtype = DType::MxFp4;
    tensor.shape = {rows, columns};
    tensor.mxfp4_blocks = std::move(block_data).value();
    tensor.mxfp4_scales = std::move(scale_data).value();
    return tensor;
}

} // namespace moe
} // namespace ncnn
