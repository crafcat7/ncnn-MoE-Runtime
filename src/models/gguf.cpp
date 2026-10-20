#include "gguf.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <fstream>
#include <limits>
#include <regex>
#include <utility>

namespace ncnn {
namespace moe {

std::optional<uint64_t> GgufMetadataValue::unsigned_integer() const noexcept
{
    if (const auto* number = std::get_if<uint64_t>(&value))
        return *number;
    if (const auto* number = std::get_if<int64_t>(&value); number && *number >= 0)
        return static_cast<uint64_t>(*number);
    return std::nullopt;
}

std::optional<int64_t> GgufMetadataValue::integer() const noexcept
{
    if (const auto* number = std::get_if<int64_t>(&value))
        return *number;
    if (const auto* number = std::get_if<uint64_t>(&value); number && *number <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return static_cast<int64_t>(*number);
    return std::nullopt;
}

std::optional<double> GgufMetadataValue::number() const noexcept
{
    if (const auto* number = std::get_if<double>(&value))
        return *number;
    if (const auto* number = std::get_if<uint64_t>(&value))
        return static_cast<double>(*number);
    if (const auto* number = std::get_if<int64_t>(&value))
        return static_cast<double>(*number);
    return std::nullopt;
}

std::optional<bool> GgufMetadataValue::boolean() const noexcept
{
    if (const auto* flag = std::get_if<bool>(&value))
        return *flag;
    return std::nullopt;
}

const std::string* GgufMetadataValue::string() const noexcept
{
    return std::get_if<std::string>(&value);
}

const GgufMetadataValue::Array* GgufMetadataValue::array() const noexcept
{
    return std::get_if<Array>(&value);
}

constexpr uint64_t maximum_header_size = UINT64_C(256) * 1024 * 1024;
constexpr uint64_t maximum_string_size = UINT64_C(16) * 1024 * 1024;
constexpr uint64_t maximum_metadata_values = UINT64_C(4) * 1024 * 1024;
constexpr uint64_t maximum_tensor_count = UINT64_C(1024) * 1024;

class GgufReader
{
public:
    GgufReader(const std::filesystem::path& path, uint64_t size)
        : stream(path, std::ios::binary), file_size(size), source(path)
    {
        if (!stream)
            fail("cannot open GGUF file", ErrorCode::IoError);
    }

    bool read_unsigned(uint64_t& value, size_t bytes)
    {
        std::array<uint8_t, 8> buffer{};
        if (bytes > buffer.size() || !read_bytes(buffer.data(), bytes))
            return false;
        value = 0;
        for (size_t i = 0; i < bytes; ++i)
            value |= static_cast<uint64_t>(buffer[i]) << (i * 8);
        return true;
    }

    bool read_string(std::string& value, uint64_t maximum = maximum_string_size)
    {
        uint64_t length = 0;
        if (!read_unsigned(length, 8))
            return false;
        if (length > maximum || length > remaining())
            return fail("GGUF string exceeds its bounded header range");
        value.resize(static_cast<size_t>(length));
        return read_bytes(value.data(), static_cast<size_t>(length));
    }

    bool read_bytes(void* destination, size_t bytes)
    {
        if (!failure.message.empty())
            return false;
        if (bytes > remaining())
            return fail("truncated or oversized GGUF header");
        if (bytes != 0)
        {
            stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
            if (!stream)
                return fail("cannot read GGUF header", ErrorCode::IoError);
        }
        position += bytes;
        return true;
    }

    bool fail(const std::string& message, ErrorCode code = ErrorCode::InvalidModel)
    {
        if (failure.message.empty())
            failure = Error{code, message + ": " + source.string()};
        return false;
    }

    [[nodiscard]] uint64_t offset() const noexcept
    {
        return position;
    }

    [[nodiscard]] uint64_t remaining() const noexcept
    {
        return position < std::min(file_size, maximum_header_size)
                   ? std::min(file_size, maximum_header_size) - position
                   : 0;
    }

    [[nodiscard]] const Error& error() const noexcept
    {
        return failure;
    }

private:
    std::ifstream stream;
    uint64_t file_size;
    std::filesystem::path source;
    uint64_t position = 0;
    Error failure;
};

static bool read_metadata_value(GgufReader& reader, uint64_t type,
                                GgufMetadataValue& result, uint64_t& value_count)
{
    if (type > static_cast<uint32_t>(GgufMetadataType::Float64)
        || value_count >= maximum_metadata_values)
        return reader.fail("invalid GGUF metadata type or value count");
    ++value_count;
    result.type = static_cast<GgufMetadataType>(type);
    if (result.type == GgufMetadataType::String)
    {
        std::string text;
        if (!reader.read_string(text))
            return false;
        result.value = std::move(text);
        return true;
    }
    if (result.type == GgufMetadataType::Array)
    {
        uint64_t element_type = 0;
        uint64_t count = 0;
        if (!reader.read_unsigned(element_type, 4) || !reader.read_unsigned(count, 8))
            return false;
        if (element_type > static_cast<uint32_t>(GgufMetadataType::Float64)
            || element_type == static_cast<uint32_t>(GgufMetadataType::Array)
            || count > maximum_metadata_values - value_count || count > reader.remaining())
            return reader.fail("invalid GGUF metadata array type or length");
        result.array_type = static_cast<GgufMetadataType>(element_type);
        GgufMetadataValue::Array values;
        values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
        {
            GgufMetadataValue value;
            if (!read_metadata_value(reader, element_type, value, value_count))
                return false;
            values.push_back(std::move(value));
        }
        result.value = std::move(values);
        return true;
    }
    static constexpr size_t scalar_sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    uint64_t raw = 0;
    if (!reader.read_unsigned(raw, scalar_sizes[type]))
        return false;
    switch (result.type)
    {
    case GgufMetadataType::Int8: result.value = static_cast<int64_t>(std::bit_cast<int8_t>(static_cast<uint8_t>(raw))); break;
    case GgufMetadataType::Int16: result.value = static_cast<int64_t>(std::bit_cast<int16_t>(static_cast<uint16_t>(raw))); break;
    case GgufMetadataType::Int32: result.value = static_cast<int64_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(raw))); break;
    case GgufMetadataType::Int64: result.value = std::bit_cast<int64_t>(raw); break;
    case GgufMetadataType::Float32: result.value = static_cast<double>(std::bit_cast<float>(static_cast<uint32_t>(raw))); break;
    case GgufMetadataType::Float64: result.value = std::bit_cast<double>(raw); break;
    case GgufMetadataType::Bool:
        if (raw > 1)
            return reader.fail("invalid GGUF boolean value");
        result.value = raw != 0;
        break;
    default: result.value = raw; break;
    }
    return true;
}

struct ParsedGguf
{
    std::unordered_map<std::string, GgufMetadataValue> metadata;
    std::vector<GgufTensorInfo> tensors;
};

static Result<std::pair<DType, std::pair<uint64_t, uint64_t>>> tensor_encoding(uint64_t type)
{
    using Encoding = std::pair<DType, std::pair<uint64_t, uint64_t>>;
    switch (type)
    {
    case 0: return Encoding{DType::Float32, {1, 4}};
    case 1: return Encoding{DType::Float16, {1, 2}};
    case 10: return Encoding{DType::Q2K, {256, 84}};
    case 11: return Encoding{DType::Q3K, {256, 110}};
    case 12: return Encoding{DType::Q4K, {256, 144}};
    case 13: return Encoding{DType::Q5K, {256, 176}};
    case 14: return Encoding{DType::Q6K, {256, 210}};
    case 15: return Encoding{DType::Q8K, {256, 292}};
    case 26: return Encoding{DType::Int32, {1, 4}};
    case 30: return Encoding{DType::BFloat16, {1, 2}};
    case 39: return Encoding{DType::MxFp4, {32, 17}};
    default: return Error{ErrorCode::UnsupportedModel, "unsupported GGUF tensor type: " + std::to_string(type)};
    }
}

static Result<ParsedGguf> parse_gguf(const std::filesystem::path& path)
{
    std::error_code error;
    const uint64_t file_size = std::filesystem::file_size(path, error);
    if (error)
        return Error{ErrorCode::IoError, "cannot inspect GGUF file: " + path.string()};
    GgufReader reader(path, file_size);
    std::array<char, 4> magic{};
    uint64_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_count = 0;
    if (!reader.read_bytes(magic.data(), magic.size()) || !reader.read_unsigned(version, 4)
        || !reader.read_unsigned(tensor_count, 8) || !reader.read_unsigned(metadata_count, 8))
        return reader.error();
    if (magic != std::array<char, 4>{'G', 'G', 'U', 'F'} || version != 3)
        return Error{ErrorCode::UnsupportedModel, "GGUF requires version 3 little-endian encoding: " + path.string()};
    if (tensor_count > maximum_tensor_count || tensor_count > reader.remaining() / 24
        || metadata_count > maximum_metadata_values || metadata_count > reader.remaining() / 13)
        return Error{ErrorCode::InvalidModel, "invalid GGUF tensor or metadata count: " + path.string()};
    ParsedGguf parsed;
    uint64_t value_count = 0;
    for (uint64_t i = 0; i < metadata_count; ++i)
    {
        std::string key;
        uint64_t type = 0;
        GgufMetadataValue value;
        if (!reader.read_string(key, 1024) || !reader.read_unsigned(type, 4)
            || !read_metadata_value(reader, type, value, value_count))
            return reader.error();
        if (key.empty() || key.find('\0') != std::string::npos || !parsed.metadata.emplace(std::move(key), std::move(value)).second)
            return Error{ErrorCode::InvalidModel, "empty or duplicate GGUF metadata key: " + path.string()};
    }
    uint64_t alignment = 32;
    if (const auto found = parsed.metadata.find("general.alignment"); found != parsed.metadata.end())
    {
        const auto value = found->second.unsigned_integer();
        if (!value || *value == 0 || *value > 1024 * 1024 || (*value & (*value - 1)) != 0)
            return Error{ErrorCode::InvalidModel, "invalid GGUF tensor alignment: " + path.string()};
        alignment = *value;
    }
    parsed.tensors.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i)
    {
        GgufTensorInfo tensor;
        tensor.path = path;
        uint64_t dimensions = 0;
        if (!reader.read_string(tensor.name, 1024) || !reader.read_unsigned(dimensions, 4))
            return reader.error();
        if (tensor.name.empty() || tensor.name.find('\0') != std::string::npos || dimensions == 0 || dimensions > 4)
            return Error{ErrorCode::InvalidModel, "invalid GGUF tensor name or dimension count: " + path.string()};
        uint64_t elements = 1;
        uint64_t contiguous_dimension = 0;
        for (uint64_t dimension = 0; dimension < dimensions; ++dimension)
        {
            uint64_t extent = 0;
            if (!reader.read_unsigned(extent, 8))
                return reader.error();
            if (extent == 0 || extent > std::numeric_limits<uint32_t>::max() || elements > std::numeric_limits<uint64_t>::max() / extent)
                return Error{ErrorCode::InvalidModel, "invalid GGUF tensor shape: " + tensor.name};
            elements *= extent;
            if (dimension == 0)
                contiguous_dimension = extent;
            tensor.shape.push_back(static_cast<uint32_t>(extent));
        }
        std::reverse(tensor.shape.begin(), tensor.shape.end());
        uint64_t type = 0;
        if (!reader.read_unsigned(type, 4) || !reader.read_unsigned(tensor.offset, 8))
            return reader.error();
        auto encoding = tensor_encoding(type);
        if (!encoding)
            return encoding.error();
        tensor.dtype = encoding.value().first;
        tensor.ggml_type = static_cast<uint32_t>(type);
        const auto [block_size, block_bytes] = encoding.value().second;
        if (contiguous_dimension % block_size != 0 || elements / block_size > std::numeric_limits<uint64_t>::max() / block_bytes
            || tensor.offset % alignment != 0)
            return Error{ErrorCode::InvalidModel, "invalid GGUF quantized shape or tensor offset: " + tensor.name};
        tensor.size = elements / block_size * block_bytes;
        parsed.tensors.push_back(std::move(tensor));
    }
    const uint64_t padding = reader.offset() % alignment == 0 ? 0 : alignment - reader.offset() % alignment;
    if (reader.offset() > file_size || padding > file_size - reader.offset())
        return Error{ErrorCode::InvalidModel, "truncated GGUF tensor data alignment: " + path.string()};
    const uint64_t data_offset = reader.offset() + padding;
    for (GgufTensorInfo& tensor : parsed.tensors)
    {
        if (tensor.offset > file_size - data_offset || tensor.size > file_size - data_offset - tensor.offset)
            return Error{ErrorCode::InvalidModel, "truncated GGUF tensor payload: " + tensor.name};
        tensor.offset += data_offset;
    }
    std::vector<const GgufTensorInfo*> ranges;
    ranges.reserve(parsed.tensors.size());
    for (const GgufTensorInfo& tensor : parsed.tensors)
        ranges.push_back(&tensor);
    std::sort(ranges.begin(), ranges.end(), [](const GgufTensorInfo* left, const GgufTensorInfo* right) { return left->offset < right->offset; });
    for (size_t i = 1; i < ranges.size(); ++i)
    {
        if (ranges[i - 1]->size > ranges[i]->offset - ranges[i - 1]->offset)
            return Error{ErrorCode::InvalidModel, "overlapping GGUF tensor payloads: " + path.string()};
    }
    return parsed;
}

static bool gguf_extension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension == ".gguf";
}

static Result<std::filesystem::path> discover_gguf(const std::filesystem::path& path)
{
    std::error_code error;
    if (std::filesystem::is_regular_file(path, error))
    {
        if (!gguf_extension(path))
            return Error{ErrorCode::InvalidArgument, "GGUF source must be a .gguf file: " + path.string()};
        return path;
    }
    if (error || !std::filesystem::is_directory(path, error))
        return Error{ErrorCode::IoError, "cannot inspect GGUF source: " + path.string()};
    std::vector<std::filesystem::path> candidates;
    const auto collect = [&](const std::filesystem::path& directory) {
        for (const auto& entry : std::filesystem::directory_iterator(directory, error))
        {
            if (entry.is_regular_file(error) && gguf_extension(entry.path()))
                candidates.push_back(entry.path());
            if (error)
                break;
        }
    };
    collect(path);
    if (candidates.empty() && !error)
    {
        for (const auto& entry : std::filesystem::directory_iterator(path, error))
        {
            if (entry.is_directory(error))
                collect(entry.path());
            if (error)
                break;
        }
    }
    if (error)
        return Error{ErrorCode::IoError, "cannot enumerate GGUF source: " + path.string()};
    if (candidates.empty())
        return Error{ErrorCode::InvalidModel, "model directory contains no GGUF files: " + path.string()};
    std::sort(candidates.begin(), candidates.end());
    const std::regex split_pattern("^(.+)-([0-9]{5})-of-([0-9]{5})\\.gguf$", std::regex::icase);
    std::smatch first;
    const std::string first_name = candidates.front().filename().string();
    const bool split = std::regex_match(first_name, first, split_pattern);
    for (size_t i = 1; i < candidates.size(); ++i)
    {
        std::smatch current;
        const std::string name = candidates[i].filename().string();
        if (!split || !std::regex_match(name, current, split_pattern)
            || candidates[i].parent_path() != candidates.front().parent_path()
            || current[1].str() != first[1].str() || current[3].str() != first[3].str())
            return Error{ErrorCode::InvalidArgument, "multiple GGUF packages found; select a .gguf file explicitly: " + path.string()};
    }
    return candidates.front();
}

static std::optional<uint64_t> metadata_integer(const ParsedGguf& parsed, const std::string& key)
{
    const auto found = parsed.metadata.find(key);
    return found == parsed.metadata.end() ? std::nullopt : found->second.unsigned_integer();
}

Result<GgufArchive> GgufArchive::open(const std::filesystem::path& root_or_file)
{
    std::error_code error;
    const auto absolute = std::filesystem::absolute(root_or_file, error);
    if (error)
        return Error{ErrorCode::IoError, "cannot resolve GGUF source: " + root_or_file.string()};
    auto discovered = discover_gguf(absolute);
    if (!discovered)
        return discovered.error();
    auto initial = parse_gguf(discovered.value());
    if (!initial)
        return initial.error();
    const auto count_value = metadata_integer(initial.value(), "split.count");
    if (initial.value().metadata.contains("split.count") && !count_value)
        return Error{ErrorCode::InvalidModel, "GGUF split.count must be an unsigned integer"};
    const uint64_t count = count_value.value_or(1);
    if (count == 0 || count > 99999)
        return Error{ErrorCode::InvalidModel, "invalid GGUF split.count"};
    std::smatch match;
    const std::string filename = discovered.value().filename().string();
    const std::regex split_pattern("^(.+)-([0-9]{5})-of-([0-9]{5})\\.gguf$", std::regex::icase);
    const bool numbered = std::regex_match(filename, match, split_pattern);
    uint64_t selected_index = 0;
    if (numbered)
    {
        selected_index = std::stoull(match[2].str());
        if (selected_index == 0 || selected_index > count || std::stoull(match[3].str()) != count)
            return Error{ErrorCode::InvalidModel, "GGUF split filename disagrees with split.count"};
        --selected_index;
    }
    else if (count != 1)
        return Error{ErrorCode::InvalidModel, "split GGUF requires standard -00001-of-000NN.gguf filenames"};
    const auto total = metadata_integer(initial.value(), "split.tensors.count");
    if ((count != 1 && !total) || (initial.value().metadata.contains("split.tensors.count") && !total))
        return Error{ErrorCode::InvalidModel, "invalid or missing GGUF split.tensors.count"};
    if (total && *total > maximum_tensor_count)
        return Error{ErrorCode::InvalidModel, "GGUF package exceeds the tensor count limit"};

    GgufArchive archive;
    archive.selected_source = discovered.value();
    archive.selected_root = std::filesystem::is_directory(absolute, error) ? absolute : absolute.parent_path();
    uint64_t metadata_value_count = 0;
    for (uint64_t index = 0; index < count; ++index)
    {
        std::filesystem::path path = discovered.value();
        if (numbered)
        {
            const std::string number = std::to_string(index + 1);
            path = path.parent_path() / (match[1].str() + "-" + std::string(5 - number.size(), '0') + number + "-of-" + match[3].str() + path.extension().string());
        }
        auto shard = index == selected_index ? std::move(initial) : parse_gguf(path);
        if (!shard)
            return shard.error();
        const ParsedGguf& parsed = shard.value();
        if (metadata_integer(parsed, "split.count").value_or(1) != count
            || (count != 1 && metadata_integer(parsed, "split.no") != index)
            || (parsed.metadata.contains("split.no") && metadata_integer(parsed, "split.no") != index)
            || metadata_integer(parsed, "split.tensors.count") != total)
            return Error{ErrorCode::InvalidModel, "inconsistent GGUF shard metadata: " + path.string()};
        if (parsed.tensors.size() > maximum_tensor_count - archive.tensors.size())
            return Error{ErrorCode::InvalidModel, "GGUF package exceeds the tensor count limit"};
        for (auto& [key, value] : shard.value().metadata)
        {
            if (key == "split.no")
                continue;
            const auto found = archive.metadata.find(key);
            if (found != archive.metadata.end() && found->second != value)
                return Error{ErrorCode::InvalidModel, "GGUF shards disagree on metadata: " + key};
            if (found == archive.metadata.end())
            {
                // Nested arrays are rejected by the reader. Count the array
                // itself and every primitive once across distinct keys.
                const auto* values = value.array();
                const uint64_t size = values ? static_cast<uint64_t>(values->size()) + 1 : 1;
                if (size > maximum_metadata_values - metadata_value_count)
                    return Error{ErrorCode::InvalidModel, "GGUF package exceeds the metadata value limit"};
                metadata_value_count += size;
                archive.metadata.emplace(key, std::move(value));
            }
        }
        for (GgufTensorInfo& tensor : shard.value().tensors)
        {
            const std::string name = tensor.name;
            if (!archive.tensors.emplace(name, std::move(tensor)).second)
                return Error{ErrorCode::InvalidModel, "duplicate GGUF tensor name: " + name};
        }
    }
    if (total && *total != archive.tensors.size())
        return Error{ErrorCode::InvalidModel, "GGUF split.tensors.count disagrees with tensor descriptors"};
    return archive;
}

const GgufTensorInfo* GgufArchive::find(const std::string& name) const noexcept
{
    const auto found = tensors.find(name);
    return found == tensors.end() ? nullptr : &found->second;
}

const GgufMetadataValue* GgufArchive::find_metadata(const std::string& key) const noexcept
{
    const auto found = metadata.find(key);
    return found == metadata.end() ? nullptr : &found->second;
}

const std::unordered_map<std::string, GgufTensorInfo>& GgufArchive::tensor_infos() const noexcept
{
    return tensors;
}

const std::unordered_map<std::string, GgufMetadataValue>& GgufArchive::metadata_values() const noexcept
{
    return metadata;
}

const std::filesystem::path& GgufArchive::source_path() const noexcept
{
    return selected_source;
}

const std::filesystem::path& GgufArchive::root_path() const noexcept
{
    return selected_root;
}

} // namespace moe
} // namespace ncnn
