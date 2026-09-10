#include "modeladapter.h"

#include "ncnn/moe/modeladapter.h"
#include "safetensors.h"
#include "storage/mappedfile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <utility>

namespace ncnn {
namespace moe {

static void skip_json_whitespace(const std::string& json, size_t& position) noexcept
{
    while (position < json.size()
           && (json[position] == ' ' || json[position] == '\t'
               || json[position] == '\r' || json[position] == '\n'))
    {
        ++position;
    }
}

static bool is_json_digit(char value) noexcept
{
    return value >= '0' && value <= '9';
}

static int json_hex_digit(char value) noexcept
{
    if (value >= '0' && value <= '9')
        return value - '0';
    if (value >= 'a' && value <= 'f')
        return value - 'a' + 10;
    if (value >= 'A' && value <= 'F')
        return value - 'A' + 10;
    return -1;
}

static bool append_json_codepoint(uint32_t codepoint, std::string& output)
{
    if (codepoint > 0x10ffff
        || (codepoint >= 0xd800 && codepoint <= 0xdfff))
    {
        return false;
    }

    if (codepoint <= 0x7f)
    {
        output.push_back(static_cast<char>(codepoint));
    }
    else if (codepoint <= 0x7ff)
    {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    else if (codepoint <= 0xffff)
    {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    else
    {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    return true;
}

static bool parse_json_hex_escape(const std::string& json, size_t& position, uint32_t& codepoint)
{
    if (json.size() - position < 4)
        return false;

    codepoint = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        const int digit = json_hex_digit(json[position + i]);
        if (digit < 0)
            return false;
        codepoint = (codepoint << 4) | static_cast<uint32_t>(digit);
    }
    position += 4;
    return true;
}

static bool parse_json_string(const std::string& json,
                              size_t& position,
                              std::string* decoded)
{
    if (position >= json.size() || json[position++] != '"')
        return false;
    if (decoded)
        decoded->clear();

    while (position < json.size())
    {
        const char value = json[position++];
        if (value == '"')
            return true;
        if (value == '\\')
        {
            if (position >= json.size())
                return false;
            const char escaped = json[position++];

            switch (escaped)
            {
            case '"':
            case '\\':
            case '/':
                if (decoded)
                    decoded->push_back(escaped);
                break;
            case 'b':
                if (decoded)
                    decoded->push_back('\b');
                break;
            case 'f':
                if (decoded)
                    decoded->push_back('\f');
                break;
            case 'n':
                if (decoded)
                    decoded->push_back('\n');
                break;
            case 'r':
                if (decoded)
                    decoded->push_back('\r');
                break;
            case 't':
                if (decoded)
                    decoded->push_back('\t');
                break;
            case 'u':
            {
                uint32_t codepoint = 0;
                if (!parse_json_hex_escape(json, position, codepoint))
                    return false;
                if (codepoint >= 0xd800 && codepoint <= 0xdbff)
                {
                    if (json.size() - position < 6
                        || json[position] != '\\' || json[position + 1] != 'u')
                    {
                        return false;
                    }
                    position += 2;
                    uint32_t low_surrogate = 0;
                    if (!parse_json_hex_escape(json, position, low_surrogate)
                        || low_surrogate < 0xdc00 || low_surrogate > 0xdfff)
                    {
                        return false;
                    }
                    codepoint = 0x10000
                                + ((codepoint - 0xd800) << 10)
                                + (low_surrogate - 0xdc00);
                }
                else if (codepoint >= 0xdc00 && codepoint <= 0xdfff)
                {
                    return false;
                }
                if (decoded && !append_json_codepoint(codepoint, *decoded))
                    return false;
                break;
            }
            default:
                return false;
            }
        }
        else
        {
            if (static_cast<unsigned char>(value) < 0x20)
                return false;
            if (decoded)
                decoded->push_back(value);
        }
    }
    return false;
}

static bool is_json_number(const std::string& value) noexcept
{
    size_t position = 0;
    if (position < value.size() && value[position] == '-')
        ++position;
    if (position >= value.size())
        return false;

    if (value[position] == '0')
    {
        ++position;
        if (position < value.size() && is_json_digit(value[position]))
            return false;
    }
    else
    {
        if (value[position] < '1' || value[position] > '9')
            return false;
        while (position < value.size() && is_json_digit(value[position]))
            ++position;
    }

    if (position < value.size() && value[position] == '.')
    {
        ++position;
        const size_t fraction_start = position;
        while (position < value.size() && is_json_digit(value[position]))
            ++position;
        if (position == fraction_start)
            return false;
    }

    if (position < value.size() && (value[position] == 'e' || value[position] == 'E'))
    {
        ++position;
        if (position < value.size() && (value[position] == '+' || value[position] == '-'))
            ++position;
        const size_t exponent_start = position;
        while (position < value.size() && is_json_digit(value[position]))
            ++position;
        if (position == exponent_start)
            return false;
    }
    return position == value.size();
}

static bool scan_json_value(const std::string& json, size_t& position)
{
    skip_json_whitespace(json, position);
    if (position >= json.size())
        return false;

    if (json[position] == '"')
    {
        if (!parse_json_string(json, position, nullptr))
            return false;
    }
    else if (json[position] == '{' || json[position] == '[')
    {
        const char opening = json[position++];
        std::vector<char> delimiters{opening};
        while (position < json.size() && !delimiters.empty())
        {
            if (json[position] == '"')
            {
                if (!parse_json_string(json, position, nullptr))
                    return false;
                continue;
            }
            if (json[position] == '{')
                delimiters.push_back('{');
            else if (json[position] == '[')
                delimiters.push_back('[');
            else if (json[position] == '}')
            {
                if (delimiters.empty() || delimiters.back() != '{')
                    return false;
                delimiters.pop_back();
            }
            else if (json[position] == ']')
            {
                if (delimiters.empty() || delimiters.back() != '[')
                    return false;
                delimiters.pop_back();
            }
            ++position;
        }
        if (!delimiters.empty())
            return false;
    }
    else
    {
        const size_t value_start = position;
        while (position < json.size()
               && json[position] != ',' && json[position] != '}' && json[position] != ']'
               && json[position] != ' ' && json[position] != '\t'
               && json[position] != '\r' && json[position] != '\n')
        {
            ++position;
        }
        if (position == value_start)
            return false;
    }

    return true;
}

static bool scan_json_array(const std::string& json,
                            size_t& position,
                            std::vector<std::string>& elements)
{
    if (position >= json.size() || json[position++] != '[')
        return false;
    skip_json_whitespace(json, position);
    if (position < json.size() && json[position] == ']')
    {
        ++position;
        return true;
    }

    while (position < json.size())
    {
        skip_json_whitespace(json, position);
        const size_t value_start = position;
        if (!scan_json_value(json, position))
            return false;
        elements.push_back(json.substr(value_start, position - value_start));
        skip_json_whitespace(json, position);
        if (position >= json.size())
            return false;
        if (json[position] == ']')
        {
            ++position;
            return true;
        }
        if (json[position++] != ',')
            return false;
        skip_json_whitespace(json, position);
        if (position >= json.size() || json[position] == ']')
            return false;
    }
    return false;
}

std::optional<std::string> find_manifest_member(const std::string& json, const std::string& key)
{
    size_t position = 0;
    skip_json_whitespace(json, position);
    if (position >= json.size() || json[position++] != '{')
        return std::nullopt;
    skip_json_whitespace(json, position);
    if (position < json.size() && json[position] == '}')
        return std::nullopt;

    while (position < json.size())
    {
        std::string name;
        if (!parse_json_string(json, position, &name))
            return std::nullopt;
        skip_json_whitespace(json, position);
        if (position >= json.size() || json[position++] != ':')
            return std::nullopt;
        skip_json_whitespace(json, position);
        const size_t value_start = position;
        if (!scan_json_value(json, position))
            return std::nullopt;
        const size_t value_end = position;
        size_t delimiter = value_end;
        skip_json_whitespace(json, delimiter);
        if (delimiter >= json.size() || (json[delimiter] != ',' && json[delimiter] != '}'))
            return std::nullopt;
        if (name == key)
            return json.substr(value_start, value_end - value_start);

        position = delimiter;
        if (position >= json.size())
            return std::nullopt;
        if (json[position] == '}')
            return std::nullopt;
        if (json[position++] != ',')
            return std::nullopt;
        skip_json_whitespace(json, position);
        if (position >= json.size() || json[position] == '}')
            return std::nullopt;
    }
    return std::nullopt;
}

Result<std::string> read_manifest_object(const std::string& json, const std::string& key, const char* prefix)
{
    const std::optional<std::string> value = find_manifest_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing object field: " + key};

    if (value->empty() || (*value)[0] != '{')
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "object field: " + key};
    return *value;
}

static bool is_json_uint(const std::string& value) noexcept
{
    if (value.empty())
        return false;
    if (value == "0")
        return true;
    if (value[0] < '1' || value[0] > '9')
        return false;
    for (size_t i = 1; i < value.size(); ++i)
    {
        if (!is_json_digit(value[i]))
            return false;
    }
    return true;
}

Result<uint32_t> read_manifest_uint32(const std::string& json, const std::string& key, const char* prefix)
{
    const std::optional<std::string> value = find_manifest_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing integer field: " + key};
    if (!is_json_uint(*value))
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "integer field: " + key};

    try
    {
        const unsigned long long parsed_value = std::stoull(*value);
        if (parsed_value > std::numeric_limits<uint32_t>::max())
            return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest integer is out of range: " + key};
        return static_cast<uint32_t>(parsed_value);
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "integer field: " + key};
    }
}

Result<std::string> read_manifest_string(const std::string& json, const std::string& key, const char* prefix)
{
    const std::optional<std::string> value = find_manifest_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing string field: " + key};

    size_t position = 0;
    std::string decoded;
    if (!parse_json_string(*value, position, &decoded))
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "string field: " + key};
    skip_json_whitespace(*value, position);
    if (position != value->size())
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "string field: " + key};
    if (decoded.empty())
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing string field: " + key};
    return decoded;
}

Result<float> read_manifest_float(const std::string& json, const std::string& key, const char* prefix)
{
    const std::optional<std::string> value = find_manifest_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing numeric field: " + key};
    if (!is_json_number(*value))
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "numeric field: " + key};
    try
    {
        size_t consumed = 0;
        const float parsed_value = std::stof(*value, &consumed);
        if (consumed != value->size() || !std::isfinite(parsed_value))
            return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "numeric field: " + key};
        return parsed_value;
    }
    catch (const std::exception&)
    {
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "numeric field: " + key};
    }
}

Result<bool> read_manifest_bool(const std::string& json, const std::string& key, const char* prefix)
{
    const std::optional<std::string> value = find_manifest_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing boolean field: " + key};
    if (*value != "true" && *value != "false")
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "boolean field: " + key};
    return *value == "true";
}

Result<std::vector<uint32_t>> read_manifest_uint32_array(const std::string& json, const std::string& key, const char* prefix)
{
    const std::optional<std::string> value = find_manifest_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing integer array: " + key};

    std::vector<std::string> elements;
    size_t position = 0;
    if (!scan_json_array(*value, position, elements))
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "integer array: " + key};
    skip_json_whitespace(*value, position);
    if (position != value->size())
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "integer array: " + key};

    std::vector<uint32_t> result;
    result.reserve(elements.size());
    for (const std::string& element : elements)
    {
        if (!is_json_uint(element))
            return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "integer array: " + key};
        try
        {
            const unsigned long long parsed_value = std::stoull(element);
            if (parsed_value > std::numeric_limits<uint32_t>::max())
            {
                return Error{ErrorCode::InvalidModel,
                             std::string(prefix) + "integer array value is out of range: " + key};
            }
            result.push_back(static_cast<uint32_t>(parsed_value));
        }
        catch (const std::exception&)
        {
            return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "integer array: " + key};
        }
    }
    return result;
}

Result<std::vector<std::string>> read_manifest_string_array(const std::string& json, const std::string& key, const char* prefix)
{
    const std::optional<std::string> value = find_manifest_member(json, key);
    if (!value)
        return Error{ErrorCode::InvalidModel, std::string(prefix) + "manifest is missing string array: " + key};

    std::vector<std::string> elements;
    size_t position = 0;
    if (!scan_json_array(*value, position, elements))
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "string array: " + key};
    skip_json_whitespace(*value, position);
    if (position != value->size())
        return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "string array: " + key};

    for (std::string& element : elements)
    {
        size_t element_position = 0;
        std::string decoded;
        if (!parse_json_string(element, element_position, &decoded))
            return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "string array: " + key};
        skip_json_whitespace(element, element_position);
        if (element_position != element.size())
            return Error{ErrorCode::InvalidModel, "invalid " + std::string(prefix) + "string array: " + key};
        element = std::move(decoded);
    }
    return elements;
}

float optional_manifest_float(const std::string& json, const std::string& key, float fallback)
{
    auto value = read_manifest_float(json, key);
    return value ? value.value() : fallback;
}

Result<uint32_t> get_rotary_dimension(uint32_t head_dimension,
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

static Result<uint64_t> fnv1a64_file(const std::filesystem::path& path,
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

Result<bool> optional_artifact_exists(const std::filesystem::path& path,
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

static std::string mxfp4_artifact_identity_name(const char* prefix,
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

static Result<void> validate_u8_artifact_tensor(const SafetensorsArchive& archive,
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

Result<void> validate_mxfp4_artifact_identity(const SafetensorsArchive& archive,
                                              const ModelPackage& package,
                                              const char* identity_prefix,
                                              uint32_t layer_count,
                                              uint32_t mtp_layer_count,
                                              uint32_t expert_count,
                                              uint32_t hidden_size,
                                              uint32_t intermediate_size,
                                              const char* identity_description,
                                              const char* artifact_description)
{
    uint64_t config_hash = UINT64_C(14695981039346656037);
    for (unsigned char value : package.manifest.raw_json)
    {
        config_hash ^= value;
        config_hash *= UINT64_C(1099511628211);
    }
    auto index_hash = fnv1a64_file(package.root / "model.safetensors.index.json", identity_description);
    if (!index_hash)
        return index_hash.error();
    const std::string identity = mxfp4_artifact_identity_name(identity_prefix,
                                                              layer_count,
                                                              mtp_layer_count,
                                                              expert_count,
                                                              hidden_size,
                                                              intermediate_size,
                                                              config_hash,
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

Result<void> validate_mxfp4_artifact_expert_bank(const SafetensorsArchive& archive,
                                                 const std::string& prefix,
                                                 uint32_t expert_count,
                                                 uint32_t hidden_size,
                                                 uint32_t intermediate_size,
                                                 const char* description)
{
    auto status = validate_u8_artifact_tensor(archive,
                                              prefix + "gate_up.blocks",
                                              {expert_count, intermediate_size * 2, hidden_size / 32, 16},
                                              description);
    if (!status)
        return status.error();

    status = validate_u8_artifact_tensor(archive,
                                         prefix + "gate_up.scales",
                                         {expert_count, intermediate_size * 2, hidden_size / 32},
                                         description);
    if (!status)
        return status.error();

    status = validate_u8_artifact_tensor(archive,
                                         prefix + "down.blocks",
                                         {expert_count, hidden_size, intermediate_size / 32, 16},
                                         description);
    if (!status)
        return status.error();

    return validate_u8_artifact_tensor(archive,
                                       prefix + "down.scales",
                                       {expert_count, hidden_size, intermediate_size / 32},
                                       description);
}

Result<void> add_tensor(WeightMapping& mapping,
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

Result<void> add_bfloat16_slice(WeightMapping& mapping,
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

Result<void> add_bfloat16_expert_bank(WeightMapping& mapping,
                                      const SafetensorsArchive& archive,
                                      const std::string& target_prefix,
                                      const std::string& target_suffix,
                                      const std::string& source_name,
                                      uint32_t expert_count,
                                      const std::vector<uint32_t>& shape)
{
    const SafetensorInfo* source = archive.find(source_name);
    if (!source || source->dtype != "BF16" || expert_count == 0 || shape.empty()
        || source->shape.size() != shape.size() + 1
        || source->shape.front() != expert_count)
    {
        return Error{ErrorCode::InvalidModel, "invalid BF16 Expert bank tensor: " + source_name};
    }

    uint64_t slice_elements = 1;
    for (uint32_t dimension : shape)
    {
        if (dimension == 0 || slice_elements > std::numeric_limits<uint64_t>::max() / dimension)
        {
            return Error{ErrorCode::InvalidModel, "invalid BF16 Expert bank tensor: " + source_name};
        }
        slice_elements *= dimension;
    }
    if (!std::equal(shape.begin(), shape.end(), source->shape.begin() + 1))
    {
        return Error{ErrorCode::InvalidModel, "invalid BF16 Expert bank tensor: " + source_name};
    }
    if (slice_elements > std::numeric_limits<uint64_t>::max() / sizeof(uint16_t))
    {
        return Error{ErrorCode::InvalidModel, "invalid BF16 Expert bank tensor: " + source_name};
    }
    const uint64_t slice_size = slice_elements * sizeof(uint16_t);
    if (slice_size > std::numeric_limits<uint64_t>::max() / expert_count)
    {
        return Error{ErrorCode::InvalidModel, "invalid BF16 Expert bank tensor: " + source_name};
    }
    const uint64_t bank_size = slice_size * expert_count;
    if (slice_size > std::numeric_limits<size_t>::max()
        || source->size != bank_size
        || source->offset > std::numeric_limits<uint64_t>::max() - source->size)
    {
        return Error{ErrorCode::InvalidModel, "invalid BF16 Expert bank tensor: " + source_name};
    }

    std::shared_ptr<const uint8_t> bank_data;
    {
        auto mapped = MappedFileRange::open(source->path, source->offset, bank_size);
        if (mapped && reinterpret_cast<uintptr_t>(mapped.value()->data()) % alignof(uint16_t) == 0)
            bank_data = mapped.value()->share_data();
    }

    if (bank_data)
    {
        for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
        {
            TensorData slice;
            slice.dtype = DType::BFloat16;
            slice.shape = shape;
            const size_t byte_offset = static_cast<size_t>(static_cast<uint64_t>(expert_id) * slice_size);
            slice.mapped_data = std::shared_ptr<const uint8_t>(bank_data, bank_data.get() + byte_offset);
            slice.mapped_size = slice_size;
            mapping.emplace(target_prefix + "experts." + std::to_string(expert_id) + "." + target_suffix,
                            std::move(slice));
        }
        return {};
    }

    for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
    {
        auto status = add_bfloat16_slice(mapping, archive,
                                         target_prefix + "experts." + std::to_string(expert_id) + "." + target_suffix,
                                         source_name, expert_id, shape);
        if (!status)
            return status.error();
    }
    return {};
}

static Result<void> add_query_gate(WeightMapping& mapping,
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
    TensorData gate;
    gate.dtype = DType::BFloat16;
    gate.shape = {static_cast<uint32_t>(query_rows), hidden_size};
    gate.bfloat16_data.resize(static_cast<size_t>(query_rows * hidden_size));
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

Result<void> add_qwen_attention(WeightMapping& mapping, const SafetensorsArchive& archive,
                                const std::string& source_prefix, const std::string& target_prefix,
                                uint32_t head_count, uint32_t head_dimension, uint32_t hidden_size, const char* description)
{
    auto ret = add_query_gate(mapping, archive, source_prefix + "self_attn.q_proj.weight", target_prefix,
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

Result<void> add_qwen_gated_delta_net(WeightMapping& mapping, const SafetensorsArchive& archive,
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

Result<void> add_qwen_shared_expert(WeightMapping& mapping, const SafetensorsArchive& archive,
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

Result<void> add_mxfp4_expert(WeightMapping& mapping,
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
