#ifndef NCNN_MOE_MODELS_JSON_H
#define NCNN_MOE_MODELS_JSON_H

#include "modeladapter.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

inline bool is_space(char value) noexcept
{
    return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

inline std::size_t skip_space(std::string_view value, std::size_t offset)
{
    while (offset < value.size() && is_space(value[offset]))
        ++offset;
    return offset;
}

inline std::optional<std::size_t> json_value_offset(std::string_view object, std::string_view key)
{
    const std::optional<std::string_view> value = find_manifest_member(object, key);
    if (!value)
        return std::nullopt;
    return static_cast<std::size_t>(value->data() - object.data());
}

inline std::optional<std::string> json_string(std::string_view object, std::string_view key)
{
    const std::optional<std::string_view> value = find_manifest_member(object, key);
    if (!value || value->empty() || value->front() != '"')
        return std::nullopt;

    std::size_t position = 0;
    std::string decoded;
    if (!parse_json_string(*value, position, &decoded) || position != value->size())
        return std::nullopt;
    return decoded;
}

inline std::optional<int64_t> json_integer(std::string_view object, std::string_view key)
{
    const auto offset_result = json_value_offset(object, key);
    if (!offset_result)
        return std::nullopt;
    std::size_t end = *offset_result;
    while (end < object.size()
           && object[end] != ','
           && object[end] != '}'
           && object[end] != ']'
           && !is_space(object[end]))
    {
        ++end;
    }
    int64_t result = 0;
    const auto parsed = std::from_chars(object.data() + *offset_result, object.data() + end, result);
    if (parsed.ec != std::errc() || parsed.ptr != object.data() + end)
        return std::nullopt;
    return result;
}

inline std::optional<double> json_number(std::string_view object, std::string_view key)
{
    const auto offset_result = json_value_offset(object, key);
    if (!offset_result)
        return std::nullopt;
    std::size_t end = *offset_result;
    while (end < object.size()
           && object[end] != ','
           && object[end] != '}'
           && object[end] != ']'
           && !is_space(object[end]))
    {
        ++end;
    }
    const std::string token(object.substr(*offset_result, end - *offset_result));
    char* parsed_end = nullptr;
    const double result = std::strtod(token.c_str(), &parsed_end);
    if (parsed_end == token.c_str() || *parsed_end != '\0')
        return std::nullopt;
    return result;
}

inline std::optional<bool> json_boolean(std::string_view object, std::string_view key)
{
    const std::optional<std::string_view> value = find_manifest_member(object, key);
    if (!value)
        return std::nullopt;
    if (*value == "true")
        return true;
    if (*value == "false")
        return false;
    return std::nullopt;
}

inline std::optional<std::vector<int64_t>> json_integer_array(std::string_view object, std::string_view key)
{
    const std::optional<std::string_view> value = find_manifest_member(object, key);
    if (!value || value->empty() || value->front() != '[')
        return std::nullopt;

    const std::string_view array = *value;
    std::vector<int64_t> result;
    std::size_t offset = skip_space(array, 1);
    if (offset < array.size() && array[offset] == ']')
    {
        if (offset + 1 != array.size())
            return std::nullopt;
        return result;
    }
    while (offset < array.size())
    {
        const std::size_t value_start = offset;
        while (offset < array.size()
               && array[offset] != ','
               && array[offset] != ']'
               && !is_space(array[offset]))
        {
            ++offset;
        }
        int64_t item = 0;
        const auto parsed = std::from_chars(array.data() + value_start, array.data() + offset, item);
        if (parsed.ec != std::errc() || parsed.ptr != array.data() + offset)
            return std::nullopt;
        result.push_back(item);
        offset = skip_space(array, offset);
        if (offset >= array.size())
            return std::nullopt;
        if (array[offset] == ']')
        {
            if (offset + 1 != array.size())
                return std::nullopt;
            return result;
        }
        if (array[offset] != ',')
            return std::nullopt;
        offset = skip_space(array, offset + 1);
    }
    return std::nullopt;
}

inline std::string json_escape(std::string_view value)
{
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (const char character : value)
    {
        switch (character)
        {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (static_cast<unsigned char>(character) < 0x20)
            {
                constexpr char hex[] = "0123456789abcdef";
                const unsigned char value = static_cast<unsigned char>(character);
                result += "\\u00";
                result.push_back(hex[value >> 4]);
                result.push_back(hex[value & 0x0f]);
            }
            else
                result.push_back(character);
            break;
        }
    }
    result.push_back('"');
    return result;
}

class JsonObject
{
public:
    void add_string(std::string_view name, std::string_view value)
    {
        key(name);
        text += json_escape(value);
    }

    void add_uint(std::string_view name, uint64_t value)
    {
        key(name);
        text += std::to_string(value);
    }

    void add_optional_uint(std::string_view name, const std::optional<uint64_t>& value)
    {
        if (value)
            add_uint(name, *value);
        else
            add_null(name);
    }

    void add_int(std::string_view name, int64_t value)
    {
        key(name);
        text += std::to_string(value);
    }

    void add_bool(std::string_view name, bool value)
    {
        key(name);
        text += value ? "true" : "false";
    }

    void add_double(std::string_view name, double value)
    {
        key(name);
        if (!std::isfinite(value))
        {
            text += "null";
            return;
        }
        std::ostringstream stream;
        stream << std::setprecision(10) << value;
        text += stream.str();
    }

    void add_optional_double(std::string_view name, const std::optional<double>& value)
    {
        if (value)
            add_double(name, *value);
        else
            add_null(name);
    }

    void add_null(std::string_view name)
    {
        key(name);
        text += "null";
    }

    void add_raw(std::string_view name, std::string_view value)
    {
        key(name);
        text.append(value);
    }

    [[nodiscard]] std::string finish()
    {
        text.push_back('}');
        return std::move(text);
    }

private:
    void key(std::string_view name)
    {
        if (!first)
            text.push_back(',');
        first = false;
        text += json_escape(name);
        text.push_back(':');
    }

    std::string text = "{";
    bool first = true;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELS_JSON_H
