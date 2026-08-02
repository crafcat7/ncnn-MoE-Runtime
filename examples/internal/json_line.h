#ifndef NCNN_MOE_EXAMPLES_JSON_LINE_H
#define NCNN_MOE_EXAMPLES_JSON_LINE_H

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
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
    std::string needle;
    needle.reserve(key.size() + 2);
    needle.push_back('"');
    needle.append(key);
    needle.push_back('"');
    std::size_t search = 0;
    while (true)
    {
        const std::size_t key_offset = object.find(needle, search);
        if (key_offset == std::string_view::npos)
            return std::nullopt;
        std::size_t offset = skip_space(object, key_offset + needle.size());
        if (offset < object.size() && object[offset] == ':')
            return skip_space(object, offset + 1);
        search = key_offset + needle.size();
    }
}

inline std::optional<std::string> json_string(std::string_view object, std::string_view key)
{
    const auto offset_result = json_value_offset(object, key);
    if (!offset_result || *offset_result >= object.size() || object[*offset_result] != '"')
        return std::nullopt;

    std::string result;
    for (std::size_t offset = *offset_result + 1; offset < object.size(); ++offset)
    {
        const char value = object[offset];
        if (value == '"')
            return result;
        if (value != '\\')
        {
            result.push_back(value);
            continue;
        }
        if (++offset >= object.size())
            return std::nullopt;
        switch (object[offset])
        {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        default: return std::nullopt;
        }
    }
    return std::nullopt;
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
    const auto offset_result = json_value_offset(object, key);
    if (!offset_result)
        return std::nullopt;
    if (object.substr(*offset_result, 4) == "true")
        return true;
    if (object.substr(*offset_result, 5) == "false")
        return false;
    return std::nullopt;
}

inline std::optional<std::vector<int64_t>> json_integer_array(std::string_view object, std::string_view key)
{
    const auto offset_result = json_value_offset(object, key);
    if (!offset_result || *offset_result >= object.size() || object[*offset_result] != '[')
        return std::nullopt;

    std::vector<int64_t> result;
    std::size_t offset = skip_space(object, *offset_result + 1);
    if (offset < object.size() && object[offset] == ']')
        return result;
    while (offset < object.size())
    {
        const std::size_t value_start = offset;
        while (offset < object.size()
               && object[offset] != ','
               && object[offset] != ']'
               && !is_space(object[offset]))
        {
            ++offset;
        }
        int64_t value = 0;
        const auto parsed = std::from_chars(object.data() + value_start, object.data() + offset, value);
        if (parsed.ec != std::errc() || parsed.ptr != object.data() + offset)
            return std::nullopt;
        result.push_back(value);
        offset = skip_space(object, offset);
        if (offset >= object.size())
            return std::nullopt;
        if (object[offset] == ']')
            return result;
        if (object[offset] != ',')
            return std::nullopt;
        offset = skip_space(object, offset + 1);
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
                result += "?";
            else
                result.push_back(character);
            break;
        }
    }
    result.push_back('"');
    return result;
}

inline std::optional<std::string> string(std::string_view object, std::string_view key)
{
    return json_string(object, key);
}

inline std::optional<int64_t> integer(std::string_view object, std::string_view key)
{
    return json_integer(object, key);
}

inline std::optional<double> number(std::string_view object, std::string_view key)
{
    return json_number(object, key);
}

inline std::optional<bool> boolean(std::string_view object, std::string_view key)
{
    return json_boolean(object, key);
}

inline std::optional<std::vector<int64_t>> integer_array(std::string_view object, std::string_view key)
{
    return json_integer_array(object, key);
}

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXAMPLES_JSON_LINE_H
