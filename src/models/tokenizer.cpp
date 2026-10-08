// Portions of byte-level BPE handling are derived from Palm-Infra's Apache-2.0
// tokenizer implementation. See tokenizer.LICENSE.
#include "tokenizer.h"

#if defined(NCNN_MOE_TOKENIZER_ICU)
#include "json.h"
#include "modeladapter.h"
#include "storage/mappedfile.h"
#endif

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#if defined(NCNN_MOE_TOKENIZER_ICU)
#include <unicode/normalizer2.h>
#include <unicode/parseerr.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>
#include <unicode/uchar.h>
#include <unicode/uniset.h>
#endif

namespace ncnn {
namespace moe {

// Mix both token IDs before using them as an unordered_map key. The default
// integer hash leaves low bits unchanged, which clusters packed IDs when the
// implementation uses power-of-two bucket counts.
static uint64_t merge_key(uint32_t left, uint32_t right) noexcept
{
    uint64_t key = (static_cast<uint64_t>(left) << 32) | right;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

#if defined(NCNN_MOE_TOKENIZER_ICU)
static const char qwen36_template[] = R"QWEN({%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %})QWEN";

static const char qwen38_template[] = R"QWEN38({%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + content + '<|im_end|>\n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %})QWEN38";

#endif

struct Tokenizer::UnicodeProfile
{
#if defined(NCNN_MOE_TOKENIZER_ICU)
    std::vector<std::unique_ptr<const icu::RegexPattern>> patterns;
    std::unique_ptr<icu::UnicodeSet> filter;
    std::unique_ptr<icu::FilteredNormalizer2> nfc;
#endif
};

struct Tokenizer::EncodeScratch
{
    using Candidate = std::tuple<uint32_t, size_t, size_t, int32_t, int32_t, int32_t>;

    struct Node
    {
        int32_t id;
        size_t prev;
        size_t next;
    };

    std::vector<Candidate> candidates;
    std::vector<Node> nodes;
    std::string chunk;
#if defined(NCNN_MOE_TOKENIZER_ICU)
    std::vector<icu::UnicodeString> pieces;
    std::vector<icu::UnicodeString> split;
#endif
};

Tokenizer::Tokenizer() = default;

Tokenizer::~Tokenizer() = default;

Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;

Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

bool Tokenizer::load(const std::string& model_directory, size_t vocabulary_size)
{
    Tokenizer loaded;
    try
    {
        if (!loaded.load_impl(model_directory, vocabulary_size))
            return false;
    }
    catch (...)
    {
        return false;
    }
    *this = std::move(loaded);
    return true;
}

bool Tokenizer::load_impl(const std::string& model_directory, size_t vocabulary_size)
{
#if !defined(NCNN_MOE_TOKENIZER_ICU)
    (void)model_directory;
    (void)vocabulary_size;
    return false;
#else
    static const std::string qwen_regex = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    static const std::string icu_regex = R"NMR((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n[\p{L}&&\p{Age=16.0}][\p{N}&&\p{Age=16.0}]]?[[\p{L}&&\p{Age=16.0}][\p{M}&&\p{Age=16.0}]]+|[\p{N}&&\p{Age=16.0}]| ?[^\s[\p{L}&&\p{Age=16.0}][\p{M}&&\p{Age=16.0}][\p{N}&&\p{Age=16.0}]]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)NMR";
    static const std::string gpt_regex = R"GPT([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)GPT";
    static const std::array<std::string, 3> deepseek_regexes = {
        R"(\p{N}{1,3})",
        R"([一-龥぀-ゟ゠-ヿ]+)",
        [] {
            std::string value = R"DS([!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)DS";
            for (size_t pos = 0; (pos = value.find("\\r", pos)) != std::string::npos;)
            {
                value.replace(pos, 2, "\r");
                ++pos;
            }
            for (size_t pos = 0; (pos = value.find("\\n", pos)) != std::string::npos;)
            {
                value.replace(pos, 2, "\n");
                ++pos;
            }
            return value;
        }()};
    static constexpr uint32_t max_id = 1000000;
    static constexpr size_t max_bytes = 64 * 1024 * 1024;

    auto tokenizer_path = std::filesystem::path(model_directory) / "tokenizer.json";
    const uint64_t tokenizer_size = std::filesystem::file_size(tokenizer_path);
    auto mapping_result = MappedFileRange::open(tokenizer_path, 0, tokenizer_size);
    if (!mapping_result)
        return false;
    const std::shared_ptr<MappedFileRange> mapping = std::move(mapping_result).value();
    const std::string_view tokenizer(reinterpret_cast<const char*>(mapping->data()), mapping->size());

    const std::filesystem::path config_path = std::filesystem::path(model_directory) / "tokenizer_config.json";
    std::ifstream config_input(config_path, std::ios::binary);
    if (!config_input)
        return false;
    const std::string config_text((std::istreambuf_iterator<char>(config_input)), std::istreambuf_iterator<char>());
    if (config_input.bad())
        return false;
    const std::string_view config(config_text);

    std::ifstream model_input(std::filesystem::path(model_directory) / "config.json", std::ios::binary);
    if (!model_input)
        return false;
    const std::string model_config((std::istreambuf_iterator<char>(model_input)), std::istreambuf_iterator<char>());
    if (model_input.bad())
        return false;
    const std::optional<std::string> model_type = json_string(model_config, "model_type");
    Family new_family = Family::None;
    size_t base_vocab_size = 0;
    size_t total_vocab_size = 0;
    bool has_nfc = false;
    std::vector<std::string> split_regexes;
    if (model_type == "gpt_oss")
    {
        new_family = Family::GptOss;
        base_vocab_size = 199998;
        total_vocab_size = 201088;
        split_regexes.push_back(gpt_regex);
    }
    else if (model_type == "deepseek_v4")
    {
        new_family = Family::DeepSeekV4;
        base_vocab_size = 128000;
        total_vocab_size = 129280;
        split_regexes.assign(deepseek_regexes.begin(), deepseek_regexes.end());
    }
    else if (model_type == "qwen3_5_moe")
    {
        new_family = Family::Qwen36;
        base_vocab_size = 248044;
        total_vocab_size = 248320;
        has_nfc = true;
        split_regexes.push_back(qwen_regex);
    }
    else if (model_type == "qwen4_exp")
    {
        new_family = Family::Qwen38;
        base_vocab_size = 248044;
        total_vocab_size = 248320;
        has_nfc = true;
        split_regexes.push_back(qwen_regex);
    }
    else
        return false;
    std::optional<std::string_view> text_config;
    if (new_family == Family::Qwen36 || new_family == Family::Qwen38)
    {
        text_config = find_manifest_member(std::string_view(model_config), "text_config");
        const std::string expected_text_type = new_family == Family::Qwen36 ? "qwen3_5_moe_text" : "qwen4_exp_text";
        if (!text_config || json_integer(*text_config, "vocab_size") != std::optional<int64_t>(248320) || json_string(*text_config, "model_type") != std::optional<std::string>(expected_text_type))
            return false;
    }
    else if (json_integer(model_config, "vocab_size") != std::optional<int64_t>(static_cast<int64_t>(total_vocab_size)))
        return false;

    if (new_family == Family::Qwen36 || new_family == Family::Qwen38)
    {
        std::ifstream template_input(std::filesystem::path(model_directory) / "chat_template.jinja", std::ios::binary);
        if (!template_input)
            return false;
        const std::string template_file((std::istreambuf_iterator<char>(template_input)), std::istreambuf_iterator<char>());
        if (template_input.bad())
            return false;
        const std::string_view pinned_template = new_family == Family::Qwen36
                                                     ? std::string_view(qwen36_template, sizeof(qwen36_template) - 1)
                                                     : std::string_view(qwen38_template, sizeof(qwen38_template) - 1);
        const std::optional<std::string> config_template = json_string(config, "chat_template");
        if (template_file != pinned_template || !config_template || *config_template != pinned_template)
            return false;
    }

    auto take_value = [](std::string_view text, size_t& position, std::string_view& value) {
        position = skip_space(text, position);
        if (position >= text.size())
            return false;
        const size_t begin = position;
        if (text[position] == '"')
        {
            if (!parse_json_string(text, position, nullptr))
                return false;
        }
        else if (text[position] == '{' || text[position] == '[')
        {
            std::vector<char> delimiters{text[position++]};
            while (position < text.size() && !delimiters.empty())
            {
                if (text[position] == '"')
                {
                    if (!parse_json_string(text, position, nullptr))
                        return false;
                    continue;
                }
                const char current = text[position++];
                if (current == '{' || current == '[')
                    delimiters.push_back(current);
                else if (current == '}' || current == ']')
                {
                    if ((current == '}' && delimiters.back() != '{') || (current == ']' && delimiters.back() != '['))
                        return false;
                    delimiters.pop_back();
                }
            }
            if (!delimiters.empty())
                return false;
        }
        else
        {
            while (position < text.size() && text[position] != ',' && text[position] != '}' && text[position] != ']' && !is_space(text[position]))
                ++position;
            if (begin == position)
                return false;
        }
        value = text.substr(begin, position - begin);
        return true;
    };
    auto object_size = [&](std::string_view object, size_t& count) {
        count = 0;
        std::unordered_set<std::string> keys;
        size_t position = skip_space(object, 0);
        if (position >= object.size() || object[position++] != '{')
            return false;
        position = skip_space(object, position);
        if (position < object.size() && object[position] == '}')
            return skip_space(object, position + 1) == object.size();
        for (;;)
        {
            std::string key;
            if (!parse_json_string(object, position, &key) || !keys.emplace(std::move(key)).second)
                return false;
            position = skip_space(object, position);
            if (position >= object.size() || object[position++] != ':')
                return false;
            std::string_view value;
            if (!take_value(object, position, value))
                return false;
            ++count;
            position = skip_space(object, position);
            if (position >= object.size())
                return false;
            if (object[position] == '}')
                return skip_space(object, position + 1) == object.size();
            if (object[position++] != ',')
                return false;
            position = skip_space(object, position);
        }
    };
    auto member = [](std::string_view object, std::string_view key) {
        const std::optional<std::string_view> value = find_manifest_member(object, key);
        if (!value)
            throw std::runtime_error("missing tokenizer metadata member");
        return *value;
    };
    auto require_size = [&](std::string_view object, size_t expected) {
        size_t count = 0;
        if (!object_size(object, count) || count != expected)
            throw std::runtime_error("unsupported tokenizer metadata object");
    };
    auto is_byte_level = [&](std::string_view value, bool add_prefix_space, bool trim_offsets, bool use_regex) {
        size_t count = 0;
        return object_size(value, count) && count == 4 && json_string(value, "type") == std::optional<std::string>("ByteLevel") && json_boolean(value, "add_prefix_space") == std::optional<bool>(add_prefix_space) && json_boolean(value, "trim_offsets") == std::optional<bool>(trim_offsets) && json_boolean(value, "use_regex") == std::optional<bool>(use_regex);
    };

    require_size(tokenizer, 9);
    if (json_string(tokenizer, "version") != std::optional<std::string>("1.0") || member(tokenizer, "truncation") != "null" || member(tokenizer, "padding") != "null")
        return false;
    const std::string_view normalizer = member(tokenizer, "normalizer");
    if (has_nfc)
    {
        require_size(normalizer, 1);
        if (json_string(normalizer, "type") != std::optional<std::string>("NFC"))
            return false;
    }
    else if (new_family == Family::GptOss)
    {
        if (normalizer != "null")
            return false;
    }
    else
    {
        require_size(normalizer, 2);
        if (json_string(normalizer, "type") != std::optional<std::string>("Sequence") || member(normalizer, "normalizers") != "[]")
            return false;
    }
    const std::string_view pre = member(tokenizer, "pre_tokenizer");
    require_size(pre, 2);
    if (json_string(pre, "type") != std::optional<std::string>("Sequence"))
        return false;
    const std::string_view pretokenizers = member(pre, "pretokenizers");
    std::vector<std::string_view> stages;
    size_t position = skip_space(pretokenizers, 0);
    if (position >= pretokenizers.size() || pretokenizers[position++] != '[')
        return false;
    for (;;)
    {
        position = skip_space(pretokenizers, position);
        if (position < pretokenizers.size() && pretokenizers[position] == ']')
        {
            ++position;
            if (skip_space(pretokenizers, position) != pretokenizers.size())
                return false;
            break;
        }
        std::string_view stage;
        if (!take_value(pretokenizers, position, stage))
            return false;
        stages.push_back(stage);
        position = skip_space(pretokenizers, position);
        if (position >= pretokenizers.size())
            return false;
        if (pretokenizers[position] == ']')
        {
            ++position;
            if (skip_space(pretokenizers, position) != pretokenizers.size())
                return false;
            break;
        }
        if (pretokenizers[position++] != ',')
            return false;
    }
    if (stages.size() != split_regexes.size() + 1)
        return false;
    for (size_t i = 0; i < split_regexes.size(); ++i)
    {
        require_size(stages[i], 4);
        const std::string_view pattern = member(stages[i], "pattern");
        require_size(pattern, 1);
        if (json_string(stages[i], "type") != std::optional<std::string>("Split") || json_string(stages[i], "behavior") != std::optional<std::string>("Isolated") || json_boolean(stages[i], "invert") != std::optional<bool>(false) || json_string(pattern, "Regex") != std::optional<std::string>(split_regexes[i]))
            return false;
    }
    const bool qwen_bytelevel = new_family == Family::Qwen36 || new_family == Family::Qwen38;
    if (!is_byte_level(stages.back(), false, !qwen_bytelevel, false)
        || !is_byte_level(member(tokenizer, "post_processor"), !qwen_bytelevel, false, !qwen_bytelevel)
        || !is_byte_level(member(tokenizer, "decoder"), !qwen_bytelevel, !qwen_bytelevel, !qwen_bytelevel))
        return false;

    if (new_family == Family::Qwen36 || new_family == Family::Qwen38)
    {
        require_size(config, 16);
        if (json_string(config, "tokenizer_class") != std::optional<std::string>("Qwen2Tokenizer") || json_boolean(config, "clean_up_tokenization_spaces") != std::optional<bool>(false) || json_boolean(config, "add_prefix_space") != std::optional<bool>(false) || json_boolean(config, "split_special_tokens") != std::optional<bool>(false) || json_boolean(config, "add_bos_token") != std::optional<bool>(false) || json_string(config, "errors") != std::optional<std::string>("replace") || member(config, "bos_token") != "null" || member(config, "unk_token") != "null" || json_string(config, "eos_token") != std::optional<std::string>("<|im_end|>") || json_string(config, "pad_token") != std::optional<std::string>("<|endoftext|>") || json_string(config, "pretokenize_regex") != std::optional<std::string>(qwen_regex) || json_integer(config, "model_max_length") != std::optional<int64_t>(262144))
            return false;
    }
    else if (new_family == Family::GptOss)
    {
        require_size(config, 9);
        if (json_string(config, "tokenizer_class") != std::optional<std::string>("PreTrainedTokenizerFast") || json_boolean(config, "clean_up_tokenization_spaces") != std::optional<bool>(false) || json_string(config, "bos_token") != std::optional<std::string>("<|startoftext|>") || json_string(config, "eos_token") != std::optional<std::string>("<|return|>") || json_string(config, "pad_token") != std::optional<std::string>("<|endoftext|>") || member(config, "extra_special_tokens") != "{}")
            return false;
    }
    else
    {
        const std::string_view bos = member(config, "bos_token");
        const std::string_view eos = member(config, "eos_token");
        const std::string_view pad = member(config, "pad_token");
        const auto valid_deepseek_token = [&](std::string_view item, std::string_view content) {
            size_t count = 0;
            return object_size(item, count) && count == 6
                   && json_string(item, "__type") == std::optional<std::string>("AddedToken")
                   && json_string(item, "content") == std::optional<std::string>(content)
                   && json_boolean(item, "lstrip") == std::optional<bool>(false)
                   && json_boolean(item, "normalized") == std::optional<bool>(true)
                   && json_boolean(item, "rstrip") == std::optional<bool>(false)
                   && json_boolean(item, "single_word") == std::optional<bool>(false);
        };
        require_size(config, 11);
        if (json_string(config, "tokenizer_class") != std::optional<std::string>("PreTrainedTokenizerFast") || json_boolean(config, "clean_up_tokenization_spaces") != std::optional<bool>(false) || json_boolean(config, "add_bos_token") != std::optional<bool>(false) || json_boolean(config, "add_eos_token") != std::optional<bool>(false) || json_boolean(config, "legacy") != std::optional<bool>(true) || json_integer(config, "model_max_length") != std::optional<int64_t>(1048576) || !valid_deepseek_token(bos, "<｜begin▁of▁sentence｜>") || !valid_deepseek_token(eos, "<｜end▁of▁sentence｜>") || !valid_deepseek_token(pad, "<｜end▁of▁sentence｜>") || member(config, "sp_model_kwargs") != "{}" || member(config, "unk_token") != "null")
            return false;
    }
    const std::array<std::string_view, 13> additional_special_tokens = {
        "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"};
    const std::array<std::pair<std::string_view, std::string_view>, 7> extra_expected = {{{"audio_bos_token", "<|audio_start|>"}, {"audio_eos_token", "<|audio_end|>"}, {"audio_token", "<|audio_pad|>"}, {"image_token", "<|image_pad|>"}, {"video_token", "<|video_pad|>"}, {"vision_bos_token", "<|vision_start|>"}, {"vision_eos_token", "<|vision_end|>"}}};
    if (new_family == Family::Qwen36 || new_family == Family::Qwen38)
    {
        const std::string_view extra_special_tokens = member(config, "additional_special_tokens");
        position = skip_space(extra_special_tokens, 0);
        if (position >= extra_special_tokens.size() || extra_special_tokens[position++] != '[')
            return false;
        for (const std::string_view expected : additional_special_tokens)
        {
            position = skip_space(extra_special_tokens, position);
            std::string value;
            if (!parse_json_string(extra_special_tokens, position, &value) || value != expected)
                return false;
            position = skip_space(extra_special_tokens, position);
            if (expected != additional_special_tokens.back())
            {
                if (position >= extra_special_tokens.size() || extra_special_tokens[position++] != ',')
                    return false;
            }
        }
        position = skip_space(extra_special_tokens, position);
        if (position >= extra_special_tokens.size() || extra_special_tokens[position++] != ']' || skip_space(extra_special_tokens, position) != extra_special_tokens.size())
            return false;
        const std::string_view extra_map = member(config, "extra_special_tokens");
        require_size(extra_map, 7);
        for (const auto& expected : extra_expected)
        {
            if (json_string(extra_map, expected.first) != std::optional<std::string>(expected.second))
                return false;
        }
    }

    const std::string_view model = member(tokenizer, "model");
    require_size(model, new_family == Family::DeepSeekV4 ? 9 : 10);
    const bool profile_ignore_merges = new_family == Family::GptOss;
    const bool null_affixes = new_family == Family::GptOss || new_family == Family::DeepSeekV4;
    const bool bad_ignore_merges = new_family == Family::DeepSeekV4
                                       ? find_manifest_member(model, "ignore_merges").has_value()
                                       : json_boolean(model, "ignore_merges") != std::optional<bool>(profile_ignore_merges);
    if (json_string(model, "type") != std::optional<std::string>("BPE") || member(model, "dropout") != "null" || member(model, "unk_token") != "null" || (null_affixes ? member(model, "continuing_subword_prefix") != "null" || member(model, "end_of_word_suffix") != "null" : json_string(model, "continuing_subword_prefix") != std::optional<std::string>("") || json_string(model, "end_of_word_suffix") != std::optional<std::string>("")) || json_boolean(model, "fuse_unk") != std::optional<bool>(false) || json_boolean(model, "byte_fallback") != std::optional<bool>(false) || bad_ignore_merges)
        return false;
    const std::string_view vocab_json = member(model, "vocab");
    const std::string_view merges_json = member(model, "merges");

    UVersionInfo unicode_version{};
    u_getUnicodeVersion(unicode_version);
    if (unicode_version[0] < 16)
        return false;
    std::unique_ptr<UnicodeProfile> unicode_profile(new UnicodeProfile);
    UErrorCode status = U_ZERO_ERROR;
    for (const std::string& regex : split_regexes)
    {
        UParseError parse_error{};
        const std::string& compile_regex = has_nfc ? icu_regex : regex;
        std::unique_ptr<const icu::RegexPattern> pattern(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(compile_regex), 0, parse_error, status));
        if (U_FAILURE(status) || !pattern)
            return false;
        unicode_profile->patterns.push_back(std::move(pattern));
        status = U_ZERO_ERROR;
    }
    if (has_nfc)
    {
        std::unique_ptr<icu::UnicodeSet> nfc_filter(new icu::UnicodeSet(icu::UnicodeString::fromUTF8("[\\p{Age=9.0}]"), status));
        if (U_FAILURE(status) || !nfc_filter)
            return false;
        nfc_filter->freeze();
        const icu::Normalizer2* base_nfc = icu::Normalizer2::getNFCInstance(status);
        if (U_FAILURE(status) || !base_nfc)
            return false;
        unicode_profile->nfc.reset(new icu::FilteredNormalizer2(*base_nfc, *nfc_filter));
        unicode_profile->filter = std::move(nfc_filter);
    }

    std::array<int16_t, 512> byte_map;
    byte_map.fill(-1);
    std::array<uint32_t, 256> byte_codepoint{};
    std::array<bool, 256> included{};
    for (int byte = 33; byte <= 126; ++byte)
    {
        byte_map[static_cast<size_t>(byte)] = static_cast<int16_t>(byte);
        byte_codepoint[static_cast<size_t>(byte)] = static_cast<uint32_t>(byte);
        included[static_cast<size_t>(byte)] = true;
    }
    for (int byte = 161; byte <= 172; ++byte)
    {
        byte_map[static_cast<size_t>(byte)] = static_cast<int16_t>(byte);
        byte_codepoint[static_cast<size_t>(byte)] = static_cast<uint32_t>(byte);
        included[static_cast<size_t>(byte)] = true;
    }
    for (int byte = 174; byte <= 255; ++byte)
    {
        byte_map[static_cast<size_t>(byte)] = static_cast<int16_t>(byte);
        byte_codepoint[static_cast<size_t>(byte)] = static_cast<uint32_t>(byte);
        included[static_cast<size_t>(byte)] = true;
    }
    uint32_t codepoint = 256;
    for (int byte = 0; byte < 256; ++byte)
    {
        if (!included[static_cast<size_t>(byte)])
        {
            byte_map[codepoint] = static_cast<int16_t>(byte);
            byte_codepoint[static_cast<size_t>(byte)] = codepoint++;
        }
    }
    auto utf8_codepoint = [](uint32_t value) {
        std::string text;
        if (value < 0x80)
            text.push_back(static_cast<char>(value));
        else if (value < 0x800)
        {
            text.push_back(static_cast<char>(0xc0 | (value >> 6)));
            text.push_back(static_cast<char>(0x80 | (value & 0x3f)));
        }
        else
        {
            text.push_back(static_cast<char>(0xe0 | (value >> 12)));
            text.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
            text.push_back(static_cast<char>(0x80 | (value & 0x3f)));
        }
        return text;
    };

    std::vector<Entry> new_index;
    std::vector<uint8_t> new_bytes;
    std::unordered_map<uint64_t, Merge> new_merges;
    std::unordered_map<std::string, uint32_t> vocab;
    std::unordered_map<uint32_t, std::string> added_by_id;
    std::unordered_map<std::string, uint32_t> added_by_content;
    std::unordered_map<uint32_t, std::pair<bool, bool>> added_flags_by_id;
    auto store = [&](uint32_t id, std::string_view token, bool is_special, bool raw_utf8 = false) {
        if (id > max_id)
            throw std::runtime_error("token ID exceeds supported direct index");
        if (token.size() > max_bytes || new_bytes.size() > max_bytes - token.size())
            throw std::runtime_error("tokenizer byte blob exceeds supported size");
        size_t needed = static_cast<size_t>(id) + 1;
        if (needed > new_index.size())
            new_index.resize(needed);
        Entry& entry = new_index[id];
        const uint32_t start = static_cast<uint32_t>(new_bytes.size());
        if (raw_utf8)
        {
            for (const char byte : token)
                new_bytes.push_back(static_cast<uint8_t>(static_cast<unsigned char>(byte)));
        }
        else
        {
            size_t offset = 0;
            while (offset < token.size())
            {
                const size_t begin = offset;
                const uint8_t first = static_cast<uint8_t>(static_cast<unsigned char>(token[offset++]));
                uint32_t cp = 0;
                size_t length = 1;
                if (first <= 0x7f)
                    cp = first;
                else if (first >= 0xc2 && first <= 0xdf)
                {
                    cp = first & 0x1f;
                    length = 2;
                }
                else if (first >= 0xe0 && first <= 0xef)
                {
                    cp = first & 0x0f;
                    length = 3;
                }
                else if (first >= 0xf0 && first <= 0xf4)
                {
                    cp = first & 7;
                    length = 4;
                }
                else
                    throw std::runtime_error("invalid UTF-8 token piece");
                if (length > 1)
                {
                    if (token.size() - begin < length)
                        throw std::runtime_error("truncated UTF-8 token piece");
                    for (size_t i = 1; i < length; ++i)
                    {
                        const uint8_t next = static_cast<uint8_t>(static_cast<unsigned char>(token[begin + i]));
                        if ((next & 0xc0) != 0x80 || (i == 1 && ((first == 0xe0 && next < 0xa0) || (first == 0xed && next >= 0xa0) || (first == 0xf0 && next < 0x90) || (first == 0xf4 && next > 0x8f))))
                            throw std::runtime_error("invalid UTF-8 token piece");
                        cp = (cp << 6) | (next & 0x3f);
                    }
                    offset = begin + length;
                }
                if (cp < byte_map.size() && byte_map[cp] >= 0)
                    new_bytes.push_back(static_cast<uint8_t>(byte_map[cp]));
                else
                {
                    for (size_t i = begin; i < offset; ++i)
                        new_bytes.push_back(static_cast<uint8_t>(static_cast<unsigned char>(token[i])));
                }
            }
        }
        entry.offset = start;
        entry.length = static_cast<uint32_t>(new_bytes.size() - start);
        entry.flags = static_cast<uint8_t>(Present | (is_special ? Special : 0));
    };

    if (vocab_json.empty() || vocab_json.front() != '{')
        return false;
    vocab.reserve(base_vocab_size);
    std::vector<bool> base_ids(base_vocab_size, false);
    position = skip_space(vocab_json, 0);
    if (position >= vocab_json.size() || vocab_json[position++] != '{')
        return false;
    std::string piece;
    for (;;)
    {
        position = skip_space(vocab_json, position);
        if (position >= vocab_json.size())
            return false;
        if (vocab_json[position] == '}')
        {
            ++position;
            if (skip_space(vocab_json, position) != vocab_json.size())
                return false;
            break;
        }
        if (!parse_json_string(vocab_json, position, &piece))
            return false;
        position = skip_space(vocab_json, position);
        if (position >= vocab_json.size() || vocab_json[position++] != ':')
            return false;
        position = skip_space(vocab_json, position);
        const size_t begin = position;
        while (position < vocab_json.size() && vocab_json[position] != ',' && vocab_json[position] != '}' && !is_space(vocab_json[position]))
            ++position;
        uint32_t id = 0;
        const auto parsed = std::from_chars(vocab_json.data() + begin, vocab_json.data() + position, id);
        if (parsed.ec != std::errc() || parsed.ptr != vocab_json.data() + position || id >= base_ids.size() || base_ids[id] || !vocab.emplace(piece, id).second)
            return false;
        base_ids[id] = true;
        store(id, piece, false);
        if (new_family == Family::GptOss)
        {
            const Entry& entry = new_index[id];
            const char* raw_data = entry.length == 0 ? "" : reinterpret_cast<const char*>(new_bytes.data() + entry.offset);
            if (!raw_piece_ids.emplace(std::string(raw_data, entry.length), id).second)
                return false;
        }
        position = skip_space(vocab_json, position);
        if (position >= vocab_json.size())
            return false;
        if (vocab_json[position] == '}')
        {
            ++position;
            if (skip_space(vocab_json, position) != vocab_json.size())
                return false;
            break;
        }
        if (vocab_json[position++] != ',')
            return false;
        position = skip_space(vocab_json, position);
        if (position >= vocab_json.size() || vocab_json[position] == '}')
            return false;
    }
    if (vocab.empty())
        return false;

    auto read_added = [&](uint32_t id, std::string_view item, bool id_in_object) {
        if (id >= total_vocab_size || ((new_family == Family::Qwen36 || new_family == Family::Qwen38) && (id < base_vocab_size || id > 248076)) || (new_family == Family::GptOss && (id < 199998 || id > 200018)))
            throw std::runtime_error("AddedToken ID outside the supported range");
        size_t count = 0;
        if (!object_size(item, count) || count != (id_in_object ? 7u : 6u))
            throw std::runtime_error("unsupported AddedToken metadata");
        const std::optional<std::string> content = json_string(item, "content");
        const std::optional<bool> single_word = json_boolean(item, "single_word");
        const std::optional<bool> lstrip = json_boolean(item, "lstrip");
        const std::optional<bool> rstrip = json_boolean(item, "rstrip");
        const std::optional<bool> normalized = json_boolean(item, "normalized");
        const std::optional<bool> is_special = json_boolean(item, "special");
        if (id_in_object && json_integer(item, "id") != std::optional<int64_t>(id))
            throw std::runtime_error("AddedToken ID mismatch");
        if (!content || content->empty() || !single_word || *single_word || !lstrip || *lstrip || !rstrip || *rstrip || !normalized || !is_special)
            throw std::runtime_error("unsupported AddedToken matching metadata");
        const auto base = vocab.find(*content);
        if (base != vocab.end() && base->second != id)
            throw std::runtime_error("AddedToken conflicts with base vocabulary");
        const auto old_id = added_by_content.find(*content);
        const auto old_content = added_by_id.find(id);
        if ((old_id != added_by_content.end() && old_id->second != id) || (old_content != added_by_id.end() && old_content->second != *content))
            throw std::runtime_error("conflicting AddedToken overlay");
        const auto old_flags = added_flags_by_id.find(id);
        if (old_flags != added_flags_by_id.end() && (old_flags->second.first != *is_special || old_flags->second.second != *normalized))
            throw std::runtime_error("conflicting AddedToken flags");
        const bool is_new_added_token = old_content == added_by_id.end();
        added_by_content[*content] = id;
        added_by_id[id] = *content;
        added_flags_by_id[id] = {*is_special, *normalized};
        store(id, *content, *is_special, true);
        if (is_new_added_token)
            added_tokens.push_back({*content, id});
    };

    const std::string_view added_json = member(tokenizer, "added_tokens");
    position = skip_space(added_json, 0);
    if (position >= added_json.size() || added_json[position++] != '[')
        return false;
    size_t added_count = 0;
    for (;;)
    {
        position = skip_space(added_json, position);
        if (position >= added_json.size())
            return false;
        if (added_json[position] == ']')
        {
            ++position;
            if (skip_space(added_json, position) != added_json.size())
                return false;
            break;
        }
        std::string_view item;
        if (!take_value(added_json, position, item) || item.empty() || item.front() != '{')
            return false;
        const std::optional<int64_t> id_value = json_integer(item, "id");
        if (!id_value || *id_value < 0 || static_cast<uint64_t>(*id_value) > max_id)
            return false;
        read_added(static_cast<uint32_t>(*id_value), item, true);
        ++added_count;
        position = skip_space(added_json, position);
        if (position >= added_json.size())
            return false;
        if (added_json[position] == ']')
        {
            ++position;
            if (skip_space(added_json, position) != added_json.size())
                return false;
            break;
        }
        if (added_json[position++] != ',')
            return false;
        position = skip_space(added_json, position);
        if (position >= added_json.size() || added_json[position] == ']')
            return false;
    }
    const size_t expected_tokenizer_added_count = new_family == Family::Qwen36 ? 26 : new_family == Family::Qwen38 ? 33
                                                                                  : new_family == Family::GptOss   ? 21
                                                                                                                   : 1283;
    if (added_count != expected_tokenizer_added_count)
        return false;

    size_t config_added_count = 0;
    if (new_family != Family::DeepSeekV4)
    {
        const std::string_view config_added = member(config, "added_tokens_decoder");
        const size_t expected_config_added_count = new_family == Family::Qwen36 || new_family == Family::Qwen38 ? 33 : 21;
        require_size(config_added, expected_config_added_count);
        position = skip_space(config_added, 0);
        if (position >= config_added.size() || config_added[position++] != '{')
            return false;
        for (;;)
        {
            position = skip_space(config_added, position);
            if (position >= config_added.size())
                return false;
            if (config_added[position] == '}')
            {
                ++position;
                if (skip_space(config_added, position) != config_added.size())
                    return false;
                break;
            }
            std::string key;
            if (!parse_json_string(config_added, position, &key))
                return false;
            position = skip_space(config_added, position);
            if (position >= config_added.size() || config_added[position++] != ':')
                return false;
            uint32_t id = 0;
            const auto parsed = std::from_chars(key.data(), key.data() + key.size(), id);
            if (parsed.ec != std::errc() || parsed.ptr != key.data() + key.size() || id > max_id)
                return false;
            std::string_view item;
            if (!take_value(config_added, position, item) || item.empty() || item.front() != '{')
                return false;
            const bool has_id = find_manifest_member(item, "id").has_value();
            read_added(id, item, has_id);
            ++config_added_count;
            position = skip_space(config_added, position);
            if (position >= config_added.size())
                return false;
            if (config_added[position] == '}')
            {
                ++position;
                if (skip_space(config_added, position) != config_added.size())
                    return false;
                break;
            }
            if (config_added[position++] != ',')
                return false;
            position = skip_space(config_added, position);
            if (position >= config_added.size() || config_added[position] == '}')
                return false;
        }
    }
    const size_t expected_union_added_count = new_family == Family::Qwen36 || new_family == Family::Qwen38 ? 33 : expected_tokenizer_added_count;
    if ((new_family != Family::DeepSeekV4 && config_added_count != (new_family == Family::GptOss ? 21u : 33u)) || added_by_id.size() != expected_union_added_count)
        return false;
    if (vocabulary_size != 0 && vocabulary_size != total_vocab_size)
        return false;

    for (size_t byte = 0; byte < byte_ids.size(); ++byte)
    {
        const std::string symbol = utf8_codepoint(byte_codepoint[byte]);
        const auto found = vocab.find(symbol);
        if (found == vocab.end())
            return false;
        byte_ids[byte] = static_cast<int32_t>(found->second);
    }

    position = skip_space(merges_json, 0);
    if (position >= merges_json.size() || merges_json[position++] != '[')
        return false;
    new_merges.reserve(base_vocab_size);
    size_t merge_rank = 0;
    for (;;)
    {
        position = skip_space(merges_json, position);
        if (position >= merges_json.size())
            return false;
        if (merges_json[position] == ']')
        {
            ++position;
            if (skip_space(merges_json, position) != merges_json.size())
                return false;
            break;
        }
        position = skip_space(merges_json, position);
        std::string left_piece;
        std::string right_piece;
        if (new_family == Family::GptOss)
        {
            if (position >= merges_json.size() || merges_json[position++] != '[')
                return false;
            position = skip_space(merges_json, position);
            if (!parse_json_string(merges_json, position, &left_piece))
                return false;
            position = skip_space(merges_json, position);
            if (position >= merges_json.size() || merges_json[position++] != ',')
                return false;
            position = skip_space(merges_json, position);
            if (!parse_json_string(merges_json, position, &right_piece))
                return false;
            position = skip_space(merges_json, position);
            if (position >= merges_json.size() || merges_json[position++] != ']')
                return false;
        }
        else
        {
            std::string merge_text;
            if (!parse_json_string(merges_json, position, &merge_text))
                return false;
            const size_t separator = merge_text.find(' ');
            if (separator == std::string::npos || separator == 0 || separator + 1 == merge_text.size())
                return false;
            left_piece = merge_text.substr(0, separator);
            right_piece = merge_text.substr(separator + 1);
        }
        if (left_piece.empty() || right_piece.empty())
            return false;
        const auto left = vocab.find(left_piece);
        const auto right = vocab.find(right_piece);
        const auto merged = vocab.find(left_piece + right_piece);
        if (left == vocab.end() || right == vocab.end() || merged == vocab.end() || left->second >= base_vocab_size || right->second >= base_vocab_size || merged->second >= base_vocab_size || merge_rank > std::numeric_limits<uint32_t>::max())
            return false;
        const uint64_t key = merge_key(left->second, right->second);
        if (!new_merges.emplace(key, Merge{static_cast<uint32_t>(merge_rank), merged->second}).second)
            return false;
        ++merge_rank;
        position = skip_space(merges_json, position);
        if (position >= merges_json.size())
            return false;
        if (merges_json[position] == ']')
        {
            ++position;
            if (skip_space(merges_json, position) != merges_json.size())
                return false;
            break;
        }
        if (merges_json[position++] != ',')
            return false;
        position = skip_space(merges_json, position);
        if (position >= merges_json.size() || merges_json[position] == ']')
            return false;
    }
    if (new_merges.size() != merge_rank)
        return false;

    if (new_family == Family::Qwen36 || new_family == Family::Qwen38)
    {
        const std::string_view extra = member(config, "extra_special_tokens");
        for (const auto& expected : extra_expected)
        {
            const std::optional<std::string> content = json_string(extra, expected.first);
            if (!content)
                return false;
            const auto found = added_by_content.find(*content);
            if (*content != expected.second || found == added_by_content.end() || (new_index[found->second].flags & Special) == 0)
                return false;
        }
        for (const std::string_view content : additional_special_tokens)
        {
            const auto found = added_by_content.find(std::string(content));
            if (found == added_by_content.end() || (new_index[found->second].flags & Special) == 0)
                return false;
        }
    }
    if (new_family == Family::GptOss)
    {
        static const std::array<std::string_view, 21> expected_contents = {
            "<|startoftext|>", "<|endoftext|>", "<|reserved_200000|>", "<|reserved_200001|>", "<|return|>",
            "<|constrain|>", "<|reserved_200004|>", "<|channel|>", "<|start|>", "<|end|>", "<|message|>",
            "<|reserved_200009|>", "<|reserved_200010|>", "<|reserved_200011|>", "<|call|>",
            "<|reserved_200013|>", "<|reserved_200014|>", "<|reserved_200015|>", "<|reserved_200016|>",
            "<|reserved_200017|>", "<|endofprompt|>"};
        for (size_t i = 0; i < expected_contents.size(); ++i)
        {
            const uint32_t id = static_cast<uint32_t>(199998 + i);
            const auto found = added_by_id.find(id);
            if (found == added_by_id.end() || found->second != expected_contents[i] || (new_index[id].flags & Special) == 0)
                return false;
        }
    }
    else if (new_family == Family::DeepSeekV4)
    {
        static const std::array<std::pair<uint32_t, std::string_view>, 10> expected_deepseek_tokens = {{{0, "<｜begin▁of▁sentence｜>"}, {1, "<｜end▁of▁sentence｜>"}, {2, "<｜▁pad▁｜>"}, {128803, "<｜User｜>"}, {128804, "<｜Assistant｜>"}, {128821, "<think>"}, {128822, "</think>"}, {128825, "｜DSML｜"}, {128828, "<｜latest_reminder｜>"}, {129279, "<｜image｜>"}}};
        for (const auto& expected : expected_deepseek_tokens)
        {
            const auto found = added_by_id.find(expected.first);
            if (found == added_by_id.end() || found->second != expected.second)
                return false;
            const bool expected_special = expected.first <= 2 || expected.first == 129279;
            if (((new_index[expected.first].flags & Special) != 0) != expected_special)
                return false;
        }
    }
    else
    {
        static const std::array<std::string_view, 33> expected_qwen_tokens = {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>",
            "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>",
            "</tool_response>", "<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>",
            "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>"};
        for (size_t i = 0; i < expected_qwen_tokens.size(); ++i)
        {
            const uint32_t id = static_cast<uint32_t>(248044 + i);
            const auto found = added_by_id.find(id);
            const bool expected_special = i < 14 || i >= 26;
            if (found == added_by_id.end() || found->second != expected_qwen_tokens[i] || (((new_index[id].flags & Special) != 0) != expected_special))
                return false;
        }
    }

    std::vector<int32_t> new_stops;
    if (new_family == Family::GptOss)
    {
        for (const std::string_view spelling : {"<|return|>", "<|call|>"})
        {
            const auto found = added_by_content.find(std::string(spelling));
            if (found == added_by_content.end() || (new_index[found->second].flags & Special) == 0)
                return false;
            new_stops.push_back(static_cast<int32_t>(found->second));
        }
        // Harmony reserves ID 201088, outside the model's 0..201087 logits vocabulary.
        for (uint32_t id = 200018; id <= total_vocab_size; ++id)
        {
            const std::string spelling = "<|reserved_" + std::to_string(id) + "|>";
            store(id, spelling, true, true);
            added_tokens.push_back({spelling, id});
        }
        added_tokens.erase(std::remove_if(added_tokens.begin(), added_tokens.end(), [](const AddedToken& token) {
                               return token.content == "<|endofprompt|>";
                           }),
                           added_tokens.end());
    }
    else if (new_family == Family::DeepSeekV4)
    {
        const auto eos = added_by_content.find("<｜end▁of▁sentence｜>");
        if (eos == added_by_content.end() || eos->second != 1 || (new_index[1].flags & Special) == 0)
            return false;
        new_stops.push_back(1);
    }
    else
    {
        const auto eos = added_by_content.find("<|im_end|>");
        if (eos == added_by_content.end() || eos->second != 248046 || (new_index[eos->second].flags & Special) == 0)
            return false;
        new_stops.push_back(static_cast<int32_t>(eos->second));
    }
    const std::filesystem::path generation_path = std::filesystem::path(model_directory) / "generation_config.json";
    std::error_code generation_error;
    const bool has_generation_config = std::filesystem::exists(generation_path, generation_error);
    if (generation_error)
        return false;
    if (new_family == Family::Qwen36 || new_family == Family::Qwen38)
    {
        if (has_generation_config)
        {
            std::ifstream generation_input(generation_path, std::ios::binary);
            if (!generation_input)
                return false;
            const std::string generation_config((std::istreambuf_iterator<char>(generation_input)),
                                                std::istreambuf_iterator<char>());
            if (generation_input.bad())
                return false;
            size_t generation_member_count = 0;
            if (!object_size(generation_config, generation_member_count))
                return false;
            (void)generation_member_count;
            const std::optional<std::string_view> eos_config = find_manifest_member(std::string_view(generation_config), "eos_token_id");
            if (!eos_config)
                return false;
            std::vector<uint32_t> ids;
            position = skip_space(*eos_config, 0);
            if (position < eos_config->size() && (*eos_config)[position] == '[')
            {
                ++position;
                for (;;)
                {
                    position = skip_space(*eos_config, position);
                    if (position >= eos_config->size() || (*eos_config)[position] == ']')
                        return false;
                    const size_t begin = position;
                    while (position < eos_config->size() && (*eos_config)[position] != ',' && (*eos_config)[position] != ']' && !is_space((*eos_config)[position]))
                        ++position;
                    uint32_t token_id = 0;
                    const auto parsed = std::from_chars(eos_config->data() + begin,
                                                        eos_config->data() + position, token_id);
                    if (parsed.ec != std::errc() || parsed.ptr != eos_config->data() + position)
                        return false;
                    ids.push_back(token_id);
                    position = skip_space(*eos_config, position);
                    if (position >= eos_config->size())
                        return false;
                    if ((*eos_config)[position] == ']')
                    {
                        ++position;
                        if (skip_space(*eos_config, position) != eos_config->size())
                            return false;
                        break;
                    }
                    if ((*eos_config)[position++] != ',')
                        return false;
                    position = skip_space(*eos_config, position);
                    if (position >= eos_config->size() || (*eos_config)[position] == ']')
                        return false;
                }
            }
            else
            {
                const size_t begin = position;
                while (position < eos_config->size() && !is_space((*eos_config)[position]))
                    ++position;
                uint32_t token_id = 0;
                const auto parsed = std::from_chars(eos_config->data() + begin,
                                                    eos_config->data() + position, token_id);
                if (parsed.ec != std::errc() || parsed.ptr != eos_config->data() + position || skip_space(*eos_config, position) != eos_config->size())
                    return false;
                ids.push_back(token_id);
            }
            if (ids.empty())
                return false;
            if (ids != std::vector<uint32_t>{248046, 248044})
                return false;
            for (const uint32_t token_id : ids)
            {
                if (token_id >= total_vocab_size || static_cast<size_t>(token_id) >= new_index.size() || (new_index[static_cast<size_t>(token_id)].flags & Present) == 0)
                    return false;
                const int32_t id = static_cast<int32_t>(token_id);
                if (std::find(new_stops.begin(), new_stops.end(), id) == new_stops.end())
                    new_stops.push_back(id);
            }
        }
    }

    std::sort(added_tokens.begin(), added_tokens.end(), [](const AddedToken& left, const AddedToken& right) {
        const unsigned char left_first = static_cast<unsigned char>(left.content.front());
        const unsigned char right_first = static_cast<unsigned char>(right.content.front());
        if (left_first != right_first)
            return left_first < right_first;
        if (left.content.size() != right.content.size())
            return left.content.size() > right.content.size();
        return left.content < right.content;
    });
    added_tokens.erase(std::unique(added_tokens.begin(), added_tokens.end(), [](const AddedToken& left, const AddedToken& right) {
                           return left.id == right.id && left.content == right.content;
                       }),
                       added_tokens.end());
    size_t added_token_index = 0;
    for (size_t first_byte = 0; first_byte < added_token_ranges.size(); ++first_byte)
    {
        auto& range = added_token_ranges[first_byte];
        range.first = added_token_index;
        while (added_token_index < added_tokens.size() && static_cast<unsigned char>(added_tokens[added_token_index].content.front()) == first_byte)
            ++added_token_index;
        range.second = added_token_index;
    }
    if (added_token_index != added_tokens.size())
        return false;

    index.swap(new_index);
    bytes.swap(new_bytes);
    merges.swap(new_merges);
    stops.swap(new_stops);
    family = new_family;
    ignore_merges = profile_ignore_merges;
    unicode = std::move(unicode_profile);
    return true;
#endif
}

void Tokenizer::bpe(std::string_view raw,
                    std::vector<int32_t>& ids,
                    EncodeScratch& scratch) const
{
    scratch.candidates.clear();
    const size_t count = raw.size();
    if (count == 0)
        return;

    if (ignore_merges)
    {
        const auto direct = raw_piece_ids.find(std::string(raw));
        if (direct != raw_piece_ids.end())
        {
            ids.push_back(static_cast<int32_t>(direct->second));
            return;
        }
    }

    using Candidate = EncodeScratch::Candidate;
    if (scratch.candidates.capacity() < count)
        scratch.candidates.reserve(count);
    std::vector<EncodeScratch::Node>& nodes = scratch.nodes;
    nodes.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        nodes[i] = {byte_ids[static_cast<uint8_t>(raw[i])], i == 0 ? count : i - 1,
                    i + 1 == count ? count : i + 1};
    }

    auto add = [&](size_t left) {
        if (left >= count || nodes[left].id < 0)
            return;
        const size_t right = nodes[left].next;
        if (right == count || nodes[right].id < 0)
            return;
        const uint64_t key = merge_key(static_cast<uint32_t>(nodes[left].id), static_cast<uint32_t>(nodes[right].id));
        const auto merge = merges.find(key);
        if (merge != merges.end())
        {
            scratch.candidates.emplace_back(merge->second.rank, left, right,
                                            nodes[left].id, nodes[right].id,
                                            static_cast<int32_t>(merge->second.id));
            std::push_heap(scratch.candidates.begin(), scratch.candidates.end(),
                           std::greater<Candidate>{});
        }
    };
    for (size_t i = 0; i + 1 < count; ++i)
        add(i);

    while (!scratch.candidates.empty())
    {
        std::pop_heap(scratch.candidates.begin(), scratch.candidates.end(),
                      std::greater<Candidate>{});
        const Candidate candidate = scratch.candidates.back();
        scratch.candidates.pop_back();
        const size_t left = std::get<1>(candidate);
        const size_t right = std::get<2>(candidate);
        if (nodes[left].id < 0 || nodes[right].id < 0 || nodes[left].next != right || nodes[left].id != std::get<3>(candidate) || nodes[right].id != std::get<4>(candidate))
            continue;

        nodes[left].id = std::get<5>(candidate);
        nodes[right].id = -1;
        nodes[left].next = nodes[right].next;
        if (nodes[left].next != count)
            nodes[nodes[left].next].prev = left;
        if (nodes[left].prev != count)
            add(nodes[left].prev);
        add(left);
    }

    for (size_t i = 0; i != count; i = nodes[i].next)
        ids.push_back(nodes[i].id);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const
{
    EncodeScratch scratch;
    std::vector<int32_t> ids;
    encode_impl(text, true, scratch, ids);
    return ids;
}

void Tokenizer::encode_impl(std::string_view text,
                            bool recognize_added_tokens,
                            EncodeScratch& scratch,
                            std::vector<int32_t>& ids) const
{
#if !defined(NCNN_MOE_TOKENIZER_ICU)
    (void)text;
    (void)recognize_added_tokens;
    (void)scratch;
    (void)ids;
    throw std::runtime_error("tokenizer requires the ICU Runtime dependency");
#else
    if (index.empty() || !unicode)
        throw std::runtime_error("tokenizer is not loaded");
    if (text.empty())
        return;

    constexpr size_t cache_entries = 256;
    constexpr size_t cache_chunk_bytes = 256;
    constexpr size_t cache_limit = 64 * 1024;
    std::unordered_map<std::string, std::vector<int32_t>> cache;
    size_t cache_size = 0;

    auto encode_range = [&](const icu::UnicodeString& piece, int32_t start, int32_t end) {
        if (end <= start)
            return;
        std::string& chunk = scratch.chunk;
        chunk.clear();
        piece.tempSubStringBetween(start, end).toUTF8String(chunk);
        if (chunk.empty())
            return;
        if (chunk.size() <= cache_chunk_bytes)
        {
            const auto found = cache.find(chunk);
            if (found != cache.end())
            {
                ids.insert(ids.end(), found->second.begin(), found->second.end());
                return;
            }
        }
        const size_t first_id = ids.size();
        bpe(chunk, ids, scratch);
        const size_t id_count = ids.size() - first_id;
        if (chunk.size() <= cache_chunk_bytes && cache.size() < cache_entries && id_count <= (cache_limit - chunk.size()) / sizeof(int32_t))
        {
            const size_t payload = chunk.size() + id_count * sizeof(int32_t);
            if (payload <= cache_limit - cache_size)
            {
                cache.emplace(chunk, std::vector<int32_t>(ids.begin() + first_id, ids.end()));
                cache_size += payload;
            }
        }
    };

    auto encode_segment = [&](std::string_view segment) {
        if (segment.empty())
            return;
        icu::UnicodeString input = icu::UnicodeString::fromUTF8(segment);
        if (unicode->nfc)
        {
            UErrorCode status = U_ZERO_ERROR;
            icu::UnicodeString normalized;
            unicode->nfc->normalize(input, normalized, status);
            if (U_FAILURE(status))
                throw std::runtime_error("tokenizer NFC normalization failed");
            input = std::move(normalized);
        }

        std::vector<icu::UnicodeString>& pieces = scratch.pieces;
        std::vector<icu::UnicodeString>& split = scratch.split;
        pieces.clear();
        split.clear();
        pieces.emplace_back(std::move(input));
        for (size_t pattern_index = 0; pattern_index < unicode->patterns.size(); ++pattern_index)
        {
            const auto& pattern = unicode->patterns[pattern_index];
            const bool final_stage = pattern_index + 1 == unicode->patterns.size();
            for (const icu::UnicodeString& piece : pieces)
            {
                auto emit_range = [&](int32_t start, int32_t end) {
                    if (end <= start)
                        return;
                    if (final_stage)
                        encode_range(piece, start, end);
                    else
                        split.emplace_back(piece, start, end - start);
                };
                UErrorCode status = U_ZERO_ERROR;
                std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(piece, status));
                if (U_FAILURE(status) || !matcher)
                    throw std::runtime_error("could not create tokenizer regex matcher");
                int32_t cursor = 0;
                while (matcher->find(status))
                {
                    const int32_t start = matcher->start(status);
                    const int32_t end = matcher->end(status);
                    if (U_FAILURE(status) || start < cursor || end <= start)
                        throw std::runtime_error("tokenizer regex returned invalid span");
                    if (start > cursor)
                        emit_range(cursor, start);
                    emit_range(start, end);
                    cursor = end;
                }
                if (U_FAILURE(status))
                    throw std::runtime_error("tokenizer regex matching failed");
                if (cursor < piece.length())
                    emit_range(cursor, piece.length());
            }
            if (final_stage)
                return;
            pieces.swap(split);
            split.clear();
        }
        for (const icu::UnicodeString& piece : pieces)
            encode_range(piece, 0, piece.length());
    };

    if (!recognize_added_tokens || added_tokens.empty())
    {
        encode_segment(text);
        return;
    }

    size_t cursor = 0;
    while (cursor < text.size())
    {
        size_t best_pos = std::string_view::npos;
        size_t best_length = 0;
        const AddedToken* best_token = nullptr;
        for (size_t position = cursor; position < text.size(); ++position)
        {
            const auto& range = added_token_ranges[static_cast<unsigned char>(text[position])];
            for (size_t token_index = range.first; token_index < range.second; ++token_index)
            {
                const AddedToken& token = added_tokens[token_index];
                const size_t token_length = token.content.size();
                if (token_length > text.size() - position || text.compare(position, token_length, token.content.data(), token_length) != 0)
                    continue;
                best_pos = position;
                best_length = token_length;
                best_token = &token;
                break;
            }
            if (best_token)
                break;
        }
        if (best_pos == std::string_view::npos)
        {
            encode_segment(text.substr(cursor));
            break;
        }
        if (best_pos > cursor)
            encode_segment(text.substr(cursor, best_pos - cursor));
        ids.push_back(static_cast<int32_t>(best_token->id));
        cursor = best_pos + best_length;
    }
#endif
}

std::vector<int32_t> Tokenizer::apply_chat(std::string_view messages_json, bool enable_thinking) const
{
#if !defined(NCNN_MOE_TOKENIZER_ICU)
    (void)messages_json;
    (void)enable_thinking;
    throw std::runtime_error("tokenizer requires the ICU Runtime dependency");
#else
    if (index.empty() || !unicode)
        throw std::runtime_error("tokenizer is not loaded");
    EncodeScratch scratch;
    struct Message
    {
        std::string role;
        std::string content;
    };
    std::vector<Message> messages;
    size_t position = skip_space(messages_json, 0);
    if (position >= messages_json.size() || messages_json[position++] != '[')
        throw std::invalid_argument("messages must be a JSON array");
    for (;;)
    {
        position = skip_space(messages_json, position);
        if (position >= messages_json.size())
            throw std::invalid_argument("invalid messages array");
        if (messages_json[position] == ']')
        {
            ++position;
            if (skip_space(messages_json, position) != messages_json.size())
                throw std::invalid_argument("trailing data after messages array");
            break;
        }
        if (messages_json[position++] != '{')
            throw std::invalid_argument("each message must be an object");
        Message message;
        bool has_role = false;
        bool has_content = false;
        for (;;)
        {
            position = skip_space(messages_json, position);
            if (position >= messages_json.size())
                throw std::invalid_argument("invalid message object");
            if (messages_json[position] == '}')
            {
                ++position;
                break;
            }
            std::string key;
            if (!parse_json_string(messages_json, position, &key))
                throw std::invalid_argument("invalid message member name");
            position = skip_space(messages_json, position);
            if (position >= messages_json.size() || messages_json[position++] != ':')
                throw std::invalid_argument("invalid message member");
            position = skip_space(messages_json, position);
            if (key != "role" && key != "content")
                throw std::invalid_argument("message fields are limited to role and content strings");
            if (position >= messages_json.size() || messages_json[position] != '"')
                throw std::invalid_argument("message role and content must be strings");
            std::string value;
            if (!parse_json_string(messages_json, position, &value))
                throw std::invalid_argument("invalid message string");
            if (key == "role")
            {
                if (has_role)
                    throw std::invalid_argument("duplicate message role");
                has_role = true;
                message.role = std::move(value);
            }
            else
            {
                if (has_content)
                    throw std::invalid_argument("duplicate message content");
                has_content = true;
                message.content = std::move(value);
            }
            position = skip_space(messages_json, position);
            if (position >= messages_json.size())
                throw std::invalid_argument("invalid message object");
            if (messages_json[position] == '}')
            {
                ++position;
                break;
            }
            if (messages_json[position++] != ',')
                throw std::invalid_argument("invalid message object delimiter");
            position = skip_space(messages_json, position);
            if (position >= messages_json.size() || messages_json[position] == '}')
                throw std::invalid_argument("invalid message object delimiter");
        }
        if (!has_role || !has_content)
            throw std::invalid_argument("each message needs role and content strings");
        messages.push_back(std::move(message));
        position = skip_space(messages_json, position);
        if (position >= messages_json.size())
            throw std::invalid_argument("invalid messages array");
        if (messages_json[position] == ']')
        {
            ++position;
            if (skip_space(messages_json, position) != messages_json.size())
                throw std::invalid_argument("trailing data after messages array");
            break;
        }
        if (messages_json[position++] != ',')
            throw std::invalid_argument("invalid messages array delimiter");
        position = skip_space(messages_json, position);
        if (position >= messages_json.size() || messages_json[position] == ']')
            throw std::invalid_argument("invalid messages array delimiter");
    }

    if (family == Family::GptOss)
    {
        std::vector<int32_t> prompt;
        const auto append_raw = [&](std::string_view text) {
            encode_impl(text, false, scratch, prompt);
        };
        static constexpr std::string_view default_system = "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.";
        for (const Message& message : messages)
        {
            std::string role = message.role;
            for (char& character : role)
            {
                if (character >= 'A' && character <= 'Z')
                    character = static_cast<char>(character - 'A' + 'a');
            }
            if (role == "tool")
                throw std::invalid_argument("Tools should have a name!");
            if (role != "system" && role != "developer" && role != "user" && role != "assistant")
                throw std::invalid_argument("Unknown Harmony role");
            prompt.push_back(200006);
            append_raw(role);
            prompt.push_back(200008);
            append_raw(role == "system" && message.content.empty() ? default_system : std::string_view(message.content));
            prompt.push_back(200007);
        }
        prompt.push_back(200006);
        append_raw("assistant");
        return prompt;
    }

    if (family == Family::DeepSeekV4)
    {
        std::vector<Message> normalized;
        for (const Message& message : messages)
        {
            if (message.role != "system" && message.role != "developer" && message.role != "user" && message.role != "assistant" && message.role != "tool")
                throw std::invalid_argument("Unknown DeepSeek-V4 role");
            if (message.role == "tool")
            {
                const std::string tool_result = "<tool_result>" + message.content + "</tool_result>";
                if (!normalized.empty() && normalized.back().role == "user")
                    normalized.back().content += "\n\n" + tool_result;
                else
                    normalized.push_back({"user", tool_result});
            }
            else if (message.role == "user" && !normalized.empty() && normalized.back().role == "user")
            {
                normalized.back().content += "\n\n" + message.content;
            }
            else
                normalized.push_back({message.role, message.content});
        }
        int32_t last_user_index = -1;
        for (size_t i = 0; i < normalized.size(); ++i)
        {
            if (normalized[i].role == "user")
                last_user_index = static_cast<int32_t>(i);
        }
        const bool thinking = enable_thinking;
        if (thinking && last_user_index >= 0)
        {
            std::vector<Message> kept;
            kept.reserve(normalized.size());
            for (size_t i = 0; i < normalized.size(); ++i)
            {
                if (normalized[i].role == "developer" && static_cast<int32_t>(i) < last_user_index)
                    continue;
                kept.push_back(std::move(normalized[i]));
            }
            normalized.swap(kept);
            last_user_index = -1;
            for (size_t i = 0; i < normalized.size(); ++i)
            {
                if (normalized[i].role == "user")
                    last_user_index = static_cast<int32_t>(i);
            }
        }

        std::string prompt = "<｜begin▁of▁sentence｜>";
        for (size_t i = 0; i < normalized.size(); ++i)
        {
            const Message& message = normalized[i];
            if (message.role == "system")
                prompt += message.content;
            else if (message.role == "developer")
            {
                if (message.content.empty())
                    throw std::invalid_argument("Invalid message for role developer");
                prompt += "<｜User｜>" + message.content;
            }
            else if (message.role == "user")
                prompt += "<｜User｜>" + message.content;
            else if (message.role == "assistant")
            {
                if (thinking && static_cast<int32_t>(i) > last_user_index)
                    prompt += "</think>";
                prompt += message.content;
                prompt += "<｜end▁of▁sentence｜>";
            }
            const bool next_is_assistant = i + 1 < normalized.size() && normalized[i + 1].role == "assistant";
            if (i + 1 < normalized.size() && !next_is_assistant)
                continue;
            if (message.role == "user" || message.role == "developer")
            {
                prompt += "<｜Assistant｜>";
                if (thinking && static_cast<int32_t>(i) >= last_user_index)
                    prompt += "<think>";
                else
                    prompt += "</think>";
            }
        }
        std::vector<int32_t> ids;
        encode_impl(prompt, true, scratch, ids);
        return ids;
    }

    if (messages.empty())
        throw std::invalid_argument("No messages provided.");
    auto python_space = [](UChar32 cp) {
        return u_isUWhiteSpace(cp) || (cp >= 0x1c && cp <= 0x1f);
    };
    auto trim = [&](const std::string& text) {
        const icu::UnicodeString value = icu::UnicodeString::fromUTF8(text);
        int32_t begin = 0;
        int32_t end = value.length();
        while (begin < end && python_space(value.char32At(begin)))
            begin = value.moveIndex32(begin, 1);
        while (end > begin && python_space(value.char32At(value.moveIndex32(end, -1))))
            end = value.moveIndex32(end, -1);
        std::string result;
        value.tempSubStringBetween(begin, end).toUTF8String(result);
        return result;
    };

    static constexpr std::string_view qwen38_reasoning_instruction = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    std::string prompt;
    if (family == Family::Qwen38)
    {
        const std::string instruction = enable_thinking ? std::string(qwen38_reasoning_instruction) : std::string();
        if (messages.front().role == "system")
        {
            const std::string content = trim(messages.front().content);
            if (!content.empty())
            {
                prompt += "<|im_start|>system\n";
                if (!instruction.empty())
                    prompt += instruction + "\n\n";
                prompt += content + "<|im_end|>\n";
            }
            else if (!instruction.empty())
                prompt += "<|im_start|>system\n" + instruction + "<|im_end|>\n";
        }
        else if (!instruction.empty())
            prompt += "<|im_start|>system\n" + instruction + "<|im_end|>\n";
    }
    else if (messages.front().role == "system")
    {
        prompt += "<|im_start|>system\n";
        prompt += trim(messages.front().content);
        prompt += "<|im_end|>\n";
    }

    int32_t last_user_index = -1;
    for (size_t i = 0; i < messages.size(); ++i)
    {
        if (messages[i].role != "user")
            continue;
        const std::string content = trim(messages[i].content);
        if (!(content.rfind("<tool_response>", 0) == 0 && content.size() >= 16 && content.compare(content.size() - 16, 16, "</tool_response>") == 0))
            last_user_index = static_cast<int32_t>(i);
    }
    if (last_user_index < 0)
        throw std::invalid_argument("No user query found in messages.");

    for (size_t i = 0; i < messages.size(); ++i)
    {
        const Message& message = messages[i];
        const std::string content = trim(message.content);
        if (message.role == "system")
        {
            if (i != 0)
                throw std::invalid_argument("System message must be at the beginning.");
        }
        else if (message.role == "user")
        {
            prompt += "<|im_start|>user\n" + content + "<|im_end|>\n";
        }
        else if (message.role == "assistant")
        {
            std::string reasoning;
            std::string body = content;
            if (family == Family::Qwen36)
            {
                const size_t first_close = body.find("</think>");
                if (first_close != std::string::npos)
                {
                    reasoning = body.substr(0, first_close);
                    while (!reasoning.empty() && reasoning.back() == '\n')
                        reasoning.pop_back();
                    const size_t open = reasoning.rfind("<think>");
                    if (open != std::string::npos)
                        reasoning.erase(0, open + 7);
                    while (!reasoning.empty() && reasoning.front() == '\n')
                        reasoning.erase(reasoning.begin());
                    const size_t last_close = body.rfind("</think>");
                    body.erase(0, last_close + 8);
                    while (!body.empty() && body.front() == '\n')
                        body.erase(body.begin());
                    reasoning = trim(reasoning);
                }
            }
            prompt += "<|im_start|>assistant\n";
            if (family == Family::Qwen38)
                prompt += "<think>\n" + reasoning + "\n</think>\n\n" + body;
            else if (static_cast<int32_t>(i) > last_user_index)
                prompt += "<think>\n" + reasoning + "\n</think>\n\n" + body;
            else
                prompt += body;
            prompt += "<|im_end|>\n";
        }
        else if (message.role == "tool")
        {
            if (i > 0 && messages[i - 1].role != "tool")
                prompt += "<|im_start|>user";
            prompt += "\n<tool_response>\n" + content + "\n</tool_response>";
            if ((i + 1 < messages.size() && messages[i + 1].role != "tool") || i + 1 == messages.size())
                prompt += "<|im_end|>\n";
        }
        else
            throw std::invalid_argument("Unexpected message role.");
    }

    prompt += "<|im_start|>assistant\n";
    prompt += enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    std::vector<int32_t> ids;
    encode_impl(prompt, true, scratch, ids);
    return ids;
#endif
}

std::string Tokenizer::decode(int32_t id, std::string& pending, bool final) const
{
#if !defined(NCNN_MOE_TOKENIZER_ICU)
    (void)id;
    (void)pending;
    (void) final;
    throw std::runtime_error("tokenizer requires the ICU Runtime dependency");
#else
    if (index.empty() || !unicode)
        throw std::runtime_error("tokenizer is not loaded");
    if (id >= 0 && static_cast<size_t>(id) < index.size())
    {
        const Entry& entry = index[static_cast<size_t>(id)];
        const bool skip_special = family == Family::Qwen36 || family == Family::Qwen38;
        if ((entry.flags & Present) != 0 && (!skip_special || (entry.flags & Special) == 0) && entry.length != 0)
            pending.append(reinterpret_cast<const char*>(bytes.data() + entry.offset), entry.length);
    }

    std::string output;
    size_t offset = 0;
    const auto byte = [&](size_t index_value) {
        return static_cast<uint8_t>(static_cast<unsigned char>(pending[index_value]));
    };
    const auto continuation = [](uint8_t value) { return (value & 0xc0) == 0x80; };
    const auto replacement = [&] { output.append("\xef\xbf\xbd", 3); };
    while (offset < pending.size())
    {
        const uint8_t lead = byte(offset);
        if (lead <= 0x7f)
        {
            output.push_back(static_cast<char>(lead));
            ++offset;
            continue;
        }
        size_t need = 0;
        if (lead >= 0xc2 && lead <= 0xdf)
            need = 2;
        else if (lead >= 0xe0 && lead <= 0xef)
            need = 3;
        else if (lead >= 0xf0 && lead <= 0xf4)
            need = 4;
        else
        {
            replacement();
            ++offset;
            continue;
        }
        const size_t available = pending.size() - offset;
        if (!final && lead == 0xed && available == 2 && byte(offset + 1) >= 0xa0 && byte(offset + 1) <= 0xbf)
            break;
        const size_t checked = std::min(need, available);
        bool malformed = false;
        for (size_t index_value = 1; index_value < checked; ++index_value)
        {
            const uint8_t value = byte(offset + index_value);
            const bool valid_second = continuation(value) && !(lead == 0xe0 && value < 0xa0) && !(lead == 0xed && value >= 0xa0) && !(lead == 0xf0 && value < 0x90) && !(lead == 0xf4 && value > 0x8f);
            if (!(index_value == 1 ? valid_second : continuation(value)))
            {
                replacement();
                offset += index_value == 1 ? 1 : index_value;
                malformed = true;
                break;
            }
        }
        if (malformed)
            continue;
        if (available < need)
        {
            if (!final)
                break;
            replacement();
            offset += available;
            continue;
        }
        output.append(pending, offset, need);
        offset += need;
    }
    if (offset != 0)
        pending.erase(0, offset);
    return output;
#endif
}

} // namespace moe
} // namespace ncnn
