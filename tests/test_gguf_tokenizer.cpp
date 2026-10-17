#include "models/tokenizer.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static void check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

// Pinned canonical templates keep these metadata-only fixtures independent of
// source-tree assets, external model packages, and Python tokenizer libraries.
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

static void append_unsigned(std::vector<uint8_t>& bytes, uint64_t value, size_t width)
{
    for (size_t index = 0; index < width; ++index)
        bytes.push_back(static_cast<uint8_t>(value >> (index * 8)));
}

static void append_string(std::vector<uint8_t>& bytes, std::string_view value)
{
    append_unsigned(bytes, value.size(), 8);
    bytes.insert(bytes.end(), value.begin(), value.end());
}

static std::string utf8(uint32_t point)
{
    if (point < 128)
        return std::string(1, static_cast<char>(point));
    return {static_cast<char>(0xc0 | (point >> 6)), static_cast<char>(0x80 | (point & 63))};
}

struct Fixture
{
    std::string architecture;
    std::string pre;
    std::string chat;
    std::vector<std::string> tokens;
    std::vector<uint32_t> types;
    std::vector<std::string> merges = {"f o", "fo o", "1 2", "12 3"};
    uint32_t base = 0;
    uint32_t first_byte = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    bool has_bos = true;
    bool add_bos = false;

    uint32_t byte_id(uint8_t byte) const
    {
        return first_byte + byte;
    }

    uint32_t word_id() const
    {
        return first_byte + 256;
    }
};

static Fixture fixture(const std::string& architecture, bool qwen38 = false, bool dense_template = false)
{
    Fixture result;
    result.architecture = architecture;
    const bool deepseek = architecture == "deepseek4";
    const bool gpt = architecture == "gpt-oss";
    result.pre = deepseek ? "joyai-llm" : gpt ? "gpt-4o"
                                              : "qwen35";
    result.base = deepseek ? 128000 : gpt ? 199998
                                          : 248044;
    result.tokens.resize(deepseek ? 129280 : gpt ? 201088
                                                 : 248320);
    result.types.resize(result.tokens.size(), 1);
    result.first_byte = deepseek ? 3 : 0;
    result.bos = deepseek ? 0 : gpt ? 199998
                                    : 248044;
    result.eos = deepseek ? 1 : gpt ? 200002
                                    : 248046;
    result.has_bos = deepseek || gpt;
    result.chat = "This arbitrary Jinja is not evaluated by the fixed native formatter.";
    for (uint32_t id = 0; id < result.tokens.size(); ++id)
        result.tokens[id] = "fixture" + std::to_string(id);
    uint32_t next_symbol = 256;
    for (uint32_t byte = 0; byte < 256; ++byte)
    {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172)
                             || (byte >= 174 && byte <= 255);
        result.tokens[result.first_byte + byte] = utf8(visible ? byte : next_symbol++);
    }
    for (size_t offset = 0; const auto* token : {"fo", "foo", "12", "123", "zz"})
        result.tokens[result.word_id() + offset++] = token;
    if (deepseek)
    {
        for (uint32_t id = result.base; id < result.tokens.size(); ++id)
        {
            result.tokens[id] = "<|fixture_" + std::to_string(id) + "|>";
            result.types[id] = 3;
        }
        for (const auto& item : std::vector<std::pair<uint32_t, std::string>>{
                 {0, "<｜begin▁of▁sentence｜>"},
                 {1, "<｜end▁of▁sentence｜>"},
                 {2, "<｜▁pad▁｜>"},
                 {128803, "<｜User｜>"},
                 {128804, "<｜Assistant｜>"},
                 {128821, "<think>"},
                 {128822, "</think>"},
                 {128825, "｜DSML｜"},
                 {128828, "<｜latest_reminder｜>"},
                 {129279, "<｜image｜>"}})
        {
            result.tokens[item.first] = item.second;
            // Also cover exporters that describe BOS/EOS/PAD as USER_DEFINED.
            result.types[item.first] = item.first < 3 || item.first == 128821 || item.first == 128822 ? 4 : 3;
        }
    }
    else
    {
        const std::vector<std::string> added = gpt ? std::vector<std::string>{
                                                         "<|startoftext|>", "<|endoftext|>", "<|reserved_200000|>", "<|reserved_200001|>", "<|return|>",
                                                         "<|constrain|>", "<|reserved_200004|>", "<|channel|>", "<|start|>", "<|end|>", "<|message|>",
                                                         "<|reserved_200009|>", "<|reserved_200010|>", "<|reserved_200011|>", "<|call|>",
                                                         "<|reserved_200013|>", "<|reserved_200014|>", "<|reserved_200015|>", "<|reserved_200016|>",
                                                         "<|reserved_200017|>", "<|endofprompt|>"}
                                                   : std::vector<std::string>{"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>", "<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>"};
        for (uint32_t id = result.base; id < result.tokens.size(); ++id)
        {
            result.tokens[id] = "[PAD" + std::to_string(id) + "]";
            result.types[id] = 5;
        }
        for (uint32_t offset = 0; offset < added.size(); ++offset)
        {
            result.tokens[result.base + offset] = added[offset];
            // llama.cpp's exporter can promote <|fim_*|> from USER_DEFINED to
            // CONTROL. This must not change native HF-compatible visibility.
            result.types[result.base + offset] = 3;
        }
        if (!gpt)
        {
            result.chat = qwen38 ? qwen38_template : qwen36_template;
            if (dense_template)
            {
                for (const auto& change : std::vector<std::pair<std::string, std::string>>{
                         {"(preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index)", "loop.index0 > ns.last_query_index"},
                         {"args_value | string if args_value is string else args_value | tojson | safe", "args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string"}})
                {
                    const size_t position = result.chat.find(change.first);
                    check(position != std::string::npos, "pinned dense Qwen template change is absent");
                    result.chat.replace(position, change.first.size(), change.second);
                }
            }
        }
    }
    return result;
}

static void write_fixture(const std::filesystem::path& path, const Fixture& value)
{
    std::vector<uint8_t> bytes{'G', 'G', 'U', 'F'};
    append_unsigned(bytes, 3, 4);
    append_unsigned(bytes, 0, 8);
    append_unsigned(bytes, 9 + (value.has_bos ? 1 : 0), 8);
    const auto string_metadata = [&](std::string_view key, std::string_view text) {
        append_string(bytes, key);
        append_unsigned(bytes, 8, 4);
        append_string(bytes, text);
    };
    const auto integer_metadata = [&](std::string_view key, uint32_t number) {
        append_string(bytes, key);
        append_unsigned(bytes, 4, 4);
        append_unsigned(bytes, number, 4);
    };
    string_metadata("general.architecture", value.architecture);
    string_metadata("tokenizer.ggml.model", "gpt2");
    string_metadata("tokenizer.ggml.pre", value.pre);
    string_metadata("tokenizer.chat_template", value.chat);
    integer_metadata("tokenizer.ggml.eos_token_id", value.eos);
    if (value.has_bos)
        integer_metadata("tokenizer.ggml.bos_token_id", value.bos);
    append_string(bytes, "tokenizer.ggml.add_bos_token");
    append_unsigned(bytes, 7, 4);
    bytes.push_back(static_cast<uint8_t>(value.add_bos));
    const auto string_array = [&](std::string_view key, const std::vector<std::string>& entries) {
        append_string(bytes, key);
        append_unsigned(bytes, 9, 4);
        append_unsigned(bytes, 8, 4);
        append_unsigned(bytes, entries.size(), 8);
        for (const auto& entry : entries)
            append_string(bytes, entry);
    };
    string_array("tokenizer.ggml.tokens", value.tokens);
    string_array("tokenizer.ggml.merges", value.merges);
    append_string(bytes, "tokenizer.ggml.token_type");
    append_unsigned(bytes, 9, 4);
    append_unsigned(bytes, 4, 4);
    append_unsigned(bytes, value.types.size(), 8);
    for (uint32_t type : value.types)
        append_unsigned(bytes, type, 4);
    bytes.resize(bytes.size() + (32 - bytes.size() % 32) % 32);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    check(static_cast<bool>(stream), "failed to write tokenizer GGUF fixture");
}

static std::string decode(const Tokenizer& tokenizer, const std::vector<int32_t>& ids)
{
    std::string pending;
    std::string output;
    for (size_t index = 0; index < ids.size(); ++index)
        output += tokenizer.decode(ids[index], pending, index + 1 == ids.size());
    output += tokenizer.decode(-1, pending, true);
    check(pending.empty(), "decoded UTF-8 remained buffered");
    return output;
}

static void check_profile(const std::filesystem::path& root, Fixture value, bool qwen38 = false)
{
    const auto path = root / (value.architecture + (qwen38 ? "38" : "") + ".gguf");
    write_fixture(path, value);
    Tokenizer tokenizer;
    check(tokenizer.load(path.string(), value.tokens.size()), value.architecture + ": GGUF tokenizer load failed");
    const bool deepseek = value.architecture == "deepseek4";
    const bool gpt = value.architecture == "gpt-oss";
    const bool qwen = !deepseek && !gpt;
    check(tokenizer.encode("foo") == std::vector<int32_t>{static_cast<int32_t>(value.word_id() + 1)}, "BPE merge order changed");
    const std::string text = "CPU 中文 café 😀 1234\n";
    check(decode(tokenizer, tokenizer.encode(text)) == text, "ByteLevel UTF-8 round trip changed");
    const std::string decomposed = "cafe\xcc\x81";
    check(decode(tokenizer, tokenizer.encode(decomposed)) == (qwen ? "café" : decomposed), "family NFC behavior changed");
    const auto zz = tokenizer.encode("zz");
    check(zz == (gpt ? std::vector<int32_t>{static_cast<int32_t>(value.word_id() + 4)} : std::vector<int32_t>{static_cast<int32_t>(value.byte_id('z')), static_cast<int32_t>(value.byte_id('z'))}),
          "family ignore_merges behavior changed");
    const auto digits = tokenizer.encode("123");
    check(digits == (qwen ? std::vector<int32_t>{static_cast<int32_t>(value.byte_id('1')), static_cast<int32_t>(value.byte_id('2')), static_cast<int32_t>(value.byte_id('3'))} : std::vector<int32_t>{static_cast<int32_t>(value.word_id() + 3)}),
          "family numeric pre-tokenization changed");
    std::string pending;
    check(tokenizer.decode(static_cast<int32_t>(value.byte_id(0xf0)), pending).empty() && pending.size() == 1,
          "UTF-8 prefix was emitted prematurely");
    check(tokenizer.decode(static_cast<int32_t>(value.byte_id(0x9f)), pending).empty() && pending.size() == 2,
          "UTF-8 partial sequence was emitted prematurely");
    check(tokenizer.decode(static_cast<int32_t>(value.byte_id(0x98)), pending).empty() && pending.size() == 3,
          "UTF-8 partial sequence was emitted prematurely");
    check(tokenizer.decode(static_cast<int32_t>(value.byte_id(0x80)), pending, true) == "😀" && pending.empty(),
          "incremental UTF-8 decoding changed");
    if (gpt)
    {
        check(tokenizer.stop_tokens() == std::vector<int32_t>({200002, 200012}), "Harmony stop tokens changed");
        check(tokenizer.encode("<|reserved_200018|><|reserved_201088|>") == std::vector<int32_t>({200018, 201088}),
              "Harmony reserved overlay is absent");
        check(decode(tokenizer, {200018, 201088}) == "<|reserved_200018|><|reserved_201088|>", "Harmony reserved decode changed");
        check(tokenizer.encode("<|endofprompt|>") != std::vector<int32_t>{200018}, "replaced Harmony alias remains active");
        std::vector<int32_t> expected{200006};
        const auto append = [&](std::string_view raw) {
            const auto ids = tokenizer.encode(raw);
            expected.insert(expected.end(), ids.begin(), ids.end());
        };
        append("user");
        expected.push_back(200008);
        append("hi");
        expected.insert(expected.end(), {200007, 200006});
        append("assistant");
        check(tokenizer.encode_chat(R"([{"role":"user","content":"hi"}])", false) == expected, "Harmony chat boundaries changed");
    }
    else if (deepseek)
    {
        check(tokenizer.stop_tokens() == std::vector<int32_t>{1}, "DeepSeek stop tokens changed");
        check(decode(tokenizer, {1}) == "<｜end▁of▁sentence｜>", "DeepSeek raw EOS visibility differs from HF");
        check(tokenizer.encode("<think></think>") == std::vector<int32_t>({128821, 128822}), "DeepSeek think tokens changed");
        check(tokenizer.encode_chat(R"([{"role":"user","content":"hi"}])", true)
                  == tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜><think>"),
              "DeepSeek chat formatter changed");
        check(tokenizer.encode_chat(R"([{"role":"tool","content":"result"}])", true)
                  == tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜><tool_result>result</tool_result><｜Assistant｜><think>"),
              "DeepSeek standalone tool formatting changed");
        check(tokenizer.encode_chat(R"([{"role":"user","content":"first"},{"role":"tool","content":"one"},{"role":"tool","content":"two"},{"role":"user","content":"last"}])", true)
                  == tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜>first\n\n<tool_result>one</tool_result>\n\n<tool_result>two</tool_result>\n\nlast<｜Assistant｜><think>"),
              "DeepSeek user and consecutive tool merging changed");
        const std::string developer_messages = R"([{"role":"developer","content":"context"},{"role":"user","content":"query"}])";
        check(tokenizer.encode_chat(developer_messages, true)
                  == tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜>query<｜Assistant｜><think>"),
              "DeepSeek thinking did not filter the earlier developer message");
        check(tokenizer.encode_chat(developer_messages, false)
                  == tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜>context<｜User｜>query<｜Assistant｜></think>"),
              "DeepSeek non-thinking filtered the earlier developer message");
        const std::string empty_developer_messages = R"([{"role":"developer","content":""},{"role":"user","content":"query"}])";
        check(tokenizer.encode_chat(empty_developer_messages, true)
                  == tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜>query<｜Assistant｜><think>"),
              "DeepSeek thinking did not filter the empty earlier developer message");
        bool rejected_empty_developer = false;
        try
        {
            (void)tokenizer.encode_chat(empty_developer_messages, false);
        }
        catch (const std::invalid_argument&)
        {
            rejected_empty_developer = true;
        }
        check(rejected_empty_developer, "DeepSeek non-thinking accepted an empty developer message");
    }
    else
    {
        check(tokenizer.stop_tokens() == std::vector<int32_t>({248046, 248044}), "Qwen stop tokens changed");
        check(decode(tokenizer, {248045, 248068, 248060, 248069, 248046, 248320 - 1}) == "<think><|fim_prefix|></think>",
              "Qwen control/user-defined/unused visibility differs from HF");
        check(tokenizer.encode("<think><|fim_prefix|></think>") == std::vector<int32_t>({248068, 248060, 248069}),
              "Qwen user-defined token recognition changed");
        std::string expected;
        if (qwen38)
            expected = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n";
        expected += "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        check(tokenizer.encode_chat(R"([{"role":"user","content":" hi "}])", true) == tokenizer.encode(expected), "Qwen chat revision selection changed");
    }
    check(!tokenizer.load(path.string(), value.tokens.size() - 1), "mismatched logits vocabulary was accepted");
    check(tokenizer.encode("foo") == std::vector<int32_t>{static_cast<int32_t>(value.word_id() + 1)}, "failed reload changed active tokenizer");
}

static void check_invalid_profiles(const std::filesystem::path& root)
{
    auto value = fixture("qwen35");
    const auto path = root / "invalid.gguf";
    write_fixture(path, value);
    Tokenizer tokenizer;
    check(tokenizer.load(path.string()), "negative-case baseline did not load");
    const auto expected = tokenizer.encode("foo");
    const auto reject = [&] {
        write_fixture(path, value);
        check(!tokenizer.load(path.string()), "unsupported GGUF tokenizer profile was accepted");
        check(tokenizer.encode("foo") == expected, "failed profile validation changed active tokenizer");
    };
    value.pre = "qwen2";
    reject();
    value.pre = "qwen35";
    const auto canonical = value.chat;
    value.chat = "{{ messages }}";
    reject();
    value.chat.clear();
    reject();
    value.chat = canonical;
    value.eos = 248044;
    reject();
    value.eos = 248046;
    value.types[value.byte_id('a')] = 6;
    reject();
    value.types[value.byte_id('a')] = 1;
    value.tokens[248045] = "<|different_start|>";
    reject();
    value.tokens[248045] = "<|im_start|>";
    value.merges.push_back("missing pair");
    reject();
    value.merges.pop_back();
    value.merges.push_back(value.merges.front());
    reject();
    value.merges.pop_back();
    value.add_bos = true;
    reject();
    value.add_bos = false;
    value.architecture = "qwen4exp";
    reject();
    value.architecture = "unsupported";
    reject();
}

} // namespace moe
} // namespace ncnn

int main()
{
    const auto root = std::filesystem::temp_directory_path()
                      / ("ncnn_moe_gguf_tokenizer_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        std::filesystem::create_directories(root);
        ncnn::moe::check_profile(root, ncnn::moe::fixture("deepseek4"));
        ncnn::moe::check_profile(root, ncnn::moe::fixture("gpt-oss"));
        ncnn::moe::check_profile(root, ncnn::moe::fixture("qwen35", false, true));
        ncnn::moe::check_profile(root, ncnn::moe::fixture("qwen35moe"));
        ncnn::moe::check_profile(root, ncnn::moe::fixture("qwen35", true), true);
        ncnn::moe::check_profile(root, ncnn::moe::fixture("qwen4exp", true), true);
        ncnn::moe::check_invalid_profiles(root);
        std::filesystem::remove_all(root);
        std::cout << "GGUF tokenizer profiles passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::error_code cleanup_error;
        std::filesystem::remove_all(root, cleanup_error);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
