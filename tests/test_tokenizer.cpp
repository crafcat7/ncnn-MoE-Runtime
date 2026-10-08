#include "models/json.h"
#include "models/tokenizer.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static std::string ids_json(const std::vector<int32_t>& ids)
{
    std::string result = "[";
    for (size_t index = 0; index < ids.size(); ++index)
    {
        if (index != 0)
            result.push_back(',');
        result += std::to_string(ids[index]);
    }
    result.push_back(']');
    return result;
}

static std::string bytes_json(std::string_view bytes)
{
    std::string result = "[";
    for (size_t index = 0; index < bytes.size(); ++index)
    {
        if (index != 0)
            result.push_back(',');
        result += std::to_string(static_cast<unsigned char>(bytes[index]));
    }
    result.push_back(']');
    return result;
}

static std::string strings_json(const std::vector<std::string>& values)
{
    std::string result = "[";
    for (size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0)
            result.push_back(',');
        result += json_escape(values[index]);
    }
    result.push_back(']');
    return result;
}

static std::string decode_ids(const Tokenizer& tokenizer, const std::vector<int32_t>& ids)
{
    std::string pending;
    std::string decoded;
    for (size_t index = 0; index < ids.size(); ++index)
        decoded += tokenizer.decode(ids[index], pending, index + 1 == ids.size());
    if (ids.empty())
        decoded += tokenizer.decode(-1, pending, true);
    return decoded;
}

static std::string error_record(std::string_view message)
{
    JsonObject result;
    result.add_bool("ok", false);
    result.add_string("error", message);
    return result.finish();
}

static std::string process_line(const Tokenizer& tokenizer, std::string_view line)
{
    const auto operation = json_string(line, "op");
    if (!operation)
        return error_record("request requires a string op");

    try
    {
        if (*operation == "encode")
        {
            const auto text = json_string(line, "text");
            if (!text)
                return error_record("encode requires a string text");
            const std::vector<int32_t> ids = tokenizer.encode(*text);
            JsonObject result;
            result.add_bool("ok", true);
            result.add_raw("ids", ids_json(ids));
            result.add_string("decoded", decode_ids(tokenizer, ids));
            return result.finish();
        }

        if (*operation == "chat")
        {
            const auto messages = find_manifest_member(line, "messages");
            if (!messages || messages->empty() || messages->front() != '[')
                return error_record("chat requires a messages array");
            const auto thinking = json_boolean(line, "enable_thinking");
            const std::vector<int32_t> ids = tokenizer.apply_chat(*messages, thinking.value_or(true));
            JsonObject result;
            result.add_bool("ok", true);
            result.add_raw("ids", ids_json(ids));
            result.add_string("decoded", decode_ids(tokenizer, ids));
            return result.finish();
        }

        if (*operation == "decode")
        {
            const auto parsed_ids = json_integer_array(line, "ids");
            if (!parsed_ids)
                return error_record("decode requires an integer ids array");
            std::vector<int32_t> ids;
            ids.reserve(parsed_ids->size());
            for (const int64_t id : *parsed_ids)
            {
                if (id < std::numeric_limits<int32_t>::min() || id > std::numeric_limits<int32_t>::max())
                    return error_record("decode token id is outside int32 range");
                ids.push_back(static_cast<int32_t>(id));
            }

            const auto frame = json_boolean(line, "stop_frame");
            const auto should_flush = json_boolean(line, "final");
            const bool final = should_flush.value_or(true);
            size_t visible_count = ids.size();
            bool stopped = false;
            if (frame.value_or(false))
            {
                const std::vector<int32_t>& stops = tokenizer.stop_tokens();
                for (size_t index = 0; index < ids.size(); ++index)
                {
                    if (std::find(stops.begin(), stops.end(), ids[index]) != stops.end())
                    {
                        visible_count = index;
                        stopped = true;
                        break;
                    }
                }
                ids.resize(visible_count);
            }

            std::string pending;
            std::string decoded;
            std::vector<std::string> pieces;
            pieces.reserve(ids.size());
            for (size_t index = 0; index < ids.size(); ++index)
            {
                std::string piece = tokenizer.decode(ids[index], pending, final && index + 1 == ids.size());
                decoded += piece;
                pieces.push_back(std::move(piece));
            }
            std::string flush;
            if (ids.empty() && final)
                flush = tokenizer.decode(-1, pending, true);
            decoded += flush;

            JsonObject result;
            result.add_bool("ok", true);
            result.add_string("decoded", decoded);
            result.add_raw("pieces", strings_json(pieces));
            result.add_string("flush", flush);
            result.add_raw("pending_bytes", bytes_json(pending));
            result.add_bool("final", final);
            result.add_bool("stopped", stopped);
            result.add_uint("consumed", visible_count);
            return result.finish();
        }

        if (*operation == "stops")
        {
            JsonObject result;
            result.add_bool("ok", true);
            result.add_raw("ids", ids_json(tokenizer.stop_tokens()));
            return result.finish();
        }

        return error_record("unsupported op");
    }
    catch (const std::exception& error)
    {
        return error_record(error.what());
    }
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: ncnn_moe_tokenizer_test MODEL_DIRECTORY < requests.jsonl\n";
        return 2;
    }

    ncnn::moe::Tokenizer tokenizer;
    if (!tokenizer.load(argv[1]))
    {
        std::cerr << "failed to load tokenizer assets from " << argv[1] << '\n';
        return 2;
    }

    std::vector<std::string> concurrency_texts = {
        "",
        "x",
        "CPU x86/ARM，中文English混排！",
        "café cafe\u0301 😀 <|im_start|>user",
        "<|im_start|>user\nhello<|im_end|>\n",
    };
    std::string long_text;
    for (size_t index = 0; index < 32; ++index)
    {
        long_text += "naïve 😀 中文 CPU/";
        if (index % 2 == 0)
            long_text += "cafe\u0301 ";
    }
    concurrency_texts.push_back(std::move(long_text));
    const std::string concurrency_chat = R"([{"role":"user","content":"中文 café 😀 <|im_start|>user"}])";
    const ncnn::moe::Tokenizer& const_tokenizer = tokenizer;
    std::vector<std::vector<int32_t>> expected_concurrency_ids;
    expected_concurrency_ids.reserve(concurrency_texts.size());
    for (const std::string& text : concurrency_texts)
        expected_concurrency_ids.push_back(const_tokenizer.encode(text));
    const std::vector<int32_t> expected_chat_ids = const_tokenizer.apply_chat(concurrency_chat, true);
    std::atomic<bool> concurrency_ok{true};
    std::vector<std::thread> workers;
    try
    {
        for (size_t worker_index = 0; worker_index < 4; ++worker_index)
        {
            workers.emplace_back([&, worker_index] {
                try
                {
                    for (size_t pass = 0; pass < 2; ++pass)
                    {
                        for (size_t offset = 0; offset < concurrency_texts.size(); ++offset)
                        {
                            const size_t text_index = (worker_index + pass * 3 + offset)
                                                      % concurrency_texts.size();
                            if (const_tokenizer.encode(concurrency_texts[text_index])
                                != expected_concurrency_ids[text_index])
                            {
                                concurrency_ok.store(false, std::memory_order_relaxed);
                                return;
                            }
                        }
                        if (const_tokenizer.apply_chat(concurrency_chat, true) != expected_chat_ids)
                        {
                            concurrency_ok.store(false, std::memory_order_relaxed);
                            return;
                        }
                    }
                }
                catch (...)
                {
                    concurrency_ok.store(false, std::memory_order_relaxed);
                }
            });
        }
    }
    catch (...)
    {
        concurrency_ok.store(false, std::memory_order_relaxed);
    }
    for (std::thread& worker : workers)
        worker.join();
    if (!concurrency_ok.load(std::memory_order_relaxed))
    {
        std::cerr << "concurrent tokenizer output differed from serial token IDs\n";
        return 2;
    }

    std::string line;
    while (std::getline(std::cin, line))
        std::cout << ncnn::moe::process_line(tokenizer, line) << '\n';
    if (!std::cin.eof())
    {
        std::cerr << "failed while reading JSONL requests\n";
        return 2;
    }
    return 0;
}
