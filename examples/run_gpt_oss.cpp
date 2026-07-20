#include "ncnn/moe/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static int32_t greedy_token(const LogitsOutput& logits)
{
    const auto iterator = std::max_element(logits.values.begin(), logits.values.end());
    return static_cast<int32_t>(std::distance(logits.values.begin(), iterator));
}

static double elapsed_seconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static bool contains_token(const std::vector<int32_t>& tokens, int32_t token)
{
    return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: ncnn_moe_gpt_oss <model-directory> <token-id> [token-id ...]"
                     " [--max-new-tokens N] [--stop-token ID ...]\n";
        return 2;
    }

    try {
        uint32_t max_new_tokens = 1;
        std::vector<int32_t> prompt;
        std::vector<int32_t> stop_tokens;
        for (int index = 2; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--max-new-tokens") {
                if (++index >= argc) {
                    std::cerr << "--max-new-tokens requires a value\n";
                    return 2;
                }
                max_new_tokens = static_cast<uint32_t>(std::stoul(argv[index]));
            }
            else if (argument == "--stop-token") {
                if (++index >= argc) {
                    std::cerr << "--stop-token requires a value\n";
                    return 2;
                }
                stop_tokens.push_back(std::stoi(argv[index]));
            }
            else {
                prompt.push_back(std::stoi(argument));
            }
        }
        if (prompt.empty() || max_new_tokens == 0) {
            std::cerr << "at least one prompt token and one generated token are required\n";
            return 2;
        }

        ncnn::moe::Runtime runtime;
        const auto load_start = std::chrono::steady_clock::now();
        auto model = runtime.load_model(std::filesystem::path(argv[1]));
        if (!model) {
            std::cerr << "load failed: " << model.error().message << '\n';
            return 1;
        }
        std::cout << "loaded " << model.value()->descriptor().model_type
                  << " in " << ncnn::moe::elapsed_seconds(load_start) << " s\n";

        auto session = runtime.create_session(model.value());
        if (!session) {
            std::cerr << "session creation failed: " << session.error().message << '\n';
            return 1;
        }

        const auto prefill_start = std::chrono::steady_clock::now();
        auto prefill = session.value()->prefill(prompt);
        if (!prefill) {
            std::cerr << "prefill failed: " << prefill.error().message << '\n';
            return 1;
        }
        std::cout << "prefill " << prompt.size() << " token(s) in "
                  << ncnn::moe::elapsed_seconds(prefill_start) << " s\n";

        ncnn::moe::LogitsOutput logits = std::move(prefill).value().logits;
        std::vector<int32_t> generated_tokens;
        for (uint32_t index = 0; index < max_new_tokens; ++index) {
            const int32_t token = ncnn::moe::greedy_token(logits);
            generated_tokens.push_back(token);
            if (ncnn::moe::contains_token(stop_tokens, token) || index + 1 == max_new_tokens)
                break;

            const auto decode_start = std::chrono::steady_clock::now();
            auto decoded = session.value()->decode(token);
            if (!decoded) {
                std::cerr << "\ndecode failed: " << decoded.error().message << '\n';
                return 1;
            }
            logits = std::move(decoded).value().logits;
            std::cout << "decoded token " << token << " in "
                      << ncnn::moe::elapsed_seconds(decode_start) << " s\n";
        }
        std::cout << "generated token ids:";
        for (int32_t token : generated_tokens)
            std::cout << ' ' << token;
        std::cout << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "invalid argument: " << error.what() << '\n';
        return 2;
    }
}
