#include "ncnn/moe/runtime.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static double elapsed_seconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static const char* hybrid_mode_name(HybridMode mode)
{
    if (mode == HybridMode::HybridExperts)
        return "vulkan-dense/cpu-experts";
    if (mode == HybridMode::VulkanWithCpuPrefetch)
        return "vulkan-dense/cpu-experts-prefetch";
    return "cpu";
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: ncnn_moe_gpt_oss <model-directory> <token-id> [token-id ...]"
                     " [--max-new-tokens N] [--stop-token ID ...] [--temperature T]"
                     " [--top-k K] [--top-p P] [--min-p P] [--seed N]"
                     " [--cpu|--hybrid|--hybrid-prefetch]"
                     " [--stream-token-ids]\n";
        return 2;
    }

    try {
        uint32_t max_new_tokens = 1;
        uint64_t sampling_seed = 0;
        ncnn::moe::SamplingOptions sampling;
        sampling.temperature = 0.0f;
        std::vector<int32_t> prompt;
        std::vector<int32_t> stop_tokens;
        bool stream_token_ids = false;
        ncnn::moe::HybridMode requested_mode = ncnn::moe::HybridMode::Auto;
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
            else if (argument == "--temperature") {
                if (++index >= argc) {
                    std::cerr << "--temperature requires a value\n";
                    return 2;
                }
                sampling.temperature = std::stof(argv[index]);
            }
            else if (argument == "--top-k") {
                if (++index >= argc) {
                    std::cerr << "--top-k requires a value\n";
                    return 2;
                }
                sampling.top_k = static_cast<uint32_t>(std::stoul(argv[index]));
            }
            else if (argument == "--top-p") {
                if (++index >= argc) {
                    std::cerr << "--top-p requires a value\n";
                    return 2;
                }
                sampling.top_p = std::stof(argv[index]);
            }
            else if (argument == "--min-p") {
                if (++index >= argc) {
                    std::cerr << "--min-p requires a value\n";
                    return 2;
                }
                sampling.min_p = std::stof(argv[index]);
            }
            else if (argument == "--seed") {
                if (++index >= argc) {
                    std::cerr << "--seed requires a value\n";
                    return 2;
                }
                sampling_seed = std::stoull(argv[index]);
            }
            else if (argument == "--stream-token-ids") {
                stream_token_ids = true;
            }
            else if (argument == "--cpu") {
                requested_mode = ncnn::moe::HybridMode::CpuOnly;
            }
            else if (argument == "--hybrid") {
                requested_mode = ncnn::moe::HybridMode::HybridExperts;
            }
            else if (argument == "--hybrid-prefetch") {
                requested_mode = ncnn::moe::HybridMode::VulkanWithCpuPrefetch;
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
        ncnn::moe::RuntimeOptions runtime_options;
        runtime_options.hybrid_mode = requested_mode;
        const auto load_start = std::chrono::steady_clock::now();
        auto model = runtime.load_model(std::filesystem::path(argv[1]), runtime_options);
        if (!model) {
            std::cerr << "load failed: " << model.error().message << '\n';
            return 1;
        }
        std::cout << "loaded " << model.value()->descriptor().model_type
                  << " in " << ncnn::moe::elapsed_seconds(load_start) << " s, backend "
                  << ncnn::moe::hybrid_mode_name(model.value()->hybrid_mode()) << '\n';

        ncnn::moe::SessionOptions session_options;
        session_options.sampling_seed = sampling_seed;
        auto session = runtime.create_session(model.value(), session_options);
        if (!session) {
            std::cerr << "session creation failed: " << session.error().message << '\n';
            return 1;
        }

        ncnn::moe::GenerationOptions generation_options;
        generation_options.max_new_tokens = max_new_tokens;
        generation_options.sampling = sampling;
        generation_options.stop_tokens = std::move(stop_tokens);
        const auto generation_start = std::chrono::steady_clock::now();
        auto generated = session.value()->generate(
            prompt,
            generation_options,
            [stream_token_ids](const ncnn::moe::StreamToken& token) {
                if (stream_token_ids)
                    std::cout << "generated token id: " << token.token_id << '\n' << std::flush;
                return true;
            });
        if (!generated) {
            std::cerr << "generation failed: " << generated.error().message << '\n';
            return 1;
        }
        std::cout << "generated " << generated.value().tokens.size() << " token(s) in "
                  << ncnn::moe::elapsed_seconds(generation_start) << " s\n";
        std::cout << "Vulkan linear dispatches: "
                  << session.value()->statistics().vulkan_linear_dispatches << '\n';
        std::cout << "Vulkan attention blocks: "
                  << session.value()->statistics().vulkan_attention_blocks << '\n';
        std::cout << "Attention time: "
                  << session.value()->statistics().attention_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Router time: "
                  << session.value()->statistics().router_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert time: "
                  << session.value()->statistics().expert_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert prefetches: " << session.value()->statistics().expert_prefetches
                  << " (" << session.value()->statistics().expert_prefetch_bytes << " bytes hinted)\n";
        std::cout << "generated token ids:";
        for (const ncnn::moe::StreamToken& token : generated.value().tokens)
            std::cout << ' ' << token.token_id;
        std::cout << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "invalid argument: " << error.what() << '\n';
        return 2;
    }
}
