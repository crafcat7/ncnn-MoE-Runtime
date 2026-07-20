#include "ncnn/moe/runtime.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: ncnn_moe_tiny <model-directory> [token-id ...]\n";
        return 2;
    }

    try {
        std::vector<int32_t> token_ids;
        for (int index = 2; index < argc; ++index)
            token_ids.push_back(std::stoi(argv[index]));
        if (token_ids.empty())
            token_ids.push_back(0);

        ncnn::moe::Runtime runtime;
        auto model = runtime.load_model(std::filesystem::path(argv[1]));
        if (!model) {
            std::cerr << "load failed: " << model.error().message << '\n';
            return 1;
        }

        auto session = runtime.create_session(model.value());
        if (!session) {
            std::cerr << "session creation failed: " << session.error().message << '\n';
            return 1;
        }

        auto result = session.value()->prefill(token_ids);
        if (!result) {
            std::cerr << "prefill failed: " << result.error().message << '\n';
            return 1;
        }

        std::cout << "processed " << result.value().processed_tokens
                  << " token(s), sequence length " << session.value()->sequence_length() << '\n';
        const size_t shown = std::min<size_t>(result.value().logits.values.size(), 16);
        std::cout << "last-token logits[0:" << shown << "]:";
        for (size_t index = 0; index < shown; ++index)
            std::cout << ' ' << result.value().logits.values[index];
        std::cout << '\n';

        const auto& stats = session.value()->statistics();
        std::cout << "expert assignments: " << stats.expert_assignments << '\n';
        std::cout << "expert batches: " << stats.expert_batches << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "invalid argument: " << error.what() << '\n';
        return 2;
    }
}
