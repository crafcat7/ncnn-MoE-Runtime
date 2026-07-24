#include "ncnn/moe/runtime.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
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
                     " [--expert-cache-mb N] [--expert-io-workers N]"
                     " [--cpu|--hybrid|--hybrid-prefetch]"
                     " [--stream-token-ids]\n";
        return 2;
    }

    try {
        uint32_t max_new_tokens = 1;
        uint64_t sampling_seed = 0;
        uint64_t expert_cache_bytes = 0;
        uint32_t expert_io_workers = 0;
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
            else if (argument == "--expert-cache-mb") {
                if (++index >= argc) {
                    std::cerr << "--expert-cache-mb requires a value\n";
                    return 2;
                }
                const uint64_t megabytes = std::stoull(argv[index]);
                if (megabytes > std::numeric_limits<uint64_t>::max() / (1024 * 1024)) {
                    std::cerr << "--expert-cache-mb is too large\n";
                    return 2;
                }
                expert_cache_bytes = megabytes * 1024 * 1024;
            }
            else if (argument == "--expert-io-workers") {
                if (++index >= argc) {
                    std::cerr << "--expert-io-workers requires a value\n";
                    return 2;
                }
                const uint64_t workers = std::stoull(argv[index]);
                if (workers > 64) {
                    std::cerr << "--expert-io-workers must be between 0 and 64\n";
                    return 2;
                }
                expert_io_workers = static_cast<uint32_t>(workers);
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
        runtime_options.expert_cache_bytes = expert_cache_bytes;
        runtime_options.expert_io_workers = expert_io_workers;
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
        std::cout << "Vulkan compute submissions: "
                  << session.value()->statistics().vulkan_compute_submissions << '\n';
        std::cout << "Vulkan batch transfers: "
                  << session.value()->statistics().vulkan_batch_uploads << " upload(s), "
                  << session.value()->statistics().vulkan_batch_downloads << " download(s)\n";
        std::cout << "Vulkan auxiliary uploads: "
                  << session.value()->statistics().vulkan_auxiliary_uploads << " upload(s), "
                  << session.value()->statistics().vulkan_auxiliary_upload_bytes << " bytes\n";
        std::cout << "Vulkan staging slots: "
                  << session.value()->statistics().vulkan_staging_slot_resizes << " resize(s), "
                  << session.value()->statistics().vulkan_staging_slot_reuses << " reuse(s), "
                  << session.value()->statistics().vulkan_staging_slot_acquisitions << " acquisition(s), "
                  << session.value()->statistics().vulkan_staging_slot_contentions << " contention(s)\n";
        std::cout << "Attention time: "
                  << session.value()->statistics().attention_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Router time: "
                  << session.value()->statistics().router_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert time: "
                  << session.value()->statistics().expert_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert prefetches: " << session.value()->statistics().expert_prefetches
                  << " (" << session.value()->statistics().expert_prefetch_bytes << " bytes hinted)\n";
        std::cout << "Expert cache: " << session.value()->statistics().expert_cache_hits
                  << " hit(s), " << session.value()->statistics().expert_cache_misses
                  << " miss(es), " << session.value()->statistics().expert_cache_evictions
                  << " eviction(s), " << session.value()->statistics().expert_cache_bytes_read
                  << " bytes read, " << session.value()->statistics().expert_cache_resident_bytes
                  << " bytes resident, "
                  << session.value()->statistics().expert_cache_queued_reads
                  << " queued read(s), "
                  << session.value()->statistics().expert_cache_speculative_reads
                  << " speculative read(s)\n";
        std::cout << "Parallel CPU experts: "
                  << (runtime.capabilities().openmp_expert_parallelism
                          ? "OpenMP enabled"
                          : "single-thread fallback")
                  << '\n';
        std::cout << "MXFP4 CPU kernel: " << runtime.capabilities().mxfp4_kernel << '\n';
        std::cout << "MXFP4 decode GEMV rows: "
                  << session.value()->statistics().mxfp4_decode_gemv_rows << '\n';
        std::cout << "MXFP4 prefill GEMM rows: "
                  << session.value()->statistics().mxfp4_prefill_gemm_rows << '\n';
        std::cout << "MXFP4 paired rows: "
                  << session.value()->statistics().mxfp4_paired_rows << '\n';
        std::cout << "MXFP4 fused Gate/Up rows: "
                  << session.value()->statistics().mxfp4_fused_gate_up_rows << '\n';
        std::cout << "Parallel expert tasks: "
                  << session.value()->statistics().expert_parallel_tasks << '\n';
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
