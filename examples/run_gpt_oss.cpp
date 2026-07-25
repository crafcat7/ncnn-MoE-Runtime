#include "ncnn/moe/runtime.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
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

static const char* expert_memory_mode_name(ExpertMemoryMode mode)
{
    if (mode == ExpertMemoryMode::OnDemand)
        return "on-demand";
    if (mode == ExpertMemoryMode::Eager)
        return "eager";
    return "auto";
}

enum RunnerFlag : uint32_t
{
    RunnerStreamTokenIds = 1u << 0
};

static const char* require_value(
    int argc,
    char** argv,
    int& index,
    const char* option)
{
    if (++index >= argc)
        throw std::invalid_argument(std::string(option) + " requires a value");
    return argv[index];
}

static uint64_t mebibytes(
    const char* value,
    const char* option)
{
    const uint64_t count = std::stoull(value);
    if (count > std::numeric_limits<uint64_t>::max() / (1024 * 1024))
        throw std::out_of_range(std::string(option) + " is too large");
    return count * 1024 * 1024;
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: ncnn_moe_gpt_oss <model-directory> <token-id> [token-id ...]"
                     " [--max-new-tokens N] [--stop-token ID ...] [--temperature T]"
                     " [--top-k K] [--top-p P] [--min-p P] [--seed N]"
                     " [--expert-memory auto|eager|on-demand]"
                     " [--host-memory-mb N] [--expert-cache-mb N]"
                     " [--expert-gpu-cache-mb N]"
                     " [--expert-io-workers N]"
                     " [--mmap-experts]"
                     " [--cpu|--hybrid|--hybrid-prefetch]"
                     " [--stream-token-ids]\n";
        return 2;
    }

    try {
        ncnn::moe::RuntimeOptions runtime_options;
        ncnn::moe::SessionOptions session_options;
        ncnn::moe::GenerationOptions generation_options;
        generation_options.sampling.temperature = 0.0f;
        std::vector<int32_t> prompt;
        uint32_t runner_flags = 0;
        for (int index = 2; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--max-new-tokens") {
                generation_options.max_new_tokens = static_cast<uint32_t>(
                    std::stoul(ncnn::moe::require_value(
                        argc,
                        argv,
                        index,
                        "--max-new-tokens")));
            }
            else if (argument == "--stop-token") {
                generation_options.stop_tokens.push_back(std::stoi(
                    ncnn::moe::require_value(
                        argc,
                        argv,
                        index,
                        "--stop-token")));
            }
            else if (argument == "--temperature") {
                generation_options.sampling.temperature = std::stof(
                    ncnn::moe::require_value(
                        argc,
                        argv,
                        index,
                        "--temperature"));
            }
            else if (argument == "--top-k") {
                generation_options.sampling.top_k = static_cast<uint32_t>(
                    std::stoul(ncnn::moe::require_value(
                        argc,
                        argv,
                        index,
                        "--top-k")));
            }
            else if (argument == "--top-p") {
                generation_options.sampling.top_p = std::stof(
                    ncnn::moe::require_value(
                        argc,
                        argv,
                        index,
                        "--top-p"));
            }
            else if (argument == "--min-p") {
                generation_options.sampling.min_p = std::stof(
                    ncnn::moe::require_value(
                        argc,
                        argv,
                        index,
                        "--min-p"));
            }
            else if (argument == "--seed") {
                session_options.sampling_seed = std::stoull(
                    ncnn::moe::require_value(
                        argc,
                        argv,
                        index,
                        "--seed"));
            }
            else if (argument == "--expert-cache-mb") {
                runtime_options.expert_cache_bytes = ncnn::moe::mebibytes(
                    ncnn::moe::require_value(
                        argc,
                        argv,
                        index,
                        "--expert-cache-mb"),
                    "--expert-cache-mb");
            }
            else if (argument == "--host-memory-mb") {
                runtime_options.host_memory_budget_bytes
                    = ncnn::moe::mebibytes(
                        ncnn::moe::require_value(
                            argc,
                            argv,
                            index,
                            "--host-memory-mb"),
                        "--host-memory-mb");
            }
            else if (argument == "--expert-gpu-cache-mb") {
                runtime_options.expert_gpu_cache_bytes
                    = ncnn::moe::mebibytes(
                        ncnn::moe::require_value(
                            argc,
                            argv,
                            index,
                            "--expert-gpu-cache-mb"),
                        "--expert-gpu-cache-mb");
            }
            else if (argument == "--expert-memory") {
                const std::string mode = ncnn::moe::require_value(
                    argc,
                    argv,
                    index,
                    "--expert-memory");
                if (mode == "auto") {
                    runtime_options.expert_memory_mode
                        = ncnn::moe::ExpertMemoryMode::Auto;
                }
                else if (mode == "eager") {
                    runtime_options.expert_memory_mode
                        = ncnn::moe::ExpertMemoryMode::Eager;
                }
                else if (mode == "on-demand") {
                    runtime_options.expert_memory_mode
                        = ncnn::moe::ExpertMemoryMode::OnDemand;
                }
                else {
                    std::cerr << "--expert-memory must be auto, eager, or on-demand\n";
                    return 2;
                }
            }
            else if (argument == "--expert-io-workers") {
                const uint64_t workers = std::stoull(
                    ncnn::moe::require_value(
                        argc,
                        argv,
                        index,
                        "--expert-io-workers"));
                if (workers > 64) {
                    std::cerr << "--expert-io-workers must be between 0 and 64\n";
                    return 2;
                }
                runtime_options.expert_io_workers
                    = static_cast<uint32_t>(workers);
            }
            else if (argument == "--stream-token-ids") {
                runner_flags |= ncnn::moe::RunnerStreamTokenIds;
            }
            else if (argument == "--mmap-experts") {
                runtime_options.flags
                    |= ncnn::moe::RuntimeOptionMemoryMapExperts;
            }
            else if (argument == "--cpu") {
                runtime_options.hybrid_mode
                    = ncnn::moe::HybridMode::CpuOnly;
            }
            else if (argument == "--hybrid") {
                runtime_options.hybrid_mode
                    = ncnn::moe::HybridMode::HybridExperts;
            }
            else if (argument == "--hybrid-prefetch") {
                runtime_options.hybrid_mode
                    = ncnn::moe::HybridMode::VulkanWithCpuPrefetch;
            }
            else {
                prompt.push_back(std::stoi(argument));
            }
        }
        if (prompt.empty() || generation_options.max_new_tokens == 0) {
            std::cerr << "at least one prompt token and one generated token are required\n";
            return 2;
        }

        ncnn::moe::Runtime runtime;
        const auto load_start = std::chrono::steady_clock::now();
        auto model = runtime.load_model(std::filesystem::path(argv[1]), runtime_options);
        if (!model) {
            std::cerr << "load failed: " << model.error().message << '\n';
            return 1;
        }
        ncnn::moe::ModelPtr loaded_model = std::move(model).value();
        std::cout << "loaded " << loaded_model->descriptor().model_type
                  << " in " << ncnn::moe::elapsed_seconds(load_start) << " s, backend "
                  << ncnn::moe::hybrid_mode_name(loaded_model->hybrid_mode()) << '\n';
        const ncnn::moe::ModelMemoryPlan& memory_plan
            = loaded_model->memory_plan();
        std::cout << "Expert memory: "
                  << ncnn::moe::expert_memory_mode_name(memory_plan.selected_mode)
                  << ", " << memory_plan.estimated_expert_bytes / (1024 * 1024)
                  << " MiB estimated";
        if (ncnn::moe::has_flag(
                memory_plan.flags,
                ncnn::moe::ModelMemoryFileBackedExperts)) {
            std::cout << ", " << memory_plan.expert_cache_bytes / (1024 * 1024)
                      << " MiB cache";
        }
        std::cout << '\n';
        std::cout << "Host memory budget: "
                  << memory_plan.host_memory_budget_bytes / (1024 * 1024)
                  << " MiB";
        if (memory_plan.physical_memory_bytes != 0) {
            std::cout << " of "
                      << memory_plan.physical_memory_bytes / (1024 * 1024)
                      << " MiB detected";
        }
        std::cout << ", dense estimate "
                  << memory_plan.estimated_dense_bytes / (1024 * 1024)
                  << " MiB\n";
        if (runtime.capabilities().vulkan_heap_budget_bytes != 0) {
            std::cout << "Vulkan heap budget: "
                      << runtime.capabilities().vulkan_heap_budget_bytes
                             / (1024 * 1024)
                      << " MiB\n";
        }
        if (runtime_options.expert_gpu_cache_bytes != 0) {
            std::cout << "Expert GPU victim cache: "
                      << runtime_options.expert_gpu_cache_bytes / (1024 * 1024)
                      << " MiB configured\n";
        }

        auto session = runtime.create_session(loaded_model, session_options);
        if (!session) {
            std::cerr << "session creation failed: " << session.error().message << '\n';
            return 1;
        }

        ncnn::moe::SessionPtr active_session = std::move(session).value();
        const auto generation_start = std::chrono::steady_clock::now();
        auto generated = active_session->generate(
            prompt,
            generation_options,
            [runner_flags](const ncnn::moe::StreamToken& token) {
                if (ncnn::moe::has_flag(
                        runner_flags,
                        ncnn::moe::RunnerStreamTokenIds)) {
                    std::cout << "generated token id: " << token.token_id << '\n'
                              << std::flush;
                }
                return true;
            });
        if (!generated) {
            std::cerr << "generation failed: " << generated.error().message << '\n';
            return 1;
        }
        ncnn::moe::GenerationResult generation = std::move(generated).value();
        const ncnn::moe::SessionStatistics statistics
            = active_session->statistics();
        std::cout << "generated " << generation.tokens.size() << " token(s) in "
                  << ncnn::moe::elapsed_seconds(generation_start) << " s\n";
        std::cout << "Vulkan linear dispatches: "
                  << statistics.vulkan_linear_dispatches << '\n';
        std::cout << "Vulkan attention blocks: "
                  << statistics.vulkan_attention_blocks << '\n';
        std::cout << "Vulkan compute submissions: "
                  << statistics.vulkan_compute_submissions << '\n';
        std::cout << "Vulkan batch transfers: "
                  << statistics.vulkan_batch_uploads << " upload(s), "
                  << statistics.vulkan_batch_downloads << " download(s)\n";
        std::cout << "Vulkan auxiliary uploads: "
                  << statistics.vulkan_auxiliary_uploads << " upload(s), "
                  << statistics.vulkan_auxiliary_upload_bytes << " bytes\n";
        std::cout << "Vulkan staging slots: "
                  << statistics.vulkan_staging_slot_resizes << " resize(s), "
                  << statistics.vulkan_staging_slot_reuses << " reuse(s), "
                  << statistics.vulkan_staging_slot_acquisitions << " acquisition(s), "
                  << statistics.vulkan_staging_slot_contentions << " contention(s)\n";
        std::cout << "Attention time: "
                  << statistics.attention_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Router time: "
                  << statistics.router_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert time: "
                  << statistics.expert_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert prefetches: " << statistics.expert_prefetches
                  << " (" << statistics.expert_prefetch_bytes << " bytes hinted)\n";
        std::cout << "Expert cache: " << statistics.expert_cache_hits
                  << " hit(s), " << statistics.expert_cache_misses
                  << " miss(es), " << statistics.expert_cache_evictions
                  << " eviction(s), " << statistics.expert_cache_bytes_read
                  << " bytes read, " << statistics.expert_cache_resident_bytes
                  << " bytes resident, "
                  << statistics.expert_cache_queued_reads
                  << " queued read(s), "
                  << statistics.expert_cache_speculative_reads
                  << " speculative read(s)\n";
        std::cout << "Expert mmap: "
                  << statistics.expert_cache_mapped_ranges
                  << " range(s), "
                  << statistics.expert_cache_mapped_bytes
                  << " bytes\n";
        std::cout << "Expert GPU cache: "
                  << statistics.expert_gpu_cache_hits
                  << " hit(s), "
                  << statistics.expert_gpu_cache_misses
                  << " miss(es), "
                  << statistics.expert_gpu_cache_admissions
                  << " admission(s), "
                  << statistics.expert_gpu_cache_stores
                  << " store(s), "
                  << statistics.expert_gpu_cache_evictions
                  << " eviction(s), "
                  << statistics.expert_gpu_cache_dropped_admissions
                  << " dropped admission(s), "
                  << statistics.expert_gpu_cache_restore_failures
                  << " restore failure(s), "
                  << statistics.expert_gpu_cache_bytes_uploaded
                  << " bytes uploaded, "
                  << statistics.expert_gpu_cache_bytes_downloaded
                  << " bytes downloaded, "
                  << statistics.expert_gpu_cache_restore_time_microseconds / 1000.0
                  << " ms restoring, "
                  << statistics.expert_gpu_cache_mapped_stores
                  << " mapped store(s), "
                  << statistics.expert_gpu_cache_mapped_restores
                  << " mapped restore(s), "
                  << statistics.expert_gpu_cache_resident_bytes
                  << " bytes resident, "
                  << statistics.expert_gpu_cache_pending_bytes
                  << " bytes pending\n";
        std::cout << "Parallel CPU experts: "
                  << (ncnn::moe::has_flag(
                          runtime.capabilities().flags,
                          ncnn::moe::RuntimeCapabilityOpenmpExpertParallelism)
                          ? "OpenMP enabled"
                          : "single-thread fallback")
                  << '\n';
        std::cout << "MXFP4 CPU kernel: " << runtime.capabilities().mxfp4_kernel << '\n';
        std::cout << "MXFP4 decode GEMV rows: "
                  << statistics.mxfp4_decode_gemv_rows << '\n';
        std::cout << "MXFP4 prefill GEMM rows: "
                  << statistics.mxfp4_prefill_gemm_rows << '\n';
        std::cout << "MXFP4 paired rows: "
                  << statistics.mxfp4_paired_rows << '\n';
        std::cout << "MXFP4 fused Gate/Up rows: "
                  << statistics.mxfp4_fused_gate_up_rows << '\n';
        std::cout << "Parallel expert tasks: "
                  << statistics.expert_parallel_tasks << '\n';
        std::cout << "generated token ids:";
        for (const ncnn::moe::StreamToken& token : generation.tokens)
            std::cout << ' ' << token.token_id;
        std::cout << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "invalid argument: " << error.what() << '\n';
        return 2;
    }
}
