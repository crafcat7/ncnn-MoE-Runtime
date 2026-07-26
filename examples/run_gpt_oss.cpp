#include "ncnn/moe/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <sstream>
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

#define NCNN_MOE_RUNNER_STREAM_TOKEN_BIT 0
#define NCNN_MOE_RUNNER_CROSS_CALL_BIT   1

enum RunnerFlag : uint32_t
{
    RunnerStreamTokenIds = UINT32_C(1) << NCNN_MOE_RUNNER_STREAM_TOKEN_BIT,
    RunnerCrossCallScheduling = UINT32_C(1) << NCNN_MOE_RUNNER_CROSS_CALL_BIT
};

static const char* require_value(int argc, char** argv, int& index, const char* option)
{
    if (++index >= argc)
        throw std::invalid_argument(std::string(option) + " requires a value");
    return argv[index];
}

static uint64_t mebibytes(const char* value, const char* option)
{
    const uint64_t count = std::stoull(value);
    if (count > std::numeric_limits<uint64_t>::max() / (1024 * 1024))
        throw std::out_of_range(std::string(option) + " is too large");
    return count * 1024 * 1024;
}

static std::vector<uint32_t> parse_device_indices(const char* value, const char* option)
{
    std::vector<uint32_t> result;
    std::stringstream input(value);
    std::string item;
    while (std::getline(input, item, ','))
    {
        if (item.empty())
        {
            throw std::invalid_argument(std::string(option) + " contains an empty device index");
        }
        result.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    if (result.empty())
    {
        throw std::invalid_argument(std::string(option) + " requires at least one device index");
    }
    return result;
}

static void append_feature(std::string& features, const char* name)
{
    if (!features.empty())
        features += ',';
    features += name;
}

static std::string vulkan_kernel_features(const VulkanDeviceCapabilities& device)
{
    std::string result;
    if (has_flag(device.flags, VulkanDeviceInt8Storage))
        append_feature(result, "int8-storage");
    if (has_flag(device.flags, VulkanDeviceInt8Arithmetic))
        append_feature(result, "int8-arithmetic");
    if (has_flag(device.flags, VulkanDeviceIntegerDotProduct))
        append_feature(result, "integer-dot");
    if (has_flag(device.flags, VulkanDeviceSubgroupOperations))
        append_feature(result, "subgroup");
    if (has_flag(device.flags, VulkanDeviceCooperativeMatrix))
        append_feature(result, "cooperative-matrix");
    if (has_flag(device.flags, VulkanDeviceInt8CooperativeMatrix))
        append_feature(result, "int8-cooperative-matrix");
    return result.empty() ? "baseline-fp32" : result;
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: ncnn_moe_gpt_oss <model-directory> <token-id> [token-id ...]"
                     " [--max-new-tokens N] [--stop-token ID ...] [--temperature T]"
                     " [--top-k K] [--top-p P] [--min-p P] [--seed N]"
                     " [--expert-memory auto|eager|on-demand]"
                     " [--host-memory-mb N] [--expert-cache-mb N]"
                     " [--expert-gpu-cache-mb N]"
                     " [--expert-gpu-victim-cache-mb N]"
                     " [--expert-gpu-victim-reuse-probe N]"
                     " [--disable-gpu-victim-execution]"
                     " [--expert-io-workers N]"
                     " [--vulkan-device N]"
                     " [--vulkan-devices N[,N...]]"
                     " [--mmap-experts]"
                     " [--direct-expert-io]"
                     " [--buffered-expert-io]"
                     " [--cache-warmup-runs N]"
                     " [--parallel-sessions N]"
                     " [--scheduler-expert-threads N]"
                     " [--scheduler-staging auto|force|off]"
                     " [--scheduler-cross-call]"
                     " [--scheduler-collection-us N]"
                     " [--scheduler-max-micro-batch N]"
                     " [--cpu|--hybrid|--hybrid-prefetch]"
                     " [--stream-token-ids]\n";
        return 2;
    }

    try
    {
        ncnn::moe::RuntimeOptions runtime_options;
        ncnn::moe::SessionOptions session_options;
        ncnn::moe::GenerationOptions generation_options;
        generation_options.sampling.temperature = 0.0f;
        std::vector<int32_t> prompt;
        uint32_t runner_flags = 0;
        uint32_t cache_warmup_runs = 0;
        uint32_t parallel_sessions = 1;
        uint32_t scheduler_expert_threads = 0;
        uint32_t scheduler_collection_microseconds = 200;
        uint32_t scheduler_max_micro_batch = 0;
        uint32_t scheduler_flags = 0;
        for (int index = 2; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (argument == "--max-new-tokens")
            {
                generation_options.max_new_tokens = static_cast<uint32_t>(std::stoul(ncnn::moe::require_value(argc, argv, index, "--max-new-tokens")));
            }
            else if (argument == "--stop-token")
            {
                generation_options.stop_tokens.push_back(std::stoi(ncnn::moe::require_value(argc, argv, index, "--stop-token")));
            }
            else if (argument == "--temperature")
            {
                generation_options.sampling.temperature = std::stof(ncnn::moe::require_value(argc, argv, index, "--temperature"));
            }
            else if (argument == "--top-k")
            {
                generation_options.sampling.top_k = static_cast<uint32_t>(std::stoul(ncnn::moe::require_value(argc, argv, index, "--top-k")));
            }
            else if (argument == "--top-p")
            {
                generation_options.sampling.top_p = std::stof(ncnn::moe::require_value(argc, argv, index, "--top-p"));
            }
            else if (argument == "--min-p")
            {
                generation_options.sampling.min_p = std::stof(ncnn::moe::require_value(argc, argv, index, "--min-p"));
            }
            else if (argument == "--seed")
            {
                session_options.sampling_seed = std::stoull(ncnn::moe::require_value(argc, argv, index, "--seed"));
            }
            else if (argument == "--expert-cache-mb")
            {
                runtime_options.expert_cache_bytes = ncnn::moe::mebibytes(ncnn::moe::require_value(argc, argv, index, "--expert-cache-mb"), "--expert-cache-mb");
            }
            else if (argument == "--host-memory-mb")
            {
                runtime_options.host_memory_budget_bytes = ncnn::moe::mebibytes(ncnn::moe::require_value(argc, argv, index, "--host-memory-mb"), "--host-memory-mb");
            }
            else if (argument == "--expert-gpu-cache-mb")
            {
                runtime_options.expert_gpu_cache_bytes = ncnn::moe::mebibytes(ncnn::moe::require_value(argc, argv, index, "--expert-gpu-cache-mb"), "--expert-gpu-cache-mb");
            }
            else if (argument == "--expert-gpu-victim-cache-mb")
            {
                runtime_options.expert_gpu_victim_cache_bytes = ncnn::moe::mebibytes(ncnn::moe::require_value(argc, argv, index, "--expert-gpu-victim-cache-mb"), "--expert-gpu-victim-cache-mb");
            }
            else if (argument == "--expert-gpu-victim-reuse-probe")
            {
                runtime_options.expert_gpu_victim_reuse_probe_interval = static_cast<uint32_t>(std::stoul(ncnn::moe::require_value(argc, argv, index, "--expert-gpu-victim-reuse-probe")));
            }
            else if (argument == "--disable-gpu-victim-execution")
            {
                runtime_options.flags |= ncnn::moe::RuntimeOptionDisableGpuVictimExecution;
            }
            else if (argument == "--expert-memory")
            {
                const std::string mode = ncnn::moe::require_value(argc, argv, index, "--expert-memory");
                if (mode == "auto")
                {
                    runtime_options.expert_memory_mode = ncnn::moe::ExpertMemoryMode::Auto;
                }
                else if (mode == "eager")
                {
                    runtime_options.expert_memory_mode = ncnn::moe::ExpertMemoryMode::Eager;
                }
                else if (mode == "on-demand")
                {
                    runtime_options.expert_memory_mode = ncnn::moe::ExpertMemoryMode::OnDemand;
                }
                else
                {
                    std::cerr << "--expert-memory must be auto, eager, or on-demand\n";
                    return 2;
                }
            }
            else if (argument == "--expert-io-workers")
            {
                const uint64_t workers = std::stoull(ncnn::moe::require_value(argc, argv, index, "--expert-io-workers"));
                if (workers > 64)
                {
                    std::cerr << "--expert-io-workers must be between 0 and 64\n";
                    return 2;
                }
                runtime_options.expert_io_workers = static_cast<uint32_t>(workers);
            }
            else if (argument == "--vulkan-device")
            {
                runtime_options.vulkan_device_index = static_cast<uint32_t>(std::stoul(ncnn::moe::require_value(argc, argv, index, "--vulkan-device")));
            }
            else if (argument == "--vulkan-devices")
            {
                runtime_options.vulkan_device_indices = ncnn::moe::parse_device_indices(ncnn::moe::require_value(argc, argv, index, "--vulkan-devices"), "--vulkan-devices");
            }
            else if (argument == "--stream-token-ids")
            {
                runner_flags |= ncnn::moe::RunnerStreamTokenIds;
            }
            else if (argument == "--mmap-experts")
            {
                runtime_options.flags |= ncnn::moe::RuntimeOptionMemoryMapExperts;
            }
            else if (argument == "--direct-expert-io")
            {
                runtime_options.flags |= ncnn::moe::RuntimeOptionDirectExpertIo;
            }
            else if (argument == "--buffered-expert-io")
            {
                runtime_options.flags |= ncnn::moe::RuntimeOptionBufferedExpertIo;
            }
            else if (argument == "--cache-warmup-runs")
            {
                cache_warmup_runs = static_cast<uint32_t>(std::stoul(ncnn::moe::require_value(argc, argv, index, "--cache-warmup-runs")));
            }
            else if (argument == "--parallel-sessions")
            {
                const uint64_t count = std::stoull(ncnn::moe::require_value(argc, argv, index, "--parallel-sessions"));
                if (count == 0 || count > 64)
                {
                    std::cerr << "--parallel-sessions must be between 1 and 64\n";
                    return 2;
                }
                parallel_sessions = static_cast<uint32_t>(count);
            }
            else if (argument == "--scheduler-expert-threads")
            {
                const uint64_t count = std::stoull(ncnn::moe::require_value(argc, argv, index, "--scheduler-expert-threads"));
                if (count == 0 || count > 1024)
                {
                    std::cerr << "--scheduler-expert-threads must be between 1 and 1024\n";
                    return 2;
                }
                scheduler_expert_threads = static_cast<uint32_t>(count);
            }
            else if (argument == "--scheduler-staging")
            {
                const std::string mode = ncnn::moe::require_value(argc, argv, index, "--scheduler-staging");
                if (mode == "auto")
                {
                    scheduler_flags = 0;
                }
                else if (mode == "force")
                {
                    scheduler_flags = ncnn::moe::SchedulerOptionForceStagedBatching;
                }
                else if (mode == "off")
                {
                    scheduler_flags = ncnn::moe::SchedulerOptionDisableStagedBatching;
                }
                else
                {
                    std::cerr << "--scheduler-staging must be auto, force, or off\n";
                    return 2;
                }
            }
            else if (argument == "--scheduler-cross-call")
            {
                runner_flags |= ncnn::moe::RunnerCrossCallScheduling;
            }
            else if (argument == "--scheduler-collection-us")
            {
                scheduler_collection_microseconds = static_cast<uint32_t>(std::stoul(ncnn::moe::require_value(argc, argv, index, "--scheduler-collection-us")));
            }
            else if (argument == "--scheduler-max-micro-batch")
            {
                scheduler_max_micro_batch = static_cast<uint32_t>(std::stoul(ncnn::moe::require_value(argc, argv, index, "--scheduler-max-micro-batch")));
            }
            else if (argument == "--cpu")
            {
                runtime_options.hybrid_mode = ncnn::moe::HybridMode::CpuOnly;
            }
            else if (argument == "--hybrid")
            {
                runtime_options.hybrid_mode = ncnn::moe::HybridMode::HybridExperts;
            }
            else if (argument == "--hybrid-prefetch")
            {
                runtime_options.hybrid_mode = ncnn::moe::HybridMode::VulkanWithCpuPrefetch;
            }
            else
            {
                prompt.push_back(std::stoi(argument));
            }
        }
        if (prompt.empty() || generation_options.max_new_tokens == 0)
        {
            std::cerr << "at least one prompt token and one generated token are required\n";
            return 2;
        }
        runtime_options.expected_concurrency = parallel_sessions;

        ncnn::moe::Runtime runtime;
        const auto load_start = std::chrono::steady_clock::now();
        auto model = runtime.load_model(std::filesystem::path(argv[1]), runtime_options);
        if (!model)
        {
            std::cerr << "load failed: " << model.error().message << '\n';
            return 1;
        }
        ncnn::moe::ModelPtr loaded_model = std::move(model).value();
        std::cout << "loaded " << loaded_model->descriptor().model_type << " in " << ncnn::moe::elapsed_seconds(load_start) << " s, backend " << ncnn::moe::hybrid_mode_name(loaded_model->hybrid_mode()) << '\n';
        const ncnn::moe::ModelMemoryPlan& memory_plan = loaded_model->memory_plan();
        std::cout << "Expert memory: " << ncnn::moe::expert_memory_mode_name(memory_plan.selected_mode) << ", " << memory_plan.estimated_expert_bytes / (1024 * 1024) << " MiB estimated";
        if (ncnn::moe::has_flag(memory_plan.flags, ncnn::moe::ModelMemoryFileBackedExperts))
        {
            std::cout << ", " << memory_plan.expert_cache_bytes / (1024 * 1024) << " MiB cache";
        }
        std::cout << '\n';
        std::cout << "Host memory budget: " << memory_plan.host_memory_budget_bytes / (1024 * 1024) << " MiB";
        if (memory_plan.physical_memory_bytes != 0)
        {
            std::cout << " of " << memory_plan.physical_memory_bytes / (1024 * 1024) << " MiB detected";
        }
        std::cout << ", dense estimate " << memory_plan.estimated_dense_bytes / (1024 * 1024) << " MiB\n";
        if (runtime.capabilities().vulkan_heap_budget_bytes != 0)
        {
            std::cout << "Vulkan heap budget: " << runtime.capabilities().vulkan_heap_budget_bytes / (1024 * 1024) << " MiB\n";
        }
        if (loaded_model->vulkan_device_index() != ncnn::moe::automatic_vulkan_device_index)
        {
            std::cout << "Vulkan model device: " << loaded_model->vulkan_device_index() << '\n';
            const uint32_t index = loaded_model->vulkan_device_index();
            if (index < runtime.capabilities().vulkan_devices.size())
            {
                std::cout << "Vulkan kernel features: " << ncnn::moe::vulkan_kernel_features(runtime.capabilities().vulkan_devices[index]) << '\n';
            }
        }
        if (!loaded_model->vulkan_device_indices().empty())
        {
            std::cout << "Vulkan model devices: ";
            for (size_t index = 0; index < loaded_model->vulkan_device_indices().size(); ++index)
            {
                if (index != 0)
                    std::cout << ',';
                std::cout << loaded_model->vulkan_device_indices()[index];
            }
            std::cout << "; layer placement: ";
            bool first = true;
            for (uint32_t device : loaded_model->vulkan_device_indices())
            {
                uint32_t count = 0;
                for (const auto& layer : loaded_model->execution_plan())
                {
                    if (layer.vulkan_device_index == device)
                        ++count;
                }
                if (count == 0)
                    continue;
                if (!first)
                    std::cout << ',';
                std::cout << device << ':' << count;
                first = false;
            }
            std::cout << '\n';
        }
        if (runtime_options.expert_gpu_cache_bytes != 0)
        {
            std::cout << "Executable Expert GPU cache: " << runtime_options.expert_gpu_cache_bytes / (1024 * 1024) << " MiB configured\n";
        }
        if (runtime_options.expert_gpu_victim_cache_bytes != 0)
        {
            std::cout << "Expert GPU victim cache: " << runtime_options.expert_gpu_victim_cache_bytes / (1024 * 1024) << " MiB configured\n";
            std::cout << "Expert GPU victim reuse-probe interval: " << runtime_options.expert_gpu_victim_reuse_probe_interval << '\n';
        }

        for (uint32_t warmup = 0; warmup < cache_warmup_runs; ++warmup)
        {
            auto warmup_session = runtime.create_session(loaded_model, session_options);
            if (!warmup_session)
            {
                std::cerr << "cache warm-up session creation failed: " << warmup_session.error().message << '\n';
                return 1;
            }
            auto warmed = warmup_session.value()->generate(prompt, generation_options);
            if (!warmed)
            {
                std::cerr << "cache warm-up generation failed: " << warmed.error().message << '\n';
                return 1;
            }
        }
        if (cache_warmup_runs != 0)
        {
            auto synchronized = runtime.synchronize_model_caches(loaded_model);
            if (!synchronized)
            {
                std::cerr << "cache warm-up synchronization failed: " << synchronized.error().message << '\n';
                return 1;
            }
        }

        std::vector<ncnn::moe::SessionPtr> active_sessions;
        active_sessions.reserve(parallel_sessions);
        for (uint32_t session_index = 0; session_index < parallel_sessions; ++session_index)
        {
            auto session = runtime.create_session(loaded_model, session_options);
            if (!session)
            {
                std::cerr << "session creation failed: " << session.error().message << '\n';
                return 1;
            }
            active_sessions.push_back(std::move(session).value());
        }

        const auto generation_start = std::chrono::steady_clock::now();
        std::vector<ncnn::moe::GenerationResult> session_generations(parallel_sessions);
        ncnn::moe::SchedulerStatistics scheduler_statistics;
        if (parallel_sessions == 1)
        {
            auto generated = active_sessions.front()->generate(prompt, generation_options, [runner_flags](const ncnn::moe::StreamToken& token) {
                if (ncnn::moe::has_flag(runner_flags, ncnn::moe::RunnerStreamTokenIds))
                {
                    std::cout << "generated token id: " << token.token_id << '\n'
                              << std::flush;
                }
                return true;
            });
            if (!generated)
            {
                std::cerr << "generation failed: " << generated.error().message << '\n';
                return 1;
            }
            session_generations.front() = std::move(generated).value();
        }
        else
        {
            ncnn::moe::SchedulerOptions scheduler_options;
            scheduler_options.worker_count = parallel_sessions;
            scheduler_options.expert_threads_per_worker = scheduler_expert_threads;
            scheduler_options.cross_call_window_microseconds = scheduler_collection_microseconds;
            scheduler_options.cross_call_max_batch_size = scheduler_max_micro_batch;
            scheduler_options.flags = scheduler_flags;
            auto scheduler = runtime.create_scheduler(scheduler_options);
            if (!scheduler)
            {
                std::cerr << "scheduler creation failed: " << scheduler.error().message << '\n';
                return 1;
            }

            std::vector<ncnn::moe::LogitsOutput> logits(parallel_sessions);
            for (uint32_t session_index = 0; session_index < parallel_sessions; ++session_index)
            {
                auto prefilled = active_sessions[session_index]->prefill(prompt);
                if (!prefilled)
                {
                    std::cerr << "parallel prefill failed: " << prefilled.error().message << '\n';
                    return 1;
                }
                logits[session_index] = std::move(prefilled).value().logits;
            }

            std::vector<bool> active(parallel_sessions, true);
            for (uint32_t token_index = 0; token_index < generation_options.max_new_tokens; ++token_index)
            {
                std::vector<ncnn::moe::DecodeBatchRequest> requests;
                std::vector<uint32_t> request_sessions;
                requests.reserve(parallel_sessions);
                request_sessions.reserve(parallel_sessions);
                for (uint32_t session_index = 0; session_index < parallel_sessions; ++session_index)
                {
                    if (!active[session_index])
                        continue;
                    auto sampled = active_sessions[session_index]->sample(logits[session_index], generation_options.sampling);
                    if (!sampled)
                    {
                        std::cerr << "parallel sampling failed: " << sampled.error().message << '\n';
                        return 1;
                    }
                    ncnn::moe::StreamToken token;
                    token.index = token_index;
                    token.token_id = sampled.value().token_id;
                    token.probability = sampled.value().probability;
                    token.is_stop_token = std::find(generation_options.stop_tokens.begin(), generation_options.stop_tokens.end(), token.token_id) != generation_options.stop_tokens.end();
                    session_generations[session_index].tokens.push_back(token);
                    if (ncnn::moe::has_flag(runner_flags, ncnn::moe::RunnerStreamTokenIds))
                    {
                        std::cout << "generated session " << session_index << " token id: " << token.token_id << '\n'
                                  << std::flush;
                    }
                    if (token.is_stop_token)
                    {
                        session_generations[session_index].stopped_by_stop_token = true;
                        active[session_index] = false;
                        continue;
                    }
                    if (token_index + 1 == generation_options.max_new_tokens)
                    {
                        continue;
                    }
                    requests.push_back({active_sessions[session_index], token.token_id});
                    request_sessions.push_back(session_index);
                }
                if (requests.empty())
                    break;
                std::vector<ncnn::moe::Result<ncnn::moe::DecodeResult>> decoded;
                if (ncnn::moe::has_flag(runner_flags, ncnn::moe::RunnerCrossCallScheduling))
                {
                    std::vector<std::future<std::vector<ncnn::moe::Result<ncnn::moe::DecodeResult>>>> futures;
                    futures.reserve(requests.size());
                    for (ncnn::moe::DecodeBatchRequest& request : requests)
                    {
                        futures.push_back(scheduler.value()->submit_decode({std::move(request)}));
                    }
                    decoded.reserve(futures.size());
                    for (auto& future : futures)
                    {
                        auto result = future.get();
                        if (result.size() != 1)
                        {
                            std::cerr << "cross-call decode returned an invalid result count\n";
                            return 1;
                        }
                        decoded.push_back(std::move(result.front()));
                    }
                }
                else
                {
                    decoded = scheduler.value()->submit_decode(std::move(requests)).get();
                }
                for (size_t request_index = 0; request_index < decoded.size(); ++request_index)
                {
                    if (!decoded[request_index])
                    {
                        std::cerr << "parallel decode failed: " << decoded[request_index].error().message << '\n';
                        return 1;
                    }
                    logits[request_sessions[request_index]] = std::move(decoded[request_index]).value().logits;
                }
            }
            scheduler_statistics = scheduler.value()->statistics();
        }

        ncnn::moe::GenerationResult generation;
        for (ncnn::moe::GenerationResult& session_generation : session_generations)
        {
            generation.tokens.insert(generation.tokens.end(), session_generation.tokens.begin(), session_generation.tokens.end());
        }
        const double generation_seconds = ncnn::moe::elapsed_seconds(generation_start);
        const ncnn::moe::SessionStatistics statistics = active_sessions.front()->statistics();
        std::cout << "generated " << generation.tokens.size() << " token(s) in " << generation_seconds << " s\n";
        std::cout << "Parallel sessions: " << parallel_sessions << ", aggregate throughput: " << generation.tokens.size() / generation_seconds << " token/s\n";
        if (parallel_sessions > 1)
        {
            std::cout << "Scheduler: " << scheduler_statistics.worker_count << " worker(s), " << scheduler_statistics.expert_threads_per_worker << " Expert thread(s) per worker, " << scheduler_statistics.max_in_flight << " max in flight\n";
            std::cout << "Scheduler staging: " << scheduler_statistics.staged_batches << " batch(es), " << scheduler_statistics.staged_requests << " request(s), " << scheduler_statistics.staging_bypassed_batches << " bypassed batch(es), "
                      << scheduler_statistics.logical_expert_batches << " logical Expert batch(es) -> " << scheduler_statistics.physical_expert_batches << " physical, " << scheduler_statistics.coalesced_expert_routes
                      << " coalesced route(s), max " << scheduler_statistics.max_coalesced_expert_batch_size << " row(s)\n";
            const double adaptive_staged_ms = scheduler_statistics.adaptive_staged_observations == 0 ? 0.0 : static_cast<double>(scheduler_statistics.adaptive_staged_time_microseconds) / scheduler_statistics.adaptive_staged_observations / 1000.0;
            const double adaptive_independent_ms = scheduler_statistics.adaptive_independent_observations == 0 ? 0.0 : static_cast<double>(scheduler_statistics.adaptive_independent_time_microseconds) / scheduler_statistics.adaptive_independent_observations / 1000.0;
            std::cout << "Scheduler adaptive policy: " << scheduler_statistics.adaptive_staged_decisions << " staged decision(s), " << scheduler_statistics.adaptive_independent_decisions << " independent decision(s), "
                      << scheduler_statistics.adaptive_probe_decisions << " probe(s), " << scheduler_statistics.adaptive_policy_switches << " switch(es), " << scheduler_statistics.adaptive_staged_observations << '/'
                      << scheduler_statistics.adaptive_independent_observations << " staged/independent observation(s), " << adaptive_staged_ms << '/' << adaptive_independent_ms << " mean ms/request\n";
            std::cout << "Scheduler adaptive phases: " << scheduler_statistics.adaptive_resident_decisions << '/' << scheduler_statistics.adaptive_mixed_decisions << '/' << scheduler_statistics.adaptive_storage_decisions
                      << " resident/mixed/storage decision(s), " << scheduler_statistics.adaptive_resident_observations << '/' << scheduler_statistics.adaptive_mixed_observations << '/' << scheduler_statistics.adaptive_storage_observations
                      << " observation(s), " << scheduler_statistics.adaptive_phase_changes << " phase change(s), " << scheduler_statistics.adaptive_noisy_switch_rejections << " noisy switch rejection(s)\n";
            std::cout << "Scheduler cross-call: " << scheduler_statistics.cross_call_collected_batches << " collected batch(es), " << scheduler_statistics.cross_call_collected_requests << " collected request(s), "
                      << scheduler_statistics.cross_call_collection_probes << " probe(s), " << scheduler_statistics.cross_call_collection_timeouts << " timeout(s), " << scheduler_statistics.cross_call_collection_bypasses << " bypass(es), "
                      << scheduler_statistics.cross_call_collection_wait_microseconds << " us waiting, max batch " << scheduler_statistics.max_cross_call_batch_size << ", max pending " << scheduler_statistics.max_cross_call_pending << '\n';
        }
        std::cout << "Vulkan linear dispatches: " << statistics.vulkan_linear_dispatches << '\n';
        std::cout << "Vulkan attention blocks: " << statistics.vulkan_attention_blocks << '\n';
        std::cout << "Vulkan compute submissions: " << statistics.vulkan_compute_submissions << '\n';
        std::cout << "Vulkan async submissions: " << statistics.vulkan_async_submissions << '\n';
        std::cout << "Vulkan batch transfers: " << statistics.vulkan_batch_uploads << " upload(s), " << statistics.vulkan_batch_downloads << " download(s)\n";
        std::cout << "Vulkan auxiliary uploads: " << statistics.vulkan_auxiliary_uploads << " upload(s), " << statistics.vulkan_auxiliary_upload_bytes << " bytes\n";
        std::cout << "Vulkan staging slots: " << statistics.vulkan_staging_slot_resizes << " resize(s), " << statistics.vulkan_staging_slot_reuses << " reuse(s), " << statistics.vulkan_staging_slot_acquisitions << " acquisition(s), "
                  << statistics.vulkan_staging_slot_contentions << " contention(s)\n";
        std::cout << "Vulkan command buffer reuses: " << statistics.vulkan_command_buffer_reuses << '\n';
        std::cout << "Vulkan attention fusion: " << statistics.vulkan_attention_qkv_rope_fusions << " QKV+RoPE block(s), " << statistics.vulkan_attention_qkv_ring_fusions << " QKV->ring block(s), "
                  << statistics.vulkan_attention_decode_sdpa_fusions << " Decode-SDPA block(s)\n";
        std::cout << "Vulkan KV ring: " << statistics.vulkan_kv_ring_appends << " append(s), " << statistics.vulkan_kv_ring_resizes << " resize(s), " << statistics.vulkan_kv_ring_wrapped_views << " wrapped view(s)\n";
        std::cout << "Attention time: " << statistics.attention_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Router time: " << statistics.router_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert time: " << statistics.expert_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert weight demand: " << statistics.expert_batch_weight_bytes << " batched bytes, " << statistics.expert_route_weight_bytes << " route bytes\n";
        std::cout << "Expert cache wait time: " << statistics.expert_cache_wait_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert cache management time: " << statistics.expert_cache_management_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert engine wall time: " << statistics.expert_engine_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert compute wall time: " << statistics.expert_compute_time_microseconds / 1000.0 << " ms\n";
        const uint64_t expert_accounted_time = statistics.expert_compute_time_microseconds + statistics.expert_regroup_time_microseconds + statistics.expert_combine_time_microseconds + statistics.expert_cache_wait_time_microseconds
                                               + statistics.expert_cache_management_time_microseconds;
        const uint64_t expert_orchestration_time = statistics.expert_time_microseconds > expert_accounted_time ? statistics.expert_time_microseconds - expert_accounted_time : 0;
        std::cout << "Expert orchestration wall time: " << expert_orchestration_time / 1000.0 << " ms\n";
        std::cout << "Expert regroup time: " << statistics.expert_regroup_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert combine time: " << statistics.expert_combine_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Embedding time: " << statistics.embedding_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Final norm time: " << statistics.final_norm_time_microseconds / 1000.0 << " ms\n";
        std::cout << "LM head time: " << statistics.lm_head_time_microseconds / 1000.0 << " ms\n";
        std::cout << "Expert prefetches: " << statistics.expert_prefetches << " (" << statistics.expert_prefetch_bytes << " bytes hinted)\n";
        std::cout << "Expert route prediction: " << statistics.expert_route_predictions << " prediction(s), " << statistics.expert_route_prediction_matches << " match(es), " << statistics.expert_route_prediction_cache_hits
                  << " cache-ready, " << statistics.expert_route_prediction_cache_misses << " not-ready\n";
        std::cout << "Expert cache: " << statistics.expert_cache_hits << " hit(s), " << statistics.expert_cache_misses << " miss(es), " << statistics.expert_cache_evictions << " eviction(s), " << statistics.expert_cache_bytes_read
                  << " bytes read, " << statistics.expert_cache_resident_bytes << " bytes resident, " << statistics.expert_cache_queued_reads << " queued read(s), " << statistics.expert_cache_speculative_reads << " speculative read(s), "
                  << statistics.expert_cache_cancelled_speculative_reads << " cancelled speculative read(s), " << statistics.expert_cache_dropped_speculative_admissions << " dropped speculative admission(s)\n";
        std::cout << "Expert ARC: T1 " << statistics.expert_cache_arc_recent_bytes << " bytes, T2 " << statistics.expert_cache_arc_frequent_bytes << " bytes, target T1 " << statistics.expert_cache_arc_recent_target_bytes << " bytes, B1 "
                  << statistics.expert_cache_arc_recent_ghost_bytes << " bytes, B2 " << statistics.expert_cache_arc_frequent_ghost_bytes << " bytes, B1 " << statistics.expert_cache_arc_recent_ghost_hits << " hit(s), B2 "
                  << statistics.expert_cache_arc_frequent_ghost_hits << " hit(s)\n";
        std::cout << "Expert mmap: " << statistics.expert_cache_mapped_ranges << " range(s), " << statistics.expert_cache_mapped_bytes << " bytes\n";
        const char* expert_read_policy = "sampling";
        if (statistics.expert_cache_read_policy == 1)
            expert_read_policy = "buffered";
        else if (statistics.expert_cache_read_policy == 2)
            expert_read_policy = "direct";
        else if (statistics.expert_cache_read_policy == 3)
            expert_read_policy = "mmap";
        std::cout << "Expert I/O policy: " << expert_read_policy << ", " << statistics.expert_cache_direct_read_ranges << " direct range(s), " << statistics.expert_cache_direct_read_bytes << " direct byte(s), "
                  << statistics.expert_cache_direct_read_fallbacks << " fallback(s), " << statistics.expert_cache_buffered_read_ranges << " buffered range(s), " << statistics.expert_cache_buffered_read_bytes << " buffered byte(s)\n";
        std::cout << "Expert GPU execution cache: " << statistics.expert_gpu_cache_hits << " hit(s), " << statistics.expert_gpu_cache_misses << " miss(es), " << statistics.expert_gpu_cache_admissions << " admission(s), "
                  << statistics.expert_gpu_cache_stores << " store(s), " << statistics.expert_gpu_cache_evictions << " eviction(s), " << statistics.expert_gpu_cache_dropped_admissions << " dropped admission(s), "
                  << statistics.expert_gpu_cache_bytes_uploaded << " bytes uploaded, " << statistics.expert_gpu_cache_resident_bytes << " bytes resident, " << statistics.expert_gpu_cache_pending_bytes << " bytes pending\n";
        std::cout << "Expert GPU victim cache: " << statistics.expert_gpu_victim_cache_hits << " hit(s), " << statistics.expert_gpu_victim_cache_misses << " miss(es), " << statistics.expert_gpu_victim_cache_admissions << " admission(s), "
                  << statistics.expert_gpu_victim_cache_filtered_admissions << " filtered admission(s), " << statistics.expert_gpu_victim_cache_reused_admissions << " reused admission(s), "
                  << statistics.expert_gpu_victim_cache_probe_admissions << " probe admission(s), " << statistics.expert_gpu_victim_cache_stores << " store(s), " << statistics.expert_gpu_victim_cache_evictions << " eviction(s), "
                  << statistics.expert_gpu_victim_cache_dropped_admissions << " dropped admission(s), " << statistics.expert_gpu_victim_cache_restore_failures << " restore failure(s), " << statistics.expert_gpu_victim_cache_bytes_uploaded
                  << " bytes uploaded, " << statistics.expert_gpu_victim_cache_bytes_downloaded << " bytes downloaded, " << statistics.expert_gpu_victim_cache_restore_time_microseconds / 1000.0 << " ms restoring, "
                  << statistics.expert_gpu_victim_cache_mapped_stores << " mapped store(s), " << statistics.expert_gpu_victim_cache_mapped_restores << " mapped restore(s), " << statistics.expert_gpu_victim_cache_resident_bytes
                  << " bytes resident, " << statistics.expert_gpu_victim_cache_pending_bytes << " bytes pending\n";
        std::cout << "Expert GPU execution: " << statistics.expert_gpu_executions << " execution(s), " << statistics.expert_gpu_execution_failures << " failure(s), " << statistics.expert_gpu_cpu_preferred << " CPU-preferred decision(s), "
                  << statistics.expert_gpu_execution_time_microseconds / 1000.0 << " ms executing\n";
        std::cout << "Expert GPU device source: " << statistics.expert_gpu_device_source_hits << " hit(s), " << statistics.expert_gpu_device_source_misses << " miss(es), " << statistics.expert_gpu_device_source_executions
                  << " execution(s), " << statistics.expert_gpu_device_source_execution_failures << " failure(s)\n";
        std::cout << "Expert GPU ARC: " << statistics.expert_gpu_arc_recent_bytes << " recent byte(s), " << statistics.expert_gpu_arc_frequent_bytes << " frequent byte(s), " << statistics.expert_gpu_arc_recent_target_bytes
                  << " recent target byte(s), " << statistics.expert_gpu_arc_recent_ghost_bytes << " recent ghost byte(s), " << statistics.expert_gpu_arc_frequent_ghost_bytes << " frequent ghost byte(s)\n";
        std::cout << "Parallel CPU experts: " << (ncnn::moe::has_flag(runtime.capabilities().flags, ncnn::moe::RuntimeCapabilityOpenmpExperts) ? "OpenMP enabled" : "single-thread fallback") << '\n';
        std::cout << "CPU topology: " << runtime.capabilities().physical_cpu_core_count << " physical core(s), " << runtime.capabilities().logical_cpu_count << " logical processor(s), " << runtime.capabilities().openmp_thread_count
                  << " OpenMP thread(s)\n";
        std::cout << "CPU ISA capabilities: " << runtime.capabilities().cpu_isa << '\n';
        std::cout << "MXFP4 CPU kernel: " << runtime.capabilities().mxfp4_kernel << '\n';
        std::cout << "MXFP4 decode row-pair group: " << runtime.capabilities().mxfp4_decode_row_pair_group_size << '\n';
        std::cout << "Activation CPU kernel: " << runtime.capabilities().activation_kernel << '\n';
        std::cout << "MXFP4 decode GEMV rows: " << statistics.mxfp4_decode_gemv_rows << '\n';
        std::cout << "MXFP4 prefill GEMM rows: " << statistics.mxfp4_prefill_gemm_rows << '\n';
        std::cout << "MXFP4 paired rows: " << statistics.mxfp4_paired_rows << '\n';
        std::cout << "MXFP4 fused Gate/Up rows: " << statistics.mxfp4_fused_gate_up_rows << '\n';
        std::cout << "Parallel expert tasks: " << statistics.expert_parallel_tasks << '\n';
        const uint64_t vulkan_budget = runtime.capabilities().vulkan_heap_budget_bytes;
        if (vulkan_budget != 0)
        {
            for (uint64_t divisor : {UINT64_C(8), UINT64_C(4), UINT64_C(3), UINT64_C(2)})
            {
                const uint64_t capacity = vulkan_budget / divisor;
                const ncnn::moe::ExpertHotsetEstimate estimate = loaded_model->expert_store().estimate_hotset(capacity);
                const double batch_coverage = estimate.requested_batch_weight_bytes == 0 ? 0.0 : 100.0 * static_cast<double>(estimate.covered_batch_weight_bytes) / static_cast<double>(estimate.requested_batch_weight_bytes);
                const double route_coverage = estimate.requested_route_weight_bytes == 0 ? 0.0 : 100.0 * static_cast<double>(estimate.covered_route_weight_bytes) / static_cast<double>(estimate.requested_route_weight_bytes);
                std::cout << "Expert static hotset at " << capacity / (1024 * 1024) << " MiB: " << estimate.resident_expert_count << '/' << estimate.active_expert_count << " Expert(s), " << estimate.resident_bytes << " bytes resident, "
                          << batch_coverage << "% batch-byte coverage, " << route_coverage << "% route-byte coverage\n";
            }
        }
        std::cout << "generated token ids:";
        for (const ncnn::moe::StreamToken& token : generation.tokens)
            std::cout << ' ' << token.token_id;
        std::cout << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "invalid argument: " << error.what() << '\n';
        return 2;
    }
}
