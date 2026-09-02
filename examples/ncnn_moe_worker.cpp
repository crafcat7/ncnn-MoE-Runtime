#include "internal/json_line.h"
#include "internal/gpu_telemetry.h"
#include "ncnn/moe/runtime.h"
#include "engine/cpu_features.h"
#include "kernels/cpu_bfloat16.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace ncnn {
namespace moe {

class JsonObject
{
private:
    std::string value_ = "{";
    bool first_ = true;

    void separator()
    {
        if (!first_)
            value_.push_back(',');
        first_ = false;
    }

    void key(std::string_view name)
    {
        separator();
        value_ += json_escape(name);
        value_.push_back(':');
    }

public:
    void add_string(std::string_view name, std::string_view value)
    {
        key(name);
        value_ += json_escape(value);
    }

    void add_uint(std::string_view name, uint64_t value)
    {
        key(name);
        value_ += std::to_string(value);
    }

    void add_optional_uint(std::string_view name, const std::optional<uint64_t>& value)
    {
        if (value)
            add_uint(name, *value);
        else
            add_null(name);
    }

    void add_int(std::string_view name, int64_t value)
    {
        key(name);
        value_ += std::to_string(value);
    }

    void add_bool(std::string_view name, bool value)
    {
        key(name);
        value_ += value ? "true" : "false";
    }

    void add_double(std::string_view name, double value)
    {
        key(name);
        if (!std::isfinite(value))
        {
            value_ += "null";
            return;
        }
        std::ostringstream stream;
        stream << std::setprecision(10) << value;
        value_ += stream.str();
    }

    void add_optional_double(std::string_view name, const std::optional<double>& value)
    {
        if (value)
            add_double(name, *value);
        else
            add_null(name);
    }

    void add_null(std::string_view name)
    {
        key(name);
        value_ += "null";
    }

    void add_raw(std::string_view name, std::string_view value)
    {
        key(name);
        value_.append(value);
    }

    [[nodiscard]] std::string finish()
    {
        value_.push_back('}');
        return std::move(value_);
    }
};

static std::string uint_array(const std::vector<uint32_t>& values)
{
    std::string result = "[";
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0)
            result.push_back(',');
        result += std::to_string(values[index]);
    }
    result.push_back(']');
    return result;
}

static const char* error_code_name(ErrorCode code) noexcept
{
    switch (code)
    {
    case ErrorCode::InvalidArgument: return "invalid_argument";
    case ErrorCode::IoError: return "io_error";
    case ErrorCode::InvalidModel: return "invalid_model";
    case ErrorCode::UnsupportedModel: return "unsupported_model";
    case ErrorCode::InternalError: return "internal_error";
    }
    return "unknown";
}

static const char* hybrid_mode_name(HybridMode mode) noexcept
{
    switch (mode)
    {
    case HybridMode::CpuOnly: return "cpu";
    case HybridMode::HybridExperts: return "hybrid";
    case HybridMode::Auto: return "auto";
    }
    return "unknown";
}

static const char* expert_memory_mode_name(ExpertMemoryMode mode) noexcept
{
    switch (mode)
    {
    case ExpertMemoryMode::Auto: return "auto";
    case ExpertMemoryMode::Eager: return "eager";
    case ExpertMemoryMode::OnDemand: return "on-demand";
    }
    return "unknown";
}

static const char* cpu_packed_weight_mode_name(
    CpuPackedWeightMode mode) noexcept
{
    return mode == CpuPackedWeightMode::Enabled ? "on" : "off";
}

static const char* vulkan_type_name(VulkanDeviceType type) noexcept
{
    switch (type)
    {
    case VulkanDeviceType::Discrete: return "discrete";
    case VulkanDeviceType::Integrated: return "integrated";
    case VulkanDeviceType::Virtual: return "virtual";
    case VulkanDeviceType::Cpu: return "cpu";
    case VulkanDeviceType::Unknown: return "unknown";
    }
    return "unknown";
}

static std::string gpu_infos_json(const std::vector<VulkanDeviceCapabilities>& gpu_infos)
{
    std::string result = "[";
    for (std::size_t i = 0; i < gpu_infos.size(); ++i)
    {
        if (i != 0)
            result.push_back(',');
        const VulkanDeviceCapabilities& gpu_info = gpu_infos[i];
        JsonObject item;
        item.add_uint("index", gpu_info.device_index);
        item.add_string("name", gpu_info.device_name);
        item.add_string("type", vulkan_type_name(gpu_info.type));
        item.add_uint("vendor_id", gpu_info.vendor_id);
        item.add_uint("device_id", gpu_info.device_id);
        item.add_uint("rough_score", gpu_info.rough_score);
        item.add_uint("compute_queue_count", gpu_info.compute_queue_count);
        item.add_uint("transfer_queue_count", gpu_info.transfer_queue_count);
        item.add_uint("heap_budget_bytes", gpu_info.heap_budget);
        item.add_uint("heap_usage_bytes", gpu_info.heap_usage);
        item.add_uint("heap_available_bytes", gpu_info.heap_available);
        item.add_uint("flags", gpu_info.flags);
        result += item.finish();
    }
    result.push_back(']');
    return result;
}

static std::string expert_metrics_json(const RuntimeMetricCounters& counters)
{
    JsonObject result;
    result.add_uint("cache_hit", counters.expert_cache_hits);
    result.add_uint("cache_miss", counters.expert_cache_misses);
    result.add_uint("io_bytes", counters.expert_io_bytes);
    result.add_uint("cache_resident_bytes", counters.expert_cache_resident_size);
    result.add_uint("gpu_cache_hit", counters.expert_gpu_cache_hits);
    result.add_uint("gpu_cache_miss", counters.expert_gpu_cache_misses);
    result.add_uint("gpu_cache_admissions", counters.expert_gpu_cache_admissions);
    result.add_uint("gpu_cache_stores", counters.expert_gpu_cache_stores);
    result.add_uint("gpu_cache_dropped_admissions", counters.expert_gpu_cache_dropped_admissions);
    result.add_uint("gpu_cache_resident_bytes", counters.expert_gpu_cache_resident_size);
    result.add_uint("gpu_cache_pending_bytes", counters.expert_gpu_cache_pending_size);
    result.add_uint("gpu_executions", counters.expert_gpu_executions);
    result.add_uint("gpu_execution_failures", counters.expert_gpu_execution_failures);
    result.add_uint("gpu_cpu_preferred", counters.expert_gpu_cpu_preferred);
    result.add_uint("gpu_route_aggregation_batches", counters.expert_gpu_route_aggregation_batches);
    result.add_uint("gpu_route_aggregation_routes", counters.expert_gpu_route_aggregation_routes);
    result.add_uint("gpu_route_aggregation_bytes_saved", counters.expert_gpu_route_aggregation_bytes_saved);
    const uint64_t cache_requests = counters.expert_cache_hits + counters.expert_cache_misses;
    if (cache_requests == 0)
        result.add_null("cache_hit_rate");
    else
        result.add_double("cache_hit_rate", static_cast<double>(counters.expert_cache_hits) / static_cast<double>(cache_requests));
    return result.finish();
}

static std::string cpu_metrics_json(const RuntimeMetricCounters& counters)
{
    JsonObject result;
    result.add_uint("expert_compute_time_microseconds", counters.expert_compute_time_microseconds);
    return result.finish();
}

static std::string gpu_metrics_json(const RuntimeMetricCounters& counters, bool available)
{
    JsonObject result;
    result.add_bool("available", available);
    if (available)
    {
        result.add_uint("submit_count", counters.gpu_submit_count);
        result.add_uint("wait_time_microseconds", counters.gpu_wait_time_microseconds);
    }
    else
    {
        result.add_null("submit_count");
        result.add_null("wait_time_microseconds");
    }
    if (available && counters.gpu_kernel_time_available)
        result.add_uint("kernel_time_microseconds", counters.gpu_kernel_time_microseconds);
    else
        result.add_null("kernel_time_microseconds");
    result.add_bool("kernel_time_available", counters.gpu_kernel_time_available);
    result.add_uint("linear_dispatches", counters.vulkan_linear_dispatches);
    result.add_uint("attention_blocks", counters.vulkan_attention_blocks);
    result.add_uint("gated_delta_fusions", counters.vulkan_gated_delta_fusions);
    result.add_uint("gated_delta_submissions", counters.vulkan_gated_delta_submissions);
    result.add_uint("batch_uploads", counters.vulkan_batch_uploads);
    result.add_uint("batch_downloads", counters.vulkan_batch_downloads);
    result.add_uint("attention_qkv_rope_fusions", counters.vulkan_attention_qkv_rope_fusions);
    result.add_uint("attention_device_rope_fusions", counters.vulkan_attention_device_rope_fusions);
    result.add_uint("attention_qkv_ring_fusions", counters.vulkan_attention_qkv_ring_fusions);
    result.add_uint("attention_qkv_rope_pipeline_failures", counters.vulkan_attention_qkv_rope_pipeline_failures);
    result.add_uint("attention_qkv_rope_shape_failures", counters.vulkan_attention_qkv_rope_shape_failures);
    result.add_uint("attention_qkv_rope_source_failures", counters.vulkan_attention_qkv_rope_source_failures);
    result.add_uint("attention_qkv_rope_norm_failures", counters.vulkan_attention_qkv_rope_norm_failures);
    result.add_uint("attention_qkv_rope_ring_failures", counters.vulkan_attention_qkv_rope_ring_failures);
    result.add_uint("attention_qkv_rope_allocation_failures", counters.vulkan_attention_qkv_rope_allocation_failures);
    result.add_uint("attention_precondition_failures", counters.vulkan_attention_precondition_failures);
    result.add_uint("attention_staging_failures", counters.vulkan_attention_staging_failures);
    result.add_uint("attention_norm_failures", counters.vulkan_attention_norm_failures);
    result.add_uint("attention_qkv_failures", counters.vulkan_attention_qkv_failures);
    result.add_uint("attention_cache_failures", counters.vulkan_attention_cache_failures);
    result.add_uint("attention_sdpa_failures", counters.vulkan_attention_sdpa_failures);
    result.add_uint("attention_projection_failures", counters.vulkan_attention_projection_failures);
    result.add_uint("attention_output_failures", counters.vulkan_attention_output_failures);
    result.add_uint("attention_submit_failures", counters.vulkan_attention_submit_failures);
    result.add_uint("attention_cache_materializations", counters.vulkan_attention_cache_materializations);
    result.add_uint("attention_cpu_fallbacks", counters.vulkan_attention_cpu_fallbacks);
    if (available)
        result.add_string("reason", counters.gpu_kernel_time_available ? "" : "gpu_expert_execution_not_observed");
    else
        result.add_string("reason", "runtime_backend_cpu_only");
    return result.finish();
}

static std::string generation_timing_json(const GenerationTimingMetrics& timing)
{
    JsonObject result;
    result.add_bool("active", timing.active);
    result.add_uint("input_tokens", timing.input_tokens);
    result.add_uint("output_tokens", timing.output_tokens);
    result.add_uint("elapsed_microseconds", timing.elapsed_microseconds);
    result.add_optional_uint("prompt_elapsed_microseconds", timing.prompt_elapsed_microseconds);
    result.add_optional_uint("generation_elapsed_microseconds", timing.generation_elapsed_microseconds);
    result.add_optional_double("prompt_tokens_per_second", timing.prompt_tokens_per_second);
    result.add_optional_double("generation_tokens_per_second", timing.generation_tokens_per_second);
    result.add_optional_uint("ttft_microseconds", timing.ttft_microseconds);
    result.add_optional_double("tpot_microseconds", timing.tpot_microseconds);
    result.add_optional_double("decode_tokens_per_second", timing.decode_tokens_per_second);
    return result.finish();
}

static std::string stats_json(const SessionMetrics& metrics)
{
    JsonObject result;
    result.add_raw("generation", expert_metrics_json(metrics.generation));
    result.add_raw("generation_cpu", cpu_metrics_json(metrics.generation));
    result.add_raw("generation_gpu", gpu_metrics_json(metrics.generation, metrics.gpu_available));
    result.add_raw("cumulative", expert_metrics_json(metrics.cumulative));
    result.add_raw("cumulative_cpu", cpu_metrics_json(metrics.cumulative));
    result.add_raw("cumulative_gpu", gpu_metrics_json(metrics.cumulative, metrics.gpu_available));
    result.add_raw("timing", generation_timing_json(metrics.timing));
    result.add_bool("gpu_available", metrics.gpu_available);
    result.add_uint("kv_cache_logical_bytes", metrics.cumulative.kv_cache_logical_size);
    result.add_uint("kv_cache_allocated_bytes", metrics.cumulative.kv_cache_allocated_size);

    return result.finish();
}

static std::string memory_statistics_json(const MemoryManagerStatistics& statistics)
{
    JsonObject result;
    result.add_uint("registered_tensors", statistics.registered_tensors);
    result.add_uint("cpu_bytes", statistics.cpu_size);
    result.add_uint("vulkan_bytes", statistics.vulkan_size);
    result.add_uint("shared_bytes", statistics.shared_size);
    result.add_uint("transitions", statistics.transitions);
    result.add_uint("tensor_uses", statistics.tensor_uses);
    return result.finish();
}

struct ProcessTelemetry
{
    std::optional<double> cpu_percent;
    std::optional<uint64_t> rss_bytes;
    std::optional<uint64_t> read_bytes;
    std::optional<uint64_t> write_bytes;
    std::string cpu_reason;
    std::string rss_reason;
    std::string io_reason;
};

class ProcessTelemetrySampler
{
private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point previous_wall_ = Clock::now();
    uint64_t previous_cpu_microseconds_ = 0;
    bool has_previous_cpu_ = false;

#if defined(_WIN32)
    static uint64_t filetime_microseconds(const FILETIME& value)
    {
        ULARGE_INTEGER integer;
        integer.LowPart = value.dwLowDateTime;
        integer.HighPart = value.dwHighDateTime;
        return static_cast<uint64_t>(integer.QuadPart / 10);
    }
#endif

public:
    ProcessTelemetry sample()
    {
        ProcessTelemetry result;
#if defined(_WIN32)
        FILETIME creation_time{};
        FILETIME exit_time{};
        FILETIME kernel_time{};
        FILETIME user_time{};
        if (GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time))
        {
            const uint64_t cpu_microseconds = filetime_microseconds(kernel_time) + filetime_microseconds(user_time);
            const auto now = Clock::now();
            const uint64_t wall_microseconds = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - previous_wall_).count());
            if (has_previous_cpu_ && wall_microseconds != 0)
                result.cpu_percent = 100.0 * static_cast<double>(cpu_microseconds - previous_cpu_microseconds_) / static_cast<double>(wall_microseconds);
            else
                result.cpu_reason = "warming_up";
            previous_cpu_microseconds_ = cpu_microseconds;
            previous_wall_ = now;
            has_previous_cpu_ = true;
        }
        else
        {
            result.cpu_reason = "GetProcessTimes_failed";
        }
        PROCESS_MEMORY_COUNTERS memory{};
        memory.cb = sizeof(memory);
        if (GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
            result.rss_bytes = static_cast<uint64_t>(memory.WorkingSetSize);
        else
            result.rss_reason = "GetProcessMemoryInfo_failed";
        IO_COUNTERS io{};
        if (GetProcessIoCounters(GetCurrentProcess(), &io))
        {
            result.read_bytes = static_cast<uint64_t>(io.ReadTransferCount);
            result.write_bytes = static_cast<uint64_t>(io.WriteTransferCount);
        }
        else
        {
            result.io_reason = "GetProcessIoCounters_failed";
        }
#elif defined(__linux__)
        struct rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0)
        {
            const uint64_t cpu_microseconds = static_cast<uint64_t>(usage.ru_utime.tv_sec) * 1000000
                                              + static_cast<uint64_t>(usage.ru_utime.tv_usec)
                                              + static_cast<uint64_t>(usage.ru_stime.tv_sec) * 1000000
                                              + static_cast<uint64_t>(usage.ru_stime.tv_usec);
            const auto now = Clock::now();
            const uint64_t wall_microseconds = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - previous_wall_).count());
            if (has_previous_cpu_ && wall_microseconds != 0)
                result.cpu_percent = 100.0 * static_cast<double>(cpu_microseconds - previous_cpu_microseconds_) / static_cast<double>(wall_microseconds);
            else
                result.cpu_reason = "warming_up";
            previous_cpu_microseconds_ = cpu_microseconds;
            previous_wall_ = now;
            has_previous_cpu_ = true;
            result.rss_bytes = static_cast<uint64_t>(usage.ru_maxrss) * 1024;
            result.rss_reason = "peak_rss";
        }
        else
        {
            result.cpu_reason = "getrusage_failed";
            result.rss_reason = "getrusage_failed";
        }
        std::ifstream io_stream("/proc/self/io");
        std::string line;
        while (std::getline(io_stream, line))
        {
            const auto separator = line.find(':');
            if (separator == std::string::npos)
                continue;
            uint64_t value = 0;
            try
            {
                value = std::stoull(line.substr(separator + 1));
            }
            catch (const std::exception&)
            {
                continue;
            }
            if (line.starts_with("read_bytes"))
                result.read_bytes = value;
            else if (line.starts_with("write_bytes"))
                result.write_bytes = value;
        }
        if (!result.read_bytes || !result.write_bytes)
            result.io_reason = "/proc/self/io_unavailable";
#else
        result.cpu_reason = "process_cpu_provider_unavailable";
        result.rss_reason = "process_memory_provider_unavailable";
        result.io_reason = "process_io_provider_unavailable";
#endif
        return result;
    }
};

static std::string process_telemetry_json(ProcessTelemetrySampler& sampler)
{
    const ProcessTelemetry process = sampler.sample();
    JsonObject result;
    if (process.cpu_percent)
        result.add_double("cpu_percent", *process.cpu_percent);
    else
        result.add_null("cpu_percent");
    if (process.rss_bytes)
        result.add_uint("rss_bytes", *process.rss_bytes);
    else
        result.add_null("rss_bytes");
    if (process.read_bytes)
        result.add_uint("read_bytes", *process.read_bytes);
    else
        result.add_null("read_bytes");
    if (process.write_bytes)
        result.add_uint("write_bytes", *process.write_bytes);
    else
        result.add_null("write_bytes");
    result.add_null("io_utilization_percent");
    result.add_string("io_utilization_reason", "process_provider_exposes_counters_only");
    result.add_string("cpu_reason", process.cpu_percent ? "" : process.cpu_reason);
    result.add_string("rss_reason", process.rss_bytes ? "" : process.rss_reason);
    result.add_string("io_reason", process.read_bytes && process.write_bytes ? "" : process.io_reason);
    return result.finish();
}

static std::string gpu_telemetry_json(const GpuTelemetrySampler& sampler)
{
    const GpuTelemetrySample telemetry = sampler.sample();
    JsonObject result;
    result.add_bool("active", telemetry.active);
    if (telemetry.device_index)
        result.add_uint("device_index", *telemetry.device_index);
    else
        result.add_null("device_index");
    result.add_uint("vendor_id", telemetry.vendor_id);
    result.add_string("device_name", telemetry.device_name);
    result.add_string("provider", telemetry.provider);
    if (telemetry.utilization_percent)
        result.add_double("utilization_percent", *telemetry.utilization_percent);
    else
        result.add_null("utilization_percent");
    if (telemetry.memory_utilization_percent)
        result.add_double("memory_utilization_percent", *telemetry.memory_utilization_percent);
    else
        result.add_null("memory_utilization_percent");
    if (telemetry.used_bytes)
        result.add_uint("used_bytes", *telemetry.used_bytes);
    else
        result.add_null("used_bytes");
    if (telemetry.total_bytes)
        result.add_uint("total_bytes", *telemetry.total_bytes);
    else
        result.add_null("total_bytes");
    result.add_string("reason", telemetry.reason);
    return result.finish();
}

static std::string runtime_metrics_json(
    ProcessTelemetrySampler& sampler,
    const GpuTelemetrySampler& gpu_sampler,
    const SessionMetrics& metrics)
{
    JsonObject result;
    result.add_optional_double("prompt_tok_per_second", metrics.timing.prompt_tokens_per_second);
    result.add_optional_double("prompt_tokens_per_second", metrics.timing.prompt_tokens_per_second);
    result.add_optional_double("generation_tok_per_second", metrics.timing.generation_tokens_per_second);
    result.add_optional_double("generation_tokens_per_second", metrics.timing.generation_tokens_per_second);
    result.add_optional_double("decode_tok_per_second", metrics.timing.decode_tokens_per_second);
    result.add_optional_double("tokens_per_second", metrics.timing.decode_tokens_per_second);
    result.add_optional_double("token_per_second", metrics.timing.decode_tokens_per_second);
    result.add_optional_uint("prompt_elapsed_microseconds", metrics.timing.prompt_elapsed_microseconds);
    result.add_optional_uint("generation_elapsed_microseconds", metrics.timing.generation_elapsed_microseconds);
    result.add_optional_uint("ttft_microseconds", metrics.timing.ttft_microseconds);
    result.add_optional_double("tpot_microseconds", metrics.timing.tpot_microseconds);
    result.add_uint("input_tokens", metrics.timing.input_tokens);
    result.add_uint("output_tokens", metrics.timing.output_tokens);
    result.add_uint("elapsed_microseconds", metrics.timing.elapsed_microseconds);
    result.add_raw("expert", expert_metrics_json(metrics.generation));
    result.add_raw("cpu", cpu_metrics_json(metrics.generation));
    result.add_raw("gpu", gpu_metrics_json(metrics.generation, metrics.gpu_available));
    result.add_raw("process", process_telemetry_json(sampler));
    result.add_raw("gpu_device", gpu_telemetry_json(gpu_sampler));
    result.add_raw("cumulative", expert_metrics_json(metrics.cumulative));
    result.add_raw("cumulative_cpu", cpu_metrics_json(metrics.cumulative));
    result.add_raw("cumulative_gpu", gpu_metrics_json(metrics.cumulative, metrics.gpu_available));
    return result.finish();
}

static uint64_t require_mebibytes(const char* value, const char* option)
{
    const uint64_t count = std::stoull(value);
    if (count > std::numeric_limits<uint64_t>::max() / (1024 * 1024))
        throw std::out_of_range(std::string(option) + " is too large");
    return count * 1024 * 1024;
}

static const char* require_value(int argc, char** argv, int& index, const char* option)
{
    if (++index >= argc)
        throw std::invalid_argument(std::string(option) + " requires a value");
    return argv[index];
}

static std::vector<uint32_t> parse_device_indices(const char* value)
{
    std::vector<uint32_t> result;
    std::stringstream input(value);
    std::string item;
    while (std::getline(input, item, ','))
    {
        if (item.empty())
            throw std::invalid_argument("--vulkan-devices contains an empty device index");
        result.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    if (result.empty())
        throw std::invalid_argument("--vulkan-devices requires at least one device index");
    return result;
}

static HybridMode parse_hybrid_mode(const std::string& value)
{
    if (value == "auto") return HybridMode::Auto;
    if (value == "cpu") return HybridMode::CpuOnly;
    if (value == "hybrid") return HybridMode::HybridExperts;
    throw std::invalid_argument("unknown backend: " + value);
}

static ExpertMemoryMode parse_expert_memory_mode(const std::string& value)
{
    if (value == "auto") return ExpertMemoryMode::Auto;
    if (value == "eager") return ExpertMemoryMode::Eager;
    if (value == "on-demand") return ExpertMemoryMode::OnDemand;
    throw std::invalid_argument("unknown Expert memory mode: " + value);
}

static CpuPackedWeightMode parse_cpu_packed_weight_mode(
    const std::string& value)
{
    if (value == "on") return CpuPackedWeightMode::Enabled;
    if (value == "off") return CpuPackedWeightMode::Disabled;
    throw std::invalid_argument(
        "CPU packed-weight mode must be on or off");
}

static Option parse_options(int argc, char** argv, int first_argument)
{
    Option result;
    const auto set_expert_io_mode = [&result](uint32_t mode) {
        result.flags = (result.flags & ~OptionExpertIoMask) | mode;
    };
    for (int index = first_argument; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--backend")
            result.hybrid_mode = parse_hybrid_mode(require_value(argc, argv, index, "--backend"));
        else if (argument == "--cpu")
            result.hybrid_mode = HybridMode::CpuOnly;
        else if (argument == "--hybrid")
            result.hybrid_mode = HybridMode::HybridExperts;
        else if (argument == "--host-memory-mb")
            result.host_memory_budget = require_mebibytes(require_value(argc, argv, index, "--host-memory-mb"), "--host-memory-mb");
        else if (argument == "--expert-cache-mb")
            result.expert_cache_size = require_mebibytes(require_value(argc, argv, index, "--expert-cache-mb"), "--expert-cache-mb");
        else if (argument == "--expert-gpu-cache-mb")
            result.expert_gpu_cache_size = require_mebibytes(require_value(argc, argv, index, "--expert-gpu-cache-mb"), "--expert-gpu-cache-mb");
        else if (argument == "--expert-gpu-victim-cache-mb")
            result.expert_gpu_victim_cache_size = require_mebibytes(require_value(argc, argv, index, "--expert-gpu-victim-cache-mb"), "--expert-gpu-victim-cache-mb");
        else if (argument == "--expert-gpu-victim-reuse-probe")
            result.expert_gpu_victim_reuse_probe_interval = static_cast<uint32_t>(std::stoul(require_value(argc, argv, index, "--expert-gpu-victim-reuse-probe")));
        else if (argument == "--expert-io-workers")
            result.num_expert_io_threads = static_cast<uint32_t>(std::stoul(require_value(argc, argv, index, "--expert-io-workers")));
        else if (argument == "--expert-memory")
            result.expert_memory_mode = parse_expert_memory_mode(require_value(argc, argv, index, "--expert-memory"));
        else if (argument == "--cpu-packed-weights")
            result.cpu_packed_weight_mode = parse_cpu_packed_weight_mode(
                require_value(argc, argv, index, "--cpu-packed-weights"));
        else if (argument == "--vulkan-device")
            result.vulkan_device_index = static_cast<uint32_t>(std::stoul(require_value(argc, argv, index, "--vulkan-device")));
        else if (argument == "--vulkan-devices")
            result.vulkan_device_indices = parse_device_indices(require_value(argc, argv, index, "--vulkan-devices"));
        else if (argument == "--expected-concurrency")
            result.num_concurrent_sessions = static_cast<uint32_t>(std::stoul(require_value(argc, argv, index, "--expected-concurrency")));
        else if (argument == "--mmap-experts")
            set_expert_io_mode(OptionMemoryMapExperts);
        else if (argument == "--direct-expert-io")
            set_expert_io_mode(OptionDirectExpertIo);
        else if (argument == "--buffered-expert-io")
            set_expert_io_mode(OptionBufferedExpertIo);
        else if (argument == "--disable-gpu-victim-execution")
            result.flags |= ncnn::moe::OptionDisableGpuVictimExecution;
        else if (argument == "--disable-gpu-expert-execution")
            result.flags |= ncnn::moe::OptionDisableGpuExpertExecution;
        else if (argument == "--release-vulkan-dense-host")
            result.flags |= ncnn::moe::OptionReleaseVulkanDenseHostStorage;
        else
            throw std::invalid_argument("unknown worker option: " + argument);
    }
    return result;
}

static uint32_t request_uint(std::string_view request, std::string_view key, uint32_t default_value)
{
    const auto value = integer(request, key);
    if (!value)
        return default_value;
    if (*value < 0 || static_cast<uint64_t>(*value) > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(std::string(key) + " is outside the uint32 range");
    return static_cast<uint32_t>(*value);
}

static bool request_bool(std::string_view request, std::string_view key, bool default_value)
{
    const auto value = boolean(request, key);
    return value ? *value : default_value;
}

static std::vector<int32_t> request_tokens(std::string_view request, std::string_view key, bool required)
{
    const auto values = integer_array(request, key);
    if (!values)
    {
        if (required)
            throw std::invalid_argument(std::string(key) + " must be an integer array");
        return {};
    }
    std::vector<int32_t> result;
    result.reserve(values->size());
    for (const int64_t value : *values)
    {
        if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
            throw std::invalid_argument(std::string(key) + " contains an out-of-range token ID");
        result.push_back(static_cast<int32_t>(value));
    }
    if (required && result.empty())
        throw std::invalid_argument(std::string(key) + " must not be empty");
    return result;
}

struct GenerateRequest
{
    std::string request_id;
    std::string session_id;
    std::vector<int32_t> prompt_tokens;
    GenerationOptions options;
    bool metrics_enabled = true;
    uint32_t metrics_interval_ms = 1000;
};

class Worker
{
private:
    Runtime& runtime_;
    ModelPtr model_;
    std::unordered_map<std::string, SessionPtr> sessions_;
    std::mutex output_mutex_;
    std::mutex generation_state_mutex_;
    std::thread generation_thread_;
    std::atomic<bool> generation_running_{false};
    std::atomic<bool> cancel_requested_{false};
    std::string active_request_id_;
    ProcessTelemetrySampler telemetry_sampler_;
    GpuTelemetrySampler gpu_telemetry_sampler_;

    void emit(std::string value)
    {
        const std::lock_guard<std::mutex> lock(output_mutex_);
        std::cout << value << '\n'
                  << std::flush;
    }

    void emit_error(std::string_view request_id, std::string_view message, std::string_view code = "invalid_request")
    {
        JsonObject result;
        result.add_string("event", "error");
        if (!request_id.empty())
            result.add_string("request_id", request_id);
        result.add_string("code", code);
        result.add_string("message", message);
        emit(result.finish());
    }

    void join_finished_generation()
    {
        if (generation_thread_.joinable() && !generation_running_.load())
            generation_thread_.join();
    }

    [[nodiscard]] bool generation_active(std::string_view request_id = {})
    {
        if (!generation_running_.load())
            return false;
        if (request_id.empty())
            return true;
        const std::lock_guard<std::mutex> lock(generation_state_mutex_);
        return active_request_id_ == request_id;
    }

    void emit_stats(std::string_view event, std::string_view request_id, std::string_view session_id, const SessionPtr& session)
    {
        JsonObject result;
        result.add_string("event", event);
        if (!request_id.empty())
            result.add_string("request_id", request_id);
        result.add_string("session_id", session_id);
        result.add_uint("sequence_length", session->sequence_length());
        result.add_raw("stats", stats_json(session->metrics()));
        result.add_raw("memory", memory_statistics_json(session->memory_statistics()));
        emit(result.finish());
    }

    void emit_ready()
    {
        const RuntimeInfo& info = runtime_.info();
        const ncnn::moe::EffectiveOption& effective = model_->effective_option();
        const ncnn::moe::ModelMemoryPlan& memory_plan = model_->memory_plan();
        JsonObject model;
        model.add_string("model_type", model_->descriptor().model_type);
        model.add_uint("vocabulary_size", model_->descriptor().vocabulary_size);
        model.add_uint("hidden_size", model_->descriptor().hidden_size);
        model.add_uint("layer_count", model_->descriptor().layer_count);
        model.add_uint("expert_count", model_->descriptor().expert_count);
        model.add_uint("experts_per_token", model_->descriptor().experts_per_token);
        model.add_uint("max_context_tokens", maximum_context_tokens());

        JsonObject resources;
        resources.add_string("backend", hybrid_mode_name(effective.hybrid_mode));
        resources.add_string("requested_expert_memory", expert_memory_mode_name(effective.requested_expert_memory_mode));
        resources.add_string("selected_expert_memory", expert_memory_mode_name(effective.selected_expert_memory_mode));
        resources.add_string("requested_cpu_packed_weights", cpu_packed_weight_mode_name(effective.requested_cpu_packed_weight_mode));
        resources.add_string("selected_cpu_packed_weights", cpu_packed_weight_mode_name(effective.selected_cpu_packed_weight_mode));
        resources.add_uint("host_memory_budget_bytes", effective.host_memory_budget);
        resources.add_uint("expert_cache_bytes", effective.expert_cache_size);
        resources.add_uint("expert_gpu_cache_bytes", effective.expert_gpu_cache_size);
        resources.add_uint("expert_gpu_victim_cache_bytes", effective.expert_gpu_victim_cache_size);
        resources.add_uint("expert_io_workers", effective.num_expert_io_threads);
        resources.add_uint("expected_concurrency", effective.num_concurrent_sessions);
        resources.add_uint("flags", effective.flags);
        resources.add_uint("optimization_flags", effective.optimization_flags);
        resources.add_bool("file_backed_experts", effective.use_file_backed_experts);
        if (effective.vulkan_device_index == ncnn::moe::automatic_vulkan_device_index)
            resources.add_null("vulkan_device_index");
        else
            resources.add_uint("vulkan_device_index", effective.vulkan_device_index);
        resources.add_raw("vulkan_device_indices", uint_array(effective.vulkan_device_indices));
        resources.add_uint("estimated_dense_bytes", memory_plan.estimated_dense_size);
        resources.add_uint("estimated_expert_bytes", memory_plan.estimated_expert_size);
        resources.add_uint("expert_pair_bytes", memory_plan.expert_pair_size);
        resources.add_uint("expert_pair_resident_bytes", memory_plan.expert_pair_resident_size);
        resources.add_uint("estimated_cpu_packed_expert_bytes", memory_plan.estimated_cpu_packed_expert_size);
        resources.add_uint("estimated_expert_resident_bytes", memory_plan.estimated_expert_resident_size);
        resources.add_uint("minimum_active_expert_bytes", memory_plan.minimum_active_expert_size);

        JsonObject info_json;
        info_json.add_uint("physical_memory_bytes", info.physical_memory_size);
        info_json.add_uint("logical_cpu_count", info.cpu_count);
        info_json.add_uint("physical_cpu_core_count", info.physical_cpu_count);
        info_json.add_uint("openmp_thread_count", info.num_threads);
        info_json.add_uint("flags", info.flags);
        info_json.add_string("cpu_isa", cpu_isa_names(info.cpu_isa_flags));
        info_json.add_string(
            "bfloat16_batched_linear_kernel",
            bfloat16_batched_linear_kernel_name(effective.optimization_flags));
        info_json.add_uint("vulkan_device_count", info.gpu_infos.size());
        info_json.add_uint("selected_vulkan_device_index", info.default_gpu_index);
        info_json.add_raw("vulkan_devices", gpu_infos_json(info.gpu_infos));

        JsonObject result;
        result.add_string("event", "ready");
        result.add_raw("model", model.finish());
        result.add_raw("resources", resources.finish());
        result.add_raw("capabilities", info_json.finish());
        result.add_raw("telemetry", gpu_telemetry_json(gpu_telemetry_sampler_));
        emit(result.finish());
    }

    [[nodiscard]] uint32_t maximum_context_tokens() const
    {
        uint32_t maximum = 0;
        for (const ncnn::moe::LayerDescriptor& layer : model_->descriptor().layers)
            maximum = std::max(maximum, layer.attention.max_context_length);
        return maximum;
    }

    void execute_generate(GenerateRequest request, SessionPtr session)
    {
        const auto started = std::chrono::steady_clock::now();
        std::atomic<bool> metrics_stop{false};
        std::mutex metrics_mutex;
        std::condition_variable metrics_condition;
        const bool metrics_enabled = request.metrics_enabled && request.metrics_interval_ms != 0;
        std::thread metrics_thread;
        if (metrics_enabled)
        {
            metrics_thread = std::thread([&]() {
                std::unique_lock<std::mutex> lock(metrics_mutex);
                while (!metrics_stop.load())
                {
                    if (metrics_condition.wait_for(
                            lock,
                            std::chrono::milliseconds(request.metrics_interval_ms),
                            [&]() { return metrics_stop.load(); }))
                    {
                        break;
                    }
                    lock.unlock();
                    emit_runtime_metrics(request, session, started);
                    lock.lock();
                }
            });
        }
        const auto stop_metrics = [&]() {
            if (!metrics_enabled)
                return;
            metrics_stop.store(true);
            metrics_condition.notify_one();
            if (metrics_thread.joinable())
                metrics_thread.join();
        };
        try
        {
            auto generated = session->generate(
                request.prompt_tokens,
                request.options,
                [this, &request, started](const ncnn::moe::StreamToken& token) {
                    JsonObject event;
                    event.add_string("event", "token");
                    event.add_string("request_id", request.request_id);
                    event.add_string("session_id", request.session_id);
                    event.add_uint("index", token.index);
                    event.add_int("token_id", token.token_id);
                    event.add_double("probability", token.probability);
                    event.add_bool("is_stop_token", token.is_stop_token);
                    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                    event.add_double("elapsed_seconds", elapsed);
                    emit(event.finish());

                    if (cancel_requested_.load())
                        return false;
                    return true;
                });
            stop_metrics();
            if (!generated)
            {
                emit_error(request.request_id, generated.error().message, error_code_name(generated.error().code));
            }
            else
            {
                const ncnn::moe::GenerationResult& generation = generated.value();
                JsonObject done;
                done.add_string("event", "done");
                done.add_string("request_id", request.request_id);
                done.add_string("session_id", request.session_id);
                done.add_bool("ok", true);
                done.add_bool("metrics_enabled", metrics_enabled);
                done.add_uint("generated_tokens", generation.tokens.size());
                done.add_bool("stopped_by_stop_token", generation.stopped_by_stop_token);
                done.add_bool("stopped_by_callback", generation.stopped_by_callback);
                done.add_bool("cancelled", generation.stopped_by_callback && cancel_requested_.load());
                const SessionMetrics metrics = session->metrics();
                done.add_double("elapsed_seconds", static_cast<double>(metrics.timing.elapsed_microseconds) / 1000000.0);
                done.add_optional_double("prompt_tok_per_second", metrics.timing.prompt_tokens_per_second);
                done.add_optional_double("prompt_tokens_per_second", metrics.timing.prompt_tokens_per_second);
                done.add_optional_double("generation_tok_per_second", metrics.timing.generation_tokens_per_second);
                done.add_optional_double("generation_tokens_per_second", metrics.timing.generation_tokens_per_second);
                done.add_optional_double("tokens_per_second", metrics.timing.decode_tokens_per_second);
                done.add_optional_double("decode_tok_per_second", metrics.timing.decode_tokens_per_second);
                done.add_optional_uint("prompt_elapsed_microseconds", metrics.timing.prompt_elapsed_microseconds);
                done.add_optional_uint("generation_elapsed_microseconds", metrics.timing.generation_elapsed_microseconds);
                done.add_optional_uint("ttft_microseconds", metrics.timing.ttft_microseconds);
                done.add_optional_double("tpot_microseconds", metrics.timing.tpot_microseconds);
                done.add_uint("sequence_length", session->sequence_length());
                done.add_raw("metrics", runtime_metrics_json(telemetry_sampler_, gpu_telemetry_sampler_, metrics));
                done.add_raw("stats", stats_json(metrics));
                done.add_raw("memory", memory_statistics_json(session->memory_statistics()));
                done.add_raw("telemetry", process_telemetry_json(telemetry_sampler_));
                done.add_raw("gpu", gpu_telemetry_json(gpu_telemetry_sampler_));
                emit(done.finish());
            }
        }
        catch (const std::exception& exception)
        {
            stop_metrics();
            emit_error(request.request_id, exception.what(), "worker_exception");
        }
        catch (...)
        {
            stop_metrics();
            emit_error(request.request_id, "unknown exception in generation", "worker_exception");
        }
        {
            const std::lock_guard<std::mutex> lock(generation_state_mutex_);
            active_request_id_.clear();
        }
        generation_running_.store(false);
    }

    void emit_runtime_metrics(const GenerateRequest& request, const SessionPtr& session, std::chrono::steady_clock::time_point started)
    {
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const SessionMetrics metrics = session->metrics();
        JsonObject event;
        event.add_string("event", "metrics");
        event.add_string("request_id", request.request_id);
        event.add_string("session_id", request.session_id);
        event.add_double("elapsed_seconds", elapsed);
        event.add_raw("metrics", runtime_metrics_json(telemetry_sampler_, gpu_telemetry_sampler_, metrics));
        emit(event.finish());
    }

    void handle_create_session(std::string_view request)
    {
        const auto session_id = string(request, "session_id");
        if (!session_id || session_id->empty())
            throw std::invalid_argument("create_session requires a non-empty session_id");
        if (sessions_.contains(*session_id))
            throw std::invalid_argument("session already exists: " + *session_id);
        SessionOptions options;
        options.sampling_seed = integer(request, "seed").value_or(0);
        options.prefill_chunk_size = request_uint(request, "prefill_chunk_size", options.prefill_chunk_size);
        options.enable_speculative_context = request_bool(request, "enable_speculative_context", true);
        auto session = runtime_.create_session(model_, options);
        if (!session)
            throw std::runtime_error(session.error().message);
        sessions_.emplace(*session_id, std::move(session).value());
        JsonObject result;
        result.add_string("event", "session_created");
        result.add_string("session_id", *session_id);
        result.add_uint("sequence_length", 0);
        emit(result.finish());
    }

    void handle_reset(std::string_view request)
    {
        const auto session_id = string(request, "session_id");
        if (!session_id)
            throw std::invalid_argument("reset requires session_id");
        if (generation_active())
            throw std::invalid_argument("cannot reset while a generation is active; cancel it first");
        const auto iterator = sessions_.find(*session_id);
        if (iterator == sessions_.end())
            throw std::invalid_argument("unknown session: " + *session_id);
        auto reset = iterator->second->reset();
        if (!reset)
            throw std::runtime_error(reset.error().message);
        JsonObject result;
        result.add_string("event", "reset");
        result.add_string("session_id", *session_id);
        emit(result.finish());
    }

    void handle_compact(std::string_view request)
    {
        const auto session_id = string(request, "session_id");
        if (!session_id)
            throw std::invalid_argument("compact requires session_id");
        if (generation_active())
            throw std::invalid_argument("cannot compact while a generation is active; cancel it first");
        const auto iterator = sessions_.find(*session_id);
        if (iterator == sessions_.end())
            throw std::invalid_argument("unknown session: " + *session_id);
        std::vector<int32_t> replay_tokens = request_tokens(request, "replay_tokens", false);
        auto reset = iterator->second->reset();
        if (!reset)
            throw std::runtime_error(reset.error().message);
        if (!replay_tokens.empty())
        {
            auto replay = iterator->second->prefill(replay_tokens);
            if (!replay)
                throw std::runtime_error(replay.error().message);
        }
        JsonObject result;
        result.add_string("event", "compacted");
        result.add_string("session_id", *session_id);
        result.add_uint("replayed_tokens", replay_tokens.size());
        result.add_uint("sequence_length", iterator->second->sequence_length());
        emit(result.finish());
    }

    void handle_stats(std::string_view request)
    {
        const auto session_id = string(request, "session_id");
        if (!session_id)
            throw std::invalid_argument("stats requires session_id");
        const auto iterator = sessions_.find(*session_id);
        if (iterator == sessions_.end())
            throw std::invalid_argument("unknown session: " + *session_id);
        emit_stats("stats", {}, *session_id, iterator->second);
    }

    void handle_generate(std::string_view request)
    {
        join_finished_generation();
        if (generation_running_.load())
            throw std::invalid_argument("only one generation can run at a time; cancel it first");
        GenerateRequest parsed;
        parsed.request_id = string(request, "request_id").value_or("generate");
        parsed.session_id = string(request, "session_id").value_or("main");
        parsed.prompt_tokens = request_tokens(request, "prompt_tokens", true);
        parsed.options.max_new_tokens = request_uint(request, "max_new_tokens", parsed.options.max_new_tokens);
        parsed.options.enable_speculative = request_bool(request, "enable_speculative", true);
        parsed.metrics_enabled = request_bool(request, "metrics_enabled", true);
        parsed.options.stop_tokens = request_tokens(request, "stop_tokens", false);
        if (const auto value = number(request, "temperature"))
            parsed.options.sampling.temperature = static_cast<float>(*value);
        else
            parsed.options.sampling.temperature = 0.0f;
        if (const auto value = integer(request, "top_k"))
            parsed.options.sampling.top_k = request_uint(request, "top_k", 0);
        if (const auto value = number(request, "top_p"))
            parsed.options.sampling.top_p = static_cast<float>(*value);
        if (const auto value = number(request, "min_p"))
            parsed.options.sampling.min_p = static_cast<float>(*value);
        if (const auto value = number(request, "speculative_confidence"))
            parsed.options.speculative_confidence_threshold = static_cast<float>(*value);
        parsed.options.speculative_max_draft_tokens = request_uint(request, "speculative_max_draft", 0);
        parsed.metrics_interval_ms = request_uint(request, "metrics_interval_ms", 1000);
        if (parsed.metrics_interval_ms == 0)
            parsed.metrics_enabled = false;
        const auto iterator = sessions_.find(parsed.session_id);
        if (iterator == sessions_.end())
            throw std::invalid_argument("unknown session: " + parsed.session_id);
        if (parsed.options.max_new_tokens == 0)
        {
            const uint32_t max_context = maximum_context_tokens();
            const uint64_t sequence_length = iterator->second->sequence_length();
            if (max_context == 0 || sequence_length >= max_context)
                throw std::invalid_argument("unlimited generation requires remaining model context");
            parsed.options.max_new_tokens = static_cast<uint32_t>(max_context - sequence_length);
        }
        {
            const std::lock_guard<std::mutex> lock(generation_state_mutex_);
            active_request_id_ = parsed.request_id;
        }
        cancel_requested_.store(false);
        generation_running_.store(true);
        generation_thread_ = std::thread(&Worker::execute_generate, this, std::move(parsed), iterator->second);
    }

    void handle_cancel(std::string_view request)
    {
        const std::string request_id = string(request, "request_id").value_or("");
        if (!generation_running_.load())
            throw std::invalid_argument("no generation is active");
        if (!request_id.empty() && !generation_active(request_id))
            throw std::invalid_argument("request_id does not match the active generation");
        cancel_requested_.store(true);
        JsonObject result;
        result.add_string("event", "cancel_requested");
        if (!request_id.empty())
            result.add_string("request_id", request_id);
        emit(result.finish());
    }

public:
    Worker(Runtime& runtime, ModelPtr model)
        : runtime_(runtime), model_(std::move(model)), gpu_telemetry_sampler_(runtime.info(), model_->effective_option())
    {
        (void)telemetry_sampler_.sample();
        emit_ready();
    }

    ~Worker()
    {
        cancel_requested_.store(true);
        if (generation_thread_.joinable())
            generation_thread_.join();
    }

    int run()
    {
        std::string line;
        while (std::getline(std::cin, line))
        {
            join_finished_generation();
            if (line.empty())
                continue;
            try
            {
                const auto operation = string(line, "op");
                if (!operation)
                    throw std::invalid_argument("request requires an op string");
                if (*operation == "create_session")
                    handle_create_session(line);
                else if (*operation == "generate")
                    handle_generate(line);
                else if (*operation == "reset")
                    handle_reset(line);
                else if (*operation == "compact")
                    handle_compact(line);
                else if (*operation == "stats")
                    handle_stats(line);
                else if (*operation == "cancel")
                    handle_cancel(line);
                else if (*operation == "shutdown")
                {
                    cancel_requested_.store(true);
                    if (generation_thread_.joinable())
                        generation_thread_.join();
                    JsonObject result;
                    result.add_string("event", "shutdown");
                    emit(result.finish());
                    return 0;
                }
                else
                    throw std::invalid_argument("unknown op: " + *operation);
            }
            catch (const std::exception& exception)
            {
                const auto request_id = string(line, "request_id").value_or("");
                emit_error(request_id, exception.what());
            }
        }
        cancel_requested_.store(true);
        if (generation_thread_.joinable())
            generation_thread_.join();
        return 0;
    }
};

static void print_usage(const char* executable)
{
    std::cerr << "usage: " << executable << " <model-directory> [runtime options]\n"
              << "  runtime: --backend auto|cpu|hybrid, --host-memory-mb N,\n"
              << "           --expert-cache-mb N, --expert-gpu-cache-mb N, --expert-io-workers N,\n"
              << "           --expert-memory auto|eager|on-demand, --cpu-packed-weights on|off,\n"
              << "           --vulkan-device N, --vulkan-devices N[,N...]\n"
              << "  io/cache: --mmap-experts, --direct-expert-io, --buffered-expert-io,\n"
              << "            --release-vulkan-dense-host, --disable-gpu-expert-execution,\n"
              << "            --expected-concurrency N\n";
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        ncnn::moe::print_usage(argv[0]);
        return 2;
    }
    try
    {
        const ncnn::moe::Option opt = ncnn::moe::parse_options(argc, argv, 2);
        ncnn::moe::Runtime runtime;
        auto model = runtime.load_model(std::filesystem::path(argv[1]), opt);
        if (!model)
        {
            ncnn::moe::JsonObject error;
            error.add_string("event", "error");
            error.add_string("code", ncnn::moe::error_code_name(model.error().code));
            error.add_string("message", model.error().message);
            std::cout << error.finish() << '\n'
                      << std::flush;
            return 1;
        }
        ncnn::moe::Worker worker(runtime, std::move(model).value());
        return worker.run();
    }
    catch (const std::exception& exception)
    {
        ncnn::moe::JsonObject error;
        error.add_string("event", "error");
        error.add_string("code", "invalid_arguments");
        error.add_string("message", exception.what());
        std::cout << error.finish() << '\n'
                  << std::flush;
        return 2;
    }
}
