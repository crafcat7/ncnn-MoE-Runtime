#include "gpu_telemetry.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace ncnn {
namespace moe {

constexpr uint32_t nvidia_vendor_id = 0x10de;
constexpr uint32_t nvml_success = 0;

using NvmlReturn = unsigned int;
using NvmlDevice = void*;

struct NvmlUtilization
{
    unsigned int gpu = 0;
    unsigned int memory = 0;
};

struct NvmlMemory
{
    unsigned long long total = 0;
    unsigned long long free = 0;
    unsigned long long used = 0;
};

using NvmlInitFn = NvmlReturn (*)();
using NvmlShutdownFn = NvmlReturn (*)();
using NvmlGetCountFn = NvmlReturn (*)(unsigned int*);
using NvmlGetHandleByIndexFn = NvmlReturn (*)(unsigned int, NvmlDevice*);
using NvmlGetUtilizationFn = NvmlReturn (*)(NvmlDevice, NvmlUtilization*);
using NvmlGetMemoryInfoFn = NvmlReturn (*)(NvmlDevice, NvmlMemory*);
using NvmlGetNameFn = NvmlReturn (*)(NvmlDevice, char*, unsigned int);
using NvmlErrorStringFn = const char* (*)(NvmlReturn);

#if defined(_WIN32)
using LibraryHandle = HMODULE;
#else
using LibraryHandle = void*;
#endif

void* load_symbol(LibraryHandle library, const char* name)
{
#if defined(_WIN32)
    return library == nullptr ? nullptr : reinterpret_cast<void*>(GetProcAddress(library, name));
#elif defined(__linux__) || defined(__APPLE__)
    return library == nullptr ? nullptr : dlsym(library, name);
#else
    (void)library;
    (void)name;
    return nullptr;
#endif
}

LibraryHandle load_library()
{
#if defined(_WIN32)
    return LoadLibraryA("nvml.dll");
#elif defined(__linux__)
    for (const char* name : {"libnvidia-ml.so.1", "libnvidia-ml.so"})
    {
        if (LibraryHandle library = dlopen(name, RTLD_LOCAL | RTLD_NOW))
            return library;
    }
    return nullptr;
#else
    return nullptr;
#endif
}

void unload_library(LibraryHandle library)
{
    if (library == nullptr)
        return;
#if defined(_WIN32)
    FreeLibrary(library);
#elif defined(__linux__) || defined(__APPLE__)
    dlclose(library);
#else
    (void)library;
#endif
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

struct GpuTelemetrySampler::Impl
{
    GpuTelemetrySample base;
    LibraryHandle library = nullptr;
    NvmlInitFn init = nullptr;
    NvmlShutdownFn shutdown = nullptr;
    NvmlGetCountFn get_count = nullptr;
    NvmlGetHandleByIndexFn get_handle_by_index = nullptr;
    NvmlGetUtilizationFn get_utilization = nullptr;
    NvmlGetMemoryInfoFn get_memory_info = nullptr;
    NvmlGetNameFn get_name = nullptr;
    NvmlErrorStringFn error_string = nullptr;
    NvmlDevice device = nullptr;
    bool initialized = false;

    Impl(const RuntimeCapabilities& capabilities, const EffectiveRuntimeConfig& effective)
    {
        base.active = effective.hybrid_mode != HybridMode::CpuOnly;
        if (!base.active)
        {
            base.reason = "backend_cpu";
            return;
        }

        std::optional<uint32_t> selected_index;
        if (!effective.vulkan_device_indices.empty())
            selected_index = effective.vulkan_device_indices.front();
        else if (effective.vulkan_device_index != automatic_vulkan_device_index)
            selected_index = effective.vulkan_device_index;
        else if (capabilities.vulkan_device_count != 0)
            selected_index = capabilities.selected_vulkan_device_index;

        if (!selected_index || *selected_index >= capabilities.vulkan_devices.size())
        {
            base.reason = "no_vulkan_device";
            return;
        }

        const VulkanDeviceCapabilities& vulkan_device = capabilities.vulkan_devices[*selected_index];
        base.device_index = vulkan_device.index;
        base.vendor_id = vulkan_device.vendor_id;
        base.device_name = vulkan_device.name;
        if (base.vendor_id != nvidia_vendor_id)
        {
            base.reason = "selected_device_not_nvidia";
            return;
        }

        library = load_library();
        if (library == nullptr)
        {
            base.reason = "nvml_library_not_found";
            return;
        }
        base.provider = "nvml";

        init = reinterpret_cast<NvmlInitFn>(load_symbol(library, "nvmlInit_v2"));
        if (init == nullptr)
            init = reinterpret_cast<NvmlInitFn>(load_symbol(library, "nvmlInit"));
        shutdown = reinterpret_cast<NvmlShutdownFn>(load_symbol(library, "nvmlShutdown"));
        get_count = reinterpret_cast<NvmlGetCountFn>(load_symbol(library, "nvmlDeviceGetCount_v2"));
        if (get_count == nullptr)
            get_count = reinterpret_cast<NvmlGetCountFn>(load_symbol(library, "nvmlDeviceGetCount"));
        get_handle_by_index = reinterpret_cast<NvmlGetHandleByIndexFn>(
            load_symbol(library, "nvmlDeviceGetHandleByIndex_v2"));
        if (get_handle_by_index == nullptr)
            get_handle_by_index = reinterpret_cast<NvmlGetHandleByIndexFn>(
                load_symbol(library, "nvmlDeviceGetHandleByIndex"));
        get_utilization = reinterpret_cast<NvmlGetUtilizationFn>(
            load_symbol(library, "nvmlDeviceGetUtilizationRates"));
        get_memory_info = reinterpret_cast<NvmlGetMemoryInfoFn>(load_symbol(library, "nvmlDeviceGetMemoryInfo"));
        get_name = reinterpret_cast<NvmlGetNameFn>(load_symbol(library, "nvmlDeviceGetName"));
        error_string = reinterpret_cast<NvmlErrorStringFn>(load_symbol(library, "nvmlErrorString"));

        if (init == nullptr || shutdown == nullptr || get_count == nullptr || get_handle_by_index == nullptr
            || get_utilization == nullptr || get_memory_info == nullptr || get_name == nullptr)
        {
            base.reason = "nvml_symbol_missing";
            return;
        }

        const NvmlReturn init_result = init();
        if (init_result != nvml_success)
        {
            base.reason = error_reason("nvml_init", init_result);
            return;
        }
        initialized = true;

        unsigned int device_count = 0;
        const NvmlReturn count_result = get_count(&device_count);
        if (count_result != nvml_success)
        {
            base.reason = error_reason("nvml_device_count", count_result);
            return;
        }

        const std::string target_name = lower_copy(base.device_name);
        for (unsigned int index = 0; index < device_count; ++index)
        {
            NvmlDevice candidate = nullptr;
            if (get_handle_by_index(index, &candidate) != nvml_success || candidate == nullptr)
                continue;

            char name[256] = {};
            if (get_name(candidate, name, sizeof(name)) != nvml_success)
                continue;
            const std::string candidate_name = lower_copy(name);
            if (!target_name.empty() && (candidate_name == target_name || candidate_name.find(target_name) != std::string::npos
                                          || target_name.find(candidate_name) != std::string::npos))
            {
                device = candidate;
                break;
            }
        }

        // Vulkan and NVML usually enumerate NVIDIA devices in the same order.
        // Use the index as a conservative fallback when the driver changes the
        // reported device name format.
        if (device == nullptr && *selected_index < device_count)
        {
            NvmlDevice candidate = nullptr;
            if (get_handle_by_index(*selected_index, &candidate) == nvml_success)
                device = candidate;
        }

        if (device == nullptr)
        {
            base.reason = "nvml_device_not_found";
            return;
        }
        base.reason.clear();
    }

    ~Impl()
    {
        if (initialized && shutdown != nullptr)
            (void)shutdown();
        unload_library(library);
    }

    [[nodiscard]] std::string error_reason(const char* operation, NvmlReturn result) const
    {
        std::string reason = operation;
        reason += "_failed";
        reason += "_";
        if (error_string != nullptr)
        {
            const char* description = error_string(result);
            if (description != nullptr && *description != '\0')
            {
                reason += lower_copy(description);
                return reason;
            }
        }
        reason += std::to_string(result);
        return reason;
    }

    [[nodiscard]] GpuTelemetrySample sample() const
    {
        GpuTelemetrySample result = base;
        if (!initialized || device == nullptr)
            return result;

        NvmlUtilization utilization{};
        const NvmlReturn utilization_result = get_utilization(device, &utilization);
        if (utilization_result == nvml_success)
        {
            result.utilization_percent = static_cast<double>(utilization.gpu);
            result.memory_utilization_percent = static_cast<double>(utilization.memory);
        }
        else
        {
            result.reason = error_reason("nvml_utilization", utilization_result);
        }

        NvmlMemory memory{};
        const NvmlReturn memory_result = get_memory_info(device, &memory);
        if (memory_result == nvml_success)
        {
            result.used_bytes = static_cast<uint64_t>(memory.used);
            result.total_bytes = static_cast<uint64_t>(memory.total);
        }
        else if (result.reason.empty())
        {
            result.reason = error_reason("nvml_memory", memory_result);
        }
        return result;
    }
};

GpuTelemetrySampler::GpuTelemetrySampler(const RuntimeCapabilities& capabilities, const EffectiveRuntimeConfig& effective)
    : impl_(std::make_unique<Impl>(capabilities, effective))
{
}

GpuTelemetrySampler::~GpuTelemetrySampler() = default;

GpuTelemetrySample GpuTelemetrySampler::sample() const
{
    return impl_->sample();
}

} // namespace moe
} // namespace ncnn
