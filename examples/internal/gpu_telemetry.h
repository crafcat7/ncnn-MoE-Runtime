#ifndef NCNN_MOE_GPU_TELEMETRY_H
#define NCNN_MOE_GPU_TELEMETRY_H

#include "ncnn/moe/runtime.h"
#include "ncnn/moe/option.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ncnn {
namespace moe {

struct GpuTelemetrySample
{
    bool active = false;
    std::optional<uint32_t> device_index;
    uint32_t vendor_id = 0;
    std::string device_name;
    std::string provider = "none";
    std::string reason = "unavailable";
    std::optional<double> utilization_percent;
    std::optional<double> memory_utilization_percent;
    std::optional<uint64_t> used_bytes;
    std::optional<uint64_t> total_bytes;
};

// Samples vendor GPU telemetry without making NVML a build-time dependency.
// NVIDIA's NVML library is loaded from the installed driver at runtime; other
// vendors and missing providers remain observable as an explicit reason.
class GpuTelemetrySampler
{
private:
    struct Impl;
    std::unique_ptr<Impl> d;

public:
    GpuTelemetrySampler(const RuntimeInfo& info, const EffectiveOption& opt);
    ~GpuTelemetrySampler();

    GpuTelemetrySampler(const GpuTelemetrySampler&) = delete;
    GpuTelemetrySampler& operator=(const GpuTelemetrySampler&) = delete;

    [[nodiscard]] GpuTelemetrySample sample() const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GPU_TELEMETRY_H
