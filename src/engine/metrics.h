#ifndef NCNN_MOE_METRICS_H
#define NCNN_MOE_METRICS_H

#include <cstddef>
#include <span>

namespace ncnn {
namespace moe {

struct CompiledModel;
struct DecodeBatchEntry;
struct ExpertBackendStatistics;
struct ExpertCacheStatistics;
struct SessionStatistics;
struct RuntimeMetricCounters;
struct VulkanStatistics;

// Shared resource snapshots describe the interval, not exclusive Session work.
void record_resource_delta(const CompiledModel& model,
                           SessionStatistics& stats,
                           const ExpertCacheStatistics& cache_before,
                           const ExpertBackendStatistics& backend_before);
void record_resource_delta(const CompiledModel& model,
                           std::span<const DecodeBatchEntry> entries,
                           const ExpertCacheStatistics& cache_before,
                           const ExpertBackendStatistics& backend_before);

void record_vulkan_delta(SessionStatistics& stats,
                         const VulkanStatistics& before,
                         const VulkanStatistics& after);

RuntimeMetricCounters metric_counters(const SessionStatistics& stats,
                                      const RuntimeMetricCounters* baseline);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_METRICS_H
