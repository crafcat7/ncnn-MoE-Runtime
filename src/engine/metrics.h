#ifndef NCNN_MOE_METRICS_H
#define NCNN_MOE_METRICS_H

#include "backends/ncnn/vulkancontext.h"

#include <cstddef>
#include <span>

namespace ncnn {
namespace moe {

struct CompiledModel;
struct CpuDecodeBatchEntry;
struct ExpertBackendStatistics;
struct ExpertCacheStatistics;
struct ExpertVictimCacheStatistics;
struct ExpertPlan;
struct SessionStatistics;

// Shared resource snapshots describe the interval, not exclusive Session work.
void record_model_resource_delta(
    const CompiledModel& model,
    SessionStatistics& statistics,
    const ExpertCacheStatistics& execution_cache_before,
    const ExpertBackendStatistics& expert_backend_before);
void record_batch_resource_delta(
    const CompiledModel& model,
    std::span<const CpuDecodeBatchEntry> entries,
    const ExpertCacheStatistics& cache_before,
    const ExpertBackendStatistics& backend_before);

void record_expert_weight_demand(const ExpertPlan& expert, size_t route_count, SessionStatistics& statistics) noexcept;

void record_expert_cache_delta(
    SessionStatistics& statistics,
    const ExpertCacheStatistics& before,
    const ExpertCacheStatistics& after);

void record_vulkan_execution_delta(
    SessionStatistics& statistics,
    const NcnnVulkanExecutionSnapshot& before,
    const NcnnVulkanContextInstancePtr& context_instance);
void record_expert_backend_delta(
    SessionStatistics& statistics,
    const ExpertBackendStatistics& before,
    const ExpertBackendStatistics& after);
void record_expert_victim_cache_delta(
    SessionStatistics& statistics,
    const ExpertVictimCacheStatistics& before,
    const ExpertVictimCacheStatistics& after);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_METRICS_H
