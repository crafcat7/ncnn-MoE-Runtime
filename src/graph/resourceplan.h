#ifndef NCNN_MOE_RESOURCEPLAN_H
#define NCNN_MOE_RESOURCEPLAN_H

#include <cstdint>
#include <limits>

namespace ncnn {
namespace moe {

struct GpuResourceSnapshot
{
    uint32_t device_index = 0;
    uint64_t heap_budget = 0;
    uint64_t heap_usage = 0;
};

struct GpuResourceRequest
{
    uint64_t persistent_size = 0;
    uint64_t session_state_size = 0;
    uint64_t workspace_size = 0;
};

struct GpuResourcePlan
{
    GpuResourceSnapshot initial;
    uint32_t num_concurrent_sessions = 1;
    uint64_t expert_cache_reserve = 0;
    GpuResourceRequest admitted;
    // Physically reserved graph/state storage is already present in live heap
    // usage. Keep it in the baseline but never charge it twice to that heap.
    uint64_t allocated_execution_size = 0;
};

[[nodiscard]] GpuResourcePlan make_gpu_resource_plan(GpuResourceSnapshot initial,
                                                     uint32_t num_concurrent_sessions,
                                                     uint64_t expert_cache_reserve = 0) noexcept;

// Weights are shared; state and peak workspace are reserved for each Session.
// The baseline estimate and live usage are independent limits, so already
// allocated weights are never subtracted twice from the live heap.
[[nodiscard]] bool can_admit_gpu_resources(const GpuResourcePlan& plan,
                                           GpuResourceSnapshot live,
                                           GpuResourceRequest request) noexcept;

// Commit only after an entire operator group has been created successfully.
void commit_gpu_resources(GpuResourcePlan& plan, GpuResourceRequest request) noexcept;

[[nodiscard]] uint64_t gpu_exec_reserve(const GpuResourcePlan& plan) noexcept;

// Called after persistent operators are allocated. Explicit cache reservations
// are released into this capacity, rather than subtracted a second time.
[[nodiscard]] uint64_t gpu_resource_cache_capacity(const GpuResourcePlan& plan,
                                                   GpuResourceSnapshot live,
                                                   uint64_t minimum_entry_size,
                                                   uint64_t total_size = std::numeric_limits<uint64_t>::max()) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RESOURCEPLAN_H
