#include "resourceplan.h"

#include <algorithm>

namespace ncnn {
namespace moe {

static uint64_t add_size(uint64_t left, uint64_t right) noexcept
{
    return right > std::numeric_limits<uint64_t>::max() - left
               ? std::numeric_limits<uint64_t>::max()
               : left + right;
}

static uint64_t multiply_size(uint64_t size, uint64_t count) noexcept
{
    return count != 0 && size > std::numeric_limits<uint64_t>::max() / count
               ? std::numeric_limits<uint64_t>::max()
               : size * count;
}

static GpuResourceRequest combine_requests(GpuResourceRequest admitted, GpuResourceRequest request) noexcept
{
    return {add_size(admitted.persistent_size, request.persistent_size),
            add_size(admitted.session_state_size, request.session_state_size),
            std::max(admitted.workspace_size, request.workspace_size)};
}

GpuResourcePlan make_gpu_resource_plan(GpuResourceSnapshot initial,
                                       uint32_t num_concurrent_sessions,
                                       uint64_t expert_cache_reserve) noexcept
{
    return {initial, std::max(1u, num_concurrent_sessions), expert_cache_reserve, {}};
}

static uint64_t execution_reserve(const GpuResourcePlan& plan, GpuResourceRequest request) noexcept
{
    const uint64_t workspace = multiply_size(request.workspace_size, plan.num_concurrent_sessions);
    return add_size(multiply_size(request.session_state_size, plan.num_concurrent_sessions), workspace);
}

uint64_t gpu_resource_execution_reserve(const GpuResourcePlan& plan) noexcept
{
    return execution_reserve(plan, plan.admitted);
}

static uint64_t remaining_size(uint64_t budget, uint64_t used, uint64_t reserved) noexcept
{
    if (used >= budget || reserved >= budget - used)
        return 0;
    return budget - used - reserved;
}

bool can_admit_gpu_resources(const GpuResourcePlan& plan,
                             GpuResourceSnapshot live,
                             GpuResourceRequest request) noexcept
{
    if (plan.initial.device_index != live.device_index || plan.initial.heap_budget == 0 || live.heap_budget == 0)
        return false;
    const uint64_t budget = std::min(plan.initial.heap_budget, live.heap_budget);
    const GpuResourceRequest combined = combine_requests(plan.admitted, request);
    if (combined.persistent_size < plan.admitted.persistent_size
        || request.persistent_size > std::numeric_limits<uint64_t>::max() - plan.admitted.persistent_size
        || request.session_state_size > std::numeric_limits<uint64_t>::max() - plan.admitted.session_state_size
        || combined.session_state_size > std::numeric_limits<uint64_t>::max() / plan.num_concurrent_sessions
        || combined.workspace_size > std::numeric_limits<uint64_t>::max() / plan.num_concurrent_sessions)
        return false;
    const uint64_t state_size = combined.session_state_size * plan.num_concurrent_sessions;
    const uint64_t workspace_size = combined.workspace_size * plan.num_concurrent_sessions;
    if (workspace_size > std::numeric_limits<uint64_t>::max() - state_size
        || plan.expert_cache_reserve > std::numeric_limits<uint64_t>::max() - state_size - workspace_size)
        return false;
    const uint64_t execution = std::max(state_size + workspace_size, plan.allocated_execution_size);
    if (plan.expert_cache_reserve > std::numeric_limits<uint64_t>::max() - execution)
        return false;
    const uint64_t reserved = execution + plan.expert_cache_reserve;
    const uint64_t live_reserved = reserved - plan.allocated_execution_size;
    if (plan.initial.heap_usage > budget || live.heap_usage > budget
        || reserved > budget - plan.initial.heap_usage || live_reserved > budget - live.heap_usage)
        return false;
    const uint64_t baseline_available = remaining_size(budget, plan.initial.heap_usage, reserved);
    const uint64_t live_available = remaining_size(budget, live.heap_usage, live_reserved);
    return combined.persistent_size <= baseline_available && request.persistent_size <= live_available;
}

void commit_gpu_resources(GpuResourcePlan& plan, GpuResourceRequest request) noexcept
{
    plan.admitted = combine_requests(plan.admitted, request);
}

uint64_t gpu_resource_cache_capacity(const GpuResourcePlan& plan,
                                     GpuResourceSnapshot live,
                                     uint64_t minimum_entry_size,
                                     uint64_t total_size) noexcept
{
    if (plan.initial.device_index != live.device_index || plan.initial.heap_budget == 0
        || live.heap_budget == 0 || minimum_entry_size == 0)
        return 0;
    const uint64_t budget = std::min(plan.initial.heap_budget, live.heap_budget);
    const uint64_t reserved = std::max(execution_reserve(plan, plan.admitted), plan.allocated_execution_size);
    const uint64_t baseline_available = remaining_size(budget,
                                                       add_size(plan.initial.heap_usage, plan.admitted.persistent_size), reserved);
    const uint64_t live_available = remaining_size(budget, live.heap_usage, reserved - plan.allocated_execution_size);
    const uint64_t capacity = std::min({baseline_available, live_available, total_size});
    return capacity - capacity % minimum_entry_size;
}

} // namespace moe
} // namespace ncnn
