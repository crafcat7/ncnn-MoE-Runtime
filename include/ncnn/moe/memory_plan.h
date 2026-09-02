#ifndef NCNN_MOE_MEMORY_PLAN_H
#define NCNN_MOE_MEMORY_PLAN_H

#include <cstdint>

namespace ncnn {
namespace moe {

enum class ExpertMemoryMode
{
    Auto,
    Eager,
    OnDemand
};

#define NCNN_MOE_MEMORY_FILE_BACKED_EXPERT_BIT 0

enum ModelMemoryFlag : uint32_t
{
    ModelMemoryFileBackedExperts = UINT32_C(1) << NCNN_MOE_MEMORY_FILE_BACKED_EXPERT_BIT
};

struct ModelMemoryPlan
{
    ExpertMemoryMode requested_mode = ExpertMemoryMode::Auto;
    ExpertMemoryMode selected_mode = ExpertMemoryMode::Eager;
    uint64_t physical_memory_size = 0;
    uint64_t host_memory_budget = 0;
    uint64_t estimated_dense_size = 0;
    uint64_t estimated_expert_size = 0;
    uint64_t estimated_cpu_packed_expert_size = 0;
    uint64_t estimated_expert_resident_size = 0;
    uint64_t expert_pair_size = 0;
    uint64_t expert_pair_resident_size = 0;
    uint64_t minimum_active_expert_size = 0;
    uint64_t expert_cache_size = 0;
    uint32_t flags = 0;
    uint64_t available_memory_size = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MEMORY_PLAN_H
