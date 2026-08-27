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
#define NCNN_MOE_MEMORY_FILE_BACKED_PLE_BIT    1

enum ModelMemoryFlag : uint32_t
{
    ModelMemoryFileBackedExperts = UINT32_C(1) << NCNN_MOE_MEMORY_FILE_BACKED_EXPERT_BIT,
    ModelMemoryFileBackedPle = UINT32_C(1) << NCNN_MOE_MEMORY_FILE_BACKED_PLE_BIT
};

struct ModelMemoryPlan
{
    ExpertMemoryMode requested_mode = ExpertMemoryMode::Auto;
    ExpertMemoryMode selected_mode = ExpertMemoryMode::Eager;
    uint64_t physical_memory_bytes = 0;
    uint64_t host_memory_budget_bytes = 0;
    uint64_t estimated_dense_bytes = 0;
    uint64_t estimated_file_backed_dense_bytes = 0;
    uint64_t estimated_expert_bytes = 0;
    uint64_t estimated_cpu_packed_expert_bytes = 0;
    uint64_t estimated_expert_resident_bytes = 0;
    uint64_t expert_pair_bytes = 0;
    uint64_t expert_pair_resident_bytes = 0;
    uint64_t minimum_active_expert_bytes = 0;
    uint64_t expert_cache_bytes = 0;
    uint32_t flags = 0;
    uint64_t available_memory_bytes = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MEMORY_PLAN_H
