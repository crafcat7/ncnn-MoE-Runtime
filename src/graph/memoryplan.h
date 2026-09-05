#ifndef NCNN_MOE_MEMORYPLAN_H
#define NCNN_MOE_MEMORYPLAN_H

#include "ncnn/moe/option.h"
#include "ncnn/moe/modeldescriptor.h"
#include "ncnn/moe/result.h"

#include <cstdint>

namespace ncnn {
namespace moe {

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
    uint64_t available_memory_size = 0;
};

struct Option;

[[nodiscard]] Result<ModelMemoryPlan> plan_model_memory(const MoeModelDescriptor& descriptor, const Option& opt, uint64_t physical_memory_size,
                                                        bool release_vulkan_dense_host_storage = false,
                                                        uint64_t available_memory_size = 0,
                                                        bool reserve_cpu_packed_weights = false);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MEMORYPLAN_H
