#ifndef NCNN_MOE_COMPILER_H
#define NCNN_MOE_COMPILER_H

#include "compiledmodel.h"
#include "resourceplan.h"

namespace ncnn {
namespace moe {

[[nodiscard]] Result<void> validate_model_descriptor(const MoeModelDescriptor& descriptor);

// Retired bit 4 stays unused; resident dense host copies are reclaimed automatically.
#define NCNN_MOE_COMPILER_BACKEND_CPU_BIT           0
#define NCNN_MOE_COMPILER_BACKEND_VULKAN_DENSE_BIT  1
#define NCNN_MOE_COMPILER_BACKEND_VULKAN_ATTN_BIT   2
#define NCNN_MOE_COMPILER_BACKEND_MXFP4_CPU_BIT     3
#define NCNN_MOE_COMPILER_BACKEND_VULKAN_EXPERT_BIT 5
#define NCNN_MOE_COMPILER_BACKEND_FILE_EXPERT_BIT   6

enum BackendFlag : uint32_t
{
    BackendCpuExecution = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_CPU_BIT,
    BackendVulkanDense = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_VULKAN_DENSE_BIT,
    BackendVulkanAttention = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_VULKAN_ATTN_BIT,
    BackendMxfp4CpuKernel = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_MXFP4_CPU_BIT,
    BackendVulkanExperts = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_VULKAN_EXPERT_BIT,
    BackendFileBackedExperts = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_FILE_EXPERT_BIT
};

struct CompilerOption
{
    uint32_t flags = BackendCpuExecution | BackendMxfp4CpuKernel;
    uint32_t device_index = automatic_vulkan_device_index;
    uint32_t num_concurrent_sessions = 1;
    // Direct-compiler allocation allowance, measured from this compilation's
    // live usage baseline. Zero uses the live device heap. Runtime loading
    // supplies absolute, independent snapshots and leaves this allowance zero.
    uint64_t gpu_heap_budget = 0;
    // Preserve explicitly requested executable/victim cache capacity while
    // placing persistent attention and output projections.
    uint64_t gpu_expert_cache_reserve = 0;
    std::vector<GpuResourceSnapshot> gpu_resources;
    // Parallel to gpu_resources; explicit executable/victim cache reservations.
    // Resource planning adds the automatic minimum GPU Expert working set.
    std::vector<uint64_t> gpu_expert_cache_reserves;
    uint64_t optimization_flags = OptimizationDefaultFlags;
    VulkanRuntimePtr vulkan_runtime;
    std::vector<uint32_t> device_indices;
    std::vector<uint32_t> device_scores;
};

// Compile into a fresh model. On failure, discard the partially populated output.
[[nodiscard]] Result<void> compile_model(MoeModelDescriptor descriptor, WeightMapping mapping, CompiledModel& compiled, HybridMode hybrid_mode = HybridMode::CpuOnly);

[[nodiscard]] Result<void> compile_model(MoeModelDescriptor descriptor, WeightMapping mapping, CompiledModel& compiled, HybridMode hybrid_mode, const CompilerOption& opt);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_COMPILER_H
