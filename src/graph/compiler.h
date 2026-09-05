#ifndef NCNN_MOE_COMPILER_H
#define NCNN_MOE_COMPILER_H

#include "compiledmodel.h"

namespace ncnn {
namespace moe {

[[nodiscard]] Result<void> validate_model_descriptor(const MoeModelDescriptor& descriptor);

#define NCNN_MOE_COMPILER_BACKEND_CPU_BIT              0
#define NCNN_MOE_COMPILER_BACKEND_VULKAN_DENSE_BIT     1
#define NCNN_MOE_COMPILER_BACKEND_VULKAN_ATTN_BIT      2
#define NCNN_MOE_COMPILER_BACKEND_MXFP4_CPU_BIT        3
#define NCNN_MOE_COMPILER_BACKEND_RETAIN_CPU_DENSE_BIT 4
#define NCNN_MOE_COMPILER_BACKEND_RELEASE_DENSE_BIT    5
#define NCNN_MOE_COMPILER_BACKEND_VULKAN_EXPERT_BIT    6
#define NCNN_MOE_COMPILER_BACKEND_FILE_EXPERT_BIT      7

enum BackendFlag : uint32_t
{
    BackendCpuExecution = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_CPU_BIT,
    BackendVulkanDense = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_VULKAN_DENSE_BIT,
    BackendVulkanAttention = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_VULKAN_ATTN_BIT,
    BackendMxfp4CpuKernel = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_MXFP4_CPU_BIT,
    BackendRetainCpuDenseCopies = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_RETAIN_CPU_DENSE_BIT,
    BackendReleaseVulkanDenseHostStorage = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_RELEASE_DENSE_BIT,
    BackendVulkanExperts = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_VULKAN_EXPERT_BIT,
    BackendFileBackedExperts = UINT32_C(1) << NCNN_MOE_COMPILER_BACKEND_FILE_EXPERT_BIT
};

struct CompilerOption
{
    uint32_t flags = BackendCpuExecution | BackendMxfp4CpuKernel | BackendRetainCpuDenseCopies;
    uint32_t device_index = automatic_vulkan_device_index;
    uint32_t num_concurrent_sessions = 1;
    // Total heap budget available to the selected Vulkan placement.  The
    // compiler uses this as a conservative admission limit for optional
    // persistent GPU attention state; zero means that no budget-based
    // promotion should be attempted.
    uint64_t gpu_heap_budget = 0;
    uint64_t optimization_flags = OptimizationDefaultFlags;
    NcnnVulkanContextInstancePtr vkctx;
    std::vector<uint32_t> device_indices;
    std::vector<uint32_t> device_scores;
};

[[nodiscard]] Result<CompiledModel> compile_model(MoeModelDescriptor descriptor, WeightMapping mapping, HybridMode hybrid_mode = HybridMode::CpuOnly);

[[nodiscard]] Result<CompiledModel> compile_model(MoeModelDescriptor descriptor, WeightMapping mapping, HybridMode hybrid_mode, const CompilerOption& opt);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_COMPILER_H
