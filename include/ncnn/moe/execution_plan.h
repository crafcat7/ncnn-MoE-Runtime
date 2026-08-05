#ifndef NCNN_MOE_EXECUTION_PLAN_H
#define NCNN_MOE_EXECUTION_PLAN_H

#include "ncnn/moe/compiled_layer_plan.h"
#include "ncnn/moe/expert.h"
#include "ncnn/moe/compiled_operator.h"
#include "ncnn/moe/execution_graph.h"
#include "ncnn/moe/memory_plan.h"
#include "ncnn/moe/moe_ir.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/runtime_config.h"
#include "ncnn/moe/types.h"
#include "ncnn/moe/vulkan_context.h"
#include "ncnn/moe/weight_store.h"

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

class Mxfp4ExpertCache;
class IExpertExecutionBackend;

struct SpeculativeModelPlan
{
    // Speculative execution is a separate graph region because its layer
    // payloads are not part of the target model's main layer vector.  The
    // graph is still the sole owner of order, backend placement, and
    // prefetch policy for this region.
    ExecutionGraph graph;
    ExecutionSchedule schedule;
    std::vector<uint32_t> target_layer_ids;
    TensorHandle mtp_embedding_norm_weight = invalid_tensor_handle;
    TensorHandle mtp_hidden_norm_weight = invalid_tensor_handle;
    TensorHandle mtp_input_projection_weight = invalid_tensor_handle;
    TensorHandle main_projection_weight = invalid_tensor_handle;
    TensorHandle main_norm_weight = invalid_tensor_handle;
    TensorHandle final_norm_weight = invalid_tensor_handle;
    TensorHandle hyper_head_function = invalid_tensor_handle;
    TensorHandle hyper_head_base = invalid_tensor_handle;
    TensorHandle hyper_head_scale = invalid_tensor_handle;
    TensorHandle markov_embedding_weight = invalid_tensor_handle;
    TensorHandle markov_head_weight = invalid_tensor_handle;
    TensorHandle confidence_weight = invalid_tensor_handle;
    uint32_t block_size = 0;
    uint32_t noise_token_id = 0;
    uint32_t markov_rank = 0;
    SpeculativeModelKind kind = SpeculativeModelKind::None;

    [[nodiscard]] bool enabled() const noexcept
    {
        return kind != SpeculativeModelKind::None
               && !graph.layer_plans.empty()
               && block_size != 0;
    }
};

struct CompiledModel
{
    MoeIR descriptor;
    ExecutionGraph graph;
    ExecutionSchedule schedule;
    ModelMemoryPlan memory_plan;
    WeightStore weights;
    CompiledOperatorTable operators;
    TensorHandle token_embedding = invalid_tensor_handle;
    TensorHandle final_norm_weight = invalid_tensor_handle;
    TensorHandle lm_head_weight = invalid_tensor_handle;
    TensorHandle hyper_head_function = invalid_tensor_handle;
    TensorHandle hyper_head_base = invalid_tensor_handle;
    TensorHandle hyper_head_scale = invalid_tensor_handle;
    SpeculativeModelPlan speculative;
    std::shared_ptr<Mxfp4ExpertCache> expert_cache;
    std::shared_ptr<IExpertExecutionBackend> expert_backend;
    std::shared_ptr<ExpertStore> expert_store;
    NcnnVulkanContextInstancePtr vulkan_context_instance;
    HybridMode hybrid_mode = HybridMode::CpuOnly;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    std::vector<uint32_t> vulkan_device_indices;
    uint32_t runtime_option_flags = 0;
    uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
    uint32_t expected_concurrency = 1;
    EffectiveRuntimeConfig effective_runtime_config;
};

#define NCNN_MOE_BACKEND_CAP_CPU_BIT              0
#define NCNN_MOE_BACKEND_CAP_VULKAN_DENSE_BIT     1
#define NCNN_MOE_BACKEND_CAP_VULKAN_ATTN_BIT      2
#define NCNN_MOE_BACKEND_CAP_MXFP4_CPU_BIT        3
#define NCNN_MOE_BACKEND_CAP_RETAIN_CPU_DENSE_BIT 4
#define NCNN_MOE_BACKEND_CAP_RELEASE_DENSE_BIT     5
#define NCNN_MOE_BACKEND_CAP_VULKAN_EXPERT_BIT     6

class ModelCompiler
{
public:
    enum BackendCapabilityFlag : uint32_t
    {
        BackendCapabilityCpuExecution = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_CPU_BIT,
        BackendCapabilityVulkanDense = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_VULKAN_DENSE_BIT,
        BackendCapabilityVulkanAttention = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_VULKAN_ATTN_BIT,
        BackendCapabilityMxfp4CpuKernel = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_MXFP4_CPU_BIT,
        BackendCapabilityRetainCpuDenseCopies = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_RETAIN_CPU_DENSE_BIT,
        BackendCapabilityReleaseVulkanDenseHostStorage = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_RELEASE_DENSE_BIT,
        BackendCapabilityVulkanExperts = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_VULKAN_EXPERT_BIT
    };

    struct BackendCapabilities
    {
        uint32_t flags = BackendCapabilityCpuExecution | BackendCapabilityMxfp4CpuKernel | BackendCapabilityRetainCpuDenseCopies;
        uint32_t cpu_parallelism = 1;
        uint32_t vulkan_queue_count = 0;
        uint32_t vulkan_device_index = automatic_vulkan_device_index;
        uint32_t expected_concurrency = 1;
        uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
        NcnnVulkanContextInstancePtr vulkan_context_instance;
        std::vector<uint32_t> vulkan_device_indices;
        std::vector<uint32_t> vulkan_device_scores;
    };

    [[nodiscard]] Result<CompiledModel> compile(MoeIR ir, WeightMapping mapping, HybridMode hybrid_mode = HybridMode::CpuOnly) const;

    [[nodiscard]] Result<CompiledModel> compile(MoeIR ir, WeightMapping mapping, HybridMode hybrid_mode, const BackendCapabilities& capabilities) const;
};

using MoeCompiler = ModelCompiler;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXECUTION_PLAN_H
