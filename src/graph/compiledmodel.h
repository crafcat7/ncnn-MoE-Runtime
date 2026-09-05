#ifndef NCNN_MOE_COMPILEDMODEL_H
#define NCNN_MOE_COMPILEDMODEL_H

#include "layerplan.h"
#include "compiledoperator.h"
#include "graph.h"
#include "memoryplan.h"
#include "ncnn/moe/modeldescriptor.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/option.h"
#include "ncnn/moe/types.h"
#include "backends/ncnn/vulkancontext.h"
#include "storage/weightstore.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

class Model;
class ExpertCache;
class ExpertBackend;

// Runtime settings after Auto values have been resolved.
struct EffectiveOption
{
    HybridMode hybrid_mode = HybridMode::CpuOnly;
    uint64_t expert_gpu_cache_size = 0;
    uint64_t expert_gpu_victim_cache_size = 0;
    uint32_t expert_gpu_victim_reuse_probe_interval = 1;
    uint32_t num_expert_io_threads = 0;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    std::vector<uint32_t> vulkan_device_indices;
    uint32_t flags = 0;
    uint64_t optimization_flags = OptimizationDefaultFlags;
    uint32_t num_concurrent_sessions = 1;
};

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
    MoeModelDescriptor descriptor;
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
    GatedResidualPlan gated_residual_head;
    SpeculativeModelPlan speculative;
    std::shared_ptr<ExpertCache> expert_cache;
    std::shared_ptr<ExpertBackend> expert_backend;
    NcnnVulkanContextInstancePtr vulkan_context_instance;
    EffectiveOption opt;
};

// Internal access to the compiled representation hidden by Model.
[[nodiscard]] const CompiledModel& model_compiled(const Model& model) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_COMPILEDMODEL_H
