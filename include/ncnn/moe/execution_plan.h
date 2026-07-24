#ifndef NCNN_MOE_EXECUTION_PLAN_H
#define NCNN_MOE_EXECUTION_PLAN_H

#include "ncnn/moe/model_descriptor.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace ncnn {
namespace moe {

class Mxfp4ExpertCache;

using TensorHandle = uint32_t;
inline constexpr TensorHandle invalid_tensor_handle = std::numeric_limits<TensorHandle>::max();

class WeightTable
{
public:
    [[nodiscard]] Result<TensorHandle> add(std::string name, TensorData tensor);
    [[nodiscard]] const TensorData& at(TensorHandle handle) const;
    [[nodiscard]] TensorData& at_mutable(TensorHandle handle);
    [[nodiscard]] const TensorData* find(const std::string& name) const noexcept;
    [[nodiscard]] TensorHandle find_handle(const std::string& name) const noexcept;
    [[nodiscard]] size_t size() const noexcept
    {
        return tensors_.size();
    }

private:
    std::vector<TensorData> tensors_;
    std::unordered_map<std::string, TensorHandle> handles_;
};

struct ExpertPlan
{
    TensorHandle gate_weight = invalid_tensor_handle;
    TensorHandle up_weight = invalid_tensor_handle;
    TensorHandle gate_up_weight = invalid_tensor_handle;
    TensorHandle down_weight = invalid_tensor_handle;
    TensorHandle gate_up_bias = invalid_tensor_handle;
    TensorHandle down_bias = invalid_tensor_handle;
    ExpertActivation activation = ExpertActivation::Silu;
    float activation_limit = 0.0f;
    bool gated = true;
};

struct AttentionBlockPlan
{
    std::shared_ptr<NcnnVulkanAttentionOperator> vulkan_attention_operator;
    std::shared_ptr<NcnnLinearOperator> fused_qkv_operator;
    TensorHandle pre_attention_norm_weight = invalid_tensor_handle;
    TensorHandle query_weight = invalid_tensor_handle;
    TensorHandle query_bias = invalid_tensor_handle;
    TensorHandle key_weight = invalid_tensor_handle;
    TensorHandle key_bias = invalid_tensor_handle;
    TensorHandle value_weight = invalid_tensor_handle;
    TensorHandle value_bias = invalid_tensor_handle;
    TensorHandle output_weight = invalid_tensor_handle;
    TensorHandle output_bias = invalid_tensor_handle;
    TensorHandle sinks = invalid_tensor_handle;

    uint32_t head_count = 0;
    uint32_t kv_head_count = 0;
    uint32_t head_dimension = 0;
    uint32_t sliding_window = 0;
    uint32_t initial_context_length = 0;
    uint32_t max_context_length = 0;
    float rope_theta = 10000.0f;
    float rope_scaling_factor = 1.0f;
    float rope_ntk_alpha = 1.0f;
    float rope_ntk_beta = 32.0f;
    bool use_attention_sink = false;
};

struct MoeBlockPlan
{
    TensorHandle pre_ffn_norm_weight = invalid_tensor_handle;
    TensorHandle router_weight = invalid_tensor_handle;
    TensorHandle router_bias = invalid_tensor_handle;
    std::vector<ExpertPlan> experts;

    uint32_t top_k = 0;
    uint32_t hidden_size = 0;
    RouterNormalization normalization = RouterNormalization::SelectedExperts;
    bool normalize_topk_weights = true;
};

enum class ExecutionBackend
{
    Cpu,
    Vulkan
};

struct CompiledNodePlan
{
    ModelNodeType type = ModelNodeType::RmsNorm;
    ExecutionBackend backend = ExecutionBackend::Cpu;
};

struct CompiledLayerPlan
{
    uint32_t layer_id = 0;
    std::vector<CompiledNodePlan> nodes;
    AttentionBlockPlan attention;
    MoeBlockPlan moe;
    bool use_attention = false;
};

struct CompiledModel
{
    MoeModelDescriptor descriptor;
    std::vector<CompiledLayerPlan> layers;
    WeightTable weights;
    TensorHandle token_embedding = invalid_tensor_handle;
    TensorHandle final_norm_weight = invalid_tensor_handle;
    TensorHandle lm_head_weight = invalid_tensor_handle;
    std::shared_ptr<Mxfp4ExpertCache> expert_cache;
    HybridMode hybrid_mode = HybridMode::CpuOnly;
};

class ModelCompiler
{
public:
    struct BackendCapabilities
    {
        bool cpu_execution = true;
        bool vulkan_dense = false;
        bool vulkan_attention = false;
        bool mxfp4_cpu_kernel = true;
        // File-backed expert mode prioritizes fitting the model over retaining
        // an additional transformed ncnn copy of every CPU dense matrix.
        bool retain_cpu_dense_copies = true;
    };

    [[nodiscard]] Result<CompiledModel> compile(
        MoeModelDescriptor descriptor,
        WeightMapping mapping,
        HybridMode hybrid_mode = HybridMode::CpuOnly) const;

    [[nodiscard]] Result<CompiledModel> compile(
        MoeModelDescriptor descriptor,
        WeightMapping mapping,
        HybridMode hybrid_mode,
        const BackendCapabilities& capabilities) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXECUTION_PLAN_H
