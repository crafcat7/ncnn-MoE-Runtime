#ifndef NCNN_MOE_EXECUTION_PLAN_H
#define NCNN_MOE_EXECUTION_PLAN_H

#include "ncnn/moe/expert.h"
#include "ncnn/moe/execution_graph.h"
#include "ncnn/moe/memory_plan.h"
#include "ncnn/moe/moe_ir.h"
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
class IExpertExecutionBackend;

using TensorHandle = uint32_t;
inline constexpr TensorHandle invalid_tensor_handle = std::numeric_limits<TensorHandle>::max();

class WeightTable
{
private:
    std::vector<TensorData> tensors_;
    std::unordered_map<std::string, TensorHandle> handles_;

public:
    [[nodiscard]] Result<TensorHandle> add(std::string name, TensorData tensor);
    [[nodiscard]] const TensorData& at(TensorHandle handle) const;
    [[nodiscard]] TensorData& at_mutable(TensorHandle handle);
    [[nodiscard]] TensorHandle find_handle(const std::string& name) const noexcept;
    [[nodiscard]] size_t size() const noexcept
    {
        return tensors_.size();
    }
};

#define NCNN_MOE_EXPERT_PLAN_GATED_BIT          0
#define NCNN_MOE_EXPERT_PLAN_PACKED_GATE_UP_BIT 1

enum ExpertPlanFlag : uint32_t
{
    ExpertPlanGated = UINT32_C(1) << NCNN_MOE_EXPERT_PLAN_GATED_BIT,
    ExpertPlanPackedGateUp = UINT32_C(1) << NCNN_MOE_EXPERT_PLAN_PACKED_GATE_UP_BIT
};

struct ExpertPlan
{
    std::shared_ptr<Expert> runtime;
    std::string cache_key;
    TensorHandle gate_weight = invalid_tensor_handle;
    TensorHandle up_weight = invalid_tensor_handle;
    TensorHandle gate_up_weight = invalid_tensor_handle;
    TensorHandle down_weight = invalid_tensor_handle;
    TensorHandle gate_up_bias = invalid_tensor_handle;
    TensorHandle down_bias = invalid_tensor_handle;
    ExpertActivation activation = ExpertActivation::Silu;
    float activation_limit = 0.0f;
    uint64_t weight_bytes = 0;
    uint32_t flags = ExpertPlanGated;
};

#define NCNN_MOE_ATTN_PLAN_SINK_BIT       0
#define NCNN_MOE_ATTN_PLAN_QK_NORM_BIT    1
#define NCNN_MOE_ATTN_PLAN_LATENT_BIT     2
#define NCNN_MOE_ATTN_PLAN_COMPRESSED_BIT 3
#define NCNN_MOE_ATTN_PLAN_DELTA_BIT      4
#define NCNN_MOE_ATTN_PLAN_GATE_BIT       5

enum AttentionBlockFlag : uint32_t
{
    AttentionBlockSink = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_SINK_BIT,
    AttentionBlockQueryKeyNorm = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_QK_NORM_BIT,
    AttentionBlockLatent = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_LATENT_BIT,
    AttentionBlockCompressed = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_COMPRESSED_BIT,
    AttentionBlockGatedDeltaNet = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_DELTA_BIT,
    AttentionBlockOutputGate = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_GATE_BIT
};

struct AttentionBlockPlan
{
    std::shared_ptr<NcnnVulkanAttentionOperator> vulkan_attention_operator;
    std::shared_ptr<NcnnLinearOperator> fused_qkv_operator;
    std::shared_ptr<NcnnVulkanBfloat16Operator>
        fused_qkv_bfloat16_operator;
    std::shared_ptr<NcnnVulkanBfloat16Operator>
        fused_qkv_gate_bfloat16_operator;
    std::shared_ptr<NcnnLinearOperator> fused_delta_input_operator;
    std::shared_ptr<NcnnVulkanBfloat16Operator>
        fused_delta_input_bfloat16_operator;
    TensorHandle pre_attention_norm_weight = invalid_tensor_handle;
    TensorHandle query_weight = invalid_tensor_handle;
    TensorHandle query_bias = invalid_tensor_handle;
    TensorHandle query_norm_weight = invalid_tensor_handle;
    TensorHandle key_weight = invalid_tensor_handle;
    TensorHandle key_bias = invalid_tensor_handle;
    TensorHandle key_norm_weight = invalid_tensor_handle;
    TensorHandle value_weight = invalid_tensor_handle;
    TensorHandle value_bias = invalid_tensor_handle;
    TensorHandle output_weight = invalid_tensor_handle;
    TensorHandle output_bias = invalid_tensor_handle;
    TensorHandle output_gate_weight = invalid_tensor_handle;
    TensorHandle sinks = invalid_tensor_handle;
    TensorHandle delta_qkv_weight = invalid_tensor_handle;
    TensorHandle delta_z_weight = invalid_tensor_handle;
    TensorHandle delta_beta_weight = invalid_tensor_handle;
    TensorHandle delta_alpha_weight = invalid_tensor_handle;
    TensorHandle delta_convolution_weight = invalid_tensor_handle;
    TensorHandle delta_time_bias = invalid_tensor_handle;
    TensorHandle delta_decay_log = invalid_tensor_handle;
    TensorHandle delta_norm_weight = invalid_tensor_handle;
    TensorHandle query_a_weight = invalid_tensor_handle;
    TensorHandle query_b_weight = invalid_tensor_handle;
    TensorHandle key_value_weight = invalid_tensor_handle;
    TensorHandle key_value_norm_weight = invalid_tensor_handle;
    TensorHandle output_a_weight = invalid_tensor_handle;
    TensorHandle output_b_weight = invalid_tensor_handle;
    TensorHandle compressor_position = invalid_tensor_handle;
    TensorHandle compressor_norm_weight = invalid_tensor_handle;
    TensorHandle compressor_key_value_weight = invalid_tensor_handle;
    TensorHandle compressor_gate_weight = invalid_tensor_handle;
    TensorHandle indexer_compressor_position = invalid_tensor_handle;
    TensorHandle indexer_compressor_norm_weight = invalid_tensor_handle;
    TensorHandle indexer_compressor_key_value_weight = invalid_tensor_handle;
    TensorHandle indexer_compressor_gate_weight = invalid_tensor_handle;
    TensorHandle indexer_query_weight = invalid_tensor_handle;
    TensorHandle indexer_weights_weight = invalid_tensor_handle;

    uint32_t head_count = 0;
    uint32_t kv_head_count = 0;
    uint32_t head_dimension = 0;
    uint32_t value_head_dimension = 0;
    uint32_t sliding_window = 0;
    uint32_t initial_context_length = 0;
    uint32_t max_context_length = 0;
    uint32_t query_lora_rank = 0;
    uint32_t rope_head_dimension = 0;
    uint32_t output_lora_rank = 0;
    uint32_t output_group_count = 0;
    uint32_t compression_ratio = 0;
    uint32_t index_head_count = 0;
    uint32_t index_head_dimension = 0;
    uint32_t index_top_k = 0;
    uint32_t convolution_kernel_size = 0;
    float rope_theta = 10000.0f;
    float compressed_rope_theta = 10000.0f;
    float rope_scaling_factor = 1.0f;
    float rope_ntk_alpha = 1.0f;
    float rope_ntk_beta = 32.0f;
    float norm_weight_offset = 0.0f;
    uint32_t flags = 0;
};

struct HyperConnectionPlan
{
    TensorHandle attention_function = invalid_tensor_handle;
    TensorHandle attention_base = invalid_tensor_handle;
    TensorHandle attention_scale = invalid_tensor_handle;
    TensorHandle ffn_function = invalid_tensor_handle;
    TensorHandle ffn_base = invalid_tensor_handle;
    TensorHandle ffn_scale = invalid_tensor_handle;
};

#define NCNN_MOE_BLOCK_NORMALIZE_TOPK_BIT 0

enum MoeBlockFlag : uint32_t
{
    MoeBlockNormalizeTopKWeights = UINT32_C(1) << NCNN_MOE_BLOCK_NORMALIZE_TOPK_BIT
};

struct MoeBlockPlan
{
    TensorHandle pre_ffn_norm_weight = invalid_tensor_handle;
    TensorHandle router_weight = invalid_tensor_handle;
    TensorHandle router_bias = invalid_tensor_handle;
    TensorHandle router_selection_bias = invalid_tensor_handle;
    TensorHandle token_experts = invalid_tensor_handle;
    TensorHandle shared_expert_gate_weight = invalid_tensor_handle;
    std::shared_ptr<NcnnVulkanBfloat16Operator>
        fused_shared_input_bfloat16_operator;
    std::vector<ExpertPlan> experts;
    ExpertPlan shared_expert;

    uint32_t top_k = 0;
    uint32_t hidden_size = 0;
    RouterScoreFunction score_function = RouterScoreFunction::Softmax;
    RouterNormalization normalization = RouterNormalization::SelectedExperts;
    float routed_scaling_factor = 1.0f;
    bool has_shared_expert = false;
    uint32_t flags = MoeBlockNormalizeTopKWeights;
};

struct CompiledNodePlan
{
    ModelNodeType type = ModelNodeType::RmsNorm;
    ExecutionBackend backend = ExecutionBackend::Cpu;
};

#define NCNN_MOE_COMPILED_LAYER_ATTENTION_BIT 0

enum CompiledLayerFlag : uint32_t
{
    CompiledLayerAttention = UINT32_C(1) << NCNN_MOE_COMPILED_LAYER_ATTENTION_BIT
};

struct CompiledLayerPlan
{
    uint32_t layer_id = 0;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    std::vector<CompiledNodePlan> nodes;
    AttentionBlockPlan attention;
    HyperConnectionPlan hyper_connection;
    MoeBlockPlan moe;
    uint32_t flags = 0;
};

struct SpeculativeModelPlan
{
    std::vector<CompiledLayerPlan> layers;
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
               && !layers.empty()
               && block_size != 0;
    }
};

struct CompiledModel
{
    MoeIR descriptor;
    std::vector<CompiledLayerPlan> layers;
    ExecutionGraph graph;
    ExecutionSchedule schedule;
    ModelMemoryPlan memory_plan;
    WeightTable weights;
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
    HybridMode hybrid_mode = HybridMode::CpuOnly;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    std::vector<uint32_t> vulkan_device_indices;
    uint32_t runtime_option_flags = 0;
    uint32_t expected_concurrency = 1;
};

#define NCNN_MOE_BACKEND_CAP_CPU_BIT              0
#define NCNN_MOE_BACKEND_CAP_VULKAN_DENSE_BIT     1
#define NCNN_MOE_BACKEND_CAP_VULKAN_ATTN_BIT      2
#define NCNN_MOE_BACKEND_CAP_MXFP4_CPU_BIT        3
#define NCNN_MOE_BACKEND_CAP_RETAIN_CPU_DENSE_BIT 4

class ModelCompiler
{
public:
    enum BackendCapabilityFlag : uint32_t
    {
        BackendCapabilityCpuExecution = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_CPU_BIT,
        BackendCapabilityVulkanDense = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_VULKAN_DENSE_BIT,
        BackendCapabilityVulkanAttention = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_VULKAN_ATTN_BIT,
        BackendCapabilityMxfp4CpuKernel = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_MXFP4_CPU_BIT,
        BackendCapabilityRetainCpuDenseCopies = UINT32_C(1) << NCNN_MOE_BACKEND_CAP_RETAIN_CPU_DENSE_BIT
    };

    struct BackendCapabilities
    {
        uint32_t flags = BackendCapabilityCpuExecution | BackendCapabilityMxfp4CpuKernel | BackendCapabilityRetainCpuDenseCopies;
        uint32_t cpu_parallelism = 1;
        uint32_t vulkan_queue_count = 0;
        uint32_t vulkan_device_index = automatic_vulkan_device_index;
        uint32_t expected_concurrency = 1;
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
