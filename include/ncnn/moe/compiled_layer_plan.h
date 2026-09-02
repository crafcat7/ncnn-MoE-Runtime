#ifndef NCNN_MOE_COMPILED_LAYER_PLAN_H
#define NCNN_MOE_COMPILED_LAYER_PLAN_H

#include "ncnn/moe/compiled_operator.h"
#include "ncnn/moe/expert.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

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
    uint64_t weight_size = 0;
    uint32_t flags = ExpertPlanGated;
};

#define NCNN_MOE_ATTN_PLAN_SINK_BIT       0
#define NCNN_MOE_ATTN_PLAN_QK_NORM_BIT    1
#define NCNN_MOE_ATTN_PLAN_LATENT_BIT     2
#define NCNN_MOE_ATTN_PLAN_COMPRESSED_BIT 3
#define NCNN_MOE_ATTN_PLAN_DELTA_BIT      4
#define NCNN_MOE_ATTN_PLAN_GATE_BIT       5
#define NCNN_MOE_ATTN_PLAN_QSA_BIT        6
#define NCNN_MOE_ATTN_PLAN_EXTERNAL_BIT   7
#define NCNN_MOE_ATTN_PLAN_SIGMOID_BIT    8

enum AttentionBlockFlag : uint32_t
{
    AttentionBlockSink = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_SINK_BIT,
    AttentionBlockQueryKeyNorm = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_QK_NORM_BIT,
    AttentionBlockLatent = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_LATENT_BIT,
    AttentionBlockCompressed = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_COMPRESSED_BIT,
    AttentionBlockGatedDeltaNet = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_DELTA_BIT,
    AttentionBlockOutputGate = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_GATE_BIT,
    AttentionBlockQsa = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_QSA_BIT,
    AttentionBlockExternalResidual = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_EXTERNAL_BIT,
    AttentionBlockSigmoidGate = UINT32_C(1) << NCNN_MOE_ATTN_PLAN_SIGMOID_BIT
};

struct AttentionBlockPlan
{
    CompiledOperatorHandle vulkan_attention_operator = invalid_compiled_operator_handle;
    CompiledOperatorHandle fused_qkv_operator = invalid_compiled_operator_handle;
    CompiledOperatorHandle fused_qkv_bfloat16_operator = invalid_compiled_operator_handle;
    CompiledOperatorHandle fused_qkv_gate_bfloat16_operator = invalid_compiled_operator_handle;
    CompiledOperatorHandle fused_delta_input_operator = invalid_compiled_operator_handle;
    CompiledOperatorHandle fused_delta_input_bfloat16_operator = invalid_compiled_operator_handle;
    CompiledOperatorHandle gated_delta_vulkan_operator = invalid_compiled_operator_handle;
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
    TensorHandle qsa_query_key_weight = invalid_tensor_handle;
    TensorHandle qsa_query_norm_weight = invalid_tensor_handle;
    TensorHandle qsa_key_norm_weight = invalid_tensor_handle;

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
    uint32_t index_token_budget = 0;
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

struct GatedResidualPlan
{
    TensorHandle norm_weight = invalid_tensor_handle;
    TensorHandle mix_down_weight = invalid_tensor_handle;
    TensorHandle mix_up_weight = invalid_tensor_handle;
    TensorHandle inject_weight = invalid_tensor_handle;
};

struct PleBlockPlan
{
    TensorHandle key_weight = invalid_tensor_handle;
    TensorHandle value_weight = invalid_tensor_handle;
    TensorHandle key_norm_weight = invalid_tensor_handle;
    TensorHandle query_norm_weight = invalid_tensor_handle;
    TensorHandle convolution_norm_weight = invalid_tensor_handle;
    TensorHandle convolution_weight = invalid_tensor_handle;
    TensorHandle hash_multipliers = invalid_tensor_handle;
    TensorHandle head_vocabulary_sizes = invalid_tensor_handle;
    TensorHandle head_offsets = invalid_tensor_handle;
    std::vector<TensorHandle> embedding_shards;
    uint32_t embedding_dimension = 0;
    uint32_t convolution_kernel_size = 0;
    uint32_t ngram_size = 0;
    uint32_t heads_per_ngram = 0;
    uint32_t eos_token_id = 0;

    [[nodiscard]] bool enabled() const noexcept
    {
        return embedding_dimension != 0;
    }
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
    CompiledOperatorHandle fused_shared_input_bfloat16_operator = invalid_compiled_operator_handle;
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

#define NCNN_MOE_COMPILED_LAYER_ATTENTION_BIT 0

enum CompiledLayerFlag : uint32_t
{
    CompiledLayerAttention = UINT32_C(1) << NCNN_MOE_COMPILED_LAYER_ATTENTION_BIT
};

struct CompiledLayerPlan
{
    uint32_t layer_id = 0;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    AttentionBlockPlan attention;
    HyperConnectionPlan hyper_connection;
    GatedResidualPlan attention_gated_residual;
    GatedResidualPlan ffn_gated_residual;
    PleBlockPlan ple;
    MoeBlockPlan moe;
    uint32_t flags = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_COMPILED_LAYER_PLAN_H
