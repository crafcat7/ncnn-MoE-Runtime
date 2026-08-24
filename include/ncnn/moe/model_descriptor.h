#ifndef NCNN_MOE_MODEL_DESCRIPTOR_H
#define NCNN_MOE_MODEL_DESCRIPTOR_H

#include "ncnn/moe/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

#define NCNN_MOE_ATTN_BIAS_BIT             0
#define NCNN_MOE_ATTN_SINK_BIT             1
#define NCNN_MOE_ATTN_QK_NORM_BIT          2
#define NCNN_MOE_ATTN_INTERLEAVED_ROPE_BIT 3
#define NCNN_MOE_ATTN_OUTPUT_GATE_BIT      4

enum AttentionDescriptorFlag : uint32_t
{
    AttentionDescriptorBias = UINT32_C(1) << NCNN_MOE_ATTN_BIAS_BIT,
    AttentionDescriptorSinks = UINT32_C(1) << NCNN_MOE_ATTN_SINK_BIT,
    AttentionDescriptorQueryKeyNorm = UINT32_C(1) << NCNN_MOE_ATTN_QK_NORM_BIT,
    AttentionDescriptorRopeInterleaved = UINT32_C(1) << NCNN_MOE_ATTN_INTERLEAVED_ROPE_BIT,
    AttentionDescriptorOutputGate = UINT32_C(1) << NCNN_MOE_ATTN_OUTPUT_GATE_BIT
};

enum class AttentionKind
{
    Standard,
    GatedDeltaNet,
    MultiHeadLatent
};

struct AttentionDescriptor
{
    AttentionKind kind = AttentionKind::Standard;
    uint32_t head_count = 0;
    uint32_t kv_head_count = 0;
    uint32_t head_dimension = 0;
    uint32_t sliding_window = 0;
    uint32_t initial_context_length = 0;
    uint32_t max_context_length = 0;
    uint32_t query_lora_rank = 0;
    uint32_t kv_lora_rank = 0;
    uint32_t qk_nope_head_dimension = 0;
    uint32_t qk_rope_head_dimension = 0;
    uint32_t value_head_dimension = 0;
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
    DType projection_weight_dtype = DType::Float32;
    uint32_t flags = 0;
};

#define NCNN_MOE_DESC_NORMALIZE_TOPK_BIT  0
#define NCNN_MOE_DESC_SHARED_EXPERT_BIT   1
#define NCNN_MOE_DESC_EXPERT_BIAS_BIT     2
#define NCNN_MOE_DESC_ROUTER_BIAS_BIT     3
#define NCNN_MOE_DESC_PROJECTION_BIAS_BIT 4
#define NCNN_MOE_DESC_SHARED_GATE_BIT     5

enum MoeDescriptorFlag : uint32_t
{
    MoeDescriptorNormalizeTopKWeights = UINT32_C(1) << NCNN_MOE_DESC_NORMALIZE_TOPK_BIT,
    MoeDescriptorSharedExpert = UINT32_C(1) << NCNN_MOE_DESC_SHARED_EXPERT_BIT,
    MoeDescriptorExpertBias = UINT32_C(1) << NCNN_MOE_DESC_EXPERT_BIAS_BIT,
    MoeDescriptorRouterBias = UINT32_C(1) << NCNN_MOE_DESC_ROUTER_BIAS_BIT,
    MoeDescriptorProjectionBias = UINT32_C(1) << NCNN_MOE_DESC_PROJECTION_BIAS_BIT,
    MoeDescriptorSharedExpertGate = UINT32_C(1) << NCNN_MOE_DESC_SHARED_GATE_BIT
};

struct MoeDescriptor
{
    uint32_t expert_count = 0;
    uint32_t top_k = 0;
    uint32_t intermediate_size = 0;
    uint32_t shared_expert_count = 0;
    uint32_t router_group_count = 0;
    uint32_t router_top_k_groups = 0;

    RouterScoreFunction score_function = RouterScoreFunction::Softmax;
    RouterNormalization normalization = RouterNormalization::SelectedExperts;
    ExpertActivation activation = ExpertActivation::Silu;
    ExpertLayout layout = ExpertLayout::GateUpDown;
    DType expert_weight_dtype = DType::Float32;
    DType shared_expert_weight_dtype = DType::Float32;

    float activation_limit = 0.0f;
    float routed_scaling_factor = 1.0f;
    uint32_t flags = MoeDescriptorNormalizeTopKWeights;
};

struct FfnDescriptor
{
    MoeDescriptor moe;
    uint32_t dense_intermediate_size = 0;
};

#define NCNN_MOE_LAYER_ATTENTION_BIT 0
#define NCNN_MOE_LAYER_MOE_BIT       1
#define NCNN_MOE_LAYER_DENSE_FFN_BIT 2

enum LayerDescriptorFlag : uint32_t
{
    LayerDescriptorAttention = UINT32_C(1) << NCNN_MOE_LAYER_ATTENTION_BIT,
    LayerDescriptorMoe = UINT32_C(1) << NCNN_MOE_LAYER_MOE_BIT,
    LayerDescriptorDenseFfn = UINT32_C(1) << NCNN_MOE_LAYER_DENSE_FFN_BIT
};

struct LayerDescriptor
{
    AttentionDescriptor attention;
    FfnDescriptor ffn;

    NormType pre_attention_norm = NormType::None;
    NormType pre_ffn_norm = NormType::RmsNorm;
    uint32_t flags = LayerDescriptorMoe;
};

enum class SpeculativeModelKind
{
    None,
    DSpark,
    Mtp
};

struct MoeModelDescriptor
{
    std::string model_type;

    uint32_t vocabulary_size = 0;
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t layer_count = 0;

    uint32_t attention_head_count = 0;
    uint32_t kv_head_count = 0;
    uint32_t head_dimension = 0;

    uint32_t expert_count = 0;
    uint32_t experts_per_token = 0;

    DType activation_dtype = DType::Float32;
    DType kv_cache_dtype = DType::Float32;
    float norm_epsilon = 1e-5f;
    float norm_weight_offset = 0.0f;
    uint32_t hyper_connection_multiplier = 1;
    uint32_t hyper_connection_iterations = 0;
    float hyper_connection_epsilon = 1e-6f;
    uint32_t hash_routing_layer_count = 0;
    SpeculativeModelKind speculative_kind = SpeculativeModelKind::None;
    uint32_t speculative_layer_count = 0;
    uint32_t speculative_block_size = 0;
    uint32_t speculative_noise_token_id = 0;
    uint32_t speculative_markov_rank = 0;

    std::vector<LayerDescriptor> layers;
    std::vector<uint32_t> speculative_target_layer_ids;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODEL_DESCRIPTOR_H
