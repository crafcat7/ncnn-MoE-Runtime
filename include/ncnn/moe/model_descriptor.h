#ifndef NCNN_MOE_MODEL_DESCRIPTOR_H
#define NCNN_MOE_MODEL_DESCRIPTOR_H

#include "ncnn/moe/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

enum class ModelNodeType
{
    RmsNorm,
    FusedQkv,
    Rope,
    AttentionSink,
    Sdpa,
    Projection,
    Router,
    TopK,
    ExpertGroup,
    Combine
};

struct ModelNodeDescriptor
{
    ModelNodeType type = ModelNodeType::RmsNorm;
};

enum AttentionDescriptorFlag : uint32_t
{
    AttentionDescriptorBias = 1u << 0,
    AttentionDescriptorSinks = 1u << 1
};

struct AttentionDescriptor
{
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
    uint32_t flags = 0;
};

enum MoeDescriptorFlag : uint32_t
{
    MoeDescriptorNormalizeTopKWeights = 1u << 0,
    MoeDescriptorSharedExpert = 1u << 1,
    MoeDescriptorExpertBias = 1u << 2,
    MoeDescriptorRouterBias = 1u << 3,
    MoeDescriptorProjectionBias = 1u << 4
};

struct MoeDescriptor
{
    uint32_t expert_count = 0;
    uint32_t top_k = 0;
    uint32_t intermediate_size = 0;

    RouterScoreFunction score_function = RouterScoreFunction::Softmax;
    RouterNormalization normalization = RouterNormalization::SelectedExperts;
    ExpertActivation activation = ExpertActivation::Silu;
    ExpertLayout layout = ExpertLayout::GateUpDown;
    DType expert_weight_dtype = DType::Float32;

    float activation_limit = 0.0f;
    uint32_t flags = MoeDescriptorNormalizeTopKWeights;
};

struct FfnDescriptor
{
    MoeDescriptor moe;
};

enum LayerDescriptorFlag : uint32_t
{
    LayerDescriptorAttention = 1u << 0,
    LayerDescriptorMoe = 1u << 1
};

struct LayerDescriptor
{
    AttentionDescriptor attention;
    FfnDescriptor ffn;
    std::vector<ModelNodeDescriptor> nodes;

    NormType pre_attention_norm = NormType::None;
    NormType pre_ffn_norm = NormType::RmsNorm;
    uint32_t flags = LayerDescriptorMoe;
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

    std::vector<LayerDescriptor> layers;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODEL_DESCRIPTOR_H
