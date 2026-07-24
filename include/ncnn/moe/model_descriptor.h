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

    bool use_bias = false;
    bool use_sinks = false;
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

    bool normalize_topk_weights = true;
    bool use_shared_expert = false;
    bool use_expert_bias = false;
    bool use_router_bias = false;
    bool use_projection_bias = false;
    float activation_limit = 0.0f;
};

struct FfnDescriptor
{
    MoeDescriptor moe;
};

struct LayerDescriptor
{
    AttentionDescriptor attention;
    FfnDescriptor ffn;
    std::vector<ModelNodeDescriptor> nodes;

    NormType pre_attention_norm = NormType::None;
    NormType pre_ffn_norm = NormType::RmsNorm;

    // Adapter-side package hints used while constructing nodes and mapping
    // weights. ModelCompiler derives execution structure from nodes.
    bool use_attention = false;
    bool use_moe = true;
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
