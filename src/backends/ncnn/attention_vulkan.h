#ifndef NCNN_MOE_ATTENTION_VULKAN_H
#define NCNN_MOE_ATTENTION_VULKAN_H

#include "kernels/activationbuffer.h"

#include "ncnn/moe/types.h"
#include "ncnn/moe/option.h"

#if NCNN_MOE_WITH_VULKAN
#include <mat.h>
#endif

#include <cstdint>
#include <memory>
#include <span>

namespace ncnn {
#if NCNN_MOE_WITH_VULKAN
class VkCompute;
#endif
namespace moe {

struct LayerCache;
class Linear;
class Bfloat16Linear_vulkan;

#if NCNN_MOE_WITH_VULKAN
class AttentionCache_vulkan
{
public:
    // Double-written rows keep wrapped cache views contiguous.
    ncnn::VkMat key;
    ncnn::VkMat value;
};
#else
class AttentionCache_vulkan;
#endif

#define NCNN_MOE_VULKAN_ATTN_SINK_BIT        0
#define NCNN_MOE_VULKAN_ATTN_QK_NORM_BIT     1
#define NCNN_MOE_VULKAN_ATTN_OUTPUT_GATE_BIT 2

enum AttentionFlag : uint32_t
{
    AttentionSink = UINT32_C(1) << NCNN_MOE_VULKAN_ATTN_SINK_BIT,
    AttentionQueryKeyNorm = UINT32_C(1) << NCNN_MOE_VULKAN_ATTN_QK_NORM_BIT,
    AttentionOutputGate = UINT32_C(1) << NCNN_MOE_VULKAN_ATTN_OUTPUT_GATE_BIT
};

struct AttentionConfig_vulkan
{
    uint32_t hidden_size = 0;
    uint32_t head_count = 0;
    uint32_t kv_head_count = 0;
    uint32_t head_dimension = 0;
    uint32_t rope_head_dimension = 0;
    uint32_t sliding_window = 0;
    uint32_t initial_context_length = 0;
    float norm_epsilon = 1e-5f;
    float norm_weight_offset = 0.0f;
    float rope_theta = 10000.0f;
    float rope_scaling_factor = 1.0f;
    float rope_ntk_alpha = 1.0f;
    float rope_ntk_beta = 32.0f;
    DType activation_dtype = DType::Float32;
    DType kv_cache_dtype = DType::Float32;
    uint32_t flags = 0;
    uint64_t optimization_flags = OptimizationDefaultFlags;
};

enum class AttentionBatchResult_vulkan
{
    NotExecuted,
    Executed,
    Failed
};

struct AttentionBatchEntry_vulkan
{
    uint64_t position_offset = 0;
    LayerCache* cache = nullptr;
    const ActivationBuffer* input = nullptr;
    ActivationBuffer* output = nullptr;
};

class Attention_vulkan
{
public:
    ~Attention_vulkan();

    [[nodiscard]] static std::shared_ptr<Attention_vulkan> create(const TensorData& norm_weight, const TensorData* sinks,
                                                                  std::shared_ptr<Linear> fused_qkv,
                                                                  std::shared_ptr<Linear> output_projection,
                                                                  const AttentionConfig_vulkan& config);
    [[nodiscard]] static std::shared_ptr<Attention_vulkan> create(
        const TensorData& norm_weight,
        const TensorData& query_norm_weight,
        const TensorData& key_norm_weight,
        const TensorData* sinks,
        std::shared_ptr<Bfloat16Linear_vulkan> fused_qkv_gate,
        std::shared_ptr<Bfloat16Linear_vulkan> output_projection,
        const AttentionConfig_vulkan& config);
    // One upload, one submission, and one download per Attention block.
    [[nodiscard]] bool forward(uint64_t position_offset, LayerCache& cache, const ActivationBuffer& input, ActivationBuffer& output) const;
    // Materializes an otherwise valid device KV ring into the CPU cache.  This
    // is the recovery boundary used when a new Vulkan dispatch cannot be
    // committed but the previous device ring is still authoritative.
    [[nodiscard]] bool materialize_device_cache(LayerCache& cache) const;
    void record_cpu_fallback() const noexcept;
    // Independent one-row Session caches recorded into one queue submission.
    [[nodiscard]] AttentionBatchResult_vulkan forward_batch(
        std::span<const AttentionBatchEntry_vulkan> entries) const;

private:
    class Implementation;

#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] bool support_qkv_rope(size_t token_count) const noexcept;

    [[nodiscard]] bool record_qkv_rope(
        const ncnn::VkMat& fused_qkv,
        const ncnn::VkMat& cosine,
        const ncnn::VkMat& sine,
        size_t token_count,
        uint64_t position_offset,
        bool device_rope,
        const AttentionCache_vulkan* ring,
        uint64_t ring_capacity,
        uint64_t destination_start,
        ncnn::VkMat& query,
        ncnn::VkMat& key,
        ncnn::VkMat& value,
        ncnn::VkCompute& cmd) const;

    [[nodiscard]] bool record_qkv_norm_rope(
        const ncnn::VkMat& fused_qkv,
        const ncnn::VkMat& cosine,
        const ncnn::VkMat& sine,
        size_t token_count,
        uint64_t position_offset,
        bool device_rope,
        const AttentionCache_vulkan* ring,
        uint64_t ring_capacity,
        uint64_t destination_start,
        ncnn::VkMat& query,
        ncnn::VkMat& key,
        ncnn::VkMat& value,
        ncnn::VkMat& gate,
        ncnn::VkCompute& cmd) const;
#endif

    Attention_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_ATTENTION_VULKAN_H
