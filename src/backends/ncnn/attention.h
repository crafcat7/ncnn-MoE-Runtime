#ifndef NCNN_MOE_NCNN_ATTENTION_H
#define NCNN_MOE_NCNN_ATTENTION_H

#include "kernels/activation.h"

#include "ncnn/moe/types.h"
#include "ncnn/moe/option.h"

#include <cstdint>
#include <memory>
#include <span>

namespace ncnn {
namespace moe {

struct CpuLayerCache;
class NcnnLinearOperator;
class NcnnVulkanBfloat16Operator;

#define NCNN_MOE_NCNN_ATTN_SINK_BIT        0
#define NCNN_MOE_NCNN_ATTN_QK_NORM_BIT     1
#define NCNN_MOE_NCNN_ATTN_OUTPUT_GATE_BIT 2

enum NcnnAttentionFlag : uint32_t
{
    NcnnAttentionSink = UINT32_C(1) << NCNN_MOE_NCNN_ATTN_SINK_BIT,
    NcnnAttentionQueryKeyNorm = UINT32_C(1) << NCNN_MOE_NCNN_ATTN_QK_NORM_BIT,
    NcnnAttentionOutputGate = UINT32_C(1) << NCNN_MOE_NCNN_ATTN_OUTPUT_GATE_BIT
};

struct NcnnVulkanAttentionConfig
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

enum class NcnnVulkanAttentionBatchResult
{
    NotExecuted,
    Executed,
    Failed
};

struct NcnnVulkanAttentionBatchEntry
{
    uint64_t position_offset = 0;
    CpuLayerCache* cache = nullptr;
    const ActivationBuffer* input = nullptr;
    ActivationBuffer* output = nullptr;
};

class NcnnVulkanAttentionOperator
{
private:
    class Implementation;

    NcnnVulkanAttentionOperator();
    std::unique_ptr<Implementation> d;

public:
    ~NcnnVulkanAttentionOperator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanAttentionOperator> create(const TensorData& norm_weight, const TensorData* sinks,
                                                                             std::shared_ptr<NcnnLinearOperator> fused_qkv,
                                                                             std::shared_ptr<NcnnLinearOperator> output_projection,
                                                                             const NcnnVulkanAttentionConfig& config);
    [[nodiscard]] static std::shared_ptr<NcnnVulkanAttentionOperator> create_with_query_key_norm_and_gate(
        const TensorData& norm_weight,
        const TensorData& query_norm_weight,
        const TensorData& key_norm_weight,
        const TensorData* sinks,
        std::shared_ptr<NcnnVulkanBfloat16Operator> fused_qkv_gate,
        std::shared_ptr<NcnnVulkanBfloat16Operator> output_projection,
        const NcnnVulkanAttentionConfig& config);
    // One upload, one submission, and one download per Attention block.
    [[nodiscard]] bool forward(uint64_t position_offset, CpuLayerCache& cache, const ActivationBuffer& input, ActivationBuffer& output) const;
    // Materializes an otherwise valid device KV ring into the CPU cache.  This
    // is the recovery boundary used when a new Vulkan dispatch cannot be
    // committed but the previous device ring is still authoritative.
    [[nodiscard]] bool materialize_device_cache(CpuLayerCache& cache) const;
    void record_cpu_fallback() const noexcept;
    // Independent one-row Session caches recorded into one queue submission.
    [[nodiscard]] NcnnVulkanAttentionBatchResult forward_batch(
        std::span<const NcnnVulkanAttentionBatchEntry> entries) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_ATTENTION_H
