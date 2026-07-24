#ifndef NCNN_MOE_NCNN_ATTENTION_H
#define NCNN_MOE_NCNN_ATTENTION_H

#include "cpu_batch.h"

#include "ncnn/moe/types.h"

#include <cstdint>
#include <memory>

namespace ncnn {
namespace moe {

struct CpuLayerCache;
class NcnnLinearOperator;

struct NcnnVulkanAttentionConfig
{
    uint32_t hidden_size = 0;
    uint32_t head_count = 0;
    uint32_t kv_head_count = 0;
    uint32_t head_dimension = 0;
    uint32_t sliding_window = 0;
    uint32_t initial_context_length = 0;
    float norm_epsilon = 1e-5f;
    float rope_theta = 10000.0f;
    float rope_scaling_factor = 1.0f;
    float rope_ntk_alpha = 1.0f;
    float rope_ntk_beta = 32.0f;
    DType activation_dtype = DType::Float32;
    DType kv_cache_dtype = DType::Float32;
    bool use_attention_sink = false;
};

class NcnnVulkanAttentionOperator
{
public:
    ~NcnnVulkanAttentionOperator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanAttentionOperator> create(
        const TensorData& norm_weight,
        const TensorData* sinks,
        std::shared_ptr<NcnnLinearOperator> fused_qkv,
        std::shared_ptr<NcnnLinearOperator> output_projection,
        const NcnnVulkanAttentionConfig& config);
    [[nodiscard]] static uint64_t current_thread_blocks() noexcept;

    // Executes the complete dense attention block across one heterogeneous
    // boundary: one hidden-state upload, one VkCompute submission containing
    // every ncnn layer, and one hidden-state download for CPU routing/experts.
    [[nodiscard]] bool forward(
        uint64_t position_offset,
        CpuLayerCache& cache,
        const CpuBatch& input,
        CpuBatch& output) const;

private:
    class Implementation;

    NcnnVulkanAttentionOperator();
    std::unique_ptr<Implementation> implementation_;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_ATTENTION_H
