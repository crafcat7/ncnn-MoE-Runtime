#ifndef NCNN_MOE_LATENTATTENTION_VULKAN_H
#define NCNN_MOE_LATENTATTENTION_VULKAN_H

#include "kernels/activationbuffer.h"
#include "graph/layerplan.h"

#if NCNN_MOE_WITH_VULKAN
#include <mat.h>
namespace ncnn {
class VkCompute;
class Option;
} // namespace ncnn
#endif

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct LayerCache;
struct LatentAttentionRowContext;
struct CompiledOperator;
class Float8Linear_vulkan;
class DeviceTensor_vulkan;
class VulkanContext;
class WeightStore;
class CompiledOperatorTable;

class LatentCache_vulkan
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::VkMat window;
    ncnn::VkMat compressed;
    uint64_t token_count = 0;
    uint32_t compressed_count = 0;
    uint32_t compressed_capacity = 0;
    uint32_t dimension = 0;
    uint32_t window_capacity = 0;
    ncnn::VkMat compressor_state;
    ncnn::VkMat index_compressor_state;
    ncnn::VkMat index_compressed;
    bool complete_state = false;
#endif
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;
};

class LatentValueWork_vulkan
{
public:
    LatentValueWork_vulkan();
    ~LatentValueWork_vulkan();
    LatentValueWork_vulkan(const LatentValueWork_vulkan&) = delete;
    LatentValueWork_vulkan& operator=(const LatentValueWork_vulkan&) = delete;
    // Publish the appended latent rows only after the segment fence.
    [[nodiscard]] bool commit() noexcept;
    void abort() noexcept;

private:
    friend class LatentAttention_vulkan;
    class Implementation;
    std::unique_ptr<Implementation> d;
};

class LatentAttentionWork_vulkan
{
public:
    LatentAttentionWork_vulkan();
    ~LatentAttentionWork_vulkan();
    LatentAttentionWork_vulkan(const LatentAttentionWork_vulkan&) = delete;
    LatentAttentionWork_vulkan& operator=(const LatentAttentionWork_vulkan&) = delete;
    // Publish cache shadows only after the caller has successfully waited.
    [[nodiscard]] bool commit();

private:
    friend class LatentAttention_vulkan;
    class Implementation;
    std::unique_ptr<Implementation> d;
};

class LatentAttention_vulkan
{
public:
    // Private FP32 buffers retained through the fence, one element count per
    // allocation. Cache/state and the logical output are planned separately.
    [[nodiscard]] static std::vector<uint64_t> scratch_elements(const AttentionBlockPlan& plan, uint32_t hidden,
                                                                uint32_t rows, uint32_t context_tokens);
    ~LatentAttention_vulkan();

    [[nodiscard]] static std::shared_ptr<LatentAttention_vulkan> create(const AttentionBlockPlan& plan,
                                                                        const TensorData* sinks,
                                                                        std::shared_ptr<Float8Linear_vulkan> output_a,
                                                                        std::shared_ptr<Float8Linear_vulkan> output_b);
    [[nodiscard]] static std::shared_ptr<LatentAttention_vulkan> create(const AttentionBlockPlan& plan,
                                                                        const TensorData* sinks,
                                                                        const CompiledOperator& output_a,
                                                                        const CompiledOperator& output_b);
    // CPU caches have already committed the current row. Mirrors are optional
    // and may be discarded without affecting fallback or speculative undo.
    [[nodiscard]] bool forward_batch(std::span<const uint64_t> positions,
                                     std::span<LayerCache* const> caches,
                                     std::span<const LatentAttentionRowContext> contexts,
                                     const ActivationBuffer& query,
                                     const DeviceTensor_vulkan* device_query,
                                     ActivationBuffer& output) const;

    // Expanded-value MLA: query and key_value already carry the absorbed
    // representation, norm, and RoPE. gate is the per-head output gate logit.
    [[nodiscard]] bool forward_value_batch(std::span<const uint64_t> positions,
                                           std::span<LayerCache* const> caches,
                                           const ActivationBuffer& query,
                                           const ActivationBuffer& key_value,
                                           const ActivationBuffer& gate,
                                           ActivationBuffer& output) const;

    [[nodiscard]] bool forward_projected_batch(const ActivationBuffer& input,
                                               std::span<const uint64_t> positions,
                                               std::span<LayerCache* const> caches,
                                               ActivationBuffer& output) const;

    [[nodiscard]] bool prepare(const WeightStore& weights, const CompiledOperatorTable& operators, uint64_t optimization_flags);
    // Expanded-value MLA keeps activations on the device for the resident
    // command graph. The caller supplies the pre-attention normalized input.
    [[nodiscard]] bool prepare_value(const WeightStore& weights, const CompiledOperatorTable& operators);
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] bool record_value_forward(const ncnn::VkMat& input,
                                            std::span<const uint64_t> positions,
                                            std::span<LayerCache* const> caches,
                                            ncnn::VkMat& output,
                                            ncnn::VkCompute& cmd,
                                            LatentValueWork_vulkan& work,
                                            std::vector<ncnn::VkMat>& storage) const;
    [[nodiscard]] bool can_record() const noexcept;
    // The caller owns the recorder and holds the context command lock.
    [[nodiscard]] bool record_batch(const ncnn::VkMat& input,
                                    std::span<const uint64_t> positions,
                                    std::span<LayerCache* const> caches,
                                    ncnn::VkMat& output,
                                    ncnn::VkCompute& cmd,
                                    LatentAttentionWork_vulkan& work) const;
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
#endif

private:
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] static std::shared_ptr<VulkanContext> projection_context(const CompiledOperator& op) noexcept;
    [[nodiscard]] static const ncnn::Option* projection_option(const CompiledOperator& op) noexcept;
    [[nodiscard]] static uint32_t projection_input_columns(const CompiledOperator& op) noexcept;
    [[nodiscard]] static uint32_t projection_output_columns(const CompiledOperator& op) noexcept;
    [[nodiscard]] bool record_projection(const CompiledOperator& op,
                                         const ncnn::VkMat& input,
                                         ncnn::VkMat& output,
                                         ncnn::VkCompute& cmd,
                                         std::vector<ncnn::VkMat>& storage) const;
#endif
    class Implementation;
    LatentAttention_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LATENTATTENTION_VULKAN_H
