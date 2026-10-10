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

namespace ncnn {
namespace moe {

struct LayerCache;
struct LatentAttentionRowContext;
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
    ~LatentAttention_vulkan();

    [[nodiscard]] static std::shared_ptr<LatentAttention_vulkan> create(const AttentionBlockPlan& plan,
                                                                        const TensorData& sinks,
                                                                        std::shared_ptr<Float8Linear_vulkan> output_a,
                                                                        std::shared_ptr<Float8Linear_vulkan> output_b);
    // CPU caches have already committed the current row. Mirrors are optional
    // and may be discarded without affecting fallback or speculative undo.
    [[nodiscard]] bool forward_batch(std::span<const uint64_t> positions,
                                     std::span<LayerCache* const> caches,
                                     std::span<const LatentAttentionRowContext> contexts,
                                     const ActivationBuffer& query,
                                     const DeviceTensor_vulkan* device_query,
                                     ActivationBuffer& output) const;

    [[nodiscard]] bool forward_projected_batch(const ActivationBuffer& input,
                                               std::span<const uint64_t> positions,
                                               std::span<LayerCache* const> caches,
                                               ActivationBuffer& output) const;

    [[nodiscard]] bool prepare(const WeightStore& weights, const CompiledOperatorTable& operators, uint64_t optimization_flags);
#if NCNN_MOE_WITH_VULKAN
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
    class Implementation;
    LatentAttention_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LATENTATTENTION_VULKAN_H
