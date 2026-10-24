#ifndef NCNN_MOE_QSAATTENTION_VULKAN_H
#define NCNN_MOE_QSAATTENTION_VULKAN_H

#include "graph/layerplan.h"
#include "vulkan.h"

#if NCNN_MOE_WITH_VULKAN
#include <mat.h>
namespace ncnn {
class VkCompute;
class Option;
} // namespace ncnn
#endif

#include <cstdint>
#include <memory>

namespace ncnn {
namespace moe {

class WeightStore;
class VulkanContext;
struct LayerCache;

// Device-authoritative QSA state. All KV words preserve the configured cache
// precision; QSA raw keys are rounded to BF16 before block averaging.
class QsaCache_vulkan
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::VkMat keys;
    ncnn::VkMat values;
    ncnn::VkMat index_raw;
    ncnn::VkMat pooled_keys;
    uint64_t start_position = 0;
    uint32_t token_count = 0;
    uint32_t capacity = 0;
    uint32_t columns = 0;
    uint32_t index_dimension = 0;
    uint32_t compression_ratio = 0;
    DType dtype = DType::Float32;
#endif
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;
};

class QsaAttentionWork_vulkan
{
public:
    QsaAttentionWork_vulkan();
    ~QsaAttentionWork_vulkan();
    QsaAttentionWork_vulkan(const QsaAttentionWork_vulkan&) = delete;
    QsaAttentionWork_vulkan& operator=(const QsaAttentionWork_vulkan&) = delete;
    // The caller must successfully wait for the recorded command first.
    [[nodiscard]] bool commit();

private:
    friend class QsaAttention_vulkan;
    class Implementation;
    std::unique_ptr<Implementation> d;
};

class QsaAttention_vulkan
{
public:
    ~QsaAttention_vulkan();
    [[nodiscard]] static std::shared_ptr<QsaAttention_vulkan> create(const WeightStore& weights,
                                                                     const AttentionBlockPlan& plan,
                                                                     uint32_t device_index,
                                                                     const VulkanRuntimePtr& runtime,
                                                                     uint64_t optimization_flags);
    [[nodiscard]] uint64_t device_bytes() const noexcept;
    [[nodiscard]] uint64_t initial_state_bytes() const noexcept;
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] const std::shared_ptr<VulkanContext>& vulkan_context() const noexcept;
    [[nodiscard]] const ncnn::Option& option() const noexcept;
    // Caller owns the context command lock and retains work until fence wait.
    // Input/output are row-major FP32 device activations. No CPU fallback or
    // implicit cache promotion is performed. Transactions and sliding QSA are
    // rejected because the established QSA contract retains the full prefix.
    [[nodiscard]] bool record_forward(const ncnn::VkMat& input,
                                      LayerCache& cache,
                                      uint64_t position_offset,
                                      ncnn::VkMat& output,
                                      ncnn::VkCompute& command,
                                      QsaAttentionWork_vulkan& work) const;
#endif

private:
    QsaAttention_vulkan();
    class Implementation;
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif
