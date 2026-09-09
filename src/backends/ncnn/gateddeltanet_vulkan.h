#ifndef NCNN_MOE_GATEDDELTANET_VULKAN_H
#define NCNN_MOE_GATEDDELTANET_VULKAN_H

#include "vulkan.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
class Option;

namespace moe {

class ActivationBuffer;
class Bfloat16Linear_vulkan;
class VulkanContext;
struct LayerCache;
struct TensorData;

struct GatedDeltaBatchEntry_vulkan
{
    const ActivationBuffer* normalized = nullptr;
    LayerCache* cache = nullptr;
    ActivationBuffer* projected = nullptr;
};

enum class GatedDeltaBatchResult_vulkan
{
    NotExecuted,
    Executed,
    Failed
};

class GatedDeltaState_vulkan
{
public:
    [[nodiscard]] static std::shared_ptr<GatedDeltaState_vulkan> create(
        const std::shared_ptr<VulkanContext>& context,
        uint32_t convolution_size,
        uint32_t kernel_size,
        uint32_t head_count,
        uint32_t head_dimension,
        uint32_t value_head_dimension,
        const ncnn::Option& opt);

    ~GatedDeltaState_vulkan();

    [[nodiscard]] bool begin_transaction(size_t expected_rows) noexcept;
    [[nodiscard]] bool prepare_transaction_finish(
        size_t committed_rows,
        size_t recorded_rows) noexcept;
    void complete_transaction() noexcept;
    [[nodiscard]] bool prepare_cpu_state(
        const std::vector<float>& convolution,
        const std::vector<float>& recurrent);
    [[nodiscard]] bool download(
        std::vector<float>& convolution,
        std::vector<float>& recurrent) const;
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;

private:
    friend class GatedDeltaNet_vulkan;
    class Implementation;
    GatedDeltaState_vulkan();
    std::unique_ptr<Implementation> d;
};

class GatedDeltaNet_vulkan
{
public:
    ~GatedDeltaNet_vulkan();

    [[nodiscard]] static std::shared_ptr<GatedDeltaNet_vulkan> create(
        const std::shared_ptr<Bfloat16Linear_vulkan>& fused_input,
        const TensorData& convolution_weight,
        const TensorData& time_bias,
        const TensorData& decay_log,
        const TensorData& norm_weight,
        const std::shared_ptr<Bfloat16Linear_vulkan>& output_projection,
        uint32_t head_count,
        uint32_t kv_head_count,
        uint32_t head_dimension,
        uint32_t value_head_dimension,
        uint32_t convolution_kernel_size,
        float norm_epsilon,
        bool sigmoid_gate,
        uint32_t vulkan_device_index,
        const VulkanRuntimePtr& vulkan_runtime,
        uint64_t optimization_flags);

    [[nodiscard]] bool forward(
        const ActivationBuffer& normalized,
        LayerCache& cache,
        ActivationBuffer& projected) const;
    [[nodiscard]] bool forward_input_rms_norm(
        const ActivationBuffer& input,
        LayerCache& cache,
        ActivationBuffer& projected) const;
    [[nodiscard]] bool has_input_rms_norm() const noexcept;
    [[nodiscard]] GatedDeltaBatchResult_vulkan forward_batch(
        std::span<const GatedDeltaBatchEntry_vulkan> entries) const;

private:
    class Implementation;

    [[nodiscard]] bool forward_impl(
        const ActivationBuffer& input,
        LayerCache& cache,
        ActivationBuffer& projected,
        bool apply_input_rms_norm) const;

    GatedDeltaNet_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GATEDDELTANET_VULKAN_H
