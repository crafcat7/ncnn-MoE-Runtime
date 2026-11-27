#ifndef NCNN_MOE_LINEAR_MXFP4_H
#define NCNN_MOE_LINEAR_MXFP4_H

#include "backends/ncnn/linear/core.h"
#include "kernels/ops.h"

#include <algorithm>
#include <cstring>

namespace ncnn {
namespace moe {

// Packs MXFP4 blocks, scales, and float/BF16 bias into one device-ready buffer.
// Caller validates segment sizes and float alignment of the bias offset.
inline void pack_mxfp4_weight_storage(std::span<uint8_t> storage,
                                      size_t packed_segment_size, size_t scales_segment_size,
                                      std::span<const uint8_t> blocks, std::span<const uint8_t> scales,
                                      std::span<const float> float_bias, std::span<const uint16_t> bfloat16_bias) noexcept
{
    std::memcpy(storage.data(), blocks.data(), blocks.size());
    std::memset(storage.data() + blocks.size(), 0, packed_segment_size - blocks.size());
    uint8_t* scale_values = storage.data() + packed_segment_size;
    std::memcpy(scale_values, scales.data(), scales.size());
    std::memset(scale_values + scales.size(), 0, scales_segment_size - scales.size());

    const size_t bias_offset = packed_segment_size + scales_segment_size;
    float* bias_values = reinterpret_cast<float*>(storage.data() + bias_offset);
    if (!float_bias.empty())
        std::copy(float_bias.begin(), float_bias.end(), bias_values);
    else
        for (size_t index = 0; index < bfloat16_bias.size(); ++index)
            bias_values[index] = bfloat16_to_float(bfloat16_bias[index]);
    const size_t bias_size = (float_bias.size() + bfloat16_bias.size()) * sizeof(float);
    std::memset(storage.data() + bias_offset + bias_size, 0, storage.size() - bias_offset - bias_size);
}

class Mxfp4Linear_vulkan
{
public:
    ~Mxfp4Linear_vulkan();

    [[nodiscard]] static std::shared_ptr<Mxfp4Linear_vulkan> create(const TensorData& matrix, const TensorData* bias,
                                                                    uint32_t vulkan_device_index,
                                                                    const VulkanRuntimePtr& vulkan_runtime,
                                                                    uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;

private:
    friend class Mxfp4Expert_vulkan;
    friend class VulkanExpertBackend;

    [[nodiscard]] static std::shared_ptr<Mxfp4Linear_vulkan> create_with_allocator(const TensorData& matrix,
                                                                                   const TensorData* bias,
                                                                                   uint32_t vulkan_device_index,
                                                                                   ncnn::VkAllocator* weight_allocator,
                                                                                   const VulkanRuntimePtr& vulkan_runtime,
                                                                                   uint64_t optimization_flags,
                                                                                   VkWeightBatch* upload_batch = nullptr);
    class Implementation;

    Mxfp4Linear_vulkan();
    std::unique_ptr<Implementation> d;
};

struct Mxfp4DeviceMatrixView_vulkan
{
    uint32_t output_columns = 0;
    uint32_t input_columns = 0;
    uint64_t packed_size = 0;
    uint64_t scales_size = 0;
    size_t packed_offset = 0;
    size_t scales_offset = 0;
};

class Mxfp4Expert_vulkan
{
public:
    ~Mxfp4Expert_vulkan();

    [[nodiscard]] static std::shared_ptr<Mxfp4Expert_vulkan> create(const TensorData& gate_up,
                                                                    const TensorData* gate_up_bias,
                                                                    const TensorData& down,
                                                                    const TensorData* down_bias,
                                                                    float activation_limit,
                                                                    uint32_t vulkan_device_index,
                                                                    ExpertActivation activation,
                                                                    const VulkanRuntimePtr& vulkan_runtime,
                                                                    uint64_t optimization_flags);
    [[nodiscard]] static std::shared_ptr<Mxfp4Expert_vulkan> create_from_device_storage(const Mxfp4DeviceMatrixView_vulkan& gate_up,
                                                                                        const TensorData* gate_up_bias,
                                                                                        const Mxfp4DeviceMatrixView_vulkan& down,
                                                                                        const TensorData* down_bias,
                                                                                        float activation_limit,
                                                                                        uint32_t vulkan_device_index,
                                                                                        const ncnn::VkMat& storage,
                                                                                        ExpertActivation activation,
                                                                                        const VulkanRuntimePtr& vulkan_runtime,
                                                                                        uint64_t optimization_flags);
    [[nodiscard]] bool forward(const ActivationBuffer& input, ActivationBuffer& output) const;

private:
    friend class VulkanExpertBackend;

    // Caller owns scratch vectors and retains bound VkMats until submit.
    void record(const ncnn::VkMat& input, ncnn::VkMat& intermediate, ncnn::VkMat& output,
                ncnn::VkCompute& cmd, std::vector<ncnn::VkMat>& bindings,
                std::vector<ncnn::vk_constant_type>& constants) const;

    [[nodiscard]] static std::shared_ptr<Mxfp4Expert_vulkan> create_with_allocator(const TensorData& gate_up,
                                                                                   const TensorData* gate_up_bias,
                                                                                   const TensorData& down,
                                                                                   const TensorData* down_bias,
                                                                                   float activation_limit,
                                                                                   uint32_t vulkan_device_index,
                                                                                   ncnn::VkAllocator* weight_allocator,
                                                                                   ExpertActivation activation,
                                                                                   const VulkanRuntimePtr& vulkan_runtime,
                                                                                   uint64_t optimization_flags,
                                                                                   VkWeightBatch* upload_batch = nullptr);

    class Implementation;

    static bool init_pipelines(Implementation& impl, uint64_t optimization_flags, bool hold_lock);

    Mxfp4Expert_vulkan();
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LINEAR_MXFP4_H
