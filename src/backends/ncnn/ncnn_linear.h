#ifndef NCNN_MOE_NCNN_LINEAR_H
#define NCNN_MOE_NCNN_LINEAR_H

#include "kernels/cpu_batch.h"

#include "ncnn/moe/types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace ncnn {
class VkAllocator;
class VkMat;

namespace moe {

enum class NcnnLinearDevice
{
    Cpu,
    Vulkan
};

struct NcnnVulkanRuntimeCounters
{
    uint64_t compute_submissions = 0;
    uint64_t asynchronous_submissions = 0;
    uint64_t batch_uploads = 0;
    uint64_t batch_downloads = 0;
    uint64_t auxiliary_uploads = 0;
    uint64_t auxiliary_upload_bytes = 0;
    uint64_t staging_slot_resizes = 0;
    uint64_t staging_slot_reuses = 0;
    uint64_t staging_slot_acquisitions = 0;
    uint64_t staging_slot_contentions = 0;
    uint64_t command_buffer_reuses = 0;
    uint64_t attention_qkv_rope_fusions = 0;
    uint64_t attention_qkv_ring_fusions = 0;
    uint64_t attention_decode_sdpa_fusions = 0;
    uint64_t kv_ring_appends = 0;
    uint64_t kv_ring_resizes = 0;
    uint64_t kv_ring_wrapped_views = 0;
};

class NcnnLinearOperator
{
private:
    friend class NcnnVulkanAttentionOperator;

    class Implementation;

    NcnnLinearOperator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnLinearOperator();

    [[nodiscard]] static std::shared_ptr<NcnnLinearOperator> create(const TensorData& matrix, const TensorData* bias,
                                                                    NcnnLinearDevice device = NcnnLinearDevice::Cpu,
                                                                    uint32_t vulkan_device_index = automatic_vulkan_device_index);
    [[nodiscard]] static std::shared_ptr<NcnnLinearOperator> create_fused(const std::vector<const TensorData*>& matrices,
                                                                          const std::vector<const TensorData*>& biases, NcnnLinearDevice device,
                                                                          uint32_t vulkan_device_index = automatic_vulkan_device_index);
    [[nodiscard]] static uint32_t vulkan_device_count() noexcept;
    [[nodiscard]] static uint64_t vulkan_heap_budget_bytes() noexcept;
    [[nodiscard]] static std::vector<VulkanDeviceCapabilities> vulkan_device_capabilities();
    [[nodiscard]] static uint64_t current_thread_vulkan_dispatches() noexcept;
    [[nodiscard]] static NcnnVulkanRuntimeCounters current_thread_vulkan_runtime_counters() noexcept;
    [[nodiscard]] bool forward(const CpuBatch& input, CpuBatch& output) const;
    [[nodiscard]] bool uses_vulkan() const noexcept;
};

class NcnnVulkanMxfp4Operator
{
private:
    friend class NcnnVulkanMxfp4ExpertOperator;
    friend class VulkanMxfp4ExpertBackend;

    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4Operator> create_with_allocator(const TensorData& matrix, const TensorData* bias,
                                                                                        uint32_t vulkan_device_index, ncnn::VkAllocator* weight_allocator);
    class Implementation;

    NcnnVulkanMxfp4Operator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnVulkanMxfp4Operator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4Operator> create(const TensorData& matrix, const TensorData* bias = nullptr,
                                                                         uint32_t vulkan_device_index = automatic_vulkan_device_index);
    [[nodiscard]] bool forward(const CpuBatch& input, CpuBatch& output) const;
    [[nodiscard]] uint32_t input_columns() const noexcept;
    [[nodiscard]] uint32_t output_columns() const noexcept;
};

struct NcnnVulkanMxfp4DeviceMatrixView
{
    uint32_t output_columns = 0;
    uint32_t input_columns = 0;
    uint64_t packed_bytes = 0;
    uint64_t scales_bytes = 0;
    size_t packed_offset = 0;
    size_t scales_offset = 0;
};

class NcnnVulkanMxfp4ExpertOperator
{
private:
    friend class VulkanMxfp4ExpertBackend;

    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> create_with_allocator(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                                              const TensorData& down, const TensorData* down_bias,
                                                                                              float activation_limit, uint32_t vulkan_device_index,
                                                                                              ncnn::VkAllocator* weight_allocator);

    class Implementation;

    NcnnVulkanMxfp4ExpertOperator();
    std::unique_ptr<Implementation> implementation_;

public:
    ~NcnnVulkanMxfp4ExpertOperator();

    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> create(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                               const TensorData& down, const TensorData* down_bias, float activation_limit,
                                                                               uint32_t vulkan_device_index = automatic_vulkan_device_index);
    [[nodiscard]] static std::shared_ptr<NcnnVulkanMxfp4ExpertOperator> create_from_device_storage(
        const NcnnVulkanMxfp4DeviceMatrixView& gate_up, const TensorData* gate_up_bias, const NcnnVulkanMxfp4DeviceMatrixView& down,
        const TensorData* down_bias, float activation_limit, uint32_t vulkan_device_index, const ncnn::VkMat& storage);
    [[nodiscard]] bool forward(const CpuBatch& input, CpuBatch& output) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_LINEAR_H
