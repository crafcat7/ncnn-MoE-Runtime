#ifndef NCNN_MOE_NCNN_LINEAR_H
#define NCNN_MOE_NCNN_LINEAR_H

#include "kernels/cpu_batch.h"

#include "ncnn/moe/types.h"

#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

enum class NcnnLinearDevice
{
    Cpu,
    Vulkan
};

struct NcnnVulkanRuntimeCounters
{
    uint64_t compute_submissions = 0;
    uint64_t batch_uploads = 0;
    uint64_t batch_downloads = 0;
    uint64_t auxiliary_uploads = 0;
    uint64_t auxiliary_upload_bytes = 0;
    uint64_t staging_slot_resizes = 0;
    uint64_t staging_slot_reuses = 0;
    uint64_t staging_slot_acquisitions = 0;
    uint64_t staging_slot_contentions = 0;
};

class NcnnLinearOperator
{
public:
    ~NcnnLinearOperator();

    [[nodiscard]] static std::shared_ptr<NcnnLinearOperator> create(
        const TensorData& matrix,
        const TensorData* bias,
        NcnnLinearDevice device = NcnnLinearDevice::Cpu);
    [[nodiscard]] static std::shared_ptr<NcnnLinearOperator> create_fused(
        const std::vector<const TensorData*>& matrices,
        const std::vector<const TensorData*>& biases,
        NcnnLinearDevice device);
    [[nodiscard]] static uint32_t vulkan_device_count() noexcept;
    [[nodiscard]] static uint64_t vulkan_heap_budget_bytes() noexcept;
    [[nodiscard]] static uint64_t current_thread_vulkan_dispatches() noexcept;
    [[nodiscard]] static NcnnVulkanRuntimeCounters current_thread_vulkan_runtime_counters() noexcept;
    [[nodiscard]] bool forward(const CpuBatch& input, CpuBatch& output) const;
    [[nodiscard]] bool uses_vulkan() const noexcept;

private:
    friend class NcnnVulkanAttentionOperator;

    class Implementation;

    NcnnLinearOperator();
    std::unique_ptr<Implementation> implementation_;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NCNN_LINEAR_H
