#ifndef NCNN_MOE_NCNN_LINEAR_H
#define NCNN_MOE_NCNN_LINEAR_H

#include "cpu_batch.h"

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
    [[nodiscard]] static uint64_t current_thread_vulkan_dispatches() noexcept;
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
