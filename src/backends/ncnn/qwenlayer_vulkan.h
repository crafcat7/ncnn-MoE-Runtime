#ifndef NCNN_MOE_QWENLAYER_VULKAN_H
#define NCNN_MOE_QWENLAYER_VULKAN_H
#include "kernels/activationbuffer.h"
#include "graph/router.h"
#include "vulkan.h"
#include <memory>
#include <span>
#include <string>
namespace ncnn {
namespace moe {
struct CompiledModel;
struct CompiledLayerPlan;
struct LayerCache;
class DeviceTensor_vulkan;
struct MoeCombineInput_vulkan;
class PleState_vulkan
{
public:
    PleState_vulkan();
    ~PleState_vulkan();
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;

private:
    friend class QwenLayer_vulkan;
    class Implementation;
    std::unique_ptr<Implementation> d;
};
class QwenLayerWorkspace_vulkan
{
public:
    QwenLayerWorkspace_vulkan();
    ~QwenLayerWorkspace_vulkan();
    void reset();
    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;
    [[nodiscard]] const std::shared_ptr<const DeviceTensor_vulkan>& device_input() const noexcept;

private:
    friend class QwenLayer_vulkan;
    class Implementation;
    std::unique_ptr<Implementation> d;
};
class QwenLayer_vulkan
{
public:
    ~QwenLayer_vulkan();
    [[nodiscard]] static std::shared_ptr<QwenLayer_vulkan> create(const CompiledModel& model,
                                                                  const CompiledLayerPlan& layer);
    [[nodiscard]] static std::shared_ptr<QwenLayer_vulkan> create_head(const CompiledModel& model);
    [[nodiscard]] uint64_t device_bytes() const noexcept;
    [[nodiscard]] bool forward(const ActivationBuffer& initial_hidden, std::span<const int32_t> tokens,
                               uint64_t position, LayerCache& cache, const ExpertDispatchOptions& options,
                               ExpertDispatchPlan& routes, QwenLayerWorkspace_vulkan& workspace, std::string& error) const;
    [[nodiscard]] static bool defer_combine(std::span<const MoeCombineInput_vulkan> inputs,
                                            size_t rows, uint32_t columns, std::shared_ptr<const DeviceTensor_vulkan> shared,
                                            QwenLayerWorkspace_vulkan& workspace);
    [[nodiscard]] static bool finish(const CompiledModel& model, bool produce_logits, bool last_only,
                                     ActivationBuffer& logits, QwenLayerWorkspace_vulkan& workspace);

private:
    class Implementation;
    QwenLayer_vulkan();
    std::unique_ptr<Implementation> d;
};
} // namespace moe
} // namespace ncnn
#endif
