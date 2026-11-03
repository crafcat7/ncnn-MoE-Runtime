#ifndef NCNN_MOE_GATEDRESIDUALLAYER_VULKAN_H
#define NCNN_MOE_GATEDRESIDUALLAYER_VULKAN_H
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
    friend class GatedResidualLayer_vulkan;
    class Implementation;
    std::unique_ptr<Implementation> d;
};
class GatedResidualWorkspace_vulkan
{
public:
    GatedResidualWorkspace_vulkan();
    ~GatedResidualWorkspace_vulkan();
    void reset();
    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] const std::shared_ptr<const DeviceTensor_vulkan>& device_input() const noexcept;

private:
    friend class GatedResidualLayer_vulkan;
    class Implementation;
    std::unique_ptr<Implementation> d;
};
class GatedResidualLayer_vulkan
{
public:
    ~GatedResidualLayer_vulkan();
    [[nodiscard]] static std::shared_ptr<GatedResidualLayer_vulkan> create(const CompiledModel& model,
                                                                           const CompiledLayerPlan& layer);
    [[nodiscard]] static std::shared_ptr<GatedResidualLayer_vulkan> create_head(const CompiledModel& model);
    [[nodiscard]] bool forward(const ActivationBuffer& initial_hidden, std::span<const int32_t> tokens,
                               uint64_t position, LayerCache& cache, const ExpertDispatchOptions& options,
                               ExpertDispatchPlan& routes, GatedResidualWorkspace_vulkan& workspace, std::string& error) const;
    [[nodiscard]] static bool defer_combine(std::span<const MoeCombineInput_vulkan> inputs,
                                            size_t rows, uint32_t columns, std::shared_ptr<const DeviceTensor_vulkan> shared,
                                            GatedResidualWorkspace_vulkan& workspace);
    [[nodiscard]] static bool finish(const CompiledModel& model, bool produce_logits, bool last_only,
                                     ActivationBuffer& logits, GatedResidualWorkspace_vulkan& workspace);

private:
    class Implementation;
    GatedResidualLayer_vulkan();
    std::unique_ptr<Implementation> d;
};
} // namespace moe
} // namespace ncnn
#endif
