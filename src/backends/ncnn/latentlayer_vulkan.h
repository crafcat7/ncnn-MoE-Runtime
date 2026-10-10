#ifndef NCNN_MOE_LATENTLAYER_VULKAN_H
#define NCNN_MOE_LATENTLAYER_VULKAN_H

#include "kernels/activationbuffer.h"
#include "graph/router.h"

#include <memory>
#include <span>

namespace ncnn {
namespace moe {
class LatentAttention_vulkan;
class HyperConnection_vulkan;
class RmsNorm_vulkan;
class Router_vulkan;
class DeviceTensor_vulkan;
class LayerHead_vulkan;
class MoeCombine_vulkan;
struct MoeCombineInput_vulkan;
struct LayerCache;

class LatentLayerWorkspace_vulkan
{
public:
    LatentLayerWorkspace_vulkan();
    ~LatentLayerWorkspace_vulkan();
    LatentLayerWorkspace_vulkan(const LatentLayerWorkspace_vulkan&) = delete;
    LatentLayerWorkspace_vulkan& operator=(const LatentLayerWorkspace_vulkan&) = delete;
    // End an active segment without discarding reusable Host publication storage.
    void reset();
    // Explicit session/budget boundary: also discard retained Host capacities.
    void release();
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;
    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] const std::shared_ptr<const DeviceTensor_vulkan>& device_input() const noexcept;

private:
    friend class LatentLayer_vulkan;
    class Implementation;
    std::unique_ptr<Implementation> d;
};

// A continuous GPU segment ends only at the host ExpertStore scheduling boundary.
class LatentLayer_vulkan
{
public:
    [[nodiscard]] static std::shared_ptr<LatentLayer_vulkan> create(std::shared_ptr<LatentAttention_vulkan> attention,
                                                                    std::shared_ptr<HyperConnection_vulkan> attention_hyper,
                                                                    std::shared_ptr<HyperConnection_vulkan> ffn_hyper,
                                                                    std::shared_ptr<RmsNorm_vulkan> ffn_norm,
                                                                    std::shared_ptr<Router_vulkan> router);
    [[nodiscard]] bool forward(const ActivationBuffer& initial_hidden,
                               std::span<const uint64_t> positions,
                               std::span<LayerCache* const> caches,
                               const ExpertDispatchOptions& options,
                               ActivationBuffer& normalized,
                               ExpertDispatchPlan& routes,
                               LatentLayerWorkspace_vulkan& workspace) const;
    [[nodiscard]] static bool combine_device(std::span<const MoeCombineInput_vulkan> inputs,
                                             size_t rows, uint32_t columns,
                                             const ActivationBuffer* shared_host,
                                             std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                                             std::shared_ptr<DeviceTensor_vulkan>& output,
                                             const LatentLayerWorkspace_vulkan& workspace);
    // Capture the canonical Expert/Shared sum for the next layer or final Head.
    // No command is recorded and no device work is submitted here.
    [[nodiscard]] static bool defer_combine(std::span<const MoeCombineInput_vulkan> inputs,
                                            size_t rows, uint32_t columns,
                                            const ActivationBuffer* shared_host,
                                            std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                                            LatentLayerWorkspace_vulkan& workspace);
    [[nodiscard]] static bool defer_combine(ActivationBuffer& combined,
                                            LatentLayerWorkspace_vulkan& workspace);
    [[nodiscard]] static bool defer_combine(std::shared_ptr<const DeviceTensor_vulkan> combined,
                                            LatentLayerWorkspace_vulkan& workspace);
    // Used only at a CPU consumer or a capability boundary.
    [[nodiscard]] static bool materialize_hidden(LatentLayerWorkspace_vulkan& workspace,
                                                 ActivationBuffer& hidden);
    [[nodiscard]] static bool can_finish(const LatentLayerWorkspace_vulkan& workspace,
                                         const LayerHead_vulkan& head) noexcept;
    // Complete the last FFN post, HC head, norm and optional LM head in one submission.
    [[nodiscard]] static bool finish(LatentLayerWorkspace_vulkan& workspace,
                                     const LayerHead_vulkan& head,
                                     bool produce_logits,
                                     bool last_row_only,
                                     ActivationBuffer& hidden,
                                     ActivationBuffer& logits);

private:
    std::shared_ptr<LatentAttention_vulkan> attention;
    std::shared_ptr<HyperConnection_vulkan> attention_hyper;
    std::shared_ptr<HyperConnection_vulkan> ffn_hyper;
    std::shared_ptr<RmsNorm_vulkan> ffn_norm;
    std::shared_ptr<Router_vulkan> router;
    std::shared_ptr<MoeCombine_vulkan> combiner;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LATENTLAYER_VULKAN_H
