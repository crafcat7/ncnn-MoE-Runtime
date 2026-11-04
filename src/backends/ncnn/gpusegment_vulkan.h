#ifndef NCNN_MOE_GPUSEGMENT_VULKAN_H
#define NCNN_MOE_GPUSEGMENT_VULKAN_H

#include "vulkan.h"
#include "residentffn_vulkan.h"
#include "graph/router.h"
#include "kernels/activationbuffer.h"

#include <memory>
#include <span>
#include <string>

namespace ncnn {
namespace moe {

class DeviceTensor_vulkan;
class VulkanContext;
struct CompiledModel;
struct CompiledLayerPlan;
struct LayerCache;
class GpuSegmentContinuation_vulkan;

class GpuSegmentWorkspace_vulkan
{
public:
    GpuSegmentWorkspace_vulkan();
    ~GpuSegmentWorkspace_vulkan();
    GpuSegmentWorkspace_vulkan(const GpuSegmentWorkspace_vulkan&) = delete;
    GpuSegmentWorkspace_vulkan& operator=(const GpuSegmentWorkspace_vulkan&) = delete;
    void reset();
    void reserve_host_output(uint64_t slot_capacity) noexcept;
    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] uint64_t allocated_bytes() const noexcept;
    [[nodiscard]] const std::shared_ptr<const DeviceTensor_vulkan>& device_input() const noexcept;

private:
    friend class GpuSegment_vulkan;
    friend class GpuSegmentContinuation_vulkan;
    class Implementation;
    std::unique_ptr<Implementation> d;
};

// A model-neutral device segment can continue through resident dense or routed
// FFNs, or end at a host ExpertStore boundary with validated compact routes.
class GpuSegment_vulkan
{
public:
    ~GpuSegment_vulkan();
    [[nodiscard]] static bool supports(const CompiledModel& model, const CompiledLayerPlan& layer) noexcept;
    [[nodiscard]] static bool supports_head(const CompiledModel& model) noexcept;
    [[nodiscard]] static std::shared_ptr<GpuSegment_vulkan> create(const CompiledModel& model, const CompiledLayerPlan& layer);
    [[nodiscard]] static std::shared_ptr<GpuSegment_vulkan> create_head(const CompiledModel& model);
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept;
    // Shared by execution and lifetime planning. Returns a target and writes
    // its Router/LM-head fence. A legal non-applicable boundary returns
    // invalid_execution_node_id; a matched target without a valid fence or
    // device range is an error.
    [[nodiscard]] static Result<uint32_t> continuation_target(const CompiledModel& model, uint32_t expert_node_id,
                                                              uint32_t& fence_id);

    // Execute an admitted resident chain. Cache pointers are layer-major,
    // with one entry per input row. Routed layers require layer-major options;
    // an all-dense chain also accepts an empty options span.
    // The owner submits bounded recording chunks and validates each fence
    // before publishing cache state. Hidden remains on the device between chunks.
    [[nodiscard]] static bool forward_resident(const CompiledModel& model, const ActivationBuffer& initial_hidden,
                                               std::span<const uint64_t> positions, std::span<LayerCache* const> caches,
                                               std::span<const ExpertDispatchOptions> options,
                                               bool produce_logits, bool last_row_only,
                                               ActivationBuffer& logits, std::string& error,
                                               GpuSegmentWorkspace_vulkan* workspace = nullptr,
                                               uint32_t layer_limit = 0);

    // Prepare a private transaction before entering the Expert command lock.
    // The Region records it only in the wave that completes its aggregate.
    [[nodiscard]] std::shared_ptr<GpuSegmentContinuation_vulkan> prepare_continuation(std::span<const uint64_t> positions,
                                                                                      std::span<LayerCache* const> caches,
                                                                                      const ExpertDispatchOptions& options,
                                                                                      const GpuSegmentWorkspace_vulkan& workspace,
                                                                                      std::string& error) const;
    [[nodiscard]] std::shared_ptr<GpuSegmentContinuation_vulkan> prepare_head_continuation(bool produce_logits, bool last_row_only,
                                                                                           const GpuSegmentWorkspace_vulkan& workspace,
                                                                                           std::string& error) const;

    // Prefill repeats one cache with consecutive positions; staged decode
    // supplies one independent cache per row. Both use one segment submit.
    [[nodiscard]] bool forward(const ActivationBuffer& initial_hidden,
                               std::span<const uint64_t> positions,
                               std::span<LayerCache* const> caches,
                               const ExpertDispatchOptions& options,
                               ExpertDispatchPlan& routes,
                               GpuSegmentWorkspace_vulkan& workspace,
                               std::string& error) const;
    // No recording/submission: the next segment records routed/shared FFN
    // output plus residual before consuming hidden on the same device.
    [[nodiscard]] static bool defer_combine(std::shared_ptr<const DeviceTensor_vulkan> combined,
                                            GpuSegmentWorkspace_vulkan& workspace);
    [[nodiscard]] static bool materialize_hidden(GpuSegmentWorkspace_vulkan& workspace,
                                                 ActivationBuffer& hidden, std::string& error);
    [[nodiscard]] bool finish(bool produce_logits, bool last_row_only,
                              ActivationBuffer& hidden, ActivationBuffer& logits,
                              GpuSegmentWorkspace_vulkan& workspace, std::string& error) const;

private:
    friend class GpuSegmentContinuation_vulkan;
    GpuSegment_vulkan();
    class Implementation;
    std::shared_ptr<Implementation> d;
};

// A pending operand is never represented as a completed DeviceTensor. The
// Expert wave owns submit/wait; the executor publishes after Expert commit.
class GpuSegmentContinuation_vulkan final : public ResidentFfnContinuation_vulkan
{
public:
    ~GpuSegmentContinuation_vulkan() override;
    [[nodiscard]] const std::shared_ptr<VulkanContext>& context() const noexcept override;
#if NCNN_MOE_WITH_VULKAN
    [[nodiscard]] bool record(const ncnn::VkMat& combined, ncnn::VkCompute& command, const ncnn::Option& option) override;
#endif
    void mark_submit_attempted() noexcept override;
    void mark_completed() noexcept override;
    void abort() noexcept override;
    [[nodiscard]] bool publish(ExpertDispatchPlan& routes, GpuSegmentWorkspace_vulkan& workspace, std::string& error);
    [[nodiscard]] bool publish_head(ActivationBuffer& logits, GpuSegmentWorkspace_vulkan& workspace, std::string& error);

private:
    friend class GpuSegment_vulkan;
    GpuSegmentContinuation_vulkan();
    class Implementation;
    std::unique_ptr<Implementation> d;
};

} // namespace moe
} // namespace ncnn

#endif
