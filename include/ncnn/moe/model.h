#ifndef NCNN_MOE_MODEL_H
#define NCNN_MOE_MODEL_H

#include "ncnn/moe/execution_plan.h"

#include <memory>

namespace ncnn {
namespace moe {

class SessionBatchAccess;

class Model
{
private:
    explicit Model(std::shared_ptr<const CompiledModel> compiled)
        : compiled_(std::move(compiled))
    {
    }

    std::shared_ptr<const CompiledModel> compiled_;

    friend class Runtime;
    friend class Session;
    friend class SessionBatchAccess;

public:
    [[nodiscard]] const MoeIR& ir() const noexcept
    {
        return compiled_->descriptor;
    }
    [[nodiscard]] const MoeModelDescriptor& descriptor() const noexcept
    {
        return compiled_->descriptor;
    }
    [[nodiscard]] HybridMode hybrid_mode() const noexcept
    {
        return compiled_->hybrid_mode;
    }
    [[nodiscard]] uint32_t vulkan_device_index() const noexcept
    {
        return compiled_->vulkan_device_index;
    }
    [[nodiscard]] const std::vector<uint32_t>& vulkan_device_indices() const noexcept
    {
        return compiled_->vulkan_device_indices;
    }
    [[nodiscard]] const std::vector<CompiledLayerPlan>& execution_plan() const noexcept
    {
        return compiled_->layers;
    }
    [[nodiscard]] const ExecutionGraph& execution_graph() const noexcept
    {
        return compiled_->graph;
    }
    [[nodiscard]] const ExecutionSchedule& execution_schedule() const noexcept
    {
        return compiled_->schedule;
    }
    [[nodiscard]] const ModelMemoryPlan& memory_plan() const noexcept
    {
        return compiled_->memory_plan;
    }
    [[nodiscard]] const EffectiveRuntimeOptions& effective_runtime_options() const noexcept
    {
        return compiled_->effective_runtime_options;
    }
    [[nodiscard]] const ExpertStore& expert_store() const noexcept
    {
        return *compiled_->expert_store;
    }
};

using ModelPtr = std::shared_ptr<Model>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODEL_H
