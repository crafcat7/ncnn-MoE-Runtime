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
    explicit Model(std::shared_ptr<const CompiledModel> _compiled)
        : compiled(std::move(_compiled))
    {
    }

    std::shared_ptr<const CompiledModel> compiled;

    friend class Runtime;
    friend class Session;
    friend class SessionBatchAccess;

public:
    [[nodiscard]] const MoeIR& ir() const noexcept
    {
        return compiled->descriptor;
    }
    [[nodiscard]] const MoeModelDescriptor& descriptor() const noexcept
    {
        return compiled->descriptor;
    }
    [[nodiscard]] HybridMode hybrid_mode() const noexcept
    {
        return compiled->hybrid_mode;
    }
    [[nodiscard]] uint32_t vulkan_device_index() const noexcept
    {
        return compiled->vulkan_device_index;
    }
    [[nodiscard]] const std::vector<uint32_t>& vulkan_device_indices() const noexcept
    {
        return compiled->vulkan_device_indices;
    }
    [[nodiscard]] const std::vector<CompiledLayerPlan>& execution_plan() const noexcept
    {
        return compiled->graph.layer_plans;
    }
    [[nodiscard]] const ExecutionGraph& execution_graph() const noexcept
    {
        return compiled->graph;
    }
    [[nodiscard]] const ExecutionSchedule& execution_schedule() const noexcept
    {
        return compiled->schedule;
    }
    [[nodiscard]] const ExecutionGraph& speculative_execution_graph() const noexcept
    {
        return compiled->speculative.graph;
    }
    [[nodiscard]] const ExecutionSchedule& speculative_execution_schedule() const noexcept
    {
        return compiled->speculative.schedule;
    }
    [[nodiscard]] const ModelMemoryPlan& memory_plan() const noexcept
    {
        return compiled->memory_plan;
    }
    [[nodiscard]] const EffectiveOption& effective_option() const noexcept
    {
        return compiled->effective_option;
    }

    [[nodiscard]] const ExpertStore& expert_store() const noexcept
    {
        return *compiled->expert_store;
    }
};

using ModelPtr = std::shared_ptr<Model>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODEL_H
