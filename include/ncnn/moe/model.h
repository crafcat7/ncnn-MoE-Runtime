#ifndef NCNN_MOE_MODEL_H
#define NCNN_MOE_MODEL_H

#include "ncnn/moe/execution_plan.h"

#include <memory>

namespace ncnn {
namespace moe {

class Model
{
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

private:
    explicit Model(std::shared_ptr<const CompiledModel> compiled) : compiled_(std::move(compiled))
    {
    }

    std::shared_ptr<const CompiledModel> compiled_;

    friend class Runtime;
    friend class Session;
};

using ModelPtr = std::shared_ptr<Model>;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODEL_H
