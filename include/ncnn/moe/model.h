#ifndef NCNN_MOE_MODEL_H
#define NCNN_MOE_MODEL_H

#include "ncnn/moe/execution_plan.h"

#include <memory>

namespace ncnn {
namespace moe {

class Model
{
public:
    [[nodiscard]] const MoeModelDescriptor& descriptor() const noexcept
    {
        return compiled_->descriptor;
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
