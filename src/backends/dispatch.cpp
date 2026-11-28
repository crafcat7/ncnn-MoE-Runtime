#include "backends/dispatch.h"

#include "backends/ncnn/linear/linear.h"
#include "graph/compiledoperator.h"
#include "graph/graph.h"
#include "kernels/activationbuffer.h"

#include <stdexcept>

namespace ncnn {
namespace moe {

bool has_vulkan_linear_operator(const CompiledOperator* op) noexcept
{
    return op != nullptr && (op->linear || op->bfloat16 || op->float8 || op->qnk);
}

bool has_vulkan_float8_or_linear_operator(const CompiledOperator* op) noexcept
{
    return op != nullptr && (op->float8 || op->linear);
}

bool dispatch_vulkan_linear(const CompiledOperator* op,
                            const ActivationBuffer& input,
                            ActivationBuffer& output,
                            ExecutionBackend backend)
{
    if (backend != ExecutionBackend::Vulkan)
        return false;
    if (!op)
        throw std::runtime_error("Vulkan linear was selected without a compiled operator");
    bool executed = false;
    if (op->bfloat16)
        executed = op->bfloat16->forward(input, output);
    else if (op->float8)
        executed = op->float8->forward(input, output);
    else if (op->qnk)
        executed = op->qnk->forward(input, output);
    else if (op->linear)
        executed = op->linear->forward(input, output);
    else
        throw std::runtime_error("Vulkan linear was selected without a device operator");

    if (!executed)
        throw std::runtime_error("Vulkan linear execution failed");
    return true;
}

} // namespace moe
} // namespace ncnn
