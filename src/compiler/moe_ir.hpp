#ifndef NCNN_MOE_COMPILER_MOE_IR_HPP
#define NCNN_MOE_COMPILER_MOE_IR_HPP

#include "ncnn/moe/execution_plan.h"
#include "ncnn/moe/moe_ir.h"
#include "ncnn/moe/result.h"

namespace ncnn {
namespace moe {

[[nodiscard]] Result<void> normalize_moe_ir(MoeIR& ir);
[[nodiscard]] Result<void> build_compiled_execution_graph(CompiledModel& compiled, const ModelCompiler::BackendCapabilities& caps);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_COMPILER_MOE_IR_HPP
