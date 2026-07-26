#ifndef NCNN_MOE_MEMORY_PLANNER_H
#define NCNN_MOE_MEMORY_PLANNER_H

#include "ncnn/moe/memory_plan.h"
#include "ncnn/moe/moe_ir.h"
#include "ncnn/moe/result.h"

#include <cstdint>

namespace ncnn {
namespace moe {

struct RuntimeOptions;

[[nodiscard]] Result<ModelMemoryPlan> plan_model_memory(const MoeIR& ir, const RuntimeOptions& options, uint64_t physical_memory_bytes);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MEMORY_PLANNER_H
