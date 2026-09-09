#ifndef NCNN_MOE_MODELPIPELINE_H
#define NCNN_MOE_MODELPIPELINE_H

#include "ncnn/moe/result.h"

#include <cstdint>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
class WeightStore;
struct MoeBlockPlan;
struct CompiledModel;
struct AttentionBlockPlan;
struct CompilerOption;

Result<void> prepare_model_pipeline(
    CompiledModel& compiled,
    const CompilerOption& opt);

void release_vulkan_dense_host_copies(CompiledModel& compiled);

[[nodiscard]] bool support_vulkan_experts(
    const WeightStore& weights,
    const MoeBlockPlan& moe,
    uint64_t optimization_flags) noexcept;

[[nodiscard]] bool support_vulkan_shared_experts(
    const CompiledOperatorTable& operators,
    const MoeBlockPlan& moe) noexcept;

[[nodiscard]] bool support_vulkan_attention(
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& attention) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELPIPELINE_H
