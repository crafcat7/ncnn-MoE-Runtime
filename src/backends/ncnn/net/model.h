#ifndef NCNN_MOE_NET_MODEL_H
#define NCNN_MOE_NET_MODEL_H

#include "backends/ncnn/expert/shared_ffn.h"
#include "backends/ncnn/net/memory.h"
#include "backends/ncnn/net/segment.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
class WeightStore;
struct AttentionBlockPlan;
struct CompiledModel;
struct CompilerOption;
struct MoeBlockPlan;

Result<void> prepare_model_pipeline(CompiledModel& compiled,
                                    const CompilerOption& opt);

void release_host_copies(CompiledModel& compiled);

[[nodiscard]] bool support_vulkan_experts(const WeightStore& weights,
                                          const MoeBlockPlan& moe,
                                          uint64_t optimization_flags) noexcept;

[[nodiscard]] bool support_vulkan_attention(const CompiledOperatorTable& operators,
                                            const AttentionBlockPlan& attention) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_NET_MODEL_H
