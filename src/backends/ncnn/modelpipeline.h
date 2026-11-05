#ifndef NCNN_MOE_MODELPIPELINE_H
#define NCNN_MOE_MODELPIPELINE_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"
#include "graph/memoryplan.h"

#include <cstdint>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
class WeightStore;
struct MoeBlockPlan;
struct CompiledModel;
struct AttentionBlockPlan;
struct CompilerOption;
struct ExpertPlan;
struct CompiledLayerPlan;

Result<void> prepare_model_pipeline(CompiledModel& compiled,
                                    const CompilerOption& opt);

// Recording work is bounded independently of persistent weight residency.
[[nodiscard]] uint32_t resident_chunk_layers(const CompiledModel& model, uint32_t rows) noexcept;

// Measure the already scheduled placement for representative prefill and
// decode execution. Persistent KV/recurrent state is reported separately from
// reusable graph buffers. Zero context_tokens uses prefill_rows.
[[nodiscard]] Result<ModelGraphMemoryPlan> plan_model_graph_memory(const CompiledModel& compiled,
                                                                   uint32_t prefill_rows,
                                                                   uint32_t sequence_count = 1,
                                                                   uint32_t context_tokens = 0);

void release_host_copies(CompiledModel& compiled);

[[nodiscard]] bool support_vulkan_experts(const WeightStore& weights,
                                          const MoeBlockPlan& moe,
                                          uint64_t optimization_flags) noexcept;

[[nodiscard]] bool support_vulkan_shared_experts(const CompiledOperatorTable& operators,
                                                 const MoeBlockPlan& moe) noexcept;

[[nodiscard]] bool support_vulkan_ffn(const CompiledOperatorTable& operators,
                                      const ExpertPlan& ffn) noexcept;

[[nodiscard]] bool support_vulkan_dense_ffn(const CompiledOperatorTable& operators,
                                            const ExpertPlan& ffn) noexcept;

[[nodiscard]] uint64_t attn_vk_size(const CompiledModel& compiled,
                                    const CompiledLayerPlan& layer,
                                    uint64_t alignment,
                                    uint32_t device_index = automatic_vulkan_device_index) noexcept;

[[nodiscard]] uint64_t dense_weight_size(const CompiledModel& compiled,
                                         const TensorData& matrix,
                                         uint64_t alignment,
                                         uint32_t device_index = automatic_vulkan_device_index,
                                         bool has_bias = false) noexcept;

[[nodiscard]] uint64_t dense_pending_size(const CompiledModel& compiled,
                                          uint32_t device_index) noexcept;

// The common resource plan's unallocated per-Session state and peak execution
// workspace for this device. Shape estimates use the default prefill chunk;
// larger chunks/contexts remain checked when their buffers are allocated.
[[nodiscard]] uint64_t vulkan_pending_reserve(const CompiledModel& compiled,
                                              uint32_t device_index) noexcept;

[[nodiscard]] bool support_vulkan_attention(const CompiledOperatorTable& operators,
                                            const AttentionBlockPlan& attention) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELPIPELINE_H
