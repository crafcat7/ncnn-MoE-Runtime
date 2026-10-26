#ifndef NCNN_MOE_MODELPIPELINE_H
#define NCNN_MOE_MODELPIPELINE_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

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

void release_vulkan_dense_host_copies(CompiledModel& compiled);

[[nodiscard]] bool support_vulkan_experts(const WeightStore& weights,
                                          const MoeBlockPlan& moe,
                                          uint64_t optimization_flags) noexcept;

[[nodiscard]] bool support_vulkan_shared_experts(const CompiledOperatorTable& operators,
                                                 const MoeBlockPlan& moe) noexcept;

[[nodiscard]] bool support_vulkan_ffn(const CompiledOperatorTable& operators,
                                      const ExpertPlan& ffn) noexcept;

[[nodiscard]] bool support_vulkan_dense_ffn(const CompiledOperatorTable& operators,
                                            const ExpertPlan& ffn) noexcept;

[[nodiscard]] uint64_t dense_vulkan_attention_storage_size(const CompiledModel& compiled,
                                                           const CompiledLayerPlan& layer,
                                                           uint64_t alignment,
                                                           uint32_t device_index = automatic_vulkan_device_index) noexcept;

[[nodiscard]] uint64_t dense_vulkan_weight_storage_size(const CompiledModel& compiled,
                                                        const TensorData& matrix,
                                                        uint64_t alignment,
                                                        uint32_t device_index = automatic_vulkan_device_index,
                                                        bool has_bias = false) noexcept;

[[nodiscard]] uint64_t dense_vulkan_workspace_reserve(uint64_t heap_budget) noexcept;

[[nodiscard]] uint64_t dense_vulkan_pending_state_size(const CompiledModel& compiled,
                                                       uint32_t device_index) noexcept;

// Per-Session recurrent/default-chunk KV state, peak-layer default prefill
// workspace, and physical Expert allocator slack not present in live heap use.
// Uses SessionOptions defaults; larger chunks/contexts remain allocation-checked.
// Per-device state/work reserve for Vulkan attention. Qwen's continuous chain
// owns a larger specialized reserve; ordinary attention uses its initial KV
// rings plus the common execution workspace.
[[nodiscard]] uint64_t vulkan_pending_reserve(const CompiledModel& compiled,
                                              uint32_t device_index,
                                              uint64_t heap_budget) noexcept;

[[nodiscard]] bool support_vulkan_attention(const CompiledOperatorTable& operators,
                                            const AttentionBlockPlan& attention) noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELPIPELINE_H
