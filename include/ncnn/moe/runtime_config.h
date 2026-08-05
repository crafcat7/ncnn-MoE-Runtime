#ifndef NCNN_MOE_RUNTIME_CONFIG_H
#define NCNN_MOE_RUNTIME_CONFIG_H

#include "ncnn/moe/memory_plan.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

// RuntimeOptionFlag belongs with RuntimeConfig so applications can include
// the reusable configuration contract without pulling in Runtime itself.
#define NCNN_MOE_RUNTIME_MMAP_EXPERT_BIT         0
#define NCNN_MOE_RUNTIME_DIRECT_IO_BIT           1
#define NCNN_MOE_RUNTIME_BUFFERED_IO_BIT         2
#define NCNN_MOE_RUNTIME_DISABLE_VICTIM_EXEC_BIT 3
#define NCNN_MOE_RUNTIME_ROUTER_PRED_BIT         4
#define NCNN_MOE_RUNTIME_FORWARD_ARC_BIT        5
#define NCNN_MOE_RUNTIME_RANK_ADAPT_BIT         6
#define NCNN_MOE_RUNTIME_READ_MERGE_BIT         7
#define NCNN_MOE_RUNTIME_ASYNC_ROUTER_PRED_BIT  8
#define NCNN_MOE_RUNTIME_RELEASE_DENSE_BIT      9
#define NCNN_MOE_RUNTIME_DISABLE_GPU_EXPERT_BIT 10

enum RuntimeOptionFlag : uint32_t
{
    RuntimeOptionMemoryMapExperts = UINT32_C(1) << NCNN_MOE_RUNTIME_MMAP_EXPERT_BIT,
    RuntimeOptionDirectExpertIo = UINT32_C(1) << NCNN_MOE_RUNTIME_DIRECT_IO_BIT,
    RuntimeOptionBufferedExpertIo = UINT32_C(1) << NCNN_MOE_RUNTIME_BUFFERED_IO_BIT,
    RuntimeOptionDisableGpuVictimExecution = UINT32_C(1) << NCNN_MOE_RUNTIME_DISABLE_VICTIM_EXEC_BIT,
    RuntimeOptionRouterPrediction = UINT32_C(1) << NCNN_MOE_RUNTIME_ROUTER_PRED_BIT,
    RuntimeOptionForwardAwareCache = UINT32_C(1) << NCNN_MOE_RUNTIME_FORWARD_ARC_BIT,
    RuntimeOptionRankAdaptivePrefetch = UINT32_C(1) << NCNN_MOE_RUNTIME_RANK_ADAPT_BIT,
    RuntimeOptionCrossExpertReadCoalescing = UINT32_C(1) << NCNN_MOE_RUNTIME_READ_MERGE_BIT,
    RuntimeOptionAsyncRouterPrediction = UINT32_C(1) << NCNN_MOE_RUNTIME_ASYNC_ROUTER_PRED_BIT,
    RuntimeOptionReleaseVulkanDenseHostStorage = UINT32_C(1) << NCNN_MOE_RUNTIME_RELEASE_DENSE_BIT,
    RuntimeOptionDisableGpuExpertExecution = UINT32_C(1) << NCNN_MOE_RUNTIME_DISABLE_GPU_EXPERT_BIT
};

// Optimization switches are deliberately separate from RuntimeOptionFlag:
// the latter describes storage, I/O, and admission policy, while this mask
// controls optional CPU/Vulkan kernels.  Every switch is a bit so callers can
// construct a reproducible configuration without process environment state.
#define NCNN_MOE_OPT_CPU_SIMD_RMS_NORM_BIT             0
#define NCNN_MOE_OPT_CPU_FAST_SILU_BIT                 1
#define NCNN_MOE_OPT_CPU_MXFP4_ROW_PAIRS_BIT           2
#define NCNN_MOE_OPT_CPU_FLOAT8_FUSED_GATE_UP_BIT      3
#define NCNN_MOE_OPT_CPU_FLOAT8_BF16_DOT_BIT           4
#define NCNN_MOE_OPT_CPU_FLOAT8_BATCH_TILE_BIT         5
#define NCNN_MOE_OPT_CPU_FLOAT8_SIMD_QUANTIZE_BIT      6
#define NCNN_MOE_OPT_CPU_FLOAT8_FUSED_PROJECTIONS_BIT  7
#define NCNN_MOE_OPT_CPU_BFLOAT16_BATCHED_BIT          8
#define NCNN_MOE_OPT_CPU_BFLOAT16_SINGLE_TOKEN_BIT     9
#define NCNN_MOE_OPT_CPU_ROPE_CACHE_BIT                10
#define NCNN_MOE_OPT_CPU_BF16_DIRECT_ATTENTION_BIT     11
#define NCNN_MOE_OPT_CPU_BF16_ATTENTION_DOT_BIT        12
#define NCNN_MOE_OPT_CPU_GATED_DELTA_SIMD_BIT          13
#define NCNN_MOE_OPT_CPU_LATENT_PREPARED_ROPE_BIT      14
#define NCNN_MOE_OPT_CPU_LATENT_SIMD_NORM_BIT          15
#define NCNN_MOE_OPT_CPU_LATENT_ONLINE_SOFTMAX_BIT     16
#define NCNN_MOE_OPT_CPU_LATENT_OUTPUT_GROUPS_BIT      17
#define NCNN_MOE_OPT_NCNN_CPU_BFLOAT16_LINEAR_BIT      18
#define NCNN_MOE_OPT_VULKAN_BFLOAT16_COOP_MATRIX_BIT   19
#define NCNN_MOE_OPT_VULKAN_FP16_ACTIVATIONS_BIT       21
#define NCNN_MOE_OPT_VULKAN_ATTENTION_BIT              22
#define NCNN_MOE_OPT_VULKAN_ATTENTION_BATCH_BIT        23
#define NCNN_MOE_OPT_VULKAN_KV_PROMOTION_BIT           24
#define NCNN_MOE_OPT_VULKAN_DEVICE_ROPE_BIT            25
#define NCNN_MOE_OPT_VULKAN_DIRECT_HOST_INPUT_BIT      26
#define NCNN_MOE_OPT_VULKAN_DIRECT_HOST_OUTPUT_BIT     27
#define NCNN_MOE_OPT_VULKAN_INPUT_QUANTIZE_BIT         28
#define NCNN_MOE_OPT_VULKAN_FP8_TILE4_BIT              29
#define NCNN_MOE_OPT_VULKAN_DECODE_SDPA_BIT            30
#define NCNN_MOE_OPT_VULKAN_QKV_RING_BIT               31
#define NCNN_MOE_OPT_VULKAN_ROUTE_AGGREGATION_BIT      32
#define NCNN_MOE_OPT_VULKAN_EXPERT_BATCH_ADMISSION_BIT 33
#define NCNN_MOE_OPT_VULKAN_EXPERT_EAGER_ADMISSION_BIT 34
#define NCNN_MOE_OPT_VULKAN_LATENT_COMPRESSOR_BIT      35
#define NCNN_MOE_OPT_CPU_BFLOAT16_FORCE_SMALL_BIT      36
#define NCNN_MOE_OPT_VULKAN_ROUTE_AGGREGATION_FORCE_BIT 37
#define NCNN_MOE_OPT_VULKAN_DECODE_SDPA_FORCE_BIT      38
#define NCNN_MOE_OPT_NCNN_CPU_BFLOAT16_LINEAR_FORCE_BIT 39
#define NCNN_MOE_OPT_VULKAN_PIPELINE_BIND_ELISION_BIT  40
#define NCNN_MOE_OPT_VULKAN_READONLY_BINDINGS_BIT      41
#define NCNN_MOE_OPT_VULKAN_BATCH_BUFFER_BARRIERS_BIT  42
#define NCNN_MOE_OPT_VULKAN_STACK_DESCRIPTOR_PAYLOAD_BIT 43
#define NCNN_MOE_OPT_VULKAN_COMMAND_GRAPH_BIT          44
#define NCNN_MOE_OPT_CPU_LATENT_VECTOR_SOFTMAX_BIT    45
#define NCNN_MOE_OPT_VULKAN_LATENT_INPUT_RMS_NORM_BIT  46
#define NCNN_MOE_OPT_VULKAN_EXPERT_GPU_PRIORITY_BIT   47

enum RuntimeOptimizationFlag : uint64_t
{
    RuntimeOptimizationCpuSimdRmsNorm = UINT64_C(1) << NCNN_MOE_OPT_CPU_SIMD_RMS_NORM_BIT,
    RuntimeOptimizationCpuFastSilu = UINT64_C(1) << NCNN_MOE_OPT_CPU_FAST_SILU_BIT,
    RuntimeOptimizationCpuMxfp4RowPairs = UINT64_C(1) << NCNN_MOE_OPT_CPU_MXFP4_ROW_PAIRS_BIT,
    RuntimeOptimizationCpuFloat8FusedGateUp = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_FUSED_GATE_UP_BIT,
    RuntimeOptimizationCpuFloat8Bf16Dot = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_BF16_DOT_BIT,
    RuntimeOptimizationCpuFloat8BatchTile = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_BATCH_TILE_BIT,
    RuntimeOptimizationCpuFloat8SimdQuantize = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_SIMD_QUANTIZE_BIT,
    RuntimeOptimizationCpuFloat8FusedProjections = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_FUSED_PROJECTIONS_BIT,
    RuntimeOptimizationCpuBfloat16Batched = UINT64_C(1) << NCNN_MOE_OPT_CPU_BFLOAT16_BATCHED_BIT,
    RuntimeOptimizationCpuBfloat16SingleToken = UINT64_C(1) << NCNN_MOE_OPT_CPU_BFLOAT16_SINGLE_TOKEN_BIT,
    RuntimeOptimizationCpuRopeCache = UINT64_C(1) << NCNN_MOE_OPT_CPU_ROPE_CACHE_BIT,
    RuntimeOptimizationCpuBf16DirectAttention = UINT64_C(1) << NCNN_MOE_OPT_CPU_BF16_DIRECT_ATTENTION_BIT,
    RuntimeOptimizationCpuBf16AttentionDot = UINT64_C(1) << NCNN_MOE_OPT_CPU_BF16_ATTENTION_DOT_BIT,
    RuntimeOptimizationCpuGatedDeltaSimd = UINT64_C(1) << NCNN_MOE_OPT_CPU_GATED_DELTA_SIMD_BIT,
    RuntimeOptimizationCpuLatentPreparedRope = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_PREPARED_ROPE_BIT,
    RuntimeOptimizationCpuLatentSimdNorm = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_SIMD_NORM_BIT,
    RuntimeOptimizationCpuLatentOnlineSoftmax = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_ONLINE_SOFTMAX_BIT,
    RuntimeOptimizationCpuLatentOutputGroups = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_OUTPUT_GROUPS_BIT,
    RuntimeOptimizationNcnnCpuBfloat16Linear = UINT64_C(1) << NCNN_MOE_OPT_NCNN_CPU_BFLOAT16_LINEAR_BIT,
    RuntimeOptimizationVulkanBfloat16CoopMatrix = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_BFLOAT16_COOP_MATRIX_BIT,
    RuntimeOptimizationVulkanFp16Activations = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_FP16_ACTIVATIONS_BIT,
    RuntimeOptimizationVulkanAttention = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_ATTENTION_BIT,
    RuntimeOptimizationVulkanAttentionBatch = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_ATTENTION_BATCH_BIT,
    RuntimeOptimizationVulkanKvPromotion = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_KV_PROMOTION_BIT,
    RuntimeOptimizationVulkanDeviceRope = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_DEVICE_ROPE_BIT,
    RuntimeOptimizationVulkanDirectHostInput = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_DIRECT_HOST_INPUT_BIT,
    RuntimeOptimizationVulkanDirectHostOutput = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_DIRECT_HOST_OUTPUT_BIT,
    RuntimeOptimizationVulkanInputQuantize = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_INPUT_QUANTIZE_BIT,
    RuntimeOptimizationVulkanFp8Tile4 = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_FP8_TILE4_BIT,
    RuntimeOptimizationVulkanDecodeSdpa = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_DECODE_SDPA_BIT,
    RuntimeOptimizationVulkanQkvRing = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_QKV_RING_BIT,
    RuntimeOptimizationVulkanRouteAggregation = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_ROUTE_AGGREGATION_BIT,
    RuntimeOptimizationVulkanExpertBatchAdmission = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_EXPERT_BATCH_ADMISSION_BIT,
    RuntimeOptimizationVulkanExpertEagerAdmission = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_EXPERT_EAGER_ADMISSION_BIT,
    RuntimeOptimizationVulkanLatentCompressor = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_LATENT_COMPRESSOR_BIT,
    RuntimeOptimizationCpuBfloat16ForceSmall = UINT64_C(1) << NCNN_MOE_OPT_CPU_BFLOAT16_FORCE_SMALL_BIT,
    RuntimeOptimizationVulkanRouteAggregationForce = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_ROUTE_AGGREGATION_FORCE_BIT,
    RuntimeOptimizationVulkanDecodeSdpaForce = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_DECODE_SDPA_FORCE_BIT,
    RuntimeOptimizationNcnnCpuBfloat16LinearForce = UINT64_C(1) << NCNN_MOE_OPT_NCNN_CPU_BFLOAT16_LINEAR_FORCE_BIT,
    RuntimeOptimizationVulkanPipelineBindElision = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_PIPELINE_BIND_ELISION_BIT,
    RuntimeOptimizationVulkanReadonlyBindings = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_READONLY_BINDINGS_BIT,
    RuntimeOptimizationVulkanBatchBufferBarriers = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_BATCH_BUFFER_BARRIERS_BIT,
    RuntimeOptimizationVulkanStackDescriptorPayload = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_STACK_DESCRIPTOR_PAYLOAD_BIT,
    RuntimeOptimizationVulkanCommandGraph = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_COMMAND_GRAPH_BIT,
    RuntimeOptimizationCpuLatentVectorSoftmax = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_VECTOR_SOFTMAX_BIT,
    RuntimeOptimizationVulkanLatentInputRmsNorm = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_LATENT_INPUT_RMS_NORM_BIT,
    RuntimeOptimizationVulkanExpertGpuPriority = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_EXPERT_GPU_PRIORITY_BIT
};

inline constexpr uint64_t RuntimeOptimizationDefaultFlags =
    RuntimeOptimizationCpuSimdRmsNorm
    | RuntimeOptimizationCpuFastSilu
    | RuntimeOptimizationCpuFloat8FusedGateUp
    | RuntimeOptimizationCpuFloat8Bf16Dot
    | RuntimeOptimizationCpuFloat8BatchTile
    | RuntimeOptimizationCpuFloat8SimdQuantize
    | RuntimeOptimizationCpuFloat8FusedProjections
    | RuntimeOptimizationCpuBfloat16Batched
    | RuntimeOptimizationCpuRopeCache
    | RuntimeOptimizationCpuBf16DirectAttention
    | RuntimeOptimizationCpuGatedDeltaSimd
    | RuntimeOptimizationCpuLatentPreparedRope
    | RuntimeOptimizationCpuLatentSimdNorm
    | RuntimeOptimizationCpuLatentOnlineSoftmax
    | RuntimeOptimizationCpuLatentOutputGroups
    | RuntimeOptimizationNcnnCpuBfloat16Linear
    | RuntimeOptimizationVulkanBfloat16CoopMatrix
    | RuntimeOptimizationVulkanAttention
    | RuntimeOptimizationVulkanAttentionBatch
    | RuntimeOptimizationVulkanKvPromotion
    | RuntimeOptimizationVulkanDeviceRope
    | RuntimeOptimizationVulkanDecodeSdpa
    | RuntimeOptimizationVulkanQkvRing
    | RuntimeOptimizationVulkanRouteAggregation
    | RuntimeOptimizationVulkanExpertBatchAdmission
    | RuntimeOptimizationVulkanExpertEagerAdmission
    | RuntimeOptimizationVulkanLatentCompressor
    | RuntimeOptimizationVulkanPipelineBindElision
    | RuntimeOptimizationVulkanReadonlyBindings
    | RuntimeOptimizationVulkanBatchBufferBarriers
    | RuntimeOptimizationVulkanStackDescriptorPayload
    | RuntimeOptimizationVulkanCommandGraph
    | RuntimeOptimizationCpuLatentVectorSoftmax
    | RuntimeOptimizationVulkanLatentInputRmsNorm
    | RuntimeOptimizationVulkanExpertGpuPriority;

[[nodiscard]] inline bool runtime_optimization_enabled(
    uint64_t optimization_flags,
    uint64_t flag) noexcept
{
    return has_flag(optimization_flags, flag);
}

// User-supplied runtime configuration. Zero-valued memory settings and Auto
// enum values leave hardware- and model-specific decisions to Runtime.
// Tokenizer, prompt, chat, and sampling state deliberately do not belong here.
struct RuntimeConfig
{
    HybridMode hybrid_mode = HybridMode::Auto;
    ExpertMemoryMode expert_memory_mode = ExpertMemoryMode::Auto;
    uint64_t host_memory_budget_bytes = 0;
    uint64_t expert_cache_bytes = 0;
    uint64_t expert_gpu_cache_bytes = 0;
    uint64_t expert_gpu_victim_cache_bytes = 0;
    uint32_t expert_gpu_victim_reuse_probe_interval = 1;
    uint32_t expert_io_workers = 0;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    uint32_t expected_concurrency = 1;
    std::vector<uint32_t> vulkan_device_indices;
    uint32_t flags = 0;
    uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
};

// The concrete settings selected after RuntimeConfig::Auto has been resolved
// and the model has been compiled.  This is intentionally separate from
// RuntimeConfig so applications can show the effective plan without having
// to duplicate the runtime's hardware and memory decisions.
struct EffectiveRuntimeConfig
{
    HybridMode hybrid_mode = HybridMode::CpuOnly;
    ExpertMemoryMode requested_expert_memory_mode = ExpertMemoryMode::Auto;
    ExpertMemoryMode selected_expert_memory_mode = ExpertMemoryMode::Eager;
    uint64_t host_memory_budget_bytes = 0;
    uint64_t expert_cache_bytes = 0;
    uint64_t expert_gpu_cache_bytes = 0;
    uint64_t expert_gpu_victim_cache_bytes = 0;
    uint32_t expert_gpu_victim_reuse_probe_interval = 1;
    uint32_t expert_io_workers = 0;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    std::vector<uint32_t> vulkan_device_indices;
    uint32_t flags = 0;
    uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
    uint32_t expected_concurrency = 1;
    bool file_backed_experts = false;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RUNTIME_CONFIG_H
