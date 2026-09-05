#ifndef NCNN_MOE_OPTION_H
#define NCNN_MOE_OPTION_H

#include "ncnn/moe/types.h"

#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

// Model loading option flags are kept in this public header.
#define NCNN_MOE_OPTION_MMAP_EXPERT_BIT         0
#define NCNN_MOE_OPTION_DIRECT_IO_BIT           1
#define NCNN_MOE_OPTION_BUFFERED_IO_BIT         2
#define NCNN_MOE_OPTION_DISABLE_VICTIM_EXEC_BIT 3
#define NCNN_MOE_OPTION_ROUTER_PRED_BIT         4
#define NCNN_MOE_OPTION_FORWARD_ARC_BIT         5
#define NCNN_MOE_OPTION_RANK_ADAPT_BIT          6
#define NCNN_MOE_OPTION_READ_MERGE_BIT          7
#define NCNN_MOE_OPTION_ASYNC_ROUTER_PRED_BIT   8
#define NCNN_MOE_OPTION_RELEASE_DENSE_BIT       9
#define NCNN_MOE_OPTION_DISABLE_GPU_EXPERT_BIT  10

enum OptionFlag : uint32_t
{
    OptionMemoryMapExperts = UINT32_C(1) << NCNN_MOE_OPTION_MMAP_EXPERT_BIT,
    OptionDirectExpertIo = UINT32_C(1) << NCNN_MOE_OPTION_DIRECT_IO_BIT,
    OptionBufferedExpertIo = UINT32_C(1) << NCNN_MOE_OPTION_BUFFERED_IO_BIT,
    OptionDisableGpuVictimExecution = UINT32_C(1) << NCNN_MOE_OPTION_DISABLE_VICTIM_EXEC_BIT,
    OptionRouterPrediction = UINT32_C(1) << NCNN_MOE_OPTION_ROUTER_PRED_BIT,
    OptionForwardAwareCache = UINT32_C(1) << NCNN_MOE_OPTION_FORWARD_ARC_BIT,
    OptionRankAdaptivePrefetch = UINT32_C(1) << NCNN_MOE_OPTION_RANK_ADAPT_BIT,
    OptionCrossExpertReadCoalescing = UINT32_C(1) << NCNN_MOE_OPTION_READ_MERGE_BIT,
    OptionAsyncRouterPrediction = UINT32_C(1) << NCNN_MOE_OPTION_ASYNC_ROUTER_PRED_BIT,
    OptionReleaseVulkanDenseHostStorage = UINT32_C(1) << NCNN_MOE_OPTION_RELEASE_DENSE_BIT,
    OptionDisableGpuExpertExecution = UINT32_C(1) << NCNN_MOE_OPTION_DISABLE_GPU_EXPERT_BIT
};

inline constexpr uint32_t OptionExpertIoMask = OptionMemoryMapExperts
                                               | OptionDirectExpertIo
                                               | OptionBufferedExpertIo;

// Kernel choices use a separate bitmap from storage and admission policy.
#define NCNN_MOE_OPT_CPU_SIMD_RMS_NORM_BIT               0
#define NCNN_MOE_OPT_CPU_FAST_SILU_BIT                   1
#define NCNN_MOE_OPT_CPU_MXFP4_ROW_PAIRS_BIT             2
#define NCNN_MOE_OPT_CPU_FLOAT8_FUSED_GATE_UP_BIT        3
#define NCNN_MOE_OPT_CPU_FLOAT8_BF16_DOT_BIT             4
#define NCNN_MOE_OPT_CPU_FLOAT8_BATCH_TILE_BIT           5
#define NCNN_MOE_OPT_CPU_FLOAT8_SIMD_QUANTIZE_BIT        6
#define NCNN_MOE_OPT_CPU_FLOAT8_FUSED_PROJECTIONS_BIT    7
#define NCNN_MOE_OPT_CPU_BFLOAT16_BATCHED_BIT            8
#define NCNN_MOE_OPT_CPU_ROPE_CACHE_BIT                  9
#define NCNN_MOE_OPT_CPU_BF16_DIRECT_ATTENTION_BIT       10
#define NCNN_MOE_OPT_CPU_GATED_DELTA_SIMD_BIT            11
#define NCNN_MOE_OPT_CPU_LATENT_PREPARED_ROPE_BIT        12
#define NCNN_MOE_OPT_CPU_LATENT_SIMD_NORM_BIT            13
#define NCNN_MOE_OPT_CPU_LATENT_ONLINE_SOFTMAX_BIT       14
#define NCNN_MOE_OPT_CPU_LATENT_OUTPUT_GROUPS_BIT        15
#define NCNN_MOE_OPT_NCNN_CPU_BFLOAT16_LINEAR_BIT        16
#define NCNN_MOE_OPT_VULKAN_BFLOAT16_COOP_MATRIX_BIT     17
#define NCNN_MOE_OPT_VULKAN_ATTENTION_BIT                18
#define NCNN_MOE_OPT_VULKAN_ATTENTION_BATCH_BIT          19
#define NCNN_MOE_OPT_VULKAN_KV_PROMOTION_BIT             20
#define NCNN_MOE_OPT_VULKAN_DEVICE_ROPE_BIT              21
#define NCNN_MOE_OPT_VULKAN_DECODE_SDPA_BIT              22
#define NCNN_MOE_OPT_VULKAN_QKV_RING_BIT                 23
#define NCNN_MOE_OPT_VULKAN_ROUTE_AGGREGATION_BIT        24
#define NCNN_MOE_OPT_VULKAN_EXPERT_BATCH_ADMISSION_BIT   25
#define NCNN_MOE_OPT_VULKAN_EXPERT_EAGER_ADMISSION_BIT   26
#define NCNN_MOE_OPT_VULKAN_LATENT_COMPRESSOR_BIT        27
#define NCNN_MOE_OPT_VULKAN_PIPELINE_BIND_ELISION_BIT    28
#define NCNN_MOE_OPT_VULKAN_READONLY_BINDINGS_BIT        29
#define NCNN_MOE_OPT_VULKAN_BATCH_BUFFER_BARRIERS_BIT    30
#define NCNN_MOE_OPT_VULKAN_STACK_DESCRIPTOR_PAYLOAD_BIT 31
#define NCNN_MOE_OPT_VULKAN_COMMAND_GRAPH_BIT            32
#define NCNN_MOE_OPT_CPU_LATENT_VECTOR_SOFTMAX_BIT       33
#define NCNN_MOE_OPT_VULKAN_LATENT_INPUT_RMS_NORM_BIT    34
#define NCNN_MOE_OPT_VULKAN_EXPERT_GPU_PRIORITY_BIT      35
#define NCNN_MOE_OPT_CPU_MXFP4_Q8_BIT                    36
#define NCNN_MOE_OPT_CPU_FLASH_ATTENTION_BIT             37
#define NCNN_MOE_OPT_CPU_SPLIT_KV_ATTENTION_BIT          38
#define NCNN_MOE_OPT_VULKAN_INDEXED_EXPERTS_BIT          39
#define NCNN_MOE_OPT_VULKAN_QNK_BIT                      40
#define NCNN_MOE_OPT_CPU_PACKED_WEIGHTS_BIT              41

enum OptimizationFlag : uint64_t
{
    OptimizationCpuSimdRmsNorm = UINT64_C(1) << NCNN_MOE_OPT_CPU_SIMD_RMS_NORM_BIT,
    OptimizationCpuFastSilu = UINT64_C(1) << NCNN_MOE_OPT_CPU_FAST_SILU_BIT,
    OptimizationCpuMxfp4RowPairs = UINT64_C(1) << NCNN_MOE_OPT_CPU_MXFP4_ROW_PAIRS_BIT,
    OptimizationCpuFloat8FusedGateUp = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_FUSED_GATE_UP_BIT,
    OptimizationCpuFloat8Bf16Dot = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_BF16_DOT_BIT,
    OptimizationCpuFloat8BatchTile = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_BATCH_TILE_BIT,
    OptimizationCpuFloat8SimdQuantize = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_SIMD_QUANTIZE_BIT,
    OptimizationCpuFloat8FusedProjections = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLOAT8_FUSED_PROJECTIONS_BIT,
    OptimizationCpuBfloat16Batched = UINT64_C(1) << NCNN_MOE_OPT_CPU_BFLOAT16_BATCHED_BIT,
    OptimizationCpuRopeCache = UINT64_C(1) << NCNN_MOE_OPT_CPU_ROPE_CACHE_BIT,
    OptimizationCpuBf16DirectAttention = UINT64_C(1) << NCNN_MOE_OPT_CPU_BF16_DIRECT_ATTENTION_BIT,
    OptimizationCpuGatedDeltaSimd = UINT64_C(1) << NCNN_MOE_OPT_CPU_GATED_DELTA_SIMD_BIT,
    OptimizationCpuLatentPreparedRope = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_PREPARED_ROPE_BIT,
    OptimizationCpuLatentSimdNorm = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_SIMD_NORM_BIT,
    OptimizationCpuLatentOnlineSoftmax = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_ONLINE_SOFTMAX_BIT,
    OptimizationCpuLatentOutputGroups = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_OUTPUT_GROUPS_BIT,
    OptimizationNcnnCpuBfloat16Linear = UINT64_C(1) << NCNN_MOE_OPT_NCNN_CPU_BFLOAT16_LINEAR_BIT,
    OptimizationVulkanBfloat16CoopMatrix = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_BFLOAT16_COOP_MATRIX_BIT,
    OptimizationVulkanAttention = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_ATTENTION_BIT,
    OptimizationVulkanAttentionBatch = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_ATTENTION_BATCH_BIT,
    OptimizationVulkanKvPromotion = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_KV_PROMOTION_BIT,
    OptimizationVulkanDeviceRope = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_DEVICE_ROPE_BIT,
    OptimizationVulkanDecodeSdpa = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_DECODE_SDPA_BIT,
    OptimizationVulkanQkvRing = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_QKV_RING_BIT,
    OptimizationVulkanRouteAggregation = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_ROUTE_AGGREGATION_BIT,
    OptimizationVulkanExpertBatchAdmission = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_EXPERT_BATCH_ADMISSION_BIT,
    OptimizationVulkanExpertEagerAdmission = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_EXPERT_EAGER_ADMISSION_BIT,
    OptimizationVulkanLatentCompressor = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_LATENT_COMPRESSOR_BIT,
    OptimizationVulkanPipelineBindElision = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_PIPELINE_BIND_ELISION_BIT,
    OptimizationVulkanReadonlyBindings = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_READONLY_BINDINGS_BIT,
    OptimizationVulkanBatchBufferBarriers = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_BATCH_BUFFER_BARRIERS_BIT,
    OptimizationVulkanStackDescriptorPayload = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_STACK_DESCRIPTOR_PAYLOAD_BIT,
    OptimizationVulkanCommandGraph = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_COMMAND_GRAPH_BIT,
    OptimizationCpuLatentVectorSoftmax = UINT64_C(1) << NCNN_MOE_OPT_CPU_LATENT_VECTOR_SOFTMAX_BIT,
    OptimizationVulkanLatentInputRmsNorm = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_LATENT_INPUT_RMS_NORM_BIT,
    OptimizationVulkanExpertGpuPriority = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_EXPERT_GPU_PRIORITY_BIT,
    OptimizationCpuMxfp4Q8 = UINT64_C(1) << NCNN_MOE_OPT_CPU_MXFP4_Q8_BIT,
    OptimizationCpuFlashAttention = UINT64_C(1) << NCNN_MOE_OPT_CPU_FLASH_ATTENTION_BIT,
    OptimizationCpuSplitKvAttention = UINT64_C(1) << NCNN_MOE_OPT_CPU_SPLIT_KV_ATTENTION_BIT,
    OptimizationVulkanIndexedExperts = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_INDEXED_EXPERTS_BIT,
    OptimizationVulkanQnK = UINT64_C(1) << NCNN_MOE_OPT_VULKAN_QNK_BIT,
    // Internal execution bit selected from CpuPackedWeightMode. It is not
    // part of OptimizationDefaultFlags.
    OptimizationCpuPackedWeights = UINT64_C(1) << NCNN_MOE_OPT_CPU_PACKED_WEIGHTS_BIT
};

inline constexpr uint64_t OptimizationDefaultFlags = OptimizationCpuSimdRmsNorm
                                                     | OptimizationCpuFastSilu
                                                     | OptimizationCpuMxfp4RowPairs
                                                     | OptimizationCpuFloat8FusedGateUp
                                                     | OptimizationCpuFloat8Bf16Dot
                                                     | OptimizationCpuFloat8BatchTile
                                                     | OptimizationCpuFloat8SimdQuantize
                                                     | OptimizationCpuFloat8FusedProjections
                                                     | OptimizationCpuBfloat16Batched
                                                     | OptimizationCpuRopeCache
                                                     | OptimizationCpuBf16DirectAttention
                                                     | OptimizationCpuFlashAttention
                                                     | OptimizationCpuSplitKvAttention
                                                     | OptimizationCpuGatedDeltaSimd
                                                     | OptimizationCpuLatentPreparedRope
                                                     | OptimizationCpuLatentSimdNorm
                                                     | OptimizationCpuLatentOnlineSoftmax
                                                     | OptimizationCpuLatentOutputGroups
                                                     | OptimizationCpuMxfp4Q8
                                                     | OptimizationNcnnCpuBfloat16Linear
                                                     | OptimizationVulkanBfloat16CoopMatrix
                                                     | OptimizationVulkanAttention
                                                     | OptimizationVulkanAttentionBatch
                                                     | OptimizationVulkanKvPromotion
                                                     | OptimizationVulkanDeviceRope
                                                     | OptimizationVulkanDecodeSdpa
                                                     | OptimizationVulkanQkvRing
                                                     | OptimizationVulkanRouteAggregation
                                                     | OptimizationVulkanExpertBatchAdmission
                                                     | OptimizationVulkanExpertEagerAdmission
                                                     | OptimizationVulkanLatentCompressor
                                                     | OptimizationVulkanPipelineBindElision
                                                     | OptimizationVulkanReadonlyBindings
                                                     | OptimizationVulkanBatchBufferBarriers
                                                     | OptimizationVulkanStackDescriptorPayload
                                                     | OptimizationVulkanCommandGraph
                                                     | OptimizationCpuLatentVectorSoftmax
                                                     | OptimizationVulkanLatentInputRmsNorm
                                                     | OptimizationVulkanExpertGpuPriority
                                                     | OptimizationVulkanIndexedExperts
                                                     | OptimizationVulkanQnK;

inline constexpr uint64_t OptimizationAllFlags = (UINT64_C(1) << (NCNN_MOE_OPT_CPU_PACKED_WEIGHTS_BIT + 1)) - 1;
inline constexpr uint64_t OptimizationOptionalFlags = OptimizationCpuPackedWeights;
static_assert(
    (OptimizationDefaultFlags | OptimizationOptionalFlags)
    == OptimizationAllFlags);
static_assert(
    (OptimizationDefaultFlags & OptimizationOptionalFlags) == 0);

// User-supplied model loading options. Zero-valued memory settings and Auto
// enum values leave hardware- and model-specific decisions to Runtime.
// Tokenizer, prompt, chat, and sampling state deliberately do not belong here.
enum class ExpertMemoryMode
{
    Auto,
    Eager,
    OnDemand
};

enum class CpuPackedWeightMode
{
    Disabled,
    Enabled
};

struct Option
{
    HybridMode hybrid_mode = HybridMode::Auto;
    ExpertMemoryMode expert_memory_mode = ExpertMemoryMode::Auto;
    CpuPackedWeightMode cpu_packed_weight_mode = CpuPackedWeightMode::Disabled;
    uint64_t host_memory_budget = 0;
    uint64_t expert_cache_size = 0;
    uint64_t expert_gpu_cache_size = 0;
    uint64_t expert_gpu_victim_cache_size = 0;
    uint32_t expert_gpu_victim_reuse_probe_interval = 1;
    uint32_t num_expert_io_threads = 0;
    uint32_t vulkan_device_index = automatic_vulkan_device_index;
    uint32_t num_concurrent_sessions = 1;
    std::vector<uint32_t> vulkan_device_indices;
    uint32_t flags = 0;
    uint64_t optimization_flags = OptimizationDefaultFlags;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_OPTION_H
