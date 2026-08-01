#ifndef NCNN_MOE_RUNTIME_H
#define NCNN_MOE_RUNTIME_H

#include "ncnn/moe/model.h"
#include "ncnn/moe/model_adapter.h"
#include "ncnn/moe/memory_plan.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/scheduler.h"
#include "ncnn/moe/session.h"
#include "ncnn/moe/types.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

#define NCNN_MOE_RUNTIME_MMAP_EXPERT_BIT         0
#define NCNN_MOE_RUNTIME_DIRECT_IO_BIT           1
#define NCNN_MOE_RUNTIME_BUFFERED_IO_BIT         2
#define NCNN_MOE_RUNTIME_DISABLE_VICTIM_EXEC_BIT 3
#define NCNN_MOE_RUNTIME_ROUTER_PRED_BIT         4
#define NCNN_MOE_RUNTIME_FORWARD_ARC_BIT         5
#define NCNN_MOE_RUNTIME_RANK_ADAPT_BIT          6
#define NCNN_MOE_RUNTIME_READ_MERGE_BIT          7
#define NCNN_MOE_RUNTIME_ASYNC_ROUTER_PRED_BIT   8
#define NCNN_MOE_RUNTIME_RELEASE_DENSE_BIT       9

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
    RuntimeOptionReleaseVulkanDenseHostStorage = UINT32_C(1) << NCNN_MOE_RUNTIME_RELEASE_DENSE_BIT
};

struct RuntimeOptions
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
};

#define NCNN_MOE_RUNTIME_CAP_CPU_BIT              0
#define NCNN_MOE_RUNTIME_CAP_NCNN_LINEAR_BIT      1
#define NCNN_MOE_RUNTIME_CAP_VULKAN_BIT           2
#define NCNN_MOE_RUNTIME_CAP_VULKAN_CPU_BIT       3
#define NCNN_MOE_RUNTIME_CAP_VULKAN_PREFETCH_BIT  4
#define NCNN_MOE_RUNTIME_CAP_VULKAN_ATTENTION_BIT 5
#define NCNN_MOE_RUNTIME_CAP_VULKAN_VICTIM_BIT    6
#define NCNN_MOE_RUNTIME_CAP_MXFP4_CPU_BIT        7
#define NCNN_MOE_RUNTIME_CAP_NEON_BIT             8
#define NCNN_MOE_RUNTIME_CAP_AVX2_BIT             9
#define NCNN_MOE_RUNTIME_CAP_AVX512_BIT           10
#define NCNN_MOE_RUNTIME_CAP_OPENMP_BIT           11
#define NCNN_MOE_RUNTIME_CAP_CROSS_SESSION_BIT    12
#define NCNN_MOE_RUNTIME_CAP_DOUBLE_BUFFER_BIT    14
#define NCNN_MOE_RUNTIME_CAP_MXFP4_VULKAN_BIT     15
#define NCNN_MOE_RUNTIME_CAP_SVE2_BIT             16
#define NCNN_MOE_RUNTIME_CAP_MULTI_VULKAN_BIT     17

enum RuntimeCapabilityFlag : uint32_t
{
    RuntimeCapabilityCpuExecution = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_CPU_BIT,
    RuntimeCapabilityNcnnCpuLinear = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_NCNN_LINEAR_BIT,
    RuntimeCapabilityVulkanExecution = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_VULKAN_BIT,
    RuntimeCapabilityVulkanCpuMix = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_VULKAN_CPU_BIT,
    RuntimeCapabilityVulkanCpuPrefetch = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_VULKAN_PREFETCH_BIT,
    RuntimeCapabilityVulkanAttention = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_VULKAN_ATTENTION_BIT,
    RuntimeCapabilityVulkanVictimCache = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_VULKAN_VICTIM_BIT,
    RuntimeCapabilityMxfp4CpuKernel = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_MXFP4_CPU_BIT,
    RuntimeCapabilityMxfp4ArmNeon = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_NEON_BIT,
    RuntimeCapabilityMxfp4X86Avx2 = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_AVX2_BIT,
    RuntimeCapabilityMxfp4X86Avx512 = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_AVX512_BIT,
    RuntimeCapabilityOpenmpExperts = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_OPENMP_BIT,
    RuntimeCapabilityCrossSessionScheduling = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_CROSS_SESSION_BIT,
    RuntimeCapabilityVulkanDoubleBuffering = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_DOUBLE_BUFFER_BIT,
    RuntimeCapabilityMxfp4VulkanProjection = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_MXFP4_VULKAN_BIT,
    RuntimeCapabilityMxfp4ArmSve2 = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_SVE2_BIT,
    RuntimeCapabilityMultiVulkanPlacement = UINT32_C(1) << NCNN_MOE_RUNTIME_CAP_MULTI_VULKAN_BIT
};

#define NCNN_MOE_ISA_ARM_NEON_BIT        0
#define NCNN_MOE_ISA_ARM_SVE_BIT         1
#define NCNN_MOE_ISA_ARM_SVE2_BIT        2
#define NCNN_MOE_ISA_X86_AVX2_FMA_BIT    3
#define NCNN_MOE_ISA_X86_AVX512_BIT      4
#define NCNN_MOE_ISA_X86_AVX_VNNI_BIT    5
#define NCNN_MOE_ISA_X86_AVX512_VNNI_BIT 6
#define NCNN_MOE_ISA_X86_AVX512_BF16_BIT 7
#define NCNN_MOE_ISA_X86_AMX_TILE_BIT    8
#define NCNN_MOE_ISA_X86_AMX_INT8_BIT    9
#define NCNN_MOE_ISA_X86_AMX_BF16_BIT    10

enum CpuIsaCapabilityFlag : uint64_t
{
    CpuIsaArmNeon = UINT64_C(1) << NCNN_MOE_ISA_ARM_NEON_BIT,
    CpuIsaArmSve = UINT64_C(1) << NCNN_MOE_ISA_ARM_SVE_BIT,
    CpuIsaArmSve2 = UINT64_C(1) << NCNN_MOE_ISA_ARM_SVE2_BIT,
    CpuIsaX86Avx2Fma = UINT64_C(1) << NCNN_MOE_ISA_X86_AVX2_FMA_BIT,
    CpuIsaX86Avx512 = UINT64_C(1) << NCNN_MOE_ISA_X86_AVX512_BIT,
    CpuIsaX86AvxVnni = UINT64_C(1) << NCNN_MOE_ISA_X86_AVX_VNNI_BIT,
    CpuIsaX86Avx512Vnni = UINT64_C(1) << NCNN_MOE_ISA_X86_AVX512_VNNI_BIT,
    CpuIsaX86Avx512Bf16 = UINT64_C(1) << NCNN_MOE_ISA_X86_AVX512_BF16_BIT,
    CpuIsaX86AmxTile = UINT64_C(1) << NCNN_MOE_ISA_X86_AMX_TILE_BIT,
    CpuIsaX86AmxInt8 = UINT64_C(1) << NCNN_MOE_ISA_X86_AMX_INT8_BIT,
    CpuIsaX86AmxBf16 = UINT64_C(1) << NCNN_MOE_ISA_X86_AMX_BF16_BIT
};

struct RuntimeCapabilities
{
    uint64_t physical_memory_bytes = 0;
    uint64_t vulkan_heap_budget_bytes = 0;
    uint32_t vulkan_device_count = 0;
    uint32_t selected_vulkan_device_index = 0;
    uint32_t logical_cpu_count = 1;
    uint32_t physical_cpu_core_count = 1;
    uint32_t openmp_thread_count = 1;
    uint32_t cpu_linear_thread_limit = 1;
    uint32_t float8_linear_thread_limit = 1;
    uint32_t float8_linear_row_group_size = 1;
    uint32_t mxfp4_decode_row_pair_group_size = 1;
    uint64_t cpu_isa_flags = 0;
    uint32_t flags = RuntimeCapabilityCpuExecution | RuntimeCapabilityMxfp4CpuKernel | RuntimeCapabilityCrossSessionScheduling;
    std::string mxfp4_kernel;
    std::string float8_kernel;
    std::string cpu_isa;
    std::string activation_kernel;
    std::vector<VulkanDeviceCapabilities> vulkan_devices;
};

class Runtime
{
private:
    std::vector<std::shared_ptr<IMoeModelAdapter>> adapters_;
    RuntimeCapabilities capabilities_;

public:
    Runtime();

    [[nodiscard]] const RuntimeCapabilities& capabilities() const noexcept
    {
        return capabilities_;
    }

    void register_adapter(std::shared_ptr<IMoeModelAdapter> adapter);

    [[nodiscard]] Result<ModelPtr> load_model(const std::filesystem::path& model_path, const RuntimeOptions& options = {});

    // Synchronization point for warm-up and traffic transitions.
    [[nodiscard]] Result<void> synchronize_model_caches(const ModelPtr& model);

    [[nodiscard]] Result<SessionPtr> create_session(const ModelPtr& model, const SessionOptions& options = {});

    [[nodiscard]] Result<BatchSchedulerPtr> create_scheduler(const SchedulerOptions& options = {});
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RUNTIME_H
