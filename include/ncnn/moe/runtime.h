#ifndef NCNN_MOE_RUNTIME_H
#define NCNN_MOE_RUNTIME_H

#include "ncnn/moe/model.h"
#include "ncnn/moe/model_adapter.h"
#include "ncnn/moe/memory_plan.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/option.h"
#include "ncnn/moe/scheduler.h"
#include "ncnn/moe/session.h"
#include "ncnn/moe/types.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace ncnn {
namespace moe {

// Runtime capability bit positions.
#define NCNN_MOE_RUNTIME_CPU_BIT              0
#define NCNN_MOE_RUNTIME_NCNN_LINEAR_BIT      1
#define NCNN_MOE_RUNTIME_VULKAN_BIT           2
#define NCNN_MOE_RUNTIME_VULKAN_CPU_BIT       3
#define NCNN_MOE_RUNTIME_VULKAN_ATTENTION_BIT 4
#define NCNN_MOE_RUNTIME_VULKAN_VICTIM_BIT    5
#define NCNN_MOE_RUNTIME_MXFP4_CPU_BIT       6
#define NCNN_MOE_RUNTIME_NEON_BIT             7
#define NCNN_MOE_RUNTIME_AVX2_BIT             8
#define NCNN_MOE_RUNTIME_AVX512_BIT           9
#define NCNN_MOE_RUNTIME_OPENMP_BIT           10
#define NCNN_MOE_RUNTIME_CROSS_SESSION_BIT    11
#define NCNN_MOE_RUNTIME_DOUBLE_BUFFER_BIT    12
#define NCNN_MOE_RUNTIME_MXFP4_VULKAN_BIT     13
#define NCNN_MOE_RUNTIME_SVE2_BIT             14
#define NCNN_MOE_RUNTIME_MULTI_VULKAN_BIT     15

// CPU ISA bit positions.
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

// Runtime capability flags.
enum RuntimeFlag : uint32_t
{
    RuntimeCpu = UINT32_C(1) << NCNN_MOE_RUNTIME_CPU_BIT,
    RuntimeNcnnLinear = UINT32_C(1) << NCNN_MOE_RUNTIME_NCNN_LINEAR_BIT,
    RuntimeVulkan = UINT32_C(1) << NCNN_MOE_RUNTIME_VULKAN_BIT,
    RuntimeVulkanCpu = UINT32_C(1) << NCNN_MOE_RUNTIME_VULKAN_CPU_BIT,
    RuntimeVulkanAttention = UINT32_C(1) << NCNN_MOE_RUNTIME_VULKAN_ATTENTION_BIT,
    RuntimeVulkanVictim = UINT32_C(1) << NCNN_MOE_RUNTIME_VULKAN_VICTIM_BIT,
    RuntimeMxfp4Cpu = UINT32_C(1) << NCNN_MOE_RUNTIME_MXFP4_CPU_BIT,
    RuntimeMxfp4Neon = UINT32_C(1) << NCNN_MOE_RUNTIME_NEON_BIT,
    RuntimeMxfp4Avx2 = UINT32_C(1) << NCNN_MOE_RUNTIME_AVX2_BIT,
    RuntimeMxfp4Avx512 = UINT32_C(1) << NCNN_MOE_RUNTIME_AVX512_BIT,
    RuntimeOpenmp = UINT32_C(1) << NCNN_MOE_RUNTIME_OPENMP_BIT,
    RuntimeCrossSession = UINT32_C(1) << NCNN_MOE_RUNTIME_CROSS_SESSION_BIT,
    RuntimeVulkanDoubleBuffer = UINT32_C(1) << NCNN_MOE_RUNTIME_DOUBLE_BUFFER_BIT,
    RuntimeMxfp4Vulkan = UINT32_C(1) << NCNN_MOE_RUNTIME_MXFP4_VULKAN_BIT,
    RuntimeMxfp4Sve2 = UINT32_C(1) << NCNN_MOE_RUNTIME_SVE2_BIT,
    RuntimeVulkanMultiDevice = UINT32_C(1) << NCNN_MOE_RUNTIME_MULTI_VULKAN_BIT
};

// CPU ISA flags detected at runtime.
enum CpuIsaFlag : uint64_t
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

struct RuntimeInfo
{
    uint64_t physical_memory_size = 0;
    uint64_t available_memory_size = 0;

    uint32_t cpu_count = 1;
    uint32_t physical_cpu_count = 1;
    uint32_t num_threads = 1;
    uint32_t cpu_linear_num_threads = 1;
    uint32_t float8_linear_num_threads = 1;
    uint32_t float8_linear_row_group_size = 1;
    uint32_t mxfp4_decode_row_pair_group_size = 1;
    uint64_t cpu_isa_flags = 0;

    uint32_t default_gpu_index = 0;
    std::vector<VulkanDeviceCapabilities> gpu_infos;

    uint32_t flags = RuntimeCpu | RuntimeMxfp4Cpu | RuntimeCrossSession;
};

class Runtime
{
private:
    std::vector<std::shared_ptr<IMoeModelAdapter>> adapters;
    RuntimeInfo runtime_info;

public:
    Runtime();

    [[nodiscard]] const RuntimeInfo& info() const noexcept
    {
        return runtime_info;
    }

    void register_adapter(std::shared_ptr<IMoeModelAdapter> adapter);

    [[nodiscard]] Result<ModelPtr> load_model(const std::filesystem::path& model_path, const Option& opt = {});

    // Synchronization point for warm-up and traffic transitions.
    [[nodiscard]] Result<void> synchronize_model_caches(const ModelPtr& model);

    [[nodiscard]] Result<SessionPtr> create_session(const ModelPtr& model, const SessionOptions& opt = {});

    [[nodiscard]] Result<BatchSchedulerPtr> create_scheduler(const SchedulerOptions& opt = {});
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RUNTIME_H
