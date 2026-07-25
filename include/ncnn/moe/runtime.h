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

enum RuntimeOptionFlag : uint32_t
{
    RuntimeOptionMemoryMapExperts = 1u << 0
};

struct RuntimeOptions
{
    HybridMode hybrid_mode = HybridMode::Auto;
    ExpertMemoryMode expert_memory_mode = ExpertMemoryMode::Auto;
    uint64_t host_memory_budget_bytes = 0;
    uint64_t expert_cache_bytes = 0;
    uint64_t expert_gpu_cache_bytes = 0;
    uint32_t expert_io_workers = 0;
    uint32_t flags = 0;
};

enum RuntimeCapabilityFlag : uint32_t
{
    RuntimeCapabilityCpuExecution = 1u << 0,
    RuntimeCapabilityNcnnCpuLinear = 1u << 1,
    RuntimeCapabilityVulkanExecution = 1u << 2,
    RuntimeCapabilityVulkanCpuMix = 1u << 3,
    RuntimeCapabilityVulkanCpuPrefetch = 1u << 4,
    RuntimeCapabilityVulkanAttention = 1u << 5,
    RuntimeCapabilityVulkanExpertVictimCache = 1u << 6,
    RuntimeCapabilityMxfp4CpuKernel = 1u << 7,
    RuntimeCapabilityMxfp4ArmNeon = 1u << 8,
    RuntimeCapabilityMxfp4X86Avx2 = 1u << 9,
    RuntimeCapabilityMxfp4X86Avx512 = 1u << 10,
    RuntimeCapabilityOpenmpExpertParallelism = 1u << 11,
    RuntimeCapabilityCrossSessionScheduling = 1u << 12
};

struct RuntimeCapabilities
{
    uint64_t physical_memory_bytes = 0;
    uint64_t vulkan_heap_budget_bytes = 0;
    uint32_t vulkan_device_count = 0;
    uint32_t flags = RuntimeCapabilityCpuExecution
                     | RuntimeCapabilityMxfp4CpuKernel
                     | RuntimeCapabilityCrossSessionScheduling;
    std::string mxfp4_kernel;
};

class Runtime
{
public:
    Runtime();

    [[nodiscard]] const RuntimeCapabilities& capabilities() const noexcept
    {
        return capabilities_;
    }

    void register_adapter(std::shared_ptr<IMoeModelAdapter> adapter);

    [[nodiscard]] Result<ModelPtr> load_model(
        const std::filesystem::path& model_path,
        const RuntimeOptions& options = {});

    [[nodiscard]] Result<SessionPtr> create_session(
        const ModelPtr& model,
        const SessionOptions& options = {});

    [[nodiscard]] Result<BatchSchedulerPtr> create_scheduler(
        const SchedulerOptions& options = {});

private:
    std::vector<std::shared_ptr<IMoeModelAdapter> > adapters_;
    RuntimeCapabilities capabilities_;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_RUNTIME_H
