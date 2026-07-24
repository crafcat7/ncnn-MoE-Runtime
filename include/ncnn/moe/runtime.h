#ifndef NCNN_MOE_RUNTIME_H
#define NCNN_MOE_RUNTIME_H

#include "ncnn/moe/model.h"
#include "ncnn/moe/model_adapter.h"
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

struct RuntimeOptions
{
    HybridMode hybrid_mode = HybridMode::Auto;
};

struct RuntimeCapabilities
{
    bool cpu_execution = true;
    bool ncnn_cpu_linear = false;
    bool vulkan_execution = false;
    bool vulkan_cpu_mix = false;
    bool vulkan_cpu_prefetch = false;
    bool vulkan_attention = false;
    bool mxfp4_cpu_kernel = true;
    bool mxfp4_arm_neon = false;
    bool mxfp4_x86_avx2 = false;
    bool mxfp4_x86_avx512 = false;
    bool openmp_expert_parallelism = false;
    bool cross_session_scheduling = true;
    uint32_t vulkan_device_count = 0;
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
