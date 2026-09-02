#ifndef NCNN_MOE_MODELLOADER_H
#define NCNN_MOE_MODELLOADER_H

#include "ncnn/moe/execution_plan.h"
#include "ncnn/moe/model_adapter.h"
#include "ncnn/moe/runtime.h"

namespace ncnn {
namespace moe {

class IExpertVictimCache;

class ModelLoader
{
public:
    ModelLoader(
        const RuntimeInfo& _info,
        const std::vector<std::shared_ptr<IMoeModelAdapter>>& _adapters,
        const Option& _opt);

    [[nodiscard]] Result<CompiledModel> load(const std::filesystem::path& model_path);

private:
    [[nodiscard]] Result<void> sanitize_option();
    [[nodiscard]] Result<void> resolve_gpu_devices();
    [[nodiscard]] Result<void> load_package(const std::filesystem::path& model_path);
    [[nodiscard]] Result<void> plan_memory();
    [[nodiscard]] Result<void> compile_model();
    [[nodiscard]] Result<void> configure_expert_cache();
    void prepare_gpu_expert_cache();
    [[nodiscard]] Result<void> resolve_gpu_cache_sizes();
    [[nodiscard]] Result<void> create_expert_victim_cache();
    [[nodiscard]] Result<void> create_expert_backend();
    [[nodiscard]] Result<void> create_host_expert_cache(uint32_t num_io_threads, uint32_t flags);
    [[nodiscard]] Result<void> configure_resident_qnk_backend();

    void set_effective_option();
    [[nodiscard]] uint32_t resolve_expert_io_threads() const;
    [[nodiscard]] uint32_t expert_cache_flags() const noexcept;

    const RuntimeInfo& info;
    const std::vector<std::shared_ptr<IMoeModelAdapter>>& adapters;
    Option opt;
    ModelPackage package;
    const IMoeModelAdapter* adapter = nullptr;
    MoeIR ir;
    ModelMemoryPlan plan;
    CompiledModel model;

    CpuPackedWeightMode cpu_packed_weight_mode = CpuPackedWeightMode::Disabled;
    uint64_t optimization_flags = OptimizationDefaultFlags;
    uint64_t requested_gpu_cache_size = 0;
    bool use_vulkan_compute = false;
    bool use_vulkan_dense_host_release = false;
    bool use_file_backed_experts = false;
    bool use_auto_gpu_cache = false;

    uint64_t expert_gpu_cache_size = 0;
    uint64_t expert_gpu_victim_cache_size = 0;
    std::vector<VulkanDeviceCapabilities> gpu_infos;
    std::vector<uint64_t> gpu_cache_sizes;
    std::vector<uint64_t> gpu_victim_cache_sizes;
    std::vector<std::shared_ptr<IExpertVictimCache>> victim_caches;
    std::shared_ptr<IExpertVictimCache> victim_cache;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELLOADER_H
