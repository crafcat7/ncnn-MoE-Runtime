#ifndef NCNN_MOE_MODELLOADER_H
#define NCNN_MOE_MODELLOADER_H

#include "graph/compiledmodel.h"
#include "ncnn/moe/modeladapter.h"
#include "ncnn/moe/runtime.h"

namespace ncnn {
namespace moe {

class ExpertVictimCache;

class ModelLoader
{
public:
    ModelLoader(
        const RuntimeInfo& _info,
        const std::vector<std::shared_ptr<ModelAdapter>>& _adapters,
        const Option& _opt);

    [[nodiscard]] Result<CompiledModel> load(const std::filesystem::path& model_path);

private:
    [[nodiscard]] Result<void> sanitize_option();
    [[nodiscard]] Result<void> resolve_gpu_devices();
    [[nodiscard]] Result<void> load_package(const std::filesystem::path& model_path);
    [[nodiscard]] Result<void> plan_memory();
    [[nodiscard]] Result<void> compile_model();
    [[nodiscard]] Result<void> configure_expert_cache();
    [[nodiscard]] Result<void> resolve_gpu_cache_sizes(
        std::vector<uint64_t>& gpu_cache_sizes,
        std::vector<uint64_t>& gpu_victim_cache_sizes);
    [[nodiscard]] Result<std::shared_ptr<ExpertVictimCache>> create_expert_victim_cache(
        const std::vector<uint64_t>& gpu_victim_cache_sizes,
        std::vector<std::shared_ptr<ExpertVictimCache>>& victim_caches);
    [[nodiscard]] Result<void> create_expert_backend(
        const std::vector<uint64_t>& gpu_cache_sizes,
        const std::vector<std::shared_ptr<ExpertVictimCache>>& victim_caches);
    [[nodiscard]] Result<void> configure_resident_qnk_backend();

    void set_effective_option();
    [[nodiscard]] uint32_t resolve_expert_io_threads() const;
    [[nodiscard]] uint32_t expert_cache_flags() const noexcept;

    const RuntimeInfo& info;
    const std::vector<std::shared_ptr<ModelAdapter>>& adapters;
    Option opt;
    ModelPackage package;
    const ModelAdapter* adapter = nullptr;
    MoeModelDescriptor descriptor;
    ModelMemoryPlan plan;
    CompiledModel model;

    uint64_t requested_gpu_cache_size = 0;
    bool use_auto_gpu_cache = false;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MODELLOADER_H
