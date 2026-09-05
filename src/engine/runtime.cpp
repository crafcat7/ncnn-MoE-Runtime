#include "ncnn/moe/runtime.h"

#include "graph/compiledmodel.h"
#include "cpu.h"
#include "expertbackend.h"
#include "modelloader.h"
#include "models/modeladapter_builtin.h"
#include "models/modeladapter_deepseekv4.h"
#include "models/modeladapter_qwen3_5.h"
#include "models/modeladapter_qwen4exp.h"
#include "kernels/mxfp4.h"
#include "kernels/float8.h"
#include "kernels/ops.h"
#include "storage/expertcache.h"
#include "backends/ncnn/vulkancontext.h"

#include <algorithm>
#include <thread>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

Model::Model(std::shared_ptr<const CompiledModel> _compiled)
    : compiled(std::move(_compiled))
{
}

const MoeModelDescriptor& Model::descriptor() const noexcept
{
    return compiled->descriptor;
}

HybridMode Model::hybrid_mode() const noexcept
{
    return compiled->opt.hybrid_mode;
}

uint32_t Model::vulkan_device_index() const noexcept
{
    return compiled->opt.vulkan_device_index;
}

const std::vector<uint32_t>& Model::vulkan_device_indices() const noexcept
{
    return compiled->opt.vulkan_device_indices;
}

const CompiledModel& model_compiled(const Model& model) noexcept
{
    return *model.compiled;
}

Runtime::Runtime()
{
    runtime_info.physical_memory_size = physical_memory_size();
    runtime_info.available_memory_size = available_memory_size();
    runtime_info.cpu_count = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t physical_cpu_count = get_physical_cpu_count();
    runtime_info.physical_cpu_count = physical_cpu_count == 0 ? runtime_info.cpu_count : physical_cpu_count;
    runtime_info.cpu_isa_flags = cpu_isa_flags();
    runtime_info.cpu_linear_num_threads = cpu_linear_num_threads();
    runtime_info.float8_linear_row_group_size = float8_linear_row_group_size(OptimizationDefaultFlags);
    runtime_info.mxfp4_decode_row_pair_group_size = mxfp4_decode_row_pair_group_size();
    const MxFp4KernelKind kernel = mxfp4_kernel_kind();
    if (kernel == MxFp4KernelKind::ArmNeon)
        runtime_info.flags |= RuntimeMxfp4Neon;
    else if (kernel == MxFp4KernelKind::ArmSve2)
        runtime_info.flags |= RuntimeMxfp4Sve2;
    if (kernel == MxFp4KernelKind::X86Avx2)
        runtime_info.flags |= RuntimeMxfp4Avx2;
    if (kernel == MxFp4KernelKind::X86Avx512)
        runtime_info.flags |= RuntimeMxfp4Avx512;
#if defined(_OPENMP)
    runtime_info.flags |= RuntimeOpenmp;
    runtime_info.num_threads = static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
#endif
#if defined(NCNN_MOE_USE_NCNN) && NCNN_MOE_USE_NCNN
    runtime_info.flags |= RuntimeNcnnLinear;
#endif
#if defined(NCNN_MOE_WITH_VULKAN) && NCNN_MOE_WITH_VULKAN
    runtime_info.gpu_infos = get_gpu_infos();
    if (!runtime_info.gpu_infos.empty())
    {
        runtime_info.flags |= RuntimeVulkan | RuntimeVulkanCpu | RuntimeVulkanAttention | RuntimeVulkanVictim
                              | RuntimeVulkanDoubleBuffer | RuntimeMxfp4Vulkan;
        const uint32_t default_gpu_index = get_default_gpu_index();
        if (default_gpu_index < runtime_info.gpu_infos.size())
            runtime_info.default_gpu_index = default_gpu_index;
    }
    if (runtime_info.gpu_infos.size() > 1)
    {
        runtime_info.flags |= RuntimeVulkanMultiDevice;
    }
#endif
    register_adapter(std::make_shared<BuiltinModelAdapter>());
    register_adapter(std::make_shared<DeepSeekV4ModelAdapter>());
    register_adapter(std::make_shared<Qwen3_5MoeModelAdapter>());
    register_adapter(std::make_shared<Qwen4ExpModelAdapter>());
}

void Runtime::register_adapter(std::shared_ptr<ModelAdapter> adapter)
{
    if (adapter)
        adapters.push_back(std::move(adapter));
}

Result<ModelPtr> Runtime::load_model(const std::filesystem::path& model_path, const Option& opt)
{
    ModelLoader loader(runtime_info, adapters, opt);
    auto compiled = loader.load(model_path);
    if (!compiled)
        return compiled.error();

    auto compiled_model = std::make_shared<const CompiledModel>(std::move(compiled).value());
    return ModelPtr(new Model(std::move(compiled_model)));
}

Result<void> Runtime::synchronize_model_caches(const ModelPtr& model)
{
    if (!model)
    {
        return Error{ErrorCode::InvalidArgument, "model cannot be null"};
    }
    const CompiledModel& compiled = model_compiled(*model);
    if (compiled.expert_cache)
        compiled.expert_cache->wait_for_background_work();
    if (compiled.expert_backend)
        compiled.expert_backend->wait_for_background_work();
    return {};
}

Result<SessionPtr> Runtime::create_session(const ModelPtr& model, const SessionOptions& opt)
{
    if (!model)
        return Error{ErrorCode::InvalidArgument, "model cannot be null"};
    return SessionPtr(new Session(model, opt));
}

Result<BatchSchedulerPtr> Runtime::create_scheduler(const SchedulerOptions& opt)
{
    return BatchSchedulerPtr(new BatchScheduler(opt));
}

} // namespace moe
} // namespace ncnn
