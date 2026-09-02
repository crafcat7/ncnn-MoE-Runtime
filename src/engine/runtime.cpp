#include "ncnn/moe/runtime.h"

#include "ncnn/moe/execution_plan.h"
#include "cpu_features.h"
#include "cpu_topology.h"
#include "expert_backend.h"
#include "modelloader.h"
#include "models/builtin_model_adapter.h"
#include "models/deepseek_v4_model_adapter.h"
#include "models/qwen3_5_moe_model_adapter.h"
#include "models/qwen4_exp_model_adapter.h"
#include "kernels/cpu_mxfp4.h"
#include "kernels/cpu_float8.h"
#include "kernels/cpu_ops.h"
#include "storage/expert_cache.h"
#include "backends/ncnn/ncnn_linear.h"
#include "storage/system_memory.h"

#include <algorithm>
#include <thread>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

Runtime::Runtime()
{
    runtime_info.physical_memory_size = physical_memory_size();
    runtime_info.available_memory_size = available_memory_size();
    runtime_info.cpu_count = std::max(1u, std::thread::hardware_concurrency());
    const CpuTopology topology = discover_cpu_topology();
    runtime_info.physical_cpu_count = topology.physical_cpu_count == 0 ? runtime_info.cpu_count : topology.physical_cpu_count;
    runtime_info.cpu_isa_flags = cpu_isa_flags();
    runtime_info.cpu_linear_num_threads = cpu_linear_num_threads();
    runtime_info.float8_linear_num_threads = float8_linear_num_threads();
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
    runtime_info.gpu_infos = NcnnLinearOperator::gpu_infos();
    if (!runtime_info.gpu_infos.empty())
    {
        runtime_info.flags |= RuntimeVulkan | RuntimeVulkanCpu | RuntimeVulkanAttention | RuntimeVulkanVictim
                              | RuntimeVulkanDoubleBuffer | RuntimeMxfp4Vulkan;
        const uint32_t default_gpu_index = NcnnLinearOperator::default_gpu_index();
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

void Runtime::register_adapter(std::shared_ptr<IMoeModelAdapter> adapter)
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
    if (model->compiled->expert_cache)
        model->compiled->expert_cache->wait_for_background_work();
    if (model->compiled->expert_backend)
        model->compiled->expert_backend->wait_for_background_work();
    return {};
}

Result<SessionPtr> Runtime::create_session(const ModelPtr& model, const SessionOptions& opt)
{
    if (!model)
        return Error{ErrorCode::InvalidArgument, "model cannot be null"};
    if (opt.logits_output_mode != LogitsOutputMode::FullLogits)
        return Error{ErrorCode::UnsupportedModel, "the current executor supports full logits output only"};
    return SessionPtr(new Session(model, opt));
}

Result<BatchSchedulerPtr> Runtime::create_scheduler(const SchedulerOptions& opt)
{
    if (has_flag(opt.flags, SchedulerOptionDisableStagedBatching) && has_flag(opt.flags, SchedulerOptionForceStagedBatching))
    {
        return Error{ErrorCode::InvalidArgument, "scheduler staged batching cannot be both disabled and forced"};
    }
    if (opt.worker_count > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler worker_count exceeds 1024"};
    if (opt.expert_threads_per_worker > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler expert_threads_per_worker exceeds 1024"};
    if (opt.reserved_io_threads > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler reserved_io_threads exceeds 1024"};
    if (opt.reserved_service_threads > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler reserved_service_threads exceeds 1024"};
    if (opt.adaptive_probe_interval > 1000000)
    {
        return Error{ErrorCode::InvalidArgument, "scheduler adaptive_probe_interval exceeds 1000000"};
    }
    if (opt.cross_call_window_microseconds > 1000000)
    {
        return Error{ErrorCode::InvalidArgument, "scheduler cross_call_window_microseconds exceeds 1000000"};
    }
    if (opt.cross_call_max_batch_size > 1024)
    {
        return Error{ErrorCode::InvalidArgument, "scheduler cross_call_max_batch_size exceeds 1024"};
    }
    if (opt.worker_cpu_sets.size() > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler worker_cpu_sets exceeds 1024 entries"};
    if (!opt.worker_cpu_sets.empty() && opt.worker_count != 0 && opt.worker_count != opt.worker_cpu_sets.size())
    {
        return Error{ErrorCode::InvalidArgument, "scheduler worker_count must match worker_cpu_sets"};
    }
    for (const std::vector<uint32_t>& cpu_set : opt.worker_cpu_sets)
    {
        if (cpu_set.empty())
            return Error{ErrorCode::InvalidArgument, "scheduler worker_cpu_sets entries cannot be empty"};
    }
#if !defined(__linux__)
    if (!opt.worker_cpu_sets.empty())
        return Error{ErrorCode::UnsupportedModel, "scheduler worker_cpu_sets are supported only on Linux"};
#endif
    return std::make_shared<BatchScheduler>(opt);
}

} // namespace moe
} // namespace ncnn
