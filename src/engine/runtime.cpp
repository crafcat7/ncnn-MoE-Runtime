#include "ncnn/moe/runtime.h"

#include "ncnn/moe/execution_plan.h"
#include "models/builtin_model_adapter.h"
#include "kernels/cpu_mxfp4.h"
#include "storage/expert_cache.h"
#include "storage/expert_victim_cache.h"
#include "graph/memory_planner.h"
#include "backends/ncnn/ncnn_linear.h"
#include "storage/system_memory.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <utility>

namespace ncnn {
namespace moe {

static Result<std::string> read_text_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return Error{ErrorCode::IoError, "cannot open manifest: " + path.string()};

    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof())
        return Error{ErrorCode::IoError, "cannot read manifest: " + path.string()};
    return contents.str();
}

static Result<std::string> required_json_string(const std::string& json, const std::string& key)
{
    const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        return Error{ErrorCode::InvalidModel, "manifest is missing string field: " + key};
    return match[1].str();
}

Runtime::Runtime()
{
    capabilities_.physical_memory_bytes = physical_memory_bytes();
    capabilities_.mxfp4_kernel = mxfp4_kernel_name();
    const MxFp4KernelKind kernel = mxfp4_kernel_kind();
    if (kernel == MxFp4KernelKind::ArmNeon)
        capabilities_.flags |= RuntimeCapabilityMxfp4ArmNeon;
    if (kernel == MxFp4KernelKind::X86Avx2)
        capabilities_.flags |= RuntimeCapabilityMxfp4X86Avx2;
    if (kernel == MxFp4KernelKind::X86Avx512)
        capabilities_.flags |= RuntimeCapabilityMxfp4X86Avx512;
#if defined(_OPENMP)
    capabilities_.flags |= RuntimeCapabilityOpenmpExpertParallelism;
#endif
#if defined(NCNN_MOE_USE_NCNN) && NCNN_MOE_USE_NCNN
    capabilities_.flags |= RuntimeCapabilityNcnnCpuLinear;
#endif
#if defined(NCNN_MOE_WITH_VULKAN) && NCNN_MOE_WITH_VULKAN
    capabilities_.vulkan_device_count = NcnnLinearOperator::vulkan_device_count();
    if (capabilities_.vulkan_device_count > 0) {
        capabilities_.flags |= RuntimeCapabilityVulkanExecution
                               | RuntimeCapabilityVulkanCpuMix
                               | RuntimeCapabilityVulkanCpuPrefetch
                               | RuntimeCapabilityVulkanAttention
                               | RuntimeCapabilityVulkanExpertVictimCache;
    }
    capabilities_.vulkan_heap_budget_bytes
        = NcnnLinearOperator::vulkan_heap_budget_bytes();
#endif
    register_adapter(std::make_shared<BuiltinModelAdapter>());
}

void Runtime::register_adapter(std::shared_ptr<IMoeModelAdapter> adapter)
{
    if (adapter)
        adapters_.push_back(std::move(adapter));
}

Result<ModelPtr> Runtime::load_model(
    const std::filesystem::path& model_path,
    const RuntimeOptions& options)
{
    HybridMode resolved_mode = options.hybrid_mode;
    if (resolved_mode == HybridMode::Auto)
        resolved_mode = has_flag(
                            capabilities_.flags,
                            RuntimeCapabilityVulkanCpuMix)
                            ? HybridMode::HybridExperts
                            : HybridMode::CpuOnly;
    if (resolved_mode == HybridMode::VulkanOnly)
        return Error{ErrorCode::UnsupportedModel, "Vulkan-only execution is not implemented; use HybridExperts"};
    if (resolved_mode == HybridMode::HybridExperts
        && !has_flag(
            capabilities_.flags,
            RuntimeCapabilityVulkanCpuMix)) {
        return Error{ErrorCode::UnsupportedModel, "HybridExperts requires a Vulkan device and Vulkan-enabled ncnn"};
    }
    if (resolved_mode == HybridMode::VulkanWithCpuPrefetch
        && !has_flag(
            capabilities_.flags,
            RuntimeCapabilityVulkanCpuPrefetch)) {
        return Error{ErrorCode::UnsupportedModel, "VulkanWithCpuPrefetch requires a Vulkan device and Vulkan-enabled ncnn"};
    }

    std::filesystem::path root = model_path;
    std::filesystem::path manifest_path;
    std::error_code filesystem_error;
    if (std::filesystem::is_directory(model_path, filesystem_error))
        manifest_path = model_path / "config.json";
    else {
        manifest_path = model_path;
        root = model_path.parent_path();
    }

    auto manifest_text = read_text_file(manifest_path);
    if (!manifest_text)
        return manifest_text.error();

    auto model_type = required_json_string(manifest_text.value(), "model_type");
    if (!model_type)
        return model_type.error();

    ModelPackage package{
        root,
        ModelManifest{std::move(model_type).value(), std::move(manifest_text).value()},
        0};

    const IMoeModelAdapter* selected_adapter = nullptr;
    for (const auto& adapter : adapters_) {
        if (adapter->can_load(package.manifest)) {
            selected_adapter = adapter.get();
            break;
        }
    }
    if (!selected_adapter)
        return Error{ErrorCode::UnsupportedModel, "no adapter registered for model_type: " + package.manifest.model_type};

    auto parsed_ir = selected_adapter->parse_model(package);
    if (!parsed_ir)
        return parsed_ir.error();

    auto memory_plan = plan_model_memory(
        parsed_ir.value(),
        options,
        capabilities_.physical_memory_bytes);
    if (!memory_plan)
        return memory_plan.error();
    ModelMemoryPlan plan = std::move(memory_plan).value();
    const bool file_backed_experts
        = has_flag(plan.flags, ModelMemoryFileBackedExperts);
    if (file_backed_experts)
        package.flags |= ModelPackageDeferMxfp4Experts;

    const bool use_vulkan_dense = resolved_mode == HybridMode::HybridExperts
                                  || resolved_mode == HybridMode::VulkanWithCpuPrefetch;
    if (options.expert_gpu_cache_bytes != 0) {
        if (!file_backed_experts) {
            return Error{
                ErrorCode::InvalidArgument,
                "the Expert GPU cache requires on-demand Expert memory"};
        }
        if (!use_vulkan_dense
            || !has_flag(
                capabilities_.flags,
                RuntimeCapabilityVulkanExpertVictimCache)) {
            return Error{
                ErrorCode::UnsupportedModel,
                "the Expert GPU cache requires Vulkan mixed execution"};
        }
        if (options.expert_gpu_cache_bytes
            < plan.expert_pair_bytes) {
            return Error{
                ErrorCode::InvalidArgument,
                "the Expert GPU cache cannot hold one Expert pair"};
        }
        const uint64_t maximum_safe_capacity
            = capabilities_.vulkan_heap_budget_bytes / 4;
        if (maximum_safe_capacity == 0
            || options.expert_gpu_cache_bytes
                   > maximum_safe_capacity) {
            return Error{
                ErrorCode::InvalidArgument,
                "the Expert GPU cache exceeds one quarter of the Vulkan heap budget"};
        }
    }

    auto weights = selected_adapter->map_weights(package, parsed_ir.value());
    if (!weights)
        return weights.error();

    ModelCompiler compiler;
    ModelCompiler::BackendCapabilities compiler_capabilities;
    compiler_capabilities.flags = 0;
    if (has_flag(capabilities_.flags, RuntimeCapabilityCpuExecution))
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityCpuExecution;
    if (use_vulkan_dense
        && has_flag(capabilities_.flags, RuntimeCapabilityVulkanExecution)) {
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityVulkanDense;
    }
    if (use_vulkan_dense
        && has_flag(capabilities_.flags, RuntimeCapabilityVulkanAttention)) {
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityVulkanAttention;
    }
    if (has_flag(capabilities_.flags, RuntimeCapabilityMxfp4CpuKernel))
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityMxfp4CpuKernel;
    if (!file_backed_experts)
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityRetainCpuDenseCopies;
    auto compiled = compiler.compile(
        std::move(parsed_ir).value(),
        std::move(weights).value(),
        resolved_mode,
        compiler_capabilities);
    if (!compiled)
        return compiled.error();
    CompiledModel compiled_model = std::move(compiled).value();
    compiled_model.memory_plan = plan;
    if (file_backed_experts) {
        std::shared_ptr<IExpertVictimCache> victim_cache;
        uint32_t expert_cache_flags = 0;
        if (has_flag(
                options.flags,
                RuntimeOptionMemoryMapExperts)) {
            expert_cache_flags |= ExpertCacheMemoryMapRanges;
        }
        if (options.expert_gpu_cache_bytes != 0) {
            victim_cache = create_vulkan_expert_victim_cache(
                options.expert_gpu_cache_bytes);
            if (!victim_cache) {
                return Error{
                    ErrorCode::UnsupportedModel,
                    "cannot create the Vulkan Expert victim cache"};
            }
        }
        compiled_model.expert_cache
            = std::make_shared<Mxfp4ExpertCache>(
                plan.expert_cache_bytes,
                options.expert_io_workers,
                std::move(victim_cache),
                expert_cache_flags);
    }

    auto immutable = std::make_shared<const CompiledModel>(
        std::move(compiled_model));
    return ModelPtr(new Model(std::move(immutable)));
}

Result<SessionPtr> Runtime::create_session(const ModelPtr& model, const SessionOptions& options)
{
    if (!model)
        return Error{ErrorCode::InvalidArgument, "model cannot be null"};
    if (options.logits_output_mode != LogitsOutputMode::FullLogits)
        return Error{ErrorCode::UnsupportedModel, "the current executor supports full logits output only"};
    return SessionPtr(new Session(model, options));
}

Result<BatchSchedulerPtr> Runtime::create_scheduler(const SchedulerOptions& options)
{
    if (options.worker_count > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler worker_count exceeds 1024"};
    if (options.expert_threads_per_worker > 1024)
        return Error{
            ErrorCode::InvalidArgument,
            "scheduler expert_threads_per_worker exceeds 1024"};
    if (options.worker_cpu_sets.size() > 1024)
        return Error{
            ErrorCode::InvalidArgument,
            "scheduler worker_cpu_sets exceeds 1024 entries"};
    if (!options.worker_cpu_sets.empty()
        && options.worker_count != 0
        && options.worker_count != options.worker_cpu_sets.size()) {
        return Error{
            ErrorCode::InvalidArgument,
            "scheduler worker_count must match worker_cpu_sets"};
    }
    for (const std::vector<uint32_t>& cpu_set : options.worker_cpu_sets) {
        if (cpu_set.empty())
            return Error{
                ErrorCode::InvalidArgument,
                "scheduler worker_cpu_sets entries cannot be empty"};
    }
#if !defined(__linux__)
    if (!options.worker_cpu_sets.empty())
        return Error{
            ErrorCode::UnsupportedModel,
            "scheduler worker_cpu_sets are supported only on Linux"};
#endif
    return std::make_shared<BatchScheduler>(options);
}

} // namespace moe
} // namespace ncnn
