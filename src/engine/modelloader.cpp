#include "modelloader.h"

#include "cpu_thread_budget.h"
#include "expert_backend.h"
#include "compiler/moe_ir.hpp"
#include "kernels/cpu_mxfp4.h"
#include "storage/expert_cache.h"
#include "storage/expert_victim_cache.h"
#include "graph/memory_planner.h"
#include "backends/ncnn/ncnn_linear.h"
#include "storage/system_memory.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace ncnn {
namespace moe {

static bool cpu_packed_weights_supported(
    const MoeIR& ir,
    const Option& opt) noexcept
{
    for (const LayerDescriptor& layer : ir.layers)
    {
        if (!has_flag(layer.flags, LayerDescriptorMoe))
            continue;
        const DType dtype = layer.ffn.moe.expert_weight_dtype;
        if (is_qnk_dtype(dtype))
            return true;
        if (dtype == DType::MxFp4
            && has_flag(
                opt.optimization_flags,
                OptimizationCpuMxfp4Q8)
            && mxfp4_q8_packed_kernel_available())
        {
            return true;
        }
    }
    return false;
}

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

static Result<std::vector<uint64_t>> distribute_gpu_cache_size(
    uint64_t total_size,
    uint64_t expert_pair_size,
    const std::vector<uint32_t>& device_indices,
    const std::vector<VulkanDeviceCapabilities>& gpu_infos,
    std::string_view cache_name)
{
    std::vector<uint64_t> sizes(device_indices.size(), 0);
    if (total_size == 0)
        return sizes;
    if (device_indices.empty())
    {
        return Error{ErrorCode::InvalidArgument, std::string(cache_name) + " requires at least one active Vulkan device"};
    }
    if (expert_pair_size == 0 || device_indices.size() > total_size / expert_pair_size)
    {
        return Error{ErrorCode::InvalidArgument, std::string(cache_name) + " cannot hold one Expert pair per active Vulkan device"};
    }

    uint64_t total_score = 0;
    for (uint32_t device_index : device_indices)
    {
        const uint64_t score = std::max(1u, gpu_infos[device_index].rough_score);
        if (total_score > std::numeric_limits<uint64_t>::max() - score)
            return Error{ErrorCode::InvalidArgument, "Vulkan device score sum overflows"};
        total_score += score;
    }

    const uint64_t minimum_size = expert_pair_size * device_indices.size();
    const uint64_t remaining_size = total_size - minimum_size;
    uint64_t assigned_size = 0;
    for (size_t i = 0; i < device_indices.size(); ++i)
    {
        const uint64_t score = std::max(1u, gpu_infos[device_indices[i]].rough_score);
        const uint64_t extra_size = i + 1 == device_indices.size()
                                        ? remaining_size - assigned_size
                                        : remaining_size / total_score * score + remaining_size % total_score * score / total_score;
        sizes[i] = expert_pair_size + extra_size;
        assigned_size += extra_size;
    }
    return sizes;
}

static std::vector<uint64_t> get_auto_gpu_cache_sizes(
    uint64_t expert_pair_size,
    const std::vector<uint32_t>& device_indices,
    const std::vector<VulkanDeviceCapabilities>& gpu_infos,
    uint32_t num_concurrent_sessions)
{
    const uint64_t conservative_cache_percent = 85;
    const uint64_t resource_rich_cache_percent = 92;
    const uint64_t resource_rich_heap_budget = UINT64_C(12) * 1024 * 1024 * 1024;
    const uint64_t percent_base = 100;

    std::vector<uint64_t> sizes(device_indices.size(), 0);
    if (expert_pair_size == 0 || device_indices.empty())
        return sizes;

    // Leave headroom for activations, KV, and Vulkan workspace.
    for (size_t i = 0; i < device_indices.size(); ++i)
    {
        if (device_indices[i] >= gpu_infos.size())
            return {};
        const VulkanDeviceCapabilities& gpu_info = gpu_infos[device_indices[i]];
        const bool resource_rich_single_session = num_concurrent_sessions == 1
                                                  && gpu_info.heap_budget >= resource_rich_heap_budget;
        const uint64_t cache_percent = resource_rich_single_session
                                           ? resource_rich_cache_percent
                                           : conservative_cache_percent;
        const uint64_t available_size = gpu_info.heap_available;
        const uint64_t available_whole_percent = available_size / percent_base;
        const uint64_t available_remainder = available_size % percent_base;
        const uint64_t cache_budget = available_whole_percent * cache_percent
                                      + available_remainder * cache_percent / percent_base;
        sizes[i] = cache_budget - cache_budget % expert_pair_size;
    }
    return sizes;
}

static uint64_t sum_sizes(const std::vector<uint64_t>& sizes) noexcept
{
    uint64_t total = 0;
    for (uint64_t size : sizes)
    {
        if (size > std::numeric_limits<uint64_t>::max() - total)
            return std::numeric_limits<uint64_t>::max();
        total += size;
    }
    return total;
}

ModelLoader::ModelLoader(
    const RuntimeInfo& _info,
    const std::vector<std::shared_ptr<IMoeModelAdapter>>& _adapters,
    const Option& _opt)
    : info(_info),
      adapters(_adapters),
      opt(_opt)
{
}

Result<void> ModelLoader::sanitize_option()
{
    const bool use_mmap = has_flag(opt.flags, OptionMemoryMapExperts);
    const bool use_direct_io = has_flag(opt.flags, OptionDirectExpertIo);
    const bool use_buffered_io = has_flag(opt.flags, OptionBufferedExpertIo);
    if (use_mmap && (use_direct_io || use_buffered_io))
        return Error{ErrorCode::InvalidArgument, "memory-mapped and explicit Expert I/O modes are mutually exclusive"};
    if (use_direct_io && use_buffered_io)
        return Error{ErrorCode::InvalidArgument, "direct and buffered Expert I/O are mutually exclusive"};
    if (opt.expert_gpu_victim_reuse_probe_interval == 0 || opt.expert_gpu_victim_reuse_probe_interval > 1024)
        return Error{ErrorCode::InvalidArgument, "Expert GPU victim reuse-probe interval must be between 1 and 1024"};
    if (opt.num_concurrent_sessions == 0 || opt.num_concurrent_sessions > 1024)
        return Error{ErrorCode::InvalidArgument, "num_concurrent_sessions must be between 1 and 1024"};

    return {};
}

Result<void> ModelLoader::resolve_gpu_devices()
{
    const bool support_vulkan = has_flag(info.flags, RuntimeVulkanCpu);
    if (opt.hybrid_mode == HybridMode::HybridExperts && !support_vulkan)
        return Error{ErrorCode::UnsupportedModel, "HybridExperts requires a Vulkan device and Vulkan-enabled ncnn"};

    if (!opt.vulkan_device_indices.empty())
    {
        if (opt.vulkan_device_index != automatic_vulkan_device_index && opt.vulkan_device_index != opt.vulkan_device_indices.front())
            return Error{ErrorCode::InvalidArgument, "vulkan_device_index must match the first explicit Vulkan device"};
    }
    else
    {
        uint32_t device_index = opt.vulkan_device_index;
        if (device_index == automatic_vulkan_device_index && !info.gpu_infos.empty())
            device_index = info.default_gpu_index;
        if (device_index != automatic_vulkan_device_index)
            opt.vulkan_device_indices.push_back(device_index);
    }

    std::set<uint32_t> unique_indices;
    for (uint32_t device_index : opt.vulkan_device_indices)
    {
        if (device_index >= info.gpu_infos.size())
            return Error{ErrorCode::InvalidArgument, "a Vulkan device index is out of range"};
        if (!unique_indices.insert(device_index).second)
            return Error{ErrorCode::InvalidArgument, "Vulkan device indices must be unique"};
    }

    if (!opt.vulkan_device_indices.empty())
        opt.vulkan_device_index = opt.vulkan_device_indices.front();

    if (opt.hybrid_mode == HybridMode::Auto)
    {
        const bool use_vulkan = support_vulkan && opt.vulkan_device_index != automatic_vulkan_device_index && info.gpu_infos[opt.vulkan_device_index].type != VulkanDeviceType::Cpu;
        opt.hybrid_mode = use_vulkan ? HybridMode::HybridExperts : HybridMode::CpuOnly;
    }

    if (opt.hybrid_mode != HybridMode::CpuOnly)
    {
        for (uint32_t device_index : opt.vulkan_device_indices)
        {
            if (info.gpu_infos[device_index].type == VulkanDeviceType::Cpu)
                return Error{ErrorCode::UnsupportedModel, "Vulkan CPU devices cannot participate in heterogeneous placement"};
        }
    }

    return {};
}

Result<void> ModelLoader::load_package(const std::filesystem::path& model_path)
{
    std::filesystem::path root = model_path;
    std::filesystem::path manifest_path;
    std::error_code ec;
    if (std::filesystem::is_directory(model_path, ec))
        manifest_path = model_path / "config.json";
    else
    {
        manifest_path = model_path;
        root = model_path.parent_path();
    }

    auto manifest_json = read_text_file(manifest_path);
    if (!manifest_json)
        return manifest_json.error();

    auto model_type = required_json_string(manifest_json.value(), "model_type");
    if (!model_type)
        return model_type.error();

    package = ModelPackage{root, ModelManifest{std::move(model_type).value(), std::move(manifest_json).value()}, 0};

    for (const auto& candidate : adapters)
    {
        if (candidate->can_load(package.manifest))
        {
            adapter = candidate.get();
            break;
        }
    }
    if (!adapter)
        return Error{ErrorCode::UnsupportedModel, "no adapter registered for model_type: " + package.manifest.model_type};

    auto parsed = adapter->parse_model(package);
    if (!parsed)
        return parsed.error();
    ir = std::move(parsed).value();

    auto normalized = normalize_moe_ir(ir);
    if (!normalized)
        return normalized.error();

    return {};
}

Result<void> ModelLoader::plan_memory()
{
    use_vulkan_compute = opt.hybrid_mode == HybridMode::HybridExperts;
    use_vulkan_dense_host_release = use_vulkan_compute && has_flag(opt.flags, OptionReleaseVulkanDenseHostStorage);

    const uint64_t available_size = available_memory_size();
    auto planned = plan_model_memory(
        ir,
        opt,
        info.physical_memory_size,
        use_vulkan_dense_host_release,
        available_size,
        false);
    if (!planned)
        return planned.error();
    plan = std::move(planned).value();

    if (opt.cpu_packed_weight_mode == CpuPackedWeightMode::Enabled)
    {
        if (!cpu_packed_weights_supported(ir, opt))
        {
            return Error{
                ErrorCode::UnsupportedModel,
                "CPU packed weights are unavailable for this model or CPU"};
        }

        auto packed_plan = plan_model_memory(
            ir,
            opt,
            info.physical_memory_size,
            use_vulkan_dense_host_release,
            available_size,
            true);
        if (!packed_plan)
            return packed_plan.error();
        plan = std::move(packed_plan).value();
        cpu_packed_weight_mode = CpuPackedWeightMode::Enabled;
    }

    optimization_flags = opt.optimization_flags & ~OptimizationCpuPackedWeights;
    if (cpu_packed_weight_mode == CpuPackedWeightMode::Enabled)
        optimization_flags |= OptimizationCpuPackedWeights;

    use_file_backed_experts = has_flag(plan.flags, ModelMemoryFileBackedExperts);
    const bool use_file_backed_mxfp4 = use_file_backed_experts
                                       && std::all_of(
                                           ir.layers.begin(),
                                           ir.layers.end(),
                                           [](const LayerDescriptor& layer) {
                                               return !has_flag(layer.flags, LayerDescriptorMoe)
                                                      || layer.ffn.moe.expert_weight_dtype == DType::MxFp4;
                                           });
    const bool support_gpu_cache = use_file_backed_experts
                                   && std::all_of(
                                       ir.layers.begin(),
                                       ir.layers.end(),
                                       [](const LayerDescriptor& layer) {
                                           if (!has_flag(layer.flags, LayerDescriptorMoe))
                                               return true;
                                           const MoeDescriptor& moe = layer.ffn.moe;
                                           return moe.expert_weight_dtype == DType::MxFp4
                                                  || (moe.expert_weight_dtype == DType::BFloat16
                                                      && has_flag(moe.flags, MoeDescriptorFileBackedExperts));
                                       });
    if (use_file_backed_mxfp4)
        package.flags |= ModelPackageDeferMxfp4Experts;

    if (opt.expert_gpu_cache_size > std::numeric_limits<uint64_t>::max() - opt.expert_gpu_victim_cache_size)
        return Error{ErrorCode::InvalidArgument, "the combined Expert GPU cache capacity overflows"};

    requested_gpu_cache_size = opt.expert_gpu_cache_size + opt.expert_gpu_victim_cache_size;
    if (has_flag(opt.flags, OptionDisableGpuExpertExecution) && requested_gpu_cache_size != 0)
        return Error{ErrorCode::InvalidArgument, "GPU Expert execution cannot be disabled while an Expert GPU cache is configured"};

    // Automatic GPU caching is limited to MXFP4. Large BF16 uploads cause a
    // cold cache to thrash during prefill, but explicit BF16 cache sizes remain
    // supported.
    use_auto_gpu_cache = use_vulkan_compute
                         && !has_flag(opt.flags, OptionDisableGpuExpertExecution)
                         && use_file_backed_mxfp4
                         && requested_gpu_cache_size == 0;
    if (requested_gpu_cache_size == 0)
        return {};
    if (!use_file_backed_experts)
        return Error{ErrorCode::InvalidArgument, "the Expert GPU cache requires on-demand Expert memory"};
    if (!support_gpu_cache)
        return Error{ErrorCode::UnsupportedModel, "the Expert GPU cache requires file-backed MXFP4 or BF16 Expert weights"};
    if (!use_vulkan_compute || !has_flag(info.flags, RuntimeVulkanVictim))
        return Error{ErrorCode::UnsupportedModel, "the Expert GPU cache requires Vulkan mixed execution"};
    if (requested_gpu_cache_size < plan.expert_pair_size)
        return Error{ErrorCode::InvalidArgument, "the Expert GPU cache cannot hold one Expert pair"};

    return {};
}

Result<void> ModelLoader::compile_model()
{
    auto weights = adapter->map_weights(package, ir);
    if (!weights)
        return weights.error();

    const VulkanDeviceCapabilities* gpu_info = nullptr;
    if (opt.vulkan_device_index != automatic_vulkan_device_index)
        gpu_info = &info.gpu_infos[opt.vulkan_device_index];

    const NcnnVulkanContextInstancePtr vkctx = create_ncnn_vulkan_context_instance();
    ModelCompiler::BackendCapabilities backend_caps;
    backend_caps.flags = 0;
    backend_caps.num_threads = info.num_threads;
    backend_caps.device_index = opt.vulkan_device_index;
    backend_caps.num_concurrent_sessions = opt.num_concurrent_sessions;
    backend_caps.gpu_heap_budget = gpu_info ? gpu_info->heap_budget : 0;

    // Use the smallest heap as the common multi-device budget so persistent
    // operators cannot over-admit on a weaker shard.
    for (uint32_t device_index : opt.vulkan_device_indices)
    {
        const uint64_t heap_budget = info.gpu_infos[device_index].heap_budget;
        if (heap_budget == 0)
        {
            backend_caps.gpu_heap_budget = 0;
            break;
        }
        if (backend_caps.gpu_heap_budget == 0)
            backend_caps.gpu_heap_budget = heap_budget;
        else
            backend_caps.gpu_heap_budget = std::min(
                backend_caps.gpu_heap_budget,
                heap_budget);
    }

    backend_caps.optimization_flags = optimization_flags;
    backend_caps.vkctx = vkctx;
    backend_caps.device_indices = opt.vulkan_device_indices;
    backend_caps.device_scores.reserve(opt.vulkan_device_indices.size());
    for (uint32_t device_index : opt.vulkan_device_indices)
    {
        backend_caps.device_scores.push_back(std::max(1u, info.gpu_infos[device_index].rough_score));
    }
    if (gpu_info)
        backend_caps.compute_queue_count = gpu_info->compute_queue_count;

    if (has_flag(info.flags, RuntimeCpu))
        backend_caps.flags |= ModelCompiler::BackendCpuExecution;
    if (use_vulkan_compute && has_flag(info.flags, RuntimeVulkan))
        backend_caps.flags |= ModelCompiler::BackendVulkanDense;
    if (use_vulkan_compute
        && has_flag(info.flags, RuntimeVulkanAttention)
        && has_flag(optimization_flags, OptimizationVulkanAttention))
    {
        backend_caps.flags |= ModelCompiler::BackendVulkanAttention;
    }
    if (requested_gpu_cache_size != 0
        || (use_auto_gpu_cache && use_file_backed_experts)
        || (use_vulkan_compute && has_flag(optimization_flags, OptimizationVulkanQnK)))
    {
        backend_caps.flags |= ModelCompiler::BackendVulkanExperts;
    }
    if (has_flag(info.flags, RuntimeMxfp4Cpu))
        backend_caps.flags |= ModelCompiler::BackendMxfp4CpuKernel;
    if (use_file_backed_experts)
        backend_caps.flags |= ModelCompiler::BackendFileBackedExperts;
    else
        backend_caps.flags |= ModelCompiler::BackendRetainCpuDenseCopies;
    if (use_vulkan_dense_host_release && use_file_backed_experts)
        backend_caps.flags |= ModelCompiler::BackendReleaseVulkanDenseHostStorage;

    ModelCompiler compiler;
    auto compiled = compiler.compile(
        std::move(ir),
        std::move(weights).value(),
        opt.hybrid_mode,
        backend_caps);
    if (!compiled)
        return compiled.error();

    model = std::move(compiled).value();
    model.memory_plan = plan;
    model.option_flags = opt.flags;
    model.optimization_flags = optimization_flags;
    model.num_concurrent_sessions = opt.num_concurrent_sessions;
    return {};
}

void ModelLoader::prepare_gpu_expert_cache()
{
    expert_gpu_cache_size = opt.expert_gpu_cache_size;
    expert_gpu_victim_cache_size = opt.expert_gpu_victim_cache_size;
    gpu_infos = info.gpu_infos;

#if defined(NCNN_MOE_WITH_VULKAN) && NCNN_MOE_WITH_VULKAN
    if (use_auto_gpu_cache && use_file_backed_experts)
    {
        // Dense weights have already been allocated, so use the live budget.
        const std::vector<VulkanDeviceCapabilities> live_gpu_infos = NcnnLinearOperator::gpu_infos();
        if (live_gpu_infos.size() == gpu_infos.size())
            gpu_infos = live_gpu_infos;
    }
#endif

    if (!use_auto_gpu_cache || !use_file_backed_experts)
        return;

    std::vector<uint64_t> sizes = get_auto_gpu_cache_sizes(
        plan.expert_pair_size,
        model.vulkan_device_indices,
        gpu_infos,
        opt.num_concurrent_sessions);
    const bool valid = !sizes.empty()
                       && std::all_of(
                           sizes.begin(),
                           sizes.end(),
                           [pair_size = plan.expert_pair_size](uint64_t size) {
                               return size >= pair_size;
                           });
    if (valid)
    {
        expert_gpu_cache_size = sum_sizes(sizes);
        gpu_cache_sizes = std::move(sizes);
    }
}

void ModelLoader::set_effective_option()
{
    EffectiveOption& effective_opt = model.effective_option;
    effective_opt.hybrid_mode = model.hybrid_mode;
    effective_opt.requested_expert_memory_mode = plan.requested_mode;
    effective_opt.selected_expert_memory_mode = plan.selected_mode;
    effective_opt.requested_cpu_packed_weight_mode = opt.cpu_packed_weight_mode;
    effective_opt.selected_cpu_packed_weight_mode = cpu_packed_weight_mode;
    effective_opt.host_memory_budget = plan.host_memory_budget;
    effective_opt.expert_cache_size = plan.expert_cache_size;
    effective_opt.expert_gpu_cache_size = expert_gpu_cache_size;
    effective_opt.expert_gpu_victim_cache_size = expert_gpu_victim_cache_size;
    effective_opt.expert_gpu_victim_reuse_probe_interval = opt.expert_gpu_victim_reuse_probe_interval;
    effective_opt.vulkan_device_index = model.vulkan_device_index;
    effective_opt.vulkan_device_indices = model.vulkan_device_indices;
    effective_opt.flags = opt.flags;
    effective_opt.optimization_flags = optimization_flags;
    effective_opt.num_concurrent_sessions = opt.num_concurrent_sessions;
    effective_opt.use_file_backed_experts = use_file_backed_experts;
}

uint32_t ModelLoader::resolve_expert_io_threads() const
{
    if (opt.num_expert_io_threads != 0)
    {
        return std::min(
            opt.num_expert_io_threads,
            resolve_cpu_thread_budget(opt.num_expert_io_threads).reserved_io_threads);
    }

    uint32_t max_active_experts = 1;
    for (const CompiledLayerPlan& layer : model.graph.layer_plans)
        max_active_experts = std::max(max_active_experts, layer.moe.top_k);
    for (const CompiledLayerPlan& layer : model.speculative.graph.layer_plans)
        max_active_experts = std::max(max_active_experts, layer.moe.top_k);

    const uint32_t max_io_threads = max_active_experts > std::numeric_limits<uint32_t>::max() / 2
                                        ? std::numeric_limits<uint32_t>::max()
                                        : max_active_experts * 2;
    return std::min(max_io_threads, resolve_cpu_thread_budget().reserved_io_threads);
}

uint32_t ModelLoader::expert_cache_flags() const noexcept
{
    uint32_t flags = 0;
    if (has_flag(opt.flags, OptionMemoryMapExperts))
        flags |= ExpertCacheMemoryMapRanges;
    if (has_flag(opt.flags, OptionDirectExpertIo))
        flags |= ExpertCacheDirectReads;
    if (has_flag(opt.flags, OptionBufferedExpertIo))
        flags |= ExpertCacheBufferedReads;
    if (has_flag(opt.flags, OptionForwardAwareCache))
        flags |= ExpertCacheForwardAwareEviction;
    if (has_flag(opt.flags, OptionRouterPrediction))
        flags |= ExpertCacheAllowSpeculativeEviction;
    if (has_flag(opt.flags, OptionCrossExpertReadCoalescing))
        flags |= ExpertCacheCrossExpertReadCoalescing;
    return flags;
}

Result<void> ModelLoader::resolve_gpu_cache_sizes()
{
    const std::vector<uint32_t>& device_indices = model.vulkan_device_indices;
    if (gpu_cache_sizes.empty())
    {
        auto sizes = distribute_gpu_cache_size(
            expert_gpu_cache_size,
            plan.expert_pair_size,
            device_indices,
            gpu_infos,
            "the executable Expert GPU cache");
        if (!sizes)
            return sizes.error();
        gpu_cache_sizes = std::move(sizes).value();
    }

    auto sizes = distribute_gpu_cache_size(
        expert_gpu_victim_cache_size,
        plan.expert_pair_size,
        device_indices,
        gpu_infos,
        "the Expert GPU victim cache");
    if (!sizes)
        return sizes.error();
    gpu_victim_cache_sizes = std::move(sizes).value();

    for (size_t i = 0; i < device_indices.size(); ++i)
    {
        const uint64_t cache_size = gpu_cache_sizes[i];
        const uint64_t victim_size = gpu_victim_cache_sizes[i];
        if (cache_size > std::numeric_limits<uint64_t>::max() - victim_size)
            return Error{ErrorCode::InvalidArgument, "the combined per-device Expert GPU cache capacity overflows"};

        const uint64_t heap_budget = gpu_infos[device_indices[i]].heap_budget;
        if (heap_budget == 0 || cache_size + victim_size > heap_budget)
            return Error{ErrorCode::InvalidArgument, "combined per-device Expert GPU caches exceed the Vulkan heap budget"};
    }

    return {};
}

Result<void> ModelLoader::create_expert_victim_cache()
{
    if (expert_gpu_victim_cache_size == 0)
        return {};

    const std::vector<uint32_t>& device_indices = model.vulkan_device_indices;
    victim_caches.reserve(device_indices.size());
    for (size_t i = 0; i < device_indices.size(); ++i)
    {
        auto shard = create_vulkan_victim_cache(
            gpu_victim_cache_sizes[i],
            device_indices[i],
            model.vulkan_context_instance,
            optimization_flags);
        if (!shard)
            return Error{ErrorCode::UnsupportedModel, "cannot create an Expert GPU victim-cache shard"};
        victim_caches.push_back(std::move(shard));
    }

    victim_cache = create_sharded_victim_cache(victim_caches);
    if (!victim_cache)
        return Error{ErrorCode::UnsupportedModel, "cannot create the sharded Expert GPU victim cache"};
    if (opt.expert_gpu_victim_reuse_probe_interval > 1)
    {
        victim_cache = create_reuse_victim_cache(
            std::move(victim_cache),
            opt.expert_gpu_victim_reuse_probe_interval);
    }

    return {};
}

Result<void> ModelLoader::create_expert_backend()
{
    const bool use_victim_source = !has_flag(opt.flags, OptionDisableGpuVictimExecution)
                                   && !victim_caches.empty();
    if (expert_gpu_cache_size == 0 && !use_victim_source)
        return {};

    const std::vector<uint32_t>& device_indices = model.vulkan_device_indices;
    std::vector<std::shared_ptr<IExpertExecutionBackend>> backends;
    std::vector<uint32_t> backend_device_indices;
    for (size_t i = 0; i < device_indices.size(); ++i)
    {
        auto backend = create_vulkan_mxfp4_expert_backend(
            gpu_cache_sizes[i],
            device_indices[i],
            !use_victim_source ? std::shared_ptr<IExpertVictimCache>() : victim_caches[i],
            model.vulkan_context_instance,
            optimization_flags);
        if (!backend)
            return Error{ErrorCode::UnsupportedModel, "cannot create a Vulkan Expert execution cache/source backend"};
        backends.push_back(std::move(backend));
        backend_device_indices.push_back(device_indices[i]);
    }

    if (use_victim_source)
    {
        model.expert_backend = create_key_sharded_expert_backend(
            std::move(backends),
            std::move(backend_device_indices));
    }
    else
    {
        std::vector<uint32_t> group_device_indices;
        group_device_indices.reserve(model.graph.layer_plans.size() + model.speculative.graph.layer_plans.size());
        for (const CompiledLayerPlan& layer : model.graph.layer_plans)
            group_device_indices.push_back(layer.vulkan_device_index);
        for (const CompiledLayerPlan& layer : model.speculative.graph.layer_plans)
            group_device_indices.push_back(layer.vulkan_device_index);

        model.expert_backend = create_multi_device_expert_backend(
            std::move(backends),
            std::move(backend_device_indices),
            std::move(group_device_indices));
    }

    if (!model.expert_backend)
        return Error{ErrorCode::UnsupportedModel, "cannot create the Vulkan Expert execution cache/source backend"};
    return {};
}

Result<void> ModelLoader::create_host_expert_cache(uint32_t num_io_threads, uint32_t flags)
{
    const size_t group_count = model.graph.layer_plans.size() + model.speculative.graph.layer_plans.size();
    if (group_count > std::numeric_limits<uint32_t>::max())
        return Error{ErrorCode::InvalidModel, "the total Expert residency-group count overflows"};

    model.expert_cache = std::make_shared<Mxfp4ExpertCache>(
        plan.expert_cache_size,
        num_io_threads,
        std::move(victim_cache),
        flags,
        static_cast<uint32_t>(group_count),
        cpu_packed_weight_mode == CpuPackedWeightMode::Enabled);
    return {};
}

Result<void> ModelLoader::configure_expert_cache()
{
    prepare_gpu_expert_cache();

    set_effective_option();
    if (!use_file_backed_experts)
        return {};

    const uint32_t num_io_threads = resolve_expert_io_threads();
    const uint32_t cache_flags = expert_cache_flags();
    model.effective_option.num_expert_io_threads = num_io_threads;

    auto ret = resolve_gpu_cache_sizes();
    if (!ret)
        return ret.error();
    ret = create_expert_victim_cache();
    if (!ret)
        return ret.error();
    ret = create_expert_backend();
    if (!ret)
        return ret.error();
    return create_host_expert_cache(num_io_threads, cache_flags);
}

Result<void> ModelLoader::configure_resident_qnk_backend()
{
    bool has_qnk = false;
    uint64_t max_pair_size = 0;
    uint32_t max_top_k = 1;
    const auto inspect_graph = [&](const ExecutionGraph& graph) {
        for (const CompiledLayerPlan& layer : graph.layer_plans)
        {
            max_top_k = std::max(max_top_k, layer.moe.top_k);
            for (const ExpertPlan& expert : layer.moe.experts)
            {
                if (expert.gate_up_weight == invalid_tensor_handle
                    || expert.down_weight == invalid_tensor_handle)
                {
                    continue;
                }

                const TensorData& gate_up = model.weights.at(expert.gate_up_weight);
                const TensorData& down = model.weights.at(expert.down_weight);
                if (is_qnk_dtype(gate_up.dtype) && gate_up.dtype == down.dtype)
                {
                    has_qnk = true;
                    max_pair_size = std::max(max_pair_size, expert.weight_size);
                }
            }
        }
    };
    inspect_graph(model.graph);
    inspect_graph(model.speculative.graph);

    if (!has_qnk
        || !use_vulkan_compute
        || !has_flag(optimization_flags, OptimizationVulkanQnK)
        || model.expert_backend)
    {
        return {};
    }
    if (max_pair_size == 0
        || max_pair_size > std::numeric_limits<uint64_t>::max() / max_top_k)
    {
        return Error{ErrorCode::InvalidArgument, "resident Qn_K Expert GPU capacity overflows"};
    }

    const uint64_t cache_size = max_pair_size * max_top_k;
    auto backend = create_vulkan_mxfp4_expert_backend(
        cache_size,
        opt.vulkan_device_index,
        nullptr,
        model.vulkan_context_instance,
        optimization_flags);
    if (!backend)
        return Error{ErrorCode::UnsupportedModel, "cannot create the Vulkan Qn_K Expert execution backend"};

    model.expert_backend = std::move(backend);
    model.effective_option.expert_gpu_cache_size = cache_size;
    return {};
}

Result<CompiledModel> ModelLoader::load(const std::filesystem::path& model_path)
{
    auto ret = sanitize_option();
    if (!ret)
        return ret.error();
    ret = resolve_gpu_devices();
    if (!ret)
        return ret.error();

    ret = load_package(model_path);
    if (!ret)
        return ret.error();

    ret = plan_memory();
    if (!ret)
        return ret.error();

    ret = compile_model();
    if (!ret)
        return ret.error();

    ret = configure_expert_cache();
    if (!ret)
        return ret.error();
    ret = configure_resident_qnk_backend();
    if (!ret)
        return ret.error();

    return std::move(model);
}

} // namespace moe
} // namespace ncnn
