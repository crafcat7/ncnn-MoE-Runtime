#include "ncnn/moe/runtime.h"

#include "ncnn/moe/execution_plan.h"
#include "cpu_features.h"
#include "cpu_thread_budget.h"
#include "cpu_topology.h"
#include "expert_backend.h"
#include "compiler/moe_ir.hpp"
#include "models/builtin_model_adapter.h"
#include "models/deepseek_v4_model_adapter.h"
#include "models/qwen3_5_moe_model_adapter.h"
#include "kernels/cpu_bfloat16.h"
#include "kernels/cpu_mxfp4.h"
#include "kernels/cpu_float8.h"
#include "kernels/cpu_ops.h"
#include "storage/expert_cache.h"
#include "storage/expert_victim_cache.h"
#include "graph/memory_planner.h"
#include "backends/ncnn/ncnn_linear.h"
#include "storage/system_memory.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ncnn {
namespace moe {

static bool cpu_packed_weights_supported(
    const MoeIR& ir,
    const RuntimeConfig& config) noexcept
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
                config.optimization_flags,
                RuntimeOptimizationCpuMxfp4Q8)
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

static Result<std::vector<uint64_t>> distribute_gpu_capacity(uint64_t total_capacity, uint64_t expert_pair_bytes, const std::vector<uint32_t>& device_indices, const std::vector<VulkanDeviceCapabilities>& devices,
                                                             std::string_view cache_name)
{
    std::vector<uint64_t> capacities(device_indices.size(), 0);
    if (total_capacity == 0)
        return capacities;
    if (device_indices.empty())
    {
        return Error{ErrorCode::InvalidArgument, std::string(cache_name) + " requires at least one active Vulkan device"};
    }
    if (expert_pair_bytes == 0 || device_indices.size() > total_capacity / expert_pair_bytes)
    {
        return Error{ErrorCode::InvalidArgument, std::string(cache_name) + " cannot hold one Expert pair per active Vulkan device"};
    }

    uint64_t total_score = 0;
    for (uint32_t device_index : device_indices)
    {
        const uint64_t score = std::max(1u, devices[device_index].rough_score);
        if (total_score > std::numeric_limits<uint64_t>::max() - score)
            return Error{ErrorCode::InvalidArgument, "Vulkan device score sum overflows"};
        total_score += score;
    }

    const uint64_t minimum_capacity = expert_pair_bytes * device_indices.size();
    const uint64_t distributable_capacity = total_capacity - minimum_capacity;
    uint64_t assigned_extra_capacity = 0;
    for (size_t index = 0; index < device_indices.size(); ++index)
    {
        const uint64_t score = std::max(1u, devices[device_indices[index]].rough_score);
        const uint64_t extra_capacity = index + 1 == device_indices.size()
                                            ? distributable_capacity - assigned_extra_capacity
                                            : distributable_capacity / total_score * score + distributable_capacity % total_score * score / total_score;
        capacities[index] = expert_pair_bytes + extra_capacity;
        assigned_extra_capacity += extra_capacity;
    }
    return capacities;
}

static std::vector<uint64_t> automatic_gpu_expert_capacities(
    uint64_t expert_pair_bytes,
    const std::vector<uint32_t>& device_indices,
    const std::vector<VulkanDeviceCapabilities>& devices,
    uint32_t expected_concurrency)
{
    constexpr uint64_t kConservativeExpertCachePercent = 85;
    constexpr uint64_t kResourceRichSingleSessionCachePercent = 92;
    constexpr uint64_t kResourceRichHeapBudgetBytes = UINT64_C(12) * 1024 * 1024 * 1024;
    constexpr uint64_t kPercentageBase = 100;

    std::vector<uint64_t> capacities(device_indices.size(), 0);
    if (expert_pair_bytes == 0 || device_indices.empty())
        return capacities;

    // Leave headroom for activations, KV, and Vulkan workspace.
    for (size_t index = 0; index < device_indices.size(); ++index)
    {
        if (device_indices[index] >= devices.size())
            return {};
        const VulkanDeviceCapabilities& device = devices[device_indices[index]];
        const bool resource_rich_single_session = expected_concurrency == 1
                                                  && device.heap_budget_bytes >= kResourceRichHeapBudgetBytes;
        const uint64_t cache_percent = resource_rich_single_session
                                           ? kResourceRichSingleSessionCachePercent
                                           : kConservativeExpertCachePercent;
        const uint64_t available = device.heap_available_bytes;
        const uint64_t available_whole_percent = available / kPercentageBase;
        const uint64_t available_remainder = available % kPercentageBase;
        const uint64_t cache_budget = available_whole_percent * cache_percent
                                      + available_remainder * cache_percent / kPercentageBase;
        capacities[index] = cache_budget - cache_budget % expert_pair_bytes;
    }
    return capacities;
}

static uint64_t sum_capacities(const std::vector<uint64_t>& capacities) noexcept
{
    uint64_t total = 0;
    for (uint64_t capacity : capacities)
    {
        if (capacity > std::numeric_limits<uint64_t>::max() - total)
            return std::numeric_limits<uint64_t>::max();
        total += capacity;
    }
    return total;
}

Runtime::Runtime()
{
    capabilities_.physical_memory_bytes = physical_memory_bytes();
    capabilities_.available_memory_bytes = available_memory_bytes();
    capabilities_.logical_cpu_count = std::max(1u, std::thread::hardware_concurrency());
    const CpuTopology cpu_topology = discover_cpu_topology();
    capabilities_.physical_cpu_core_count = cpu_topology.physical_core_count == 0 ? capabilities_.logical_cpu_count : cpu_topology.physical_core_count;
    const CpuIsaCapabilities cpu_isa = detect_cpu_isa_capabilities();
    capabilities_.cpu_isa_flags = cpu_isa.flags;
    capabilities_.cpu_isa = cpu_isa.names;
    capabilities_.mxfp4_kernel = mxfp4_kernel_name();
    capabilities_.float8_kernel = float8_linear_kernel_name(RuntimeOptimizationDefaultFlags);
    capabilities_.bfloat16_dot_kernel = bfloat16_dot_kernel_name();
    capabilities_.bfloat16_batched_linear_kernel = bfloat16_batched_linear_kernel_name(RuntimeOptimizationDefaultFlags);
    capabilities_.cpu_small_bfloat16_linear_policy = NcnnLinearOperator::cpu_small_bfloat16_linear_policy(
        RuntimeOptimizationDefaultFlags);
    capabilities_.cpu_linear_thread_limit = cpu_linear_thread_limit();
    capabilities_.float8_linear_thread_limit = float8_linear_thread_limit();
    capabilities_.float8_linear_row_group_size = float8_linear_row_group_size(RuntimeOptimizationDefaultFlags);
    capabilities_.mxfp4_decode_row_pair_group_size = mxfp4_decode_row_pair_group_size();
    capabilities_.activation_kernel = scaled_silu_kernel_name(RuntimeOptimizationDefaultFlags);
    const MxFp4KernelKind kernel = mxfp4_kernel_kind();
    if (kernel == MxFp4KernelKind::ArmNeon)
        capabilities_.flags |= RuntimeCapabilityMxfp4ArmNeon;
    else if (kernel == MxFp4KernelKind::ArmSve2)
        capabilities_.flags |= RuntimeCapabilityMxfp4ArmSve2;
    if (kernel == MxFp4KernelKind::X86Avx2)
        capabilities_.flags |= RuntimeCapabilityMxfp4X86Avx2;
    if (kernel == MxFp4KernelKind::X86Avx512)
        capabilities_.flags |= RuntimeCapabilityMxfp4X86Avx512;
#if defined(_OPENMP)
    capabilities_.flags |= RuntimeCapabilityOpenmpExperts;
    capabilities_.openmp_thread_count = static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
#endif
#if defined(NCNN_MOE_USE_NCNN) && NCNN_MOE_USE_NCNN
    capabilities_.flags |= RuntimeCapabilityNcnnCpuLinear;
#endif
#if defined(NCNN_MOE_WITH_VULKAN) && NCNN_MOE_WITH_VULKAN
    capabilities_.vulkan_device_count = NcnnLinearOperator::vulkan_device_count();
    if (capabilities_.vulkan_device_count > 0)
    {
        capabilities_.flags |= RuntimeCapabilityVulkanExecution | RuntimeCapabilityVulkanCpuMix | RuntimeCapabilityVulkanAttention | RuntimeCapabilityVulkanVictimCache
                               | RuntimeCapabilityVulkanDoubleBuffering | RuntimeCapabilityMxfp4VulkanProjection;
    }
    capabilities_.vulkan_heap_budget_bytes = NcnnLinearOperator::vulkan_heap_budget_bytes();
    capabilities_.vulkan_devices = NcnnLinearOperator::vulkan_device_capabilities();
    if (capabilities_.vulkan_devices.size() > 1)
    {
        capabilities_.flags |= RuntimeCapabilityMultiVulkanPlacement;
    }
    for (const VulkanDeviceCapabilities& device : capabilities_.vulkan_devices)
    {
        if (has_flag(device.flags, VulkanDeviceSelected))
        {
            capabilities_.selected_vulkan_device_index = device.index;
            break;
        }
    }
#endif
    register_adapter(std::make_shared<BuiltinModelAdapter>());
    register_adapter(std::make_shared<DeepSeekV4ModelAdapter>());
    register_adapter(std::make_shared<Qwen3_5MoeModelAdapter>());
}

void Runtime::register_adapter(std::shared_ptr<IMoeModelAdapter> adapter)
{
    if (adapter)
        adapters_.push_back(std::move(adapter));
}

Result<ModelPtr> Runtime::load_model(
    const std::filesystem::path& model_path,
    const RuntimeConfig& config,
    RuntimeLoadProgressCallback on_progress)
{
    constexpr uint32_t total_load_steps = 9;
    const auto report_progress = [&](uint32_t completed_steps, std::string_view phase, std::string_view message) {
        if (!on_progress)
            return;
        RuntimeLoadProgress progress;
        progress.completed_steps = completed_steps;
        progress.total_steps = total_load_steps;
        progress.phase = phase;
        progress.message = message;
        on_progress(progress);
    };

    report_progress(0, "validate", "Validating runtime configuration");
    if (has_flag(config.flags, RuntimeOptionMemoryMapExperts)
        && (has_flag(config.flags, RuntimeOptionDirectExpertIo)
            || has_flag(config.flags, RuntimeOptionBufferedExpertIo)))
    {
        return Error{ErrorCode::InvalidArgument, "memory-mapped and explicit Expert I/O modes are mutually exclusive"};
    }
    if (has_flag(config.flags, RuntimeOptionDirectExpertIo) && has_flag(config.flags, RuntimeOptionBufferedExpertIo))
    {
        return Error{ErrorCode::InvalidArgument, "direct and buffered Expert I/O are mutually exclusive"};
    }
    if (config.expert_gpu_victim_reuse_probe_interval == 0 || config.expert_gpu_victim_reuse_probe_interval > 1024)
    {
        return Error{ErrorCode::InvalidArgument, "Expert GPU victim reuse-probe interval must be between 1 and 1024"};
    }
    if (config.expected_concurrency == 0 || config.expected_concurrency > 1024)
    {
        return Error{ErrorCode::InvalidArgument, "expected_concurrency must be between 1 and 1024"};
    }
    HybridMode resolved_mode = config.hybrid_mode;
    if (resolved_mode == HybridMode::Auto)
        resolved_mode = has_flag(capabilities_.flags, RuntimeCapabilityVulkanCpuMix) ? HybridMode::HybridExperts : HybridMode::CpuOnly;
    if (resolved_mode == HybridMode::HybridExperts && !has_flag(capabilities_.flags, RuntimeCapabilityVulkanCpuMix))
    {
        return Error{ErrorCode::UnsupportedModel, "HybridExperts requires a Vulkan device and Vulkan-enabled ncnn"};
    }
    std::vector<uint32_t> selected_vulkan_device_indices;
    if (!config.vulkan_device_indices.empty())
    {
        if (config.vulkan_device_index != automatic_vulkan_device_index && config.vulkan_device_index != config.vulkan_device_indices.front())
        {
            return Error{ErrorCode::InvalidArgument, "vulkan_device_index must match the first explicit Vulkan device"};
        }
        selected_vulkan_device_indices = config.vulkan_device_indices;
    }
    else
    {
        if (config.vulkan_device_index != automatic_vulkan_device_index)
        {
            selected_vulkan_device_indices.push_back(config.vulkan_device_index);
        }
        else if (!capabilities_.vulkan_devices.empty())
        {
            selected_vulkan_device_indices.push_back(capabilities_.selected_vulkan_device_index);
        }
    }
    std::set<uint32_t> unique_device_indices;
    for (uint32_t device_index : selected_vulkan_device_indices)
    {
        if (device_index >= capabilities_.vulkan_devices.size())
        {
            return Error{ErrorCode::InvalidArgument, "a Vulkan device index is out of range"};
        }
        if (!unique_device_indices.insert(device_index).second)
        {
            return Error{ErrorCode::InvalidArgument, "Vulkan device indices must be unique"};
        }
    }
    const uint32_t selected_vulkan_device_index = selected_vulkan_device_indices.empty()
                                                      ? automatic_vulkan_device_index
                                                      : selected_vulkan_device_indices.front();
    const VulkanDeviceCapabilities* selected_vulkan_device = selected_vulkan_device_index < capabilities_.vulkan_devices.size()
                                                                 ? &capabilities_.vulkan_devices[selected_vulkan_device_index]
                                                                 : nullptr;
    if (config.hybrid_mode == HybridMode::Auto && (!selected_vulkan_device || selected_vulkan_device->type == VulkanDeviceType::Cpu))
    {
        resolved_mode = HybridMode::CpuOnly;
    }
    if (resolved_mode != HybridMode::CpuOnly)
    {
        for (uint32_t device_index : selected_vulkan_device_indices)
        {
            if (capabilities_.vulkan_devices[device_index].type == VulkanDeviceType::Cpu)
            {
                return Error{ErrorCode::UnsupportedModel, "Vulkan CPU devices cannot participate in heterogeneous placement"};
            }
        }
    }

    report_progress(1, "hardware", "Selecting CPU and Vulkan execution devices");
    std::filesystem::path root = model_path;
    std::filesystem::path manifest_path;
    std::error_code filesystem_error;
    if (std::filesystem::is_directory(model_path, filesystem_error))
        manifest_path = model_path / "config.json";
    else
    {
        manifest_path = model_path;
        root = model_path.parent_path();
    }

    report_progress(2, "manifest", "Reading model manifest");
    auto manifest_text = read_text_file(manifest_path);
    if (!manifest_text)
        return manifest_text.error();

    auto model_type = required_json_string(manifest_text.value(), "model_type");
    if (!model_type)
        return model_type.error();

    ModelPackage package{root, ModelManifest{std::move(model_type).value(), std::move(manifest_text).value()}, 0};

    const IMoeModelAdapter* selected_adapter = nullptr;
    for (const auto& adapter : adapters_)
    {
        if (adapter->can_load(package.manifest))
        {
            selected_adapter = adapter.get();
            break;
        }
    }
    if (!selected_adapter)
        return Error{ErrorCode::UnsupportedModel, "no adapter registered for model_type: " + package.manifest.model_type};

    report_progress(3, "architecture", "Parsing model architecture");
    auto parsed_ir = selected_adapter->parse_model(package);
    if (!parsed_ir)
        return parsed_ir.error();
    auto normalized_ir = normalize_moe_ir(parsed_ir.value());
    if (!normalized_ir)
        return normalized_ir.error();

    const bool use_vulkan_dense = resolved_mode == HybridMode::HybridExperts;
    const bool release_vulkan_dense_host_storage = use_vulkan_dense && has_flag(config.flags, RuntimeOptionReleaseVulkanDenseHostStorage);
    report_progress(4, "memory", "Planning host memory and Expert cache");
    const uint64_t current_available_memory_bytes = available_memory_bytes();
    auto raw_memory_plan = plan_model_memory(
        parsed_ir.value(),
        config,
        capabilities_.physical_memory_bytes,
        release_vulkan_dense_host_storage,
        current_available_memory_bytes,
        false);
    if (!raw_memory_plan)
        return raw_memory_plan.error();
    ModelMemoryPlan plan = std::move(raw_memory_plan).value();
    CpuPackedWeightMode selected_cpu_packed_weight_mode =
        CpuPackedWeightMode::Disabled;
    if (config.cpu_packed_weight_mode == CpuPackedWeightMode::Enabled)
    {
        if (!cpu_packed_weights_supported(parsed_ir.value(), config))
        {
            return Error{
                ErrorCode::UnsupportedModel,
                "CPU packed weights are unavailable for this model or CPU"};
        }
        auto packed_memory_plan = plan_model_memory(
            parsed_ir.value(),
            config,
            capabilities_.physical_memory_bytes,
            release_vulkan_dense_host_storage,
            current_available_memory_bytes,
            true);
        if (!packed_memory_plan)
            return packed_memory_plan.error();
        plan = std::move(packed_memory_plan).value();
        selected_cpu_packed_weight_mode = CpuPackedWeightMode::Enabled;
    }
    uint64_t effective_optimization_flags =
        config.optimization_flags & ~RuntimeOptimizationCpuPackedWeights;
    if (selected_cpu_packed_weight_mode == CpuPackedWeightMode::Enabled)
        effective_optimization_flags |= RuntimeOptimizationCpuPackedWeights;
    const bool file_backed_experts = has_flag(plan.flags, ModelMemoryFileBackedExperts);
    if (file_backed_experts)
        package.flags |= ModelPackageDeferMxfp4Experts;

    if (config.expert_gpu_cache_bytes > std::numeric_limits<uint64_t>::max() - config.expert_gpu_victim_cache_bytes)
    {
        return Error{ErrorCode::InvalidArgument, "the combined Expert GPU cache capacity overflows"};
    }
    const uint64_t requested_expert_gpu_bytes = config.expert_gpu_cache_bytes + config.expert_gpu_victim_cache_bytes;
    if (has_flag(config.flags, RuntimeOptionDisableGpuExpertExecution) && requested_expert_gpu_bytes != 0)
    {
        return Error{ErrorCode::InvalidArgument, "GPU Expert execution cannot be disabled while an Expert GPU cache is configured"};
    }
    const bool automatic_gpu_expert_cache = resolved_mode == HybridMode::HybridExperts
                                            && !has_flag(config.flags, RuntimeOptionDisableGpuExpertExecution)
                                            && requested_expert_gpu_bytes == 0;
    if (requested_expert_gpu_bytes != 0)
    {
        if (!file_backed_experts)
        {
            return Error{ErrorCode::InvalidArgument, "the Expert GPU cache requires on-demand Expert memory"};
        }
        if (!use_vulkan_dense || !has_flag(capabilities_.flags, RuntimeCapabilityVulkanVictimCache))
        {
            return Error{ErrorCode::UnsupportedModel, "the Expert GPU cache requires Vulkan mixed execution"};
        }
        if (requested_expert_gpu_bytes < plan.expert_pair_bytes)
        {
            return Error{ErrorCode::InvalidArgument, "the Expert GPU cache cannot hold one Expert pair"};
        }
    }

    report_progress(5, "weights", "Mapping model weights");
    auto weights = selected_adapter->map_weights(package, parsed_ir.value());
    if (!weights)
        return weights.error();

    ModelCompiler compiler;
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    ModelCompiler::BackendCapabilities compiler_capabilities;
    compiler_capabilities.flags = 0;
    compiler_capabilities.cpu_parallelism = capabilities_.openmp_thread_count;
    compiler_capabilities.vulkan_device_index = selected_vulkan_device_index;
    compiler_capabilities.expected_concurrency = config.expected_concurrency;
    compiler_capabilities.optimization_flags = effective_optimization_flags;
    compiler_capabilities.vulkan_context_instance = context_instance;
    compiler_capabilities.vulkan_device_indices = selected_vulkan_device_indices;
    compiler_capabilities.vulkan_device_scores.reserve(selected_vulkan_device_indices.size());
    for (uint32_t device_index : selected_vulkan_device_indices)
    {
        compiler_capabilities.vulkan_device_scores.push_back(std::max(1u, capabilities_.vulkan_devices[device_index].rough_score));
    }
    if (selected_vulkan_device)
    {
        compiler_capabilities.vulkan_queue_count = selected_vulkan_device->compute_queue_count;
    }
    if (has_flag(capabilities_.flags, RuntimeCapabilityCpuExecution))
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityCpuExecution;
    if (use_vulkan_dense && has_flag(capabilities_.flags, RuntimeCapabilityVulkanExecution))
    {
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityVulkanDense;
    }
    if (use_vulkan_dense && has_flag(capabilities_.flags, RuntimeCapabilityVulkanAttention))
    {
        if (has_flag(effective_optimization_flags, RuntimeOptimizationVulkanAttention))
            compiler_capabilities.flags |= ModelCompiler::BackendCapabilityVulkanAttention;
    }
    if (requested_expert_gpu_bytes != 0
        || (automatic_gpu_expert_cache && file_backed_experts)
        || (use_vulkan_dense
            && has_flag(effective_optimization_flags, RuntimeOptimizationVulkanQnK)))
    {
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityVulkanExperts;
    }
    if (has_flag(capabilities_.flags, RuntimeCapabilityMxfp4CpuKernel))
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityMxfp4CpuKernel;
    if (!file_backed_experts)
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityRetainCpuDenseCopies;
    if (release_vulkan_dense_host_storage && file_backed_experts)
        compiler_capabilities.flags |= ModelCompiler::BackendCapabilityReleaseVulkanDenseHostStorage;
    report_progress(6, "compile", "Compiling execution graph");
    auto compiled = compiler.compile(std::move(parsed_ir).value(), std::move(weights).value(), resolved_mode, compiler_capabilities);
    if (!compiled)
        return compiled.error();
    CompiledModel compiled_model = std::move(compiled).value();
    compiled_model.memory_plan = plan;
    compiled_model.runtime_option_flags = config.flags;
    compiled_model.optimization_flags = effective_optimization_flags;
    compiled_model.expected_concurrency = config.expected_concurrency;
    uint64_t expert_gpu_cache_bytes = config.expert_gpu_cache_bytes;
    uint64_t expert_gpu_victim_cache_bytes = config.expert_gpu_victim_cache_bytes;
    std::vector<VulkanDeviceCapabilities> cache_devices = capabilities_.vulkan_devices;
#if defined(NCNN_MOE_WITH_VULKAN) && NCNN_MOE_WITH_VULKAN
    if (automatic_gpu_expert_cache && file_backed_experts)
    {
        // Re-read the budget after dense weights are allocated.
        const std::vector<VulkanDeviceCapabilities> live_devices = NcnnLinearOperator::vulkan_device_capabilities();
        if (live_devices.size() == cache_devices.size())
            cache_devices = live_devices;
    }
#endif
    std::vector<uint64_t> automatic_executable_capacities;
    if (automatic_gpu_expert_cache && file_backed_experts)
    {
        automatic_executable_capacities = automatic_gpu_expert_capacities(
            plan.expert_pair_bytes,
            compiled_model.vulkan_device_indices,
            cache_devices,
            config.expected_concurrency);
        const bool every_device_can_hold_one_pair = !automatic_executable_capacities.empty()
                                                    && std::all_of(
                                                        automatic_executable_capacities.begin(),
                                                        automatic_executable_capacities.end(),
                                                        [pair_bytes = plan.expert_pair_bytes](uint64_t capacity) {
                                                            return capacity >= pair_bytes;
                                                        });
        if (every_device_can_hold_one_pair)
            expert_gpu_cache_bytes = sum_capacities(automatic_executable_capacities);
        else
            automatic_executable_capacities.clear();
    }
    compiled_model.effective_runtime_config.hybrid_mode = compiled_model.hybrid_mode;
    compiled_model.effective_runtime_config.requested_expert_memory_mode = plan.requested_mode;
    compiled_model.effective_runtime_config.selected_expert_memory_mode = plan.selected_mode;
    compiled_model.effective_runtime_config.requested_cpu_packed_weight_mode =
        config.cpu_packed_weight_mode;
    compiled_model.effective_runtime_config.selected_cpu_packed_weight_mode =
        selected_cpu_packed_weight_mode;
    compiled_model.effective_runtime_config.host_memory_budget_bytes = plan.host_memory_budget_bytes;
    compiled_model.effective_runtime_config.expert_cache_bytes = plan.expert_cache_bytes;
    compiled_model.effective_runtime_config.expert_gpu_cache_bytes = expert_gpu_cache_bytes;
    compiled_model.effective_runtime_config.expert_gpu_victim_cache_bytes = expert_gpu_victim_cache_bytes;
    compiled_model.effective_runtime_config.expert_gpu_victim_reuse_probe_interval = config.expert_gpu_victim_reuse_probe_interval;
    compiled_model.effective_runtime_config.vulkan_device_index = compiled_model.vulkan_device_index;
    compiled_model.effective_runtime_config.vulkan_device_indices = compiled_model.vulkan_device_indices;
    compiled_model.effective_runtime_config.flags = config.flags;
    compiled_model.effective_runtime_config.optimization_flags = effective_optimization_flags;
    compiled_model.effective_runtime_config.expected_concurrency = config.expected_concurrency;
    compiled_model.effective_runtime_config.file_backed_experts = file_backed_experts;
    report_progress(7, "cache", "Preparing Expert storage and caches");
    if (file_backed_experts)
    {
        uint32_t expert_io_workers = config.expert_io_workers;
        if (expert_io_workers == 0)
        {
            uint32_t maximum_active_experts = 1;
            for (const CompiledLayerPlan& layer : compiled_model.graph.layer_plans)
            {
                maximum_active_experts = std::max(maximum_active_experts, layer.moe.top_k);
            }
            for (const CompiledLayerPlan& layer : compiled_model.speculative.graph.layer_plans)
            {
                maximum_active_experts = std::max(maximum_active_experts, layer.moe.top_k);
            }
            const uint32_t exact_and_speculative_budget = maximum_active_experts > std::numeric_limits<uint32_t>::max() / 2
                                                              ? std::numeric_limits<uint32_t>::max()
                                                              : maximum_active_experts * 2;
            // Keep file-backed reads inside the reserved I/O pool.
            const CpuThreadBudget thread_budget = resolve_cpu_thread_budget();
            expert_io_workers = std::min(exact_and_speculative_budget, thread_budget.reserved_io_threads);
        }
        else
        {
            expert_io_workers = std::min(
                expert_io_workers,
                resolve_cpu_thread_budget(expert_io_workers).reserved_io_threads);
        }
        compiled_model.effective_runtime_config.expert_io_workers = expert_io_workers;
        uint32_t expert_cache_flags = 0;
        if (has_flag(config.flags, RuntimeOptionMemoryMapExperts))
        {
            expert_cache_flags |= ExpertCacheMemoryMapRanges;
        }
        if (has_flag(config.flags, RuntimeOptionDirectExpertIo))
        {
            expert_cache_flags |= ExpertCacheDirectReads;
        }
        if (has_flag(config.flags, RuntimeOptionBufferedExpertIo))
        {
            expert_cache_flags |= ExpertCacheBufferedReads;
        }
        if (has_flag(config.flags, RuntimeOptionForwardAwareCache))
        {
            expert_cache_flags |= ExpertCacheForwardAwareEviction;
        }
        if (has_flag(config.flags, RuntimeOptionRouterPrediction))
        {
            expert_cache_flags |= ExpertCacheAllowSpeculativeEviction;
        }
        if (has_flag(config.flags, RuntimeOptionCrossExpertReadCoalescing))
        {
            expert_cache_flags |= ExpertCacheCrossExpertReadCoalescing;
        }
        const std::vector<uint32_t>& expert_vulkan_device_indices = compiled_model.vulkan_device_indices;
        auto executable_capacities_result = [&]() -> Result<std::vector<uint64_t>> {
            if (!automatic_executable_capacities.empty())
                return std::move(automatic_executable_capacities);
            return distribute_gpu_capacity(
                expert_gpu_cache_bytes,
                plan.expert_pair_bytes,
                expert_vulkan_device_indices,
                cache_devices,
                "the executable Expert GPU cache");
        }();
        if (!executable_capacities_result)
            return executable_capacities_result.error();
        auto victim_capacities_result = distribute_gpu_capacity(
            expert_gpu_victim_cache_bytes,
            plan.expert_pair_bytes,
            expert_vulkan_device_indices,
            cache_devices,
            "the Expert GPU victim cache");
        if (!victim_capacities_result)
            return victim_capacities_result.error();
        std::vector<uint64_t> executable_capacities = std::move(executable_capacities_result).value();
        std::vector<uint64_t> victim_capacities = std::move(victim_capacities_result).value();
        for (size_t index = 0; index < expert_vulkan_device_indices.size(); ++index)
        {
            const uint64_t executable_capacity = executable_capacities[index];
            const uint64_t victim_capacity = victim_capacities[index];
            if (executable_capacity > std::numeric_limits<uint64_t>::max() - victim_capacity)
            {
                return Error{ErrorCode::InvalidArgument, "the combined per-device Expert GPU cache capacity overflows"};
            }
            const uint64_t maximum_safe_capacity = cache_devices[expert_vulkan_device_indices[index]].heap_budget_bytes;
            if (maximum_safe_capacity == 0 || executable_capacity + victim_capacity > maximum_safe_capacity)
            {
                return Error{ErrorCode::InvalidArgument, "combined per-device Expert GPU caches exceed the Vulkan heap budget"};
            }
        }
        std::vector<std::shared_ptr<IExpertVictimCache>> expert_victim_cache_shards;
        std::shared_ptr<IExpertVictimCache> expert_victim_cache;
        if (expert_gpu_victim_cache_bytes != 0)
        {
            expert_victim_cache_shards.reserve(expert_vulkan_device_indices.size());
            for (size_t index = 0; index < expert_vulkan_device_indices.size(); ++index)
            {
                const uint32_t device_index = expert_vulkan_device_indices[index];
                const uint64_t capacity = victim_capacities[index];
                auto shard = create_vulkan_victim_cache(
                    capacity,
                    device_index,
                    compiled_model.vulkan_context_instance,
                    effective_optimization_flags);
                if (!shard)
                {
                    return Error{ErrorCode::UnsupportedModel, "cannot create an Expert GPU victim-cache shard"};
                }
                expert_victim_cache_shards.push_back(std::move(shard));
            }
            expert_victim_cache = create_sharded_victim_cache(expert_victim_cache_shards);
            if (!expert_victim_cache)
            {
                return Error{ErrorCode::UnsupportedModel, "cannot create the sharded Expert GPU victim cache"};
            }
            if (config.expert_gpu_victim_reuse_probe_interval > 1)
            {
                expert_victim_cache = create_reuse_victim_cache(std::move(expert_victim_cache), config.expert_gpu_victim_reuse_probe_interval);
            }
        }
        const bool use_victim_device_source = !has_flag(config.flags, RuntimeOptionDisableGpuVictimExecution) && !expert_victim_cache_shards.empty();
        if (expert_gpu_cache_bytes != 0 || use_victim_device_source)
        {
            std::vector<std::shared_ptr<IExpertExecutionBackend>> device_backends;
            std::vector<uint32_t> expert_device_indices;
            for (size_t index = 0; index < expert_vulkan_device_indices.size(); ++index)
            {
                const uint32_t placement_device_index = expert_vulkan_device_indices[index];
                const uint64_t capacity = executable_capacities[index];
                auto backend = create_vulkan_mxfp4_expert_backend(
                    capacity,
                    placement_device_index,
                    !use_victim_device_source ? std::shared_ptr<IExpertVictimCache>() : expert_victim_cache_shards[index],
                    compiled_model.vulkan_context_instance,
                    effective_optimization_flags);
                if (!backend)
                {
                    return Error{ErrorCode::UnsupportedModel, "cannot create a Vulkan MXFP4 Expert execution cache/source backend"};
                }
                device_backends.push_back(std::move(backend));
                expert_device_indices.push_back(placement_device_index);
            }
            std::vector<uint32_t> residency_group_devices;
            residency_group_devices.reserve(compiled_model.graph.layer_plans.size() + compiled_model.speculative.graph.layer_plans.size());
            for (const CompiledLayerPlan& layer : compiled_model.graph.layer_plans)
            {
                residency_group_devices.push_back(layer.vulkan_device_index);
            }
            for (const CompiledLayerPlan& layer : compiled_model.speculative.graph.layer_plans)
            {
                residency_group_devices.push_back(layer.vulkan_device_index);
            }
            if (!use_victim_device_source)
            {
                compiled_model.expert_backend = create_multi_device_expert_backend(std::move(device_backends), std::move(expert_device_indices), std::move(residency_group_devices));
            }
            else
            {
                compiled_model.expert_backend = create_key_sharded_expert_backend(std::move(device_backends), std::move(expert_device_indices));
            }
            if (!compiled_model.expert_backend)
            {
                return Error{ErrorCode::UnsupportedModel, "cannot create the Vulkan MXFP4 Expert execution cache/source backend"};
            }
        }
        const size_t residency_group_count = compiled_model.graph.layer_plans.size() + compiled_model.speculative.graph.layer_plans.size();
        if (residency_group_count > std::numeric_limits<uint32_t>::max())
        {
            return Error{ErrorCode::InvalidModel, "the total Expert residency-group count overflows"};
        }
        compiled_model.expert_cache = std::make_shared<Mxfp4ExpertCache>(
            plan.expert_cache_bytes,
            expert_io_workers,
            std::move(expert_victim_cache),
            expert_cache_flags,
            static_cast<uint32_t>(residency_group_count),
            selected_cpu_packed_weight_mode == CpuPackedWeightMode::Enabled);
    }

    bool has_resident_qnk_experts = false;
    uint64_t maximum_qnk_pair_bytes = 0;
    uint32_t maximum_qnk_top_k = 1;
    const auto inspect_qnk_graph = [&](const ExecutionGraph& graph) {
        for (const CompiledLayerPlan& layer : graph.layer_plans)
        {
            maximum_qnk_top_k = std::max(maximum_qnk_top_k, layer.moe.top_k);
            for (const ExpertPlan& expert : layer.moe.experts)
            {
                if (expert.gate_up_weight == invalid_tensor_handle
                    || expert.down_weight == invalid_tensor_handle)
                {
                    continue;
                }
                const TensorData& gate_up = compiled_model.weights.at(expert.gate_up_weight);
                const TensorData& down = compiled_model.weights.at(expert.down_weight);
                if (is_qnk_dtype(gate_up.dtype) && gate_up.dtype == down.dtype)
                {
                    has_resident_qnk_experts = true;
                    maximum_qnk_pair_bytes = std::max(maximum_qnk_pair_bytes, expert.weight_bytes);
                }
            }
        }
    };
    inspect_qnk_graph(compiled_model.graph);
    inspect_qnk_graph(compiled_model.speculative.graph);
    if (has_resident_qnk_experts
        && use_vulkan_dense
        && has_flag(effective_optimization_flags, RuntimeOptimizationVulkanQnK)
        && !compiled_model.expert_backend)
    {
        if (maximum_qnk_pair_bytes == 0
            || maximum_qnk_pair_bytes > std::numeric_limits<uint64_t>::max() / maximum_qnk_top_k)
        {
            return Error{ErrorCode::InvalidArgument, "resident Qn_K Expert GPU capacity overflows"};
        }
        const uint64_t qnk_capacity = maximum_qnk_pair_bytes * maximum_qnk_top_k;
        auto backend = create_vulkan_mxfp4_expert_backend(
            qnk_capacity,
            selected_vulkan_device_index,
            nullptr,
            compiled_model.vulkan_context_instance,
            effective_optimization_flags);
        if (!backend)
            return Error{ErrorCode::UnsupportedModel, "cannot create the Vulkan Qn_K Expert execution backend"};
        compiled_model.expert_backend = std::move(backend);
        compiled_model.effective_runtime_config.expert_gpu_cache_bytes = qnk_capacity;
    }

    report_progress(8, "finalize", "Finalizing runtime model");
    auto immutable = std::make_shared<const CompiledModel>(std::move(compiled_model));
    report_progress(9, "ready", "Model initialization complete");
    return ModelPtr(new Model(std::move(immutable)));
}

Result<void> Runtime::synchronize_model_caches(const ModelPtr& model)
{
    if (!model)
    {
        return Error{ErrorCode::InvalidArgument, "model cannot be null"};
    }
    if (model->compiled_->expert_cache)
        model->compiled_->expert_cache->wait_for_background_work();
    if (model->compiled_->expert_backend)
        model->compiled_->expert_backend->wait_for_background_work();
    return {};
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
    if (has_flag(options.flags, SchedulerOptionDisableStagedBatching) && has_flag(options.flags, SchedulerOptionForceStagedBatching))
    {
        return Error{ErrorCode::InvalidArgument, "scheduler staged batching cannot be both disabled and forced"};
    }
    if (options.worker_count > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler worker_count exceeds 1024"};
    if (options.expert_threads_per_worker > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler expert_threads_per_worker exceeds 1024"};
    if (options.reserved_io_threads > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler reserved_io_threads exceeds 1024"};
    if (options.reserved_service_threads > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler reserved_service_threads exceeds 1024"};
    if (options.adaptive_probe_interval > 1000000)
    {
        return Error{ErrorCode::InvalidArgument, "scheduler adaptive_probe_interval exceeds 1000000"};
    }
    if (options.cross_call_window_microseconds > 1000000)
    {
        return Error{ErrorCode::InvalidArgument, "scheduler cross_call_window_microseconds exceeds 1000000"};
    }
    if (options.cross_call_max_batch_size > 1024)
    {
        return Error{ErrorCode::InvalidArgument, "scheduler cross_call_max_batch_size exceeds 1024"};
    }
    if (options.worker_cpu_sets.size() > 1024)
        return Error{ErrorCode::InvalidArgument, "scheduler worker_cpu_sets exceeds 1024 entries"};
    if (!options.worker_cpu_sets.empty() && options.worker_count != 0 && options.worker_count != options.worker_cpu_sets.size())
    {
        return Error{ErrorCode::InvalidArgument, "scheduler worker_count must match worker_cpu_sets"};
    }
    for (const std::vector<uint32_t>& cpu_set : options.worker_cpu_sets)
    {
        if (cpu_set.empty())
            return Error{ErrorCode::InvalidArgument, "scheduler worker_cpu_sets entries cannot be empty"};
    }
#if !defined(__linux__)
    if (!options.worker_cpu_sets.empty())
        return Error{ErrorCode::UnsupportedModel, "scheduler worker_cpu_sets are supported only on Linux"};
#endif
    return std::make_shared<BatchScheduler>(options);
}

} // namespace moe
} // namespace ncnn
