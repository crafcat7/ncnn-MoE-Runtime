#include "modelloader.h"

#include "cpu.h"
#include "expertbackend.h"
#include "expert.h"
#include "graph/compiler.h"
#include "kernels/mxfp4.h"
#include "models/modeladapter.h"
#include "models/safetensors.h"
#include "models/tokenizer.h"
#include "models/ggufmodel.h"
#include "storage/expertcache.h"
#include "storage/expertresidency.h"
#include "storage/expertcache_victim.h"
#include "storage/mappedfile.h"
#include "graph/memoryplan.h"
#include "backends/ncnn/vulkan.h"
#include "backends/ncnn/modelpipeline.h"
#include "backends/ncnn/expertbackend_vulkan.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace ncnn {
namespace moe {

static bool support_cpu_packed_weights(const MoeModelDescriptor& descriptor,
                                       const Option& opt) noexcept
{
    for (const LayerDescriptor& layer : descriptor.layers)
    {
        if (layer.dense.enabled())
            continue;
        const DType dtype = layer.moe.expert_weight_dtype;
        if (is_qnk_dtype(dtype))
            return true;
        if (dtype == DType::MxFp4
            && has_flag(opt.optimization_flags,
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

static Result<std::vector<uint64_t>> distribute_gpu_cache_size(uint64_t total_size,
                                                               uint64_t expert_pair_size,
                                                               const std::vector<uint32_t>& device_indices,
                                                               const std::vector<GpuInfo>& gpu_infos,
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

static std::vector<uint64_t> get_auto_gpu_cache_sizes(uint64_t expert_pair_size,
                                                      const std::vector<uint32_t>& device_indices,
                                                      const std::vector<GpuInfo>& gpu_infos,
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
        const GpuInfo& gpu_info = gpu_infos[device_indices[i]];
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

ModelLoader::ModelLoader(const RuntimeInfo& _info,
                         const std::vector<std::shared_ptr<ModelAdapter>>& _adapters,
                         const Option& _opt,
                         CompiledModel& _model)
    : info(_info),
      adapters(_adapters),
      opt(_opt),
      model(_model)
{
}

Result<void> ModelLoader::validate_option()
{
    if (opt.hybrid_mode != HybridMode::CpuOnly
        && opt.hybrid_mode != HybridMode::HybridExperts
        && opt.hybrid_mode != HybridMode::Auto)
        return Error{ErrorCode::InvalidArgument, "invalid HybridMode"};
    if (opt.cpu_packed_weight_mode != CpuPackedWeightMode::Disabled
        && opt.cpu_packed_weight_mode != CpuPackedWeightMode::Enabled)
        return Error{ErrorCode::InvalidArgument, "invalid CpuPackedWeightMode"};
    if (opt.expert_io_mode != ExpertIoMode::Auto
        && opt.expert_io_mode != ExpertIoMode::Mmap
        && opt.expert_io_mode != ExpertIoMode::Direct
        && opt.expert_io_mode != ExpertIoMode::Buffered)
        return Error{ErrorCode::InvalidArgument, "invalid ExpertIoMode"};
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
    if (model_path.empty())
        return Error{ErrorCode::InvalidArgument, "model path cannot be empty"};

    std::error_code absolute_error;
    const std::filesystem::path resolved_model_path = std::filesystem::absolute(model_path, absolute_error);
    if (absolute_error || resolved_model_path.empty())
        return Error{ErrorCode::IoError, "cannot resolve model path: " + model_path.string()};

    std::filesystem::path root = resolved_model_path;
    std::filesystem::path manifest_path;
    std::error_code ec;
    const bool model_directory = std::filesystem::is_directory(resolved_model_path, ec);
    if (model_directory)
        manifest_path = resolved_model_path / "config.json";
    else
    {
        manifest_path = resolved_model_path;
        root = resolved_model_path.parent_path();
    }

    const bool gguf = has_gguf_source(resolved_model_path);
    if (model_directory && !gguf && !std::filesystem::is_regular_file(manifest_path, ec))
    {
        return Error{ErrorCode::InvalidArgument,
                     "no config.json or GGUF model files found directly in model directory: " + resolved_model_path.string()
                         + "; subdirectories are not searched; select the directory containing model files or a GGUF file"};
    }
    auto manifest_json = [&]() -> Result<std::string> {
        if (!gguf)
            return read_text_file(manifest_path);
        auto archive = GgufArchive::open(resolved_model_path);
        if (!archive)
            return archive.error();
        return gguf_model_manifest(archive.value());
    }();
    if (!manifest_json)
        return manifest_json.error();

    auto model_type = read_manifest_string(manifest_json.value(), "model_type");
    if (!model_type)
        return model_type.error();

    package = ModelPackage{root, ModelManifest{std::move(model_type).value(), std::move(manifest_json).value()}, 0};
    if (gguf)
        package.weights_path = resolved_model_path;

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
    descriptor = std::move(parsed).value();

    auto valid = validate_model_descriptor(descriptor);
    if (!valid)
        return valid.error();

    return {};
}

Result<void> ModelLoader::plan_memory()
{
    const bool reserve_cpu_packed_weights = opt.cpu_packed_weight_mode == CpuPackedWeightMode::Enabled;
    // ModelLoader sets weights_path only for GGUF, which always takes the native-weight path.
    bool use_native_weights = !package.weights_path.empty();
    if (!use_native_weights)
    {
        auto archive = SafetensorsArchive::open(package.root, package.manifest.raw_json);
        use_native_weights = archive
                             && (archive.value().has_native_quantized_tensors()
                                 || (!reserve_cpu_packed_weights && archive.value().has_float8_tensors()));
    }
    if (use_native_weights)
    {
        const uint32_t old_flags = package.flags;
        package.flags |= ModelPackageDeferMxfp4Experts;
        auto mapped = adapter->map_weights(package, descriptor);
        package.flags = old_flags;
        if (!mapped)
            return mapped.error();
        native_weights = std::move(mapped).value();
        native_quantized_experts = std::any_of(native_weights->begin(), native_weights->end(), [](const auto& item) {
            return item.first.find(".experts.") != std::string::npos
                   && (item.second.dtype == DType::Int4 || is_qnk_dtype(item.second.dtype));
        });
    }
    const bool use_vulkan_compute = opt.hybrid_mode == HybridMode::HybridExperts;
    const bool use_vulkan_dense_host_release = use_vulkan_compute && has_flag(info.flags, RuntimeVulkan);

    if (reserve_cpu_packed_weights
        && std::any_of(descriptor.layers.begin(), descriptor.layers.end(), [](const LayerDescriptor& layer) {
               return layer.dense.enabled()
                      && (is_qnk_dtype(layer.dense.gate_weight_dtype)
                          || is_qnk_dtype(layer.dense.up_weight_dtype)
                          || is_qnk_dtype(layer.dense.down_weight_dtype));
           }))
    {
        return Error{
            ErrorCode::UnsupportedModel,
            "dense Qn_K FFNs require CpuPackedWeightMode::Disabled; CPU packed weights are supported for routed Experts only"};
    }
    if (reserve_cpu_packed_weights && !support_cpu_packed_weights(descriptor, opt))
    {
        return Error{
            ErrorCode::UnsupportedModel,
            "CPU packed weights are unavailable for this model or CPU"};
    }

    const uint64_t available_size = available_memory_size();
    auto planned = plan_model_memory(descriptor,
                                     opt,
                                     info.physical_memory_size,
                                     use_vulkan_dense_host_release,
                                     available_size,
                                     reserve_cpu_packed_weights,
                                     native_weights ? &*native_weights : nullptr,
                                     use_vulkan_compute && has_flag(info.flags, RuntimeVulkanAttention));
    if (!planned)
        return planned.error();
    plan = std::move(planned).value();

    opt.optimization_flags &= ~OptimizationCpuPackedWeights;
    if (reserve_cpu_packed_weights)
        opt.optimization_flags |= OptimizationCpuPackedWeights;

    const bool use_file_backed_experts = plan.selected_mode == ExpertMemoryMode::OnDemand;
    const bool use_file_backed_mxfp4 = use_file_backed_experts
                                       && std::all_of(descriptor.layers.begin(),
                                                      descriptor.layers.end(),
                                                      [](const LayerDescriptor& layer) {
                                                          return layer.dense.enabled()
                                                                 || layer.moe.expert_weight_dtype == DType::MxFp4;
                                                      });
    const bool native_file_backed_qnk_experts = native_weights
                                                && std::any_of(native_weights->begin(),
                                                               native_weights->end(),
                                                               [](const auto& item) {
                                                                   return item.first.find(".experts.") != std::string::npos
                                                                          && is_qnk_dtype(item.second.dtype)
                                                                          && static_cast<bool>(item.second.mxfp4_file_storage);
                                                               });
    const bool use_file_backed_qnk = use_file_backed_experts && native_file_backed_qnk_experts;
    if (plan.selected_mode == ExpertMemoryMode::Eager && native_weights
        && std::any_of(native_weights->begin(), native_weights->end(), [](const auto& item) {
               return static_cast<bool>(item.second.mxfp4_file_storage);
           }))
    {
        bool ready = std::all_of(native_weights->begin(), native_weights->end(), [](const auto& item) {
            return !item.second.mxfp4_file_storage || is_qnk_dtype(item.second.dtype);
        });
        if (ready)
        {
            for (auto& [name, tensor] : *native_weights)
            {
                const auto storage = tensor.mxfp4_file_storage;
                if (!storage)
                    continue;
                auto primary = MappedFileRange::open(storage->blocks_path, storage->blocks_offset, storage->blocks_size);
                if (!primary)
                {
                    ready = false;
                    break;
                }
                if (storage->secondary_blocks_path.empty())
                {
                    if (package.weights_path.empty())
                        primary.value()->prefault();
                    tensor.mapped_data = primary.value()->share_data();
                    tensor.mapped_size = storage->blocks_size;
                }
                else
                {
                    auto secondary = MappedFileRange::open(storage->secondary_blocks_path, storage->secondary_blocks_offset,
                                                           storage->secondary_blocks_size);
                    if (!secondary
                        || storage->secondary_blocks_size > std::numeric_limits<size_t>::max()
                        || storage->blocks_size > std::numeric_limits<size_t>::max() - storage->secondary_blocks_size)
                    {
                        ready = false;
                        break;
                    }
                    tensor.quantized_data.resize(static_cast<size_t>(storage->blocks_size + storage->secondary_blocks_size));
                    std::copy_n(primary.value()->data(), static_cast<size_t>(storage->blocks_size), tensor.quantized_data.data());
                    std::copy_n(secondary.value()->data(), static_cast<size_t>(storage->secondary_blocks_size),
                                tensor.quantized_data.data() + static_cast<size_t>(storage->blocks_size));
                }
                tensor.mxfp4_file_storage.reset();
            }
        }
        if (!ready)
            native_weights.reset();
    }
    const bool support_gpu_cache = use_file_backed_experts
                                   && std::all_of(descriptor.layers.begin(),
                                                  descriptor.layers.end(),
                                                  [](const LayerDescriptor& layer) {
                                                      if (layer.dense.enabled())
                                                          return true;
                                                      const MoeDescriptor& moe = layer.moe;
                                                      return moe.expert_weight_dtype == DType::MxFp4
                                                             || (moe.expert_weight_dtype == DType::BFloat16
                                                                 && has_flag(moe.flags, MoeDescriptorFileBackedExperts));
                                                  });
    if (use_file_backed_mxfp4 || use_file_backed_qnk)
        package.flags |= ModelPackageDeferMxfp4Experts;

    if (opt.expert_gpu_cache_size > std::numeric_limits<uint64_t>::max() - opt.expert_gpu_victim_cache_size)
        return Error{ErrorCode::InvalidArgument, "the combined Expert GPU cache capacity overflows"};

    requested_gpu_cache_size = opt.expert_gpu_cache_size + opt.expert_gpu_victim_cache_size;
    if (has_flag(opt.flags, OptionDisableGpuExpertExecution) && requested_gpu_cache_size != 0)
        return Error{ErrorCode::InvalidArgument, "GPU Expert execution cannot be disabled while an Expert GPU cache is configured"};
    if (native_quantized_experts && opt.expert_gpu_victim_cache_size != 0)
        return Error{ErrorCode::UnsupportedModel, "the Expert GPU victim cache requires MXFP4 Expert weights"};

    // Automatic GPU caching is limited to MXFP4. Large BF16 uploads cause a
    // cold cache to thrash during prefill, but explicit BF16 cache sizes remain
    // supported.
    use_auto_gpu_cache = use_vulkan_compute
                         && !has_flag(opt.flags, OptionDisableGpuExpertExecution)
                         && use_file_backed_mxfp4
                         && requested_gpu_cache_size == 0;
    if (requested_gpu_cache_size == 0)
        return {};
    const bool resident_dense_ffn = std::any_of(descriptor.layers.begin(), descriptor.layers.end(), [](const LayerDescriptor& layer) {
        return layer.dense.enabled();
    });
    if (native_quantized_experts || resident_dense_ffn)
    {
        if (!use_vulkan_compute || !has_flag(info.flags, RuntimeVulkan))
            return Error{ErrorCode::UnsupportedModel, "the Expert GPU cache requires Vulkan mixed execution"};
        if (resident_dense_ffn && !use_file_backed_experts && opt.expert_gpu_victim_cache_size != 0)
            return Error{ErrorCode::UnsupportedModel, "the Expert GPU victim cache requires file-backed Expert weights"};
        return {};
    }
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
    const bool use_vulkan_compute = opt.hybrid_mode == HybridMode::HybridExperts;
    auto weights = native_weights ? Result<WeightMapping>(std::move(*native_weights)) : adapter->map_weights(package, descriptor);
    native_weights.reset();
    if (!weights)
        return weights.error();
    const bool use_file_backed_experts = plan.selected_mode == ExpertMemoryMode::OnDemand;
    const bool has_dense_ffn = std::any_of(descriptor.layers.begin(), descriptor.layers.end(), [](const LayerDescriptor& layer) {
        return layer.dense.enabled();
    });

    const GpuInfo* gpu_info = nullptr;
    if (opt.vulkan_device_index != automatic_vulkan_device_index)
        gpu_info = &info.gpu_infos[opt.vulkan_device_index];

    CompilerOption compiler_opt;
    compiler_opt.flags = 0;
    if (use_vulkan_compute)
        compiler_opt.vulkan_runtime = create_vulkan_runtime();
    compiler_opt.device_index = opt.vulkan_device_index;
    compiler_opt.num_concurrent_sessions = opt.num_concurrent_sessions;
    compiler_opt.explicit_expert_gpu_cache = requested_gpu_cache_size != 0;
    compiler_opt.gpu_expert_cache_reserve = requested_gpu_cache_size;
    compiler_opt.gpu_heap_budget = gpu_info ? gpu_info->heap_budget : 0;

    // Use the smallest heap as the common multi-device budget so persistent
    // operators cannot over-admit on a weaker shard.
    for (uint32_t device_index : opt.vulkan_device_indices)
    {
        const uint64_t heap_budget = info.gpu_infos[device_index].heap_budget;
        if (heap_budget == 0)
        {
            compiler_opt.gpu_heap_budget = 0;
            break;
        }
        if (compiler_opt.gpu_heap_budget == 0)
            compiler_opt.gpu_heap_budget = heap_budget;
        else
            compiler_opt.gpu_heap_budget = std::min(compiler_opt.gpu_heap_budget,
                                                    heap_budget);
    }

    compiler_opt.optimization_flags = opt.optimization_flags;
    compiler_opt.device_indices = opt.vulkan_device_indices;
    compiler_opt.device_scores.reserve(opt.vulkan_device_indices.size());
    for (uint32_t device_index : opt.vulkan_device_indices)
    {
        compiler_opt.device_scores.push_back(std::max(1u, info.gpu_infos[device_index].rough_score));
    }
    if (has_flag(info.flags, RuntimeCpu))
        compiler_opt.flags |= BackendCpuExecution;
    if (use_vulkan_compute && has_flag(info.flags, RuntimeVulkan))
        compiler_opt.flags |= BackendVulkanDense;
    if (use_vulkan_compute
        && has_flag(info.flags, RuntimeVulkanAttention)
        && has_flag(opt.optimization_flags, OptimizationVulkanAttention))
    {
        compiler_opt.flags |= BackendVulkanAttention;
    }
    if (requested_gpu_cache_size != 0
        || use_auto_gpu_cache
        || (use_vulkan_compute
            && (has_dense_ffn || has_flag(opt.optimization_flags, OptimizationVulkanQnK))
            && !has_flag(opt.flags, OptionDisableGpuExpertExecution)))
    {
        compiler_opt.flags |= BackendVulkanExperts;
    }
    if (has_flag(info.flags, RuntimeMxfp4Cpu))
        compiler_opt.flags |= BackendMxfp4CpuKernel;
    if (use_file_backed_experts)
        compiler_opt.flags |= BackendFileBackedExperts;

    auto ret = ncnn::moe::compile_model(std::move(descriptor),
                                        std::move(weights).value(),
                                        model,
                                        opt.hybrid_mode,
                                        compiler_opt);
    if (!ret)
        return ret.error();

    auto finalized_plan = finalize_model_memory_plan(plan, opt, model.released_dense_host_size);
    if (!finalized_plan)
        return finalized_plan.error();
    model.memory_plan = plan;
    return {};
}

uint32_t ModelLoader::resolve_expert_io_threads() const
{
    if (opt.num_expert_io_threads != 0)
    {
        return std::min(opt.num_expert_io_threads,
                        resolve_cpu_thread_budget(opt.num_expert_io_threads).num_io_threads);
    }

    uint32_t max_active_experts = 1;
    for (const CompiledLayerPlan& layer : model.graph.layer_plans)
        max_active_experts = std::max(max_active_experts, layer.moe.top_k);
    for (const CompiledLayerPlan& layer : model.speculative.graph.layer_plans)
        max_active_experts = std::max(max_active_experts, layer.moe.top_k);

    const uint32_t max_io_threads = max_active_experts > std::numeric_limits<uint32_t>::max() / 2
                                        ? std::numeric_limits<uint32_t>::max()
                                        : max_active_experts * 2;
    return std::min(max_io_threads, resolve_cpu_thread_budget().num_io_threads);
}

Result<void> ModelLoader::resolve_gpu_cache_sizes(std::vector<uint64_t>& gpu_cache_sizes,
                                                  std::vector<uint64_t>& gpu_victim_cache_sizes)
{
    const std::vector<uint32_t>& device_indices = model.opt.vulkan_device_indices;
    std::vector<GpuInfo> live_gpu_infos;

#if defined(NCNN_MOE_WITH_VULKAN) && NCNN_MOE_WITH_VULKAN
    if (use_auto_gpu_cache || model.qwen_device_chain)
    {
        // Dense weights have already been allocated, so use the live budget.
        live_gpu_infos = get_gpu_infos();
    }
#endif
    const bool use_live_gpu_infos = live_gpu_infos.size() == info.gpu_infos.size()
                                    && std::all_of(device_indices.begin(), device_indices.end(),
                                                   [&live_gpu_infos](uint32_t device_index) {
                                                       return device_index < live_gpu_infos.size();
                                                   });
    const std::vector<GpuInfo>& gpu_infos = use_live_gpu_infos ? live_gpu_infos : info.gpu_infos;
    if (model.qwen_device_chain && !use_live_gpu_infos)
        return Error{ErrorCode::UnsupportedModel, "Qwen device execution requires a live Vulkan heap budget for Expert cache admission"};

    // Loading-time free heap predates dense and persistent attention uploads.
    // If the live query is unavailable, leave the automatic cache disabled
    // instead of admitting Experts against stale free-memory information.
    if (use_auto_gpu_cache && use_live_gpu_infos)
    {
        std::vector<GpuInfo> cache_gpu_infos = gpu_infos;
        for (uint32_t device : device_indices)
        {
            auto& gpu = cache_gpu_infos[device];
            const uint64_t reserve = qwen_device_pending_reserve(model, device);
            gpu.heap_available = gpu.heap_available > reserve ? gpu.heap_available - reserve : 0;
        }
        std::vector<uint64_t> sizes = get_auto_gpu_cache_sizes(plan.expert_pair_size,
                                                               device_indices,
                                                               cache_gpu_infos,
                                                               opt.num_concurrent_sessions);
        const bool valid = !sizes.empty()
                           && std::all_of(sizes.begin(),
                                          sizes.end(),
                                          [pair_size = plan.expert_pair_size](uint64_t size) {
                                              return size >= pair_size;
                                          });
        if (valid)
        {
            model.opt.expert_gpu_cache_size = sum_sizes(sizes);
            gpu_cache_sizes = std::move(sizes);
        }
    }

    if (gpu_cache_sizes.empty())
    {
        auto sizes = distribute_gpu_cache_size(model.opt.expert_gpu_cache_size,
                                               plan.expert_pair_size,
                                               device_indices,
                                               gpu_infos,
                                               "the executable Expert GPU cache");
        if (!sizes)
            return sizes.error();
        gpu_cache_sizes = std::move(sizes).value();
    }

    auto sizes = distribute_gpu_cache_size(model.opt.expert_gpu_victim_cache_size,
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
        const auto& gpu = gpu_infos[device_indices[i]];
        const uint64_t pending = qwen_device_pending_reserve(model, device_indices[i]);
        if (pending != 0 && (gpu.heap_available <= pending || cache_size + victim_size > gpu.heap_available - pending))
            return Error{ErrorCode::InvalidArgument, "combined Expert GPU caches leave insufficient memory for Qwen device execution"};
    }

    return {};
}

Result<std::shared_ptr<ExpertVictimCache>> ModelLoader::create_expert_victim_cache(const std::vector<uint64_t>& gpu_victim_cache_sizes,
                                                                                   std::vector<std::shared_ptr<ExpertVictimCache>>& victim_caches)
{
    if (model.opt.expert_gpu_victim_cache_size == 0)
        return std::shared_ptr<ExpertVictimCache>();

    const std::vector<uint32_t>& device_indices = model.opt.vulkan_device_indices;
    victim_caches.reserve(device_indices.size());
    for (size_t i = 0; i < device_indices.size(); ++i)
    {
        auto shard = create_vulkan_victim_cache(gpu_victim_cache_sizes[i],
                                                device_indices[i],
                                                model.vulkan_runtime,
                                                model.opt.optimization_flags);
        if (!shard)
            return Error{ErrorCode::UnsupportedModel, "cannot create an Expert GPU victim-cache shard"};
        victim_caches.push_back(std::move(shard));
    }

    std::shared_ptr<ExpertVictimCache> victim_cache = create_sharded_victim_cache(victim_caches);
    if (!victim_cache)
        return Error{ErrorCode::UnsupportedModel, "cannot create the sharded Expert GPU victim cache"};
    if (opt.expert_gpu_victim_reuse_probe_interval > 1)
    {
        victim_cache = create_reuse_victim_cache(std::move(victim_cache),
                                                 opt.expert_gpu_victim_reuse_probe_interval);
    }

    return victim_cache;
}

Result<void> ModelLoader::create_expert_backend(const std::vector<uint64_t>& gpu_cache_sizes,
                                                const std::vector<std::shared_ptr<ExpertVictimCache>>& victim_caches)
{
    const bool use_victim_source = !has_flag(opt.flags, OptionDisableGpuVictimExecution)
                                   && !victim_caches.empty();
    if (model.opt.expert_gpu_cache_size == 0 && !use_victim_source)
        return {};

    const std::vector<uint32_t>& device_indices = model.opt.vulkan_device_indices;
    std::vector<std::shared_ptr<ExpertBackend>> backends;
    backends.reserve(device_indices.size());
    for (size_t i = 0; i < device_indices.size(); ++i)
    {
        auto backend = create_vulkan_expert_backend(gpu_cache_sizes[i],
                                                    device_indices[i],
                                                    !use_victim_source ? std::shared_ptr<ExpertVictimCache>() : victim_caches[i],
                                                    model.vulkan_runtime,
                                                    model.opt.optimization_flags);
        if (!backend)
            return Error{ErrorCode::UnsupportedModel, "cannot create a Vulkan Expert execution cache/source backend"};
        backends.push_back(std::move(backend));
    }

    if (use_victim_source)
    {
        model.expert_backend = create_key_sharded_expert_backend(std::move(backends),
                                                                 device_indices);
    }
    else
    {
        std::vector<uint32_t> group_device_indices;
        group_device_indices.reserve(model.graph.layer_plans.size() + model.speculative.graph.layer_plans.size());
        for (const CompiledLayerPlan& layer : model.graph.layer_plans)
            group_device_indices.push_back(layer.vulkan_device_index);
        for (const CompiledLayerPlan& layer : model.speculative.graph.layer_plans)
            group_device_indices.push_back(layer.vulkan_device_index);

        model.expert_backend = create_multi_device_expert_backend(std::move(backends),
                                                                  device_indices,
                                                                  std::move(group_device_indices));
    }

    if (!model.expert_backend)
        return Error{ErrorCode::UnsupportedModel, "cannot create the Vulkan Expert execution cache/source backend"};
    return {};
}

Result<void> ModelLoader::configure_expert_cache()
{
    model.opt.expert_io_mode = opt.expert_io_mode;
    model.opt.expert_gpu_cache_size = opt.expert_gpu_cache_size;
    model.opt.expert_gpu_victim_cache_size = opt.expert_gpu_victim_cache_size;
    model.opt.expert_gpu_victim_reuse_probe_interval = opt.expert_gpu_victim_reuse_probe_interval;
    model.opt.flags = opt.flags;
    if (model.memory_plan.selected_mode != ExpertMemoryMode::OnDemand)
        return {};

    const uint32_t num_io_threads = resolve_expert_io_threads();
    model.opt.num_expert_io_threads = num_io_threads;

    std::vector<uint64_t> gpu_cache_sizes;
    std::vector<uint64_t> gpu_victim_cache_sizes;
    auto ret = resolve_gpu_cache_sizes(gpu_cache_sizes, gpu_victim_cache_sizes);
    if (!ret)
        return ret.error();
    std::vector<std::shared_ptr<ExpertVictimCache>> victim_caches;
    auto victim_cache = create_expert_victim_cache(gpu_victim_cache_sizes, victim_caches);
    if (!victim_cache)
        return victim_cache.error();
    ret = create_expert_backend(gpu_cache_sizes, victim_caches);
    if (!ret)
        return ret.error();

    const size_t group_count = model.graph.layer_plans.size() + model.speculative.graph.layer_plans.size();
    if (group_count > std::numeric_limits<uint32_t>::max())
        return Error{ErrorCode::InvalidModel, "the total Expert residency-group count overflows"};

    uint32_t cache_flags = 0;
    if (has_flag(opt.flags, OptionForwardAwareCache))
        cache_flags |= ExpertCacheForwardAwareEviction;
    if (has_flag(opt.flags, OptionRouterPrediction))
        cache_flags |= ExpertCacheAllowSpeculativeEviction;
    if (has_flag(opt.flags, OptionCrossExpertReadCoalescing))
        cache_flags |= ExpertCacheCrossExpertReadCoalescing;

    const bool coordinate_gpu_caches = model.expert_backend || static_cast<bool>(victim_cache.value());
    model.expert_cache = std::make_shared<ExpertCache>(plan.expert_cache_size,
                                                       num_io_threads,
                                                       std::move(victim_cache).value(),
                                                       opt.expert_io_mode,
                                                       cache_flags,
                                                       static_cast<uint32_t>(group_count),
                                                       has_flag(model.opt.optimization_flags, OptimizationCpuPackedWeights));
    if (coordinate_gpu_caches)
    {
        auto residency = std::make_shared<ExpertResidencyCoordinator>();
        model.expert_cache->set_residency_coordinator(residency);
        if (model.expert_backend)
            model.expert_backend->set_residency_coordinator(std::move(residency));
    }
    return {};
}

Result<void> ModelLoader::configure_resident_quantized_backend()
{
    if (opt.hybrid_mode != HybridMode::HybridExperts
        || has_flag(opt.flags, OptionDisableGpuExpertExecution)
        || model.expert_backend)
    {
        return {};
    }

    // Effective capacity describes a constructed cache, including when an
    // undersized dense request leaves execution on the CPU.
    model.opt.expert_gpu_cache_size = 0;
    std::vector<uint32_t> device_indices = model.opt.vulkan_device_indices;
    if (device_indices.empty())
        device_indices.push_back(model.opt.vulkan_device_index);
    std::vector<uint64_t> max_pair_sizes(device_indices.size());
    std::vector<uint64_t> total_sizes(device_indices.size());
    uint64_t smallest_dense_entry_size = std::numeric_limits<uint64_t>::max();
    bool invalid_storage_size = false;
    const auto add_size = [&](uint64_t& total, uint64_t size) {
        if (size > std::numeric_limits<uint64_t>::max() - total)
            invalid_storage_size = true;
        else
            total += size;
    };
    const auto inspect_graph = [&](const ExecutionGraph& graph) {
        for (const CompiledLayerPlan& layer : graph.layer_plans)
        {
            const auto found = std::find(device_indices.begin(), device_indices.end(), layer.vulkan_device_index);
            const size_t device = found == device_indices.end() ? 0 : static_cast<size_t>(found - device_indices.begin());
            uint64_t layer_total = 0;
            uint64_t layer_maximum = 0;
            std::vector<const ExpertPlan*> experts;
            experts.reserve(layer.moe.experts.size() + 1);
            if (layer.dense_ffn.use_gpu_cache)
                experts.push_back(&layer.dense_ffn);
            for (const ExpertPlan& expert : layer.moe.experts)
                experts.push_back(&expert);
            for (const ExpertPlan* expert_ptr : experts)
            {
                const ExpertPlan& expert = *expert_ptr;
                const TensorHandle gate_handle = expert.gate_up_weight != invalid_tensor_handle
                                                     ? expert.gate_up_weight
                                                     : expert.gate_weight;
                if (gate_handle == invalid_tensor_handle || expert.down_weight == invalid_tensor_handle
                    || (expert.gate_up_weight == invalid_tensor_handle && expert.up_weight == invalid_tensor_handle))
                {
                    continue;
                }

                const TensorData& gate_up = model.weights.at(gate_handle);
                const TensorData& down = model.weights.at(expert.down_weight);
                const TensorData* up = expert.gate_up_weight == invalid_tensor_handle
                                           ? &model.weights.at(expert.up_weight)
                                           : nullptr;
                const auto quantized = [&](const TensorData& matrix) {
                    return matrix.dtype == DType::Int4 || is_qnk_dtype(matrix.dtype)
                           || (model.qwen_device_chain && (matrix.dtype == DType::BFloat16 || matrix.dtype == DType::Float32 || matrix.dtype == DType::MxFp4));
                };
                if (expert.use_gpu_cache
                    || ((quantized(gate_up) || quantized(down) || (up && quantized(*up)))
                        && support_vulkan_expert(expert, gate_up, down, model.opt.optimization_flags, up)))
                {
                    const uint64_t bytes = native_quantized_expert_storage_bytes(gate_up, up, down,
                                                                                 device_indices[device],
                                                                                 model.vulkan_runtime,
                                                                                 model.opt.optimization_flags,
                                                                                 (up ? expert.gate_bias : expert.gate_up_bias) == invalid_tensor_handle
                                                                                     ? nullptr
                                                                                     : &model.weights.at(up ? expert.gate_bias : expert.gate_up_bias),
                                                                                 expert.up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.up_bias),
                                                                                 expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias),
                                                                                 expert.activation, expert.activation_limit);
                    invalid_storage_size = invalid_storage_size || bytes == 0;
                    if (expert.use_gpu_cache && bytes != 0)
                        smallest_dense_entry_size = std::min(smallest_dense_entry_size, bytes);
                    add_size(layer_total, bytes);
                    layer_maximum = std::max(layer_maximum, bytes);
                }
            }
            if (layer_maximum == 0)
                continue;
            add_size(total_sizes[device], layer_total);
            max_pair_sizes[device] = std::max(max_pair_sizes[device], layer_maximum);
        }
    };
    inspect_graph(model.graph);
    inspect_graph(model.speculative.graph);
    if (invalid_storage_size)
        return Error{ErrorCode::InvalidArgument, "resident quantized Expert GPU storage size is invalid or overflows"};

    std::vector<size_t> active_indices;
    std::vector<uint32_t> resident_devices;
    uint64_t maximum_pair_size = 0;
    for (size_t index = 0; index < device_indices.size(); ++index)
    {
        if (max_pair_sizes[index] == 0)
            continue;
        active_indices.push_back(index);
        resident_devices.push_back(device_indices[index]);
        maximum_pair_size = std::max(maximum_pair_size, max_pair_sizes[index]);
    }
    if (active_indices.empty())
    {
        if (native_quantized_experts && opt.expert_gpu_cache_size != 0)
            return Error{ErrorCode::UnsupportedModel, "the Expert GPU cache has no eligible quantized Experts"};
        return {};
    }

    // Query after dense and persistent attention uploads. Automatic admission
    // must not reserve against the stale loading-time free heap.
    const std::vector<GpuInfo> live_gpu_infos = get_gpu_infos();
    const bool live_budget_available = live_gpu_infos.size() == info.gpu_infos.size()
                                       && std::all_of(resident_devices.begin(), resident_devices.end(), [&](uint32_t device) {
                                              return device < live_gpu_infos.size();
                                          });
    const std::vector<GpuInfo>& gpu_infos = live_budget_available ? live_gpu_infos : info.gpu_infos;
    std::vector<uint64_t> cache_sizes(resident_devices.size());
    const bool explicit_cache = opt.expert_gpu_cache_size != 0;
    if (explicit_cache)
    {
        if (std::any_of(resident_devices.begin(), resident_devices.end(), [&](uint32_t device) { return device >= gpu_infos.size(); }))
            return Error{ErrorCode::InvalidArgument, "the Expert GPU cache references an unavailable Vulkan device"};
        const bool pure_dense = model.descriptor.expert_count == 0
                                && smallest_dense_entry_size != std::numeric_limits<uint64_t>::max();
        const uint64_t minimum_entry_size = pure_dense ? smallest_dense_entry_size : maximum_pair_size;
        if (pure_dense && opt.expert_gpu_cache_size / resident_devices.size() < minimum_entry_size)
            return {};
        auto distributed = distribute_gpu_cache_size(opt.expert_gpu_cache_size, minimum_entry_size,
                                                     resident_devices, gpu_infos, "the executable quantized Expert GPU cache");
        if (!distributed)
            return distributed.error();
        cache_sizes = std::move(distributed).value();
        for (size_t index = 0; index < cache_sizes.size(); ++index)
        {
            const uint64_t heap_budget = gpu_infos[resident_devices[index]].heap_budget;
            if (heap_budget == 0 || cache_sizes[index] > heap_budget)
                return Error{ErrorCode::InvalidArgument, "the quantized Expert GPU cache exceeds the Vulkan heap budget"};
            const auto& gpu = gpu_infos[resident_devices[index]];
            const uint64_t pending = qwen_device_pending_reserve(model, resident_devices[index]);
            if (pending != 0 && (gpu.heap_available <= pending || cache_sizes[index] > gpu.heap_available - pending))
                return Error{ErrorCode::InvalidArgument, "the quantized Expert GPU cache leaves insufficient memory for the Qwen device execution state"};
            cache_sizes[index] = std::min(cache_sizes[index], total_sizes[active_indices[index]]);
        }
    }
    else
    {
        if (!live_budget_available)
            return {};
        std::vector<GpuInfo> cache_gpu_infos = gpu_infos;
        for (uint32_t device : resident_devices)
        {
            GpuInfo& gpu = cache_gpu_infos[device];
            const uint64_t reserve = qwen_device_pending_reserve(model, device);
            gpu.heap_available = gpu.heap_available > reserve ? gpu.heap_available - reserve : 0;
        }
        if (smallest_dense_entry_size != std::numeric_limits<uint64_t>::max())
        {
            for (uint32_t device : resident_devices)
            {
                GpuInfo& gpu = cache_gpu_infos[device];
                const uint64_t reserve = dense_vulkan_workspace_reserve(gpu.heap_budget);
                gpu.heap_available = gpu.heap_available > reserve ? gpu.heap_available - reserve : 0;
                const uint64_t state = dense_vulkan_pending_state_size(model, device);
                gpu.heap_available = gpu.heap_available > state ? gpu.heap_available - state : 0;
            }
        }
        for (size_t index = 0; index < cache_sizes.size(); ++index)
        {
            const size_t device = active_indices[index];
            const auto available = get_auto_gpu_cache_sizes(max_pair_sizes[device], {resident_devices[index]},
                                                            cache_gpu_infos, opt.num_concurrent_sessions);
            if (available.size() != 1)
                continue;
            cache_sizes[index] = native_quantized_gpu_cache_size(available.front(), total_sizes[device],
                                                                 max_pair_sizes[device]);
        }
    }

    std::vector<std::shared_ptr<ExpertBackend>> backends;
    std::vector<uint32_t> cache_devices;
    uint64_t total_cache_size = 0;
    for (size_t index = 0; index < resident_devices.size(); ++index)
    {
        const uint64_t cache_size = cache_sizes[index];
        if (cache_size == 0)
            continue;
        if (cache_size > std::numeric_limits<uint64_t>::max() - total_cache_size)
            return Error{ErrorCode::InvalidArgument, "resident quantized Expert GPU capacities overflow"};
        auto backend = create_vulkan_expert_backend(cache_size, resident_devices[index], nullptr,
                                                    model.vulkan_runtime, model.opt.optimization_flags);
        if (!backend)
        {
            if (explicit_cache)
                return Error{ErrorCode::UnsupportedModel, "cannot create a Vulkan quantized Expert execution backend"};
            continue;
        }
        total_cache_size += cache_size;
        backends.push_back(std::move(backend));
        cache_devices.push_back(resident_devices[index]);
    }
    if (backends.empty())
        return {};
    std::vector<uint32_t> group_device_indices;
    group_device_indices.reserve(model.graph.layer_plans.size() + model.speculative.graph.layer_plans.size());
    for (const CompiledLayerPlan& layer : model.graph.layer_plans)
        group_device_indices.push_back(layer.vulkan_device_index);
    for (const CompiledLayerPlan& layer : model.speculative.graph.layer_plans)
        group_device_indices.push_back(layer.vulkan_device_index);
    model.expert_backend = create_multi_device_expert_backend(std::move(backends), std::move(cache_devices),
                                                              std::move(group_device_indices));
    if (!model.expert_backend)
        return Error{ErrorCode::UnsupportedModel, "cannot create the Vulkan quantized Expert execution backend"};
    model.opt.expert_gpu_cache_size = total_cache_size;
    return {};
}

Result<void> ModelLoader::load(const std::filesystem::path& model_path)
{
    auto ret = validate_option();
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
    ret = configure_resident_quantized_backend();
    if (!ret)
        return ret.error();
    if (model.qwen_device_chain && !model.expert_backend)
        return Error{ErrorCode::UnsupportedModel, "continuous Qwen Vulkan execution requires an executable GPU Expert cache within the remaining memory budget"};

    auto tokenizer = std::make_shared<Tokenizer>();
    if (tokenizer->load((package.weights_path.empty() ? package.root : package.weights_path).string(), model.descriptor.vocabulary_size))
        model.tokenizer = std::move(tokenizer);

    return {};
}

} // namespace moe
} // namespace ncnn
