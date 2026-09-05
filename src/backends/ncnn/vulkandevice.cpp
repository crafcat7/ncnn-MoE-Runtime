#include "vulkandevice.h"

#include <chrono>
#include <utility>

namespace ncnn {
namespace moe {

NcnnVulkanContextInstancePtr create_ncnn_vulkan_context_instance()
{
    return std::make_shared<NcnnVulkanContextInstance>();
}

uint32_t get_gpu_count() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::create_gpu_instance() != 0)
        return 0;
    return static_cast<uint32_t>(ncnn::get_gpu_count());
#else
    return 0;
#endif
}

uint32_t get_default_gpu_index() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::create_gpu_instance() != 0 || ncnn::get_gpu_count() == 0)
        return 0;
    return static_cast<uint32_t>(ncnn::get_default_gpu_index());
#else
    return 0;
#endif
}

std::vector<GpuInfo> get_gpu_infos()
{
    std::vector<GpuInfo> infos;
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::create_gpu_instance() != 0)
        return infos;
    const int gpu_count = ncnn::get_gpu_count();
    infos.reserve(static_cast<size_t>(gpu_count));
    for (int i = 0; i < gpu_count; ++i)
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
        GpuInfo gpu_info;
        gpu_info.device_index = static_cast<uint32_t>(i);
        gpu_info.vendor_id = info.vendor_id();
        gpu_info.device_id = info.device_id();
        gpu_info.device_name = info.device_name();
        gpu_info.rough_score = info.rough_score();
        gpu_info.compute_queue_count = info.compute_queue_count();
        gpu_info.transfer_queue_count = info.transfer_queue_count();
        switch (info.type())
        {
        case 0: gpu_info.type = VulkanDeviceType::Discrete; break;
        case 1: gpu_info.type = VulkanDeviceType::Integrated; break;
        case 2: gpu_info.type = VulkanDeviceType::Virtual; break;
        case 3: gpu_info.type = VulkanDeviceType::Cpu; break;
        default: gpu_info.type = VulkanDeviceType::Unknown; break;
        }
        if (info.support_fp16_storage())
            gpu_info.flags |= VulkanDeviceFp16Storage;
        if (info.support_fp16_arithmetic())
            gpu_info.flags |= VulkanDeviceFp16Arithmetic;
        if (info.support_bf16_storage())
            gpu_info.flags |= VulkanDeviceBf16Storage;
        if (info.support_int8_storage())
            gpu_info.flags |= VulkanDeviceInt8Storage;
        if (info.support_int8_arithmetic())
            gpu_info.flags |= VulkanDeviceInt8Arithmetic;
        if (info.support_VK_KHR_shader_integer_dot_product())
            gpu_info.flags |= VulkanDeviceIntegerDotProduct;
        if (info.support_subgroup_ops() != 0)
            gpu_info.flags |= VulkanDeviceSubgroupOperations;
        if (info.support_cooperative_matrix())
            gpu_info.flags |= VulkanDeviceCooperativeMatrix;
        if (info.support_int8_cooperative_matrix())
            gpu_info.flags |= VulkanDeviceInt8CooperativeMatrix;
        if (info.unified_compute_transfer_queue())
            gpu_info.flags |= VulkanDeviceUnifiedComputeTransfer;
        if (info.resizable_bar_enabled())
            gpu_info.flags |= VulkanDeviceResizableBar;
        ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(i);
        if (vkdev)
        {
            gpu_info.heap_budget = static_cast<uint64_t>(vkdev->get_heap_budget()) * 1024 * 1024;
            gpu_info.heap_usage = static_cast<uint64_t>(vkdev->get_heap_usage()) * 1024 * 1024;
            gpu_info.heap_available = gpu_info.heap_usage < gpu_info.heap_budget
                                          ? gpu_info.heap_budget - gpu_info.heap_usage
                                          : 0;
        }
        infos.push_back(std::move(gpu_info));
    }
#endif
    return infos;
}

NcnnVulkanExecutionSnapshot get_vulkan_execution_snapshot(
    const NcnnVulkanContextInstancePtr& context_instance) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return context_instance ? context_instance->state.snapshot()
                            : NcnnVulkanExecutionSnapshot{};
#else
    (void)context_instance;
    return {};
#endif
}

#if NCNN_MOE_WITH_VULKAN
static uint32_t vulkan_command_optimization_flags(uint64_t optimization_flags) noexcept
{
    uint32_t flags = 0;
    if (has_flag(
            optimization_flags,
            OptimizationVulkanPipelineBindElision))
    {
        flags |= ncnn::VkComputeOptimizationPipelineBindElision;
    }
    if (has_flag(
            optimization_flags,
            OptimizationVulkanReadonlyBindings))
    {
        flags |= ncnn::VkComputeOptimizationReadonlyBindings;
    }
    if (has_flag(
            optimization_flags,
            OptimizationVulkanBatchBufferBarriers))
    {
        flags |= ncnn::VkComputeOptimizationBatchBufferBarriers;
    }
    if (has_flag(
            optimization_flags,
            OptimizationVulkanStackDescriptorPayload))
    {
        flags |= ncnn::VkComputeOptimizationStackDescriptorPayload;
    }
    return flags;
}

NcnnVulkanExecutionSnapshot NcnnVulkanRuntimeState::snapshot() const noexcept
{
    NcnnVulkanExecutionSnapshot result;
    result.dispatches = dispatches.load();
    result.attention_blocks = attention_blocks.load();
    result.counters.compute_submissions = compute_submissions.load();
    result.counters.submit_wait_time_microseconds = submit_wait_time_microseconds.load();
    result.counters.batch_uploads = batch_uploads.load();
    result.counters.batch_downloads = batch_downloads.load();
    result.counters.auxiliary_uploads = auxiliary_uploads.load();
    result.counters.auxiliary_upload_bytes = auxiliary_upload_bytes.load();
    result.counters.staging_slot_resizes = staging_slot_resizes.load();
    result.counters.staging_slot_reuses = staging_slot_reuses.load();
    result.counters.staging_slot_acquisitions = staging_slot_acquisitions.load();
    result.counters.staging_slot_contentions = staging_slot_contentions.load();
    result.counters.command_buffer_reuses = command_buffer_reuses.load();
    result.counters.command_graph_submissions = command_graph_submissions.load();
    result.counters.command_graph_operations = command_graph_operations.load();
    result.counters.direct_host_input_bindings = direct_host_input_bindings.load();
    result.counters.direct_host_output_bindings = direct_host_output_bindings.load();
    result.counters.attention_qkv_rope_fusions = attention_qkv_rope_fusions.load();
    result.counters.attention_device_rope_fusions = attention_device_rope_fusions.load();
    result.counters.attention_qkv_ring_fusions = attention_qkv_ring_fusions.load();
    result.counters.attention_qkv_rope_pipeline_failures = attention_qkv_rope_pipeline_failures.load();
    result.counters.attention_qkv_rope_shape_failures = attention_qkv_rope_shape_failures.load();
    result.counters.attention_qkv_rope_source_failures = attention_qkv_rope_source_failures.load();
    result.counters.attention_qkv_rope_norm_failures = attention_qkv_rope_norm_failures.load();
    result.counters.attention_qkv_rope_ring_failures = attention_qkv_rope_ring_failures.load();
    result.counters.attention_qkv_rope_allocation_failures = attention_qkv_rope_allocation_failures.load();
    result.counters.attention_precondition_failures = attention_precondition_failures.load();
    result.counters.attention_staging_failures = attention_staging_failures.load();
    result.counters.attention_norm_failures = attention_norm_failures.load();
    result.counters.attention_qkv_failures = attention_qkv_failures.load();
    result.counters.attention_cache_failures = attention_cache_failures.load();
    result.counters.attention_sdpa_failures = attention_sdpa_failures.load();
    result.counters.attention_projection_failures = attention_projection_failures.load();
    result.counters.attention_output_failures = attention_output_failures.load();
    result.counters.attention_submit_failures = attention_submit_failures.load();
    result.counters.attention_decode_sdpa_fusions = attention_decode_sdpa_fusions.load();
    result.counters.attention_cache_materializations = attention_cache_materializations.load();
    result.counters.attention_cpu_fallbacks = attention_cpu_fallbacks.load();
    result.counters.shared_expert_swiglu_fusions = shared_expert_swiglu_fusions.load();
    result.counters.gated_delta_fusions = gated_delta_fusions.load();
    result.counters.gated_delta_submissions = gated_delta_submissions.load();
    result.counters.rms_norm_linear_fusions = rms_norm_linear_fusions.load();
    result.counters.kv_ring_appends = kv_ring_appends.load();
    result.counters.kv_ring_resizes = kv_ring_resizes.load();
    result.counters.kv_ring_wrapped_views = kv_ring_wrapped_views.load();
    result.counters.kv_cache_promotions = kv_cache_promotions.load();
    result.counters.kv_cache_promotion_bytes = kv_cache_promotion_bytes.load();
    result.counters.bfloat16_cooperative_matrix_dispatches = bfloat16_cooperative_matrix_dispatches.load();
    result.counters.command_dispatches = command_dispatches.load();
    result.counters.command_pipeline_binds = command_pipeline_binds.load();
    result.counters.command_redundant_pipeline_binds = command_redundant_pipeline_binds.load();
    result.counters.command_descriptor_bindings = command_descriptor_bindings.load();
    result.counters.command_push_constant_updates = command_push_constant_updates.load();
    result.counters.command_resource_barrier_calls = command_resource_barrier_calls.load();
    result.counters.command_buffer_resource_barriers = command_buffer_resource_barriers.load();
    result.counters.command_image_resource_barriers = command_image_resource_barriers.load();
    return result;
}

int submit_compute_and_wait(
    ncnn::VkCompute& command,
    NcnnVulkanRuntimeState& runtime_state)
{
    const ncnn::VkComputeCommandStatistics command_recording = command.command_statistics();
    const auto started = std::chrono::steady_clock::now();
    const int result = command.submit_and_wait();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started);
    runtime_state.submit_wait_time_microseconds += static_cast<uint64_t>(elapsed.count());
    runtime_state.command_dispatches += command_recording.dispatches;
    runtime_state.command_pipeline_binds += command_recording.pipeline_binds;
    runtime_state.command_redundant_pipeline_binds += command_recording.redundant_pipeline_binds;
    runtime_state.command_descriptor_bindings += command_recording.descriptor_bindings;
    runtime_state.command_push_constant_updates += command_recording.push_constant_updates;
    runtime_state.command_resource_barrier_calls += command_recording.resource_barrier_calls;
    runtime_state.command_buffer_resource_barriers += command_recording.buffer_resource_barriers;
    runtime_state.command_image_resource_barriers += command_recording.image_resource_barriers;
    return result;
}

NcnnVulkanTransferLease::NcnnVulkanTransferLease(NcnnVulkanTransferSlot& slot, std::unique_lock<std::mutex> _lock)
    : transfer_slot(&slot), lock(std::move(_lock))
{
}

NcnnVulkanContext::~NcnnVulkanContext()
{
    const std::lock_guard<std::mutex> lock(command_lock);
    for (NcnnVulkanTransferSlot& slot : transfer_slots)
    {
        delete slot.command;
        slot.command = nullptr;
    }
}

std::shared_ptr<NcnnVulkanContext> NcnnVulkanContext::acquire(
    uint32_t requested_device_index,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags)
{
    if (!context_instance)
        return {};
    {
        const std::lock_guard<std::mutex> lock(
            context_instance->initialization_mutex);
        if (!context_instance->initialization_attempted)
        {
            context_instance->initialization_attempted = true;
#if defined(__APPLE__) && defined(NCNN_MOE_MOLTENVK_LIBRARY_PATH)
            context_instance->instance_ready = ncnn::create_gpu_instance(NCNN_MOE_MOLTENVK_LIBRARY_PATH) == 0 && ncnn::get_gpu_count() > 0;
#else
            context_instance->instance_ready = ncnn::create_gpu_instance() == 0 && ncnn::get_gpu_count() > 0;
#endif
        }
    }
    if (!context_instance->instance_ready)
        return {};

    const uint32_t device_index = requested_device_index == automatic_vulkan_device_index ? static_cast<uint32_t>(ncnn::get_default_gpu_index()) : requested_device_index;
    if (device_index >= static_cast<uint32_t>(ncnn::get_gpu_count()))
    {
        return {};
    }
    const uint32_t command_flags = vulkan_command_optimization_flags(optimization_flags);
    ncnn::VulkanDevice* device = ncnn::get_gpu_device(static_cast<int>(device_index));
    if (!device)
        return {};
    const NcnnVulkanContextCacheKey context_key{
        device_index,
        optimization_flags};
    const std::lock_guard<std::mutex> lock(
        context_instance->context_mutex);
    const auto existing = context_instance->contexts.find(context_key);
    if (existing != context_instance->contexts.end())
    {
        if (const std::shared_ptr<NcnnVulkanContext> context = existing->second.lock())
        {
            return context;
        }
        context_instance->contexts.erase(existing);
    }
    auto context = std::shared_ptr<NcnnVulkanContext>(new NcnnVulkanContext(
        device,
        context_instance,
        optimization_flags,
        command_flags));
    context_instance->contexts.emplace(context_key, context);
    return context;
}

std::shared_ptr<const std::vector<uint32_t>> NcnnVulkanContext::shader_binary(
    const char* source,
    int source_length,
    const ncnn::Option& option,
    uint64_t variant)
{
    if (!source || source_length <= 0)
        return {};
    const ShaderCacheKey key{source, variant};
    const std::lock_guard<std::mutex> lock(pipeline_cache_mutex);
    const auto cached = shader_binaries.find(key);
    if (cached != shader_binaries.end())
        return cached->second;
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(
            source,
            source_length,
            option,
            spirv)
            != 0
        || spirv.empty())
    {
        const std::shared_ptr<const std::vector<uint32_t>> failed = std::make_shared<const std::vector<uint32_t>>();
        shader_binaries.emplace(key, failed);
        return failed;
    }
    const std::shared_ptr<const std::vector<uint32_t>> binary = std::make_shared<const std::vector<uint32_t>>(std::move(spirv));
    shader_binaries.emplace(key, binary);
    return binary;
}

std::shared_ptr<ncnn::Pipeline> NcnnVulkanContext::find_pipeline(
    const char* source,
    uint64_t variant) const
{
    const std::lock_guard<std::mutex> lock(pipeline_cache_mutex);
    const auto cached = pipelines.find({source, variant});
    if (cached == pipelines.end())
        return {};
    return cached->second.lock();
}

void NcnnVulkanContext::cache_pipeline(
    const char* source,
    uint64_t variant,
    const std::shared_ptr<ncnn::Pipeline>& pipeline)
{
    const std::lock_guard<std::mutex> lock(pipeline_cache_mutex);
    pipelines[{source, variant}] = pipeline;
}

NcnnVulkanTransferLease NcnnVulkanContext::acquire_transfer_slot()
{
    const size_t slot_index = next_transfer_slot.fetch_add(1, std::memory_order_relaxed) % transfer_slots.size();
    NcnnVulkanTransferSlot& slot = transfer_slots[slot_index];
    std::unique_lock<std::mutex> lock(slot.mutex, std::try_to_lock);
    if (!lock.owns_lock())
    {
        ++runtime_state().staging_slot_contentions;
        lock.lock();
    }
    ++runtime_state().staging_slot_acquisitions;
    return NcnnVulkanTransferLease(slot, std::move(lock));
}

NcnnVulkanContext::NcnnVulkanContext(
    ncnn::VulkanDevice* device,
    NcnnVulkanContextInstancePtr _context_instance,
    uint64_t optimization_flags,
    uint32_t command_optimization_flags)
    : vkdev(device),
      context_instance(std::move(_context_instance)),
      flags(optimization_flags),
      command_flags(command_optimization_flags),
      blob_alloc(device->acquire_blob_allocator()),
      staging_alloc(device->acquire_staging_allocator())
{
    for (NcnnVulkanTransferSlot& slot : transfer_slots)
        slot.staging_allocator = device->acquire_staging_allocator();
    for (NcnnVulkanTransferSlot& slot : transfer_slots)
    {
        slot.command = new ncnn::VkCompute(device, command_flags);
    }
}

NcnnVulkanWeightUploadBatch::NcnnVulkanWeightUploadBatch(const std::shared_ptr<NcnnVulkanContext>& _context)
    : context(_context),
      cmd(_context ? _context->device() : nullptr),
      command_lock(_context ? std::unique_lock<std::mutex>(_context->command_mutex()) : std::unique_lock<std::mutex>())
{
}

bool NcnnVulkanWeightUploadBatch::record(
    const ncnn::Mat& source,
    ncnn::VkMat& destination,
    const ncnn::Option& option,
    ncnn::VkAllocator* weight_allocator)
{
    if (!context || !weight_allocator || source.empty())
        return false;
    auto staging_allocator = std::make_unique<ncnn::VkWeightStagingAllocator>(context->device());
    if (!staging_allocator)
        return false;
    ncnn::Option upload_option = option;
    upload_option.blob_vkallocator = weight_allocator;
    upload_option.workspace_vkallocator = weight_allocator;
    upload_option.staging_vkallocator = staging_allocator.get();
    cmd.record_upload(source, destination, upload_option);
    if (destination.empty())
        return false;
    staging_allocators.push_back(std::move(staging_allocator));
    return true;
}

bool NcnnVulkanWeightUploadBatch::submit()
{
    return context && cmd.submit_and_wait() == 0;
}
#endif // NCNN_MOE_WITH_VULKAN

} // namespace moe
} // namespace ncnn
