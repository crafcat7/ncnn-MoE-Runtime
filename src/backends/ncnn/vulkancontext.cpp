#include "vulkancontext.h"
#include "kernels/activationbuffer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN

static bool has_batch_shape(
    const ncnn::VkMat& buffer,
    size_t rows,
    uint32_t columns,
    size_t element_size)
{
    return buffer.dims == 2
           && buffer.w == static_cast<int>(columns)
           && buffer.h == static_cast<int>(rows)
           && buffer.elemsize == element_size
           && buffer.elempack == 1;
}

bool prepare_staging_batch(
    ncnn::VkMat& buffer,
    size_t rows,
    uint32_t columns,
    ncnn::VkAllocator* allocator,
    VulkanRuntimeState& runtime_state,
    size_t element_size)
{
    const bool reused = has_batch_shape(buffer, rows, columns, element_size);
    buffer.create(
        static_cast<int>(columns),
        static_cast<int>(rows),
        element_size,
        allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++runtime_state.staging_slot_reuses;
    else
        ++runtime_state.staging_slot_resizes;
    return true;
}

bool prepare_staging_tensor(
    ncnn::VkMat& buffer,
    int width,
    int height,
    int channels,
    size_t element_size,
    ncnn::VkAllocator* allocator,
    VulkanRuntimeState& runtime_state)
{
    const bool reused = buffer.dims == 3
                        && buffer.w == width
                        && buffer.h == height
                        && buffer.c == channels
                        && buffer.elemsize == element_size
                        && buffer.elempack == 1;
    buffer.create(width, height, channels, element_size, allocator);
    if (buffer.empty() || !buffer.mapped_ptr())
        return false;
    if (reused)
        ++runtime_state.staging_slot_reuses;
    else
        ++runtime_state.staging_slot_resizes;
    return true;
}

bool record_mapped_upload(
    ncnn::VkMat& staging,
    ncnn::VkMat& destination,
    ncnn::VkCompute& command,
    const ncnn::Option& option)
{
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    command.record_clone(staging, destination, option);
    return !destination.empty();
}

static ncnn::Option activation_source_option(
    const ncnn::Option& option,
    DType source_dtype)
{
    ncnn::Option source_option = option;
    if (source_dtype == DType::BFloat16)
    {
        source_option.use_fp16_storage = false;
        source_option.use_fp16_packed = false;
        source_option.use_bf16_storage = true;
        source_option.use_bf16_packed = true;
    }
    else if (source_dtype == DType::Float16)
    {
        source_option.use_bf16_storage = false;
        source_option.use_bf16_packed = false;
        source_option.use_fp16_storage = true;
        source_option.use_fp16_packed = true;
    }
    return source_option;
}

bool record_mapped_activation_upload(
    ncnn::VkMat& staging,
    ncnn::VkMat& destination,
    ncnn::VkCompute& command,
    ncnn::VulkanDevice* device,
    const ncnn::Option& option,
    DType source_dtype)
{
    if (vulkan_activation_storage_variant(option) == 0)
    {
        if (source_dtype == DType::Float32)
            return record_mapped_upload(staging, destination, command, option);
        if (!device || staging.empty() || !staging.mapped_ptr())
            return false;
        staging.allocator->flush(staging.data);
        staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        const ncnn::Option source_option = activation_source_option(option, source_dtype);
        const int destination_elempack = staging.h % 4 == 0 ? 4 : 1;
        device->convert_packing(
            staging,
            destination,
            destination_elempack,
            1,
            command,
            source_option);
        return !destination.empty();
    }
    if (!device || staging.empty() || !staging.mapped_ptr())
        return false;
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    const int cast_type = option.use_bf16_storage ? 5 : 2;
    const ncnn::Option source_option = activation_source_option(option, source_dtype);
    device->convert_packing(
        staging,
        destination,
        1,
        cast_type,
        command,
        source_option);
    return !destination.empty()
           && destination.elemsize == vulkan_activation_element_size(option);
}

ncnn::VkMat bind_direct_host_input(
    ncnn::VkMat& staging,
    VulkanRuntimeState& runtime_state)
{
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    ++runtime_state.direct_host_input_bindings;
    return staging;
}

ncnn::VkMat prepare_direct_host_output(
    ncnn::VkMat& staging,
    VulkanRuntimeState& runtime_state)
{
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    ++runtime_state.direct_host_output_bindings;
    return staging;
}

bool fill_staging_upload(
    const ActivationBuffer& input,
    ncnn::VkMat& staging,
    ncnn::VkAllocator* allocator,
    VulkanRuntimeState& runtime_state)
{
    const size_t element_size = input.element_size();
    if (element_size == 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || input.bytes().size()
               != input.rows() * static_cast<size_t>(input.columns()) * element_size)
    {
        return false;
    }
    if (!prepare_staging_batch(
            staging,
            input.rows(),
            input.columns(),
            allocator,
            runtime_state,
            element_size))
    {
        return false;
    }

    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty() || mapped.total() * mapped.elemsize < input.bytes().size())
        return false;
    std::memcpy(mapped.data, input.bytes().data(), input.bytes().size());
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

bool fill_staging_values(
    const void* source,
    size_t count,
    size_t element_size,
    ncnn::VkMat& staging,
    ncnn::VkAllocator* allocator,
    VulkanRuntimeState& runtime_state)
{
    if (!source
        || count == 0
        || element_size == 0
        || count > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    if (!prepare_staging_batch(
            staging,
            1,
            static_cast<uint32_t>(count),
            allocator,
            runtime_state,
            element_size))
    {
        return false;
    }
    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty() || mapped.total() < count)
        return false;
    std::memcpy(mapped.data, source, count * element_size);
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

bool record_prepared_staging_upload(
    const ncnn::VkMat& staging,
    size_t rows,
    ncnn::VkMat& destination,
    ncnn::VkCompute& command,
    ncnn::VulkanDevice* device,
    const ncnn::Option& option,
    DType source_dtype)
{
    if (!device || staging.empty() || !staging.mapped_ptr())
        return false;
    const int packed_rows = static_cast<int>(rows);
    const int destination_elempack = packed_rows % 4 == 0 && (option.use_packing_layout || packed_rows <= 4) ? 4 : 1;
    int cast_type = source_dtype == DType::Float32 ? 0 : 1;
    if (option.use_bf16_storage || option.use_bf16_packed)
        cast_type = 5;
    else if (option.use_fp16_storage || option.use_fp16_packed)
        cast_type = 2;
    else if (device->info.type() != 0)
    {
        cast_type = 1;
    }
    const ncnn::Option source_option = activation_source_option(option, source_dtype);
    device->convert_packing(
        staging,
        destination,
        destination_elempack,
        cast_type,
        command,
        source_option);
    return !destination.empty();
}

bool record_prepared_staging_download(
    const ncnn::VkMat& source,
    size_t rows,
    uint32_t columns,
    ncnn::VkMat& staging,
    ncnn::VkCompute& command,
    const ncnn::Option& option)
{
    if (!has_batch_shape(staging, rows, columns, sizeof(float)))
        return false;
    ncnn::Option staging_option = option;
    staging_option.blob_vkallocator = staging.allocator;
    staging_option.workspace_vkallocator = staging.allocator;
    staging_option.staging_vkallocator = staging.allocator;
    command.record_clone(source, staging, staging_option);
    return !staging.empty();
}

bool record_prepared_activation_staging_download(
    const ncnn::VkMat& source,
    size_t rows,
    uint32_t columns,
    ncnn::VkMat& staging,
    ncnn::VkCompute& command,
    ncnn::VulkanDevice* device,
    const ncnn::Option& option,
    DType output_dtype)
{
    const size_t output_element_size = output_dtype == DType::Float32
                                           ? sizeof(float)
                                           : (output_dtype == DType::Float16 || output_dtype == DType::BFloat16
                                                  ? sizeof(uint16_t)
                                                  : 0);
    if (output_element_size == 0)
        return false;

    const int output_cast_type = output_dtype == DType::Float32
                                     ? 1
                                 : output_dtype == DType::Float16 ? 2
                                                                  : 5;
    const bool source_matches_cpu_batch = source.dims == 2
                                          && source.w == static_cast<int>(columns)
                                          && source.h == static_cast<int>(rows)
                                          && source.elempack == 1
                                          && source.elemsize == output_element_size;
    if (vulkan_activation_storage_variant(option) == 0
        && output_dtype == DType::Float32
        && source_matches_cpu_batch)
    {
        return record_prepared_staging_download(
            source,
            rows,
            columns,
            staging,
            command,
            option);
    }
    if (!device)
        return false;
    if (!staging.allocator)
        return false;
    staging.create(
        static_cast<int>(columns),
        static_cast<int>(rows),
        output_element_size,
        staging.allocator);
    if (staging.empty() || !staging.mapped_ptr())
        return false;
    if (!has_batch_shape(staging, rows, columns, output_element_size))
        return false;
    ncnn::Option staging_option = option;
    staging_option.blob_vkallocator = staging.allocator;
    staging_option.workspace_vkallocator = staging.allocator;
    staging_option.staging_vkallocator = staging.allocator;
    device->convert_packing(
        source,
        staging,
        1,
        output_cast_type,
        command,
        staging_option);
    const bool matches = has_batch_shape(
        staging,
        rows,
        columns,
        output_element_size);
    return matches;
}

bool copy_staging_to_cpu_batch(
    ncnn::VkMat& staging,
    ActivationBuffer& output)
{
    const size_t element_size = output.element_size();
    if (element_size == 0 || staging.elemsize != element_size)
        return false;
    staging.allocator->invalidate(staging.data);
    const ncnn::Mat mapped = staging.mapped();
    if (mapped.empty()
        || mapped.dims != 2
        || mapped.w != static_cast<int>(output.columns())
        || mapped.h != static_cast<int>(output.rows())
        || mapped.elempack != 1
        || mapped.elemsize != element_size)
    {
        return false;
    }
    const size_t row_bytes = static_cast<size_t>(output.columns()) * element_size;
    const auto* source = static_cast<const std::byte*>(mapped.data);
    for (size_t row_index = 0; row_index < output.rows(); ++row_index)
        std::memcpy(
            output.mutable_row_bytes(row_index).data(),
            source + row_index * row_bytes,
            row_bytes);
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

bool copy_staging_to_cpu_batches(
    ncnn::VkMat& staging,
    std::span<const ActivationBuffer*> inputs,
    std::span<ActivationBuffer*> outputs,
    uint32_t columns)
{
    if (inputs.empty()
        || inputs.size() != outputs.size()
        || columns == 0
        || staging.empty()
        || !staging.allocator
        || !staging.data)
    {
        return false;
    }

    ActivationBuffer* first_output = outputs.front();
    if (!first_output)
        return false;
    const size_t element_size = first_output->element_size();
    if (element_size == 0
        || static_cast<size_t>(columns)
               > std::numeric_limits<size_t>::max() / element_size)
    {
        return false;
    }
    const size_t row_bytes = static_cast<size_t>(columns) * element_size;

    size_t total_rows = 0;
    for (size_t index = 0; index < inputs.size(); ++index)
    {
        const ActivationBuffer* input = inputs[index];
        ActivationBuffer* output = outputs[index];
        if (!input
            || !output
            || input->rows() == 0
            || output->dtype() != first_output->dtype()
            || output->element_size() != element_size
            || input->rows() > std::numeric_limits<size_t>::max() / row_bytes
            || total_rows
                   > std::numeric_limits<size_t>::max() - input->rows())
        {
            return false;
        }
        for (size_t previous = 0; previous < index; ++previous)
        {
            if (output == outputs[previous])
                return false;
        }
        for (size_t other_index = 0; other_index < inputs.size(); ++other_index)
        {
            if (other_index != index && output == inputs[other_index])
                return false;
        }
        total_rows += input->rows();
    }
    if (total_rows == 0
        || total_rows > static_cast<size_t>(std::numeric_limits<int>::max())
        || total_rows > std::numeric_limits<size_t>::max() / row_bytes
        || columns > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || !has_batch_shape(staging, total_rows, columns, element_size))
    {
        return false;
    }

    staging.allocator->invalidate(staging.data);
    const ncnn::Mat mapped = staging.mapped();
    if (mapped.empty()
        || mapped.dims != 2
        || mapped.w != static_cast<int>(columns)
        || mapped.h != static_cast<int>(total_rows)
        || mapped.elempack != 1
        || mapped.elemsize != element_size)
    {
        return false;
    }

    for (size_t index = 0; index < outputs.size(); ++index)
        outputs[index]->reset(inputs[index]->rows(), columns, false);

    const auto* source = static_cast<const std::byte*>(mapped.data);
    size_t row_offset = 0;
    for (size_t index = 0; index < outputs.size(); ++index)
    {
        ActivationBuffer& output = *outputs[index];
        const size_t output_bytes = output.rows() * row_bytes;
        std::memcpy(
            output.mutable_bytes().data(),
            source + row_offset * row_bytes,
            output_bytes);
        row_offset += output.rows();
    }
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

bool prepare_float_tensor_upload(
    const TensorData& source,
    ncnn::Mat& destination)
{
    if (source.element_count() == 0
        || source.element_count()
               > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    destination.create(static_cast<int>(source.element_count()), sizeof(float));
    if (destination.empty())
        return false;
    float* values = static_cast<float*>(destination.data);
    if (source.dtype == DType::Float32)
    {
        const std::span<const float> source_values = source.float32_values();
        if (source_values.size() != source.element_count())
            return false;
        std::copy(source_values.begin(), source_values.end(), values);
        return true;
    }
    if (source.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> source_values = source.bfloat16_values();
        if (source_values.size() != source.element_count())
            return false;
        for (size_t index = 0; index < source_values.size(); ++index)
            values[index] = ncnn::bfloat16_to_float32(source_values[index]);
        return true;
    }
    return false;
}

#endif // NCNN_MOE_WITH_VULKAN

VulkanRuntimePtr create_vulkan_runtime()
{
    return std::make_shared<VulkanRuntime>();
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

VulkanStatistics get_vulkan_statistics(
    const VulkanRuntimePtr& vulkan_runtime) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return vulkan_runtime ? vulkan_runtime->state.snapshot()
                          : VulkanStatistics{};
#else
    (void)vulkan_runtime;
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

VulkanStatistics VulkanRuntimeState::snapshot() const noexcept
{
    VulkanStatistics result;
    result.dispatches = dispatches.load();
    result.attention_blocks = attention_blocks.load();
    result.compute_submissions = compute_submissions.load();
    result.submit_wait_time_microseconds = submit_wait_time_microseconds.load();
    result.batch_uploads = batch_uploads.load();
    result.batch_downloads = batch_downloads.load();
    result.auxiliary_uploads = auxiliary_uploads.load();
    result.auxiliary_upload_bytes = auxiliary_upload_bytes.load();
    result.staging_slot_resizes = staging_slot_resizes.load();
    result.staging_slot_reuses = staging_slot_reuses.load();
    result.staging_slot_acquisitions = staging_slot_acquisitions.load();
    result.staging_slot_contentions = staging_slot_contentions.load();
    result.command_buffer_reuses = command_buffer_reuses.load();
    result.command_graph_submissions = command_graph_submissions.load();
    result.command_graph_operations = command_graph_operations.load();
    result.direct_host_input_bindings = direct_host_input_bindings.load();
    result.direct_host_output_bindings = direct_host_output_bindings.load();
    result.attention_qkv_rope_fusions = attention_qkv_rope_fusions.load();
    result.attention_device_rope_fusions = attention_device_rope_fusions.load();
    result.attention_qkv_ring_fusions = attention_qkv_ring_fusions.load();
    result.attention_decode_sdpa_fusions = attention_decode_sdpa_fusions.load();
    result.attention_cache_materializations = attention_cache_materializations.load();
    result.attention_cpu_fallbacks = attention_cpu_fallbacks.load();
    result.shared_expert_swiglu_fusions = shared_expert_swiglu_fusions.load();
    result.gated_delta_fusions = gated_delta_fusions.load();
    result.gated_delta_submissions = gated_delta_submissions.load();
    result.rms_norm_linear_fusions = rms_norm_linear_fusions.load();
    result.kv_ring_appends = kv_ring_appends.load();
    result.kv_ring_resizes = kv_ring_resizes.load();
    result.kv_ring_wrapped_views = kv_ring_wrapped_views.load();
    result.kv_cache_promotions = kv_cache_promotions.load();
    result.kv_cache_promotion_bytes = kv_cache_promotion_bytes.load();
    result.bfloat16_cooperative_matrix_dispatches = bfloat16_cooperative_matrix_dispatches.load();
    result.command_dispatches = command_dispatches.load();
    result.command_pipeline_binds = command_pipeline_binds.load();
    result.command_redundant_pipeline_binds = command_redundant_pipeline_binds.load();
    result.command_descriptor_bindings = command_descriptor_bindings.load();
    result.command_push_constant_updates = command_push_constant_updates.load();
    result.command_resource_barrier_calls = command_resource_barrier_calls.load();
    result.command_buffer_resource_barriers = command_buffer_resource_barriers.load();
    result.command_image_resource_barriers = command_image_resource_barriers.load();
    return result;
}

int submit_compute_and_wait(
    ncnn::VkCompute& command,
    VulkanRuntimeState& runtime_state)
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

VulkanTransferLease::VulkanTransferLease(VulkanTransferSlot& slot, std::unique_lock<std::mutex> _lock)
    : transfer_slot(&slot), lock(std::move(_lock))
{
}

VulkanContext::~VulkanContext()
{
    const std::lock_guard<std::mutex> lock(command_lock);
    for (VulkanTransferSlot& slot : transfer_slots)
    {
        delete slot.command;
        slot.command = nullptr;
    }
}

std::shared_ptr<VulkanContext> VulkanContext::acquire(
    uint32_t requested_device_index,
    const VulkanRuntimePtr& vulkan_runtime,
    uint64_t optimization_flags)
{
    if (!vulkan_runtime)
        return {};
    {
        const std::lock_guard<std::mutex> lock(
            vulkan_runtime->initialization_mutex);
        if (!vulkan_runtime->initialization_attempted)
        {
            vulkan_runtime->initialization_attempted = true;
#if defined(__APPLE__) && defined(NCNN_MOE_MOLTENVK_LIBRARY_PATH)
            vulkan_runtime->instance_ready = ncnn::create_gpu_instance(NCNN_MOE_MOLTENVK_LIBRARY_PATH) == 0 && ncnn::get_gpu_count() > 0;
#else
            vulkan_runtime->instance_ready = ncnn::create_gpu_instance() == 0 && ncnn::get_gpu_count() > 0;
#endif
        }
    }
    if (!vulkan_runtime->instance_ready)
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
    const VulkanContextCacheKey context_key{
        device_index,
        optimization_flags};
    const std::lock_guard<std::mutex> lock(
        vulkan_runtime->context_mutex);
    const auto existing = vulkan_runtime->contexts.find(context_key);
    if (existing != vulkan_runtime->contexts.end())
    {
        if (const std::shared_ptr<VulkanContext> context = existing->second.lock())
        {
            return context;
        }
        vulkan_runtime->contexts.erase(existing);
    }
    auto context = std::shared_ptr<VulkanContext>(new VulkanContext(
        device,
        vulkan_runtime,
        optimization_flags,
        command_flags));
    vulkan_runtime->contexts.emplace(context_key, context);
    return context;
}

std::shared_ptr<const std::vector<uint32_t>> VulkanContext::shader_binary(
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

std::shared_ptr<ncnn::Pipeline> VulkanContext::find_pipeline(
    const char* source,
    uint64_t variant) const
{
    const std::lock_guard<std::mutex> lock(pipeline_cache_mutex);
    const auto cached = pipelines.find({source, variant});
    if (cached == pipelines.end())
        return {};
    return cached->second.lock();
}

void VulkanContext::cache_pipeline(
    const char* source,
    uint64_t variant,
    const std::shared_ptr<ncnn::Pipeline>& pipeline)
{
    const std::lock_guard<std::mutex> lock(pipeline_cache_mutex);
    pipelines[{source, variant}] = pipeline;
}

VulkanTransferLease VulkanContext::acquire_transfer_slot()
{
    const size_t slot_index = next_transfer_slot.fetch_add(1, std::memory_order_relaxed) % transfer_slots.size();
    VulkanTransferSlot& slot = transfer_slots[slot_index];
    std::unique_lock<std::mutex> lock(slot.mutex, std::try_to_lock);
    if (!lock.owns_lock())
    {
        ++runtime_state().staging_slot_contentions;
        lock.lock();
    }
    ++runtime_state().staging_slot_acquisitions;
    return VulkanTransferLease(slot, std::move(lock));
}

VulkanContext::VulkanContext(
    ncnn::VulkanDevice* device,
    VulkanRuntimePtr _vulkan_runtime,
    uint64_t optimization_flags,
    uint32_t command_optimization_flags)
    : vkdev(device),
      vulkan_runtime(std::move(_vulkan_runtime)),
      flags(optimization_flags),
      command_flags(command_optimization_flags),
      blob_alloc(device->acquire_blob_allocator()),
      staging_alloc(device->acquire_staging_allocator())
{
    for (VulkanTransferSlot& slot : transfer_slots)
        slot.staging_allocator = device->acquire_staging_allocator();
    for (VulkanTransferSlot& slot : transfer_slots)
    {
        slot.command = new ncnn::VkCompute(device, command_flags);
    }
}

VulkanWeightUploadBatch::VulkanWeightUploadBatch(const std::shared_ptr<VulkanContext>& _context)
    : context(_context),
      cmd(_context ? _context->device() : nullptr),
      command_lock(_context ? std::unique_lock<std::mutex>(_context->command_mutex()) : std::unique_lock<std::mutex>())
{
}

bool VulkanWeightUploadBatch::record(
    const ncnn::Mat& source,
    ncnn::VkMat& destination,
    const ncnn::Option& option,
    ncnn::VkAllocator* weight_allocator)
{
    if (!context || !weight_allocator || source.empty())
        return false;
    if (!staging_allocator)
        staging_allocator = std::make_unique<ncnn::VkWeightStagingAllocator>(context->device());
    if (!staging_allocator)
        return false;
    ncnn::Option upload_option = option;
    upload_option.blob_vkallocator = weight_allocator;
    upload_option.workspace_vkallocator = weight_allocator;
    upload_option.staging_vkallocator = staging_allocator.get();
    cmd.record_upload(source, destination, upload_option);
    if (destination.empty())
        return false;
    return true;
}

bool VulkanWeightUploadBatch::submit()
{
    return context && cmd.submit_and_wait() == 0;
}
#endif // NCNN_MOE_WITH_VULKAN

} // namespace moe
} // namespace ncnn
