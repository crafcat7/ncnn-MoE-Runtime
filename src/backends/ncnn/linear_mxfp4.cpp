#include "linear.h"
#include "sharedffn_vulkan.h"
#include "mxfp4_weightpacking.h"
#include "rmsnorm_vulkan.h"
#include "vulkancontext.h"
#include "experttransfer_vulkan.h"
#include "expertbackend_vulkan.h"
#include "innerproductweights_vulkan.h"
#include "moecombine_vulkan.h"
#include "residentffn_vulkan.h"

#include "ncnn/moe/runtime.h"
#include "engine/expertbackend.h"
#include "kernels/float8.h"
#include "kernels/int4.h"
#include "kernels/mxfp4.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"

#if NCNN_MOE_WITH_VULKAN
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#if NCNN_MOE_USE_NCNN
#include <layer.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif
#endif

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/expert_gather.comp.hex.h"
#include "kernels/vulkan/mxfp4_gate_up.comp.hex.h"
#include "kernels/vulkan/mxfp4_row_tile.comp.hex.h"
#include "kernels/vulkan/mxfp4_indexed.comp.hex.h"
#include "kernels/vulkan/mxfp4_projection.comp.hex.h"
#include "kernels/vulkan/mxfp4_route_aggregation.comp.hex.h"
#endif
#include "linear_impl.h"
#include "linear_internal.h"

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN

static bool create_mxfp4_projection_pipeline(const std::shared_ptr<VulkanContext>& context, const ncnn::Option& option,
                                             std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(mxfp4_projection_shader,
                                                                                      static_cast<int>(sizeof(mxfp4_projection_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    const size_t pipeline_key = option.use_subgroup_ops ? 1u : 0u;
    destination = context->find_pipeline(mxfp4_projection_shader, pipeline_key);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = pipeline->local_size_x() * pipeline->local_size_y() * pipeline->local_size_z();
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        // Weak cache entries retain the deleter control block after disposal.
        context.reset();
    });
    context->cache_pipeline(mxfp4_projection_shader, pipeline_key, destination);
    return true;
}

static size_t align_to_uint32(size_t bytes) noexcept
{
    if (bytes > std::numeric_limits<size_t>::max() - 3)
        return 0;
    return (bytes + 3) & ~static_cast<size_t>(3);
}
#endif

Mxfp4Linear_vulkan::Mxfp4Linear_vulkan()
    : d(new Implementation)
{
}

Mxfp4Linear_vulkan::~Mxfp4Linear_vulkan() = default;

std::shared_ptr<Mxfp4Linear_vulkan> Mxfp4Linear_vulkan::create(const TensorData& matrix, const TensorData* bias,
                                                               uint32_t vulkan_device_index,
                                                               const VulkanRuntimePtr& vulkan_runtime,
                                                               uint64_t optimization_flags)
{
    return create_with_allocator(matrix,
                                 bias,
                                 vulkan_device_index,
                                 nullptr,
                                 vulkan_runtime,
                                 optimization_flags);
}

std::shared_ptr<Mxfp4Linear_vulkan> Mxfp4Linear_vulkan::create_with_allocator(const TensorData& matrix, const TensorData* bias,
                                                                              uint32_t vulkan_device_index, ncnn::VkAllocator* weight_allocator,
                                                                              const VulkanRuntimePtr& vulkan_runtime,
                                                                              uint64_t optimization_flags,
                                                                              VulkanWeightUploadBatch* upload_batch)
{
#if NCNN_MOE_WITH_VULKAN
    if (matrix.dtype != DType::MxFp4 || matrix.shape.size() != 2 || matrix.shape[0] == 0 || matrix.shape[1] == 0 || matrix.shape[1] % 32 != 0
        || !matrix.input_column_indices.empty()
        || matrix.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 64)
        || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t block_count = input_columns / 32;
    const uint64_t expected_blocks = static_cast<uint64_t>(output_columns) * block_count * 16;
    const uint64_t expected_scales = static_cast<uint64_t>(output_columns) * block_count;
    if (matrix.mxfp4_blocks.size() != expected_blocks || matrix.mxfp4_scales.size() != expected_scales)
    {
        return {};
    }
    if (bias && (bias->shape.size() != 1 || bias->shape[0] != output_columns || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    std::shared_ptr<Mxfp4Linear_vulkan> result(new Mxfp4Linear_vulkan);
    if (upload_batch)
        upload_batch->retain_owner(result);
    Implementation& implementation = *result->d;
    implementation.input_columns = input_columns;
    implementation.output_columns = output_columns;
    implementation.block_count = block_count;
    implementation.vulkan_context = VulkanContext::acquire(vulkan_device_index,
                                                           vulkan_runtime,
                                                           optimization_flags);
    if (!implementation.vulkan_context)
        return {};
    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    implementation.option.use_vulkan_compute = true;
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage = false;
    implementation.option.use_fp16_arithmetic = false;
    implementation.option.use_bf16_packed = false;
    implementation.option.use_bf16_storage = false;
    implementation.option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.staging_vkallocator = implementation.vulkan_context->staging_allocator();
    implementation.option.use_cooperative_matrix = device->info.support_cooperative_matrix();
    implementation.option.use_subgroup_ops = device->info.support_subgroup_ops();
    const uint64_t preferred_weight_size = expected_blocks + expected_scales + static_cast<uint64_t>(output_columns) * sizeof(float);
    if (preferred_weight_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        return {};
    }
    if (!weight_allocator)
    {
        implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(device, static_cast<size_t>(preferred_weight_size)));
        weight_allocator = implementation.weight_allocator.get();
    }
    if (!upload_batch)
        implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));

    const size_t packed_size = align_to_uint32(matrix.mxfp4_blocks.size());
    const size_t scales_size = align_to_uint32(matrix.mxfp4_scales.size());
    const size_t bias_size = static_cast<size_t>(output_columns) * sizeof(float);
    if (packed_size == 0 || scales_size == 0
        || packed_size > static_cast<size_t>(std::numeric_limits<int>::max())
        || scales_size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    std::span<const float> float_bias_values;
    std::span<const uint16_t> bfloat16_bias_values;
    if (bias && bias->dtype == DType::Float32)
    {
        float_bias_values = bias->float32_values();
        if (float_bias_values.size() != output_columns)
            return {};
    }
    else if (bias)
    {
        bfloat16_bias_values = bias->bfloat16_values();
        if (bfloat16_bias_values.size() != output_columns)
            return {};
    }

    const size_t storage_alignment = std::max<size_t>(4,
                                                      device->info.buffer_offset_alignment());
    const auto aligned_segment = [storage_alignment](size_t bytes) -> size_t {
        const size_t remainder = bytes % storage_alignment;
        if (remainder == 0)
            return bytes;
        const size_t padding = storage_alignment - remainder;
        if (bytes > std::numeric_limits<size_t>::max() - padding)
            return 0;
        return bytes + padding;
    };
    const size_t packed_segment_size = aligned_segment(packed_size);
    const size_t scales_segment_size = aligned_segment(scales_size);
    const size_t bias_segment_size = aligned_segment(bias_size);
    if (packed_segment_size == 0 || scales_segment_size == 0 || bias_segment_size == 0
        || packed_segment_size > std::numeric_limits<size_t>::max() - scales_segment_size
        || packed_segment_size + scales_segment_size > std::numeric_limits<size_t>::max() - bias_segment_size)
    {
        return {};
    }
    const size_t scales_offset = packed_segment_size;
    const size_t bias_offset = packed_segment_size + scales_segment_size;
    const size_t storage_size = bias_offset + bias_segment_size;
    if (storage_size == 0 || storage_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {};

    const bool direct_staging = upload_batch
                                && dynamic_cast<VulkanExpertWeightAllocator*>(weight_allocator)
                                && has_flag(optimization_flags, OptimizationVulkanExpertDirectStaging);
    ncnn::Mat combined;
    std::span<uint8_t> packed_storage;
    if (direct_staging)
        packed_storage = upload_batch->prepare_storage(storage_size, weight_allocator);
    else
    {
        combined = upload_batch ? upload_batch->host_storage(storage_size)
                                : ncnn::Mat(static_cast<int>(storage_size), sizeof(uint8_t));
        if (!combined.empty())
            packed_storage = {static_cast<uint8_t*>(combined.data), storage_size};
    }
    if (packed_storage.size() != storage_size)
        return {};
    pack_mxfp4_weight_storage(packed_storage,
                              packed_segment_size, scales_segment_size,
                              {matrix.mxfp4_blocks.data(), matrix.mxfp4_blocks.size()},
                              {matrix.mxfp4_scales.data(), matrix.mxfp4_scales.size()},
                              float_bias_values, bfloat16_bias_values);

    if (upload_batch)
    {
        if (!create_mxfp4_projection_pipeline(implementation.vulkan_context, implementation.option, implementation.pipeline))
            return {};
    }
    else
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_mxfp4_projection_pipeline(implementation.vulkan_context, implementation.option, implementation.pipeline))
        {
            return {};
        }
    }
    const uint64_t local_size_x = implementation.pipeline->local_size_x();
    const uint64_t max_dispatch_width = static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (local_size_x == 0 || local_size_x > max_dispatch_width
        || static_cast<uint64_t>(output_columns)
               > (max_dispatch_width - (local_size_x - 1)) / local_size_x)
    {
        return {};
    }
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = weight_allocator;
    upload_option.workspace_vkallocator = weight_allocator;
    bool uploaded = false;
    if (upload_batch)
    {
        uploaded = direct_staging
                       ? upload_batch->record_prepared_storage(implementation.storage, weight_allocator)
                       : upload_batch->record(combined, implementation.storage, implementation.option, weight_allocator);
    }
    else
    {
        upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
        // Executable-cache admission runs on a background thread. Serialize
        // weight uploads with foreground compute and other context users;
        // ncnn's transfer command and Vulkan allocators are not an independent
        // command domain for this shared device context.
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(device);
        command.record_upload(combined, implementation.storage, upload_option);
        uploaded = !implementation.storage.empty() && command.submit_and_wait() == 0;
    }
    if (!upload_batch)
    {
        implementation.weight_staging_allocator.reset();
    }
    if (!uploaded)
    {
        return {};
    }

    const auto storage_view = [&implementation](size_t offset, size_t bytes) {
        ncnn::VkMat view = implementation.storage;
        view.dims = 1;
        view.w = static_cast<int>(bytes);
        view.h = 1;
        view.d = 1;
        view.c = 1;
        view.elemsize = sizeof(uint8_t);
        view.elempack = 1;
        view.cstep = bytes;
#if NCNN_BATCH
        view.n = 1;
        view.nstep = bytes;
#endif
        view.offset += offset;
        return view;
    };
    implementation.packed = storage_view(0, packed_size);
    implementation.scales = storage_view(scales_offset, scales_size);
    implementation.bias = storage_view(bias_offset, bias_size);
    implementation.indexed_storage = true;
    implementation.indexed_scales_word_offset = static_cast<uint32_t>(scales_offset / sizeof(uint32_t));
    implementation.indexed_bias_word_offset = static_cast<uint32_t>(bias_offset / sizeof(uint32_t));
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)vulkan_runtime;
    (void)optimization_flags;
    (void)upload_batch;
    return {};
#endif
}

bool Mxfp4Linear_vulkan::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context || !implementation.pipeline || input.rows() == 0 || input.dtype() != DType::Float32
        || input.columns() != implementation.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * implementation.input_columns
                                                                                                 * sizeof(float),
                                                                                             input.dtype());
    const bool direct_host_output = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * implementation.output_columns
                                                                                                  * sizeof(float),
                                                                                              output.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output.reset(input.rows(), implementation.output_columns, false);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = bind_host_out(transfer_slot.download);
    else
        output_gpu.create(static_cast<int>(implementation.output_columns), static_cast<int>(input.rows()), sizeof(float),
                          implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings(5);
    bindings[0] = input_gpu;
    bindings[1] = implementation.packed;
    bindings[2] = implementation.scales;
    bindings[3] = implementation.bias;
    bindings[4] = output_gpu;
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(static_cast<uint64_t>(implementation.output_columns)
                                    * implementation.pipeline->local_size_x());
    dispatcher.h = static_cast<int>(input.rows());
    dispatcher.c = 1;
    command.record_pipeline(implementation.pipeline.get(), bindings, constants, dispatcher);
    if ((!direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     implementation.output_columns,
                                     transfer_slot.download,
                                     command,
                                     implementation.vulkan_context->device(),
                                     implementation.option,
                                     output_dtype))
        || submit_compute_and_wait(command, implementation.vulkan_context->device()) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    ++runtime_state.dispatches;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

uint32_t Mxfp4Linear_vulkan::input_columns() const noexcept
{
    return d->input_columns;
}

uint32_t Mxfp4Linear_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

#if NCNN_MOE_WITH_VULKAN

static bool create_mxfp4_gate_up_pipeline(const std::shared_ptr<VulkanContext>& context, const ncnn::Option& option,
                                          std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(mxfp4_gate_up_shader,
                                                                                      static_cast<int>(sizeof(mxfp4_gate_up_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;
    ncnn::VulkanDevice* device = context->device();
    const size_t pipeline_key = option.use_subgroup_ops ? 1u : 0u;
    destination = context->find_pipeline(mxfp4_gate_up_shader, pipeline_key);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = pipeline->local_size_x() * pipeline->local_size_y() * pipeline->local_size_z();
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        // Weak cache entries retain the deleter control block after disposal.
        context.reset();
    });
    context->cache_pipeline(mxfp4_gate_up_shader, pipeline_key, destination);
    return true;
}

static bool create_mxfp4_row_tile_pipeline(const std::shared_ptr<VulkanContext>& context, const ncnn::Option& option,
                                           std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(mxfp4_row_tile_shader,
                                                                                      static_cast<int>(sizeof(mxfp4_row_tile_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;
    ncnn::VulkanDevice* device = context->device();
    const size_t pipeline_key = option.use_subgroup_ops ? 1u : 0u;
    destination = context->find_pipeline(mxfp4_row_tile_shader, pipeline_key);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = pipeline->local_size_x() * pipeline->local_size_y() * pipeline->local_size_z();
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        // Weak cache entries retain the deleter control block after disposal.
        context.reset();
    });
    context->cache_pipeline(mxfp4_row_tile_shader, pipeline_key, destination);
    return true;
}

bool create_gather_pipeline(const std::shared_ptr<VulkanContext>& context,
                            const ncnn::Option& opt,
                            std::shared_ptr<ncnn::Pipeline>& destination)
{
    const auto spirv = context->shader_binary(expert_gather_shader, static_cast<int>(sizeof(expert_gather_shader) - 1), opt, 0);
    if (!spirv || spirv->empty())
        return false;
    destination = context->find_pipeline(expert_gather_shader, 0);
    if (destination)
        return true;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(context->device()));
    pipeline->set_local_size_xyz(128, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        // Weak cache entries retain the deleter control block after disposal.
        context.reset();
    });
    context->cache_pipeline(expert_gather_shader, 0, destination);
    return true;
}

bool can_share_row(size_t available_rows, size_t request_count,
                   std::span<const uint32_t> input_rows)
{
    // Every validated request has at least one route. Equal request/route
    // counts therefore prove one route per request, all reading the same row.
    return available_rows == 1 && request_count != 0 && input_rows.size() == request_count
           && std::all_of(input_rows.begin(), input_rows.end(), [](uint32_t row) { return row == 0; });
}

bool VulkanExpertBackend::forward_device_batch(std::span<const ExpertRequest> requests,
                                               std::span<const Selection> selected)
{
    if (selected.empty()) return true;
    auto valid_entry = [](const Entry* entry) {
        return entry && (entry->qnk_operation || entry->bfloat16_operation || entry->operation);
    };
    if (!valid_entry(selected.front().entry.get()) || selected.front().request_index >= requests.size()) return false;
    const auto& first = *selected.front().entry;
    const auto context = first.qnk_operation        ? first.qnk_operation->d->vulkan_context
                         : first.bfloat16_operation ? first.bfloat16_operation->d->vulkan_context
                                                    : first.operation->d->gate_up->d->vulkan_context;
    const auto& option = first.qnk_operation        ? first.qnk_operation->d->option
                         : first.bfloat16_operation ? first.bfloat16_operation->d->option
                                                    : first.operation->d->gate_up->d->option;
    if (!context || activation_variant(option) != 0) return false;
    const uint32_t input_columns = first.qnk_operation        ? first.qnk_operation->d->gate_up->input_columns()
                                   : first.bfloat16_operation ? first.bfloat16_operation->d->input_columns
                                                              : first.operation->d->gate_up->d->input_columns;
    const uint32_t output_columns = first.qnk_operation        ? first.qnk_operation->d->output_columns
                                    : first.bfloat16_operation ? first.bfloat16_operation->d->output_columns
                                                               : first.operation->d->down->d->output_columns;
    const auto& first_request = requests[selected.front().request_index];
    const auto region = first_request.region;
    const auto retained_input = first_request.device_input;
    const ncnn::VkMat* source = retained_input ? retained_input->value() : nullptr;
    size_t available_rows = 0;
    if (region)
    {
        if (region->columns() != input_columns || region->rows() == 0) return false;
        available_rows = region->rows();
        if (!resident_combiner) resident_combiner = MoeCombine_vulkan::create(context);
        if (!region->bind_context(context, resident_combiner)) return false;
    }
    else
    {
        if (!source || source->dims != 2 || source->w != static_cast<int>(input_columns)
            || source->h <= 0 || (source->elempack != 1 && source->elempack != 4)
            || source->elemsize != sizeof(float) * static_cast<size_t>(source->elempack)
            || retained_input->vulkan_context() != context) return false;
        available_rows = static_cast<size_t>(source->h) * source->elempack;
        if (retained_input->rows() != available_rows || retained_input->columns() != input_columns) return false;
    }
    size_t total_rows = 0;
    uint64_t dispatches = 0;
    uint64_t int4_dispatches = 0;
    std::vector<uint32_t> input_rows;
    for (const Selection& selection : selected)
    {
        if (selection.request_index >= requests.size() || !valid_entry(selection.entry.get())) return false;
        const auto& request = requests[selection.request_index];
        const auto* qnk = selection.entry->qnk_operation.get();
        const auto* bfloat16 = selection.entry->bfloat16_operation.get();
        const auto* mxfp4 = selection.entry->operation.get();
        const auto& expert_context = qnk        ? qnk->d->vulkan_context
                                     : bfloat16 ? bfloat16->d->vulkan_context
                                                : mxfp4->d->gate_up->d->vulkan_context;
        const uint32_t expert_input_columns = qnk        ? qnk->d->gate_up->input_columns()
                                              : bfloat16 ? bfloat16->d->input_columns
                                                         : mxfp4->d->gate_up->d->input_columns;
        const uint32_t expert_output_columns = qnk        ? qnk->d->output_columns
                                               : bfloat16 ? bfloat16->d->output_columns
                                                          : mxfp4->d->down->d->output_columns;
        if (!request.output || !request.device_output || request.region != region
            || (!region && request.device_input != retained_input)
            || request.rows() == 0 || request.rows() != request.device_routes.size()
            || request.columns() != input_columns || expert_context != context
            || expert_input_columns != input_columns || expert_output_columns != output_columns
            || request.rows() > context->device()->info.max_workgroup_count_y()
            || request.rows() > static_cast<size_t>(std::numeric_limits<int>::max()) - total_rows) return false;
        for (const ExpertRoute& route : request.device_routes)
        {
            if (route.token_index >= available_rows) return false;
            input_rows.push_back(route.token_index);
        }
        total_rows += request.rows();
        if (qnk)
        {
            dispatches += (qnk->d->up ? 4u : 3u) + (qnk->d->router_gate ? 2u : 0u);
            int4_dispatches += qnk->d->gate_up->dtype() == DType::Int4 ? 1u : 0u;
            int4_dispatches += qnk->d->down->dtype() == DType::Int4 ? 1u : 0u;
            int4_dispatches += qnk->d->up && qnk->d->up->dtype() == DType::Int4 ? 1u : 0u;
        }
        else
            dispatches += 2;
    }
    if (input_columns == 0 || output_columns == 0 || total_rows == 0) return false;
    const bool shared_single_row = can_share_row(available_rows, selected.size(), input_rows);
    dispatches += !shared_single_row;
    auto transfer_lease = context->acquire_transfer_slot();
    auto& slot = transfer_lease.slot();
    ncnn::VkMat rows_staging;
    if (!shared_single_row
        && !fill_staging_values(input_rows.data(), input_rows.size(), sizeof(uint32_t), rows_staging, slot.staging_allocator)) return false;
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if (!shared_single_row && !input_gather_pipeline && !create_gather_pipeline(context, option, input_gather_pipeline)) return false;
    if (slot.command_used && slot.command->reset() != 0) return false;
    slot.command_used = true;
    auto& command = *slot.command;
    ncnn::VkMat region_input;
    ncnn::VkMat rows_gpu;
    ncnn::VkMat input_gpu;
    std::vector<ncnn::VkMat> workspace, aligned_inputs, bindings, outputs;
    std::vector<ncnn::vk_constant_type> constants(3);
    std::vector<ResidentFfnRegion_vulkan::Output> region_outputs;
    outputs.reserve(selected.size());
    region_outputs.reserve(selected.size());
    // Reset all command references before releasing any recorded operand.
    // submit_compute_and_wait drains submitted work, including wait failures.
    bool successful = false;
    const auto cleanup = [&](void*) {
        if (!successful)
        {
            (void)command.reset();
            if (region) region->abort_wave();
        }
    };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);
    bool uploaded = false, downloaded = false;
    if (region)
    {
        if (!region->record_input(region_input, command, option, uploaded)) return false;
        source = &region_input;
    }
    else
        retained_input->prepare_read();
    if (shared_single_row)
    {
        input_gpu = aligned_row_view(*source, 0, 1, command, option, context, aligned_inputs, selected.size());
        if (input_gpu.empty()) return false;
    }
    else
    {
        if (!record_mapped_upload(rows_staging, rows_gpu, command, option)) return false;
        input_gpu.create(static_cast<int>(input_columns), static_cast<int>(total_rows), sizeof(float), context->blob_allocator());
        if (input_gpu.empty()) return false;
        constants[0].u32 = input_columns;
        constants[1].u32 = static_cast<uint32_t>(total_rows);
        constants[2].u32 = static_cast<uint32_t>(source->elempack);
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(input_columns);
        dispatcher.h = static_cast<int>(std::min(total_rows, size_t(65535)));
        dispatcher.c = static_cast<int>((total_rows + 65534) / 65535);
        command.record_pipeline_readonly(input_gather_pipeline.get(), {*source, rows_gpu, input_gpu}, {1, 1, 0}, constants, dispatcher);
    }
    size_t offset = 0;
    for (const Selection& selection : selected)
    {
        const auto& request = requests[selection.request_index];
        ncnn::VkMat input_view = aligned_row_view(input_gpu, shared_single_row ? 0 : offset, request.rows(), command,
                                                  option, context, aligned_inputs, selected.size());
        ncnn::VkMat output_gpu;
        output_gpu.create(static_cast<int>(output_columns), static_cast<int>(request.rows()), sizeof(float), context->blob_allocator());
        if (input_view.empty() || output_gpu.empty()) return false;
        outputs.push_back(output_gpu);
        if (selection.entry->qnk_operation)
        {
            if (!selection.entry->qnk_operation->record(input_view, output_gpu, command, workspace, bindings, constants)) return false;
        }
        else if (selection.entry->bfloat16_operation)
        {
            if (!selection.entry->bfloat16_operation->record(input_view, output_gpu, command, workspace, bindings, constants)) return false;
        }
        else
        {
            auto& intermediate = workspace.emplace_back();
            intermediate.create(static_cast<int>(selection.entry->operation->d->gate_up->d->output_columns / 2),
                                static_cast<int>(request.rows()), sizeof(float), context->blob_allocator());
            if (intermediate.empty()) return false;
            selection.entry->operation->record(input_view, intermediate, output_gpu, command, bindings, constants);
        }
        if (region) region_outputs.push_back({request.region_index, output_gpu, request.device_routes});
        offset += request.rows();
    }
    if (region && !region->record_outputs(region_outputs, command, option, downloaded)) return false;
    if (region) region->mark_submit_attempted();
    if (submit_compute_and_wait(command, context->device()) != 0) return false;
    if (region && !region->complete_wave()) return false;
    std::vector<std::shared_ptr<DeviceTensor_vulkan>> completed;
    completed.reserve(selected.size());
    for (const ncnn::VkMat& output : outputs)
    {
        auto value = std::make_shared<DeviceTensor_vulkan>();
        if (!value->assign_completed(output, context)) return false;
        completed.push_back(std::move(value));
    }
    for (size_t index = 0; index < selected.size(); ++index)
        *requests[selected[index].request_index].device_output = std::move(completed[index]);
    auto& state = context->runtime_state();
    state.dispatches += dispatches + (region && region->complete() ? region->completed_combine_dispatches() : 0);
    state.int4_dispatches += int4_dispatches;
    state.batch_uploads += uploaded ? 1 : 0;
    state.batch_downloads += downloaded ? 1 : 0;
    ++state.compute_submissions;
    successful = true;
    return true;
}

bool create_route_agg_pipeline(const std::shared_ptr<VulkanContext>& context, const ncnn::Option& option,
                               std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(mxfp4_route_aggregation_shader,
                                                                                      static_cast<int>(sizeof(mxfp4_route_aggregation_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(mxfp4_route_aggregation_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(128, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        // Weak cache entries retain the deleter control block after disposal.
        context.reset();
    });
    context->cache_pipeline(mxfp4_route_aggregation_shader, 0, destination);
    return true;
}

// Indexed Expert dispatch uses one storage-buffer binding per selected Expert
// slot. Larger top-k batches are chunked and reuse this fixed descriptor table.

bool create_indexed_pipeline(const std::shared_ptr<VulkanContext>& context,
                             const ncnn::Option& option,
                             std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(mxfp4_indexed_shader,
                                                                                      static_cast<int>(sizeof(mxfp4_indexed_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    const size_t pipeline_key = option.use_subgroup_ops ? 1u : 0u;
    destination = context->find_pipeline(mxfp4_indexed_shader, pipeline_key);
    if (destination)
        return true;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(context->device()));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = pipeline->local_size_x() * pipeline->local_size_y() * pipeline->local_size_z();
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context = context](ncnn::Pipeline* value) mutable {
                                                      {
                                                          const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                          delete value;
                                                      }
                                                      // Weak cache entries retain the deleter control block after disposal.
                                                      context.reset();
                                                  });
    context->cache_pipeline(mxfp4_indexed_shader, pipeline_key, destination);
    return true;
}
#endif

Mxfp4Expert_vulkan::Mxfp4Expert_vulkan()
    : d(new Implementation)
{
}

Mxfp4Expert_vulkan::~Mxfp4Expert_vulkan() = default;

std::shared_ptr<Mxfp4Expert_vulkan> Mxfp4Expert_vulkan::create(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                               const TensorData& down, const TensorData* down_bias,
                                                               float activation_limit, uint32_t vulkan_device_index,
                                                               ExpertActivation activation,
                                                               const VulkanRuntimePtr& vulkan_runtime,
                                                               uint64_t optimization_flags)
{
    return create_with_allocator(gate_up,
                                 gate_up_bias,
                                 down,
                                 down_bias,
                                 activation_limit,
                                 vulkan_device_index,
                                 nullptr,
                                 activation,
                                 vulkan_runtime,
                                 optimization_flags);
}

std::shared_ptr<Mxfp4Expert_vulkan> Mxfp4Expert_vulkan::create_with_allocator(const TensorData& gate_up, const TensorData* gate_up_bias,
                                                                              const TensorData& down, const TensorData* down_bias,
                                                                              float activation_limit, uint32_t vulkan_device_index,
                                                                              ncnn::VkAllocator* weight_allocator,
                                                                              ExpertActivation activation,
                                                                              const VulkanRuntimePtr& vulkan_runtime,
                                                                              uint64_t optimization_flags,
                                                                              VulkanWeightUploadBatch* upload_batch)
{
#if NCNN_MOE_WITH_VULKAN
    if (gate_up.shape.size() != 2 || gate_up.shape[0] % 2 != 0 || down.shape.size() != 2 || down.shape[1] != gate_up.shape[0] / 2 || activation_limit < 0.0f
        || !gate_up.input_column_indices.empty() || !down.input_column_indices.empty())
    {
        return {};
    }

    std::shared_ptr<Mxfp4Linear_vulkan> gate_up_projection = Mxfp4Linear_vulkan::create_with_allocator(gate_up,
                                                                                                       gate_up_bias,
                                                                                                       vulkan_device_index,
                                                                                                       weight_allocator,
                                                                                                       vulkan_runtime,
                                                                                                       optimization_flags,
                                                                                                       upload_batch);
    std::shared_ptr<Mxfp4Linear_vulkan> down_projection = Mxfp4Linear_vulkan::create_with_allocator(down,
                                                                                                    down_bias,
                                                                                                    vulkan_device_index,
                                                                                                    weight_allocator,
                                                                                                    vulkan_runtime,
                                                                                                    optimization_flags,
                                                                                                    upload_batch);
    if (!gate_up_projection || !down_projection)
        return {};
    if (gate_up_projection->d->vulkan_context != down_projection->d->vulkan_context)
    {
        return {};
    }

    std::shared_ptr<Mxfp4Expert_vulkan> result(new Mxfp4Expert_vulkan);
    if (upload_batch)
        upload_batch->retain_owner(result);
    Implementation& implementation = *result->d;
    implementation.gate_up = std::move(gate_up_projection);
    implementation.down = std::move(down_projection);
    implementation.activation_limit = activation_limit;
    implementation.activation = activation;
    Mxfp4Linear_vulkan::Implementation& gate_implementation = *implementation.gate_up->d;
    if (upload_batch)
    {
        if (!create_mxfp4_gate_up_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.gate_up_pipeline)
            || (has_flag(optimization_flags, OptimizationVulkanMxfp4RowTile)
                && !create_mxfp4_row_tile_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.row_tile_pipeline)))
            return {};
    }
    else
    {
        const std::lock_guard<std::mutex> lock(gate_implementation.vulkan_context->command_mutex());
        if (!create_mxfp4_gate_up_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.gate_up_pipeline)
            || (has_flag(optimization_flags, OptimizationVulkanMxfp4RowTile)
                && !create_mxfp4_row_tile_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.row_tile_pipeline)))
        {
            return {};
        }
    }
    const uint64_t max_dispatch_width = static_cast<uint64_t>(std::numeric_limits<int>::max());
    const uint64_t gate_local_size_x = implementation.gate_up_pipeline->local_size_x();
    const uint64_t intermediate_columns = gate_implementation.output_columns / 2;
    if (gate_local_size_x == 0 || gate_local_size_x > max_dispatch_width
        || intermediate_columns > (max_dispatch_width - (gate_local_size_x - 1)) / gate_local_size_x)
    {
        return {};
    }
    if (implementation.row_tile_pipeline)
    {
        const uint64_t row_tile_local_size_x = implementation.row_tile_pipeline->local_size_x();
        if (row_tile_local_size_x == 0 || row_tile_local_size_x > max_dispatch_width
            || intermediate_columns > (max_dispatch_width - (row_tile_local_size_x - 1)) / row_tile_local_size_x
            || static_cast<uint64_t>(implementation.down->d->output_columns)
                   > (max_dispatch_width - (row_tile_local_size_x - 1)) / row_tile_local_size_x)
        {
            return {};
        }
    }
    return result;
#else
    (void)gate_up;
    (void)gate_up_bias;
    (void)down;
    (void)down_bias;
    (void)activation_limit;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)activation;
    (void)vulkan_runtime;
    (void)optimization_flags;
    (void)upload_batch;
    return {};
#endif
}

std::shared_ptr<Mxfp4Expert_vulkan> Mxfp4Expert_vulkan::create_from_device_storage(const Mxfp4DeviceMatrixView_vulkan& gate_up, const TensorData* gate_up_bias, const Mxfp4DeviceMatrixView_vulkan& down,
                                                                                   const TensorData* down_bias, float activation_limit, uint32_t vulkan_device_index, const ncnn::VkMat& storage,
                                                                                   ExpertActivation activation,
                                                                                   const VulkanRuntimePtr& vulkan_runtime,
                                                                                   uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN && NCNN_BATCH
    if (storage.empty() || gate_up.output_columns == 0 || gate_up.output_columns % 2 != 0 || gate_up.input_columns == 0 || down.output_columns == 0
        || down.input_columns != gate_up.output_columns / 2 || activation_limit < 0.0f)
    {
        return {};
    }
    const auto create_projection = [&](const Mxfp4DeviceMatrixView_vulkan& matrix, const TensorData* bias) -> std::shared_ptr<Mxfp4Linear_vulkan> {
        if (matrix.output_columns == 0 || matrix.input_columns == 0 || matrix.input_columns % 32 != 0)
        {
            return {};
        }
        const uint32_t output_columns = matrix.output_columns;
        const uint32_t input_columns = matrix.input_columns;
        const uint32_t block_count = input_columns / 32;
        const uint64_t expected_blocks = static_cast<uint64_t>(output_columns) * block_count * 16;
        const uint64_t expected_scales = static_cast<uint64_t>(output_columns) * block_count;
        if (matrix.packed_size != expected_blocks || matrix.scales_size != expected_scales || expected_blocks > std::numeric_limits<int>::max()
            || expected_scales > std::numeric_limits<int>::max() || matrix.packed_offset > storage.buffer_capacity()
            || matrix.scales_offset > storage.buffer_capacity() || expected_blocks > storage.buffer_capacity() - matrix.packed_offset
            || expected_scales > storage.buffer_capacity() - matrix.scales_offset)
        {
            return {};
        }
        if (bias && (bias->shape.size() != 1 || bias->shape[0] != output_columns || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
        {
            return {};
        }

        std::shared_ptr<Mxfp4Linear_vulkan> projection(new Mxfp4Linear_vulkan);
        Mxfp4Linear_vulkan::Implementation& implementation = *projection->d;
        implementation.input_columns = input_columns;
        implementation.output_columns = output_columns;
        implementation.block_count = block_count;
        implementation.vulkan_context = VulkanContext::acquire(vulkan_device_index,
                                                               vulkan_runtime,
                                                               optimization_flags);
        if (!implementation.vulkan_context)
            return {};
        ncnn::VulkanDevice* device = implementation.vulkan_context->device();
        const size_t alignment = std::max<size_t>(4, device->info.buffer_offset_alignment());
        if ((storage.buffer_offset() + matrix.packed_offset) % alignment != 0 || (storage.buffer_offset() + matrix.scales_offset) % alignment != 0)
        {
            return {};
        }
        implementation.option.use_vulkan_compute = true;
        implementation.option.use_fp16_packed = false;
        implementation.option.use_fp16_storage = false;
        implementation.option.use_fp16_arithmetic = false;
        implementation.option.use_bf16_packed = false;
        implementation.option.use_bf16_storage = false;
        implementation.option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
        implementation.option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
        implementation.option.staging_vkallocator = implementation.vulkan_context->staging_allocator();
        implementation.option.use_cooperative_matrix = device->info.support_cooperative_matrix();
        implementation.option.use_subgroup_ops = device->info.support_subgroup_ops();

        const auto view = [&storage](size_t offset, uint64_t bytes) {
            ncnn::VkMat result = storage;
            result.dims = 1;
            result.w = static_cast<int>(align_to_uint32(static_cast<size_t>(bytes)));
            result.h = 1;
            result.d = 1;
            result.c = 1;
            result.elemsize = sizeof(uint8_t);
            result.elempack = 1;
            result.cstep = result.w;
            result.offset += offset;
            result.n = 1;
            result.nstep = result.w;
            return result;
        };
        implementation.packed = view(matrix.packed_offset, expected_blocks);
        implementation.scales = view(matrix.scales_offset, expected_scales);

        ncnn::Mat biases(static_cast<int>(output_columns), sizeof(float));
        if (biases.empty())
            return {};
        float* bias_values = static_cast<float*>(biases.data);
        if (!bias)
        {
            std::fill_n(bias_values, output_columns, 0.0f);
        }
        else if (bias->dtype == DType::Float32)
        {
            const auto values = bias->float32_values();
            if (values.size() != output_columns)
            {
                return {};
            }
            std::copy(values.begin(), values.end(), bias_values);
        }
        else
        {
            const auto values = bias->bfloat16_values();
            if (values.size() != output_columns)
            {
                return {};
            }
            for (uint32_t index = 0; index < output_columns; ++index)
            {
                bias_values[index] = bfloat16_to_float(values[index]);
            }
        }
        implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));
        ncnn::Option upload_option = implementation.option;
        upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
        {
            const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
            if (!create_mxfp4_projection_pipeline(implementation.vulkan_context, implementation.option, implementation.pipeline))
            {
                return {};
            }
            const uint64_t local_size_x = implementation.pipeline->local_size_x();
            const uint64_t max_dispatch_width = static_cast<uint64_t>(std::numeric_limits<int>::max());
            if (local_size_x == 0 || local_size_x > max_dispatch_width
                || static_cast<uint64_t>(output_columns)
                       > (max_dispatch_width - (local_size_x - 1)) / local_size_x)
            {
                return {};
            }
            ncnn::VkTransfer command(device);
            command.record_upload(biases, implementation.bias, upload_option);
            if (implementation.bias.empty() || command.submit_and_wait() != 0)
            {
                return {};
            }
        }
        implementation.weight_staging_allocator.reset();
        return projection;
    };

    auto gate_projection = create_projection(gate_up, gate_up_bias);
    auto down_projection = create_projection(down, down_bias);
    if (!gate_projection || !down_projection || gate_projection->d->vulkan_context != down_projection->d->vulkan_context)
    {
        return {};
    }
    std::shared_ptr<Mxfp4Expert_vulkan> result(new Mxfp4Expert_vulkan);
    Implementation& implementation = *result->d;
    implementation.gate_up = std::move(gate_projection);
    implementation.down = std::move(down_projection);
    implementation.activation_limit = activation_limit;
    implementation.activation = activation;
    auto& gate_implementation = *implementation.gate_up->d;
    {
        const std::lock_guard<std::mutex> lock(gate_implementation.vulkan_context->command_mutex());
        if (!create_mxfp4_gate_up_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.gate_up_pipeline)
            || (has_flag(optimization_flags, OptimizationVulkanMxfp4RowTile)
                && !create_mxfp4_row_tile_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.row_tile_pipeline)))
        {
            return {};
        }
    }
    const uint64_t max_dispatch_width = static_cast<uint64_t>(std::numeric_limits<int>::max());
    const uint64_t gate_local_size_x = implementation.gate_up_pipeline->local_size_x();
    const uint64_t intermediate_columns = gate_implementation.output_columns / 2;
    if (gate_local_size_x == 0 || gate_local_size_x > max_dispatch_width
        || intermediate_columns > (max_dispatch_width - (gate_local_size_x - 1)) / gate_local_size_x)
    {
        return {};
    }
    if (implementation.row_tile_pipeline)
    {
        const uint64_t row_tile_local_size_x = implementation.row_tile_pipeline->local_size_x();
        if (row_tile_local_size_x == 0 || row_tile_local_size_x > max_dispatch_width
            || intermediate_columns > (max_dispatch_width - (row_tile_local_size_x - 1)) / row_tile_local_size_x
            || static_cast<uint64_t>(implementation.down->d->output_columns)
                   > (max_dispatch_width - (row_tile_local_size_x - 1)) / row_tile_local_size_x)
        {
            return {};
        }
    }
    return result;
#else
    (void)gate_up;
    (void)gate_up_bias;
    (void)down;
    (void)down_bias;
    (void)activation_limit;
    (void)vulkan_device_index;
    (void)storage;
    (void)activation;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

bool Mxfp4Expert_vulkan::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.gate_up || !implementation.down || !implementation.gate_up_pipeline)
    {
        return false;
    }
    const Mxfp4Linear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const Mxfp4Linear_vulkan::Implementation& down = *implementation.down->d;
    if (!gate.vulkan_context || gate.vulkan_context != down.vulkan_context || input.rows() == 0
        || input.dtype() != DType::Float32 || input.columns() != gate.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    const uint32_t intermediate_columns = gate.output_columns / 2;
    if (down.input_columns != intermediate_columns)
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = gate.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = gate.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), down.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output.reset(input.rows(), down.output_columns, false);
    std::unique_lock<std::mutex> lock(gate.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, gate.option))
    {
        return false;
    }
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(intermediate_columns), static_cast<int>(input.rows()), sizeof(float), gate.vulkan_context->blob_allocator());
    if (intermediate_gpu.empty())
        return false;
    ncnn::VkMat output_gpu;
    output_gpu.create(static_cast<int>(down.output_columns), static_cast<int>(input.rows()), sizeof(float), gate.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;
    std::vector<ncnn::VkMat> bindings(5);
    std::vector<ncnn::vk_constant_type> constants(7);
    record(input_gpu, intermediate_gpu, output_gpu, command, bindings, constants);

    if (!record_staging_download(output_gpu,
                                 input.rows(),
                                 down.output_columns,
                                 transfer_slot.download,
                                 command,
                                 gate.vulkan_context->device(),
                                 down.option,
                                 output_dtype)
        || submit_compute_and_wait(command, gate.vulkan_context->device()) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

#if NCNN_MOE_WITH_VULKAN
void Mxfp4Expert_vulkan::record(const ncnn::VkMat& input,
                                ncnn::VkMat& intermediate,
                                ncnn::VkMat& output,
                                ncnn::VkCompute& cmd,
                                std::vector<ncnn::VkMat>& bindings,
                                std::vector<ncnn::vk_constant_type>& constants) const
{
    const Implementation& implementation = *d;
    const Mxfp4Linear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const Mxfp4Linear_vulkan::Implementation& down = *implementation.down->d;
    const uint32_t intermediate_columns = gate.output_columns / 2;
    const uint32_t rows = static_cast<uint32_t>(input.h);

    // Four-row register reuse amortizes its extra registers on larger
    // prefill batches. Keep short prompts on the scalar decode pipeline.
    if (rows >= 16 && implementation.row_tile_pipeline)
    {
        bindings.resize(5);
        bindings[0] = input;
        bindings[1] = gate.packed;
        bindings[2] = gate.scales;
        bindings[3] = gate.bias;
        bindings[4] = intermediate;
        constants.resize(7);
        constants[0].u32 = gate.input_columns;
        constants[1].u32 = intermediate_columns;
        constants[2].u32 = gate.block_count;
        constants[3].u32 = rows;
        constants[4].u32 = 0;
        constants[5].u32 = implementation.activation == ExpertActivation::Silu
                               ? 2
                           : implementation.activation == ExpertActivation::DeepSeekSwiGlu ? 1
                                                                                           : 0;
        constants[6].f = implementation.activation_limit;
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(static_cast<uint64_t>(intermediate_columns)
                                        * implementation.row_tile_pipeline->local_size_x());
        dispatcher.h = static_cast<int>((rows + 3) / 4);
        dispatcher.c = 1;
        cmd.record_pipeline(implementation.row_tile_pipeline.get(), bindings, constants, dispatcher);

        bindings[0] = intermediate;
        bindings[1] = down.packed;
        bindings[2] = down.scales;
        bindings[3] = down.bias;
        bindings[4] = output;
        constants[0].u32 = down.input_columns;
        constants[1].u32 = down.output_columns;
        constants[2].u32 = down.block_count;
        constants[4].u32 = 1;
        dispatcher.w = static_cast<int>(static_cast<uint64_t>(down.output_columns)
                                        * implementation.row_tile_pipeline->local_size_x());
        cmd.record_pipeline(implementation.row_tile_pipeline.get(), bindings, constants, dispatcher);
        return;
    }

    bindings.resize(5);
    bindings[0] = input;
    bindings[1] = gate.packed;
    bindings[2] = gate.scales;
    bindings[3] = gate.bias;
    bindings[4] = intermediate;
    constants.resize(6);
    constants[0].u32 = gate.input_columns;
    constants[1].u32 = intermediate_columns;
    constants[2].u32 = gate.block_count;
    constants[3].u32 = rows;
    constants[4].u32 = implementation.activation == ExpertActivation::Silu
                           ? 2
                       : implementation.activation == ExpertActivation::DeepSeekSwiGlu ? 1
                                                                                       : 0;
    constants[5].f = implementation.activation_limit;
    ncnn::VkMat gate_dispatcher;
    gate_dispatcher.w = static_cast<int>(static_cast<uint64_t>(intermediate_columns)
                                         * implementation.gate_up_pipeline->local_size_x());
    gate_dispatcher.h = static_cast<int>(rows);
    gate_dispatcher.c = 1;
    cmd.record_pipeline(implementation.gate_up_pipeline.get(), bindings, constants, gate_dispatcher);

    bindings[0] = intermediate;
    bindings[1] = down.packed;
    bindings[2] = down.scales;
    bindings[3] = down.bias;
    bindings[4] = output;
    constants.resize(4);
    constants[0].u32 = down.input_columns;
    constants[1].u32 = down.output_columns;
    constants[2].u32 = down.block_count;
    constants[3].u32 = rows;
    ncnn::VkMat down_dispatcher;
    down_dispatcher.w = static_cast<int>(static_cast<uint64_t>(down.output_columns)
                                         * down.pipeline->local_size_x());
    down_dispatcher.h = static_cast<int>(rows);
    down_dispatcher.c = 1;
    cmd.record_pipeline(down.pipeline.get(), bindings, constants, down_dispatcher);
}
#endif

} // namespace moe
} // namespace ncnn
