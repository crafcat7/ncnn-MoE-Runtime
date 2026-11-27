#include "backends/ncnn/core/context.h"
#include "backends/ncnn/core/memory.h"
#include "backends/ncnn/core/transfer.h"
#include "backends/ncnn/linear/internal.h"
#include "backends/ncnn/linear/mxfp4.h"
#include "kernels/ops.h"
#include "kernels/vulkan/mxfp4_gate_up.comp.hex.h"
#include "kernels/vulkan/mxfp4_projection.comp.hex.h"
#include "kernels/vulkan/mxfp4_row_tile.comp.hex.h"

#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <pipeline.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

// --- MXFP4 pipeline & layout helpers -----------------------------------------

static bool create_mxfp4_pipeline(const std::shared_ptr<VulkanContext>& context,
                                  const ncnn::Option& option,
                                  const char* src, int len,
                                  std::shared_ptr<ncnn::Pipeline>& dst)
{
    const auto spirv = context->shader_binary(src, len, option, 0);
    if (!spirv || spirv->empty())
        return false;

    const uint64_t key = option.use_subgroup_ops ? 1u : 0u;
    dst = context->find_pipeline(src, key);
    if (dst)
        return true;

    std::unique_ptr<ncnn::Pipeline> pipe(new ncnn::Pipeline(context->device()));
    pipe->set_optimal_local_size_xyz(32, 1, 1);
    std::vector<ncnn::vk_specialization_type> specs(1);
    specs[0].u32 = pipe->local_size_x() * pipe->local_size_y() * pipe->local_size_z();
    if (pipe->create(spirv->data(), spirv->size() * sizeof(uint32_t), specs) != 0)
        return false;

    dst = context->cache_pipeline(src, key, std::shared_ptr<ncnn::Pipeline>(pipe.release()));
    return true;
}

static size_t align_bytes(size_t bytes, size_t alignment) noexcept
{
    const size_t remainder = bytes % alignment;
    if (remainder == 0)
        return bytes;
    const size_t padding = alignment - remainder;
    if (bytes > std::numeric_limits<size_t>::max() - padding)
        return 0;
    return bytes + padding;
}

static ncnn::VkMat byte_storage_view(const ncnn::VkMat& storage, size_t offset, size_t bytes)
{
    ncnn::VkMat view = storage;
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
}

static bool valid_dispatch_columns(const ncnn::Pipeline* pipe, uint64_t columns) noexcept
{
    const uint64_t lx = pipe ? pipe->local_size_x() : 0;
    const uint64_t max_w = static_cast<uint64_t>(std::numeric_limits<int>::max());
    return lx > 0 && lx <= max_w && columns <= (max_w - (lx - 1)) / lx;
}

// --- Mxfp4Linear_vulkan ------------------------------------------------------

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
    return create_with_allocator(matrix, bias, vulkan_device_index, nullptr,
                                 vulkan_runtime, optimization_flags);
}

std::shared_ptr<Mxfp4Linear_vulkan> Mxfp4Linear_vulkan::create_with_allocator(const TensorData& matrix,
                                                                              const TensorData* bias,
                                                                              uint32_t vulkan_device_index,
                                                                              ncnn::VkAllocator* weight_allocator,
                                                                              const VulkanRuntimePtr& vulkan_runtime,
                                                                              uint64_t optimization_flags,
                                                                              VkWeightBatch* upload_batch)
{
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t block_count = input_columns / 32;
    const uint64_t expected_blocks = static_cast<uint64_t>(output_columns) * block_count * 16;
    const uint64_t expected_scales = static_cast<uint64_t>(output_columns) * block_count;
    if (matrix.mxfp4_blocks.size() != expected_blocks || matrix.mxfp4_scales.size() != expected_scales)
        return {};

    std::shared_ptr<Mxfp4Linear_vulkan> result(new Mxfp4Linear_vulkan);
    if (upload_batch)
        upload_batch->retain_owner(result);

    Implementation& impl = *result->d;
    impl.input_columns = input_columns;
    impl.output_columns = output_columns;
    impl.block_count = block_count;
    impl.vulkan_context = VulkanContext::acquire(vulkan_device_index, vulkan_runtime, optimization_flags);
    if (!impl.vulkan_context)
        return {};

    ncnn::VulkanDevice* device = impl.vulkan_context->device();
    impl.option = make_vk_opt(*impl.vulkan_context);
    impl.option.use_cooperative_matrix = device->info.support_cooperative_matrix();

    const uint64_t preferred_weight_size = expected_blocks + expected_scales
                                           + static_cast<uint64_t>(output_columns) * sizeof(float);
    if (preferred_weight_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return {};

    if (!weight_allocator)
    {
        impl.weight_allocator.reset(new ncnn::VkWeightAllocator(device, static_cast<size_t>(preferred_weight_size)));
        weight_allocator = impl.weight_allocator.get();
    }
    if (!upload_batch)
        impl.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));

    const size_t packed_size = align_bytes(matrix.mxfp4_blocks.size(), 4);
    const size_t scales_size = align_bytes(matrix.mxfp4_scales.size(), 4);
    const size_t bias_size = static_cast<size_t>(output_columns) * sizeof(float);
    if (packed_size == 0 || scales_size == 0
        || packed_size > static_cast<size_t>(std::numeric_limits<int>::max())
        || scales_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {};

    std::span<const float> float_bias;
    std::span<const uint16_t> bf16_bias;
    if (bias && bias->dtype == DType::Float32)
    {
        float_bias = bias->float32_values();
        if (float_bias.size() != output_columns)
            return {};
    }
    else if (bias)
    {
        bf16_bias = bias->bfloat16_values();
        if (bf16_bias.size() != output_columns)
            return {};
    }

    const size_t storage_alignment = std::max<size_t>(4, device->info.buffer_offset_alignment());
    const size_t packed_seg = align_bytes(packed_size, storage_alignment);
    const size_t scales_seg = align_bytes(scales_size, storage_alignment);
    const size_t bias_seg = align_bytes(bias_size, storage_alignment);
    if (packed_seg == 0 || scales_seg == 0 || bias_seg == 0
        || packed_seg > std::numeric_limits<size_t>::max() - scales_seg
        || packed_seg + scales_seg > std::numeric_limits<size_t>::max() - bias_seg)
        return {};

    const size_t scales_offset = packed_seg;
    const size_t bias_offset = packed_seg + scales_seg;
    const size_t storage_size = bias_offset + bias_seg;
    if (storage_size == 0 || storage_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {};

    const bool direct_staging = upload_batch
                                && dynamic_cast<VkWeightAllocator*>(weight_allocator)
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

    pack_mxfp4_weight_storage(packed_storage, packed_seg, scales_seg,
                              {matrix.mxfp4_blocks.data(), matrix.mxfp4_blocks.size()},
                              {matrix.mxfp4_scales.data(), matrix.mxfp4_scales.size()},
                              float_bias, bf16_bias);

    const int shader_len = static_cast<int>(sizeof(mxfp4_projection_shader) - 1);
    if (upload_batch)
    {
        if (!create_mxfp4_pipeline(impl.vulkan_context, impl.option, mxfp4_projection_shader, shader_len, impl.pipeline))
            return {};
    }
    else
    {
        const std::lock_guard<std::mutex> lock(impl.vulkan_context->command_mutex());
        if (!create_mxfp4_pipeline(impl.vulkan_context, impl.option, mxfp4_projection_shader, shader_len, impl.pipeline))
            return {};
    }

    if (!valid_dispatch_columns(impl.pipeline.get(), output_columns))
        return {};

    bool uploaded = false;
    if (upload_batch)
    {
        uploaded = direct_staging
                       ? upload_batch->record_prepared_storage(impl.storage, weight_allocator)
                       : upload_batch->record(combined, impl.storage, impl.option, weight_allocator);
    }
    else
    {
        ncnn::Option upload_opt = impl.option;
        upload_opt.blob_vkallocator = weight_allocator;
        upload_opt.workspace_vkallocator = weight_allocator;
        upload_opt.staging_vkallocator = impl.weight_staging_allocator.get();

        // Serialize background weight uploads with foreground context compute users.
        const std::lock_guard<std::mutex> lock(impl.vulkan_context->command_mutex());
        ncnn::VkTransfer cmd(device);
        cmd.record_upload(combined, impl.storage, upload_opt);
        uploaded = !impl.storage.empty() && cmd.submit_and_wait() == 0;
        impl.weight_staging_allocator.reset();
    }
    if (!uploaded)
        return {};

    impl.packed = byte_storage_view(impl.storage, 0, packed_size);
    impl.scales = byte_storage_view(impl.storage, scales_offset, scales_size);
    impl.bias = byte_storage_view(impl.storage, bias_offset, bias_size);
    impl.indexed_storage = true;
    impl.indexed_scales_word_offset = static_cast<uint32_t>(scales_offset / sizeof(uint32_t));
    impl.indexed_bias_word_offset = static_cast<uint32_t>(bias_offset / sizeof(uint32_t));
    return result;
}

bool Mxfp4Linear_vulkan::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
    const Implementation& impl = *d;
    VulkanRuntimeState& state = impl.vulkan_context->runtime_state();
    VkTransferLease lease = impl.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& slot = lease.slot();

    const size_t in_bytes = static_cast<size_t>(input.rows()) * impl.input_columns * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(input.rows()) * impl.output_columns * sizeof(float);
    const bool direct_in = impl.vulkan_context->support_direct_host_buffer(in_bytes, input.dtype());
    const bool direct_out = impl.vulkan_context->support_direct_host_buffer(out_bytes, output.dtype());
    if (!fill_staging_upload(input, slot.upload, slot.staging_allocator)
        || !prepare_staging_batch(slot.download, input.rows(), impl.output_columns, slot.staging_allocator))
        return false;

    const DType output_dtype = output.dtype();
    output.reset(input.rows(), impl.output_columns, false);

    const std::lock_guard<std::mutex> lock(impl.vulkan_context->command_mutex());
    ncnn::VkCompute& cmd = *slot.command;
    if (slot.command_used && cmd.reset() != 0)
        return false;
    slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (direct_in)
        input_gpu = bind_host_in(slot.upload);
    else if (!record_mapped_upload(slot.upload, input_gpu, cmd, impl.option))
        return false;

    ncnn::VkMat output_gpu;
    if (direct_out)
        output_gpu = bind_host_out(slot.download);
    else
        output_gpu.create(static_cast<int>(impl.output_columns), static_cast<int>(input.rows()),
                          sizeof(float), impl.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings = {input_gpu, impl.packed, impl.scales, impl.bias, output_gpu};
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = impl.input_columns;
    constants[1].u32 = impl.output_columns;
    constants[2].u32 = impl.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.rows());

    ncnn::VkMat dispatch;
    dispatch.w = static_cast<int>(static_cast<uint64_t>(impl.output_columns) * impl.pipeline->local_size_x());
    dispatch.h = static_cast<int>(input.rows());
    dispatch.c = 1;
    cmd.record_pipeline(impl.pipeline.get(), bindings, constants, dispatch);

    if ((!direct_out && !record_staging_download(output_gpu, input.rows(), impl.output_columns, slot.download, cmd, impl.vulkan_context->device(), impl.option, output_dtype))
        || submit_wait(cmd, impl.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(slot.download, output))
        return false;

    ++state.dispatches;
    ++state.compute_submissions;
    ++state.batch_uploads;
    ++state.batch_downloads;
    return true;
}

uint32_t Mxfp4Linear_vulkan::input_columns() const noexcept
{
    return d->input_columns;
}

uint32_t Mxfp4Linear_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

// --- Mxfp4Expert_vulkan ------------------------------------------------------

bool Mxfp4Expert_vulkan::init_pipelines(Implementation& impl, uint64_t optimization_flags, bool hold_lock)
{
    auto& gate = *impl.gate_up->d;
    const bool row_tile = has_flag(optimization_flags, OptimizationVulkanMxfp4RowTile);
    const int gate_len = static_cast<int>(sizeof(mxfp4_gate_up_shader) - 1);
    const int tile_len = static_cast<int>(sizeof(mxfp4_row_tile_shader) - 1);

    std::unique_lock<std::mutex> lock(gate.vulkan_context->command_mutex(), std::defer_lock);
    if (hold_lock)
        lock.lock();

    if (!create_mxfp4_pipeline(gate.vulkan_context, gate.option, mxfp4_gate_up_shader, gate_len, impl.gate_up_pipeline)
        || (row_tile && !create_mxfp4_pipeline(gate.vulkan_context, gate.option, mxfp4_row_tile_shader, tile_len, impl.row_tile_pipeline)))
        return false;

    const uint32_t intermediate_columns = gate.output_columns / 2;
    if (!valid_dispatch_columns(impl.gate_up_pipeline.get(), intermediate_columns))
        return false;
    if (impl.row_tile_pipeline
        && (!valid_dispatch_columns(impl.row_tile_pipeline.get(), intermediate_columns)
            || !valid_dispatch_columns(impl.row_tile_pipeline.get(), impl.down->d->output_columns)))
        return false;

    return true;
}

Mxfp4Expert_vulkan::Mxfp4Expert_vulkan()
    : d(new Implementation)
{
}

Mxfp4Expert_vulkan::~Mxfp4Expert_vulkan() = default;

std::shared_ptr<Mxfp4Expert_vulkan> Mxfp4Expert_vulkan::create(const TensorData& gate_up,
                                                               const TensorData* gate_up_bias,
                                                               const TensorData& down,
                                                               const TensorData* down_bias,
                                                               float activation_limit,
                                                               uint32_t vulkan_device_index,
                                                               ExpertActivation activation,
                                                               const VulkanRuntimePtr& vulkan_runtime,
                                                               uint64_t optimization_flags)
{
    return create_with_allocator(gate_up, gate_up_bias, down, down_bias, activation_limit,
                                 vulkan_device_index, nullptr, activation,
                                 vulkan_runtime, optimization_flags);
}

std::shared_ptr<Mxfp4Expert_vulkan> Mxfp4Expert_vulkan::create_with_allocator(const TensorData& gate_up,
                                                                              const TensorData* gate_up_bias,
                                                                              const TensorData& down,
                                                                              const TensorData* down_bias,
                                                                              float activation_limit,
                                                                              uint32_t vulkan_device_index,
                                                                              ncnn::VkAllocator* weight_allocator,
                                                                              ExpertActivation activation,
                                                                              const VulkanRuntimePtr& vulkan_runtime,
                                                                              uint64_t optimization_flags,
                                                                              VkWeightBatch* upload_batch)
{
    if (gate_up.shape.size() != 2 || gate_up.shape[0] % 2 != 0 || down.shape.size() != 2
        || down.shape[1] != gate_up.shape[0] / 2 || activation_limit < 0.0f
        || !gate_up.input_column_indices.empty() || !down.input_column_indices.empty())
        return {};

    auto gate_proj = Mxfp4Linear_vulkan::create_with_allocator(gate_up,
                                                               gate_up_bias,
                                                               vulkan_device_index,
                                                               weight_allocator,
                                                               vulkan_runtime,
                                                               optimization_flags,
                                                               upload_batch);
    auto down_proj = Mxfp4Linear_vulkan::create_with_allocator(down,
                                                               down_bias,
                                                               vulkan_device_index,
                                                               weight_allocator,
                                                               vulkan_runtime,
                                                               optimization_flags,
                                                               upload_batch);
    if (!gate_proj || !down_proj || gate_proj->d->vulkan_context != down_proj->d->vulkan_context)
        return {};

    std::shared_ptr<Mxfp4Expert_vulkan> result(new Mxfp4Expert_vulkan);
    if (upload_batch)
        upload_batch->retain_owner(result);

    Implementation& impl = *result->d;
    impl.gate_up = std::move(gate_proj);
    impl.down = std::move(down_proj);
    impl.activation_limit = activation_limit;
    impl.activation = activation;

    if (!init_pipelines(impl, optimization_flags, !upload_batch))
        return {};
    return result;
}

std::shared_ptr<Mxfp4Expert_vulkan> Mxfp4Expert_vulkan::create_from_device_storage(const Mxfp4DeviceMatrixView_vulkan& gate_up,
                                                                                   const TensorData* gate_up_bias,
                                                                                   const Mxfp4DeviceMatrixView_vulkan& down,
                                                                                   const TensorData* down_bias,
                                                                                   float activation_limit,
                                                                                   uint32_t vulkan_device_index,
                                                                                   const ncnn::VkMat& storage,
                                                                                   ExpertActivation activation,
                                                                                   const VulkanRuntimePtr& vulkan_runtime,
                                                                                   uint64_t optimization_flags)
{
#if NCNN_BATCH
    if (storage.empty() || gate_up.output_columns == 0 || gate_up.output_columns % 2 != 0
        || gate_up.input_columns == 0 || down.output_columns == 0
        || down.input_columns != gate_up.output_columns / 2 || activation_limit < 0.0f)
        return {};

    const auto create_projection = [&](const Mxfp4DeviceMatrixView_vulkan& matrix,
                                       const TensorData* bias) -> std::shared_ptr<Mxfp4Linear_vulkan> {
        if (matrix.output_columns == 0 || matrix.input_columns == 0 || matrix.input_columns % 32 != 0)
            return {};

        const uint32_t output_columns = matrix.output_columns;
        const uint32_t input_columns = matrix.input_columns;
        const uint32_t block_count = input_columns / 32;
        const uint64_t expected_blocks = static_cast<uint64_t>(output_columns) * block_count * 16;
        const uint64_t expected_scales = static_cast<uint64_t>(output_columns) * block_count;
        if (matrix.packed_size != expected_blocks || matrix.scales_size != expected_scales
            || expected_blocks > std::numeric_limits<int>::max()
            || expected_scales > std::numeric_limits<int>::max()
            || matrix.packed_offset > storage.buffer_capacity()
            || matrix.scales_offset > storage.buffer_capacity()
            || expected_blocks > storage.buffer_capacity() - matrix.packed_offset
            || expected_scales > storage.buffer_capacity() - matrix.scales_offset)
            return {};

        if (bias && (bias->shape.size() != 1 || bias->shape[0] != output_columns || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
            return {};

        std::shared_ptr<Mxfp4Linear_vulkan> proj(new Mxfp4Linear_vulkan);
        Mxfp4Linear_vulkan::Implementation& impl = *proj->d;
        impl.input_columns = input_columns;
        impl.output_columns = output_columns;
        impl.block_count = block_count;
        impl.vulkan_context = VulkanContext::acquire(vulkan_device_index, vulkan_runtime, optimization_flags);
        if (!impl.vulkan_context)
            return {};

        ncnn::VulkanDevice* device = impl.vulkan_context->device();
        const size_t alignment = std::max<size_t>(4, device->info.buffer_offset_alignment());
        if ((storage.buffer_offset() + matrix.packed_offset) % alignment != 0
            || (storage.buffer_offset() + matrix.scales_offset) % alignment != 0)
            return {};

        impl.option = make_vk_opt(*impl.vulkan_context);
        impl.option.use_cooperative_matrix = device->info.support_cooperative_matrix();
        impl.packed = byte_storage_view(storage, matrix.packed_offset,
                                        align_bytes(static_cast<size_t>(expected_blocks), 4));
        impl.scales = byte_storage_view(storage, matrix.scales_offset,
                                        align_bytes(static_cast<size_t>(expected_scales), 4));

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
                return {};
            std::copy(values.begin(), values.end(), bias_values);
        }
        else
        {
            const auto values = bias->bfloat16_values();
            if (values.size() != output_columns)
                return {};
            for (uint32_t i = 0; i < output_columns; ++i)
                bias_values[i] = bfloat16_to_float(values[i]);
        }

        impl.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));
        ncnn::Option upload_opt = impl.option;
        upload_opt.staging_vkallocator = impl.weight_staging_allocator.get();
        {
            const std::lock_guard<std::mutex> lock(impl.vulkan_context->command_mutex());
            const int shader_len = static_cast<int>(sizeof(mxfp4_projection_shader) - 1);
            if (!create_mxfp4_pipeline(impl.vulkan_context, impl.option, mxfp4_projection_shader, shader_len, impl.pipeline)
                || !valid_dispatch_columns(impl.pipeline.get(), output_columns))
                return {};

            ncnn::VkTransfer cmd(device);
            cmd.record_upload(biases, impl.bias, upload_opt);
            if (impl.bias.empty() || cmd.submit_and_wait() != 0)
                return {};
        }
        impl.weight_staging_allocator.reset();
        return proj;
    };

    auto gate_proj = create_projection(gate_up, gate_up_bias);
    auto down_proj = create_projection(down, down_bias);
    if (!gate_proj || !down_proj || gate_proj->d->vulkan_context != down_proj->d->vulkan_context)
        return {};

    std::shared_ptr<Mxfp4Expert_vulkan> result(new Mxfp4Expert_vulkan);
    Implementation& impl = *result->d;
    impl.gate_up = std::move(gate_proj);
    impl.down = std::move(down_proj);
    impl.activation_limit = activation_limit;
    impl.activation = activation;

    if (!init_pipelines(impl, optimization_flags, true))
        return {};
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
    const Implementation& impl = *d;
    const Mxfp4Linear_vulkan::Implementation& gate = *impl.gate_up->d;
    const Mxfp4Linear_vulkan::Implementation& down = *impl.down->d;
    const uint32_t intermediate_columns = gate.output_columns / 2;
    VulkanRuntimeState& state = gate.vulkan_context->runtime_state();

    VkTransferLease lease = gate.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& slot = lease.slot();
    if (!fill_staging_upload(input, slot.upload, slot.staging_allocator)
        || !prepare_staging_batch(slot.download, input.rows(), down.output_columns, slot.staging_allocator))
        return false;

    const DType output_dtype = output.dtype();
    output.reset(input.rows(), down.output_columns, false);

    const std::lock_guard<std::mutex> lock(gate.vulkan_context->command_mutex());
    ncnn::VkCompute& cmd = *slot.command;
    if (slot.command_used && cmd.reset() != 0)
        return false;
    slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(slot.upload, input_gpu, cmd, gate.option))
        return false;

    ncnn::VkMat intermediate_gpu(static_cast<int>(intermediate_columns), static_cast<int>(input.rows()),
                                 sizeof(float), gate.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu(static_cast<int>(down.output_columns), static_cast<int>(input.rows()),
                           sizeof(float), gate.vulkan_context->blob_allocator());
    if (intermediate_gpu.empty() || output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings(5);
    std::vector<ncnn::vk_constant_type> constants(7);
    record(input_gpu, intermediate_gpu, output_gpu, cmd, bindings, constants);

    if (!record_staging_download(output_gpu, input.rows(), down.output_columns,
                                 slot.download, cmd, gate.vulkan_context->device(),
                                 down.option, output_dtype)
        || submit_wait(cmd, gate.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(slot.download, output))
        return false;

    state.dispatches += 2;
    ++state.compute_submissions;
    ++state.batch_uploads;
    ++state.batch_downloads;
    return true;
}

void Mxfp4Expert_vulkan::record(const ncnn::VkMat& input,
                                ncnn::VkMat& intermediate,
                                ncnn::VkMat& output,
                                ncnn::VkCompute& cmd,
                                std::vector<ncnn::VkMat>& bindings,
                                std::vector<ncnn::vk_constant_type>& constants) const
{
    const Implementation& impl = *d;
    const Mxfp4Linear_vulkan::Implementation& gate = *impl.gate_up->d;
    const Mxfp4Linear_vulkan::Implementation& down = *impl.down->d;
    const uint32_t intermediate_columns = gate.output_columns / 2;
    const uint32_t rows = static_cast<uint32_t>(input.h);
    const uint32_t act_mode = impl.activation == ExpertActivation::Silu             ? 2
                              : impl.activation == ExpertActivation::DeepSeekSwiGlu ? 1
                                                                                    : 0;

    // Four-row register reuse amortizes overhead on prefill; short prompts stay scalar.
    if (rows >= 16 && impl.row_tile_pipeline)
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
        constants[5].u32 = act_mode;
        constants[6].f = impl.activation_limit;

        ncnn::VkMat dispatch;
        dispatch.w = static_cast<int>(static_cast<uint64_t>(intermediate_columns) * impl.row_tile_pipeline->local_size_x());
        dispatch.h = static_cast<int>((rows + 3) / 4);
        dispatch.c = 1;
        cmd.record_pipeline(impl.row_tile_pipeline.get(), bindings, constants, dispatch);

        bindings[0] = intermediate;
        bindings[1] = down.packed;
        bindings[2] = down.scales;
        bindings[3] = down.bias;
        bindings[4] = output;
        constants[0].u32 = down.input_columns;
        constants[1].u32 = down.output_columns;
        constants[2].u32 = down.block_count;
        constants[4].u32 = 1;
        dispatch.w = static_cast<int>(static_cast<uint64_t>(down.output_columns) * impl.row_tile_pipeline->local_size_x());
        cmd.record_pipeline(impl.row_tile_pipeline.get(), bindings, constants, dispatch);
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
    constants[4].u32 = act_mode;
    constants[5].f = impl.activation_limit;

    ncnn::VkMat gate_dispatch;
    gate_dispatch.w = static_cast<int>(static_cast<uint64_t>(intermediate_columns) * impl.gate_up_pipeline->local_size_x());
    gate_dispatch.h = static_cast<int>(rows);
    gate_dispatch.c = 1;
    cmd.record_pipeline(impl.gate_up_pipeline.get(), bindings, constants, gate_dispatch);

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

    ncnn::VkMat down_dispatch;
    down_dispatch.w = static_cast<int>(static_cast<uint64_t>(down.output_columns) * down.pipeline->local_size_x());
    down_dispatch.h = static_cast<int>(rows);
    down_dispatch.c = 1;
    cmd.record_pipeline(down.pipeline.get(), bindings, constants, down_dispatch);
}

} // namespace moe
} // namespace ncnn
