#include "backends/ncnn/core/context.h"
#include "backends/ncnn/core/tensor.h"
#include "backends/ncnn/linear/core.h"
#include "backends/ncnn/linear/internal.h"
#include "kernels/ops.h"
#include "kernels/vulkan/projection_matvec.comp.hex.h"

#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <layer.h>
#include <layer/vulkan/innerproduct_vulkan.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>
#include <pipeline.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

// --- Shared linear projection helpers ----------------------------------------

bool matvec_supported(const std::shared_ptr<VulkanContext>& context,
                      uint32_t input_columns, uint32_t output_columns,
                      bool bfloat16)
{
    const auto& limits = context->device()->info.physicalDeviceProperties().limits;
    const uint64_t outputs_per_group = bfloat16 ? 2 : 1;
    const uint64_t groups = (static_cast<uint64_t>(output_columns) + outputs_per_group - 1) / outputs_per_group;
    const uint64_t weight_bytes = static_cast<uint64_t>(input_columns) * output_columns * (bfloat16 ? 2 : 4);
    return input_columns != 0 && output_columns != 0
           && limits.maxComputeWorkGroupInvocations >= 128 && limits.maxComputeWorkGroupSize[0] >= 128
           && limits.maxComputeSharedMemorySize >= 2 * 128 * sizeof(float)
           && groups <= limits.maxComputeWorkGroupCount[0]
           && groups <= static_cast<uint64_t>(std::numeric_limits<int>::max() / 128)
           && weight_bytes <= limits.maxStorageBufferRange
           && static_cast<uint64_t>(input_columns) * sizeof(float) <= limits.maxStorageBufferRange
           && static_cast<uint64_t>(output_columns) * sizeof(float) <= limits.maxStorageBufferRange;
}

bool create_matvec_pipeline(const std::shared_ptr<VulkanContext>& context,
                            const ncnn::Option& option, bool bfloat16,
                            uint32_t input_columns, uint32_t output_columns,
                            std::shared_ptr<ncnn::Pipeline>& output)
{
    const uint32_t input_pack = !bfloat16 && input_columns % 4 == 0 ? 4 : 1;
    const uint32_t output_pack = !bfloat16 && output_columns % 4 == 0 ? 4 : 1;
    const uint64_t key = static_cast<uint64_t>(bfloat16) | (static_cast<uint64_t>(input_pack) << 8)
                         | (static_cast<uint64_t>(output_pack) << 12)
                         | (static_cast<uint64_t>(option.use_subgroup_ops) << 16);

    std::vector<ncnn::vk_specialization_type> specs(3);
    specs[0].u32 = bfloat16 ? 1 : 0;
    specs[1].u32 = input_pack;
    specs[2].u32 = output_pack;

    return create_pipeline(context, projection_matvec_shader,
                           static_cast<int>(sizeof(projection_matvec_shader) - 1),
                           option, 0, key, 128, output, specs, 0, 1, 1, true);
}

bool single_row_input(const ncnn::VkMat& input, uint32_t columns)
{
    return !input.empty() && input.dims == 2 && input.w == static_cast<int>(columns)
           && input.h == 1 && input.elempack == 1 && input.elemsize == sizeof(float)
#if NCNN_BATCH
           && input.n == 1
#endif
        ;
}

void record_matvec(const ncnn::Pipeline* pipeline, bool bfloat16,
                   uint32_t input_columns, uint32_t output_columns, bool has_bias,
                   const ncnn::VkMat& input, const ncnn::VkMat& weights,
                   const ncnn::VkMat& bias, ncnn::VkMat& output, ncnn::VkCompute& command)
{
    std::vector<ncnn::vk_constant_type> constants(3);
    constants[0].u32 = input_columns;
    constants[1].u32 = output_columns;
    constants[2].u32 = has_bias ? 1 : 0;

    ncnn::VkMat dispatch;
    const uint64_t outputs_per_group = bfloat16 ? 2 : 1;
    dispatch.w = static_cast<int>(((static_cast<uint64_t>(output_columns) + outputs_per_group - 1) / outputs_per_group) * 128);
    dispatch.h = 1;
    dispatch.c = 1;
    command.record_pipeline_readonly(pipeline, {input, weights, bias, output}, {1, 0, 0, 0}, constants, dispatch);
}

ncnn::VkMat aligned_row_view(const ncnn::VkMat& source,
                             size_t first_row, size_t rows,
                             ncnn::VkCompute& command,
                             const ncnn::Option& option,
                             const std::shared_ptr<VulkanContext>& context,
                             std::vector<ncnn::VkMat>& retained,
                             size_t reserve_size)
{
    // Storage descriptors may require stricter offset alignment than row byte stride.
    ncnn::VkMat view = row_view(source, first_row, rows);
    const size_t alignment = std::max<size_t>(4, context->device()->info.buffer_offset_alignment());
    if (view.empty() || view.buffer_offset() % alignment == 0)
        return view;
    if (view.buffer_offset() % 4 != 0 || view.buffer_capacity() % 4 != 0)
        return {};
    if (retained.capacity() == 0 && reserve_size != 0)
        retained.reserve(reserve_size);

    ncnn::VkMat aligned;
    command.record_clone(view, aligned, option);
    if (aligned.empty())
        return {};
    retained.push_back(aligned);
    return aligned;
}

bool record_row_copy(const ncnn::VkMat& source, ncnn::VkMat& destination,
                     ncnn::VkCompute& command, const ncnn::Option& option)
{
    if (source.empty() || destination.empty() || source.buffer_offset() % 4 != 0
        || destination.buffer_offset() % 4 != 0 || destination.buffer_capacity() % 4 != 0)
        return false;

    // Bind blob_vkallocator to destination.allocator so record_clone writes in place.
    ncnn::Option copy_opt = option;
    copy_opt.blob_vkallocator = destination.allocator;
    const VkBuffer buffer = destination.buffer();
    const size_t offset = destination.buffer_offset();
    command.record_clone(source, destination, copy_opt);
    return !destination.empty() && destination.buffer() == buffer && destination.buffer_offset() == offset;
}

// --- Linear ------------------------------------------------------------------

class Linear::Implementation
{
public:
    ~Implementation()
    {
        if (layer)
        {
            const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
            if (pipeline_created)
                layer->destroy_pipeline(option);
            delete layer;
        }
        weight_staging_allocator.reset();
        weight_allocator.reset();
    }

    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> single_row_pipeline;
    ncnn::Layer* layer = nullptr;
    ncnn::Option option;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    bool pipeline_created = false;
    bool has_bias = false;
};

Linear::Linear()
    : d(new Implementation)
{
}

Linear::~Linear() = default;

std::shared_ptr<Linear> Linear::create(const TensorData& matrix, const TensorData* bias,
                                       uint32_t vulkan_device_index,
                                       const VulkanRuntimePtr& vulkan_runtime,
                                       uint64_t optimization_flags)
{
    std::shared_ptr<Linear> linear(new Linear);
    Implementation& impl = *linear->d;
    impl.input_columns = matrix.shape[1];
    impl.output_columns = matrix.shape[0];
    impl.has_bias = bias != nullptr;

    impl.vulkan_context = VulkanContext::acquire(vulkan_device_index, vulkan_runtime, optimization_flags);
    if (!impl.vulkan_context)
        return {};

    ncnn::VulkanDevice* vkdev = impl.vulkan_context->device();
    impl.option = make_vk_opt(*impl.vulkan_context);
    impl.option.use_cooperative_matrix = vkdev->info.support_cooperative_matrix();

    impl.layer = ncnn::create_layer_vulkan(ncnn::LayerType::InnerProduct);
    if (!impl.layer)
        return {};
    impl.layer->vkdev = vkdev;

    ncnn::ParamDict params;
    params.set(0, static_cast<int>(impl.output_columns));
    params.set(1, bias ? 1 : 0);
    params.set(2, static_cast<int>(matrix.element_count()));
    if (impl.layer->load_param(params) != 0)
        return {};

    ncnn::Mat model_data[2];
    if (!prepare_float_upload(matrix, model_data[0]))
        return {};
    if (bias && !prepare_float_upload(*bias, model_data[1]))
        return {};

    if (impl.layer->load_model(ncnn::ModelBinFromMatArray(model_data)) != 0
        || impl.layer->create_pipeline(impl.option) != 0)
        return {};
    impl.pipeline_created = true;

    impl.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
    impl.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(vkdev));
    {
        const std::lock_guard<std::mutex> lock(impl.vulkan_context->command_mutex());
        ncnn::VkTransfer cmd(vkdev);
        ncnn::Option upload_opt = impl.option;
        upload_opt.blob_vkallocator = impl.weight_allocator.get();
        upload_opt.workspace_vkallocator = impl.weight_allocator.get();
        upload_opt.staging_vkallocator = impl.weight_staging_allocator.get();
        if (impl.layer->upload_model(cmd, upload_opt) != 0 || cmd.submit_and_wait() != 0)
            return {};

        // InnerProduct_vulkan::upload_model can succeed despite an empty GPU allocation.
        const auto& inner_product = *static_cast<const ncnn::InnerProduct_vulkan*>(impl.layer);
        if (inner_product.weight_data_gpu.empty() || (bias && inner_product.bias_data_gpu.empty()))
            return {};
    }

    if (matvec_supported(impl.vulkan_context, impl.input_columns, impl.output_columns, false)
        && !create_matvec_pipeline(impl.vulkan_context, impl.option, false,
                                   impl.input_columns, impl.output_columns, impl.single_row_pipeline))
        return {};

    return linear;
}

std::shared_ptr<Linear> Linear::create_fused(const std::vector<const TensorData*>& matrices,
                                             const std::vector<const TensorData*>& biases,
                                             uint32_t vulkan_device_index,
                                             const VulkanRuntimePtr& vulkan_runtime,
                                             uint64_t optimization_flags)
{
    if (matrices.empty() || matrices.size() != biases.size() || !matrices.front())
        return {};

    const DType dtype = matrices.front()->dtype;
    if (matrices.front()->shape.size() != 2 || (dtype != DType::Float32 && dtype != DType::BFloat16))
        return {};

    const uint32_t input_columns = matrices.front()->shape[1];
    const bool has_bias = biases.front() != nullptr;
    uint64_t output_columns = 0;
    uint64_t element_count = 0;

    for (size_t i = 0; i < matrices.size(); ++i)
    {
        const TensorData* matrix = matrices[i];
        const TensorData* bias = biases[i];
        if (!matrix || matrix->dtype != dtype || matrix->shape.size() != 2
            || matrix->shape[1] != input_columns || (bias != nullptr) != has_bias)
            return {};
        if (bias && (bias->dtype != dtype || bias->shape.size() != 1 || bias->shape[0] != matrix->shape[0]))
            return {};
        output_columns += matrix->shape[0];
        element_count += matrix->element_count();
    }
    if (output_columns > std::numeric_limits<uint32_t>::max())
        return {};

    TensorData fused_matrix;
    fused_matrix.dtype = dtype;
    fused_matrix.shape = {static_cast<uint32_t>(output_columns), input_columns};

    TensorData fused_bias;
    fused_bias.dtype = dtype;
    fused_bias.shape = {static_cast<uint32_t>(output_columns)};

    if (dtype == DType::Float32)
    {
        fused_matrix.float32_data.reserve(element_count);
        if (has_bias)
            fused_bias.float32_data.reserve(output_columns);

        for (size_t i = 0; i < matrices.size(); ++i)
        {
            const std::span<const float> m = matrices[i]->float32_values();
            fused_matrix.float32_data.insert(fused_matrix.float32_data.end(), m.begin(), m.end());
            if (has_bias)
            {
                const std::span<const float> b = biases[i]->float32_values();
                fused_bias.float32_data.insert(fused_bias.float32_data.end(), b.begin(), b.end());
            }
        }
    }
    else
    {
        fused_matrix.bfloat16_data.reserve(element_count);
        if (has_bias)
            fused_bias.bfloat16_data.reserve(output_columns);

        for (size_t i = 0; i < matrices.size(); ++i)
        {
            const std::span<const uint16_t> m = matrices[i]->bfloat16_values();
            fused_matrix.bfloat16_data.insert(fused_matrix.bfloat16_data.end(), m.begin(), m.end());
            if (has_bias)
            {
                const std::span<const uint16_t> b = biases[i]->bfloat16_values();
                fused_bias.bfloat16_data.insert(fused_bias.bfloat16_data.end(), b.begin(), b.end());
            }
        }
    }

    return create(fused_matrix, has_bias ? &fused_bias : nullptr,
                  vulkan_device_index, vulkan_runtime, optimization_flags);
}

bool Linear::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
    const Implementation& impl = *d;
    VulkanRuntimeState& state = impl.vulkan_context->runtime_state();
    VkTransferLease lease = impl.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& slot = lease.slot();
    const DType output_dtype = output.dtype();

    if (!fill_staging_upload(input, slot.upload, slot.staging_allocator)
        || !prepare_staging_batch(slot.download, input.rows(), impl.output_columns,
                                  slot.staging_allocator, output.element_size()))
        return false;

    output.reset(input.rows(), impl.output_columns, false);
    const std::lock_guard<std::mutex> lock(impl.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev = impl.vulkan_context->device();
    ncnn::VkCompute& cmd = *slot.command;

    if (slot.command_used && cmd.reset() != 0)
        return false;
    slot.command_used = true;

    ncnn::VkMat bottom_gpu;
    if (!record_prepared_staging_upload(slot.upload, input.rows(), bottom_gpu, cmd,
                                        vkdev, impl.option, input.dtype()))
        return false;

    ncnn::VkMat top_gpu;
    if (forward(bottom_gpu, top_gpu, cmd, impl.option) != 0)
        return false;

    ncnn::VkMat download_gpu = top_gpu;
    if (top_gpu.elempack != 1)
    {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(top_gpu, unpacked, 1, cmd, impl.option);
        download_gpu = unpacked;
    }

    if (!record_staging_download(download_gpu, input.rows(), impl.output_columns,
                                 slot.download, cmd, vkdev, impl.option, output_dtype)
        || submit_wait(cmd, vkdev) != 0
        || !copy_staging_to_cpu_batch(slot.download, output))
        return false;

    ++state.dispatches;
    ++state.compute_submissions;
    ++state.batch_uploads;
    ++state.batch_downloads;
    return true;
}

int Linear::forward(const ncnn::VkMat& input, ncnn::VkMat& output,
                    ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    if (d->single_row_pipeline && single_row_input(input, d->input_columns))
    {
        ncnn::VkMat dst(static_cast<int>(d->output_columns), 1, sizeof(float), opt.blob_vkallocator);
        if (dst.empty())
            return -100;

        const auto& ip = *static_cast<const ncnn::InnerProduct_vulkan*>(d->layer);
        record_matvec(d->single_row_pipeline.get(), false, d->input_columns, d->output_columns,
                      d->has_bias, input, ip.weight_data_gpu, ip.bias_data_gpu, dst, cmd);
        output = dst;
        return 0;
    }
    return d->layer->forward(input, output, cmd, opt);
}

const std::shared_ptr<VulkanContext>& Linear::context() const noexcept
{
    return d->vulkan_context;
}

const ncnn::Option& Linear::option() const noexcept
{
    return d->option;
}

uint32_t Linear::input_columns() const noexcept
{
    return d->input_columns;
}

uint32_t Linear::output_columns() const noexcept
{
    return d->output_columns;
}

// --- CommandGraph_vulkan -----------------------------------------------------

class CommandGraph_vulkan::Implementation
{
public:
    struct PendingDownload
    {
        ncnn::VkMat staging;
        ActivationBuffer* output = nullptr;
    };

    std::shared_ptr<VulkanContext> context;
    ncnn::Option option;
    std::shared_ptr<CommandGraphState_vulkan> state;
    std::unique_ptr<VkTransferLease> transfer_lease;
    std::vector<ncnn::VkMat> upload_staging;
    std::vector<PendingDownload> pending_downloads;
    bool recorded = false;
    bool submitted = false;
    bool completed = false;
};

CommandGraph_vulkan::CommandGraph_vulkan(std::unique_ptr<Implementation> implementation)
    : d(std::move(implementation))
{
}

CommandGraph_vulkan::~CommandGraph_vulkan()
{
    if (d && d->submitted && !d->completed && d->context && d->transfer_lease)
        (void)wait();
    if (d && !d->submitted && d->context && d->transfer_lease)
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        (void)d->transfer_lease->slot().command->reset();
    }
}

CommandGraph_vulkan::CommandGraph_vulkan(CommandGraph_vulkan&&) noexcept = default;

CommandGraph_vulkan& CommandGraph_vulkan::operator=(CommandGraph_vulkan&& other) noexcept
{
    if (this != &other)
    {
        // Drain the replaced graph before dropping its command lease and owners.
        CommandGraph_vulkan retired(std::move(d));
        d = std::move(other.d);
    }
    return *this;
}

std::unique_ptr<CommandGraph_vulkan> CommandGraph_vulkan::create(const Linear& seed)
{
    if (!seed.context())
        return {};

    auto impl = std::make_unique<Implementation>();
    impl->context = seed.context();
    impl->option = seed.option();
    impl->state = std::make_shared<CommandGraphState_vulkan>();
    impl->state->storage_variant = activation_variant(impl->option);
    impl->transfer_lease = std::make_unique<VkTransferLease>(impl->context->acquire_transfer_slot());

    VkTransferSlot& slot = impl->transfer_lease->slot();
    const std::lock_guard<std::mutex> lock(impl->context->command_mutex());
    if (slot.command_used && slot.command->reset() != 0)
        return {};
    slot.command_used = true;
    return std::unique_ptr<CommandGraph_vulkan>(new CommandGraph_vulkan(std::move(impl)));
}

bool CommandGraph_vulkan::upload(const ActivationBuffer& input, DeviceTensor_vulkan& output)
{
    if (!d || d->submitted || d->state->failed || !d->context
        || input.rows() == 0 || input.columns() == 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;

    ncnn::VkMat staging;
    if (!fill_staging_upload(input, staging, d->transfer_lease->slot().staging_allocator))
        return false;

    ncnn::VkMat device_input;
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (!record_prepared_staging_upload(staging, input.rows(), device_input,
                                            *d->transfer_lease->slot().command,
                                            d->context->device(), d->option, input.dtype()))
            return false;
    }

    output.assign_pending(device_input, input.rows(), input.columns(), d->context, d->state);
    d->upload_staging.push_back(std::move(staging));
    d->recorded = true;
    return true;
}

bool CommandGraph_vulkan::linear(const Linear& op, const DeviceTensor_vulkan& input, DeviceTensor_vulkan& output)
{
    if (!d || d->submitted || d->state->failed || &input == &output || input.empty())
        return false;
    if (input.context().get() != d->context.get()
        || !input.readable_from(d->state, activation_variant(d->option)))
        return false;

    const std::shared_ptr<VulkanContext>& context = op.context();
    const ncnn::Option& opt = op.option();
    if (!context || context.get() != d->context.get() || op.input_columns() != input.columns()
        || activation_variant(opt) != activation_variant(d->option))
        return false;

    const std::shared_ptr<CommandGraphState_vulkan>& input_graph = input.graph_state();
    ncnn::VkMat device_output;
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (input_graph.get() != d->state.get())
            input.prepare_read();
        if (op.forward(input.raw_value(), device_output, *d->transfer_lease->slot().command, opt) != 0)
            return false;
        if (device_output.elempack != 1)
        {
            ncnn::VkMat unpacked;
            d->context->device()->convert_packing(device_output, unpacked, 1,
                                                  *d->transfer_lease->slot().command, opt);
            device_output = unpacked;
        }
    }
    if (device_output.empty())
        return false;

    output.assign_pending(device_output, input.rows(), op.output_columns(), d->context, d->state);
    d->recorded = true;
    return true;
}

bool CommandGraph_vulkan::download(const DeviceTensor_vulkan& input, ActivationBuffer& output)
{
    if (!d || d->submitted || d->state->failed || input.empty())
        return false;
    if (input.context().get() != d->context.get()
        || !input.readable_from(d->state, activation_variant(d->option)))
        return false;

    const std::shared_ptr<CommandGraphState_vulkan>& input_graph = input.graph_state();
    Implementation::PendingDownload pending;
    const DType output_dtype = output.dtype();
    if (!prepare_staging_batch(pending.staging, input.rows(), input.columns(),
                               d->transfer_lease->slot().staging_allocator, output.element_size()))
        return false;

    output.reset(input.rows(), input.columns(), false);
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (input_graph.get() != d->state.get())
            input.prepare_read();
        if (!record_staging_download(input.raw_value(), input.rows(), input.columns(),
                                     pending.staging, *d->transfer_lease->slot().command,
                                     d->context->device(), d->option, output_dtype))
            return false;
    }

    pending.output = &output;
    d->pending_downloads.push_back(std::move(pending));
    return true;
}

bool CommandGraph_vulkan::submit()
{
    if (!d || d->submitted || d->state->failed
        || (!d->recorded && d->pending_downloads.empty())
        || !d->context || !d->transfer_lease)
        return false;

    ncnn::VkCompute& cmd = *d->transfer_lease->slot().command;
    const std::lock_guard<std::mutex> lock(d->context->command_mutex());
    VulkanRuntimeState& state = d->context->runtime_state();
    if (cmd.submit() != 0)
    {
        d->state->failed = true;
        return false;
    }

    d->submitted = true;
    d->state->submitted = true;
    d->completed = false;
    ++state.compute_submissions;
    state.batch_uploads += d->upload_staging.size();
    state.batch_downloads += d->pending_downloads.size();
    return true;
}

bool CommandGraph_vulkan::wait()
{
    if (!d || !d->submitted || d->completed || !d->context || !d->transfer_lease)
        return false;

    ncnn::VkCompute& cmd = *d->transfer_lease->slot().command;
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        const int wait_result = wait_cmd(cmd, d->context->device());
        // Native CB is now complete (or lost); avoid a second wait on CPU copy errors.
        d->completed = true;
        if (wait_result != 0)
        {
            d->state->failed = true;
            return false;
        }
    }

    for (Implementation::PendingDownload& pending : d->pending_downloads)
    {
        if (!pending.output || !copy_staging_to_cpu_batch(pending.staging, *pending.output))
        {
            d->state->failed = true;
            return false;
        }
    }

    d->state->completed = true;
    return true;
}

} // namespace moe
} // namespace ncnn
