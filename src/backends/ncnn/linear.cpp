#include "linear_core.h"
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
#include "kernels/vulkan/projection_matvec.comp.hex.h"
#endif
#include "linear_internal.h"

namespace ncnn {
namespace moe {

class Linear::Implementation
{
public:
#if NCNN_MOE_USE_NCNN
    ~Implementation()
    {
        if (layer)
        {
#if NCNN_MOE_WITH_VULKAN
            const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
            if (pipeline_created)
                layer->destroy_pipeline(option);
#endif
            delete layer;
        }
#if NCNN_MOE_WITH_VULKAN
        weight_staging_allocator.reset();
        weight_allocator.reset();
#endif
    }

    ncnn::Layer* layer = nullptr;
    ncnn::Option option;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    bool pipeline_created = false;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> single_row_pipeline;
    bool has_bias = false;
#endif
#endif
};

#if NCNN_MOE_WITH_VULKAN
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
    const auto binary = context->shader_binary(projection_matvec_shader,
                                               static_cast<int>(sizeof(projection_matvec_shader) - 1), option, 0);
    if (!binary || binary->empty()) return false;
    auto pipeline = std::shared_ptr<ncnn::Pipeline>(new ncnn::Pipeline(context->device()), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        context.reset();
    });
    const uint32_t input_pack = !bfloat16 && input_columns % 4 == 0 ? 4 : 1;
    const uint32_t output_pack = !bfloat16 && output_columns % 4 == 0 ? 4 : 1;
    const uint64_t key = static_cast<uint64_t>(bfloat16) | (static_cast<uint64_t>(input_pack) << 8)
                         | (static_cast<uint64_t>(output_pack) << 12) | (static_cast<uint64_t>(option.use_subgroup_ops) << 16);
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    output = context->find_pipeline(projection_matvec_shader, key);
    if (output) return true;
    std::vector<ncnn::vk_specialization_type> specializations(3);
    specializations[0].u32 = bfloat16 ? 1 : 0;
    specializations[1].u32 = input_pack;
    specializations[2].u32 = output_pack;
    pipeline->set_local_size_xyz(128, 1, 1);
    if (pipeline->create(binary->data(), binary->size() * sizeof(uint32_t), specializations) != 0) return false;
    output = pipeline;
    context->cache_pipeline(projection_matvec_shader, key, output);
    return true;
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
#endif

#if NCNN_MOE_WITH_VULKAN
struct CommandGraphState_vulkan
{
    size_t storage_variant = 0;
    bool submitted = false;
    bool completed = false;
    bool failed = false;
};
#endif

#if NCNN_MOE_WITH_VULKAN
// Row offsets are byte offsets, while storage descriptors have a stricter
// device alignment. Keep the normal view path and copy only misaligned reads.
ncnn::VkMat aligned_row_view(const ncnn::VkMat& source,
                             size_t first_row, size_t rows,
                             ncnn::VkCompute& command,
                             const ncnn::Option& option,
                             const std::shared_ptr<VulkanContext>& context,
                             std::vector<ncnn::VkMat>& retained,
                             size_t reserve_size)
{
    ncnn::VkMat view = row_view(source, first_row, rows);
    const size_t alignment = std::max<size_t>(4, context->device()->info.buffer_offset_alignment());
    if (view.empty() || view.buffer_offset() % alignment == 0) return view;
    if (view.buffer_offset() % 4 != 0 || view.buffer_capacity() % 4 != 0) return {};
    if (retained.capacity() == 0 && reserve_size != 0)
        retained.reserve(reserve_size);
    ncnn::VkMat aligned;
    command.record_clone(view, aligned, option);
    if (aligned.empty()) return {};
    retained.push_back(aligned);
    return aligned;
}

bool record_row_copy(const ncnn::VkMat& source, ncnn::VkMat& destination,
                     ncnn::VkCompute& command, const ncnn::Option& option)
{
    if (source.empty() || destination.empty() || source.buffer_offset() % 4 != 0
        || destination.buffer_offset() % 4 != 0 || destination.buffer_capacity() % 4 != 0)
        return false;
    // Keep create_like on the existing row view, including a staging-backed
    // destination, instead of silently allocating a different destination.
    ncnn::Option copy_option = option;
    copy_option.blob_vkallocator = destination.allocator;
    const VkBuffer buffer = destination.buffer();
    const size_t offset = destination.buffer_offset();
    command.record_clone(source, destination, copy_option);
    return !destination.empty() && destination.buffer() == buffer && destination.buffer_offset() == offset;
}

ncnn::VkMat bind_host_out(ncnn::VkMat& staging)
{
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return staging;
}
#endif

class DeviceTensor_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    // Retain the allocator domain and completion record after the producer
    // graph releases its command/staging resources.
    std::shared_ptr<VulkanContext> context;
    // Null marks an intrinsically completed unpacked FP32 publication.
    // Pending graph results always retain their producer's mutable state.
    std::shared_ptr<CommandGraphState_vulkan> graph;
    ncnn::VkMat value;
    size_t rows = 0;
    uint32_t columns = 0;

    [[nodiscard]] bool readable_from(const std::shared_ptr<CommandGraphState_vulkan>& target,
                                     size_t storage_variant) const noexcept
    {
        if (!graph) return storage_variant == 0;
        return !graph->failed && graph->storage_variant == storage_variant
               && (graph == target || (graph->submitted && graph->completed));
    }
#endif
};

class CommandGraph_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    struct PendingDownload
    {
        ncnn::VkMat staging;
        ActivationBuffer* output = nullptr;
    };

    std::shared_ptr<VulkanContext> context;
    ncnn::Option option;
    std::shared_ptr<CommandGraphState_vulkan> state;
    std::unique_ptr<VulkanTransferLease> transfer_lease;
    std::vector<ncnn::VkMat> upload_staging;
    std::vector<PendingDownload> pending_downloads;
    bool recorded = false;
    bool submitted = false;
    bool completed = false;
#endif
};

Linear::Linear()
    : d(new Implementation)
{
}

Linear::~Linear() = default;

DeviceTensor_vulkan::DeviceTensor_vulkan() = default;

DeviceTensor_vulkan::~DeviceTensor_vulkan() = default;

DeviceTensor_vulkan::DeviceTensor_vulkan(DeviceTensor_vulkan&&) noexcept = default;

DeviceTensor_vulkan& DeviceTensor_vulkan::operator=(DeviceTensor_vulkan&&) noexcept = default;

bool DeviceTensor_vulkan::empty() const noexcept
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    return !d || d->value.empty();
#else
    return true;
#endif
}

size_t DeviceTensor_vulkan::rows() const noexcept
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    return d ? d->rows : 0;
#else
    return 0;
#endif
}

uint32_t DeviceTensor_vulkan::columns() const noexcept
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    return d ? d->columns : 0;
#else
    return 0;
#endif
}

bool DeviceTensor_vulkan::assign_completed(const ncnn::VkMat& value,
                                           std::shared_ptr<VulkanContext> context)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!context || value.empty() || (value.dims != 1 && value.dims != 2)
        || value.elempack != 1 || value.elemsize != sizeof(float))
    {
        d.reset();
        return false;
    }
    auto tensor = std::make_unique<Implementation>();
    tensor->context = std::move(context);
    tensor->value = value;
    tensor->rows = value.dims == 1 ? 1 : static_cast<size_t>(value.h);
    tensor->columns = static_cast<uint32_t>(value.w);
    d = std::move(tensor);
    return true;
#else
    (void)value;
    (void)context;
    return false;
#endif
}

const ncnn::VkMat* DeviceTensor_vulkan::value() const noexcept
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    return d && (!d->graph || (d->graph->completed && !d->graph->failed)) && !d->value.empty()
               ? &d->value
               : nullptr;
#else
    return nullptr;
#endif
}

void DeviceTensor_vulkan::prepare_read() const noexcept
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    const ncnn::VkMat* completed = value();
    if (!completed || !completed->data)
        return;

    // ncnn changes shared buffer metadata while recording, so an abandoned
    // consumer can leave flags for a barrier that never ran. Re-arm every
    // completed import instead of restoring a snapshot that may overwrite a
    // later recorder's state. The producer may have used transfer or compute.
    completed->data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    completed->data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
#endif
}

const std::shared_ptr<VulkanContext>& DeviceTensor_vulkan::vulkan_context() const noexcept
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (d)
        return d->context;
#endif
    static const std::shared_ptr<VulkanContext> empty_context;
    return empty_context;
}

CommandGraph_vulkan::CommandGraph_vulkan(std::unique_ptr<Implementation> implementation)
    : d(std::move(implementation))
{
}

CommandGraph_vulkan::~CommandGraph_vulkan()
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (d && d->submitted
        && !d->completed
        && d->context
        && d->transfer_lease)
    {
        (void)wait();
    }
    if (d && !d->submitted
        && d->context && d->transfer_lease)
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        (void)d->transfer_lease->slot().command->reset();
    }
#endif
}

CommandGraph_vulkan::CommandGraph_vulkan(CommandGraph_vulkan&&) noexcept = default;

CommandGraph_vulkan& CommandGraph_vulkan::operator=(CommandGraph_vulkan&& other) noexcept
{
    if (this != &other)
    {
        // Preserve the old Graph's submitted-command cleanup when replacing
        // its implementation. A plain unique_ptr move would release owners
        // and return the old command slot without waiting for its fence.
        CommandGraph_vulkan retired(std::move(d));
        d = std::move(other.d);
    }
    return *this;
}

std::unique_ptr<CommandGraph_vulkan> CommandGraph_vulkan::create(const Linear& seed)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    auto implementation = std::make_unique<Implementation>();
    implementation->context = seed.vulkan_context();
    implementation->option = seed.option();
    implementation->state = std::make_shared<CommandGraphState_vulkan>();
    implementation->state->storage_variant = activation_variant(implementation->option);
    implementation->transfer_lease = std::make_unique<VulkanTransferLease>(implementation->context->acquire_transfer_slot());

    VulkanTransferSlot& transfer_slot = implementation->transfer_lease->slot();
    const std::lock_guard<std::mutex> lock(implementation->context->command_mutex());
    if (transfer_slot.command_used)
    {
        if (transfer_slot.command->reset() != 0)
            return {};
    }
    transfer_slot.command_used = true;
    return std::unique_ptr<CommandGraph_vulkan>(new CommandGraph_vulkan(std::move(implementation)));
#else
    (void)seed;
    return {};
#endif
}

bool CommandGraph_vulkan::upload(const ActivationBuffer& input,
                                 DeviceTensor_vulkan& output)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!d || d->submitted || d->state->failed
        || !d->context || input.rows() == 0
        || input.columns() == 0
        || input.rows()
               > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    ncnn::VkMat staging;
    if (!fill_staging_upload(input,
                             staging,
                             d->transfer_lease->slot().staging_allocator))
    {
        return false;
    }

    ncnn::VkMat device_input;
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (!record_prepared_staging_upload(staging,
                                            input.rows(),
                                            device_input,
                                            *d->transfer_lease->slot().command,
                                            d->context->device(),
                                            d->option,
                                            input.dtype()))
        {
            return false;
        }
    }

    auto tensor = std::make_unique<DeviceTensor_vulkan::Implementation>();
    tensor->context = d->context;
    tensor->graph = d->state;
    tensor->value = std::move(device_input);
    tensor->rows = input.rows();
    tensor->columns = input.columns();
    output.d = std::move(tensor);
    d->upload_staging.push_back(std::move(staging));
    d->recorded = true;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool CommandGraph_vulkan::linear(const Linear& op,
                                 const DeviceTensor_vulkan& input,
                                 DeviceTensor_vulkan& output)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!d || d->submitted || d->state->failed
        || !input.d
        || &input == &output || input.empty())
    {
        return false;
    }
    const std::shared_ptr<CommandGraphState_vulkan>& input_graph = input.d->graph;
    if (input.d->context.get() != d->context.get()
        || !input.d->readable_from(d->state, activation_variant(d->option)))
    {
        return false;
    }

    const std::shared_ptr<VulkanContext>& context = op.vulkan_context();
    const ncnn::Option& opt = op.option();
    if (!context
        || context.get()
               != d->context.get()
        || op.input_columns() != input.columns()
        || activation_variant(opt)
               != activation_variant(d->option))
    {
        return false;
    }

    ncnn::VkMat device_output;
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (input_graph.get() != d->state.get())
            input.prepare_read();
        if (op.forward(input.d->value,
                       device_output,
                       *d->transfer_lease->slot().command,
                       opt)
            != 0)
        {
            return false;
        }
        if (device_output.elempack != 1)
        {
            ncnn::VkMat unpacked;
            d->context->device()->convert_packing(device_output,
                                                  unpacked,
                                                  1,
                                                  *d->transfer_lease->slot().command,
                                                  opt);
            device_output = unpacked;
        }
    }
    if (device_output.empty())
        return false;

    auto tensor = std::make_unique<DeviceTensor_vulkan::Implementation>();
    tensor->context = d->context;
    tensor->graph = d->state;
    tensor->value = std::move(device_output);
    tensor->rows = input.rows();
    tensor->columns = op.output_columns();
    output.d = std::move(tensor);
    d->recorded = true;
    return true;
#else
    (void)op;
    (void)input;
    (void)output;
    return false;
#endif
}

bool CommandGraph_vulkan::download(const DeviceTensor_vulkan& input,
                                   ActivationBuffer& output)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!d || d->submitted || d->state->failed
        || !input.d || input.empty())
    {
        return false;
    }
    const std::shared_ptr<CommandGraphState_vulkan>& input_graph = input.d->graph;
    if (input.d->context.get() != d->context.get()
        || !input.d->readable_from(d->state, activation_variant(d->option)))
    {
        return false;
    }

    CommandGraph_vulkan::Implementation::PendingDownload pending;
    const DType output_dtype = output.dtype();
    if (!prepare_staging_batch(pending.staging,
                               input.rows(),
                               input.columns(),
                               d->transfer_lease->slot().staging_allocator,
                               output.element_size()))
    {
        return false;
    }
    output.reset(input.rows(), input.columns(), false);
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (input_graph.get() != d->state.get())
            input.prepare_read();
        if (!record_staging_download(input.d->value,
                                     input.rows(),
                                     input.columns(),
                                     pending.staging,
                                     *d->transfer_lease->slot().command,
                                     d->context->device(),
                                     d->option,
                                     output_dtype))
        {
            return false;
        }
    }
    pending.output = &output;
    d->pending_downloads.push_back(std::move(pending));
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool CommandGraph_vulkan::submit()
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!d || d->submitted || d->state->failed
        || (!d->recorded && d->pending_downloads.empty())
        || !d->context || !d->transfer_lease)
    {
        return false;
    }

    ncnn::VkCompute& command = *d->transfer_lease->slot().command;
    const std::lock_guard<std::mutex> lock(d->context->command_mutex());
    VulkanRuntimeState& runtime_state = d->context->runtime_state();
    if (command.submit() != 0)
    {
        d->state->failed = true;
        return false;
    }
    d->submitted = true;
    d->state->submitted = true;
    d->completed = false;
    ++runtime_state.compute_submissions;
    runtime_state.batch_uploads += d->upload_staging.size();
    runtime_state.batch_downloads += d->pending_downloads.size();
    return true;
#else
    return false;
#endif
}

bool CommandGraph_vulkan::wait()
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!d || !d->submitted
        || d->completed
        || !d->context || !d->transfer_lease)
    {
        return false;
    }

    ncnn::VkCompute& command = *d->transfer_lease->slot().command;
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        const int wait_result = wait_submitted_compute(command, d->context->device());
        // The native CB is now complete (or its device is lost). Later CPU
        // download failure must not wait it again after ncnn cleared submitted.
        d->completed = true;
        if (wait_result != 0)
        {
            d->state->failed = true;
            return false;
        }
    }

    for (CommandGraph_vulkan::Implementation::PendingDownload& pending :
         d->pending_downloads)
    {
        if (!pending.output
            || !copy_staging_to_cpu_batch(pending.staging, *pending.output))
        {
            d->state->failed = true;
            return false;
        }
    }
    d->completed = true;
    d->state->completed = true;
    return true;
#else
    return false;
#endif
}

std::shared_ptr<Linear> Linear::create(const TensorData& matrix, const TensorData* bias,
                                       uint32_t vulkan_device_index,
                                       const VulkanRuntimePtr& vulkan_runtime,
                                       uint64_t optimization_flags)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (matrix.shape.size() != 2 || (matrix.dtype != DType::Float32 && matrix.dtype != DType::BFloat16))
        return {};
    if (matrix.element_count() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return {};

    std::shared_ptr<Linear> linear(new Linear);
    Implementation& implementation = *linear->d;
    implementation.input_columns = matrix.shape[1];
    implementation.output_columns = matrix.shape[0];
    implementation.has_bias = bias != nullptr;
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage = false;
    implementation.option.use_fp16_arithmetic = false;
    implementation.option.use_bf16_packed = false;
    implementation.option.use_bf16_storage = false;

    implementation.vulkan_context = VulkanContext::acquire(vulkan_device_index,
                                                           vulkan_runtime,
                                                           optimization_flags);
    if (!implementation.vulkan_context)
        return {};

    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    implementation.option.use_vulkan_compute = true;
    implementation.option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.staging_vkallocator = implementation.vulkan_context->staging_allocator();
    implementation.option.use_cooperative_matrix = vkdev->info.support_cooperative_matrix();
    implementation.option.use_subgroup_ops = vkdev->info.support_subgroup_ops();
    implementation.layer = ncnn::create_layer_vulkan(ncnn::LayerType::InnerProduct);
    if (implementation.layer)
        implementation.layer->vkdev = vkdev;
    if (!implementation.layer)
        return {};

    ncnn::ParamDict parameters;
    parameters.set(0, static_cast<int>(implementation.output_columns));
    parameters.set(1, bias ? 1 : 0);
    parameters.set(2, static_cast<int>(matrix.element_count()));
    if (implementation.layer->load_param(parameters) != 0)
        return {};

    ncnn::Mat model_data[2];
    model_data[0].create(static_cast<int>(matrix.element_count()), sizeof(float));
    if (model_data[0].empty())
        return {};
    float* weight_data = static_cast<float*>(model_data[0].data);
    const std::span<const float> float32_weights = matrix.float32_values();
    const std::span<const uint16_t> bfloat16_weights = matrix.bfloat16_values();
    for (size_t index = 0; index < matrix.element_count(); ++index)
        weight_data[index] = matrix.dtype == DType::Float32 ? float32_weights[index] : bfloat16_to_float(bfloat16_weights[index]);

    if (bias)
    {
        model_data[1].create(static_cast<int>(implementation.output_columns), sizeof(float));
        if (model_data[1].empty())
            return {};
        float* bias_data = static_cast<float*>(model_data[1].data);
        const std::span<const float> float32_bias = bias->float32_values();
        const std::span<const uint16_t> bfloat16_bias = bias->bfloat16_values();
        for (uint32_t column = 0; column < implementation.output_columns; ++column)
        {
            bias_data[column] = bias->dtype == DType::Float32 ? float32_bias[column] : bfloat16_to_float(bfloat16_bias[column]);
        }
    }

    if (implementation.layer->load_model(ncnn::ModelBinFromMatArray(model_data)) != 0 || implementation.layer->create_pipeline(implementation.option) != 0)
        return {};
    implementation.pipeline_created = true;

    implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(vkdev));
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(vkdev);
        ncnn::Option upload_option = implementation.option;
        upload_option.blob_vkallocator = implementation.weight_allocator.get();
        upload_option.workspace_vkallocator = implementation.weight_allocator.get();
        upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
        if (implementation.layer->upload_model(command, upload_option) != 0 || command.submit_and_wait() != 0)
            return {};
        // ncnn's upload_model can succeed after a failed weight allocation.
        // An empty binding would otherwise read the device's dummy buffer.
        const auto& inner_product = *static_cast<const ncnn::InnerProduct_vulkan*>(implementation.layer);
        if (!inner_product_weights_available(inner_product, bias != nullptr))
            return {};
    }
    if (matvec_supported(implementation.vulkan_context, implementation.input_columns, implementation.output_columns, false)
        && !create_matvec_pipeline(implementation.vulkan_context, implementation.option, false,
                                   implementation.input_columns, implementation.output_columns,
                                   implementation.single_row_pipeline)) return {};
    return linear;
#else
    (void)matrix;
    (void)bias;
    (void)vulkan_device_index;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

std::shared_ptr<Linear> Linear::create_fused(const std::vector<const TensorData*>& matrices,
                                             const std::vector<const TensorData*>& biases,
                                             uint32_t vulkan_device_index,
                                             const VulkanRuntimePtr& vulkan_runtime,
                                             uint64_t optimization_flags)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (matrices.empty() || matrices.size() != biases.size() || !matrices.front())
        return {};

    const DType dtype = matrices.front()->dtype;
    if (matrices.front()->shape.size() != 2 || (dtype != DType::Float32 && dtype != DType::BFloat16))
        return {};
    const uint32_t input_columns = matrices.front()->shape[1];
    const bool has_bias = biases.front() != nullptr;
    uint64_t output_columns = 0;
    uint64_t element_count = 0;
    for (size_t index = 0; index < matrices.size(); ++index)
    {
        const TensorData* matrix = matrices[index];
        const TensorData* bias = biases[index];
        if (!matrix || matrix->dtype != dtype || matrix->shape.size() != 2 || matrix->shape[1] != input_columns || (bias != nullptr) != has_bias)
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
        for (size_t index = 0; index < matrices.size(); ++index)
        {
            const std::span<const float> matrix_values = matrices[index]->float32_values();
            fused_matrix.float32_data.insert(fused_matrix.float32_data.end(), matrix_values.begin(), matrix_values.end());
            if (has_bias)
            {
                const std::span<const float> bias_values = biases[index]->float32_values();
                fused_bias.float32_data.insert(fused_bias.float32_data.end(), bias_values.begin(), bias_values.end());
            }
        }
    }
    else
    {
        fused_matrix.bfloat16_data.reserve(element_count);
        if (has_bias)
            fused_bias.bfloat16_data.reserve(output_columns);
        for (size_t index = 0; index < matrices.size(); ++index)
        {
            const std::span<const uint16_t> matrix_values = matrices[index]->bfloat16_values();
            fused_matrix.bfloat16_data.insert(fused_matrix.bfloat16_data.end(), matrix_values.begin(), matrix_values.end());
            if (has_bias)
            {
                const std::span<const uint16_t> bias_values = biases[index]->bfloat16_values();
                fused_bias.bfloat16_data.insert(fused_bias.bfloat16_data.end(), bias_values.begin(), bias_values.end());
            }
        }
    }
    return create(fused_matrix,
                  has_bias ? &fused_bias : nullptr,
                  vulkan_device_index,
                  vulkan_runtime,
                  optimization_flags);
#else
    (void)matrices;
    (void)biases;
    (void)vulkan_device_index;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

bool Linear::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.layer || input.columns() != implementation.input_columns)
        return false;

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const DType output_dtype = output.dtype();
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  implementation.output_columns,
                                  transfer_slot.staging_allocator,
                                  output.element_size()))
        return false;

    output.reset(input.rows(), implementation.output_columns, false);
    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat bottom_gpu;
    if (!record_prepared_staging_upload(transfer_slot.upload, input.rows(), bottom_gpu, command, vkdev, implementation.option, input.dtype()))
        return false;

    ncnn::VkMat top_gpu;
    if (forward(bottom_gpu, top_gpu, command, implementation.option) != 0)
        return false;
    ncnn::VkMat download_gpu = top_gpu;
    if (top_gpu.elempack != 1)
    {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(top_gpu, unpacked, 1, command, implementation.option);
        download_gpu = unpacked;
    }
    if (!record_staging_download(download_gpu,
                                 input.rows(),
                                 implementation.output_columns,
                                 transfer_slot.download,
                                 command,
                                 vkdev,
                                 implementation.option,
                                 output_dtype))
        return false;
    if (submit_compute_and_wait(command, implementation.vulkan_context->device()) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
        return false;
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

int Linear::forward(const ncnn::VkMat& input,
                    ncnn::VkMat& output,
                    ncnn::VkCompute& cmd,
                    const ncnn::Option& opt) const
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!d || !d->layer)
        return -1;
#if NCNN_BATCH
    if (input.n != 1) return -1;
#endif
    if (input.empty()
        || (input.dims == 2 && input.buffer_capacity() < static_cast<uint64_t>(input.w) * input.h * input.elemsize)) return -1;
    if (d->single_row_pipeline && single_row_input(input, d->input_columns))
    {
        if (&input == &output) return -1;
        ncnn::VkMat destination(static_cast<int>(d->output_columns), 1, sizeof(float), opt.blob_vkallocator);
        if (destination.empty()) return -100;
        if (destination.buffer_capacity() < static_cast<uint64_t>(d->output_columns) * sizeof(float)) return -1;
        const auto& inner_product = *static_cast<const ncnn::InnerProduct_vulkan*>(d->layer);
        record_matvec(d->single_row_pipeline.get(), false, d->input_columns, d->output_columns, d->has_bias,
                      input, inner_product.weight_data_gpu, inner_product.bias_data_gpu, destination, cmd);
        output = destination;
        return 0;
    }
    return d->layer->forward(input, output, cmd, opt);
#else
    (void)input;
    (void)output;
    (void)cmd;
    (void)opt;
    return -1;
#endif
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& Linear::vulkan_context() const noexcept
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
#endif

} // namespace moe
} // namespace ncnn
