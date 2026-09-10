#include "linear.h"
#include "vulkancontext.h"
#include "expertbackend_vulkan.h"

#include "ncnn/moe/runtime.h"
#include "engine/expertbackend.h"
#include "engine/cpu.h"
#include "kernels/float8.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/bfloat16_cooperative_projection.comp.hex.h"
#include "kernels/vulkan/bfloat16_projection.comp.hex.h"
#include "kernels/vulkan/bfloat16_rms_norm_projection.comp.hex.h"
#include "kernels/vulkan/bfloat16_swiglu_down.comp.hex.h"
#include "kernels/vulkan/float8_projection.comp.hex.h"
#include "kernels/vulkan/float8_quantize.comp.hex.h"
#include "kernels/vulkan/float8_rms_norm_quantize.comp.hex.h"
#include "kernels/vulkan/float8_swiglu_quantize.comp.hex.h"
#include "kernels/vulkan/mxfp4_gate_up.comp.hex.h"
#include "kernels/vulkan/mxfp4_indexed.comp.hex.h"
#include "kernels/vulkan/mxfp4_projection.comp.hex.h"
#include "kernels/vulkan/mxfp4_route_aggregation.comp.hex.h"
#include "kernels/vulkan/qnk_projection.comp.hex.h"
#include "kernels/vulkan/qnk_swiglu.comp.hex.h"
#endif

#if NCNN_MOE_USE_NCNN
#include <layer.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif
#endif

namespace ncnn {
namespace moe {

#if NCNN_MOE_USE_NCNN
static constexpr uint64_t max_ncnn_linear_weight_size = 64ull * 1024ull * 1024ull;

static bool ncnn_cpu_bfloat16_linear_enabled(uint64_t optimization_flags) noexcept
{
    const uint64_t isa = cpu_isa_flags();
    // Keep ncnn's FP32-expanded path as the conservative default on
    // unbenchmarked ISAs. The direct BF16 path is default only where its
    // AVX512 implementation has been validated end to end.
    const bool default_enabled = (isa & CpuIsaX86Avx512) == 0;
    return default_enabled
           && has_flag(optimization_flags,
                       OptimizationNcnnCpuBfloat16Linear);
}
#endif

class Linear::Implementation
{
public:
#if NCNN_MOE_USE_NCNN
    ~Implementation()
    {
        if (layer)
        {
#if NCNN_MOE_WITH_VULKAN
            if (vulkan_context)
            {
                const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
                if (pipeline_created)
                    layer->destroy_pipeline(option);
            }
            else
#endif
                if (pipeline_created)
            {
                layer->destroy_pipeline(option);
            }
            delete layer;
        }
#if NCNN_MOE_WITH_VULKAN
        weight_staging_allocator.reset();
        weight_allocator.reset();
#endif
    }

    ncnn::Layer* layer = nullptr;
    ncnn::Option option;
    mutable std::mutex forward_mutex;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    bool pipeline_created = false;
    uint64_t optimization_flags = OptimizationDefaultFlags;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
#endif
#endif
};

#if NCNN_MOE_WITH_VULKAN
struct CommandGraphState_vulkan
{
};
#endif

class DeviceTensor_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::weak_ptr<CommandGraphState_vulkan> graph;
    ncnn::VkMat value;
    size_t rows = 0;
    uint32_t columns = 0;
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
    uint64_t recorded_operations = 0;
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

CommandGraph_vulkan& CommandGraph_vulkan::operator=(CommandGraph_vulkan&&) noexcept = default;

std::unique_ptr<CommandGraph_vulkan> CommandGraph_vulkan::create(const Linear& seed)
{
#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
    if (!seed.uses_vulkan())
        return {};

    auto implementation = std::make_unique<Implementation>();
    implementation->context = seed.vulkan_context();
    implementation->option = seed.option();
    implementation->state = std::make_shared<CommandGraphState_vulkan>();
    implementation->transfer_lease = std::make_unique<VulkanTransferLease>(implementation->context->acquire_transfer_slot());

    VulkanTransferSlot& transfer_slot = implementation->transfer_lease->slot();
    VulkanRuntimeState& runtime_state = implementation->context->runtime_state();
    const std::lock_guard<std::mutex> lock(implementation->context->command_mutex());
    if (transfer_slot.command_used)
    {
        if (transfer_slot.command->reset() != 0)
            return {};
        ++runtime_state.command_buffer_reuses;
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
    if (!d || d->submitted
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
                             d->transfer_lease->slot().staging_allocator,
                             d->context->runtime_state()))
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
    tensor->graph = d->state;
    tensor->value = std::move(device_input);
    tensor->rows = input.rows();
    tensor->columns = input.columns();
    output.d = std::move(tensor);
    d->upload_staging.push_back(std::move(staging));
    ++d->recorded_operations;
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
    if (!d || d->submitted
        || !input.d
        || &input == &output || input.empty())
    {
        return false;
    }
    const std::shared_ptr<CommandGraphState_vulkan> input_graph = input.d->graph.lock();
    if (!input_graph
        || input_graph.get() != d->state.get())
    {
        return false;
    }

    const std::shared_ptr<VulkanContext>& context = op.vulkan_context();
    const ncnn::Option& opt = op.option();
    if (!context
        || context.get()
               != d->context.get()
        || op.input_columns() != input.columns()
        || vulkan_activation_storage_variant(opt)
               != vulkan_activation_storage_variant(d->option))
    {
        return false;
    }

    ncnn::VkMat device_output;
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
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
    tensor->graph = d->state;
    tensor->value = std::move(device_output);
    tensor->rows = input.rows();
    tensor->columns = op.output_columns();
    output.d = std::move(tensor);
    ++d->recorded_operations;
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
    if (!d || d->submitted
        || !input.d || input.empty())
    {
        return false;
    }
    const std::shared_ptr<CommandGraphState_vulkan> input_graph = input.d->graph.lock();
    if (!input_graph
        || input_graph.get() != d->state.get())
    {
        return false;
    }

    CommandGraph_vulkan::Implementation::PendingDownload pending;
    const DType output_dtype = output.dtype();
    if (!prepare_staging_batch(pending.staging,
                               input.rows(),
                               input.columns(),
                               d->transfer_lease->slot().staging_allocator,
                               d->context->runtime_state(),
                               output.element_size()))
    {
        return false;
    }
    output.reset(input.rows(), input.columns(), false);
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (!record_prepared_activation_staging_download(input.d->value,
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
    if (!d || d->submitted
        || d->recorded_operations == 0
        || !d->context || !d->transfer_lease)
    {
        return false;
    }

    ncnn::VkCompute& command = *d->transfer_lease->slot().command;
    const std::lock_guard<std::mutex> lock(d->context->command_mutex());
    const ncnn::VkComputeCommandStatistics command_recording = command.command_statistics();
    VulkanRuntimeState& runtime_state = d->context->runtime_state();
    if (command.submit() != 0)
        return false;
    d->submitted = true;
    d->completed = false;
    runtime_state.dispatches += command_recording.dispatches;
    ++runtime_state.compute_submissions;
    runtime_state.batch_uploads += d->upload_staging.size();
    runtime_state.batch_downloads += d->pending_downloads.size();
    ++runtime_state.command_graph_submissions;
    runtime_state.command_graph_operations += d->recorded_operations;
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
    VulkanRuntimeState& runtime_state = d->context->runtime_state();
    const auto started = std::chrono::steady_clock::now();
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        if (command.wait() != 0)
            return false;
    }
    const auto submit_wait = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started);
    runtime_state.submit_wait_time_microseconds += static_cast<uint64_t>(submit_wait.count());

    for (CommandGraph_vulkan::Implementation::PendingDownload& pending :
         d->pending_downloads)
    {
        if (!pending.output
            || !copy_staging_to_cpu_batch(pending.staging, *pending.output))
        {
            return false;
        }
    }
    d->completed = true;
    return true;
#else
    return false;
#endif
}

const char* Linear::cpu_small_bfloat16_linear_policy(uint64_t optimization_flags) noexcept
{
#if NCNN_MOE_USE_NCNN
    return ncnn_cpu_bfloat16_linear_enabled(optimization_flags)
               ? "ncnn-fp32-expanded"
               : "moe-direct-bfloat16";
#else
    (void)optimization_flags;
    return "moe-direct-bfloat16";
#endif
}

std::shared_ptr<Linear> Linear::create(const TensorData& matrix, const TensorData* bias, LinearDevice device,
                                       uint32_t vulkan_device_index,
                                       const VulkanRuntimePtr& vulkan_runtime,
                                       uint64_t optimization_flags)
{
#if !NCNN_MOE_WITH_VULKAN
    (void)vulkan_device_index;
    (void)vulkan_runtime;
#endif
#if NCNN_MOE_USE_NCNN
    if (matrix.shape.size() != 2 || (matrix.dtype != DType::Float32 && matrix.dtype != DType::BFloat16))
        return {};
    if (device == LinearDevice::Cpu
        && matrix.dtype == DType::BFloat16
        && !ncnn_cpu_bfloat16_linear_enabled(optimization_flags))
    {
        return {};
    }

    const uint64_t element_size = matrix.dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
    if (device == LinearDevice::Cpu && matrix.element_count() > max_ncnn_linear_weight_size / element_size)
        return {};
    if (matrix.element_count() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return {};

    std::shared_ptr<Linear> linear(new Linear);
    Implementation& implementation = *linear->d;
    implementation.optimization_flags = optimization_flags;
    implementation.input_columns = matrix.shape[1];
    implementation.output_columns = matrix.shape[0];
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage = false;
    implementation.option.use_fp16_arithmetic = false;
    implementation.option.use_bf16_packed = false;
    implementation.option.use_bf16_storage = false;

#if NCNN_MOE_WITH_VULKAN
    if (device == LinearDevice::Vulkan)
    {
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
    }
    else
#endif
    {
        implementation.layer = ncnn::create_layer_cpu(ncnn::LayerType::InnerProduct);
    }
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

#if NCNN_MOE_WITH_VULKAN
    if (implementation.vulkan_context)
    {
        ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
        implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
        implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(vkdev));
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(vkdev);
        ncnn::Option upload_option = implementation.option;
        upload_option.blob_vkallocator = implementation.weight_allocator.get();
        upload_option.workspace_vkallocator = implementation.weight_allocator.get();
        upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
        if (implementation.layer->upload_model(command, upload_option) != 0 || command.submit_and_wait() != 0)
            return {};
    }
#endif
    return linear;
#else
    (void)matrix;
    (void)bias;
    (void)device;
    (void)vulkan_device_index;
    (void)vulkan_runtime;
    return {};
#endif
}

std::shared_ptr<Linear> Linear::create_fused(const std::vector<const TensorData*>& matrices,
                                             const std::vector<const TensorData*>& biases, LinearDevice device,
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
                  device,
                  vulkan_device_index,
                  vulkan_runtime,
                  optimization_flags);
}

bool Linear::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_USE_NCNN
    const Implementation& implementation = *d;
    if (!implementation.layer || input.columns() != implementation.input_columns)
        return false;

#if NCNN_MOE_WITH_VULKAN
    if (implementation.vulkan_context)
    {
        VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
        VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
        VulkanTransferSlot& transfer_slot = transfer_lease.slot();
        const DType output_dtype = output.dtype();
        if (!fill_staging_upload(input,
                                 transfer_slot.upload,
                                 transfer_slot.staging_allocator,
                                 runtime_state)
            || !prepare_staging_batch(transfer_slot.download,
                                      input.rows(),
                                      implementation.output_columns,
                                      transfer_slot.staging_allocator,
                                      runtime_state,
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
            ++runtime_state.command_buffer_reuses;
        }
        transfer_slot.command_used = true;
        ncnn::VkMat bottom_gpu;
        if (!record_prepared_staging_upload(transfer_slot.upload, input.rows(), bottom_gpu, command, vkdev, implementation.option, input.dtype()))
            return false;

        ncnn::VkMat top_gpu;
        if (implementation.layer->forward(bottom_gpu, top_gpu, command, implementation.option) != 0)
            return false;
        ncnn::VkMat download_gpu = top_gpu;
        if (top_gpu.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(top_gpu, unpacked, 1, command, implementation.option);
            download_gpu = unpacked;
        }
        if (!record_prepared_activation_staging_download(download_gpu,
                                                         input.rows(),
                                                         implementation.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         vkdev,
                                                         implementation.option,
                                                         output_dtype))
            return false;
        if (submit_compute_and_wait(command, runtime_state) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
            return false;
        ++runtime_state.dispatches;
        ++runtime_state.compute_submissions;
        ++runtime_state.batch_uploads;
        ++runtime_state.batch_downloads;
        return true;
    }
#endif

    const std::lock_guard<std::mutex> lock(implementation.forward_mutex);
    ncnn::Mat bottom(static_cast<int>(input.columns()), static_cast<int>(input.rows()), sizeof(float));
    if (bottom.empty())
        return false;
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
        std::copy_n(input.row(row_index), input.columns(), bottom.row<float>(static_cast<int>(row_index)));

    ncnn::Mat top;
    if (implementation.layer->forward(bottom, top, implementation.option) != 0 || top.empty()
        || top.total() * top.elempack != input.rows() * implementation.output_columns)
        return false;
    output.reset(input.rows(), implementation.output_columns, false);
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        const float* source = top.row<float>(static_cast<int>(row_index));
        std::copy_n(source, implementation.output_columns, output.row(row_index));
    }
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

bool Linear::uses_vulkan() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->vulkan_context != nullptr;
#else
    return false;
#endif
}

class Bfloat16Linear_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> cooperative_pipeline;
    mutable std::shared_ptr<ncnn::Pipeline> swiglu_down_pipeline;
    std::shared_ptr<ncnn::Pipeline> rms_norm_projection_pipeline;
    ncnn::VkMat packed;
    ncnn::VkMat bias;
    ncnn::VkMat rms_norm_weight;
    ncnn::Option option;
    float rms_norm_epsilon = 0.0f;
    uint32_t rms_norm_output_subgroups = 0;
    uint32_t cooperative_tile_m = 0;
    uint32_t cooperative_tile_n = 0;
    uint32_t cooperative_tile_k = 0;
    uint32_t cooperative_subgroup_size = 0;
    uint64_t optimization_flags = OptimizationDefaultFlags;

    [[nodiscard]] bool record_projection(const ncnn::VkMat& input,
                                         ncnn::VkMat& output,
                                         uint32_t token_count,
                                         ncnn::VkCompute& command) const;
#endif
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
};

#if NCNN_MOE_WITH_VULKAN

static bool create_bfloat16_projection_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                const ncnn::Option& option,
                                                std::shared_ptr<ncnn::Pipeline>& destination)
{
    const size_t shader_variant = (option.use_subgroup_ops ? 1u : 0u) * 3
                                  + vulkan_activation_storage_variant(option);
    ncnn::Option compile_option = option;
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(bfloat16_projection_shader,
                                                                                      static_cast<int>(sizeof(bfloat16_projection_shader) - 1),
                                                                                      compile_option,
                                                                                      shader_variant);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(bfloat16_projection_shader,
                                         shader_variant);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context](ncnn::Pipeline* value) {
                                                      const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                      delete value;
                                                  });
    context->cache_pipeline(bfloat16_projection_shader,
                            shader_variant,
                            destination);
    return true;
}

static bool create_bfloat16_cooperative_projection_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                            const ncnn::Option& option,
                                                            uint32_t input_columns,
                                                            uint32_t output_columns,
                                                            std::shared_ptr<ncnn::Pipeline>& destination,
                                                            uint32_t& tile_m,
                                                            uint32_t& tile_n,
                                                            uint32_t& tile_k,
                                                            uint32_t& subgroup_size)
{
    ncnn::VulkanDevice* device = context->device();
    // The cooperative shader uses the BF16 storage macro family for its
    // matrix tiles. Activation storage must use the scalar/subgroup path until
    // activation and weight storage types are independently parameterized.
    if (vulkan_activation_storage_variant(option) != 0)
        return false;
    if (!device->info.support_VK_KHR_cooperative_matrix()
        || !device->info.support_bf16_cooperative_matrix()
        || !device->info.support_bf16_storage())
    {
        return false;
    }

    int selected_m = 0;
    int selected_n = 0;
    int selected_k = 0;
    int selected_subgroup_size = 0;
    device->info.get_optimal_cooperative_matrix_mnk(16,
                                                    static_cast<int>(output_columns),
                                                    static_cast<int>(input_columns),
                                                    VK_COMPONENT_TYPE_BFLOAT16_KHR,
                                                    VK_COMPONENT_TYPE_FLOAT32_KHR,
                                                    VK_SCOPE_SUBGROUP_KHR,
                                                    selected_m,
                                                    selected_n,
                                                    selected_k,
                                                    selected_subgroup_size);
    if (selected_m <= 0
        || selected_n <= 0
        || selected_k <= 0
        || selected_subgroup_size <= 0
        || selected_m % 4 != 0
        || selected_n % 4 != 0
        || selected_k % 4 != 0)
    {
        return false;
    }
    const uint64_t shared_size = (static_cast<uint64_t>(selected_m) * selected_k
                                  + static_cast<uint64_t>(selected_k) * selected_n)
                                     * sizeof(uint16_t)
                                 + static_cast<uint64_t>(selected_m) * selected_n
                                       * sizeof(float);
    if (shared_size > device->info.max_shared_memory_size()
        || selected_m > std::numeric_limits<uint16_t>::max()
        || selected_n > std::numeric_limits<uint16_t>::max()
        || selected_k > std::numeric_limits<uint16_t>::max()
        || selected_subgroup_size > std::numeric_limits<uint16_t>::max())
    {
        return false;
    }

    ncnn::Option compile_option = option;
    compile_option.vulkan_device_index = device->info.device_index();
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_storage = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    compile_option.use_bf16_storage = true;
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(bfloat16_cooperative_projection_shader,
                                                                                      static_cast<int>(sizeof(bfloat16_cooperative_projection_shader) - 1),
                                                                                      compile_option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    const uint64_t pipeline_key = static_cast<uint64_t>(selected_m)
                                  | static_cast<uint64_t>(selected_n) << 16
                                  | static_cast<uint64_t>(selected_k) << 32
                                  | static_cast<uint64_t>(selected_subgroup_size) << 48;
    destination = context->find_pipeline(bfloat16_cooperative_projection_shader,
                                         pipeline_key);
    if (destination)
    {
        tile_m = static_cast<uint32_t>(selected_m);
        tile_n = static_cast<uint32_t>(selected_n);
        tile_k = static_cast<uint32_t>(selected_k);
        subgroup_size = static_cast<uint32_t>(selected_subgroup_size);
        return true;
    }

    std::vector<ncnn::vk_specialization_type> specializations(3);
    specializations[0].u32 = static_cast<uint32_t>(selected_m);
    specializations[1].u32 = static_cast<uint32_t>(selected_n);
    specializations[2].u32 = static_cast<uint32_t>(selected_k);
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(selected_subgroup_size);
    pipeline->set_local_size_xyz(selected_subgroup_size, 1, 1);
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context](ncnn::Pipeline* value) {
                                                      const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                      delete value;
                                                  });
    context->cache_pipeline(bfloat16_cooperative_projection_shader,
                            pipeline_key,
                            destination);
    tile_m = static_cast<uint32_t>(selected_m);
    tile_n = static_cast<uint32_t>(selected_n);
    tile_k = static_cast<uint32_t>(selected_k);
    subgroup_size = static_cast<uint32_t>(selected_subgroup_size);
    return true;
}

static bool create_bfloat16_rms_norm_projection_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                         const ncnn::Option& option,
                                                         std::shared_ptr<ncnn::Pipeline>& destination,
                                                         uint32_t& output_subgroups)
{
    const size_t shader_variant = (option.use_subgroup_ops ? 1u : 0u) * 3
                                  + vulkan_activation_storage_variant(option);
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(bfloat16_rms_norm_projection_shader,
                                                                                      static_cast<int>(sizeof(bfloat16_rms_norm_projection_shader) - 1),
                                                                                      option,
                                                                                      shader_variant);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    const uint32_t local_size = device->info.max_workgroup_invocations() >= 256
                                        && device->info.max_workgroup_size_x() >= 256
                                    ? 256
                                    : 128;
    output_subgroups = local_size / 32;
    destination = context->find_pipeline(bfloat16_rms_norm_projection_shader,
                                         shader_variant);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(static_cast<int>(local_size), 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context](ncnn::Pipeline* value) {
                                                      const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                      delete value;
                                                  });
    context->cache_pipeline(bfloat16_rms_norm_projection_shader,
                            shader_variant,
                            destination);
    return true;
}

static bool create_bfloat16_swiglu_down_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                 const ncnn::Option& option,
                                                 std::shared_ptr<ncnn::Pipeline>& destination)
{
    const size_t shader_variant = (option.use_subgroup_ops ? 1u : 0u) * 3
                                  + vulkan_activation_storage_variant(option);
    ncnn::Option compile_option = option;
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(bfloat16_swiglu_down_shader,
                                                                                      static_cast<int>(sizeof(bfloat16_swiglu_down_shader) - 1),
                                                                                      compile_option,
                                                                                      shader_variant);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(bfloat16_swiglu_down_shader,
                                         shader_variant);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context](ncnn::Pipeline* value) {
                                                      const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                      delete value;
                                                  });
    context->cache_pipeline(bfloat16_swiglu_down_shader,
                            shader_variant,
                            destination);
    return true;
}

static bool prepare_bfloat16_upload(std::span<const uint16_t> source,
                                    ncnn::Mat& destination)
{
    if (source.empty()
        || source.size() > static_cast<size_t>(std::numeric_limits<int>::max()) * 2)
    {
        return false;
    }
    const size_t word_count = (source.size() + 1) / 2;
    if (word_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    destination.create(static_cast<int>(word_count), sizeof(uint32_t));
    if (destination.empty())
        return false;
    uint32_t* words = static_cast<uint32_t*>(destination.data);
    for (size_t index = 0; index < word_count; ++index)
    {
        const size_t first = index * 2;
        const uint32_t low = source[first];
        const uint32_t high = first + 1 < source.size() ? source[first + 1] : 0;
        words[index] = low | high << 16;
    }
    return true;
}
#endif

Bfloat16Linear_vulkan::Bfloat16Linear_vulkan()
    : d(new Implementation)
{
}

Bfloat16Linear_vulkan::~Bfloat16Linear_vulkan() = default;

bool Bfloat16Linear_vulkan::prepare_rms_norm(const TensorData& weight,
                                             float epsilon,
                                             float weight_offset)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || implementation.rms_norm_projection_pipeline
        || epsilon <= 0.0f
        || weight.shape.size() != 1 || weight.shape[0] != implementation.input_columns
        || (weight.dtype != DType::Float32
            && weight.dtype != DType::BFloat16))
    {
        return false;
    }

    ncnn::Mat values;
    values.create(static_cast<int>(implementation.input_columns), sizeof(float));
    if (values.empty())
        return false;
    float* destination = static_cast<float*>(values.data);
    if (weight.dtype == DType::Float32)
    {
        const std::span<const float> source = weight.float32_values();
        if (source.size() != implementation.input_columns)
            return false;
        for (uint32_t index = 0; index < implementation.input_columns; ++index)
            destination[index] = source[index] + weight_offset;
    }
    else
    {
        const std::span<const uint16_t> source = weight.bfloat16_values();
        if (source.size() != implementation.input_columns)
            return false;
        for (uint32_t index = 0; index < implementation.input_columns; ++index)
            destination[index] = bfloat16_to_float(source[index]) + weight_offset;
    }

    if (!create_bfloat16_rms_norm_projection_pipeline(implementation.vulkan_context,
                                                      implementation.option,
                                                      implementation.rms_norm_projection_pipeline,
                                                      implementation.rms_norm_output_subgroups))
        return false;

    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(implementation.vulkan_context->device()));
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(implementation.vulkan_context->device());
        command.record_upload(values,
                              implementation.rms_norm_weight,
                              upload_option);
        uploaded = !implementation.rms_norm_weight.empty()
                   && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
    {
        implementation.rms_norm_projection_pipeline.reset();
        return false;
    }
    implementation.rms_norm_epsilon = epsilon;
    return true;
#else
    (void)weight;
    (void)epsilon;
    (void)weight_offset;
    return false;
#endif
}

bool Bfloat16Linear_vulkan::has_rms_norm_chain() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->rms_norm_projection_pipeline != nullptr
           && !d->rms_norm_weight.empty();
#else
    return false;
#endif
}

bool Bfloat16Linear_vulkan::forward_rms_norm_chain(const ActivationBuffer& input,
                                                   ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || !implementation.rms_norm_projection_pipeline
        || implementation.rms_norm_weight.empty()
        || implementation.rms_norm_output_subgroups == 0
        || input.rows() == 0
        || input.columns() != implementation.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = vulkan_activation_storage_variant(implementation.option) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                    * implementation.input_columns * sizeof(float),
                                                                                                input.dtype());
    const bool direct_host_output = vulkan_activation_storage_variant(implementation.option) == 0
                                    && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                     * implementation.output_columns * sizeof(float),
                                                                                                 output.dtype());
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator,
                             runtime_state)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  implementation.output_columns,
                                  transfer_slot.staging_allocator,
                                  runtime_state))
    {
        return false;
    }
    output.reset(input.rows(), implementation.output_columns, false);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
        {
            return false;
        }
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_activation_upload(transfer_slot.upload,
                                              input_gpu,
                                              command,
                                              implementation.vulkan_context->device(),
                                              implementation.option,
                                              input.dtype()))
    {
        return false;
    }

    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(implementation.output_columns),
                          static_cast<int>(input.rows()),
                          vulkan_activation_element_size(implementation.option),
                          implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;
    const std::vector<ncnn::VkMat> bindings = {
        input_gpu,
        implementation.packed,
        implementation.bias,
        implementation.rms_norm_weight,
        output_gpu};
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.rows());
    constants[4].f = implementation.rms_norm_epsilon;
    constants[5].u32 = implementation.rms_norm_output_subgroups;
    const uint64_t dispatch_width = ((static_cast<uint64_t>(implementation.output_columns)
                                      + implementation.rms_norm_output_subgroups - 1)
                                     / implementation.rms_norm_output_subgroups)
                                    * implementation.rms_norm_output_subgroups * 32;
    if (dispatch_width > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(dispatch_width);
    dispatcher.h = static_cast<int>(input.rows());
    dispatcher.c = 1;
    command.record_pipeline(implementation.rms_norm_projection_pipeline.get(),
                            bindings,
                            constants,
                            dispatcher);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         implementation.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         implementation.vulkan_context->device(),
                                                         implementation.option,
                                                         output.dtype()))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    ++runtime_state.dispatches;
    ++runtime_state.rms_norm_linear_fusions;
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

std::shared_ptr<Bfloat16Linear_vulkan>
Bfloat16Linear_vulkan::create(const TensorData& matrix,
                              const TensorData* bias,
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

std::shared_ptr<Bfloat16Linear_vulkan>
Bfloat16Linear_vulkan::create_with_allocator(const TensorData& matrix,
                                             const TensorData* bias,
                                             uint32_t vulkan_device_index,
                                             ncnn::VkAllocator* weight_allocator,
                                             const VulkanRuntimePtr& vulkan_runtime,
                                             uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (matrix.dtype != DType::BFloat16
        || matrix.shape.size() != 2
        || matrix.shape[0] == 0
        || matrix.shape[1] == 0
        || matrix.shape[1] % 4 != 0
        || matrix.shape[0]
               > static_cast<uint32_t>(std::numeric_limits<int>::max() / 32)
        || matrix.shape[1]
               > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const std::span<const uint16_t> weights = matrix.bfloat16_values();
    if (weights.size() != matrix.element_count())
        return {};
    if (bias
        && (bias->shape.size() != 1
            || bias->shape[0] != output_columns
            || (bias->dtype != DType::Float32
                && bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    std::shared_ptr<Bfloat16Linear_vulkan> result(new Bfloat16Linear_vulkan);
    Implementation& implementation = *result->d;
    implementation.input_columns = input_columns;
    implementation.output_columns = output_columns;
    implementation.block_count = (input_columns + 127) / 128;
    implementation.optimization_flags = optimization_flags;
    implementation.vulkan_context = VulkanContext::acquire(vulkan_device_index,
                                                           vulkan_runtime,
                                                           optimization_flags);
    if (!implementation.vulkan_context)
        return {};
    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    if (device->info.subgroup_size() != 32)
        return {};
    implementation.option.vulkan_device_index = device->info.device_index();
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

    const uint64_t weight_size = static_cast<uint64_t>(weights.size()) * sizeof(uint16_t);
    const uint64_t preferred_weight_size = weight_size
                                           + static_cast<uint64_t>(output_columns) * sizeof(float)
                                           + static_cast<uint64_t>(input_columns) * sizeof(float);
    if (preferred_weight_size
        > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        return {};
    }
    if (!weight_allocator)
    {
        implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(device,
                                                                          static_cast<size_t>(preferred_weight_size)));
        weight_allocator = implementation.weight_allocator.get();
    }
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));

    ncnn::Mat packed;
    ncnn::Mat biases;
    if (!prepare_bfloat16_upload(weights, packed))
        return {};
    biases.create(static_cast<int>(output_columns), sizeof(float));
    if (biases.empty())
        return {};
    float* bias_values = static_cast<float*>(biases.data);
    if (!bias)
    {
        std::fill_n(bias_values, output_columns, 0.0f);
    }
    else if (bias->dtype == DType::Float32)
    {
        const std::span<const float> values = bias->float32_values();
        if (values.size() != output_columns)
            return {};
        std::copy(values.begin(), values.end(), bias_values);
    }
    else
    {
        const std::span<const uint16_t> values = bias->bfloat16_values();
        if (values.size() != output_columns)
            return {};
        for (uint32_t index = 0; index < output_columns; ++index)
            bias_values[index] = bfloat16_to_float(values[index]);
    }

    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_bfloat16_projection_pipeline(implementation.vulkan_context,
                                                 implementation.option,
                                                 implementation.pipeline))
        {
            return {};
        }
        if (has_flag(implementation.optimization_flags,
                     OptimizationVulkanBfloat16CoopMatrix))
        {
            create_bfloat16_cooperative_projection_pipeline(implementation.vulkan_context,
                                                            implementation.option,
                                                            input_columns,
                                                            output_columns,
                                                            implementation.cooperative_pipeline,
                                                            implementation.cooperative_tile_m,
                                                            implementation.cooperative_tile_n,
                                                            implementation.cooperative_tile_k,
                                                            implementation.cooperative_subgroup_size);
        }
    }
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = weight_allocator;
    upload_option.workspace_vkallocator = weight_allocator;
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    ncnn::Option packed_upload_option = upload_option;
    packed_upload_option.use_fp16_storage = false;
    packed_upload_option.use_bf16_storage = false;
    bool uploaded = false;
    {
        ncnn::VkTransfer command(device);
        command.record_upload(packed,
                              implementation.packed,
                              packed_upload_option);
        command.record_upload(biases,
                              implementation.bias,
                              upload_option);
        uploaded = !implementation.packed.empty()
                   && !implementation.bias.empty()
                   && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return {};
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

std::shared_ptr<Bfloat16Linear_vulkan>
Bfloat16Linear_vulkan::create_fused(const std::vector<const TensorData*>& matrices,
                                    const std::vector<const TensorData*>& biases,
                                    uint32_t vulkan_device_index,
                                    const VulkanRuntimePtr& vulkan_runtime,
                                    uint64_t optimization_flags)
{
    if (matrices.empty()
        || matrices.size() != biases.size()
        || !matrices.front()
        || matrices.front()->dtype != DType::BFloat16
        || matrices.front()->shape.size() != 2)
    {
        return {};
    }
    const uint32_t input_columns = matrices.front()->shape[1];
    const bool has_bias = biases.front() != nullptr;
    uint64_t output_columns = 0;
    uint64_t element_count = 0;
    for (size_t index = 0; index < matrices.size(); ++index)
    {
        const TensorData* matrix = matrices[index];
        const TensorData* bias = biases[index];
        if (!matrix
            || matrix->dtype != DType::BFloat16
            || matrix->shape.size() != 2
            || matrix->shape[1] != input_columns
            || (bias != nullptr) != has_bias
            || (bias
                && (bias->shape.size() != 1
                    || bias->shape[0] != matrix->shape[0]
                    || (bias->dtype != DType::Float32
                        && bias->dtype != DType::BFloat16))))
        {
            return {};
        }
        output_columns += matrix->shape[0];
        element_count += matrix->element_count();
    }
    if (output_columns > std::numeric_limits<uint32_t>::max()
        || element_count > std::numeric_limits<size_t>::max())
    {
        return {};
    }

    TensorData fused_matrix;
    fused_matrix.dtype = DType::BFloat16;
    fused_matrix.shape = {
        static_cast<uint32_t>(output_columns),
        input_columns};
    fused_matrix.bfloat16_data.reserve(static_cast<size_t>(element_count));
    TensorData fused_bias;
    fused_bias.dtype = DType::Float32;
    fused_bias.shape = {static_cast<uint32_t>(output_columns)};
    if (has_bias)
        fused_bias.float32_data.reserve(static_cast<size_t>(output_columns));
    for (size_t index = 0; index < matrices.size(); ++index)
    {
        const std::span<const uint16_t> matrix_values = matrices[index]->bfloat16_values();
        if (matrix_values.size() != matrices[index]->element_count())
            return {};
        fused_matrix.bfloat16_data.insert(fused_matrix.bfloat16_data.end(),
                                          matrix_values.begin(),
                                          matrix_values.end());
        if (!has_bias)
            continue;
        const TensorData& bias = *biases[index];
        if (bias.dtype == DType::Float32)
        {
            const std::span<const float> bias_values = bias.float32_values();
            fused_bias.float32_data.insert(fused_bias.float32_data.end(),
                                           bias_values.begin(),
                                           bias_values.end());
        }
        else
        {
            const std::span<const uint16_t> bias_values = bias.bfloat16_values();
            for (uint16_t value : bias_values)
                fused_bias.float32_data.push_back(bfloat16_to_float(value));
        }
    }
    return create(fused_matrix,
                  has_bias ? &fused_bias : nullptr,
                  vulkan_device_index,
                  vulkan_runtime,
                  optimization_flags);
}

#if NCNN_MOE_WITH_VULKAN
bool Bfloat16Linear_vulkan::Implementation::record_projection(const ncnn::VkMat& input,
                                                              ncnn::VkMat& output,
                                                              uint32_t token_count,
                                                              ncnn::VkCompute& command) const
{
    std::vector<ncnn::VkMat> bindings = {
        input,
        packed,
        bias,
        output};
    const uint64_t cooperative_output_work = static_cast<uint64_t>(token_count) * output_columns;
    if (cooperative_pipeline
        && token_count >= cooperative_tile_m
        && input_columns >= cooperative_tile_k
        && output_columns >= cooperative_tile_n
        && cooperative_output_work >= UINT64_C(65536))
    {
        const uint64_t token_tile_count = (token_count + cooperative_tile_m - 1)
                                          / cooperative_tile_m;
        const uint64_t output_tile_count = (output_columns + cooperative_tile_n - 1)
                                           / cooperative_tile_n;
        const uint64_t invocation_count = token_tile_count * output_tile_count
                                          * cooperative_subgroup_size;
        if (invocation_count
            <= static_cast<uint64_t>(std::numeric_limits<int>::max()))
        {
            std::vector<ncnn::vk_constant_type> constants(4);
            constants[0].u32 = input_columns;
            constants[1].u32 = output_columns;
            constants[2].u32 = token_count;
            constants[3].u32 = static_cast<uint32_t>(token_tile_count);
            ncnn::VkMat dispatcher;
            dispatcher.w = static_cast<int>(invocation_count);
            dispatcher.h = 1;
            dispatcher.c = 1;
            command.record_pipeline_readonly(cooperative_pipeline.get(),
                                             bindings,
                                             {1, 1, 1, 0},
                                             constants,
                                             dispatcher);
            return true;
        }
    }

    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = input_columns;
    constants[1].u32 = output_columns;
    constants[2].u32 = block_count;
    constants[3].u32 = token_count;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(output_columns * 32);
    dispatcher.h = static_cast<int>(token_count);
    dispatcher.c = 1;
    command.record_pipeline_readonly(pipeline.get(),
                                     bindings,
                                     {1, 1, 1, 0},
                                     constants,
                                     dispatcher);
    return false;
}
#endif

void Bfloat16Linear_vulkan::record_scalar_projection(const ncnn::VkMat& input,
                                                     ncnn::VkMat& output,
                                                     ncnn::VkCompute& cmd) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    const std::vector<ncnn::VkMat> bindings = {
        input,
        implementation.packed,
        implementation.bias,
        output};
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.h);
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(implementation.output_columns * 32);
    dispatcher.h = input.h;
    dispatcher.c = 1;
    cmd.record_pipeline_readonly(implementation.pipeline.get(),
                                 bindings,
                                 {1, 1, 1, 0},
                                 constants,
                                 dispatcher);
#else
    (void)input;
    (void)output;
    (void)cmd;
#endif
}

void Bfloat16Linear_vulkan::record_rms_norm_projection(const ncnn::VkMat& input,
                                                       ncnn::VkMat& output,
                                                       ncnn::VkCompute& cmd) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    const std::vector<ncnn::VkMat> bindings = {
        input,
        implementation.packed,
        implementation.bias,
        implementation.rms_norm_weight,
        output};
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.h);
    constants[4].f = implementation.rms_norm_epsilon;
    constants[5].u32 = implementation.rms_norm_output_subgroups;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(((static_cast<uint64_t>(implementation.output_columns)
                                      + implementation.rms_norm_output_subgroups - 1)
                                     / implementation.rms_norm_output_subgroups)
                                    * implementation.rms_norm_output_subgroups * 32);
    dispatcher.h = input.h;
    dispatcher.c = 1;
    cmd.record_pipeline_readonly(implementation.rms_norm_projection_pipeline.get(),
                                 bindings,
                                 {1, 1, 1, 1, 0},
                                 constants,
                                 dispatcher);
#else
    (void)input;
    (void)output;
    (void)cmd;
#endif
}

int Bfloat16Linear_vulkan::forward(const ncnn::VkMat& input,
                                   ncnn::VkMat& output,
                                   ncnn::VkCompute& cmd,
                                   const ncnn::Option& opt) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || input.empty()
        || input.dims != 2
        || input.w != static_cast<int>(implementation.input_columns)
        || input.h <= 0
        || input.elempack != 1
        || input.elemsize != sizeof(float))
    {
        return -1;
    }

    output.create(static_cast<int>(implementation.output_columns),
                  input.h,
                  sizeof(float),
                  opt.blob_vkallocator);
    if (output.empty())
        return -100;

    const std::vector<ncnn::VkMat> bindings = {
        input,
        implementation.packed,
        implementation.bias,
        output};
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.h);
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(implementation.output_columns * 32);
    dispatcher.h = input.h;
    dispatcher.c = 1;
    cmd.record_pipeline(implementation.pipeline.get(),
                        bindings,
                        constants,
                        dispatcher);
    return 0;
#else
    (void)input;
    (void)output;
    (void)cmd;
    (void)opt;
    return -1;
#endif
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& Bfloat16Linear_vulkan::vulkan_context() const noexcept
{
    return d->vulkan_context;
}

const ncnn::Option& Bfloat16Linear_vulkan::option() const noexcept
{
    return d->option;
}

uint32_t Bfloat16Linear_vulkan::input_columns() const noexcept
{
    return d->input_columns;
}

uint32_t Bfloat16Linear_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}
#endif

bool Bfloat16Linear_vulkan::forward(const ActivationBuffer& input,
                                    ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || input.rows() == 0
        || input.columns() != implementation.input_columns
        || input.rows()
               > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = vulkan_activation_storage_variant(implementation.option) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                    * implementation.input_columns * sizeof(float),
                                                                                                input.dtype());
    const bool direct_host_output = vulkan_activation_storage_variant(implementation.option) == 0
                                    && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                     * implementation.output_columns * sizeof(float),
                                                                                                 output.dtype());
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator,
                             runtime_state)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  implementation.output_columns,
                                  transfer_slot.staging_allocator,
                                  runtime_state))
    {
        return false;
    }
    output.reset(input.rows(), implementation.output_columns, false);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
        {
            return false;
        }
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_activation_upload(transfer_slot.upload,
                                              input_gpu,
                                              command,
                                              implementation.vulkan_context->device(),
                                              implementation.option,
                                              input.dtype()))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(implementation.output_columns),
                          static_cast<int>(input.rows()),
                          vulkan_activation_element_size(implementation.option),
                          implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    const bool used_cooperative_matrix = implementation.record_projection(input_gpu,
                                                                          output_gpu,
                                                                          static_cast<uint32_t>(input.rows()),
                                                                          command);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         implementation.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         implementation.vulkan_context->device(),
                                                         implementation.option,
                                                         output.dtype()))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download,
                                      output))
    {
        return false;
    }
    ++runtime_state.dispatches;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    if (used_cooperative_matrix)
    {
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    }
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool Bfloat16Linear_vulkan::forward_parallel(const ActivationBuffer& input,
                                             const Bfloat16Linear_vulkan& parallel_operator,
                                             ActivationBuffer& output,
                                             ActivationBuffer& parallel_output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *d;
    const Implementation& parallel = *parallel_operator.d;
    if (!first.vulkan_context
        || !parallel.vulkan_context
        || first.vulkan_context.get() != parallel.vulkan_context.get()
        || !first.pipeline
        || !parallel.pipeline
        || vulkan_activation_storage_variant(first.option)
               != vulkan_activation_storage_variant(parallel.option)
        || input.rows() == 0
        || input.columns() != first.input_columns
        || input.columns() != parallel.input_columns
        || input.rows()
               > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    ncnn::VkMat parallel_download;
    const bool direct_host_input = vulkan_activation_storage_variant(first.option) == 0
                                   && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * first.input_columns
                                                                                           * sizeof(float),
                                                                                       input.dtype());
    const bool direct_host_output = vulkan_activation_storage_variant(first.option) == 0
                                    && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * first.output_columns
                                                                                            * sizeof(float),
                                                                                        output.dtype());
    const bool parallel_direct_host_output = vulkan_activation_storage_variant(parallel.option) == 0
                                             && parallel.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * parallel.output_columns
                                                                                                        * sizeof(float),
                                                                                                    parallel_output.dtype());
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator,
                             runtime_state)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  first.output_columns,
                                  transfer_slot.staging_allocator,
                                  runtime_state)
        || !prepare_staging_batch(parallel_download,
                                  input.rows(),
                                  parallel.output_columns,
                                  transfer_slot.staging_allocator,
                                  runtime_state))
    {
        return false;
    }
    output.reset(input.rows(), first.output_columns, false);
    parallel_output.reset(input.rows(), parallel.output_columns, false);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_activation_upload(transfer_slot.upload,
                                              input_gpu,
                                              command,
                                              first.vulkan_context->device(),
                                              first.option,
                                              input.dtype()))
    {
        return false;
    }

    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(first.output_columns),
                          static_cast<int>(input.rows()),
                          vulkan_activation_element_size(first.option),
                          first.vulkan_context->blob_allocator());
    ncnn::VkMat parallel_output_gpu;
    if (parallel_direct_host_output)
        parallel_output_gpu = prepare_direct_host_output(parallel_download, runtime_state);
    else
        parallel_output_gpu.create(static_cast<int>(parallel.output_columns),
                                   static_cast<int>(input.rows()),
                                   vulkan_activation_element_size(parallel.option),
                                   parallel.vulkan_context->blob_allocator());
    if (output_gpu.empty() || parallel_output_gpu.empty())
        return false;

    const bool first_used_cooperative_matrix = first.record_projection(input_gpu,
                                                                       output_gpu,
                                                                       static_cast<uint32_t>(input.rows()),
                                                                       command);
    const bool parallel_used_cooperative_matrix = parallel.record_projection(input_gpu,
                                                                             parallel_output_gpu,
                                                                             static_cast<uint32_t>(input.rows()),
                                                                             command);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         first.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         first.vulkan_context->device(),
                                                         first.option,
                                                         output.dtype()))
        || (!parallel_direct_host_output
            && !record_prepared_activation_staging_download(parallel_output_gpu,
                                                            input.rows(),
                                                            parallel.output_columns,
                                                            parallel_download,
                                                            command,
                                                            parallel.vulkan_context->device(),
                                                            parallel.option,
                                                            parallel_output.dtype()))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output)
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }
    runtime_state.dispatches += 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads += 2;
    if (first_used_cooperative_matrix)
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    if (parallel_used_cooperative_matrix)
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    return true;
#else
    (void)input;
    (void)parallel_operator;
    (void)output;
    (void)parallel_output;
    return false;
#endif
}

bool Bfloat16Linear_vulkan::forward_swiglu_chain(const ActivationBuffer& input,
                                                 const Bfloat16Linear_vulkan& down_operator,
                                                 uint32_t intermediate_columns,
                                                 ExpertActivation activation,
                                                 float activation_limit,
                                                 bool apply_router_gate,
                                                 ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *d;
    const Implementation& down = *down_operator.d;
    const uint64_t expected_fused_columns = static_cast<uint64_t>(intermediate_columns) * 2
                                            + (apply_router_gate ? 1u : 0u);
    (void)activation_limit;
    if (activation != ExpertActivation::Silu
        || !first.vulkan_context
        || !down.vulkan_context
        || first.vulkan_context.get() != down.vulkan_context.get()
        || vulkan_activation_storage_variant(first.option)
               != vulkan_activation_storage_variant(down.option)
        || !first.pipeline
        || first.output_columns != expected_fused_columns
        || down.input_columns != intermediate_columns
        || intermediate_columns == 0
        || intermediate_columns % 128 != 0
        || input.rows() == 0
        || input.columns() != first.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
        || down.packed.empty()
        || down.bias.empty())
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = vulkan_activation_storage_variant(first.option) == 0
                                   && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * first.input_columns
                                                                                           * sizeof(float),
                                                                                       input.dtype());
    const bool direct_host_output = vulkan_activation_storage_variant(first.option) == 0
                                    && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * down.output_columns
                                                                                            * sizeof(float),
                                                                                        output.dtype());
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator,
                             runtime_state)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  down.output_columns,
                                  transfer_slot.staging_allocator,
                                  runtime_state))
    {
        return false;
    }
    output.reset(input.rows(), down.output_columns, false);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    if (!first.swiglu_down_pipeline
        && !create_bfloat16_swiglu_down_pipeline(first.vulkan_context,
                                                 first.option,
                                                 first.swiglu_down_pipeline))
    {
        return false;
    }
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_activation_upload(transfer_slot.upload,
                                              input_gpu,
                                              command,
                                              first.vulkan_context->device(),
                                              first.option,
                                              input.dtype()))
    {
        return false;
    }
    ncnn::VkMat fused_gpu;
    fused_gpu.create(static_cast<int>(first.output_columns),
                     static_cast<int>(input.rows()),
                     vulkan_activation_element_size(first.option),
                     first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(down.output_columns),
                          static_cast<int>(input.rows()),
                          vulkan_activation_element_size(first.option),
                          first.vulkan_context->blob_allocator());
    if (fused_gpu.empty() || output_gpu.empty())
        return false;

    const bool used_cooperative_matrix = first.record_projection(input_gpu,
                                                                 fused_gpu,
                                                                 static_cast<uint32_t>(input.rows()),
                                                                 command);

    std::vector<ncnn::VkMat> swiglu_bindings = {
        fused_gpu,
        down.packed,
        down.bias,
        output_gpu};
    std::vector<ncnn::vk_constant_type> swiglu_constants(5);
    swiglu_constants[0].u32 = intermediate_columns;
    swiglu_constants[1].u32 = first.output_columns;
    swiglu_constants[2].u32 = down.output_columns;
    swiglu_constants[3].u32 = static_cast<uint32_t>(input.rows());
    swiglu_constants[4].u32 = apply_router_gate ? 1u : 0u;
    ncnn::VkMat swiglu_dispatcher;
    swiglu_dispatcher.w = static_cast<int>(down.output_columns * 32);
    swiglu_dispatcher.h = static_cast<int>(input.rows());
    swiglu_dispatcher.c = 1;
    command.record_pipeline(first.swiglu_down_pipeline.get(),
                            swiglu_bindings,
                            swiglu_constants,
                            swiglu_dispatcher);

    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         down.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         first.vulkan_context->device(),
                                                         first.option,
                                                         output.dtype()))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 2;
    ++runtime_state.shared_expert_swiglu_fusions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    if (used_cooperative_matrix)
    {
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    }
    return true;
#else
    (void)input;
    (void)down_operator;
    (void)intermediate_columns;
    (void)activation;
    (void)activation_limit;
    (void)apply_router_gate;
    (void)output;
    return false;
#endif
}

class Bfloat16Expert_vulkan::Implementation
{
public:
    std::shared_ptr<Bfloat16Linear_vulkan> gate_up;
    std::shared_ptr<Bfloat16Linear_vulkan> down;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
#endif
    uint32_t intermediate_columns = 0;
    uint32_t output_columns = 0;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::Silu;
};

Bfloat16Expert_vulkan::Bfloat16Expert_vulkan()
    : d(new Implementation)
{
}

Bfloat16Expert_vulkan::~Bfloat16Expert_vulkan() = default;

uint32_t Bfloat16Expert_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

std::shared_ptr<Bfloat16Expert_vulkan>
Bfloat16Expert_vulkan::create(const TensorData& gate_up,
                              const TensorData* gate_up_bias,
                              const TensorData& down,
                              const TensorData* down_bias,
                              float activation_limit,
                              uint32_t vulkan_device_index,
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

std::shared_ptr<Bfloat16Expert_vulkan>
Bfloat16Expert_vulkan::create_with_allocator(const TensorData& gate_up,
                                             const TensorData* gate_up_bias,
                                             const TensorData& down,
                                             const TensorData* down_bias,
                                             float activation_limit,
                                             uint32_t vulkan_device_index,
                                             ncnn::VkAllocator* weight_allocator,
                                             ExpertActivation activation,
                                             const VulkanRuntimePtr& vulkan_runtime,
                                             uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (activation != ExpertActivation::Silu
        || gate_up.dtype != DType::BFloat16
        || down.dtype != DType::BFloat16
        || gate_up.shape.size() != 2
        || down.shape.size() != 2
        || gate_up.shape[0] == 0
        || gate_up.shape[0] % 2 != 0
        || down.shape[0] == 0
        || down.shape[1] != gate_up.shape[0] / 2
        || (gate_up.shape[0] / 2) % 128 != 0
        || activation_limit < 0.0f)
    {
        return {};
    }

    std::shared_ptr<Bfloat16Expert_vulkan> result(new Bfloat16Expert_vulkan);
    Implementation& implementation = *result->d;
    implementation.gate_up = Bfloat16Linear_vulkan::create_with_allocator(gate_up,
                                                                          gate_up_bias,
                                                                          vulkan_device_index,
                                                                          weight_allocator,
                                                                          vulkan_runtime,
                                                                          optimization_flags);
    implementation.down = Bfloat16Linear_vulkan::create_with_allocator(down,
                                                                       down_bias,
                                                                       vulkan_device_index,
                                                                       weight_allocator,
                                                                       vulkan_runtime,
                                                                       optimization_flags);
    if (!implementation.gate_up || !implementation.down)
        return {};

    const Bfloat16Linear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const Bfloat16Linear_vulkan::Implementation& down_projection = *implementation.down->d;
    if (!gate.vulkan_context
        || gate.vulkan_context != down_projection.vulkan_context
        || gate.output_columns != gate_up.shape[0]
        || down_projection.input_columns != gate_up.shape[0] / 2
        || down_projection.output_columns != down.shape[0])
    {
        return {};
    }

    implementation.vulkan_context = gate.vulkan_context;
    implementation.intermediate_columns = gate.output_columns / 2;
    implementation.output_columns = down_projection.output_columns;
    implementation.activation_limit = activation_limit;
    implementation.activation = activation;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_bfloat16_swiglu_down_pipeline(implementation.vulkan_context,
                                                  gate.option,
                                                  gate.swiglu_down_pipeline))
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
    return {};
#endif
}

bool Bfloat16Expert_vulkan::forward(const ActivationBuffer& input,
                                    ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    return implementation.gate_up
           && implementation.down
           && implementation.gate_up->forward_swiglu_chain(input,
                                                           *implementation.down,
                                                           implementation.intermediate_columns,
                                                           implementation.activation,
                                                           implementation.activation_limit,
                                                           false,
                                                           output);
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool Bfloat16Expert_vulkan::forward_batch(std::span<const Bfloat16Expert_vulkan*> experts,
                                          std::span<const ActivationBuffer*> inputs,
                                          std::span<ActivationBuffer*> outputs)
{
#if NCNN_MOE_WITH_VULKAN
    if (experts.empty()
        || experts.size() != inputs.size()
        || experts.size() != outputs.size())
    {
        return false;
    }

    const Bfloat16Expert_vulkan* first_operator = experts.front();
    if (!first_operator || !first_operator->d)
        return false;
    const Implementation& first_expert = *first_operator->d;
    if (!first_expert.gate_up
        || !first_expert.down
        || !first_expert.vulkan_context)
    {
        return false;
    }
    const Bfloat16Linear_vulkan::Implementation& first_gate = *first_expert.gate_up->d;
    const Bfloat16Linear_vulkan::Implementation& first_down = *first_expert.down->d;
    if (!first_gate.pipeline
        || !first_gate.swiglu_down_pipeline
        || first_down.packed.empty()
        || first_down.bias.empty()
        || first_gate.output_columns % 2 != 0
        || first_down.input_columns != first_expert.intermediate_columns
        || first_down.output_columns != first_expert.output_columns)
    {
        return false;
    }

    const uint32_t input_columns = first_gate.input_columns;
    const uint32_t output_columns = first_down.output_columns;
    if (input_columns == 0
        || output_columns == 0
        || first_expert.intermediate_columns == 0
        || first_expert.intermediate_columns % 128 != 0)
    {
        return false;
    }

    size_t total_rows = 0;
    for (size_t index = 0; index < experts.size(); ++index)
    {
        const Bfloat16Expert_vulkan* operator_instance = experts[index];
        const ActivationBuffer* input = inputs[index];
        const ActivationBuffer* output = outputs[index];
        if (!operator_instance
            || !operator_instance->d
            || !input
            || !output
            || input->rows() == 0
            || input->rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
            || input->rows() > static_cast<size_t>(std::numeric_limits<int>::max())
            || input->dtype() != DType::Float32
            || input->columns() != input_columns
            || output->dtype() != DType::Float32)
        {
            return false;
        }
        const Implementation& expert = *operator_instance->d;
        if (!expert.gate_up
            || !expert.down
            || !expert.vulkan_context
            || expert.vulkan_context != first_expert.vulkan_context
            || expert.intermediate_columns != first_expert.intermediate_columns
            || expert.output_columns != output_columns
            || expert.activation != first_expert.activation
            || expert.activation_limit != first_expert.activation_limit)
        {
            return false;
        }
        const Bfloat16Linear_vulkan::Implementation& gate = *expert.gate_up->d;
        const Bfloat16Linear_vulkan::Implementation& down = *expert.down->d;
        if (!gate.pipeline
            || !gate.swiglu_down_pipeline
            || down.packed.empty()
            || down.bias.empty()
            || gate.vulkan_context != first_expert.vulkan_context
            || down.vulkan_context != first_expert.vulkan_context
            || gate.input_columns != input_columns
            || gate.output_columns != first_gate.output_columns
            || down.input_columns != first_down.input_columns
            || down.output_columns != output_columns
            || vulkan_activation_storage_variant(gate.option)
                   != vulkan_activation_storage_variant(first_gate.option))
        {
            return false;
        }
        if (total_rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
                             - input->rows()
            || total_rows > static_cast<size_t>(std::numeric_limits<int>::max())
                                - input->rows())
        {
            return false;
        }
        total_rows += input->rows();
    }
    if (total_rows == 0
        || total_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = first_expert.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = first_expert.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!prepare_staging_batch(transfer_slot.upload,
                               total_rows,
                               input_columns,
                               transfer_slot.staging_allocator,
                               runtime_state,
                               sizeof(float))
        || !prepare_staging_batch(transfer_slot.download,
                                  total_rows,
                                  output_columns,
                                  transfer_slot.staging_allocator,
                                  runtime_state,
                                  sizeof(float)))
    {
        return false;
    }

    ncnn::Mat mapped_input = transfer_slot.upload.mapped();
    if (mapped_input.empty()
        || mapped_input.dims != 2
        || mapped_input.w != static_cast<int>(input_columns)
        || mapped_input.h != static_cast<int>(total_rows)
        || mapped_input.elemsize != sizeof(float)
        || mapped_input.elempack != 1)
    {
        return false;
    }
    auto* mapped_input_data = static_cast<std::byte*>(mapped_input.data);
    size_t row_offset = 0;
    for (const ActivationBuffer* input : inputs)
    {
        const size_t input_size = input->rows() * static_cast<size_t>(input_columns) * sizeof(float);
        if (input->bytes().size() != input_size)
            return false;
        std::memcpy(mapped_input_data
                        + row_offset * static_cast<size_t>(input_columns) * sizeof(float),
                    input->bytes().data(),
                    input_size);
        row_offset += input->rows();
    }
    transfer_slot.upload.allocator->flush(transfer_slot.upload.data);
    transfer_slot.upload.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    transfer_slot.upload.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    std::unique_lock<std::mutex> lock(first_expert.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (!record_mapped_activation_upload(transfer_slot.upload,
                                         input_gpu,
                                         command,
                                         first_expert.vulkan_context->device(),
                                         first_gate.option,
                                         DType::Float32))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
    output_gpu.create(static_cast<int>(output_columns),
                      static_cast<int>(total_rows),
                      vulkan_activation_element_size(first_gate.option),
                      first_expert.vulkan_context->blob_allocator());
    if (input_gpu.empty() || output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> intermediates;
    intermediates.reserve(experts.size());
    std::vector<ncnn::VkMat> swiglu_bindings(4);
    std::vector<ncnn::vk_constant_type> swiglu_constants(5);
    row_offset = 0;
    for (size_t index = 0; index < experts.size(); ++index)
    {
        const Implementation& expert = *experts[index]->d;
        const Bfloat16Linear_vulkan::Implementation& gate = *expert.gate_up->d;
        const Bfloat16Linear_vulkan::Implementation& down = *expert.down->d;
        const size_t rows = inputs[index]->rows();
        ncnn::VkMat input_view = row_view(input_gpu, row_offset, rows);
        ncnn::VkMat output_view = row_view(output_gpu, row_offset, rows);
        if (input_view.empty() || output_view.empty())
            return false;

        ncnn::VkMat& fused_gpu = intermediates.emplace_back();
        fused_gpu.create(static_cast<int>(gate.output_columns),
                         static_cast<int>(rows),
                         vulkan_activation_element_size(first_gate.option),
                         first_expert.vulkan_context->blob_allocator());
        if (fused_gpu.empty())
            return false;

        const bool used_cooperative_matrix = gate.record_projection(input_view,
                                                                    fused_gpu,
                                                                    static_cast<uint32_t>(rows),
                                                                    command);
        if (used_cooperative_matrix)
            ++runtime_state.bfloat16_cooperative_matrix_dispatches;

        swiglu_bindings[0] = fused_gpu;
        swiglu_bindings[1] = down.packed;
        swiglu_bindings[2] = down.bias;
        swiglu_bindings[3] = output_view;
        swiglu_constants[0].u32 = expert.intermediate_columns;
        swiglu_constants[1].u32 = gate.output_columns;
        swiglu_constants[2].u32 = down.output_columns;
        swiglu_constants[3].u32 = static_cast<uint32_t>(rows);
        swiglu_constants[4].u32 = 0;
        ncnn::VkMat swiglu_dispatcher;
        swiglu_dispatcher.w = static_cast<int>(down.output_columns * 32);
        swiglu_dispatcher.h = static_cast<int>(rows);
        swiglu_dispatcher.c = 1;
        command.record_pipeline(gate.swiglu_down_pipeline.get(),
                                swiglu_bindings,
                                swiglu_constants,
                                swiglu_dispatcher);
        row_offset += rows;
    }

    if (!record_prepared_activation_staging_download(output_gpu,
                                                     total_rows,
                                                     output_columns,
                                                     transfer_slot.download,
                                                     command,
                                                     first_expert.vulkan_context->device(),
                                                     first_down.option,
                                                     DType::Float32)
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batches(transfer_slot.download,
                                        inputs,
                                        outputs,
                                        output_columns))
    {
        return false;
    }
    lock.unlock();
    runtime_state.dispatches += static_cast<uint64_t>(experts.size()) * 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)experts;
    (void)inputs;
    (void)outputs;
    return false;
#endif
}

class Float8Linear_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> quantize_pipeline;
    std::shared_ptr<ncnn::Pipeline> swiglu_quantize_pipeline;
    std::shared_ptr<ncnn::Pipeline> rms_norm_quantize_pipeline;
    ncnn::VkMat packed;
    ncnn::VkMat scales;
    ncnn::VkMat bias;
    ncnn::VkMat rms_norm_weight;
    ncnn::VkMat input_rms_norm_weight;
    ncnn::Option option;
    uint64_t optimization_flags = OptimizationDefaultFlags;

    void record_projection(const ncnn::VkMat& input,
                           const ncnn::VkMat& output,
                           uint32_t token_count,
                           ncnn::VkCompute& command) const;
#endif
    uint32_t matrix_input_columns = 0;
    uint32_t logical_input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t output_columns_per_group = 0;
    uint32_t block_count = 0;
    uint32_t input_group_count = 1;
    float rms_norm_epsilon = 0.0f;
    float input_rms_norm_epsilon = 0.0f;
};

#if NCNN_MOE_WITH_VULKAN

void Float8Linear_vulkan::Implementation::record_projection(const ncnn::VkMat& input,
                                                            const ncnn::VkMat& output,
                                                            uint32_t token_count,
                                                            ncnn::VkCompute& command) const
{
    std::vector<ncnn::VkMat> bindings = {input, packed, scales, bias, output};
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = matrix_input_columns;
    constants[1].u32 = logical_input_columns;
    constants[2].u32 = output_columns;
    constants[3].u32 = output_columns_per_group;
    constants[4].u32 = block_count;
    constants[5].u32 = token_count;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(output_columns * 32);
    dispatcher.h = static_cast<int>(token_count);
    dispatcher.c = 1;
    command.record_pipeline_readonly(pipeline.get(),
                                     bindings,
                                     {1, 1, 1, 1, 0},
                                     constants,
                                     dispatcher);
}

static bool create_float8_projection_pipeline(const std::shared_ptr<VulkanContext>& context,
                                              const ncnn::Option& option,
                                              std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(float8_projection_shader,
                                                                                      static_cast<int>(sizeof(float8_projection_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(float8_projection_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(float8_projection_shader, 0, destination);
    return true;
}

static bool create_float8_quantize_pipeline(const std::shared_ptr<VulkanContext>& context,
                                            const ncnn::Option& option,
                                            std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(float8_quantize_shader,
                                                                                      static_cast<int>(sizeof(float8_quantize_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(float8_quantize_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(float8_quantize_shader, 0, destination);
    return true;
}

static bool create_float8_rms_norm_quantize_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                     const ncnn::Option& option,
                                                     std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(float8_rms_norm_quantize_shader,
                                                                                      static_cast<int>(sizeof(float8_rms_norm_quantize_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(float8_rms_norm_quantize_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(float8_rms_norm_quantize_shader, 0, destination);
    return true;
}

static bool create_float8_swiglu_quantize_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                   const ncnn::Option& option,
                                                   std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(float8_swiglu_quantize_shader,
                                                                                      static_cast<int>(sizeof(float8_swiglu_quantize_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(float8_swiglu_quantize_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(float8_swiglu_quantize_shader, 0, destination);
    return true;
}

static size_t align_float8_upload(size_t bytes) noexcept
{
    return (bytes + 3) & ~static_cast<size_t>(3);
}

static bool prepare_float8_upload(std::span<const uint8_t> source, ncnn::Mat& destination)
{
    const size_t padded_size = align_float8_upload(source.size());
    if (padded_size == 0 || padded_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    destination.create(static_cast<int>(padded_size), sizeof(uint8_t));
    if (destination.empty())
        return false;
    std::memset(destination.data, 0, padded_size);
    std::memcpy(destination.data, source.data(), source.size());
    return true;
}

#endif

Float8Linear_vulkan::Float8Linear_vulkan()
    : d(new Implementation)
{
}

Float8Linear_vulkan::~Float8Linear_vulkan() = default;

std::shared_ptr<Float8Linear_vulkan> Float8Linear_vulkan::create(const TensorData& matrix,
                                                                 const TensorData* bias,
                                                                 uint32_t input_group_count,
                                                                 uint32_t vulkan_device_index,
                                                                 const VulkanRuntimePtr& vulkan_runtime,
                                                                 uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (matrix.dtype != DType::Float8E4M3 || matrix.shape.size() != 2 || matrix.shape[0] == 0 || matrix.shape[1] == 0
        || matrix.shape[1] % 128 != 0 || input_group_count == 0 || matrix.shape[0] % input_group_count != 0
        || matrix.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 32)
        || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t block_count = input_columns / 128;
    const uint32_t output_block_count = (output_columns + 127) / 128;
    const std::span<const uint8_t> weights = matrix.float8_values();
    if (weights.size() != matrix.element_count()
        || matrix.quantization_scales.size() != static_cast<size_t>(output_block_count) * block_count)
    {
        return {};
    }
    if (bias && (bias->shape.size() != 1 || bias->shape[0] != output_columns || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
    {
        return {};
    }
    const uint64_t logical_input_columns = static_cast<uint64_t>(input_columns) * input_group_count;
    if (logical_input_columns > std::numeric_limits<uint32_t>::max())
        return {};

    std::shared_ptr<Float8Linear_vulkan> result(new Float8Linear_vulkan);
    Implementation& implementation = *result->d;
    implementation.matrix_input_columns = input_columns;
    implementation.logical_input_columns = static_cast<uint32_t>(logical_input_columns);
    implementation.output_columns = output_columns;
    implementation.output_columns_per_group = output_columns / input_group_count;
    implementation.block_count = block_count;
    implementation.input_group_count = input_group_count;
    implementation.optimization_flags = optimization_flags;
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

    const uint64_t preferred_weight_size = align_float8_upload(weights.size())
                                           + static_cast<uint64_t>(matrix.quantization_scales.size()) * sizeof(float)
                                           + static_cast<uint64_t>(output_columns) * sizeof(float);
    if (preferred_weight_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return {};
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));

    ncnn::Mat packed;
    ncnn::Mat scales;
    ncnn::Mat biases;
    if (!prepare_float8_upload(weights, packed))
        return {};
    scales.create(static_cast<int>(matrix.quantization_scales.size()), sizeof(float));
    biases.create(static_cast<int>(output_columns), sizeof(float));
    if (scales.empty() || biases.empty())
        return {};
    std::copy(matrix.quantization_scales.begin(), matrix.quantization_scales.end(), static_cast<float*>(scales.data));
    float* bias_values = static_cast<float*>(biases.data);
    if (!bias)
    {
        std::fill_n(bias_values, output_columns, 0.0f);
    }
    else if (bias->dtype == DType::Float32)
    {
        const std::span<const float> values = bias->float32_values();
        std::copy(values.begin(), values.end(), bias_values);
    }
    else
    {
        const std::span<const uint16_t> values = bias->bfloat16_values();
        for (uint32_t index = 0; index < output_columns; ++index)
            bias_values[index] = bfloat16_to_float(values[index]);
    }

    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_float8_projection_pipeline(implementation.vulkan_context, implementation.option, implementation.pipeline)
            || !create_float8_quantize_pipeline(implementation.vulkan_context, implementation.option, implementation.quantize_pipeline)
            || !create_float8_swiglu_quantize_pipeline(implementation.vulkan_context,
                                                       implementation.option,
                                                       implementation.swiglu_quantize_pipeline))
            return {};
    }
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
    upload_option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        ncnn::VkTransfer command(device);
        command.record_upload(packed, implementation.packed, upload_option);
        command.record_upload(scales, implementation.scales, upload_option);
        command.record_upload(biases, implementation.bias, upload_option);
        uploaded = !implementation.packed.empty() && !implementation.scales.empty() && !implementation.bias.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return {};
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)input_group_count;
    (void)vulkan_device_index;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

bool Float8Linear_vulkan::prepare_rms_norm_weight(const TensorData& weight,
                                                  uint32_t expected_columns,
                                                  float epsilon,
                                                  ncnn::VkMat& destination)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    if (!implementation.vulkan_context || epsilon <= 0.0f
        || expected_columns == 0
        || weight.shape.size() != 1 || weight.shape[0] != expected_columns
        || (weight.dtype != DType::Float32 && weight.dtype != DType::BFloat16))
    {
        return false;
    }
    ncnn::Mat values;
    values.create(static_cast<int>(expected_columns), sizeof(float));
    if (values.empty())
        return false;
    float* values_destination = static_cast<float*>(values.data);
    if (weight.dtype == DType::Float32)
    {
        const std::span<const float> source = weight.float32_values();
        if (source.size() != expected_columns)
            return false;
        std::copy(source.begin(), source.end(), values_destination);
    }
    else
    {
        const std::span<const uint16_t> source = weight.bfloat16_values();
        if (source.size() != expected_columns)
            return false;
        for (uint32_t index = 0; index < expected_columns; ++index)
            values_destination[index] = bfloat16_to_float(source[index]);
    }
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_float8_rms_norm_quantize_pipeline(implementation.vulkan_context,
                                                      implementation.option,
                                                      implementation.rms_norm_quantize_pipeline))
        {
            return false;
        }
    }
    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
    upload_option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        ncnn::VkTransfer command(device);
        command.record_upload(values, destination, upload_option);
        uploaded = !destination.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    return uploaded;
#else
    (void)weight;
    (void)expected_columns;
    (void)epsilon;
    (void)destination;
    return false;
#endif
}

bool Float8Linear_vulkan::prepare_rms_norm(const TensorData& weight,
                                           float epsilon)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    if (!prepare_rms_norm_weight(weight,
                                 implementation.output_columns,
                                 epsilon,
                                 implementation.rms_norm_weight))
        return false;
    implementation.rms_norm_epsilon = epsilon;
    return true;
#else
    (void)weight;
    (void)epsilon;
    return false;
#endif
}

bool Float8Linear_vulkan::prepare_input_rms_norm(const TensorData& weight,
                                                 float epsilon)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    if (!prepare_rms_norm_weight(weight,
                                 implementation.logical_input_columns,
                                 epsilon,
                                 implementation.input_rms_norm_weight))
        return false;
    implementation.input_rms_norm_epsilon = epsilon;
    return true;
#else
    (void)weight;
    (void)epsilon;
    return false;
#endif
}

#if NCNN_MOE_WITH_VULKAN
static bool fill_float8_quantized_staging(const ActivationBuffer& input,
                                          uint32_t columns,
                                          uint64_t optimization_flags,
                                          ncnn::VkMat& staging,
                                          ncnn::VkAllocator* allocator,
                                          VulkanRuntimeState& runtime_state)
{
    if (input.dtype() != DType::Float32
        || input.rows() == 0
        || input.columns() != columns
        || columns == 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    const size_t row_bytes = static_cast<size_t>(columns) * sizeof(float);
    if (input.rows() > std::numeric_limits<size_t>::max() / row_bytes
        || input.bytes().size() != input.rows() * row_bytes)
    {
        return false;
    }
    if (!prepare_staging_batch(staging,
                               input.rows(),
                               columns,
                               allocator,
                               runtime_state,
                               sizeof(float)))
    {
        return false;
    }

    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty()
        || mapped.dims != 2
        || mapped.w != static_cast<int>(columns)
        || mapped.h != static_cast<int>(input.rows())
        || mapped.elempack != 1
        || mapped.elemsize != sizeof(float))
    {
        return false;
    }
    for (size_t row_index = 0; row_index < input.rows(); ++row_index)
    {
        float* destination = mapped.row<float>(static_cast<int>(row_index));
        quantize_float8_e4m3(input.row(row_index),
                             destination,
                             columns,
                             128,
                             true,
                             optimization_flags);
    }
    staging.allocator->flush(staging.data);
    staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}
#endif

bool Float8Linear_vulkan::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context || !implementation.pipeline || input.rows() == 0
        || input.columns() != implementation.logical_input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                 * implementation.logical_input_columns * sizeof(float),
                                                                                             input.dtype());
    const bool direct_host_output = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * implementation.output_columns
                                                                                                  * sizeof(float),
                                                                                              output.dtype());
    if (!fill_float8_quantized_staging(input,
                                       implementation.logical_input_columns,
                                       implementation.optimization_flags,
                                       transfer_slot.upload,
                                       transfer_slot.staging_allocator,
                                       runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator, runtime_state))
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
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
        return false;
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(implementation.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    implementation.record_projection(input_gpu,
                                     output_gpu,
                                     static_cast<uint32_t>(input.rows()),
                                     command);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         implementation.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         implementation.vulkan_context->device(),
                                                         implementation.option,
                                                         output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
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

bool Float8Linear_vulkan::forward_chain(const ActivationBuffer& input, const Float8Linear_vulkan& next, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *d;
    const Implementation& second = *next.d;
    if (!first.vulkan_context || !second.vulkan_context || first.vulkan_context.get() != second.vulkan_context.get()
        || !first.pipeline || !second.pipeline || !first.quantize_pipeline || input.rows() == 0
        || input.columns() != first.logical_input_columns
        || first.output_columns != second.logical_input_columns
        || first.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * second.output_columns
                                                                                         * sizeof(float),
                                                                                     output.dtype());
    if (!fill_float8_quantized_staging(input,
                                       first.logical_input_columns,
                                       first.optimization_flags,
                                       transfer_slot.upload,
                                       transfer_slot.staging_allocator,
                                       runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output.reset(input.rows(), second.output_columns, false);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(first.output_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(second.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          first.vulkan_context->blob_allocator());
    if (intermediate_gpu.empty() || output_gpu.empty())
        return false;

    first.record_projection(input_gpu,
                            intermediate_gpu,
                            static_cast<uint32_t>(input.rows()),
                            command);

    std::vector<ncnn::VkMat> quantize_bindings = {intermediate_gpu};
    std::vector<ncnn::vk_constant_type> quantize_constants(2);
    quantize_constants[0].u32 = first.output_columns;
    quantize_constants[1].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat quantize_dispatcher;
    quantize_dispatcher.w = static_cast<int>((first.output_columns / 128) * 32);
    quantize_dispatcher.h = static_cast<int>(input.rows());
    quantize_dispatcher.c = 1;
    command.record_pipeline(first.quantize_pipeline.get(), quantize_bindings, quantize_constants, quantize_dispatcher);

    second.record_projection(intermediate_gpu,
                             output_gpu,
                             static_cast<uint32_t>(input.rows()),
                             command);

    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         second.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         first.vulkan_context->device(),
                                                         first.option,
                                                         output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
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
    (void)next;
    (void)output;
    return false;
#endif
}

bool Float8Linear_vulkan::forward_rms_norm_chain(const ActivationBuffer& input,
                                                 const Float8Linear_vulkan& next,
                                                 ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *d;
    const Implementation& second = *next.d;
    if (!first.vulkan_context || !second.vulkan_context || first.vulkan_context.get() != second.vulkan_context.get()
        || !first.pipeline || !second.pipeline || !first.rms_norm_quantize_pipeline || first.rms_norm_weight.empty()
        || first.rms_norm_epsilon <= 0.0f || input.rows() == 0
        || input.columns() != first.logical_input_columns
        || first.output_columns != second.logical_input_columns
        || first.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * second.output_columns
                                                                                         * sizeof(float),
                                                                                     output.dtype());
    if (!fill_float8_quantized_staging(input,
                                       first.logical_input_columns,
                                       first.optimization_flags,
                                       transfer_slot.upload,
                                       transfer_slot.staging_allocator,
                                       runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output.reset(input.rows(), second.output_columns, false);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(first.output_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(second.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          first.vulkan_context->blob_allocator());
    if (intermediate_gpu.empty() || output_gpu.empty())
        return false;

    first.record_projection(input_gpu,
                            intermediate_gpu,
                            static_cast<uint32_t>(input.rows()),
                            command);

    std::vector<ncnn::VkMat> norm_bindings = {intermediate_gpu, first.rms_norm_weight};
    std::vector<ncnn::vk_constant_type> norm_constants(3);
    norm_constants[0].u32 = first.output_columns;
    norm_constants[1].u32 = static_cast<uint32_t>(input.rows());
    norm_constants[2].f = first.rms_norm_epsilon;
    ncnn::VkMat norm_dispatcher;
    norm_dispatcher.w = 32;
    norm_dispatcher.h = static_cast<int>(input.rows());
    norm_dispatcher.c = 1;
    command.record_pipeline_readonly(first.rms_norm_quantize_pipeline.get(),
                                     norm_bindings,
                                     {0, 1},
                                     norm_constants,
                                     norm_dispatcher);

    second.record_projection(intermediate_gpu,
                             output_gpu,
                             static_cast<uint32_t>(input.rows()),
                             command);

    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         second.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         first.vulkan_context->device(),
                                                         first.option,
                                                         output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
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
    (void)next;
    (void)output;
    return false;
#endif
}

bool Float8Linear_vulkan::forward_rms_norm_chain_parallel_impl(const ActivationBuffer& input,
                                                               const Float8Linear_vulkan& next,
                                                               const Float8Linear_vulkan& parallel_operator,
                                                               ActivationBuffer& output,
                                                               ActivationBuffer& parallel_output,
                                                               bool normalize_input) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *d;
    const Implementation& second = *next.d;
    const Implementation& parallel = *parallel_operator.d;
    if (normalize_input
        && !has_flag(first.optimization_flags,
                     OptimizationVulkanLatentInputRmsNorm))
    {
        return false;
    }
    if (!first.vulkan_context || !second.vulkan_context || !parallel.vulkan_context
        || first.vulkan_context.get() != second.vulkan_context.get()
        || first.vulkan_context.get() != parallel.vulkan_context.get()
        || !first.pipeline || !second.pipeline || !parallel.pipeline
        || !first.rms_norm_quantize_pipeline || first.rms_norm_weight.empty()
        || (normalize_input && first.input_rms_norm_weight.empty())
        || first.rms_norm_epsilon <= 0.0f || input.rows() == 0
        || (normalize_input && first.input_rms_norm_epsilon <= 0.0f)
        || input.columns() != first.logical_input_columns
        || input.columns() != parallel.logical_input_columns
        || first.output_columns != second.logical_input_columns
        || first.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * second.output_columns
                                                                                         * sizeof(float),
                                                                                     output.dtype());
    const bool parallel_direct_host_output = parallel.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * parallel.output_columns
                                                                                                     * sizeof(float),
                                                                                                 parallel_output.dtype());
    ncnn::VkMat parallel_download;
    if (!(normalize_input
              ? fill_staging_upload(input,
                                    transfer_slot.upload,
                                    transfer_slot.staging_allocator,
                                    runtime_state)
              : fill_float8_quantized_staging(input,
                                              first.logical_input_columns,
                                              first.optimization_flags,
                                              transfer_slot.upload,
                                              transfer_slot.staging_allocator,
                                              runtime_state))
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(parallel_download, input.rows(), parallel.output_columns, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    const DType parallel_output_dtype = parallel_output.dtype();
    output.reset(input.rows(), second.output_columns, false);
    parallel_output.reset(input.rows(), parallel.output_columns, false);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;
    if (normalize_input)
    {
        std::vector<ncnn::VkMat> input_norm_bindings = {
            input_gpu,
            first.input_rms_norm_weight};
        std::vector<ncnn::vk_constant_type> input_norm_constants(3);
        input_norm_constants[0].u32 = first.logical_input_columns;
        input_norm_constants[1].u32 = static_cast<uint32_t>(input.rows());
        input_norm_constants[2].f = first.input_rms_norm_epsilon;
        ncnn::VkMat input_norm_dispatcher;
        input_norm_dispatcher.w = 32;
        input_norm_dispatcher.h = static_cast<int>(input.rows());
        input_norm_dispatcher.c = 1;
        command.record_pipeline_readonly(first.rms_norm_quantize_pipeline.get(),
                                         input_norm_bindings,
                                         {0, 1},
                                         input_norm_constants,
                                         input_norm_dispatcher);
    }
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(first.output_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(second.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          first.vulkan_context->blob_allocator());
    ncnn::VkMat parallel_gpu;
    if (parallel_direct_host_output)
        parallel_gpu = prepare_direct_host_output(parallel_download, runtime_state);
    else
        parallel_gpu.create(static_cast<int>(parallel.output_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            first.vulkan_context->blob_allocator());
    if (intermediate_gpu.empty() || output_gpu.empty() || parallel_gpu.empty())
        return false;

    first.record_projection(input_gpu,
                            intermediate_gpu,
                            static_cast<uint32_t>(input.rows()),
                            command);

    std::vector<ncnn::VkMat> norm_bindings = {intermediate_gpu, first.rms_norm_weight};
    std::vector<ncnn::vk_constant_type> norm_constants(3);
    norm_constants[0].u32 = first.output_columns;
    norm_constants[1].u32 = static_cast<uint32_t>(input.rows());
    norm_constants[2].f = first.rms_norm_epsilon;
    ncnn::VkMat norm_dispatcher;
    norm_dispatcher.w = 32;
    norm_dispatcher.h = static_cast<int>(input.rows());
    norm_dispatcher.c = 1;
    command.record_pipeline_readonly(first.rms_norm_quantize_pipeline.get(),
                                     norm_bindings,
                                     {0, 1},
                                     norm_constants,
                                     norm_dispatcher);

    second.record_projection(intermediate_gpu,
                             output_gpu,
                             static_cast<uint32_t>(input.rows()),
                             command);
    parallel.record_projection(input_gpu,
                               parallel_gpu,
                               static_cast<uint32_t>(input.rows()),
                               command);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         second.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         first.vulkan_context->device(),
                                                         first.option,
                                                         output_dtype))
        || (!parallel_direct_host_output
            && !record_prepared_activation_staging_download(parallel_gpu,
                                                            input.rows(),
                                                            parallel.output_columns,
                                                            parallel_download,
                                                            command,
                                                            first.vulkan_context->device(),
                                                            first.option,
                                                            parallel_output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output)
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }
    runtime_state.dispatches += 3;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads += 2;
    if (normalize_input)
        ++runtime_state.rms_norm_linear_fusions;
    return true;
#else
    (void)input;
    (void)next;
    (void)parallel_operator;
    (void)output;
    (void)parallel_output;
    (void)normalize_input;
    return false;
#endif
}

bool Float8Linear_vulkan::forward_rms_norm_chain_parallel(const ActivationBuffer& input,
                                                          const Float8Linear_vulkan& next,
                                                          const Float8Linear_vulkan& parallel_operator,
                                                          ActivationBuffer& output,
                                                          ActivationBuffer& parallel_output) const
{
    return forward_rms_norm_chain_parallel_impl(input,
                                                next,
                                                parallel_operator,
                                                output,
                                                parallel_output,
                                                false);
}

bool Float8Linear_vulkan::forward_input_rms_norm_chain_parallel(const ActivationBuffer& input,
                                                                const Float8Linear_vulkan& next,
                                                                const Float8Linear_vulkan& parallel_operator,
                                                                ActivationBuffer& output,
                                                                ActivationBuffer& parallel_output) const
{
    return forward_rms_norm_chain_parallel_impl(input,
                                                next,
                                                parallel_operator,
                                                output,
                                                parallel_output,
                                                true);
}

bool Float8Linear_vulkan::forward_rms_norm_chain_parallel_bfloat16(const ActivationBuffer& input,
                                                                   const Float8Linear_vulkan& next,
                                                                   const Float8Linear_vulkan& parallel_operator,
                                                                   std::span<const Bfloat16Linear_vulkan*> extra_operators,
                                                                   std::span<ActivationBuffer*> extra_outputs,
                                                                   ActivationBuffer& output,
                                                                   ActivationBuffer& parallel_output) const
{
#if NCNN_MOE_WITH_VULKAN
    if (extra_operators.empty())
        return forward_rms_norm_chain_parallel(input,
                                               next,
                                               parallel_operator,
                                               output,
                                               parallel_output);
    if (extra_operators.size() != extra_outputs.size())
        return false;

    const Implementation& first = *d;
    const Implementation& second = *next.d;
    const Implementation& parallel = *parallel_operator.d;
    if (!first.vulkan_context || !second.vulkan_context || !parallel.vulkan_context
        || first.vulkan_context.get() != second.vulkan_context.get()
        || first.vulkan_context.get() != parallel.vulkan_context.get()
        || !first.pipeline || !second.pipeline || !parallel.pipeline
        || !first.quantize_pipeline
        || !first.rms_norm_quantize_pipeline || first.rms_norm_weight.empty()
        || first.rms_norm_epsilon <= 0.0f || input.rows() == 0
        || input.columns() != first.logical_input_columns
        || input.columns() != parallel.logical_input_columns
        || first.output_columns != second.logical_input_columns
        || first.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    size_t extra_storage_variant = 0;
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const Bfloat16Linear_vulkan* extra_operator = extra_operators[index];
        ActivationBuffer* extra_output = extra_outputs[index];
        if (!extra_operator || !extra_output)
            return false;
        const Bfloat16Linear_vulkan::Implementation& extra = *extra_operator->d;
        if (!extra.vulkan_context
            || extra.vulkan_context.get() != first.vulkan_context.get()
            || !extra.pipeline
            || extra.input_columns != input.columns())
        {
            return false;
        }
        const size_t variant = vulkan_activation_storage_variant(extra.option);
        if (index == 0)
            extra_storage_variant = variant;
        else if (variant != extra_storage_variant)
            return false;
    }
    if (extra_storage_variant != vulkan_activation_storage_variant(first.option)
        || extra_storage_variant != 0)
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * second.output_columns
                                                                                         * sizeof(float),
                                                                                     output.dtype());
    const bool parallel_direct_host_output = parallel.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * parallel.output_columns
                                                                                                     * sizeof(float),
                                                                                                 parallel_output.dtype());
    ncnn::VkMat parallel_download;
    std::vector<ncnn::VkMat> extra_downloads(extra_operators.size());
    std::vector<bool> extra_direct_host_outputs(extra_operators.size(), false);
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  second.output_columns,
                                  transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(parallel_download,
                                  input.rows(),
                                  parallel.output_columns,
                                  transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->d;
        if (!prepare_staging_batch(extra_downloads[index],
                                   input.rows(),
                                   extra.output_columns,
                                   transfer_slot.staging_allocator, runtime_state))
        {
            return false;
        }
        extra_direct_host_outputs[index] = extra.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * extra.output_columns
                                                                                                * sizeof(float),
                                                                                            extra_outputs[index]->dtype());
    }
    const DType output_dtype = output.dtype();
    const DType parallel_output_dtype = parallel_output.dtype();
    output.reset(input.rows(), second.output_columns, false);
    parallel_output.reset(input.rows(), parallel.output_columns, false);
    for (size_t index = 0; index < extra_outputs.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->d;
        extra_outputs[index]->reset(input.rows(), extra.output_columns, false);
    }

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    ncnn::VkMat original_input_gpu;
    if (!record_mapped_upload(transfer_slot.upload,
                              original_input_gpu,
                              command,
                              first.option))
    {
        return false;
    }
    ncnn::VkMat input_gpu;
    command.record_clone(original_input_gpu, input_gpu, first.option);
    if (input_gpu.empty())
        return false;
    std::vector<ncnn::VkMat> input_quantize_bindings = {input_gpu};
    std::vector<ncnn::vk_constant_type> input_quantize_constants(2);
    input_quantize_constants[0].u32 = first.logical_input_columns;
    input_quantize_constants[1].u32 = static_cast<uint32_t>(input.rows());
    ncnn::VkMat input_quantize_dispatcher;
    input_quantize_dispatcher.w = static_cast<int>((first.logical_input_columns / 128) * 32);
    input_quantize_dispatcher.h = static_cast<int>(input.rows());
    input_quantize_dispatcher.c = 1;
    command.record_pipeline(first.quantize_pipeline.get(),
                            input_quantize_bindings,
                            input_quantize_constants,
                            input_quantize_dispatcher);

    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(first.output_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(second.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          first.vulkan_context->blob_allocator());
    ncnn::VkMat parallel_gpu;
    if (parallel_direct_host_output)
        parallel_gpu = prepare_direct_host_output(parallel_download, runtime_state);
    else
        parallel_gpu.create(static_cast<int>(parallel.output_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            first.vulkan_context->blob_allocator());
    std::vector<ncnn::VkMat> extra_gpu(extra_operators.size());
    if (intermediate_gpu.empty() || output_gpu.empty() || parallel_gpu.empty())
        return false;
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->d;
        if (extra_direct_host_outputs[index])
            extra_gpu[index] = prepare_direct_host_output(extra_downloads[index], runtime_state);
        else
            extra_gpu[index].create(static_cast<int>(extra.output_columns),
                                    static_cast<int>(input.rows()),
                                    vulkan_activation_element_size(extra.option),
                                    first.vulkan_context->blob_allocator());
        if (extra_gpu[index].empty())
            return false;
    }

    first.record_projection(input_gpu,
                            intermediate_gpu,
                            static_cast<uint32_t>(input.rows()),
                            command);

    std::vector<ncnn::VkMat> norm_bindings = {
        intermediate_gpu,
        first.rms_norm_weight};
    std::vector<ncnn::vk_constant_type> norm_constants(3);
    norm_constants[0].u32 = first.output_columns;
    norm_constants[1].u32 = static_cast<uint32_t>(input.rows());
    norm_constants[2].f = first.rms_norm_epsilon;
    ncnn::VkMat norm_dispatcher;
    norm_dispatcher.w = 32;
    norm_dispatcher.h = static_cast<int>(input.rows());
    norm_dispatcher.c = 1;
    command.record_pipeline_readonly(first.rms_norm_quantize_pipeline.get(),
                                     norm_bindings,
                                     {0, 1},
                                     norm_constants,
                                     norm_dispatcher);
    second.record_projection(intermediate_gpu,
                             output_gpu,
                             static_cast<uint32_t>(input.rows()),
                             command);
    parallel.record_projection(input_gpu,
                               parallel_gpu,
                               static_cast<uint32_t>(input.rows()),
                               command);

    std::vector<bool> cooperative_dispatches(extra_operators.size(), false);
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->d;
        cooperative_dispatches[index] = extra.record_projection(original_input_gpu,
                                                                extra_gpu[index],
                                                                static_cast<uint32_t>(input.rows()),
                                                                command);
    }

    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         second.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         first.vulkan_context->device(),
                                                         first.option,
                                                         output_dtype))
        || (!parallel_direct_host_output
            && !record_prepared_activation_staging_download(parallel_gpu,
                                                            input.rows(),
                                                            parallel.output_columns,
                                                            parallel_download,
                                                            command,
                                                            first.vulkan_context->device(),
                                                            first.option,
                                                            parallel_output_dtype)))
    {
        return false;
    }
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->d;
        if (!extra_direct_host_outputs[index]
            && !record_prepared_activation_staging_download(extra_gpu[index],
                                                            input.rows(),
                                                            extra.output_columns,
                                                            extra_downloads[index],
                                                            command,
                                                            extra.vulkan_context->device(),
                                                            extra.option,
                                                            extra_outputs[index]->dtype()))
        {
            return false;
        }
    }
    if (submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output)
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        if (!copy_staging_to_cpu_batch(extra_downloads[index],
                                       *extra_outputs[index]))
        {
            return false;
        }
    }
    runtime_state.dispatches += 4 + static_cast<uint64_t>(extra_operators.size());
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads += 2 + extra_operators.size();
    for (bool used_cooperative_matrix : cooperative_dispatches)
    {
        if (used_cooperative_matrix)
            ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    }
    return true;
#else
    (void)input;
    (void)next;
    (void)parallel_operator;
    (void)extra_operators;
    (void)extra_outputs;
    (void)output;
    (void)parallel_output;
    return false;
#endif
}

bool Float8Linear_vulkan::forward_swiglu_chain(const ActivationBuffer& input,
                                               const Float8Linear_vulkan& up_operator,
                                               const Float8Linear_vulkan& down_operator,
                                               ExpertActivation activation,
                                               float activation_limit,
                                               ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& gate = *d;
    const Implementation& up = *up_operator.d;
    const Implementation& down = *down_operator.d;
    if (activation != ExpertActivation::DeepSeekSwiGlu
        || !gate.vulkan_context || !up.vulkan_context || !down.vulkan_context
        || gate.vulkan_context.get() != up.vulkan_context.get()
        || gate.vulkan_context.get() != down.vulkan_context.get()
        || !gate.pipeline || !up.pipeline || !down.pipeline || !gate.swiglu_quantize_pipeline
        || input.rows() == 0 || input.columns() != gate.logical_input_columns
        || input.columns() != up.logical_input_columns
        || gate.output_columns != up.output_columns
        || gate.output_columns != down.logical_input_columns
        || gate.output_columns % 128 != 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = gate.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = gate.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = gate.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * down.output_columns
                                                                                        * sizeof(float),
                                                                                    output.dtype());
    if (!fill_float8_quantized_staging(input,
                                       gate.logical_input_columns,
                                       gate.optimization_flags,
                                       transfer_slot.upload,
                                       transfer_slot.staging_allocator,
                                       runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), down.output_columns, transfer_slot.staging_allocator, runtime_state))
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
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, gate.option))
        return false;
    ncnn::VkMat gate_gpu;
    gate_gpu.create(static_cast<int>(gate.output_columns),
                    static_cast<int>(input.rows()),
                    sizeof(float),
                    gate.vulkan_context->blob_allocator());
    ncnn::VkMat up_gpu;
    up_gpu.create(static_cast<int>(up.output_columns),
                  static_cast<int>(input.rows()),
                  sizeof(float),
                  gate.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(down.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          gate.vulkan_context->blob_allocator());
    if (gate_gpu.empty() || up_gpu.empty() || output_gpu.empty())
        return false;

    gate.record_projection(input_gpu,
                           gate_gpu,
                           static_cast<uint32_t>(input.rows()),
                           command);
    up.record_projection(input_gpu,
                         up_gpu,
                         static_cast<uint32_t>(input.rows()),
                         command);

    std::vector<ncnn::VkMat> activation_bindings = {gate_gpu, up_gpu};
    std::vector<ncnn::vk_constant_type> activation_constants(3);
    activation_constants[0].u32 = gate.output_columns;
    activation_constants[1].u32 = static_cast<uint32_t>(input.rows());
    activation_constants[2].f = activation_limit;
    ncnn::VkMat activation_dispatcher;
    activation_dispatcher.w = static_cast<int>((gate.output_columns / 128) * 32);
    activation_dispatcher.h = static_cast<int>(input.rows());
    activation_dispatcher.c = 1;
    command.record_pipeline(gate.swiglu_quantize_pipeline.get(),
                            activation_bindings,
                            activation_constants,
                            activation_dispatcher);

    down.record_projection(gate_gpu,
                           output_gpu,
                           static_cast<uint32_t>(input.rows()),
                           command);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         down.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         gate.vulkan_context->device(),
                                                         gate.option,
                                                         output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 3;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)input;
    (void)up_operator;
    (void)down_operator;
    (void)activation;
    (void)activation_limit;
    (void)output;
    return false;
#endif
}

class Mxfp4Linear_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    // Pack the read-only MXFP4 projection inputs into one device buffer.
    ncnn::VkMat storage;
    ncnn::VkMat packed;
    ncnn::VkMat scales;
    ncnn::VkMat bias;
    ncnn::Option option;
    bool indexed_storage = false;
    uint32_t indexed_scales_word_offset = 0;
    uint32_t indexed_bias_word_offset = 0;
#endif
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
};

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
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
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

    ncnn::Mat combined;
    combined.create(static_cast<int>(storage_size), sizeof(uint8_t));
    if (combined.empty())
        return {};
    std::memset(combined.data, 0, storage_size);
    std::memcpy(combined.data, matrix.mxfp4_blocks.data(), matrix.mxfp4_blocks.size());
    std::memcpy(static_cast<std::byte*>(combined.data) + scales_offset,
                matrix.mxfp4_scales.data(),
                matrix.mxfp4_scales.size());
    float* bias_values = reinterpret_cast<float*>(static_cast<std::byte*>(combined.data) + bias_offset);
    if (bias && bias->dtype == DType::Float32)
    {
        std::copy(float_bias_values.begin(), float_bias_values.end(), bias_values);
    }
    else if (bias)
    {
        for (uint32_t index = 0; index < output_columns; ++index)
            bias_values[index] = bfloat16_to_float(bfloat16_bias_values[index]);
    }

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
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = weight_allocator;
    upload_option.workspace_vkallocator = weight_allocator;
    bool uploaded = false;
    if (upload_batch)
    {
        uploaded = upload_batch->record(combined, implementation.storage, implementation.option, weight_allocator);
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
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator, runtime_state))
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
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
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
    dispatcher.w = static_cast<int>(implementation.output_columns * 32);
    dispatcher.h = static_cast<int>(input.rows());
    dispatcher.c = 1;
    command.record_pipeline(implementation.pipeline.get(), bindings, constants, dispatcher);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         implementation.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         implementation.vulkan_context->device(),
                                                         implementation.option,
                                                         output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
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

class QnkLinear_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    ncnn::VkMat storage;
    ncnn::VkMat quantized;
    ncnn::VkMat bias;
    ncnn::Option option;
    uint64_t optimization_flags = OptimizationDefaultFlags;
#endif
    DType dtype = DType::Q4K;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
    uint32_t block_size = 0;
};

#if NCNN_MOE_WITH_VULKAN

static bool create_qnk_projection_pipeline(const std::shared_ptr<VulkanContext>& context,
                                           const ncnn::Option& option,
                                           std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(qnk_projection_shader,
                                                                                      static_cast<int>(sizeof(qnk_projection_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    const size_t pipeline_key = option.use_subgroup_ops ? 1u : 0u;
    destination = context->find_pipeline(qnk_projection_shader, pipeline_key);
    if (destination)
        return true;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context](ncnn::Pipeline* value) {
                                                      const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                      delete value;
                                                  });
    context->cache_pipeline(qnk_projection_shader, pipeline_key, destination);
    return true;
}

static bool create_qnk_swiglu_pipeline(const std::shared_ptr<VulkanContext>& context,
                                       const ncnn::Option& option,
                                       std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(qnk_swiglu_shader,
                                                                                      static_cast<int>(sizeof(qnk_swiglu_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(qnk_swiglu_shader, 0);
    if (destination)
        return true;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(128, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context](ncnn::Pipeline* value) {
                                                      const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                      delete value;
                                                  });
    context->cache_pipeline(qnk_swiglu_shader, 0, destination);
    return true;
}

static size_t qnk_shader_type(DType dtype) noexcept
{
    return static_cast<size_t>(dtype) - static_cast<size_t>(DType::Q2K);
}

#endif

QnkLinear_vulkan::QnkLinear_vulkan()
    : d(new Implementation)
{
}

QnkLinear_vulkan::~QnkLinear_vulkan() = default;

std::shared_ptr<QnkLinear_vulkan> QnkLinear_vulkan::create(const TensorData& matrix,
                                                           const TensorData* bias,
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

std::shared_ptr<QnkLinear_vulkan> QnkLinear_vulkan::create_with_allocator(const TensorData& matrix,
                                                                          const TensorData* bias,
                                                                          uint32_t vulkan_device_index,
                                                                          ncnn::VkAllocator* weight_allocator,
                                                                          const VulkanRuntimePtr& vulkan_runtime,
                                                                          uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (!is_qnk_dtype(matrix.dtype)
        || matrix.shape.size() != 2
        || matrix.shape[0] == 0
        || matrix.shape[1] == 0
        || matrix.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 32)
        || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || !qnk_shape_supported(matrix.dtype, matrix.shape[0], matrix.shape[1]))
    {
        return {};
    }
    const std::span<const uint8_t> raw_values = matrix.qnk_values();
    const uint64_t expected_size = qnk_storage_bytes(matrix.dtype, matrix.shape[0], matrix.shape[1]);
    if (expected_size == 0
        || expected_size != raw_values.size()
        || expected_size > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    if (bias
        && (bias->shape.size() != 1
            || bias->shape[0] != output_columns
            || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    std::shared_ptr<QnkLinear_vulkan> result(new QnkLinear_vulkan);
    Implementation& implementation = *result->d;
    implementation.dtype = matrix.dtype;
    implementation.input_columns = input_columns;
    implementation.output_columns = output_columns;
    implementation.block_count = input_columns / qnk_block_elements;
    implementation.block_size = static_cast<uint32_t>(qnk_block_bytes(matrix.dtype));
    implementation.optimization_flags = optimization_flags;
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

    const size_t raw_size = static_cast<size_t>(expected_size);
    const size_t bias_size = static_cast<size_t>(output_columns) * sizeof(float);
    const size_t storage_alignment = std::max<size_t>(4, device->info.buffer_offset_alignment());
    const auto aligned_segment = [storage_alignment](size_t bytes) -> size_t {
        const size_t remainder = bytes % storage_alignment;
        if (remainder == 0)
            return bytes;
        const size_t padding = storage_alignment - remainder;
        if (bytes > std::numeric_limits<size_t>::max() - padding)
            return 0;
        return bytes + padding;
    };
    const size_t raw_segment_size = aligned_segment(raw_size);
    const size_t bias_offset = raw_segment_size;
    const size_t bias_segment_size = aligned_segment(bias_size);
    if (raw_segment_size == 0
        || bias_segment_size == 0
        || raw_segment_size > std::numeric_limits<size_t>::max() - bias_segment_size)
    {
        return {};
    }
    const size_t storage_size = raw_segment_size + bias_segment_size;
    if (storage_size == 0 || storage_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {};
    if (!weight_allocator)
    {
        implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(device, storage_size));
        weight_allocator = implementation.weight_allocator.get();
    }
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));

    ncnn::Mat raw;
    raw.create(static_cast<int>(raw_size), sizeof(uint8_t));
    if (raw.empty())
        return {};
    std::memcpy(raw.data, raw_values.data(), raw_size);

    ncnn::Mat biases;
    biases.create(static_cast<int>(output_columns), sizeof(float));
    if (biases.empty())
        return {};
    float* bias_values = static_cast<float*>(biases.data);
    if (!bias)
    {
        std::fill_n(bias_values, output_columns, 0.0f);
    }
    else if (bias->dtype == DType::Float32)
    {
        const std::span<const float> values = bias->float32_values();
        if (values.size() != output_columns)
            return {};
        std::copy(values.begin(), values.end(), bias_values);
    }
    else
    {
        const std::span<const uint16_t> values = bias->bfloat16_values();
        if (values.size() != output_columns)
            return {};
        for (uint32_t index = 0; index < output_columns; ++index)
            bias_values[index] = bfloat16_to_float(values[index]);
    }

    ncnn::Mat combined;
    combined.create(static_cast<int>(storage_size), sizeof(uint8_t));
    if (combined.empty())
        return {};
    std::memset(combined.data, 0, storage_size);
    std::memcpy(combined.data, raw.data, raw_size);
    std::memcpy(static_cast<std::byte*>(combined.data) + bias_offset, biases.data, bias_size);

    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_qnk_projection_pipeline(implementation.vulkan_context, implementation.option, implementation.pipeline))
            return {};
    }
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = weight_allocator;
    upload_option.workspace_vkallocator = weight_allocator;
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(device);
        command.record_upload(combined, implementation.storage, upload_option);
        uploaded = !implementation.storage.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return {};

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
    implementation.quantized = storage_view(0, raw_segment_size);
    implementation.bias = storage_view(bias_offset, bias_size);
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

bool QnkLinear_vulkan::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || input.dtype() != DType::Float32
        || input.rows() == 0
        || input.columns() != implementation.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * implementation.input_columns * sizeof(float),
                                                                                             input.dtype());
    const bool direct_host_output = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * implementation.output_columns * sizeof(float),
                                                                                              output.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator, runtime_state))
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
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
        return false;
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(implementation.output_columns), static_cast<int>(input.rows()), sizeof(float), implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings(4);
    bindings[0] = input_gpu;
    bindings[1] = implementation.quantized;
    bindings[2] = implementation.bias;
    bindings[3] = output_gpu;
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.rows());
    constants[4].u32 = implementation.block_size;
    constants[5].u32 = static_cast<uint32_t>(qnk_shader_type(implementation.dtype));
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(implementation.output_columns * 32);
    dispatcher.h = static_cast<int>(input.rows());
    dispatcher.c = 1;
    command.record_pipeline(implementation.pipeline.get(), bindings, constants, dispatcher);
    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         implementation.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         implementation.vulkan_context->device(),
                                                         implementation.option,
                                                         output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
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

DType QnkLinear_vulkan::dtype() const noexcept
{
    return d->dtype;
}

uint32_t QnkLinear_vulkan::input_columns() const noexcept
{
    return d->input_columns;
}

uint32_t QnkLinear_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

class QnkExpert_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<QnkLinear_vulkan> gate_up;
    std::shared_ptr<QnkLinear_vulkan> down;
    std::shared_ptr<VulkanContext> vulkan_context;
    std::shared_ptr<ncnn::Pipeline> swiglu_pipeline;
    ncnn::Option option;
#endif
    uint32_t intermediate_columns = 0;
    uint32_t output_columns = 0;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
    bool interleave_gate_up_rows = true;
};

QnkExpert_vulkan::QnkExpert_vulkan()
    : d(new Implementation)
{
}

QnkExpert_vulkan::~QnkExpert_vulkan() = default;

uint32_t QnkExpert_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

std::shared_ptr<QnkExpert_vulkan> QnkExpert_vulkan::create(const TensorData& gate_up,
                                                           const TensorData* gate_up_bias,
                                                           const TensorData& down,
                                                           const TensorData* down_bias,
                                                           float activation_limit,
                                                           uint32_t vulkan_device_index,
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

std::shared_ptr<QnkExpert_vulkan> QnkExpert_vulkan::create_with_allocator(const TensorData& gate_up,
                                                                          const TensorData* gate_up_bias,
                                                                          const TensorData& down,
                                                                          const TensorData* down_bias,
                                                                          float activation_limit,
                                                                          uint32_t vulkan_device_index,
                                                                          ncnn::VkAllocator* weight_allocator,
                                                                          ExpertActivation activation,
                                                                          const VulkanRuntimePtr& vulkan_runtime,
                                                                          uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if ((activation != ExpertActivation::Silu
         && activation != ExpertActivation::GptOssSwiGlu
         && activation != ExpertActivation::DeepSeekSwiGlu)
        || !is_qnk_dtype(gate_up.dtype)
        || gate_up.dtype != down.dtype
        || gate_up.shape.size() != 2
        || down.shape.size() != 2
        || gate_up.shape[0] == 0
        || gate_up.shape[0] % 2 != 0
        || down.shape[0] == 0
        || down.shape[1] != gate_up.shape[0] / 2
        || !qnk_shape_supported(gate_up.dtype, gate_up.shape[0], gate_up.shape[1])
        || !qnk_shape_supported(down.dtype, down.shape[0], down.shape[1])
        || activation_limit < 0.0f)
    {
        return {};
    }
    if (gate_up_bias
        && (gate_up_bias->shape.size() != 1
            || gate_up_bias->shape[0] != gate_up.shape[0]
            || (gate_up_bias->dtype != DType::Float32 && gate_up_bias->dtype != DType::BFloat16)))
    {
        return {};
    }
    if (down_bias
        && (down_bias->shape.size() != 1
            || down_bias->shape[0] != down.shape[0]
            || (down_bias->dtype != DType::Float32 && down_bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    std::shared_ptr<QnkExpert_vulkan> result(new QnkExpert_vulkan);
    Implementation& implementation = *result->d;
    implementation.gate_up = QnkLinear_vulkan::create_with_allocator(gate_up,
                                                                     gate_up_bias,
                                                                     vulkan_device_index,
                                                                     weight_allocator,
                                                                     vulkan_runtime,
                                                                     optimization_flags);
    implementation.down = QnkLinear_vulkan::create_with_allocator(down,
                                                                  down_bias,
                                                                  vulkan_device_index,
                                                                  weight_allocator,
                                                                  vulkan_runtime,
                                                                  optimization_flags);
    if (!implementation.gate_up || !implementation.down)
        return {};

    const QnkLinear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const QnkLinear_vulkan::Implementation& down_projection = *implementation.down->d;
    if (!gate.vulkan_context || gate.vulkan_context != down_projection.vulkan_context)
        return {};
    implementation.vulkan_context = gate.vulkan_context;
    implementation.option = gate.option;
    implementation.intermediate_columns = gate.output_columns / 2;
    implementation.output_columns = implementation.down->output_columns();
    implementation.activation_limit = activation_limit;
    implementation.activation = activation;
    implementation.interleave_gate_up_rows = gate_up.qnk_interleave_rows;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_qnk_swiglu_pipeline(implementation.vulkan_context,
                                        implementation.option,
                                        implementation.swiglu_pipeline))
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
    return {};
#endif
}

bool QnkExpert_vulkan::forward_batch(std::span<const QnkExpert_vulkan*> experts,
                                     std::span<const ActivationBuffer*> inputs,
                                     std::span<ActivationBuffer*> outputs)
{
#if NCNN_MOE_WITH_VULKAN
    if (experts.empty() || experts.size() != inputs.size() || experts.size() != outputs.size())
        return false;

    const QnkExpert_vulkan* first_operator = experts.front();
    if (!first_operator || !first_operator->d)
        return false;
    const Implementation& first_expert = *first_operator->d;
    if (!first_expert.gate_up
        || !first_expert.down
        || !first_expert.vulkan_context
        || !first_expert.swiglu_pipeline)
    {
        return false;
    }

    const QnkLinear_vulkan::Implementation& first_gate = *first_expert.gate_up->d;
    const QnkLinear_vulkan::Implementation& first_down = *first_expert.down->d;
    if (!first_gate.pipeline
        || !first_down.pipeline
        || first_gate.output_columns % 2 != 0
        || first_down.input_columns != first_expert.intermediate_columns
        || first_down.output_columns != first_expert.output_columns)
    {
        return false;
    }

    const uint32_t input_columns = first_gate.input_columns;
    const uint32_t output_columns = first_down.output_columns;
    const DType output_dtype = outputs.front() ? outputs.front()->dtype() : DType::Int32;
    if (input_columns == 0 || output_columns == 0 || output_dtype != DType::Float32)
        return false;

    size_t total_rows = 0;
    for (size_t index = 0; index < experts.size(); ++index)
    {
        const QnkExpert_vulkan* operator_instance = experts[index];
        const ActivationBuffer* input = inputs[index];
        const ActivationBuffer* output = outputs[index];
        if (!operator_instance
            || !operator_instance->d
            || !input
            || !output
            || input->rows() == 0
            || input->rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
            || input->rows() > static_cast<size_t>(std::numeric_limits<int>::max())
            || input->dtype() != DType::Float32
            || input->columns() != input_columns
            || output->dtype() != output_dtype)
        {
            return false;
        }
        const Implementation& expert = *operator_instance->d;
        if (!expert.gate_up
            || !expert.down
            || !expert.vulkan_context
            || !expert.swiglu_pipeline
            || expert.vulkan_context != first_expert.vulkan_context
            || expert.intermediate_columns != first_expert.intermediate_columns
            || expert.output_columns != output_columns
            || expert.activation != first_expert.activation
            || expert.activation_limit != first_expert.activation_limit
            || expert.interleave_gate_up_rows != first_expert.interleave_gate_up_rows)
        {
            return false;
        }
        const QnkLinear_vulkan::Implementation& gate = *expert.gate_up->d;
        const QnkLinear_vulkan::Implementation& down = *expert.down->d;
        if (!gate.pipeline
            || !down.pipeline
            || gate.vulkan_context != first_expert.vulkan_context
            || down.vulkan_context != first_expert.vulkan_context
            || gate.input_columns != input_columns
            || gate.output_columns != first_gate.output_columns
            || down.input_columns != first_down.input_columns
            || down.output_columns != output_columns)
        {
            return false;
        }
        if (total_rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - input->rows())
            return false;
        total_rows += input->rows();
    }
    if (total_rows == 0 || total_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;

    VulkanRuntimeState& runtime_state = first_expert.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = first_expert.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!prepare_staging_batch(transfer_slot.upload,
                               total_rows,
                               input_columns,
                               transfer_slot.staging_allocator,
                               runtime_state,
                               sizeof(float))
        || !prepare_staging_batch(transfer_slot.download,
                                  total_rows,
                                  output_columns,
                                  transfer_slot.staging_allocator,
                                  runtime_state,
                                  sizeof(float)))
    {
        return false;
    }

    ncnn::Mat mapped_input = transfer_slot.upload.mapped();
    if (mapped_input.empty()
        || mapped_input.dims != 2
        || mapped_input.w != static_cast<int>(input_columns)
        || mapped_input.h != static_cast<int>(total_rows)
        || mapped_input.elemsize != sizeof(float)
        || mapped_input.elempack != 1)
    {
        return false;
    }
    auto* mapped_input_data = static_cast<std::byte*>(mapped_input.data);
    size_t row_offset = 0;
    for (const ActivationBuffer* input : inputs)
    {
        const size_t input_size = input->rows() * static_cast<size_t>(input_columns) * sizeof(float);
        if (input->bytes().size() != input_size)
            return false;
        std::memcpy(mapped_input_data + row_offset * static_cast<size_t>(input_columns) * sizeof(float),
                    input->bytes().data(),
                    input_size);
        row_offset += input->rows();
    }
    transfer_slot.upload.allocator->flush(transfer_slot.upload.data);
    transfer_slot.upload.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    transfer_slot.upload.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    std::unique_lock<std::mutex> lock(first_expert.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first_gate.option))
        return false;
    ncnn::VkMat output_gpu;
    output_gpu.create(static_cast<int>(output_columns),
                      static_cast<int>(total_rows),
                      sizeof(float),
                      first_expert.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> intermediates;
    intermediates.reserve(experts.size() * 2);
    std::vector<ncnn::VkMat> bindings(4);
    std::vector<ncnn::vk_constant_type> constants(6);
    row_offset = 0;
    for (size_t index = 0; index < experts.size(); ++index)
    {
        const Implementation& expert = *experts[index]->d;
        const QnkLinear_vulkan::Implementation& gate = *expert.gate_up->d;
        const QnkLinear_vulkan::Implementation& down = *expert.down->d;
        const size_t rows = inputs[index]->rows();
        ncnn::VkMat input_view = row_view(input_gpu, row_offset, rows);
        ncnn::VkMat output_view = row_view(output_gpu, row_offset, rows);
        if (input_view.empty() || output_view.empty())
            return false;

        ncnn::VkMat gate_up_gpu;
        gate_up_gpu.create(static_cast<int>(gate.output_columns),
                           static_cast<int>(rows),
                           sizeof(float),
                           first_expert.vulkan_context->blob_allocator());
        ncnn::VkMat intermediate_gpu;
        intermediate_gpu.create(static_cast<int>(expert.intermediate_columns),
                                static_cast<int>(rows),
                                sizeof(float),
                                first_expert.vulkan_context->blob_allocator());
        if (gate_up_gpu.empty() || intermediate_gpu.empty())
            return false;
        intermediates.push_back(gate_up_gpu);
        intermediates.push_back(intermediate_gpu);

        bindings[0] = input_view;
        bindings[1] = gate.quantized;
        bindings[2] = gate.bias;
        bindings[3] = gate_up_gpu;
        constants[0].u32 = gate.input_columns;
        constants[1].u32 = gate.output_columns;
        constants[2].u32 = gate.block_count;
        constants[3].u32 = static_cast<uint32_t>(rows);
        constants[4].u32 = gate.block_size;
        constants[5].u32 = static_cast<uint32_t>(qnk_shader_type(gate.dtype));
        ncnn::VkMat gate_dispatcher;
        gate_dispatcher.w = static_cast<int>(gate.output_columns * 32);
        gate_dispatcher.h = static_cast<int>(rows);
        gate_dispatcher.c = 1;
        command.record_pipeline(gate.pipeline.get(), bindings, constants, gate_dispatcher);

        bindings.resize(2);
        constants.resize(5);
        bindings[0] = gate_up_gpu;
        bindings[1] = intermediate_gpu;
        constants[0].u32 = expert.intermediate_columns;
        constants[1].u32 = static_cast<uint32_t>(rows);
        constants[2].u32 = expert.activation == ExpertActivation::GptOssSwiGlu ? 0u : 1u;
        constants[3].f = expert.activation_limit;
        constants[4].u32 = expert.interleave_gate_up_rows ? 0u : 1u;
        ncnn::VkMat activation_dispatcher;
        activation_dispatcher.w = static_cast<int>(expert.intermediate_columns * 128);
        activation_dispatcher.h = static_cast<int>(rows);
        activation_dispatcher.c = 1;
        command.record_pipeline(expert.swiglu_pipeline.get(),
                                bindings,
                                constants,
                                activation_dispatcher);

        bindings.resize(4);
        constants.resize(6);
        bindings[0] = intermediate_gpu;
        bindings[1] = down.quantized;
        bindings[2] = down.bias;
        bindings[3] = output_view;
        constants[0].u32 = down.input_columns;
        constants[1].u32 = down.output_columns;
        constants[2].u32 = down.block_count;
        constants[3].u32 = static_cast<uint32_t>(rows);
        constants[4].u32 = down.block_size;
        constants[5].u32 = static_cast<uint32_t>(qnk_shader_type(down.dtype));
        ncnn::VkMat down_dispatcher;
        down_dispatcher.w = static_cast<int>(down.output_columns * 32);
        down_dispatcher.h = static_cast<int>(rows);
        down_dispatcher.c = 1;
        command.record_pipeline(down.pipeline.get(), bindings, constants, down_dispatcher);
        row_offset += rows;
    }

    if (!record_prepared_activation_staging_download(output_gpu,
                                                     total_rows,
                                                     output_columns,
                                                     transfer_slot.download,
                                                     command,
                                                     first_expert.vulkan_context->device(),
                                                     first_down.option,
                                                     output_dtype)
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batches(transfer_slot.download,
                                        inputs,
                                        outputs,
                                        output_columns))
    {
        return false;
    }
    lock.unlock();
    runtime_state.dispatches += static_cast<uint64_t>(experts.size()) * 3;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)experts;
    (void)inputs;
    (void)outputs;
    return false;
#endif
}

bool QnkExpert_vulkan::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.gate_up
        || !implementation.down
        || !implementation.vulkan_context
        || !implementation.swiglu_pipeline
        || input.rows() == 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
        || input.columns() != implementation.gate_up->input_columns())
    {
        return false;
    }
    const QnkLinear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const QnkLinear_vulkan::Implementation& down = *implementation.down->d;
    if (gate.output_columns % 2 != 0
        || down.input_columns != implementation.intermediate_columns
        || down.output_columns != implementation.output_columns)
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * gate.input_columns * sizeof(float),
                                                                                             input.dtype());
    const bool direct_host_output = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * down.output_columns * sizeof(float),
                                                                                              output.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  down.output_columns,
                                  transfer_slot.staging_allocator,
                                  runtime_state))
    {
        return false;
    }

    const DType output_dtype = output.dtype();
    output.reset(input.rows(), down.output_columns, false);
    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, gate.option))
        return false;

    ncnn::VkMat gate_up_gpu;
    gate_up_gpu.create(static_cast<int>(gate.output_columns),
                       static_cast<int>(input.rows()),
                       sizeof(float),
                       implementation.vulkan_context->blob_allocator());
    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(implementation.intermediate_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            implementation.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
    {
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    }
    else
    {
        output_gpu.create(static_cast<int>(down.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          implementation.vulkan_context->blob_allocator());
    }
    if (gate_up_gpu.empty() || intermediate_gpu.empty() || output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> gate_bindings = {
        input_gpu,
        gate.quantized,
        gate.bias,
        gate_up_gpu,
    };
    std::vector<ncnn::vk_constant_type> gate_constants(6);
    gate_constants[0].u32 = gate.input_columns;
    gate_constants[1].u32 = gate.output_columns;
    gate_constants[2].u32 = gate.block_count;
    gate_constants[3].u32 = static_cast<uint32_t>(input.rows());
    gate_constants[4].u32 = gate.block_size;
    gate_constants[5].u32 = static_cast<uint32_t>(qnk_shader_type(gate.dtype));
    ncnn::VkMat gate_dispatcher;
    gate_dispatcher.w = static_cast<int>(gate.output_columns * 32);
    gate_dispatcher.h = static_cast<int>(input.rows());
    gate_dispatcher.c = 1;
    command.record_pipeline(gate.pipeline.get(), gate_bindings, gate_constants, gate_dispatcher);

    std::vector<ncnn::VkMat> activation_bindings = {
        gate_up_gpu,
        intermediate_gpu,
    };
    std::vector<ncnn::vk_constant_type> activation_constants(5);
    activation_constants[0].u32 = implementation.intermediate_columns;
    activation_constants[1].u32 = static_cast<uint32_t>(input.rows());
    activation_constants[2].u32 = implementation.activation == ExpertActivation::GptOssSwiGlu ? 0u : 1u;
    activation_constants[3].f = implementation.activation_limit;
    activation_constants[4].u32 = implementation.interleave_gate_up_rows ? 0u : 1u;
    ncnn::VkMat activation_dispatcher;
    activation_dispatcher.w = static_cast<int>(implementation.intermediate_columns * 128);
    activation_dispatcher.h = static_cast<int>(input.rows());
    activation_dispatcher.c = 1;
    command.record_pipeline(implementation.swiglu_pipeline.get(),
                            activation_bindings,
                            activation_constants,
                            activation_dispatcher);

    std::vector<ncnn::VkMat> down_bindings = {
        intermediate_gpu,
        down.quantized,
        down.bias,
        output_gpu,
    };
    std::vector<ncnn::vk_constant_type> down_constants(6);
    down_constants[0].u32 = down.input_columns;
    down_constants[1].u32 = down.output_columns;
    down_constants[2].u32 = down.block_count;
    down_constants[3].u32 = static_cast<uint32_t>(input.rows());
    down_constants[4].u32 = down.block_size;
    down_constants[5].u32 = static_cast<uint32_t>(qnk_shader_type(down.dtype));
    ncnn::VkMat down_dispatcher;
    down_dispatcher.w = static_cast<int>(down.output_columns * 32);
    down_dispatcher.h = static_cast<int>(input.rows());
    down_dispatcher.c = 1;
    command.record_pipeline(down.pipeline.get(), down_bindings, down_constants, down_dispatcher);

    if ((!direct_host_output
         && !record_prepared_activation_staging_download(output_gpu,
                                                         input.rows(),
                                                         down.output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         implementation.vulkan_context->device(),
                                                         down.option,
                                                         output_dtype))
        || submit_compute_and_wait(command, runtime_state) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 3;
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

class Mxfp4Expert_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<ncnn::Pipeline> gate_up_pipeline;
#endif
    std::shared_ptr<Mxfp4Linear_vulkan> gate_up;
    std::shared_ptr<Mxfp4Linear_vulkan> down;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
};

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
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(mxfp4_gate_up_shader, pipeline_key, destination);
    return true;
}

static bool create_mxfp4_route_aggregation_pipeline(const std::shared_ptr<VulkanContext>& context, const ncnn::Option& option,
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
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context](ncnn::Pipeline* value) {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        delete value;
    });
    context->cache_pipeline(mxfp4_route_aggregation_shader, 0, destination);
    return true;
}

// Indexed Expert dispatch uses one storage-buffer binding per selected Expert
// slot. Larger top-k batches are chunked and reuse this fixed descriptor table.
static constexpr uint32_t mxfp4_indexed_max_experts = 8;

static bool create_mxfp4_indexed_pipeline(const std::shared_ptr<VulkanContext>& context,
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
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context](ncnn::Pipeline* value) {
                                                      const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                      delete value;
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
    if (gate_up.shape.size() != 2 || gate_up.shape[0] % 2 != 0 || down.shape.size() != 2 || down.shape[1] != gate_up.shape[0] / 2 || activation_limit < 0.0f)
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
    Implementation& implementation = *result->d;
    implementation.gate_up = std::move(gate_up_projection);
    implementation.down = std::move(down_projection);
    implementation.activation_limit = activation_limit;
    implementation.activation = activation;
    Mxfp4Linear_vulkan::Implementation& gate_implementation = *implementation.gate_up->d;
    if (upload_batch)
    {
        if (!create_mxfp4_gate_up_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.gate_up_pipeline))
            return {};
    }
    else
    {
        const std::lock_guard<std::mutex> lock(gate_implementation.vulkan_context->command_mutex());
        if (!create_mxfp4_gate_up_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.gate_up_pipeline))
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
        if (!create_mxfp4_gate_up_pipeline(gate_implementation.vulkan_context, gate_implementation.option, implementation.gate_up_pipeline))
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
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), down.output_columns, transfer_slot.staging_allocator, runtime_state))
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
        ++runtime_state.command_buffer_reuses;
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
    record(input_gpu, intermediate_gpu, output_gpu, command);

    if (!record_prepared_activation_staging_download(output_gpu,
                                                     input.rows(),
                                                     down.output_columns,
                                                     transfer_slot.download,
                                                     command,
                                                     gate.vulkan_context->device(),
                                                     down.option,
                                                     output_dtype)
        || submit_compute_and_wait(command, runtime_state) != 0 || !copy_staging_to_cpu_batch(transfer_slot.download, output))
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
                                ncnn::VkCompute& cmd) const
{
    const Implementation& implementation = *d;
    const Mxfp4Linear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const Mxfp4Linear_vulkan::Implementation& down = *implementation.down->d;
    const uint32_t intermediate_columns = gate.output_columns / 2;
    const uint32_t rows = static_cast<uint32_t>(input.h);

    std::vector<ncnn::VkMat> bindings(5);
    bindings[0] = input;
    bindings[1] = gate.packed;
    bindings[2] = gate.scales;
    bindings[3] = gate.bias;
    bindings[4] = intermediate;
    std::vector<ncnn::vk_constant_type> constants(6);
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
    gate_dispatcher.w = static_cast<int>(intermediate_columns * 32);
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
    down_dispatcher.w = static_cast<int>(down.output_columns * 32);
    down_dispatcher.h = static_cast<int>(rows);
    down_dispatcher.c = 1;
    cmd.record_pipeline(down.pipeline.get(), bindings, constants, down_dispatcher);
}

bool VulkanExpertBackend::forward_batch(std::span<const ExpertBackendRequest> requests,
                                        std::span<const Selection> selected)
{
    if (selected.empty())
        return true;
    if (!selected.front().entry || !selected.front().entry->operation)
        return false;
    const Mxfp4Expert_vulkan::Implementation& first_expert = *selected.front().entry->operation->d;
    if (!first_expert.gate_up || !first_expert.down || !first_expert.gate_up_pipeline)
    {
        return false;
    }
    const Mxfp4Linear_vulkan::Implementation& first_gate = *first_expert.gate_up->d;
    const Mxfp4Linear_vulkan::Implementation& first_down = *first_expert.down->d;
    if (!first_gate.vulkan_context || first_gate.vulkan_context != first_down.vulkan_context)
    {
        return false;
    }

    const uint32_t input_columns = first_gate.input_columns;
    const uint32_t output_columns = first_down.output_columns;
    bool use_indexed = has_flag(optimization_flags, OptimizationVulkanIndexedExperts)
                       && selected.size() >= 4
                       && selected.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    if (use_indexed
        && (input_columns == 0 || first_gate.output_columns / 2 == 0 || output_columns == 0
            || !first_gate.indexed_storage || !first_down.indexed_storage
            || first_gate.storage.empty() || first_down.storage.empty()))
    {
        use_indexed = false;
    }

    VulkanRuntimeState& runtime_state = first_gate.vulkan_context->runtime_state();
    size_t total_rows = 0;
    size_t maximum_request_rows = 0;
    for (const Selection& selection : selected)
    {
        if (selection.request_index >= requests.size()
            || !selection.entry
            || !selection.entry->operation)
        {
            return false;
        }
        const ExpertBackendRequest& request = requests[selection.request_index];
        const auto& expert = *selection.entry->operation->d;
        if (!expert.gate_up || !expert.down)
            return false;
        const auto& gate = *expert.gate_up->d;
        const auto& down = *expert.down->d;
        if (!request.input || !request.output || request.input->rows() == 0
            || request.input->dtype() != DType::Float32 || request.input->columns() != input_columns
            || request.input->bytes().size() != request.input->rows() * static_cast<size_t>(input_columns) * sizeof(float)
            || gate.vulkan_context != first_gate.vulkan_context || down.vulkan_context != first_gate.vulkan_context
            || gate.input_columns != input_columns
            || gate.output_columns % 2 != 0 || down.input_columns != gate.output_columns / 2
            || down.output_columns != output_columns
            || request.input->rows() > static_cast<size_t>(std::numeric_limits<int>::max()) - total_rows)
        {
            return false;
        }

        if (use_indexed)
        {
            if (expert.activation != first_expert.activation
                || expert.activation_limit != first_expert.activation_limit
                || !gate.indexed_storage || !down.indexed_storage
                || gate.storage.empty() || down.storage.empty()
                || gate.output_columns != first_gate.output_columns
                || gate.block_count != first_gate.block_count
                || down.block_count != first_down.block_count
                || gate.indexed_scales_word_offset != first_gate.indexed_scales_word_offset
                || gate.indexed_bias_word_offset != first_gate.indexed_bias_word_offset
                || down.indexed_scales_word_offset != first_down.indexed_scales_word_offset
                || down.indexed_bias_word_offset != first_down.indexed_bias_word_offset)
            {
                use_indexed = false;
            }
            else
            {
                maximum_request_rows = std::max(maximum_request_rows, request.input->rows());
            }
        }
        total_rows += request.input->rows();
    }
    bool use_tiled_indexed = false;
    size_t indexed_chunk_count = 0;
    if (use_indexed)
    {
        // Tile only when the largest selected Expert has row reuse.
        use_tiled_indexed = maximum_request_rows >= 2;
        indexed_chunk_count = (selected.size() + mxfp4_indexed_max_experts - 1)
                              / mxfp4_indexed_max_experts;
        // The indexed shader has a generic slot switch, while the regular
        // path uses a specialized projection pipeline. Keep the dynamic
        // capability model-neutral, but only select it when the dispatch
        // reduction is large enough to amortize that selector overhead.
        // This is intentionally an internal policy (not a model option):
        // top-k=10 decode uses the regular path, whereas top-k=16/24 can
        // benefit from 8+8/8+8+8 chunking.
        const uint64_t regular_dispatches = static_cast<uint64_t>(selected.size()) * 2;
        const uint64_t indexed_dispatches = static_cast<uint64_t>(indexed_chunk_count) * 2;
        const uint64_t selector_overhead_factor = use_tiled_indexed ? 4 : 6;
        if (indexed_dispatches == 0
            || regular_dispatches <= indexed_dispatches * selector_overhead_factor)
        {
            use_indexed = false;
            use_tiled_indexed = false;
            indexed_chunk_count = 0;
        }
    }

    if (use_indexed
        && !indexed_pipeline
        && !create_mxfp4_indexed_pipeline(first_gate.vulkan_context,
                                          first_gate.option,
                                          indexed_pipeline))
    {
        use_indexed = false;
        use_tiled_indexed = false;
        indexed_chunk_count = 0;
    }

    std::vector<uint32_t> expert_row_offsets;
    std::vector<uint32_t> row_slots;
    if (use_indexed)
    {
        expert_row_offsets.resize(selected.size() + 1);
        row_slots.resize(total_rows);
    }

    ActivationBuffer* aggregated_output = nullptr;
    uint32_t aggregated_token_count = 0;
    const bool use_route_aggregation = route_aggregation_enabled(requests,
                                                                 selected,
                                                                 output_columns,
                                                                 aggregated_output,
                                                                 aggregated_token_count,
                                                                 optimization_flags);
    std::vector<uint32_t> route_offsets;
    std::vector<uint32_t> route_rows;
    std::vector<float> route_weights;
    if (use_route_aggregation
        && !build_route_aggregation_metadata(requests,
                                             selected,
                                             route_offsets,
                                             route_rows,
                                             route_weights))
    {
        return false;
    }

    VulkanTransferLease transfer_lease = first_gate.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!prepare_staging_batch(transfer_slot.upload,
                               total_rows,
                               input_columns,
                               transfer_slot.staging_allocator,
                               runtime_state,
                               sizeof(float)))
    {
        return false;
    }
    ncnn::Mat mapped_input = transfer_slot.upload.mapped();
    if (mapped_input.empty()
        || mapped_input.dims != 2
        || mapped_input.w != static_cast<int>(input_columns)
        || mapped_input.h != static_cast<int>(total_rows)
        || mapped_input.elemsize != sizeof(float)
        || mapped_input.elempack != 1)
    {
        return false;
    }

    size_t row_offset = 0;
    auto* mapped_input_data = static_cast<std::byte*>(mapped_input.data);
    for (size_t slot = 0; slot < selected.size(); ++slot)
    {
        const ExpertBackendRequest& request = requests[selected[slot].request_index];
        const ActivationBuffer& input = *request.input;
        if (use_indexed)
            expert_row_offsets[slot] = static_cast<uint32_t>(row_offset);
        std::memcpy(mapped_input_data + row_offset * static_cast<size_t>(input_columns) * sizeof(float),
                    input.bytes().data(),
                    input.bytes().size());
        if (use_indexed)
        {
            std::fill(row_slots.begin() + static_cast<std::ptrdiff_t>(row_offset),
                      row_slots.begin() + static_cast<std::ptrdiff_t>(row_offset + input.rows()),
                      static_cast<uint32_t>(slot));
        }
        row_offset += input.rows();
    }
    if (use_indexed)
        expert_row_offsets[selected.size()] = static_cast<uint32_t>(row_offset);
    transfer_slot.upload.allocator->flush(transfer_slot.upload.data);
    transfer_slot.upload.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    transfer_slot.upload.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    const bool direct_host_input = first_gate.vulkan_context->support_direct_host_buffer(total_rows * input_columns * sizeof(float),
                                                                                         DType::Float32);
    const bool direct_host_output = !use_route_aggregation
                                    && first_gate.vulkan_context->support_direct_host_buffer(total_rows * output_columns * sizeof(float),
                                                                                             DType::Float32);
    const size_t download_rows = use_route_aggregation ? aggregated_token_count : total_rows;
    if (!prepare_staging_batch(transfer_slot.download,
                               download_rows,
                               output_columns,
                               transfer_slot.staging_allocator,
                               runtime_state))
        return false;

    if (use_indexed)
    {
        if (!fill_staging_values(row_slots.data(),
                                 row_slots.size(),
                                 sizeof(uint32_t),
                                 transfer_slot.expert_slots,
                                 transfer_slot.staging_allocator,
                                 runtime_state))
            return false;
        if (use_tiled_indexed
            && !fill_staging_values(expert_row_offsets.data(),
                                    expert_row_offsets.size(),
                                    sizeof(uint32_t),
                                    transfer_slot.expert_row_offsets,
                                    transfer_slot.staging_allocator,
                                    runtime_state))
            return false;
    }

    if (use_route_aggregation)
    {
        if (!fill_staging_values(route_offsets.data(),
                                 route_offsets.size(),
                                 sizeof(uint32_t),
                                 transfer_slot.route_offsets,
                                 transfer_slot.staging_allocator,
                                 runtime_state)
            || !fill_staging_values(route_rows.data(),
                                    route_rows.size(),
                                    sizeof(uint32_t),
                                    transfer_slot.route_rows,
                                    transfer_slot.staging_allocator,
                                    runtime_state)
            || !fill_staging_values(route_weights.data(),
                                    route_weights.size(),
                                    sizeof(float),
                                    transfer_slot.route_weights,
                                    transfer_slot.staging_allocator,
                                    runtime_state))
            return false;
    }

    const bool use_direct_output = !use_route_aggregation && selected.size() == 1;
    ActivationBuffer combined_output;
    // A single request already owns a private output until Submission::commit().
    ActivationBuffer& host_output = use_direct_output ? *requests[selected.front().request_index].output : combined_output;
    if (!use_route_aggregation && !use_direct_output)
        host_output.reset(total_rows, output_columns, false);

    std::unique_lock<std::mutex> lock(first_gate.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first_gate.option))
    {
        return false;
    }

    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = prepare_direct_host_output(transfer_slot.download, runtime_state);
    else
        output_gpu.create(static_cast<int>(output_columns),
                          static_cast<int>(total_rows),
                          sizeof(float),
                          first_gate.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    ncnn::VkMat expert_ids_gpu;
    ncnn::VkMat expert_row_offsets_gpu;
    ncnn::VkMat intermediate_gpu;
    std::vector<ncnn::VkMat> intermediates;
    if (use_indexed)
    {
        if (!record_mapped_upload(transfer_slot.expert_slots,
                                  expert_ids_gpu,
                                  command,
                                  first_gate.option))
        {
            return false;
        }
        if (use_tiled_indexed
            && !record_mapped_upload(transfer_slot.expert_row_offsets,
                                     expert_row_offsets_gpu,
                                     command,
                                     first_gate.option))
        {
            return false;
        }

        intermediate_gpu.create(static_cast<int>(first_gate.output_columns / 2),
                                static_cast<int>(total_rows),
                                sizeof(float),
                                first_gate.vulkan_context->blob_allocator());
        if (intermediate_gpu.empty())
            return false;

        std::vector<ncnn::VkMat> bindings(mxfp4_indexed_max_experts + 3);
        std::vector<unsigned char> readonly_bindings(bindings.size(), 1);
        readonly_bindings.back() = 0;
        std::vector<ncnn::vk_constant_type> constants(20);
        constants[0].u32 = first_gate.input_columns;
        constants[1].u32 = first_gate.output_columns;
        constants[2].u32 = first_down.input_columns;
        constants[3].u32 = first_down.output_columns;
        constants[4].u32 = first_gate.block_count;
        constants[5].u32 = first_down.block_count;
        constants[6].u32 = static_cast<uint32_t>(total_rows);
        constants[7].u32 = 0;
        constants[8].u32 = 0;
        constants[9].u32 = first_gate.indexed_scales_word_offset;
        constants[10].u32 = first_gate.indexed_bias_word_offset;
        constants[11].u32 = 0;
        constants[12].u32 = first_down.indexed_scales_word_offset;
        constants[13].u32 = first_down.indexed_bias_word_offset;
        constants[14].u32 = 0;
        constants[15].u32 = first_expert.activation == ExpertActivation::GptOssSwiGlu ? 0u : 1u;
        constants[16].f = first_expert.activation_limit;
        constants[17].u32 = 0;
        constants[18].u32 = 0;
        constants[19].u32 = 0;

        for (size_t chunk_begin = 0;
             chunk_begin < selected.size();
             chunk_begin += mxfp4_indexed_max_experts)
        {
            const size_t chunk_end = std::min(selected.size(),
                                              chunk_begin + static_cast<size_t>(mxfp4_indexed_max_experts));
            const uint32_t chunk_count = static_cast<uint32_t>(chunk_end - chunk_begin);
            const uint32_t row_base = expert_row_offsets[chunk_begin];
            const uint32_t row_end = expert_row_offsets[chunk_end];
            const uint32_t chunk_rows = row_end - row_base;
            uint32_t chunk_maximum_request_rows = 0;
            for (size_t slot = chunk_begin; slot < chunk_end; ++slot)
            {
                const size_t request_index = selected[slot].request_index;
                chunk_maximum_request_rows = std::max(chunk_maximum_request_rows,
                                                      static_cast<uint32_t>(requests[request_index].input->rows()));
            }
            const uint32_t chunk_tile_mode = use_tiled_indexed && chunk_maximum_request_rows >= 2 ? 1u : 0u;

            bindings[0] = input_gpu;
            for (uint32_t slot = 0; slot < mxfp4_indexed_max_experts; ++slot)
            {
                if (slot < chunk_count)
                {
                    bindings[slot + 1] = selected[chunk_begin + slot].entry->operation->d->gate_up->d->storage;
                }
                else
                {
                    bindings[slot + 1].release();
                }
            }
            bindings[mxfp4_indexed_max_experts + 1] = chunk_tile_mode != 0 ? expert_row_offsets_gpu : expert_ids_gpu;
            bindings.back() = intermediate_gpu;
            constants[7].u32 = chunk_count;
            constants[14].u32 = 0;
            constants[17].u32 = chunk_tile_mode;
            constants[18].u32 = row_base;
            constants[19].u32 = static_cast<uint32_t>(chunk_begin);
            ncnn::VkMat gate_dispatcher;
            gate_dispatcher.w = static_cast<int>(first_gate.output_columns / 2 * 32);
            gate_dispatcher.h = static_cast<int>(chunk_tile_mode != 0 ? (chunk_maximum_request_rows + 1) / 2 : chunk_rows);
            gate_dispatcher.c = static_cast<int>(chunk_tile_mode != 0 ? chunk_count : 1);
            command.record_pipeline_readonly(indexed_pipeline.get(),
                                             bindings,
                                             readonly_bindings,
                                             constants,
                                             gate_dispatcher);

            bindings[0] = intermediate_gpu;
            for (uint32_t slot = 0; slot < mxfp4_indexed_max_experts; ++slot)
            {
                if (slot < chunk_count)
                {
                    bindings[slot + 1] = selected[chunk_begin + slot].entry->operation->d->down->d->storage;
                }
                else
                {
                    bindings[slot + 1].release();
                }
            }
            bindings.back() = output_gpu;
            constants[14].u32 = 1;
            ncnn::VkMat down_dispatcher;
            down_dispatcher.w = static_cast<int>(output_columns * 32);
            down_dispatcher.h = static_cast<int>(chunk_tile_mode != 0 ? (chunk_maximum_request_rows + 1) / 2 : chunk_rows);
            down_dispatcher.c = static_cast<int>(chunk_tile_mode != 0 ? chunk_count : 1);
            command.record_pipeline_readonly(indexed_pipeline.get(),
                                             bindings,
                                             readonly_bindings,
                                             constants,
                                             down_dispatcher);
        }
    }
    else
    {
        intermediates.reserve(selected.size());
        row_offset = 0;
        for (const Selection& selection : selected)
        {
            const ExpertBackendRequest& request = requests[selection.request_index];
            const auto& expert = *selection.entry->operation->d;
            const auto& gate = *expert.gate_up->d;
            const uint32_t intermediate_columns = gate.output_columns / 2;
            ncnn::VkMat input_view = row_view(input_gpu, row_offset, request.input->rows());
            ncnn::VkMat output_view = row_view(output_gpu, row_offset, request.input->rows());
            if (input_view.empty() || output_view.empty())
            {
                return false;
            }
            ncnn::VkMat& intermediate = intermediates.emplace_back();
            intermediate.create(static_cast<int>(intermediate_columns),
                                static_cast<int>(request.input->rows()),
                                sizeof(float),
                                first_gate.vulkan_context->blob_allocator());
            if (intermediate.empty())
                return false;
            selection.entry->operation->record(input_view, intermediate, output_view, command);
            row_offset += request.input->rows();
        }
    }

    if (use_route_aggregation)
    {
        ncnn::VkMat route_offsets_gpu;
        ncnn::VkMat route_rows_gpu;
        ncnn::VkMat route_weights_gpu;
        if (!record_mapped_upload(transfer_slot.route_offsets,
                                  route_offsets_gpu,
                                  command,
                                  first_gate.option)
            || !record_mapped_upload(transfer_slot.route_rows,
                                     route_rows_gpu,
                                     command,
                                     first_gate.option)
            || !record_mapped_upload(transfer_slot.route_weights,
                                     route_weights_gpu,
                                     command,
                                     first_gate.option))
        {
            return false;
        }
        if (!route_aggregation_pipeline
            && !create_mxfp4_route_aggregation_pipeline(first_gate.vulkan_context,
                                                        first_gate.option,
                                                        route_aggregation_pipeline))
        {
            return false;
        }
        ncnn::VkMat aggregated_output_gpu;
        aggregated_output_gpu.create(static_cast<int>(output_columns),
                                     static_cast<int>(aggregated_token_count),
                                     sizeof(float),
                                     first_gate.vulkan_context->blob_allocator());
        if (aggregated_output_gpu.empty())
            return false;
        std::vector<ncnn::VkMat> aggregation_bindings = {
            output_gpu,
            route_offsets_gpu,
            route_rows_gpu,
            route_weights_gpu,
            aggregated_output_gpu,
        };
        std::vector<ncnn::vk_constant_type> aggregation_constants(2);
        aggregation_constants[0].u32 = output_columns;
        aggregation_constants[1].u32 = aggregated_token_count;
        ncnn::VkMat aggregation_dispatcher;
        aggregation_dispatcher.w = static_cast<int>(output_columns * 128);
        aggregation_dispatcher.h = static_cast<int>(aggregated_token_count);
        aggregation_dispatcher.c = 1;
        command.record_pipeline(route_aggregation_pipeline.get(),
                                aggregation_bindings,
                                aggregation_constants,
                                aggregation_dispatcher);
        if (!record_prepared_activation_staging_download(aggregated_output_gpu,
                                                         aggregated_token_count,
                                                         output_columns,
                                                         transfer_slot.download,
                                                         command,
                                                         first_down.vulkan_context->device(),
                                                         first_down.option,
                                                         aggregated_output->dtype())
            || submit_compute_and_wait(command, runtime_state) != 0
            || !copy_staging_to_cpu_batch(transfer_slot.download, *aggregated_output))
        {
            return false;
        }
    }
    else
    {
        if (!direct_host_output
            && !record_prepared_activation_staging_download(output_gpu,
                                                            total_rows,
                                                            output_columns,
                                                            transfer_slot.download,
                                                            command,
                                                            first_down.vulkan_context->device(),
                                                            first_down.option,
                                                            host_output.dtype()))
        {
            return false;
        }
        if (submit_compute_and_wait(command, runtime_state) != 0)
            return false;
        if (use_direct_output)
            host_output.reset(total_rows, output_columns, false);
        if (!copy_staging_to_cpu_batch(transfer_slot.download, host_output))
            return false;
    }
    lock.unlock();

    if (use_route_aggregation)
    {
        for (const Selection& selection : selected)
        {
            const ExpertBackendRequest& request = requests[selection.request_index];
            if (request.route_aggregation.completed)
                *request.route_aggregation.completed = 1;
        }
    }
    else if (!use_direct_output)
    {
        row_offset = 0;
        for (const Selection& selection : selected)
        {
            const ExpertBackendRequest& request = requests[selection.request_index];
            request.output->reset(request.input->rows(), output_columns, false);
            for (size_t row = 0; row < request.input->rows(); ++row)
            {
                std::copy_n(combined_output.row(row_offset + row),
                            output_columns,
                            request.output->row(row));
            }
            row_offset += request.input->rows();
        }
    }
    const uint64_t projection_dispatches = use_indexed
                                               ? static_cast<uint64_t>(indexed_chunk_count) * 2
                                               : static_cast<uint64_t>(selected.size()) * 2;
    runtime_state.dispatches += projection_dispatches + static_cast<uint64_t>(use_route_aggregation);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
}
#endif

} // namespace moe
} // namespace ncnn
