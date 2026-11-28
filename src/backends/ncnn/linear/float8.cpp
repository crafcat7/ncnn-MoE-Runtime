#include "backends/ncnn/core/context.h"
#include "backends/ncnn/core/tensor.h"
#include "backends/ncnn/layer/rmsnorm_vulkan.h"
#include "backends/ncnn/linear/bf16.h"
#include "backends/ncnn/linear/float8.h"
#include "backends/ncnn/linear/internal.h"
#include "kernels/bfloat16.h"
#include "kernels/float8.h"
#include "kernels/vulkan/float8_projection.comp.hex.h"
#include "kernels/vulkan/float8_quantize.comp.hex.h"
#include "kernels/vulkan/float8_rms_norm_quantize.comp.hex.h"
#include "kernels/vulkan/float8_swiglu_quantize.comp.hex.h"

#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>
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

// --- Internal Vulkan pipeline, staging, and dispatch helpers -----------------

static bool create_float8_pipeline(const std::shared_ptr<VulkanContext>& context,
                                   const ncnn::Option& option,
                                   const char* src, int len,
                                   std::shared_ptr<ncnn::Pipeline>& destination)
{
    std::vector<ncnn::vk_specialization_type> specs(1);
    specs[0].u32 = 32;
    return create_pipeline(context, src, len, option, 0, 0, 32, destination,
                           specs, 0, 1, 1, true);
}

static size_t align_float8_upload(size_t bytes) noexcept
{
    if (bytes > std::numeric_limits<size_t>::max() - 3)
        return 0;
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

static bool copy_tensor_floats(const TensorData& source, uint32_t expected_count, float* destination)
{
    if (source.dtype == DType::Float32)
    {
        const std::span<const float> values = source.float32_values();
        if (values.size() != expected_count)
            return false;
        std::copy(values.begin(), values.end(), destination);
        return true;
    }
    if (source.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> values = source.bfloat16_values();
        if (values.size() != expected_count)
            return false;
        for (uint32_t index = 0; index < expected_count; ++index)
            destination[index] = bfloat16_to_float(values[index]);
        return true;
    }
    return false;
}

static bool fill_float8_quantized_staging(const ActivationBuffer& input,
                                          uint32_t columns,
                                          uint64_t optimization_flags,
                                          ncnn::VkMat& staging,
                                          ncnn::VkAllocator* allocator)
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
    if (!prepare_staging_batch(staging, input.rows(), columns, allocator, sizeof(float)))
        return false;

    ncnn::Mat mapped = staging.mapped();
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
    (void)bind_host_in(staging);
    return true;
}

static bool float8_linear_scratch_shape(uint32_t input_columns, uint64_t rows,
                                        uint32_t& scratch_columns) noexcept
{
    scratch_columns = input_columns;
    return input_columns != 0 && rows != 0
           && saturating_multiply_u64(input_columns, rows) != UINT64_MAX;
}

static size_t fp8_swiglu_shapes(uint32_t input_columns, uint32_t intermediate_columns,
                                uint64_t rows, bool device_input,
                                uint64_t columns[3]) noexcept
{
    if (intermediate_columns == 0 || rows == 0)
        return 0;
    if (device_input)
    {
        uint32_t quantized_columns = 0;
        if (!float8_linear_scratch_shape(input_columns, rows, quantized_columns))
            return 0;
        columns[0] = quantized_columns;
        columns[1] = intermediate_columns;
        columns[2] = intermediate_columns;
        return 3;
    }
    columns[0] = intermediate_columns;
    columns[1] = intermediate_columns;
    return 2;
}

static bool reset_slot_command(VkTransferSlot& slot)
{
    if (slot.command_used && slot.command->reset() != 0)
        return false;
    slot.command_used = true;
    return true;
}

static bool prepare_output_buffer(const VulkanContext& context,
                                  uint32_t columns,
                                  size_t rows,
                                  size_t elem_size,
                                  bool direct_host_output,
                                  ncnn::VkMat& download,
                                  ncnn::VkMat& output_gpu)
{
    if (direct_host_output)
        output_gpu = bind_host_out(download);
    else
        output_gpu.create(static_cast<int>(columns), static_cast<int>(rows),
                          elem_size, context.blob_allocator());
    return !output_gpu.empty();
}

static bool record_optional_download(const ncnn::VkMat& output_gpu,
                                     size_t rows,
                                     uint32_t columns,
                                     ncnn::VkMat& download,
                                     ncnn::VkCompute& command,
                                     const VulkanContext& context,
                                     const ncnn::Option& option,
                                     DType output_dtype,
                                     bool direct_host_output)
{
    return direct_host_output
           || record_staging_download(output_gpu, rows, columns, download, command,
                                      context.device(), option, output_dtype);
}

static bool prepare_host_io(const VulkanContext& context,
                            const ActivationBuffer& input,
                            uint32_t input_columns,
                            uint64_t optimization_flags,
                            bool quantize_input,
                            uint32_t output_columns,
                            bool retain_output,
                            VkTransferSlot& slot,
                            bool& direct_host_output,
                            DType& output_dtype,
                            ActivationBuffer& output)
{
    const size_t output_bytes = static_cast<size_t>(input.rows()) * output_columns * sizeof(float);
    direct_host_output = !retain_output && context.support_direct_host_buffer(output_bytes, output.dtype());
    const bool upload_ready = quantize_input
                                  ? fill_float8_quantized_staging(input,
                                                                  input_columns,
                                                                  optimization_flags,
                                                                  slot.upload,
                                                                  slot.staging_allocator)
                                  : fill_staging_upload(input, slot.upload, slot.staging_allocator);
    if (!upload_ready
        || (!retain_output
            && !prepare_staging_batch(slot.download,
                                      input.rows(),
                                      output_columns,
                                      slot.staging_allocator)))
    {
        return false;
    }
    output_dtype = output.dtype();
    output.reset(input.rows(), output_columns, false);
    return true;
}

static void record_float8_quantize(const ncnn::Pipeline* pipeline,
                                   const ncnn::VkMat& data,
                                   uint32_t columns,
                                   uint32_t rows,
                                   ncnn::VkCompute& command)
{
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].u32 = columns;
    constants[1].u32 = rows;
    ncnn::VkMat dispatch;
    dispatch.w = static_cast<int>((static_cast<uint64_t>(columns) / 128)
                                  * pipeline->local_size_x());
    dispatch.h = static_cast<int>(rows);
    dispatch.c = 1;
    command.record_pipeline(pipeline, {data}, constants, dispatch);
}

static void record_float8_rms_norm_quantize(const ncnn::Pipeline* pipeline,
                                            const ncnn::VkMat& data,
                                            const ncnn::VkMat& weight,
                                            uint32_t columns,
                                            uint32_t rows,
                                            float epsilon,
                                            ncnn::VkCompute& command)
{
    std::vector<ncnn::vk_constant_type> constants(3);
    constants[0].u32 = columns;
    constants[1].u32 = rows;
    constants[2].f = epsilon;
    ncnn::VkMat dispatch;
    dispatch.w = static_cast<int>(pipeline->local_size_x());
    dispatch.h = static_cast<int>(rows);
    dispatch.c = 1;
    command.record_pipeline_readonly(pipeline, {data, weight}, {0, 1}, constants, dispatch);
}

static void record_float8_swiglu_quantize(const ncnn::Pipeline* pipeline,
                                          const ncnn::VkMat& gate,
                                          const ncnn::VkMat& up,
                                          uint32_t columns,
                                          uint32_t rows,
                                          float activation_limit,
                                          ncnn::VkCompute& command)
{
    std::vector<ncnn::vk_constant_type> constants(3);
    constants[0].u32 = columns;
    constants[1].u32 = rows;
    constants[2].f = activation_limit;
    ncnn::VkMat dispatch;
    dispatch.w = static_cast<int>((static_cast<uint64_t>(columns) / 128)
                                  * pipeline->local_size_x());
    dispatch.h = static_cast<int>(rows);
    dispatch.c = 1;
    command.record_pipeline(pipeline, {gate, up}, constants, dispatch);
}

struct Float8ProjectionParams
{
    const ncnn::Pipeline* pipeline = nullptr;
    const ncnn::VkMat* packed = nullptr;
    const ncnn::VkMat* scales = nullptr;
    const ncnn::VkMat* bias = nullptr;
    uint32_t matrix_input_columns = 0;
    uint32_t logical_input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t output_columns_per_group = 0;
    uint32_t block_count = 0;
    uint32_t scale_row_size = 128;
};

static void record_float8_projection(const Float8ProjectionParams& params,
                                     const ncnn::VkMat& input,
                                     const ncnn::VkMat& output,
                                     uint32_t token_count,
                                     ncnn::VkCompute& command,
                                     uint32_t output_stride = 0,
                                     uint32_t output_offset = 0)
{
    if (output_stride == 0)
        output_stride = params.output_columns;
    std::vector<ncnn::VkMat> bindings = {
        input,
        *params.packed,
        *params.scales,
        *params.bias,
        output};
    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].u32 = params.matrix_input_columns;
    constants[1].u32 = params.logical_input_columns;
    constants[2].u32 = params.output_columns;
    constants[3].u32 = params.output_columns_per_group;
    constants[4].u32 = params.block_count;
    constants[5].u32 = token_count;
    constants[6].u32 = params.scale_row_size;
    constants[7].u32 = output_stride;
    constants[8].u32 = output_offset;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(static_cast<uint64_t>(params.output_columns)
                                    * params.pipeline->local_size_x());
    dispatcher.h = static_cast<int>(token_count);
    dispatcher.c = 1;
    command.record_pipeline_readonly(params.pipeline,
                                     bindings,
                                     {1, 0, 0, 0, 0},
                                     constants,
                                     dispatcher);
}

static bool forward_float8_chain(const VulkanContext& context,
                                 const ncnn::Option& option,
                                 uint64_t optimization_flags,
                                 const Float8ProjectionParams& first,
                                 const Float8ProjectionParams& second,
                                 const ncnn::Pipeline* quantize_pipeline,
                                 const ncnn::Pipeline* rms_norm_quantize_pipeline,
                                 const ncnn::VkMat& rms_norm_weight,
                                 float rms_norm_epsilon,
                                 const ActivationBuffer& input,
                                 ActivationBuffer& output,
                                 bool use_norm)
{
    if (use_norm)
    {
        if (!rms_norm_quantize_pipeline || rms_norm_weight.empty())
            return false;
    }

    VulkanContext& mutable_context = const_cast<VulkanContext&>(context);
    VulkanRuntimeState& runtime_state = mutable_context.runtime_state();
    VkTransferLease transfer_lease = mutable_context.acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    bool direct_host_output = false;
    DType output_dtype = DType::Float32;
    if (!prepare_host_io(context,
                         input,
                         first.logical_input_columns,
                         optimization_flags,
                         true,
                         second.output_columns,
                         false,
                         transfer_slot,
                         direct_host_output,
                         output_dtype,
                         output))
    {
        return false;
    }

    const std::lock_guard<std::mutex> lock(mutable_context.command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (!reset_slot_command(transfer_slot))
        return false;

    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, option))
        return false;

    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(first.output_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            context.blob_allocator());
    ncnn::VkMat output_gpu;
    if (intermediate_gpu.empty()
        || !prepare_output_buffer(context,
                                  second.output_columns,
                                  input.rows(),
                                  sizeof(float),
                                  direct_host_output,
                                  transfer_slot.download,
                                  output_gpu))
    {
        return false;
    }

    const uint32_t rows = static_cast<uint32_t>(input.rows());
    record_float8_projection(first, input_gpu, intermediate_gpu, rows, command);
    if (use_norm)
    {
        record_float8_rms_norm_quantize(rms_norm_quantize_pipeline,
                                        intermediate_gpu,
                                        rms_norm_weight,
                                        first.output_columns,
                                        rows,
                                        rms_norm_epsilon,
                                        command);
    }
    else
    {
        record_float8_quantize(quantize_pipeline,
                               intermediate_gpu,
                               first.output_columns,
                               rows,
                               command);
    }
    record_float8_projection(second, intermediate_gpu, output_gpu, rows, command);

    if (!record_optional_download(output_gpu,
                                  input.rows(),
                                  second.output_columns,
                                  transfer_slot.download,
                                  command,
                                  context,
                                  option,
                                  output_dtype,
                                  direct_host_output)
        || submit_wait(command, context.device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }

    runtime_state.dispatches += 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
}

// --- Float8Linear_vulkan::Implementation -------------------------------------

void Float8Linear_vulkan::Implementation::record_projection(const ncnn::VkMat& input,
                                                            const ncnn::VkMat& output,
                                                            uint32_t token_count,
                                                            ncnn::VkCompute& command,
                                                            uint32_t output_stride,
                                                            uint32_t output_offset) const
{
    const Float8ProjectionParams params{
        pipeline.get(),
        &packed,
        &scales,
        &bias,
        matrix_input_columns,
        logical_input_columns,
        output_columns,
        output_columns_per_group,
        block_count,
        scale_row_size};
    record_float8_projection(params, input, output, token_count, command,
                             output_stride, output_offset);
}

// --- Float8Linear_vulkan -----------------------------------------------------

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
    if (matrix.dtype != DType::Float8E4M3 || matrix.shape.size() != 2
        || matrix.shape[0] == 0 || matrix.shape[1] == 0 || matrix.shape[1] % 128 != 0
        || matrix.float8_scale_row_size == 0 || input_group_count == 0
        || matrix.shape[0] % input_group_count != 0)
    {
        return {};
    }

    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t block_count = input_columns / 128;
    const uint32_t output_block_count = output_columns / matrix.float8_scale_row_size
                                        + (output_columns % matrix.float8_scale_row_size != 0);
    const std::span<const uint8_t> weights = matrix.float8_values();
    if (weights.size() != matrix.element_count()
        || matrix.quantization_scales.size() != static_cast<size_t>(output_block_count) * block_count)
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
    implementation.scale_row_size = matrix.float8_scale_row_size;
    implementation.optimization_flags = optimization_flags;
    implementation.vulkan_context = VulkanContext::acquire(vulkan_device_index,
                                                           vulkan_runtime,
                                                           optimization_flags);
    if (!implementation.vulkan_context)
        return {};

    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    implementation.option = make_vk_opt(*implementation.vulkan_context);
    implementation.option.use_cooperative_matrix = device->info.support_cooperative_matrix();

    const size_t aligned_weights = align_float8_upload(weights.size());
    if (aligned_weights == 0)
        return {};
    const uint64_t preferred_weight_size = static_cast<uint64_t>(aligned_weights)
                                           + static_cast<uint64_t>(matrix.quantization_scales.size()) * sizeof(float)
                                           + static_cast<uint64_t>(output_columns) * sizeof(float);
    if (preferred_weight_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return {};

    ncnn::Mat packed;
    ncnn::Mat scales;
    ncnn::Mat biases;
    if (!prepare_float8_upload(weights, packed))
        return {};
    scales.create(static_cast<int>(matrix.quantization_scales.size()), sizeof(float));
    biases.create(static_cast<int>(output_columns), sizeof(float));
    if (scales.empty() || biases.empty())
        return {};
    std::copy(matrix.quantization_scales.begin(), matrix.quantization_scales.end(),
              static_cast<float*>(scales.data));

    float* bias_values = static_cast<float*>(biases.data);
    if (!bias)
        std::fill_n(bias_values, output_columns, 0.0f);
    else if (!copy_tensor_floats(*bias, output_columns, bias_values))
        return {};

    if (!create_float8_pipeline(implementation.vulkan_context, implementation.option,
                                float8_projection_shader,
                                static_cast<int>(sizeof(float8_projection_shader) - 1),
                                implementation.pipeline)
        || !create_float8_pipeline(implementation.vulkan_context, implementation.option,
                                   float8_quantize_shader,
                                   static_cast<int>(sizeof(float8_quantize_shader) - 1),
                                   implementation.quantize_pipeline)
        || !create_float8_pipeline(implementation.vulkan_context, implementation.option,
                                   float8_swiglu_quantize_shader,
                                   static_cast<int>(sizeof(float8_swiglu_quantize_shader) - 1),
                                   implementation.swiglu_quantize_pipeline))
    {
        return {};
    }

    const uint64_t projection_local_size = implementation.pipeline->local_size_x();
    const uint64_t quantize_local_size = std::max(implementation.quantize_pipeline->local_size_x(),
                                                  implementation.swiglu_quantize_pipeline->local_size_x());
    const uint64_t max_dispatch_width = static_cast<uint64_t>(std::numeric_limits<int>::max());
    const uint64_t max_quantize_blocks = std::max(implementation.logical_input_columns / 128,
                                                  implementation.output_columns / 128);
    if (projection_local_size == 0 || quantize_local_size == 0
        || projection_local_size > max_dispatch_width || quantize_local_size > max_dispatch_width
        || static_cast<uint64_t>(implementation.output_columns)
               > (max_dispatch_width - (projection_local_size - 1)) / projection_local_size
        || max_quantize_blocks > (max_dispatch_width - (quantize_local_size - 1)) / quantize_local_size)
    {
        return {};
    }

    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));
    ncnn::Option upload_option = implementation.option;
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(device);
        command.record_upload(packed, implementation.packed, upload_option);
        command.record_upload(scales, implementation.scales, upload_option);
        command.record_upload(biases, implementation.bias, upload_option);
        uploaded = !implementation.packed.empty()
                   && !implementation.scales.empty()
                   && !implementation.bias.empty()
                   && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return {};
    return result;
}

bool Float8Linear_vulkan::prepare_rms_norm_weight(const TensorData& weight,
                                                  uint32_t expected_columns,
                                                  float epsilon,
                                                  ncnn::VkMat& destination)
{
    (void)epsilon;
    Implementation& implementation = *d;
    ncnn::Mat values;
    values.create(static_cast<int>(expected_columns), sizeof(float));
    if (values.empty()
        || !copy_tensor_floats(weight, expected_columns, static_cast<float*>(values.data)))
    {
        return false;
    }

    if (!create_float8_pipeline(implementation.vulkan_context,
                                implementation.option,
                                float8_rms_norm_quantize_shader,
                                static_cast<int>(sizeof(float8_rms_norm_quantize_shader) - 1),
                                implementation.rms_norm_quantize_pipeline))
    {
        return false;
    }

    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));
    ncnn::Option upload_option = implementation.option;
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(device);
        command.record_upload(values, destination, upload_option);
        uploaded = !destination.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    return uploaded;
}

bool Float8Linear_vulkan::prepare_rms_norm(const TensorData& weight, float epsilon)
{
    Implementation& implementation = *d;
    if (!prepare_rms_norm_weight(weight,
                                 implementation.output_columns,
                                 epsilon,
                                 implementation.rms_norm_weight))
    {
        return false;
    }
    implementation.rms_norm = RmsNorm_vulkan::create(weight, epsilon, 0.0f,
                                                     implementation.vulkan_context,
                                                     implementation.option);
    if (!implementation.rms_norm)
        return false;
    implementation.rms_norm_epsilon = epsilon;
    return true;
}

bool Float8Linear_vulkan::prepare_input_rms_norm(const TensorData& weight, float epsilon)
{
    Implementation& implementation = *d;
    if (!prepare_rms_norm_weight(weight,
                                 implementation.logical_input_columns,
                                 epsilon,
                                 implementation.input_rms_norm_weight))
    {
        return false;
    }
    implementation.input_rms_norm = RmsNorm_vulkan::create(weight, epsilon, 0.0f,
                                                           implementation.vulkan_context,
                                                           implementation.option);
    if (!implementation.input_rms_norm)
        return false;
    implementation.input_rms_norm_epsilon = epsilon;
    return true;
}

std::vector<uint64_t> Float8Linear_vulkan::scratch_elements(uint32_t input_columns, uint64_t rows)
{
    uint32_t scratch_columns = 0;
    if (!float8_linear_scratch_shape(input_columns, rows, scratch_columns))
        return input_columns == 0 || rows == 0 ? std::vector<uint64_t>{}
                                               : std::vector<uint64_t>{UINT64_MAX};
    return {saturating_multiply_u64(scratch_columns, rows)};
}

std::vector<uint64_t> Float8Linear_vulkan::swiglu_scratch_elements(uint32_t input_columns,
                                                                   uint32_t intermediate_columns,
                                                                   uint64_t rows,
                                                                   bool device_input)
{
    if (input_columns == 0 || intermediate_columns == 0 || rows == 0)
        return {};
    uint64_t columns[3]{};
    const size_t count = fp8_swiglu_shapes(input_columns, intermediate_columns,
                                           rows, device_input, columns);
    if (count != (device_input ? 3u : 2u))
        return {UINT64_MAX};
    std::vector<uint64_t> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index)
    {
        const uint64_t elements = saturating_multiply_u64(columns[index], rows);
        if (elements == UINT64_MAX)
            return {UINT64_MAX};
        if (elements != 0)
            result.push_back(elements);
    }
    return result;
}

const std::shared_ptr<VulkanContext>& Float8Linear_vulkan::context() const noexcept
{
    return d->vulkan_context;
}

const ncnn::Option& Float8Linear_vulkan::option() const noexcept
{
    return d->option;
}

uint32_t Float8Linear_vulkan::input_columns() const noexcept
{
    return d->logical_input_columns;
}

uint32_t Float8Linear_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

bool Float8Linear_vulkan::record_input_normalization(const ncnn::VkMat& input,
                                                     ncnn::VkMat& output,
                                                     ncnn::VkCompute& cmd) const
{
    return d->input_rms_norm && d->input_rms_norm->record(input, output, cmd);
}

bool Float8Linear_vulkan::record_rms_norm_chain_parallel(const ncnn::VkMat& input,
                                                         const Float8Linear_vulkan& next,
                                                         const Float8Linear_vulkan& parallel,
                                                         ncnn::VkMat& query,
                                                         ncnn::VkMat& kv,
                                                         ncnn::VkMat& query_rank,
                                                         ncnn::VkCompute& cmd,
                                                         std::vector<ncnn::VkMat>& workspace) const
{
    if (!d->rms_norm || d->vulkan_context != next.d->vulkan_context
        || d->vulkan_context != parallel.d->vulkan_context)
    {
        return false;
    }
    ncnn::VkMat intermediate;
    if (!record_forward(input, intermediate, cmd, workspace)
        || !d->rms_norm->record(intermediate, query_rank, cmd)
        || !next.record_forward(query_rank, query, cmd, workspace)
        || !parallel.record_forward(input, kv, cmd, workspace))
    {
        return false;
    }
    workspace.push_back(std::move(intermediate));
    return true;
}

bool Float8Linear_vulkan::record_forward(const ncnn::VkMat& input,
                                         ncnn::VkMat& output,
                                         ncnn::VkCompute& command,
                                         std::vector<ncnn::VkMat>& workspace,
                                         uint32_t output_column_offset,
                                         uint32_t output_stride) const
{
    uint32_t scratch_columns = 0;
    const uint64_t rows = input.h > 0 ? static_cast<uint64_t>(input.h) : 0;
    if (!float8_linear_scratch_shape(d->logical_input_columns, rows, scratch_columns))
    {
        return false;
    }
    if (output_stride == 0)
        output_stride = d->output_columns;

    ncnn::VkMat quantized;
    command.record_clone(input, quantized, d->option);
    if (output_column_offset == 0 && output_stride == d->output_columns)
    {
        output.create(static_cast<int>(d->output_columns), input.h, sizeof(float),
                      d->vulkan_context->blob_allocator());
    }
    if (quantized.empty() || output.empty())
        return false;

    const uint32_t row_count = static_cast<uint32_t>(rows);
    record_float8_quantize(d->quantize_pipeline.get(), quantized, scratch_columns, row_count, command);
    d->record_projection(quantized,
                         output,
                         row_count,
                         command,
                         output_stride,
                         output_column_offset);
    workspace.push_back(std::move(quantized));
    return true;
}

bool Float8Linear_vulkan::materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output) const
{
    return input.materialize(output);
}

bool Float8Linear_vulkan::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
    const Implementation& implementation = *d;
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();

    const size_t input_bytes = static_cast<size_t>(input.rows()) * implementation.logical_input_columns * sizeof(float);
    const bool direct_host_input = implementation.vulkan_context->support_direct_host_buffer(input_bytes, input.dtype());
    bool direct_host_output = false;
    DType output_dtype = DType::Float32;
    if (!prepare_host_io(*implementation.vulkan_context,
                         input,
                         implementation.logical_input_columns,
                         implementation.optimization_flags,
                         true,
                         implementation.output_columns,
                         false,
                         transfer_slot,
                         direct_host_output,
                         output_dtype,
                         output))
    {
        return false;
    }

    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (!reset_slot_command(transfer_slot))
        return false;

    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_host_in(transfer_slot.upload);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
        return false;

    ncnn::VkMat output_gpu;
    if (!prepare_output_buffer(*implementation.vulkan_context,
                               implementation.output_columns,
                               input.rows(),
                               sizeof(float),
                               direct_host_output,
                               transfer_slot.download,
                               output_gpu))
    {
        return false;
    }

    implementation.record_projection(input_gpu,
                                     output_gpu,
                                     static_cast<uint32_t>(input.rows()),
                                     command);
    if (!record_optional_download(output_gpu,
                                  input.rows(),
                                  implementation.output_columns,
                                  transfer_slot.download,
                                  command,
                                  *implementation.vulkan_context,
                                  implementation.option,
                                  output_dtype,
                                  direct_host_output)
        || submit_wait(command, implementation.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }

    ++runtime_state.dispatches;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
}

bool Float8Linear_vulkan::forward_chain(const ActivationBuffer& input,
                                        const Float8Linear_vulkan& next,
                                        ActivationBuffer& output) const
{
    const Implementation& first = *d;
    const Implementation& second = *next.d;
    const Float8ProjectionParams first_params{
        first.pipeline.get(),
        &first.packed,
        &first.scales,
        &first.bias,
        first.matrix_input_columns,
        first.logical_input_columns,
        first.output_columns,
        first.output_columns_per_group,
        first.block_count,
        first.scale_row_size};
    const Float8ProjectionParams second_params{
        second.pipeline.get(),
        &second.packed,
        &second.scales,
        &second.bias,
        second.matrix_input_columns,
        second.logical_input_columns,
        second.output_columns,
        second.output_columns_per_group,
        second.block_count,
        second.scale_row_size};
    return forward_float8_chain(*first.vulkan_context,
                                first.option,
                                first.optimization_flags,
                                first_params,
                                second_params,
                                first.quantize_pipeline.get(),
                                first.rms_norm_quantize_pipeline.get(),
                                first.rms_norm_weight,
                                first.rms_norm_epsilon,
                                input,
                                output,
                                false);
}

bool Float8Linear_vulkan::forward_rms_norm_chain(const ActivationBuffer& input,
                                                 const Float8Linear_vulkan& next,
                                                 ActivationBuffer& output) const
{
    const Implementation& first = *d;
    const Implementation& second = *next.d;
    const Float8ProjectionParams first_params{
        first.pipeline.get(),
        &first.packed,
        &first.scales,
        &first.bias,
        first.matrix_input_columns,
        first.logical_input_columns,
        first.output_columns,
        first.output_columns_per_group,
        first.block_count,
        first.scale_row_size};
    const Float8ProjectionParams second_params{
        second.pipeline.get(),
        &second.packed,
        &second.scales,
        &second.bias,
        second.matrix_input_columns,
        second.logical_input_columns,
        second.output_columns,
        second.output_columns_per_group,
        second.block_count,
        second.scale_row_size};
    return forward_float8_chain(*first.vulkan_context,
                                first.option,
                                first.optimization_flags,
                                first_params,
                                second_params,
                                first.quantize_pipeline.get(),
                                first.rms_norm_quantize_pipeline.get(),
                                first.rms_norm_weight,
                                first.rms_norm_epsilon,
                                input,
                                output,
                                true);
}

bool Float8Linear_vulkan::forward_rms_norm_chain_parallel_impl(const ActivationBuffer& input,
                                                               const Float8Linear_vulkan& next,
                                                               const Float8Linear_vulkan& parallel_operator,
                                                               ActivationBuffer& output,
                                                               ActivationBuffer& parallel_output,
                                                               bool normalize_input,
                                                               DeviceTensor_vulkan* retained_output) const
{
    const Implementation& first = *d;
    const Implementation& second = *next.d;
    const Implementation& parallel = *parallel_operator.d;
    if (normalize_input
        && !has_flag(first.optimization_flags, OptimizationVulkanLatentInputRmsNorm))
    {
        return false;
    }
    if (!first.rms_norm_quantize_pipeline || first.rms_norm_weight.empty()
        || (normalize_input && first.input_rms_norm_weight.empty()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    bool direct_host_output = false;
    DType output_dtype = DType::Float32;
    const size_t parallel_bytes = static_cast<size_t>(input.rows()) * parallel.output_columns * sizeof(float);
    const bool parallel_direct_host_output = parallel.vulkan_context->support_direct_host_buffer(parallel_bytes, parallel_output.dtype());

    ncnn::VkMat parallel_download;
    if (!prepare_host_io(*first.vulkan_context,
                         input,
                         first.logical_input_columns,
                         first.optimization_flags,
                         !normalize_input,
                         second.output_columns,
                         retained_output != nullptr,
                         transfer_slot,
                         direct_host_output,
                         output_dtype,
                         output)
        || !prepare_staging_batch(parallel_download,
                                  input.rows(),
                                  parallel.output_columns,
                                  transfer_slot.staging_allocator))
    {
        return false;
    }

    const DType parallel_output_dtype = parallel_output.dtype();
    parallel_output.reset(input.rows(), parallel.output_columns, false);

    const std::lock_guard<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (!reset_slot_command(transfer_slot))
        return false;

    ncnn::VkMat input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first.option))
        return false;

    const uint32_t rows = static_cast<uint32_t>(input.rows());
    if (normalize_input)
    {
        record_float8_rms_norm_quantize(first.rms_norm_quantize_pipeline.get(),
                                        input_gpu,
                                        first.input_rms_norm_weight,
                                        first.logical_input_columns,
                                        rows,
                                        first.input_rms_norm_epsilon,
                                        command);
    }

    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(first.output_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    ncnn::VkMat parallel_gpu;
    if (intermediate_gpu.empty()
        || !prepare_output_buffer(*first.vulkan_context,
                                  second.output_columns,
                                  input.rows(),
                                  sizeof(float),
                                  direct_host_output,
                                  transfer_slot.download,
                                  output_gpu)
        || !prepare_output_buffer(*first.vulkan_context,
                                  parallel.output_columns,
                                  input.rows(),
                                  sizeof(float),
                                  parallel_direct_host_output,
                                  parallel_download,
                                  parallel_gpu))
    {
        return false;
    }

    first.record_projection(input_gpu, intermediate_gpu, rows, command);
    record_float8_rms_norm_quantize(first.rms_norm_quantize_pipeline.get(),
                                    intermediate_gpu,
                                    first.rms_norm_weight,
                                    first.output_columns,
                                    rows,
                                    first.rms_norm_epsilon,
                                    command);
    second.record_projection(intermediate_gpu, output_gpu, rows, command);
    parallel.record_projection(input_gpu, parallel_gpu, rows, command);

    if ((!retained_output
         && !record_optional_download(output_gpu,
                                      input.rows(),
                                      second.output_columns,
                                      transfer_slot.download,
                                      command,
                                      *first.vulkan_context,
                                      first.option,
                                      output_dtype,
                                      direct_host_output))
        || !record_optional_download(parallel_gpu,
                                     input.rows(),
                                     parallel.output_columns,
                                     parallel_download,
                                     command,
                                     *first.vulkan_context,
                                     first.option,
                                     parallel_output_dtype,
                                     parallel_direct_host_output)
        || submit_wait(command, first.vulkan_context->device()) != 0
        || (!retained_output && !copy_staging_to_cpu_batch(transfer_slot.download, output))
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }

    if (retained_output)
    {
        if (!retained_output->assign_completed(output_gpu, first.vulkan_context))
            return false;
        output.clear();
    }

    runtime_state.dispatches += 3;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads += retained_output ? 1 : 2;
    if (normalize_input)
        ++runtime_state.rms_norm_linear_fusions;
    return true;
}

bool Float8Linear_vulkan::forward_rms_norm_chain_parallel(const ActivationBuffer& input,
                                                          const Float8Linear_vulkan& next,
                                                          const Float8Linear_vulkan& parallel_operator,
                                                          ActivationBuffer& output,
                                                          ActivationBuffer& parallel_output,
                                                          DeviceTensor_vulkan* retained_output) const
{
    return forward_rms_norm_chain_parallel_impl(input,
                                                next,
                                                parallel_operator,
                                                output,
                                                parallel_output,
                                                false,
                                                retained_output);
}

bool Float8Linear_vulkan::forward_input_rms_norm_chain_parallel(const ActivationBuffer& input,
                                                                const Float8Linear_vulkan& next,
                                                                const Float8Linear_vulkan& parallel_operator,
                                                                ActivationBuffer& output,
                                                                ActivationBuffer& parallel_output,
                                                                DeviceTensor_vulkan* retained_output) const
{
    return forward_rms_norm_chain_parallel_impl(input,
                                                next,
                                                parallel_operator,
                                                output,
                                                parallel_output,
                                                true,
                                                retained_output);
}

bool Float8Linear_vulkan::forward_rms_norm_chain_parallel_bfloat16(const ActivationBuffer& input,
                                                                   const Float8Linear_vulkan& next,
                                                                   const Float8Linear_vulkan& parallel_operator,
                                                                   std::span<const Bfloat16Linear_vulkan*> extra_operators,
                                                                   std::span<ActivationBuffer*> extra_outputs,
                                                                   ActivationBuffer& output,
                                                                   ActivationBuffer& parallel_output,
                                                                   DeviceTensor_vulkan* retained_output) const
{
    if (extra_operators.empty())
    {
        return forward_rms_norm_chain_parallel(input,
                                               next,
                                               parallel_operator,
                                               output,
                                               parallel_output,
                                               retained_output);
    }
    if (extra_operators.size() != extra_outputs.size())
        return false;

    const Implementation& first = *d;
    const Implementation& second = *next.d;
    const Implementation& parallel = *parallel_operator.d;
    if (!first.rms_norm_quantize_pipeline || first.rms_norm_weight.empty())
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    bool direct_host_output = false;
    DType output_dtype = DType::Float32;
    const size_t parallel_bytes = static_cast<size_t>(input.rows()) * parallel.output_columns * sizeof(float);
    const bool parallel_direct_host_output = parallel.vulkan_context->support_direct_host_buffer(parallel_bytes, parallel_output.dtype());

    ncnn::VkMat parallel_download;
    std::vector<ncnn::VkMat> extra_downloads(extra_operators.size());
    std::vector<bool> extra_direct_host_outputs(extra_operators.size(), false);
    if (!prepare_host_io(*first.vulkan_context,
                         input,
                         first.logical_input_columns,
                         first.optimization_flags,
                         false,
                         second.output_columns,
                         retained_output != nullptr,
                         transfer_slot,
                         direct_host_output,
                         output_dtype,
                         output)
        || !prepare_staging_batch(parallel_download,
                                  input.rows(),
                                  parallel.output_columns,
                                  transfer_slot.staging_allocator))
    {
        return false;
    }

    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const Bfloat16Linear_vulkan::Implementation& extra = *extra_operators[index]->d;
        if (!prepare_staging_batch(extra_downloads[index],
                                   input.rows(),
                                   extra.output_columns,
                                   transfer_slot.staging_allocator))
        {
            return false;
        }
        const size_t extra_bytes = static_cast<size_t>(input.rows()) * extra.output_columns * sizeof(float);
        extra_direct_host_outputs[index] = extra.vulkan_context->support_direct_host_buffer(extra_bytes, extra_outputs[index]->dtype());
    }

    const DType parallel_output_dtype = parallel_output.dtype();
    parallel_output.reset(input.rows(), parallel.output_columns, false);
    for (size_t index = 0; index < extra_outputs.size(); ++index)
    {
        const Bfloat16Linear_vulkan::Implementation& extra = *extra_operators[index]->d;
        extra_outputs[index]->reset(input.rows(), extra.output_columns, false);
    }

    const std::lock_guard<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (!reset_slot_command(transfer_slot))
        return false;

    ncnn::VkMat original_input_gpu;
    if (!record_mapped_upload(transfer_slot.upload, original_input_gpu, command, first.option))
        return false;

    ncnn::VkMat input_gpu;
    command.record_clone(original_input_gpu, input_gpu, first.option);
    if (input_gpu.empty())
        return false;

    const uint32_t rows = static_cast<uint32_t>(input.rows());
    record_float8_quantize(first.quantize_pipeline.get(),
                           input_gpu,
                           first.logical_input_columns,
                           rows,
                           command);

    ncnn::VkMat intermediate_gpu;
    intermediate_gpu.create(static_cast<int>(first.output_columns),
                            static_cast<int>(input.rows()),
                            sizeof(float),
                            first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    ncnn::VkMat parallel_gpu;
    if (intermediate_gpu.empty()
        || !prepare_output_buffer(*first.vulkan_context,
                                  second.output_columns,
                                  input.rows(),
                                  sizeof(float),
                                  direct_host_output,
                                  transfer_slot.download,
                                  output_gpu)
        || !prepare_output_buffer(*first.vulkan_context,
                                  parallel.output_columns,
                                  input.rows(),
                                  sizeof(float),
                                  parallel_direct_host_output,
                                  parallel_download,
                                  parallel_gpu))
    {
        return false;
    }

    std::vector<ncnn::VkMat> extra_gpu(extra_operators.size());
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const Bfloat16Linear_vulkan::Implementation& extra = *extra_operators[index]->d;
        if (!prepare_output_buffer(*first.vulkan_context,
                                   extra.output_columns,
                                   input.rows(),
                                   act_element_size(extra.option),
                                   extra_direct_host_outputs[index],
                                   extra_downloads[index],
                                   extra_gpu[index]))
        {
            return false;
        }
    }

    first.record_projection(input_gpu, intermediate_gpu, rows, command);
    record_float8_rms_norm_quantize(first.rms_norm_quantize_pipeline.get(),
                                    intermediate_gpu,
                                    first.rms_norm_weight,
                                    first.output_columns,
                                    rows,
                                    first.rms_norm_epsilon,
                                    command);
    second.record_projection(intermediate_gpu, output_gpu, rows, command);
    parallel.record_projection(input_gpu, parallel_gpu, rows, command);

    std::vector<ncnn::VkMat> projection_bindings;
    std::vector<ncnn::vk_constant_type> projection_constants;
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const Bfloat16Linear_vulkan::Implementation& extra = *extra_operators[index]->d;
        (void)extra.record_projection(original_input_gpu,
                                      extra_gpu[index],
                                      rows,
                                      command,
                                      projection_bindings,
                                      projection_constants);
    }

    if ((!retained_output
         && !record_optional_download(output_gpu,
                                      input.rows(),
                                      second.output_columns,
                                      transfer_slot.download,
                                      command,
                                      *first.vulkan_context,
                                      first.option,
                                      output_dtype,
                                      direct_host_output))
        || !record_optional_download(parallel_gpu,
                                     input.rows(),
                                     parallel.output_columns,
                                     parallel_download,
                                     command,
                                     *first.vulkan_context,
                                     first.option,
                                     parallel_output_dtype,
                                     parallel_direct_host_output))
    {
        return false;
    }

    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const Bfloat16Linear_vulkan::Implementation& extra = *extra_operators[index]->d;
        if (!record_optional_download(extra_gpu[index],
                                      input.rows(),
                                      extra.output_columns,
                                      extra_downloads[index],
                                      command,
                                      *extra.vulkan_context,
                                      extra.option,
                                      extra_outputs[index]->dtype(),
                                      extra_direct_host_outputs[index]))
        {
            return false;
        }
    }

    if (submit_wait(command, first.vulkan_context->device()) != 0
        || (!retained_output && !copy_staging_to_cpu_batch(transfer_slot.download, output))
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }

    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        if (!copy_staging_to_cpu_batch(extra_downloads[index], *extra_outputs[index]))
            return false;
    }

    if (retained_output)
    {
        if (!retained_output->assign_completed(output_gpu, first.vulkan_context))
            return false;
        output.clear();
    }

    runtime_state.dispatches += 4 + static_cast<uint64_t>(extra_operators.size());
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads += (retained_output ? 1 : 2) + extra_operators.size();
    return true;
}

bool Float8Linear_vulkan::forward_swiglu_chain(const ActivationBuffer& input,
                                               const Float8Linear_vulkan& up_operator,
                                               const Float8Linear_vulkan& down_operator,
                                               ExpertActivation activation,
                                               float activation_limit,
                                               ActivationBuffer& output) const
{
    (void)activation;
    const Implementation& gate = *d;
    const Implementation& up = *up_operator.d;
    const Implementation& down = *down_operator.d;

    VulkanRuntimeState& runtime_state = gate.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = gate.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    bool direct_host_output = false;
    DType output_dtype = DType::Float32;
    if (!prepare_host_io(*gate.vulkan_context,
                         input,
                         gate.logical_input_columns,
                         gate.optimization_flags,
                         true,
                         down.output_columns,
                         false,
                         transfer_slot,
                         direct_host_output,
                         output_dtype,
                         output))
    {
        return false;
    }

    const std::lock_guard<std::mutex> lock(gate.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (!reset_slot_command(transfer_slot))
        return false;

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
    if (gate_gpu.empty()
        || up_gpu.empty()
        || !prepare_output_buffer(*gate.vulkan_context,
                                  down.output_columns,
                                  input.rows(),
                                  sizeof(float),
                                  direct_host_output,
                                  transfer_slot.download,
                                  output_gpu))
    {
        return false;
    }

    const uint32_t rows = static_cast<uint32_t>(input.rows());
    gate.record_projection(input_gpu, gate_gpu, rows, command);
    up.record_projection(input_gpu, up_gpu, rows, command);
    record_float8_swiglu_quantize(gate.swiglu_quantize_pipeline.get(),
                                  gate_gpu,
                                  up_gpu,
                                  gate.output_columns,
                                  rows,
                                  activation_limit,
                                  command);
    down.record_projection(gate_gpu, output_gpu, rows, command);

    if (!record_optional_download(output_gpu,
                                  input.rows(),
                                  down.output_columns,
                                  transfer_slot.download,
                                  command,
                                  *gate.vulkan_context,
                                  gate.option,
                                  output_dtype,
                                  direct_host_output)
        || submit_wait(command, gate.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }

    runtime_state.dispatches += 3;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    ++runtime_state.shared_expert_swiglu_fusions;
    return true;
}

bool Float8Linear_vulkan::record_swiglu_chain(const ncnn::VkMat& input,
                                              const Float8Linear_vulkan& up_operator,
                                              const Float8Linear_vulkan& down_operator,
                                              ExpertActivation activation,
                                              float activation_limit,
                                              ncnn::VkMat& output,
                                              ncnn::VkCompute& command,
                                              std::vector<ncnn::VkMat>& workspace,
                                              FfnRecordStatistics_vulkan& statistics) const
{
    (void)activation;
    const Implementation& gate = *d;
    const Implementation& up = *up_operator.d;
    const Implementation& down = *down_operator.d;
    const uint64_t rows = input.h > 0 ? static_cast<uint64_t>(input.h) : 0;
    uint64_t scratch_columns[3]{};
    if (fp8_swiglu_shapes(gate.logical_input_columns, gate.output_columns, rows, true, scratch_columns) != 3)
        return false;

    workspace.reserve(workspace.size() + 4);
    ncnn::VkMat gate_gpu;
    if (!record_forward(input, gate_gpu, command, workspace))
        return false;

    const ncnn::VkMat quantized = workspace.back();
    workspace.push_back(gate_gpu);

    ncnn::VkMat up_gpu;
    up_gpu.create(static_cast<int>(scratch_columns[2]),
                  input.h,
                  sizeof(float),
                  gate.vulkan_context->blob_allocator());
    ncnn::VkMat result;
    result.create(static_cast<int>(down.output_columns),
                  input.h,
                  sizeof(float),
                  gate.vulkan_context->blob_allocator());
    if (up_gpu.empty() || result.empty())
        return false;

    workspace.push_back(up_gpu);
    workspace.push_back(result);

    const uint32_t row_count = static_cast<uint32_t>(rows);
    up.record_projection(quantized, up_gpu, row_count, command);
    record_float8_swiglu_quantize(gate.swiglu_quantize_pipeline.get(),
                                  gate_gpu,
                                  up_gpu,
                                  gate.output_columns,
                                  row_count,
                                  activation_limit,
                                  command);
    down.record_projection(gate_gpu, result, row_count, command);
    output = result;
    statistics.dispatches += 5; // Input quantization, Gate, Up, gated quantization, Down.
    ++statistics.shared_expert_swiglu_fusions;
    return true;
}

bool Float8Linear_vulkan::forward_swiglu_chain_device(const DeviceTensor_vulkan& input,
                                                      const Float8Linear_vulkan& up_operator,
                                                      const Float8Linear_vulkan& down_operator,
                                                      ExpertActivation activation,
                                                      float activation_limit,
                                                      DeviceTensor_vulkan& output) const
{
    const std::shared_ptr<VulkanContext>& context = d->vulkan_context;
    const ncnn::VkMat* source = input.value();
    if (!source || input.context() != context)
        return false;

    const ncnn::VkMat retained_input = *source;
    VkTransferLease lease = context->acquire_transfer_slot();
    VkTransferSlot& slot = lease.slot();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if (!reset_slot_command(slot))
        return false;

    struct ResetOnFailure
    {
        ncnn::VkCompute& cmd;
        bool active = true;
        ~ResetOnFailure()
        {
            if (active)
                (void)cmd.reset();
        }
    };

    ncnn::VkCompute& command = *slot.command;
    ResetOnFailure reset_guard{command};
    rearm_read(retained_input);

    ncnn::VkMat input_gpu = retained_input;
    if (input_gpu.elempack != 1)
    {
        context->device()->convert_packing(retained_input, input_gpu, 1, 1, command, d->option);
        if (input_gpu.empty())
            return false;
    }

    ncnn::VkMat output_gpu;
    std::vector<ncnn::VkMat> workspace;
    FfnRecordStatistics_vulkan statistics;
    if (!record_swiglu_chain(input_gpu,
                             up_operator,
                             down_operator,
                             activation,
                             activation_limit,
                             output_gpu,
                             command,
                             workspace,
                             statistics)
        || submit_wait(command, context->device()) != 0)
    {
        return false;
    }

    DeviceTensor_vulkan completed;
    if (!completed.assign_completed(output_gpu, context))
        return false;

    output = std::move(completed);
    VulkanRuntimeState& state = context->runtime_state();
    state.dispatches += statistics.dispatches;
    state.int4_dispatches += statistics.int4_dispatches;
    state.shared_expert_swiglu_fusions += statistics.shared_expert_swiglu_fusions;
    ++state.compute_submissions;
    reset_guard.active = false;
    return true;
}

} // namespace moe
} // namespace ncnn
