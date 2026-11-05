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
#include "kernels/vulkan/float8_projection.comp.hex.h"
#include "kernels/vulkan/float8_quantize.comp.hex.h"
#include "kernels/vulkan/float8_rms_norm_quantize.comp.hex.h"
#include "kernels/vulkan/float8_swiglu_quantize.comp.hex.h"
#endif
#include "linear_impl.h"
#include "linear_internal.h"

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN

void Float8Linear_vulkan::Implementation::record_projection(const ncnn::VkMat& input,
                                                            const ncnn::VkMat& output,
                                                            uint32_t token_count,
                                                            ncnn::VkCompute& command,
                                                            uint32_t output_stride,
                                                            uint32_t output_offset) const
{
    if (output_stride == 0)
        output_stride = output_columns;
    std::vector<ncnn::VkMat> bindings = {input, packed, scales, bias, output};
    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].u32 = matrix_input_columns;
    constants[1].u32 = logical_input_columns;
    constants[2].u32 = output_columns;
    constants[3].u32 = output_columns_per_group;
    constants[4].u32 = block_count;
    constants[5].u32 = token_count;
    constants[6].u32 = scale_row_size;
    constants[7].u32 = output_stride;
    constants[8].u32 = output_offset;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(static_cast<uint64_t>(output_columns) * pipeline->local_size_x());
    dispatcher.h = static_cast<int>(token_count);
    dispatcher.c = 1;
    command.record_pipeline_readonly(pipeline.get(),
                                     bindings,
                                     {1, 0, 0, 0, 0},
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
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = pipeline->local_size_x() * pipeline->local_size_y() * pipeline->local_size_z();
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
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = pipeline->local_size_x() * pipeline->local_size_y() * pipeline->local_size_z();
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
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = pipeline->local_size_x() * pipeline->local_size_y() * pipeline->local_size_z();
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
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = pipeline->local_size_x() * pipeline->local_size_y() * pipeline->local_size_z();
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
        || (matrix.float8_scale_row_size != 1 && matrix.float8_scale_row_size != 128)
        || matrix.shape[1] % 128 != 0 || input_group_count == 0 || matrix.shape[0] % input_group_count != 0
        || matrix.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 32)
        || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
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
    implementation.scale_row_size = matrix.float8_scale_row_size;
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
    implementation.rms_norm = RmsNorm_vulkan::create(weight, epsilon, 0.0f,
                                                     implementation.vulkan_context, implementation.option);
    if (!implementation.rms_norm)
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
    implementation.input_rms_norm = RmsNorm_vulkan::create(weight, epsilon, 0.0f,
                                                           implementation.vulkan_context, implementation.option);
    if (!implementation.input_rms_norm)
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
    if (!prepare_staging_batch(staging,
                               input.rows(),
                               columns,
                               allocator,
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

static bool float8_linear_scratch_shape(uint32_t input_columns, uint64_t rows,
                                        uint32_t& scratch_columns) noexcept
{
    scratch_columns = input_columns;
    return input_columns != 0 && rows != 0
           && saturating_multiply_u64(input_columns, rows) != UINT64_MAX;
}

std::vector<uint64_t> Float8Linear_vulkan::scratch_elements(uint32_t input_columns,
                                                            uint64_t rows)
{
    uint32_t scratch_columns = 0;
    if (!float8_linear_scratch_shape(input_columns, rows, scratch_columns))
        return input_columns == 0 || rows == 0 ? std::vector<uint64_t>{}
                                               : std::vector<uint64_t>{UINT64_MAX};
    return {saturating_multiply_u64(scratch_columns, rows)};
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
        if (elements == UINT64_MAX) return {UINT64_MAX};
        if (elements != 0) result.push_back(elements);
    }
    return result;
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& Float8Linear_vulkan::vulkan_context() const noexcept
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
        return false;
    ncnn::VkMat intermediate;
    if (!record_forward(input, intermediate, cmd, workspace)
        || !d->rms_norm->record(intermediate, query_rank, cmd)
        || !next.record_forward(query_rank, query, cmd, workspace)
        || !parallel.record_forward(input, kv, cmd, workspace))
        return false;
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
    if (!d->vulkan_context || !d->pipeline || !d->quantize_pipeline
        || input.empty() || input.dims != 2 || input.elempack != 1
        || input.elemsize != sizeof(float) || input.h <= 0
        || input.w != static_cast<int>(d->logical_input_columns)
        || !float8_linear_scratch_shape(d->logical_input_columns, rows, scratch_columns)
        || d->logical_input_columns % 128 != 0)
        return false;
    if (output_stride == 0)
        output_stride = d->output_columns;
    if (output_stride > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || output_column_offset > output_stride
        || d->output_columns > output_stride - output_column_offset)
        return false;
    ncnn::VkMat quantized;
    command.record_clone(input, quantized, d->option);
    if (output_column_offset == 0 && output_stride == d->output_columns)
    {
        output.create(static_cast<int>(d->output_columns), input.h, sizeof(float),
                      d->vulkan_context->blob_allocator());
    }
    else if (output.empty()
             || output.dims != 2
             || output.elempack != 1
             || output.elemsize != sizeof(float)
             || output.w != static_cast<int>(output_stride)
             || output.h != input.h)
    {
        return false;
    }
    if (quantized.empty() || output.empty())
        return false;
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].u32 = scratch_columns;
    constants[1].u32 = static_cast<uint32_t>(rows);
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>((static_cast<uint64_t>(scratch_columns) / 128)
                                    * d->quantize_pipeline->local_size_x());
    dispatcher.h = static_cast<int>(rows);
    dispatcher.c = 1;
    command.record_pipeline(d->quantize_pipeline.get(), {quantized}, constants, dispatcher);
    d->record_projection(quantized,
                         output,
                         static_cast<uint32_t>(rows),
                         command,
                         output_stride,
                         output_column_offset);
    workspace.push_back(std::move(quantized));
    return true;
}
#endif

bool Float8Linear_vulkan::materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const ncnn::VkMat* value = input.value();
    if (!value || !d->vulkan_context || input.vulkan_context() != d->vulkan_context)
        return false;
    VulkanTransferLease lease = d->vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& slot = lease.slot();
    VulkanRuntimeState& state = d->vulkan_context->runtime_state();
    if (!prepare_staging_batch(slot.download, input.rows(), input.columns(),
                               slot.staging_allocator, output.element_size()))
        return false;
    const DType dtype = output.dtype();
    output.reset(input.rows(), input.columns(), false);
    const std::lock_guard<std::mutex> lock(d->vulkan_context->command_mutex());
    input.prepare_read();
    if (slot.command_used && slot.command->reset() != 0)
        return false;
    slot.command_used = true;
    if (!record_staging_download(*value, input.rows(), input.columns(),
                                 slot.download, *slot.command,
                                 d->vulkan_context->device(), d->option, dtype)
        || submit_compute_and_wait(*slot.command, d->vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(slot.download, output))
        return false;
    ++state.compute_submissions;
    ++state.batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

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
                                       transfer_slot.staging_allocator)
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
        return false;
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = bind_host_out(transfer_slot.download);
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
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     implementation.output_columns,
                                     transfer_slot.download,
                                     command,
                                     implementation.vulkan_context->device(),
                                     implementation.option,
                                     output_dtype))
        || submit_compute_and_wait(command, implementation.vulkan_context->device()) != 0
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
                                       transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator))
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
        output_gpu = bind_host_out(transfer_slot.download);
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
    quantize_dispatcher.w = static_cast<int>((static_cast<uint64_t>(first.output_columns) / 128)
                                             * first.quantize_pipeline->local_size_x());
    quantize_dispatcher.h = static_cast<int>(input.rows());
    quantize_dispatcher.c = 1;
    command.record_pipeline(first.quantize_pipeline.get(), quantize_bindings, quantize_constants, quantize_dispatcher);

    second.record_projection(intermediate_gpu,
                             output_gpu,
                             static_cast<uint32_t>(input.rows()),
                             command);

    if ((!direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     second.output_columns,
                                     transfer_slot.download,
                                     command,
                                     first.vulkan_context->device(),
                                     first.option,
                                     output_dtype))
        || submit_compute_and_wait(command, first.vulkan_context->device()) != 0
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
                                       transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator))
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
        output_gpu = bind_host_out(transfer_slot.download);
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
    norm_dispatcher.w = static_cast<int>(first.rms_norm_quantize_pipeline->local_size_x());
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
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     second.output_columns,
                                     transfer_slot.download,
                                     command,
                                     first.vulkan_context->device(),
                                     first.option,
                                     output_dtype))
        || submit_compute_and_wait(command, first.vulkan_context->device()) != 0
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
                                                               bool normalize_input,
                                                               DeviceTensor_vulkan* retained_output) const
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
    const bool direct_host_output = !retained_output && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * second.output_columns * sizeof(float), output.dtype());
    const bool parallel_direct_host_output = parallel.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * parallel.output_columns
                                                                                                     * sizeof(float),
                                                                                                 parallel_output.dtype());
    ncnn::VkMat parallel_download;
    if (!(normalize_input
              ? fill_staging_upload(input,
                                    transfer_slot.upload,
                                    transfer_slot.staging_allocator)
              : fill_float8_quantized_staging(input,
                                              first.logical_input_columns,
                                              first.optimization_flags,
                                              transfer_slot.upload,
                                              transfer_slot.staging_allocator))
        || (!retained_output && !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator))
        || !prepare_staging_batch(parallel_download, input.rows(), parallel.output_columns, transfer_slot.staging_allocator))
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
        input_norm_dispatcher.w = static_cast<int>(first.rms_norm_quantize_pipeline->local_size_x());
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
        output_gpu = bind_host_out(transfer_slot.download);
    else
        output_gpu.create(static_cast<int>(second.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          first.vulkan_context->blob_allocator());
    ncnn::VkMat parallel_gpu;
    if (parallel_direct_host_output)
        parallel_gpu = bind_host_out(parallel_download);
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
    norm_dispatcher.w = static_cast<int>(first.rms_norm_quantize_pipeline->local_size_x());
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
    if ((!retained_output && !direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     second.output_columns,
                                     transfer_slot.download,
                                     command,
                                     first.vulkan_context->device(),
                                     first.option,
                                     output_dtype))
        || (!parallel_direct_host_output
            && !record_staging_download(parallel_gpu,
                                        input.rows(),
                                        parallel.output_columns,
                                        parallel_download,
                                        command,
                                        first.vulkan_context->device(),
                                        first.option,
                                        parallel_output_dtype))
        || submit_compute_and_wait(command, first.vulkan_context->device()) != 0
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
#else
    (void)input;
    (void)next;
    (void)parallel_operator;
    (void)output;
    (void)parallel_output;
    (void)retained_output;
    (void)normalize_input;
    return false;
#endif
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
                                                false, retained_output);
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
                                                true, retained_output);
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
#if NCNN_MOE_WITH_VULKAN
    if (extra_operators.empty())
        return forward_rms_norm_chain_parallel(input,
                                               next,
                                               parallel_operator,
                                               output,
                                               parallel_output, retained_output);
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
        const size_t variant = activation_variant(extra.option);
        if (index == 0)
            extra_storage_variant = variant;
        else if (variant != extra_storage_variant)
            return false;
    }
    if (extra_storage_variant != activation_variant(first.option)
        || extra_storage_variant != 0)
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_output = !retained_output && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * second.output_columns * sizeof(float), output.dtype());
    const bool parallel_direct_host_output = parallel.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * parallel.output_columns
                                                                                                     * sizeof(float),
                                                                                                 parallel_output.dtype());
    ncnn::VkMat parallel_download;
    std::vector<ncnn::VkMat> extra_downloads(extra_operators.size());
    std::vector<bool> extra_direct_host_outputs(extra_operators.size(), false);
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator)
        || (!retained_output && !prepare_staging_batch(transfer_slot.download, input.rows(), second.output_columns, transfer_slot.staging_allocator))
        || !prepare_staging_batch(parallel_download,
                                  input.rows(),
                                  parallel.output_columns,
                                  transfer_slot.staging_allocator))
    {
        return false;
    }
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->d;
        if (!prepare_staging_batch(extra_downloads[index],
                                   input.rows(),
                                   extra.output_columns,
                                   transfer_slot.staging_allocator))
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
    input_quantize_dispatcher.w = static_cast<int>((static_cast<uint64_t>(first.logical_input_columns) / 128)
                                                   * first.quantize_pipeline->local_size_x());
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
        output_gpu = bind_host_out(transfer_slot.download);
    else
        output_gpu.create(static_cast<int>(second.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          first.vulkan_context->blob_allocator());
    ncnn::VkMat parallel_gpu;
    if (parallel_direct_host_output)
        parallel_gpu = bind_host_out(parallel_download);
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
            extra_gpu[index] = bind_host_out(extra_downloads[index]);
        else
            extra_gpu[index].create(static_cast<int>(extra.output_columns),
                                    static_cast<int>(input.rows()),
                                    act_element_size(extra.option),
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
    norm_dispatcher.w = static_cast<int>(first.rms_norm_quantize_pipeline->local_size_x());
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
    std::vector<ncnn::VkMat> projection_bindings;
    std::vector<ncnn::vk_constant_type> projection_constants;
    for (size_t index = 0; index < extra_operators.size(); ++index)
    {
        const auto& extra = *extra_operators[index]->d;
        cooperative_dispatches[index] = extra.record_projection(original_input_gpu,
                                                                extra_gpu[index],
                                                                static_cast<uint32_t>(input.rows()),
                                                                command,
                                                                projection_bindings,
                                                                projection_constants);
    }

    if ((!retained_output && !direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     second.output_columns,
                                     transfer_slot.download,
                                     command,
                                     first.vulkan_context->device(),
                                     first.option,
                                     output_dtype))
        || (!parallel_direct_host_output
            && !record_staging_download(parallel_gpu,
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
            && !record_staging_download(extra_gpu[index],
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
    if (submit_compute_and_wait(command, first.vulkan_context->device()) != 0
        || (!retained_output && !copy_staging_to_cpu_batch(transfer_slot.download, output))
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
    (void)retained_output;
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
                                       transfer_slot.staging_allocator)
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
        output_gpu = bind_host_out(transfer_slot.download);
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
    activation_dispatcher.w = static_cast<int>((static_cast<uint64_t>(gate.output_columns) / 128)
                                               * gate.swiglu_quantize_pipeline->local_size_x());
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
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     down.output_columns,
                                     transfer_slot.download,
                                     command,
                                     gate.vulkan_context->device(),
                                     gate.option,
                                     output_dtype))
        || submit_compute_and_wait(command, gate.vulkan_context->device()) != 0
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

#if NCNN_MOE_WITH_VULKAN
bool Float8Linear_vulkan::record_swiglu_chain(const ncnn::VkMat& input,
                                              const Float8Linear_vulkan& up_operator,
                                              const Float8Linear_vulkan& down_operator,
                                              ExpertActivation activation, float activation_limit,
                                              ncnn::VkMat& output, ncnn::VkCompute& command,
                                              std::vector<ncnn::VkMat>& workspace,
                                              FfnRecordStatistics_vulkan& statistics) const
{
    const auto& gate = *d;
    const auto& up = *up_operator.d;
    const auto& down = *down_operator.d;
    uint64_t scratch_columns[3]{};
    const size_t scratch_count = fp8_swiglu_shapes(gate.logical_input_columns,
                                                   gate.output_columns,
                                                   input.h > 0 ? static_cast<uint64_t>(input.h) : 0,
                                                   true,
                                                   scratch_columns);
    if (&input == &output || activation != ExpertActivation::DeepSeekSwiGlu
        || !gate.vulkan_context || gate.vulkan_context != up.vulkan_context || gate.vulkan_context != down.vulkan_context
        || !gate.pipeline || !gate.quantize_pipeline || !up.pipeline || !down.pipeline || !gate.swiglu_quantize_pipeline
        || gate.input_group_count != 1 || up.input_group_count != 1 || down.input_group_count != 1
        || input.empty() || input.dims != 2 || input.elempack != 1 || input.elemsize != sizeof(float)
        || input.h <= 0 || scratch_count != 3 || input.w != static_cast<int>(scratch_columns[0])
        || !input.allocator || input.allocator->vkdev != gate.vulkan_context->device()
        || static_cast<uint32_t>(input.h) > gate.vulkan_context->device()->info.max_workgroup_count_y()
        || gate.logical_input_columns != up.logical_input_columns || gate.output_columns != up.output_columns
        || gate.output_columns != down.logical_input_columns || gate.output_columns != scratch_columns[1]
        || up.output_columns != scratch_columns[2] || gate.output_columns % 128 != 0)
        return false;
    workspace.reserve(workspace.size() + 4);
    ncnn::VkMat gate_gpu, up_gpu, result;
    if (!record_forward(input, gate_gpu, command, workspace)) return false;
    const ncnn::VkMat quantized = workspace.back();
    workspace.push_back(gate_gpu);
    up_gpu.create(static_cast<int>(scratch_columns[2]), input.h, sizeof(float), gate.vulkan_context->blob_allocator());
    result.create(static_cast<int>(down.output_columns), input.h, sizeof(float), gate.vulkan_context->blob_allocator());
    if (up_gpu.empty() || result.empty()) return false;
    workspace.push_back(up_gpu);
    workspace.push_back(result);
    up.record_projection(quantized, up_gpu, static_cast<uint32_t>(input.h), command);
    std::vector<ncnn::vk_constant_type> constants(3);
    constants[0].u32 = gate.output_columns;
    constants[1].u32 = static_cast<uint32_t>(input.h);
    constants[2].f = activation_limit;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>((static_cast<uint64_t>(gate.output_columns) / 128)
                                    * gate.swiglu_quantize_pipeline->local_size_x());
    dispatcher.h = input.h;
    dispatcher.c = 1;
    command.record_pipeline(gate.swiglu_quantize_pipeline.get(), {gate_gpu, up_gpu}, constants, dispatcher);
    down.record_projection(gate_gpu, result, static_cast<uint32_t>(input.h), command);
    output = result;
    statistics.dispatches += 5; // Input quantization, Gate, Up, gated quantization, Down.
    ++statistics.shared_expert_swiglu_fusions;
    return true;
}
#endif

bool Float8Linear_vulkan::forward_swiglu_chain_device(const DeviceTensor_vulkan& input,
                                                      const Float8Linear_vulkan& up_operator,
                                                      const Float8Linear_vulkan& down_operator,
                                                      ExpertActivation activation,
                                                      float activation_limit,
                                                      DeviceTensor_vulkan& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const auto& context = d->vulkan_context;
    const auto* source = input.value();
    if (&input == &output || !context || !source || source->dims != 2
        || source->elempack <= 0 || source->elemsize != sizeof(float) * static_cast<size_t>(source->elempack)
        || input.vulkan_context() != context)
        return false;
    const ncnn::VkMat retained_input = *source;
    auto lease = context->acquire_transfer_slot();
    auto& slot = lease.slot();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if (slot.command_used && slot.command->reset() != 0) return false;
    slot.command_used = true;
    auto& command = *slot.command;
    ncnn::VkMat input_gpu = retained_input;
    ncnn::VkMat output_gpu;
    std::vector<ncnn::VkMat> workspace;
    FfnRecordStatistics_vulkan statistics;
    bool successful = false;
    const auto cleanup = [&](void*) {
        if (!successful) (void)command.reset();
    };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);
    input.prepare_read();
    if (input_gpu.elempack != 1)
    {
        context->device()->convert_packing(retained_input, input_gpu, 1, 1, command, d->option);
        if (input_gpu.empty()) return false;
    }
    if (!record_swiglu_chain(input_gpu, up_operator, down_operator, activation, activation_limit,
                             output_gpu, command, workspace, statistics)
        || submit_compute_and_wait(command, context->device()) != 0)
        return false;
    DeviceTensor_vulkan completed;
    if (!completed.assign_completed(output_gpu, context)) return false;
    output = std::move(completed);
    auto& state = context->runtime_state();
    state.dispatches += statistics.dispatches;
    state.int4_dispatches += statistics.int4_dispatches;
    state.shared_expert_swiglu_fusions += statistics.shared_expert_swiglu_fusions;
    state.bfloat16_cooperative_matrix_dispatches += statistics.bfloat16_cooperative_matrix_dispatches;
    ++state.compute_submissions;
    successful = true;
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

} // namespace moe
} // namespace ncnn
