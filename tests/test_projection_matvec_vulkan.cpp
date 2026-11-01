#include "backends/ncnn/linear.h"
#include "backends/ncnn/vulkancontext.h"
#include "kernels/bfloat16.h"
#include "kernels/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static void projection_check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

static float projection_value(const TensorData& tensor, size_t index)
{
    return tensor.dtype == DType::Float32 ? tensor.float32_values()[index]
                                          : bfloat16_to_float(tensor.bfloat16_values()[index]);
}

static TensorData projection_weight(DType dtype, uint32_t columns, uint32_t input_columns)
{
    TensorData result;
    result.dtype = dtype;
    result.shape = {columns, input_columns};
    for (size_t index = 0; index < static_cast<size_t>(columns) * input_columns; ++index)
    {
        const float value = std::sin(static_cast<float>((index * 17 + index / 13 + 5) % 997) * 0.031f) * 0.13f;
        if (dtype == DType::Float32)
            result.float32_data.push_back(value);
        else
            result.bfloat16_data.push_back(float_to_bfloat16(value));
    }
    return result;
}

static TensorData projection_bias(DType dtype, uint32_t columns)
{
    TensorData result;
    result.dtype = dtype;
    result.shape = {columns};
    for (uint32_t column = 0; column < columns; ++column)
    {
        const float value = static_cast<float>(static_cast<int>(column % 11) - 5) * 0.017f;
        if (dtype == DType::Float32)
            result.float32_data.push_back(value);
        else
            result.bfloat16_data.push_back(float_to_bfloat16(value));
    }
    return result;
}

static ncnn::Option projection_option(const std::shared_ptr<VulkanContext>& context)
{
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.blob_vkallocator = context->blob_allocator();
    option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    return option;
}

static int record_projection(const Linear* float32, const Bfloat16Linear_vulkan* bfloat16,
                             const ncnn::VkMat& input, ncnn::VkMat& output,
                             ncnn::VkCompute& command, const ncnn::Option& option)
{
    return float32 ? float32->forward(input, output, command, option)
                   : bfloat16->forward(input, output, command, option);
}

static ActivationBuffer projection_execute(const Linear* float32, const Bfloat16Linear_vulkan* bfloat16,
                                           const std::shared_ptr<VulkanContext>& context,
                                           const VulkanRuntimePtr& runtime, const ActivationBuffer& input,
                                           uint32_t output_columns, bool reject_bad_descriptors)
{
    const auto before = get_vulkan_statistics(runtime);
    ActivationBuffer result(input.rows(), output_columns);
    ncnn::VkMat input_staging, output_staging;
    projection_check(fill_staging_upload(input, input_staging, context->staging_allocator())
                         && prepare_staging_batch(output_staging, input.rows(), output_columns, context->staging_allocator()),
                     "cannot allocate projection staging");
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        const auto option = projection_option(context);
        ncnn::VkCompute command(context->device());
        ncnn::VkMat device_input, device_output;
        projection_check(record_prepared_staging_upload(input_staging, input.rows(), device_input, command,
                                                        context->device(), option, input.dtype()),
                         "cannot record projection input upload");
#if NCNN_BATCH
        if (reject_bad_descriptors)
        {
            ncnn::VkMat invalid = device_input;
            invalid.nstep = 0;
            projection_check(record_projection(float32, bfloat16, invalid, device_output, command, option) != 0
                                 && device_output.empty(),
                             "projection accepted a truncated descriptor");
            invalid = device_input;
            invalid.n = 2;
            projection_check(record_projection(float32, bfloat16, invalid, device_output, command, option) != 0
                                 && device_output.empty(),
                             "projection accepted an unsupported batch dimension");
        }
#else
        (void)reject_bad_descriptors;
#endif
        projection_check(record_projection(float32, bfloat16, device_input, device_output, command, option) == 0,
                         "device projection recording failed");
        const size_t logical_bytes = input.rows() * output_columns * sizeof(float);
        const std::string descriptor_label = std::to_string(input.columns()) + "x" + std::to_string(output_columns)
                                             + " rows=" + std::to_string(input.rows())
                                             + " dims=" + std::to_string(device_output.dims)
                                             + " cstep=" + std::to_string(device_output.cstep)
                                             + " bytes=" + std::to_string(device_output.buffer_capacity());
        projection_check(device_output.dims == 2 && device_output.w == static_cast<int>(output_columns)
                             && device_output.h == static_cast<int>(input.rows())
                             && device_output.elempack == 1 && device_output.elemsize == sizeof(float)
                             && device_output.buffer_capacity() >= logical_bytes
                             && device_output.buffer_capacity() <= device_output.data->capacity,
                         "projection output lost its full FP32 descriptor range: " + descriptor_label);
#if NCNN_BATCH
        projection_check(device_output.n == 1 && device_output.nstep == device_output.cstep,
                         "projection output has incorrect batch stride: " + descriptor_label);
#endif
        projection_check(device_output.data != device_input.data,
                         "projection output aliases its readonly input");
        projection_check(record_prepared_activation_staging_download(device_output, input.rows(), output_columns,
                                                                     output_staging, command, context->device(), option)
                             && submit_compute_and_wait(command, context->device()) == 0
                             && copy_staging_to_cpu_batch(output_staging, result),
                         "cannot complete projection output download");
    }
    const auto after = get_vulkan_statistics(runtime);
    projection_check(after.dispatches == before.dispatches && after.compute_submissions == before.compute_submissions
                         && after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads
                         && after.weight_upload_bytes == before.weight_upload_bytes
                         && after.weight_transfer_submissions == before.weight_transfer_submissions,
                     "record-only projection published counters or uploaded weights");
    return result;
}

static void projection_compare(const TensorData& matrix, const TensorData* bias,
                               const ActivationBuffer& input, const ActivationBuffer& output,
                               const std::string& label)
{
    const auto cpu = bias ? forward_linear(matrix, *bias, input, OptimizationDefaultFlags)
                          : forward_linear(matrix, input, OptimizationDefaultFlags);
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t row = 0; row < input.rows(); ++row)
        for (uint32_t column = 0; column < matrix.shape[0]; ++column)
        {
            double expected = bias ? projection_value(*bias, column) : 0.0;
            double magnitude = std::abs(expected);
            for (uint32_t k = 0; k < matrix.shape[1]; ++k)
            {
                const double term = static_cast<double>(projection_value(matrix, static_cast<size_t>(column) * matrix.shape[1] + k)) * input.row(row)[k];
                expected += term;
                magnitude += std::abs(term);
            }
            const float actual = output.row(row)[column];
            if (std::isnan(expected))
            {
                projection_check(std::isnan(actual), label + ": NaN was discarded");
                continue;
            }
            if (std::isinf(expected))
            {
                projection_check(std::isinf(actual) && std::signbit(actual) == std::signbit(expected), label + ": infinity changed");
                continue;
            }
            projection_check(std::isfinite(actual), label + ": finite projection became nonfinite");
            const double tolerance = 2e-6 + 2e-6 * std::abs(expected) + 2e-7 * magnitude;
            projection_check(std::abs(static_cast<double>(actual) - expected) <= tolerance,
                             label + ": FP64 elementwise projection mismatch");
            projection_check(std::abs(static_cast<double>(actual) - cpu.row(row)[column]) <= tolerance * 2,
                             label + ": CPU projection mismatch");
            squared_error += (static_cast<double>(actual) - expected) * (static_cast<double>(actual) - expected);
            squared_reference += expected * expected;
        }
    projection_check(std::sqrt(squared_error / std::max(1.0, squared_reference)) <= 2e-5,
                     label + ": FP64 relative L2 mismatch");
}

static void projection_case(uint32_t device, const VulkanRuntimePtr& runtime,
                            const std::shared_ptr<VulkanContext>& context,
                            uint32_t input_columns, uint32_t output_columns,
                            DType dtype, bool native_bfloat16, bool has_bias)
{
    const auto matrix = projection_weight(dtype, output_columns, input_columns);
    const auto bias = projection_bias(has_bias && output_columns % 2 == 0 ? DType::BFloat16 : DType::Float32, output_columns);
    const auto float32 = native_bfloat16 ? std::shared_ptr<Linear>{}
                                         : Linear::create(matrix, has_bias ? &bias : nullptr, device, runtime, OptimizationDefaultFlags);
    const auto bfloat16 = native_bfloat16 ? Bfloat16Linear_vulkan::create(matrix, has_bias ? &bias : nullptr, device, runtime, OptimizationDefaultFlags)
                                          : std::shared_ptr<Bfloat16Linear_vulkan>{};
    projection_check(float32 || bfloat16, "projection factory failed");
    ActivationBuffer input(2, input_columns);
    for (size_t row = 0; row < input.rows(); ++row)
        for (uint32_t k = 0; k < input_columns; ++k)
            input.row(row)[k] = std::cos(static_cast<float>((k * 29 + row * 7) % 1021) * 0.023f) * 0.21f;
    ActivationBuffer single(1, input_columns);
    std::copy_n(input.row(0), input_columns, single.row(0));
    const auto snapshot = single;
    const auto actual = projection_execute(float32.get(), bfloat16.get(), context, runtime, single, output_columns, true);
    const auto multirow = projection_execute(float32.get(), bfloat16.get(), context, runtime, input, output_columns, false);
    const std::string label = (native_bfloat16 ? "native BF16 " : dtype == DType::BFloat16 ? "expanded BF16 "
                                                                                           : "FP32 ")
                              + std::to_string(input_columns) + "x" + std::to_string(output_columns);
    projection_compare(matrix, has_bias ? &bias : nullptr, single, actual, label);
    projection_compare(matrix, has_bias ? &bias : nullptr, input, multirow, label + " multirow");
    for (uint32_t column = 0; column < output_columns; ++column)
        projection_check(std::abs(actual.row(0)[column] - multirow.row(0)[column]) <= 4e-5f * (1.0f + std::abs(multirow.row(0)[column])),
                         label + ": single-row and multirow projections differ");
    projection_check(std::equal(single.values().begin(), single.values().end(), snapshot.values().begin()),
                     "projection modified its host input");
}

static void projection_nonfinite(uint32_t device, const VulkanRuntimePtr& runtime,
                                 const std::shared_ptr<VulkanContext>& context, bool native_bfloat16)
{
    TensorData matrix;
    matrix.dtype = native_bfloat16 ? DType::BFloat16 : DType::Float32;
    matrix.shape = {3, 4};
    if (native_bfloat16)
        matrix.bfloat16_data.assign(12, float_to_bfloat16(0.25f));
    else
        matrix.float32_data.assign(12, 0.25f);
    const auto float32 = native_bfloat16 ? std::shared_ptr<Linear>{}
                                         : Linear::create(matrix, nullptr, device, runtime, OptimizationDefaultFlags);
    const auto bfloat16 = native_bfloat16 ? Bfloat16Linear_vulkan::create(matrix, nullptr, device, runtime, OptimizationDefaultFlags)
                                          : std::shared_ptr<Bfloat16Linear_vulkan>{};
    projection_check(float32 || bfloat16, "nonfinite projection factory failed");
    for (const float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
    {
        ActivationBuffer input(1, 4);
        input.row(0)[0] = value;
        const auto result = projection_execute(float32.get(), bfloat16.get(), context, runtime, input, 3, false);
        projection_compare(matrix, nullptr, input, result, "nonfinite projection");
    }
}

static void test_projection_matvec(uint32_t device)
{
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    projection_check(static_cast<bool>(context), "cannot acquire projection Context");
    struct Case
    {
        uint32_t input_columns, output_columns;
        bool bias;
    };
    const std::array<Case, 8> float32_cases = {{{1, 1, false}, {3, 5, true}, {4, 4, false}, {7, 4, true}, {8, 3, false}, {129, 7, true}, {513, 13, false}, {4096, 33, true}}};
    for (const auto& test : float32_cases)
        projection_case(device, runtime, context, test.input_columns, test.output_columns, DType::Float32, false, test.bias);
    projection_case(device, runtime, context, 129, 7, DType::BFloat16, false, true);
    projection_nonfinite(device, runtime, context, false);
    if (context->device()->info.subgroup_size() == 32)
    {
        const std::array<Case, 4> bfloat16_cases = {{{4, 1, false}, {132, 17, true}, {2880, 32, true}, {4096, 33, true}}};
        for (const auto& test : bfloat16_cases)
            projection_case(device, runtime, context, test.input_columns, test.output_columns, DType::BFloat16, true, test.bias);
        projection_nonfinite(device, runtime, context, true);
    }
    else
        std::cout << "SKIP: native BF16 projection requires the existing 32-lane capability; FP32 matvec tested\n";
}
#endif

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::moe::get_gpu_count() == 0)
    {
        std::cout << "SKIP: Vulkan GPU unavailable\n";
        return 77;
    }
    try
    {
        const uint32_t device = argc == 3 && std::string(argv[1]) == "--device"
                                    ? static_cast<uint32_t>(std::stoul(argv[2]))
                                    : ncnn::moe::get_default_gpu_index();
        ncnn::moe::test_projection_matvec(device);
        std::cout << "Projection matvec FP64/CPU, descriptor, tails and multirow comparisons passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Projection matvec test failed: " << error.what() << '\n';
        return 1;
    }
#else
    (void)argc;
    (void)argv;
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
