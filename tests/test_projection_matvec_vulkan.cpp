#include "backends/ncnn/linear.h"
#include "backends/ncnn/vulkancontext.h"
#include "kernels/bfloat16.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

static void projection_quantized(uint32_t device, const VulkanRuntimePtr& runtime,
                                 DType dtype, uint32_t columns, uint32_t outputs,
                                 uint32_t groups = 1, bool permuted = false)
{
    TensorData matrix;
    matrix.dtype = dtype;
    matrix.shape = {outputs, columns};
    matrix.quantized_data.resize(static_cast<size_t>(qnk_storage_bytes(dtype, outputs, columns)));
    for (size_t i = 0; i < matrix.quantized_data.size(); ++i)
        matrix.quantized_data[i] = static_cast<uint8_t>((i * 37u + i / 11u + 53u) & 255u);
    const auto half = [](uint8_t* data) { data[0] = 0x00; data[1] = 0x24; };
    const uint32_t block_size = qnk_block_size(dtype);
    const size_t block_bytes = qnk_block_bytes(dtype);
    for (size_t offset = 0; offset < matrix.quantized_data.size(); offset += block_bytes)
    {
        uint8_t* block = matrix.quantized_data.data() + offset;
        if (dtype == DType::Q2K)
        {
            half(block + 80);
            half(block + 82);
        }
        else if (dtype == DType::Q3K)
            half(block + 108);
        else if (dtype == DType::Q6K)
            half(block + 208);
        else if (dtype == DType::Q8K)
        {
            const float scale = 0.015625f;
            std::memcpy(block, &scale, sizeof(scale));
        }
        else
        {
            half(block);
            if (dtype == DType::Q4K || dtype == DType::Q5K || dtype == DType::Q4_1 || dtype == DType::Q5_1)
                half(block + 2);
        }
    }
    if (permuted)
        for (uint32_t k = 0; k < columns; ++k)
            matrix.input_column_indices.push_back(columns - k - 1);

    // Decode with the independent CPU block decoder, then compare against an
    // FP64 dot product. Grouping and input permutations are explicit here.
    TensorData reference;
    reference.dtype = DType::Float32;
    reference.shape = {outputs, columns * groups};
    reference.float32_data.resize(static_cast<size_t>(outputs) * columns * groups, 0.0f);
    std::array<float, qnk_block_elements> decoded;
    for (uint32_t output = 0; output < outputs; ++output)
        for (uint32_t k = 0; k < columns; k += block_size)
        {
            qnk_dequantize_block(dtype, matrix.quantized_data.data() + (static_cast<size_t>(output) * columns + k) / block_size * block_bytes, decoded.data());
            const uint32_t group = output / (outputs / groups);
            for (uint32_t i = 0; i < block_size; ++i)
            {
                const uint32_t input_column = permuted ? columns - (k + i) - 1 : k + i;
                reference.float32_data[(static_cast<size_t>(output) * groups + group) * columns + input_column] = decoded[i];
            }
        }
    const auto bias = projection_bias(DType::Float32, outputs);
    const bool has_bias = outputs % 2 != 0;
    const auto operation = QnkLinear_vulkan::create(matrix, has_bias ? &bias : nullptr, device, runtime, OptimizationDefaultFlags, groups);
    projection_check(static_cast<bool>(operation), "native quantized projection factory failed");
    const auto before = get_vulkan_statistics(runtime);
    std::vector<float> single_row;
    for (const size_t rows : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(5), size_t(9)})
    {
        ActivationBuffer input(rows, columns * groups), actual;
        for (size_t row = 0; row < rows; ++row)
            for (uint32_t k = 0; k < input.columns(); ++k)
                input.row(row)[k] = std::cos(static_cast<float>((k * 29 + row * 7) % 1021) * 0.023f) * 0.21f;
        projection_check(operation->forward(input, actual), "native quantized projection execution failed");
        projection_check(actual.rows() == rows && actual.columns() == outputs, "native quantized projection shape mismatch");
        projection_compare(reference, has_bias ? &bias : nullptr, input, actual,
                           "native dtype=" + std::to_string(static_cast<int>(dtype)) + " rows=" + std::to_string(rows));
        if (rows == 1)
            single_row.assign(actual.row(0), actual.row(0) + outputs);
        else
            projection_check(std::equal(single_row.begin(), single_row.end(), actual.row(0)),
                             "native quantized prefill changed the single-row accumulation order");
        if (outputs > 65535) break;
    }
    const auto after = get_vulkan_statistics(runtime);
    projection_check(after.weight_upload_bytes == before.weight_upload_bytes
                         && after.weight_transfer_submissions == before.weight_transfer_submissions,
                     "native quantized projection reuploaded weights during execution");
}

static void projection_quantized_fusion(uint32_t device, const VulkanRuntimePtr& runtime, bool permuted)
{
    const std::array<DType, 4> types = {DType::IQ4NL, DType::Q5_0, DType::Q4_0, DType::Q8_0};
    std::array<TensorData, 4> matrices, biases;
    std::array<const TensorData*, 4> weights, bias_ptrs;
    TensorData reference, reference_bias;
    reference.dtype = reference_bias.dtype = DType::Float32;
    reference.shape = {0, 256};
    uint64_t planned = 0;
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    for (size_t m = 0; m < matrices.size(); ++m)
    {
        auto& matrix = matrices[m];
        matrix.dtype = types[m];
        matrix.shape = {static_cast<uint32_t>(m * 2 + 1), 256};
        matrix.quantized_data.resize(static_cast<size_t>(qnk_storage_bytes(matrix.dtype, matrix.shape[0], 256)));
        for (size_t i = 0; i < matrix.quantized_data.size(); ++i)
            matrix.quantized_data[i] = static_cast<uint8_t>(i * 43 + i / 7 + m * 11);
        const size_t bytes = qnk_block_bytes(matrix.dtype);
        for (size_t i = 0; i < matrix.quantized_data.size(); i += bytes)
        {
            // Include zero, negative and non-power-of-two scales.
            const uint16_t scale = i / bytes % 3 == 0 ? 0 : i / bytes % 3 == 1 ? 0xa521
                                                                               : 0x26ab;
            std::memcpy(matrix.quantized_data.data() + i, &scale, sizeof(scale));
        }
        if (permuted)
            for (uint32_t k = 0; k < 256; ++k) matrix.input_column_indices.push_back(255 - k);
        weights[m] = &matrix;
        biases[m] = projection_bias(m % 2 == 0 ? DType::Float32 : DType::BFloat16, matrix.shape[0]);
        bias_ptrs[m] = m == 2 ? nullptr : &biases[m];
        const size_t first = reference.float32_data.size();
        reference.shape[0] += matrix.shape[0];
        reference.float32_data.resize(static_cast<size_t>(reference.shape[0]) * 256);
        std::array<float, qnk_block_elements> block;
        for (size_t offset = 0; offset < matrix.quantized_data.size(); offset += bytes)
        {
            qnk_dequantize_block(matrix.dtype, matrix.quantized_data.data() + offset, block.data());
            for (size_t k = 0; k < 32; ++k)
            {
                const size_t column = offset / bytes * 32 + k;
                reference.float32_data[first + column / 256 * 256 + (permuted ? 255 - column % 256 : column % 256)] = block[k];
            }
        }
        for (uint32_t row = 0; row < matrix.shape[0]; ++row)
            reference_bias.float32_data.push_back(bias_ptrs[m] ? projection_value(biases[m], row) : 0.0f);
        const auto single = QnkLinear_vulkan::create(matrix, bias_ptrs[m], device, runtime, OptimizationDefaultFlags);
        const uint64_t size = QnkLinear_vulkan::projection_storage_bytes(matrix, alignment);
        projection_check(single && single->device_bytes() == size, "prepared projection exceeds its storage estimate");
        planned += size;
    }
    reference_bias.shape = {reference.shape[0]};
    projection_check(QnkLinear_vulkan::can_fuse(weights), "compatible mixed formats cannot fuse");
    projection_check(!QnkLinear_vulkan::can_fuse(weights, alignment, planned - 1), "fused projections exceeded the storage binding limit");
    projection_check(!QnkLinear_vulkan::can_fuse(weights, UINT64_MAX), "fused projections accepted overflowing alignment");
    const auto fused = QnkLinear_vulkan::create_fused(weights, bias_ptrs, device, runtime, OptimizationDefaultFlags);
    projection_check(fused && fused->device_bytes() <= planned, "fused projections exceed their resource reservation");
    const auto before = get_vulkan_statistics(runtime);
    for (const size_t rows : {size_t(1), size_t(3), size_t(4), size_t(5), size_t(9)})
    {
        ActivationBuffer input(rows, 256), output;
        for (size_t i = 0; i < input.values().size(); ++i)
            input.row(i / 256)[i % 256] = std::sin(static_cast<float>(i % 971) * 0.013f);
        projection_check(fused->forward(input, output), "fused quantized projection failed");
        projection_compare(reference, &reference_bias, input, output, "mixed quantized fusion");
    }
    const auto after = get_vulkan_statistics(runtime);
    projection_check(after.weight_upload_bytes == before.weight_upload_bytes, "fused projection uploaded weights during forward");
    matrices[1].shape[1] = 512;
    projection_check(!QnkLinear_vulkan::can_fuse(weights), "fused incompatible input widths");
    matrices[1].shape[1] = 256;
    matrices[1].input_column_indices = permuted ? std::vector<uint32_t>{} : matrices[0].input_column_indices;
    if (!permuted) matrices[1].input_column_indices = {0};
    projection_check(!QnkLinear_vulkan::can_fuse(weights), "fused incompatible input permutations");
    matrices[1].input_column_indices = matrices[0].input_column_indices;
    biases[0].shape[0] += 1;
    projection_check(!QnkLinear_vulkan::create_fused(weights, bias_ptrs, device, runtime, OptimizationDefaultFlags), "accepted invalid fused bias");
    matrices[1].dtype = DType::Q3K;
    projection_check(!QnkLinear_vulkan::can_fuse(weights), "fused incompatible native block layouts");
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
        const auto runtime = ncnn::moe::create_vulkan_runtime();
        using ncnn::moe::DType;
        for (const auto dtype : {DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8K,
                                 DType::Q8_0, DType::IQ4NL, DType::IQ4XS, DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::Q5_1})
        {
            ncnn::moe::projection_quantized(device, runtime, dtype, 512, 7);
            ncnn::moe::projection_quantized(device, runtime, dtype, 256, 15, 3, true);
        }
        for (const auto dtype : {DType::Q8_0, DType::IQ4NL, DType::Q5_0})
        {
            ncnn::moe::projection_quantized(device, runtime, dtype, 32, 3);
            ncnn::moe::projection_quantized(device, runtime, dtype, 2880, 32);
        }
        ncnn::moe::projection_quantized_fusion(device, runtime, false);
        ncnn::moe::projection_quantized_fusion(device, runtime, true);
        ncnn::moe::projection_quantized(device, runtime, DType::Q3K, 4096, 33);
        ncnn::moe::projection_quantized(device, runtime, DType::Q8_0, 32, 262145);
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
