#include "backends/ncnn/rmsnorm_vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "kernels/bfloat16.h"
#include "kernels/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static void norm_check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

class NormWeightAllocator final : public ncnn::VkWeightAllocator
{
public:
    explicit NormWeightAllocator(const ncnn::VulkanDevice* device)
        : ncnn::VkWeightAllocator(device)
    {
    }
    ncnn::VkBufferMemory* fastMalloc(size_t bytes) override
    {
        auto* result = ncnn::VkWeightAllocator::fastMalloc(bytes);
        if (result) ++allocations;
        return result;
    }
    void fastFree(ncnn::VkBufferMemory* memory) override
    {
        if (memory) ++releases;
        ncnn::VkWeightAllocator::fastFree(memory);
    }
    size_t allocations = 0;
    size_t releases = 0;
};

static ncnn::Option norm_option(const std::shared_ptr<VulkanContext>& context, bool subgroup)
{
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.use_subgroup_ops = subgroup;
    option.blob_vkallocator = context->blob_allocator();
    option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    return option;
}

static ActivationBuffer norm_reference(const ActivationBuffer& input, const TensorData& weight,
                                       float epsilon, float weight_offset)
{
    ActivationBuffer result(input.rows(), input.columns());
    for (size_t row = 0; row < input.rows(); ++row)
    {
        double squares = 0.0;
        for (uint32_t column = 0; column < input.columns(); ++column)
            squares += static_cast<double>(input.row(row)[column]) * input.row(row)[column];
        const double scale = 1.0 / std::sqrt(squares / input.columns() + epsilon);
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            const float value = weight.dtype == DType::Float32 ? weight.float32_data[column]
                                                               : bfloat16_to_float(weight.bfloat16_data[column]);
            result.row(row)[column] = static_cast<float>(input.row(row)[column] * scale * (value + weight_offset));
        }
    }
    return result;
}

static void norm_compare(const ActivationBuffer& actual, const ActivationBuffer& expected, const std::string& label)
{
    norm_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + ": shape mismatch");
    for (size_t index = 0; index < actual.values().size(); ++index)
    {
        const float got = actual.values()[index];
        const float want = expected.values()[index];
        const double tolerance = 2e-6 + 5e-6 * std::abs(static_cast<double>(want));
        norm_check(std::isfinite(got) && std::abs(static_cast<double>(got) - want) <= tolerance,
                   label + ": FP64 reference mismatch at " + std::to_string(index)
                       + " actual=" + std::to_string(got) + " expected=" + std::to_string(want));
    }
}

static void norm_case(const std::shared_ptr<VulkanContext>& context, const VulkanRuntimePtr& runtime,
                      uint32_t columns, uint32_t rows, DType weight_type, bool subgroup,
                      float epsilon, float weight_offset, bool use_offset)
{
    TensorData weight;
    weight.dtype = weight_type;
    weight.shape = {columns};
    for (uint32_t column = 0; column < columns; ++column)
    {
        const float value = 0.9f + std::sin(static_cast<float>((column * 13) % 383) * 0.037f) * 0.7f;
        if (weight_type == DType::Float32)
            weight.float32_data.push_back(value);
        else
            weight.bfloat16_data.push_back(float_to_bfloat16(value));
    }
    ActivationBuffer input(rows, columns);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
        {
            const float scale = columns == 33 ? 0.0001f : 1.3f;
            input.row(row)[column] = rows > 1 && row + 1 == rows
                                         ? 0.0f
                                         : std::sin(static_cast<float>((column * 29 + row * 19 + 7) % 1021) * 0.023f) * scale;
        }
    const auto reference = norm_reference(input, weight, epsilon, weight_offset);
    const auto second_reference = norm_reference(reference, weight, epsilon, weight_offset);
    const auto option = norm_option(context, subgroup);
    NormWeightAllocator weight_allocator(context->device());
    auto norm = RmsNorm_vulkan::create(weight, epsilon, weight_offset, context, option, &weight_allocator);
    norm_check(static_cast<bool>(norm) && weight_allocator.allocations == 1 && weight_allocator.releases == 0,
               "RMSNorm did not retain exactly one weight allocation in the supplied allocator");

    size_t prefix_bytes = 0;
#if NCNN_BATCH
    if (use_offset)
        prefix_bytes = std::max<size_t>(16, context->device()->info.buffer_offset_alignment());
#else
    (void)use_offset;
#endif
    ActivationBuffer backing(1, static_cast<uint32_t>(prefix_bytes / sizeof(float) + input.values().size()));
    std::fill_n(backing.row(0), backing.columns(), -23.0f);
    std::copy(input.values().begin(), input.values().end(), backing.row(0) + prefix_bytes / sizeof(float));
    ncnn::VkMat input_staging, first_staging, second_staging, preserved_staging;
    norm_check(fill_staging_upload(backing, input_staging, context->staging_allocator())
                   && prepare_staging_batch(first_staging, rows, columns, context->staging_allocator())
                   && prepare_staging_batch(second_staging, rows, columns, context->staging_allocator())
                   && prepare_staging_batch(preserved_staging, rows, columns, context->staging_allocator()),
               "RMSNorm staging allocation failed");
    ActivationBuffer first_result(rows, columns), second_result(rows, columns), preserved(rows, columns);
    const auto before = get_vulkan_statistics(runtime);
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device());
        ncnn::VkMat device_backing;
        norm_check(record_prepared_staging_upload(input_staging, 1, device_backing, command,
                                                  context->device(), option, DType::Float32),
                   "RMSNorm input upload recording failed");
        ncnn::VkMat device_input = device_backing;
        device_input.w = static_cast<int>(columns);
        device_input.h = static_cast<int>(rows);
        device_input.cstep = static_cast<size_t>(columns) * rows;
#if NCNN_BATCH
        device_input.nstep = device_input.cstep;
        device_input.offset += prefix_bytes;
#endif
        ncnn::VkMat first = device_input;
        const auto* source_allocation = device_input.data;
        norm_check(!norm->record(device_input, device_input, command) && device_input.data == source_allocation,
                   "RMSNorm accepted the same input/output variable");
        const auto reject = [&](const ncnn::VkMat& invalid, const char* reason) {
            norm_check(!norm->record(invalid, first, command) && first.data == source_allocation,
                       std::string("RMSNorm accepted or published an invalid descriptor: ") + reason);
        };
        auto invalid = device_input;
        invalid.w += 1;
        reject(invalid, "width");
        invalid = device_input;
        invalid.elemsize = sizeof(uint16_t);
        reject(invalid, "FP16 storage");
        invalid = device_input;
        invalid.elempack = 4;
        reject(invalid, "packed rows");
#if NCNN_BATCH
        invalid = device_input;
        invalid.nstep = 0;
        reject(invalid, "truncated range");
        invalid = device_input;
        invalid.n = 2;
        reject(invalid, "batch count");
        invalid = device_input;
        invalid.nstep = device_input.data->capacity / sizeof(float) + 1;
        reject(invalid, "range beyond allocation");
        if (context->device()->info.buffer_offset_alignment() > sizeof(float))
        {
            invalid = device_input;
            invalid.offset += sizeof(float);
            reject(invalid, "unaligned offset");
        }
#endif
        norm_check(norm->record(device_input, first, command) && first.data != source_allocation,
                   "RMSNorm alias output overwrote the residual input");
        const size_t logical_bytes = static_cast<size_t>(rows) * columns * sizeof(float);
        norm_check(first.dims == 2 && first.w == static_cast<int>(columns) && first.h == static_cast<int>(rows)
                       && first.elempack == 1 && first.elemsize == sizeof(float)
                       && first.buffer_capacity() >= logical_bytes && first.buffer_capacity() <= first.data->capacity,
                   "RMSNorm output descriptor does not cover its FP32 values");
#if NCNN_BATCH
        norm_check(first.n == 1 && first.nstep == first.cstep, "RMSNorm output has invalid batch stride");
#endif
        ncnn::VkMat second;
        norm_check(norm->record(first, second, command) && second.data != first.data,
                   "RMSNorm dependent recording failed");
        norm_check(record_prepared_activation_staging_download(first, rows, columns, first_staging, command, context->device(), option)
                       && record_prepared_activation_staging_download(second, rows, columns, second_staging, command, context->device(), option)
                       && record_prepared_activation_staging_download(device_input, rows, columns, preserved_staging, command, context->device(), option)
                       && submit_compute_and_wait(command, context->device()) == 0
                       && copy_staging_to_cpu_batch(first_staging, first_result)
                       && copy_staging_to_cpu_batch(second_staging, second_result)
                       && copy_staging_to_cpu_batch(preserved_staging, preserved),
                   "RMSNorm submission or output download failed");
    }
    const auto after = get_vulkan_statistics(runtime);
    norm_check(after.compute_submissions == before.compute_submissions && after.dispatches == before.dispatches
                   && after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads,
               "RMSNorm recorder published counters or performed its own submission");
    norm_check(std::equal(input.values().begin(), input.values().end(), preserved.values().begin()),
               "RMSNorm changed the residual input on GPU");
    const std::string label = "K=" + std::to_string(columns) + " rows=" + std::to_string(rows)
                              + " subgroup=" + std::to_string(subgroup) + " offset=" + std::to_string(prefix_bytes);
    norm_compare(first_result, reference, label);
    norm_compare(second_result, second_reference, label + " dependent");
    norm.reset();
    norm_check(weight_allocator.allocations == 1 && weight_allocator.releases == 1,
               "RMSNorm did not release its externally allocated weight before allocator destruction");
}

static void norm_device(uint32_t device)
{
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(device, runtime, OptimizationDefaultFlags);
    norm_check(static_cast<bool>(context), "cannot acquire RMSNorm Context");
    struct Case
    {
        uint32_t columns, rows;
        DType dtype;
    };
    // Representative scalar, aligned, tail and long rows. Both reduction
    // capabilities run against the independent FP64 reference for each case.
    const std::array<Case, 10> cases = {{{1, 1, DType::Float32}, {33, 3, DType::BFloat16}, {128, 1, DType::Float32}, {129, 3, DType::Float32}, {255, 1, DType::BFloat16}, {256, 3, DType::BFloat16}, {257, 3, DType::Float32}, {2880, 14, DType::BFloat16}, {4097, 3, DType::Float32}, {8192, 1, DType::BFloat16}}};
    size_t index = 0;
    for (const auto& test : cases)
    {
        const float epsilon = index % 2 == 0 ? 1e-5f : 1e-3f;
        const float offset = index % 3 == 0 ? 0.0f : index % 3 == 1 ? 1.0f
                                                                    : -0.5f;
        for (const bool subgroup : {false, true})
            norm_case(context, runtime, test.columns, test.rows, test.dtype, subgroup, epsilon, offset, index % 2 != 0);
        ++index;
    }
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
        if (argc == 3 && std::string(argv[1]) == "--device")
            ncnn::moe::norm_device(static_cast<uint32_t>(std::stoul(argv[2])));
        else
            for (uint32_t device = 0; device < ncnn::moe::get_gpu_count(); ++device)
                ncnn::moe::norm_device(device);
        std::cout << "RMSNorm FP64, tails, offsets, aliases, descriptors and allocator lifetime passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "RMSNorm test failed: " << error.what() << '\n';
        return 1;
    }
#else
    (void)argc;
    (void)argv;
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
