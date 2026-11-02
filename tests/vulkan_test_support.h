#ifndef NCNN_MOE_VULKAN_TEST_SUPPORT_H
#define NCNN_MOE_VULKAN_TEST_SUPPORT_H

#include "backends/ncnn/vulkancontext.h"
#include "kernels/activationbuffer.h"

#if NCNN_MOE_WITH_VULKAN
#include <option.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ncnn {
namespace moe {

static void segment_check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

static void segment_compare(std::span<const float> actual, std::span<const float> expected, const std::string& context)
{
    segment_check(actual.size() == expected.size(), context + " shape mismatch");
    float scale = 0.0f;
    for (const float value : expected) scale = std::max(scale, std::abs(value));
    // Both use FP32 activations and BF16 KV; weight kernels differ in reduction order.
    const float tolerance = 2e-4f + 2e-3f * scale;
    for (size_t index = 0; index < actual.size(); ++index)
        segment_check(std::isfinite(actual[index]) && std::abs(actual[index] - expected[index]) <= tolerance,
                      context + " numerical mismatch at " + std::to_string(index) + ": " + std::to_string(actual[index])
                          + " vs " + std::to_string(expected[index]) + ", tolerance " + std::to_string(tolerance));
}

#if NCNN_MOE_WITH_VULKAN
static constexpr uint64_t segment_flags = OptimizationDefaultFlags & ~OptimizationCpuFastSilu;

static ncnn::Option vulkan_test_option(const VulkanContext& context)
{
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_packing_layout = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.blob_vkallocator = context.blob_allocator();
    option.workspace_vkallocator = context.blob_allocator();
    option.staging_vkallocator = context.staging_allocator();
    return option;
}

// Raw VkMat fixtures use one counted staging upload and submit. This is
// intentionally separate from the DeviceTensor CommandGraph fixture.
static bool vulkan_test_upload_raw(const std::shared_ptr<VulkanContext>& context,
                                   const ActivationBuffer& source, ncnn::VkMat& output)
{
    if (!context) return false;
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const auto option = vulkan_test_option(*context);
    ncnn::VkCompute command(context->device());
    ncnn::VkMat staging, uploaded;
    if (!fill_staging_upload(source, staging, option.staging_vkallocator)
        || !record_mapped_upload(staging, uploaded, command, option)
        || submit_compute_and_wait(command, context->device()) != 0)
        return false;
    ++context->runtime_state().compute_submissions;
    ++context->runtime_state().batch_uploads;
    output = std::move(uploaded);
    return true;
}
#endif

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_VULKAN_TEST_SUPPORT_H
