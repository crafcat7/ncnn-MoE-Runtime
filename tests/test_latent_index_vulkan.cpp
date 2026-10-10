#include "backends/ncnn/vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "kernels/activationbuffer.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/latent_attention_index_sort.comp.hex.h"
#include "kernels/vulkan/latent_attention_index_merge.comp.hex.h"
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static void index_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static std::unique_ptr<ncnn::Pipeline> index_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                      const ncnn::Option& option, const char* source, int size)
{
    const auto binary = context->shader_binary(source, size, option, 0);
    index_check(binary && !binary->empty(), "compile parallel index shader");
    auto result = std::make_unique<ncnn::Pipeline>(context->device());
    result->set_local_size_xyz(128, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    index_check(result->create(binary->data(), binary->size() * sizeof(uint32_t), specializations) == 0,
                "create parallel index shader pipeline");
    return result;
}

static void index_test(const std::vector<float>& values, uint32_t top_k, const std::string& label)
{
    const auto runtime = create_vulkan_runtime();
    const auto context = VulkanContext::acquire(get_default_gpu_index(), runtime, OptimizationDefaultFlags);
    index_check(static_cast<bool>(context), label + ": Vulkan context");
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_packing_layout = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.blob_vkallocator = context->blob_allocator();
    option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    auto sort = index_pipeline(context, option, latent_attention_index_sort_shader,
                               static_cast<int>(sizeof(latent_attention_index_sort_shader) - 1));
    auto merge = index_pipeline(context, option, latent_attention_index_merge_shader,
                                static_cast<int>(sizeof(latent_attention_index_merge_shader) - 1));
    const uint32_t count = static_cast<uint32_t>(values.size());
    ActivationBuffer host(1, std::max(1u, count));
    std::copy(values.begin(), values.end(), host.row(0));
    ncnn::VkMat staging;
    ncnn::VkMat scores;
    ncnn::VkCompute command(context->device(), context->command_optimization_flags());
    index_check(fill_staging_upload(host, staging, context->staging_allocator()), label + ": scores staging");
    index_check(record_prepared_staging_upload(staging, 1, scores, command, context->device(), option), label + ": scores upload");
    uint32_t runs = std::max(1u, (count + 1023u) / 1024u);
    ncnn::VkMat temporary[2];
    temporary[0].create(static_cast<int>(runs * top_k), sizeof(float), context->blob_allocator());
    temporary[1].create(static_cast<int>(((runs + 1u) / 2u) * top_k), sizeof(float), context->blob_allocator());
    index_check(!temporary[0].empty() && !temporary[1].empty(), label + ": parallel index scratch");
    ncnn::VkMat sorted = temporary[0];
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].u32 = count;
    constants[1].u32 = top_k;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(runs * 128u);
    dispatcher.h = 1;
    dispatcher.c = 1;
    command.record_pipeline(sort.get(), {scores, sorted}, constants, dispatcher);
    uint32_t span = 1024;
    uint32_t next = 1;
    while (runs > 1)
    {
        ncnn::VkMat merged = temporary[next];
        constants.resize(4);
        constants[0].u32 = count;
        constants[1].u32 = top_k;
        constants[2].u32 = runs;
        constants[3].u32 = span;
        dispatcher.w = static_cast<int>(runs * top_k);
        command.record_pipeline(merge.get(), {scores, sorted, merged}, constants, dispatcher);
        sorted = merged;
        runs = (runs + 1u) / 2u;
        span *= 2u;
        next ^= 1u;
    }
    ncnn::Mat observed;
    command.record_download(sorted, observed, option);
    index_check(submit_compute_and_wait(command, context->device()) == 0, label + ": parallel index submit");
    index_check(!observed.empty() && observed.elempack == 1 && observed.elemsize == sizeof(uint32_t), label + ": index download");
    std::vector<uint32_t> expected(count);
    std::iota(expected.begin(), expected.end(), 0u);
    std::sort(expected.begin(), expected.end(), [&](uint32_t left, uint32_t right) {
        return values[left] > values[right] || (values[left] == values[right] && left < right);
    });
    const auto* indices = static_cast<const uint32_t*>(observed.data);
    for (uint32_t rank = 0; rank < top_k; ++rank)
    {
        const uint32_t reference = rank < count ? expected[rank] : std::numeric_limits<uint32_t>::max();
        index_check(indices[rank] == reference, label + ": rank " + std::to_string(rank)
                                                    + " actual=" + std::to_string(indices[rank]) + " expected=" + std::to_string(reference));
    }
}

static std::vector<float> index_values(uint32_t count)
{
    std::vector<float> result(count);
    for (uint32_t index = 0; index < count; ++index)
        result[index] = static_cast<float>(static_cast<int>((index * 37u + index / 31u) % 997u) - 498) * 0.03125f;
    return result;
}

static void index_tests()
{
    index_test({}, 512, "empty candidate padding");
    index_test({-4.0f}, 1, "one candidate");
    index_test(index_values(127), 512, "small candidate padding");
    index_test(index_values(513), 512, "single block K512");
    index_test(index_values(1025), 512, "two blocks odd tail");
    index_test(index_values(3001), 512, "three blocks hierarchical merge");
    index_test(index_values(8193), 512, "nine blocks hierarchical merge");
    index_test(index_values(4097), 1024, "maximum parallel K and odd run");
    index_test(index_values(5127), 1, "single winner across blocks");
    index_test(std::vector<float>(3001, 0.0f), 512, "cross block equal score stable indices");
    auto extremes = index_values(3001);
    extremes[0] = -std::numeric_limits<float>::infinity();
    extremes[1000] = std::numeric_limits<float>::max();
    extremes[1023] = std::numeric_limits<float>::infinity();
    extremes[1024] = std::numeric_limits<float>::infinity();
    extremes[2048] = std::numeric_limits<float>::max();
    extremes[3000] = std::numeric_limits<float>::lowest();
    index_test(extremes, 512, "maximum finite values and infinities");
}
#endif

} // namespace moe
} // namespace ncnn

int main()
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::moe::get_gpu_count() == 0)
    {
        std::cout << "SKIP: Vulkan GPU unavailable\n";
        return 77;
    }
    try
    {
        ncnn::moe::index_tests();
        std::cout << "Parallel latent index Vulkan tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Parallel latent index Vulkan test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
