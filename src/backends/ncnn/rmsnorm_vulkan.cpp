#include "rmsnorm_vulkan.h"
#include "vulkancontext.h"
#include "ncnn/moe/types.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/activation_rms_norm.comp.hex.h"
#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#endif

namespace ncnn {
namespace moe {

class RmsNorm_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    ncnn::VkMat weight;
    uint32_t columns = 0;
    uint32_t workgroup_size = 0;
    bool add_norm = false;
    mutable std::vector<ncnn::VkMat> bindings;
    std::vector<unsigned char> readonly;
    mutable std::vector<ncnn::vk_constant_type> constants;
#endif
};

RmsNorm_vulkan::RmsNorm_vulkan()
    : d(new Implementation)
{
}

RmsNorm_vulkan::~RmsNorm_vulkan() = default;

std::shared_ptr<RmsNorm_vulkan> RmsNorm_vulkan::create(const TensorData& weight,
                                                       float epsilon,
                                                       float weight_offset,
                                                       std::shared_ptr<VulkanContext> context,
                                                       const ncnn::Option& opt,
                                                       ncnn::VkAllocator* weight_allocator,
                                                       bool add_norm)
{
#if NCNN_MOE_WITH_VULKAN
    if (!context || epsilon <= 0.0f || weight.shape.size() != 1
        || weight.shape[0] == 0 || weight.shape[0] > 0x7fffffff
        || activation_variant(opt) != 0)
        return {};
    const auto& limits = context->device()->info.physicalDeviceProperties().limits;
    const uint32_t workgroup_size = limits.maxComputeWorkGroupInvocations >= 256
                                            && limits.maxComputeWorkGroupSize[0] >= 256
                                        ? 256
                                        : 128;
    if (limits.maxComputeWorkGroupInvocations < workgroup_size
        || limits.maxComputeWorkGroupSize[0] < workgroup_size
        || limits.maxComputeSharedMemorySize < 256 * sizeof(float)
        || static_cast<uint64_t>(weight.shape[0]) * sizeof(float) > limits.maxStorageBufferRange
        || (add_norm
            && (limits.maxPerStageDescriptorStorageBuffers < 5
                || limits.maxDescriptorSetStorageBuffers < 5)))
        return {};
    ncnn::Mat values;
    if (!prepare_float_tensor_upload(weight, values))
        return {};
    auto result = std::shared_ptr<RmsNorm_vulkan>(new RmsNorm_vulkan);
    auto& impl = *result->d;
    impl.context = std::move(context);
    impl.columns = static_cast<uint32_t>(weight.shape[0]);
    impl.workgroup_size = workgroup_size;
    impl.add_norm = add_norm;
    impl.bindings.resize(add_norm ? 5 : 3);
    impl.readonly = add_norm ? std::vector<unsigned char>{1, 1, 0, 1, 0}
                             : std::vector<unsigned char>{1, 1, 0};
    impl.constants.resize(5);
    impl.constants[0].u32 = impl.columns;
    impl.constants[2].f = epsilon;
    impl.constants[3].f = weight_offset;
    impl.constants[4].u32 = workgroup_size;
    ncnn::Option compile_opt = opt;
    constexpr uint32_t required_subgroup_operations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    compile_opt.use_subgroup_ops = opt.use_subgroup_ops
                                   && (impl.context->device()->info.support_subgroup_ops() & required_subgroup_operations)
                                          == required_subgroup_operations;
    const uint64_t shader_variant = compile_opt.use_subgroup_ops ? 1 : 0;
    const uint64_t pipeline_variant = shader_variant | (static_cast<uint64_t>(workgroup_size) << 1);
    static const std::string residual_add_shader = [] {
        std::string source(activation_rms_norm_shader, sizeof(activation_rms_norm_shader) - 1);
        const size_t version_end = source.find('\n');
        if (version_end == std::string::npos)
            return std::string();
        source.insert(version_end + 1, "#define NCNN_RMS_NORM_RESIDUAL_ADD\n");
        return source;
    }();
    if (add_norm && residual_add_shader.empty())
        return {};
    const char* shader_source = add_norm ? residual_add_shader.c_str() : activation_rms_norm_shader;
    const int shader_source_size = add_norm ? static_cast<int>(residual_add_shader.size())
                                            : static_cast<int>(sizeof(activation_rms_norm_shader) - 1);
    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    impl.pipeline = impl.context->find_pipeline(shader_source, pipeline_variant);
    if (!impl.pipeline)
    {
        const auto binary = impl.context->shader_binary(shader_source, shader_source_size,
                                                        compile_opt, shader_variant);
        if (!binary || binary->empty())
            return {};
        auto pipeline = std::make_unique<ncnn::Pipeline>(impl.context->device());
        pipeline->set_local_size_xyz(static_cast<int>(workgroup_size), 1, 1);
        std::vector<ncnn::vk_specialization_type> specializations(1);
        specializations[0].u32 = compile_opt.use_subgroup_ops ? 1 : 0;
        if (pipeline->create(binary->data(), binary->size() * sizeof(uint32_t), specializations) != 0)
            return {};
        impl.pipeline = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = impl.context](ncnn::Pipeline* value) mutable {
            {
                const std::lock_guard<std::mutex> lock(context->command_mutex());
                delete value;
            }
            // Weak cache entries retain the deleter control block after disposal.
            context.reset();
        });
        impl.context->cache_pipeline(shader_source, pipeline_variant, impl.pipeline);
    }
    ncnn::VkWeightStagingAllocator staging(impl.context->device());
    ncnn::Option upload_opt = opt;
    upload_opt.blob_vkallocator = weight_allocator ? weight_allocator : impl.context->blob_allocator();
    upload_opt.workspace_vkallocator = upload_opt.blob_vkallocator;
    upload_opt.staging_vkallocator = &staging;
    ncnn::VkTransfer cmd(impl.context->device());
    cmd.record_upload(values, impl.weight, upload_opt);
    if (impl.weight.empty() || cmd.submit_and_wait() != 0)
        return {};
    impl.weight.data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    impl.weight.data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    return result;
#else
    (void)weight;
    (void)epsilon;
    (void)weight_offset;
    (void)context;
    (void)opt;
    (void)weight_allocator;
    (void)add_norm;
    return {};
#endif
}

#if NCNN_MOE_WITH_VULKAN
static bool rms_norm_buffer_supported(const ncnn::VkMat& value, uint64_t required_bytes,
                                      const VkPhysicalDeviceLimits& limits,
                                      const ncnn::VulkanDevice* device)
{
    if (!value.data || !value.allocator || value.allocator->vkdev != device
        || value.buffer_offset() < value.data->offset)
        return false;
    const size_t offset = value.buffer_offset() - value.data->offset;
    const size_t descriptor_bytes = value.buffer_capacity();
    return descriptor_bytes >= required_bytes
           && descriptor_bytes <= limits.maxStorageBufferRange
           && offset <= value.data->capacity
           && descriptor_bytes <= value.data->capacity - offset
           && value.buffer_offset() % std::max<VkDeviceSize>(1, limits.minStorageBufferOffsetAlignment) == 0;
}

static bool rms_norm_ranges_overlap(const ncnn::VkMat& first, uint64_t first_bytes,
                                    const ncnn::VkMat& second, uint64_t second_bytes)
{
    if (first.empty() || second.empty() || !first.data || !second.data
        || first.buffer() != second.buffer())
        return false;
    const uint64_t first_offset = first.buffer_offset();
    const uint64_t second_offset = second.buffer_offset();
    return first_offset <= second_offset ? second_offset - first_offset < first_bytes
                                         : first_offset - second_offset < second_bytes;
}
#endif

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& RmsNorm_vulkan::vulkan_context() const noexcept
{
    return d->context;
}
#endif

bool RmsNorm_vulkan::record(const ncnn::VkMat& input,
                            ncnn::VkMat& output,
                            ncnn::VkCompute& cmd,
                            const ncnn::VkMat* branch,
                            ncnn::VkMat* residual) const
{
#if NCNN_MOE_WITH_VULKAN
    const auto& impl = *d;
    if (&input == &output || !impl.context || !impl.pipeline || input.empty() || input.dims != 2
        || input.elempack != 1 || input.elemsize != sizeof(float)
        || input.w != static_cast<int>(impl.columns) || input.h <= 0)
        return false;
    if (impl.add_norm)
    {
        if (!branch || !residual)
            return false;
        if (branch == &output || residual == &output
            || residual == branch || residual == &input)
            return false;
    }
    else if (branch || residual)
        return false;
    if (impl.add_norm
        && (branch->empty() || branch->dims != 2 || branch->elempack != 1
            || branch->elemsize != sizeof(float) || branch->w != input.w || branch->h != input.h))
        return false;
#if NCNN_BATCH
    if (input.n != 1 || (impl.add_norm && branch->n != 1))
        return false;
#endif
    const auto& limits = impl.context->device()->info.physicalDeviceProperties().limits;
    const uint64_t elements = static_cast<uint64_t>(impl.columns) * input.h;
    const uint64_t required_bytes = elements * sizeof(float);
    if (static_cast<uint32_t>(input.h) > limits.maxComputeWorkGroupCount[1]
        || elements > std::numeric_limits<uint32_t>::max()
        || !rms_norm_buffer_supported(input, required_bytes, limits, impl.context->device())
        || (impl.add_norm
            && !rms_norm_buffer_supported(*branch, required_bytes, limits, impl.context->device())))
        return false;
    // VkMat::create preserves an existing allocation with matching shape.
    // A caller may pass a separate view of input as its output variable;
    // release that view before allocating so residual input stays intact.
    ncnn::VkMat destination = output;
    ncnn::VkMat residual_destination;
    if (impl.add_norm)
        residual_destination = *residual;
    const uint64_t weight_bytes = static_cast<uint64_t>(impl.columns) * sizeof(float);
    const auto overlaps_source = [&](const ncnn::VkMat& value) {
        return rms_norm_ranges_overlap(value, required_bytes, input, required_bytes)
               || rms_norm_ranges_overlap(value, required_bytes, impl.weight, weight_bytes)
               || (impl.add_norm
                   && rms_norm_ranges_overlap(value, required_bytes, *branch, required_bytes));
    };
    if (overlaps_source(destination)
        || (impl.add_norm
            && rms_norm_ranges_overlap(destination, required_bytes, residual_destination, required_bytes)))
        destination.release();
    destination.create(input.w, input.h, sizeof(float), impl.context->blob_allocator());
    if (destination.empty() || !rms_norm_buffer_supported(destination, required_bytes, limits, impl.context->device())
        || overlaps_source(destination))
        return false;
    if (impl.add_norm)
    {
        if (overlaps_source(residual_destination)
            || rms_norm_ranges_overlap(residual_destination, required_bytes, destination, required_bytes))
            residual_destination.release();
        residual_destination.create(input.w, input.h, sizeof(float), impl.context->blob_allocator());
        if (residual_destination.empty()
            || !rms_norm_buffer_supported(residual_destination, required_bytes, limits, impl.context->device())
            || overlaps_source(residual_destination)
            || rms_norm_ranges_overlap(residual_destination, required_bytes, destination, required_bytes))
            return false;
    }
    auto& bindings = impl.bindings;
    const auto& readonly = impl.readonly;
    auto& constants = impl.constants;
    try
    {
        bindings[0] = input;
        bindings[1] = impl.weight;
        bindings[2] = destination;
        if (impl.add_norm)
        {
            bindings[3] = *branch;
            bindings[4] = residual_destination;
        }
        constants[1].u32 = static_cast<uint32_t>(input.h);
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(impl.workgroup_size);
        dispatcher.h = input.h;
        dispatcher.c = 1;
        cmd.record_pipeline_readonly(impl.pipeline.get(), bindings, readonly, constants, dispatcher);
    }
    catch (...)
    {
        for (auto& binding : bindings)
            binding.release();
        throw;
    }
    for (auto& binding : bindings)
        binding.release();
    output = destination;
    if (impl.add_norm)
        *residual = residual_destination;
    return true;
#else
    (void)input;
    (void)output;
    (void)cmd;
    (void)branch;
    (void)residual;
    return false;
#endif
}

} // namespace moe
} // namespace ncnn
