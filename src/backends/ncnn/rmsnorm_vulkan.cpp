#include "rmsnorm_vulkan.h"
#include "vulkancontext.h"
#include "ncnn/moe/types.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/activation_rms_norm.comp.hex.h"
#include <algorithm>
#include <limits>
#include <mutex>
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
                                                       ncnn::VkAllocator* weight_allocator)
{
#if NCNN_MOE_WITH_VULKAN
    if (!context || epsilon <= 0.0f || weight.shape.size() != 1
        || weight.shape[0] == 0 || weight.shape[0] > 0x7fffffff
        || vulkan_activation_storage_variant(opt) != 0)
        return {};
    const auto& limits = context->device()->info.physicalDeviceProperties().limits;
    const uint32_t workgroup_size = limits.maxComputeWorkGroupInvocations >= 256
                                            && limits.maxComputeWorkGroupSize[0] >= 256
                                        ? 256
                                        : 128;
    if (limits.maxComputeWorkGroupInvocations < workgroup_size
        || limits.maxComputeWorkGroupSize[0] < workgroup_size
        || limits.maxComputeSharedMemorySize < 256 * sizeof(float)
        || static_cast<uint64_t>(weight.shape[0]) * sizeof(float) > limits.maxStorageBufferRange)
        return {};
    ncnn::Mat values;
    if (!prepare_float_tensor_upload(weight, values))
        return {};
    auto result = std::shared_ptr<RmsNorm_vulkan>(new RmsNorm_vulkan);
    auto& impl = *result->d;
    impl.context = std::move(context);
    impl.columns = static_cast<uint32_t>(weight.shape[0]);
    impl.workgroup_size = workgroup_size;
    impl.bindings.resize(3);
    impl.readonly = {1, 1, 0};
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
    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    impl.pipeline = impl.context->find_pipeline(activation_rms_norm_shader, pipeline_variant);
    if (!impl.pipeline)
    {
        const auto binary = impl.context->shader_binary(activation_rms_norm_shader,
                                                        sizeof(activation_rms_norm_shader) - 1,
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
        impl.context->cache_pipeline(activation_rms_norm_shader, pipeline_variant, impl.pipeline);
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
    return result;
#else
    (void)weight;
    (void)epsilon;
    (void)weight_offset;
    (void)context;
    (void)opt;
    (void)weight_allocator;
    return {};
#endif
}

#if NCNN_MOE_WITH_VULKAN
static bool rms_norm_buffer_supported(const ncnn::VkMat& value, uint64_t required_bytes,
                                      const VkPhysicalDeviceLimits& limits)
{
    if (!value.data || value.buffer_offset() < value.data->offset)
        return false;
    const size_t offset = value.buffer_offset() - value.data->offset;
    const size_t descriptor_bytes = value.buffer_capacity();
    return descriptor_bytes >= required_bytes
           && descriptor_bytes <= limits.maxStorageBufferRange
           && offset <= value.data->capacity
           && descriptor_bytes <= value.data->capacity - offset
           && value.buffer_offset() % std::max<VkDeviceSize>(1, limits.minStorageBufferOffsetAlignment) == 0;
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
                            ncnn::VkCompute& cmd) const
{
#if NCNN_MOE_WITH_VULKAN
    const auto& impl = *d;
    if (&input == &output || !impl.context || !impl.pipeline || input.empty() || input.dims != 2
        || input.elempack != 1 || input.elemsize != sizeof(float)
        || input.w != static_cast<int>(impl.columns) || input.h <= 0)
        return false;
#if NCNN_BATCH
    if (input.n != 1)
        return false;
#endif
    const auto& limits = impl.context->device()->info.physicalDeviceProperties().limits;
    const uint64_t elements = static_cast<uint64_t>(impl.columns) * input.h;
    const uint64_t required_bytes = elements * sizeof(float);
    if (static_cast<uint32_t>(input.h) > limits.maxComputeWorkGroupCount[1]
        || elements > std::numeric_limits<uint32_t>::max()
        || !rms_norm_buffer_supported(input, required_bytes, limits))
        return false;
    // VkMat::create preserves an existing allocation with matching shape.
    // A caller may pass a separate view of input as its output variable;
    // release that view before allocating so residual input stays intact.
    ncnn::VkMat destination = output;
    if (destination.data == input.data)
        destination.release();
    destination.create(input.w, input.h, sizeof(float), impl.context->blob_allocator());
    if (destination.empty() || !rms_norm_buffer_supported(destination, required_bytes, limits))
        return false;
    auto& bindings = impl.bindings;
    const auto& readonly = impl.readonly;
    auto& constants = impl.constants;
    try
    {
        bindings[0] = input;
        bindings[1] = impl.weight;
        bindings[2] = destination;
        constants[1].u32 = static_cast<uint32_t>(input.h);
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(impl.workgroup_size);
        dispatcher.h = input.h;
        dispatcher.c = 1;
        if (impl.weight.data)
        {
            impl.weight.data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            impl.weight.data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
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
    return true;
#else
    (void)input;
    (void)output;
    (void)cmd;
    return false;
#endif
}

} // namespace moe
} // namespace ncnn
