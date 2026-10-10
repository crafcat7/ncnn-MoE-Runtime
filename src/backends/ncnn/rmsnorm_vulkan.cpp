#include "rmsnorm_vulkan.h"
#include "vulkancontext.h"
#include "ncnn/moe/types.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/activation_rms_norm.comp.hex.h"
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
    ncnn::Option opt;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    ncnn::VkMat weight;
    uint32_t columns = 0;
    float epsilon = 0.0f;
    float weight_offset = 0.0f;
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
                                                       const ncnn::Option& opt)
{
#if NCNN_MOE_WITH_VULKAN
    if (!context || epsilon <= 0.0f || weight.shape.size() != 1
        || weight.shape[0] == 0 || weight.shape[0] > 0x7fffffff
        || vulkan_activation_storage_variant(opt) != 0)
        return {};
    ncnn::Mat values;
    if (!prepare_float_tensor_upload(weight, values))
        return {};
    auto result = std::shared_ptr<RmsNorm_vulkan>(new RmsNorm_vulkan);
    auto& impl = *result->d;
    impl.context = std::move(context);
    impl.opt = opt;
    impl.columns = static_cast<uint32_t>(weight.shape[0]);
    impl.epsilon = epsilon;
    impl.weight_offset = weight_offset;
    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    impl.pipeline = impl.context->find_pipeline(activation_rms_norm_shader, 0);
    if (!impl.pipeline)
    {
        const auto binary = impl.context->shader_binary(activation_rms_norm_shader,
                                                        sizeof(activation_rms_norm_shader) - 1,
                                                        opt, 0);
        if (!binary || binary->empty())
            return {};
        auto pipeline = std::make_unique<ncnn::Pipeline>(impl.context->device());
        pipeline->set_optimal_local_size_xyz(32, 1, 1);
        if (pipeline->create(binary->data(), binary->size() * sizeof(uint32_t), {}) != 0)
            return {};
        impl.pipeline = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = impl.context](ncnn::Pipeline* value) mutable {
            {
                const std::lock_guard<std::mutex> lock(context->command_mutex());
                delete value;
            }
            // Weak cache entries retain the deleter control block after disposal.
            context.reset();
        });
        impl.context->cache_pipeline(activation_rms_norm_shader, 0, impl.pipeline);
    }
    ncnn::VkWeightStagingAllocator staging(impl.context->device());
    ncnn::Option upload_opt = opt;
    upload_opt.blob_vkallocator = impl.context->blob_allocator();
    upload_opt.workspace_vkallocator = impl.context->blob_allocator();
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
    return {};
#endif
}

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
    if (!impl.context || !impl.pipeline || input.empty() || input.dims != 2
        || input.elempack != 1 || input.elemsize != sizeof(float)
        || input.w != static_cast<int>(impl.columns) || input.h <= 0)
        return false;
    output.create(input.w, input.h, sizeof(float), impl.context->blob_allocator());
    if (output.empty())
        return false;
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = impl.columns;
    constants[1].u32 = input.h;
    constants[2].f = impl.epsilon;
    constants[3].f = impl.weight_offset;
    ncnn::VkMat dispatcher;
    dispatcher.w = 32;
    dispatcher.h = input.h;
    dispatcher.c = 1;
    if (impl.weight.data)
    {
        impl.weight.data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        impl.weight.data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    cmd.record_pipeline_readonly(impl.pipeline.get(), {input, impl.weight, output}, {1, 1, 0}, constants, dispatcher);
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
