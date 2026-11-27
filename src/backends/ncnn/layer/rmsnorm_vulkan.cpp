#include "backends/ncnn/layer/rmsnorm_vulkan.h"

#include "backends/ncnn/core/context.h"
#include "kernels/vulkan/activation_rms_norm.comp.hex.h"
#include "ncnn/moe/types.h"

#include <mutex>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

class RmsNorm_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<ncnn::Pipeline> pipe[2]; // 0: fp32 out, 1: bf16 out
    ncnn::VkMat weight;
    uint32_t columns = 0;
    uint32_t workgroup_size = 0;
    bool add_norm = false;
    mutable std::vector<ncnn::VkMat> bindings;
    std::vector<unsigned char> readonly;
    mutable std::vector<ncnn::vk_constant_type> constants;
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
    const auto& limits = context->device()->info.physicalDeviceProperties().limits;
    const uint32_t workgroup_size = limits.maxComputeWorkGroupInvocations >= 256
                                            && limits.maxComputeWorkGroupSize[0] >= 256
                                        ? 256
                                        : 128;

    ncnn::Mat values;
    if (!prepare_float_upload(weight, values))
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

    constexpr uint32_t required_subgroup_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    ncnn::Option compile_opt = opt;
    compile_opt.use_subgroup_ops = opt.use_subgroup_ops
                                   && (impl.context->device()->info.support_subgroup_ops() & required_subgroup_ops)
                                          == required_subgroup_ops;

    const uint32_t aligned4 = (impl.columns & 3u) == 0u ? 1u : 0u;
    const uint64_t shader_variant = compile_opt.use_subgroup_ops ? 1 : 0;
    const uint64_t pipe_variant = shader_variant | (static_cast<uint64_t>(workgroup_size) << 1)
                                  | (static_cast<uint64_t>(aligned4) << 16);
    std::vector<ncnn::vk_specialization_type> specs(2);
    specs[0].u32 = compile_opt.use_subgroup_ops ? 1u : 0u;
    specs[1].u32 = aligned4;

    const int base_len = static_cast<int>(sizeof(activation_rms_norm_shader) - 1);
    static const std::string variants[4] = {
        std::string(activation_rms_norm_shader, base_len),
        prefix_shader(activation_rms_norm_shader, base_len, "#define NCNN_RMS_NORM_RESIDUAL_ADD\n"),
        prefix_shader(activation_rms_norm_shader, base_len, "#define NCNN_RMS_NORM_BF16_OUT\n"),
        prefix_shader(activation_rms_norm_shader, base_len,
                      "#define NCNN_RMS_NORM_RESIDUAL_ADD\n#define NCNN_RMS_NORM_BF16_OUT\n"),
    };

    const std::string& f32_src = variants[add_norm ? 1 : 0];
    if (f32_src.empty())
        return {};

    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    if (!create_pipeline(impl.context, f32_src.c_str(), static_cast<int>(f32_src.size()),
                         compile_opt, shader_variant, pipe_variant,
                         static_cast<int>(workgroup_size), impl.pipe[0], specs))
        return {};

    if ((impl.columns & 1u) == 0u)
    {
        const std::string& bf16_src = variants[add_norm ? 3 : 2];
        if (!bf16_src.empty())
        {
            (void)create_pipeline(impl.context, bf16_src.c_str(), static_cast<int>(bf16_src.size()),
                                  compile_opt, shader_variant, pipe_variant,
                                  static_cast<int>(workgroup_size), impl.pipe[1], specs);
        }
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

    rearm_read(impl.weight);
    return result;
}

const std::shared_ptr<VulkanContext>& RmsNorm_vulkan::context() const noexcept
{
    return d->context;
}

bool RmsNorm_vulkan::record(const ncnn::VkMat& input,
                            ncnn::VkMat& output,
                            ncnn::VkCompute& cmd,
                            const ncnn::VkMat* branch,
                            ncnn::VkMat* residual,
                            bool bf16_out) const
{
    const auto& impl = *d;
    const auto& pipe = impl.pipe[bf16_out ? 1 : 0];
    if (!pipe)
        return false;

    output.create(input.w, input.h, bf16_out ? sizeof(uint16_t) : sizeof(float), 1,
                  impl.context->blob_allocator());
    if (output.empty())
        return false;

    if (impl.add_norm)
    {
        residual->create(input.w, input.h, sizeof(float), 1, impl.context->blob_allocator());
        if (residual->empty())
            return false;
    }

    auto& bindings = impl.bindings;
    bindings[0] = input;
    bindings[1] = impl.weight;
    bindings[2] = output;
    if (impl.add_norm)
    {
        bindings[3] = *branch;
        bindings[4] = *residual;
    }
    impl.constants[1].u32 = static_cast<uint32_t>(input.h);

    ncnn::VkMat dispatch;
    dispatch.w = static_cast<int>(impl.workgroup_size);
    dispatch.h = input.h;
    dispatch.c = 1;
    cmd.record_pipeline_readonly(pipe.get(), bindings, impl.readonly, impl.constants, dispatch);

    for (auto& binding : bindings)
        binding.release();
    return true;
}

} // namespace moe
} // namespace ncnn
