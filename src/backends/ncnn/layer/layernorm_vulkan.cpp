#include "backends/ncnn/layer/layernorm_vulkan.h"

#include "backends/ncnn/core/context.h"
#include "ncnn/moe/types.h"

#include <layer.h>
#include <layer_type.h>
#include <layernorm.h>
#include <paramdict.h>

#include <mutex>

namespace ncnn {
namespace moe {

class LayerNorm_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<ncnn::Layer> layer;
    ncnn::Option option;
    uint32_t columns = 0;
};

LayerNorm_vulkan::LayerNorm_vulkan()
    : d(new Implementation)
{
}

LayerNorm_vulkan::~LayerNorm_vulkan() = default;

std::shared_ptr<LayerNorm_vulkan> LayerNorm_vulkan::create(const TensorData& weight,
                                                           const TensorData& bias,
                                                           float epsilon,
                                                           std::shared_ptr<VulkanContext> context,
                                                           const ncnn::Option& opt,
                                                           ncnn::VkAllocator* weight_allocator)
{
    ncnn::Mat gamma;
    ncnn::Mat beta;
    if (!prepare_float_upload(weight, gamma) || !prepare_float_upload(bias, beta))
        return {};

    auto result = std::shared_ptr<LayerNorm_vulkan>(new LayerNorm_vulkan);
    auto& impl = *result->d;
    impl.context = std::move(context);
    impl.columns = weight.shape[0];
    impl.option = opt;
    impl.option.use_vulkan_compute = true;
    impl.option.blob_vkallocator = impl.context->blob_allocator();
    impl.option.workspace_vkallocator = impl.context->blob_allocator();
    impl.option.staging_vkallocator = impl.context->staging_allocator();

    ncnn::Option upload_opt = impl.option;
    upload_opt.blob_vkallocator = weight_allocator ? weight_allocator : impl.context->blob_allocator();
    upload_opt.workspace_vkallocator = upload_opt.blob_vkallocator;

    ncnn::Layer* raw_layer = ncnn::create_layer_vulkan(ncnn::LayerType::LayerNorm);
    if (!raw_layer)
        return {};

    impl.layer = own_vk_layer(impl.context, raw_layer, upload_opt);

    raw_layer->vkdev = impl.context->device();
    ncnn::ParamDict params;
    params.set(0, static_cast<int>(impl.columns));
    params.set(1, epsilon);
    params.set(2, 1);
    if (raw_layer->load_param(params) != 0)
        return {};

    auto* norm = static_cast<ncnn::LayerNorm*>(raw_layer);
    norm->gamma_data = gamma;
    norm->beta_data = beta;

    {
        const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
        if (raw_layer->create_pipeline(upload_opt) != 0)
            return {};
    }

    ncnn::VkTransfer transfer(impl.context->device());
    if (raw_layer->upload_model(transfer, upload_opt) != 0 || transfer.submit_and_wait() != 0)
        return {};

    return result;
}

bool LayerNorm_vulkan::record(const ncnn::VkMat& input,
                              ncnn::VkMat& output,
                              ncnn::VkCompute& cmd) const
{
    const auto& impl = *d;
    output.create(input.w, input.h, sizeof(float), impl.context->blob_allocator());
    if (output.empty())
        return false;

    cmd.record_clone(input, output, impl.option);
    return impl.layer->forward_inplace(output, cmd, impl.option) == 0;
}

} // namespace moe
} // namespace ncnn
