#include "layernorm_vulkan.h"
#include "vulkancontext.h"
#include "ncnn/moe/types.h"

#if NCNN_MOE_WITH_VULKAN
#include <layer.h>
#include <layer_type.h>
#include <layernorm.h>
#include <paramdict.h>

#include <limits>
#include <mutex>
#endif

namespace ncnn {
namespace moe {

class LayerNorm_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<ncnn::Layer> layer;
    ncnn::Option option;
    uint32_t columns = 0;
#endif
};

LayerNorm_vulkan::LayerNorm_vulkan()
    : d(new Implementation)
{
}

LayerNorm_vulkan::~LayerNorm_vulkan() = default;

std::shared_ptr<LayerNorm_vulkan>
LayerNorm_vulkan::create(const TensorData& weight,
                         const TensorData& bias,
                         float epsilon,
                         std::shared_ptr<VulkanContext> context,
                         const ncnn::Option& opt,
                         ncnn::VkAllocator* weight_allocator)
{
#if NCNN_MOE_WITH_VULKAN
    if (!context || epsilon <= 0.0f
        || weight.shape.size() != 1 || weight.shape[0] == 0
        || weight.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || bias.shape != weight.shape || bias.dtype != weight.dtype
        || activation_variant(opt) != 0)
        return {};

    ncnn::Mat gamma;
    ncnn::Mat beta;
    if (!prepare_float_tensor_upload(weight, gamma)
        || !prepare_float_tensor_upload(bias, beta))
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

    ncnn::Option upload_option = impl.option;
    upload_option.blob_vkallocator = weight_allocator ? weight_allocator : impl.context->blob_allocator();
    upload_option.workspace_vkallocator = upload_option.blob_vkallocator;

    ncnn::Layer* raw_layer = ncnn::create_layer_vulkan(ncnn::LayerType::LayerNorm);
    if (!raw_layer)
        return {};
    impl.layer = std::shared_ptr<ncnn::Layer>(raw_layer, [context = impl.context, option = upload_option](ncnn::Layer* layer) mutable {
        if (!layer)
            return;
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            layer->destroy_pipeline(option);
        }
        delete layer;
    });

    raw_layer->vkdev = impl.context->device();
    ncnn::ParamDict parameters;
    parameters.set(0, static_cast<int>(impl.columns)); // affine_size
    parameters.set(1, epsilon);                        // eps
    parameters.set(2, 1);                              // affine
    if (raw_layer->load_param(parameters) != 0)
        return {};

    auto* norm = static_cast<ncnn::LayerNorm*>(raw_layer);
    norm->gamma_data = gamma;
    norm->beta_data = beta;

    {
        const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
        if (raw_layer->create_pipeline(upload_option) != 0)
            return {};
    }

    ncnn::VkTransfer transfer(impl.context->device());
    if (raw_layer->upload_model(transfer, upload_option) != 0
        || transfer.submit_and_wait() != 0)
        return {};
    return result;
#else
    (void)weight;
    (void)bias;
    (void)epsilon;
    (void)context;
    (void)opt;
    (void)weight_allocator;
    return {};
#endif
}

bool LayerNorm_vulkan::record(const ncnn::VkMat& input,
                              ncnn::VkMat& output,
                              ncnn::VkCompute& cmd) const
{
#if NCNN_MOE_WITH_VULKAN
    const auto& impl = *d;
    if (&input == &output || !impl.context || !impl.layer
        || input.empty() || input.dims != 2 || input.elempack != 1
        || input.elemsize != sizeof(float)
        || input.w != static_cast<int>(impl.columns) || input.h <= 0)
        return false;
    if (output.data && output.buffer() == input.buffer())
        output.release();
    output.create(input.w, input.h, sizeof(float), impl.context->blob_allocator());
    if (output.empty())
        return false;
    cmd.record_clone(input, output, impl.option);
    if (impl.layer->forward_inplace(output, cmd, impl.option) != 0)
        return false;
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
