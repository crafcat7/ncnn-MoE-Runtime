#ifndef NCNN_MOE_INNERPRODUCTWEIGHTS_VULKAN_H
#define NCNN_MOE_INNERPRODUCTWEIGHTS_VULKAN_H

#if NCNN_MOE_WITH_VULKAN
#include <layer/vulkan/innerproduct_vulkan.h>

namespace ncnn {
namespace moe {

inline bool inner_product_weights_available(const ncnn::InnerProduct_vulkan& layer, bool has_bias) noexcept
{
    return !layer.weight_data_gpu.empty() && (!has_bias || !layer.bias_data_gpu.empty());
}

} // namespace moe
} // namespace ncnn
#endif

#endif // NCNN_MOE_INNERPRODUCTWEIGHTS_VULKAN_H
