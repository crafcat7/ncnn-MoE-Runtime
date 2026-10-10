#include "layerhead_vulkan.h"
#include "hyperconnection_vulkan.h"
#include "linear.h"
#include "rmsnorm_vulkan.h"
#include "vulkancontext.h"
#include "graph/compiledoperator.h"
#include "ncnn/moe/types.h"

#include <utility>

namespace ncnn {
namespace moe {

class LayerHead_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    uint32_t columns = 0;
    std::shared_ptr<HyperConnection_vulkan> hyper;
    std::shared_ptr<RmsNorm_vulkan> norm;
    std::shared_ptr<Linear> linear;
    std::shared_ptr<Bfloat16Linear_vulkan> bfloat16;
    std::shared_ptr<Float8Linear_vulkan> float8;
#endif
};

LayerHead_vulkan::LayerHead_vulkan()
    : d(new Implementation)
{
}

LayerHead_vulkan::~LayerHead_vulkan() = default;

std::shared_ptr<LayerHead_vulkan> LayerHead_vulkan::create(const TensorData& function,
                                                           const TensorData& scale,
                                                           const TensorData& base,
                                                           const TensorData& norm_weight,
                                                           const CompiledOperator& lm_head,
                                                           uint32_t multiplier,
                                                           float norm_epsilon,
                                                           float hyper_epsilon,
                                                           uint32_t vulkan_device_index,
                                                           const VulkanRuntimePtr& vulkan_runtime,
                                                           uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (multiplier == 0 || function.shape.size() != 2 || function.shape[0] != multiplier
        || function.shape[1] == 0 || function.shape[1] % multiplier != 0
        || norm_weight.shape.size() != 1 || norm_weight.shape[0] != function.shape[1] / multiplier)
        return {};
    const uint32_t hidden_size = function.shape[1] / multiplier;
    auto result = std::shared_ptr<LayerHead_vulkan>(new LayerHead_vulkan);
    Implementation& implementation = *result->d;
    if (lm_head.float8 && lm_head.float8->vulkan_context()
        && lm_head.float8->input_columns() == hidden_size
        && hidden_size % 128 == 0 && vulkan_activation_storage_variant(lm_head.float8->option()) == 0)
    {
        implementation.float8 = lm_head.float8;
        implementation.context = lm_head.float8->vulkan_context();
        implementation.opt = lm_head.float8->option();
    }
    else if (lm_head.bfloat16 && lm_head.bfloat16->vulkan_context()
             && lm_head.bfloat16->input_columns() == hidden_size
             && vulkan_activation_storage_variant(lm_head.bfloat16->option()) == 0)
    {
        implementation.bfloat16 = lm_head.bfloat16;
        implementation.context = lm_head.bfloat16->vulkan_context();
        implementation.opt = lm_head.bfloat16->option();
    }
    else if (lm_head.linear && lm_head.linear->uses_vulkan() && lm_head.linear->vulkan_context()
             && lm_head.linear->input_columns() == hidden_size
             && vulkan_activation_storage_variant(lm_head.linear->option()) == 0)
    {
        implementation.linear = lm_head.linear;
        implementation.context = lm_head.linear->vulkan_context();
        implementation.opt = lm_head.linear->option();
    }
    else
        return {};
    implementation.columns = hidden_size;
    implementation.opt.use_packing_layout = false;
    implementation.opt.use_fp16_packed = false;
    implementation.opt.use_fp16_storage = false;
    implementation.opt.use_fp16_arithmetic = false;
    implementation.opt.use_bf16_packed = false;
    implementation.opt.use_bf16_storage = false;
    implementation.opt.blob_vkallocator = implementation.context->blob_allocator();
    implementation.opt.workspace_vkallocator = implementation.context->blob_allocator();
    implementation.opt.staging_vkallocator = implementation.context->staging_allocator();
    implementation.hyper = HyperConnection_vulkan::create(function, scale, base, multiplier, 0, norm_epsilon, hyper_epsilon,
                                                          vulkan_device_index, vulkan_runtime, optimization_flags);
    if (!implementation.hyper || implementation.hyper->vulkan_context() != implementation.context)
        return {};
    implementation.norm = RmsNorm_vulkan::create(norm_weight, norm_epsilon, 0.0f, implementation.context, implementation.opt);
    if (!implementation.norm)
        return {};
    return result;
#else
    (void)function;
    (void)scale;
    (void)base;
    (void)norm_weight;
    (void)lm_head;
    (void)multiplier;
    (void)norm_epsilon;
    (void)hyper_epsilon;
    (void)vulkan_device_index;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& LayerHead_vulkan::vulkan_context() const noexcept
{
    return d->context;
}

const ncnn::Option& LayerHead_vulkan::option() const noexcept
{
    return d->opt;
}

bool LayerHead_vulkan::record_hidden(const ncnn::VkMat& input,
                                     ncnn::VkMat& normalized,
                                     HyperConnectionWorkspace_vulkan& hc_workspace,
                                     std::vector<ncnn::VkMat>& temporaries,
                                     ncnn::VkCompute& cmd) const
{
    if (&normalized == &input || &normalized == &hc_workspace.projection || &normalized == &hc_workspace.pre)
        return false;
    ncnn::VkMat headed;
    if (!d->hyper->record_head(input, headed, hc_workspace, cmd))
        return false;
    temporaries.push_back(headed);
    ncnn::VkMat next;
    if (!d->norm->record(headed, next, cmd))
        return false;
    temporaries.push_back(next);
    normalized = std::move(next);
    return true;
}

bool LayerHead_vulkan::record(const ncnn::VkMat& input,
                              ncnn::VkMat& logits,
                              HyperConnectionWorkspace_vulkan& hc_workspace,
                              std::vector<ncnn::VkMat>& temporaries,
                              ncnn::VkCompute& cmd) const
{
    if (&logits == &input || &logits == &hc_workspace.projection || &logits == &hc_workspace.pre)
        return false;
    ncnn::VkMat normalized;
    if (!record_hidden(input, normalized, hc_workspace, temporaries, cmd))
        return false;
    return record_logits(normalized, logits, temporaries, cmd);
}

bool LayerHead_vulkan::record_logits(const ncnn::VkMat& normalized,
                                     ncnn::VkMat& logits,
                                     std::vector<ncnn::VkMat>& temporaries,
                                     ncnn::VkCompute& cmd) const
{
    if (&normalized == &logits || normalized.empty() || normalized.dims != 2
        || normalized.w != static_cast<int>(d->columns) || normalized.h <= 0
        || normalized.elempack != 1 || normalized.elemsize != sizeof(float))
        return false;
    // The input may be an element of temporaries. Recording FP8 projection appends its
    // quantized workspace, so keep a separate owner before vector reallocation.
    const ncnn::VkMat input = normalized;
    if (temporaries.empty() || temporaries.back().data != input.data || temporaries.back().offset != input.offset)
        temporaries.push_back(input);
    ncnn::VkMat next;
    if (d->float8)
    {
        if (!d->float8->record_forward(input, next, cmd, temporaries))
            return false;
    }
    else if (d->bfloat16)
    {
        if (d->bfloat16->forward(input, next, cmd, d->opt) != 0)
            return false;
    }
    else if (!d->linear || d->linear->forward(input, next, cmd, d->opt) != 0)
        return false;
    if (next.empty() || next.dims != 2 || next.h != input.h || next.elempack != 1 || next.elemsize != sizeof(float))
        return false;
    logits = std::move(next);
    return true;
}
#endif

} // namespace moe
} // namespace ncnn
