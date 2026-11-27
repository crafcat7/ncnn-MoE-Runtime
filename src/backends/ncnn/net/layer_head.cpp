#include "backends/ncnn/net/layer_head.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/layer/hyperconnection_vulkan.h"
#include "backends/ncnn/layer/rmsnorm_vulkan.h"
#include "backends/ncnn/linear/linear.h"
#include "graph/compiledoperator.h"
#include "ncnn/moe/types.h"

#include <utility>

namespace ncnn {
namespace moe {

class LayerHead_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    uint32_t columns = 0;
    std::shared_ptr<HyperConnection_vulkan> hyper;
    std::shared_ptr<RmsNorm_vulkan> norm;
    std::shared_ptr<Linear> linear;
    std::shared_ptr<Bfloat16Linear_vulkan> bfloat16;
    std::shared_ptr<Float8Linear_vulkan> float8;
    std::shared_ptr<QnkLinear_vulkan> qnk;
};

LayerHead_vulkan::LayerHead_vulkan()
    : d(new Implementation)
{
}

LayerHead_vulkan::~LayerHead_vulkan() = default;

uint32_t LayerHead_vulkan::int4_ops(bool include_logits) const noexcept
{
    return d->hyper->int4_ops() + (include_logits && d->qnk && d->qnk->dtype() == DType::Int4 ? 1u : 0u);
}

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
    const uint32_t hidden_size = function.shape[1] / multiplier;
    auto result = std::shared_ptr<LayerHead_vulkan>(new LayerHead_vulkan);
    Implementation& impl = *result->d;

    if (lm_head.qnk && lm_head.qnk->context()
        && lm_head.qnk->input_columns() == hidden_size
        && activation_variant(lm_head.qnk->option()) == 0)
    {
        impl.qnk = lm_head.qnk;
        impl.context = lm_head.qnk->context();
        impl.opt = lm_head.qnk->option();
    }
    else if (lm_head.float8 && lm_head.float8->context()
             && lm_head.float8->input_columns() == hidden_size
             && hidden_size % 128 == 0
             && activation_variant(lm_head.float8->option()) == 0)
    {
        impl.float8 = lm_head.float8;
        impl.context = lm_head.float8->context();
        impl.opt = lm_head.float8->option();
    }
    else if (lm_head.bfloat16 && lm_head.bfloat16->context()
             && lm_head.bfloat16->input_columns() == hidden_size
             && activation_variant(lm_head.bfloat16->option()) == 0)
    {
        impl.bfloat16 = lm_head.bfloat16;
        impl.context = lm_head.bfloat16->context();
        impl.opt = lm_head.bfloat16->option();
    }
    else if (lm_head.linear && lm_head.linear->context()
             && lm_head.linear->input_columns() == hidden_size
             && activation_variant(lm_head.linear->option()) == 0)
    {
        impl.linear = lm_head.linear;
        impl.context = lm_head.linear->context();
        impl.opt = lm_head.linear->option();
    }
    else
    {
        return {};
    }

    impl.columns = hidden_size;
    impl.opt = make_vk_opt(*impl.context);
    impl.hyper = HyperConnection_vulkan::create(function, scale, base, multiplier, 0,
                                                norm_epsilon, hyper_epsilon,
                                                vulkan_device_index, vulkan_runtime,
                                                optimization_flags);
    if (!impl.hyper || impl.hyper->context() != impl.context)
        return {};

    impl.norm = RmsNorm_vulkan::create(norm_weight, norm_epsilon, 0.0f, impl.context, impl.opt);
    if (!impl.norm)
        return {};

    return result;
}

const std::shared_ptr<VulkanContext>& LayerHead_vulkan::context() const noexcept
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

bool LayerHead_vulkan::record_logits(const ncnn::VkMat& normalized,
                                     ncnn::VkMat& logits,
                                     std::vector<ncnn::VkMat>& temporaries,
                                     ncnn::VkCompute& cmd) const
{
    // Retain an independent copy before FP8 quantization appends to temporaries.
    const ncnn::VkMat input = normalized;
    if (temporaries.empty() || temporaries.back().data != input.data || temporaries.back().offset != input.offset)
        temporaries.push_back(input);

    ncnn::VkMat next;
    if (d->qnk)
    {
        next.create(static_cast<int>(d->qnk->output_columns()), input.h, sizeof(float),
                    d->context->blob_allocator());
        if (next.empty())
            return false;
        d->qnk->record(input, next, cmd);
    }
    else if (d->float8)
    {
        if (!d->float8->record_forward(input, next, cmd, temporaries))
            return false;
    }
    else if (d->bfloat16)
    {
        if (d->bfloat16->forward(input, next, cmd, d->opt) != 0)
            return false;
    }
    else if (d->linear->forward(input, next, cmd, d->opt) != 0)
    {
        return false;
    }

    logits = std::move(next);
    return true;
}

} // namespace moe
} // namespace ncnn
