#include "sharedffn_vulkan.h"
#include "linear.h"
#include "vulkancontext.h"
#include "graph/compiledmodel.h"

#if NCNN_MOE_WITH_VULKAN
#include <command.h>
#include <mat.h>
#endif

namespace ncnn {
namespace moe {
class SharedFfn_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<QnkExpert_vulkan> qnk;
    std::shared_ptr<Bfloat16Linear_vulkan> bfloat16, bfloat16_down;
    std::shared_ptr<Float8Linear_vulkan> float8, float8_up, float8_down;
    ExpertActivation activation = ExpertActivation::Silu;
    float activation_limit = 0.0f;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t intermediate_columns = 0;
    bool router_gate = false;
};

SharedFfn_vulkan::SharedFfn_vulkan() : d(new Implementation)
{
}
SharedFfn_vulkan::~SharedFfn_vulkan() = default;

std::shared_ptr<SharedFfn_vulkan> SharedFfn_vulkan::create(const CompiledModel& model, const MoeBlockPlan& moe)
{
#if NCNN_MOE_WITH_VULKAN
    if (model.opt.hybrid_mode != HybridMode::HybridExperts || !moe.has_shared_expert) return {};
    auto result = std::shared_ptr<SharedFfn_vulkan>(new SharedFfn_vulkan);
    auto& prepared = *result->d;
    const auto& expert = moe.shared_expert;
    prepared.activation = expert.activation;
    prepared.activation_limit = expert.activation_limit;
    prepared.router_gate = moe.shared_expert_gate_weight != invalid_tensor_handle;
    const auto& native = model.operators.at(expert.vulkan_operator).qnk_expert;
    if (native && (!prepared.router_gate || native->has_router_gate()))
    {
        prepared.qnk = native;
        prepared.context = native->vulkan_context();
        prepared.input_columns = native->input_columns();
        prepared.output_columns = native->output_columns();
    }
    else
    {
        const auto& fp8_gate = model.operators.at_weight(expert.gate_weight).float8;
        const auto& fp8_up = model.operators.at_weight(expert.up_weight).float8;
        const auto& fp8_down = model.operators.at_weight(expert.down_weight).float8;
        const auto& bf16_gate = model.operators.at(moe.fused_shared_input_bfloat16_operator).bfloat16;
        const auto& bf16_down = model.operators.at_weight(expert.down_weight).bfloat16;
        if (!prepared.router_gate && expert.activation == ExpertActivation::DeepSeekSwiGlu
            && fp8_gate && fp8_up && fp8_down
            && fp8_gate->vulkan_context() == fp8_up->vulkan_context()
            && fp8_gate->vulkan_context() == fp8_down->vulkan_context()
            && fp8_gate->input_columns() == fp8_up->input_columns()
            && fp8_gate->output_columns() == fp8_up->output_columns()
            && fp8_gate->output_columns() == fp8_down->input_columns()
            && fp8_gate->input_columns() % 128 == 0 && fp8_gate->output_columns() % 128 == 0)
        {
            prepared.float8 = fp8_gate;
            prepared.float8_up = fp8_up;
            prepared.float8_down = fp8_down;
            prepared.context = fp8_gate->vulkan_context();
            prepared.input_columns = fp8_gate->input_columns();
            prepared.output_columns = fp8_down->output_columns();
        }
        else if ((expert.activation == ExpertActivation::Silu || expert.activation == ExpertActivation::DeepSeekSwiGlu)
                 && bf16_gate && bf16_down && bf16_gate->vulkan_context() == bf16_down->vulkan_context()
                 && bf16_down->input_columns() != 0 && bf16_down->input_columns() % 128 == 0
                 && static_cast<uint64_t>(bf16_gate->output_columns())
                        == static_cast<uint64_t>(bf16_down->input_columns()) * 2 + (prepared.router_gate ? 1u : 0u))
        {
            prepared.bfloat16 = bf16_gate;
            prepared.bfloat16_down = bf16_down;
            prepared.context = bf16_gate->vulkan_context();
            prepared.input_columns = bf16_gate->input_columns();
            prepared.output_columns = bf16_down->output_columns();
            prepared.intermediate_columns = bf16_down->input_columns();
        }
    }
    if (!prepared.context || prepared.input_columns == 0 || prepared.output_columns == 0) return {};
    return result;
#else
    (void)model;
    (void)moe;
    return {};
#endif
}

std::shared_ptr<SharedFfn_vulkan> SharedFfn_vulkan::create_dense(const CompiledModel& model, const ExpertPlan& ffn)
{
#if NCNN_MOE_WITH_VULKAN
    if (model.opt.hybrid_mode != HybridMode::HybridExperts || ffn.use_gpu_cache) return {};
    const auto& native = model.operators.at(ffn.vulkan_operator).qnk_expert;
    if (!native || !native->vulkan_context()) return {};
    auto result = std::shared_ptr<SharedFfn_vulkan>(new SharedFfn_vulkan);
    auto& prepared = *result->d;
    prepared.qnk = native;
    prepared.context = native->vulkan_context();
    prepared.input_columns = native->input_columns();
    prepared.output_columns = native->output_columns();
    return result;
#else
    (void)model;
    (void)ffn;
    return {};
#endif
}

const std::shared_ptr<VulkanContext>& SharedFfn_vulkan::context() const noexcept
{
    return d->context;
}

#if NCNN_MOE_WITH_VULKAN
bool SharedFfn_vulkan::record(const ncnn::VkMat& normalized, ncnn::VkMat& output,
                              ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace,
                              FfnRecordStatistics_vulkan& statistics) const
{
    if (!d->context || &normalized == &output || normalized.empty()
        || normalized.dims != 2 || normalized.elempack != 1 || normalized.elemsize != sizeof(float)
        || normalized.h <= 0 || normalized.w != static_cast<int>(d->input_columns)
        || !normalized.allocator || normalized.allocator->vkdev != d->context->device())
        return false;
    if (d->qnk)
    {
        workspace.reserve(workspace.size() + 7);
        ncnn::VkMat result;
        result.create(static_cast<int>(d->output_columns), normalized.h, sizeof(float), d->context->blob_allocator());
        if (result.empty()) return false;
        workspace.push_back(result);
        std::vector<ncnn::VkMat> bindings;
        std::vector<ncnn::vk_constant_type> constants;
        if (!d->qnk->record(normalized, result, command, workspace, bindings, constants)) return false;
        output = result;
        statistics.dispatches += d->qnk->record_dispatches();
        statistics.int4_dispatches += d->qnk->int4_operations();
        return true;
    }
    if (d->bfloat16)
        return d->bfloat16->record_swiglu_chain(normalized, *d->bfloat16_down, d->intermediate_columns,
                                                d->activation, d->activation_limit, d->router_gate,
                                                output, command, workspace, statistics);
    return d->float8
           && d->float8->record_swiglu_chain(normalized, *d->float8_up, *d->float8_down,
                                             d->activation, d->activation_limit, output, command, workspace, statistics);
}
#endif
} // namespace moe
} // namespace ncnn
