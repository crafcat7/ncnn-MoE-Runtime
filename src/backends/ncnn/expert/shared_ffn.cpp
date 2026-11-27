#include "backends/ncnn/expert/shared_ffn.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/linear/linear.h"
#include "graph/compiledmodel.h"

#include <command.h>
#include <mat.h>

namespace ncnn {
namespace moe {

bool support_vulkan_ffn(const CompiledOperatorTable& operators, const ExpertPlan& ffn) noexcept
{
    const CompiledOperator& fused = operators.at(ffn.vulkan_operator);
    if (fused.qnk_expert)
        return true;
    if (fused.bfloat16 && operators.at_weight(ffn.down_weight).bfloat16)
        return true;
    return fused.float8 && operators.at_weight(ffn.up_weight).float8
           && operators.at_weight(ffn.down_weight).float8
           && (ffn.activation == ExpertActivation::Silu || ffn.activation == ExpertActivation::DeepSeekSwiGlu);
}

bool support_vulkan_shared_experts(const CompiledOperatorTable& operators,
                                   const MoeBlockPlan& moe) noexcept
{
    if (moe.fused_shared_input_bfloat16_operator != invalid_compiled_operator_handle
        && operators.at(moe.fused_shared_input_bfloat16_operator).bfloat16)
        return static_cast<bool>(operators.at_weight(moe.shared_expert.down_weight).bfloat16);

    const ExpertPlan& shared = moe.shared_expert;
    const auto& qnk = operators.at(shared.vulkan_operator).qnk_expert;
    if (qnk)
        return moe.shared_expert_gate_weight == invalid_tensor_handle || qnk->has_router_gate();

    return moe.shared_expert_gate_weight == invalid_tensor_handle
           && shared.activation == ExpertActivation::DeepSeekSwiGlu
           && operators.at_weight(shared.gate_weight).float8
           && operators.at_weight(shared.up_weight).float8
           && operators.at_weight(shared.down_weight).float8;
}

class SharedFfn_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<QnkExpert_vulkan> qnk;
    std::shared_ptr<Bfloat16Linear_vulkan> bfloat16;
    std::shared_ptr<Bfloat16Linear_vulkan> bfloat16_down;
    std::shared_ptr<Float8Linear_vulkan> float8;
    std::shared_ptr<Float8Linear_vulkan> float8_up;
    std::shared_ptr<Float8Linear_vulkan> float8_down;
    ExpertActivation activation = ExpertActivation::Silu;
    float activation_limit = 0.0f;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t intermediate_columns = 0;
    bool router_gate = false;
};

SharedFfn_vulkan::SharedFfn_vulkan()
    : d(new Implementation)
{
}

SharedFfn_vulkan::~SharedFfn_vulkan() = default;

std::shared_ptr<SharedFfn_vulkan> SharedFfn_vulkan::create(const CompiledModel& model, const MoeBlockPlan& moe)
{
    if (model.opt.hybrid_mode != HybridMode::HybridExperts || !moe.has_shared_expert)
        return {};

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
        prepared.context = native->context();
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
            && fp8_gate->context() == fp8_up->context()
            && fp8_gate->context() == fp8_down->context()
            && fp8_gate->input_columns() == fp8_up->input_columns()
            && fp8_gate->output_columns() == fp8_up->output_columns()
            && fp8_gate->output_columns() == fp8_down->input_columns()
            && fp8_gate->input_columns() % 128 == 0 && fp8_gate->output_columns() % 128 == 0)
        {
            prepared.float8 = fp8_gate;
            prepared.float8_up = fp8_up;
            prepared.float8_down = fp8_down;
            prepared.context = fp8_gate->context();
            prepared.input_columns = fp8_gate->input_columns();
            prepared.output_columns = fp8_down->output_columns();
        }
        else if ((expert.activation == ExpertActivation::Silu || expert.activation == ExpertActivation::DeepSeekSwiGlu)
                 && bf16_gate && bf16_down && bf16_gate->context() == bf16_down->context()
                 && bf16_down->input_columns() != 0 && bf16_down->input_columns() % 128 == 0
                 && static_cast<uint64_t>(bf16_gate->output_columns())
                        == static_cast<uint64_t>(bf16_down->input_columns()) * 2 + (prepared.router_gate ? 1u : 0u))
        {
            prepared.bfloat16 = bf16_gate;
            prepared.bfloat16_down = bf16_down;
            prepared.context = bf16_gate->context();
            prepared.input_columns = bf16_gate->input_columns();
            prepared.output_columns = bf16_down->output_columns();
            prepared.intermediate_columns = bf16_down->input_columns();
        }
    }

    if (!prepared.context || prepared.input_columns == 0 || prepared.output_columns == 0)
        return {};
    return result;
}

std::shared_ptr<SharedFfn_vulkan> SharedFfn_vulkan::create(const CompiledModel& model, const ExpertPlan& ffn)
{
    if (model.opt.hybrid_mode != HybridMode::HybridExperts || ffn.use_gpu_cache)
        return {};

    auto result = std::shared_ptr<SharedFfn_vulkan>(new SharedFfn_vulkan);
    auto& prepared = *result->d;
    prepared.activation = ffn.activation;
    prepared.activation_limit = ffn.activation_limit;

    const auto& native = model.operators.at(ffn.vulkan_operator).qnk_expert;
    if (native && native->context())
    {
        prepared.qnk = native;
        prepared.context = native->context();
        prepared.input_columns = native->input_columns();
        prepared.output_columns = native->output_columns();
    }
    else
    {
        const auto& bf16_gate = model.operators.at(ffn.vulkan_operator).bfloat16;
        const auto& bf16_down = model.operators.at_weight(ffn.down_weight).bfloat16;
        if ((ffn.activation == ExpertActivation::Silu || ffn.activation == ExpertActivation::DeepSeekSwiGlu)
            && bf16_gate && bf16_down && bf16_gate->context() == bf16_down->context()
            && bf16_down->input_columns() != 0 && bf16_down->input_columns() % 128 == 0
            && static_cast<uint64_t>(bf16_gate->output_columns()) == static_cast<uint64_t>(bf16_down->input_columns()) * 2)
        {
            prepared.bfloat16 = bf16_gate;
            prepared.bfloat16_down = bf16_down;
            prepared.context = bf16_gate->context();
            prepared.input_columns = bf16_gate->input_columns();
            prepared.output_columns = bf16_down->output_columns();
            prepared.intermediate_columns = bf16_down->input_columns();
        }
    }

    if (!prepared.context || prepared.input_columns == 0 || prepared.output_columns == 0)
        return {};
    return result;
}

const std::shared_ptr<VulkanContext>& SharedFfn_vulkan::context() const noexcept
{
    return d->context;
}

bool SharedFfn_vulkan::use_cm2() const noexcept
{
    if (d->bfloat16)
        return d->bfloat16->use_cm2();
    if (d->qnk)
        return d->qnk->use_cm2();
    return false;
}

bool SharedFfn_vulkan::record(const ncnn::VkMat& normalized, ncnn::VkMat& output,
                              ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace,
                              FfnRecordStatistics_vulkan& statistics,
                              const ncnn::VkMat* res,
                              bool* res_fused) const
{
    if (res_fused)
        *res_fused = false;

    if (d->qnk)
    {
        workspace.reserve(workspace.size() + 7);
        ncnn::VkMat result;
        result.create(static_cast<int>(d->output_columns), normalized.h, sizeof(float), d->context->blob_allocator());
        if (result.empty())
            return false;
        workspace.push_back(result);

        std::vector<ncnn::VkMat> bindings;
        std::vector<ncnn::vk_constant_type> constants;
        if (!d->qnk->record(normalized, result, command, workspace, bindings, constants, res, res_fused))
            return false;

        output = result;
        statistics.dispatches += d->qnk->record_dispatches();
        statistics.int4_dispatches += d->qnk->int4_operations();
        return true;
    }

    if (d->bfloat16)
    {
        return d->bfloat16->record_swiglu_chain(normalized, *d->bfloat16_down, d->intermediate_columns,
                                                d->activation, d->activation_limit, d->router_gate,
                                                output, command, workspace, statistics, res, res_fused);
    }

    return d->float8
           && d->float8->record_swiglu_chain(normalized, *d->float8_up, *d->float8_down,
                                             d->activation, d->activation_limit, output, command,
                                             workspace, statistics);
}

} // namespace moe
} // namespace ncnn
