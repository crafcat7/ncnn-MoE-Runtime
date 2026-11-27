#include "backends/ncnn/layer/router_vulkan.h"

#include "backends/ncnn/linear/linear.h"
#include "kernels/vulkan/router_select.comp.hex.h"
#include "kernels/vulkan/router_select_raw.comp.hex.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

class Router_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<Linear> projection;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> raw_pipeline;
    ncnn::Option opt;
    std::vector<float> selection_values;
    ncnn::VkMat selection_bias;
    ncnn::VkMat dummy;
    uint32_t expert_count = 0;
    uint32_t input_columns = 0;
    uint32_t raw_subgroup_size = 0;
};

static uint32_t router_raw_subgroup_size(const ncnn::GpuInfo& info, const ncnn::Option& option)
{
    const uint32_t subgroup = info.subgroup_size();
    const uint32_t required_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    if (!option.use_subgroup_ops || (info.support_subgroup_ops() & required_ops) != required_ops
        || !info.queryMaintenance4Features().maintenance4
        || !info.support_subgroup_size_control() || !info.support_compute_full_subgroups()
        || (info.querySubgroupSizeControlProperties().requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0
        || subgroup < 32 || subgroup > 128 || (subgroup & (subgroup - 1)) != 0
        || subgroup < info.min_subgroup_size() || subgroup > info.max_subgroup_size()
        || subgroup > info.max_workgroup_invocations() || subgroup > info.max_workgroup_size_x())
        return 0;
    return subgroup;
}

Router_vulkan::Router_vulkan()
    : d(new Implementation)
{
}

Router_vulkan::~Router_vulkan() = default;

uint32_t Router_vulkan::selected_columns(uint32_t top_k) noexcept
{
    return top_k > 0 && top_k <= 64 ? top_k * 2 + 1 : 0;
}

std::shared_ptr<Router_vulkan> Router_vulkan::create(const TensorData& weight,
                                                     const TensorData* bias,
                                                     const TensorData* selection_bias,
                                                     uint32_t device_index,
                                                     const VulkanRuntimePtr& runtime,
                                                     uint64_t optimization_flags)
{
    const auto projection = Linear::create(weight, bias, device_index, runtime, optimization_flags);
    if (!projection)
        return {};

    std::shared_ptr<Router_vulkan> result(new Router_vulkan);
    Implementation& impl = *result->d;
    impl.projection = projection;
    impl.context = projection->context();
    impl.opt = projection->option();
    impl.expert_count = weight.shape[0];
    impl.input_columns = weight.shape[1];

    if (!create_pipeline(impl.context, router_select_shader,
                         static_cast<int>(sizeof(router_select_shader) - 1),
                         impl.opt, 0, 0, 128, impl.pipeline))
        return {};

    impl.raw_subgroup_size = router_raw_subgroup_size(impl.context->device()->info, impl.opt);
    if (impl.raw_subgroup_size != 0)
    {
        const uint32_t subgroup = impl.raw_subgroup_size;
        const uint32_t experts_per_lane = (impl.expert_count + subgroup - 1) / subgroup;
        const uint64_t variant = (static_cast<uint64_t>(experts_per_lane) << 32) | subgroup;
        ncnn::Option compile_opt = impl.opt;
        compile_opt.use_subgroup_ops = true;

        // maintenance4 triggers SPIR-V 1.6 so local_size_x == subgroup makes all lanes active.
        std::vector<ncnn::vk_specialization_type> specs(1);
        specs[0].u32 = experts_per_lane;
        if (!create_pipeline(impl.context, router_select_raw_shader,
                             static_cast<int>(sizeof(router_select_raw_shader) - 1),
                             compile_opt, 0, variant, static_cast<int>(subgroup),
                             impl.raw_pipeline, specs, static_cast<int>(subgroup)))
            return {};
    }

    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    ncnn::VkTransfer transfer(impl.context->device());
    ncnn::Mat dummy(1, sizeof(float));
    dummy[0] = 0.0f;
    transfer.record_upload(dummy, impl.dummy, impl.opt);

    if (selection_bias)
    {
        ncnn::Mat values;
        if (!prepare_float_upload(*selection_bias, values))
            return {};
        const float* data = static_cast<const float*>(values.data);
        impl.selection_values.assign(data, data + impl.expert_count);
        for (float value : impl.selection_values)
        {
            if (!std::isfinite(value))
                return {};
        }
        transfer.record_upload(values, impl.selection_bias, impl.opt);
    }

    if (transfer.submit_and_wait() != 0 || impl.dummy.empty()
        || (selection_bias && impl.selection_bias.empty()))
        return {};

    return result;
}

Result<void> Router_vulkan::decode_selected(const ActivationBuffer& selected,
                                            const ExpertDispatchOptions& options,
                                            ExpertDispatchPlan& result)
{
    if (result.route_scratch.size() < options.expert_count)
        result.route_scratch.resize(options.expert_count);
    for (uint32_t expert = 0; expert < options.expert_count; ++expert)
        result.route_scratch[expert].clear();

    for (size_t row = 0; row < selected.rows(); ++row)
    {
        const float* values = selected.row(row);
        for (uint32_t rank = 0; rank < options.top_k; ++rank)
        {
            const uint32_t expert = static_cast<uint32_t>(values[rank]);
            auto& entries = result.route_scratch[expert];
            entries.push_back({static_cast<uint32_t>(row), rank, values[options.top_k + rank]});
        }
    }

    size_t batch_count = 0;
    for (uint32_t expert = 0; expert < options.expert_count; ++expert)
        batch_count += !result.route_scratch[expert].empty();
    if (result.batches.size() < batch_count)
        result.batches.resize(batch_count);

    size_t batch_index = 0;
    for (uint32_t expert = 0; expert < options.expert_count; ++expert)
    {
        if (result.route_scratch[expert].empty())
            continue;
        ExpertBatch& batch = result.batches[batch_index++];
        batch.expert_id = expert;
        batch.routes.swap(result.route_scratch[expert]);
    }

    for (size_t i = batch_count; i < result.batches.size(); ++i)
    {
        ExpertBatch& batch = result.batches[i];
        if (batch.expert_id < result.route_scratch.size()
            && batch.routes.capacity() > result.route_scratch[batch.expert_id].capacity())
            batch.routes.swap(result.route_scratch[batch.expert_id]);
    }
    result.batches.resize(batch_count);
    result.assignment_count = selected.rows() * options.top_k;
    return {};
}

const std::shared_ptr<VulkanContext>& Router_vulkan::context() const noexcept
{
    return d->context;
}

const ncnn::Option& Router_vulkan::option() const noexcept
{
    return d->opt;
}

bool Router_vulkan::record(const ncnn::VkMat& input,
                           const ExpertDispatchOptions& options,
                           RouterWorkspace_vulkan& workspace,
                           ncnn::VkCompute& cmd) const
{
    const Implementation& impl = *d;
    workspace.context = impl.context;
    workspace.selection_bias = impl.dummy;
    workspace.explicit_ids = impl.dummy;

    if (!options.selection_bias.empty())
    {
        const bool same_as_weights = impl.selection_values.size() == options.selection_bias.size()
                                     && std::equal(impl.selection_values.begin(),
                                                   impl.selection_values.end(),
                                                   options.selection_bias.begin());
        if (same_as_weights)
        {
            workspace.selection_bias = impl.selection_bias;
        }
        else
        {
            workspace.selection_bias = {};
            if (!fill_staging_values(options.selection_bias.data(), options.selection_bias.size(),
                                     sizeof(float), workspace.selection_bias_staging,
                                     impl.opt.staging_vkallocator)
                || !record_mapped_upload(workspace.selection_bias_staging,
                                         workspace.selection_bias, cmd, impl.opt))
                return false;
        }
    }

    if (!options.explicit_expert_ids.empty())
    {
        workspace.explicit_ids = {};
        if (!fill_staging_values(options.explicit_expert_ids.data(),
                                 options.explicit_expert_ids.size(),
                                 sizeof(uint32_t), workspace.explicit_ids_staging,
                                 impl.opt.staging_vkallocator)
            || !record_mapped_upload(workspace.explicit_ids_staging,
                                     workspace.explicit_ids, cmd, impl.opt))
            return false;
    }

    if (impl.projection->forward(input, workspace.projected, cmd, impl.opt) != 0
        || workspace.projected.empty())
        return false;

    workspace.logits = workspace.projected;
    if (workspace.projected.elempack != 1)
        impl.context->device()->convert_packing(workspace.projected, workspace.logits, 1, cmd, impl.opt);

    workspace.selected.create(static_cast<int>(selected_columns(options.top_k)),
                              input.h, sizeof(float), impl.opt.blob_vkallocator);
    if (workspace.selected.empty())
        return false;

    const bool raw_topk = options.score_function == RouterScoreFunction::Softmax
                          && options.normalization == RouterNormalization::SelectedExperts
                          && options.group_count == 0
                          && options.selection_bias.empty() && options.explicit_expert_ids.empty();
    if (raw_topk && impl.raw_pipeline)
    {
        std::vector<ncnn::vk_constant_type> constants(3);
        constants[0].u32 = options.expert_count;
        constants[1].u32 = options.top_k;
        constants[2].f = options.routed_scaling_factor;

        ncnn::VkMat dispatch;
        dispatch.w = impl.raw_subgroup_size;
        dispatch.h = input.h;
        dispatch.c = 1;
        cmd.record_pipeline_readonly(impl.raw_pipeline.get(),
                                     {workspace.logits, workspace.selected}, {1, 0}, constants, dispatch);
        return true;
    }

    std::vector<ncnn::vk_constant_type> constants(10);
    constants[0].u32 = options.expert_count;
    constants[1].u32 = options.top_k;
    constants[2].u32 = static_cast<uint32_t>(options.score_function);
    constants[3].u32 = options.normalization == RouterNormalization::SelectedExperts ? 1u : 0u;
    constants[4].u32 = !options.selection_bias.empty() ? 1u : 0u;
    constants[5].u32 = !options.explicit_expert_ids.empty() ? 1u : 0u;
    constants[6].f = options.routed_scaling_factor;
    constants[7].f = options.jitter;
    constants[8].u32 = options.group_count;
    constants[9].u32 = options.top_k_groups;

    ncnn::VkMat dispatch;
    dispatch.w = 128;
    dispatch.h = input.h;
    dispatch.c = 1;
    cmd.record_pipeline_readonly(impl.pipeline.get(),
                                 {workspace.logits, workspace.selection_bias,
                                  workspace.explicit_ids, workspace.selected},
                                 {1, 1, 1, 0}, constants, dispatch);
    return true;
}

} // namespace moe
} // namespace ncnn
