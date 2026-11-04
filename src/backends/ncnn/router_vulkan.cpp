#include "router_vulkan.h"
#include "linear.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/router_select.comp.hex.h"
#include "kernels/vulkan/router_select_raw.comp.hex.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static bool valid_router_options(const ExpertDispatchOptions& options,
                                 size_t rows)
{
    const bool known_score = options.score_function == RouterScoreFunction::Softmax
                             || options.score_function == RouterScoreFunction::Sigmoid
                             || options.score_function == RouterScoreFunction::SqrtSoftplus
                             || options.score_function == RouterScoreFunction::SparseMixer;
    if (rows == 0 || options.expert_count == 0 || options.expert_count > 2048
        || options.top_k == 0 || options.top_k > options.expert_count || options.top_k > 64
        || !std::isfinite(options.routed_scaling_factor) || options.routed_scaling_factor <= 0.0f
        || (options.normalization != RouterNormalization::None
            && options.normalization != RouterNormalization::SelectedExperts)
        || !known_score
        || (!options.selection_bias.empty() && options.selection_bias.size() != options.expert_count)
        || (!options.explicit_expert_ids.empty() && options.explicit_expert_ids.size() != rows * options.top_k))
        return false;
    if (options.score_function == RouterScoreFunction::SparseMixer
        && (options.top_k != 2 || !std::isfinite(options.jitter) || options.jitter < 0.0f))
        return false;
    for (float value : options.selection_bias)
        if (!std::isfinite(value))
            return false;
    for (uint32_t id : options.explicit_expert_ids)
        if (id >= options.expert_count)
            return false;
    return true;
}

class Router_vulkan::Implementation
{
public:
    uint32_t expert_count = 0;
    uint32_t input_columns = 0;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<Linear> projection;
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    std::vector<float> selection_values;
    ncnn::VkMat selection_bias;
    ncnn::VkMat dummy;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> raw_pipeline;
    uint32_t raw_subgroup_size = 0;
#endif
};

#if NCNN_MOE_WITH_VULKAN
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

static bool create_router_pipeline(const std::shared_ptr<VulkanContext>& context,
                                   const ncnn::Option& option, const char* shader, size_t size,
                                   uint64_t variant, uint32_t group_size, uint32_t subgroup_size,
                                   const std::vector<ncnn::vk_specialization_type>& specializations,
                                   std::shared_ptr<ncnn::Pipeline>& output)
{
    const auto binary = context->shader_binary(shader, static_cast<int>(size), option, 0);
    if (!binary || binary->empty())
        return false;
    // The deleter also locks for unprepared pipelines, so allocation of its
    // shared ownership control block must precede the command lock.
    auto pipeline = std::shared_ptr<ncnn::Pipeline>(new ncnn::Pipeline(context->device()),
                                                    [context = context](ncnn::Pipeline* value) mutable {
                                                        {
                                                            const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                            delete value;
                                                        }
                                                        // Weak cache entries retain the deleter after disposal.
                                                        context.reset();
                                                    });
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    output = context->find_pipeline(shader, variant);
    if (output)
        return true;
    if (subgroup_size != 0)
        pipeline->set_subgroup_size(subgroup_size);
    pipeline->set_local_size_xyz(group_size, 1, 1);
    if (pipeline->create(binary->data(), binary->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    output = pipeline;
    context->cache_pipeline(shader, variant, output);
    return true;
}
#endif

Router_vulkan::Router_vulkan() : d(new Implementation)
{
}

Router_vulkan::~Router_vulkan() = default;

uint32_t Router_vulkan::selected_columns(uint32_t top_k) noexcept
{
    return top_k > 0 && top_k <= 64 ? top_k * 2 + 1 : 0;
}

std::shared_ptr<Router_vulkan>
Router_vulkan::create(const TensorData& weight, const TensorData* bias,
                      const TensorData* selection_bias, uint32_t device_index,
                      const VulkanRuntimePtr& runtime,
                      uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (weight.shape.size() != 2 || weight.shape[0] == 0 || weight.shape[0] > 2048 || weight.shape[1] == 0 || (weight.dtype != DType::Float32 && weight.dtype != DType::BFloat16) || (weight.dtype == DType::Float32 && weight.float32_values().size() != weight.element_count()) || (weight.dtype == DType::BFloat16 && weight.bfloat16_values().size() != weight.element_count()))
        return {};
    for (const TensorData* vector : {bias, selection_bias})
    {
        if (vector && (vector->shape.size() != 1 || vector->shape[0] != weight.shape[0] || (vector->dtype != DType::Float32 && vector->dtype != DType::BFloat16) || (vector->dtype == DType::Float32 && vector->float32_values().size() != weight.shape[0]) || (vector->dtype == DType::BFloat16 && vector->bfloat16_values().size() != weight.shape[0])))
            return {};
    }
    const auto projection = Linear::create(weight, bias, device_index, runtime,
                                           optimization_flags);
    if (!projection)
        return {};
    std::shared_ptr<Router_vulkan> result(new Router_vulkan);
    Implementation& implementation = *result->d;
    implementation.projection = projection;
    implementation.context = projection->vulkan_context();
    implementation.opt = projection->option();
    implementation.expert_count = weight.shape[0];
    implementation.input_columns = weight.shape[1];
    if (!create_router_pipeline(implementation.context, implementation.opt,
                                router_select_shader, sizeof(router_select_shader) - 1,
                                0, 128, 0, {}, implementation.pipeline))
        return {};
    implementation.raw_subgroup_size = router_raw_subgroup_size(implementation.context->device()->info,
                                                                implementation.opt);
    if (implementation.raw_subgroup_size != 0)
    {
        const uint32_t subgroup = implementation.raw_subgroup_size;
        const uint32_t experts_per_lane = (implementation.expert_count + subgroup - 1) / subgroup;
        const uint64_t variant = static_cast<uint64_t>(experts_per_lane) << 32 | subgroup;
        ncnn::Option compile_option = implementation.opt;
        compile_option.use_subgroup_ops = true;
        // Enabled maintenance4 makes ncnn emit SPIR-V 1.6. Together with a
        // required subgroup size equal to local_size_x, this guarantees every
        // shuffle source is active without a REQUIRE_FULL_SUBGROUPS stage flag.
        std::vector<ncnn::vk_specialization_type> specializations(1);
        specializations[0].u32 = experts_per_lane;
        if (!create_router_pipeline(implementation.context, compile_option,
                                    router_select_raw_shader, sizeof(router_select_raw_shader) - 1,
                                    variant, subgroup, subgroup, specializations,
                                    implementation.raw_pipeline))
            return {};
    }
    const std::lock_guard<std::mutex> lock(implementation.context->command_mutex());
    ncnn::VkTransfer transfer(implementation.context->device());
    ncnn::Mat dummy(1, sizeof(float));
    dummy[0] = 0.0f;
    transfer.record_upload(dummy, implementation.dummy, implementation.opt);
    if (selection_bias)
    {
        ncnn::Mat values;
        if (!prepare_float_tensor_upload(*selection_bias, values))
            return {};
        const float* data = static_cast<const float*>(values.data);
        implementation.selection_values.assign(data,
                                               data + implementation.expert_count);
        for (float value : implementation.selection_values)
            if (!std::isfinite(value))
                return {};
        transfer.record_upload(values, implementation.selection_bias,
                               implementation.opt);
    }
    if (transfer.submit_and_wait() != 0 || implementation.dummy.empty() || (selection_bias && implementation.selection_bias.empty()))
        return {};
    return result;
#else
    (void)weight;
    (void)bias;
    (void)selection_bias;
    (void)device_index;
    (void)runtime;
    (void)optimization_flags;
    return {};
#endif
}

Result<void>
Router_vulkan::decode_selected(const ActivationBuffer& selected,
                               const ExpertDispatchOptions& options,
                               ExpertDispatchPlan& result)
{
    if (selected.dtype() != DType::Float32 || selected.columns() != selected_columns(options.top_k) || !valid_router_options(options, selected.rows()))
        return Error{ErrorCode::InvalidArgument,
                     "GPU router selected shape or options are invalid"};
    // Validate every row before touching the published plan or route scratch.
    for (size_t row = 0; row < selected.rows(); ++row)
    {
        const float* values = selected.row(row);
        const float status = values[options.top_k * 2];
        if (status == 1.0f)
            return Error{ErrorCode::InvalidArgument, "router logits must be finite"};
        if (status == 2.0f)
            return Error{ErrorCode::InvalidModel,
                         "selected router weights have a non-positive sum"};
        if (status != 0.0f)
            return Error{ErrorCode::InternalError,
                         "GPU router produced an invalid status"};
        for (uint32_t rank = 0; rank < options.top_k; ++rank)
        {
            const float id = values[rank];
            const float weight = values[options.top_k + rank];
            if (!std::isfinite(id) || id < 0.0f || id >= static_cast<float>(options.expert_count) || std::floor(id) != id || !std::isfinite(weight) || weight < 0.0f)
                return Error{ErrorCode::InternalError,
                             "GPU router produced an invalid selected route"};
        }
    }
    if (result.route_scratch.size() < options.expert_count)
        result.route_scratch.resize(options.expert_count);
    for (uint32_t expert = 0; expert < options.expert_count; ++expert)
        result.route_scratch[expert].clear();
    for (size_t row = 0; row < selected.rows(); ++row)
    {
        const float* values = selected.row(row);
        for (uint32_t rank = 0; rank < options.top_k; ++rank)
            result.route_scratch[static_cast<uint32_t>(values[rank])].push_back({static_cast<uint32_t>(row), rank, values[options.top_k + rank]});
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
    for (size_t index = batch_count; index < result.batches.size(); ++index)
    {
        ExpertBatch& batch = result.batches[index];
        if (batch.expert_id < result.route_scratch.size() && batch.routes.capacity() > result.route_scratch[batch.expert_id].capacity())
            batch.routes.swap(result.route_scratch[batch.expert_id]);
    }
    result.batches.resize(batch_count);
    result.assignment_count = selected.rows() * options.top_k;
    return {};
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& Router_vulkan::context() const noexcept
{
    return d->context;
}

const ncnn::Option& Router_vulkan::option() const noexcept
{ return d->opt; }

bool Router_vulkan::record(const ncnn::VkMat& input,
                           const ExpertDispatchOptions& options,
                           RouterWorkspace_vulkan& workspace,
                           ncnn::VkCompute& cmd) const
{
    const Implementation& implementation = *d;
    if (input.empty() || input.dims != 2 || input.w != static_cast<int>(implementation.input_columns) || input.h <= 0 || input.elempack != 1 || input.elemsize != sizeof(float) || options.expert_count != implementation.expert_count || !valid_router_options(options, static_cast<size_t>(input.h)))
        return false;
    workspace.context = implementation.context;
    workspace.selection_bias = implementation.dummy;
    workspace.explicit_ids = implementation.dummy;
    if (!options.selection_bias.empty())
    {
        if (implementation.selection_values.size() == options.selection_bias.size() && std::equal(implementation.selection_values.begin(), implementation.selection_values.end(), options.selection_bias.begin()))
            workspace.selection_bias = implementation.selection_bias;
        else
        {
            workspace.selection_bias = {};
            if (!fill_staging_values(options.selection_bias.data(), options.selection_bias.size(),
                                     sizeof(float), workspace.selection_bias_staging,
                                     implementation.opt.staging_vkallocator)
                || !record_mapped_upload(workspace.selection_bias_staging,
                                         workspace.selection_bias, cmd,
                                         implementation.opt))
                return false;
        }
    }
    if (!options.explicit_expert_ids.empty())
    {
        workspace.explicit_ids = {};
        if (!fill_staging_values(options.explicit_expert_ids.data(),
                                 options.explicit_expert_ids.size(),
                                 sizeof(uint32_t), workspace.explicit_ids_staging,
                                 implementation.opt.staging_vkallocator)
            || !record_mapped_upload(workspace.explicit_ids_staging,
                                     workspace.explicit_ids, cmd, implementation.opt))
            return false;
    }
    if (implementation.projection->forward(input, workspace.projected, cmd,
                                           implementation.opt)
            != 0
        || workspace.projected.empty())
        return false;
    workspace.logits = workspace.projected;
    if (workspace.projected.elempack != 1)
        implementation.context->device()->convert_packing(workspace.projected, workspace.logits, 1, cmd, implementation.opt);
    if (workspace.logits.empty() || workspace.logits.dims != 2 || workspace.logits.w != static_cast<int>(implementation.expert_count) || workspace.logits.h != input.h || workspace.logits.elempack != 1 || workspace.logits.elemsize != sizeof(float))
        return false;
    workspace.selected.create(static_cast<int>(selected_columns(options.top_k)),
                              input.h, sizeof(float),
                              implementation.opt.blob_vkallocator);
    if (workspace.selected.empty())
        return false;
    const bool raw_topk = options.score_function == RouterScoreFunction::Softmax
                          && options.normalization == RouterNormalization::SelectedExperts
                          && options.selection_bias.empty() && options.explicit_expert_ids.empty();
    if (raw_topk && implementation.raw_pipeline)
    {
        std::vector<ncnn::vk_constant_type> constants(3);
        constants[0].u32 = options.expert_count;
        constants[1].u32 = options.top_k;
        constants[2].f = options.routed_scaling_factor;
        ncnn::VkMat dispatcher;
        dispatcher.w = implementation.raw_subgroup_size;
        dispatcher.h = input.h;
        dispatcher.c = 1;
        cmd.record_pipeline_readonly(implementation.raw_pipeline.get(),
                                     {workspace.logits, workspace.selected}, {1, 0}, constants, dispatcher);
        return true;
    }
    std::vector<ncnn::vk_constant_type> constants(8);
    constants[0].u32 = options.expert_count;
    constants[1].u32 = options.top_k;
    constants[2].u32 = static_cast<uint32_t>(options.score_function);
    constants[3].u32 = options.normalization == RouterNormalization::SelectedExperts ? 1u : 0u;
    constants[4].u32 = !options.selection_bias.empty() ? 1u : 0u;
    constants[5].u32 = !options.explicit_expert_ids.empty() ? 1u : 0u;
    constants[6].f = options.routed_scaling_factor;
    constants[7].f = options.jitter;
    ncnn::VkMat dispatcher;
    dispatcher.w = 128;
    dispatcher.h = input.h;
    dispatcher.c = 1;
    cmd.record_pipeline_readonly(implementation.pipeline.get(),
                                 {workspace.logits, workspace.selection_bias,
                                  workspace.explicit_ids, workspace.selected},
                                 {1, 1, 1, 0}, constants, dispatcher);
    return true;
}
#endif

} // namespace moe
} // namespace ncnn
