#include "router_vulkan.h"
#include "linear.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/router_select.comp.hex.h"
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
    if (rows == 0 || options.expert_count == 0 || options.expert_count > 2048 || options.top_k == 0 || options.top_k > options.expert_count || options.top_k > 64 || !std::isfinite(options.routed_scaling_factor) || options.routed_scaling_factor <= 0.0f || (options.normalization != RouterNormalization::None && options.normalization != RouterNormalization::SelectedExperts) || (options.score_function != RouterScoreFunction::Softmax && options.score_function != RouterScoreFunction::Sigmoid && options.score_function != RouterScoreFunction::SqrtSoftplus) || (!options.selection_bias.empty() && options.selection_bias.size() != options.expert_count) || (!options.explicit_expert_ids.empty() && options.explicit_expert_ids.size() != rows * options.top_k))
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
#endif
};

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
    const auto projection = Linear::create(weight, bias, LinearDevice::Vulkan, device_index, runtime,
                                           optimization_flags);
    if (!projection || !projection->uses_vulkan())
        return {};
    std::shared_ptr<Router_vulkan> result(new Router_vulkan);
    Implementation& implementation = *result->d;
    implementation.projection = projection;
    implementation.context = projection->vulkan_context();
    implementation.opt = projection->option();
    implementation.expert_count = weight.shape[0];
    implementation.input_columns = weight.shape[1];
    const std::lock_guard<std::mutex> lock(implementation.context->command_mutex());
    const auto spirv = implementation.context->shader_binary(router_select_shader, static_cast<int>(sizeof(router_select_shader) - 1),
                                                             implementation.opt, 0);
    if (!spirv || spirv->empty())
        return {};
    implementation.pipeline = implementation.context->find_pipeline(router_select_shader, 0);
    if (!implementation.pipeline)
    {
        std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(implementation.context->device()));
        pipeline->set_local_size_xyz(128, 1, 1);
        const std::vector<ncnn::vk_specialization_type> specializations;
        if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t),
                             specializations)
            != 0)
            return {};
        implementation.pipeline = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                                  [context = implementation.context](ncnn::Pipeline* value) mutable {
                                                                      {
                                                                          const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                                          delete value;
                                                                      }
                                                                      // Weak cache entries retain the deleter control block after disposal.
                                                                      context.reset();
                                                                  });
        implementation.context->cache_pipeline(router_select_shader, 0,
                                               implementation.pipeline);
    }
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
    std::vector<ncnn::vk_constant_type> constants(7);
    constants[0].u32 = options.expert_count;
    constants[1].u32 = options.top_k;
    constants[2].u32 = static_cast<uint32_t>(options.score_function);
    constants[3].u32 = options.normalization == RouterNormalization::SelectedExperts ? 1u : 0u;
    constants[4].u32 = !options.selection_bias.empty() ? 1u : 0u;
    constants[5].u32 = !options.explicit_expert_ids.empty() ? 1u : 0u;
    constants[6].f = options.routed_scaling_factor;
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
