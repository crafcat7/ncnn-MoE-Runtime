#include "hyperconnection_vulkan.h"
#include "vulkancontext.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/hyper_connection_project.comp.hex.h"
#include "kernels/vulkan/hyper_connection_mix.comp.hex.h"
#include "kernels/vulkan/hyper_connection_reduce.comp.hex.h"
#include "kernels/vulkan/hyper_connection_post.comp.hex.h"

#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif

#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static bool create_hyper_pipeline(const std::shared_ptr<VulkanContext>& context,
                                  const ncnn::Option& opt,
                                  const char* shader,
                                  int shader_size,
                                  std::shared_ptr<ncnn::Pipeline>& destination)
{
    const auto spirv = context->shader_binary(shader, shader_size, opt, 0);
    if (!spirv || spirv->empty())
        return false;
    destination = context->find_pipeline(shader, 0);
    if (destination)
        return true;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(context->device()));
    pipeline->set_local_size_xyz(128, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        // Weak cache entries retain the deleter control block after disposal.
        context.reset();
    });
    context->cache_pipeline(shader, 0, destination);
    return true;
}

static bool valid_hyper_buffer(const ncnn::VkMat& input, uint32_t columns, int rows)
{
    return !input.empty() && input.dims == 2 && input.w == static_cast<int>(columns)
           && input.h == rows && rows > 0 && input.elempack == 1 && input.elemsize == sizeof(float)
           && static_cast<uint64_t>(columns) * static_cast<uint64_t>(rows) <= std::numeric_limits<uint32_t>::max();
}

static bool allocate_hyper_buffer(ncnn::VkMat& output, uint32_t columns, int rows,
                                  const std::shared_ptr<VulkanContext>& context)
{
    if (columns == 0 || columns > static_cast<uint32_t>(std::numeric_limits<int>::max()) || rows <= 0
        || static_cast<uint64_t>(columns) * static_cast<uint64_t>(rows) > std::numeric_limits<uint32_t>::max())
        return false;
    output.create(static_cast<int>(columns), rows, sizeof(float), context->blob_allocator());
    return !output.empty();
}

static void prepare_hyper_weight_read(const ncnn::VkMat& weight)
{
    if (weight.data)
    {
        // Recording mutates shared access metadata before the command runs.
        // Re-arm immutable uploads when a preceding recorder was abandoned.
        weight.data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        weight.data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
}

static void record_hyper_pipeline(const ncnn::Pipeline* pipeline,
                                  const std::vector<ncnn::VkMat>& bindings,
                                  const std::vector<ncnn::vk_constant_type>& constants,
                                  uint32_t width, uint32_t height, uint32_t channels,
                                  ncnn::VkCompute& cmd)
{
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(width);
    dispatcher.h = static_cast<int>(height);
    dispatcher.c = static_cast<int>(channels);
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
}
#endif

class HyperConnection_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    ncnn::VkMat function;
    ncnn::VkMat scale;
    ncnn::VkMat base;
    uint32_t columns = 0;
    uint32_t output_columns = 0;
    uint32_t copies = 0;
    uint32_t sinkhorn_iterations = 0;
    float norm_epsilon = 0.0f;
    float hyper_epsilon = 0.0f;
    bool head = false;
    std::shared_ptr<ncnn::Pipeline> project_pipeline;
    std::shared_ptr<ncnn::Pipeline> mix_pipeline;
    std::shared_ptr<ncnn::Pipeline> reduce_pipeline;
    std::shared_ptr<ncnn::Pipeline> post_pipeline;

    void record_project(const ncnn::VkMat& input, ncnn::VkMat& output, ncnn::VkCompute& cmd) const
    {
        std::vector<ncnn::vk_constant_type> constants(3);
        constants[0].u32 = columns;
        constants[1].u32 = output_columns;
        constants[2].f = norm_epsilon;
        prepare_hyper_weight_read(function);
        record_hyper_pipeline(project_pipeline.get(), {input, function, output}, constants, 128, output_columns, input.h, cmd);
    }

    void record_mix(const ncnn::VkMat& projection, ncnn::VkMat& pre,
                    ncnn::VkMat& post, ncnn::VkMat& combine, ncnn::VkCompute& cmd) const
    {
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].u32 = copies;
        constants[1].u32 = sinkhorn_iterations;
        constants[2].f = hyper_epsilon;
        constants[3].u32 = head ? 1 : 0;
        prepare_hyper_weight_read(scale);
        prepare_hyper_weight_read(base);
        record_hyper_pipeline(mix_pipeline.get(), {projection, scale, base, pre, post, combine}, constants, 128, 1, projection.h, cmd);
    }

    void record_reduce(const ncnn::VkMat& input, const ncnn::VkMat& pre, ncnn::VkMat& output, ncnn::VkCompute& cmd) const
    {
        std::vector<ncnn::vk_constant_type> constants(2);
        constants[0].u32 = columns / copies;
        constants[1].u32 = copies;
        record_hyper_pipeline(reduce_pipeline.get(), {input, pre, output}, constants, columns / copies, input.h, 1, cmd);
    }
#endif
};

HyperConnection_vulkan::HyperConnection_vulkan()
    : d(new Implementation)
{
}

HyperConnection_vulkan::~HyperConnection_vulkan() = default;

std::shared_ptr<HyperConnection_vulkan> HyperConnection_vulkan::create(const TensorData& function,
                                                                       const TensorData& scale,
                                                                       const TensorData& base,
                                                                       uint32_t multiplier,
                                                                       uint32_t sinkhorn_iterations,
                                                                       float norm_epsilon,
                                                                       float hyper_epsilon,
                                                                       uint32_t vulkan_device_index,
                                                                       const VulkanRuntimePtr& vulkan_runtime,
                                                                       uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (multiplier == 0 || multiplier > 8 || norm_epsilon <= 0.0f || hyper_epsilon <= 0.0f
        || !std::isfinite(norm_epsilon) || !std::isfinite(hyper_epsilon)
        || function.dtype != DType::Float32 || function.shape.size() != 2 || function.shape[1] == 0
        || function.shape[1] % multiplier != 0 || function.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || function.float32_values().size() != function.element_count()
        || base.dtype != DType::Float32 || base.shape.size() != 1
        || base.shape[0] != function.shape[0] || base.float32_values().size() != function.shape[0]
        || scale.dtype != DType::Float32)
        return {};
    const bool head = function.shape[0] == multiplier;
    if ((!head && function.shape[0] != multiplier * (multiplier + 2))
        || scale.float32_values().size() != (head ? 1u : 3u)
        || (!head && sinkhorn_iterations == 0)
        || function.element_count() > std::numeric_limits<uint32_t>::max())
        return {};
    std::shared_ptr<VulkanContext> context = VulkanContext::acquire(vulkan_device_index, vulkan_runtime, optimization_flags);
    if (!context
        || (static_cast<uint64_t>(function.shape[1]) + 127) / 128 > context->device()->info.max_workgroup_count_x())
        return {};
    auto result = std::shared_ptr<HyperConnection_vulkan>(new HyperConnection_vulkan);
    Implementation& implementation = *result->d;
    implementation.context = context;
    implementation.columns = function.shape[1];
    implementation.output_columns = function.shape[0];
    implementation.copies = multiplier;
    implementation.sinkhorn_iterations = sinkhorn_iterations;
    implementation.norm_epsilon = norm_epsilon;
    implementation.hyper_epsilon = hyper_epsilon;
    implementation.head = head;
    implementation.opt.use_packing_layout = false;
    implementation.opt.use_fp16_packed = false;
    implementation.opt.use_fp16_storage = false;
    implementation.opt.use_fp16_arithmetic = false;
    implementation.opt.use_bf16_packed = false;
    implementation.opt.use_bf16_storage = false;
    implementation.opt.blob_vkallocator = context->blob_allocator();
    implementation.opt.workspace_vkallocator = context->blob_allocator();
    implementation.opt.staging_vkallocator = context->staging_allocator();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if (!create_hyper_pipeline(context, implementation.opt, hyper_connection_project_shader, static_cast<int>(sizeof(hyper_connection_project_shader) - 1), implementation.project_pipeline)
        || !create_hyper_pipeline(context, implementation.opt, hyper_connection_mix_shader, static_cast<int>(sizeof(hyper_connection_mix_shader) - 1), implementation.mix_pipeline)
        || !create_hyper_pipeline(context, implementation.opt, hyper_connection_reduce_shader, static_cast<int>(sizeof(hyper_connection_reduce_shader) - 1), implementation.reduce_pipeline)
        || !create_hyper_pipeline(context, implementation.opt, hyper_connection_post_shader, static_cast<int>(sizeof(hyper_connection_post_shader) - 1), implementation.post_pipeline))
        return {};
    ncnn::Mat function_host(static_cast<int>(implementation.columns), static_cast<int>(implementation.output_columns), sizeof(float));
    ncnn::Mat scale_host(static_cast<int>(scale.float32_values().size()), sizeof(float));
    ncnn::Mat base_host(static_cast<int>(base.float32_values().size()), sizeof(float));
    if (function_host.empty() || scale_host.empty() || base_host.empty())
        return {};
    function_host.fill(0.0f);
    std::memcpy(function_host.data, function.float32_values().data(), static_cast<size_t>(function.element_count()) * sizeof(float));
    std::memcpy(scale_host.data, scale.float32_values().data(), scale_host.total() * sizeof(float));
    std::memcpy(base_host.data, base.float32_values().data(), base_host.total() * sizeof(float));
    ncnn::VkCompute cmd(context->device(), context->command_optimization_flags());
    // record_upload packs a 2-D matrix along its rows even when packing layout is disabled.
    // These shaders address the original row-major FP32 values, so preserve pack1 storage.
    cmd.record_clone(function_host, implementation.function, implementation.opt);
    cmd.record_clone(scale_host, implementation.scale, implementation.opt);
    cmd.record_clone(base_host, implementation.base, implementation.opt);
    if (implementation.function.empty() || implementation.scale.empty() || implementation.base.empty()
        || submit_compute_and_wait(cmd, context->device()) != 0)
        return {};
    ++context->runtime_state().compute_submissions;
    return result;
#else
    (void)function;
    (void)scale;
    (void)base;
    (void)multiplier;
    (void)sinkhorn_iterations;
    (void)norm_epsilon;
    (void)hyper_epsilon;
    (void)vulkan_device_index;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& HyperConnection_vulkan::vulkan_context() const noexcept
{
    return d->context;
}

const ncnn::Option& HyperConnection_vulkan::option() const noexcept
{
    return d->opt;
}

uint32_t HyperConnection_vulkan::input_columns() const noexcept
{
    return d->columns;
}

uint32_t HyperConnection_vulkan::multiplier() const noexcept
{
    return d->copies;
}

bool HyperConnection_vulkan::is_head() const noexcept
{
    return d->head;
}

bool HyperConnection_vulkan::record_pre(const ncnn::VkMat& input,
                                        HyperConnectionMix_vulkan& result,
                                        HyperConnectionWorkspace_vulkan& workspace,
                                        ncnn::VkCompute& cmd) const
{
    if (d->head || !valid_hyper_buffer(input, d->columns, input.h)
        || static_cast<uint32_t>(input.h) > d->context->device()->info.max_workgroup_count_y()
        || static_cast<uint32_t>(input.h) > d->context->device()->info.max_workgroup_count_z()
        || &input == &result.reduced || &input == &result.post || &input == &result.combine
        || &input == &workspace.projection || &input == &workspace.pre)
        return false;
    HyperConnectionMix_vulkan next;
    HyperConnectionWorkspace_vulkan work;
    next.context = d->context;
    work.context = d->context;
    if (!allocate_hyper_buffer(next.reduced, d->columns / d->copies, input.h, d->context)
        || !allocate_hyper_buffer(next.post, d->copies, input.h, d->context)
        || !allocate_hyper_buffer(next.combine, d->copies * d->copies, input.h, d->context)
        || !allocate_hyper_buffer(work.projection, d->output_columns, input.h, d->context)
        || !allocate_hyper_buffer(work.pre, d->copies, input.h, d->context))
        return false;
    d->record_project(input, work.projection, cmd);
    d->record_mix(work.projection, work.pre, next.post, next.combine, cmd);
    d->record_reduce(input, work.pre, next.reduced, cmd);
    result = std::move(next);
    workspace = std::move(work);
    return true;
}

bool HyperConnection_vulkan::record_post(const ncnn::VkMat& branch,
                                         const ncnn::VkMat& residual,
                                         const HyperConnectionMix_vulkan& mix,
                                         ncnn::VkMat& output,
                                         ncnn::VkCompute& cmd) const
{
    if (d->head || mix.context != d->context
        || static_cast<uint32_t>(branch.h) > d->context->device()->info.max_workgroup_count_y()
        || !valid_hyper_buffer(branch, d->columns / d->copies, branch.h)
        || !valid_hyper_buffer(residual, d->columns, branch.h)
        || !valid_hyper_buffer(mix.post, d->copies, branch.h)
        || !valid_hyper_buffer(mix.combine, d->copies * d->copies, branch.h)
        || &output == &branch || &output == &residual || &output == &mix.post || &output == &mix.combine)
        return false;
    ncnn::VkMat next;
    if (!allocate_hyper_buffer(next, d->columns, branch.h, d->context))
        return false;
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].u32 = d->columns / d->copies;
    constants[1].u32 = d->copies;
    record_hyper_pipeline(d->post_pipeline.get(), {branch, residual, mix.post, mix.combine, next}, constants, d->columns, branch.h, 1, cmd);
    output = std::move(next);
    return true;
}

bool HyperConnection_vulkan::record_head(const ncnn::VkMat& input,
                                         ncnn::VkMat& output,
                                         HyperConnectionWorkspace_vulkan& workspace,
                                         ncnn::VkCompute& cmd) const
{
    if (!d->head || !valid_hyper_buffer(input, d->columns, input.h) || &output == &input
        || static_cast<uint32_t>(input.h) > d->context->device()->info.max_workgroup_count_y()
        || static_cast<uint32_t>(input.h) > d->context->device()->info.max_workgroup_count_z()
        || &input == &workspace.projection || &input == &workspace.pre
        || &output == &workspace.projection || &output == &workspace.pre)
        return false;
    HyperConnectionWorkspace_vulkan work;
    work.context = d->context;
    ncnn::VkMat next;
    if (!allocate_hyper_buffer(next, d->columns / d->copies, input.h, d->context)
        || !allocate_hyper_buffer(work.projection, d->output_columns, input.h, d->context)
        || !allocate_hyper_buffer(work.pre, d->copies, input.h, d->context))
        return false;
    d->record_project(input, work.projection, cmd);
    ncnn::VkMat unused_post;
    ncnn::VkMat unused_combine;
    d->record_mix(work.projection, work.pre, unused_post, unused_combine, cmd);
    d->record_reduce(input, work.pre, next, cmd);
    output = std::move(next);
    workspace = std::move(work);
    return true;
}
#endif

} // namespace moe
} // namespace ncnn
