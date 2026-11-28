#include "backends/ncnn/layer/hyperconnection_vulkan.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/layer/rmsnorm_vulkan.h"
#include "backends/ncnn/linear/qnk.h"
#include "kernels/vulkan/hyper_connection_mix.comp.hex.h"
#include "kernels/vulkan/hyper_connection_post.comp.hex.h"
#include "kernels/vulkan/hyper_connection_project.comp.hex.h"
#include "kernels/vulkan/hyper_connection_reduce.comp.hex.h"

#include <command.h>
#include <gpu.h>
#include <pipeline.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static bool allocate_hyper_buffer(ncnn::VkMat& output, uint32_t columns, int rows,
                                  const std::shared_ptr<VulkanContext>& context)
{
    output.create(static_cast<int>(columns), rows, sizeof(float), context->blob_allocator());
    return !output.empty();
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

class HyperConnection_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    ncnn::VkMat function;
    ncnn::VkMat scale;
    ncnn::VkMat base;
    std::shared_ptr<QnkLinear_vulkan> native_projection;
    std::shared_ptr<RmsNorm_vulkan> norm;
    uint32_t columns = 0;
    uint32_t output_columns = 0;
    uint32_t copies = 0;
    uint32_t sinkhorn_iterations = 0;
    float norm_epsilon = 0.0f;
    float hyper_epsilon = 0.0f;
    bool head = false;
    uint64_t device_bytes = 0;
    std::shared_ptr<ncnn::Pipeline> project_pipeline;
    std::shared_ptr<ncnn::Pipeline> mix_pipeline;
    std::shared_ptr<ncnn::Pipeline> reduce_pipeline;
    std::shared_ptr<ncnn::Pipeline> post_pipeline;

    bool record_project(const ncnn::VkMat& input, ncnn::VkMat& output,
                        ncnn::VkMat& normalized, ncnn::VkCompute& cmd) const
    {
        if (native_projection)
        {
            if (!norm->record(input, normalized, cmd))
                return false;
            native_projection->record(normalized, output, cmd);
            return true;
        }

        std::vector<ncnn::vk_constant_type> constants(3);
        constants[0].u32 = columns;
        constants[1].u32 = output_columns;
        constants[2].f = norm_epsilon;
        // Recording mutates shared access metadata before the command runs.
        // Re-arm immutable uploads when a preceding recorder was abandoned.
        rearm_read(function);
        record_hyper_pipeline(project_pipeline.get(), {input, function, output}, constants, 128, output_columns, input.h, cmd);
        return true;
    }

    void record_mix(const ncnn::VkMat& projection, ncnn::VkMat& pre,
                    ncnn::VkMat& post, ncnn::VkMat& combine, ncnn::VkCompute& cmd) const
    {
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].u32 = copies;
        constants[1].u32 = sinkhorn_iterations;
        constants[2].f = hyper_epsilon;
        constants[3].u32 = head ? 1 : 0;
        rearm_read(scale);
        rearm_read(base);
        record_hyper_pipeline(mix_pipeline.get(), {projection, scale, base, pre, post, combine}, constants, 128, 1, projection.h, cmd);
    }

    void record_reduce(const ncnn::VkMat& input, const ncnn::VkMat& pre, ncnn::VkMat& output, ncnn::VkCompute& cmd) const
    {
        std::vector<ncnn::vk_constant_type> constants(2);
        constants[0].u32 = columns / copies;
        constants[1].u32 = copies;
        record_hyper_pipeline(reduce_pipeline.get(), {input, pre, output}, constants, columns / copies, input.h, 1, cmd);
    }
};

HyperConnection_vulkan::HyperConnection_vulkan()
    : d(new Implementation)
{
}

HyperConnection_vulkan::~HyperConnection_vulkan() = default;

uint64_t HyperConnection_vulkan::device_bytes() const noexcept
{
    return d->device_bytes;
}

uint32_t HyperConnection_vulkan::int4_ops() const noexcept
{
    return d->native_projection && d->native_projection->dtype() == DType::Int4 ? 1u : 0u;
}

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
    const bool native_projection = function.dtype == DType::Int4 || is_qnk_dtype(function.dtype)
                                   || function.dtype == DType::BFloat16
                                   || (function.dtype == DType::Float32 && !function.input_column_indices.empty());
    const bool head = function.shape[0] == multiplier;
    std::shared_ptr<VulkanContext> context = VulkanContext::acquire(vulkan_device_index, vulkan_runtime, optimization_flags);
    if (!context)
        return {};

    // Declare the owner before locking: on failure its pipeline deleters must
    // run after the command lock is released.
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
    implementation.opt = make_vk_opt(*context);

    if (native_projection)
    {
        implementation.native_projection = QnkLinear_vulkan::create(function, nullptr, vulkan_device_index,
                                                                    vulkan_runtime, optimization_flags);
        if (!implementation.native_projection || implementation.native_projection->context() != context
            || activation_variant(implementation.native_projection->option()) != 0)
            return {};

        TensorData norm_weight;
        norm_weight.dtype = DType::Float32;
        norm_weight.shape = {implementation.columns};
        norm_weight.float32_data.assign(implementation.columns, 1.0f);
        implementation.norm = RmsNorm_vulkan::create(norm_weight, norm_epsilon, 0.0f, context, implementation.opt);
        if (!implementation.norm)
            return {};
    }

    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if ((!native_projection
         && !create_pipeline(context, hyper_connection_project_shader,
                             static_cast<int>(sizeof(hyper_connection_project_shader) - 1),
                             implementation.opt, 0, 0, 128, implementation.project_pipeline))
        || !create_pipeline(context, hyper_connection_mix_shader,
                            static_cast<int>(sizeof(hyper_connection_mix_shader) - 1),
                            implementation.opt, 0, 0, 128, implementation.mix_pipeline)
        || !create_pipeline(context, hyper_connection_reduce_shader,
                            static_cast<int>(sizeof(hyper_connection_reduce_shader) - 1),
                            implementation.opt, 0, 0, 128, implementation.reduce_pipeline)
        || !create_pipeline(context, hyper_connection_post_shader,
                            static_cast<int>(sizeof(hyper_connection_post_shader) - 1),
                            implementation.opt, 0, 0, 128, implementation.post_pipeline))
        return {};

    ncnn::Mat function_host;
    ncnn::Mat scale_host;
    ncnn::Mat base_host;
    if (!prepare_float_upload(scale, scale_host) || !prepare_float_upload(base, base_host))
        return {};

    if (!native_projection)
    {
        if (function.float32_values().size() != static_cast<size_t>(function.element_count()))
            return {};
        function_host.create(static_cast<int>(implementation.columns),
                             static_cast<int>(implementation.output_columns), sizeof(float));
        if (function_host.empty())
            return {};
        std::memcpy(function_host.data, function.float32_values().data(),
                    static_cast<size_t>(function.element_count()) * sizeof(float));
    }

    ncnn::VkCompute cmd(context->device());
    // record_upload packs a 2-D matrix along its rows even when packing layout is disabled.
    // These shaders address the original row-major FP32 values, so preserve pack1 storage.
    if (!native_projection)
        cmd.record_clone(function_host, implementation.function, implementation.opt);
    cmd.record_clone(scale_host, implementation.scale, implementation.opt);
    cmd.record_clone(base_host, implementation.base, implementation.opt);
    if ((!native_projection && implementation.function.empty())
        || implementation.scale.empty() || implementation.base.empty()
        || submit_wait(cmd, context->device()) != 0)
        return {};

    const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
    implementation.device_bytes = align_up(scale.element_count() * sizeof(float), alignment)
                                  + align_up(base.element_count() * sizeof(float), alignment);
    if (native_projection)
        implementation.device_bytes += implementation.native_projection->device_bytes()
                                       + align_up(static_cast<uint64_t>(implementation.columns) * sizeof(float), alignment);
    else
        implementation.device_bytes += align_up(function.element_count() * sizeof(float), alignment);

    ++context->runtime_state().compute_submissions;
    return result;
}

const std::shared_ptr<VulkanContext>& HyperConnection_vulkan::context() const noexcept
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
    if (d->head
        || static_cast<uint32_t>(input.h) > d->context->device()->info.max_workgroup_count_y()
        || static_cast<uint32_t>(input.h) > d->context->device()->info.max_workgroup_count_z())
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

    if (!d->record_project(input, work.projection, work.normalized, cmd))
        return false;
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
    if (d->head
        || static_cast<uint32_t>(branch.h) > d->context->device()->info.max_workgroup_count_y()
        || residual.h != branch.h)
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
    if (!d->head
        || static_cast<uint32_t>(input.h) > d->context->device()->info.max_workgroup_count_y()
        || static_cast<uint32_t>(input.h) > d->context->device()->info.max_workgroup_count_z())
        return false;

    HyperConnectionWorkspace_vulkan work;
    work.context = d->context;
    ncnn::VkMat next;
    if (!allocate_hyper_buffer(next, d->columns / d->copies, input.h, d->context)
        || !allocate_hyper_buffer(work.projection, d->output_columns, input.h, d->context)
        || !allocate_hyper_buffer(work.pre, d->copies, input.h, d->context))
        return false;

    if (!d->record_project(input, work.projection, work.normalized, cmd))
        return false;
    ncnn::VkMat unused_post;
    ncnn::VkMat unused_combine;
    d->record_mix(work.projection, work.pre, unused_post, unused_combine, cmd);
    d->record_reduce(input, work.pre, next, cmd);
    output = std::move(next);
    workspace = std::move(work);
    return true;
}

} // namespace moe
} // namespace ncnn
