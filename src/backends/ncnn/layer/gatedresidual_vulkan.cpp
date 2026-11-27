#include "backends/ncnn/layer/gatedresidual_vulkan.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/linear/linear.h"
#include "kernels/vulkan/residual_ple.comp.hex.h"
#include "storage/weightstore.h"

#include <command.h>
#include <pipeline.h>

#include <limits>
#include <mutex>
#include <vector>

namespace ncnn {
namespace moe {

std::shared_ptr<ncnn::Pipeline> residual_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                  const ncnn::Option& opt)
{
    std::shared_ptr<ncnn::Pipeline> pipe;
    (void)create_pipeline(context, residual_ple_shader,
                          static_cast<int>(sizeof(residual_ple_shader) - 1),
                          opt, 0, 0, 128, pipe);
    return pipe;
}

void residual_record(const ncnn::Pipeline* pipeline, uint32_t mode, uint32_t hidden, uint32_t copies,
                     uint32_t rows, uint32_t columns, float epsilon, float offset,
                     std::initializer_list<ncnn::VkMat> values, ncnn::VkCompute& cmd,
                     uint32_t kernel, uint32_t ngram, uint32_t length, uint32_t slot)
{
    static thread_local std::vector<ncnn::VkMat> bindings(6);
    static thread_local std::vector<ncnn::vk_constant_type> constants(11);
    try
    {
        bindings.assign(values.begin(), values.end());
        constants[0].u32 = mode;
        constants[1].u32 = hidden;
        constants[2].u32 = copies;
        constants[3].u32 = rows;
        constants[4].u32 = columns;
        constants[5].u32 = kernel;
        constants[6].u32 = ngram;
        constants[7].u32 = length;
        constants[8].u32 = slot;
        constants[9].f = epsilon;
        constants[10].f = offset;

        ncnn::VkMat dispatch;
        dispatch.w = static_cast<int>((mode == 0 || mode == 5) ? copies * 128 : columns);
        dispatch.h = static_cast<int>(mode == 6 ? 1 : rows);
        dispatch.c = 1;
        cmd.record_pipeline(pipeline, bindings, constants, dispatch);
    }
    catch (...)
    {
        bindings.clear();
        throw;
    }
    bindings.clear();
}

bool residual_allocate(ncnn::VkMat& value, uint32_t columns, int rows,
                       const std::shared_ptr<VulkanContext>& context)
{
    if (columns == 0 || rows <= 0 || columns > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || static_cast<uint64_t>(columns) * rows > std::numeric_limits<uint32_t>::max())
        return false;

    value.create(static_cast<int>(columns), rows, sizeof(float), context->blob_allocator());
    return !value.empty();
}

bool residual_float_weight(const TensorData& tensor, ncnn::VkMat& value,
                           ncnn::VkCompute& command, const ncnn::Option& opt)
{
    ncnn::Mat host;
    if (!prepare_float_upload(tensor, host))
        return false;
    command.record_upload(host, value, opt);
    return !value.empty();
}

class GatedResidual_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    std::shared_ptr<QnkLinear_vulkan> down;
    std::shared_ptr<QnkLinear_vulkan> up;
    std::shared_ptr<QnkLinear_vulkan> inject;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    ncnn::VkMat norm;
    uint32_t hidden = 0;
    uint32_t copies = 0;
    float epsilon = 0.0f;
    float offset = 0.0f;
};

GatedResidual_vulkan::GatedResidual_vulkan()
    : d(new Implementation)
{
}

GatedResidual_vulkan::~GatedResidual_vulkan()
{
    if (d->context)
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        d->norm.release();
    }
}

std::shared_ptr<GatedResidual_vulkan> GatedResidual_vulkan::create(const WeightStore& weights,
                                                                   const GatedResidualPlan& plan,
                                                                   const MoeModelDescriptor& descriptor,
                                                                   uint32_t device,
                                                                   const VulkanRuntimePtr& runtime,
                                                                   uint64_t flags)
{
    auto result = std::shared_ptr<GatedResidual_vulkan>(new GatedResidual_vulkan);
    auto& impl = *result->d;
    impl.down = QnkLinear_vulkan::create(weights.at(plan.mix_down_weight), nullptr, device, runtime, flags);
    impl.up = QnkLinear_vulkan::create(weights.at(plan.mix_up_weight), nullptr, device, runtime, flags);
    if (!impl.down || !impl.up)
        return {};

    if (plan.inject_weight != invalid_tensor_handle)
    {
        impl.inject = QnkLinear_vulkan::create(weights.at(plan.inject_weight), nullptr, device, runtime, flags);
        if (!impl.inject)
            return {};
    }

    impl.context = impl.down->context();
    impl.opt = impl.down->option();
    impl.hidden = descriptor.hidden_size;
    impl.copies = descriptor.hyper_connection_multiplier;
    impl.epsilon = descriptor.norm_epsilon;
    impl.offset = descriptor.norm_weight_offset;

    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    impl.pipeline = residual_pipeline(impl.context, impl.opt);
    if (!impl.pipeline)
        return {};

    ncnn::VkCompute cmd(impl.context->device());
    if (!residual_float_weight(weights.at(plan.norm_weight), impl.norm, cmd, impl.opt)
        || submit_wait(cmd, impl.context->device()) != 0)
        return {};

    return result;
}

const std::shared_ptr<VulkanContext>& GatedResidual_vulkan::context() const noexcept
{
    return d->context;
}

const ncnn::Option& GatedResidual_vulkan::option() const noexcept
{
    return d->opt;
}

bool GatedResidual_vulkan::record_pre(const ncnn::VkMat& input, ncnn::VkMat& reduced,
                                      ncnn::VkMat& injection, ncnn::VkCompute& cmd,
                                      std::vector<ncnn::VkMat>& storage) const
{
    const auto& impl = *d;
    ncnn::VkMat normalized, projected, activated, mixed, raw_inject;
    if (!residual_allocate(normalized, input.w, input.h, impl.context)
        || !residual_allocate(reduced, impl.hidden, input.h, impl.context))
        return false;

    rearm_read(impl.norm);
    residual_record(impl.pipeline.get(), 0, impl.hidden, impl.copies, input.h, input.w,
                    impl.epsilon, impl.offset, {input, impl.norm, input, input, input, normalized}, cmd);

    if (!impl.down->record_forward(normalized, projected, cmd)
        || !residual_allocate(activated, projected.w, input.h, impl.context))
        return false;
    residual_record(impl.pipeline.get(), 1, impl.hidden, impl.copies, input.h, projected.w,
                    0, 0, {projected, projected, projected, projected, projected, activated}, cmd);

    if (!impl.up->record_forward(activated, mixed, cmd))
        return false;
    residual_record(impl.pipeline.get(), 3, impl.hidden, impl.copies, input.h, impl.hidden,
                    0, 0, {normalized, mixed, mixed, mixed, mixed, reduced}, cmd);

    if (impl.inject)
    {
        if (!impl.inject->record_forward(normalized, raw_inject, cmd)
            || !residual_allocate(injection, impl.copies, input.h, impl.context))
            return false;
        residual_record(impl.pipeline.get(), 2, impl.hidden, impl.copies, input.h, impl.copies,
                        0, 0, {raw_inject, raw_inject, raw_inject, raw_inject, raw_inject, injection}, cmd);
    }

    for (const auto& val : {normalized, projected, activated, mixed, raw_inject})
    {
        if (!val.empty())
            storage.push_back(val);
    }
    return true;
}

bool GatedResidual_vulkan::record_post(const ncnn::VkMat& branch, const ncnn::VkMat& residual,
                                       const ncnn::VkMat& injection, ncnn::VkMat& output,
                                       ncnn::VkCompute& cmd) const
{
    const auto& impl = *d;
    if (branch.h != residual.h || injection.h != residual.h
        || !residual_allocate(output, residual.w, residual.h, impl.context))
        return false;

    residual_record(impl.pipeline.get(), 4, impl.hidden, impl.copies, residual.h, residual.w,
                    0, 0, {residual, branch, injection, residual, residual, output}, cmd);
    return true;
}

} // namespace moe
} // namespace ncnn
