#include "gatedresidual_vulkan.h"
#include "linear.h"
#include "vulkancontext.h"
#include "qwen_vulkan_internal.h"
#include "storage/weightstore.h"
#include "kernels/bfloat16.h"
#include "kernels/ops.h"
#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/qwen_elementwise.comp.hex.h"
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif
#include <algorithm>
#include <limits>
#include <mutex>

namespace ncnn {
namespace moe {
#if NCNN_MOE_WITH_VULKAN
// Shared by the residual and PLE record chains. Every operation consumes and
// produces unpacked FP32 activations; immutable native matrices stay packed.
std::shared_ptr<ncnn::Pipeline> qwen_pipeline(const std::shared_ptr<VulkanContext>& context, const ncnn::Option& opt)
{
    const auto binary = context->shader_binary(qwen_elementwise_shader, sizeof(qwen_elementwise_shader) - 1, opt, 0);
    if (!binary || binary->empty()) return {};
    auto result = context->find_pipeline(qwen_elementwise_shader, 0);
    if (result) return result;
    auto pipeline = std::make_unique<ncnn::Pipeline>(context->device());
    pipeline->set_local_size_xyz(128, 1, 1);
    if (pipeline->create(binary->data(), binary->size() * sizeof(uint32_t), {}) != 0) return {};
    result = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [owner = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(owner->command_mutex());
            delete value;
        }
        owner.reset();
    });
    context->cache_pipeline(qwen_elementwise_shader, 0, result);
    return result;
}

void qwen_record(const ncnn::Pipeline* pipeline, uint32_t mode, uint32_t hidden, uint32_t copies,
                 uint32_t rows, uint32_t columns, float epsilon, float offset, const std::vector<ncnn::VkMat>& bindings,
                 ncnn::VkCompute& cmd, uint32_t kernel, uint32_t ngram, uint32_t length, uint32_t slot)
{
    std::vector<ncnn::vk_constant_type> constants(11);
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
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>((mode == 0 || mode == 5) ? copies * 128 : columns);
    dispatcher.h = static_cast<int>(mode == 6 ? 1 : rows);
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
}

bool qwen_allocate(ncnn::VkMat& value, uint32_t columns, int rows, const std::shared_ptr<VulkanContext>& context)
{
    if (columns == 0 || rows <= 0 || columns > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || static_cast<uint64_t>(columns) * rows > std::numeric_limits<uint32_t>::max()) return false;
    value.create(static_cast<int>(columns), rows, sizeof(float), context->blob_allocator());
    return !value.empty();
}

bool qwen_float_weight(const TensorData& tensor, ncnn::VkMat& value,
                       ncnn::VkCompute& command, const ncnn::Option& opt)
{
    if (tensor.element_count() == 0 || tensor.element_count() > static_cast<uint64_t>(std::numeric_limits<int>::max())) return false;
    ncnn::Mat host(static_cast<int>(tensor.element_count()), sizeof(float));
    float* destination = host;
    if (tensor.dtype == DType::BFloat16)
    {
        const auto source = tensor.bfloat16_values();
        if (source.size() != tensor.element_count()) return false;
        for (size_t i = 0; i < source.size(); ++i) destination[i] = bfloat16_to_float(source[i]);
    }
    else if (tensor.dtype == DType::Float32)
    {
        const auto source = tensor.float32_values();
        if (source.size() != tensor.element_count()) return false;
        std::copy(source.begin(), source.end(), destination);
    }
    else
        return false;
    command.record_upload(host, value, opt);
    return !value.empty();
}

void qwen_prepare_read(const ncnn::VkMat& value)
{
    if (value.data)
    {
        value.data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        value.data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
}
#endif

class GatedResidual_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    std::shared_ptr<QnkLinear_vulkan> down, up, inject;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    ncnn::VkMat norm;
    uint32_t hidden = 0, copies = 0;
    float epsilon = 0, offset = 0;
#endif
};
GatedResidual_vulkan::GatedResidual_vulkan() : d(new Implementation)
{
}
GatedResidual_vulkan::~GatedResidual_vulkan()
{
#if NCNN_MOE_WITH_VULKAN
    if (d->context)
    {
        const std::lock_guard<std::mutex> lock(d->context->command_mutex());
        d->norm.release();
    }
#endif
}
std::shared_ptr<GatedResidual_vulkan> GatedResidual_vulkan::create(const WeightStore& weights,
                                                                   const GatedResidualPlan& plan, const MoeModelDescriptor& descriptor, uint32_t device,
                                                                   const VulkanRuntimePtr& runtime, uint64_t flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (plan.norm_weight >= weights.size() || plan.mix_down_weight >= weights.size()
        || plan.mix_up_weight >= weights.size() || descriptor.hidden_size == 0 || descriptor.hyper_connection_multiplier == 0) return {};
    auto result = std::shared_ptr<GatedResidual_vulkan>(new GatedResidual_vulkan);
    auto& impl = *result->d;
    impl.down = QnkLinear_vulkan::create(weights.at(plan.mix_down_weight), nullptr, device, runtime, flags);
    impl.up = QnkLinear_vulkan::create(weights.at(plan.mix_up_weight), nullptr, device, runtime, flags);
    if (!impl.down || !impl.up) return {};
    if (plan.inject_weight != invalid_tensor_handle)
    {
        if (plan.inject_weight >= weights.size()) return {};
        impl.inject = QnkLinear_vulkan::create(weights.at(plan.inject_weight), nullptr, device, runtime, flags);
        if (!impl.inject) return {};
    }
    impl.context = impl.down->vulkan_context();
    impl.opt = impl.down->option();
    impl.hidden = descriptor.hidden_size;
    impl.copies = descriptor.hyper_connection_multiplier;
    impl.epsilon = descriptor.norm_epsilon;
    impl.offset = descriptor.norm_weight_offset;
    if (impl.down->input_columns() != impl.hidden * impl.copies
        || impl.up->input_columns() != impl.down->output_columns() || impl.up->output_columns() != impl.hidden * impl.copies
        || (impl.inject && (impl.inject->input_columns() != impl.hidden * impl.copies || impl.inject->output_columns() != impl.copies))) return {};
    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    impl.pipeline = qwen_pipeline(impl.context, impl.opt);
    if (!impl.pipeline || weights.at(plan.norm_weight).element_count() != static_cast<uint64_t>(impl.hidden) * impl.copies) return {};
    ncnn::VkCompute command(impl.context->device());
    if (!qwen_float_weight(weights.at(plan.norm_weight), impl.norm, command, impl.opt)
        || submit_compute_and_wait(command, impl.context->device()) != 0) return {};
    return result;
#else
    (void)weights;
    (void)plan;
    (void)descriptor;
    (void)device;
    (void)runtime;
    (void)flags;
    return {};
#endif
}
uint64_t GatedResidual_vulkan::device_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->down->device_bytes() + d->up->device_bytes() + (d->inject ? d->inject->device_bytes() : 0) + d->norm.total() * d->norm.elemsize;
#else
    return 0;
#endif
}
#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& GatedResidual_vulkan::context() const noexcept
{ return d->context; }
const ncnn::Option& GatedResidual_vulkan::option() const noexcept
{ return d->opt; }
bool GatedResidual_vulkan::record_pre(const ncnn::VkMat& input, ncnn::VkMat& reduced,
                                      ncnn::VkMat& injection, ncnn::VkCompute& cmd, std::vector<ncnn::VkMat>& storage) const
{
    const auto& impl = *d;
    if (input.dims != 2 || input.w != static_cast<int>(impl.hidden * impl.copies) || input.h <= 0 || input.elemsize != sizeof(float) || input.elempack != 1) return false;
    ncnn::VkMat normalized, projected, activated, mixed, raw_inject;
    if (!qwen_allocate(normalized, input.w, input.h, impl.context)
        || !qwen_allocate(reduced, impl.hidden, input.h, impl.context)) return false;
    qwen_prepare_read(impl.norm);
    qwen_record(impl.pipeline.get(), 0, impl.hidden, impl.copies, input.h, input.w,
                impl.epsilon, impl.offset, {input, impl.norm, input, input, input, normalized}, cmd);
    if (!impl.down->record_forward(normalized, projected, cmd, storage)
        || !qwen_allocate(activated, projected.w, input.h, impl.context)) return false;
    qwen_record(impl.pipeline.get(), 1, impl.hidden, impl.copies, input.h, projected.w,
                0, 0, {projected, projected, projected, projected, projected, activated}, cmd);
    if (!impl.up->record_forward(activated, mixed, cmd, storage)) return false;
    qwen_record(impl.pipeline.get(), 3, impl.hidden, impl.copies, input.h, impl.hidden,
                0, 0, {normalized, mixed, mixed, mixed, mixed, reduced}, cmd);
    if (impl.inject)
    {
        if (!impl.inject->record_forward(normalized, raw_inject, cmd, storage)
            || !qwen_allocate(injection, impl.copies, input.h, impl.context)) return false;
        qwen_record(impl.pipeline.get(), 2, impl.hidden, impl.copies, input.h, impl.copies,
                    0, 0, {raw_inject, raw_inject, raw_inject, raw_inject, raw_inject, injection}, cmd);
    }
    for (const auto& value : {normalized, projected, activated, mixed, raw_inject})
        if (!value.empty()) storage.push_back(value);
    return true;
}
bool GatedResidual_vulkan::record_post(const ncnn::VkMat& branch, const ncnn::VkMat& residual,
                                       const ncnn::VkMat& injection, ncnn::VkMat& output, ncnn::VkCompute& cmd) const
{
    const auto& impl = *d;
    if (branch.w != static_cast<int>(impl.hidden) || branch.h != residual.h || residual.w != static_cast<int>(impl.hidden * impl.copies)
        || injection.w != static_cast<int>(impl.copies) || injection.h != residual.h || !qwen_allocate(output, residual.w, residual.h, impl.context)) return false;
    qwen_record(impl.pipeline.get(), 4, impl.hidden, impl.copies, residual.h, residual.w,
                0, 0, {residual, branch, injection, residual, residual, output}, cmd);
    return true;
}
#endif
} // namespace moe
} // namespace ncnn
