#include "backends/ncnn/expert/ffn_region.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/core/tensor.h"
#include "backends/ncnn/layer/moecombine_vulkan.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace ncnn {
namespace moe {

class ResidentFfnRegion_vulkan::Implementation
{
public:
    struct Slot
    {
        size_t index = 0;
        ncnn::VkMat value;
        std::vector<ExpertRoute> routes;
    };

    bool reject_wave() noexcept
    {
        if (continuation && !continuation_failed)
        {
            continuation_failed = true;
            continuation->abort();
        }
        return false;
    }

    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<MoeCombine_vulkan> combiner;
    std::shared_ptr<const DeviceTensor_vulkan> source_device;
    std::shared_ptr<const DeviceTensor_vulkan> device_output;
    std::shared_ptr<ResidentFfnContinuation_vulkan> continuation;
    ActivationBuffer source;
    ActivationBuffer aggregate;
    ncnn::VkMat input;
    ncnn::VkMat pending_input;
    ncnn::VkMat input_staging;
    ncnn::VkMat combined;
    ncnn::VkMat download;
    std::vector<Slot> slots;
    std::vector<Slot> pending;
    std::vector<ncnn::VkMat> storage;
    size_t rows = 0;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint64_t pending_combine_dispatches = 0;
    uint64_t completed_combine_dispatches = 0;
    ResidentFfnOutput_vulkan output_mode = ResidentFfnOutput_vulkan::Host;
    bool completed = false;
    bool continuation_failed = false;
    bool final_wave = false;
    bool recording = false;
    bool outputs_recorded = false;
    bool wave_started = false;
    bool continuation_recorded = false;
    bool continuation_submit_attempted = false;
};

ResidentFfnRegion_vulkan::ResidentFfnRegion_vulkan()
    : d(new Implementation)
{
}

ResidentFfnRegion_vulkan::~ResidentFfnRegion_vulkan()
{
    if (d->continuation && !d->continuation_failed)
        d->continuation->abort();
}

std::shared_ptr<ResidentFfnRegion_vulkan> ResidentFfnRegion_vulkan::create(const ActivationBuffer& normalized,
                                                                           size_t expert_count,
                                                                           uint32_t output_columns)
{
    if (normalized.dtype() != DType::Float32 || normalized.rows() == 0 || normalized.columns() == 0
        || normalized.rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || normalized.columns() > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || output_columns == 0 || output_columns > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || expert_count == 0)
        return {};

    auto result = std::shared_ptr<ResidentFfnRegion_vulkan>(new ResidentFfnRegion_vulkan);
    result->d->source = normalized;
    result->d->rows = normalized.rows();
    result->d->input_columns = normalized.columns();
    result->d->output_columns = output_columns;
    result->d->slots.resize(expert_count);
    return result;
}

std::shared_ptr<ResidentFfnRegion_vulkan> ResidentFfnRegion_vulkan::create(std::shared_ptr<const DeviceTensor_vulkan> normalized,
                                                                           size_t expert_count,
                                                                           uint32_t output_columns,
                                                                           ResidentFfnOutput_vulkan output)
{
    if ((output != ResidentFfnOutput_vulkan::Host && output != ResidentFfnOutput_vulkan::Device)
        || !normalized || normalized->empty() || !normalized->value() || !normalized->context()
        || normalized->rows() == 0 || normalized->rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || normalized->columns() == 0 || normalized->columns() > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || output_columns == 0 || output_columns > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || expert_count == 0)
        return {};

    const ncnn::VkMat* value = normalized->value();
    if (value->dims != 2 || value->elempack != 1 || value->elemsize != sizeof(float)
        || value->w != static_cast<int>(normalized->columns()) || value->h != static_cast<int>(normalized->rows()))
        return {};

    auto result = std::shared_ptr<ResidentFfnRegion_vulkan>(new ResidentFfnRegion_vulkan);
    result->d->rows = normalized->rows();
    result->d->input_columns = normalized->columns();
    result->d->output_columns = output_columns;
    result->d->output_mode = output;
    result->d->context = normalized->context();
    result->d->source_device = std::move(normalized);
    result->d->slots.resize(expert_count);
    return result;
}

bool ResidentFfnRegion_vulkan::complete() const noexcept
{
    return d->completed && !d->continuation_failed;
}

size_t ResidentFfnRegion_vulkan::rows() const noexcept
{
    return d->rows;
}

uint32_t ResidentFfnRegion_vulkan::columns() const noexcept
{
    return d->input_columns;
}

uint64_t ResidentFfnRegion_vulkan::pending_combine_dispatches() const noexcept
{
    return d->pending_combine_dispatches;
}

uint64_t ResidentFfnRegion_vulkan::completed_combine_dispatches() const noexcept
{
    return d->continuation_failed ? 0 : d->completed_combine_dispatches;
}

bool ResidentFfnRegion_vulkan::set_continuation(std::shared_ptr<ResidentFfnContinuation_vulkan> continuation)
{
    if (!continuation || d->continuation || d->continuation_failed
        || d->output_mode != ResidentFfnOutput_vulkan::Device || !d->context
        || continuation->context() != d->context || d->wave_started || d->recording || d->completed)
        return false;

    d->continuation = std::move(continuation);
    return true;
}

void ResidentFfnRegion_vulkan::mark_submit_attempted() noexcept
{
    if (d->continuation && !d->continuation_failed && d->recording && d->outputs_recorded
        && d->final_wave && d->continuation_recorded && !d->continuation_submit_attempted)
    {
        d->continuation_submit_attempted = true;
        d->continuation->mark_submit_attempted();
    }
}

bool ResidentFfnRegion_vulkan::take_output(ActivationBuffer& output)
{
    if (!d->completed || d->continuation_failed || d->output_mode != ResidentFfnOutput_vulkan::Host
        || d->aggregate.rows() != d->rows || d->aggregate.columns() != d->output_columns)
        return false;

    output.swap(d->aggregate);
    d->completed = false;
    d->completed_combine_dispatches = 0;
    d->input.release();
    d->pending_input.release();
    d->input_staging.release();
    d->combined.release();
    d->download.release();
    d->storage.clear();
    d->pending.clear();
    d->slots.clear();
    d->source = {};
    d->source_device.reset();
    d->aggregate = {};
    return true;
}

bool ResidentFfnRegion_vulkan::take_device_output(std::shared_ptr<const DeviceTensor_vulkan>& output)
{
    if (!d->completed || d->continuation_failed || d->output_mode != ResidentFfnOutput_vulkan::Device || !d->device_output)
        return false;

    output.swap(d->device_output);
    d->device_output.reset();
    d->completed = false;
    d->completed_combine_dispatches = 0;
    d->input.release();
    d->pending_input.release();
    d->slots.clear();
    d->source_device.reset();
    d->source = {};
    return true;
}

bool ResidentFfnRegion_vulkan::bind_context(std::shared_ptr<VulkanContext> context,
                                            std::shared_ptr<MoeCombine_vulkan> combiner)
{
    if (!context || !combiner || (d->context && d->context != context))
        return d->reject_wave();

    d->context = std::move(context);
    d->combiner = std::move(combiner);
    return true;
}

bool ResidentFfnRegion_vulkan::record_input(ncnn::VkMat& input, ncnn::VkCompute& command,
                                            const ncnn::Option& option, bool& uploaded)
{
    uploaded = false;
    if (!d->context || d->completed || d->recording || d->continuation_failed)
        return d->reject_wave();

    d->wave_started = true;
    d->recording = true;
    d->outputs_recorded = false;

    if (d->source_device)
    {
        if (d->source_device->context() != d->context || !d->source_device->value())
            return d->reject_wave();
        d->source_device->prepare_read();
        input = *d->source_device->value();
        return true;
    }

    if (!d->input.empty())
    {
        rearm_read(d->input);
        input = d->input;
        return true;
    }

    if (!fill_staging_upload(d->source, d->input_staging, option.staging_vkallocator)
        || !record_mapped_upload(d->input_staging, d->pending_input, command, option))
        return d->reject_wave();

    input = d->pending_input;
    uploaded = true;
    return true;
}

bool ResidentFfnRegion_vulkan::record_outputs(std::span<const Output> outputs, ncnn::VkCompute& command,
                                              const ncnn::Option& option, bool& downloaded)
{
    downloaded = false;
    if (!d->recording || outputs.empty() || !d->pending.empty() || !d->combiner || d->continuation_failed)
        return d->reject_wave();

    for (const Output& output : outputs)
    {
        if (output.index >= d->slots.size() || !d->slots[output.index].value.empty()
            || output.value.empty() || output.value.h <= 0
            || static_cast<size_t>(output.value.h) != output.routes.size()
            || std::any_of(d->pending.begin(), d->pending.end(), [&](const auto& prior) { return prior.index == output.index; }))
            return d->reject_wave();

        for (const ExpertRoute& route : output.routes)
        {
            if (route.token_index >= d->rows)
                return d->reject_wave();
        }

        auto& pending = d->pending.emplace_back();
        pending.index = output.index;
        pending.value = output.value;
        pending.routes.assign(output.routes.begin(), output.routes.end());
    }

    const size_t committed = std::count_if(d->slots.begin(), d->slots.end(), [](const auto& slot) { return !slot.value.empty(); });
    d->final_wave = committed + d->pending.size() == d->slots.size();
    if (!d->final_wave)
    {
        d->outputs_recorded = true;
        return true;
    }

    std::vector<MoeCombineDeviceInput_vulkan> inputs(d->slots.size());
    for (size_t i = 0; i < d->slots.size(); ++i)
    {
        if (!d->slots[i].value.empty())
        {
            rearm_read(d->slots[i].value);
            inputs[i] = {&d->slots[i].value, d->slots[i].routes};
        }
    }
    for (const auto& pending : d->pending)
        inputs[pending.index] = {&pending.value, pending.routes};

    uint64_t dispatches = 0;
    if (!d->combiner->record_device_inputs(inputs, d->rows, d->output_columns, d->combined,
                                           command, d->storage, nullptr, &dispatches))
        return d->reject_wave();

    if (d->continuation)
    {
        if (!d->continuation->record(d->combined, command, option))
            return d->reject_wave();
        d->continuation_recorded = true;
    }

    if (d->output_mode == ResidentFfnOutput_vulkan::Host)
    {
        if (!prepare_staging_batch(d->download, d->rows, d->output_columns, option.staging_vkallocator)
            || !record_staging_download(d->combined, d->rows, d->output_columns, d->download,
                                        command, d->context->device(), option))
            return d->reject_wave();
        downloaded = true;
    }

    d->pending_combine_dispatches = dispatches;
    d->outputs_recorded = true;
    return true;
}

bool ResidentFfnRegion_vulkan::complete_wave()
{
    if (!d->recording || !d->outputs_recorded || d->pending.empty())
        return d->reject_wave();
    if (d->continuation_failed
        || (d->final_wave && d->continuation && (!d->continuation_recorded || !d->continuation_submit_attempted)))
        return d->reject_wave();

    if (d->final_wave && d->output_mode == ResidentFfnOutput_vulkan::Host)
    {
        d->aggregate.reset(d->rows, d->output_columns, false);
        if (!copy_staging_to_cpu_batch(d->download, d->aggregate))
            return d->reject_wave();
    }

    if (d->final_wave && d->output_mode == ResidentFfnOutput_vulkan::Device)
    {
        auto completed = std::make_shared<DeviceTensor_vulkan>();
        if (!completed->assign_completed(d->combined, d->context))
            return d->reject_wave();
        d->device_output = std::move(completed);
    }

    if (!d->pending_input.empty())
    {
        d->input = d->pending_input;
        d->pending_input.release();
    }
    for (auto& pending : d->pending)
        d->slots[pending.index] = std::move(pending);
    d->pending.clear();
    d->input_staging.release();
    d->combined.release();
    d->download.release();
    d->storage.clear();
    d->completed = d->final_wave;
    d->completed_combine_dispatches = d->final_wave ? d->pending_combine_dispatches : 0;
    d->pending_combine_dispatches = 0;
    d->final_wave = false;
    d->recording = false;
    d->outputs_recorded = false;
    if (d->completed && d->continuation)
        d->continuation->mark_completed();
    return true;
}

void ResidentFfnRegion_vulkan::abort_wave() noexcept
{
    (void)d->reject_wave();
    d->pending.clear();
    d->pending_input.release();
    d->input_staging.release();
    d->combined.release();
    d->download.release();
    d->storage.clear();
    d->final_wave = false;
    d->recording = false;
    d->outputs_recorded = false;
    d->pending_combine_dispatches = 0;
    d->continuation_recorded = false;
    d->continuation_submit_attempted = false;
}

} // namespace moe
} // namespace ncnn
