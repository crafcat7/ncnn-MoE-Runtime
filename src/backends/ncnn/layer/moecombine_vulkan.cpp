#include "backends/ncnn/layer/moecombine_vulkan.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/core/tensor.h"
#include "kernels/vulkan/moe_combine.comp.hex.h"
#include "kernels/vulkan/moe_combine_single_row.comp.hex.h"

#include <algorithm>
#include <bit>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

class MoeCombinePending_vulkan::Implementation
{
public:
    struct Input
    {
        ActivationBuffer host;
        std::shared_ptr<const DeviceTensor_vulkan> device;
        ExpertRoute single_route;
        std::vector<ExpertRoute> routes;
    };

    std::shared_ptr<VulkanContext> context;
    std::vector<Input> inputs;
    ActivationBuffer shared_host;
    std::shared_ptr<const DeviceTensor_vulkan> shared_device;
    size_t rows = 0;
    uint32_t columns = 0;
};

MoeCombinePending_vulkan::MoeCombinePending_vulkan()
    : d(new Implementation)
{
}

MoeCombinePending_vulkan::~MoeCombinePending_vulkan() = default;

size_t MoeCombinePending_vulkan::rows() const noexcept
{
    return d->rows;
}

uint32_t MoeCombinePending_vulkan::columns() const noexcept
{
    return d->columns;
}

class MoeCombine_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> single_row_pipeline;
};

MoeCombine_vulkan::MoeCombine_vulkan()
    : d(new Implementation)
{
}

MoeCombine_vulkan::~MoeCombine_vulkan() = default;

std::shared_ptr<MoeCombine_vulkan> MoeCombine_vulkan::create(std::shared_ptr<VulkanContext> context)
{
    if (!context)
        return {};

    // Declare the owner before locking: on failure its pipeline deleter must
    // run after the command lock is released.
    auto result = std::shared_ptr<MoeCombine_vulkan>(new MoeCombine_vulkan);
    auto& impl = *result->d;
    impl.context = std::move(context);
    impl.opt = make_vk_opt(*impl.context);

    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    if (!create_pipeline(impl.context, moe_combine_shader,
                         static_cast<int>(sizeof(moe_combine_shader) - 1),
                         impl.opt, 0, 0, 128, impl.pipeline))
        return {};

    // The compact shader binds four sources and one output. Limits are cached
    // device properties; unsupported devices retain the existing GPU CSR plan.
    const auto& limits = impl.context->device()->info.physicalDeviceProperties().limits;
    if (limits.maxPerStageDescriptorStorageBuffers >= 5
        && limits.maxDescriptorSetStorageBuffers >= 5)
    {
        if (!create_pipeline(impl.context, moe_combine_single_row_shader,
                             static_cast<int>(sizeof(moe_combine_single_row_shader) - 1),
                             impl.opt, 0, 0, 128, impl.single_row_pipeline))
            return {};
    }

    return result;
}

bool MoeCombine_vulkan::supports_grouped_single_row() const noexcept
{
    return static_cast<bool>(d->single_row_pipeline);
}

const std::shared_ptr<ncnn::Pipeline>& MoeCombine_vulkan::pipeline() const noexcept
{
    return d->pipeline;
}

bool MoeCombine_vulkan::forward(std::span<const MoeCombineInput_vulkan> inputs,
                                size_t rows, uint32_t columns,
                                const ActivationBuffer* shared_host,
                                std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                                std::shared_ptr<DeviceTensor_vulkan>& output)
{
    std::shared_ptr<VulkanContext> context;
    if (shared_device && !shared_device->empty())
        context = shared_device->context();
    for (const auto& input : inputs)
    {
        if (!input.device || input.device->empty())
            continue;
        if (context && context != input.device->context())
            return false;
        context = input.device->context();
    }
    if (!context)
        return false;

    const auto prepared = create(std::move(context));
    return prepared && prepared->combine(inputs, rows, columns, shared_host, std::move(shared_device), output);
}

bool MoeCombine_vulkan::record_inputs(std::span<const MoeCombineInput_vulkan> inputs,
                                      size_t rows, uint32_t columns,
                                      const ActivationBuffer* shared_host,
                                      const std::shared_ptr<const DeviceTensor_vulkan>& shared_device,
                                      ncnn::VkMat& combined, ncnn::VkCompute& cmd,
                                      std::vector<ncnn::VkMat>& temporaries, uint64_t& uploads) const
{
    const auto& context = d->context;
    const auto& pipeline = d->pipeline;
    const auto& opt = d->opt;
    const bool has_shared_device = shared_device && !shared_device->empty();
    const bool has_shared_host = shared_host && shared_host->rows() != 0;

    combined.create(static_cast<int>(columns), static_cast<int>(rows), sizeof(float), context->blob_allocator());
    if (combined.empty())
        return false;

    temporaries.reserve(temporaries.size() + inputs.size() * 4 + 2);
    std::vector<ncnn::vk_constant_type> constants(5);
    std::vector<ncnn::VkMat> bindings(3);
    const std::vector<unsigned char> readonly_bindings = {1, 1, 0};
    constants[0].u32 = columns;
    constants[1].u32 = static_cast<uint32_t>(rows);

    bool initialize = true;
    auto record = [&](const ncnn::VkMat& source, const ncnn::VkMat& routes, uint32_t count, bool shared) {
        constants[2].u32 = count;
        constants[3].u32 = static_cast<uint32_t>(shared);
        constants[4].u32 = static_cast<uint32_t>(initialize);
        bindings[0] = source;
        bindings[1] = routes;
        bindings[2] = combined;
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(columns);
        dispatcher.h = static_cast<int>(std::min(rows, size_t(65535)));
        dispatcher.c = static_cast<int>((rows + 65534) / 65535);
        cmd.record_pipeline_readonly(pipeline.get(), bindings, readonly_bindings, constants, dispatcher);
        initialize = false;
    };

    for (const auto& input : inputs)
    {
        ncnn::VkMat source;
        if (input.device && !input.device->empty())
        {
            input.device->prepare_read();
            source = *input.device->value();
        }
        else
        {
            auto& staging = temporaries.emplace_back();
            if (!fill_staging_upload(*input.host, staging, context->staging_allocator())
                || !record_mapped_upload(staging, source, cmd, opt))
                return false;
            temporaries.push_back(source);
            ++uploads;
        }

        auto& staging = temporaries.emplace_back();
        const uint32_t route_value_count = static_cast<uint32_t>(input.routes.size() * 2);
        staging.create(static_cast<int>(route_value_count), sizeof(uint32_t), context->staging_allocator());
        if (staging.empty() || !staging.mapped_ptr())
            return false;

        auto* route_data = static_cast<uint32_t*>(staging.mapped_ptr());
        for (size_t route_index = 0; route_index < input.routes.size(); ++route_index)
        {
            route_data[route_index * 2] = input.routes[route_index].token_index;
            route_data[route_index * 2 + 1] = std::bit_cast<uint32_t>(input.routes[route_index].weight);
        }

        ncnn::VkMat routes;
        if (!record_mapped_upload(staging, routes, cmd, opt))
            return false;
        temporaries.push_back(routes);
        record(source, routes, static_cast<uint32_t>(input.routes.size()), false);
    }

    if (has_shared_device || has_shared_host)
    {
        ncnn::VkMat source;
        if (has_shared_device)
        {
            shared_device->prepare_read();
            source = *shared_device->value();
        }
        else
        {
            auto& staging = temporaries.emplace_back();
            if (!fill_staging_upload(*shared_host, staging, context->staging_allocator())
                || !record_mapped_upload(staging, source, cmd, opt))
                return false;
            temporaries.push_back(source);
            ++uploads;
        }
        record(source, source, 0, true);
    }

    return !initialize;
}

bool MoeCombine_vulkan::record(const MoeCombinePending_vulkan& pending, ncnn::VkMat& output,
                               ncnn::VkCompute& command, std::vector<ncnn::VkMat>& storage,
                               uint64_t& uploads) const
{
    std::vector<MoeCombineInput_vulkan> views;
    views.reserve(pending.d->inputs.size());
    for (const auto& input : pending.d->inputs)
    {
        const std::span<const ExpertRoute> routes = input.routes.empty()
                                                        ? std::span<const ExpertRoute>(&input.single_route, 1)
                                                        : std::span<const ExpertRoute>(input.routes);
        views.push_back({&input.host, input.device, routes});
    }

    return record_inputs(views, pending.rows(), pending.columns(), &pending.d->shared_host,
                         pending.d->shared_device, output, command, storage, uploads);
}

bool MoeCombine_vulkan::record_device_inputs(std::span<const MoeCombineDeviceInput_vulkan> inputs,
                                             size_t rows, uint32_t columns, ncnn::VkMat& output,
                                             ncnn::VkCompute& command, std::vector<ncnn::VkMat>& storage,
                                             const ncnn::VkMat* shared, uint64_t* recorded_dispatches) const
{
    output.create(static_cast<int>(columns), static_cast<int>(rows), sizeof(float), d->context->blob_allocator());
    if (output.empty())
        return false;

    const bool single_row = rows == 1 && !shared && d->single_row_pipeline
                            && std::all_of(inputs.begin(), inputs.end(), [](const auto& input) {
                                   return input.routes.size() == 1 && input.routes.front().token_index == 0;
                               });
    if (single_row)
    {
        std::vector<ncnn::vk_constant_type> constants(7);
        constants[0].u32 = columns;
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(columns);
        dispatcher.h = 1;
        dispatcher.c = 1;

        uint64_t dispatches = 0;
        for (size_t first = 0; first < inputs.size(); first += 4)
        {
            const size_t count = std::min(size_t(4), inputs.size() - first);
            std::vector<ncnn::VkMat> bindings(5);
            for (size_t slot = 0; slot < 4; ++slot)
            {
                // Unused descriptors still bind a valid read-only source.
                bindings[slot] = *inputs[first + std::min(slot, count - 1)].value;
                constants[3 + slot].f = slot < count ? inputs[first + slot].routes.front().weight : 0.0f;
            }
            bindings[4] = output;
            constants[1].u32 = static_cast<uint32_t>(count);
            constants[2].u32 = static_cast<uint32_t>(first == 0);
            command.record_pipeline_readonly(d->single_row_pipeline.get(), bindings, {1, 1, 1, 1, 0}, constants, dispatcher);
            ++dispatches;
        }

        if (recorded_dispatches)
            *recorded_dispatches = dispatches;
        return true;
    }

    storage.reserve(storage.size() + inputs.size() * 2);
    std::vector<ncnn::vk_constant_type> constants(5);
    constants[0].u32 = columns;
    constants[1].u32 = static_cast<uint32_t>(rows);
    // Mode 2 routes are stable per-token CSR: offsets[rows + 1], followed
    // by (source row, weight) pairs. Each output visits only its own routes.
    constants[3].u32 = 2;

    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(columns);
    dispatcher.h = static_cast<int>(std::min(rows, size_t(65535)));
    dispatcher.c = static_cast<int>((rows + 65534) / 65535);

    bool initialize = true;
    std::vector<uint32_t> next(rows);
    for (const auto& input : inputs)
    {
        auto& staging = storage.emplace_back();
        const size_t offset_words = rows + 1;
        staging.create(static_cast<int>(offset_words + input.routes.size() * 2),
                       sizeof(uint32_t), d->context->staging_allocator());
        if (staging.empty() || !staging.mapped_ptr())
            return false;

        auto* values = static_cast<uint32_t*>(staging.mapped_ptr());
        std::fill_n(values, offset_words, uint32_t(0));
        for (const auto& route : input.routes)
            ++values[route.token_index + 1];
        for (size_t row = 1; row <= rows; ++row)
            values[row] += values[row - 1];
        std::copy_n(values, rows, next.begin());

        for (size_t index = 0; index < input.routes.size(); ++index)
        {
            // Stable insertion preserves repeated routes' original summation
            // order within this Expert, even when token ids are unsorted.
            const size_t destination = offset_words + next[input.routes[index].token_index]++ * size_t(2);
            values[destination] = static_cast<uint32_t>(index);
            values[destination + 1] = std::bit_cast<uint32_t>(input.routes[index].weight);
        }

        ncnn::VkMat routes;
        if (!record_mapped_upload(staging, routes, command, d->opt))
            return false;
        storage.push_back(routes);

        constants[2].u32 = static_cast<uint32_t>(input.routes.size());
        constants[4].u32 = static_cast<uint32_t>(initialize);
        command.record_pipeline_readonly(d->pipeline.get(), {*input.value, routes, output}, {1, 1, 0}, constants, dispatcher);
        initialize = false;
    }

    if (shared)
    {
        // Match the CPU order: all routed Experts, then the Shared Expert.
        constants[2].u32 = 0;
        constants[3].u32 = 1;
        constants[4].u32 = static_cast<uint32_t>(initialize);
        command.record_pipeline_readonly(d->pipeline.get(), {*shared, *shared, output}, {1, 1, 0}, constants, dispatcher);
    }

    if (recorded_dispatches)
        *recorded_dispatches = inputs.size() + static_cast<uint64_t>(shared != nullptr);
    return true;
}

std::shared_ptr<MoeCombinePending_vulkan> MoeCombine_vulkan::prepare(std::span<const MoeCombineInput_vulkan> inputs,
                                                                     size_t rows, uint32_t columns,
                                                                     const ActivationBuffer* shared_host,
                                                                     std::shared_ptr<const DeviceTensor_vulkan> shared_device) const
{
    try
    {
        auto pending = std::shared_ptr<MoeCombinePending_vulkan>(new MoeCombinePending_vulkan);
        pending->d->context = d->context;
        pending->d->rows = rows;
        pending->d->columns = columns;
        pending->d->inputs.reserve(inputs.size());
        for (const auto& input : inputs)
        {
            auto& owned = pending->d->inputs.emplace_back();
            if (input.routes.size() == 1)
                owned.single_route = input.routes.front();
            else
                owned.routes.assign(input.routes.begin(), input.routes.end());

            if (input.device && !input.device->empty())
                owned.device = input.device;
            else
                owned.host = *input.host;
        }

        if (shared_device && !shared_device->empty())
            pending->d->shared_device = std::move(shared_device);
        else if (shared_host && shared_host->rows() != 0)
            pending->d->shared_host = *shared_host;

        return pending;
    }
    catch (const std::bad_alloc&)
    {
        return {};
    }
}

bool MoeCombine_vulkan::combine(std::span<const MoeCombineInput_vulkan> inputs,
                                size_t rows, uint32_t columns,
                                const ActivationBuffer* shared_host,
                                std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                                std::shared_ptr<DeviceTensor_vulkan>& output) const
{
    VkTransferLease lease = d->context->acquire_transfer_slot();
    VkTransferSlot& slot = lease.slot();
    VulkanRuntimeState& state = d->context->runtime_state();
    const std::lock_guard<std::mutex> lock(d->context->command_mutex());

    if (slot.command_used && slot.command->reset() != 0)
        return false;
    slot.command_used = true;

    ncnn::VkCompute& cmd = *slot.command;
    ncnn::VkMat combined;
    std::vector<ncnn::VkMat> storage;
    uint64_t uploads = 0;
    if (!record_inputs(inputs, rows, columns, shared_host, shared_device,
                       combined, cmd, storage, uploads))
        return false;

    if (submit_wait(cmd, d->context->device()) != 0)
        return false;

    auto completed = std::make_shared<DeviceTensor_vulkan>();
    if (!completed->assign_completed(combined, d->context))
        return false;

    ++state.compute_submissions;
    state.batch_uploads += uploads;
    output = std::move(completed);
    return true;
}

bool MoeCombine_vulkan::materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output)
{
    return input.materialize(output);
}

} // namespace moe
} // namespace ncnn
