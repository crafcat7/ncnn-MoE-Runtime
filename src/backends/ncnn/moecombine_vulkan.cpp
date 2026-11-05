#include "moecombine_vulkan.h"
#include "linear.h"
#include "vulkancontext.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <vector>
#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/moe_combine.comp.hex.h"
#include "kernels/vulkan/moe_combine_single_row.comp.hex.h"
#endif

namespace ncnn {
namespace moe {
#if NCNN_MOE_WITH_VULKAN
static ncnn::Option combine_option(const std::shared_ptr<VulkanContext>& context)
{
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_packing_layout = false;
    opt.blob_vkallocator = context->blob_allocator();
    opt.workspace_vkallocator = context->blob_allocator();
    opt.staging_vkallocator = context->staging_allocator();
    return opt;
}

static bool reset_combine_command(VulkanTransferSlot& slot)
{
    if (slot.command_used)
    {
        if (slot.command->reset() != 0) return false;
    }
    slot.command_used = true;
    return true;
}
#endif

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
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> single_row_pipeline;
#endif
};

MoeCombine_vulkan::MoeCombine_vulkan()
    : d(new Implementation)
{
}

MoeCombine_vulkan::~MoeCombine_vulkan() = default;

std::shared_ptr<MoeCombine_vulkan> MoeCombine_vulkan::create(std::shared_ptr<VulkanContext> context)
{
#if NCNN_MOE_WITH_VULKAN
    if (!context) return {};
    // Declare the owner before locking: on failure its pipeline deleter must
    // run after the command lock is released.
    auto result = std::shared_ptr<MoeCombine_vulkan>(new MoeCombine_vulkan);
    auto& impl = *result->d;
    impl.context = context;
    impl.opt = combine_option(context);
    const auto& opt = impl.opt;
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    impl.pipeline = context->find_pipeline(moe_combine_shader, 0);
    if (!impl.pipeline)
    {
        const auto binary = context->shader_binary(moe_combine_shader, sizeof(moe_combine_shader) - 1, opt, 0);
        if (!binary || binary->empty()) return {};
        auto prepared = std::make_unique<ncnn::Pipeline>(context->device());
        prepared->set_optimal_local_size_xyz(128, 1, 1);
        if (prepared->create(binary->data(), binary->size() * sizeof(uint32_t), {}) != 0) return {};
        impl.pipeline = std::shared_ptr<ncnn::Pipeline>(prepared.release(), [context](ncnn::Pipeline* value) mutable {
            {
                const std::lock_guard<std::mutex> lock(context->command_mutex());
                delete value;
            }
            // Weak cache entries retain the deleter's control block after disposal.
            context.reset();
        });
        context->cache_pipeline(moe_combine_shader, 0, impl.pipeline);
    }
    // The compact shader binds four sources and one output. Limits are cached
    // device properties; unsupported devices retain the existing GPU CSR plan.
    const auto& limits = context->device()->info.physicalDeviceProperties().limits;
    if (limits.maxPerStageDescriptorStorageBuffers >= 5 && limits.maxDescriptorSetStorageBuffers >= 5)
    {
        impl.single_row_pipeline = context->find_pipeline(moe_combine_single_row_shader, 0);
        if (!impl.single_row_pipeline)
        {
            const auto binary = context->shader_binary(moe_combine_single_row_shader,
                                                       sizeof(moe_combine_single_row_shader) - 1, opt, 0);
            if (!binary || binary->empty()) return {};
            auto prepared = std::make_unique<ncnn::Pipeline>(context->device());
            prepared->set_optimal_local_size_xyz(128, 1, 1);
            if (prepared->create(binary->data(), binary->size() * sizeof(uint32_t), {}) != 0) return {};
            impl.single_row_pipeline = std::shared_ptr<ncnn::Pipeline>(prepared.release(), [context](ncnn::Pipeline* value) mutable {
                {
                    const std::lock_guard<std::mutex> lock(context->command_mutex());
                    delete value;
                }
                context.reset();
            });
            context->cache_pipeline(moe_combine_single_row_shader, 0, impl.single_row_pipeline);
        }
    }
    return result;
#else
    (void)context;
    return {};
#endif
}

bool MoeCombine_vulkan::supports_grouped_single_row() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return static_cast<bool>(d->single_row_pipeline);
#else
    return false;
#endif
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<ncnn::Pipeline>& MoeCombine_vulkan::pipeline() const noexcept
{
    return d->pipeline;
}
#endif

bool MoeCombine_vulkan::forward(std::span<const MoeCombineInput_vulkan> inputs,
                                size_t rows, uint32_t columns,
                                const ActivationBuffer* shared_host,
                                std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                                std::shared_ptr<DeviceTensor_vulkan>& output)
{
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    if (shared_device && !shared_device->empty()) context = shared_device->vulkan_context();
    for (const auto& input : inputs)
    {
        if (!input.device || input.device->empty()) continue;
        if (context && context != input.device->vulkan_context()) return false;
        context = input.device->vulkan_context();
    }
    if (!context) return false;
    const auto prepared = create(std::move(context));
    return prepared && prepared->combine(inputs, rows, columns, shared_host, std::move(shared_device), output);
#else
    (void)inputs;
    (void)rows;
    (void)columns;
    (void)shared_host;
    (void)shared_device;
    (void)output;
    return false;
#endif
}

#if NCNN_MOE_WITH_VULKAN
bool MoeCombine_vulkan::valid_inputs(std::span<const MoeCombineInput_vulkan> inputs,
                                     size_t rows, uint32_t columns, const ActivationBuffer* shared_host,
                                     const std::shared_ptr<const DeviceTensor_vulkan>& shared_device) const
{
    if (rows == 0 || rows > static_cast<size_t>(std::numeric_limits<int>::max())
        || columns == 0 || columns > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || rows > std::numeric_limits<uint32_t>::max() / columns) return false;
    const auto& context = d->context;
    const auto& pipeline = d->pipeline;
    if (!context || !pipeline) return false;
    for (const auto& input : inputs)
    {
        if (input.routes.empty() || input.routes.size() > static_cast<size_t>(std::numeric_limits<int>::max() / 2)
            || input.routes.size() > std::numeric_limits<uint32_t>::max() / columns) return false;
        if (input.device && !input.device->empty())
        {
            if (input.device->vulkan_context() != context) return false;
            const auto* value = input.device->value();
            if (!value || value->dims != 2 || value->elempack != 1 || value->elemsize != sizeof(float)
                || input.device->rows() != input.routes.size() || input.device->columns() != columns) return false;
        }
        else if (!input.host || input.host->dtype() != DType::Float32
                 || input.host->rows() != input.routes.size() || input.host->columns() != columns)
            return false;
        for (const auto& route : input.routes)
            if (route.token_index >= rows) return false;
    }
    const bool has_shared_device = shared_device && !shared_device->empty();
    const bool has_shared_host = shared_host && shared_host->rows() != 0;
    if (has_shared_device)
    {
        const auto* value = shared_device->value();
        if (!value || value->dims != 2 || value->elempack != 1 || value->elemsize != sizeof(float)
            || shared_device->vulkan_context() != context || shared_device->rows() != rows || shared_device->columns() != columns) return false;
    }
    if (!has_shared_device && has_shared_host && (shared_host->dtype() != DType::Float32 || shared_host->rows() != rows || shared_host->columns() != columns)) return false;
    return !inputs.empty() || has_shared_device || has_shared_host;
}

bool MoeCombine_vulkan::record_inputs(std::span<const MoeCombineInput_vulkan> inputs,
                                      size_t rows, uint32_t columns, const ActivationBuffer* shared_host,
                                      const std::shared_ptr<const DeviceTensor_vulkan>& shared_device,
                                      ncnn::VkMat& combined, ncnn::VkCompute& cmd,
                                      std::vector<ncnn::VkMat>& temporaries, uint64_t& uploads) const
{
    if (!valid_inputs(inputs, rows, columns, shared_host, shared_device)) return false;
    const auto& context = d->context;
    const auto& pipeline = d->pipeline;
    const auto& opt = d->opt;
    const bool has_shared_device = shared_device && !shared_device->empty();
    const bool has_shared_host = shared_host && shared_host->rows() != 0;
    combined.create(static_cast<int>(columns), static_cast<int>(rows), sizeof(float), context->blob_allocator());
    if (combined.empty()) return false;
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
                || !record_mapped_upload(staging, source, cmd, opt)) return false;
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
        if (!record_mapped_upload(staging, routes, cmd, opt)) return false;
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
                || !record_mapped_upload(staging, source, cmd, opt)) return false;
            temporaries.push_back(source);
            ++uploads;
        }
        record(source, source, 0, true);
    }
    if (initialize) return false;
    return true;
}

bool MoeCombine_vulkan::record(const MoeCombinePending_vulkan& pending, ncnn::VkMat& output,
                               ncnn::VkCompute& command, std::vector<ncnn::VkMat>& storage,
                               uint64_t& uploads) const
{
    if (pending.d->context != d->context) return false;
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
    if (!d->context || !d->pipeline || (inputs.empty() && !shared) || rows == 0
        || rows >= static_cast<size_t>(std::numeric_limits<int>::max()) || columns == 0
        || columns > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || rows > std::numeric_limits<uint32_t>::max() / columns)
        return false;
    if (shared && (shared == &output || shared->empty() || shared->dims != 2 || shared->elempack != 1 || shared->elemsize != sizeof(float) || shared->w != static_cast<int>(columns) || shared->h != static_cast<int>(rows) || !shared->allocator || shared->allocator->vkdev != d->context->device()))
        return false;
    for (const auto& input : inputs)
    {
        const auto* value = input.value;
        if (!value || value == &output || value->empty() || value->dims != 2 || value->elempack != 1
            || value->elemsize != sizeof(float) || value->w != static_cast<int>(columns)
            || value->h <= 0 || static_cast<size_t>(value->h) != input.routes.size()
            || !value->allocator || value->allocator->vkdev != d->context->device()
            || input.routes.size() > (static_cast<size_t>(std::numeric_limits<int>::max()) - rows - 1) / 2
            || input.routes.size() > std::numeric_limits<uint32_t>::max() / columns)
            return false;
        for (const auto& route : input.routes)
            if (route.token_index >= rows) return false;
    }
    output.create(static_cast<int>(columns), static_cast<int>(rows), sizeof(float), d->context->blob_allocator());
    if (output.empty()) return false;
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
        if (recorded_dispatches) *recorded_dispatches = dispatches;
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
        staging.create(static_cast<int>(offset_words + input.routes.size() * 2), sizeof(uint32_t), d->context->staging_allocator());
        if (staging.empty() || !staging.mapped_ptr()) return false;
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
        if (!record_mapped_upload(staging, routes, command, d->opt)) return false;
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
    if (recorded_dispatches) *recorded_dispatches = inputs.size() + static_cast<uint64_t>(shared != nullptr);
    return true;
}
#endif

std::shared_ptr<MoeCombinePending_vulkan> MoeCombine_vulkan::prepare(std::span<const MoeCombineInput_vulkan> inputs,
                                                                     size_t rows, uint32_t columns,
                                                                     const ActivationBuffer* shared_host,
                                                                     std::shared_ptr<const DeviceTensor_vulkan> shared_device) const
{
#if NCNN_MOE_WITH_VULKAN
    if (!valid_inputs(inputs, rows, columns, shared_host, shared_device)) return {};
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
#else
    (void)inputs;
    (void)rows;
    (void)columns;
    (void)shared_host;
    (void)shared_device;
    return {};
#endif
}

bool MoeCombine_vulkan::combine(std::span<const MoeCombineInput_vulkan> inputs,
                                size_t rows, uint32_t columns,
                                const ActivationBuffer* shared_host,
                                std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                                std::shared_ptr<DeviceTensor_vulkan>& output) const
{
#if NCNN_MOE_WITH_VULKAN
    if (!valid_inputs(inputs, rows, columns, shared_host, shared_device)) return false;
    auto lease = d->context->acquire_transfer_slot();
    auto& slot = lease.slot();
    auto& state = d->context->runtime_state();
    const std::lock_guard<std::mutex> lock(d->context->command_mutex());
    if (!reset_combine_command(slot)) return false;
    auto& cmd = *slot.command;
    ncnn::VkMat combined;
    std::vector<ncnn::VkMat> storage;
    uint64_t uploads = 0;
    if (!record_inputs(inputs, rows, columns, shared_host, shared_device, combined, cmd, storage, uploads)) return false;
    if (submit_compute_and_wait(cmd, d->context->device()) != 0) return false;
    auto completed = std::make_shared<DeviceTensor_vulkan>();
    if (!completed->assign_completed(combined, d->context)) return false;
    ++state.compute_submissions;
    state.batch_uploads += uploads;
    output = std::move(completed);
    return true;
#else
    (void)inputs;
    (void)rows;
    (void)columns;
    (void)shared_host;
    (void)shared_device;
    (void)output;
    return false;
#endif
}

bool MoeCombine_vulkan::materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output)
{
#if NCNN_MOE_WITH_VULKAN
    if (input.empty()) return false;
    const auto context = input.vulkan_context();
    if (!context) return false;
    auto lease = context->acquire_transfer_slot();
    auto& slot = lease.slot();
    auto& state = context->runtime_state();
    const auto opt = combine_option(context);
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if (!reset_combine_command(slot)) return false;
    input.prepare_read();
    auto& cmd = *slot.command;
    if (!prepare_staging_batch(slot.download, input.rows(), input.columns(), slot.staging_allocator)
        || !record_staging_download(*input.value(), input.rows(), input.columns(), slot.download, cmd, context->device(), opt)) return false;
    if (submit_compute_and_wait(cmd, context->device()) != 0) return false;
    ActivationBuffer completed(input.rows(), input.columns());
    if (!copy_staging_to_cpu_batch(slot.download, completed)) return false;
    ++state.compute_submissions;
    ++state.batch_downloads;
    output.swap(completed);
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}
} // namespace moe
} // namespace ncnn
