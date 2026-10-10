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

static bool reset_combine_command(VulkanTransferSlot& slot, VulkanRuntimeState& state)
{
    if (slot.command_used)
    {
        if (slot.command->reset() != 0) return false;
        ++state.command_buffer_reuses;
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
    return result;
#else
    (void)context;
    return {};
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
    bool initialize = true;
    auto record = [&](const ncnn::VkMat& source, const ncnn::VkMat& routes, uint32_t count, bool shared) {
        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].u32 = columns;
        constants[1].u32 = static_cast<uint32_t>(rows);
        constants[2].u32 = count;
        constants[3].u32 = static_cast<uint32_t>(shared);
        constants[4].u32 = static_cast<uint32_t>(initialize);
        ncnn::VkMat dispatcher;
        dispatcher.w = static_cast<int>(columns);
        dispatcher.h = static_cast<int>(std::min(rows, size_t(65535)));
        dispatcher.c = static_cast<int>((rows + 65534) / 65535);
        cmd.record_pipeline_readonly(pipeline.get(), {source, routes, combined}, {1, 1, 0}, constants, dispatcher);
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
        std::vector<uint32_t> route_data;
        route_data.reserve(input.routes.size() * 2);
        for (const auto& route : input.routes)
        {
            route_data.push_back(route.token_index);
            route_data.push_back(std::bit_cast<uint32_t>(route.weight));
        }
        auto& staging = temporaries.emplace_back();
        ncnn::VkMat routes;
        if (!fill_staging_values(route_data.data(), route_data.size(), sizeof(uint32_t), staging, context->staging_allocator())
            || !record_mapped_upload(staging, routes, cmd, opt)) return false;
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
        views.push_back({&input.host, input.device, input.routes});
    return record_inputs(views, pending.rows(), pending.columns(), &pending.d->shared_host,
                         pending.d->shared_device, output, command, storage, uploads);
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
    if (!reset_combine_command(slot, state)) return false;
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
    if (!reset_combine_command(slot, state)) return false;
    input.prepare_read();
    auto& cmd = *slot.command;
    if (!prepare_staging_batch(slot.download, input.rows(), input.columns(), slot.staging_allocator)
        || !record_prepared_activation_staging_download(*input.value(), input.rows(), input.columns(), slot.download, cmd, context->device(), opt)) return false;
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
