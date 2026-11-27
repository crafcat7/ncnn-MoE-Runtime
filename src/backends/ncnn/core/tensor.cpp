#include "backends/ncnn/core/tensor.h"

#include <algorithm>
#include <climits>
#include <limits>
#include <utility>

namespace ncnn {
namespace moe {

// --- DeviceTensor_vulkan -----------------------------------------------------

class DeviceTensor_vulkan::Impl
{
public:
    std::shared_ptr<VulkanContext> context;
    // Null marks a completed unpacked FP32 tensor; non-null tracks a pending graph.
    std::shared_ptr<CommandGraphState_vulkan> graph;
    ncnn::VkMat value;
    size_t rows = 0;
    uint32_t columns = 0;
};

DeviceTensor_vulkan::DeviceTensor_vulkan() = default;
DeviceTensor_vulkan::~DeviceTensor_vulkan() = default;
DeviceTensor_vulkan::DeviceTensor_vulkan(DeviceTensor_vulkan&&) noexcept = default;
DeviceTensor_vulkan& DeviceTensor_vulkan::operator=(DeviceTensor_vulkan&&) noexcept = default;

bool DeviceTensor_vulkan::empty() const noexcept
{
    return !d || d->value.empty();
}

size_t DeviceTensor_vulkan::rows() const noexcept
{
    return d ? d->rows : 0;
}

uint32_t DeviceTensor_vulkan::columns() const noexcept
{
    return d ? d->columns : 0;
}

bool DeviceTensor_vulkan::assign_completed(const ncnn::VkMat& val, std::shared_ptr<VulkanContext> ctx)
{
    if (!ctx || val.empty() || (val.dims != 1 && val.dims != 2)
        || val.elempack != 1 || val.elemsize != sizeof(float))
    {
        d.reset();
        return false;
    }

    auto impl = std::make_unique<Impl>();
    impl->context = std::move(ctx);
    impl->value = val;
    impl->rows = val.dims == 1 ? 1 : static_cast<size_t>(val.h);
    impl->columns = static_cast<uint32_t>(val.w);
    d = std::move(impl);
    return true;
}

void DeviceTensor_vulkan::assign_pending(const ncnn::VkMat& val, size_t rows, uint32_t cols,
                                         std::shared_ptr<VulkanContext> ctx,
                                         std::shared_ptr<CommandGraphState_vulkan> graph)
{
    auto impl = std::make_unique<Impl>();
    impl->context = std::move(ctx);
    impl->graph = std::move(graph);
    impl->value = val;
    impl->rows = rows;
    impl->columns = cols;
    d = std::move(impl);
}

const ncnn::VkMat* DeviceTensor_vulkan::value() const noexcept
{
    return d && (!d->graph || (d->graph->completed && !d->graph->failed)) && !d->value.empty()
               ? &d->value
               : nullptr;
}

const ncnn::VkMat& DeviceTensor_vulkan::raw_value() const noexcept
{
    return d->value;
}

const std::shared_ptr<VulkanContext>& DeviceTensor_vulkan::context() const noexcept
{
    if (d)
        return d->context;
    static const std::shared_ptr<VulkanContext> none;
    return none;
}

const std::shared_ptr<CommandGraphState_vulkan>& DeviceTensor_vulkan::graph_state() const noexcept
{
    if (d)
        return d->graph;
    static const std::shared_ptr<CommandGraphState_vulkan> none;
    return none;
}

bool DeviceTensor_vulkan::readable_from(const std::shared_ptr<CommandGraphState_vulkan>& target,
                                        size_t storage_variant) const noexcept
{
    if (!d)
        return false;
    if (!d->graph)
        return storage_variant == 0;
    return !d->graph->failed && d->graph->storage_variant == storage_variant
           && (d->graph == target || (d->graph->submitted && d->graph->completed));
}

void DeviceTensor_vulkan::prepare_read() const noexcept
{
    const ncnn::VkMat* completed = value();
    if (completed)
        rearm_read(*completed);
}

bool DeviceTensor_vulkan::materialize(ActivationBuffer& output) const
{
    const ncnn::VkMat* src = value();
    if (!src || !d->context)
        return false;

    const auto& ctx = d->context;
    VkTransferLease lease = ctx->acquire_transfer_slot();
    VkTransferSlot& slot = lease.slot();

    const DType dst_dtype = output.dtype();
    const size_t elem_size = output.element_size();
    if (!prepare_staging_batch(slot.download, d->rows, d->columns, slot.staging_allocator, elem_size))
        return false;

    const ncnn::Option opt = make_vk_opt(*ctx);
    {
        const std::lock_guard<std::mutex> lock(ctx->command_mutex());
        if (slot.command_used && slot.command->reset() != 0)
            return false;
        slot.command_used = true;

        prepare_read();
        if (!record_staging_download(*src, d->rows, d->columns, slot.download,
                                     *slot.command, ctx->device(), opt, dst_dtype))
        {
            (void)slot.command->reset();
            return false;
        }

        if (submit_wait(*slot.command, ctx->device()) != 0)
            return false;
    }

    output.reset(d->rows, d->columns, false);
    if (!copy_staging_to_cpu_batch(slot.download, output))
        return false;

    VulkanRuntimeState& stats = ctx->runtime_state();
    ++stats.compute_submissions;
    ++stats.batch_downloads;
    return true;
}

// --- VkComputeBatch ----------------------------------------------------------

VkComputeBatch::VkComputeBatch(VulkanContext& ctx)
    : context(ctx)
{
    queue = context.device()->acquire_queue(context.device()->info.compute_queue_family_index());
}

VkComputeBatch::~VkComputeBatch()
{
    (void)reset();
    if (queue)
        context.device()->reclaim_queue(context.device()->info.compute_queue_family_index(), queue);
}

ncnn::VkCompute* VkComputeBatch::next()
{
    if (!queue || completion_error || recorded != submitted)
        return nullptr;

    if (recorded == context.ordered_commands.size())
        context.ordered_commands.push_back(std::make_unique<ncnn::VkCompute>(context.device()));

    return context.ordered_commands[recorded++].get();
}

int VkComputeBatch::submit()
{
    if (!queue || completion_error || recorded != submitted + 1)
        return -1;

    const int res = context.ordered_commands[submitted]->submit(queue);
    if (res == 0)
        ++submitted;
    else
        completion_error = res;
    return res;
}

int VkComputeBatch::reset() noexcept
{
    for (size_t i = 0; i < submitted; ++i)
    {
        auto& cmd = *context.ordered_commands[i];
        int res = wait_cmd(cmd, context.device(), queue);
        if (res == 0)
            res = cmd.reset();
        if (res != 0)
            completion_error = res;
    }

    int res = completion_error;
    for (size_t i = submitted; i < recorded; ++i)
        if (context.ordered_commands[i]->reset() != 0)
            res = -1;

    recorded = 0;
    submitted = 0;
    completion_error = res;
    context.drain_retired();
    return res;
}

// --- VkHostOutput ------------------------------------------------------------

struct VkHostOutput::Storage
{
    explicit Storage(std::shared_ptr<VulkanContext> owner)
        : context(std::move(owner)),
          allocator(std::make_unique<ncnn::VkStagingAllocator>(context->device()))
    {
    }

    // Destruction order: buffer -> allocator -> context.
    std::shared_ptr<VulkanContext> context;
    std::unique_ptr<ncnn::VkStagingAllocator> allocator;
    ncnn::VkMat buffer;
};

VkHostOutput::VkHostOutput() = default;
VkHostOutput::~VkHostOutput() = default;

bool VkHostOutput::prepare(const std::shared_ptr<VulkanContext>& ctx,
                           size_t rows, uint32_t cols, uint64_t slot_limit,
                           ncnn::VkMat& dst, ActivationBuffer& pending)
{
    if (cols == 0 || rows > static_cast<size_t>(INT_MAX)
        || cols > static_cast<uint32_t>(INT_MAX)
        || rows > std::numeric_limits<size_t>::max() / sizeof(float) / cols)
    {
        return false;
    }

    const size_t bytes = rows * static_cast<size_t>(cols) * sizeof(float);
    if (bytes > std::numeric_limits<uint64_t>::max() - 63)
        return false;

    const uint64_t planned = align_up(bytes, 64);
    if (planned > slot_limit)
        return false;

    std::shared_ptr<Storage>* chosen = nullptr;
    for (auto& slot : slots)
    {
        if (slot && slot.use_count() != 1)
            continue;
        if (slot && slot->context == ctx && slot->buffer.total() * slot->buffer.elemsize >= bytes)
        {
            chosen = &slot;
            break;
        }
        if (!chosen)
            chosen = &slot;
    }

    if (!chosen)
        return false;

    if (!*chosen || (*chosen)->context != ctx
        || (*chosen)->buffer.total() * (*chosen)->buffer.elemsize < bytes)
    {
        auto replacement = std::make_shared<Storage>(ctx);
        if (!prepare_staging_batch(replacement->buffer, rows, cols, replacement->allocator.get())
            || replacement->buffer.data->capacity > slot_limit)
        {
            return false;
        }
        *chosen = std::move(replacement);
    }

    const auto& slot = *chosen;
    ncnn::VkMat view = slot->buffer;
    view.w = static_cast<int>(cols);
    view.h = static_cast<int>(rows);
    view.cstep = rows * static_cast<size_t>(cols);
#if NCNN_BATCH
    view.nstep = view.cstep;
#endif

    ActivationBuffer prepared;
    prepared.assign_external(rows, cols, DType::Float32,
                             {static_cast<std::byte*>(view.mapped_ptr()),
                              slot->buffer.total() * slot->buffer.elemsize},
                             slot);

    dst = std::move(view);
    pending.swap(prepared);
    slot_capacity = std::max(slot_capacity, slot_limit);
    return true;
}

bool VkHostOutput::complete(ncnn::VkMat& dst, const ActivationBuffer& pending)
{
    (void)pending;
    dst.allocator->invalidate(dst.data);
    dst.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    dst.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

uint64_t VkHostOutput::allocated_bytes() const noexcept
{
    uint64_t total = 0;
    for (const auto& slot : slots)
        if (slot && slot->buffer.data)
            total += slot->buffer.data->capacity;
    return total;
}

} // namespace moe
} // namespace ncnn
