#include "latentlayer_vulkan.h"
#include "latentattention_vulkan.h"
#include "hyperconnection_vulkan.h"
#include "rmsnorm_vulkan.h"
#include "router_vulkan.h"
#include "layerhead_vulkan.h"
#include "moecombine_vulkan.h"
#include "linear.h"
#include "vulkancontext.h"

#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

class LatentLayerWorkspace_vulkan::Implementation
{
public:
    std::shared_ptr<const DeviceTensor_vulkan> normalized_device;
    // Publication swaps old caller storage back into these pending buffers.
    // They belong to the session workspace rather than one layer invocation.
    ActivationBuffer normalized_host;
    ActivationBuffer selected_host;
    ExpertDispatchPlan routes;
    ActivationBuffer hidden_host;
    ActivationBuffer logits_host;

    static void prepare_host(ActivationBuffer& buffer, size_t rows, uint32_t columns)
    {
        // Endpoint outputs may previously have held another activation dtype.
        if (buffer.dtype() != DType::Float32)
            buffer = ActivationBuffer();
        buffer.reset(rows, columns, false);
    }

    uint64_t allocated_bytes() const noexcept
    {
        uint64_t bytes = normalized_host.allocated_bytes() + selected_host.allocated_bytes()
                         + hidden_host.allocated_bytes() + logits_host.allocated_bytes();
        bytes += static_cast<uint64_t>(routes.batches.capacity()) * sizeof(ExpertBatch)
                 + static_cast<uint64_t>(routes.route_scratch.capacity()) * sizeof(std::vector<ExpertRoute>)
                 + static_cast<uint64_t>(routes.scores.capacity()) * sizeof(float)
                 + static_cast<uint64_t>(routes.selected.capacity()) * sizeof(RouteCandidate);
        for (const auto& batch : routes.batches)
            bytes += static_cast<uint64_t>(batch.routes.capacity()) * sizeof(ExpertRoute);
        for (const auto& scratch : routes.route_scratch)
            bytes += static_cast<uint64_t>(scratch.capacity()) * sizeof(ExpertRoute);
#if NCNN_MOE_WITH_VULKAN
        bytes += pending.allocated_bytes();
#endif
        return bytes;
    }

    void release_host()
    {
        normalized_host = ActivationBuffer();
        selected_host = ActivationBuffer();
        routes = ExpertDispatchPlan();
        hidden_host = ActivationBuffer();
        logits_host = ActivationBuffer();
#if NCNN_MOE_WITH_VULKAN
        pending = ActivationBuffer();
#endif
    }
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<HyperConnection_vulkan> ffn_post;
    std::shared_ptr<MoeCombine_vulkan> combiner;
    ncnn::VkMat residual;
    HyperConnectionMix_vulkan mix;
    ActivationBuffer pending;
    std::shared_ptr<const DeviceTensor_vulkan> pending_device;
    std::shared_ptr<MoeCombinePending_vulkan> pending_combine;

    size_t pending_rows() const
    { return pending_combine ? pending_combine->rows() : (pending_device ? pending_device->rows() : pending.rows()); }

    static void prepare_read(const ncnn::VkMat& value)
    {
        if (value.data)
        {
            value.data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            value.data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
    }

    bool record_hidden(ncnn::VkMat& result, ncnn::VkMat& uploaded,
                       ncnn::VkMat& staging, ncnn::VkCompute& cmd,
                       std::vector<ncnn::VkMat>& combine_storage, uint64_t& combine_uploads)
    {
        if (!context || !ffn_post || residual.empty() || pending_rows() == 0)
            return false;
        auto& state = context->runtime_state();
        if (pending_combine)
        {
            if (!combiner || !combiner->record(*pending_combine, uploaded, cmd, combine_storage, combine_uploads))
                return false;
        }
        else if (pending_device)
        {
            pending_device->prepare_read();
            uploaded = *pending_device->value();
        }
        else
        {
            if (!fill_staging_upload(pending, staging, context->staging_allocator())
                || !record_mapped_upload(staging, uploaded, cmd, ffn_post->option())) return false;
            ++state.batch_uploads;
        }
        prepare_read(residual);
        prepare_read(mix.post);
        prepare_read(mix.combine);
        if (!ffn_post->record_post(uploaded, residual, mix, result, cmd))
            return false;
        return true;
    }

    void commit_combine(uint64_t uploads)
    {
        if (!pending_combine) return;
        auto& state = context->runtime_state();
        state.batch_uploads += uploads;
    }

    // Buffer allocator operations share the context command mutex. Operator
    // owners must be released afterwards because their deleters take that mutex.
    void clear_buffers()
    {
        pending.clear();
        pending_device.reset();
        pending_combine.reset();
        residual.release();
        mix = {};
        normalized_device.reset();
    }

    void clear()
    {
        clear_buffers();
        ffn_post.reset();
        combiner.reset();
        context.reset();
    }
#endif
};

LatentLayerWorkspace_vulkan::LatentLayerWorkspace_vulkan()
    : d(new Implementation)
{
}

LatentLayerWorkspace_vulkan::~LatentLayerWorkspace_vulkan()
{
    reset();
}

void LatentLayerWorkspace_vulkan::reset()
{
#if NCNN_MOE_WITH_VULKAN
    if (d->context)
    {
        // Operator deleters also take the command mutex. Retain their owners
        // until buffers have been released and the allocator lock is gone.
        const auto context_owner = d->context;
        const auto post_owner = d->ffn_post;
        const auto combine_owner = d->combiner;
        {
            const std::lock_guard<std::mutex> lock(context_owner->command_mutex());
            d->clear_buffers();
        }
        d->clear();
    }
#endif
}

void LatentLayerWorkspace_vulkan::release()
{
    reset();
    d->release_host();
}

uint64_t LatentLayerWorkspace_vulkan::allocated_bytes() const noexcept
{
    return d->allocated_bytes();
}

bool LatentLayerWorkspace_vulkan::active() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->context && d->ffn_post && !d->residual.empty();
#else
    return false;
#endif
}

const std::shared_ptr<const DeviceTensor_vulkan>& LatentLayerWorkspace_vulkan::device_input() const noexcept
{
    return d->normalized_device;
}

std::shared_ptr<LatentLayer_vulkan> LatentLayer_vulkan::create(std::shared_ptr<LatentAttention_vulkan> attention,
                                                               std::shared_ptr<HyperConnection_vulkan> attention_hyper,
                                                               std::shared_ptr<HyperConnection_vulkan> ffn_hyper,
                                                               std::shared_ptr<RmsNorm_vulkan> ffn_norm,
                                                               std::shared_ptr<Router_vulkan> router)
{
#if NCNN_MOE_WITH_VULKAN
    if (!attention || !attention->can_record() || !attention_hyper || !ffn_hyper || !ffn_norm || !router
        || attention->vulkan_context() != attention_hyper->vulkan_context()
        || attention->vulkan_context() != ffn_hyper->vulkan_context()
        || attention->vulkan_context() != router->context()
        || attention->vulkan_context() != ffn_norm->vulkan_context()
        || attention_hyper->input_columns() != ffn_hyper->input_columns()
        || attention_hyper->multiplier() != ffn_hyper->multiplier()
        || attention_hyper->is_head() || ffn_hyper->is_head())
        return {};
    auto result = std::shared_ptr<LatentLayer_vulkan>(new LatentLayer_vulkan);
    result->attention = std::move(attention);
    result->attention_hyper = std::move(attention_hyper);
    result->ffn_hyper = std::move(ffn_hyper);
    result->ffn_norm = std::move(ffn_norm);
    result->router = std::move(router);
    // Optional preparation failure preserves the existing layer fusion.
    // Combine can materialize Expert outputs and use its host fallback.
    result->combiner = MoeCombine_vulkan::create(result->attention->vulkan_context());
    return result;
#else
    (void)attention;
    (void)attention_hyper;
    (void)ffn_hyper;
    (void)ffn_norm;
    (void)router;
    return {};
#endif
}

#if NCNN_MOE_WITH_VULKAN
static bool reset_layer_command(VulkanTransferSlot& slot, VulkanRuntimeState& state)
{
    if (slot.command_used)
    {
        if (slot.command->reset() != 0)
            return false;
        ++state.command_buffer_reuses;
    }
    slot.command_used = true;
    return true;
}
#endif

bool LatentLayer_vulkan::forward(const ActivationBuffer& initial_hidden,
                                 std::span<const uint64_t> positions,
                                 std::span<LayerCache* const> caches,
                                 const ExpertDispatchOptions& options,
                                 ActivationBuffer& normalized,
                                 ExpertDispatchPlan& routes,
                                 LatentLayerWorkspace_vulkan& workspace) const
{
#if NCNN_MOE_WITH_VULKAN
    auto context = attention->vulkan_context();
    auto& previous = *workspace.d;
    const auto previous_post_owner = previous.ffn_post;
    const auto previous_combiner_owner = previous.combiner;
    if (positions.empty() || positions.size() != caches.size()
        || normalized.dtype() != DType::Float32
        || (previous.context && (previous.context != context || previous.pending_rows() != positions.size()))
        || (!previous.context && (initial_hidden.rows() != positions.size() || initial_hidden.dtype() != DType::Float32)))
        return false;
    auto lease = context->acquire_transfer_slot();
    auto& slot = lease.slot();
    auto& state = context->runtime_state();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    std::vector<ncnn::VkMat> combine_storage;
    uint64_t combine_uploads = 0;
    if (!reset_layer_command(slot, state))
        return false;
    auto& cmd = *slot.command;
    ncnn::VkMat input;
    ncnn::VkMat uploaded;
    if (previous.context)
    {
        if (!previous.record_hidden(input, uploaded, slot.upload, cmd, combine_storage, combine_uploads))
            return false;
    }
    else
    {
        if (!fill_staging_upload(initial_hidden, slot.upload, slot.staging_allocator)
            || !record_mapped_upload(slot.upload, input, cmd, attention->option()))
            return false;
        ++state.batch_uploads;
    }
    HyperConnectionMix_vulkan attention_mix;
    HyperConnectionMix_vulkan ffn_mix;
    HyperConnectionWorkspace_vulkan attention_work;
    HyperConnectionWorkspace_vulkan ffn_work;
    LatentAttentionWork_vulkan latent_work;
    RouterWorkspace_vulkan router_work;
    ncnn::VkMat attention_output;
    ncnn::VkMat residual;
    ncnn::VkMat normalized_gpu;
    if (!attention_hyper->record_pre(input, attention_mix, attention_work, cmd)
        || !attention->record_batch(attention_mix.reduced, positions, caches, attention_output, cmd, latent_work)
        || !attention_hyper->record_post(attention_output, input, attention_mix, residual, cmd)
        || !ffn_hyper->record_pre(residual, ffn_mix, ffn_work, cmd)
        || !ffn_norm->record(ffn_mix.reduced, normalized_gpu, cmd)
        || !router->record(normalized_gpu, options, router_work, cmd))
        return false;
    ncnn::VkMat selected_staging;
    const auto selected_columns = Router_vulkan::selected_columns(options.top_k);
    if (!prepare_staging_batch(slot.download, positions.size(), normalized_gpu.w, slot.staging_allocator)
        || !prepare_staging_batch(selected_staging, positions.size(), selected_columns, slot.staging_allocator)
        || !record_prepared_activation_staging_download(normalized_gpu, positions.size(), normalized_gpu.w, slot.download,
                                                        cmd, context->device(), attention->option())
        || !record_prepared_activation_staging_download(router_work.selected, positions.size(), selected_columns, selected_staging,
                                                        cmd, context->device(), router->option()))
        return false;
    if (submit_compute_and_wait(cmd, context->device()) != 0)
        return false;
    ++state.compute_submissions;
    state.batch_downloads += 2;
    LatentLayerWorkspace_vulkan::Implementation::prepare_host(previous.normalized_host, positions.size(), normalized_gpu.w);
    LatentLayerWorkspace_vulkan::Implementation::prepare_host(previous.selected_host, positions.size(), selected_columns);
    auto device_input = std::make_shared<DeviceTensor_vulkan>();
    if (!copy_staging_to_cpu_batch(slot.download, previous.normalized_host)
        || !copy_staging_to_cpu_batch(selected_staging, previous.selected_host)
        || !Router_vulkan::decode_selected(previous.selected_host, options, previous.routes)
        || !device_input->assign_completed(normalized_gpu, context)
        || !latent_work.commit())
        return false;
    previous.commit_combine(combine_uploads);
    previous.context = std::move(context);
    previous.ffn_post = ffn_hyper;
    previous.combiner = combiner;
    previous.residual = std::move(residual);
    previous.mix = std::move(ffn_mix);
    previous.pending.clear();
    previous.pending_device.reset();
    previous.pending_combine.reset();
    previous.normalized_device = std::move(device_input);
    normalized.swap(previous.normalized_host);
    std::swap(routes, previous.routes);
    return true;
#else
    (void)initial_hidden;
    (void)positions;
    (void)caches;
    (void)options;
    (void)normalized;
    (void)routes;
    (void)workspace;
    return false;
#endif
}

bool LatentLayer_vulkan::combine_device(std::span<const MoeCombineInput_vulkan> inputs,
                                        size_t rows, uint32_t columns,
                                        const ActivationBuffer* shared_host,
                                        std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                                        std::shared_ptr<DeviceTensor_vulkan>& output,
                                        const LatentLayerWorkspace_vulkan& workspace)
{
#if NCNN_MOE_WITH_VULKAN
    return workspace.active() && workspace.d->combiner
           && workspace.d->combiner->combine(inputs, rows, columns, shared_host, std::move(shared_device), output);
#else
    (void)inputs;
    (void)rows;
    (void)columns;
    (void)shared_host;
    (void)shared_device;
    (void)output;
    (void)workspace;
    return false;
#endif
}

bool LatentLayer_vulkan::defer_combine(std::span<const MoeCombineInput_vulkan> inputs,
                                       size_t rows, uint32_t columns,
                                       const ActivationBuffer* shared_host,
                                       std::shared_ptr<const DeviceTensor_vulkan> shared_device,
                                       LatentLayerWorkspace_vulkan& workspace)
{
#if NCNN_MOE_WITH_VULKAN
    if (!workspace.active() || !workspace.d->combiner
        || rows != static_cast<size_t>(workspace.d->residual.h)
        || static_cast<uint64_t>(columns) * workspace.d->ffn_post->multiplier() != static_cast<uint32_t>(workspace.d->residual.w)) return false;
    auto pending = workspace.d->combiner->prepare(inputs, rows, columns, shared_host, std::move(shared_device));
    if (!pending) return false;
    const std::lock_guard<std::mutex> lock(workspace.d->context->command_mutex());
    workspace.d->pending.clear();
    workspace.d->pending_device.reset();
    workspace.d->pending_combine = std::move(pending);
    return true;
#else
    (void)inputs;
    (void)rows;
    (void)columns;
    (void)shared_host;
    (void)shared_device;
    (void)workspace;
    return false;
#endif
}

bool LatentLayer_vulkan::defer_combine(ActivationBuffer& combined,
                                       LatentLayerWorkspace_vulkan& workspace)
{
#if NCNN_MOE_WITH_VULKAN
    if (!workspace.active() || combined.dtype() != DType::Float32
        || combined.rows() != static_cast<size_t>(workspace.d->residual.h)
        || combined.columns() * workspace.d->ffn_post->multiplier() != static_cast<uint32_t>(workspace.d->residual.w))
        return false;
    const std::lock_guard<std::mutex> lock(workspace.d->context->command_mutex());
    workspace.d->pending_device.reset();
    workspace.d->pending_combine.reset();
    workspace.d->pending.swap(combined);
    return true;
#else
    (void)combined;
    (void)workspace;
    return false;
#endif
}

bool LatentLayer_vulkan::defer_combine(std::shared_ptr<const DeviceTensor_vulkan> combined,
                                       LatentLayerWorkspace_vulkan& workspace)
{
#if NCNN_MOE_WITH_VULKAN
    if (!workspace.active() || !combined || combined->empty()
        || combined->vulkan_context() != workspace.d->context
        || combined->rows() != static_cast<size_t>(workspace.d->residual.h)
        || combined->columns() * workspace.d->ffn_post->multiplier() != static_cast<uint32_t>(workspace.d->residual.w)) return false;
    const std::lock_guard<std::mutex> lock(workspace.d->context->command_mutex());
    workspace.d->pending.clear();
    workspace.d->pending_combine.reset();
    workspace.d->pending_device = std::move(combined);
    return true;
#else
    (void)combined;
    (void)workspace;
    return false;
#endif
}

bool LatentLayer_vulkan::materialize_hidden(LatentLayerWorkspace_vulkan& workspace,
                                            ActivationBuffer& hidden)
{
#if NCNN_MOE_WITH_VULKAN
    if (!workspace.active())
        return true;
    auto& impl = *workspace.d;
    const auto context_owner = impl.context;
    auto lease = impl.context->acquire_transfer_slot();
    auto& slot = lease.slot();
    auto& state = impl.context->runtime_state();
    std::unique_lock<std::mutex> lock(impl.context->command_mutex());
    std::vector<ncnn::VkMat> combine_storage;
    uint64_t combine_uploads = 0;
    if (!reset_layer_command(slot, state))
        return false;
    auto& cmd = *slot.command;
    ncnn::VkMat result;
    ncnn::VkMat uploaded;
    if (!impl.record_hidden(result, uploaded, slot.upload, cmd, combine_storage, combine_uploads)
        || !prepare_staging_batch(slot.download, result.h, result.w, slot.staging_allocator)
        || !record_prepared_activation_staging_download(result, result.h, result.w, slot.download, cmd,
                                                        impl.context->device(), impl.ffn_post->option()))
        return false;
    if (submit_compute_and_wait(cmd, impl.context->device()) != 0)
        return false;
    ++state.compute_submissions;
    ++state.batch_downloads;
    LatentLayerWorkspace_vulkan::Implementation::prepare_host(impl.hidden_host, result.h, result.w);
    if (!copy_staging_to_cpu_batch(slot.download, impl.hidden_host))
        return false;
    hidden.swap(impl.hidden_host);
    impl.commit_combine(combine_uploads);
    // Release buffers under the allocator's command mutex; operator owners
    // stay alive until the mutex is released.
    result.release();
    uploaded.release();
    combine_storage.clear();
    impl.clear_buffers();
    lock.unlock();
    impl.clear();
    return true;
#else
    (void)workspace;
    (void)hidden;
    return true;
#endif
}

bool LatentLayer_vulkan::can_finish(const LatentLayerWorkspace_vulkan& workspace,
                                    const LayerHead_vulkan& head) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return workspace.active() && workspace.d->context == head.vulkan_context();
#else
    (void)workspace;
    (void)head;
    return false;
#endif
}

bool LatentLayer_vulkan::finish(LatentLayerWorkspace_vulkan& workspace,
                                const LayerHead_vulkan& head,
                                bool produce_logits,
                                bool last_row_only,
                                ActivationBuffer& hidden,
                                ActivationBuffer& logits)
{
#if NCNN_MOE_WITH_VULKAN
    if (!can_finish(workspace, head))
        return false;
    auto& impl = *workspace.d;
    const auto context_owner = impl.context;
    auto lease = impl.context->acquire_transfer_slot();
    auto& slot = lease.slot();
    auto& state = impl.context->runtime_state();
    std::unique_lock<std::mutex> lock(impl.context->command_mutex());
    std::vector<ncnn::VkMat> combine_storage;
    uint64_t combine_uploads = 0;
    if (!reset_layer_command(slot, state))
        return false;
    auto& cmd = *slot.command;
    ncnn::VkMat expanded;
    ncnn::VkMat uploaded;
    ncnn::VkMat normalized;
    ncnn::VkMat output;
    std::vector<ncnn::VkMat> temporaries;
    HyperConnectionWorkspace_vulkan hc_work;
    if (!impl.record_hidden(expanded, uploaded, slot.upload, cmd, combine_storage, combine_uploads)
        || !head.record_hidden(expanded, normalized, hc_work, temporaries, cmd))
        return false;
    if (produce_logits)
    {
        const ncnn::VkMat input = last_row_only ? row_view(normalized, normalized.h - 1, 1) : normalized;
        if (!head.record_logits(input, output, temporaries, cmd))
            return false;
    }
    ncnn::VkMat hidden_staging;
    if (!prepare_staging_batch(hidden_staging, normalized.h, normalized.w, slot.staging_allocator)
        || !record_prepared_activation_staging_download(normalized, normalized.h, normalized.w, hidden_staging,
                                                        cmd, impl.context->device(), head.option())
        || (produce_logits && (!prepare_staging_batch(slot.download, output.h, output.w, slot.staging_allocator) || !record_prepared_activation_staging_download(output, output.h, output.w, slot.download, cmd, impl.context->device(), head.option()))))
        return false;
    if (submit_compute_and_wait(cmd, impl.context->device()) != 0)
        return false;
    ++state.compute_submissions;
    state.batch_downloads += produce_logits ? 2 : 1;
    LatentLayerWorkspace_vulkan::Implementation::prepare_host(impl.hidden_host, normalized.h, normalized.w);
    if (!copy_staging_to_cpu_batch(hidden_staging, impl.hidden_host))
        return false;
    if (produce_logits)
    {
        LatentLayerWorkspace_vulkan::Implementation::prepare_host(impl.logits_host, output.h, output.w);
        if (!copy_staging_to_cpu_batch(slot.download, impl.logits_host))
            return false;
        logits.swap(impl.logits_host);
    }
    else
        logits.clear();
    hidden.swap(impl.hidden_host);
    impl.commit_combine(combine_uploads);
    // Release buffers under the allocator's command mutex; operator owners
    // stay alive until the mutex is released.
    expanded.release();
    uploaded.release();
    normalized.release();
    output.release();
    temporaries.clear();
    hc_work.projection.release();
    hc_work.pre.release();
    hidden_staging.release();
    combine_storage.clear();
    impl.clear_buffers();
    lock.unlock();
    impl.clear();
    return true;
#else
    (void)workspace;
    (void)head;
    (void)produce_logits;
    (void)last_row_only;
    (void)hidden;
    (void)logits;
    return false;
#endif
}

} // namespace moe
} // namespace ncnn
