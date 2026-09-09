#include "gateddeltanet_vulkan.h"
#include "linear.h"
#include "vulkancontext.h"

#include "kernels/statecache.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/gated_delta_net.comp.hex.h"
#endif

#if NCNN_MOE_USE_NCNN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>
#include <pipeline.h>

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>
#endif

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN

// One workgroup owns the recurrent state.  The token loop is intentionally
// sequential, while convolution channels and value heads remain parallel.
// This keeps the state transition on the device without changing the
// cross-token ordering required by DeltaNet.

static bool create_pipeline(
    const std::shared_ptr<VulkanContext>& context,
    const ncnn::Option& opt,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(
        gated_delta_net_shader,
        static_cast<int>(sizeof(gated_delta_net_shader) - 1),
        opt,
        0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(gated_delta_net_shader, 0);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_local_size_xyz(128, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(
        pipeline.release(),
        [context](ncnn::Pipeline* value) {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        });
    context->cache_pipeline(gated_delta_net_shader, 0, destination);
    return true;
}

#endif

class GatedDeltaState_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    ncnn::Option opt;
    ncnn::VkMat convolution;
    ncnn::VkMat recurrent;
    ncnn::VkMat zero_convolution;
    ncnn::VkMat zero_recurrent;
    ncnn::VkMat cpu_convolution;
    ncnn::VkMat cpu_recurrent;
    ncnn::VkMat initial_convolution;
    ncnn::VkMat initial_recurrent;
    std::vector<ncnn::VkMat> convolution_snapshots;
    std::vector<ncnn::VkMat> recurrent_snapshots;
    uint32_t convolution_size = 0;

    uint32_t kernel_size = 0;
    uint32_t head_count = 0;
    uint32_t head_dimension = 0;
    uint32_t value_head_dimension = 0;
    bool initialized = false;
    bool cpu_state_pending = false;
    bool state_unknown = false;
    bool transaction_active = false;
    bool transaction_initial_recorded = false;
    bool transaction_initial_pending = false;
    bool transaction_row_pending = false;
    bool transaction_row_submitted = false;
    bool transaction_requires_restore = false;
    bool transaction_state_unknown = false;
    size_t expected_rows = 0;
    size_t recorded_rows = 0;

    bool record_initial_state(ncnn::VkCompute& cmd)
    {
        if (!initialized)
        {
            const ncnn::VkMat& source_convolution = cpu_state_pending
                                                        ? cpu_convolution
                                                        : zero_convolution;
            const ncnn::VkMat& source_recurrent = cpu_state_pending
                                                      ? cpu_recurrent
                                                      : zero_recurrent;
            cmd.record_clone(source_convolution, convolution, opt);
            cmd.record_clone(source_recurrent, recurrent, opt);
            if (convolution.empty() || recurrent.empty())
                return false;
        }
        if (!transaction_active
            || transaction_initial_recorded
            || transaction_initial_pending)
        {
            return true;
        }
        if (initial_convolution.empty() || initial_recurrent.empty())
            return false;
        cmd.record_clone(
            convolution,
            initial_convolution,
            opt);
        cmd.record_clone(
            recurrent,
            initial_recurrent,
            opt);
        transaction_initial_pending = true;
        return true;
    }

    bool record_transaction_row(ncnn::VkCompute& cmd)
    {
        if (!transaction_active)
            return true;
        if (transaction_row_pending
            || recorded_rows >= expected_rows)
        {
            return false;
        }
        if (recorded_rows < convolution_snapshots.size())
        {
            cmd.record_clone(
                convolution,
                convolution_snapshots[recorded_rows],
                opt);
            cmd.record_clone(
                recurrent,
                recurrent_snapshots[recorded_rows],
                opt);
        }
        transaction_row_pending = true;
        return true;
    }

    bool restore_snapshot(
        const ncnn::VkMat& snapshot_convolution,
        const ncnn::VkMat& snapshot_recurrent)
    {
        VulkanRuntimeState& runtime_state = vulkan_context->runtime_state();
        std::unique_lock<std::mutex> lock(
            vulkan_context->command_mutex());
        ncnn::VkCompute cmd(
            vulkan_context->device(),
            vulkan_context->command_optimization_flags());
        cmd.record_clone(snapshot_convolution, convolution, opt);
        cmd.record_clone(snapshot_recurrent, recurrent, opt);
        if (submit_compute_and_wait(cmd, runtime_state) != 0)
            return false;
        ++runtime_state.compute_submissions;
        return true;
    }

    class RecordingGuard
    {
    public:
        explicit RecordingGuard(
            std::span<Implementation* const> _states)
            : states(_states)
        {
        }

        ~RecordingGuard()
        {
            if (completed)
                return;
            for (Implementation* state : states)
            {
                state->transaction_initial_pending = false;
                state->transaction_row_pending = false;
                state->transaction_row_submitted = false;
            }
        }

        void mark_submitted() noexcept
        {
            for (Implementation* state : states)
            {
                if (state->transaction_initial_pending)
                {
                    state->transaction_initial_pending = false;
                    state->transaction_initial_recorded = true;
                }
                if (state->transaction_row_pending)
                {
                    state->transaction_row_submitted = true;
                    state->transaction_requires_restore = true;
                }
            }
        }

        void mark_submit_failed() noexcept
        {
            for (Implementation* state : states)
            {
                if (state->transaction_active
                    && state->transaction_row_pending)
                {
                    if (state->transaction_initial_recorded)
                        state->transaction_requires_restore = true;
                    else
                        state->transaction_state_unknown = true;
                }
            }
        }

        void commit_rows() noexcept
        {
            for (Implementation* state : states)
            {
                if (!state->transaction_row_pending
                    || !state->transaction_row_submitted)
                {
                    continue;
                }
                ++state->recorded_rows;
                state->transaction_row_pending = false;
                state->transaction_row_submitted = false;
                state->transaction_requires_restore = false;
            }
            completed = true;
        }

    private:
        std::span<Implementation* const> states;
        bool completed = false;
    };
#endif
};

class GatedDeltaNet_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::shared_ptr<Bfloat16Linear_vulkan> fused_input;
    std::shared_ptr<Bfloat16Linear_vulkan> output_projection;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    ncnn::VkMat convolution_weight;
    ncnn::VkMat time_bias;
    ncnn::VkMat decay_log;
    ncnn::VkMat norm_weight;
    ncnn::Option opt;
    uint32_t convolution_size = 0;
    uint32_t value_size = 0;
    uint32_t fused_columns = 0;
    uint32_t head_count = 0;
    uint32_t kv_head_count = 0;
    uint32_t head_dimension = 0;
    uint32_t value_head_dimension = 0;
    uint32_t convolution_kernel_size = 0;
    float norm_epsilon = 0.0f;
    bool sigmoid_gate = false;
#endif
};

GatedDeltaState_vulkan::GatedDeltaState_vulkan()
    : d(new Implementation)
{
}

GatedDeltaState_vulkan::~GatedDeltaState_vulkan() = default;

GatedDeltaNet_vulkan::GatedDeltaNet_vulkan()
    : d(new Implementation)
{
}

GatedDeltaNet_vulkan::~GatedDeltaNet_vulkan() = default;

#if NCNN_MOE_WITH_VULKAN

static bool create_state_storage(
    ncnn::VkMat& destination,
    uint32_t elements,
    ncnn::VkAllocator* allocator)
{
    if (elements == 0
        || elements > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    destination.create(static_cast<int>(elements), 1, sizeof(float), allocator);
    return !destination.empty();
}

static bool fill_zero_staging(ncnn::VkMat& destination, uint32_t elements)
{
    if (destination.empty() || !destination.mapped_ptr())
        return false;
    std::fill_n(static_cast<float*>(destination.mapped_ptr()), elements, 0.0f);
    destination.allocator->flush(destination.data);
    destination.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    destination.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

static uint64_t vkmat_allocated_bytes(const ncnn::VkMat& value)
{
    return value.empty() ? 0 : static_cast<uint64_t>(value.buffer_capacity());
}

#endif

std::shared_ptr<GatedDeltaState_vulkan>
GatedDeltaState_vulkan::create(
    const std::shared_ptr<VulkanContext>& context,
    uint32_t convolution_size,
    uint32_t kernel_size,
    uint32_t head_count,
    uint32_t head_dimension,
    uint32_t value_head_dimension,
    const ncnn::Option& opt)
{
#if NCNN_MOE_WITH_VULKAN
    if (!context || convolution_size == 0 || kernel_size == 0
        || head_count == 0 || head_dimension == 0
        || value_head_dimension == 0)
    {
        return {};
    }
    const uint64_t head_elements = static_cast<uint64_t>(head_dimension)
                                   * value_head_dimension;
    const uint64_t convolution_elements = static_cast<uint64_t>(convolution_size) * kernel_size;
    const uint64_t max_elements = static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (head_elements > max_elements / head_count
        || convolution_elements > max_elements)
    {
        return {};
    }
    const uint64_t recurrent_elements = static_cast<uint64_t>(head_count) * head_elements;

    std::shared_ptr<GatedDeltaState_vulkan> result(
        new GatedDeltaState_vulkan);
    Implementation& implementation = *result->d;
    implementation.vulkan_context = context;
    implementation.opt = opt;
    implementation.convolution_size = convolution_size;
    implementation.kernel_size = kernel_size;
    implementation.head_count = head_count;
    implementation.head_dimension = head_dimension;
    implementation.value_head_dimension = value_head_dimension;
    if (!create_state_storage(
            implementation.convolution,
            static_cast<uint32_t>(convolution_elements),
            context->blob_allocator())
        || !create_state_storage(
            implementation.recurrent,
            static_cast<uint32_t>(recurrent_elements),
            context->blob_allocator())
        || !create_state_storage(
            implementation.zero_convolution,
            static_cast<uint32_t>(convolution_elements),
            context->staging_allocator())
        || !create_state_storage(
            implementation.zero_recurrent,
            static_cast<uint32_t>(recurrent_elements),
            context->staging_allocator())
        || !fill_zero_staging(
            implementation.zero_convolution,
            static_cast<uint32_t>(convolution_elements))
        || !fill_zero_staging(
            implementation.zero_recurrent,
            static_cast<uint32_t>(recurrent_elements)))
    {
        return {};
    }
    return result;
#else
    (void)context;
    (void)convolution_size;
    (void)kernel_size;
    (void)head_count;
    (void)head_dimension;
    (void)value_head_dimension;
    (void)opt;
    return {};
#endif
}

bool GatedDeltaState_vulkan::begin_transaction(
    size_t expected_rows) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    if (implementation.state_unknown
        || implementation.transaction_active)
        return false;
    implementation.transaction_active = false;
    implementation.transaction_initial_recorded = false;
    implementation.transaction_initial_pending = false;
    implementation.transaction_row_pending = false;
    implementation.transaction_row_submitted = false;
    implementation.transaction_requires_restore = false;
    implementation.transaction_state_unknown = false;
    implementation.expected_rows = 0;
    implementation.recorded_rows = 0;
    implementation.initial_convolution = ncnn::VkMat();
    implementation.initial_recurrent = ncnn::VkMat();
    implementation.convolution_snapshots.clear();
    implementation.recurrent_snapshots.clear();
    if (expected_rows == 0)
    {
        implementation.transaction_active = true;
        implementation.expected_rows = expected_rows;
        return true;
    }
    const uint64_t convolution_elements = static_cast<uint64_t>(implementation.convolution_size)
                                          * implementation.kernel_size;
    const uint64_t recurrent_elements = static_cast<uint64_t>(implementation.head_count)
                                        * implementation.head_dimension
                                        * implementation.value_head_dimension;
    try
    {
        if (!create_state_storage(
                implementation.initial_convolution,
                static_cast<uint32_t>(convolution_elements),
                implementation.vulkan_context->blob_allocator())
            || !create_state_storage(
                implementation.initial_recurrent,
                static_cast<uint32_t>(recurrent_elements),
                implementation.vulkan_context->blob_allocator()))
        {
            complete_transaction();
            return false;
        }
        const size_t snapshot_count = expected_rows > 0 ? expected_rows - 1 : 0;
        implementation.convolution_snapshots.resize(snapshot_count);
        implementation.recurrent_snapshots.resize(snapshot_count);
        for (size_t index = 0; index < snapshot_count; ++index)
        {
            if (!create_state_storage(
                    implementation.convolution_snapshots[index],
                    static_cast<uint32_t>(convolution_elements),
                    implementation.vulkan_context->blob_allocator())
                || !create_state_storage(
                    implementation.recurrent_snapshots[index],
                    static_cast<uint32_t>(recurrent_elements),
                    implementation.vulkan_context->blob_allocator()))
            {
                complete_transaction();
                return false;
            }
        }
    }
    catch (...)
    {
        complete_transaction();
        return false;
    }
    implementation.expected_rows = expected_rows;
    implementation.transaction_active = true;
    return true;
#else
    (void)expected_rows;
    return false;
#endif
}

bool GatedDeltaState_vulkan::prepare_cpu_state(
    const std::vector<float>& convolution,
    const std::vector<float>& recurrent)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    const size_t expected_convolution = static_cast<size_t>(
                                            implementation.convolution_size)
                                        * implementation.kernel_size;
    const size_t expected_recurrent = static_cast<size_t>(
                                          implementation.head_count)
                                      * implementation.head_dimension
                                      * implementation.value_head_dimension;
    if (implementation.initialized
        || convolution.size() != expected_convolution
        || recurrent.size() != expected_recurrent)
    {
        return false;
    }
    implementation.cpu_convolution.create(
        static_cast<int>(expected_convolution),
        1,
        sizeof(float),
        implementation.vulkan_context->staging_allocator());
    implementation.cpu_recurrent.create(
        static_cast<int>(expected_recurrent),
        1,
        sizeof(float),
        implementation.vulkan_context->staging_allocator());
    if (implementation.cpu_convolution.empty()
        || implementation.cpu_recurrent.empty()
        || !implementation.cpu_convolution.mapped_ptr()
        || !implementation.cpu_recurrent.mapped_ptr())
    {
        return false;
    }
    std::copy(
        convolution.begin(),
        convolution.end(),
        static_cast<float*>(implementation.cpu_convolution.mapped_ptr()));
    std::copy(
        recurrent.begin(),
        recurrent.end(),
        static_cast<float*>(implementation.cpu_recurrent.mapped_ptr()));
    implementation.cpu_convolution.allocator->flush(
        implementation.cpu_convolution.data);
    implementation.cpu_recurrent.allocator->flush(
        implementation.cpu_recurrent.data);
    implementation.cpu_convolution.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    implementation.cpu_convolution.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    implementation.cpu_recurrent.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    implementation.cpu_recurrent.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    implementation.cpu_state_pending = true;
    return true;
#else
    (void)convolution;
    (void)recurrent;
    return false;
#endif
}

bool GatedDeltaState_vulkan::prepare_transaction_finish(
    size_t committed_rows,
    size_t recorded_rows) noexcept
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    if (!implementation.transaction_active
        || recorded_rows != implementation.recorded_rows
        || committed_rows > recorded_rows
        || implementation.transaction_state_unknown)
    {
        return false;
    }
    bool restored = true;
    if (committed_rows < recorded_rows
        || implementation.transaction_requires_restore)
    {
        if (committed_rows == 0)
        {
            restored = implementation.transaction_initial_recorded
                       && implementation.restore_snapshot(
                           implementation.initial_convolution,
                           implementation.initial_recurrent);
        }
        else if (committed_rows - 1 < implementation.convolution_snapshots.size())
        {
            restored = implementation.restore_snapshot(
                implementation.convolution_snapshots[committed_rows - 1],
                implementation.recurrent_snapshots[committed_rows - 1]);
        }
        else
        {
            restored = false;
        }
    }
    return restored;
#else
    (void)committed_rows;
    (void)recorded_rows;
    return false;
#endif
}

void GatedDeltaState_vulkan::complete_transaction() noexcept
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    implementation.transaction_active = false;
    implementation.transaction_initial_recorded = false;
    implementation.transaction_initial_pending = false;
    implementation.transaction_row_pending = false;
    implementation.transaction_row_submitted = false;
    implementation.transaction_requires_restore = false;
    implementation.transaction_state_unknown = false;
    implementation.expected_rows = 0;
    implementation.recorded_rows = 0;
    implementation.initial_convolution = ncnn::VkMat();
    implementation.initial_recurrent = ncnn::VkMat();
    implementation.convolution_snapshots.clear();
    implementation.recurrent_snapshots.clear();
#endif
}

bool GatedDeltaState_vulkan::download(
    std::vector<float>& convolution,
    std::vector<float>& recurrent) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (implementation.state_unknown
        || !implementation.vulkan_context
        || implementation.convolution.empty()
        || implementation.recurrent.empty())
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const size_t convolution_count = static_cast<size_t>(
                                         implementation.convolution_size)
                                     * implementation.kernel_size;
    const size_t recurrent_count = static_cast<size_t>(
                                       implementation.head_count)
                                   * implementation.head_dimension
                                   * implementation.value_head_dimension;
    if (!prepare_staging_batch(
            transfer_slot.upload,
            1,
            static_cast<uint32_t>(convolution_count),
            transfer_slot.staging_allocator,
            runtime_state,
            sizeof(float))
        || !prepare_staging_batch(
            transfer_slot.download,
            1,
            static_cast<uint32_t>(recurrent_count),
            transfer_slot.staging_allocator,
            runtime_state,
            sizeof(float)))
    {
        return false;
    }
    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& cmd = *transfer_slot.command;
    if (transfer_slot.command_used && cmd.reset() != 0)
        return false;
    transfer_slot.command_used = true;
    // The two clones use separate staging buffers.  Reuse the slot's upload
    // and download allocations only after the first copy has completed.
    ncnn::VkMat convolution_staging = transfer_slot.upload;
    ncnn::VkMat recurrent_staging = transfer_slot.download;
    cmd.record_clone(
        implementation.convolution,
        convolution_staging,
        implementation.opt);
    cmd.record_clone(
        implementation.recurrent,
        recurrent_staging,
        implementation.opt);
    if (submit_compute_and_wait(cmd, runtime_state) != 0)
        return false;
    convolution_staging.allocator->invalidate(convolution_staging.data);
    recurrent_staging.allocator->invalidate(recurrent_staging.data);
    const ncnn::Mat convolution_mapped = convolution_staging.mapped();
    const ncnn::Mat recurrent_mapped = recurrent_staging.mapped();
    if (convolution_mapped.empty() || recurrent_mapped.empty())
        return false;
    convolution.assign(
        static_cast<const float*>(convolution_mapped.data),
        static_cast<const float*>(convolution_mapped.data) + convolution_count);
    recurrent.assign(
        static_cast<const float*>(recurrent_mapped.data),
        static_cast<const float*>(recurrent_mapped.data) + recurrent_count);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)convolution;
    (void)recurrent;
    return false;
#endif
}

uint64_t GatedDeltaState_vulkan::allocated_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    uint64_t bytes = vkmat_allocated_bytes(implementation.convolution)
                     + vkmat_allocated_bytes(implementation.recurrent)
                     + vkmat_allocated_bytes(implementation.zero_convolution)
                     + vkmat_allocated_bytes(implementation.zero_recurrent)
                     + vkmat_allocated_bytes(implementation.cpu_convolution)
                     + vkmat_allocated_bytes(implementation.cpu_recurrent)
                     + vkmat_allocated_bytes(implementation.initial_convolution)
                     + vkmat_allocated_bytes(implementation.initial_recurrent);
    for (const ncnn::VkMat& snapshot : implementation.convolution_snapshots)
        bytes += vkmat_allocated_bytes(snapshot);
    for (const ncnn::VkMat& snapshot : implementation.recurrent_snapshots)
        bytes += vkmat_allocated_bytes(snapshot);
    return bytes;
#else
    return 0;
#endif
}

std::shared_ptr<GatedDeltaNet_vulkan>
GatedDeltaNet_vulkan::create(
    const std::shared_ptr<Bfloat16Linear_vulkan>& fused_input,
    const TensorData& convolution_weight,
    const TensorData& time_bias,
    const TensorData& decay_log,
    const TensorData& norm_weight,
    const std::shared_ptr<Bfloat16Linear_vulkan>& output_projection,
    uint32_t head_count,
    uint32_t kv_head_count,
    uint32_t head_dimension,
    uint32_t value_head_dimension,
    uint32_t convolution_kernel_size,
    float norm_epsilon,
    bool sigmoid_gate,
    uint32_t vulkan_device_index,
    const VulkanRuntimePtr& vulkan_runtime,
    uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (!fused_input || !output_projection || head_count == 0
        || kv_head_count == 0 || head_count % kv_head_count != 0
        || head_dimension == 0 || value_head_dimension == 0
        || convolution_kernel_size == 0)
    {
        return {};
    }
    const uint32_t key_size = kv_head_count * head_dimension;
    const uint32_t value_size = head_count * value_head_dimension;
    const uint32_t convolution_size = key_size * 2 + value_size;
    const uint32_t fused_columns = convolution_size + value_size + head_count * 2;
    const std::shared_ptr<VulkanContext>& fused_context = fused_input->vulkan_context();
    const std::shared_ptr<VulkanContext>& output_context = output_projection->vulkan_context();
    if (!fused_context || !output_context
        || fused_context->optimization_flags() != optimization_flags
        || output_context->optimization_flags() != optimization_flags
        || fused_context->runtime().get() != vulkan_runtime.get()
        || fused_context.get() != output_context.get()
        || fused_input->output_columns() != fused_columns
        || output_projection->input_columns() != value_size
        || convolution_weight.element_count()
               != static_cast<size_t>(convolution_size)
                      * convolution_kernel_size
        || time_bias.element_count() != head_count
        || decay_log.element_count() != head_count
        || norm_weight.element_count() != value_head_dimension
        || (convolution_weight.dtype != DType::Float32
            && convolution_weight.dtype != DType::BFloat16)
        || (time_bias.dtype != DType::Float32
            && time_bias.dtype != DType::BFloat16)
        || (decay_log.dtype != DType::Float32
            && decay_log.dtype != DType::BFloat16)
        || (norm_weight.dtype != DType::Float32
            && norm_weight.dtype != DType::BFloat16))
    {
        return {};
    }

    std::shared_ptr<GatedDeltaNet_vulkan> result(
        new GatedDeltaNet_vulkan);
    Implementation& implementation = *result->d;
    implementation.vulkan_context = fused_context;
    implementation.fused_input = fused_input;
    implementation.output_projection = output_projection;
    implementation.opt = fused_input->option();
    implementation.convolution_size = convolution_size;
    implementation.value_size = value_size;
    implementation.fused_columns = fused_columns;
    implementation.head_count = head_count;
    implementation.kv_head_count = kv_head_count;
    implementation.head_dimension = head_dimension;
    implementation.value_head_dimension = value_head_dimension;
    implementation.convolution_kernel_size = convolution_kernel_size;
    implementation.norm_epsilon = norm_epsilon;
    implementation.sigmoid_gate = sigmoid_gate;
    {
        const std::lock_guard<std::mutex> lock(
            implementation.vulkan_context->command_mutex());
        if (!create_pipeline(
                implementation.vulkan_context,
                implementation.opt,
                implementation.pipeline))
        {
            return {};
        }
    }

    implementation.weight_allocator.reset(
        new ncnn::VkWeightAllocator(
            implementation.vulkan_context->device()));
    implementation.weight_staging_allocator.reset(
        new ncnn::VkWeightStagingAllocator(
            implementation.vulkan_context->device()));
    ncnn::Mat convolution_model;
    ncnn::Mat time_model;
    ncnn::Mat decay_model;
    ncnn::Mat norm_model;
    if (!prepare_float_tensor_upload(convolution_weight, convolution_model)
        || !prepare_float_tensor_upload(time_bias, time_model)
        || !prepare_float_tensor_upload(decay_log, decay_model)
        || !prepare_float_tensor_upload(norm_weight, norm_model))
    {
        return {};
    }
    ncnn::Option upload_option = implementation.opt;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    {
        const std::lock_guard<std::mutex> lock(
            implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer cmd(implementation.vulkan_context->device());
        cmd.record_upload(
            convolution_model,
            implementation.convolution_weight,
            upload_option);
        cmd.record_upload(time_model, implementation.time_bias, upload_option);
        cmd.record_upload(decay_model, implementation.decay_log, upload_option);
        cmd.record_upload(norm_model, implementation.norm_weight, upload_option);
        if (implementation.convolution_weight.empty()
            || implementation.time_bias.empty()
            || implementation.decay_log.empty()
            || implementation.norm_weight.empty()
            || cmd.submit_and_wait() != 0)
        {
            return {};
        }
    }
    implementation.weight_staging_allocator.reset();
    (void)vulkan_device_index;
    return result;
#else
    (void)fused_input;
    (void)convolution_weight;
    (void)time_bias;
    (void)decay_log;
    (void)norm_weight;
    (void)output_projection;
    (void)head_count;
    (void)kv_head_count;
    (void)head_dimension;
    (void)value_head_dimension;
    (void)convolution_kernel_size;
    (void)norm_epsilon;
    (void)sigmoid_gate;
    (void)vulkan_device_index;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

bool GatedDeltaNet_vulkan::forward_impl(
    const ActivationBuffer& input,
    LayerCache& cache,
    ActivationBuffer& projected,
    bool apply_input_rms_norm) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    const uint32_t input_columns = implementation.fused_input->input_columns();
    const uint32_t output_columns = implementation.output_projection->output_columns();
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || input.rows() == 0
        || input.columns() != input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
        || (cache.transaction.active && input.rows() != 1))
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();

    bool created_device_state = false;
    if (!cache.gated_delta_device_state)
    {
        const bool has_cpu_state = !cache.gated_delta_convolution.empty()
                                   || !cache.gated_delta_recurrent.empty();
        cache.gated_delta_device_state = GatedDeltaState_vulkan::create(
            implementation.vulkan_context,
            implementation.convolution_size,
            implementation.convolution_kernel_size,
            implementation.head_count,
            implementation.head_dimension,
            implementation.value_head_dimension,
            implementation.opt);
        if (!cache.gated_delta_device_state)
            return false;
        created_device_state = true;
        if (has_cpu_state
            && !cache.gated_delta_device_state->prepare_cpu_state(
                cache.gated_delta_convolution,
                cache.gated_delta_recurrent))
        {
            cache.gated_delta_device_state.reset();
            return false;
        }
    }
    struct DeviceStateCreationAttempt
    {
        LayerCache* cache = nullptr;

        ~DeviceStateCreationAttempt()
        {
            if (cache)
            {
                cache->gated_delta_device_state.reset();
                cache->device_allocated_size = 0;
            }
        }

        void complete() noexcept
        {
            cache = nullptr;
        }
    } creation_attempt{
        created_device_state ? &cache : nullptr};
    std::shared_ptr<GatedDeltaState_vulkan> state = cache.gated_delta_device_state;
    GatedDeltaState_vulkan::Implementation& state_implementation = *state->d;
    if (state_implementation.state_unknown)
        return false;
    if (cache.transaction.active
        && !state_implementation.transaction_active)
    {
        if (!state->begin_transaction(
                cache.transaction.expected_rows))
            return false;
    }
    cache.device_allocated_size = state->allocated_bytes();

    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = vulkan_activation_storage_variant(implementation.opt) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(
                                       input.rows() * input.columns() * sizeof(float),
                                       input.dtype());
    if (!fill_staging_upload(
            input,
            transfer_slot.upload,
            transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(
            transfer_slot.download,
            input.rows(),
            output_columns,
            transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }
    const DType projected_dtype = projected.dtype();
    projected.reset(
        input.rows(),
        output_columns,
        false);

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& cmd = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (cmd.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(
                 transfer_slot.upload,
                 input_gpu,
                 cmd,
                 implementation.opt))
    {
        return false;
    }
    std::array<GatedDeltaState_vulkan::Implementation*, 1>
        transaction_states = {&state_implementation};
    GatedDeltaState_vulkan::Implementation::RecordingGuard
        transaction_recording(transaction_states);
    if (!state_implementation.record_initial_state(cmd))
    {
        return false;
    }

    const bool use_input_rms_norm = apply_input_rms_norm
                                    && implementation.fused_input->has_rms_norm_chain();
    ncnn::VkMat fused_gpu;
    fused_gpu.create(
        static_cast<int>(implementation.fused_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    ncnn::VkMat recurrent_output_gpu;
    recurrent_output_gpu.create(
        static_cast<int>(implementation.value_size),
        static_cast<int>(input.rows()),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    output_gpu.create(
        static_cast<int>(output_columns),
        static_cast<int>(input.rows()),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    if (fused_gpu.empty() || recurrent_output_gpu.empty() || output_gpu.empty())
        return false;

    if (use_input_rms_norm)
    {
        implementation.fused_input->record_rms_norm_projection(
            input_gpu,
            fused_gpu,
            cmd);
    }
    else
    {
        implementation.fused_input->record_scalar_projection(
            input_gpu,
            fused_gpu,
            cmd);
    }

    std::vector<ncnn::VkMat> delta_bindings = {
        fused_gpu,
        state_implementation.convolution,
        state_implementation.recurrent,
        implementation.convolution_weight,
        implementation.time_bias,
        implementation.decay_log,
        implementation.norm_weight,
        recurrent_output_gpu};
    std::vector<ncnn::vk_constant_type> delta_constants(11);
    delta_constants[0].u32 = implementation.convolution_size;
    delta_constants[1].u32 = implementation.value_size;
    delta_constants[2].u32 = implementation.fused_columns;
    delta_constants[3].u32 = implementation.head_count;
    delta_constants[4].u32 = implementation.kv_head_count;
    delta_constants[5].u32 = implementation.head_dimension;
    delta_constants[6].u32 = implementation.value_head_dimension;
    delta_constants[7].u32 = implementation.convolution_kernel_size;
    delta_constants[8].u32 = static_cast<uint32_t>(input.rows());
    delta_constants[9].f = implementation.norm_epsilon;
    delta_constants[10].u32 = implementation.sigmoid_gate ? 1u : 0u;
    ncnn::VkMat delta_dispatcher;
    delta_dispatcher.w = 128;
    delta_dispatcher.h = 1;
    delta_dispatcher.c = 1;
    cmd.record_pipeline_readonly(
        implementation.pipeline.get(),
        delta_bindings,
        {0, 0, 0, 1, 1, 1, 1, 0},
        delta_constants,
        delta_dispatcher);
    if (!state_implementation.record_transaction_row(cmd))
        return false;

    implementation.output_projection->record_scalar_projection(
        recurrent_output_gpu,
        output_gpu,
        cmd);

    if (!record_prepared_activation_staging_download(
            output_gpu,
            input.rows(),
            output_columns,
            transfer_slot.download,
            cmd,
            implementation.vulkan_context->device(),
            implementation.opt,
            projected_dtype))
    {
        return false;
    }
    if (submit_compute_and_wait(cmd, runtime_state) != 0)
    {
        transaction_recording.mark_submit_failed();
        if (!state_implementation.transaction_active
            && !created_device_state)
        {
            state_implementation.state_unknown = true;
        }
        return false;
    }
    transaction_recording.mark_submitted();
    if (!copy_staging_to_cpu_batch(transfer_slot.download, projected))
    {
        if (!state_implementation.transaction_active
            && !created_device_state)
        {
            state_implementation.state_unknown = true;
        }
        return false;
    }
    transaction_recording.commit_rows();
    state_implementation.initialized = true;
    state_implementation.cpu_state_pending = false;
    state_implementation.cpu_convolution = ncnn::VkMat();
    state_implementation.cpu_recurrent = ncnn::VkMat();
    state_implementation.zero_convolution = ncnn::VkMat();
    state_implementation.zero_recurrent = ncnn::VkMat();
    runtime_state.dispatches += 3;
    ++runtime_state.gated_delta_fusions;
    ++runtime_state.gated_delta_submissions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    cache.device_allocated_size = state->allocated_bytes();
    creation_attempt.complete();
    return true;
#else
    (void)input;
    (void)cache;
    (void)projected;
    (void)apply_input_rms_norm;
    return false;
#endif
}

bool GatedDeltaNet_vulkan::forward(
    const ActivationBuffer& normalized,
    LayerCache& cache,
    ActivationBuffer& projected) const
{
    return forward_impl(normalized, cache, projected, false);
}

bool GatedDeltaNet_vulkan::forward_input_rms_norm(
    const ActivationBuffer& input,
    LayerCache& cache,
    ActivationBuffer& projected) const
{
    if (!has_input_rms_norm())
        return false;
    return forward_impl(input, cache, projected, true);
}

bool GatedDeltaNet_vulkan::has_input_rms_norm() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d
           && d->fused_input
           && d->fused_input->has_rms_norm_chain();
#else
    return false;
#endif
}

GatedDeltaBatchResult_vulkan
GatedDeltaNet_vulkan::forward_batch(
    std::span<const GatedDeltaBatchEntry_vulkan> entries) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (entries.empty())
        return GatedDeltaBatchResult_vulkan::Executed;
    if (entries.size() == 1)
    {
        const GatedDeltaBatchEntry_vulkan& entry = entries.front();
        if (!entry.normalized || !entry.cache || !entry.projected)
            return GatedDeltaBatchResult_vulkan::NotExecuted;
        if (forward(*entry.normalized, *entry.cache, *entry.projected))
            return GatedDeltaBatchResult_vulkan::Executed;
        return entry.cache->gated_delta_device_state
                   ? GatedDeltaBatchResult_vulkan::Failed
                   : GatedDeltaBatchResult_vulkan::NotExecuted;
    }
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || entries.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return GatedDeltaBatchResult_vulkan::NotExecuted;
    }
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    const uint32_t input_columns = implementation.fused_input->input_columns();
    const uint32_t output_columns = implementation.output_projection->output_columns();

    const size_t total_rows = entries.size();
    ActivationBuffer combined_normalized;
    combined_normalized.reset(
        total_rows,
        input_columns,
        false);
    std::vector<std::shared_ptr<GatedDeltaState_vulkan>> states;
    states.reserve(entries.size());
    std::vector<bool> created_states;
    created_states.reserve(entries.size());
    std::vector<LayerCache*> created_state_caches;
    created_state_caches.reserve(entries.size());
    struct BatchDeviceStateCreationAttempt
    {
        explicit BatchDeviceStateCreationAttempt(
            std::vector<LayerCache*>& _created_caches)
            : created_caches(&_created_caches)
        {
        }

        ~BatchDeviceStateCreationAttempt()
        {
            if (!created_caches)
                return;
            for (LayerCache* cache : *created_caches)
            {
                cache->gated_delta_device_state.reset();
                cache->device_allocated_size = 0;
            }
        }

        void complete() noexcept
        {
            created_caches = nullptr;
        }

    private:
        std::vector<LayerCache*>* created_caches;
    } creation_attempt(created_state_caches);

    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        const GatedDeltaBatchEntry_vulkan& entry = entries[entry_index];
        if (!entry.normalized || !entry.cache || !entry.projected
            || entry.normalized->rows() != 1
            || entry.normalized->columns() != input_columns)
        {
            return GatedDeltaBatchResult_vulkan::NotExecuted;
        }
        std::copy_n(
            entry.normalized->row(0),
            input_columns,
            combined_normalized.row(entry_index));

        LayerCache& cache = *entry.cache;
        if (cache.transaction.active
            && !cache.gated_delta_device_state)
        {
            return GatedDeltaBatchResult_vulkan::NotExecuted;
        }
        bool created_device_state = false;
        if (!cache.gated_delta_device_state)
        {
            const bool has_cpu_state = !cache.gated_delta_convolution.empty()
                                       || !cache.gated_delta_recurrent.empty();
            cache.gated_delta_device_state = GatedDeltaState_vulkan::create(
                implementation.vulkan_context,
                implementation.convolution_size,
                implementation.convolution_kernel_size,
                implementation.head_count,
                implementation.head_dimension,
                implementation.value_head_dimension,
                implementation.opt);
            if (!cache.gated_delta_device_state)
                return GatedDeltaBatchResult_vulkan::NotExecuted;
            created_device_state = true;
            if (has_cpu_state
                && !cache.gated_delta_device_state->prepare_cpu_state(
                    cache.gated_delta_convolution,
                    cache.gated_delta_recurrent))
            {
                cache.gated_delta_device_state.reset();
                return GatedDeltaBatchResult_vulkan::NotExecuted;
            }
            created_state_caches.push_back(&cache);
        }
        std::shared_ptr<GatedDeltaState_vulkan> state = cache.gated_delta_device_state;
        GatedDeltaState_vulkan::Implementation& state_implementation = *state->d;
        if (state_implementation.state_unknown)
            return GatedDeltaBatchResult_vulkan::Failed;
        if (state_implementation.vulkan_context != implementation.vulkan_context)
            return GatedDeltaBatchResult_vulkan::NotExecuted;
        if (cache.transaction.active
            && !state_implementation.transaction_active)
        {
            if (!state->begin_transaction(
                    cache.transaction.expected_rows))
                return GatedDeltaBatchResult_vulkan::NotExecuted;
        }
        cache.device_allocated_size = state->allocated_bytes();
        states.push_back(std::move(state));
        created_states.push_back(created_device_state);
    }

    const auto mark_nontransaction_existing_states_unknown = [&]() noexcept {
        for (size_t index = 0; index < states.size(); ++index)
        {
            if (!created_states[index]
                && !states[index]->d->transaction_active)
            {
                states[index]->d->state_unknown = true;
            }
        }
    };

    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = vulkan_activation_storage_variant(implementation.opt) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(
                                       total_rows
                                           * input_columns
                                           * sizeof(float),
                                       combined_normalized.dtype());
    if (!fill_staging_upload(
            combined_normalized,
            transfer_slot.upload,
            transfer_slot.staging_allocator, runtime_state)
        || !prepare_staging_batch(
            transfer_slot.download,
            total_rows,
            output_columns,
            transfer_slot.staging_allocator, runtime_state))
    {
        return GatedDeltaBatchResult_vulkan::NotExecuted;
    }
    ActivationBuffer combined_projected;
    combined_projected.reset(
        total_rows,
        output_columns,
        false);
    for (const GatedDeltaBatchEntry_vulkan& entry : entries)
    {
        entry.projected->reset(
            1,
            output_columns,
            false);
    }

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& cmd = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (cmd.reset() != 0)
            return GatedDeltaBatchResult_vulkan::NotExecuted;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_mapped_upload(
                 transfer_slot.upload,
                 input_gpu,
                 cmd,
                 implementation.opt))
    {
        return GatedDeltaBatchResult_vulkan::NotExecuted;
    }
    ncnn::VkMat fused_gpu;
    fused_gpu.create(
        static_cast<int>(implementation.fused_columns),
        static_cast<int>(total_rows),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    ncnn::VkMat recurrent_output_gpu;
    recurrent_output_gpu.create(
        static_cast<int>(implementation.value_size),
        static_cast<int>(total_rows),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    output_gpu.create(
        static_cast<int>(output_columns),
        static_cast<int>(total_rows),
        sizeof(float),
        implementation.vulkan_context->blob_allocator());
    if (fused_gpu.empty() || recurrent_output_gpu.empty() || output_gpu.empty())
        return GatedDeltaBatchResult_vulkan::NotExecuted;

    std::vector<GatedDeltaState_vulkan::Implementation*>
        transaction_states;
    transaction_states.reserve(states.size());
    for (const std::shared_ptr<GatedDeltaState_vulkan>& state : states)
        transaction_states.push_back(state->d.get());
    GatedDeltaState_vulkan::Implementation::RecordingGuard
        transaction_recording(transaction_states);

    for (const std::shared_ptr<GatedDeltaState_vulkan>& state : states)
    {
        GatedDeltaState_vulkan::Implementation& state_implementation = *state->d;
        if (!state_implementation.record_initial_state(cmd))
        {
            return GatedDeltaBatchResult_vulkan::NotExecuted;
        }
    }

    implementation.fused_input->record_scalar_projection(
        input_gpu,
        fused_gpu,
        cmd);

    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        GatedDeltaState_vulkan::Implementation& state_implementation = *states[entry_index]->d;
        ncnn::VkMat fused_view = row_view(fused_gpu, entry_index, 1);
        ncnn::VkMat recurrent_view = row_view(recurrent_output_gpu, entry_index, 1);
        if (fused_view.empty() || recurrent_view.empty())
            return GatedDeltaBatchResult_vulkan::NotExecuted;
        std::vector<ncnn::VkMat> delta_bindings = {
            fused_view,
            state_implementation.convolution,
            state_implementation.recurrent,
            implementation.convolution_weight,
            implementation.time_bias,
            implementation.decay_log,
            implementation.norm_weight,
            recurrent_view};
        std::vector<ncnn::vk_constant_type> delta_constants(11);
        delta_constants[0].u32 = implementation.convolution_size;
        delta_constants[1].u32 = implementation.value_size;
        delta_constants[2].u32 = implementation.fused_columns;
        delta_constants[3].u32 = implementation.head_count;
        delta_constants[4].u32 = implementation.kv_head_count;
        delta_constants[5].u32 = implementation.head_dimension;
        delta_constants[6].u32 = implementation.value_head_dimension;
        delta_constants[7].u32 = implementation.convolution_kernel_size;
        delta_constants[8].u32 = 1;
        delta_constants[9].f = implementation.norm_epsilon;
        delta_constants[10].u32 = implementation.sigmoid_gate ? 1u : 0u;
        ncnn::VkMat delta_dispatcher;
        delta_dispatcher.w = 128;
        delta_dispatcher.h = 1;
        delta_dispatcher.c = 1;
        cmd.record_pipeline_readonly(
            implementation.pipeline.get(),
            delta_bindings,
            {0, 0, 0, 1, 1, 1, 1, 0},
            delta_constants,
            delta_dispatcher);
        if (!state_implementation.record_transaction_row(cmd))
            return GatedDeltaBatchResult_vulkan::NotExecuted;
    }

    implementation.output_projection->record_scalar_projection(
        recurrent_output_gpu,
        output_gpu,
        cmd);

    if (!record_prepared_activation_staging_download(
            output_gpu,
            total_rows,
            output_columns,
            transfer_slot.download,
            cmd,
            implementation.vulkan_context->device(),
            implementation.opt,
            combined_projected.dtype()))
    {
        return GatedDeltaBatchResult_vulkan::NotExecuted;
    }
    if (submit_compute_and_wait(cmd, runtime_state) != 0)
    {
        transaction_recording.mark_submit_failed();
        mark_nontransaction_existing_states_unknown();
        return GatedDeltaBatchResult_vulkan::Failed;
    }
    transaction_recording.mark_submitted();
    if (!copy_staging_to_cpu_batch(
            transfer_slot.download,
            combined_projected))
    {
        mark_nontransaction_existing_states_unknown();
        return GatedDeltaBatchResult_vulkan::Failed;
    }
    transaction_recording.commit_rows();
    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        GatedDeltaState_vulkan::Implementation& state_implementation = *states[entry_index]->d;
        state_implementation.initialized = true;
        state_implementation.cpu_state_pending = false;
        state_implementation.cpu_convolution = ncnn::VkMat();
        state_implementation.cpu_recurrent = ncnn::VkMat();
        state_implementation.zero_convolution = ncnn::VkMat();
        state_implementation.zero_recurrent = ncnn::VkMat();
    }
    lock.unlock();

    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        LayerCache& cache = *entries[entry_index].cache;
        std::copy_n(
            combined_projected.row(entry_index),
            output_columns,
            entries[entry_index].projected->row(0));
        cache.gated_delta_convolution.clear();
        cache.gated_delta_recurrent.clear();
        cache.gated_delta_token_count += 1;
        cache.device_allocated_size = states[entry_index]->allocated_bytes();
    }
    runtime_state.dispatches += entries.size() + 2;
    runtime_state.gated_delta_fusions += entries.size();
    ++runtime_state.gated_delta_submissions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    creation_attempt.complete();
    return GatedDeltaBatchResult_vulkan::Executed;
#else
    (void)entries;
    return GatedDeltaBatchResult_vulkan::NotExecuted;
#endif
}

} // namespace moe
} // namespace ncnn
