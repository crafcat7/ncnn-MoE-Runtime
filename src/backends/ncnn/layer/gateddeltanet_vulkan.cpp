#include "backends/ncnn/layer/gateddeltanet_vulkan.h"
#include "backends/ncnn/core/context.h"
#include "backends/ncnn/linear/linear.h"

#include "kernels/statecache.h"
#include "kernels/vulkan/gated_delta_conv.comp.hex.h"
#include "kernels/vulkan/gated_delta_conv_history.comp.hex.h"
#include "kernels/vulkan/gated_delta_conv_parallel.comp.hex.h"
#include "kernels/vulkan/gated_delta_norm.comp.hex.h"
#include "kernels/vulkan/gated_delta_qk.comp.hex.h"
#include "kernels/vulkan/gated_delta_rule.comp.hex.h"
#include "kernels/vulkan/gated_delta_rule_fast.comp.hex.h"
#include "kernels/vulkan/gated_delta_rule_reg.comp.hex.h"

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

namespace ncnn {
namespace moe {

static bool gated_delta_record_columns(uint32_t head_count, uint32_t kv_head_count,
                                       uint32_t head_dimension, uint32_t value_head_dimension,
                                       uint32_t gate_channels,
                                       uint32_t& convolution_columns, uint32_t& value_columns,
                                       uint32_t& fused_columns) noexcept
{
    const uint64_t key_size = saturating_multiply_u64(kv_head_count, head_dimension);
    const uint64_t values = saturating_multiply_u64(head_count, value_head_dimension);
    const uint64_t int_max = static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (key_size == UINT64_MAX || values == UINT64_MAX || key_size > int_max / 2 || values > int_max)
        return false;

    const uint64_t convolution = saturating_add_u64(saturating_multiply_u64(key_size, 2), values);
    const uint64_t gates = saturating_multiply_u64(head_count, 1 + static_cast<uint64_t>(gate_channels));
    const uint64_t fused = saturating_add_u64(convolution, saturating_add_u64(values, gates));
    if (convolution == UINT64_MAX || fused == UINT64_MAX || fused > int_max)
        return false;

    convolution_columns = static_cast<uint32_t>(convolution);
    value_columns = static_cast<uint32_t>(values);
    fused_columns = static_cast<uint32_t>(fused);
    return true;
}

std::vector<uint64_t> GatedDeltaNet_vulkan::scratch_elements(uint32_t head_count,
                                                             uint32_t kv_head_count,
                                                             uint32_t head_dimension,
                                                             uint32_t value_head_dimension,
                                                             uint32_t gate_channels,
                                                             uint32_t rows)
{
    uint32_t convolution_columns = 0;
    uint32_t value_columns = 0;
    uint32_t fused_columns = 0;
    if (rows > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || !gated_delta_record_columns(head_count, kv_head_count, head_dimension, value_head_dimension,
                                       gate_channels, convolution_columns, value_columns, fused_columns))
    {
        return {UINT64_MAX};
    }

    (void)convolution_columns;
    const uint64_t fused_elements = saturating_multiply_u64(fused_columns, rows);
    const uint64_t recurrent_elements = saturating_multiply_u64(value_columns, rows);
    if (fused_elements == UINT64_MAX || recurrent_elements == UINT64_MAX)
        return {UINT64_MAX};

    // Token-parallel conv reads fused input for gates and writes windowed output
    // to a second fused-sized buffer; sequential conv rewrites fused in place.
    if (rows >= 8)
    {
        const uint64_t bf16_stage = saturating_multiply_u64((value_columns + 1u) / 2u, rows);
        return {fused_elements, recurrent_elements, fused_elements, bf16_stage, bf16_stage};
    }
    return {fused_elements, recurrent_elements, fused_elements};
}

// Records one dispatch; group counts are in workgroup units, with token count in y.
static void record_dispatch(ncnn::VkCompute& cmd,
                            const std::shared_ptr<ncnn::Pipeline>& pipeline,
                            const std::vector<ncnn::VkMat>& bindings,
                            std::vector<unsigned char> readonly,
                            const std::vector<ncnn::vk_constant_type>& constants,
                            uint32_t groups,
                            uint32_t rows = 1)
{
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(static_cast<uint32_t>(pipeline->local_size_x()) * groups);
    dispatcher.h = static_cast<int>(rows);
    dispatcher.c = 1;
    cmd.record_pipeline_readonly(pipeline.get(), bindings, readonly, constants, dispatcher);
}

static bool create_state_storage(ncnn::VkMat& destination,
                                 uint64_t elements,
                                 ncnn::VkAllocator* allocator)
{
    destination.create(static_cast<int>(elements), 1, sizeof(float), allocator);
    return !destination.empty();
}

static bool fill_zero_staging(ncnn::VkMat& destination, uint64_t elements)
{
    if (destination.empty() || !destination.mapped_ptr())
        return false;

    std::fill_n(static_cast<float*>(destination.mapped_ptr()), static_cast<size_t>(elements), 0.0f);
    (void)bind_host_in(destination);
    return true;
}

static uint64_t vkmat_allocated_bytes(const ncnn::VkMat& value) noexcept
{
    return value.empty() ? 0 : static_cast<uint64_t>(value.buffer_capacity());
}

class GatedDeltaState_vulkan::Implementation
{
public:
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
    bool recording_pending = false;
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

        cmd.record_clone(convolution, initial_convolution, opt);
        cmd.record_clone(recurrent, initial_recurrent, opt);
        transaction_initial_pending = true;
        return true;
    }

    bool record_transaction_row(ncnn::VkCompute& cmd)
    {
        if (!transaction_active)
            return true;
        if (transaction_row_pending || recorded_rows >= expected_rows)
            return false;
        if (recorded_rows < convolution_snapshots.size())
        {
            cmd.record_clone(convolution, convolution_snapshots[recorded_rows], opt);
            cmd.record_clone(recurrent, recurrent_snapshots[recorded_rows], opt);
        }
        transaction_row_pending = true;
        return true;
    }

    bool restore_snapshot(const ncnn::VkMat& snapshot_convolution,
                          const ncnn::VkMat& snapshot_recurrent)
    {
        VulkanRuntimeState& runtime_state = vulkan_context->runtime_state();
        const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
        ncnn::VkCompute cmd(vulkan_context->device());
        cmd.record_clone(snapshot_convolution, convolution, opt);
        cmd.record_clone(snapshot_recurrent, recurrent, opt);
        if (submit_wait(cmd, vulkan_context->device()) != 0)
            return false;
        ++runtime_state.compute_submissions;
        return true;
    }

    class RecordingGuard
    {
    public:
        explicit RecordingGuard(std::span<Implementation* const> _states)
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
                if (state->transaction_active && state->transaction_row_pending)
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
                if (!state->transaction_row_pending || !state->transaction_row_submitted)
                    continue;
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
};

class GatedDeltaWork_vulkan::Implementation
{
public:
    LayerCache* cache = nullptr;
    std::shared_ptr<GatedDeltaState_vulkan> state;
    ncnn::VkMat input;
    ncnn::VkMat fused;
    // Destination of the token-parallel convolution, which cannot run in place.
    ncnn::VkMat conv_out;
    ncnn::VkMat row_scale;
    ncnn::VkMat recurrent_output;
    ncnn::VkMat rec_bf16;
    ncnn::VkMat projected;
    size_t rows = 0;
    uint32_t dispatches = 0;
    uint32_t int4_ops = 0;
    uint64_t previous_state_allocated_size = 0;
    bool created_state = false;
    bool owns_recording = false;
    bool recorded = false;
    bool failed = false;
    bool completed = false;

    ~Implementation()
    {
        if (!state || completed)
            return;
        auto& implementation = *state->d;
        if (owns_recording)
        {
            implementation.recording_pending = false;
            implementation.transaction_initial_pending = false;
            implementation.transaction_row_pending = false;
            implementation.transaction_row_submitted = false;
        }
        if (created_state && cache && cache->gated_delta_device_state == state)
            cache->gated_delta_device_state.reset();
    }
};

GatedDeltaWork_vulkan::GatedDeltaWork_vulkan()
    : d(new Implementation)
{
}

GatedDeltaWork_vulkan::~GatedDeltaWork_vulkan() = default;

void GatedDeltaWork_vulkan::mark_submit_failed() noexcept
{
    if (!d->state || d->completed)
        return;

    d->failed = true;
    auto& state = *d->state->d;
    if (state.transaction_active && state.transaction_row_pending)
    {
        if (state.transaction_initial_recorded)
            state.transaction_requires_restore = true;
        else
            state.transaction_state_unknown = true;
    }
    else if (!d->created_state)
    {
        state.state_unknown = true;
    }
}

bool GatedDeltaWork_vulkan::commit() noexcept
{
    if (!d->recorded || d->failed || d->completed || !d->cache
        || d->cache->gated_delta_device_state != d->state)
    {
        return false;
    }

    auto& state = *d->state->d;
    if (state.transaction_initial_pending)
    {
        state.transaction_initial_pending = false;
        state.transaction_initial_recorded = true;
    }
    if (state.transaction_row_pending)
    {
        ++state.recorded_rows;
        state.transaction_row_pending = false;
        state.transaction_row_submitted = false;
        state.transaction_requires_restore = false;
    }
    state.initialized = true;
    state.recording_pending = false;
    state.cpu_state_pending = false;
    state.cpu_convolution = ncnn::VkMat();
    state.cpu_recurrent = ncnn::VkMat();
    state.zero_convolution = ncnn::VkMat();
    state.zero_recurrent = ncnn::VkMat();

    d->cache->gated_delta_convolution.clear();
    d->cache->gated_delta_recurrent.clear();
    for (size_t row = 0; row < d->rows; ++row)
        record_gated_delta_cache_transaction_row(*d->cache);
    d->cache->gated_delta_token_count += d->rows;
    d->cache->device_allocated_size -= std::min(d->cache->device_allocated_size, d->previous_state_allocated_size);
    d->cache->device_allocated_size += d->state->allocated_bytes();

    VulkanRuntimeState& runtime = state.vulkan_context->runtime_state();
    runtime.dispatches += d->dispatches;
    runtime.int4_dispatches += d->int4_ops;
    ++runtime.gated_delta_fusions;
    ++runtime.gated_delta_submissions;
    d->completed = true;
    return true;
}

class GatedDeltaNet_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> vulkan_context;
    std::shared_ptr<Bfloat16Linear_vulkan> fused_input;
    std::shared_ptr<Bfloat16Linear_vulkan> output_projection;
    std::array<std::shared_ptr<QnkLinear_vulkan>, 4> quantized_inputs;
    std::shared_ptr<QnkLinear_vulkan> quantized_output;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> rule;
    // Register-resident variants: fast vec4 (decode + prefill) and legacy scalar reg.
    std::shared_ptr<ncnn::Pipeline> rule_dec;
    std::shared_ptr<ncnn::Pipeline> rule_pref;
    std::shared_ptr<ncnn::Pipeline> rule_reg;
    std::shared_ptr<ncnn::Pipeline> conv;
    // Token-parallel prefill convolution plus the window rewrite that follows it.
    std::shared_ptr<ncnn::Pipeline> conv_par;
    std::shared_ptr<ncnn::Pipeline> conv_hist;
    std::shared_ptr<ncnn::Pipeline> qk;
    std::shared_ptr<ncnn::Pipeline> norm;
    std::shared_ptr<ncnn::Pipeline> norm_bf16;
    std::vector<ncnn::vk_constant_type> constants;
    ncnn::VkMat convolution_weight;
    ncnn::VkMat time_bias;
    ncnn::VkMat decay_log;
    ncnn::VkMat norm_weight;
    ncnn::Option opt;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t int4_ops = 0;
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
    uint32_t gate_channels = 1;
    float lower_bound = 0.0f;
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

std::shared_ptr<GatedDeltaState_vulkan>
GatedDeltaState_vulkan::create(const std::shared_ptr<VulkanContext>& context,
                               uint32_t convolution_size,
                               uint32_t kernel_size,
                               uint32_t head_count,
                               uint32_t head_dimension,
                               uint32_t value_head_dimension,
                               const ncnn::Option& opt)
{
    const uint64_t head_elements = static_cast<uint64_t>(head_dimension) * value_head_dimension;
    const uint64_t convolution_elements = static_cast<uint64_t>(convolution_size) * kernel_size;
    const uint64_t max_elements = static_cast<uint64_t>(std::numeric_limits<int>::max());
    const uint64_t recurrent_elements = saturating_multiply_u64(head_count, head_elements);
    if (convolution_elements == 0 || convolution_elements > max_elements
        || recurrent_elements == 0 || recurrent_elements > max_elements)
    {
        return {};
    }

    std::shared_ptr<GatedDeltaState_vulkan> result(new GatedDeltaState_vulkan);
    Implementation& implementation = *result->d;
    implementation.vulkan_context = context;
    implementation.opt = opt;
    implementation.convolution_size = convolution_size;
    implementation.kernel_size = kernel_size;
    implementation.head_count = head_count;
    implementation.head_dimension = head_dimension;
    implementation.value_head_dimension = value_head_dimension;
    if (!create_state_storage(implementation.convolution, convolution_elements, context->blob_allocator())
        || !create_state_storage(implementation.recurrent, recurrent_elements, context->blob_allocator())
        || !create_state_storage(implementation.zero_convolution, convolution_elements, context->staging_allocator())
        || !create_state_storage(implementation.zero_recurrent, recurrent_elements, context->staging_allocator())
        || !fill_zero_staging(implementation.zero_convolution, convolution_elements)
        || !fill_zero_staging(implementation.zero_recurrent, recurrent_elements))
    {
        return {};
    }
    return result;
}

bool GatedDeltaState_vulkan::begin_transaction(size_t expected_rows) noexcept
{
    Implementation& implementation = *d;
    if (implementation.state_unknown || implementation.transaction_active)
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
    const uint64_t head_elements = static_cast<uint64_t>(implementation.head_dimension)
                                   * implementation.value_head_dimension;
    const uint64_t recurrent_elements = saturating_multiply_u64(implementation.head_count, head_elements);
    try
    {
        if (!create_state_storage(implementation.initial_convolution, convolution_elements,
                                  implementation.vulkan_context->blob_allocator())
            || !create_state_storage(implementation.initial_recurrent, recurrent_elements,
                                     implementation.vulkan_context->blob_allocator()))
        {
            complete_transaction();
            return false;
        }

        const size_t snapshot_count = expected_rows - 1;
        implementation.convolution_snapshots.resize(snapshot_count);
        implementation.recurrent_snapshots.resize(snapshot_count);
        for (size_t index = 0; index < snapshot_count; ++index)
        {
            if (!create_state_storage(implementation.convolution_snapshots[index], convolution_elements,
                                      implementation.vulkan_context->blob_allocator())
                || !create_state_storage(implementation.recurrent_snapshots[index], recurrent_elements,
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
}

bool GatedDeltaState_vulkan::prepare_cpu_state(const std::vector<float>& convolution,
                                               const std::vector<float>& recurrent)
{
    Implementation& implementation = *d;
    const uint64_t expected_convolution = static_cast<uint64_t>(implementation.convolution_size)
                                          * implementation.kernel_size;
    const uint64_t head_elements = static_cast<uint64_t>(implementation.head_dimension)
                                   * implementation.value_head_dimension;
    const uint64_t expected_recurrent = saturating_multiply_u64(implementation.head_count, head_elements);
    const uint64_t max_elements = static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (implementation.initialized
        || expected_convolution == 0 || expected_convolution > max_elements
        || expected_recurrent == 0 || expected_recurrent > max_elements
        || convolution.size() != expected_convolution
        || recurrent.size() != expected_recurrent)
    {
        return false;
    }

    implementation.cpu_convolution.create(static_cast<int>(expected_convolution), 1, sizeof(float),
                                          implementation.vulkan_context->staging_allocator());
    implementation.cpu_recurrent.create(static_cast<int>(expected_recurrent), 1, sizeof(float),
                                        implementation.vulkan_context->staging_allocator());
    if (implementation.cpu_convolution.empty()
        || implementation.cpu_recurrent.empty()
        || !implementation.cpu_convolution.mapped_ptr()
        || !implementation.cpu_recurrent.mapped_ptr())
    {
        return false;
    }

    // Transpose CPU [tap][channel] layout into device [channel][tap] layout.
    float* destination = static_cast<float*>(implementation.cpu_convolution.mapped_ptr());
    const uint32_t convolution_columns = implementation.convolution_size;
    const uint32_t kernel_size = implementation.kernel_size;
    for (uint32_t channel = 0; channel < convolution_columns; ++channel)
    {
        for (uint32_t tap = 0; tap < kernel_size; ++tap)
            destination[static_cast<size_t>(channel) * kernel_size + tap] = convolution[static_cast<size_t>(tap) * convolution_columns + channel];
    }
    std::copy(recurrent.begin(), recurrent.end(), static_cast<float*>(implementation.cpu_recurrent.mapped_ptr()));
    (void)bind_host_in(implementation.cpu_convolution);
    (void)bind_host_in(implementation.cpu_recurrent);
    implementation.cpu_state_pending = true;
    return true;
}

bool GatedDeltaState_vulkan::prepare_transaction_finish(size_t committed_rows,
                                                        size_t recorded_rows) noexcept
{
    Implementation& implementation = *d;
    if (!implementation.transaction_active
        || recorded_rows != implementation.recorded_rows
        || committed_rows > recorded_rows
        || implementation.transaction_state_unknown)
    {
        return false;
    }

    bool restored = true;
    if (committed_rows < recorded_rows || implementation.transaction_requires_restore)
    {
        if (committed_rows == 0)
        {
            restored = implementation.transaction_initial_recorded
                       && implementation.restore_snapshot(implementation.initial_convolution,
                                                          implementation.initial_recurrent);
        }
        else if (committed_rows - 1 < implementation.convolution_snapshots.size())
        {
            restored = implementation.restore_snapshot(implementation.convolution_snapshots[committed_rows - 1],
                                                       implementation.recurrent_snapshots[committed_rows - 1]);
        }
        else
        {
            restored = false;
        }
    }
    return restored;
}

void GatedDeltaState_vulkan::complete_transaction() noexcept
{
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
}

bool GatedDeltaState_vulkan::download(std::vector<float>& convolution,
                                      std::vector<float>& recurrent) const
{
    const Implementation& implementation = *d;
    if (implementation.state_unknown
        || !implementation.vulkan_context
        || implementation.convolution.empty()
        || implementation.recurrent.empty())
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    const size_t convolution_count = static_cast<size_t>(implementation.convolution_size)
                                     * implementation.kernel_size;
    const size_t recurrent_count = static_cast<size_t>(implementation.head_count)
                                   * implementation.head_dimension
                                   * implementation.value_head_dimension;
    if (!prepare_staging_batch(transfer_slot.upload, 1, static_cast<uint32_t>(convolution_count),
                               transfer_slot.staging_allocator, sizeof(float))
        || !prepare_staging_batch(transfer_slot.download, 1, static_cast<uint32_t>(recurrent_count),
                                  transfer_slot.staging_allocator, sizeof(float)))
    {
        return false;
    }

    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& cmd = *transfer_slot.command;
    if (transfer_slot.command_used && cmd.reset() != 0)
        return false;
    transfer_slot.command_used = true;

    // record_clone creates its destination with blob_vkallocator even when a
    // same-shaped destination exists. Target the slot's staging allocator.
    ncnn::Option download_option = implementation.opt;
    download_option.blob_vkallocator = transfer_slot.staging_allocator;
    download_option.workspace_vkallocator = transfer_slot.staging_allocator;
    download_option.staging_vkallocator = transfer_slot.staging_allocator;
    ncnn::VkMat convolution_staging = transfer_slot.upload;
    ncnn::VkMat recurrent_staging = transfer_slot.download;
    cmd.record_clone(implementation.convolution, convolution_staging, download_option);
    cmd.record_clone(implementation.recurrent, recurrent_staging, download_option);
    if (submit_wait(cmd, implementation.vulkan_context->device()) != 0)
        return false;

    convolution_staging.allocator->invalidate(convolution_staging.data);
    recurrent_staging.allocator->invalidate(recurrent_staging.data);
    const ncnn::Mat convolution_mapped = convolution_staging.mapped();
    const ncnn::Mat recurrent_mapped = recurrent_staging.mapped();
    convolution.resize(convolution_count);
    const float* source = static_cast<const float*>(convolution_mapped.data);
    const uint32_t convolution_columns = implementation.convolution_size;
    const uint32_t kernel_size = implementation.kernel_size;
    for (uint32_t channel = 0; channel < convolution_columns; ++channel)
    {
        for (uint32_t tap = 0; tap < kernel_size; ++tap)
            convolution[static_cast<size_t>(tap) * convolution_columns + channel] = source[static_cast<size_t>(channel) * kernel_size + tap];
    }
    recurrent.assign(static_cast<const float*>(recurrent_mapped.data),
                     static_cast<const float*>(recurrent_mapped.data) + recurrent_count);
    (void)bind_host_out(convolution_staging);
    (void)bind_host_out(recurrent_staging);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_downloads;
    return true;
}

uint64_t GatedDeltaState_vulkan::allocated_bytes() const noexcept
{
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
}

std::shared_ptr<GatedDeltaNet_vulkan>
GatedDeltaNet_vulkan::create(const std::shared_ptr<Bfloat16Linear_vulkan>& fused_input,
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
                             uint32_t gate_channels,
                             float lower_bound,
                             uint32_t vulkan_device_index,
                             const VulkanRuntimePtr& vulkan_runtime,
                             uint64_t optimization_flags)
{
    return create_impl(fused_input, {}, convolution_weight, time_bias, decay_log, norm_weight,
                       output_projection, {}, head_count, kv_head_count, head_dimension,
                       value_head_dimension, convolution_kernel_size, norm_epsilon, sigmoid_gate,
                       gate_channels, lower_bound, vulkan_device_index, vulkan_runtime,
                       optimization_flags);
}

std::shared_ptr<GatedDeltaNet_vulkan>
GatedDeltaNet_vulkan::create_quantized(std::span<const std::shared_ptr<QnkLinear_vulkan>> inputs,
                                       const TensorData& convolution_weight,
                                       const TensorData& time_bias,
                                       const TensorData& decay_log,
                                       const TensorData& norm_weight,
                                       const std::shared_ptr<QnkLinear_vulkan>& output_projection,
                                       uint32_t head_count,
                                       uint32_t kv_head_count,
                                       uint32_t head_dimension,
                                       uint32_t value_head_dimension,
                                       uint32_t convolution_kernel_size,
                                       float norm_epsilon,
                                       bool sigmoid_gate,
                                       uint32_t gate_channels,
                                       float lower_bound,
                                       uint32_t vulkan_device_index,
                                       const VulkanRuntimePtr& vulkan_runtime,
                                       uint64_t optimization_flags)
{
    return create_impl({}, inputs, convolution_weight, time_bias, decay_log, norm_weight,
                       {}, output_projection, head_count, kv_head_count, head_dimension,
                       value_head_dimension, convolution_kernel_size, norm_epsilon, sigmoid_gate,
                       gate_channels, lower_bound, vulkan_device_index, vulkan_runtime,
                       optimization_flags);
}

std::shared_ptr<GatedDeltaNet_vulkan>
GatedDeltaNet_vulkan::create_impl(const std::shared_ptr<Bfloat16Linear_vulkan>& fused_input,
                                  std::span<const std::shared_ptr<QnkLinear_vulkan>> quantized_inputs,
                                  const TensorData& convolution_weight,
                                  const TensorData& time_bias,
                                  const TensorData& decay_log,
                                  const TensorData& norm_weight,
                                  const std::shared_ptr<Bfloat16Linear_vulkan>& output_projection,
                                  const std::shared_ptr<QnkLinear_vulkan>& quantized_output,
                                  uint32_t head_count,
                                  uint32_t kv_head_count,
                                  uint32_t head_dimension,
                                  uint32_t value_head_dimension,
                                  uint32_t convolution_kernel_size,
                                  float norm_epsilon,
                                  bool sigmoid_gate,
                                  uint32_t gate_channels,
                                  float lower_bound,
                                  uint32_t vulkan_device_index,
                                  const VulkanRuntimePtr& vulkan_runtime,
                                  uint64_t optimization_flags)
{
    (void)vulkan_device_index;
    (void)vulkan_runtime;
    (void)optimization_flags;

    const bool quantized = !quantized_inputs.empty();
    uint32_t convolution_size = 0;
    uint32_t value_size = 0;
    uint32_t fused_columns = 0;
    if (!gated_delta_record_columns(head_count, kv_head_count, head_dimension, value_head_dimension,
                                    gate_channels, convolution_size, value_size, fused_columns))
    {
        return {};
    }

    const std::shared_ptr<VulkanContext>& fused_context = quantized ? quantized_inputs.front()->context() : fused_input->context();
    std::shared_ptr<GatedDeltaNet_vulkan> result(new GatedDeltaNet_vulkan);
    Implementation& implementation = *result->d;
    implementation.vulkan_context = fused_context;
    implementation.fused_input = fused_input;
    implementation.output_projection = output_projection;
    if (quantized)
    {
        std::copy(quantized_inputs.begin(), quantized_inputs.end(), implementation.quantized_inputs.begin());
        implementation.quantized_output = quantized_output;
        for (const auto& projection : quantized_inputs)
            implementation.int4_ops += projection->dtype() == DType::Int4 ? 1 : 0;
        implementation.int4_ops += quantized_output->dtype() == DType::Int4 ? 1 : 0;
    }
    implementation.opt = quantized ? quantized_inputs.front()->option() : fused_input->option();
    implementation.input_columns = quantized ? quantized_inputs.front()->input_columns() : fused_input->input_columns();
    implementation.output_columns = quantized ? quantized_output->output_columns() : output_projection->output_columns();
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
    implementation.gate_channels = gate_channels;
    implementation.lower_bound = lower_bound;

    // Shader ABI constants; only token count and conv_stride change per recording.
    implementation.constants.resize(14);
    implementation.constants[0].u32 = implementation.convolution_size;
    implementation.constants[1].u32 = implementation.value_size;
    implementation.constants[2].u32 = implementation.fused_columns;
    implementation.constants[3].u32 = implementation.head_count;
    implementation.constants[4].u32 = implementation.kv_head_count;
    implementation.constants[5].u32 = implementation.head_dimension;
    implementation.constants[6].u32 = implementation.value_head_dimension;
    implementation.constants[7].u32 = implementation.convolution_kernel_size;
    implementation.constants[8].u32 = 1;
    implementation.constants[9].f = implementation.norm_epsilon;
    implementation.constants[10].u32 = implementation.sigmoid_gate ? 1u : 0u;
    implementation.constants[11].u32 = implementation.gate_channels;
    implementation.constants[12].f = implementation.lower_bound;
    implementation.constants[13].u32 = implementation.fused_columns;

    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        struct PipelineSpec
        {
            const char* src;
            int len;
            std::shared_ptr<ncnn::Pipeline>* dst;
        };
        const PipelineSpec required[] = {
            {gated_delta_conv_shader, static_cast<int>(sizeof(gated_delta_conv_shader) - 1), &implementation.conv},
            {gated_delta_conv_parallel_shader, static_cast<int>(sizeof(gated_delta_conv_parallel_shader) - 1), &implementation.conv_par},
            {gated_delta_conv_history_shader, static_cast<int>(sizeof(gated_delta_conv_history_shader) - 1), &implementation.conv_hist},
            {gated_delta_qk_shader, static_cast<int>(sizeof(gated_delta_qk_shader) - 1), &implementation.qk},
            {gated_delta_rule_shader, static_cast<int>(sizeof(gated_delta_rule_shader) - 1), &implementation.rule},
            {gated_delta_norm_shader, static_cast<int>(sizeof(gated_delta_norm_shader) - 1), &implementation.norm},
        };
        for (const PipelineSpec& entry : required)
        {
            if (!create_pipeline(implementation.vulkan_context, entry.src, entry.len,
                                 implementation.opt, 0, 0, 128, *entry.dst))
            {
                return {};
            }
        }

        const auto& dev_info = implementation.vulkan_context->device()->info;
        // Register-resident rule kernel requires 32-lane subgroup, KEY_ROWS=4,
        // and gate indexed by scalar head or per-key-row.
        if (dev_info.subgroup_size() == 32
            && head_dimension == 128u
            && value_head_dimension % 4u == 0
            && (gate_channels == 1u || gate_channels == head_dimension))
        {
            if (!create_pipeline(implementation.vulkan_context, gated_delta_rule_reg_shader,
                                 static_cast<int>(sizeof(gated_delta_rule_reg_shader) - 1),
                                 implementation.opt, 0, 0, 128, implementation.rule_reg))
            {
                return {};
            }
        }

        const uint32_t fast_required_ops = VK_SUBGROUP_FEATURE_BASIC_BIT
                                           | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT
                                           | VK_SUBGROUP_FEATURE_BALLOT_BIT
                                           | VK_SUBGROUP_FEATURE_SHUFFLE_BIT
                                           | VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
        if (dev_info.subgroup_size() == 32
            && (dev_info.support_subgroup_ops() & fast_required_ops) == fast_required_ops
            && head_dimension == 128u
            && value_head_dimension % 8u == 0
            && (convolution_size % 4u) == 0
            && (fused_columns % 4u) == 0
            && gate_channels == 1u)
        {
            ncnn::Option fast_opt = implementation.opt;
            fast_opt.use_subgroup_ops = true;
            std::vector<ncnn::vk_specialization_type> spec(1);
            spec[0].u32 = 4u;
            if (!create_pipeline(implementation.vulkan_context, gated_delta_rule_fast_shader,
                                 static_cast<int>(sizeof(gated_delta_rule_fast_shader) - 1),
                                 fast_opt, 0, 4u, 32, implementation.rule_dec, spec, 32))
            {
                implementation.rule_dec.reset();
            }
            spec[0].u32 = 16u;
            if (!create_pipeline(implementation.vulkan_context, gated_delta_rule_fast_shader,
                                 static_cast<int>(sizeof(gated_delta_rule_fast_shader) - 1),
                                 fast_opt, 0, 16u, 32, implementation.rule_pref, spec, 32))
            {
                implementation.rule_pref.reset();
            }
        }

        if (!quantized && output_projection && output_projection->use_cm2()
            && (value_head_dimension & 1u) == 0u)
        {
            if (!create_pipeline_prefixed(implementation.vulkan_context, gated_delta_norm_shader,
                                          static_cast<int>(sizeof(gated_delta_norm_shader) - 1),
                                          "#define NCNN_GDN_NORM_BF16_OUT\n",
                                          implementation.opt, 1, 1, 128, implementation.norm_bf16))
            {
                implementation.norm_bf16.reset();
            }
        }
    }

    implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(implementation.vulkan_context->device(),
                                                                      static_cast<size_t>(vulkan_weight_block_size)));
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(implementation.vulkan_context->device()));

    ncnn::Mat convolution_model;
    ncnn::Mat time_model;
    ncnn::Mat decay_model;
    ncnn::Mat norm_model;
    if (!prepare_float_upload(convolution_weight, convolution_model)
        || !prepare_float_upload(time_bias, time_model)
        || !prepare_float_upload(decay_log, decay_model)
        || !prepare_float_upload(norm_weight, norm_model))
    {
        return {};
    }

    ncnn::Option upload_option = implementation.opt;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer cmd(implementation.vulkan_context->device());
        cmd.record_upload(convolution_model, implementation.convolution_weight, upload_option);
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
    rearm_read(implementation.convolution_weight);
    rearm_read(implementation.time_bias);
    rearm_read(implementation.decay_log);
    rearm_read(implementation.norm_weight);
    implementation.weight_staging_allocator.reset();
    return result;
}

void GatedDeltaNet_vulkan::record_input_projection(const ncnn::VkMat& input, ncnn::VkMat& output,
                                                   ncnn::VkCompute& command, bool normalize,
                                                   ncnn::VkMat* row_scale_workspace) const
{
    const Implementation& implementation = *d;
    if (implementation.quantized_output)
    {
        std::vector<ncnn::VkMat> bindings;
        bindings.reserve(4);
        std::vector<ncnn::vk_constant_type> constants;
        constants.reserve(15);
        uint32_t column_offset = 0;
        for (const auto& projection : implementation.quantized_inputs)
        {
            projection->record(input, output, command, column_offset, bindings, constants);
            column_offset += projection->output_columns();
        }
    }
    else if (normalize)
    {
        // Cooperative-matrix prefill materializes the norm as its own pass; the
        // fused shader stays for decode-quality fallback shapes.
        implementation.fused_input->record_norm_projection(input, output, command, row_scale_workspace);
    }
    else
    {
        implementation.fused_input->record_scalar_projection(input, output, command);
    }
}

void GatedDeltaNet_vulkan::record_output_projection(const ncnn::VkMat& input, ncnn::VkMat& output,
                                                    ncnn::VkCompute& command) const
{
    if (d->quantized_output)
        d->quantized_output->record(input, output, command);
    else
        d->output_projection->record_scalar_projection(input, output, command);
}

const std::shared_ptr<VulkanContext>& GatedDeltaNet_vulkan::context() const noexcept
{
    return d->vulkan_context;
}

const ncnn::Option& GatedDeltaNet_vulkan::option() const noexcept
{
    return d->opt;
}

bool GatedDeltaNet_vulkan::record_forward(const ncnn::VkMat& input, LayerCache& cache,
                                          size_t token_count, ncnn::VkMat& projected,
                                          ncnn::VkCompute& command, GatedDeltaWork_vulkan& work,
                                          bool apply_input_rms_norm) const
{
    const Implementation& implementation = *d;
    auto& pending = *work.d;
    pending.cache = &cache;
    pending.rows = token_count;
    pending.previous_state_allocated_size = cache.gated_delta_device_state ? cache.gated_delta_device_state->allocated_bytes() : 0;
    if (!cache.gated_delta_device_state)
    {
        cache.gated_delta_device_state = GatedDeltaState_vulkan::create(implementation.vulkan_context,
                                                                        implementation.convolution_size,
                                                                        implementation.convolution_kernel_size,
                                                                        implementation.head_count,
                                                                        implementation.head_dimension,
                                                                        implementation.value_head_dimension,
                                                                        implementation.opt);
        if (!cache.gated_delta_device_state)
            return false;
        pending.created_state = true;
    }

    pending.state = cache.gated_delta_device_state;
    auto& state = *pending.state->d;
    if (state.state_unknown || state.recording_pending || state.vulkan_context != implementation.vulkan_context
        || (state.transaction_active && !cache.transaction.active)
        || state.transaction_initial_pending || state.transaction_row_pending)
    {
        return false;
    }

    pending.owns_recording = true;
    state.recording_pending = true;
    if (pending.created_state
        && (!cache.gated_delta_convolution.empty() || !cache.gated_delta_recurrent.empty())
        && !pending.state->prepare_cpu_state(cache.gated_delta_convolution, cache.gated_delta_recurrent))
    {
        return false;
    }
    if (cache.transaction.active && !state.transaction_active
        && !pending.state->begin_transaction(cache.transaction.expected_rows))
    {
        return false;
    }

    pending.fused.create(static_cast<int>(implementation.fused_columns), static_cast<int>(token_count),
                         sizeof(float), implementation.vulkan_context->blob_allocator());
    pending.recurrent_output.create(static_cast<int>(implementation.value_size), static_cast<int>(token_count),
                                    sizeof(float), implementation.vulkan_context->blob_allocator());
    pending.projected.create(static_cast<int>(implementation.output_columns), static_cast<int>(token_count),
                             sizeof(float), implementation.vulkan_context->blob_allocator());

    // Prefill spreads conv over (channel, token); sequential kernel stays for decode/short chunks.
    const bool par_conv = implementation.conv_par && implementation.conv_hist && token_count >= 8
                          && token_count >= implementation.convolution_kernel_size;
    const bool compact_conv = par_conv && implementation.rule_dec != nullptr;
    const uint32_t conv_stride = compact_conv ? implementation.convolution_size : implementation.fused_columns;
    if (par_conv)
    {
        pending.conv_out.create(static_cast<int>(conv_stride), static_cast<int>(token_count),
                                sizeof(float), implementation.vulkan_context->blob_allocator());
    }
    if (pending.fused.empty() || pending.recurrent_output.empty() || pending.projected.empty()
        || (par_conv && pending.conv_out.empty()))
    {
        return false;
    }

    pending.input = input;
    if (!state.record_initial_state(command))
        return false;
    record_input_projection(input, pending.fused, command, apply_input_rms_norm && has_input_rms_norm(),
                            &pending.row_scale);

    std::vector<ncnn::vk_constant_type> constants = implementation.constants;
    constants[8].u32 = static_cast<uint32_t>(token_count);
    constants[13].u32 = conv_stride;
    if (par_conv)
    {
        record_dispatch(command, implementation.conv_par,
                        {pending.fused, pending.conv_out, state.convolution, implementation.convolution_weight},
                        {1, 0, 1, 1}, constants, (conv_stride + 127) / 128,
                        static_cast<uint32_t>(token_count));
        record_dispatch(command, implementation.conv_hist, {pending.fused, state.convolution}, {1, 0}, constants,
                        (implementation.convolution_size + 127) / 128);
    }
    else
    {
        record_dispatch(command, implementation.conv,
                        {pending.fused, state.convolution, implementation.convolution_weight}, {0, 0, 1}, constants,
                        (implementation.convolution_size + 127) / 128);
    }

    const ncnn::VkMat& conv = par_conv ? pending.conv_out : pending.fused;
    record_dispatch(command, implementation.qk, {conv}, {0}, constants, implementation.kv_head_count,
                    static_cast<uint32_t>(token_count));
    if (token_count >= 16u && implementation.rule_pref)
    {
        record_dispatch(command, implementation.rule_pref,
                        {conv, state.recurrent, implementation.time_bias, implementation.decay_log,
                         pending.recurrent_output, pending.fused},
                        {1, 0, 1, 1, 0, 1}, constants,
                        implementation.head_count * (implementation.value_head_dimension / 2u));
    }
    else if (implementation.rule_dec)
    {
        record_dispatch(command, implementation.rule_dec,
                        {conv, state.recurrent, implementation.time_bias, implementation.decay_log,
                         pending.recurrent_output, pending.fused},
                        {1, 0, 1, 1, 0, 1}, constants,
                        implementation.head_count * (implementation.value_head_dimension / 8u));
    }
    else if (implementation.rule_reg)
    {
        record_dispatch(command, implementation.rule_reg,
                        {conv, state.recurrent, implementation.time_bias, implementation.decay_log,
                         pending.recurrent_output},
                        {1, 0, 1, 1, 0}, constants,
                        implementation.head_count * (implementation.value_head_dimension / 4));
    }
    else
    {
        record_dispatch(command, implementation.rule,
                        {conv, state.recurrent, implementation.time_bias, implementation.decay_log,
                         pending.recurrent_output},
                        {1, 0, 1, 1, 0}, constants,
                        implementation.head_count * ((implementation.value_head_dimension + 31) / 32));
    }

    const bool use_norm_bf16 = token_count >= 8 && implementation.norm_bf16 != nullptr;
    if (use_norm_bf16)
    {
        pending.rec_bf16.create(static_cast<int>(implementation.value_size),
                                static_cast<int>(token_count),
                                sizeof(uint16_t), 1,
                                implementation.vulkan_context->blob_allocator());
        if (pending.rec_bf16.empty())
            return false;
        record_dispatch(command, implementation.norm_bf16,
                        {pending.fused, implementation.norm_weight, pending.recurrent_output, pending.rec_bf16},
                        {1, 1, 1, 0}, constants, implementation.head_count,
                        static_cast<uint32_t>(token_count));
    }
    else
    {
        record_dispatch(command, implementation.norm,
                        {pending.fused, implementation.norm_weight, pending.recurrent_output},
                        {1, 1, 0}, constants, implementation.head_count,
                        static_cast<uint32_t>(token_count));
    }

    if (!state.record_transaction_row(command))
        return false;
    record_output_projection(use_norm_bf16 ? pending.rec_bf16 : pending.recurrent_output,
                             pending.projected, command);
    projected = pending.projected;
    pending.dispatches = (implementation.quantized_output ? 4u : 1u) + 5;
    pending.int4_ops = implementation.int4_ops;
    pending.recorded = true;
    return true;
}

bool GatedDeltaNet_vulkan::forward_impl(const ActivationBuffer& input,
                                        LayerCache& cache,
                                        ActivationBuffer& projected,
                                        bool apply_input_rms_norm) const
{
    const Implementation& implementation = *d;
    VkTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = activation_variant(implementation.opt) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(input.rows() * input.columns() * sizeof(float), input.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns,
                                  transfer_slot.staging_allocator))
    {
        return false;
    }

    const DType projected_dtype = projected.dtype();
    projected.reset(input.rows(), implementation.output_columns, false);
    GatedDeltaWork_vulkan work;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkCompute& command = *transfer_slot.command;
        if (transfer_slot.command_used && command.reset() != 0)
            return false;
        transfer_slot.command_used = true;

        ncnn::VkMat input_gpu;
        if (direct_host_input)
            input_gpu = bind_host_in(transfer_slot.upload);
        else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.opt))
            return false;

        ncnn::VkMat output_gpu;
        if (!record_forward(input_gpu, cache, input.rows(), output_gpu, command, work, apply_input_rms_norm)
            || !record_staging_download(output_gpu, input.rows(), implementation.output_columns,
                                        transfer_slot.download, command, implementation.vulkan_context->device(),
                                        implementation.opt, projected_dtype))
        {
            return false;
        }
        if (submit_wait(command, implementation.vulkan_context->device()) != 0)
        {
            work.mark_submit_failed();
            return false;
        }
    }

    if (!work.commit())
        return false;
    if (!copy_staging_to_cpu_batch(transfer_slot.download, projected))
        return false;

    VulkanRuntimeState& runtime = implementation.vulkan_context->runtime_state();
    ++runtime.compute_submissions;
    ++runtime.batch_uploads;
    ++runtime.batch_downloads;
    return true;
}

bool GatedDeltaNet_vulkan::forward(const ActivationBuffer& normalized,
                                   LayerCache& cache,
                                   ActivationBuffer& projected) const
{
    return forward_impl(normalized, cache, projected, false);
}

bool GatedDeltaNet_vulkan::forward_input_rms_norm(const ActivationBuffer& input,
                                                  LayerCache& cache,
                                                  ActivationBuffer& projected) const
{
    if (!has_input_rms_norm())
        return false;
    return forward_impl(input, cache, projected, true);
}

bool GatedDeltaNet_vulkan::has_input_rms_norm() const noexcept
{
    return d && d->fused_input && d->fused_input->has_rms_norm_chain();
}

GatedDeltaBatchResult_vulkan
GatedDeltaNet_vulkan::forward_batch(std::span<const GatedDeltaBatchEntry_vulkan> entries) const
{
    const Implementation& implementation = *d;
    if (entries.empty())
        return GatedDeltaBatchResult_vulkan::Executed;
    if (entries.size() == 1)
    {
        const GatedDeltaBatchEntry_vulkan& entry = entries.front();
        if (forward(*entry.normalized, *entry.cache, *entry.projected))
            return GatedDeltaBatchResult_vulkan::Executed;
        return GatedDeltaBatchResult_vulkan::Failed;
    }

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    const uint32_t input_columns = implementation.input_columns;
    const uint32_t output_columns = implementation.output_columns;
    const size_t total_rows = entries.size();

    ActivationBuffer combined_normalized;
    combined_normalized.reset(total_rows, input_columns, false);
    std::vector<std::shared_ptr<GatedDeltaState_vulkan>> states;
    states.reserve(entries.size());
    std::vector<bool> created_states;
    created_states.reserve(entries.size());
    std::vector<LayerCache*> created_state_caches;
    created_state_caches.reserve(entries.size());

    struct BatchDeviceStateCreationAttempt
    {
        explicit BatchDeviceStateCreationAttempt(std::vector<LayerCache*>& _created_caches)
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
        std::copy_n(entry.normalized->row(0), input_columns, combined_normalized.row(entry_index));

        LayerCache& cache = *entry.cache;
        if (cache.transaction.active && !cache.gated_delta_device_state)
            return GatedDeltaBatchResult_vulkan::NotExecuted;

        bool created_device_state = false;
        if (!cache.gated_delta_device_state)
        {
            const bool has_cpu_state = !cache.gated_delta_convolution.empty()
                                       || !cache.gated_delta_recurrent.empty();
            cache.gated_delta_device_state = GatedDeltaState_vulkan::create(implementation.vulkan_context,
                                                                            implementation.convolution_size,
                                                                            implementation.convolution_kernel_size,
                                                                            implementation.head_count,
                                                                            implementation.head_dimension,
                                                                            implementation.value_head_dimension,
                                                                            implementation.opt);
            if (!cache.gated_delta_device_state)
                return GatedDeltaBatchResult_vulkan::Failed;
            created_device_state = true;
            if (has_cpu_state
                && !cache.gated_delta_device_state->prepare_cpu_state(cache.gated_delta_convolution,
                                                                      cache.gated_delta_recurrent))
            {
                cache.gated_delta_device_state.reset();
                return GatedDeltaBatchResult_vulkan::Failed;
            }
            created_state_caches.push_back(&cache);
        }

        std::shared_ptr<GatedDeltaState_vulkan> state = cache.gated_delta_device_state;
        GatedDeltaState_vulkan::Implementation& state_implementation = *state->d;
        if (state_implementation.state_unknown || state_implementation.recording_pending)
            return GatedDeltaBatchResult_vulkan::Failed;
        if (state_implementation.vulkan_context != implementation.vulkan_context)
            return GatedDeltaBatchResult_vulkan::NotExecuted;
        if (cache.transaction.active && !state_implementation.transaction_active)
        {
            if (!state->begin_transaction(cache.transaction.expected_rows))
                return GatedDeltaBatchResult_vulkan::Failed;
        }
        cache.device_allocated_size = state->allocated_bytes();
        states.push_back(std::move(state));
        created_states.push_back(created_device_state);
    }

    const auto mark_states_unknown = [&]() noexcept {
        for (size_t index = 0; index < states.size(); ++index)
        {
            if (!created_states[index] && !states[index]->d->transaction_active)
                states[index]->d->state_unknown = true;
        }
    };

    VkTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = activation_variant(implementation.opt) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(total_rows
                                                                                                    * input_columns
                                                                                                    * sizeof(float),
                                                                                                combined_normalized.dtype());
    if (!fill_staging_upload(combined_normalized, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, total_rows, output_columns, transfer_slot.staging_allocator))
    {
        return GatedDeltaBatchResult_vulkan::Failed;
    }

    ActivationBuffer combined_projected;
    combined_projected.reset(total_rows, output_columns, false);
    for (const GatedDeltaBatchEntry_vulkan& entry : entries)
        entry.projected->reset(1, output_columns, false);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& cmd = *transfer_slot.command;
    if (transfer_slot.command_used && cmd.reset() != 0)
        return GatedDeltaBatchResult_vulkan::Failed;
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_host_in(transfer_slot.upload);
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, cmd, implementation.opt))
        return GatedDeltaBatchResult_vulkan::Failed;

    ncnn::VkMat fused_gpu;
    fused_gpu.create(static_cast<int>(implementation.fused_columns), static_cast<int>(total_rows),
                     sizeof(float), implementation.vulkan_context->blob_allocator());
    ncnn::VkMat recurrent_output_gpu;
    recurrent_output_gpu.create(static_cast<int>(implementation.value_size), static_cast<int>(total_rows),
                                sizeof(float), implementation.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    output_gpu.create(static_cast<int>(output_columns), static_cast<int>(total_rows),
                      sizeof(float), implementation.vulkan_context->blob_allocator());
    if (fused_gpu.empty() || recurrent_output_gpu.empty() || output_gpu.empty())
        return GatedDeltaBatchResult_vulkan::Failed;

    std::vector<GatedDeltaState_vulkan::Implementation*> transaction_states;
    transaction_states.reserve(states.size());
    for (const std::shared_ptr<GatedDeltaState_vulkan>& state : states)
        transaction_states.push_back(state->d.get());
    GatedDeltaState_vulkan::Implementation::RecordingGuard transaction_recording(transaction_states);

    for (const std::shared_ptr<GatedDeltaState_vulkan>& state : states)
    {
        if (!state->d->record_initial_state(cmd))
            return GatedDeltaBatchResult_vulkan::Failed;
    }

    record_input_projection(input_gpu, fused_gpu, cmd, false);

    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        GatedDeltaState_vulkan::Implementation& state_implementation = *states[entry_index]->d;
        ncnn::VkMat fused_view = row_view(fused_gpu, entry_index, 1);
        ncnn::VkMat recurrent_view = row_view(recurrent_output_gpu, entry_index, 1);

        // One row per entry, so the cached token count of one applies.
        record_dispatch(cmd, implementation.conv,
                        {fused_view, state_implementation.convolution, implementation.convolution_weight},
                        {0, 0, 1}, implementation.constants,
                        (implementation.convolution_size + 127) / 128);
        record_dispatch(cmd, implementation.qk, {fused_view}, {0}, implementation.constants,
                        implementation.kv_head_count);
        record_dispatch(cmd, implementation.rule,
                        {fused_view, state_implementation.recurrent, implementation.time_bias,
                         implementation.decay_log, recurrent_view},
                        {1, 0, 1, 1, 0}, implementation.constants,
                        implementation.head_count * ((implementation.value_head_dimension + 31) / 32));
        record_dispatch(cmd, implementation.norm,
                        {fused_view, implementation.norm_weight, recurrent_view},
                        {1, 1, 0}, implementation.constants,
                        implementation.head_count);
        if (!state_implementation.record_transaction_row(cmd))
            return GatedDeltaBatchResult_vulkan::Failed;
    }

    record_output_projection(recurrent_output_gpu, output_gpu, cmd);

    if (!record_staging_download(output_gpu, total_rows, output_columns, transfer_slot.download, cmd,
                                 implementation.vulkan_context->device(), implementation.opt,
                                 combined_projected.dtype()))
    {
        return GatedDeltaBatchResult_vulkan::Failed;
    }
    if (submit_wait(cmd, implementation.vulkan_context->device()) != 0)
    {
        transaction_recording.mark_submit_failed();
        mark_states_unknown();
        return GatedDeltaBatchResult_vulkan::Failed;
    }

    transaction_recording.mark_submitted();
    if (!copy_staging_to_cpu_batch(transfer_slot.download, combined_projected))
    {
        mark_states_unknown();
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
        std::copy_n(combined_projected.row(entry_index), output_columns, entries[entry_index].projected->row(0));
        cache.gated_delta_convolution.clear();
        cache.gated_delta_recurrent.clear();
        cache.gated_delta_token_count += 1;
        cache.device_allocated_size = states[entry_index]->allocated_bytes();
    }
    runtime_state.dispatches += entries.size() * 4 + (implementation.quantized_output ? 4u : 1u) + 1;
    if (implementation.int4_ops != 0)
        runtime_state.int4_dispatches += implementation.int4_ops;
    runtime_state.gated_delta_fusions += entries.size();
    ++runtime_state.gated_delta_submissions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    creation_attempt.complete();
    return GatedDeltaBatchResult_vulkan::Executed;
}

} // namespace moe
} // namespace ncnn
