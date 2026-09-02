#include "cpu_state_cache.h"

#include "backends/ncnn/ncnn_linear.h"
#include "engine/cpu_session_state.h"

namespace ncnn {
namespace moe {

static void capture_gated_delta_snapshot(
    const CpuLayerCache& cache,
    CpuStateCacheSnapshot& snapshot)
{
    snapshot.gated_delta_convolution = cache.gated_delta_convolution;
    snapshot.gated_delta_recurrent = cache.gated_delta_recurrent;
}

static void restore_gated_delta_snapshot(
    CpuLayerCache& cache,
    CpuStateCacheSnapshot& snapshot) noexcept
{
    cache.gated_delta_convolution.swap(
        snapshot.gated_delta_convolution);
    cache.gated_delta_recurrent.swap(
        snapshot.gated_delta_recurrent);
}

Result<void> begin_state_cache_transaction(
    std::span<CpuLayerCache> caches,
    size_t expected_rows)
{
    for (const CpuLayerCache& cache : caches)
    {
        if (cache.transaction.active)
        {
            return Error{
                ErrorCode::InternalError,
                "state cache transaction is already active"};
        }
        if (!cache.gated_delta_device_state
            && cache.gated_delta_convolution.empty()
            && cache.gated_delta_recurrent.empty()
            && cache.first_slot != 0)
        {
            return Error{
                ErrorCode::UnsupportedModel,
                "state cache transaction does not support a sliding KV ring"};
        }
        if (cache.vulkan_attention_cache && cache.token_count == 0)
        {
            return Error{
                ErrorCode::InternalError,
                "Vulkan KV cache is authoritative for an empty state cache"};
        }
    }

    std::vector<NcnnVulkanGatedDeltaState*> device_transactions;
    try
    {
        device_transactions.reserve(caches.size());
    }
    catch (...)
    {
        return Error{
            ErrorCode::InternalError,
            "failed to reserve state cache transaction metadata"};
    }
    for (CpuLayerCache& cache : caches)
    {
        if (cache.gated_delta_device_state)
        {
            if (!cache.gated_delta_device_state->begin_transaction(
                    expected_rows))
            {
                for (NcnnVulkanGatedDeltaState* state : device_transactions)
                {
                    state->complete_transaction();
                }
                for (CpuLayerCache& cleanup_cache : caches)
                {
                    if (cleanup_cache.gated_delta_device_state)
                    {
                        cleanup_cache.device_allocated_size = cleanup_cache.gated_delta_device_state->allocated_bytes();
                    }
                }
                return Error{
                    ErrorCode::InternalError,
                    "failed to begin Vulkan Gated DeltaNet transaction"};
            }
            device_transactions.push_back(
                cache.gated_delta_device_state.get());
            cache.device_allocated_size = cache.gated_delta_device_state->allocated_bytes();
        }
    }

    try
    {
        for (CpuLayerCache& cache : caches)
        {
            CpuSessionStateTransaction& transaction = cache.transaction;
            transaction.initial_start_position = cache.start_position;
            transaction.initial_token_count = cache.token_count;
            transaction.initial_first_slot = cache.first_slot;
            transaction.initial_gated_delta_token_count = cache.gated_delta_token_count;
            transaction.expected_rows = expected_rows;
            transaction.recorded_rows = 0;
            transaction.initial.gated_delta_convolution.clear();
            transaction.initial.gated_delta_recurrent.clear();
            transaction.rows.clear();
            if (!cache.gated_delta_convolution.empty()
                || !cache.gated_delta_recurrent.empty())
            {
                capture_gated_delta_snapshot(
                    cache,
                    transaction.initial);
                transaction.rows.resize(
                    expected_rows > 0 ? expected_rows - 1 : 0);
            }
            transaction.active = true;
        }
    }
    catch (...)
    {
        for (CpuLayerCache& cache : caches)
        {
            CpuSessionStateTransaction& transaction = cache.transaction;
            transaction.active = false;
            transaction.expected_rows = 0;
            transaction.recorded_rows = 0;
            transaction.initial.gated_delta_convolution.clear();
            transaction.initial.gated_delta_recurrent.clear();
            transaction.rows.clear();
        }
        for (NcnnVulkanGatedDeltaState* state : device_transactions)
            state->complete_transaction();
        for (CpuLayerCache& cleanup_cache : caches)
        {
            if (cleanup_cache.gated_delta_device_state)
            {
                cleanup_cache.device_allocated_size = cleanup_cache.gated_delta_device_state->allocated_bytes();
            }
        }
        return Error{
            ErrorCode::InternalError,
            "failed to allocate state cache transaction snapshots"};
    }
    return {};
}

void record_standard_cache_transaction_rows(
    CpuLayerCache& cache,
    size_t rows)
{
    CpuSessionStateTransaction& transaction = cache.transaction;
    if (!transaction.active)
        return;
    transaction.recorded_rows += rows;
}

void record_gated_delta_cache_transaction_row(
    CpuLayerCache& cache)
{
    CpuSessionStateTransaction& transaction = cache.transaction;
    if (!transaction.active)
        return;
    ++transaction.recorded_rows;
    if (transaction.recorded_rows < transaction.expected_rows
        && !cache.gated_delta_device_state)
    {
        const size_t index = transaction.recorded_rows - 1;
        if (transaction.rows.size() <= index)
            transaction.rows.resize(index + 1);
        capture_gated_delta_snapshot(cache, transaction.rows[index]);
    }
}

Result<void> finish_state_cache_transaction(
    std::span<CpuLayerCache> caches,
    size_t committed_rows)
{
    for (const CpuLayerCache& cache : caches)
    {
        const CpuSessionStateTransaction& transaction = cache.transaction;
        if (!transaction.active)
            continue;
        if (committed_rows > transaction.recorded_rows
            || (committed_rows != 0
                && transaction.recorded_rows
                       != transaction.expected_rows))
        {
            return Error{
                ErrorCode::InternalError,
                "state cache transaction row count is inconsistent"};
        }
        if ((!cache.gated_delta_convolution.empty()
             || !cache.gated_delta_recurrent.empty())
            && committed_rows != transaction.recorded_rows
            && committed_rows > 0
            && transaction.rows.size() < committed_rows)
        {
            return Error{
                ErrorCode::InternalError,
                "state cache transaction is missing a committed row"};
        }
        if (!cache.gated_delta_device_state
            && cache.gated_delta_convolution.empty()
            && cache.gated_delta_recurrent.empty()
            && (cache.first_slot != 0
                || transaction.initial_first_slot != 0))
        {
            return Error{
                ErrorCode::UnsupportedModel,
                "state cache transaction does not support a sliding KV ring"};
        }
    }

    for (CpuLayerCache& cache : caches)
    {
        CpuSessionStateTransaction& transaction = cache.transaction;
        if (!transaction.active || !cache.gated_delta_device_state)
            continue;
        if (!cache.gated_delta_device_state->prepare_transaction_finish(
                committed_rows,
                transaction.recorded_rows))
        {
            return Error{
                ErrorCode::InternalError,
                "failed to restore Vulkan Gated DeltaNet transaction"};
        }
    }

    for (CpuLayerCache& cache : caches)
    {
        if (cache.transaction.active
            && cache.gated_delta_device_state)
        {
            cache.gated_delta_device_state->complete_transaction();
        }
    }

    for (CpuLayerCache& cache : caches)
    {
        CpuSessionStateTransaction& transaction = cache.transaction;
        if (!transaction.active)
            continue;

        const bool gated_delta = cache.gated_delta_device_state
                                 || !transaction.initial.gated_delta_convolution.empty()
                                 || !transaction.initial.gated_delta_recurrent.empty()
                                 || !cache.gated_delta_convolution.empty()
                                 || !cache.gated_delta_recurrent.empty();
        if (gated_delta)
        {
            if (committed_rows == 0)
            {
                if (!cache.gated_delta_device_state)
                    restore_gated_delta_snapshot(cache, transaction.initial);
            }
            else if (committed_rows < transaction.recorded_rows)
            {
                if (!cache.gated_delta_device_state)
                {
                    restore_gated_delta_snapshot(
                        cache,
                        transaction.rows[committed_rows - 1]);
                }
            }
            cache.gated_delta_token_count = transaction.initial_gated_delta_token_count
                                            + committed_rows;
            if (cache.gated_delta_device_state)
            {
                cache.device_allocated_size = cache.gated_delta_device_state->allocated_bytes();
            }
        }
        else
        {
            cache.start_position = transaction.initial_start_position;
            cache.first_slot = transaction.initial_first_slot;
            cache.token_count = transaction.initial_token_count + committed_rows;
            if (transaction.initial_token_count == 0
                && committed_rows == 0
                && cache.vulkan_attention_cache)
            {
                cache.vulkan_attention_cache.reset();
                cache.capacity_tokens = 0;
                cache.device_allocated_size = 0;
                cache.vulkan_attention_state_unknown = false;
            }
        }
        transaction.active = false;
        transaction.expected_rows = 0;
        transaction.recorded_rows = 0;
        transaction.initial.gated_delta_convolution.clear();
        transaction.initial.gated_delta_recurrent.clear();
        transaction.rows.clear();
    }
    return {};
}

} // namespace moe
} // namespace ncnn
