#include "cpu_state_cache.h"

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
    const CpuStateCacheSnapshot& snapshot)
{
    cache.gated_delta_convolution = snapshot.gated_delta_convolution;
    cache.gated_delta_recurrent = snapshot.gated_delta_recurrent;
}

void begin_state_cache_transaction(
    std::span<CpuLayerCache> caches,
    size_t expected_rows)
{
    for (CpuLayerCache& cache : caches)
    {
        CpuStateCacheTransaction& transaction = cache.state_transaction;
        transaction.initial_start_position = cache.start_position;
        transaction.initial_token_count = cache.token_count;
        transaction.initial_first_slot = cache.first_slot;
        transaction.initial_gated_delta_token_count =
            cache.gated_delta_token_count;
        transaction.expected_rows = expected_rows;
        transaction.recorded_rows = 0;
        transaction.active = true;
        if (!cache.gated_delta_convolution.empty()
            || !cache.gated_delta_recurrent.empty())
        {
            capture_gated_delta_snapshot(cache, transaction.initial);
            transaction.rows.resize(
                expected_rows > 0 ? expected_rows - 1 : 0);
        }
        else
        {
            transaction.initial.gated_delta_convolution.clear();
            transaction.initial.gated_delta_recurrent.clear();
        }
    }
}

void record_standard_cache_transaction_rows(
    CpuLayerCache& cache,
    size_t rows)
{
    CpuStateCacheTransaction& transaction = cache.state_transaction;
    if (!transaction.active)
        return;
    transaction.recorded_rows += rows;
}

void record_gated_delta_cache_transaction_row(
    CpuLayerCache& cache)
{
    CpuStateCacheTransaction& transaction = cache.state_transaction;
    if (!transaction.active)
        return;
    ++transaction.recorded_rows;
    if (transaction.recorded_rows < transaction.expected_rows)
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
        const CpuStateCacheTransaction& transaction =
            cache.state_transaction;
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
        if (cache.gated_delta_convolution.empty()
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
        CpuStateCacheTransaction& transaction = cache.state_transaction;
        if (!transaction.active)
            continue;

        const bool gated_delta =
            !transaction.initial.gated_delta_convolution.empty()
            || !transaction.initial.gated_delta_recurrent.empty()
            || !cache.gated_delta_convolution.empty()
            || !cache.gated_delta_recurrent.empty();
        if (gated_delta)
        {
            if (committed_rows == 0)
            {
                restore_gated_delta_snapshot(cache, transaction.initial);
            }
            else if (committed_rows < transaction.recorded_rows)
            {
                restore_gated_delta_snapshot(
                    cache,
                    transaction.rows[committed_rows - 1]);
            }
            cache.gated_delta_token_count =
                transaction.initial_gated_delta_token_count
                + committed_rows;
        }
        else
        {
            cache.start_position = transaction.initial_start_position;
            cache.first_slot = transaction.initial_first_slot;
            cache.token_count =
                transaction.initial_token_count + committed_rows;
        }
        transaction.active = false;
        transaction.expected_rows = 0;
        transaction.recorded_rows = 0;
    }
    return {};
}

} // namespace moe
} // namespace ncnn
