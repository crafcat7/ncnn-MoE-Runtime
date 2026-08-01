#ifndef NCNN_MOE_CPU_STATE_CACHE_H
#define NCNN_MOE_CPU_STATE_CACHE_H

#include "ncnn/moe/result.h"

#include <cstddef>
#include <span>

namespace ncnn {
namespace moe {

struct CpuLayerCache;

void begin_state_cache_transaction(
    std::span<CpuLayerCache> caches,
    size_t expected_rows);

void record_standard_cache_transaction_rows(
    CpuLayerCache& cache,
    size_t rows);

void record_gated_delta_cache_transaction_row(
    CpuLayerCache& cache);

[[nodiscard]] Result<void> finish_state_cache_transaction(
    std::span<CpuLayerCache> caches,
    size_t committed_rows);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_STATE_CACHE_H
