#ifndef NCNN_MOE_ATTENTION_INTERNAL_H
#define NCNN_MOE_ATTENTION_INTERNAL_H

#include "attention.h"
#include "statecache.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ncnn {
namespace moe {

[[nodiscard]] inline uint64_t cache_slot(const LayerCache& cache, uint64_t token_index)
{
    assert(cache.capacity_tokens > 0);
    return (cache.first_slot + token_index) % cache.capacity_tokens;
}

[[nodiscard]] inline bool key_visible(uint64_t key_position, uint64_t query_position, uint64_t window) noexcept
{
    return key_position <= query_position
           && (window == 0 || key_position + window > query_position);
}

void sdpa_into(const AttentionBlockPlan& plan, const TensorData* sinks, uint64_t position_offset, const ActivationBuffer& query,
               const LayerCache& cache, ActivationBuffer& output, AttentionScratch& scratch,
               std::span<const size_t> selected_offsets,
               std::span<const uint32_t> selected_indices,
               uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_ATTENTION_INTERNAL_H
