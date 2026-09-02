#include "ncnn/moe/expert.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace ncnn {
namespace moe {

static uint64_t saturating_multiply(uint64_t first, uint64_t second) noexcept
{
    if (first != 0 && second > std::numeric_limits<uint64_t>::max() / first)
    {
        return std::numeric_limits<uint64_t>::max();
    }
    return first * second;
}

static void saturating_add(uint64_t value, uint64_t& destination) noexcept
{
    if (destination > std::numeric_limits<uint64_t>::max() - value)
        destination = std::numeric_limits<uint64_t>::max();
    else
        destination += value;
}

static bool expert_hotter(const ExpertStatistics& first, const ExpertStatistics& second)
{
    const long double first_density = static_cast<long double>(first.dispatch_count) / first.weight_size;
    const long double second_density = static_cast<long double>(second.dispatch_count) / second.weight_size;
    if (first_density != second_density)
        return first_density > second_density;
    if (first.token_count != second.token_count)
        return first.token_count > second.token_count;
    if (first.key.layer_id != second.key.layer_id)
        return first.key.layer_id < second.key.layer_id;
    return first.key.expert_id < second.key.expert_id;
}

Expert::Expert(ExpertKey _key, uint64_t _weight_size, ExpertCacheState _cache_state, TensorLocation _memory_location, ExpertKernel _kernel)
    : key(_key),
      weight_size(_weight_size),
      kernel(_kernel),
      cache_state(_cache_state),
      memory_location(_memory_location)
{
}

ExpertStatistics Expert::statistics() const
{
    ExpertStatistics result;
    result.key = key;
    result.cache_state = cache_state.load(std::memory_order_relaxed);
    result.memory_location = memory_location.load(std::memory_order_relaxed);
    result.kernel = kernel;
    result.weight_size = weight_size;
    result.dispatch_count = dispatch_count.load(std::memory_order_relaxed);
    result.token_count = token_count.load(std::memory_order_relaxed);
    result.cache_hits = cache_hits.load(std::memory_order_relaxed);
    result.cache_misses = cache_misses.load(std::memory_order_relaxed);
    result.last_used = last_used.load(std::memory_order_relaxed);
    return result;
}

void Expert::record_dispatch(uint64_t tokens, uint64_t clock)
{
    dispatch_count.fetch_add(1, std::memory_order_relaxed);
    token_count.fetch_add(tokens, std::memory_order_relaxed);
    last_used.store(clock, std::memory_order_relaxed);
}

void Expert::record_cache_hit()
{
    cache_hits.fetch_add(1, std::memory_order_relaxed);
}

void Expert::record_cache_miss()
{
    cache_misses.fetch_add(1, std::memory_order_relaxed);
}

void Expert::set_residency(ExpertCacheState state, TensorLocation location)
{
    cache_state.store(state, std::memory_order_relaxed);
    memory_location.store(location, std::memory_order_relaxed);
}

void ExpertStore::add(std::shared_ptr<Expert> expert)
{
    experts.push_back(std::move(expert));
}

ExpertStoreStatistics ExpertStore::statistics() const
{
    ExpertStoreStatistics result;
    result.expert_count = experts.size();
    for (const std::shared_ptr<Expert>& entry : experts)
    {
        const ExpertStatistics expert = entry->statistics();
        switch (expert.cache_state)
        {
        case ExpertCacheState::Unloaded:
            ++result.unloaded_experts;
            break;
        case ExpertCacheState::Loading:
            ++result.loading_experts;
            break;
        case ExpertCacheState::Resident:
            ++result.resident_experts;
            break;
        case ExpertCacheState::Failed:
            ++result.failed_experts;
            break;
        }
        result.registered_weight_size += expert.weight_size;
        result.dispatch_count += expert.dispatch_count;
        result.token_count += expert.token_count;
        result.cache_hits += expert.cache_hits;
        result.cache_misses += expert.cache_misses;
    }
    return result;
}

ExpertHotsetEstimate ExpertStore::estimate_hotset(uint64_t cache_size) const
{
    ExpertHotsetEstimate result;
    result.cache_size = cache_size;
    std::vector<ExpertStatistics> hotset;
    hotset.reserve(experts.size());
    for (const std::shared_ptr<Expert>& entry : experts)
    {
        const ExpertStatistics expert = entry->statistics();
        if (expert.dispatch_count != 0 && expert.weight_size != 0)
            hotset.push_back(expert);
    }
    result.active_expert_count = hotset.size();
    for (const ExpertStatistics& expert : hotset)
    {
        saturating_add(saturating_multiply(expert.weight_size, expert.dispatch_count), result.requested_batch_weight_bytes);
        saturating_add(saturating_multiply(expert.weight_size, expert.token_count), result.requested_route_weight_bytes);
    }

    std::sort(hotset.begin(), hotset.end(), expert_hotter);
    for (const ExpertStatistics& expert : hotset)
    {
        if (expert.weight_size > cache_size - result.resident_size)
        {
            continue;
        }
        result.resident_size += expert.weight_size;
        ++result.resident_expert_count;
        saturating_add(saturating_multiply(expert.weight_size, expert.dispatch_count), result.covered_batch_weight_bytes);
        saturating_add(saturating_multiply(expert.weight_size, expert.token_count), result.covered_route_weight_bytes);
    }
    return result;
}

} // namespace moe
} // namespace ncnn
