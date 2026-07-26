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
    const long double first_density = static_cast<long double>(first.dispatch_count) / first.weight_bytes;
    const long double second_density = static_cast<long double>(second.dispatch_count) / second.weight_bytes;
    if (first_density != second_density)
        return first_density > second_density;
    if (first.token_count != second.token_count)
        return first.token_count > second.token_count;
    if (first.key.layer_id != second.key.layer_id)
        return first.key.layer_id < second.key.layer_id;
    return first.key.expert_id < second.key.expert_id;
}

Expert::Expert(ExpertKey key, uint64_t weight_bytes, ExpertCacheState cache_state, TensorLocation memory_location, ExpertKernel kernel)
    : key_(key),
      weight_bytes_(weight_bytes),
      kernel_(kernel),
      cache_state_(cache_state),
      memory_location_(memory_location)
{
}

ExpertStatistics Expert::statistics() const
{
    ExpertStatistics result;
    result.key = key_;
    result.cache_state = cache_state_.load(std::memory_order_relaxed);
    result.memory_location = memory_location_.load(std::memory_order_relaxed);
    result.kernel = kernel_;
    result.weight_bytes = weight_bytes_;
    result.dispatch_count = dispatch_count_.load(std::memory_order_relaxed);
    result.token_count = token_count_.load(std::memory_order_relaxed);
    result.cache_hits = cache_hits_.load(std::memory_order_relaxed);
    result.cache_misses = cache_misses_.load(std::memory_order_relaxed);
    result.last_used = last_used_.load(std::memory_order_relaxed);
    return result;
}

void Expert::record_dispatch(uint64_t token_count, uint64_t clock)
{
    dispatch_count_.fetch_add(1, std::memory_order_relaxed);
    token_count_.fetch_add(token_count, std::memory_order_relaxed);
    last_used_.store(clock, std::memory_order_relaxed);
}

void Expert::record_cache_hit()
{
    cache_hits_.fetch_add(1, std::memory_order_relaxed);
}

void Expert::record_cache_miss()
{
    cache_misses_.fetch_add(1, std::memory_order_relaxed);
}

void Expert::set_residency(ExpertCacheState state, TensorLocation location)
{
    cache_state_.store(state, std::memory_order_relaxed);
    memory_location_.store(location, std::memory_order_relaxed);
}

void ExpertStore::add(std::shared_ptr<Expert> expert)
{
    experts_.push_back(std::move(expert));
}

ExpertStoreStatistics ExpertStore::statistics() const
{
    ExpertStoreStatistics result;
    result.expert_count = experts_.size();
    for (const std::shared_ptr<Expert>& entry : experts_)
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
        result.registered_weight_bytes += expert.weight_bytes;
        result.dispatch_count += expert.dispatch_count;
        result.token_count += expert.token_count;
        result.cache_hits += expert.cache_hits;
        result.cache_misses += expert.cache_misses;
    }
    return result;
}

ExpertHotsetEstimate ExpertStore::estimate_hotset(uint64_t capacity_bytes) const
{
    ExpertHotsetEstimate result;
    result.capacity_bytes = capacity_bytes;
    std::vector<ExpertStatistics> experts;
    experts.reserve(experts_.size());
    for (const std::shared_ptr<Expert>& entry : experts_)
    {
        const ExpertStatistics expert = entry->statistics();
        if (expert.dispatch_count != 0 && expert.weight_bytes != 0)
            experts.push_back(expert);
    }
    result.active_expert_count = experts.size();
    for (const ExpertStatistics& expert : experts)
    {
        saturating_add(saturating_multiply(expert.weight_bytes, expert.dispatch_count), result.requested_batch_weight_bytes);
        saturating_add(saturating_multiply(expert.weight_bytes, expert.token_count), result.requested_route_weight_bytes);
    }

    std::sort(experts.begin(), experts.end(), expert_hotter);
    for (const ExpertStatistics& expert : experts)
    {
        if (expert.weight_bytes > capacity_bytes - result.resident_bytes)
        {
            continue;
        }
        result.resident_bytes += expert.weight_bytes;
        ++result.resident_expert_count;
        saturating_add(saturating_multiply(expert.weight_bytes, expert.dispatch_count), result.covered_batch_weight_bytes);
        saturating_add(saturating_multiply(expert.weight_bytes, expert.token_count), result.covered_route_weight_bytes);
    }
    return result;
}

} // namespace moe
} // namespace ncnn
