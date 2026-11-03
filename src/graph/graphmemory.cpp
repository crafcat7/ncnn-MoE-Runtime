#include "graphmemory.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <utility>

namespace ncnn {
namespace moe {

struct LiveRange
{
    uint32_t last_node = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct LiveOrder
{
    bool operator()(const LiveRange& left, const LiveRange& right) const noexcept
    {
        return left.last_node != right.last_node ? left.last_node > right.last_node
                                                 : left.offset > right.offset;
    }
};

struct DeviceState
{
    uint64_t peak_size = 0;
    std::map<uint64_t, uint64_t> free_ranges;
    std::priority_queue<LiveRange, std::vector<LiveRange>, LiveOrder> live_ranges;
};

static bool graph_memory_align(uint64_t offset, uint64_t alignment, uint64_t& aligned) noexcept
{
    const uint64_t remainder = offset % alignment;
    const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
    if (padding > std::numeric_limits<uint64_t>::max() - offset) return false;
    aligned = offset + padding;
    return true;
}

static void graph_memory_free(DeviceState& state, uint64_t offset, uint64_t size)
{
    if (size == 0) return;
    auto next = state.free_ranges.lower_bound(offset);
    if (next != state.free_ranges.begin())
    {
        const auto previous = std::prev(next);
        if (previous->first + previous->second == offset)
        {
            offset = previous->first;
            size += previous->second;
            state.free_ranges.erase(previous);
        }
    }
    if (next != state.free_ranges.end() && offset + size == next->first)
    {
        size += next->second;
        state.free_ranges.erase(next);
    }
    state.free_ranges.emplace(offset, size);
}

static Result<uint64_t> graph_memory_allocate(DeviceState& state, const GraphMemoryBuffer& buffer)
{
    if (buffer.size == 0) return uint64_t{0};
    auto best = state.free_ranges.end();
    uint64_t best_offset = 0;
    for (auto range = state.free_ranges.begin(); range != state.free_ranges.end(); ++range)
    {
        uint64_t aligned = 0;
        if (!graph_memory_align(range->first, buffer.alignment, aligned)) continue;
        const uint64_t padding = aligned - range->first;
        if (padding > range->second || buffer.size > range->second - padding) continue;
        // Best fit minimizes the complete free range consumed; lower offsets
        // break ties independently of the caller's buffer ordering.
        if (best == state.free_ranges.end() || range->second < best->second)
        {
            best = range;
            best_offset = aligned;
        }
    }
    if (best != state.free_ranges.end())
    {
        const uint64_t begin = best->first;
        const uint64_t end = begin + best->second;
        state.free_ranges.erase(best);
        graph_memory_free(state, begin, best_offset - begin);
        graph_memory_free(state, best_offset + buffer.size, end - best_offset - buffer.size);
        return best_offset;
    }

    // A free tail can grow in place. In particular, a larger successor after
    // all buffers expire needs max(previous_peak, size), rather than their sum.
    uint64_t begin = state.peak_size;
    auto tail = state.free_ranges.end();
    if (!state.free_ranges.empty())
    {
        const auto last = std::prev(state.free_ranges.end());
        if (last->first + last->second == state.peak_size)
        {
            begin = last->first;
            tail = last;
        }
    }
    uint64_t aligned = 0;
    if (!graph_memory_align(begin, buffer.alignment, aligned)
        || buffer.size > std::numeric_limits<uint64_t>::max() - aligned)
    {
        return Error{ErrorCode::InvalidArgument,
                     "graph memory offset overflow for buffer " + std::to_string(buffer.id)
                         + " on device " + std::to_string(buffer.device_index)};
    }
    if (tail != state.free_ranges.end()) state.free_ranges.erase(tail);
    graph_memory_free(state, begin, aligned - begin);
    state.peak_size = std::max(state.peak_size, aligned + buffer.size);
    return aligned;
}

Result<GraphMemoryPlan> plan_graph_memory(std::span<const GraphMemoryBuffer> buffers,
                                          std::span<const GraphMemorySubmission> submissions)
{
    std::set<std::pair<uint32_t, uint32_t>> identities;
    for (const GraphMemoryBuffer& buffer : buffers)
    {
        if (buffer.first_node > buffer.last_node)
            return Error{ErrorCode::InvalidArgument, "graph memory lifetime ends before it begins for buffer " + std::to_string(buffer.id)};
        if (buffer.alignment == 0 || (buffer.alignment & (buffer.alignment - 1)) != 0)
            return Error{ErrorCode::InvalidArgument, "graph memory alignment must be a nonzero power of two for buffer " + std::to_string(buffer.id)};
        if (!identities.emplace(buffer.device_index, buffer.id).second)
            return Error{ErrorCode::InvalidArgument, "duplicate graph memory buffer " + std::to_string(buffer.id) + " on device " + std::to_string(buffer.device_index)};
    }
    std::map<uint32_t, std::vector<GraphMemorySubmission>> device_submissions;
    for (const GraphMemorySubmission& submission : submissions)
    {
        if (submission.first_node > submission.last_node)
            return Error{ErrorCode::InvalidArgument, "graph memory submission ends before it begins on device " + std::to_string(submission.device_index)};
        device_submissions[submission.device_index].push_back(submission);
    }
    for (auto& [device, ranges] : device_submissions)
    {
        std::sort(ranges.begin(), ranges.end(), [](const auto& left, const auto& right) {
            return left.first_node < right.first_node;
        });
        for (size_t index = 1; index < ranges.size(); ++index)
            if (ranges[index].first_node <= ranges[index - 1].last_node)
                return Error{ErrorCode::InvalidArgument, "graph memory submission ranges overlap on device " + std::to_string(device)};
    }
    std::vector<GraphMemoryBuffer> retained;
    if (!submissions.empty())
    {
        retained.assign(buffers.begin(), buffers.end());
        for (GraphMemoryBuffer& buffer : retained)
        {
            const auto device = device_submissions.find(buffer.device_index);
            if (device == device_submissions.end()) continue;
            const auto& ranges = device->second;
            auto range = std::upper_bound(ranges.begin(), ranges.end(), buffer.last_node, [](uint32_t node, const auto& submission) {
                return node < submission.first_node;
            });
            if (range == ranges.begin()) continue;
            --range;
            if (buffer.first_node <= range->last_node)
                buffer.last_node = std::max(buffer.last_node, range->last_node);
        }
        buffers = retained;
    }
    std::vector<size_t> order(buffers.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        const GraphMemoryBuffer& a = buffers[left];
        const GraphMemoryBuffer& b = buffers[right];
        if (a.device_index != b.device_index) return a.device_index < b.device_index;
        if (a.first_node != b.first_node) return a.first_node < b.first_node;
        return a.id < b.id;
    });
    GraphMemoryPlan plan;
    plan.allocations.resize(buffers.size());
    std::map<uint32_t, DeviceState> devices;
    for (size_t index : order)
    {
        const GraphMemoryBuffer& buffer = buffers[index];
        DeviceState& state = devices[buffer.device_index];
        while (!state.live_ranges.empty() && state.live_ranges.top().last_node < buffer.first_node)
        {
            const LiveRange expired = state.live_ranges.top();
            state.live_ranges.pop();
            graph_memory_free(state, expired.offset, expired.size);
        }
        auto offset = graph_memory_allocate(state, buffer);
        if (!offset) return offset.error();
        plan.allocations[index] = {buffer.id, buffer.device_index, offset.value(), buffer.size, buffer.first_node, buffer.last_node};
        if (buffer.size != 0) state.live_ranges.push({buffer.last_node, offset.value(), buffer.size});
    }
    for (const auto& device : devices)
        plan.devices.push_back({device.first, device.second.peak_size});
    return plan;
}

} // namespace moe
} // namespace ncnn
