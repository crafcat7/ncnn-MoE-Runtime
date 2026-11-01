#ifndef NCNN_MOE_GRAPHMEMORY_H
#define NCNN_MOE_GRAPHMEMORY_H

#include "ncnn/moe/result.h"

#include <cstdint>
#include <map>
#include <queue>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct GraphMemoryBuffer
{
    uint32_t id = 0;
    uint32_t device_index = 0;
    uint64_t size = 0;
    uint64_t alignment = 1;
    uint32_t first_node = 0;
    uint32_t last_node = 0;
};

struct GraphMemoryAllocation
{
    uint32_t id = 0;
    uint32_t device_index = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t first_node = 0;
    uint32_t last_node = 0;
};

struct GraphMemoryDevice
{
    uint32_t device_index = 0;
    uint64_t peak_size = 0;
};

struct GraphMemoryPlan
{
    // Allocations retain input order; identity is (device_index, id).
    std::vector<GraphMemoryAllocation> allocations;
    // Devices are sorted by index, including zero-sized device arenas.
    std::vector<GraphMemoryDevice> devices;
};

// Operations recorded into one device submission retain their operands until
// the common completion fence. Ranges on one device must be disjoint.
struct GraphMemorySubmission
{
    uint32_t device_index = 0;
    uint32_t first_node = 0;
    uint32_t last_node = 0;
};

struct GraphMemoryLiveRange
{
    uint32_t last_node = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct GraphMemoryLiveOrder
{
    bool operator()(const GraphMemoryLiveRange& left, const GraphMemoryLiveRange& right) const noexcept;
};

struct GraphMemoryDeviceState
{
    uint64_t peak_size = 0;
    std::map<uint64_t, uint64_t> free_ranges;
    std::priority_queue<GraphMemoryLiveRange, std::vector<GraphMemoryLiveRange>, GraphMemoryLiveOrder> live_ranges;
};

// Lifetimes are closed intervals: inputs and outputs used at the same node
// cannot share bytes. Alignment must be a nonzero power of two. Zero-sized
// buffers have offset zero and consume no arena capacity.
// Optional submission ranges extend every intersecting buffer's last use to
// that submission's fence; callers supply the selected execution boundaries.
[[nodiscard]] Result<GraphMemoryPlan> plan_graph_memory(std::span<const GraphMemoryBuffer> buffers,
                                                        std::span<const GraphMemorySubmission> submissions = {});

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GRAPHMEMORY_H
