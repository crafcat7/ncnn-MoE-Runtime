#ifndef NCNN_MOE_MEMORY_MANAGER_H
#define NCNN_MOE_MEMORY_MANAGER_H

#include "ncnn/moe/execution_graph.h"
#include "ncnn/moe/result.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace ncnn {
namespace moe {

struct TensorResidency
{
    ExecutionTensorId tensor_id = invalid_execution_tensor_id;
    TensorLocation location = TensorLocation::Automatic;
    uint64_t bytes = 0;
    uint64_t use_count = 0;
    uint64_t transition_count = 0;
};

struct MemoryManagerStatistics
{
    uint64_t registered_tensors = 0;
    uint64_t cpu_bytes = 0;
    uint64_t vulkan_bytes = 0;
    uint64_t shared_bytes = 0;
    uint64_t transitions = 0;
    uint64_t tensor_uses = 0;
};

class MemoryManager
{
private:
    [[nodiscard]] Result<void> transition_unlocked(ExecutionTensorId tensor_id, TensorLocation location);

    mutable std::mutex mutex_;
    std::vector<TensorResidency> tensors_;

public:
    MemoryManager() = default;
    explicit MemoryManager(const ExecutionGraph& graph);

    [[nodiscard]] Result<void> initialize(const ExecutionGraph& graph);
    [[nodiscard]] Result<void> transition(ExecutionTensorId tensor_id, TensorLocation location);
    [[nodiscard]] Result<void> record_execution(const ExecutionGraph& graph);
    [[nodiscard]] MemoryManagerStatistics statistics() const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_MEMORY_MANAGER_H
