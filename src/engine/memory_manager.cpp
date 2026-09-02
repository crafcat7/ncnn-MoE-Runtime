#include "ncnn/moe/memory_manager.h"

namespace ncnn {
namespace moe {

MemoryManager::MemoryManager(const ExecutionGraph& graph)
{
    (void)initialize(graph);
}

Result<void> MemoryManager::initialize(const ExecutionGraph& graph)
{
    auto valid = graph.validate();
    if (!valid)
        return valid.error();
    std::vector<TensorResidency> residencies;
    residencies.reserve(graph.tensors.size());
    for (const ExecutionTensor& tensor : graph.tensors)
    {
        TensorResidency residency;
        residency.tensor_id = tensor.id;
        residency.location = tensor.location;
        residency.size = tensor.estimated_size;
        residencies.push_back(residency);
    }
    const std::lock_guard<std::mutex> lock(mutex);
    tensors = std::move(residencies);
    return {};
}

Result<void> MemoryManager::transition_unlocked(ExecutionTensorId tensor_id, TensorLocation location)
{
    if (location == TensorLocation::Automatic)
    {
        return Error{ErrorCode::InvalidArgument, "tensor residency transition requires a concrete location"};
    }
    if (tensor_id >= tensors.size())
    {
        return Error{ErrorCode::InvalidArgument, "tensor residency id is out of range"};
    }
    TensorResidency& tensor = tensors[tensor_id];
    if (tensor.location != location)
    {
        tensor.location = location;
        ++tensor.transition_count;
    }
    ++tensor.use_count;
    return {};
}

Result<void> MemoryManager::transition(ExecutionTensorId tensor_id, TensorLocation location)
{
    const std::lock_guard<std::mutex> lock(mutex);
    return transition_unlocked(tensor_id, location);
}

Result<void> MemoryManager::record_execution(
    const ExecutionGraph& graph,
    const ExecutionSchedule& schedule)
{
    // The compiled model owns an immutable, already-validated reservation.
    // Do only the cheap shape checks here; rebuilding validation vectors on
    // every decode token would put the scheduler back on the hot path.
    if (schedule.node_order.size() != graph.nodes.size())
        return Error{ErrorCode::InternalError, "execution reservation does not cover the graph"};
    const std::lock_guard<std::mutex> lock(mutex);
    for (ExecutionNodeId node_id : schedule.node_order)
    {
        if (node_id >= graph.nodes.size())
            return Error{ErrorCode::InternalError, "execution reservation references an invalid node"};
        const ExecutionNode& node = graph.nodes[node_id];
        for (ExecutionTensorId input : node.inputs)
        {
            const ExecutionTensor* tensor = graph.find_tensor(input);
            if (!tensor)
                return Error{ErrorCode::InternalError, "execution input tensor is missing"};
            auto status = transition_unlocked(input, tensor->location);
            if (!status)
                return status.error();
        }
        for (ExecutionTensorId output : node.outputs)
        {
            const ExecutionTensor* tensor = graph.find_tensor(output);
            if (!tensor)
                return Error{ErrorCode::InternalError, "execution output tensor is missing"};
            auto status = transition_unlocked(output, tensor->location);
            if (!status)
                return status.error();
        }
    }
    return {};
}

MemoryManagerStatistics MemoryManager::statistics() const
{
    const std::lock_guard<std::mutex> lock(mutex);
    MemoryManagerStatistics result;
    result.registered_tensors = tensors.size();
    for (const TensorResidency& tensor : tensors)
    {
        switch (tensor.location)
        {
        case TensorLocation::Automatic: break;
        case TensorLocation::Cpu: result.cpu_size += tensor.size; break;
        case TensorLocation::Vulkan: result.vulkan_size += tensor.size; break;
        case TensorLocation::Shared: result.shared_size += tensor.size; break;
        }
        result.transitions += tensor.transition_count;
        result.tensor_uses += tensor.use_count;
    }
    return result;
}

} // namespace moe
} // namespace ncnn
