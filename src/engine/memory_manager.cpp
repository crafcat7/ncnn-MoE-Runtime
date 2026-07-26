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
    std::vector<TensorResidency> tensors;
    tensors.reserve(graph.tensors.size());
    for (const ExecutionTensor& tensor : graph.tensors)
    {
        TensorResidency residency;
        residency.tensor_id = tensor.id;
        residency.location = tensor.location;
        residency.bytes = tensor.estimated_bytes;
        tensors.push_back(residency);
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    tensors_ = std::move(tensors);
    return {};
}

Result<void> MemoryManager::transition_unlocked(ExecutionTensorId tensor_id, TensorLocation location)
{
    if (location == TensorLocation::Automatic)
    {
        return Error{ErrorCode::InvalidArgument, "tensor residency transition requires a concrete location"};
    }
    if (tensor_id >= tensors_.size())
    {
        return Error{ErrorCode::InvalidArgument, "tensor residency id is out of range"};
    }
    TensorResidency& tensor = tensors_[tensor_id];
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
    const std::lock_guard<std::mutex> lock(mutex_);
    return transition_unlocked(tensor_id, location);
}

Result<void> MemoryManager::record_execution(const ExecutionGraph& graph)
{
    const std::lock_guard<std::mutex> lock(mutex_);
    for (const ExecutionNode& node : graph.nodes)
    {
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
    const std::lock_guard<std::mutex> lock(mutex_);
    MemoryManagerStatistics result;
    result.registered_tensors = tensors_.size();
    for (const TensorResidency& tensor : tensors_)
    {
        switch (tensor.location)
        {
        case TensorLocation::Automatic: break;
        case TensorLocation::Cpu: result.cpu_bytes += tensor.bytes; break;
        case TensorLocation::Vulkan: result.vulkan_bytes += tensor.bytes; break;
        case TensorLocation::Shared: result.shared_bytes += tensor.bytes; break;
        }
        result.transitions += tensor.transition_count;
        result.tensor_uses += tensor.use_count;
    }
    return result;
}

} // namespace moe
} // namespace ncnn
