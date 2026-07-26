#include "ncnn/moe/execution_graph.h"

#include <algorithm>
#include <utility>

namespace ncnn {
namespace moe {

static Result<void> validate_graph(const ExecutionGraph& graph)
{
    if (graph.nodes.empty())
        return Error{ErrorCode::InvalidModel, "execution graph cannot be empty"};

    for (size_t tensor_index = 0; tensor_index < graph.tensors.size(); ++tensor_index)
    {
        const ExecutionTensor& tensor = graph.tensors[tensor_index];
        if (tensor.id != tensor_index)
        {
            return Error{ErrorCode::InvalidModel, "execution tensor ids must be contiguous and index-aligned"};
        }
        if (tensor.name.empty())
        {
            return Error{ErrorCode::InvalidModel, "execution tensor name cannot be empty"};
        }
        if (tensor.producer == invalid_execution_node_id || tensor.producer >= graph.nodes.size())
        {
            return Error{ErrorCode::InvalidModel, "execution tensor must have a valid producer"};
        }
        for (ExecutionNodeId consumer : tensor.consumers)
        {
            if (consumer >= graph.nodes.size())
            {
                return Error{ErrorCode::InvalidModel, "execution tensor consumer is out of range"};
            }
        }
    }

    std::vector<uint32_t> indegrees(graph.nodes.size(), 0);
    std::vector<std::vector<ExecutionNodeId>> dependents(graph.nodes.size());
    for (size_t node_index = 0; node_index < graph.nodes.size(); ++node_index)
    {
        const ExecutionNode& node = graph.nodes[node_index];
        if (node.id != node_index)
        {
            return Error{ErrorCode::InvalidModel, "execution graph node ids must be contiguous and index-aligned"};
        }
        if (node.name.empty())
            return Error{ErrorCode::InvalidModel, "execution graph node name cannot be empty"};
        const uint32_t selected_backend = node.backend == ExecutionBackend::Cpu ? ExecutionBackendCpu : ExecutionBackendVulkan;
        if (!has_flag(node.backend_mask, selected_backend))
        {
            return Error{ErrorCode::InvalidModel, "execution node selected backend is not allowed by its backend mask"};
        }
        std::vector<ExecutionNodeId> unique_dependencies;
        unique_dependencies.reserve(node.dependencies.size());
        for (ExecutionNodeId dependency : node.dependencies)
        {
            if (dependency >= graph.nodes.size())
                return Error{ErrorCode::InvalidModel, "execution graph dependency is out of range"};
            if (dependency == node.id)
                return Error{ErrorCode::InvalidModel, "execution graph node cannot depend on itself"};
            if (std::find(unique_dependencies.begin(), unique_dependencies.end(), dependency) != unique_dependencies.end())
            {
                return Error{ErrorCode::InvalidModel, "execution graph contains a duplicate dependency"};
            }
            unique_dependencies.push_back(dependency);
            dependents[dependency].push_back(node.id);
            ++indegrees[node_index];
        }
        for (ExecutionTensorId input : node.inputs)
        {
            if (input >= graph.tensors.size())
            {
                return Error{ErrorCode::InvalidModel, "execution node input tensor is out of range"};
            }
            const ExecutionTensor& tensor = graph.tensors[input];
            if (std::find(tensor.consumers.begin(), tensor.consumers.end(), node.id) == tensor.consumers.end())
            {
                return Error{ErrorCode::InvalidModel, "execution input tensor does not list its consumer"};
            }
        }
        for (ExecutionTensorId output : node.outputs)
        {
            if (output >= graph.tensors.size())
            {
                return Error{ErrorCode::InvalidModel, "execution node output tensor is out of range"};
            }
            if (graph.tensors[output].producer != node.id)
            {
                return Error{ErrorCode::InvalidModel, "execution output tensor does not list its producer"};
            }
        }
        if (node.type == ExecutionNodeType::Expert
            && (node.layer_id == invalid_execution_layer_id
                || node.expert_id == invalid_execution_expert_id
                || !has_flag(node.flags, ExecutionNodeConditional)))
        {
            return Error{ErrorCode::InvalidModel, "execution graph Expert nodes require layer, expert, and conditional metadata"};
        }
        if (node.type == ExecutionNodeType::ExpertGroup && (node.layer_id == invalid_execution_layer_id || node.expert_id != invalid_execution_expert_id))
        {
            return Error{ErrorCode::InvalidModel, "execution graph ExpertGroup nodes require layer metadata and dynamic expert selection"};
        }
    }

    for (size_t event_index = 0; event_index < graph.events.size(); ++event_index)
    {
        const ExecutionEvent& event = graph.events[event_index];
        if (event.id != event_index)
        {
            return Error{ErrorCode::InvalidModel, "execution event ids must be contiguous and index-aligned"};
        }
        if (event.producer >= graph.nodes.size() || event.consumers.empty())
        {
            return Error{ErrorCode::InvalidModel, "execution event requires a valid producer and consumers"};
        }
        if (graph.nodes[event.producer].signal_event != event.id)
        {
            return Error{ErrorCode::InvalidModel, "execution event producer does not signal the event"};
        }
        for (ExecutionNodeId consumer : event.consumers)
        {
            if (consumer >= graph.nodes.size())
            {
                return Error{ErrorCode::InvalidModel, "execution event consumer is out of range"};
            }
            const std::vector<ExecutionEventId>& waits = graph.nodes[consumer].wait_events;
            if (std::find(waits.begin(), waits.end(), event.id) == waits.end())
            {
                return Error{ErrorCode::InvalidModel, "execution event consumer does not wait for the event"};
            }
        }
    }

    std::vector<ExecutionNodeId> ready;
    ready.reserve(graph.nodes.size());
    for (ExecutionNodeId node_id = 0; node_id < graph.nodes.size(); ++node_id)
    {
        if (indegrees[node_id] == 0)
            ready.push_back(node_id);
    }

    size_t visited = 0;
    while (!ready.empty())
    {
        std::vector<ExecutionNodeId> next;
        for (ExecutionNodeId node_id : ready)
        {
            ++visited;
            for (ExecutionNodeId dependent : dependents[node_id])
            {
                if (--indegrees[dependent] == 0)
                    next.push_back(dependent);
            }
        }
        ready = std::move(next);
    }
    if (visited != graph.nodes.size())
        return Error{ErrorCode::InvalidModel, "execution graph contains a dependency cycle"};
    return {};
}

Result<void> ExecutionGraph::validate() const
{
    return validate_graph(*this);
}

const ExecutionNode* ExecutionGraph::find(ExecutionNodeId id) const noexcept
{
    return id < nodes.size() && nodes[id].id == id ? &nodes[id] : nullptr;
}

const ExecutionTensor* ExecutionGraph::find_tensor(ExecutionTensorId id) const noexcept
{
    return id < tensors.size() && tensors[id].id == id ? &tensors[id] : nullptr;
}

Result<ExecutionSchedule> MoeScheduler::schedule(const ExecutionGraph& graph) const
{
    auto valid = validate_graph(graph);
    if (!valid)
        return valid.error();

    std::vector<uint32_t> indegrees(graph.nodes.size(), 0);
    std::vector<std::vector<ExecutionNodeId>> dependents(graph.nodes.size());
    for (const ExecutionNode& node : graph.nodes)
    {
        indegrees[node.id] = static_cast<uint32_t>(node.dependencies.size());
        for (ExecutionNodeId dependency : node.dependencies)
            dependents[dependency].push_back(node.id);
    }

    std::vector<ExecutionNodeId> ready;
    for (ExecutionNodeId node_id = 0; node_id < graph.nodes.size(); ++node_id)
    {
        if (indegrees[node_id] == 0)
            ready.push_back(node_id);
    }

    ExecutionSchedule result;
    while (!ready.empty())
    {
        ExecutionWave wave;
        wave.nodes = ready;
        for (ExecutionNodeId node_id : ready)
        {
            if (graph.nodes[node_id].backend == ExecutionBackend::Vulkan)
            {
                wave.vulkan_nodes.push_back(node_id);
            }
            else
            {
                wave.cpu_nodes.push_back(node_id);
            }
        }
        result.waves.push_back(std::move(wave));

        std::vector<ExecutionNodeId> next;
        for (ExecutionNodeId node_id : ready)
        {
            for (ExecutionNodeId dependent : dependents[node_id])
            {
                if (--indegrees[dependent] == 0)
                    next.push_back(dependent);
            }
        }
        ready = std::move(next);
    }
    result.events = graph.events;
    return result;
}

static uint32_t backend_flag(ExecutionBackend backend)
{
    return backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu;
}

static bool prefer_vulkan_node(ExecutionNodeType type)
{
    return type == ExecutionNodeType::Attention || type == ExecutionNodeType::LmHead;
}

static Result<ExecutionBackend> select_backend(const ExecutionNode& node, const RuntimeSchedulingOptions& options)
{
    const uint32_t usable = node.backend_mask & options.available_backends;
    if (usable == 0)
    {
        return Error{ErrorCode::UnsupportedModel, "execution node has no available backend: " + node.name};
    }

    if (has_flag(options.flags, RuntimeSchedulingPreferVulkanDense) && prefer_vulkan_node(node.type) && has_flag(usable, ExecutionBackendVulkan))
    {
        return ExecutionBackend::Vulkan;
    }
    if (has_flag(usable, backend_flag(node.backend)))
        return node.backend;
    if (has_flag(usable, ExecutionBackendCpu))
        return ExecutionBackend::Cpu;
    return ExecutionBackend::Vulkan;
}

Result<ScheduledExecutionGraph> RuntimeScheduler::compile(ExecutionGraph graph, const RuntimeSchedulingOptions& options) const
{
    if (options.available_backends == 0)
    {
        return Error{ErrorCode::InvalidArgument, "runtime scheduler requires at least one backend"};
    }
    if (options.cpu_parallelism == 0)
    {
        return Error{ErrorCode::InvalidArgument, "runtime scheduler cpu_parallelism must be non-zero"};
    }

    graph.events.clear();
    for (ExecutionNode& node : graph.nodes)
    {
        auto backend = select_backend(node, options);
        if (!backend)
            return backend.error();
        node.backend = backend.value();
        node.wait_events.clear();
        node.signal_event = invalid_execution_event_id;
        node.flags &= ~ExecutionNodeAsync;
    }

    const bool enable_async = has_flag(options.flags, RuntimeSchedulingEnableAsyncEvents)
                              && has_flag(options.available_backends, ExecutionBackendCpu)
                              && has_flag(options.available_backends, ExecutionBackendVulkan)
                              && options.vulkan_queue_count != 0;
    if (enable_async)
    {
        for (ExecutionNode& producer : graph.nodes)
        {
            std::vector<ExecutionNodeId> consumers;
            for (const ExecutionNode& candidate : graph.nodes)
            {
                if (candidate.backend == producer.backend)
                    continue;
                if (std::find(candidate.dependencies.begin(), candidate.dependencies.end(), producer.id) != candidate.dependencies.end())
                {
                    consumers.push_back(candidate.id);
                }
            }
            if (consumers.empty())
                continue;

            ExecutionEvent event;
            event.id = static_cast<ExecutionEventId>(graph.events.size());
            event.producer = producer.id;
            event.consumers = consumers;
            producer.signal_event = event.id;
            producer.flags |= ExecutionNodeAsync;
            graph.events.push_back(event);
            for (ExecutionNodeId consumer : consumers)
            {
                graph.nodes[consumer].wait_events.push_back(event.id);
                graph.nodes[consumer].flags |= ExecutionNodeAsync;
            }
        }
    }

    auto valid = graph.validate();
    if (!valid)
        return valid.error();
    MoeScheduler scheduler;
    auto schedule = scheduler.schedule(graph);
    if (!schedule)
        return schedule.error();
    ScheduledExecutionGraph result;
    result.graph = std::move(graph);
    result.schedule = std::move(schedule).value();
    result.schedule.cpu_parallelism = options.cpu_parallelism;
    result.schedule.vulkan_queue_count = options.vulkan_queue_count;
    return result;
}

} // namespace moe
} // namespace ncnn
