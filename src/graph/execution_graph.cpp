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
    std::vector<uint8_t> referenced_layer_plans(
        graph.layer_plans.size(),
        0);
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
        const bool layer_bound =
            node.type == ExecutionNodeType::Attention
            || node.type == ExecutionNodeType::Router
            || node.type == ExecutionNodeType::ExpertDispatch
            || node.type == ExecutionNodeType::Expert
            || node.type == ExecutionNodeType::ExpertGroup
            || node.type == ExecutionNodeType::SharedExpertGroup
            || node.type == ExecutionNodeType::Combine;
        if (layer_bound
            ? node.layer_plan_index == invalid_execution_layer_id
            : node.layer_plan_index != invalid_execution_layer_id)
        {
            return Error{ErrorCode::InvalidModel, "execution graph node has an invalid layer plan binding"};
        }
        if (node.layer_plan_index != invalid_execution_layer_id
            && node.layer_plan_index >= graph.layer_plans.size())
        {
            return Error{ErrorCode::InvalidModel, "execution graph node layer plan is out of range"};
        }
        if (node.layer_plan_index != invalid_execution_layer_id)
            referenced_layer_plans[node.layer_plan_index] = 1;
        const bool weight_bound =
            node.type == ExecutionNodeType::TokenEmbedding
            || node.type == ExecutionNodeType::FinalNorm
            || node.type == ExecutionNodeType::LmHead;
        if (weight_bound && node.weight_inputs.empty())
        {
            return Error{
                ErrorCode::InvalidModel,
                "execution graph weight-bound node has no weight inputs"};
        }
        if (!weight_bound && !node.weight_inputs.empty())
        {
            return Error{
                ErrorCode::InvalidModel,
                "execution graph layer node cannot own weight inputs"};
        }
        for (TensorHandle weight : node.weight_inputs)
        {
            if (weight == invalid_tensor_handle)
            {
                return Error{
                    ErrorCode::InvalidModel,
                    "execution graph contains an invalid weight input"};
            }
        }
        if (node.type == ExecutionNodeType::Expert
            && (node.layer_plan_index == invalid_execution_layer_id
                || node.expert_id == invalid_execution_expert_id
                || !has_flag(node.flags, ExecutionNodeConditional)))
        {
            return Error{ErrorCode::InvalidModel, "execution graph Expert nodes require layer, expert, and conditional metadata"};
        }
        if ((node.type == ExecutionNodeType::ExpertGroup
             || node.type == ExecutionNodeType::SharedExpertGroup)
            && (node.layer_plan_index == invalid_execution_layer_id
                || node.expert_id != invalid_execution_expert_id))
        {
            return Error{ErrorCode::InvalidModel, "execution graph ExpertGroup nodes require layer metadata and dynamic expert selection"};
        }
    }

    for (size_t plan_index = 0;
         plan_index < referenced_layer_plans.size();
         ++plan_index)
    {
        if (referenced_layer_plans[plan_index] == 0)
        {
            return Error{
                ErrorCode::InvalidModel,
                "execution graph contains an unbound layer plan"};
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

Result<void> ExecutionSchedule::validate(const ExecutionGraph& graph) const
{
    if (graph.nodes.empty())
        return Error{ErrorCode::InvalidModel, "execution schedule cannot reference an empty graph"};
    if (cpu_parallelism == 0)
        return Error{ErrorCode::InvalidModel, "execution schedule cpu parallelism must be non-zero"};
    if (node_order.size() != graph.nodes.size())
        return Error{ErrorCode::InvalidModel, "execution schedule node order is incomplete"};

    std::vector<uint32_t> positions(graph.nodes.size(), invalid_execution_layer_id);
    for (size_t order_index = 0; order_index < node_order.size(); ++order_index)
    {
        const ExecutionNodeId node_id = node_order[order_index];
        if (node_id >= graph.nodes.size())
            return Error{ErrorCode::InvalidModel, "execution schedule node order is out of range"};
        if (positions[node_id] != invalid_execution_layer_id)
            return Error{ErrorCode::InvalidModel, "execution schedule contains a duplicate node"};
        positions[node_id] = static_cast<uint32_t>(order_index);
        for (ExecutionNodeId dependency : graph.nodes[node_id].dependencies)
        {
            if (dependency >= graph.nodes.size()
                || positions[dependency] == invalid_execution_layer_id
                || positions[dependency] >= order_index)
            {
                return Error{ErrorCode::InvalidModel, "execution schedule violates a node dependency"};
            }
        }
    }

    uint32_t covered_nodes = 0;
    ExecutionBackend previous_backend = ExecutionBackend::Cpu;
    bool has_previous_backend = false;
    for (const ExecutionBackendRun& run : backend_runs)
    {
        if (run.node_count == 0
            || run.first_node != covered_nodes
            || run.node_count > node_order.size() - covered_nodes)
        {
            return Error{ErrorCode::InvalidModel, "execution schedule backend runs are not contiguous"};
        }
        if (has_previous_backend && run.backend == previous_backend)
            return Error{ErrorCode::InvalidModel, "execution schedule contains adjacent backend runs"};
        for (uint32_t offset = 0; offset < run.node_count; ++offset)
        {
            const ExecutionNodeId node_id = node_order[run.first_node + offset];
            const ExecutionBackend node_backend = graph.nodes[node_id].backend;
            if (node_backend != run.backend)
                return Error{ErrorCode::InvalidModel, "execution schedule backend run disagrees with node placement"};
        }
        covered_nodes += run.node_count;
        previous_backend = run.backend;
        has_previous_backend = true;
    }
    if (covered_nodes != node_order.size())
        return Error{ErrorCode::InvalidModel, "execution schedule backend runs do not cover the graph"};

    std::vector<uint8_t> wave_seen(graph.nodes.size(), 0);
    std::vector<uint8_t> wave_partition_seen(graph.nodes.size(), 0);
    size_t wave_order_index = 0;
    for (const ExecutionWave& wave : waves)
    {
        if (wave.nodes.empty())
            return Error{ErrorCode::InvalidModel, "execution schedule contains an empty wave"};
        if (wave.cpu_nodes.size() + wave.vulkan_nodes.size() != wave.nodes.size())
            return Error{ErrorCode::InvalidModel, "execution schedule wave backend partition is incomplete"};
        for (ExecutionNodeId node_id : wave.nodes)
        {
            if (node_id >= graph.nodes.size()
                || wave_seen[node_id] != 0
                || wave_order_index >= node_order.size()
                || node_order[wave_order_index] != node_id)
            {
                return Error{ErrorCode::InvalidModel, "execution schedule waves disagree with node order"};
            }
            wave_seen[node_id] = 1;
            ++wave_order_index;
        }
        for (ExecutionNodeId node_id : wave.cpu_nodes)
        {
            if (node_id >= graph.nodes.size()
                || wave_partition_seen[node_id] != 0
                || graph.nodes[node_id].backend != ExecutionBackend::Cpu)
            {
                return Error{ErrorCode::InvalidModel, "execution schedule CPU wave partition is invalid"};
            }
            wave_partition_seen[node_id] = 1;
        }
        for (ExecutionNodeId node_id : wave.vulkan_nodes)
        {
            if (node_id >= graph.nodes.size()
                || wave_partition_seen[node_id] != 0
                || graph.nodes[node_id].backend != ExecutionBackend::Vulkan)
            {
                return Error{ErrorCode::InvalidModel, "execution schedule Vulkan wave partition is invalid"};
            }
            wave_partition_seen[node_id] = 1;
        }
    }
    if (wave_order_index != node_order.size())
        return Error{ErrorCode::InvalidModel, "execution schedule waves do not cover the graph"};
    return {};
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
    result.node_order.reserve(graph.nodes.size());
    while (!ready.empty())
    {
        ExecutionWave wave;
        wave.nodes = ready;
        for (ExecutionNodeId node_id : ready)
        {
            result.node_order.push_back(node_id);
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

    result.backend_runs.reserve(result.node_order.size());
    for (uint32_t order_index = 0;
         order_index < result.node_order.size();
         ++order_index)
    {
        const ExecutionBackend backend =
            graph.nodes[result.node_order[order_index]].backend;
        if (result.backend_runs.empty()
            || result.backend_runs.back().backend != backend)
        {
            result.backend_runs.push_back({
                backend,
                order_index,
                1});
        }
        else
        {
            ++result.backend_runs.back().node_count;
        }
    }
    auto scheduled = result.validate(graph);
    if (!scheduled)
        return scheduled.error();
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
    auto scheduled = result.schedule.validate(result.graph);
    if (!scheduled)
        return scheduled.error();
    return result;
}

} // namespace moe
} // namespace ncnn
