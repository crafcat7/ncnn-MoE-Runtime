#include "ncnn/moe/execution_graph.h"

#include <algorithm>
#include <utility>

namespace ncnn {
namespace moe {

static Result<void> validate_graph(const ExecutionGraph& graph)
{
    if (graph.nodes.empty())
        return Error{ErrorCode::InvalidModel, "execution graph cannot be empty"};

    std::vector<uint32_t> indegrees(graph.nodes.size(), 0);
    std::vector<std::vector<ExecutionNodeId> > dependents(graph.nodes.size());
    for (size_t node_index = 0; node_index < graph.nodes.size(); ++node_index) {
        const ExecutionNode& node = graph.nodes[node_index];
        if (node.id != node_index) {
            return Error{
                ErrorCode::InvalidModel,
                "execution graph node ids must be contiguous and index-aligned"};
        }
        if (node.name.empty())
            return Error{ErrorCode::InvalidModel, "execution graph node name cannot be empty"};
        std::vector<ExecutionNodeId> unique_dependencies;
        unique_dependencies.reserve(node.dependencies.size());
        for (ExecutionNodeId dependency : node.dependencies) {
            if (dependency >= graph.nodes.size())
                return Error{ErrorCode::InvalidModel, "execution graph dependency is out of range"};
            if (dependency == node.id)
                return Error{ErrorCode::InvalidModel, "execution graph node cannot depend on itself"};
            if (std::find(
                    unique_dependencies.begin(),
                    unique_dependencies.end(),
                    dependency)
                != unique_dependencies.end()) {
                return Error{
                    ErrorCode::InvalidModel,
                    "execution graph contains a duplicate dependency"};
            }
            unique_dependencies.push_back(dependency);
            dependents[dependency].push_back(node.id);
            ++indegrees[node_index];
        }
        if (node.type == ExecutionNodeType::Expert
            && (node.layer_id == invalid_execution_layer_id
                || node.expert_id == invalid_execution_expert_id
                || !has_flag(node.flags, ExecutionNodeConditional))) {
            return Error{
                ErrorCode::InvalidModel,
                "execution graph Expert nodes require layer, expert, and conditional metadata"};
        }
    }

    std::vector<ExecutionNodeId> ready;
    ready.reserve(graph.nodes.size());
    for (ExecutionNodeId node_id = 0; node_id < graph.nodes.size(); ++node_id) {
        if (indegrees[node_id] == 0)
            ready.push_back(node_id);
    }

    size_t visited = 0;
    while (!ready.empty()) {
        std::vector<ExecutionNodeId> next;
        for (ExecutionNodeId node_id : ready) {
            ++visited;
            for (ExecutionNodeId dependent : dependents[node_id]) {
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

Result<ExecutionSchedule> MoeScheduler::schedule(
    const ExecutionGraph& graph) const
{
    auto valid = validate_graph(graph);
    if (!valid)
        return valid.error();

    std::vector<uint32_t> indegrees(graph.nodes.size(), 0);
    std::vector<std::vector<ExecutionNodeId> > dependents(graph.nodes.size());
    for (const ExecutionNode& node : graph.nodes) {
        indegrees[node.id] = static_cast<uint32_t>(node.dependencies.size());
        for (ExecutionNodeId dependency : node.dependencies)
            dependents[dependency].push_back(node.id);
    }

    std::vector<ExecutionNodeId> ready;
    for (ExecutionNodeId node_id = 0; node_id < graph.nodes.size(); ++node_id) {
        if (indegrees[node_id] == 0)
            ready.push_back(node_id);
    }

    ExecutionSchedule result;
    while (!ready.empty()) {
        ExecutionWave wave;
        wave.nodes = ready;
        result.waves.push_back(std::move(wave));

        std::vector<ExecutionNodeId> next;
        for (ExecutionNodeId node_id : ready) {
            for (ExecutionNodeId dependent : dependents[node_id]) {
                if (--indegrees[dependent] == 0)
                    next.push_back(dependent);
            }
        }
        ready = std::move(next);
    }
    return result;
}

} // namespace moe
} // namespace ncnn
