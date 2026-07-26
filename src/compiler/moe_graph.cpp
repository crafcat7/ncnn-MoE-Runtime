#include "ncnn/moe/moe_ir.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static bool contains_moe_ir_value(const std::vector<MoeIRValueId>& values, MoeIRValueId value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

Result<void> MoeGraph::validate() const
{
    if (nodes.empty())
        return Error{ErrorCode::InvalidModel, "MoeIR graph cannot be empty"};
    if (values.empty())
        return Error{ErrorCode::InvalidModel, "MoeIR graph values cannot be empty"};

    std::vector<MoeIRNodeId> producers(values.size(), invalid_moe_ir_node_id);
    std::vector<uint32_t> indegrees(nodes.size(), 0);
    std::vector<std::vector<MoeIRNodeId>> dependents(nodes.size());

    for (size_t value_index = 0; value_index < values.size(); ++value_index)
    {
        const MoeIRValue& value = values[value_index];
        if (value.id != value_index)
        {
            return Error{ErrorCode::InvalidModel, "MoeIR value ids must be contiguous and index-aligned"};
        }
        if (value.name.empty())
            return Error{ErrorCode::InvalidModel, "MoeIR value name cannot be empty"};
        if (value.shape.empty() && !has_flag(value.flags, MoeIRValueDynamicShape))
        {
            return Error{ErrorCode::InvalidModel, "MoeIR values require a shape or the dynamic-shape flag"};
        }
        for (uint32_t dimension : value.shape)
        {
            if (dimension == 0 && !has_flag(value.flags, MoeIRValueDynamicShape))
            {
                return Error{ErrorCode::InvalidModel, "MoeIR static value dimensions must be non-zero"};
            }
        }
    }

    for (size_t node_index = 0; node_index < nodes.size(); ++node_index)
    {
        const MoeIRNode& node = nodes[node_index];
        if (node.id != node_index)
        {
            return Error{ErrorCode::InvalidModel, "MoeIR node ids must be contiguous and index-aligned"};
        }
        if (node.name.empty())
            return Error{ErrorCode::InvalidModel, "MoeIR node name cannot be empty"};
        if (node.outputs.empty())
            return Error{ErrorCode::InvalidModel, "MoeIR node must produce a value"};

        for (MoeIRValueId input : node.inputs)
        {
            if (input >= values.size())
                return Error{ErrorCode::InvalidModel, "MoeIR input value is out of range"};
        }
        for (MoeIRValueId output : node.outputs)
        {
            if (output >= values.size())
                return Error{ErrorCode::InvalidModel, "MoeIR output value is out of range"};
            if (producers[output] != invalid_moe_ir_node_id)
            {
                return Error{ErrorCode::InvalidModel, "MoeIR value has more than one producer"};
            }
            producers[output] = node.id;
        }

        if (node.operation == MoeIROperator::Attention)
        {
            if (node.layer_id == invalid_moe_ir_layer_id
                || node.attention.head_count == 0
                || node.attention.kv_head_count == 0
                || node.attention.head_dimension == 0)
            {
                return Error{ErrorCode::InvalidModel, "MoeIR Attention requires layer and head metadata"};
            }
        }
        if (node.operation == MoeIROperator::ExpertGroup)
        {
            if (node.layer_id == invalid_moe_ir_layer_id
                || node.experts.expert_count == 0
                || node.experts.top_k == 0
                || node.experts.top_k > node.experts.expert_count)
            {
                return Error{ErrorCode::InvalidModel, "MoeIR ExpertGroup requires valid layer and routing metadata"};
            }
        }
        if (node.operation == MoeIROperator::SharedExpertGroup
            && (node.layer_id == invalid_moe_ir_layer_id
                || node.experts.shared_expert_count == 0
                || node.experts.intermediate_size == 0))
        {
            return Error{ErrorCode::InvalidModel, "MoeIR SharedExpertGroup requires shared Expert metadata"};
        }
        if (node.operation == MoeIROperator::DenseFfn && (node.layer_id == invalid_moe_ir_layer_id || node.intermediate_size == 0))
        {
            return Error{ErrorCode::InvalidModel, "MoeIR DenseFfn requires layer and intermediate-size metadata"};
        }
        if (node.operation == MoeIROperator::KvCache && !has_flag(node.flags, MoeIRNodeStateful))
        {
            return Error{ErrorCode::InvalidModel, "MoeIR KV Cache node must be stateful"};
        }
    }

    for (const MoeIRNode& node : nodes)
    {
        for (MoeIRValueId input : node.inputs)
        {
            const MoeIRNodeId producer = producers[input];
            if (producer == invalid_moe_ir_node_id)
            {
                if (!has_flag(values[input].flags, MoeIRValueGraphInput) && !has_flag(values[input].flags, MoeIRValuePersistent))
                {
                    return Error{ErrorCode::InvalidModel, "MoeIR input has no producer"};
                }
                continue;
            }
            if (producer == node.id)
                return Error{ErrorCode::InvalidModel, "MoeIR node consumes its own output"};
            ++indegrees[node.id];
            dependents[producer].push_back(node.id);
        }
    }

    std::vector<MoeIRNodeId> ready;
    for (MoeIRNodeId node_id = 0; node_id < nodes.size(); ++node_id)
    {
        if (indegrees[node_id] == 0)
            ready.push_back(node_id);
    }
    size_t visited = 0;
    while (!ready.empty())
    {
        std::vector<MoeIRNodeId> next;
        for (MoeIRNodeId node_id : ready)
        {
            ++visited;
            for (MoeIRNodeId dependent : dependents[node_id])
            {
                if (--indegrees[dependent] == 0)
                    next.push_back(dependent);
            }
        }
        ready = std::move(next);
    }
    if (visited != nodes.size())
        return Error{ErrorCode::InvalidModel, "MoeIR graph contains a dependency cycle"};

    if (outputs.empty())
        return Error{ErrorCode::InvalidModel, "MoeIR graph output cannot be empty"};
    std::vector<MoeIRValueId> unique_outputs;
    for (MoeIRValueId output : outputs)
    {
        if (output >= values.size())
            return Error{ErrorCode::InvalidModel, "MoeIR graph output is out of range"};
        if (contains_moe_ir_value(unique_outputs, output))
        {
            return Error{ErrorCode::InvalidModel, "MoeIR graph contains a duplicate output"};
        }
        unique_outputs.push_back(output);
        if (!has_flag(values[output].flags, MoeIRValueGraphOutput))
        {
            return Error{ErrorCode::InvalidModel, "MoeIR graph output value is missing its output flag"};
        }
    }
    return {};
}

Result<void> MoeIR::validate() const
{
    if (model_type.empty())
        return Error{ErrorCode::InvalidModel, "MoeIR model_type cannot be empty"};
    if (vocabulary_size == 0 || hidden_size == 0 || intermediate_size == 0 || layer_count == 0 || expert_count == 0 || experts_per_token == 0)
    {
        return Error{ErrorCode::InvalidModel, "MoeIR dimensions must be non-zero"};
    }
    if (layers.size() != layer_count)
        return Error{ErrorCode::InvalidModel, "MoeIR layer_count does not match layers"};
    return graph.validate();
}

} // namespace moe
} // namespace ncnn
