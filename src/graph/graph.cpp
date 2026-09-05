#include "graph.h"

#include "backends/ncnn/modelpipeline.h"
#include "compiledmodel.h"
#include "ncnn/moe/option.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

Result<void> ExecutionGraph::validate() const
{
    if (nodes.empty())
        return Error{ErrorCode::InvalidModel, "execution graph cannot be empty"};

    for (size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index)
    {
        const ExecutionTensor& tensor = tensors[tensor_index];
        if (tensor.id != tensor_index)
        {
            return Error{ErrorCode::InvalidModel, "execution tensor ids must be contiguous and index-aligned"};
        }
        if (tensor.name.empty())
        {
            return Error{ErrorCode::InvalidModel, "execution tensor name cannot be empty"};
        }
        if (tensor.producer == invalid_execution_node_id || tensor.producer >= nodes.size())
        {
            return Error{ErrorCode::InvalidModel, "execution tensor must have a valid producer"};
        }
        for (ExecutionNodeId consumer : tensor.consumers)
        {
            if (consumer >= nodes.size())
            {
                return Error{ErrorCode::InvalidModel, "execution tensor consumer is out of range"};
            }
        }
    }

    std::vector<uint32_t> indegrees(nodes.size(), 0);
    std::vector<std::vector<ExecutionNodeId>> dependents(nodes.size());
    std::vector<uint8_t> referenced_layer_plans(
        layer_plans.size(),
        0);
    for (size_t node_index = 0; node_index < nodes.size(); ++node_index)
    {
        const ExecutionNode& node = nodes[node_index];
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
        for (auto it = node.dependencies.begin(); it != node.dependencies.end(); ++it)
        {
            const ExecutionNodeId dependency = *it;
            if (dependency >= nodes.size())
                return Error{ErrorCode::InvalidModel, "execution graph dependency is out of range"};
            if (dependency == node.id)
                return Error{ErrorCode::InvalidModel, "execution graph node cannot depend on itself"};
            if (std::find(node.dependencies.begin(), it, dependency) != it)
            {
                return Error{ErrorCode::InvalidModel, "execution graph contains a duplicate dependency"};
            }
            dependents[dependency].push_back(node.id);
            ++indegrees[node_index];
        }
        for (ExecutionTensorId input : node.inputs)
        {
            if (input >= tensors.size())
            {
                return Error{ErrorCode::InvalidModel, "execution node input tensor is out of range"};
            }
            const ExecutionTensor& tensor = tensors[input];
            if (std::find(tensor.consumers.begin(), tensor.consumers.end(), node.id) == tensor.consumers.end())
            {
                return Error{ErrorCode::InvalidModel, "execution input tensor does not list its consumer"};
            }
        }
        for (ExecutionTensorId output : node.outputs)
        {
            if (output >= tensors.size())
            {
                return Error{ErrorCode::InvalidModel, "execution node output tensor is out of range"};
            }
            if (tensors[output].producer != node.id)
            {
                return Error{ErrorCode::InvalidModel, "execution output tensor does not list its producer"};
            }
        }
        const bool layer_bound = node.type == ExecutionNodeType::Attention
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
            && node.layer_plan_index >= layer_plans.size())
        {
            return Error{ErrorCode::InvalidModel, "execution graph node layer plan is out of range"};
        }
        if (node.layer_plan_index != invalid_execution_layer_id)
            referenced_layer_plans[node.layer_plan_index] = 1;
        const bool weight_bound = node.type == ExecutionNodeType::TokenEmbedding
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

    std::vector<ExecutionNodeId> ready;
    ready.reserve(nodes.size());
    for (ExecutionNodeId node_id = 0; node_id < nodes.size(); ++node_id)
    {
        if (indegrees[node_id] == 0)
            ready.push_back(node_id);
    }

    for (size_t i = 0; i < ready.size(); ++i)
    {
        for (ExecutionNodeId dependent : dependents[ready[i]])
        {
            if (--indegrees[dependent] == 0)
                ready.push_back(dependent);
        }
    }
    if (ready.size() != nodes.size())
        return Error{ErrorCode::InvalidModel, "execution graph contains a dependency cycle"};
    return {};
}

Result<void> ExecutionSchedule::validate(const ExecutionGraph& graph) const
{
    if (graph.nodes.empty())
        return Error{ErrorCode::InvalidModel, "execution schedule cannot reference an empty graph"};
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
    return {};
}

Result<ExecutionSchedule> schedule_graph(ExecutionGraph& graph, const GraphOption& opt)
{
    if (opt.available_backends == 0)
        return Error{ErrorCode::InvalidArgument, "runtime scheduler requires at least one backend"};

    for (ExecutionNode& node : graph.nodes)
    {
        const uint32_t usable = node.backend_mask & opt.available_backends;
        if (usable == 0)
            return Error{ErrorCode::UnsupportedModel, "execution node has no available backend: " + node.name};

        const uint32_t selected = node.backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu;
        const bool dense = node.type == ExecutionNodeType::Attention || node.type == ExecutionNodeType::LmHead;
        if (opt.prefer_vulkan_dense && dense && has_flag(usable, ExecutionBackendVulkan))
            node.backend = ExecutionBackend::Vulkan;
        else if (!has_flag(usable, selected))
            node.backend = has_flag(usable, ExecutionBackendCpu) ? ExecutionBackend::Cpu : ExecutionBackend::Vulkan;
    }

    auto ret = graph.validate();
    if (!ret)
        return ret.error();

    std::vector<uint32_t> indegrees(graph.nodes.size(), 0);
    std::vector<std::vector<ExecutionNodeId>> dependents(graph.nodes.size());
    for (const ExecutionNode& node : graph.nodes)
    {
        indegrees[node.id] = static_cast<uint32_t>(node.dependencies.size());
        for (ExecutionNodeId dependency : node.dependencies)
            dependents[dependency].push_back(node.id);
    }

    ExecutionSchedule schedule;
    schedule.node_order.reserve(graph.nodes.size());
    for (ExecutionNodeId node_id = 0; node_id < graph.nodes.size(); ++node_id)
    {
        if (indegrees[node_id] == 0)
            schedule.node_order.push_back(node_id);
    }
    // Appending ready nodes keeps the same stable breadth-first order.
    for (size_t i = 0; i < schedule.node_order.size(); ++i)
    {
        for (ExecutionNodeId dependent : dependents[schedule.node_order[i]])
        {
            if (--indegrees[dependent] == 0)
                schedule.node_order.push_back(dependent);
        }
    }

    schedule.backend_runs.reserve(schedule.node_order.size());
    for (uint32_t i = 0; i < schedule.node_order.size(); ++i)
    {
        const ExecutionBackend backend = graph.nodes[schedule.node_order[i]].backend;
        if (schedule.backend_runs.empty() || schedule.backend_runs.back().backend != backend)
            schedule.backend_runs.push_back({backend, i, 1});
        else
            ++schedule.backend_runs.back().node_count;
    }
    ret = schedule.validate(graph);
    if (!ret)
        return ret.error();
    return schedule;
}

static ExecutionTensorId add_tensor(ExecutionGraph& graph, std::string name, DType dtype, std::vector<uint32_t> shape)
{
    const ExecutionTensorId id = static_cast<ExecutionTensorId>(graph.tensors.size());
    ExecutionTensor tensor;
    tensor.id = id;
    tensor.name = std::move(name);
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    graph.tensors.push_back(std::move(tensor));
    return id;
}

static ExecutionNodeId add_node(ExecutionGraph& graph, ExecutionNodeType type, ExecutionBackend backend, uint32_t backend_mask, std::string name, std::vector<ExecutionNodeId> dependencies, std::vector<ExecutionTensorId> inputs,
                                std::vector<ExecutionTensorId> outputs, uint32_t layer_plan_index, uint32_t expert_id, uint32_t flags)
{
    const ExecutionNodeId node_id = static_cast<ExecutionNodeId>(graph.nodes.size());
    ExecutionNode node;
    node.id = node_id;
    node.type = type;
    node.backend = backend;
    node.backend_mask = backend_mask;
    node.layer_plan_index = layer_plan_index;
    node.expert_id = expert_id;
    node.flags = flags;
    node.name = std::move(name);
    node.dependencies = std::move(dependencies);
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    graph.nodes.push_back(std::move(node));
    for (ExecutionTensorId input : graph.nodes.back().inputs)
        graph.tensors[input].consumers.push_back(node_id);
    for (ExecutionTensorId output : graph.nodes.back().outputs)
        graph.tensors[output].producer = node_id;
    return node_id;
}

static bool use_vulkan_attention(
    const CompiledModel& compiled,
    const AttentionBlockPlan& attention) noexcept
{
    if (compiled.opt.hybrid_mode == HybridMode::CpuOnly
        || !has_flag(
            compiled.opt.optimization_flags,
            OptimizationVulkanAttention))
    {
        return false;
    }
    return support_vulkan_attention(compiled.operators, attention);
}

static ExecutionNodeId add_moe_nodes(
    ExecutionGraph& graph,
    const CompiledModel& compiled,
    size_t plan_index,
    const std::string& prefix,
    ExecutionNodeId previous,
    ExecutionTensorId& hidden,
    bool use_vulkan_experts)
{
    const CompiledLayerPlan& layer = graph.layer_plans[plan_index];
    const ExecutionTensorId router_scores = add_tensor(
        graph,
        prefix + "router.scores",
        DType::Float32,
        {0, static_cast<uint32_t>(layer.moe.experts.size())});
    const ExecutionNodeId router = add_node(
        graph,
        ExecutionNodeType::Router,
        ExecutionBackend::Cpu,
        ExecutionBackendCpu,
        prefix + "router",
        {previous},
        {hidden},
        {router_scores},
        static_cast<uint32_t>(plan_index),
        invalid_execution_expert_id,
        0);
    const ExecutionTensorId assignments = add_tensor(
        graph,
        prefix + "expert_dispatch.assignments",
        DType::Int32,
        {0, 3});
    const ExecutionNodeId dispatch = add_node(
        graph,
        ExecutionNodeType::ExpertDispatch,
        ExecutionBackend::Cpu,
        ExecutionBackendCpu,
        prefix + "expert_dispatch",
        {router},
        {router_scores},
        {assignments},
        static_cast<uint32_t>(plan_index),
        invalid_execution_expert_id,
        0);

    const bool can_use_vulkan_experts = compiled.opt.hybrid_mode != HybridMode::CpuOnly
                                        && use_vulkan_experts
                                        && support_vulkan_experts(compiled.weights, layer.moe,
                                                                  compiled.opt.optimization_flags);
    const ExecutionBackend expert_backend = can_use_vulkan_experts ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
    const uint32_t expert_backend_mask = can_use_vulkan_experts ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
    const uint32_t expert_flags = compiled.opt.hybrid_mode == HybridMode::HybridExperts
                                      ? ExecutionNodeCpuPrefetch
                                      : 0u;
    const ExecutionTensorId expert_output = add_tensor(
        graph,
        prefix + "experts.output",
        compiled.descriptor.activation_dtype,
        {0, compiled.descriptor.hidden_size});
    const ExecutionNodeId expert_group = add_node(
        graph,
        ExecutionNodeType::ExpertGroup,
        expert_backend,
        expert_backend_mask,
        prefix + "experts",
        {dispatch},
        {hidden, assignments},
        {expert_output},
        static_cast<uint32_t>(plan_index),
        invalid_execution_expert_id,
        expert_flags);

    const ExecutionTensorId combined = add_tensor(
        graph,
        prefix + "combine.hidden",
        compiled.descriptor.activation_dtype,
        {0, compiled.descriptor.hidden_size});
    std::vector<ExecutionTensorId> combine_inputs = {hidden, expert_output};
    std::vector<ExecutionNodeId> combine_dependencies = {expert_group};
    if (layer.moe.has_shared_expert)
    {
        const bool shared_vulkan = compiled.opt.hybrid_mode != HybridMode::CpuOnly
                                   && support_vulkan_shared_experts(compiled.operators, layer.moe);
        const ExecutionBackend shared_backend = shared_vulkan ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
        const uint32_t shared_backend_mask = shared_vulkan ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
        const ExecutionTensorId shared_output = add_tensor(
            graph,
            prefix + "shared_experts.output",
            compiled.descriptor.activation_dtype,
            {0, compiled.descriptor.hidden_size});
        const ExecutionNodeId shared_node = add_node(
            graph,
            ExecutionNodeType::SharedExpertGroup,
            shared_backend,
            shared_backend_mask,
            prefix + "shared_experts",
            {expert_group},
            {hidden},
            {shared_output},
            static_cast<uint32_t>(plan_index),
            invalid_execution_expert_id,
            0);
        combine_dependencies.push_back(shared_node);
        combine_inputs.push_back(shared_output);
    }
    const ExecutionNodeId combine = add_node(
        graph,
        ExecutionNodeType::Combine,
        ExecutionBackend::Cpu,
        ExecutionBackendCpu,
        prefix + "combine",
        std::move(combine_dependencies),
        std::move(combine_inputs),
        {combined},
        static_cast<uint32_t>(plan_index),
        invalid_execution_expert_id,
        0);
    hidden = combined;
    return combine;
}

static Result<void> build_speculative_graph(
    CompiledModel& compiled,
    bool use_vulkan_experts)
{
    if (!compiled.speculative.enabled())
        return {};

    ExecutionGraph graph;
    graph.layer_plans = std::move(compiled.speculative.graph.layer_plans);
    ExecutionTensorId hidden = add_tensor(
        graph,
        "speculative.input.hidden",
        compiled.descriptor.activation_dtype,
        {0, compiled.descriptor.hidden_size});
    ExecutionNodeId previous = add_node(
        graph,
        ExecutionNodeType::TokenEmbedding,
        ExecutionBackend::Cpu,
        ExecutionBackendCpu,
        "speculative.input",
        {},
        {},
        {hidden},
        invalid_execution_layer_id,
        invalid_execution_expert_id,
        0);
    graph.nodes[previous].weight_inputs = {compiled.token_embedding};

    for (size_t plan_index = 0; plan_index < graph.layer_plans.size(); ++plan_index)
    {
        const CompiledLayerPlan& layer = graph.layer_plans[plan_index];
        const std::string prefix = "speculative.layers." + std::to_string(plan_index) + ".";
        if (layer.attention.kind != AttentionKind::None)
        {
            const bool supports_vulkan_attention = use_vulkan_attention(compiled, layer.attention);
            const ExecutionBackend backend = supports_vulkan_attention ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
            const uint32_t backend_mask = supports_vulkan_attention ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
            std::vector<uint32_t> cache_shape{
                0,
                layer.attention.kv_head_count,
                layer.attention.head_dimension,
            };
            if (layer.attention.kind == AttentionKind::GatedDeltaNet)
            {
                cache_shape = {
                    0,
                    layer.attention.head_count,
                    layer.attention.head_dimension,
                    layer.attention.value_head_dimension,
                };
            }
            const ExecutionTensorId cache = add_tensor(
                graph,
                prefix + "kv_cache",
                layer.attention.kind == AttentionKind::GatedDeltaNet ? DType::Float32 : compiled.descriptor.kv_cache_dtype,
                std::move(cache_shape));
            const ExecutionTensorId attention_output = add_tensor(
                graph,
                prefix + "attention.hidden",
                compiled.descriptor.activation_dtype,
                {0, compiled.descriptor.hidden_size});
            previous = add_node(
                graph,
                ExecutionNodeType::Attention,
                backend,
                backend_mask,
                prefix + "attention",
                {previous},
                {hidden},
                {attention_output, cache},
                static_cast<uint32_t>(plan_index),
                invalid_execution_expert_id,
                0);
            hidden = attention_output;
        }

        previous = add_moe_nodes(
            graph, compiled, plan_index, prefix, previous, hidden,
            use_vulkan_experts);
    }

    GraphOption options;
    options.available_backends = ExecutionBackendCpu;
    if (compiled.opt.hybrid_mode != HybridMode::CpuOnly)
        options.available_backends |= ExecutionBackendVulkan;
    auto schedule = schedule_graph(graph, options);
    if (!schedule)
        return schedule.error();
    compiled.speculative.graph = std::move(graph);
    compiled.speculative.schedule = std::move(schedule).value();
    return {};
}

Result<void> build_graph(CompiledModel& compiled, bool use_vulkan_experts)
{
    ExecutionGraph graph;
    graph.layer_plans = std::move(compiled.graph.layer_plans);
    ExecutionTensorId hidden = add_tensor(graph, "embedding.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size});
    ExecutionNodeId previous = add_node(graph, ExecutionNodeType::TokenEmbedding, ExecutionBackend::Cpu, ExecutionBackendCpu, "token_embedding", {}, {}, {hidden}, invalid_execution_layer_id, invalid_execution_expert_id, 0);
    graph.nodes[previous].weight_inputs = {compiled.token_embedding};

    for (size_t plan_index = 0; plan_index < graph.layer_plans.size(); ++plan_index)
    {
        const CompiledLayerPlan& layer = graph.layer_plans[plan_index];
        const std::string prefix = "layers." + std::to_string(layer.layer_id) + ".";
        if (layer.attention.kind != AttentionKind::None)
        {
            const bool gated_delta_attention = layer.attention.kind == AttentionKind::GatedDeltaNet;
            const bool supports_vulkan_attention = layer.layer_id < compiled.descriptor.layers.size()
                                                   && use_vulkan_attention(compiled, layer.attention);
            const ExecutionBackend backend = supports_vulkan_attention ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
            std::vector<uint32_t> cache_shape{
                0,
                layer.attention.kv_head_count,
                layer.attention.head_dimension,
            };
            if (gated_delta_attention)
            {
                cache_shape = {
                    0,
                    layer.attention.head_count,
                    layer.attention.head_dimension,
                    layer.attention.value_head_dimension,
                };
            }
            const ExecutionTensorId cache = add_tensor(
                graph,
                prefix + "kv_cache",
                gated_delta_attention ? DType::Float32 : compiled.descriptor.kv_cache_dtype,
                std::move(cache_shape));
            const ExecutionTensorId attention_output = add_tensor(graph, prefix + "attention.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size});
            previous = add_node(
                graph, ExecutionNodeType::Attention, backend, backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu, prefix + "attention",
                {previous}, {hidden}, {attention_output, cache},
                static_cast<uint32_t>(plan_index), invalid_execution_expert_id, 0);
            hidden = attention_output;
        }

        previous = add_moe_nodes(
            graph, compiled, plan_index, prefix, previous, hidden,
            use_vulkan_experts);
    }

    const ExecutionTensorId normalized = add_tensor(graph, "final_norm.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size});
    previous = add_node(
        graph, ExecutionNodeType::FinalNorm, ExecutionBackend::Cpu, ExecutionBackendCpu, "final_norm",
        {previous}, {hidden}, {normalized},
        invalid_execution_layer_id, invalid_execution_expert_id, 0);
    graph.nodes[previous].weight_inputs = compiled.final_norm_weight == invalid_tensor_handle
                                              ? std::vector<TensorHandle>{compiled.lm_head_weight}
                                              : std::vector<TensorHandle>{compiled.final_norm_weight,
                                                                          compiled.lm_head_weight};

    const ExecutionBackend lm_head_backend = compiled.opt.hybrid_mode == HybridMode::CpuOnly ? ExecutionBackend::Cpu : ExecutionBackend::Vulkan;
    const ExecutionTensorId logits = add_tensor(graph, "logits", DType::Float32, {0, compiled.descriptor.vocabulary_size});
    const ExecutionNodeId lm_head_node = add_node(
        graph, ExecutionNodeType::LmHead, lm_head_backend, lm_head_backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu, "lm_head",
        {previous}, {normalized}, {logits},
        invalid_execution_layer_id, invalid_execution_expert_id, 0);
    graph.nodes[lm_head_node].weight_inputs = compiled.final_norm_weight == invalid_tensor_handle
                                                  ? std::vector<TensorHandle>{compiled.lm_head_weight}
                                                  : std::vector<TensorHandle>{compiled.lm_head_weight,
                                                                              compiled.final_norm_weight};

    GraphOption options;
    options.available_backends = ExecutionBackendCpu;
    if (compiled.opt.hybrid_mode != HybridMode::CpuOnly)
        options.available_backends |= ExecutionBackendVulkan;
    auto schedule = schedule_graph(graph, options);
    if (!schedule)
        return schedule.error();
    compiled.graph = std::move(graph);
    compiled.schedule = std::move(schedule).value();
    return build_speculative_graph(compiled, use_vulkan_experts);
}

} // namespace moe
} // namespace ncnn
