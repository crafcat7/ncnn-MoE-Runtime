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

static const char* layer_kind_name(LayerKind kind) noexcept
{
    switch (kind)
    {
    case LayerKind::TokenEmbedding:
        return "TokenEmbedding";
    case LayerKind::Attention:
        return "Attention";
    case LayerKind::Router:
        return "Router";
    case LayerKind::ExpertDispatch:
        return "ExpertDispatch";
    case LayerKind::ExpertGroup:
        return "ExpertGroup";
    case LayerKind::SharedExpertGroup:
        return "SharedExpertGroup";
    case LayerKind::Combine:
        return "Combine";
    case LayerKind::DenseFfn:
        return "DenseFfn";
    case LayerKind::FinalNorm:
        return "FinalNorm";
    case LayerKind::LmHead:
        return "LmHead";
    }
    return "Unknown";
}

// Tensor edges are the single source of truth for graph dependencies and
// ordering. Layers only carry runtime placement and plan bindings.
static Result<void> validate_graph(const Net& graph, std::vector<LayerId>& order)
{
    const std::vector<Blob>& tensors = graph.blobs;
    const std::vector<Layer>& nodes = graph.layers;
    const std::vector<CompiledLayerPlan>& layer_plans = graph.layer_plans;
    if (nodes.empty())
        return Error{ErrorCode::InvalidModel, "execution graph cannot be empty"};

    for (size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index)
    {
        const Blob& tensor = tensors[tensor_index];
        if (tensor.id != tensor_index)
        {
            return Error{ErrorCode::InvalidModel, "execution tensor ids must be contiguous and index-aligned"};
        }
        if (tensor.name.empty())
        {
            return Error{ErrorCode::InvalidModel, "execution tensor name cannot be empty"};
        }
        if (tensor.producer == invalid_layer_id || tensor.producer >= nodes.size())
        {
            return Error{ErrorCode::InvalidModel, "execution tensor must have a valid producer"};
        }
        for (size_t consumer_index = 0; consumer_index < tensor.consumers.size(); ++consumer_index)
        {
            const LayerId consumer = tensor.consumers[consumer_index];
            if (consumer >= nodes.size())
            {
                return Error{ErrorCode::InvalidModel, "execution tensor consumer is out of range"};
            }
            if (consumer == tensor.producer)
            {
                return Error{ErrorCode::InvalidModel, "execution tensor consumer cannot be its producer"};
            }
            if (std::find(tensor.consumers.begin(), tensor.consumers.begin() + consumer_index, consumer)
                != tensor.consumers.begin() + consumer_index)
            {
                return Error{ErrorCode::InvalidModel, "execution tensor contains a duplicate consumer"};
            }
        }
    }

    std::vector<uint32_t> indegrees(nodes.size(), 0);
    std::vector<std::vector<LayerId>> dependents(nodes.size());
    std::vector<uint8_t> referenced_layer_plans(layer_plans.size(),
                                                0);
    for (size_t node_index = 0; node_index < nodes.size(); ++node_index)
    {
        const Layer& node = nodes[node_index];
        if (node.id != node_index)
        {
            return Error{ErrorCode::InvalidModel, "execution graph node ids must be contiguous and index-aligned"};
        }
        const uint32_t selected_backend = node.backend == ExecutionBackend::Cpu ? ExecutionBackendCpu : ExecutionBackendVulkan;
        if (!has_flag(node.backend_mask, selected_backend))
        {
            return Error{ErrorCode::InvalidModel, "execution node selected backend is not allowed by its backend mask"};
        }
        if (has_flag(node.flags, LayerRequestExperts)
            && node.kind != LayerKind::ExpertDispatch)
        {
            return Error{ErrorCode::InvalidModel, "exact Expert reads must be scheduled by ExpertDispatch"};
        }
        const bool layer_bound = node.kind == LayerKind::Attention
                                 || node.kind == LayerKind::DenseFfn
                                 || node.kind == LayerKind::Router
                                 || node.kind == LayerKind::ExpertDispatch
                                 || node.kind == LayerKind::ExpertGroup
                                 || node.kind == LayerKind::SharedExpertGroup
                                 || node.kind == LayerKind::Combine;
        if (layer_bound
                ? node.plan_index == invalid_plan_index
                : node.plan_index != invalid_plan_index)
        {
            return Error{ErrorCode::InvalidModel, "execution graph node has an invalid layer plan binding"};
        }
        if (node.plan_index != invalid_plan_index
            && node.plan_index >= layer_plans.size())
        {
            return Error{ErrorCode::InvalidModel, "execution graph node layer plan is out of range"};
        }
        if (node.plan_index != invalid_plan_index)
            referenced_layer_plans[node.plan_index] = 1;
        const bool weight_bound = node.kind == LayerKind::TokenEmbedding
                                  || node.kind == LayerKind::FinalNorm
                                  || node.kind == LayerKind::LmHead;
        if (weight_bound && node.weights.empty())
        {
            return Error{
                ErrorCode::InvalidModel,
                "execution graph weight-bound node has no weight inputs"};
        }
        if (!weight_bound && !node.weights.empty())
        {
            return Error{
                ErrorCode::InvalidModel,
                "execution graph layer node cannot own weight inputs"};
        }
        for (TensorHandle weight : node.weights)
        {
            if (weight == invalid_tensor_handle)
            {
                return Error{
                    ErrorCode::InvalidModel,
                    "execution graph contains an invalid weight input"};
            }
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

    for (const Blob& tensor : tensors)
    {
        for (LayerId consumer : tensor.consumers)
        {
            dependents[tensor.producer].push_back(consumer);
            ++indegrees[consumer];
        }
    }

    order.clear();
    order.reserve(nodes.size());
    for (LayerId node_id = 0; node_id < nodes.size(); ++node_id)
    {
        if (indegrees[node_id] == 0)
            order.push_back(node_id);
    }

    for (size_t i = 0; i < order.size(); ++i)
    {
        for (LayerId dependent : dependents[order[i]])
        {
            if (--indegrees[dependent] == 0)
                order.push_back(dependent);
        }
    }
    if (order.size() != nodes.size())
        return Error{ErrorCode::InvalidModel, "execution graph contains a dependency cycle"};
    return {};
}

// Generated order must not overlap FFN workspaces. This is the only schedule
// constraint not guaranteed by the Blob topological order.
static Result<void> validate_ffn_grouping(const Net& graph)
{
    uint32_t active_ffn = invalid_plan_index;
    for (const LayerId node_id : graph.order)
    {
        const Layer& node = graph.layers[node_id];
        if (active_ffn != invalid_plan_index && node.plan_index != active_ffn)
            return Error{ErrorCode::InvalidModel, "execution schedule overlaps an unfinished FFN layer"};
        if (node.kind == LayerKind::Router)
        {
            if (active_ffn != invalid_plan_index)
                return Error{ErrorCode::InvalidModel, "execution schedule restarts an unfinished FFN layer"};
            active_ffn = node.plan_index;
        }
        else if (node.kind == LayerKind::Combine)
        {
            if (active_ffn == invalid_plan_index)
                return Error{ErrorCode::InvalidModel, "execution schedule combines without an FFN layer"};
            active_ffn = invalid_plan_index;
        }
    }
    if (active_ffn != invalid_plan_index)
        return Error{ErrorCode::InvalidModel, "execution schedule leaves an unfinished FFN layer"};
    return {};
}

Result<void> Net::validate_order() const
{
    if (layers.empty())
        return Error{ErrorCode::InvalidModel, "execution schedule cannot reference an empty graph"};
    if (order.size() != layers.size())
        return Error{ErrorCode::InvalidModel, "execution schedule node order is incomplete"};

    std::vector<uint32_t> positions(layers.size(), invalid_plan_index);
    for (size_t order_index = 0; order_index < order.size(); ++order_index)
    {
        const LayerId node_id = order[order_index];
        if (node_id >= layers.size())
            return Error{ErrorCode::InvalidModel, "execution schedule node order is out of range"};
        if (positions[node_id] != invalid_plan_index)
            return Error{ErrorCode::InvalidModel, "execution schedule contains a duplicate node"};
        positions[node_id] = static_cast<uint32_t>(order_index);
        const Layer& node = layers[node_id];
        const uint32_t selected_backend = node.backend == ExecutionBackend::Cpu ? ExecutionBackendCpu : ExecutionBackendVulkan;
        if (!has_flag(node.backend_mask, selected_backend))
        {
            return Error{ErrorCode::InvalidModel, "execution schedule node backend is not allowed by its backend mask"};
        }
    }

    auto ffn_valid = validate_ffn_grouping(*this);
    if (!ffn_valid)
        return ffn_valid.error();

    for (const Blob& tensor : blobs)
    {
        if (tensor.producer >= positions.size())
            return Error{ErrorCode::InvalidModel, "execution schedule tensor producer is out of range"};
        const uint32_t producer_position = positions[tensor.producer];
        for (LayerId consumer : tensor.consumers)
        {
            if (consumer >= positions.size())
                return Error{ErrorCode::InvalidModel, "execution schedule tensor consumer is out of range"};
            if (producer_position == invalid_plan_index
                || positions[consumer] == invalid_plan_index
                || positions[consumer] <= producer_position)
            {
                return Error{ErrorCode::InvalidModel, "execution schedule violates a tensor dependency"};
            }
        }
    }

    return {};
}

Result<void> schedule_graph(Net& graph, uint32_t available_backends)
{
    graph.order.clear();
    if (available_backends == 0)
        return Error{ErrorCode::InvalidArgument, "runtime scheduler requires at least one backend"};

    for (Layer& node : graph.layers)
    {
        const uint32_t usable = node.backend_mask & available_backends;
        if (usable == 0)
            return Error{ErrorCode::UnsupportedModel, "execution node has no available backend: " + std::string(layer_kind_name(node.kind)) + " #" + std::to_string(node.id)};

        const uint32_t selected = node.backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu;
        if (has_flag(usable, ExecutionBackendVulkan))
            node.backend = ExecutionBackend::Vulkan;
        else if (!has_flag(usable, selected))
            node.backend = has_flag(usable, ExecutionBackendCpu) ? ExecutionBackend::Cpu : ExecutionBackend::Vulkan;
    }

    auto ret = validate_graph(graph, graph.order);
    if (!ret)
        return ret.error();

    ret = validate_ffn_grouping(graph);
    if (!ret)
        return ret.error();
    return {};
}

static BlobId add_blob(Net& graph, std::string name, DType dtype, std::vector<uint32_t> shape,
                       BlobLifetime lifetime = BlobLifetime::Activation)
{
    const BlobId id = static_cast<BlobId>(graph.blobs.size());
    Blob& tensor = graph.blobs.emplace_back();
    tensor.id = id;
    tensor.name = std::move(name);
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    tensor.lifetime = lifetime;
    return id;
}

static LayerId add_node(Net& graph, LayerKind kind, ExecutionBackend backend, uint32_t backend_mask,
                        std::vector<BlobId> inputs, std::vector<BlobId> outputs,
                        uint32_t plan_index, uint32_t flags)
{
    const LayerId node_id = static_cast<LayerId>(graph.layers.size());
    Layer& node = graph.layers.emplace_back();
    node.id = node_id;
    node.kind = kind;
    node.backend = backend;
    node.backend_mask = backend_mask;
    node.plan_index = plan_index;
    node.flags = flags;
    for (BlobId input : inputs)
        graph.blobs[input].consumers.push_back(node_id);
    for (BlobId output : outputs)
        graph.blobs[output].producer = node_id;
    return node_id;
}

static void add_control_edge(Net& graph, std::string name, LayerId producer, LayerId consumer)
{
    const BlobId id = add_blob(graph, std::move(name), DType::Int32, {1}, BlobLifetime::Control);
    graph.blobs[id].producer = producer;
    graph.blobs[id].consumers.push_back(consumer);
}

static bool use_vulkan_attention(const CompiledModel& compiled,
                                 const AttentionBlockPlan& attention) noexcept
{
    if (compiled.opt.hybrid_mode == HybridMode::CpuOnly
        || !has_flag(compiled.opt.optimization_flags,
                     OptimizationVulkanAttention))
    {
        return false;
    }
    return compiled.gated_residual_chain || support_vulkan_attention(compiled.operators, attention);
}

static LayerId add_moe_nodes(Net& graph,
                             const CompiledModel& compiled,
                             size_t plan_index,
                             const std::string& prefix,
                             BlobId& hidden,
                             bool use_vulkan_experts)
{
    const CompiledLayerPlan& layer = graph.layer_plans[plan_index];
    const bool resident = &graph == &compiled.net && plan_index < compiled.gpu_segments.size()
                          && compiled.gpu_segments[plan_index];
    const BlobId router_scores = add_blob(graph,
                                          prefix + "router.scores",
                                          DType::Float32,
                                          {0, static_cast<uint32_t>(layer.moe.experts.size())});
    const LayerId router = add_node(graph,
                                    LayerKind::Router,
                                    resident ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu,
                                    resident ? ExecutionBackendVulkan : ExecutionBackendCpu,
                                    {hidden},
                                    {router_scores},
                                    static_cast<uint32_t>(plan_index),
                                    0);
    const BlobId assignments = add_blob(graph,
                                        prefix + "expert_dispatch.assignments",
                                        DType::Int32,
                                        {0, 3},
                                        BlobLifetime::Control);
    const LayerId dispatch = add_node(graph,
                                      LayerKind::ExpertDispatch,
                                      ExecutionBackend::Cpu,
                                      ExecutionBackendCpu,
                                      {router_scores},
                                      {assignments},
                                      static_cast<uint32_t>(plan_index),
                                      0);

    const bool can_use_vulkan_experts = compiled.opt.hybrid_mode != HybridMode::CpuOnly
                                        && use_vulkan_experts
                                        && support_vulkan_experts(compiled.weights, layer.moe,
                                                                  compiled.opt.optimization_flags);
    const ExecutionBackend expert_backend = can_use_vulkan_experts ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
    const uint32_t expert_backend_mask = can_use_vulkan_experts ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
    const uint32_t expert_flags = compiled.opt.hybrid_mode == HybridMode::HybridExperts
                                      ? LayerCpuPrefetch
                                      : 0u;
    const BlobId expert_output = add_blob(graph,
                                          prefix + "experts.output",
                                          compiled.descriptor.activation_dtype,
                                          {0, compiled.descriptor.hidden_size});
    const LayerId expert_group = add_node(graph,
                                          LayerKind::ExpertGroup,
                                          expert_backend,
                                          expert_backend_mask,
                                          {hidden, assignments},
                                          {expert_output},
                                          static_cast<uint32_t>(plan_index),
                                          expert_flags);

    const BlobId combined = add_blob(graph,
                                     prefix + "combine.hidden",
                                     compiled.descriptor.activation_dtype,
                                     {0, compiled.descriptor.hidden_size});
    std::vector<BlobId> combine_inputs = {hidden, expert_output};
    if (layer.moe.has_shared_expert)
    {
        const bool shared_before_routed = compiled.opt.hybrid_mode == HybridMode::CpuOnly;
        const bool shared_vulkan = compiled.opt.hybrid_mode != HybridMode::CpuOnly
                                   && support_vulkan_shared_experts(compiled.operators, layer.moe);
        const ExecutionBackend shared_backend = shared_vulkan ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
        const uint32_t shared_backend_mask = shared_vulkan ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
        const BlobId shared_output = add_blob(graph,
                                              prefix + "shared_experts.output",
                                              compiled.descriptor.activation_dtype,
                                              {0, compiled.descriptor.hidden_size});
        const LayerId shared_node = add_node(graph,
                                             LayerKind::SharedExpertGroup,
                                             shared_backend,
                                             shared_backend_mask,
                                             {hidden},
                                             {shared_output},
                                             static_cast<uint32_t>(plan_index),
                                             0);
        if (shared_before_routed)
        {
            graph.layers[dispatch].flags |= LayerRequestExperts;
            add_control_edge(graph, prefix + "dispatch.control", dispatch, shared_node);
            add_control_edge(graph, prefix + "shared_experts.control", shared_node, expert_group);
        }
        else
        {
            add_control_edge(graph, prefix + "shared_experts.control", expert_group, shared_node);
        }
        combine_inputs.push_back(shared_output);
    }
    const LayerId combine = add_node(graph,
                                     LayerKind::Combine,
                                     resident ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu,
                                     resident ? ExecutionBackendVulkan : ExecutionBackendCpu,
                                     std::move(combine_inputs),
                                     {combined},
                                     static_cast<uint32_t>(plan_index),
                                     0);
    hidden = combined;
    return combine;
}

static Result<void> build_speculative_graph(CompiledModel& compiled,
                                            bool use_vulkan_experts)
{
    if (!compiled.speculative.enabled())
        return {};

    Net& graph = compiled.speculative.net;
    graph.layers.clear();
    graph.blobs.clear();
    BlobId hidden = add_blob(graph,
                             "speculative.input.hidden",
                             compiled.descriptor.activation_dtype,
                             {0, compiled.descriptor.hidden_size});
    const LayerId token_embedding = add_node(graph,
                                             LayerKind::TokenEmbedding,
                                             ExecutionBackend::Cpu,
                                             ExecutionBackendCpu,
                                             {},
                                             {hidden},
                                             invalid_plan_index,
                                             0);
    graph.layers[token_embedding].weights = {compiled.token_embedding};

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
            const BlobId cache = add_blob(graph,
                                          prefix + "kv_cache",
                                          layer.attention.kind == AttentionKind::GatedDeltaNet ? DType::Float32 : compiled.descriptor.kv_cache_dtype,
                                          std::move(cache_shape),
                                          BlobLifetime::Session);
            const BlobId attention_output = add_blob(graph,
                                                     prefix + "attention.hidden",
                                                     compiled.descriptor.activation_dtype,
                                                     {0, compiled.descriptor.hidden_size});
            add_node(graph,
                     LayerKind::Attention,
                     backend,
                     backend_mask,
                     {hidden},
                     {attention_output, cache},
                     static_cast<uint32_t>(plan_index),
                     0);
            hidden = attention_output;
        }

        add_moe_nodes(graph, compiled, plan_index, prefix, hidden, use_vulkan_experts);
    }

    uint32_t available_backends = ExecutionBackendCpu;
    if (compiled.opt.hybrid_mode != HybridMode::CpuOnly)
        available_backends |= ExecutionBackendVulkan;
    auto schedule_status = schedule_graph(graph, available_backends);
    if (!schedule_status)
        return schedule_status.error();

    auto& layer_nodes = compiled.speculative.layer_nodes;
    layer_nodes.clear();
    layer_nodes.resize(graph.layer_plans.size());
    for (LayerId node_id : graph.order)
    {
        if (node_id >= graph.layers.size())
        {
            return Error{
                ErrorCode::InternalError,
                "speculative schedule references an invalid node"};
        }
        const Layer& node = graph.layers[node_id];
        if (node.plan_index == invalid_plan_index)
            continue;
        if (node.plan_index >= layer_nodes.size())
        {
            return Error{
                ErrorCode::InternalError,
                "speculative node layer binding is out of range"};
        }
        SpeculativeModelPlan::LayerNodes& layer = layer_nodes[node.plan_index];
        LayerId* target = nullptr;
        switch (node.kind)
        {
        case LayerKind::Attention:
            target = &layer.attention;
            break;
        case LayerKind::Router:
            target = &layer.router;
            break;
        case LayerKind::ExpertDispatch:
            target = &layer.expert_dispatch;
            break;
        case LayerKind::ExpertGroup:
            target = &layer.expert_group;
            break;
        case LayerKind::SharedExpertGroup:
            target = &layer.shared_expert_group;
            break;
        case LayerKind::Combine:
            target = &layer.combine;
            break;
        default:
            break;
        }
        if (target && *target != invalid_layer_id)
        {
            return Error{
                ErrorCode::InternalError,
                "speculative graph contains duplicate layer nodes"};
        }
        if (target)
            *target = node_id;
    }

    for (const SpeculativeModelPlan::LayerNodes& layer : layer_nodes)
    {
        if (layer.attention == invalid_layer_id
            || layer.router == invalid_layer_id
            || layer.expert_dispatch == invalid_layer_id
            || layer.expert_group == invalid_layer_id
            || layer.combine == invalid_layer_id)
        {
            return Error{
                ErrorCode::InternalError,
                "speculative graph is missing a required layer node"};
        }
    }
    return {};
}

Result<void> build_graph(CompiledModel& compiled, bool use_vulkan_experts)
{
    Net& graph = compiled.net;
    graph.layers.clear();
    graph.blobs.clear();
    BlobId hidden = add_blob(graph, "embedding.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size});
    const LayerId token_embedding = add_node(graph,
                                             LayerKind::TokenEmbedding,
                                             ExecutionBackend::Cpu,
                                             ExecutionBackendCpu,
                                             {},
                                             {hidden},
                                             invalid_plan_index,
                                             0);
    graph.layers[token_embedding].weights = {compiled.token_embedding};

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
            const BlobId cache = add_blob(graph,
                                          prefix + "kv_cache",
                                          gated_delta_attention ? DType::Float32 : compiled.descriptor.kv_cache_dtype,
                                          std::move(cache_shape),
                                          BlobLifetime::Session);
            const BlobId attention_output = add_blob(graph, prefix + "attention.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size});
            add_node(graph,
                     LayerKind::Attention,
                     backend,
                     backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu,
                     {hidden},
                     {attention_output, cache},
                     static_cast<uint32_t>(plan_index),
                     0);
            hidden = attention_output;
        }

        if (layer.dense_ffn.down_weight != invalid_tensor_handle)
        {
            const bool vulkan = compiled.opt.hybrid_mode != HybridMode::CpuOnly
                                && ((use_vulkan_experts && layer.dense_ffn.use_gpu_cache)
                                    || support_vulkan_dense_ffn(compiled.operators, layer.dense_ffn));
            const BlobId output = add_blob(graph, prefix + "ffn.hidden", compiled.descriptor.activation_dtype,
                                           {0, compiled.descriptor.hidden_size});
            add_node(graph,
                     LayerKind::DenseFfn,
                     vulkan ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu,
                     vulkan ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu,
                     {hidden},
                     {output},
                     static_cast<uint32_t>(plan_index),
                     0);
            hidden = output;
        }
        else
        {
            add_moe_nodes(graph, compiled, plan_index, prefix, hidden, use_vulkan_experts);
        }
    }

    const BlobId normalized = add_blob(graph, "final_norm.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size});
    const LayerId final_norm = add_node(graph,
                                        LayerKind::FinalNorm,
                                        compiled.gpu_head ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu,
                                        compiled.gpu_head ? ExecutionBackendVulkan : ExecutionBackendCpu,
                                        {hidden},
                                        {normalized},
                                        invalid_plan_index,
                                        0);
    graph.layers[final_norm].weights = compiled.final_norm_weight == invalid_tensor_handle
                                           ? std::vector<TensorHandle>{compiled.lm_head_weight}
                                           : std::vector<TensorHandle>{compiled.final_norm_weight,
                                                                       compiled.lm_head_weight};

    const CompiledOperator& lm_head_operator = compiled.operators.at_weight(compiled.lm_head_weight);
    const bool vulkan_lm_head = compiled.opt.hybrid_mode != HybridMode::CpuOnly
                                && (lm_head_operator.linear || lm_head_operator.bfloat16
                                    || lm_head_operator.float8 || lm_head_operator.qnk);
    const ExecutionBackend lm_head_backend = vulkan_lm_head ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
    const BlobId logits = add_blob(graph, "logits", DType::Float32, {0, compiled.descriptor.vocabulary_size});
    const LayerId lm_head_node = add_node(graph,
                                          LayerKind::LmHead,
                                          lm_head_backend,
                                          lm_head_backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu,
                                          {normalized},
                                          {logits},
                                          invalid_plan_index,
                                          0);
    graph.layers[lm_head_node].weights = compiled.final_norm_weight == invalid_tensor_handle
                                             ? std::vector<TensorHandle>{compiled.lm_head_weight}
                                             : std::vector<TensorHandle>{compiled.lm_head_weight,
                                                                         compiled.final_norm_weight};

    uint32_t available_backends = ExecutionBackendCpu;
    if (compiled.opt.hybrid_mode != HybridMode::CpuOnly)
        available_backends |= ExecutionBackendVulkan;
    auto schedule_status = schedule_graph(graph, available_backends);
    if (!schedule_status)
        return schedule_status.error();

    return build_speculative_graph(compiled, use_vulkan_experts);
}

} // namespace moe
} // namespace ncnn
