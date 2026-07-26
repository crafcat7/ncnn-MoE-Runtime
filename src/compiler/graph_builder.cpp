#include "compiler/moe_ir.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static QuantConfig quant_config_for_dtype(DType dtype)
{
    QuantConfig config;
    config.storage_dtype = dtype;
    config.compute_dtype = DType::Float32;
    if (dtype == DType::Int8)
    {
        config.scheme = QuantizationScheme::PerChannel;
        config.group_size = 1;
    }
    else if (dtype == DType::MxFp4)
    {
        config.scheme = QuantizationScheme::BlockWise;
        config.block_size = 32;
    }
    return config;
}

static MoeIRValueId append_moe_ir_value(MoeGraph& graph, std::string name, DType dtype, std::vector<uint32_t> shape, TensorLocation preferred_location, uint32_t flags)
{
    const MoeIRValueId id = static_cast<MoeIRValueId>(graph.values.size());
    MoeIRValue value;
    value.id = id;
    value.name = std::move(name);
    value.dtype = dtype;
    value.shape = std::move(shape);
    value.preferred_location = preferred_location;
    value.flags = flags;
    graph.values.push_back(std::move(value));
    return id;
}

static MoeIRNodeId append_moe_ir_node(MoeGraph& graph, MoeIROperator operation, std::string name, uint32_t layer_id, std::vector<MoeIRValueId> inputs, std::vector<MoeIRValueId> outputs)
{
    const MoeIRNodeId id = static_cast<MoeIRNodeId>(graph.nodes.size());
    MoeIRNode node;
    node.id = id;
    node.operation = operation;
    node.name = std::move(name);
    node.layer_id = layer_id;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    graph.nodes.push_back(std::move(node));
    return id;
}

Result<MoeGraph> MoeGraphBuilder::build(const MoeModelDescriptor& descriptor) const
{
    if (descriptor.layers.size() != descriptor.layer_count)
    {
        return Error{ErrorCode::InvalidModel, "cannot build MoeIR graph when layer_count does not match layers"};
    }

    MoeGraph graph;
    const MoeIRValueId token_ids = append_moe_ir_value(graph, "token_ids", DType::Int32, {0}, TensorLocation::Cpu, MoeIRValueGraphInput | MoeIRValueDynamicShape);
    MoeIRValueId hidden = append_moe_ir_value(graph, "embedding.hidden", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Cpu, MoeIRValueDynamicShape);
    (void)append_moe_ir_node(graph, MoeIROperator::TokenEmbedding, "token_embedding", invalid_moe_ir_layer_id, {token_ids}, {hidden});

    for (uint32_t layer_id = 0; layer_id < descriptor.layer_count; ++layer_id)
    {
        const LayerDescriptor& layer = descriptor.layers[layer_id];
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";
        if (has_flag(layer.flags, LayerDescriptorAttention))
        {
            const MoeIRValueId cache = append_moe_ir_value(graph, prefix + "kv_cache", descriptor.kv_cache_dtype,
                                                           {
                                                               0,
                                                               layer.attention.kv_head_count,
                                                               layer.attention.head_dimension,
                                                           },
                                                           TensorLocation::Automatic, MoeIRValuePersistent | MoeIRValueMutableState | MoeIRValueDynamicShape);
            const MoeIRNodeId cache_node = append_moe_ir_node(graph, MoeIROperator::KvCache, prefix + "kv_cache", layer_id, {}, {cache});
            graph.nodes[cache_node].attention = layer.attention;
            graph.nodes[cache_node].flags |= MoeIRNodeStateful;

            const MoeIRValueId attention_output = append_moe_ir_value(graph, prefix + "attention.hidden", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Automatic, MoeIRValueDynamicShape);
            const MoeIRNodeId attention_node = append_moe_ir_node(graph, MoeIROperator::Attention, prefix + "attention", layer_id, {hidden, cache}, {attention_output});
            graph.nodes[attention_node].attention = layer.attention;
            graph.nodes[attention_node].flags |= MoeIRNodeStateful;
            hidden = attention_output;
        }

        if (has_flag(layer.flags, LayerDescriptorMoe))
        {
            const MoeDescriptor& moe = layer.ffn.moe;
            const MoeIRValueId router_scores = append_moe_ir_value(graph, prefix + "router.scores", DType::Float32, {0, moe.expert_count}, TensorLocation::Cpu, MoeIRValueDynamicShape);
            (void)append_moe_ir_node(graph, MoeIROperator::Router, prefix + "router", layer_id, {hidden}, {router_scores});

            const MoeIRValueId expert_output = append_moe_ir_value(graph, prefix + "experts.output", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Cpu, MoeIRValueDynamicShape);
            const MoeIRNodeId expert_node = append_moe_ir_node(graph, MoeIROperator::ExpertGroup, prefix + "experts", layer_id, {hidden, router_scores}, {expert_output});
            graph.nodes[expert_node].experts = moe;
            graph.nodes[expert_node].quantization = quant_config_for_dtype(moe.expert_weight_dtype);

            std::vector<MoeIRValueId> combine_inputs{
                hidden,
                expert_output,
            };
            if (moe.shared_expert_count != 0)
            {
                const MoeIRValueId shared_output = append_moe_ir_value(graph, prefix + "shared_experts.output", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Cpu, MoeIRValueDynamicShape);
                const MoeIRNodeId shared_node = append_moe_ir_node(graph, MoeIROperator::SharedExpertGroup, prefix + "shared_experts", layer_id, {hidden}, {shared_output});
                graph.nodes[shared_node].experts = moe;
                graph.nodes[shared_node].quantization = quant_config_for_dtype(moe.expert_weight_dtype);
                combine_inputs.push_back(shared_output);
            }

            const MoeIRValueId combined = append_moe_ir_value(graph, prefix + "combine.hidden", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Cpu, MoeIRValueDynamicShape);
            (void)append_moe_ir_node(graph, MoeIROperator::Combine, prefix + "combine", layer_id, std::move(combine_inputs), {combined});
            hidden = combined;
        }
        else if (has_flag(layer.flags, LayerDescriptorDenseFfn))
        {
            const MoeIRValueId dense_output = append_moe_ir_value(graph, prefix + "dense_ffn.hidden", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Automatic, MoeIRValueDynamicShape);
            const MoeIRNodeId dense_node = append_moe_ir_node(graph, MoeIROperator::DenseFfn, prefix + "dense_ffn", layer_id, {hidden}, {dense_output});
            graph.nodes[dense_node].intermediate_size = layer.ffn.dense_intermediate_size;
            hidden = dense_output;
        }
        else
        {
            return Error{ErrorCode::InvalidModel, "MoeIR layer requires either a dense FFN or MoE"};
        }
    }

    const MoeIRValueId normalized = append_moe_ir_value(graph, "final_norm.hidden", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Cpu, MoeIRValueDynamicShape);
    (void)append_moe_ir_node(graph, MoeIROperator::FinalNorm, "final_norm", invalid_moe_ir_layer_id, {hidden}, {normalized});

    const MoeIRValueId logits = append_moe_ir_value(graph, "logits", DType::Float32, {0, descriptor.vocabulary_size}, TensorLocation::Cpu, MoeIRValueGraphOutput | MoeIRValueDynamicShape);
    (void)append_moe_ir_node(graph, MoeIROperator::LmHead, "lm_head", invalid_moe_ir_layer_id, {normalized}, {logits});
    graph.outputs.push_back(logits);

    auto valid = graph.validate();
    if (!valid)
        return valid.error();
    return graph;
}

static void append_attention_descriptor_nodes(LayerDescriptor& layer)
{
    layer.nodes.push_back({ModelNodeType::RmsNorm});
    layer.nodes.push_back({ModelNodeType::FusedQkv});
    layer.nodes.push_back({ModelNodeType::Rope});
    if (has_flag(layer.attention.flags, AttentionDescriptorSinks))
    {
        layer.nodes.push_back({ModelNodeType::AttentionSink});
    }
    layer.nodes.push_back({ModelNodeType::Sdpa});
    layer.nodes.push_back({ModelNodeType::Projection});
}

static void append_expert_descriptor_nodes(LayerDescriptor& layer)
{
    layer.nodes.push_back({ModelNodeType::RmsNorm});
    layer.nodes.push_back({ModelNodeType::Router});
    layer.nodes.push_back({ModelNodeType::TopK});
    layer.nodes.push_back({ModelNodeType::ExpertGroup});
    if (layer.ffn.moe.shared_expert_count != 0)
    {
        layer.nodes.push_back({ModelNodeType::SharedExpertGroup});
    }
    layer.nodes.push_back({ModelNodeType::Combine});
}

static void append_dense_ffn_descriptor_nodes(LayerDescriptor& layer)
{
    layer.nodes.push_back({ModelNodeType::RmsNorm});
    layer.nodes.push_back({ModelNodeType::DenseFfn});
}

static Result<void> materialize_layers_from_graph(MoeIR& ir)
{
    ir.layers.resize(ir.layer_count);
    std::vector<bool> attention_seen(ir.layer_count, false);
    std::vector<bool> experts_seen(ir.layer_count, false);
    std::vector<bool> dense_ffn_seen(ir.layer_count, false);
    for (const MoeIRNode& node : ir.graph.nodes)
    {
        if (node.layer_id == invalid_moe_ir_layer_id)
            continue;
        if (node.layer_id >= ir.layer_count)
        {
            return Error{ErrorCode::InvalidModel, "MoeIR node layer is out of range"};
        }
        LayerDescriptor& layer = ir.layers[node.layer_id];
        if (node.operation == MoeIROperator::Attention)
        {
            if (attention_seen[node.layer_id])
            {
                return Error{ErrorCode::InvalidModel, "MoeIR layer contains duplicate Attention"};
            }
            attention_seen[node.layer_id] = true;
            layer.flags |= LayerDescriptorAttention;
            layer.pre_attention_norm = NormType::RmsNorm;
            layer.attention = node.attention;
        }
        else if (node.operation == MoeIROperator::ExpertGroup)
        {
            if (experts_seen[node.layer_id])
            {
                return Error{ErrorCode::InvalidModel, "MoeIR layer contains duplicate ExpertGroup"};
            }
            experts_seen[node.layer_id] = true;
            layer.flags |= LayerDescriptorMoe;
            layer.pre_ffn_norm = NormType::RmsNorm;
            layer.ffn.moe = node.experts;
        }
        else if (node.operation == MoeIROperator::DenseFfn)
        {
            if (dense_ffn_seen[node.layer_id])
            {
                return Error{ErrorCode::InvalidModel, "MoeIR layer contains duplicate DenseFfn"};
            }
            dense_ffn_seen[node.layer_id] = true;
            layer.flags |= LayerDescriptorDenseFfn;
            layer.pre_ffn_norm = NormType::RmsNorm;
            layer.ffn.dense_intermediate_size = node.intermediate_size;
        }
    }
    for (uint32_t layer_id = 0; layer_id < ir.layer_count; ++layer_id)
    {
        if (experts_seen[layer_id] == dense_ffn_seen[layer_id])
        {
            return Error{ErrorCode::InvalidModel, "MoeIR layer requires exactly one FFN kind"};
        }
        LayerDescriptor& layer = ir.layers[layer_id];
        if (attention_seen[layer_id])
            append_attention_descriptor_nodes(layer);
        if (experts_seen[layer_id])
            append_expert_descriptor_nodes(layer);
        else
            append_dense_ffn_descriptor_nodes(layer);
    }
    return {};
}

static bool attention_matches(const AttentionDescriptor& left, const AttentionDescriptor& right)
{
    return left.kind == right.kind
           && left.head_count == right.head_count
           && left.kv_head_count == right.kv_head_count
           && left.head_dimension == right.head_dimension
           && left.sliding_window == right.sliding_window
           && left.initial_context_length == right.initial_context_length
           && left.max_context_length == right.max_context_length
           && left.query_lora_rank == right.query_lora_rank
           && left.kv_lora_rank == right.kv_lora_rank
           && left.qk_nope_head_dimension == right.qk_nope_head_dimension
           && left.qk_rope_head_dimension == right.qk_rope_head_dimension
           && left.value_head_dimension == right.value_head_dimension
           && left.rope_theta == right.rope_theta
           && left.rope_scaling_factor == right.rope_scaling_factor
           && left.rope_ntk_alpha == right.rope_ntk_alpha
           && left.rope_ntk_beta == right.rope_ntk_beta
           && left.flags == right.flags;
}

static bool experts_match(const MoeDescriptor& left, const MoeDescriptor& right)
{
    return left.expert_count == right.expert_count
           && left.top_k == right.top_k
           && left.intermediate_size == right.intermediate_size
           && left.shared_expert_count == right.shared_expert_count
           && left.router_group_count == right.router_group_count
           && left.router_top_k_groups == right.router_top_k_groups
           && left.score_function == right.score_function
           && left.normalization == right.normalization
           && left.activation == right.activation
           && left.layout == right.layout
           && left.expert_weight_dtype == right.expert_weight_dtype
           && left.activation_limit == right.activation_limit
           && left.routed_scaling_factor == right.routed_scaling_factor
           && left.flags == right.flags;
}

static Result<void> validate_graph_descriptor_consistency(const MoeIR& ir)
{
    std::vector<uint32_t> attention_counts(ir.layer_count, 0);
    std::vector<uint32_t> expert_counts(ir.layer_count, 0);
    std::vector<uint32_t> shared_expert_counts(ir.layer_count, 0);
    std::vector<uint32_t> dense_ffn_counts(ir.layer_count, 0);
    for (const MoeIRNode& node : ir.graph.nodes)
    {
        if (node.layer_id == invalid_moe_ir_layer_id)
            continue;
        if (node.layer_id >= ir.layer_count)
        {
            return Error{ErrorCode::InvalidModel, "MoeIR graph node layer is out of range"};
        }
        const LayerDescriptor& layer = ir.layers[node.layer_id];
        if (node.operation == MoeIROperator::Attention)
        {
            ++attention_counts[node.layer_id];
            if (!attention_matches(node.attention, layer.attention))
            {
                return Error{ErrorCode::InvalidModel, "MoeIR Attention does not match its layer descriptor"};
            }
        }
        else if (node.operation == MoeIROperator::ExpertGroup)
        {
            ++expert_counts[node.layer_id];
            if (!experts_match(node.experts, layer.ffn.moe))
            {
                return Error{ErrorCode::InvalidModel, "MoeIR ExpertGroup does not match its layer descriptor"};
            }
        }
        else if (node.operation == MoeIROperator::SharedExpertGroup)
        {
            ++shared_expert_counts[node.layer_id];
            if (!experts_match(node.experts, layer.ffn.moe))
            {
                return Error{ErrorCode::InvalidModel, "MoeIR SharedExpertGroup does not match its layer descriptor"};
            }
        }
        else if (node.operation == MoeIROperator::DenseFfn)
        {
            ++dense_ffn_counts[node.layer_id];
            if (node.intermediate_size != layer.ffn.dense_intermediate_size)
            {
                return Error{ErrorCode::InvalidModel, "MoeIR DenseFfn does not match its layer descriptor"};
            }
        }
    }
    for (uint32_t layer_id = 0; layer_id < ir.layer_count; ++layer_id)
    {
        const bool has_attention = has_flag(ir.layers[layer_id].flags, LayerDescriptorAttention);
        if (attention_counts[layer_id] != (has_attention ? 1u : 0u))
        {
            return Error{ErrorCode::InvalidModel, "MoeIR Attention count does not match its layer"};
        }
        const LayerDescriptor& layer = ir.layers[layer_id];
        const bool has_moe = has_flag(layer.flags, LayerDescriptorMoe);
        const bool has_dense_ffn = has_flag(layer.flags, LayerDescriptorDenseFfn);
        if (has_moe == has_dense_ffn)
        {
            return Error{ErrorCode::InvalidModel, "MoeIR layer requires exactly one FFN kind"};
        }
        if (expert_counts[layer_id] != (has_moe ? 1u : 0u) || dense_ffn_counts[layer_id] != (has_dense_ffn ? 1u : 0u))
        {
            return Error{ErrorCode::InvalidModel, "MoeIR FFN count does not match its layer"};
        }
        const uint32_t expected_shared = has_moe && layer.ffn.moe.shared_expert_count != 0 ? 1u : 0u;
        if (shared_expert_counts[layer_id] != expected_shared)
        {
            return Error{ErrorCode::InvalidModel, "MoeIR shared Expert count does not match its layer"};
        }
    }
    return {};
}

static bool has_node(const LayerDescriptor& layer, ModelNodeType type)
{
    for (const ModelNodeDescriptor& node : layer.nodes)
    {
        if (node.type == type)
            return true;
    }
    return false;
}

Result<void> normalize_moe_ir(MoeIR& ir)
{
    if (ir.graph.nodes.empty())
    {
        for (LayerDescriptor& layer : ir.layers)
        {
            if (!has_flag(layer.flags, LayerDescriptorAttention)
                && (has_node(layer, ModelNodeType::FusedQkv)
                    || has_node(layer, ModelNodeType::MultiHeadLatentAttention)))
            {
                layer.flags |= LayerDescriptorAttention;
            }
            if (!has_flag(layer.flags, LayerDescriptorMoe) && !has_flag(layer.flags, LayerDescriptorDenseFfn))
            {
                if (has_node(layer, ModelNodeType::ExpertGroup))
                    layer.flags |= LayerDescriptorMoe;
                else if (has_node(layer, ModelNodeType::DenseFfn))
                    layer.flags |= LayerDescriptorDenseFfn;
            }
        }
        MoeGraphBuilder builder;
        auto graph = builder.build(ir);
        if (!graph)
            return graph.error();
        ir.graph = std::move(graph).value();
    }
    bool graph_driven_layers = !ir.layers.empty();
    for (const LayerDescriptor& layer : ir.layers)
    {
        if (has_flag(layer.flags, LayerDescriptorMoe) || has_flag(layer.flags, LayerDescriptorDenseFfn))
        {
            graph_driven_layers = false;
            break;
        }
    }
    if (graph_driven_layers)
        ir.layers.clear();
    if (ir.layers.empty())
    {
        auto materialized = materialize_layers_from_graph(ir);
        if (!materialized)
            return materialized.error();
    }
    ir.activation_quantization = quant_config_for_dtype(ir.activation_dtype);
    ir.kv_cache_quantization = quant_config_for_dtype(ir.kv_cache_dtype);
    if (ir.expert_quantization.scheme == QuantizationScheme::None)
    {
        for (const LayerDescriptor& layer : ir.layers)
        {
            if (!has_flag(layer.flags, LayerDescriptorMoe))
                continue;
            ir.expert_quantization = quant_config_for_dtype(layer.ffn.moe.expert_weight_dtype);
            break;
        }
    }
    auto valid = ir.validate();
    if (!valid)
        return valid.error();
    return validate_graph_descriptor_consistency(ir);
}

static uint64_t dtype_size(DType dtype)
{
    switch (dtype)
    {
    case DType::Float32:
    case DType::Int32: return 4;
    case DType::Float16:
    case DType::BFloat16: return 2;
    case DType::Int8: return 1;
    case DType::MxFp4: return 0;
    }
    return 0;
}

static uint64_t estimate_tensor_bytes(DType dtype, const std::vector<uint32_t>& shape)
{
    const uint64_t element_size = dtype_size(dtype);
    if (element_size == 0)
        return 0;
    uint64_t elements = 1;
    for (uint32_t dimension : shape)
    {
        if (dimension == 0)
            return 0;
        elements *= dimension;
    }
    return elements * element_size;
}

static ExecutionTensorId append_execution_tensor(ExecutionGraph& graph, std::string name, DType dtype, std::vector<uint32_t> shape, TensorLocation location, uint32_t flags)
{
    const ExecutionTensorId id = static_cast<ExecutionTensorId>(graph.tensors.size());
    ExecutionTensor tensor;
    tensor.id = id;
    tensor.name = std::move(name);
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    tensor.location = location;
    tensor.estimated_bytes = estimate_tensor_bytes(tensor.dtype, tensor.shape);
    tensor.flags = flags;
    graph.tensors.push_back(std::move(tensor));
    return id;
}

static ExecutionNodeId append_execution_node(ExecutionGraph& graph, ExecutionNodeType type, ExecutionBackend backend, uint32_t backend_mask, std::string name, std::vector<ExecutionNodeId> dependencies, std::vector<ExecutionTensorId> inputs,
                                             std::vector<ExecutionTensorId> outputs, uint32_t layer_id, uint32_t expert_id, uint32_t flags)
{
    const ExecutionNodeId node_id = static_cast<ExecutionNodeId>(graph.nodes.size());
    ExecutionNode node;
    node.id = node_id;
    node.type = type;
    node.backend = backend;
    node.backend_mask = backend_mask;
    node.layer_id = layer_id;
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

Result<void> build_compiled_execution_graph(CompiledModel& compiled, const ModelCompiler::BackendCapabilities& capabilities)
{
    ExecutionGraph graph;
    const uint32_t dynamic_tensor = ExecutionTensorDynamic;
    ExecutionTensorId hidden = append_execution_tensor(graph, "embedding.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, dynamic_tensor);
    ExecutionNodeId previous = append_execution_node(graph, ExecutionNodeType::TokenEmbedding, ExecutionBackend::Cpu, ExecutionBackendCpu, "token_embedding", {}, {}, {hidden}, invalid_execution_layer_id, invalid_execution_expert_id, 0);

    for (const CompiledLayerPlan& layer : compiled.layers)
    {
        const std::string prefix = "layers." + std::to_string(layer.layer_id) + ".";
        if (has_flag(layer.flags, CompiledLayerAttention))
        {
            const ExecutionBackend backend = layer.nodes.empty() ? ExecutionBackend::Cpu : layer.nodes.front().backend;
            const TensorLocation cache_location = backend == ExecutionBackend::Vulkan ? TensorLocation::Vulkan : TensorLocation::Cpu;
            const ExecutionTensorId cache = append_execution_tensor(graph, prefix + "kv_cache", compiled.descriptor.kv_cache_dtype,
                                                                    {
                                                                        0,
                                                                        layer.attention.kv_head_count,
                                                                        layer.attention.head_dimension,
                                                                    },
                                                                    cache_location, ExecutionTensorPersistent | ExecutionTensorDynamic);
            uint32_t attention_tensor_flags = ExecutionTensorDynamic;
            if (backend == ExecutionBackend::Vulkan)
            {
                attention_tensor_flags |= ExecutionTensorTransferBoundary;
            }
            const ExecutionTensorId attention_output = append_execution_tensor(graph, prefix + "attention.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, attention_tensor_flags);
            previous = append_execution_node(
                graph, ExecutionNodeType::Attention, backend, backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu, prefix + "attention",
                {previous}, {hidden}, {attention_output, cache},
                layer.layer_id, invalid_execution_expert_id, 0);
            hidden = attention_output;
        }

        const ExecutionTensorId router_scores = append_execution_tensor(graph, prefix + "router.scores", DType::Float32, {0, static_cast<uint32_t>(layer.moe.experts.size())}, TensorLocation::Cpu, ExecutionTensorDynamic);
        const ExecutionNodeId router = append_execution_node(
            graph, ExecutionNodeType::Router, ExecutionBackend::Cpu, ExecutionBackendCpu, prefix + "router",
            {previous}, {hidden}, {router_scores},
            layer.layer_id, invalid_execution_expert_id, 0);
        const ExecutionTensorId assignments = append_execution_tensor(graph, prefix + "expert_dispatch.assignments", DType::Int32, {0, 3}, TensorLocation::Cpu, ExecutionTensorDynamic);
        const ExecutionNodeId dispatch = append_execution_node(
            graph, ExecutionNodeType::ExpertDispatch, ExecutionBackend::Cpu, ExecutionBackendCpu, prefix + "expert_dispatch",
            {router}, {router_scores}, {assignments},
            layer.layer_id, invalid_execution_expert_id, 0);

        const ExecutionTensorId expert_output = append_execution_tensor(graph, prefix + "experts.output", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, ExecutionTensorDynamic);
        const ExecutionNodeId expert_group = append_execution_node(
            graph, ExecutionNodeType::ExpertGroup, ExecutionBackend::Cpu, ExecutionBackendCpu, prefix + "experts",
            {dispatch}, {hidden, assignments}, {expert_output},
            layer.layer_id, invalid_execution_expert_id, 0);

        const ExecutionTensorId combined = append_execution_tensor(graph, prefix + "combine.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, ExecutionTensorDynamic);
        previous = append_execution_node(
            graph, ExecutionNodeType::Combine, ExecutionBackend::Cpu, ExecutionBackendCpu, prefix + "combine",
            {expert_group}, {hidden, expert_output}, {combined},
            layer.layer_id, invalid_execution_expert_id, 0);
        hidden = combined;
    }

    const ExecutionTensorId normalized = append_execution_tensor(graph, "final_norm.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, ExecutionTensorDynamic);
    previous = append_execution_node(
        graph, ExecutionNodeType::FinalNorm, ExecutionBackend::Cpu, ExecutionBackendCpu, "final_norm",
        {previous}, {hidden}, {normalized},
        invalid_execution_layer_id, invalid_execution_expert_id, 0);

    const ExecutionBackend lm_head_backend = compiled.hybrid_mode == HybridMode::CpuOnly ? ExecutionBackend::Cpu : ExecutionBackend::Vulkan;
    uint32_t logits_tensor_flags = ExecutionTensorDynamic;
    if (lm_head_backend == ExecutionBackend::Vulkan)
        logits_tensor_flags |= ExecutionTensorTransferBoundary;
    const ExecutionTensorId logits = append_execution_tensor(graph, "logits", DType::Float32, {0, compiled.descriptor.vocabulary_size}, TensorLocation::Cpu, logits_tensor_flags);
    (void)append_execution_node(
        graph, ExecutionNodeType::LmHead, lm_head_backend, lm_head_backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu, "lm_head",
        {previous}, {normalized}, {logits},
        invalid_execution_layer_id, invalid_execution_expert_id, 0);

    RuntimeSchedulingOptions options;
    options.available_backends = ExecutionBackendCpu;
    options.cpu_parallelism = capabilities.cpu_parallelism;
    if (compiled.hybrid_mode != HybridMode::CpuOnly)
    {
        options.available_backends |= ExecutionBackendVulkan;
        options.vulkan_queue_count = std::max(1u, capabilities.vulkan_queue_count);
    }
    RuntimeScheduler scheduler;
    auto scheduled = scheduler.compile(std::move(graph), options);
    if (!scheduled)
        return scheduled.error();
    ScheduledExecutionGraph result = std::move(scheduled).value();
    compiled.graph = std::move(result.graph);
    compiled.schedule = std::move(result.schedule);
    return {};
}

} // namespace moe
} // namespace ncnn
