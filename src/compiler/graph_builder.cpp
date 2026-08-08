#include "compiler/moe_ir.hpp"
#include "ncnn/moe/runtime_config.h"

#include "kernels/cpu_qnk.h"

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
    else if (is_qnk_dtype(dtype))
    {
        config.scheme = QuantizationScheme::BlockWise;
        config.block_size = qnk_block_elements;
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
            const DType cache_dtype = layer.attention.kind == AttentionKind::GatedDeltaNet
                                          ? DType::Float32
                                          : descriptor.kv_cache_dtype;
            const MoeIRValueId cache = append_moe_ir_value(graph, prefix + "kv_cache", cache_dtype,
                                                           std::move(cache_shape),
                                                           TensorLocation::Automatic, MoeIRValuePersistent | MoeIRValueMutableState | MoeIRValueDynamicShape);
            const MoeIRNodeId cache_node = append_moe_ir_node(graph, MoeIROperator::KvCache, prefix + "kv_cache", layer_id, {}, {cache});
            graph.nodes[cache_node].flags |= MoeIRNodeStateful;

            const MoeIRValueId attention_output = append_moe_ir_value(graph, prefix + "attention.hidden", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Automatic, MoeIRValueDynamicShape);
            const MoeIRNodeId attention_node = append_moe_ir_node(graph, MoeIROperator::Attention, prefix + "attention", layer_id, {hidden, cache}, {attention_output});
            graph.nodes[attention_node].flags |= MoeIRNodeStateful;
            hidden = attention_output;
        }

        if (has_flag(layer.flags, LayerDescriptorMoe))
        {
            const MoeDescriptor& moe = layer.ffn.moe;
            const MoeIRValueId router_scores = append_moe_ir_value(graph, prefix + "router.scores", DType::Float32, {0, moe.expert_count}, TensorLocation::Cpu, MoeIRValueDynamicShape);
            (void)append_moe_ir_node(graph, MoeIROperator::Router, prefix + "router", layer_id, {hidden}, {router_scores});

            const MoeIRValueId expert_output = append_moe_ir_value(graph, prefix + "experts.output", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Cpu, MoeIRValueDynamicShape);
            (void)append_moe_ir_node(graph, MoeIROperator::ExpertGroup, prefix + "experts", layer_id, {hidden, router_scores}, {expert_output});

            std::vector<MoeIRValueId> combine_inputs{
                hidden,
                expert_output,
            };
            if (moe.shared_expert_count != 0)
            {
                const MoeIRValueId shared_output = append_moe_ir_value(graph, prefix + "shared_experts.output", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Cpu, MoeIRValueDynamicShape);
                (void)append_moe_ir_node(graph, MoeIROperator::SharedExpertGroup, prefix + "shared_experts", layer_id, {hidden}, {shared_output});
                combine_inputs.push_back(shared_output);
            }

            const MoeIRValueId combined = append_moe_ir_value(graph, prefix + "combine.hidden", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Cpu, MoeIRValueDynamicShape);
            (void)append_moe_ir_node(graph, MoeIROperator::Combine, prefix + "combine", layer_id, std::move(combine_inputs), {combined});
            hidden = combined;
        }
        else if (has_flag(layer.flags, LayerDescriptorDenseFfn))
        {
            const MoeIRValueId dense_output = append_moe_ir_value(graph, prefix + "dense_ffn.hidden", descriptor.activation_dtype, {0, descriptor.hidden_size}, TensorLocation::Automatic, MoeIRValueDynamicShape);
            (void)append_moe_ir_node(graph, MoeIROperator::DenseFfn, prefix + "dense_ffn", layer_id, {hidden}, {dense_output});
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
            if (!has_flag(layer.flags, LayerDescriptorAttention))
            {
                return Error{ErrorCode::InvalidModel, "MoeIR Attention is not declared by its layer descriptor"};
            }
        }
        else if (node.operation == MoeIROperator::ExpertGroup)
        {
            ++expert_counts[node.layer_id];
            if (!has_flag(layer.flags, LayerDescriptorMoe))
            {
                return Error{ErrorCode::InvalidModel, "MoeIR ExpertGroup is not declared by its layer descriptor"};
            }
        }
        else if (node.operation == MoeIROperator::SharedExpertGroup)
        {
            ++shared_expert_counts[node.layer_id];
            if (!has_flag(layer.flags, LayerDescriptorMoe)
                || layer.ffn.moe.shared_expert_count == 0)
            {
                return Error{ErrorCode::InvalidModel, "MoeIR SharedExpertGroup is not declared by its layer descriptor"};
            }
        }
        else if (node.operation == MoeIROperator::DenseFfn)
        {
            ++dense_ffn_counts[node.layer_id];
            if (!has_flag(layer.flags, LayerDescriptorDenseFfn))
            {
                return Error{ErrorCode::InvalidModel, "MoeIR DenseFfn is not declared by its layer descriptor"};
            }
        }
        else if (node.operation == MoeIROperator::KvCache
                 && !has_flag(layer.flags, LayerDescriptorAttention))
        {
            return Error{ErrorCode::InvalidModel, "MoeIR KV Cache is not declared by its layer descriptor"};
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

Result<void> normalize_moe_ir(MoeIR& ir)
{
    if (ir.graph.nodes.empty())
    {
        MoeGraphBuilder builder;
        auto graph = builder.build(ir);
        if (!graph)
            return graph.error();
        ir.graph = std::move(graph).value();
    }
    // The model adapter descriptor is input metadata only.  Once the graph
    // exists, it is the sole topology source; never materialize layers back
    // from the graph or reconcile two competing operator lists.
    if (ir.layers.size() != ir.layer_count)
        return Error{ErrorCode::InvalidModel, "canonical MoeIR requires layer metadata for every graph layer"};
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
    case DType::Int64: return 8;
    case DType::Float16:
    case DType::BFloat16: return 2;
    case DType::Float8E4M3:
    case DType::Int8: return 1;
    case DType::MxFp4: return 0;
    case DType::Q2K:
    case DType::Q3K:
    case DType::Q4K:
    case DType::Q5K:
    case DType::Q6K:
    case DType::Q8K: return 0;
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

static bool has_vulkan_dense_operator(const CompiledOperator& operator_entry) noexcept
{
    return operator_entry.linear || operator_entry.bfloat16 || operator_entry.float8;
}

static bool attention_supports_vulkan(
    const CompiledModel& compiled,
    const CompiledLayerPlan& layer) noexcept
{
    if (compiled.hybrid_mode == HybridMode::CpuOnly
        || !runtime_optimization_enabled(
            compiled.optimization_flags,
            RuntimeOptimizationVulkanAttention)
        || layer.layer_id >= compiled.descriptor.layers.size())
    {
        return false;
    }

    const AttentionBlockPlan& attention = layer.attention;
    const AttentionKind kind = compiled.descriptor.layers[layer.layer_id].attention.kind;
    if (kind == AttentionKind::GatedDeltaNet)
    {
        if (attention.gated_delta_vulkan_operator == invalid_compiled_operator_handle)
            return false;
        const CompiledOperator& gated_delta_operator = compiled.operators.at(attention.gated_delta_vulkan_operator);
        return static_cast<bool>(gated_delta_operator.gated_delta);
    }
    if (kind != AttentionKind::Standard
        && kind != AttentionKind::MultiHeadLatent)
    {
        return false;
    }

    if (attention.vulkan_attention_operator != invalid_compiled_operator_handle
        && compiled.operators.at(attention.vulkan_attention_operator).attention)
    {
        return true;
    }

    const TensorHandle dense_handles[] = {
        attention.query_weight,
        attention.key_weight,
        attention.value_weight,
        attention.output_weight,
        attention.query_a_weight,
        attention.query_b_weight,
        attention.key_value_weight,
        attention.output_a_weight,
        attention.output_b_weight,
    };
    for (TensorHandle handle : dense_handles)
    {
        if (handle != invalid_tensor_handle
            && has_vulkan_dense_operator(compiled.operators.at_weight(handle)))
        {
            return true;
        }
    }
    return false;
}

static bool speculative_attention_supports_vulkan(
    const CompiledModel& compiled,
    const AttentionBlockPlan& attention) noexcept
{
    if (compiled.hybrid_mode == HybridMode::CpuOnly
        || !runtime_optimization_enabled(
            compiled.optimization_flags,
            RuntimeOptimizationVulkanAttention))
        return false;
    if (attention.gated_delta_vulkan_operator != invalid_compiled_operator_handle
        && compiled.operators.at(attention.gated_delta_vulkan_operator).gated_delta)
    {
        return true;
    }
    if (attention.vulkan_attention_operator != invalid_compiled_operator_handle
        && compiled.operators.at(attention.vulkan_attention_operator).attention)
    {
        return true;
    }

    const TensorHandle dense_handles[] = {
        attention.query_weight,
        attention.key_weight,
        attention.value_weight,
        attention.output_weight,
        attention.query_a_weight,
        attention.query_b_weight,
        attention.key_value_weight,
        attention.output_a_weight,
        attention.output_b_weight,
    };
    for (TensorHandle handle : dense_handles)
    {
        if (handle != invalid_tensor_handle
            && has_vulkan_dense_operator(compiled.operators.at_weight(handle)))
        {
            return true;
        }
    }
    return false;
}

static Result<void> build_speculative_execution_graph(
    CompiledModel& compiled,
    const ModelCompiler::BackendCapabilities& capabilities)
{
    if (!compiled.speculative.enabled())
        return {};

    ExecutionGraph graph;
    graph.layer_plans = std::move(compiled.speculative.graph.layer_plans);
    const uint32_t dynamic_tensor = ExecutionTensorDynamic;
    ExecutionTensorId hidden = append_execution_tensor(
        graph,
        "speculative.input.hidden",
        compiled.descriptor.activation_dtype,
        {0, compiled.descriptor.hidden_size},
        TensorLocation::Cpu,
        dynamic_tensor);
    ExecutionNodeId previous = append_execution_node(
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
        if (has_flag(layer.flags, CompiledLayerAttention))
        {
            const bool supports_vulkan_attention = speculative_attention_supports_vulkan(compiled, layer.attention);
            const ExecutionBackend backend = supports_vulkan_attention ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
            const uint32_t backend_mask = supports_vulkan_attention ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
            const TensorLocation cache_location = backend == ExecutionBackend::Vulkan ? TensorLocation::Vulkan : TensorLocation::Cpu;
            std::vector<uint32_t> cache_shape{
                0,
                layer.attention.kv_head_count,
                layer.attention.head_dimension,
            };
            if (has_flag(layer.attention.flags, AttentionBlockGatedDeltaNet))
            {
                cache_shape = {
                    0,
                    layer.attention.head_count,
                    layer.attention.head_dimension,
                    layer.attention.value_head_dimension,
                };
            }
            const ExecutionTensorId cache = append_execution_tensor(
                graph,
                prefix + "kv_cache",
                has_flag(layer.attention.flags, AttentionBlockGatedDeltaNet) ? DType::Float32 : compiled.descriptor.kv_cache_dtype,
                std::move(cache_shape),
                cache_location,
                ExecutionTensorPersistent | ExecutionTensorDynamic);
            uint32_t attention_tensor_flags = ExecutionTensorDynamic;
            if (backend == ExecutionBackend::Vulkan)
                attention_tensor_flags |= ExecutionTensorTransferBoundary;
            const ExecutionTensorId attention_output = append_execution_tensor(
                graph,
                prefix + "attention.hidden",
                compiled.descriptor.activation_dtype,
                {0, compiled.descriptor.hidden_size},
                TensorLocation::Cpu,
                attention_tensor_flags);
            previous = append_execution_node(
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

        const ExecutionTensorId router_scores = append_execution_tensor(
            graph,
            prefix + "router.scores",
            DType::Float32,
            {0, static_cast<uint32_t>(layer.moe.experts.size())},
            TensorLocation::Cpu,
            ExecutionTensorDynamic);
        const ExecutionNodeId router = append_execution_node(
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
        const ExecutionTensorId assignments = append_execution_tensor(
            graph,
            prefix + "expert_dispatch.assignments",
            DType::Int32,
            {0, 3},
            TensorLocation::Cpu,
            ExecutionTensorDynamic);
        const ExecutionNodeId dispatch = append_execution_node(
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

        const bool can_use_vulkan_experts = compiled.hybrid_mode != HybridMode::CpuOnly
                                            && has_flag(capabilities.flags, ModelCompiler::BackendCapabilityVulkanExperts)
                                            && !layer.moe.experts.empty()
                                            && layer.moe.experts.front().gate_up_weight != invalid_tensor_handle
                                            && (compiled.weights.at(layer.moe.experts.front().gate_up_weight).dtype == DType::MxFp4
                                                || (runtime_optimization_enabled(
                                                        capabilities.optimization_flags,
                                                        RuntimeOptimizationVulkanQnK)
                                                    && is_qnk_dtype(compiled.weights.at(layer.moe.experts.front().gate_up_weight).dtype)));
        const ExecutionBackend expert_backend = can_use_vulkan_experts ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
        const uint32_t expert_backend_mask = can_use_vulkan_experts ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
        const uint32_t expert_flags = compiled.hybrid_mode == HybridMode::VulkanWithCpuPrefetch
                                          ? ExecutionNodeCpuPrefetch
                                          : 0u;
        const ExecutionTensorId expert_output = append_execution_tensor(
            graph,
            prefix + "experts.output",
            compiled.descriptor.activation_dtype,
            {0, compiled.descriptor.hidden_size},
            TensorLocation::Cpu,
            ExecutionTensorDynamic);
        const ExecutionNodeId expert_group = append_execution_node(
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

        const ExecutionTensorId combined = append_execution_tensor(
            graph,
            prefix + "combine.hidden",
            compiled.descriptor.activation_dtype,
            {0, compiled.descriptor.hidden_size},
            TensorLocation::Cpu,
            ExecutionTensorDynamic);
        std::vector<ExecutionTensorId> combine_inputs = {hidden, expert_output};
        std::vector<ExecutionNodeId> combine_dependencies = {expert_group};
        if (layer.moe.has_shared_expert)
        {
            const bool shared_vulkan = compiled.hybrid_mode != HybridMode::CpuOnly
                                       && layer.moe.fused_shared_input_bfloat16_operator != invalid_compiled_operator_handle
                                       && compiled.operators.at(layer.moe.fused_shared_input_bfloat16_operator).bfloat16;
            const ExecutionBackend shared_backend = shared_vulkan ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
            const uint32_t shared_backend_mask = shared_vulkan ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
            const ExecutionTensorId shared_output = append_execution_tensor(
                graph,
                prefix + "shared_experts.output",
                compiled.descriptor.activation_dtype,
                {0, compiled.descriptor.hidden_size},
                TensorLocation::Cpu,
                ExecutionTensorDynamic);
            const ExecutionNodeId shared_node = append_execution_node(
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
        previous = append_execution_node(
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
    }

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
    compiled.speculative.graph = std::move(result.graph);
    compiled.speculative.schedule = std::move(result.schedule);
    return {};
}

Result<void> build_compiled_execution_graph(CompiledModel& compiled, const ModelCompiler::BackendCapabilities& capabilities)
{
    ExecutionGraph graph;
    graph.layer_plans = std::move(compiled.graph.layer_plans);
    const uint32_t dynamic_tensor = ExecutionTensorDynamic;
    ExecutionTensorId hidden = append_execution_tensor(graph, "embedding.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, dynamic_tensor);
    ExecutionNodeId previous = append_execution_node(graph, ExecutionNodeType::TokenEmbedding, ExecutionBackend::Cpu, ExecutionBackendCpu, "token_embedding", {}, {}, {hidden}, invalid_execution_layer_id, invalid_execution_expert_id, 0);
    graph.nodes[previous].weight_inputs = {compiled.token_embedding};

    for (size_t plan_index = 0; plan_index < graph.layer_plans.size(); ++plan_index)
    {
        const CompiledLayerPlan& layer = graph.layer_plans[plan_index];
        const std::string prefix = "layers." + std::to_string(layer.layer_id) + ".";
        if (has_flag(layer.flags, CompiledLayerAttention))
        {
            const bool latent_attention = compiled.descriptor.layers[layer.layer_id].attention.kind == AttentionKind::MultiHeadLatent;
            const bool gated_delta_attention = compiled.descriptor.layers[layer.layer_id].attention.kind == AttentionKind::GatedDeltaNet;
            const bool supports_vulkan_attention = attention_supports_vulkan(compiled, layer);
            const ExecutionBackend backend = supports_vulkan_attention ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
            const TensorLocation cache_location = latent_attention
                                                      ? TensorLocation::Cpu
                                                  : backend == ExecutionBackend::Vulkan
                                                      ? TensorLocation::Vulkan
                                                      : TensorLocation::Cpu;
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
            const ExecutionTensorId cache = append_execution_tensor(
                graph,
                prefix + "kv_cache",
                gated_delta_attention ? DType::Float32 : compiled.descriptor.kv_cache_dtype,
                std::move(cache_shape),
                cache_location, ExecutionTensorPersistent | ExecutionTensorDynamic);
            uint32_t attention_tensor_flags = ExecutionTensorDynamic;
            if (backend == ExecutionBackend::Vulkan && !latent_attention)
            {
                attention_tensor_flags |= ExecutionTensorTransferBoundary;
            }
            const ExecutionTensorId attention_output = append_execution_tensor(graph, prefix + "attention.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, attention_tensor_flags);
            previous = append_execution_node(
                graph, ExecutionNodeType::Attention, backend, backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu, prefix + "attention",
                {previous}, {hidden}, {attention_output, cache},
                static_cast<uint32_t>(plan_index), invalid_execution_expert_id, 0);
            hidden = attention_output;
        }

        const ExecutionTensorId router_scores = append_execution_tensor(graph, prefix + "router.scores", DType::Float32, {0, static_cast<uint32_t>(layer.moe.experts.size())}, TensorLocation::Cpu, ExecutionTensorDynamic);
        const ExecutionNodeId router = append_execution_node(
            graph, ExecutionNodeType::Router, ExecutionBackend::Cpu, ExecutionBackendCpu, prefix + "router",
            {previous}, {hidden}, {router_scores},
            static_cast<uint32_t>(plan_index), invalid_execution_expert_id, 0);
        const ExecutionTensorId assignments = append_execution_tensor(graph, prefix + "expert_dispatch.assignments", DType::Int32, {0, 3}, TensorLocation::Cpu, ExecutionTensorDynamic);
        const ExecutionNodeId dispatch = append_execution_node(
            graph, ExecutionNodeType::ExpertDispatch, ExecutionBackend::Cpu, ExecutionBackendCpu, prefix + "expert_dispatch",
            {router}, {router_scores}, {assignments},
            static_cast<uint32_t>(plan_index), invalid_execution_expert_id, 0);

        const bool can_use_vulkan_experts = compiled.hybrid_mode != HybridMode::CpuOnly
                                            && has_flag(capabilities.flags, ModelCompiler::BackendCapabilityVulkanExperts)
                                            && !layer.moe.experts.empty()
                                            && layer.moe.experts.front().gate_up_weight != invalid_tensor_handle
                                            && (compiled.weights.at(layer.moe.experts.front().gate_up_weight).dtype == DType::MxFp4
                                                || (runtime_optimization_enabled(
                                                        capabilities.optimization_flags,
                                                        RuntimeOptimizationVulkanQnK)
                                                    && is_qnk_dtype(compiled.weights.at(layer.moe.experts.front().gate_up_weight).dtype)));
        const ExecutionBackend expert_backend = can_use_vulkan_experts ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
        const uint32_t expert_backend_mask = can_use_vulkan_experts ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
        const uint32_t expert_flags = compiled.hybrid_mode == HybridMode::VulkanWithCpuPrefetch
                                          ? ExecutionNodeCpuPrefetch
                                          : 0u;
        const ExecutionTensorId expert_output = append_execution_tensor(graph, prefix + "experts.output", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, ExecutionTensorDynamic);
        const ExecutionNodeId expert_group = append_execution_node(
            graph, ExecutionNodeType::ExpertGroup, expert_backend, expert_backend_mask, prefix + "experts",
            {dispatch}, {hidden, assignments}, {expert_output},
            static_cast<uint32_t>(plan_index), invalid_execution_expert_id, expert_flags);

        const ExecutionTensorId combined = append_execution_tensor(graph, prefix + "combine.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, ExecutionTensorDynamic);
        std::vector<ExecutionTensorId> combine_inputs = {hidden, expert_output};
        std::vector<ExecutionNodeId> combine_dependencies = {expert_group};
        if (layer.moe.has_shared_expert)
        {
            const bool shared_vulkan = compiled.hybrid_mode != HybridMode::CpuOnly
                                       && layer.moe.fused_shared_input_bfloat16_operator != invalid_compiled_operator_handle
                                       && compiled.operators.at(layer.moe.fused_shared_input_bfloat16_operator).bfloat16;
            const ExecutionBackend shared_backend = shared_vulkan ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
            const uint32_t shared_backend_mask = shared_vulkan ? ExecutionBackendCpu | ExecutionBackendVulkan : ExecutionBackendCpu;
            const ExecutionTensorId shared_output = append_execution_tensor(graph, prefix + "shared_experts.output", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, ExecutionTensorDynamic);
            const ExecutionNodeId shared_node = append_execution_node(
                graph, ExecutionNodeType::SharedExpertGroup, shared_backend, shared_backend_mask, prefix + "shared_experts",
                {expert_group}, {hidden}, {shared_output}, static_cast<uint32_t>(plan_index), invalid_execution_expert_id, 0);
            combine_dependencies.push_back(shared_node);
            combine_inputs.push_back(shared_output);
        }
        previous = append_execution_node(
            graph, ExecutionNodeType::Combine, ExecutionBackend::Cpu, ExecutionBackendCpu, prefix + "combine",
            std::move(combine_dependencies), std::move(combine_inputs), {combined},
            static_cast<uint32_t>(plan_index), invalid_execution_expert_id, 0);
        hidden = combined;
    }

    const ExecutionTensorId normalized = append_execution_tensor(graph, "final_norm.hidden", compiled.descriptor.activation_dtype, {0, compiled.descriptor.hidden_size}, TensorLocation::Cpu, ExecutionTensorDynamic);
    previous = append_execution_node(
        graph, ExecutionNodeType::FinalNorm, ExecutionBackend::Cpu, ExecutionBackendCpu, "final_norm",
        {previous}, {hidden}, {normalized},
        invalid_execution_layer_id, invalid_execution_expert_id, 0);
    graph.nodes[previous].weight_inputs = {
        compiled.final_norm_weight,
        compiled.lm_head_weight};

    const ExecutionBackend lm_head_backend = compiled.hybrid_mode == HybridMode::CpuOnly ? ExecutionBackend::Cpu : ExecutionBackend::Vulkan;
    uint32_t logits_tensor_flags = ExecutionTensorDynamic;
    if (lm_head_backend == ExecutionBackend::Vulkan)
        logits_tensor_flags |= ExecutionTensorTransferBoundary;
    const ExecutionTensorId logits = append_execution_tensor(graph, "logits", DType::Float32, {0, compiled.descriptor.vocabulary_size}, TensorLocation::Cpu, logits_tensor_flags);
    const ExecutionNodeId lm_head_node = append_execution_node(
        graph, ExecutionNodeType::LmHead, lm_head_backend, lm_head_backend == ExecutionBackend::Vulkan ? ExecutionBackendVulkan : ExecutionBackendCpu, "lm_head",
        {previous}, {normalized}, {logits},
        invalid_execution_layer_id, invalid_execution_expert_id, 0);
    graph.nodes[lm_head_node].weight_inputs = {
        compiled.lm_head_weight,
        compiled.final_norm_weight};

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
    return build_speculative_execution_graph(compiled, capabilities);
}

} // namespace moe
} // namespace ncnn
