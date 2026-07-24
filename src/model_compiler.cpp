#include "ncnn/moe/execution_plan.h"

#include "internal/tensor_names.h"
#include "ncnn_attention.h"
#include "ncnn_linear.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <utility>

namespace ncnn {
namespace moe {

static bool shape_equals(const TensorData& tensor, std::initializer_list<uint32_t> expected)
{
    return tensor.shape.size() == expected.size()
           && std::equal(tensor.shape.begin(), tensor.shape.end(), expected.begin());
}

struct AdapterGraph
{
    bool use_attention = false;
    bool use_attention_sink = false;
    bool use_moe = false;
    size_t attention_node_count = 0;
};

static Result<AdapterGraph> validate_adapter_graph(const LayerDescriptor& layer)
{
    if (layer.nodes.empty())
        return Error{ErrorCode::InvalidModel, "adapter must provide an explicit operator graph"};
    AdapterGraph graph;
    size_t cursor = 0;
    if (layer.nodes.size() >= 3
        && layer.nodes[0].type == ModelNodeType::RmsNorm
        && layer.nodes[1].type == ModelNodeType::FusedQkv
        && layer.nodes[2].type == ModelNodeType::Rope) {
        graph.use_attention = true;
        cursor = 3;
        if (cursor < layer.nodes.size()
            && layer.nodes[cursor].type == ModelNodeType::AttentionSink) {
            graph.use_attention_sink = true;
            ++cursor;
        }
        if (cursor + 1 >= layer.nodes.size()
            || layer.nodes[cursor].type != ModelNodeType::Sdpa
            || layer.nodes[cursor + 1].type != ModelNodeType::Projection) {
            return Error{
                ErrorCode::InvalidModel,
                "adapter attention graph must end with SDPA and Projection"};
        }
        cursor += 2;
        graph.attention_node_count = cursor;
    }
    static constexpr ModelNodeType moe_nodes[] = {
        ModelNodeType::RmsNorm,
        ModelNodeType::Router,
        ModelNodeType::TopK,
        ModelNodeType::ExpertGroup,
        ModelNodeType::Combine,
    };
    if (cursor + std::size(moe_nodes) <= layer.nodes.size()
        && std::equal(
            std::begin(moe_nodes),
            std::end(moe_nodes),
            layer.nodes.begin() + cursor,
            [](ModelNodeType expected, const ModelNodeDescriptor& actual) {
                return expected == actual.type;
            })) {
        graph.use_moe = true;
        cursor += std::size(moe_nodes);
    }
    if (cursor != layer.nodes.size())
        return Error{ErrorCode::InvalidModel, "adapter operator graph has an invalid node order"};
    return graph;
}

static Result<TensorHandle> require_tensor(
    const WeightTable& weights,
    const std::string& name,
    std::initializer_list<uint32_t> shape,
    DType dtype)
{
    const TensorHandle handle = weights.find_handle(name);
    if (handle == invalid_tensor_handle)
        return Error{ErrorCode::InvalidModel, "missing tensor: " + name};

    const TensorData& tensor = weights.at(handle);
    if (tensor.dtype != dtype)
        return Error{ErrorCode::InvalidModel, "unexpected dtype for tensor: " + name};
    if (!shape_equals(tensor, shape))
        return Error{ErrorCode::InvalidModel, "unexpected shape for tensor: " + name};

    if (dtype == DType::Float32) {
        if (tensor.float32_data.size() != tensor.element_count())
            return Error{ErrorCode::InvalidModel, "invalid float32 data length for tensor: " + name};
        if (!tensor.bfloat16_data.empty() || !tensor.int8_data.empty() || !tensor.quantization_scales.empty()
            || !tensor.mxfp4_blocks.empty() || !tensor.mxfp4_scales.empty())
            return Error{ErrorCode::InvalidModel, "float32 tensor contains quantized storage: " + name};
    }
    else if (dtype == DType::BFloat16) {
        if (tensor.bfloat16_data.size() != tensor.element_count())
            return Error{ErrorCode::InvalidModel, "invalid bfloat16 data length for tensor: " + name};
        if (!tensor.float32_data.empty() || !tensor.int8_data.empty() || !tensor.quantization_scales.empty()
            || !tensor.mxfp4_blocks.empty() || !tensor.mxfp4_scales.empty())
            return Error{ErrorCode::InvalidModel, "bfloat16 tensor contains unrelated storage: " + name};
    }
    else if (dtype == DType::Int8) {
        if (tensor.shape.size() != 2)
            return Error{ErrorCode::InvalidModel, "int8 tensor must be a matrix: " + name};
        if (tensor.int8_data.size() != tensor.element_count())
            return Error{ErrorCode::InvalidModel, "invalid int8 data length for tensor: " + name};
        if (tensor.quantization_scales.size() != tensor.shape[0])
            return Error{ErrorCode::InvalidModel, "invalid per-row scale count for tensor: " + name};
        if (!tensor.float32_data.empty())
            return Error{ErrorCode::InvalidModel, "int8 tensor contains float32 storage: " + name};
        for (float scale : tensor.quantization_scales) {
            if (!std::isfinite(scale) || scale <= 0.0f)
                return Error{ErrorCode::InvalidModel, "invalid int8 scale for tensor: " + name};
        }
    }
    else if (dtype == DType::MxFp4) {
        if (tensor.shape.size() != 2 || tensor.shape[1] % 32 != 0)
            return Error{ErrorCode::InvalidModel, "MXFP4 tensor must be a matrix with 32-aligned columns: " + name};
        if (tensor.mxfp4_file_storage) {
            if (!tensor.mxfp4_blocks.empty() || !tensor.mxfp4_scales.empty()
                || tensor.mxfp4_file_storage->blocks_path.empty()
                || tensor.mxfp4_file_storage->scales_path.empty()
                || tensor.mxfp4_file_storage->blocks_bytes != tensor.element_count() / 2
                || tensor.mxfp4_file_storage->scales_bytes != tensor.element_count() / 32) {
                return Error{ErrorCode::InvalidModel, "invalid file-backed MXFP4 storage: " + name};
            }
        }
        else {
            if (tensor.mxfp4_blocks.size() != tensor.element_count() / 2)
                return Error{ErrorCode::InvalidModel, "invalid MXFP4 block data length for tensor: " + name};
            if (tensor.mxfp4_scales.size() != tensor.element_count() / 32)
                return Error{ErrorCode::InvalidModel, "invalid MXFP4 scale data length for tensor: " + name};
        }
        if (!tensor.float32_data.empty() || !tensor.bfloat16_data.empty() || !tensor.int8_data.empty()
            || !tensor.quantization_scales.empty())
            return Error{ErrorCode::InvalidModel, "MXFP4 tensor contains unrelated storage: " + name};
    }
    else {
        return Error{ErrorCode::UnsupportedModel, "unsupported tensor dtype: " + name};
    }
    return handle;
}

Result<TensorHandle> WeightTable::add(std::string name, TensorData tensor)
{
    if (name.empty())
        return Error{ErrorCode::InvalidArgument, "tensor name cannot be empty"};
    if (handles_.contains(name))
        return Error{ErrorCode::InvalidModel, "duplicate tensor: " + name};

    const TensorHandle handle = static_cast<TensorHandle>(tensors_.size());
    tensors_.push_back(std::move(tensor));
    handles_.emplace(std::move(name), handle);
    return handle;
}

const TensorData& WeightTable::at(TensorHandle handle) const
{
    assert(handle < tensors_.size());
    return tensors_[handle];
}

TensorData& WeightTable::at_mutable(TensorHandle handle)
{
    assert(handle < tensors_.size());
    return tensors_[handle];
}

const TensorData* WeightTable::find(const std::string& name) const noexcept
{
    const TensorHandle handle = find_handle(name);
    return handle == invalid_tensor_handle ? nullptr : &tensors_[handle];
}

TensorHandle WeightTable::find_handle(const std::string& name) const noexcept
{
    const auto it = handles_.find(name);
    return it == handles_.end() ? invalid_tensor_handle : it->second;
}

static Result<void> prepare_linear_operator(
    WeightTable& weights,
    TensorHandle matrix_handle,
    TensorHandle bias_handle,
    NcnnLinearDevice device,
    bool retain_cpu_dense_copy)
{
    TensorData& matrix = weights.at_mutable(matrix_handle);
    if (device == NcnnLinearDevice::Cpu && !retain_cpu_dense_copy)
        return {};
    const TensorData* bias = bias_handle == invalid_tensor_handle ? nullptr : &weights.at(bias_handle);
    matrix.linear_operator = NcnnLinearOperator::create(matrix, bias, device);
    if (device == NcnnLinearDevice::Vulkan && !matrix.linear_operator)
        return Error{ErrorCode::InternalError, "failed to create Vulkan InnerProduct operator"};
    return {};
}

Result<CompiledModel> ModelCompiler::compile(
    MoeModelDescriptor descriptor,
    WeightMapping mapping,
    HybridMode hybrid_mode) const
{
    BackendCapabilities capabilities;
    capabilities.vulkan_dense = hybrid_mode == HybridMode::HybridExperts
                                 || hybrid_mode == HybridMode::VulkanWithCpuPrefetch;
    capabilities.vulkan_attention = capabilities.vulkan_dense;
    return compile(
        std::move(descriptor),
        std::move(mapping),
        hybrid_mode,
        capabilities);
}

Result<CompiledModel> ModelCompiler::compile(
    MoeModelDescriptor descriptor,
    WeightMapping mapping,
    HybridMode hybrid_mode,
    const BackendCapabilities& capabilities) const
{
    if (!capabilities.cpu_execution)
        return Error{ErrorCode::UnsupportedModel, "compiler requires a CPU execution backend"};
    if (descriptor.vocabulary_size == 0 || descriptor.hidden_size == 0 || descriptor.layer_count == 0
        || descriptor.intermediate_size == 0 || descriptor.expert_count == 0
        || descriptor.experts_per_token == 0)
        return Error{ErrorCode::InvalidModel, "model dimensions must be non-zero"};
    if (descriptor.layers.size() != descriptor.layer_count)
        return Error{ErrorCode::InvalidModel, "layer_count does not match layers"};
    if (descriptor.activation_dtype != DType::Float32 && descriptor.activation_dtype != DType::BFloat16)
        return Error{ErrorCode::UnsupportedModel, "dense weights must use float32 or bfloat16"};
    if (descriptor.kv_cache_dtype != DType::Float32 && descriptor.kv_cache_dtype != DType::BFloat16)
        return Error{ErrorCode::UnsupportedModel, "KV cache must use float32 or bfloat16"};
    if (descriptor.norm_epsilon <= 0.0f)
        return Error{ErrorCode::InvalidModel, "norm_epsilon must be positive"};

    CompiledModel compiled;
    compiled.descriptor = std::move(descriptor);
    const bool hybrid_requests_vulkan = hybrid_mode == HybridMode::HybridExperts
                                        || hybrid_mode == HybridMode::VulkanWithCpuPrefetch;
    const bool use_vulkan_dense = hybrid_requests_vulkan && capabilities.vulkan_dense;
    compiled.hybrid_mode = use_vulkan_dense ? hybrid_mode : HybridMode::CpuOnly;
    const NcnnLinearDevice dense_device = use_vulkan_dense
                                              ? NcnnLinearDevice::Vulkan
                                              : NcnnLinearDevice::Cpu;

    for (auto& [name, tensor] : mapping.tensors) {
        auto added = compiled.weights.add(name, std::move(tensor));
        if (!added)
            return added.error();
    }

    auto embedding = require_tensor(
        compiled.weights, "token_embedding.weight",
        {compiled.descriptor.vocabulary_size, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
    if (!embedding)
        return embedding.error();
    compiled.token_embedding = embedding.value();

    auto final_norm = require_tensor(
        compiled.weights, "final_norm.weight", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
    if (!final_norm)
        return final_norm.error();
    compiled.final_norm_weight = final_norm.value();

    auto lm_head = require_tensor(
        compiled.weights, "lm_head.weight",
        {compiled.descriptor.vocabulary_size, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
    if (!lm_head)
        return lm_head.error();
    compiled.lm_head_weight = lm_head.value();
    auto prepared = prepare_linear_operator(
        compiled.weights,
        compiled.lm_head_weight,
        invalid_tensor_handle,
        dense_device,
        capabilities.retain_cpu_dense_copies);
    if (!prepared)
        return prepared.error();

    compiled.layers.reserve(compiled.descriptor.layer_count);
    for (uint32_t layer_id = 0; layer_id < compiled.descriptor.layer_count; ++layer_id) {
        const LayerDescriptor& layer = compiled.descriptor.layers[layer_id];
        auto parsed_graph = validate_adapter_graph(layer);
        if (!parsed_graph)
            return parsed_graph.error();
        const AdapterGraph graph = parsed_graph.value();
        const MoeDescriptor& moe = layer.ffn.moe;
        if (!graph.use_moe)
            return Error{
                ErrorCode::UnsupportedModel,
                "the current executor requires an ExpertGroup in every layer graph"};
        if (layer.pre_ffn_norm != NormType::RmsNorm)
            return Error{ErrorCode::UnsupportedModel, "the reference runtime requires RMSNorm before each MoE block"};
        if (moe.expert_count == 0 || moe.top_k == 0 || moe.top_k > moe.expert_count)
            return Error{ErrorCode::InvalidModel, "invalid expert_count/top_k"};
        if (moe.intermediate_size == 0)
            return Error{ErrorCode::InvalidModel, "intermediate_size must be non-zero"};
        if (moe.expert_count != compiled.descriptor.expert_count
            || moe.top_k != compiled.descriptor.experts_per_token
            || moe.intermediate_size != compiled.descriptor.intermediate_size) {
            return Error{ErrorCode::InvalidModel, "layer MoE dimensions do not match the model descriptor"};
        }
        if (moe.use_shared_expert)
            return Error{ErrorCode::UnsupportedModel, "shared experts are reserved for a later phase"};
        if (moe.expert_weight_dtype != DType::Float32 && moe.expert_weight_dtype != DType::Int8
            && moe.expert_weight_dtype != DType::MxFp4)
            return Error{ErrorCode::UnsupportedModel, "expert weights must use float32, int8, or MXFP4"};
        if (moe.expert_weight_dtype == DType::MxFp4
            && !capabilities.mxfp4_cpu_kernel) {
            return Error{
                ErrorCode::UnsupportedModel,
                "backend capabilities do not provide an MXFP4 CPU expert kernel"};
        }

        CompiledLayerPlan layer_plan;
        layer_plan.layer_id = layer_id;
        layer_plan.use_attention = graph.use_attention;
        const bool use_vulkan_attention
            = graph.use_attention
              && use_vulkan_dense
              && capabilities.vulkan_attention;
        layer_plan.nodes.reserve(layer.nodes.size());
        for (size_t node_index = 0;
             node_index < layer.nodes.size();
             ++node_index) {
            const ExecutionBackend backend
                = use_vulkan_attention
                          && node_index < graph.attention_node_count
                      ? ExecutionBackend::Vulkan
                      : ExecutionBackend::Cpu;
            layer_plan.nodes.push_back({
                layer.nodes[node_index].type,
                backend,
            });
        }
        layer_plan.moe.top_k = moe.top_k;
        layer_plan.moe.hidden_size = compiled.descriptor.hidden_size;
        layer_plan.moe.normalization = moe.normalization;
        layer_plan.moe.normalize_topk_weights = moe.normalize_topk_weights;

        const std::string layer_name = layer_prefix(layer_id);
        if (graph.use_attention) {
            const AttentionDescriptor& attention = layer.attention;
            const NcnnLinearDevice attention_device
                = use_vulkan_attention
                      ? NcnnLinearDevice::Vulkan
                      : NcnnLinearDevice::Cpu;
            if (layer.pre_attention_norm != NormType::RmsNorm)
                return Error{ErrorCode::UnsupportedModel, "attention requires a pre-attention RMSNorm"};
            if (attention.head_count == 0 || attention.kv_head_count == 0 || attention.head_dimension == 0
                || attention.head_count % attention.kv_head_count != 0
                || attention.head_dimension % 2 != 0)
                return Error{ErrorCode::InvalidModel, "invalid attention dimensions"};
            if (attention.head_count != compiled.descriptor.attention_head_count
                || attention.kv_head_count != compiled.descriptor.kv_head_count
                || attention.head_dimension != compiled.descriptor.head_dimension)
                return Error{ErrorCode::InvalidModel, "layer attention dimensions do not match the model descriptor"};
            AttentionBlockPlan& plan = layer_plan.attention;
            plan.head_count = attention.head_count;
            plan.kv_head_count = attention.kv_head_count;
            plan.head_dimension = attention.head_dimension;
            plan.sliding_window = attention.sliding_window;
            plan.initial_context_length = attention.initial_context_length;
            plan.max_context_length = attention.max_context_length;
            plan.rope_theta = attention.rope_theta;
            plan.rope_scaling_factor = attention.rope_scaling_factor;
            plan.rope_ntk_alpha = attention.rope_ntk_alpha;
            plan.rope_ntk_beta = attention.rope_ntk_beta;
            plan.use_attention_sink = graph.use_attention_sink;

            const uint32_t query_size = attention.head_count * attention.head_dimension;
            const uint32_t key_value_size = attention.kv_head_count * attention.head_dimension;
            auto attention_norm = require_tensor(compiled.weights, layer_name + "pre_attention_norm.weight", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
            auto query_weight = require_tensor(compiled.weights, layer_name + "attention.query.weight", {query_size, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
            auto key_weight = require_tensor(compiled.weights, layer_name + "attention.key.weight", {key_value_size, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
            auto value_weight = require_tensor(compiled.weights, layer_name + "attention.value.weight", {key_value_size, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
            auto output_weight = require_tensor(compiled.weights, layer_name + "attention.output.weight", {compiled.descriptor.hidden_size, query_size}, compiled.descriptor.activation_dtype);
            Result<TensorHandle> query_bias = invalid_tensor_handle;
            Result<TensorHandle> key_bias = invalid_tensor_handle;
            Result<TensorHandle> value_bias = invalid_tensor_handle;
            Result<TensorHandle> output_bias = invalid_tensor_handle;
            if (attention.use_bias) {
                query_bias = require_tensor(
                    compiled.weights,
                    layer_name + "attention.query.bias",
                    {query_size},
                    compiled.descriptor.activation_dtype);
                key_bias = require_tensor(
                    compiled.weights,
                    layer_name + "attention.key.bias",
                    {key_value_size},
                    compiled.descriptor.activation_dtype);
                value_bias = require_tensor(
                    compiled.weights,
                    layer_name + "attention.value.bias",
                    {key_value_size},
                    compiled.descriptor.activation_dtype);
                output_bias = require_tensor(
                    compiled.weights,
                    layer_name + "attention.output.bias",
                    {compiled.descriptor.hidden_size},
                    compiled.descriptor.activation_dtype);
            }
            Result<TensorHandle> sinks = invalid_tensor_handle;
            if (plan.use_attention_sink) {
                sinks = require_tensor(
                    compiled.weights,
                    layer_name + "attention.sinks",
                    {attention.head_count},
                    compiled.descriptor.activation_dtype);
            }
            if (!attention_norm || !query_weight || !query_bias || !key_weight || !key_bias || !value_weight || !value_bias || !output_weight || !output_bias || !sinks) {
                const Error* error = !attention_norm  ? &attention_norm.error()
                                     : !query_weight  ? &query_weight.error()
                                     : !query_bias    ? &query_bias.error()
                                     : !key_weight    ? &key_weight.error()
                                     : !key_bias      ? &key_bias.error()
                                     : !value_weight  ? &value_weight.error()
                                     : !value_bias    ? &value_bias.error()
                                     : !output_weight ? &output_weight.error()
                                     : !output_bias   ? &output_bias.error()
                                                      : &sinks.error();
                return *error;
            }
            plan.pre_attention_norm_weight = attention_norm.value();
            plan.query_weight = query_weight.value();
            plan.query_bias = query_bias.value();
            plan.key_weight = key_weight.value();
            plan.key_bias = key_bias.value();
            plan.value_weight = value_weight.value();
            plan.value_bias = value_bias.value();
            plan.output_weight = output_weight.value();
            plan.output_bias = output_bias.value();
            plan.sinks = sinks.value();
            if (attention_device == NcnnLinearDevice::Vulkan) {
                const TensorData* query_bias_data = plan.query_bias == invalid_tensor_handle
                                                        ? nullptr
                                                        : &compiled.weights.at(plan.query_bias);
                const TensorData* key_bias_data = plan.key_bias == invalid_tensor_handle
                                                      ? nullptr
                                                      : &compiled.weights.at(plan.key_bias);
                const TensorData* value_bias_data = plan.value_bias == invalid_tensor_handle
                                                        ? nullptr
                                                        : &compiled.weights.at(plan.value_bias);
                plan.fused_qkv_operator = NcnnLinearOperator::create_fused(
                    {
                        &compiled.weights.at(plan.query_weight),
                        &compiled.weights.at(plan.key_weight),
                        &compiled.weights.at(plan.value_weight),
                    },
                    {
                        query_bias_data,
                        key_bias_data,
                        value_bias_data,
                    },
                    attention_device);
                if (!plan.fused_qkv_operator)
                    return Error{ErrorCode::InternalError, "failed to create fused Vulkan QKV operator"};
            }
            else {
                prepared = prepare_linear_operator(compiled.weights, plan.query_weight, plan.query_bias, attention_device, capabilities.retain_cpu_dense_copies);
                if (!prepared)
                    return prepared.error();
                prepared = prepare_linear_operator(compiled.weights, plan.key_weight, plan.key_bias, attention_device, capabilities.retain_cpu_dense_copies);
                if (!prepared)
                    return prepared.error();
                prepared = prepare_linear_operator(compiled.weights, plan.value_weight, plan.value_bias, attention_device, capabilities.retain_cpu_dense_copies);
                if (!prepared)
                    return prepared.error();
            }
            prepared = prepare_linear_operator(
                compiled.weights,
                plan.output_weight,
                plan.output_bias,
                attention_device,
                capabilities.retain_cpu_dense_copies);
            if (!prepared)
                return prepared.error();
            if (attention_device == NcnnLinearDevice::Vulkan) {
                NcnnVulkanAttentionConfig attention_config;
                attention_config.hidden_size = compiled.descriptor.hidden_size;
                attention_config.head_count = plan.head_count;
                attention_config.kv_head_count = plan.kv_head_count;
                attention_config.head_dimension = plan.head_dimension;
                attention_config.sliding_window = plan.sliding_window;
                attention_config.initial_context_length = plan.initial_context_length;
                attention_config.norm_epsilon = compiled.descriptor.norm_epsilon;
                attention_config.rope_theta = plan.rope_theta;
                attention_config.rope_scaling_factor = plan.rope_scaling_factor;
                attention_config.rope_ntk_alpha = plan.rope_ntk_alpha;
                attention_config.rope_ntk_beta = plan.rope_ntk_beta;
                attention_config.activation_dtype = compiled.descriptor.activation_dtype;
                attention_config.kv_cache_dtype = compiled.descriptor.kv_cache_dtype;
                attention_config.use_attention_sink = plan.use_attention_sink;
                plan.vulkan_attention_operator = NcnnVulkanAttentionOperator::create(
                    compiled.weights.at(plan.pre_attention_norm_weight),
                    plan.sinks == invalid_tensor_handle
                        ? nullptr
                        : &compiled.weights.at(plan.sinks),
                    plan.fused_qkv_operator,
                    compiled.weights.at(plan.output_weight).linear_operator,
                    attention_config);
            }
        }

        auto norm = require_tensor(
            compiled.weights, layer_name + "pre_ffn_norm.weight", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
        if (!norm)
            return norm.error();
        layer_plan.moe.pre_ffn_norm_weight = norm.value();

        auto router = require_tensor(
            compiled.weights, layer_name + "router.weight",
            {moe.expert_count, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
        if (!router)
            return router.error();
        layer_plan.moe.router_weight = router.value();

        if (moe.use_router_bias) {
            auto bias = require_tensor(compiled.weights, layer_name + "router.bias", {moe.expert_count}, compiled.descriptor.activation_dtype);
            if (!bias)
                return bias.error();
            layer_plan.moe.router_bias = bias.value();
        }
        // Routing and experts form the CPU half of the heterogeneous boundary.
        // Keeping the router on Vulkan would add an avoidable upload, submit,
        // and download between the dense attention block and CPU experts.
        prepared = prepare_linear_operator(
            compiled.weights,
            layer_plan.moe.router_weight,
            layer_plan.moe.router_bias,
            NcnnLinearDevice::Cpu,
            capabilities.retain_cpu_dense_copies);
        if (!prepared)
            return prepared.error();

        layer_plan.moe.experts.reserve(moe.expert_count);
        for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id) {
            ExpertPlan expert;
            expert.activation = moe.activation;
            expert.activation_limit = moe.activation_limit;
            expert.gated = moe.layout == ExpertLayout::GateUpDown || moe.layout == ExpertLayout::InterleavedGateUpDown;
            const std::string prefix = expert_prefix(layer_id, expert_id);

            if (moe.layout == ExpertLayout::InterleavedGateUpDown) {
                auto gate_up = require_tensor(
                    compiled.weights, prefix + "gate_up.weight",
                    {moe.intermediate_size * 2, compiled.descriptor.hidden_size}, moe.expert_weight_dtype);
                if (!gate_up)
                    return gate_up.error();
                expert.gate_up_weight = gate_up.value();
            }
            else if (expert.gated) {
                auto gate = require_tensor(
                    compiled.weights, prefix + "gate.weight",
                    {moe.intermediate_size, compiled.descriptor.hidden_size}, moe.expert_weight_dtype);
                if (!gate)
                    return gate.error();
                expert.gate_weight = gate.value();
                (void)prepare_linear_operator(
                    compiled.weights,
                    expert.gate_weight,
                    invalid_tensor_handle,
                    NcnnLinearDevice::Cpu,
                    capabilities.retain_cpu_dense_copies);
            }

            if (moe.layout != ExpertLayout::InterleavedGateUpDown) {
                auto up = require_tensor(
                    compiled.weights, prefix + "up.weight",
                    {moe.intermediate_size, compiled.descriptor.hidden_size}, moe.expert_weight_dtype);
                if (!up)
                    return up.error();
                expert.up_weight = up.value();
                (void)prepare_linear_operator(
                    compiled.weights,
                    expert.up_weight,
                    invalid_tensor_handle,
                    NcnnLinearDevice::Cpu,
                    capabilities.retain_cpu_dense_copies);
            }

            auto down = require_tensor(
                compiled.weights, prefix + "down.weight",
                {compiled.descriptor.hidden_size, moe.intermediate_size}, moe.expert_weight_dtype);
            if (!down)
                return down.error();
            expert.down_weight = down.value();

            if (moe.use_projection_bias) {
                auto gate_up_bias = require_tensor(
                    compiled.weights, prefix + "gate_up.bias", {moe.intermediate_size * 2}, compiled.descriptor.activation_dtype);
                auto down_bias = require_tensor(
                    compiled.weights, prefix + "down.bias", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
                if (!gate_up_bias)
                    return gate_up_bias.error();
                if (!down_bias)
                    return down_bias.error();
                expert.gate_up_bias = gate_up_bias.value();
                expert.down_bias = down_bias.value();
            }
            (void)prepare_linear_operator(
                compiled.weights,
                expert.down_weight,
                expert.down_bias,
                NcnnLinearDevice::Cpu,
                capabilities.retain_cpu_dense_copies);

            layer_plan.moe.experts.push_back(expert);
        }

        compiled.layers.push_back(std::move(layer_plan));
    }

    return compiled;
}

} // namespace moe
} // namespace ncnn
