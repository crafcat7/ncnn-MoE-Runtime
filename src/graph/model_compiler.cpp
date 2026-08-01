#include "ncnn/moe/execution_plan.h"

#include "compiler/moe_ir.hpp"
#include "kernels/cpu_mxfp4.h"
#include "models/internal/tensor_names.h"
#include "backends/ncnn/ncnn_attention.h"
#include "backends/ncnn/ncnn_linear.h"
#include "storage/expert_cache.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <utility>

namespace ncnn {
namespace moe {

static bool shape_equals(const TensorData& tensor, std::initializer_list<uint32_t> expected)
{
    return tensor.shape.size() == expected.size() && std::equal(tensor.shape.begin(), tensor.shape.end(), expected.begin());
}

#define NCNN_MOE_ADAPTER_GRAPH_ATTN_BIT   0
#define NCNN_MOE_ADAPTER_GRAPH_SINK_BIT   1
#define NCNN_MOE_ADAPTER_GRAPH_MOE_BIT    2
#define NCNN_MOE_ADAPTER_GRAPH_MLA_BIT    3
#define NCNN_MOE_ADAPTER_GRAPH_SHARED_BIT 4

enum AdapterGraphFlag : uint32_t
{
    AdapterGraphAttention = UINT32_C(1) << NCNN_MOE_ADAPTER_GRAPH_ATTN_BIT,
    AdapterGraphAttentionSink = UINT32_C(1) << NCNN_MOE_ADAPTER_GRAPH_SINK_BIT,
    AdapterGraphMoe = UINT32_C(1) << NCNN_MOE_ADAPTER_GRAPH_MOE_BIT,
    AdapterGraphMla = UINT32_C(1) << NCNN_MOE_ADAPTER_GRAPH_MLA_BIT,
    AdapterGraphSharedExpert = UINT32_C(1) << NCNN_MOE_ADAPTER_GRAPH_SHARED_BIT
};

struct AdapterGraph
{
    uint32_t flags = 0;
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
        && layer.nodes[2].type == ModelNodeType::Rope)
    {
        graph.flags |= AdapterGraphAttention;
        cursor = 3;
        if (cursor < layer.nodes.size() && layer.nodes[cursor].type == ModelNodeType::AttentionSink)
        {
            graph.flags |= AdapterGraphAttentionSink;
            ++cursor;
        }
        if (cursor + 1 >= layer.nodes.size() || layer.nodes[cursor].type != ModelNodeType::Sdpa || layer.nodes[cursor + 1].type != ModelNodeType::Projection)
        {
            return Error{ErrorCode::InvalidModel, "adapter attention graph must end with SDPA and Projection"};
        }
        cursor += 2;
        graph.attention_node_count = cursor;
    }
    else if (layer.nodes.size() >= 4
             && layer.nodes[0].type == ModelNodeType::RmsNorm
             && layer.nodes[1].type == ModelNodeType::MultiHeadLatentAttention
             && layer.nodes[2].type == ModelNodeType::AttentionSink
             && layer.nodes[3].type == ModelNodeType::Projection)
    {
        graph.flags |= AdapterGraphAttention | AdapterGraphAttentionSink | AdapterGraphMla;
        cursor = 4;
        graph.attention_node_count = cursor;
    }
    else if (layer.nodes.size() >= 3
             && layer.nodes[0].type == ModelNodeType::RmsNorm
             && layer.nodes[1].type == ModelNodeType::GatedDeltaNet
             && layer.nodes[2].type == ModelNodeType::Projection)
    {
        graph.flags |= AdapterGraphAttention;
        cursor = 3;
        graph.attention_node_count = cursor;
    }
    static constexpr ModelNodeType moe_prefix[] = {
        ModelNodeType::RmsNorm,
        ModelNodeType::Router,
        ModelNodeType::TopK,
        ModelNodeType::ExpertGroup,
    };
    constexpr size_t moe_prefix_count = sizeof(moe_prefix) / sizeof(moe_prefix[0]);
    bool has_moe = cursor + moe_prefix_count + 1 <= layer.nodes.size();
    for (size_t index = 0; has_moe && index < moe_prefix_count; ++index)
        has_moe = moe_prefix[index] == layer.nodes[cursor + index].type;
    if (has_moe)
    {
        graph.flags |= AdapterGraphMoe;
        cursor += moe_prefix_count;
        if (cursor < layer.nodes.size() && layer.nodes[cursor].type == ModelNodeType::SharedExpertGroup)
        {
            graph.flags |= AdapterGraphSharedExpert;
            ++cursor;
        }
        if (cursor >= layer.nodes.size() || layer.nodes[cursor].type != ModelNodeType::Combine)
            return Error{ErrorCode::InvalidModel, "adapter MoE graph must end with Combine"};
        ++cursor;
    }
    if (cursor != layer.nodes.size())
        return Error{ErrorCode::InvalidModel, "adapter operator graph has an invalid node order"};
    return graph;
}

static Result<TensorHandle> require_tensor(const WeightTable& weights, const std::string& name, std::initializer_list<uint32_t> shape, DType dtype)
{
    const TensorHandle handle = weights.find_handle(name);
    if (handle == invalid_tensor_handle)
        return Error{ErrorCode::InvalidModel, "missing tensor: " + name};

    const TensorData& tensor = weights.at(handle);
    if (tensor.dtype != dtype)
        return Error{ErrorCode::InvalidModel, "unexpected dtype for tensor: " + name};
    if (!shape_equals(tensor, shape))
        return Error{ErrorCode::InvalidModel, "unexpected shape for tensor: " + name};

    if (dtype == DType::Float32)
    {
        if (tensor.float32_values().size() != tensor.element_count())
            return Error{ErrorCode::InvalidModel, "invalid float32 data length for tensor: " + name};
        if (!tensor.float32_data.empty() && tensor.mapped_data)
            return Error{ErrorCode::InvalidModel, "float32 tensor has duplicate storage: " + name};
        if (!tensor.bfloat16_data.empty()
            || !tensor.int8_data.empty()
            || !tensor.quantization_scales.empty()
            || !tensor.mxfp4_blocks.empty()
            || !tensor.mxfp4_scales.empty())
            return Error{ErrorCode::InvalidModel, "float32 tensor contains quantized storage: " + name};
    }
    else if (dtype == DType::BFloat16)
    {
        if (tensor.bfloat16_values().size() != tensor.element_count())
            return Error{ErrorCode::InvalidModel, "invalid bfloat16 data length for tensor: " + name};
        if (!tensor.bfloat16_data.empty() && tensor.mapped_data)
            return Error{ErrorCode::InvalidModel, "bfloat16 tensor has duplicate storage: " + name};
        if (!tensor.float32_data.empty()
            || !tensor.int8_data.empty()
            || !tensor.quantization_scales.empty()
            || !tensor.mxfp4_blocks.empty()
            || !tensor.mxfp4_scales.empty())
            return Error{ErrorCode::InvalidModel, "bfloat16 tensor contains unrelated storage: " + name};
    }
    else if (dtype == DType::Int8)
    {
        if (tensor.shape.size() != 2)
            return Error{ErrorCode::InvalidModel, "int8 tensor must be a matrix: " + name};
        if (tensor.int8_values().size() != tensor.element_count())
            return Error{ErrorCode::InvalidModel, "invalid int8 data length for tensor: " + name};
        if (!tensor.int8_data.empty() && tensor.mapped_data)
            return Error{ErrorCode::InvalidModel, "int8 tensor has duplicate storage: " + name};
        if (tensor.quantization_scales.size() != tensor.shape[0])
            return Error{ErrorCode::InvalidModel, "invalid per-row scale count for tensor: " + name};
        if (!tensor.float32_data.empty())
            return Error{ErrorCode::InvalidModel, "int8 tensor contains float32 storage: " + name};
        for (float scale : tensor.quantization_scales)
        {
            if (!std::isfinite(scale) || scale <= 0.0f)
                return Error{ErrorCode::InvalidModel, "invalid int8 scale for tensor: " + name};
        }
    }
    else if (dtype == DType::Float8E4M3)
    {
        if (tensor.shape.size() != 2 || tensor.float8_values().size() != tensor.element_count())
            return Error{ErrorCode::InvalidModel, "invalid blockwise FP8 data length for tensor: " + name};
        const uint64_t output_blocks = (tensor.shape[0] + 127) / 128;
        const uint64_t input_blocks = (tensor.shape[1] + 127) / 128;
        if (tensor.quantization_scales.size() != output_blocks * input_blocks)
            return Error{ErrorCode::InvalidModel, "invalid blockwise FP8 scale count for tensor: " + name};
        for (float scale : tensor.quantization_scales)
        {
            if (!std::isfinite(scale) || scale <= 0.0f)
                return Error{ErrorCode::InvalidModel, "invalid blockwise FP8 scale for tensor: " + name};
        }
    }
    else if (dtype == DType::Int64)
    {
        if (tensor.int64_values().size() != tensor.element_count())
            return Error{ErrorCode::InvalidModel, "invalid int64 data length for tensor: " + name};
    }
    else if (dtype == DType::MxFp4)
    {
        if (tensor.shape.size() != 2 || tensor.shape[1] % 32 != 0)
            return Error{ErrorCode::InvalidModel, "MXFP4 tensor must be a matrix with 32-aligned columns: " + name};
        if (tensor.mxfp4_file_storage)
        {
            const uint64_t stored_blocks = tensor.mxfp4_file_storage->blocks_bytes + tensor.mxfp4_file_storage->secondary_blocks_bytes;
            const uint64_t stored_scales = tensor.mxfp4_file_storage->scales_bytes + tensor.mxfp4_file_storage->secondary_scales_bytes;
            if (!tensor.mxfp4_blocks.empty()
                || !tensor.mxfp4_scales.empty()
                || tensor.mxfp4_file_storage->blocks_path.empty()
                || tensor.mxfp4_file_storage->scales_path.empty()
                || stored_blocks != tensor.element_count() / 2
                || stored_scales != tensor.element_count() / 32)
            {
                return Error{ErrorCode::InvalidModel, "invalid file-backed MXFP4 storage: " + name};
            }
        }
        else
        {
            if (tensor.mxfp4_blocks.size() != tensor.element_count() / 2)
                return Error{ErrorCode::InvalidModel, "invalid MXFP4 block data length for tensor: " + name};
            if (tensor.mxfp4_scales.size() != tensor.element_count() / 32)
                return Error{ErrorCode::InvalidModel, "invalid MXFP4 scale data length for tensor: " + name};
        }
        if (!tensor.float32_data.empty()
            || !tensor.bfloat16_data.empty()
            || !tensor.int8_data.empty()
            || !tensor.quantization_scales.empty()
            || tensor.mapped_data)
            return Error{ErrorCode::InvalidModel, "MXFP4 tensor contains unrelated storage: " + name};
    }
    else
    {
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
    bool retain_cpu_dense_copy,
    uint32_t vulkan_device_index,
    uint32_t input_group_count = 1,
    bool prefer_bfloat16_vulkan = true)
{
    TensorData& matrix = weights.at_mutable(matrix_handle);
    const TensorData* bias = bias_handle == invalid_tensor_handle ? nullptr : &weights.at(bias_handle);
    if (matrix.dtype == DType::Float8E4M3)
    {
        if (device == NcnnLinearDevice::Vulkan)
        {
            matrix.float8_linear_operator = NcnnVulkanFloat8Operator::create(matrix, bias, input_group_count, vulkan_device_index);
            if (!matrix.float8_linear_operator)
                return Error{ErrorCode::InternalError, "failed to create Vulkan FP8 Linear operator"};
        }
        return {};
    }
    if (matrix.dtype != DType::Float32 && matrix.dtype != DType::BFloat16)
        return {};
    if (device == NcnnLinearDevice::Cpu && !retain_cpu_dense_copy)
        return {};
    if (device == NcnnLinearDevice::Vulkan
        && matrix.dtype == DType::BFloat16
        && prefer_bfloat16_vulkan)
    {
        matrix.bfloat16_linear_operator =
            NcnnVulkanBfloat16Operator::create(
                matrix,
                bias,
                vulkan_device_index);
        if (matrix.bfloat16_linear_operator)
            return {};
    }
    matrix.linear_operator = NcnnLinearOperator::create(matrix, bias, device, vulkan_device_index);
    if (device == NcnnLinearDevice::Vulkan && !matrix.linear_operator)
        return Error{ErrorCode::InternalError, "failed to create Vulkan InnerProduct operator"};
    return {};
}

static Result<void> prepare_shared_expert_operators(
    WeightTable& weights,
    MoeBlockPlan& moe,
    NcnnLinearDevice device,
    bool retain_cpu_dense_copy,
    uint32_t vulkan_device_index)
{
    ExpertPlan& shared = moe.shared_expert;
    const TensorData& gate = weights.at(shared.gate_weight);
    const TensorData& up = weights.at(shared.up_weight);
    const bool has_router_gate =
        moe.shared_expert_gate_weight != invalid_tensor_handle;
    if (device == NcnnLinearDevice::Vulkan
        && gate.dtype == DType::BFloat16
        && up.dtype == DType::BFloat16
        && (!has_router_gate
            || weights.at(moe.shared_expert_gate_weight).dtype
                   == DType::BFloat16))
    {
        std::vector<const TensorData*> matrices = {
            &gate,
            &up,
        };
        std::vector<const TensorData*> biases = {
            nullptr,
            nullptr,
        };
        if (has_router_gate)
        {
            matrices.push_back(
                &weights.at(moe.shared_expert_gate_weight));
            biases.push_back(nullptr);
        }
        moe.fused_shared_input_bfloat16_operator =
            NcnnVulkanBfloat16Operator::create_fused(
                matrices,
                biases,
                vulkan_device_index);
    }

    if (!moe.fused_shared_input_bfloat16_operator)
    {
        const TensorHandle input_handles[] = {
            shared.gate_weight,
            shared.up_weight,
            moe.shared_expert_gate_weight,
        };
        for (TensorHandle handle : input_handles)
        {
            if (handle == invalid_tensor_handle)
                continue;
            auto prepared = prepare_linear_operator(
                weights,
                handle,
                invalid_tensor_handle,
                device,
                retain_cpu_dense_copy,
                vulkan_device_index);
            if (!prepared)
                return prepared.error();
        }
    }
    return prepare_linear_operator(
        weights,
        shared.down_weight,
        invalid_tensor_handle,
        device,
        retain_cpu_dense_copy,
        vulkan_device_index);
}

static Result<void> assign_required_tensor(
    const WeightTable& weights,
    const std::string& name,
    std::initializer_list<uint32_t> shape,
    DType dtype,
    TensorHandle& destination)
{
    auto tensor = require_tensor(weights, name, shape, dtype);
    if (!tensor)
        return tensor.error();
    destination = tensor.value();
    return {};
}

static Result<void> compile_latent_attention(
    const WeightTable& weights,
    const std::string& layer_name,
    const MoeIR& descriptor,
    const AttentionDescriptor& attention,
    AttentionBlockPlan& plan)
{
    if (attention.query_lora_rank == 0
        || attention.head_count == 0
        || attention.head_dimension == 0
        || attention.qk_rope_head_dimension == 0
        || attention.qk_rope_head_dimension >= attention.head_dimension
        || attention.output_group_count == 0
        || attention.head_count % attention.output_group_count != 0
        || attention.output_lora_rank == 0)
        return Error{ErrorCode::InvalidModel, "invalid multi-head latent attention dimensions"};
    if (attention.compression_ratio != 0 && attention.compression_ratio != 4 && attention.compression_ratio != 128)
        return Error{ErrorCode::UnsupportedModel, "compressed latent attention only supports ratios 4 and 128"};

    plan.flags |= AttentionBlockLatent | AttentionBlockSink | AttentionBlockQueryKeyNorm;
    if (attention.compression_ratio != 0)
        plan.flags |= AttentionBlockCompressed;
    plan.head_count = attention.head_count;
    plan.kv_head_count = attention.kv_head_count;
    plan.head_dimension = attention.head_dimension;
    plan.sliding_window = attention.sliding_window;
    plan.initial_context_length = attention.initial_context_length;
    plan.max_context_length = attention.max_context_length;
    plan.query_lora_rank = attention.query_lora_rank;
    plan.rope_head_dimension = attention.qk_rope_head_dimension;
    plan.output_lora_rank = attention.output_lora_rank;
    plan.output_group_count = attention.output_group_count;
    plan.compression_ratio = attention.compression_ratio;
    plan.index_head_count = attention.index_head_count;
    plan.index_head_dimension = attention.index_head_dimension;
    plan.index_top_k = attention.index_top_k;
    plan.rope_theta = attention.rope_theta;
    plan.compressed_rope_theta = attention.compressed_rope_theta;
    plan.rope_scaling_factor = attention.rope_scaling_factor;
    plan.rope_ntk_alpha = attention.rope_ntk_alpha;
    plan.rope_ntk_beta = attention.rope_ntk_beta;

    Result<void> status = assign_required_tensor(weights, layer_name + "pre_attention_norm.weight", {descriptor.hidden_size}, descriptor.activation_dtype, plan.pre_attention_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.query_a.weight", {attention.query_lora_rank, descriptor.hidden_size}, attention.projection_weight_dtype, plan.query_a_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.query_norm.weight", {attention.query_lora_rank}, descriptor.activation_dtype, plan.query_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.query_b.weight", {attention.head_count * attention.head_dimension, attention.query_lora_rank}, attention.projection_weight_dtype, plan.query_b_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.key_value.weight", {attention.head_dimension, descriptor.hidden_size}, attention.projection_weight_dtype, plan.key_value_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.key_value_norm.weight", {attention.head_dimension}, descriptor.activation_dtype, plan.key_value_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.sinks", {attention.head_count}, DType::Float32, plan.sinks);
    if (!status)
        return status.error();
    const uint32_t group_input = attention.head_count * attention.head_dimension / attention.output_group_count;
    status = assign_required_tensor(weights, layer_name + "attention.output_a.weight", {attention.output_group_count * attention.output_lora_rank, group_input}, attention.projection_weight_dtype, plan.output_a_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.output_b.weight", {descriptor.hidden_size, attention.output_group_count * attention.output_lora_rank}, attention.projection_weight_dtype, plan.output_b_weight);
    if (!status)
        return status.error();

    if (attention.compression_ratio == 0)
        return {};

    const uint32_t compressor_multiplier = attention.compression_ratio == 4 ? 2 : 1;
    status = assign_required_tensor(weights, layer_name + "attention.compressor.position", {attention.compression_ratio, compressor_multiplier * attention.head_dimension}, DType::Float32, plan.compressor_position);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.compressor.norm.weight", {attention.head_dimension}, descriptor.activation_dtype, plan.compressor_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.compressor.key_value.weight", {compressor_multiplier * attention.head_dimension, descriptor.hidden_size}, descriptor.activation_dtype, plan.compressor_key_value_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.compressor.gate.weight", {compressor_multiplier * attention.head_dimension, descriptor.hidden_size}, descriptor.activation_dtype, plan.compressor_gate_weight);
    if (!status)
        return status.error();

    if (attention.compression_ratio != 4)
        return {};
    if (attention.index_head_count == 0 || attention.index_head_dimension == 0 || attention.index_top_k == 0)
        return Error{ErrorCode::InvalidModel, "ratio-4 compressed attention requires an indexer"};
    status = assign_required_tensor(weights, layer_name + "attention.indexer.compressor.position", {attention.compression_ratio, 2 * attention.index_head_dimension}, DType::Float32, plan.indexer_compressor_position);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.indexer.compressor.norm.weight", {attention.index_head_dimension}, descriptor.activation_dtype, plan.indexer_compressor_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.indexer.compressor.key_value.weight", {2 * attention.index_head_dimension, descriptor.hidden_size}, descriptor.activation_dtype, plan.indexer_compressor_key_value_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.indexer.compressor.gate.weight", {2 * attention.index_head_dimension, descriptor.hidden_size}, descriptor.activation_dtype, plan.indexer_compressor_gate_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(weights, layer_name + "attention.indexer.query.weight", {attention.index_head_count * attention.index_head_dimension, attention.query_lora_rank}, attention.projection_weight_dtype, plan.indexer_query_weight);
    if (!status)
        return status.error();
    return assign_required_tensor(weights, layer_name + "attention.indexer.weights.weight", {attention.index_head_count, descriptor.hidden_size}, descriptor.activation_dtype, plan.indexer_weights_weight);
}

static Result<void> compile_gated_delta_attention(
    const WeightTable& weights,
    const std::string& layer_name,
    const MoeIR& descriptor,
    const AttentionDescriptor& attention,
    AttentionBlockPlan& plan)
{
    if (attention.head_count == 0
        || attention.kv_head_count == 0
        || attention.head_count % attention.kv_head_count != 0
        || attention.head_dimension == 0
        || attention.value_head_dimension == 0
        || attention.convolution_kernel_size == 0)
    {
        return Error{ErrorCode::InvalidModel, "invalid gated DeltaNet dimensions"};
    }

    plan.flags |= AttentionBlockGatedDeltaNet;
    plan.head_count = attention.head_count;
    plan.kv_head_count = attention.kv_head_count;
    plan.head_dimension = attention.head_dimension;
    plan.value_head_dimension = attention.value_head_dimension;
    plan.convolution_kernel_size = attention.convolution_kernel_size;
    plan.max_context_length = attention.max_context_length;
    plan.norm_weight_offset = descriptor.norm_weight_offset;

    const uint32_t key_size = attention.kv_head_count * attention.head_dimension;
    const uint32_t value_size = attention.head_count * attention.value_head_dimension;
    const uint32_t convolution_size = key_size * 2 + value_size;
    Result<void> status = assign_required_tensor(
        weights,
        layer_name + "pre_attention_norm.weight",
        {descriptor.hidden_size},
        descriptor.activation_dtype,
        plan.pre_attention_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights,
        layer_name + "attention.delta.qkv.weight",
        {convolution_size, descriptor.hidden_size},
        descriptor.activation_dtype,
        plan.delta_qkv_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights,
        layer_name + "attention.delta.z.weight",
        {value_size, descriptor.hidden_size},
        descriptor.activation_dtype,
        plan.delta_z_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights,
        layer_name + "attention.delta.beta.weight",
        {attention.head_count, descriptor.hidden_size},
        descriptor.activation_dtype,
        plan.delta_beta_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights,
        layer_name + "attention.delta.alpha.weight",
        {attention.head_count, descriptor.hidden_size},
        descriptor.activation_dtype,
        plan.delta_alpha_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights,
        layer_name + "attention.delta.convolution.weight",
        {convolution_size, 1, attention.convolution_kernel_size},
        descriptor.activation_dtype,
        plan.delta_convolution_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights,
        layer_name + "attention.delta.time_bias",
        {attention.head_count},
        descriptor.activation_dtype,
        plan.delta_time_bias);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights,
        layer_name + "attention.delta.decay_log",
        {attention.head_count},
        descriptor.activation_dtype,
        plan.delta_decay_log);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights,
        layer_name + "attention.delta.norm.weight",
        {attention.value_head_dimension},
        descriptor.activation_dtype,
        plan.delta_norm_weight);
    if (!status)
        return status.error();
    return assign_required_tensor(
        weights,
        layer_name + "attention.output.weight",
        {descriptor.hidden_size, value_size},
        descriptor.activation_dtype,
        plan.output_weight);
}

static uint64_t tensor_storage_bytes(const TensorData& tensor)
{
    uint64_t bytes = tensor.mapped_byte_count;
    bytes += static_cast<uint64_t>(tensor.float32_data.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(tensor.bfloat16_data.size()) * sizeof(uint16_t);
    bytes += static_cast<uint64_t>(tensor.int64_data.size()) * sizeof(int64_t);
    bytes += tensor.int8_data.size();
    bytes += static_cast<uint64_t>(tensor.quantization_scales.size()) * sizeof(float);
    bytes += tensor.mxfp4_blocks.size();
    bytes += tensor.mxfp4_scales.size();
    if (tensor.mxfp4_file_storage)
    {
        bytes += tensor.mxfp4_file_storage->blocks_bytes;
        bytes += tensor.mxfp4_file_storage->scales_bytes;
        bytes += tensor.mxfp4_file_storage->secondary_blocks_bytes;
        bytes += tensor.mxfp4_file_storage->secondary_scales_bytes;
    }
    return bytes;
}

static bool uses_vulkan_dense_operator(const TensorData& tensor)
{
    return tensor.bfloat16_linear_operator
           || tensor.float8_linear_operator
           || (tensor.linear_operator && tensor.linear_operator->uses_vulkan());
}

static void release_tensor_host_storage(TensorData& tensor)
{
    if (tensor.dtype == DType::Float32)
    {
        std::vector<float>().swap(tensor.float32_data);
    }
    else if (tensor.dtype == DType::BFloat16)
    {
        std::vector<uint16_t>().swap(tensor.bfloat16_data);
    }
    else if (tensor.dtype == DType::Float8E4M3)
    {
        std::vector<float>().swap(tensor.quantization_scales);
    }
    else
    {
        return;
    }
    tensor.mapped_data.reset();
    tensor.mapped_byte_count = 0;
}

static void release_vulkan_dense_handle(CompiledModel& compiled, TensorHandle handle)
{
    if (handle == invalid_tensor_handle)
        return;
    release_tensor_host_storage(compiled.weights.at_mutable(handle));
}

static void release_vulkan_dense_host_copies(CompiledModel& compiled)
{
    for (TensorHandle handle = 0; handle < compiled.weights.size(); ++handle)
    {
        TensorData& tensor = compiled.weights.at_mutable(handle);
        if (uses_vulkan_dense_operator(tensor))
            release_tensor_host_storage(tensor);
    }

    const auto release_fused_layer_handles = [&compiled](CompiledLayerPlan& layer) {
        AttentionBlockPlan& attention = layer.attention;
        const bool qkv_fused = attention.fused_qkv_operator
                               || attention.fused_qkv_bfloat16_operator
                               || attention.fused_qkv_gate_bfloat16_operator;
        if (qkv_fused)
        {
            release_vulkan_dense_handle(compiled, attention.query_weight);
            release_vulkan_dense_handle(compiled, attention.key_weight);
            release_vulkan_dense_handle(compiled, attention.value_weight);
        }
        if (attention.fused_qkv_gate_bfloat16_operator)
            release_vulkan_dense_handle(compiled, attention.output_gate_weight);

        if (attention.fused_delta_input_operator || attention.fused_delta_input_bfloat16_operator)
        {
            release_vulkan_dense_handle(compiled, attention.delta_qkv_weight);
            release_vulkan_dense_handle(compiled, attention.delta_z_weight);
            release_vulkan_dense_handle(compiled, attention.delta_beta_weight);
            release_vulkan_dense_handle(compiled, attention.delta_alpha_weight);
        }

        if (attention.vulkan_attention_operator)
        {
            release_vulkan_dense_handle(compiled, attention.pre_attention_norm_weight);
            release_vulkan_dense_handle(compiled, attention.sinks);
        }

        if (layer.moe.fused_shared_input_bfloat16_operator)
        {
            release_vulkan_dense_handle(compiled, layer.moe.shared_expert.gate_weight);
            release_vulkan_dense_handle(compiled, layer.moe.shared_expert.up_weight);
            release_vulkan_dense_handle(compiled, layer.moe.shared_expert_gate_weight);
        }
    };

    for (CompiledLayerPlan& layer : compiled.layers)
        release_fused_layer_handles(layer);
    for (CompiledLayerPlan& layer : compiled.speculative.layers)
        release_fused_layer_handles(layer);
}

static uint64_t expert_weight_bytes(const WeightTable& weights, const ExpertPlan& expert)
{
    const uint32_t handles[] = {
        expert.gate_weight,
        expert.up_weight,
        expert.gate_up_weight,
        expert.down_weight,
        expert.gate_up_bias,
        expert.down_bias,
    };
    uint64_t bytes = 0;
    for (uint32_t handle : handles)
    {
        if (handle != invalid_tensor_handle)
            bytes += tensor_storage_bytes(weights.at(handle));
    }
    return bytes;
}

static ExpertKernel selected_expert_kernel(DType dtype)
{
    if (dtype != DType::MxFp4)
        return ExpertKernel::PortableCpu;
    switch (mxfp4_kernel_kind())
    {
    case MxFp4KernelKind::ArmNeon: return ExpertKernel::Mxfp4ArmNeon;
    case MxFp4KernelKind::ArmSve2: return ExpertKernel::Mxfp4ArmSve2;
    case MxFp4KernelKind::X86Avx2: return ExpertKernel::Mxfp4X86Avx2;
    case MxFp4KernelKind::X86Avx512: return ExpertKernel::Mxfp4X86Avx512;
    case MxFp4KernelKind::Scalar: return ExpertKernel::Mxfp4Scalar;
    }
    return ExpertKernel::Mxfp4Scalar;
}

static double placement_cost(const std::vector<uint32_t>& counts, const std::vector<uint32_t>& scores, uint32_t concurrency, size_t added_device)
{
    double latency = 0.0;
    double bottleneck = 0.0;
    for (size_t index = 0; index < counts.size(); ++index)
    {
        const double device_time = static_cast<double>(counts[index] + (index == added_device ? 1 : 0)) / static_cast<double>(scores[index]);
        latency += device_time;
        bottleneck = std::max(bottleneck, device_time);
    }
    return latency + static_cast<double>(concurrency - 1) * bottleneck;
}

static uint32_t choose_layer_device(bool use_vulkan, const std::vector<uint32_t>& devices, const std::vector<uint32_t>& counts, std::vector<int64_t>& smooth_scores, int64_t total_score)
{
    if (!use_vulkan)
        return automatic_vulkan_device_index;

    size_t selected = 0;
    for (size_t index = 0; index < counts.size(); ++index)
    {
        smooth_scores[index] += counts[index];
        if (smooth_scores[index] > smooth_scores[selected])
            selected = index;
    }
    smooth_scores[selected] -= total_score;
    return devices[selected];
}

static Result<void> compile_mtp_speculative_model(
    CompiledModel& compiled,
    NcnnLinearDevice dense_device,
    bool retain_cpu_dense_copies)
{
    if (compiled.descriptor.speculative_layer_count != 1
        || compiled.descriptor.speculative_block_size == 0
        || !compiled.descriptor.speculative_target_layer_ids.empty()
        || compiled.descriptor.hyper_connection_multiplier != 1
        || compiled.layers.empty()
        || has_flag(compiled.layers.back().attention.flags, AttentionBlockLatent)
        || has_flag(compiled.layers.back().attention.flags, AttentionBlockGatedDeltaNet)
        || compiled.layers.back().attention.sliding_window != 0)
    {
        return Error{ErrorCode::InvalidModel, "invalid Qwen MTP speculative model configuration"};
    }

    SpeculativeModelPlan& speculative = compiled.speculative;
    speculative.kind = SpeculativeModelKind::Mtp;
    speculative.block_size = compiled.descriptor.speculative_block_size;
    const uint32_t hidden_size = compiled.descriptor.hidden_size;
    Result<void> status = assign_required_tensor(
        compiled.weights,
        "speculative.mtp.embedding_norm.weight",
        {hidden_size},
        compiled.descriptor.activation_dtype,
        speculative.mtp_embedding_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.mtp.hidden_norm.weight",
        {hidden_size},
        compiled.descriptor.activation_dtype,
        speculative.mtp_hidden_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.mtp.input_projection.weight",
        {hidden_size, hidden_size * 2},
        compiled.descriptor.activation_dtype,
        speculative.mtp_input_projection_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.final_norm.weight",
        {hidden_size},
        compiled.descriptor.activation_dtype,
        speculative.final_norm_weight);
    if (!status)
        return status.error();
    status = prepare_linear_operator(
        compiled.weights,
        speculative.mtp_input_projection_weight,
        invalid_tensor_handle,
        dense_device,
        retain_cpu_dense_copies,
        compiled.vulkan_device_index);
    if (!status)
        return status.error();

    const std::string layer_name = speculative_layer_prefix(0);
    CompiledLayerPlan layer_plan = compiled.layers.back();
    layer_plan.layer_id = compiled.descriptor.layer_count;
    layer_plan.vulkan_device_index = compiled.vulkan_device_index;
    layer_plan.nodes.clear();
    layer_plan.hyper_connection = {};
    AttentionBlockPlan& attention = layer_plan.attention;
    attention.vulkan_attention_operator.reset();
    attention.fused_qkv_operator.reset();
    attention.fused_qkv_bfloat16_operator.reset();
    attention.fused_qkv_gate_bfloat16_operator.reset();
    attention.query_bias = invalid_tensor_handle;
    attention.key_bias = invalid_tensor_handle;
    attention.value_bias = invalid_tensor_handle;
    attention.output_bias = invalid_tensor_handle;

    status = assign_required_tensor(
        compiled.weights,
        layer_name + "pre_attention_norm.weight",
        {hidden_size},
        compiled.descriptor.activation_dtype,
        attention.pre_attention_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        layer_name + "attention.query.weight",
        {attention.head_count * attention.head_dimension, hidden_size},
        compiled.descriptor.activation_dtype,
        attention.query_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        layer_name + "attention.key.weight",
        {attention.kv_head_count * attention.head_dimension, hidden_size},
        compiled.descriptor.activation_dtype,
        attention.key_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        layer_name + "attention.value.weight",
        {attention.kv_head_count * attention.head_dimension, hidden_size},
        compiled.descriptor.activation_dtype,
        attention.value_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        layer_name + "attention.output.weight",
        {hidden_size, attention.head_count * attention.head_dimension},
        compiled.descriptor.activation_dtype,
        attention.output_weight);
    if (!status)
        return status.error();
    if (has_flag(attention.flags, AttentionBlockQueryKeyNorm))
    {
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "attention.query_norm.weight",
            {attention.head_dimension},
            compiled.descriptor.activation_dtype,
            attention.query_norm_weight);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "attention.key_norm.weight",
            {attention.head_dimension},
            compiled.descriptor.activation_dtype,
            attention.key_norm_weight);
        if (!status)
            return status.error();
    }
    if (has_flag(attention.flags, AttentionBlockOutputGate))
    {
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "attention.output_gate.weight",
            {attention.head_count * attention.head_dimension, hidden_size},
            compiled.descriptor.activation_dtype,
            attention.output_gate_weight);
        if (!status)
            return status.error();
    }
    if (dense_device == NcnnLinearDevice::Vulkan)
    {
        const std::vector<const TensorData*> qkv_matrices = {
            &compiled.weights.at(attention.query_weight),
            &compiled.weights.at(attention.key_weight),
            &compiled.weights.at(attention.value_weight),
        };
        const std::vector<const TensorData*> qkv_biases = {
            nullptr,
            nullptr,
            nullptr,
        };
        if (qkv_matrices.front()->dtype == DType::BFloat16
            && attention.output_gate_weight
                   != invalid_tensor_handle)
        {
            std::vector<const TensorData*> qkv_gate_matrices =
                qkv_matrices;
            qkv_gate_matrices.push_back(
                &compiled.weights.at(
                    attention.output_gate_weight));
            std::vector<const TensorData*> qkv_gate_biases =
                qkv_biases;
            qkv_gate_biases.push_back(nullptr);
            attention.fused_qkv_gate_bfloat16_operator =
                NcnnVulkanBfloat16Operator::create_fused(
                    qkv_gate_matrices,
                    qkv_gate_biases,
                    layer_plan.vulkan_device_index);
        }
        if (!attention.fused_qkv_gate_bfloat16_operator
            && qkv_matrices.front()->dtype == DType::BFloat16)
        {
            attention.fused_qkv_bfloat16_operator =
                NcnnVulkanBfloat16Operator::create_fused(
                    qkv_matrices,
                    qkv_biases,
                    layer_plan.vulkan_device_index);
        }
        if (!attention.fused_qkv_gate_bfloat16_operator
            && !attention.fused_qkv_bfloat16_operator)
        {
            attention.fused_qkv_operator =
                NcnnLinearOperator::create_fused(
                    qkv_matrices,
                    qkv_biases,
                    dense_device,
                    layer_plan.vulkan_device_index);
        }
        if (!attention.fused_qkv_gate_bfloat16_operator
            && !attention.fused_qkv_bfloat16_operator
            && !attention.fused_qkv_operator)
            return Error{ErrorCode::InternalError, "failed to create fused Qwen MTP Vulkan QKV operator"};
    }
    else
    {
        const TensorHandle projection_handles[] = {
            attention.query_weight,
            attention.key_weight,
            attention.value_weight,
        };
        for (TensorHandle handle : projection_handles)
        {
            status = prepare_linear_operator(
                compiled.weights,
                handle,
                invalid_tensor_handle,
                dense_device,
                retain_cpu_dense_copies,
                layer_plan.vulkan_device_index);
            if (!status)
                return status.error();
        }
    }
    const TensorHandle output_handles[] = {
        attention.output_weight,
        attention.output_gate_weight,
    };
    for (TensorHandle handle : output_handles)
    {
        if (handle == invalid_tensor_handle)
            continue;
        if (handle == attention.output_gate_weight
            && attention.fused_qkv_gate_bfloat16_operator)
        {
            continue;
        }
        status = prepare_linear_operator(
            compiled.weights,
            handle,
            invalid_tensor_handle,
            dense_device,
            retain_cpu_dense_copies,
            layer_plan.vulkan_device_index);
        if (!status)
            return status.error();
    }

    const MoeDescriptor& moe = compiled.descriptor.layers.back().ffn.moe;
    MoeBlockPlan& compiled_moe = layer_plan.moe;
    compiled_moe.router_bias = invalid_tensor_handle;
    compiled_moe.router_selection_bias = invalid_tensor_handle;
    compiled_moe.token_experts = invalid_tensor_handle;
    compiled_moe.experts.clear();
    compiled_moe.shared_expert = {};
    status = assign_required_tensor(
        compiled.weights,
        layer_name + "pre_ffn_norm.weight",
        {hidden_size},
        compiled.descriptor.activation_dtype,
        compiled_moe.pre_ffn_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        layer_name + "router.weight",
        {moe.expert_count, hidden_size},
        compiled.descriptor.activation_dtype,
        compiled_moe.router_weight);
    if (!status)
        return status.error();
    status = prepare_linear_operator(
        compiled.weights,
        compiled_moe.router_weight,
        invalid_tensor_handle,
        NcnnLinearDevice::Cpu,
        retain_cpu_dense_copies,
        layer_plan.vulkan_device_index);
    if (!status)
        return status.error();

    ExpertPlan& shared = compiled_moe.shared_expert;
    shared.activation = moe.activation;
    shared.activation_limit = moe.activation_limit;
    status = assign_required_tensor(
        compiled.weights,
        layer_name + "shared_expert.gate.weight",
        {moe.intermediate_size, hidden_size},
        moe.shared_expert_weight_dtype,
        shared.gate_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        layer_name + "shared_expert.up.weight",
        {moe.intermediate_size, hidden_size},
        moe.shared_expert_weight_dtype,
        shared.up_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        layer_name + "shared_expert.down.weight",
        {hidden_size, moe.intermediate_size},
        moe.shared_expert_weight_dtype,
        shared.down_weight);
    if (!status)
        return status.error();
    shared.weight_bytes = expert_weight_bytes(compiled.weights, shared);
    if (has_flag(moe.flags, MoeDescriptorSharedExpertGate))
    {
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "shared_expert.router_gate.weight",
            {1, hidden_size},
            compiled.descriptor.activation_dtype,
            compiled_moe.shared_expert_gate_weight);
        if (!status)
            return status.error();
    }
    status = prepare_shared_expert_operators(
        compiled.weights,
        compiled_moe,
        dense_device,
        retain_cpu_dense_copies,
        layer_plan.vulkan_device_index);
    if (!status)
        return status.error();

    compiled_moe.experts.reserve(moe.expert_count);
    for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
    {
        ExpertPlan expert;
        expert.activation = moe.activation;
        expert.activation_limit = moe.activation_limit;
        if (moe.layout == ExpertLayout::PackedGateUpDown)
            expert.flags |= ExpertPlanPackedGateUp;
        const std::string prefix = speculative_expert_prefix(0, expert_id);
        status = assign_required_tensor(
            compiled.weights,
            prefix + "gate_up.weight",
            {moe.intermediate_size * 2, hidden_size},
            moe.expert_weight_dtype,
            expert.gate_up_weight);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            prefix + "down.weight",
            {hidden_size, moe.intermediate_size},
            moe.expert_weight_dtype,
            expert.down_weight);
        if (!status)
            return status.error();
        expert.weight_bytes = expert_weight_bytes(compiled.weights, expert);
        const TensorData& gate_up_weight = compiled.weights.at(expert.gate_up_weight);
        const TensorData& down_weight = compiled.weights.at(expert.down_weight);
        if (gate_up_weight.mxfp4_file_storage && down_weight.mxfp4_file_storage)
            expert.cache_key = Mxfp4ExpertCache::make_pair_key(gate_up_weight, down_weight);
        compiled_moe.experts.push_back(std::move(expert));
    }
    speculative.layers.push_back(std::move(layer_plan));
    return {};
}

static Result<void> compile_speculative_model(
    CompiledModel& compiled,
    NcnnLinearDevice dense_device,
    bool retain_cpu_dense_copies)
{
    if (compiled.descriptor.speculative_layer_count == 0)
        return {};
    if (compiled.descriptor.speculative_kind == SpeculativeModelKind::Mtp)
        return compile_mtp_speculative_model(compiled, dense_device, retain_cpu_dense_copies);
    if (compiled.descriptor.speculative_kind != SpeculativeModelKind::DSpark)
        return Error{ErrorCode::InvalidModel, "speculative layer kind is not configured"};
    if (compiled.descriptor.speculative_target_layer_ids.size() != compiled.descriptor.speculative_layer_count
        || compiled.descriptor.speculative_block_size == 0
        || compiled.descriptor.speculative_noise_token_id >= compiled.descriptor.vocabulary_size
        || compiled.descriptor.speculative_markov_rank == 0)
    {
        return Error{ErrorCode::InvalidModel, "invalid speculative model configuration"};
    }

    SpeculativeModelPlan& speculative = compiled.speculative;
    speculative.kind = SpeculativeModelKind::DSpark;
    speculative.target_layer_ids = compiled.descriptor.speculative_target_layer_ids;
    speculative.block_size = compiled.descriptor.speculative_block_size;
    speculative.noise_token_id = compiled.descriptor.speculative_noise_token_id;
    speculative.markov_rank = compiled.descriptor.speculative_markov_rank;
    const uint32_t hidden_size = compiled.descriptor.hidden_size;
    const uint32_t multiplier = compiled.descriptor.hyper_connection_multiplier;
    const uint32_t hyper_columns = multiplier * hidden_size;
    Result<void> status = assign_required_tensor(
        compiled.weights,
        "speculative.main_projection.weight",
        {hidden_size, hidden_size * static_cast<uint32_t>(speculative.target_layer_ids.size())},
        DType::Float8E4M3,
        speculative.main_projection_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.main_norm.weight",
        {hidden_size},
        compiled.descriptor.activation_dtype,
        speculative.main_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.final_norm.weight",
        {hidden_size},
        compiled.descriptor.activation_dtype,
        speculative.final_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.hyper.head.function",
        {multiplier, hyper_columns},
        DType::Float32,
        speculative.hyper_head_function);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.hyper.head.base",
        {multiplier},
        DType::Float32,
        speculative.hyper_head_base);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.hyper.head.scale",
        {1},
        DType::Float32,
        speculative.hyper_head_scale);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.markov.embedding.weight",
        {compiled.descriptor.vocabulary_size, speculative.markov_rank},
        compiled.descriptor.activation_dtype,
        speculative.markov_embedding_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.markov.head.weight",
        {compiled.descriptor.vocabulary_size, speculative.markov_rank},
        compiled.descriptor.activation_dtype,
        speculative.markov_head_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights,
        "speculative.confidence.weight",
        {1, hidden_size + speculative.markov_rank},
        compiled.descriptor.activation_dtype,
        speculative.confidence_weight);
    if (!status)
        return status.error();

    const TensorHandle dense_handles[] = {
        speculative.main_projection_weight,
        speculative.markov_head_weight,
    };
    for (TensorHandle handle : dense_handles)
    {
        status = prepare_linear_operator(
            compiled.weights,
            handle,
            invalid_tensor_handle,
            dense_device,
            retain_cpu_dense_copies,
            compiled.vulkan_device_index);
        if (!status)
            return status.error();
    }

    LayerDescriptor draft_layer = compiled.descriptor.layers.back();
    draft_layer.attention.compression_ratio = 0;
    draft_layer.ffn.moe.flags |= MoeDescriptorRouterBias;
    speculative.layers.reserve(compiled.descriptor.speculative_layer_count);
    for (uint32_t layer_id = 0; layer_id < compiled.descriptor.speculative_layer_count; ++layer_id)
    {
        const std::string layer_name = speculative_layer_prefix(layer_id);
        const MoeDescriptor& moe = draft_layer.ffn.moe;
        CompiledLayerPlan layer_plan;
        layer_plan.layer_id = compiled.descriptor.layer_count + layer_id;
        layer_plan.vulkan_device_index = compiled.vulkan_device_index;
        layer_plan.flags |= CompiledLayerAttention;
        layer_plan.moe.top_k = moe.top_k;
        layer_plan.moe.hidden_size = hidden_size;
        layer_plan.moe.score_function = moe.score_function;
        layer_plan.moe.normalization = moe.normalization;
        layer_plan.moe.routed_scaling_factor = moe.routed_scaling_factor;
        layer_plan.moe.has_shared_expert = true;
        layer_plan.moe.flags = MoeBlockNormalizeTopKWeights;

        const uint32_t mix_count = (2 + multiplier) * multiplier;
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "hyper.attention.function",
            {mix_count, hyper_columns},
            DType::Float32,
            layer_plan.hyper_connection.attention_function);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "hyper.attention.base",
            {mix_count},
            DType::Float32,
            layer_plan.hyper_connection.attention_base);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "hyper.attention.scale",
            {3},
            DType::Float32,
            layer_plan.hyper_connection.attention_scale);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "hyper.ffn.function",
            {mix_count, hyper_columns},
            DType::Float32,
            layer_plan.hyper_connection.ffn_function);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "hyper.ffn.base",
            {mix_count},
            DType::Float32,
            layer_plan.hyper_connection.ffn_base);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "hyper.ffn.scale",
            {3},
            DType::Float32,
            layer_plan.hyper_connection.ffn_scale);
        if (!status)
            return status.error();

        status = compile_latent_attention(
            compiled.weights,
            layer_name,
            compiled.descriptor,
            draft_layer.attention,
            layer_plan.attention);
        if (!status)
            return status.error();
        const TensorHandle latent_linear_handles[] = {
            layer_plan.attention.query_a_weight,
            layer_plan.attention.query_b_weight,
            layer_plan.attention.key_value_weight,
            layer_plan.attention.output_b_weight,
        };
        for (TensorHandle handle : latent_linear_handles)
        {
            status = prepare_linear_operator(
                compiled.weights,
                handle,
                invalid_tensor_handle,
                dense_device,
                retain_cpu_dense_copies,
                layer_plan.vulkan_device_index);
            if (!status)
                return status.error();
        }
        status = prepare_linear_operator(
            compiled.weights,
            layer_plan.attention.output_a_weight,
            invalid_tensor_handle,
            dense_device,
            retain_cpu_dense_copies,
            layer_plan.vulkan_device_index,
            layer_plan.attention.output_group_count);
        if (!status)
            return status.error();
        if (dense_device == NcnnLinearDevice::Vulkan)
        {
            TensorData& query_a = compiled.weights.at_mutable(layer_plan.attention.query_a_weight);
            const TensorData& query_b = compiled.weights.at(layer_plan.attention.query_b_weight);
            if (query_a.float8_linear_operator && query_b.float8_linear_operator
                && !query_a.float8_linear_operator->prepare_rms_norm(
                    compiled.weights.at(layer_plan.attention.query_norm_weight),
                    compiled.descriptor.norm_epsilon))
            {
                return Error{ErrorCode::InternalError, "failed to prepare speculative Vulkan FP8 query RMSNorm chain"};
            }
        }

        status = assign_required_tensor(
            compiled.weights,
            layer_name + "pre_ffn_norm.weight",
            {hidden_size},
            compiled.descriptor.activation_dtype,
            layer_plan.moe.pre_ffn_norm_weight);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "router.weight",
            {moe.expert_count, hidden_size},
            compiled.descriptor.activation_dtype,
            layer_plan.moe.router_weight);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "router.selection_bias",
            {moe.expert_count},
            DType::Float32,
            layer_plan.moe.router_selection_bias);
        if (!status)
            return status.error();
        status = prepare_linear_operator(
            compiled.weights,
            layer_plan.moe.router_weight,
            invalid_tensor_handle,
            NcnnLinearDevice::Cpu,
            retain_cpu_dense_copies,
            layer_plan.vulkan_device_index);
        if (!status)
            return status.error();

        ExpertPlan& shared = layer_plan.moe.shared_expert;
        shared.activation = moe.activation;
        shared.activation_limit = moe.activation_limit;
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "shared_expert.gate.weight",
            {moe.intermediate_size, hidden_size},
            moe.shared_expert_weight_dtype,
            shared.gate_weight);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "shared_expert.up.weight",
            {moe.intermediate_size, hidden_size},
            moe.shared_expert_weight_dtype,
            shared.up_weight);
        if (!status)
            return status.error();
        status = assign_required_tensor(
            compiled.weights,
            layer_name + "shared_expert.down.weight",
            {hidden_size, moe.intermediate_size},
            moe.shared_expert_weight_dtype,
            shared.down_weight);
        if (!status)
            return status.error();
        shared.weight_bytes = expert_weight_bytes(compiled.weights, shared);
        const TensorHandle shared_handles[] = {
            shared.gate_weight,
            shared.up_weight,
            shared.down_weight,
        };
        for (TensorHandle handle : shared_handles)
        {
            status = prepare_linear_operator(
                compiled.weights,
                handle,
                invalid_tensor_handle,
                dense_device,
                retain_cpu_dense_copies,
                layer_plan.vulkan_device_index);
            if (!status)
                return status.error();
        }

        layer_plan.moe.experts.reserve(moe.expert_count);
        for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
        {
            ExpertPlan expert;
            expert.activation = moe.activation;
            expert.activation_limit = moe.activation_limit;
            const std::string prefix = speculative_expert_prefix(layer_id, expert_id);
            status = assign_required_tensor(
                compiled.weights,
                prefix + "gate_up.weight",
                {moe.intermediate_size * 2, hidden_size},
                moe.expert_weight_dtype,
                expert.gate_up_weight);
            if (!status)
                return status.error();
            status = assign_required_tensor(
                compiled.weights,
                prefix + "down.weight",
                {hidden_size, moe.intermediate_size},
                moe.expert_weight_dtype,
                expert.down_weight);
            if (!status)
                return status.error();
            expert.weight_bytes = expert_weight_bytes(compiled.weights, expert);
            const TensorData& gate_up_weight = compiled.weights.at(expert.gate_up_weight);
            const TensorData& down_weight = compiled.weights.at(expert.down_weight);
            if (gate_up_weight.mxfp4_file_storage && down_weight.mxfp4_file_storage)
                expert.cache_key = Mxfp4ExpertCache::make_pair_key(gate_up_weight, down_weight);
            layer_plan.moe.experts.push_back(std::move(expert));
        }
        speculative.layers.push_back(std::move(layer_plan));
    }
    return {};
}

Result<CompiledModel> ModelCompiler::compile(MoeIR descriptor, WeightMapping mapping, HybridMode hybrid_mode) const
{
    BackendCapabilities capabilities;
    if (hybrid_mode == HybridMode::HybridExperts || hybrid_mode == HybridMode::VulkanWithCpuPrefetch)
    {
        capabilities.flags |= BackendCapabilityVulkanDense | BackendCapabilityVulkanAttention;
    }
    return compile(std::move(descriptor), std::move(mapping), hybrid_mode, capabilities);
}

Result<CompiledModel> ModelCompiler::compile(MoeIR descriptor, WeightMapping mapping, HybridMode hybrid_mode, const BackendCapabilities& capabilities) const
{
    if (!has_flag(capabilities.flags, BackendCapabilityCpuExecution))
        return Error{ErrorCode::UnsupportedModel, "compiler requires a CPU execution backend"};
    auto normalized_ir = normalize_moe_ir(descriptor);
    if (!normalized_ir)
        return normalized_ir.error();
    if (descriptor.vocabulary_size == 0
        || descriptor.hidden_size == 0
        || descriptor.layer_count == 0
        || descriptor.intermediate_size == 0
        || descriptor.expert_count == 0
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
    for (const LayerDescriptor& layer : descriptor.layers)
    {
        if (has_flag(layer.flags, LayerDescriptorDenseFfn))
        {
            return Error{ErrorCode::UnsupportedModel, "dense FFN decoder layers are represented by MoeIR but are not yet executable"};
        }
        if (has_flag(layer.flags, LayerDescriptorMoe))
        {
            const MoeDescriptor& moe = layer.ffn.moe;
            if (moe.router_group_count != 0 || moe.router_top_k_groups != 0)
            {
                return Error{ErrorCode::UnsupportedModel, "group-limited routing is represented by MoeIR but is not yet executable"};
            }
            if (!std::isfinite(moe.routed_scaling_factor) || moe.routed_scaling_factor <= 0.0f)
                return Error{ErrorCode::InvalidModel, "routed scaling factor must be finite and positive"};
        }
    }

    CompiledModel compiled;
    compiled.descriptor = std::move(descriptor);
    compiled.expert_store = std::make_shared<ExpertStore>();
    const bool hybrid_requests_vulkan = hybrid_mode == HybridMode::HybridExperts || hybrid_mode == HybridMode::VulkanWithCpuPrefetch;
    const bool use_vulkan_dense = hybrid_requests_vulkan && has_flag(capabilities.flags, BackendCapabilityVulkanDense);
    const bool retain_cpu_dense_copies = has_flag(capabilities.flags, BackendCapabilityRetainCpuDenseCopies);
    std::vector<uint32_t> dense_device_indices = capabilities.vulkan_device_indices;
    if (dense_device_indices.empty() && capabilities.vulkan_device_index != automatic_vulkan_device_index)
    {
        dense_device_indices.push_back(capabilities.vulkan_device_index);
    }
    if (use_vulkan_dense && dense_device_indices.empty())
    {
        return Error{ErrorCode::InvalidArgument, "Vulkan dense execution requires at least one device"};
    }
    std::vector<uint32_t> dense_device_scores = capabilities.vulkan_device_scores;
    if (dense_device_scores.size() != dense_device_indices.size())
    {
        dense_device_scores.assign(dense_device_indices.size(), 1);
    }
    for (uint32_t& score : dense_device_scores)
        score = std::max(1u, score);
    std::vector<uint32_t> device_layer_counts(dense_device_indices.size(), 0);
    if (use_vulkan_dense)
    {
        if (capabilities.expected_concurrency <= 1 || dense_device_indices.size() == 1)
        {
            const size_t fastest = static_cast<size_t>(std::distance(dense_device_scores.begin(), std::max_element(dense_device_scores.begin(), dense_device_scores.end())));
            device_layer_counts[fastest] = compiled.descriptor.layer_count;
        }
        else
        {
            for (uint32_t layer = 0; layer < compiled.descriptor.layer_count; ++layer)
            {
                size_t selected = 0;
                double selected_cost = placement_cost(device_layer_counts, dense_device_scores, capabilities.expected_concurrency, 0);
                for (size_t index = 1; index < dense_device_indices.size(); ++index)
                {
                    const double cost = placement_cost(device_layer_counts, dense_device_scores, capabilities.expected_concurrency, index);
                    if (cost < selected_cost)
                    {
                        selected = index;
                        selected_cost = cost;
                    }
                }
                ++device_layer_counts[selected];
            }
        }
    }
    std::vector<int64_t> smooth_device_scores(dense_device_indices.size(), 0);
    const int64_t total_device_score = std::accumulate(device_layer_counts.begin(), device_layer_counts.end(), INT64_C(0));
    compiled.hybrid_mode = use_vulkan_dense ? hybrid_mode : HybridMode::CpuOnly;
    compiled.vulkan_device_index = use_vulkan_dense ? dense_device_indices.front() : automatic_vulkan_device_index;
    if (use_vulkan_dense)
    {
        for (size_t index = 0; index < dense_device_indices.size(); ++index)
        {
            if (device_layer_counts[index] != 0 || dense_device_indices[index] == compiled.vulkan_device_index)
            {
                compiled.vulkan_device_indices.push_back(dense_device_indices[index]);
            }
        }
    }
    const NcnnLinearDevice dense_device = use_vulkan_dense ? NcnnLinearDevice::Vulkan : NcnnLinearDevice::Cpu;

    for (auto& [name, tensor] : mapping.tensors)
    {
        auto added = compiled.weights.add(name, std::move(tensor));
        if (!added)
            return added.error();
    }

    auto embedding = require_tensor(compiled.weights, "token_embedding.weight", {compiled.descriptor.vocabulary_size, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
    if (!embedding)
        return embedding.error();
    compiled.token_embedding = embedding.value();

    auto final_norm = require_tensor(compiled.weights, "final_norm.weight", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
    if (!final_norm)
        return final_norm.error();
    compiled.final_norm_weight = final_norm.value();

    auto lm_head = require_tensor(compiled.weights, "lm_head.weight", {compiled.descriptor.vocabulary_size, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
    if (!lm_head)
        return lm_head.error();
    compiled.lm_head_weight = lm_head.value();
    auto prepared = prepare_linear_operator(compiled.weights, compiled.lm_head_weight, invalid_tensor_handle, dense_device, retain_cpu_dense_copies, compiled.vulkan_device_index);
    if (!prepared)
        return prepared.error();
    if (compiled.descriptor.hyper_connection_multiplier > 1)
    {
        const uint32_t hyper_columns = compiled.descriptor.hyper_connection_multiplier * compiled.descriptor.hidden_size;
        auto status = assign_required_tensor(compiled.weights, "hyper.head.function", {compiled.descriptor.hyper_connection_multiplier, hyper_columns}, DType::Float32, compiled.hyper_head_function);
        if (!status)
            return status.error();
        status = assign_required_tensor(compiled.weights, "hyper.head.base", {compiled.descriptor.hyper_connection_multiplier}, DType::Float32, compiled.hyper_head_base);
        if (!status)
            return status.error();
        status = assign_required_tensor(compiled.weights, "hyper.head.scale", {1}, DType::Float32, compiled.hyper_head_scale);
        if (!status)
            return status.error();
        if (compiled.descriptor.hyper_connection_iterations == 0 || compiled.descriptor.hyper_connection_epsilon <= 0.0f)
            return Error{ErrorCode::InvalidModel, "invalid hyper-connection configuration"};
    }

    compiled.layers.reserve(compiled.descriptor.layer_count);
    for (uint32_t layer_id = 0; layer_id < compiled.descriptor.layer_count; ++layer_id)
    {
        const LayerDescriptor& layer = compiled.descriptor.layers[layer_id];
        auto parsed_graph = validate_adapter_graph(layer);
        if (!parsed_graph)
            return parsed_graph.error();
        const AdapterGraph graph = parsed_graph.value();
        const MoeDescriptor& moe = layer.ffn.moe;
        if (!has_flag(graph.flags, AdapterGraphMoe))
            return Error{ErrorCode::UnsupportedModel, "the current executor requires an ExpertGroup in every layer graph"};
        if (layer.pre_ffn_norm != NormType::RmsNorm)
            return Error{ErrorCode::UnsupportedModel, "the reference runtime requires RMSNorm before each MoE block"};
        if (moe.expert_count == 0 || moe.top_k == 0 || moe.top_k > moe.expert_count)
            return Error{ErrorCode::InvalidModel, "invalid expert_count/top_k"};
        if (moe.intermediate_size == 0)
            return Error{ErrorCode::InvalidModel, "intermediate_size must be non-zero"};
        if (moe.expert_count != compiled.descriptor.expert_count
            || moe.top_k != compiled.descriptor.experts_per_token
            || moe.intermediate_size != compiled.descriptor.intermediate_size)
        {
            return Error{ErrorCode::InvalidModel, "layer MoE dimensions do not match the model descriptor"};
        }
        if (has_flag(moe.flags, MoeDescriptorSharedExpert) != has_flag(graph.flags, AdapterGraphSharedExpert))
            return Error{ErrorCode::InvalidModel, "shared Expert descriptor and operator graph do not match"};
        if (has_flag(moe.flags, MoeDescriptorSharedExpertGate)
            && !has_flag(moe.flags, MoeDescriptorSharedExpert))
        {
            return Error{ErrorCode::InvalidModel, "shared Expert gate requires a shared Expert"};
        }
        if (moe.expert_weight_dtype != DType::Float32
            && moe.expert_weight_dtype != DType::BFloat16
            && moe.expert_weight_dtype != DType::Int8
            && moe.expert_weight_dtype != DType::MxFp4)
            return Error{ErrorCode::UnsupportedModel, "expert weights must use float32, bfloat16, int8, or MXFP4"};
        if (moe.expert_weight_dtype == DType::MxFp4 && !has_flag(capabilities.flags, BackendCapabilityMxfp4CpuKernel))
        {
            return Error{ErrorCode::UnsupportedModel, "backend capabilities do not provide an MXFP4 CPU expert kernel"};
        }

        CompiledLayerPlan layer_plan;
        layer_plan.layer_id = layer_id;
        layer_plan.vulkan_device_index = choose_layer_device(use_vulkan_dense, dense_device_indices, device_layer_counts, smooth_device_scores, total_device_score);
        if (has_flag(graph.flags, AdapterGraphAttention))
            layer_plan.flags |= CompiledLayerAttention;
        const bool use_vulkan_attention = has_flag(graph.flags, AdapterGraphAttention)
                                          && layer.attention.kind == AttentionKind::Standard
                                          && use_vulkan_dense
                                          && has_flag(capabilities.flags, BackendCapabilityVulkanAttention);
        const bool use_vulkan_latent_linear = has_flag(graph.flags, AdapterGraphAttention)
                                              && layer.attention.kind == AttentionKind::MultiHeadLatent
                                              && use_vulkan_dense;
        const bool use_vulkan_delta_linear = has_flag(graph.flags, AdapterGraphAttention)
                                             && layer.attention.kind == AttentionKind::GatedDeltaNet
                                             && use_vulkan_dense;
        layer_plan.nodes.reserve(layer.nodes.size());
        for (size_t node_index = 0; node_index < layer.nodes.size(); ++node_index)
        {
            const ExecutionBackend backend = use_vulkan_attention && node_index < graph.attention_node_count ? ExecutionBackend::Vulkan : ExecutionBackend::Cpu;
            layer_plan.nodes.push_back({layer.nodes[node_index].type, backend});
        }
        layer_plan.moe.top_k = moe.top_k;
        layer_plan.moe.hidden_size = compiled.descriptor.hidden_size;
        layer_plan.moe.score_function = moe.score_function;
        layer_plan.moe.normalization = moe.normalization;
        layer_plan.moe.routed_scaling_factor = moe.routed_scaling_factor;
        layer_plan.moe.has_shared_expert = has_flag(moe.flags, MoeDescriptorSharedExpert);
        layer_plan.moe.flags = 0;
        if (has_flag(moe.flags, MoeDescriptorNormalizeTopKWeights))
        {
            layer_plan.moe.flags |= MoeBlockNormalizeTopKWeights;
        }

        const std::string layer_name = layer_prefix(layer_id);
        if (compiled.descriptor.hyper_connection_multiplier > 1)
        {
            const uint32_t multiplier = compiled.descriptor.hyper_connection_multiplier;
            const uint32_t mix_count = (2 + multiplier) * multiplier;
            const uint32_t hyper_columns = multiplier * compiled.descriptor.hidden_size;
            auto status = assign_required_tensor(compiled.weights, layer_name + "hyper.attention.function", {mix_count, hyper_columns}, DType::Float32, layer_plan.hyper_connection.attention_function);
            if (!status)
                return status.error();
            status = assign_required_tensor(compiled.weights, layer_name + "hyper.attention.base", {mix_count}, DType::Float32, layer_plan.hyper_connection.attention_base);
            if (!status)
                return status.error();
            status = assign_required_tensor(compiled.weights, layer_name + "hyper.attention.scale", {3}, DType::Float32, layer_plan.hyper_connection.attention_scale);
            if (!status)
                return status.error();
            status = assign_required_tensor(compiled.weights, layer_name + "hyper.ffn.function", {mix_count, hyper_columns}, DType::Float32, layer_plan.hyper_connection.ffn_function);
            if (!status)
                return status.error();
            status = assign_required_tensor(compiled.weights, layer_name + "hyper.ffn.base", {mix_count}, DType::Float32, layer_plan.hyper_connection.ffn_base);
            if (!status)
                return status.error();
            status = assign_required_tensor(compiled.weights, layer_name + "hyper.ffn.scale", {3}, DType::Float32, layer_plan.hyper_connection.ffn_scale);
            if (!status)
                return status.error();
        }
        if (has_flag(graph.flags, AdapterGraphAttention))
        {
            const AttentionDescriptor& attention = layer.attention;
            const NcnnLinearDevice attention_device = use_vulkan_attention || use_vulkan_latent_linear || use_vulkan_delta_linear
                                                          ? NcnnLinearDevice::Vulkan
                                                          : NcnnLinearDevice::Cpu;
            if (layer.pre_attention_norm != NormType::RmsNorm)
                return Error{ErrorCode::UnsupportedModel, "attention requires a pre-attention RMSNorm"};
            if (attention.kind != AttentionKind::GatedDeltaNet)
            {
                if (attention.head_count == 0
                    || attention.kv_head_count == 0
                    || attention.head_dimension == 0
                    || attention.head_count % attention.kv_head_count != 0
                    || attention.head_dimension % 2 != 0)
                    return Error{ErrorCode::InvalidModel, "invalid attention dimensions"};
                if (attention.head_count != compiled.descriptor.attention_head_count
                    || attention.kv_head_count != compiled.descriptor.kv_head_count
                    || attention.head_dimension != compiled.descriptor.head_dimension)
                    return Error{ErrorCode::InvalidModel, "layer attention dimensions do not match the model descriptor"};
            }
            AttentionBlockPlan& plan = layer_plan.attention;
            if (attention.kind == AttentionKind::MultiHeadLatent)
            {
                auto status = compile_latent_attention(compiled.weights, layer_name, compiled.descriptor, attention, plan);
                if (!status)
                    return status.error();
                const TensorHandle latent_linear_handles[] = {
                    plan.query_a_weight,
                    plan.query_b_weight,
                    plan.key_value_weight,
                    plan.output_b_weight,
                    plan.indexer_query_weight,
                };
                for (TensorHandle handle : latent_linear_handles)
                {
                    if (handle == invalid_tensor_handle)
                        continue;
                    prepared = prepare_linear_operator(
                        compiled.weights,
                        handle,
                        invalid_tensor_handle,
                        attention_device,
                        retain_cpu_dense_copies,
                        layer_plan.vulkan_device_index);
                    if (!prepared)
                        return prepared.error();
                }
                prepared = prepare_linear_operator(
                    compiled.weights,
                    plan.output_a_weight,
                    invalid_tensor_handle,
                    attention_device,
                    retain_cpu_dense_copies,
                    layer_plan.vulkan_device_index,
                    plan.output_group_count);
                if (!prepared)
                    return prepared.error();
                if (attention_device == NcnnLinearDevice::Vulkan)
                {
                    TensorData& query_a = compiled.weights.at_mutable(plan.query_a_weight);
                    const TensorData& query_b = compiled.weights.at(plan.query_b_weight);
                    if (query_a.float8_linear_operator && query_b.float8_linear_operator
                        && !query_a.float8_linear_operator->prepare_rms_norm(
                            compiled.weights.at(plan.query_norm_weight),
                            compiled.descriptor.norm_epsilon))
                    {
                        return Error{ErrorCode::InternalError, "failed to prepare Vulkan FP8 query RMSNorm chain"};
                    }
                }
            }
            else if (attention.kind == AttentionKind::GatedDeltaNet)
            {
                auto status = compile_gated_delta_attention(
                    compiled.weights,
                    layer_name,
                    compiled.descriptor,
                    attention,
                    plan);
                if (!status)
                    return status.error();
                const std::vector<const TensorData*> delta_input_matrices = {
                    &compiled.weights.at(plan.delta_qkv_weight),
                    &compiled.weights.at(plan.delta_z_weight),
                    &compiled.weights.at(plan.delta_beta_weight),
                    &compiled.weights.at(plan.delta_alpha_weight),
                };
                const std::vector<const TensorData*> delta_input_biases = {
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                };
                if (attention_device == NcnnLinearDevice::Vulkan
                    && delta_input_matrices.front()->dtype
                           == DType::BFloat16)
                {
                    plan.fused_delta_input_bfloat16_operator =
                        NcnnVulkanBfloat16Operator::create_fused(
                            delta_input_matrices,
                            delta_input_biases,
                            layer_plan.vulkan_device_index);
                }
                else
                {
                    plan.fused_delta_input_operator =
                        NcnnLinearOperator::create_fused(
                            delta_input_matrices,
                            delta_input_biases,
                            attention_device,
                            layer_plan.vulkan_device_index);
                }
                if (attention_device == NcnnLinearDevice::Vulkan
                    && !plan.fused_delta_input_bfloat16_operator
                    && !plan.fused_delta_input_operator)
                {
                    plan.fused_delta_input_operator =
                        NcnnLinearOperator::create_fused(
                            delta_input_matrices,
                            delta_input_biases,
                            attention_device,
                            layer_plan.vulkan_device_index);
                }
                if (attention_device == NcnnLinearDevice::Vulkan
                    && !plan.fused_delta_input_bfloat16_operator
                    && !plan.fused_delta_input_operator)
                {
                    return Error{
                        ErrorCode::InternalError,
                        "failed to create fused Vulkan Gated DeltaNet input operator"};
                }
                if (!plan.fused_delta_input_bfloat16_operator
                    && !plan.fused_delta_input_operator)
                {
                    const TensorHandle delta_input_handles[] = {
                        plan.delta_qkv_weight,
                        plan.delta_z_weight,
                        plan.delta_beta_weight,
                        plan.delta_alpha_weight,
                    };
                    for (TensorHandle handle : delta_input_handles)
                    {
                        prepared = prepare_linear_operator(
                            compiled.weights,
                            handle,
                            invalid_tensor_handle,
                            attention_device,
                            retain_cpu_dense_copies,
                            layer_plan.vulkan_device_index);
                        if (!prepared)
                            return prepared.error();
                    }
                }
                prepared = prepare_linear_operator(
                    compiled.weights,
                    plan.output_weight,
                    invalid_tensor_handle,
                    attention_device,
                    retain_cpu_dense_copies,
                    layer_plan.vulkan_device_index);
                if (!prepared)
                    return prepared.error();
            }
            else
            {
                plan.head_count = attention.head_count;
                plan.kv_head_count = attention.kv_head_count;
                plan.head_dimension = attention.head_dimension;
                plan.rope_head_dimension = attention.qk_rope_head_dimension;
                plan.sliding_window = attention.sliding_window;
                plan.initial_context_length = attention.initial_context_length;
                plan.max_context_length = attention.max_context_length;
                plan.rope_theta = attention.rope_theta;
                plan.rope_scaling_factor = attention.rope_scaling_factor;
                plan.rope_ntk_alpha = attention.rope_ntk_alpha;
                plan.rope_ntk_beta = attention.rope_ntk_beta;
                plan.norm_weight_offset = compiled.descriptor.norm_weight_offset;
                if (has_flag(graph.flags, AdapterGraphAttentionSink))
                    plan.flags |= AttentionBlockSink;
                if (has_flag(attention.flags, AttentionDescriptorQueryKeyNorm))
                {
                    plan.flags |= AttentionBlockQueryKeyNorm;
                }
                if (has_flag(attention.flags, AttentionDescriptorOutputGate))
                {
                    plan.flags |= AttentionBlockOutputGate;
                }
                if (plan.rope_head_dimension != 0
                    && (plan.rope_head_dimension > plan.head_dimension
                        || plan.rope_head_dimension % 2 != 0))
                {
                    return Error{ErrorCode::InvalidModel, "invalid partial rotary dimension"};
                }
                const bool fused_vulkan_attention_eligible =
                    attention_device == NcnnLinearDevice::Vulkan
                    && !has_flag(
                        plan.flags,
                        AttentionBlockQueryKeyNorm)
                    && !has_flag(
                        plan.flags,
                        AttentionBlockOutputGate)
                    && (plan.rope_head_dimension == 0
                        || plan.rope_head_dimension
                               == plan.head_dimension)
                    && plan.norm_weight_offset == 0.0f;

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
                Result<TensorHandle> output_gate_weight = invalid_tensor_handle;
                Result<TensorHandle> query_norm_weight = invalid_tensor_handle;
                Result<TensorHandle> key_norm_weight = invalid_tensor_handle;
                if (has_flag(attention.flags, AttentionDescriptorBias))
                {
                    query_bias = require_tensor(compiled.weights, layer_name + "attention.query.bias", {query_size}, compiled.descriptor.activation_dtype);
                    key_bias = require_tensor(compiled.weights, layer_name + "attention.key.bias", {key_value_size}, compiled.descriptor.activation_dtype);
                    value_bias = require_tensor(compiled.weights, layer_name + "attention.value.bias", {key_value_size}, compiled.descriptor.activation_dtype);
                    output_bias = require_tensor(compiled.weights, layer_name + "attention.output.bias", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
                }
                if (has_flag(plan.flags, AttentionBlockQueryKeyNorm))
                {
                    query_norm_weight = require_tensor(compiled.weights, layer_name + "attention.query_norm.weight", {attention.head_dimension}, compiled.descriptor.activation_dtype);
                    key_norm_weight = require_tensor(compiled.weights, layer_name + "attention.key_norm.weight", {attention.head_dimension}, compiled.descriptor.activation_dtype);
                }
                if (has_flag(plan.flags, AttentionBlockOutputGate))
                {
                    output_gate_weight = require_tensor(
                        compiled.weights,
                        layer_name + "attention.output_gate.weight",
                        {query_size, compiled.descriptor.hidden_size},
                        compiled.descriptor.activation_dtype);
                }
                Result<TensorHandle> sinks = invalid_tensor_handle;
                if (has_flag(plan.flags, AttentionBlockSink))
                {
                    sinks = require_tensor(compiled.weights, layer_name + "attention.sinks", {attention.head_count}, compiled.descriptor.activation_dtype);
                }
                if (!attention_norm
                    || !query_weight || !query_bias || !query_norm_weight
                    || !key_weight || !key_bias || !key_norm_weight
                    || !value_weight || !value_bias
                    || !output_weight || !output_bias || !output_gate_weight
                    || !sinks)
                {
                    const Error* error = !attention_norm       ? &attention_norm.error()
                                         : !query_weight       ? &query_weight.error()
                                         : !query_bias         ? &query_bias.error()
                                         : !query_norm_weight  ? &query_norm_weight.error()
                                         : !key_weight         ? &key_weight.error()
                                         : !key_bias           ? &key_bias.error()
                                         : !key_norm_weight    ? &key_norm_weight.error()
                                         : !value_weight       ? &value_weight.error()
                                         : !value_bias         ? &value_bias.error()
                                         : !output_weight      ? &output_weight.error()
                                         : !output_bias        ? &output_bias.error()
                                         : !output_gate_weight ? &output_gate_weight.error()
                                                               : &sinks.error();
                    return *error;
                }
                plan.pre_attention_norm_weight = attention_norm.value();
                plan.query_weight = query_weight.value();
                plan.query_bias = query_bias.value();
                plan.query_norm_weight = query_norm_weight.value();
                plan.key_weight = key_weight.value();
                plan.key_bias = key_bias.value();
                plan.key_norm_weight = key_norm_weight.value();
                plan.value_weight = value_weight.value();
                plan.value_bias = value_bias.value();
                plan.output_weight = output_weight.value();
                plan.output_bias = output_bias.value();
                plan.output_gate_weight = output_gate_weight.value();
                plan.sinks = sinks.value();
                if (attention_device == NcnnLinearDevice::Vulkan)
                {
                    const TensorData* query_bias_data = plan.query_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.query_bias);
                    const TensorData* key_bias_data = plan.key_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.key_bias);
                    const TensorData* value_bias_data = plan.value_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.value_bias);
                    const std::vector<const TensorData*> qkv_matrices = {
                        &compiled.weights.at(plan.query_weight),
                        &compiled.weights.at(plan.key_weight),
                        &compiled.weights.at(plan.value_weight),
                    };
                    const std::vector<const TensorData*> qkv_biases = {
                        query_bias_data,
                        key_bias_data,
                        value_bias_data,
                    };
                    if (!fused_vulkan_attention_eligible
                        && qkv_matrices.front()->dtype
                               == DType::BFloat16
                        && plan.output_gate_weight
                               != invalid_tensor_handle
                        && !query_bias_data
                        && !key_bias_data
                        && !value_bias_data)
                    {
                        std::vector<const TensorData*>
                            qkv_gate_matrices = qkv_matrices;
                        qkv_gate_matrices.push_back(
                            &compiled.weights.at(
                                plan.output_gate_weight));
                        std::vector<const TensorData*>
                            qkv_gate_biases = qkv_biases;
                        qkv_gate_biases.push_back(nullptr);
                        plan.fused_qkv_gate_bfloat16_operator =
                            NcnnVulkanBfloat16Operator::create_fused(
                                qkv_gate_matrices,
                                qkv_gate_biases,
                                layer_plan.vulkan_device_index);
                    }
                    if (!plan.fused_qkv_gate_bfloat16_operator
                        && !fused_vulkan_attention_eligible
                        && qkv_matrices.front()->dtype
                               == DType::BFloat16)
                    {
                        plan.fused_qkv_bfloat16_operator =
                            NcnnVulkanBfloat16Operator::create_fused(
                                qkv_matrices,
                                qkv_biases,
                                layer_plan.vulkan_device_index);
                    }
                    if (!plan.fused_qkv_gate_bfloat16_operator
                        && !plan.fused_qkv_bfloat16_operator)
                    {
                        plan.fused_qkv_operator =
                            NcnnLinearOperator::create_fused(
                                qkv_matrices,
                                qkv_biases,
                                attention_device,
                                layer_plan.vulkan_device_index);
                    }
                    if (!plan.fused_qkv_gate_bfloat16_operator
                        && !plan.fused_qkv_bfloat16_operator
                        && !plan.fused_qkv_operator)
                        return Error{ErrorCode::InternalError, "failed to create fused Vulkan QKV operator"};
                }
                else
                {
                    prepared = prepare_linear_operator(compiled.weights, plan.query_weight, plan.query_bias, attention_device, retain_cpu_dense_copies, layer_plan.vulkan_device_index);
                    if (!prepared)
                        return prepared.error();
                    prepared = prepare_linear_operator(compiled.weights, plan.key_weight, plan.key_bias, attention_device, retain_cpu_dense_copies, layer_plan.vulkan_device_index);
                    if (!prepared)
                        return prepared.error();
                    prepared = prepare_linear_operator(compiled.weights, plan.value_weight, plan.value_bias, attention_device, retain_cpu_dense_copies, layer_plan.vulkan_device_index);
                    if (!prepared)
                        return prepared.error();
                }
                prepared = prepare_linear_operator(
                    compiled.weights,
                    plan.output_weight,
                    plan.output_bias,
                    attention_device,
                    retain_cpu_dense_copies,
                    layer_plan.vulkan_device_index,
                    1,
                    !fused_vulkan_attention_eligible);
                if (!prepared)
                    return prepared.error();
                if (plan.output_gate_weight != invalid_tensor_handle)
                {
                    if (!plan.fused_qkv_gate_bfloat16_operator)
                    {
                        prepared = prepare_linear_operator(
                            compiled.weights,
                            plan.output_gate_weight,
                            invalid_tensor_handle,
                            attention_device,
                            retain_cpu_dense_copies,
                            layer_plan.vulkan_device_index);
                        if (!prepared)
                            return prepared.error();
                    }
                }
                if (fused_vulkan_attention_eligible)
                {
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
                    if (has_flag(plan.flags, AttentionBlockSink))
                        attention_config.flags |= NcnnAttentionSink;
                    plan.vulkan_attention_operator = NcnnVulkanAttentionOperator::create(compiled.weights.at(plan.pre_attention_norm_weight), plan.sinks == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.sinks),
                                                                                         plan.fused_qkv_operator, compiled.weights.at(plan.output_weight).linear_operator, attention_config);
                }
            }
        }

        auto norm = require_tensor(compiled.weights, layer_name + "pre_ffn_norm.weight", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
        if (!norm)
            return norm.error();
        layer_plan.moe.pre_ffn_norm_weight = norm.value();

        auto router = require_tensor(compiled.weights, layer_name + "router.weight", {moe.expert_count, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
        if (!router)
            return router.error();
        layer_plan.moe.router_weight = router.value();

        if (has_flag(moe.flags, MoeDescriptorRouterBias))
        {
            if (moe.score_function == RouterScoreFunction::Softmax)
            {
                auto bias = require_tensor(compiled.weights, layer_name + "router.bias", {moe.expert_count}, compiled.descriptor.activation_dtype);
                if (!bias)
                    return bias.error();
                layer_plan.moe.router_bias = bias.value();
            }
            else
            {
                auto bias = require_tensor(compiled.weights, layer_name + "router.selection_bias", {moe.expert_count}, DType::Float32);
                if (!bias)
                    return bias.error();
                layer_plan.moe.router_selection_bias = bias.value();
            }
        }
        if (layer_id < compiled.descriptor.hash_routing_layer_count)
        {
            auto token_experts = require_tensor(compiled.weights, layer_name + "router.token_experts", {compiled.descriptor.vocabulary_size, moe.top_k}, DType::Int64);
            if (!token_experts)
                return token_experts.error();
            layer_plan.moe.token_experts = token_experts.value();
        }
        // The CPU router avoids a Vulkan round trip before Expert dispatch.
        prepared = prepare_linear_operator(compiled.weights, layer_plan.moe.router_weight, layer_plan.moe.router_bias, NcnnLinearDevice::Cpu, retain_cpu_dense_copies, layer_plan.vulkan_device_index);
        if (!prepared)
            return prepared.error();

        if (layer_plan.moe.has_shared_expert)
        {
            if (moe.shared_expert_count != 1)
                return Error{ErrorCode::UnsupportedModel, "the CPU runtime supports one shared Expert per MoE block"};
            ExpertPlan& shared = layer_plan.moe.shared_expert;
            shared.activation = moe.activation;
            shared.activation_limit = moe.activation_limit;
            auto gate = require_tensor(compiled.weights, layer_name + "shared_expert.gate.weight", {moe.intermediate_size, compiled.descriptor.hidden_size}, moe.shared_expert_weight_dtype);
            auto up = require_tensor(compiled.weights, layer_name + "shared_expert.up.weight", {moe.intermediate_size, compiled.descriptor.hidden_size}, moe.shared_expert_weight_dtype);
            auto down = require_tensor(compiled.weights, layer_name + "shared_expert.down.weight", {compiled.descriptor.hidden_size, moe.intermediate_size}, moe.shared_expert_weight_dtype);
            if (!gate || !up || !down)
            {
                return !gate ? gate.error() : !up ? up.error()
                                                  : down.error();
            }
            shared.gate_weight = gate.value();
            shared.up_weight = up.value();
            shared.down_weight = down.value();
            shared.weight_bytes = expert_weight_bytes(compiled.weights, shared);
            if (has_flag(moe.flags, MoeDescriptorSharedExpertGate))
            {
                auto shared_gate = require_tensor(
                    compiled.weights,
                    layer_name + "shared_expert.router_gate.weight",
                    {1, compiled.descriptor.hidden_size},
                    compiled.descriptor.activation_dtype);
                if (!shared_gate)
                    return shared_gate.error();
                layer_plan.moe.shared_expert_gate_weight = shared_gate.value();
            }
            prepared = prepare_shared_expert_operators(
                compiled.weights,
                layer_plan.moe,
                dense_device,
                retain_cpu_dense_copies,
                layer_plan.vulkan_device_index);
            if (!prepared)
                return prepared.error();
        }

        layer_plan.moe.experts.reserve(moe.expert_count);
        const bool prepare_routed_dense_operators =
            moe.expert_weight_dtype != DType::BFloat16
            || moe.expert_count <= 64;
        for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
        {
            ExpertPlan expert;
            expert.activation = moe.activation;
            expert.activation_limit = moe.activation_limit;
            if (moe.layout == ExpertLayout::UpDown)
                expert.flags &= ~ExpertPlanGated;
            if (moe.layout == ExpertLayout::PackedGateUpDown)
                expert.flags |= ExpertPlanPackedGateUp;
            const std::string prefix = expert_prefix(layer_id, expert_id);

            if (moe.layout == ExpertLayout::InterleavedGateUpDown || moe.layout == ExpertLayout::PackedGateUpDown)
            {
                auto gate_up = require_tensor(compiled.weights, prefix + "gate_up.weight", {moe.intermediate_size * 2, compiled.descriptor.hidden_size}, moe.expert_weight_dtype);
                if (!gate_up)
                    return gate_up.error();
                expert.gate_up_weight = gate_up.value();
            }
            else if (has_flag(expert.flags, ExpertPlanGated))
            {
                auto gate = require_tensor(compiled.weights, prefix + "gate.weight", {moe.intermediate_size, compiled.descriptor.hidden_size}, moe.expert_weight_dtype);
                if (!gate)
                    return gate.error();
                expert.gate_weight = gate.value();
                if (prepare_routed_dense_operators)
                    (void)prepare_linear_operator(compiled.weights, expert.gate_weight, invalid_tensor_handle, NcnnLinearDevice::Cpu, retain_cpu_dense_copies, layer_plan.vulkan_device_index);
            }

            if (moe.layout != ExpertLayout::InterleavedGateUpDown && moe.layout != ExpertLayout::PackedGateUpDown)
            {
                auto up = require_tensor(compiled.weights, prefix + "up.weight", {moe.intermediate_size, compiled.descriptor.hidden_size}, moe.expert_weight_dtype);
                if (!up)
                    return up.error();
                expert.up_weight = up.value();
                if (prepare_routed_dense_operators)
                    (void)prepare_linear_operator(compiled.weights, expert.up_weight, invalid_tensor_handle, NcnnLinearDevice::Cpu, retain_cpu_dense_copies, layer_plan.vulkan_device_index);
            }

            auto down = require_tensor(compiled.weights, prefix + "down.weight", {compiled.descriptor.hidden_size, moe.intermediate_size}, moe.expert_weight_dtype);
            if (!down)
                return down.error();
            expert.down_weight = down.value();

            if (has_flag(moe.flags, MoeDescriptorProjectionBias))
            {
                auto gate_up_bias = require_tensor(compiled.weights, prefix + "gate_up.bias", {moe.intermediate_size * 2}, compiled.descriptor.activation_dtype);
                auto down_bias = require_tensor(compiled.weights, prefix + "down.bias", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
                if (!gate_up_bias)
                    return gate_up_bias.error();
                if (!down_bias)
                    return down_bias.error();
                expert.gate_up_bias = gate_up_bias.value();
                expert.down_bias = down_bias.value();
            }
            if (prepare_routed_dense_operators)
                (void)prepare_linear_operator(compiled.weights, expert.down_weight, expert.down_bias, NcnnLinearDevice::Cpu, retain_cpu_dense_copies, layer_plan.vulkan_device_index);

            expert.weight_bytes = expert_weight_bytes(compiled.weights, expert);
            const TensorData& down_weight = compiled.weights.at(expert.down_weight);
            if (expert.gate_up_weight != invalid_tensor_handle)
            {
                const TensorData& gate_up_weight = compiled.weights.at(expert.gate_up_weight);
                if (gate_up_weight.mxfp4_file_storage && down_weight.mxfp4_file_storage)
                {
                    expert.cache_key = Mxfp4ExpertCache::make_pair_key(gate_up_weight, down_weight);
                }
            }
            const bool file_backed = down_weight.mxfp4_file_storage
                                     || (expert.gate_up_weight != invalid_tensor_handle
                                         && compiled.weights.at(expert.gate_up_weight).mxfp4_file_storage);
            expert.runtime = std::shared_ptr<Expert>(new Expert(ExpertKey{layer_id, expert_id}, expert.weight_bytes, file_backed ? ExpertCacheState::Unloaded : ExpertCacheState::Resident,
                                                                file_backed ? TensorLocation::Automatic : TensorLocation::Cpu, selected_expert_kernel(moe.expert_weight_dtype)));
            compiled.expert_store->add(expert.runtime);
            layer_plan.moe.experts.push_back(expert);
        }

        compiled.layers.push_back(std::move(layer_plan));
    }

    auto speculative = compile_speculative_model(compiled, dense_device, retain_cpu_dense_copies);
    if (!speculative)
        return speculative.error();
    for (CompiledLayerPlan& layer_plan : compiled.speculative.layers)
    {
        const MoeDescriptor& moe = compiled.descriptor.layers.back().ffn.moe;
        for (uint32_t expert_id = 0; expert_id < layer_plan.moe.experts.size(); ++expert_id)
        {
            ExpertPlan& expert = layer_plan.moe.experts[expert_id];
            if (expert.runtime)
                continue;
            const TensorData& down_weight = compiled.weights.at(expert.down_weight);
            const bool file_backed =
                down_weight.mxfp4_file_storage
                || (expert.gate_up_weight != invalid_tensor_handle
                    && compiled.weights.at(expert.gate_up_weight).mxfp4_file_storage);
            expert.runtime = std::shared_ptr<Expert>(
                new Expert(
                    ExpertKey{layer_plan.layer_id, expert_id},
                    expert.weight_bytes,
                    file_backed ? ExpertCacheState::Unloaded : ExpertCacheState::Resident,
                    file_backed ? TensorLocation::Automatic : TensorLocation::Cpu,
                    selected_expert_kernel(moe.expert_weight_dtype)));
            compiled.expert_store->add(expert.runtime);
        }
    }

    auto graph = build_compiled_execution_graph(compiled, capabilities);
    if (!graph)
        return graph.error();
    if (has_flag(capabilities.flags, BackendCapabilityReleaseVulkanDenseHostStorage))
        release_vulkan_dense_host_copies(compiled);
    return compiled;
}

} // namespace moe
} // namespace ncnn
