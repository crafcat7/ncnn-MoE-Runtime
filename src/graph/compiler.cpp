#include "compiler.h"

#include "graph.h"
#include "kernels/qnk.h"
#include "models/tensornames.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/modelpipeline.h"
#include "storage/expertcache.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace ncnn {
namespace moe {

Result<void> validate_model_descriptor(const MoeModelDescriptor& descriptor)
{
    if (descriptor.model_type.empty())
        return Error{ErrorCode::InvalidModel, "model descriptor model_type cannot be empty"};
    if (descriptor.layers.empty())
        return Error{ErrorCode::InvalidModel, "model descriptor requires at least one layer"};
    if (descriptor.layers.size() > std::numeric_limits<uint32_t>::max())
        return Error{ErrorCode::InvalidModel, "model descriptor layer count exceeds uint32 range"};
    if (descriptor.vocabulary_size == 0 || descriptor.hidden_size == 0 || descriptor.intermediate_size == 0
        || descriptor.expert_count == 0 || descriptor.experts_per_token == 0)
    {
        return Error{ErrorCode::InvalidModel, "model descriptor dimensions must be non-zero"};
    }
    const uint32_t layer_count = static_cast<uint32_t>(descriptor.layers.size());
    if (descriptor.speculative_layer_count > std::numeric_limits<uint32_t>::max() - layer_count)
        return Error{ErrorCode::InvalidModel, "model descriptor speculative layer IDs overflow"};
    if (descriptor.speculative_layer_count == 0)
    {
        if (descriptor.speculative_kind != SpeculativeModelKind::None
            || !descriptor.speculative_target_layer_ids.empty())
        {
            return Error{ErrorCode::InvalidModel, "speculative fields require speculative layers"};
        }
    }
    else if (descriptor.speculative_kind != SpeculativeModelKind::Mtp
             && descriptor.speculative_kind != SpeculativeModelKind::DSpark)
    {
        return Error{ErrorCode::InvalidModel, "speculative layer kind is not configured"};
    }
    if (descriptor.speculative_kind == SpeculativeModelKind::DSpark)
    {
        const std::vector<uint32_t>& targets = descriptor.speculative_target_layer_ids;
        for (auto target = targets.begin(); target != targets.end(); ++target)
        {
            if (*target >= layer_count)
                return Error{ErrorCode::InvalidModel, "speculative target layer ID is out of range"};
            if (std::find(targets.begin(), target, *target) != target)
            {
                return Error{ErrorCode::InvalidModel, "speculative target layer IDs contain duplicates"};
            }
        }
    }
    for (const LayerDescriptor& layer : descriptor.layers)
    {
        if (layer.ffn.kind != FfnKind::Moe && layer.ffn.kind != FfnKind::Dense)
            return Error{ErrorCode::InvalidModel, "model descriptor layer has an invalid FFN kind"};
        if (layer.ffn.kind == FfnKind::Moe)
        {
            const MoeDescriptor& moe = layer.ffn.moe;
            if (moe.expert_count == 0 || moe.top_k == 0 || moe.top_k > moe.expert_count)
                return Error{ErrorCode::InvalidModel, "invalid expert_count/top_k"};
            if (moe.intermediate_size == 0)
                return Error{ErrorCode::InvalidModel, "intermediate_size must be non-zero"};
            if (moe.expert_count != descriptor.expert_count
                || moe.top_k != descriptor.experts_per_token
                || moe.intermediate_size != descriptor.intermediate_size)
            {
                return Error{ErrorCode::InvalidModel, "layer MoE dimensions do not match the model descriptor"};
            }
            if (moe.normalization != RouterNormalization::None
                && moe.normalization != RouterNormalization::SelectedExperts)
                return Error{ErrorCode::InvalidModel, "model descriptor layer has an invalid router normalization"};
            const uint32_t known_flags = MoeDescriptorRouterBias
                                         | MoeDescriptorProjectionBias
                                         | MoeDescriptorSharedExpertGate
                                         | MoeDescriptorFileBackedExperts;
            if ((moe.flags & ~known_flags) != 0)
                return Error{ErrorCode::InvalidModel, "model descriptor layer has unknown MoE flags"};
            if (moe.score_function != RouterScoreFunction::Softmax
                && moe.score_function != RouterScoreFunction::Sigmoid
                && moe.score_function != RouterScoreFunction::SqrtSoftplus)
                return Error{ErrorCode::InvalidModel, "model descriptor layer has an invalid router score function"};
            if (moe.activation != ExpertActivation::Relu
                && moe.activation != ExpertActivation::Silu
                && moe.activation != ExpertActivation::Gelu
                && moe.activation != ExpertActivation::ClampedSilu
                && moe.activation != ExpertActivation::DeepSeekSwiGlu
                && moe.activation != ExpertActivation::GptOssSwiGlu)
                return Error{ErrorCode::InvalidModel, "model descriptor layer has an invalid expert activation"};
            if (moe.layout != ExpertLayout::UpDown
                && moe.layout != ExpertLayout::GateUpDown
                && moe.layout != ExpertLayout::PackedGateUpDown
                && moe.layout != ExpertLayout::InterleavedGateUpDown)
                return Error{ErrorCode::InvalidModel, "model descriptor layer has an invalid expert layout"};
        }
        if (layer.attention.kind != AttentionKind::None
            && layer.attention.kind != AttentionKind::Standard
            && layer.attention.kind != AttentionKind::GatedDeltaNet
            && layer.attention.kind != AttentionKind::MultiHeadLatent)
            return Error{ErrorCode::InvalidModel, "model descriptor layer has an invalid attention kind"};
        if (layer.attention.kind != AttentionKind::None
            && layer.attention.kind != AttentionKind::GatedDeltaNet)
        {
            const AttentionDescriptor& attention = layer.attention;
            if (attention.head_count == 0
                || attention.kv_head_count == 0
                || attention.head_dimension == 0
                || attention.head_count % attention.kv_head_count != 0
                || attention.head_dimension % 2 != 0)
                return Error{ErrorCode::InvalidModel, "invalid attention dimensions"};
            if (attention.head_count != descriptor.attention_head_count
                || attention.kv_head_count != descriptor.kv_head_count
                || attention.head_dimension != descriptor.head_dimension)
                return Error{ErrorCode::InvalidModel, "layer attention dimensions do not match the model descriptor"};
        }
        if (layer.attention.kind != AttentionKind::None)
        {
            const AttentionDescriptor& attention = layer.attention;
            const uint32_t known_flags = AttentionDescriptorBias
                                         | AttentionDescriptorSinks
                                         | AttentionDescriptorQueryKeyNorm
                                         | AttentionDescriptorOutputGate
                                         | AttentionDescriptorQsa
                                         | AttentionDescriptorSigmoidGate;
            if ((attention.flags & ~known_flags) != 0)
                return Error{ErrorCode::InvalidModel, "model descriptor layer has unknown attention flags"};

            uint32_t allowed_flags = AttentionDescriptorSinks | AttentionDescriptorQueryKeyNorm;
            if (attention.kind == AttentionKind::Standard)
            {
                allowed_flags = AttentionDescriptorBias
                                | AttentionDescriptorSinks
                                | AttentionDescriptorQueryKeyNorm
                                | AttentionDescriptorOutputGate
                                | AttentionDescriptorQsa;
            }
            else if (attention.kind == AttentionKind::GatedDeltaNet)
            {
                allowed_flags = AttentionDescriptorSigmoidGate;
            }
            if ((attention.flags & ~allowed_flags) != 0)
                return Error{ErrorCode::InvalidModel, "model descriptor layer has incompatible attention flags"};
        }
    }
    if (descriptor.hyper_connection_kind != HyperConnectionKind::None
        && descriptor.hyper_connection_kind != HyperConnectionKind::Sinkhorn
        && descriptor.hyper_connection_kind != HyperConnectionKind::GatedResidual)
        return Error{ErrorCode::InvalidModel, "model descriptor has an invalid hyper-connection kind"};
    return {};
}

static uint64_t gated_delta_vulkan_budget_size(
    const CompilerOption& opt,
    bool use_vulkan_dense,
    bool protects_file_backed_experts) noexcept
{
    if (!use_vulkan_dense || opt.gpu_heap_budget == 0)
        return 0;

    // File-backed Expert execution normally consumes nearly all free heap for
    // its executable/victim caches.  Reserve a small, concurrency-scaled
    // fraction for persistent GDN projection/state; without an Expert cache,
    // more of the heap can safely be used by dense attention operators.
    const uint64_t heap_divisor = protects_file_backed_experts ? 64 : 8;
    const uint64_t concurrency = std::max(1u, opt.num_concurrent_sessions);
    return opt.gpu_heap_budget / (heap_divisor * concurrency);
}

static Result<TensorHandle> require_tensor(const WeightStore& weights, const std::string& name, std::initializer_list<uint32_t> shape, DType dtype)
{
    const TensorHandle handle = weights.find_handle(name);
    if (handle == invalid_tensor_handle)
        return Error{ErrorCode::InvalidModel, "missing tensor: " + name};

    const TensorData& tensor = weights.at(handle);
    if (tensor.dtype != dtype)
        return Error{ErrorCode::InvalidModel, "unexpected dtype for tensor: " + name};
    if (tensor.shape.size() != shape.size()
        || !std::equal(tensor.shape.begin(), tensor.shape.end(), shape.begin()))
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
            const uint64_t stored_blocks = tensor.mxfp4_file_storage->blocks_size + tensor.mxfp4_file_storage->secondary_blocks_size;
            const uint64_t stored_scales = tensor.mxfp4_file_storage->scales_size + tensor.mxfp4_file_storage->secondary_scales_size;
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
    else if (is_qnk_dtype(dtype))
    {
        if (tensor.shape.size() != 2 || !qnk_shape_supported(dtype, tensor.shape[0], tensor.shape[1])
            || tensor.qnk_values().size() != qnk_storage_bytes(dtype, tensor.shape[0], tensor.shape[1]))
        {
            return Error{ErrorCode::InvalidModel, "invalid Qn_K tensor storage: " + name};
        }
        if (!tensor.float32_data.empty()
            || !tensor.bfloat16_data.empty()
            || !tensor.int8_data.empty()
            || !tensor.quantization_scales.empty()
            || !tensor.mxfp4_blocks.empty()
            || !tensor.mxfp4_scales.empty())
        {
            return Error{ErrorCode::InvalidModel, "Qn_K tensor contains unrelated storage: " + name};
        }
    }
    else
    {
        return Error{ErrorCode::UnsupportedModel, "unsupported tensor dtype: " + name};
    }
    return handle;
}

static Result<void> assign_required_tensor(
    const WeightStore& weights,
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

static Result<void> compile_gated_residual_plan(
    const WeightStore& weights,
    const std::string& prefix,
    const MoeModelDescriptor& descriptor,
    bool with_injection,
    GatedResidualPlan& plan)
{
    const uint32_t expanded_size = descriptor.hyper_connection_multiplier * descriptor.hidden_size;
    Result<void> status = assign_required_tensor(
        weights, prefix + "norm.weight", {expanded_size},
        descriptor.activation_dtype, plan.norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights, prefix + "mix_down.weight",
        {descriptor.hyper_connection_low_rank, expanded_size},
        descriptor.activation_dtype, plan.mix_down_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        weights, prefix + "mix_up.weight",
        {expanded_size, descriptor.hyper_connection_low_rank},
        descriptor.activation_dtype, plan.mix_up_weight);
    if (!status)
        return status.error();
    if (!with_injection)
        return {};
    return assign_required_tensor(
        weights, prefix + "inject.weight",
        {descriptor.hyper_connection_multiplier, expanded_size},
        descriptor.activation_dtype, plan.inject_weight);
}

static Result<void> compile_ple_plan(
    CompiledModel& compiled,
    const std::string& layer_name,
    const PleDescriptor& descriptor,
    bool retain_cpu_dense_copies,
    uint32_t vulkan_device_index,
    PleBlockPlan& plan)
{
    if (!descriptor.enabled())
        return {};
    if (descriptor.ngram_size < 2)
        return Error{ErrorCode::InvalidModel, "invalid PLE dimensions"};
    const uint32_t head_count = (descriptor.ngram_size - 1) * descriptor.heads_per_ngram;
    if (head_count == 0
        || descriptor.embedding_dimension != compiled.descriptor.hidden_size
        || descriptor.embedding_dimension % head_count != 0
        || descriptor.convolution_kernel_size == 0
        || descriptor.embedding_shard_count == 0
        || descriptor.embedding_row_count == 0)
    {
        return Error{ErrorCode::InvalidModel, "invalid PLE dimensions"};
    }
    const uint32_t expanded_size = compiled.descriptor.hyper_connection_multiplier
                                   * compiled.descriptor.hidden_size;
    const uint32_t head_dimension = descriptor.embedding_dimension / head_count;
    Result<void> status = assign_required_tensor(
        compiled.weights, layer_name + "ple.key.weight",
        {expanded_size, compiled.descriptor.hidden_size},
        compiled.descriptor.activation_dtype, plan.key_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights, layer_name + "ple.value.weight",
        {compiled.descriptor.hidden_size, compiled.descriptor.hidden_size},
        compiled.descriptor.activation_dtype, plan.value_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights, layer_name + "ple.key_norm.weight", {expanded_size},
        compiled.descriptor.activation_dtype, plan.key_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights, layer_name + "ple.query_norm.weight", {expanded_size},
        compiled.descriptor.activation_dtype, plan.query_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights, layer_name + "ple.convolution_norm.weight",
        {expanded_size}, compiled.descriptor.activation_dtype,
        plan.convolution_norm_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights, layer_name + "ple.convolution.weight",
        {expanded_size, 1, descriptor.convolution_kernel_size},
        compiled.descriptor.activation_dtype, plan.convolution_weight);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights, layer_name + "ple.hash_multipliers",
        {descriptor.ngram_size}, DType::Int64, plan.hash_multipliers);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights, layer_name + "ple.head_vocabulary_sizes",
        {head_count}, DType::Int64, plan.head_vocabulary_sizes);
    if (!status)
        return status.error();
    status = assign_required_tensor(
        compiled.weights, layer_name + "ple.head_offsets",
        {head_count}, DType::Int64, plan.head_offsets);
    if (!status)
        return status.error();

    uint64_t embedding_rows = 0;
    plan.embedding_shards.reserve(descriptor.embedding_shard_count);
    for (uint32_t shard = 0; shard < descriptor.embedding_shard_count; ++shard)
    {
        const std::string name = layer_name + "ple.embedding_shard." + std::to_string(shard);
        const TensorHandle handle = compiled.weights.find_handle(name);
        if (handle == invalid_tensor_handle)
            return Error{ErrorCode::InvalidModel, "missing tensor: " + name};
        const TensorData& tensor = compiled.weights.at(handle);
        if (tensor.dtype != compiled.descriptor.activation_dtype
            || tensor.shape.size() != 2 || tensor.shape[0] == 0
            || tensor.shape[1] != head_dimension
            || tensor.bfloat16_values().size() != tensor.element_count())
        {
            return Error{ErrorCode::InvalidModel, "invalid PLE embedding shard: " + name};
        }
        if (tensor.shape[0] > std::numeric_limits<uint64_t>::max() - embedding_rows)
            return Error{ErrorCode::InvalidModel, "PLE embedding row count overflows"};
        embedding_rows += tensor.shape[0];
        plan.embedding_shards.push_back(handle);
    }
    if (embedding_rows != descriptor.embedding_row_count)
        return Error{ErrorCode::InvalidModel, "PLE embedding shard rows do not match the model configuration"};

    const std::span<const int64_t> vocabulary_sizes = compiled.weights.at(plan.head_vocabulary_sizes).int64_values();
    const std::span<const int64_t> offsets = compiled.weights.at(plan.head_offsets).int64_values();
    uint64_t required_rows = 0;
    for (uint32_t head = 0; head < head_count; ++head)
    {
        if (vocabulary_sizes[head] <= 0 || offsets[head] < 0
            || static_cast<uint64_t>(offsets[head]) != required_rows)
        {
            return Error{ErrorCode::InvalidModel, "invalid PLE embedding metadata"};
        }
        const uint64_t size = static_cast<uint64_t>(vocabulary_sizes[head]);
        if (size > std::numeric_limits<uint64_t>::max() - required_rows)
            return Error{ErrorCode::InvalidModel, "PLE embedding metadata overflows"};
        required_rows += size;
    }
    if (required_rows > embedding_rows)
        return Error{ErrorCode::InvalidModel, "PLE embedding metadata exceeds the embedding table"};
    plan.embedding_dimension = descriptor.embedding_dimension;
    plan.convolution_kernel_size = descriptor.convolution_kernel_size;
    plan.ngram_size = descriptor.ngram_size;
    plan.heads_per_ngram = descriptor.heads_per_ngram;
    plan.eos_token_id = descriptor.eos_token_id;

    const TensorHandle linear_handles[] = {plan.key_weight, plan.value_weight};
    for (TensorHandle handle : linear_handles)
    {
        status = prepare_linear_operator(
            compiled.weights, compiled.operators, handle,
            invalid_tensor_handle, NcnnLinearDevice::Cpu,
            retain_cpu_dense_copies, vulkan_device_index,
            compiled.vulkan_context_instance, compiled.opt.optimization_flags);
        if (!status)
            return status.error();
    }
    return {};
}

static Result<void> compile_latent_attention(
    const WeightStore& weights,
    const std::string& layer_name,
    const MoeModelDescriptor& descriptor,
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

    plan.kind = AttentionKind::MultiHeadLatent;
    plan.flags |= AttentionBlockSink | AttentionBlockQueryKeyNorm;
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
    const WeightStore& weights,
    const std::string& layer_name,
    const MoeModelDescriptor& descriptor,
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

    plan.kind = AttentionKind::GatedDeltaNet;
    if (descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
        plan.flags |= AttentionBlockExternalResidual;
    if (has_flag(attention.flags, AttentionDescriptorSigmoidGate))
        plan.flags |= AttentionBlockSigmoidGate;
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
    Result<void> status;
    if (descriptor.hyper_connection_kind != HyperConnectionKind::GatedResidual)
    {
        status = assign_required_tensor(
            weights,
            layer_name + "pre_attention_norm.weight",
            {descriptor.hidden_size},
            descriptor.activation_dtype,
            plan.pre_attention_norm_weight);
        if (!status)
            return status.error();
    }
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

static Result<void> compile_standard_attention(
    const WeightStore& weights, const std::string& layer_name,
    const MoeModelDescriptor& descriptor, const LayerDescriptor& layer,
    AttentionBlockPlan& plan)
{
    const AttentionDescriptor& attention = layer.attention;
    plan.kind = AttentionKind::Standard;
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
    plan.norm_weight_offset = descriptor.norm_weight_offset;
    if (has_flag(attention.flags, AttentionDescriptorSinks))
        plan.flags |= AttentionBlockSink;
    if (has_flag(attention.flags, AttentionDescriptorQueryKeyNorm))
        plan.flags |= AttentionBlockQueryKeyNorm;
    if (has_flag(attention.flags, AttentionDescriptorOutputGate))
        plan.flags |= AttentionBlockOutputGate;
    if (has_flag(attention.flags, AttentionDescriptorQsa))
    {
        plan.flags |= AttentionBlockQsa;
        plan.index_head_count = attention.index_head_count;
        plan.index_head_dimension = attention.index_head_dimension;
        plan.index_top_k = attention.index_top_k;
        plan.index_token_budget = attention.index_token_budget;
        plan.compression_ratio = attention.compression_ratio;
    }
    if (descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
        plan.flags |= AttentionBlockExternalResidual;
    if (plan.rope_head_dimension != 0
        && (plan.rope_head_dimension > plan.head_dimension
            || plan.rope_head_dimension % 2 != 0))
    {
        return Error{ErrorCode::InvalidModel, "invalid partial rotary dimension"};
    }
    if (has_flag(plan.flags, AttentionBlockQsa))
    {
        if (attention.sliding_window != 0)
        {
            return Error{
                ErrorCode::UnsupportedModel,
                "QSA attention does not support a sliding-window KV cache"};
        }
        if (attention.index_head_count == 0
            || attention.index_head_dimension == 0
            || attention.index_top_k == 0
            || attention.index_token_budget == 0
            || attention.compression_ratio == 0
            || plan.rope_head_dimension > attention.index_head_dimension
            || attention.index_token_budget % attention.compression_ratio != 0
            || attention.index_top_k
                   != attention.index_token_budget / attention.compression_ratio)
        {
            return Error{ErrorCode::InvalidModel, "invalid QSA indexer dimensions"};
        }
    }

    const uint32_t query_size = attention.head_count * attention.head_dimension;
    const uint32_t key_value_size = attention.kv_head_count * attention.head_dimension;
    Result<void> ret;
    if (layer.pre_attention_norm == NormType::RmsNorm)
    {
        ret = assign_required_tensor(weights, layer_name + "pre_attention_norm.weight", {descriptor.hidden_size}, descriptor.activation_dtype, plan.pre_attention_norm_weight);
        if (!ret)
            return ret.error();
    }
    ret = assign_required_tensor(weights, layer_name + "attention.query.weight", {query_size, descriptor.hidden_size}, descriptor.activation_dtype, plan.query_weight);
    if (!ret)
        return ret.error();
    if (has_flag(attention.flags, AttentionDescriptorBias))
    {
        ret = assign_required_tensor(weights, layer_name + "attention.query.bias", {query_size}, descriptor.activation_dtype, plan.query_bias);
        if (!ret)
            return ret.error();
    }
    if (has_flag(plan.flags, AttentionBlockQueryKeyNorm))
    {
        ret = assign_required_tensor(weights, layer_name + "attention.query_norm.weight", {attention.head_dimension}, descriptor.activation_dtype, plan.query_norm_weight);
        if (!ret)
            return ret.error();
    }
    ret = assign_required_tensor(weights, layer_name + "attention.key.weight", {key_value_size, descriptor.hidden_size}, descriptor.activation_dtype, plan.key_weight);
    if (!ret)
        return ret.error();
    if (has_flag(attention.flags, AttentionDescriptorBias))
    {
        ret = assign_required_tensor(weights, layer_name + "attention.key.bias", {key_value_size}, descriptor.activation_dtype, plan.key_bias);
        if (!ret)
            return ret.error();
    }
    if (has_flag(plan.flags, AttentionBlockQueryKeyNorm))
    {
        ret = assign_required_tensor(weights, layer_name + "attention.key_norm.weight", {attention.head_dimension}, descriptor.activation_dtype, plan.key_norm_weight);
        if (!ret)
            return ret.error();
    }
    ret = assign_required_tensor(weights, layer_name + "attention.value.weight", {key_value_size, descriptor.hidden_size}, descriptor.activation_dtype, plan.value_weight);
    if (!ret)
        return ret.error();
    if (has_flag(attention.flags, AttentionDescriptorBias))
    {
        ret = assign_required_tensor(weights, layer_name + "attention.value.bias", {key_value_size}, descriptor.activation_dtype, plan.value_bias);
        if (!ret)
            return ret.error();
    }
    ret = assign_required_tensor(weights, layer_name + "attention.output.weight", {descriptor.hidden_size, query_size}, descriptor.activation_dtype, plan.output_weight);
    if (!ret)
        return ret.error();
    if (has_flag(attention.flags, AttentionDescriptorBias))
    {
        ret = assign_required_tensor(weights, layer_name + "attention.output.bias", {descriptor.hidden_size}, descriptor.activation_dtype, plan.output_bias);
        if (!ret)
            return ret.error();
    }
    if (has_flag(plan.flags, AttentionBlockOutputGate))
    {
        ret = assign_required_tensor(weights, layer_name + "attention.output_gate.weight", {query_size, descriptor.hidden_size}, descriptor.activation_dtype, plan.output_gate_weight);
        if (!ret)
            return ret.error();
    }
    if (has_flag(plan.flags, AttentionBlockQsa))
    {
        ret = assign_required_tensor(
            weights,
            layer_name + "attention.qsa.query_key.weight",
            {(attention.index_head_count + 1) * attention.index_head_dimension,
             descriptor.hidden_size},
            descriptor.activation_dtype, plan.qsa_query_key_weight);
        if (!ret)
            return ret.error();
        ret = assign_required_tensor(weights, layer_name + "attention.qsa.query_norm.weight", {attention.index_head_dimension}, descriptor.activation_dtype, plan.qsa_query_norm_weight);
        if (!ret)
            return ret.error();
        ret = assign_required_tensor(weights, layer_name + "attention.qsa.key_norm.weight", {attention.index_head_dimension}, descriptor.activation_dtype, plan.qsa_key_norm_weight);
        if (!ret)
            return ret.error();
    }
    if (has_flag(plan.flags, AttentionBlockSink))
    {
        ret = assign_required_tensor(weights, layer_name + "attention.sinks", {attention.head_count}, descriptor.activation_dtype, plan.sinks);
        if (!ret)
            return ret.error();
    }
    return {};
}

static uint64_t tensor_storage_size(const TensorData& tensor)
{
    uint64_t size = tensor.mapped_size;
    size += static_cast<uint64_t>(tensor.float32_data.size()) * sizeof(float);
    size += static_cast<uint64_t>(tensor.bfloat16_data.size()) * sizeof(uint16_t);
    size += static_cast<uint64_t>(tensor.int64_data.size()) * sizeof(int64_t);
    size += tensor.int8_data.size();
    size += tensor.quantized_data.size();
    size += static_cast<uint64_t>(tensor.quantization_scales.size()) * sizeof(float);
    size += tensor.mxfp4_blocks.size();
    size += tensor.mxfp4_scales.size();
    if (tensor.mxfp4_file_storage)
    {
        size += tensor.mxfp4_file_storage->blocks_size;
        size += tensor.mxfp4_file_storage->scales_size;
        size += tensor.mxfp4_file_storage->secondary_blocks_size;
        size += tensor.mxfp4_file_storage->secondary_scales_size;
    }
    return size;
}

static uint64_t expert_weight_size(const WeightStore& weights, const ExpertPlan& expert)
{
    const uint32_t handles[] = {
        expert.gate_weight,
        expert.up_weight,
        expert.gate_up_weight,
        expert.down_weight,
        expert.gate_up_bias,
        expert.down_bias,
    };
    uint64_t size = 0;
    for (uint32_t handle : handles)
    {
        if (handle != invalid_tensor_handle)
            size += tensor_storage_size(weights.at(handle));
    }
    return size;
}

static Result<void> compile_shared_expert(
    const WeightStore& weights, const std::string& layer_name,
    const MoeModelDescriptor& descriptor, const MoeDescriptor& moe,
    ExpertPlan& shared)
{
    shared.activation = moe.activation;
    shared.activation_limit = moe.activation_limit;
    auto ret = assign_required_tensor(weights, layer_name + "shared_expert.gate.weight", {moe.intermediate_size, descriptor.hidden_size}, moe.shared_expert_weight_dtype, shared.gate_weight);
    if (!ret)
        return ret.error();
    ret = assign_required_tensor(weights, layer_name + "shared_expert.up.weight", {moe.intermediate_size, descriptor.hidden_size}, moe.shared_expert_weight_dtype, shared.up_weight);
    if (!ret)
        return ret.error();
    ret = assign_required_tensor(weights, layer_name + "shared_expert.down.weight", {descriptor.hidden_size, moe.intermediate_size}, moe.shared_expert_weight_dtype, shared.down_weight);
    if (!ret)
        return ret.error();
    shared.weight_size = expert_weight_size(weights, shared);
    return {};
}

static bool is_mapped_bfloat16_expert_tensor(const TensorData& tensor) noexcept
{
    if (tensor.dtype != DType::BFloat16
        || tensor.shape.size() != 2
        || !tensor.mapped_data)
    {
        return false;
    }
    const uint64_t elements = tensor.element_count();
    return elements != 0
           && elements <= std::numeric_limits<uint64_t>::max() / sizeof(uint16_t)
           && tensor.mapped_size == elements * sizeof(uint16_t)
           && tensor.bfloat16_values().size() == elements;
}

static bool is_cache_file_backed_expert_tensor(
    const TensorData& tensor,
    bool allow_bfloat16) noexcept
{
    return static_cast<bool>(tensor.mxfp4_file_storage)
           || (allow_bfloat16
               && is_mapped_bfloat16_expert_tensor(tensor));
}

static bool is_cache_file_backed_expert_pair(
    const TensorData& gate_up,
    const TensorData& down,
    bool allow_bfloat16) noexcept
{
    return is_cache_file_backed_expert_tensor(gate_up, allow_bfloat16)
           && is_cache_file_backed_expert_tensor(down, allow_bfloat16);
}

static Result<void> combine_qnk_gate_up_weights(
    WeightStore& weights,
    TensorHandle gate_handle,
    TensorHandle up_handle,
    uint32_t rows,
    uint32_t columns)
{
    if (gate_handle == invalid_tensor_handle || up_handle == invalid_tensor_handle)
        return Error{ErrorCode::InvalidArgument, "Qn_K gate/up handles cannot be invalid"};

    const TensorData& gate = weights.at(gate_handle);
    const TensorData& up = weights.at(up_handle);
    if (!is_qnk_dtype(gate.dtype)
        || gate.dtype != up.dtype
        || gate.shape != std::vector<uint32_t>{rows, columns}
        || up.shape != std::vector<uint32_t>{rows, columns}
        || !qnk_shape_supported(gate.dtype, rows, columns))
    {
        return Error{ErrorCode::InvalidModel, "Qn_K gate/up tensors cannot be packed"};
    }

    const uint64_t row_size = qnk_storage_bytes(gate.dtype, 1, columns);
    if (row_size == 0
        || rows > std::numeric_limits<uint32_t>::max() / 2
        || row_size > std::numeric_limits<size_t>::max() / rows
        || row_size * rows > std::numeric_limits<size_t>::max() / 2)
    {
        return Error{ErrorCode::InvalidModel, "Qn_K gate/up tensor size overflows"};
    }

    const size_t source_size = static_cast<size_t>(row_size * rows);
    const std::span<const uint8_t> gate_data = gate.qnk_values();
    const std::span<const uint8_t> up_data = up.qnk_values();
    if (gate_data.size() != source_size || up_data.size() != source_size)
    {
        return Error{ErrorCode::InvalidModel, "Qn_K gate/up tensor byte count does not match its shape"};
    }

    TensorData combined;
    combined.dtype = gate.dtype;
    combined.shape = {rows * 2, columns};
    combined.qnk_interleave_rows = false;
    combined.quantized_data.reserve(source_size * 2);
    combined.quantized_data.insert(combined.quantized_data.end(), gate_data.begin(), gate_data.end());
    combined.quantized_data.insert(combined.quantized_data.end(), up_data.begin(), up_data.end());
    weights.at_mutable(gate_handle) = std::move(combined);
    weights.at_mutable(up_handle) = TensorData{};
    return {};
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
    bool retain_cpu_dense_copies,
    bool use_file_backed_experts)
{
    if (compiled.descriptor.speculative_layer_count != 1
        || compiled.descriptor.speculative_block_size == 0
        || !compiled.descriptor.speculative_target_layer_ids.empty()
        || compiled.descriptor.hyper_connection_multiplier != 1
        || compiled.graph.layer_plans.empty()
        || compiled.graph.layer_plans.back().attention.kind != AttentionKind::Standard
        || compiled.graph.layer_plans.back().attention.sliding_window != 0)
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
    status = prepare_linear_operator(compiled.weights, compiled.operators, speculative.mtp_input_projection_weight,
                                     invalid_tensor_handle,
                                     dense_device,
                                     retain_cpu_dense_copies,
                                     compiled.opt.vulkan_device_index,
                                     compiled.vulkan_context_instance,
                                     compiled.opt.optimization_flags);
    if (!status)
        return status.error();

    const std::string layer_name = speculative_layer_prefix(0);
    CompiledLayerPlan layer_plan = compiled.graph.layer_plans.back();
    layer_plan.layer_id = static_cast<uint32_t>(compiled.descriptor.layers.size());
    layer_plan.vulkan_device_index = compiled.opt.vulkan_device_index;
    layer_plan.hyper_connection = {};
    AttentionBlockPlan& attention = layer_plan.attention;
    attention.vulkan_attention_operator = invalid_compiled_operator_handle;
    attention.fused_qkv_operator = invalid_compiled_operator_handle;
    attention.fused_qkv_bfloat16_operator = invalid_compiled_operator_handle;
    attention.fused_qkv_gate_bfloat16_operator = invalid_compiled_operator_handle;
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
        status = prepare_vulkan_qkv_operator(
            compiled, layer_plan, true,
            "failed to create fused Qwen MTP Vulkan QKV operator");
        if (!status)
            return status.error();
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
            status = prepare_linear_operator(compiled.weights, compiled.operators, handle,
                                             invalid_tensor_handle,
                                             dense_device,
                                             retain_cpu_dense_copies,
                                             layer_plan.vulkan_device_index,
                                             compiled.vulkan_context_instance,
                                             compiled.opt.optimization_flags);
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
            && attention.fused_qkv_gate_bfloat16_operator != invalid_compiled_operator_handle)
        {
            continue;
        }
        status = prepare_linear_operator(compiled.weights, compiled.operators, handle,
                                         invalid_tensor_handle,
                                         dense_device,
                                         retain_cpu_dense_copies,
                                         layer_plan.vulkan_device_index,
                                         compiled.vulkan_context_instance,
                                         compiled.opt.optimization_flags);
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
    status = prepare_linear_operator(compiled.weights, compiled.operators, compiled_moe.router_weight,
                                     invalid_tensor_handle,
                                     NcnnLinearDevice::Cpu,
                                     retain_cpu_dense_copies,
                                     layer_plan.vulkan_device_index,
                                     compiled.vulkan_context_instance,
                                     compiled.opt.optimization_flags);
    if (!status)
        return status.error();

    status = compile_shared_expert(
        compiled.weights, layer_name, compiled.descriptor, moe,
        compiled_moe.shared_expert);
    if (!status)
        return status.error();
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
        compiled.operators,
        compiled_moe,
        dense_device,
        retain_cpu_dense_copies,
        layer_plan.vulkan_device_index,
        compiled.vulkan_context_instance,
        compiled.opt.optimization_flags);
    if (!status)
        return status.error();

    compiled_moe.experts.reserve(moe.expert_count);
    for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
    {
        ExpertPlan expert;
        expert.activation = moe.activation;
        expert.layout = moe.layout;
        expert.activation_limit = moe.activation_limit;
        const std::string prefix = speculative_expert_prefix(0, expert_id);
        status = assign_required_tensor(
            compiled.weights,
            prefix + "gate_up.weight",
            {moe.intermediate_size * 2, hidden_size},
            moe.expert_weight_dtype,
            expert.gate_up_weight);
        if (!status)
            return status.error();
        if (is_qnk_dtype(compiled.weights.at(expert.gate_up_weight).dtype))
        {
            compiled.weights.at_mutable(expert.gate_up_weight).qnk_interleave_rows = moe.layout == ExpertLayout::InterleavedGateUpDown;
        }
        status = assign_required_tensor(
            compiled.weights,
            prefix + "down.weight",
            {hidden_size, moe.intermediate_size},
            moe.expert_weight_dtype,
            expert.down_weight);
        if (!status)
            return status.error();
        expert.weight_size = expert_weight_size(compiled.weights, expert);
        const TensorData& gate_up_weight = compiled.weights.at(expert.gate_up_weight);
        const TensorData& down_weight = compiled.weights.at(expert.down_weight);
        const bool use_file_backed_bfloat16 = use_file_backed_experts
                                              && moe.expert_weight_dtype == DType::BFloat16
                                              && has_flag(moe.flags, MoeDescriptorFileBackedExperts);
        if (is_cache_file_backed_expert_pair(
                gate_up_weight, down_weight, use_file_backed_bfloat16))
            expert.cache_key = ExpertCache::make_pair_key(gate_up_weight, down_weight);
        else if (is_qnk_dtype(gate_up_weight.dtype) && gate_up_weight.dtype == down_weight.dtype)
            expert.cache_key = "qnk:" + prefix;
        compiled_moe.experts.push_back(std::move(expert));
    }
    speculative.graph.layer_plans.push_back(std::move(layer_plan));
    return {};
}

static Result<void> compile_speculative_model(
    CompiledModel& compiled,
    NcnnLinearDevice dense_device,
    bool retain_cpu_dense_copies,
    bool use_file_backed_experts)
{
    if (compiled.descriptor.speculative_layer_count == 0)
        return {};
    if (compiled.descriptor.speculative_kind == SpeculativeModelKind::Mtp)
        return compile_mtp_speculative_model(
            compiled, dense_device, retain_cpu_dense_copies,
            use_file_backed_experts);
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
        status = prepare_linear_operator(compiled.weights, compiled.operators, handle,
                                         invalid_tensor_handle,
                                         dense_device,
                                         retain_cpu_dense_copies,
                                         compiled.opt.vulkan_device_index,
                                         compiled.vulkan_context_instance,
                                         compiled.opt.optimization_flags);
        if (!status)
            return status.error();
    }

    LayerDescriptor draft_layer = compiled.descriptor.layers.back();
    draft_layer.attention.compression_ratio = 0;
    draft_layer.ffn.moe.flags |= MoeDescriptorRouterBias;
    const uint32_t main_layer_count = static_cast<uint32_t>(compiled.descriptor.layers.size());
    speculative.graph.layer_plans.reserve(compiled.descriptor.speculative_layer_count);
    for (uint32_t layer_id = 0; layer_id < compiled.descriptor.speculative_layer_count; ++layer_id)
    {
        const std::string layer_name = speculative_layer_prefix(layer_id);
        const MoeDescriptor& moe = draft_layer.ffn.moe;
        CompiledLayerPlan layer_plan;
        layer_plan.layer_id = main_layer_count + layer_id;
        layer_plan.vulkan_device_index = compiled.opt.vulkan_device_index;
        layer_plan.moe.top_k = moe.top_k;
        layer_plan.moe.score_function = moe.score_function;
        layer_plan.moe.normalization = RouterNormalization::SelectedExperts;
        layer_plan.moe.routed_scaling_factor = moe.routed_scaling_factor;
        layer_plan.moe.has_shared_expert = true;

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
        status = prepare_latent_attention_operators(
            compiled, layer_plan, draft_layer.attention, dense_device,
            retain_cpu_dense_copies, "speculative ");
        if (!status)
            return status.error();

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
        status = prepare_linear_operator(compiled.weights, compiled.operators, layer_plan.moe.router_weight,
                                         invalid_tensor_handle,
                                         NcnnLinearDevice::Cpu,
                                         retain_cpu_dense_copies,
                                         layer_plan.vulkan_device_index,
                                         compiled.vulkan_context_instance,
                                         compiled.opt.optimization_flags);
        if (!status)
            return status.error();

        status = compile_shared_expert(
            compiled.weights, layer_name, compiled.descriptor, moe,
            layer_plan.moe.shared_expert);
        if (!status)
            return status.error();
        ExpertPlan& shared = layer_plan.moe.shared_expert;
        const TensorHandle shared_handles[] = {
            shared.gate_weight,
            shared.up_weight,
            shared.down_weight,
        };
        for (TensorHandle handle : shared_handles)
        {
            status = prepare_linear_operator(compiled.weights, compiled.operators, handle,
                                             invalid_tensor_handle,
                                             dense_device,
                                             retain_cpu_dense_copies,
                                             layer_plan.vulkan_device_index,
                                             compiled.vulkan_context_instance,
                                             compiled.opt.optimization_flags);
            if (!status)
                return status.error();
        }

        layer_plan.moe.experts.reserve(moe.expert_count);
        for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
        {
            ExpertPlan expert;
            expert.activation = moe.activation;
            expert.layout = ExpertLayout::InterleavedGateUpDown;
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
            expert.weight_size = expert_weight_size(compiled.weights, expert);
            const TensorData& gate_up_weight = compiled.weights.at(expert.gate_up_weight);
            const TensorData& down_weight = compiled.weights.at(expert.down_weight);
            const bool use_file_backed_bfloat16 = use_file_backed_experts
                                                  && moe.expert_weight_dtype == DType::BFloat16
                                                  && has_flag(moe.flags, MoeDescriptorFileBackedExperts);
            if (is_cache_file_backed_expert_pair(
                    gate_up_weight, down_weight, use_file_backed_bfloat16))
                expert.cache_key = ExpertCache::make_pair_key(gate_up_weight, down_weight);
            layer_plan.moe.experts.push_back(std::move(expert));
        }
        speculative.graph.layer_plans.push_back(std::move(layer_plan));
    }
    return {};
}

static Result<void> compile_moe_layer(
    CompiledModel& compiled, const std::string& layer_name,
    const LayerDescriptor& layer, CompiledLayerPlan& layer_plan,
    NcnnLinearDevice dense_device, bool retain_cpu_dense_copies,
    bool use_file_backed_experts)
{
    const MoeDescriptor& moe = layer.ffn.moe;
    if (layer.pre_ffn_norm == NormType::RmsNorm)
    {
        auto ret = assign_required_tensor(compiled.weights, layer_name + "pre_ffn_norm.weight", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype, layer_plan.moe.pre_ffn_norm_weight);
        if (!ret)
            return ret.error();
    }

    auto ret = assign_required_tensor(compiled.weights, layer_name + "router.weight", {moe.expert_count, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype, layer_plan.moe.router_weight);
    if (!ret)
        return ret.error();

    if (has_flag(moe.flags, MoeDescriptorRouterBias))
    {
        if (moe.score_function == RouterScoreFunction::Softmax)
        {
            ret = assign_required_tensor(compiled.weights, layer_name + "router.bias", {moe.expert_count}, compiled.descriptor.activation_dtype, layer_plan.moe.router_bias);
        }
        else
        {
            ret = assign_required_tensor(compiled.weights, layer_name + "router.selection_bias", {moe.expert_count}, DType::Float32, layer_plan.moe.router_selection_bias);
        }
        if (!ret)
            return ret.error();
    }
    if (layer_plan.layer_id < compiled.descriptor.hash_routing_layer_count)
    {
        ret = assign_required_tensor(compiled.weights, layer_name + "router.token_experts", {compiled.descriptor.vocabulary_size, moe.top_k}, DType::Int64, layer_plan.moe.token_experts);
        if (!ret)
            return ret.error();
    }
    // The CPU router avoids a Vulkan round trip before Expert dispatch.
    ret = prepare_linear_operator(
        compiled.weights, compiled.operators, layer_plan.moe.router_weight,
        layer_plan.moe.router_bias, NcnnLinearDevice::Cpu,
        retain_cpu_dense_copies, layer_plan.vulkan_device_index,
        compiled.vulkan_context_instance, compiled.opt.optimization_flags);
    if (!ret)
        return ret.error();

    if (layer_plan.moe.has_shared_expert)
    {
        if (moe.shared_expert_count != 1)
            return Error{ErrorCode::UnsupportedModel, "the CPU runtime supports one shared Expert per MoE block"};
        ret = compile_shared_expert(
            compiled.weights, layer_name, compiled.descriptor, moe,
            layer_plan.moe.shared_expert);
        if (!ret)
            return ret.error();
        if (has_flag(moe.flags, MoeDescriptorSharedExpertGate))
        {
            ret = assign_required_tensor(compiled.weights, layer_name + "shared_expert.router_gate.weight", {1, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype, layer_plan.moe.shared_expert_gate_weight);
            if (!ret)
                return ret.error();
        }
        ret = prepare_shared_expert_operators(
            compiled.weights,
            compiled.operators,
            layer_plan.moe,
            dense_device,
            retain_cpu_dense_copies,
            layer_plan.vulkan_device_index,
            compiled.vulkan_context_instance,
            compiled.opt.optimization_flags);
        if (!ret)
            return ret.error();
    }

    layer_plan.moe.experts.reserve(moe.expert_count);
    const bool prepare_routed_dense_operators = moe.expert_weight_dtype != DType::BFloat16
                                                || moe.expert_count <= 64;
    const bool fuse_qnk_gate_up = has_flag(
        compiled.opt.optimization_flags,
        OptimizationVulkanQnK);
    for (uint32_t expert_id = 0; expert_id < moe.expert_count; ++expert_id)
    {
        ExpertPlan expert;
        expert.activation = moe.activation;
        expert.layout = moe.layout;
        expert.activation_limit = moe.activation_limit;
        const std::string prefix = expert_prefix(layer_plan.layer_id, expert_id);

        if (expert.layout == ExpertLayout::InterleavedGateUpDown || expert.layout == ExpertLayout::PackedGateUpDown)
        {
            ret = assign_required_tensor(compiled.weights, prefix + "gate_up.weight", {moe.intermediate_size * 2, compiled.descriptor.hidden_size}, moe.expert_weight_dtype, expert.gate_up_weight);
            if (!ret)
                return ret.error();
            if (is_qnk_dtype(compiled.weights.at(expert.gate_up_weight).dtype))
            {
                compiled.weights.at_mutable(expert.gate_up_weight).qnk_interleave_rows = expert.layout == ExpertLayout::InterleavedGateUpDown;
            }
        }
        else if (expert.layout != ExpertLayout::UpDown)
        {
            ret = assign_required_tensor(compiled.weights, prefix + "gate.weight", {moe.intermediate_size, compiled.descriptor.hidden_size}, moe.expert_weight_dtype, expert.gate_weight);
            if (!ret)
                return ret.error();
        }

        if (expert.layout != ExpertLayout::InterleavedGateUpDown && expert.layout != ExpertLayout::PackedGateUpDown)
        {
            ret = assign_required_tensor(compiled.weights, prefix + "up.weight", {moe.intermediate_size, compiled.descriptor.hidden_size}, moe.expert_weight_dtype, expert.up_weight);
            if (!ret)
                return ret.error();
        }

        bool qnk_gate_up_fused = false;
        if (fuse_qnk_gate_up
            && expert.layout == ExpertLayout::GateUpDown
            && expert.gate_weight != invalid_tensor_handle
            && expert.up_weight != invalid_tensor_handle)
        {
            auto fused = combine_qnk_gate_up_weights(
                compiled.weights,
                expert.gate_weight,
                expert.up_weight,
                moe.intermediate_size,
                compiled.descriptor.hidden_size);
            if (fused)
            {
                expert.gate_up_weight = expert.gate_weight;
                expert.gate_weight = invalid_tensor_handle;
                expert.up_weight = invalid_tensor_handle;
                expert.layout = ExpertLayout::PackedGateUpDown;
                qnk_gate_up_fused = true;
            }
            else if (is_qnk_dtype(compiled.weights.at(expert.gate_weight).dtype)
                     && is_qnk_dtype(compiled.weights.at(expert.up_weight).dtype))
            {
                return fused.error();
            }
        }

        if (prepare_routed_dense_operators && !qnk_gate_up_fused)
        {
            if (expert.gate_weight != invalid_tensor_handle)
                (void)prepare_linear_operator(compiled.weights, compiled.operators, expert.gate_weight, invalid_tensor_handle, NcnnLinearDevice::Cpu,
                                              retain_cpu_dense_copies, layer_plan.vulkan_device_index,
                                              compiled.vulkan_context_instance, compiled.opt.optimization_flags);
            if (expert.up_weight != invalid_tensor_handle)
                (void)prepare_linear_operator(compiled.weights, compiled.operators, expert.up_weight, invalid_tensor_handle, NcnnLinearDevice::Cpu,
                                              retain_cpu_dense_copies, layer_plan.vulkan_device_index,
                                              compiled.vulkan_context_instance, compiled.opt.optimization_flags);
        }

        ret = assign_required_tensor(compiled.weights, prefix + "down.weight", {compiled.descriptor.hidden_size, moe.intermediate_size}, moe.expert_weight_dtype, expert.down_weight);
        if (!ret)
            return ret.error();

        if (has_flag(moe.flags, MoeDescriptorProjectionBias))
        {
            ret = assign_required_tensor(compiled.weights, prefix + "gate_up.bias", {moe.intermediate_size * 2}, compiled.descriptor.activation_dtype, expert.gate_up_bias);
            if (!ret)
                return ret.error();
            ret = assign_required_tensor(compiled.weights, prefix + "down.bias", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype, expert.down_bias);
            if (!ret)
                return ret.error();
        }
        if (prepare_routed_dense_operators)
            (void)prepare_linear_operator(compiled.weights, compiled.operators, expert.down_weight, expert.down_bias, NcnnLinearDevice::Cpu,
                                          retain_cpu_dense_copies, layer_plan.vulkan_device_index,
                                          compiled.vulkan_context_instance, compiled.opt.optimization_flags);

        expert.weight_size = expert_weight_size(compiled.weights, expert);
        const TensorData& down_weight = compiled.weights.at(expert.down_weight);
        const bool use_file_backed_bfloat16 = use_file_backed_experts
                                              && moe.expert_weight_dtype == DType::BFloat16
                                              && has_flag(moe.flags, MoeDescriptorFileBackedExperts);
        if (expert.gate_up_weight != invalid_tensor_handle)
        {
            const TensorData& gate_up_weight = compiled.weights.at(expert.gate_up_weight);
            if (use_file_backed_bfloat16
                && !is_cache_file_backed_expert_pair(
                    gate_up_weight, down_weight, true))
            {
                return Error{
                    ErrorCode::InvalidModel,
                    "file-backed BF16 Expert weights must remain memory-mapped"};
            }
            if (is_cache_file_backed_expert_pair(
                    gate_up_weight, down_weight, use_file_backed_bfloat16))
            {
                expert.cache_key = ExpertCache::make_pair_key(gate_up_weight, down_weight);
            }
            else if (is_qnk_dtype(gate_up_weight.dtype) && gate_up_weight.dtype == down_weight.dtype)
            {
                expert.cache_key = "qnk:" + prefix;
            }
        }
        layer_plan.moe.experts.push_back(expert);
    }
    return {};
}

Result<CompiledModel> compile_model(MoeModelDescriptor descriptor, WeightMapping mapping, HybridMode hybrid_mode)
{
    CompilerOption opt;
    if (hybrid_mode == HybridMode::HybridExperts)
    {
        opt.flags |= BackendVulkanDense | BackendVulkanAttention;
    }
    return compile_model(std::move(descriptor), std::move(mapping), hybrid_mode, opt);
}

Result<CompiledModel> compile_model(MoeModelDescriptor descriptor, WeightMapping mapping, HybridMode hybrid_mode, const CompilerOption& opt)
{
    if (!has_flag(opt.flags, BackendCpuExecution))
        return Error{ErrorCode::UnsupportedModel, "compiler requires a CPU execution backend"};
    auto valid = validate_model_descriptor(descriptor);
    if (!valid)
        return valid.error();
    const uint32_t layer_count = static_cast<uint32_t>(descriptor.layers.size());
    if (descriptor.activation_dtype != DType::Float32 && descriptor.activation_dtype != DType::BFloat16)
        return Error{ErrorCode::UnsupportedModel, "dense weights must use float32 or bfloat16"};
    if (descriptor.kv_cache_dtype != DType::Float32 && descriptor.kv_cache_dtype != DType::BFloat16)
        return Error{ErrorCode::UnsupportedModel, "KV cache must use float32 or bfloat16"};
    if (descriptor.norm_epsilon <= 0.0f)
        return Error{ErrorCode::InvalidModel, "norm_epsilon must be positive"};
    for (const LayerDescriptor& layer : descriptor.layers)
    {
        if (layer.ffn.kind == FfnKind::Dense)
        {
            return Error{ErrorCode::UnsupportedModel, "dense FFN decoder layers are not yet executable"};
        }
        const MoeDescriptor& moe = layer.ffn.moe;
        if (moe.router_group_count != 0 || moe.router_top_k_groups != 0)
        {
            return Error{ErrorCode::UnsupportedModel, "group-limited routing is not yet executable"};
        }
        if (!std::isfinite(moe.routed_scaling_factor) || moe.routed_scaling_factor <= 0.0f)
            return Error{ErrorCode::InvalidModel, "routed scaling factor must be finite and positive"};
    }

    CompiledModel compiled;
    compiled.descriptor = std::move(descriptor);
    compiled.opt.optimization_flags = opt.optimization_flags;
    compiled.opt.num_concurrent_sessions = std::max(1u, opt.num_concurrent_sessions);
    compiled.vulkan_context_instance = opt.vkctx;
    const bool hybrid_requests_vulkan = hybrid_mode == HybridMode::HybridExperts;
    const bool use_vulkan_dense = hybrid_requests_vulkan && has_flag(opt.flags, BackendVulkanDense);
    const bool retain_cpu_dense_copies = has_flag(opt.flags, BackendRetainCpuDenseCopies);
    const bool use_file_backed_experts = has_flag(
        opt.flags, BackendFileBackedExperts);
    std::vector<uint32_t> dense_device_indices = opt.device_indices;
    if (dense_device_indices.empty() && opt.device_index != automatic_vulkan_device_index)
    {
        dense_device_indices.push_back(opt.device_index);
    }
    if (use_vulkan_dense && dense_device_indices.empty())
    {
        return Error{ErrorCode::InvalidArgument, "Vulkan dense execution requires at least one device"};
    }
    if (use_vulkan_dense && !compiled.vulkan_context_instance)
    {
        return Error{ErrorCode::InvalidArgument, "Vulkan dense execution requires a context instance"};
    }
    std::vector<uint32_t> dense_device_scores = opt.device_scores;
    if (dense_device_scores.size() != dense_device_indices.size())
    {
        dense_device_scores.assign(dense_device_indices.size(), 1);
    }
    for (uint32_t& score : dense_device_scores)
        score = std::max(1u, score);
    std::vector<uint32_t> device_layer_counts(dense_device_indices.size(), 0);
    if (use_vulkan_dense)
    {
        if (compiled.opt.num_concurrent_sessions <= 1 || dense_device_indices.size() == 1)
        {
            const size_t fastest = static_cast<size_t>(std::distance(dense_device_scores.begin(), std::max_element(dense_device_scores.begin(), dense_device_scores.end())));
            device_layer_counts[fastest] = layer_count;
        }
        else
        {
            for (uint32_t layer = 0; layer < layer_count; ++layer)
            {
                size_t selected = 0;
                double selected_cost = placement_cost(device_layer_counts, dense_device_scores, compiled.opt.num_concurrent_sessions, 0);
                for (size_t index = 1; index < dense_device_indices.size(); ++index)
                {
                    const double cost = placement_cost(device_layer_counts, dense_device_scores, compiled.opt.num_concurrent_sessions, index);
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
    compiled.opt.hybrid_mode = use_vulkan_dense ? hybrid_mode : HybridMode::CpuOnly;
    compiled.opt.vulkan_device_index = use_vulkan_dense ? dense_device_indices.front() : automatic_vulkan_device_index;
    if (use_vulkan_dense)
    {
        for (size_t index = 0; index < dense_device_indices.size(); ++index)
        {
            if (device_layer_counts[index] != 0 || dense_device_indices[index] == compiled.opt.vulkan_device_index)
            {
                compiled.opt.vulkan_device_indices.push_back(dense_device_indices[index]);
            }
        }
    }
    const NcnnLinearDevice dense_device = use_vulkan_dense ? NcnnLinearDevice::Vulkan : NcnnLinearDevice::Cpu;
    const bool vulkan_delta_fusion_available = use_vulkan_dense
                                               && has_flag(opt.flags, BackendVulkanAttention)
                                               && has_flag(
                                                   compiled.opt.optimization_flags,
                                                   OptimizationVulkanAttention);
    const bool protect_file_backed_experts = has_flag(
                                                 opt.flags, BackendVulkanExperts)
                                             && use_file_backed_experts;
    const uint64_t gated_delta_gpu_budget = gated_delta_vulkan_budget_size(
        opt,
        vulkan_delta_fusion_available,
        protect_file_backed_experts);
    uint64_t planned_gated_delta_gpu_size = 0;

    for (auto& [name, tensor] : mapping)
    {
        auto added = compiled.weights.add(name, std::move(tensor));
        if (!added)
            return added.error();
    }
    compiled.operators.bind_weight_count(compiled.weights.size());

    auto embedding = require_tensor(compiled.weights, "token_embedding.weight", {compiled.descriptor.vocabulary_size, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
    if (!embedding)
        return embedding.error();
    compiled.token_embedding = embedding.value();

    if (compiled.descriptor.final_norm == NormType::RmsNorm)
    {
        auto final_norm = require_tensor(compiled.weights, "final_norm.weight", {compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
        if (!final_norm)
            return final_norm.error();
        compiled.final_norm_weight = final_norm.value();
    }
    else if (compiled.descriptor.final_norm != NormType::None)
    {
        return Error{ErrorCode::UnsupportedModel, "unsupported final normalization"};
    }

    auto lm_head = require_tensor(compiled.weights, "lm_head.weight", {compiled.descriptor.vocabulary_size, compiled.descriptor.hidden_size}, compiled.descriptor.activation_dtype);
    if (!lm_head)
        return lm_head.error();
    compiled.lm_head_weight = lm_head.value();
    auto prepared = prepare_lm_head_operator(compiled, dense_device, retain_cpu_dense_copies);
    if (!prepared)
        return prepared.error();
    if (compiled.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
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
    else if (compiled.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
    {
        if (compiled.descriptor.hyper_connection_multiplier <= 1
            || compiled.descriptor.hyper_connection_low_rank == 0)
        {
            return Error{ErrorCode::InvalidModel, "invalid gated-residual configuration"};
        }
        auto status = compile_gated_residual_plan(
            compiled.weights, "gated_residual.head.", compiled.descriptor,
            false, compiled.gated_residual_head);
        if (!status)
            return status.error();
    }

    compiled.graph.layer_plans.reserve(layer_count);
    for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id)
    {
        const LayerDescriptor& layer = compiled.descriptor.layers[layer_id];
        const MoeDescriptor& moe = layer.ffn.moe;
        if (layer.pre_ffn_norm != NormType::RmsNorm
            && !(compiled.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual
                 && layer.pre_ffn_norm == NormType::None))
            return Error{ErrorCode::UnsupportedModel, "the reference runtime requires RMSNorm before each MoE block"};
        if (has_flag(moe.flags, MoeDescriptorSharedExpertGate)
            && moe.shared_expert_count == 0)
        {
            return Error{ErrorCode::InvalidModel, "shared Expert gate requires a shared Expert"};
        }
        if (moe.expert_weight_dtype != DType::Float32
            && moe.expert_weight_dtype != DType::BFloat16
            && moe.expert_weight_dtype != DType::Int8
            && moe.expert_weight_dtype != DType::MxFp4
            && !is_qnk_dtype(moe.expert_weight_dtype))
            return Error{ErrorCode::UnsupportedModel, "expert weights must use float32, bfloat16, int8, MXFP4, or Qn_K"};
        if (moe.expert_weight_dtype == DType::MxFp4 && !has_flag(opt.flags, BackendMxfp4CpuKernel))
        {
            return Error{ErrorCode::UnsupportedModel, "backend capabilities do not provide an MXFP4 CPU expert kernel"};
        }

        CompiledLayerPlan layer_plan;
        layer_plan.layer_id = layer_id;
        layer_plan.vulkan_device_index = choose_layer_device(use_vulkan_dense, dense_device_indices, device_layer_counts, smooth_device_scores, total_device_score);
        const bool use_vulkan_attention = layer.attention.kind == AttentionKind::Standard
                                          && !has_flag(layer.attention.flags, AttentionDescriptorQsa)
                                          && compiled.descriptor.hyper_connection_kind != HyperConnectionKind::GatedResidual
                                          && use_vulkan_dense
                                          && has_flag(opt.flags, BackendVulkanAttention);
        const bool use_vulkan_latent_linear = layer.attention.kind == AttentionKind::MultiHeadLatent
                                              && use_vulkan_dense;
        const uint64_t gated_delta_layer_size = gated_delta_vulkan_working_set_size(
            layer.attention,
            compiled.descriptor);
        const bool use_vulkan_delta_linear = vulkan_delta_fusion_available
                                             && gated_delta_layer_size != 0
                                             && gated_delta_layer_size <= gated_delta_gpu_budget
                                             && planned_gated_delta_gpu_size
                                                    <= gated_delta_gpu_budget - gated_delta_layer_size;
        if (use_vulkan_delta_linear)
            planned_gated_delta_gpu_size += gated_delta_layer_size;
        layer_plan.moe.top_k = moe.top_k;
        layer_plan.moe.score_function = moe.score_function;
        layer_plan.moe.normalization = moe.normalization;
        layer_plan.moe.routed_scaling_factor = moe.routed_scaling_factor;
        layer_plan.moe.has_shared_expert = moe.shared_expert_count != 0;

        const std::string layer_name = layer_prefix(layer_id);
        if (compiled.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
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
        else if (compiled.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual)
        {
            auto status = compile_gated_residual_plan(
                compiled.weights, layer_name + "gated_residual.attention.",
                compiled.descriptor, true,
                layer_plan.attention_gated_residual);
            if (!status)
                return status.error();
            status = compile_gated_residual_plan(
                compiled.weights, layer_name + "gated_residual.ffn.",
                compiled.descriptor, true,
                layer_plan.ffn_gated_residual);
            if (!status)
                return status.error();
        }
        if (layer.attention.kind != AttentionKind::None)
        {
            const AttentionDescriptor& attention = layer.attention;
            const NcnnLinearDevice attention_device = use_vulkan_attention || use_vulkan_latent_linear || use_vulkan_delta_linear
                                                          ? NcnnLinearDevice::Vulkan
                                                          : NcnnLinearDevice::Cpu;
            if (layer.pre_attention_norm != NormType::RmsNorm
                && !(compiled.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual
                     && layer.pre_attention_norm == NormType::None))
                return Error{ErrorCode::UnsupportedModel, "attention requires a pre-attention RMSNorm"};
            AttentionBlockPlan& plan = layer_plan.attention;
            if (attention.kind == AttentionKind::MultiHeadLatent)
            {
                auto status = compile_latent_attention(compiled.weights, layer_name, compiled.descriptor, attention, plan);
                if (!status)
                    return status.error();
                prepared = prepare_latent_attention_operators(
                    compiled, layer_plan, attention, attention_device,
                    retain_cpu_dense_copies);
                if (!prepared)
                    return prepared.error();
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
                prepared = prepare_gated_delta_attention_operators(
                    compiled, layer_plan, attention_device,
                    retain_cpu_dense_copies);
                if (!prepared)
                    return prepared.error();
            }
            else
            {
                auto status = compile_standard_attention(
                    compiled.weights, layer_name, compiled.descriptor, layer, plan);
                if (!status)
                    return status.error();
                prepared = prepare_standard_attention_operators(
                    compiled, layer_plan, attention_device, retain_cpu_dense_copies);
                if (!prepared)
                    return prepared.error();
            }
        }

        auto ple_status = compile_ple_plan(
            compiled, layer_name, layer.ple, retain_cpu_dense_copies,
            layer_plan.vulkan_device_index, layer_plan.ple);
        if (!ple_status)
            return ple_status.error();

        auto ret = compile_moe_layer(
            compiled, layer_name, layer, layer_plan, dense_device,
            retain_cpu_dense_copies, use_file_backed_experts);
        if (!ret)
            return ret.error();

        compiled.graph.layer_plans.push_back(std::move(layer_plan));
    }

    auto speculative = compile_speculative_model(
        compiled, dense_device, retain_cpu_dense_copies,
        use_file_backed_experts);
    if (!speculative)
        return speculative.error();
    auto graph = build_graph(
        compiled, has_flag(opt.flags, BackendVulkanExperts));
    if (!graph)
        return graph.error();
    if (has_flag(opt.flags, BackendReleaseVulkanDenseHostStorage))
        release_vulkan_dense_host_copies(compiled);
    return compiled;
}

} // namespace moe
} // namespace ncnn
