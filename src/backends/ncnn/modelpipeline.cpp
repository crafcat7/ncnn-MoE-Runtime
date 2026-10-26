#include "modelpipeline.h"

#include "kernels/qnk.h"
#include "kernels/bfloat16.h"
#include "linear.h"
#include "attention_vulkan.h"
#include "latentattention_vulkan.h"
#include "latentlayer_vulkan.h"
#include "hyperconnection_vulkan.h"
#include "layerhead_vulkan.h"
#include "rmsnorm_vulkan.h"
#include "router_vulkan.h"
#include "gateddeltanet_vulkan.h"
#include "qsaattention_vulkan.h"
#include "qwenlayer_vulkan.h"
#include "expertbackend_vulkan.h"
#include "graph/compiler.h"
#include "engine/expert.h"
#include "graph/compiledmodel.h"
#include "ncnn/moe/session.h"
#include "storage/weightstore.h"
#include "vulkancontext.h"

#if NCNN_MOE_WITH_VULKAN
#include <gpu.h>
#endif

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static Result<void> prepare_vulkan_linear_operator(WeightStore& weights,
                                                   CompiledOperatorTable& operators,
                                                   TensorHandle matrix_handle,
                                                   TensorHandle bias_handle,
                                                   uint32_t vulkan_device_index,
                                                   const VulkanRuntimePtr& vulkan_runtime,
                                                   uint64_t optimization_flags,
                                                   uint32_t input_group_count = 1,
                                                   bool prefer_bfloat16_vulkan = true,
                                                   bool allow_float32_fallback = true)
{
    TensorData& matrix = weights.at_mutable(matrix_handle);
    CompiledOperator& compiled_operator = operators.at_weight_mutable(matrix_handle);
    const TensorData* bias = bias_handle == invalid_tensor_handle ? nullptr : &weights.at(bias_handle);
    if (matrix.dtype == DType::Float8E4M3)
    {
        compiled_operator.float8 = Float8Linear_vulkan::create(matrix,
                                                               bias,
                                                               input_group_count,
                                                               vulkan_device_index,
                                                               vulkan_runtime,
                                                               optimization_flags);
        if (!compiled_operator.float8)
            return Error{ErrorCode::InternalError, "failed to create Vulkan FP8 Linear operator"};
        return {};
    }
    if (matrix.dtype == DType::Int4 || is_qnk_dtype(matrix.dtype)
        || (input_group_count > 1 && (matrix.dtype == DType::BFloat16 || matrix.dtype == DType::Float32)))
    {
        if (matrix.dtype == DType::Int4 || !is_qnk_dtype(matrix.dtype)
            || has_flag(optimization_flags, OptimizationVulkanQnK))
        {
            compiled_operator.qnk = QnkLinear_vulkan::create(matrix,
                                                             bias,
                                                             vulkan_device_index,
                                                             vulkan_runtime,
                                                             optimization_flags, input_group_count);
            if (!compiled_operator.qnk)
                return Error{ErrorCode::InternalError, "failed to create Vulkan Qn_K Linear operator"};
        }
        return {};
    }
    if (matrix.dtype != DType::Float32 && matrix.dtype != DType::BFloat16)
        return {};
    if (matrix.dtype == DType::BFloat16
        && prefer_bfloat16_vulkan)
    {
        compiled_operator.bfloat16 = Bfloat16Linear_vulkan::create(matrix,
                                                                   bias,
                                                                   vulkan_device_index,
                                                                   vulkan_runtime,
                                                                   optimization_flags);
        if (compiled_operator.bfloat16)
            return {};
        if (!allow_float32_fallback)
            return Error{ErrorCode::InternalError, "failed to create Vulkan BF16 Linear operator"};
    }
    compiled_operator.linear = Linear::create(matrix,
                                              bias,
                                              vulkan_device_index,
                                              vulkan_runtime,
                                              optimization_flags);
    if (!compiled_operator.linear)
        return Error{ErrorCode::InternalError, "failed to create Vulkan InnerProduct operator"};
    return {};
}

static Result<void> prepare_lm_head_operator(CompiledModel& compiled, bool allow_float32_fallback = true,
                                             bool prefer_bfloat16_vulkan = true)
{
    auto prepared = prepare_vulkan_linear_operator(compiled.weights, compiled.operators, compiled.lm_head_weight, invalid_tensor_handle,
                                                   compiled.opt.vulkan_device_index, compiled.vulkan_runtime,
                                                   compiled.opt.optimization_flags, 1, prefer_bfloat16_vulkan, allow_float32_fallback);
    if (!prepared)
        return prepared.error();
    const CompiledOperator& lm_head_operator = compiled.operators.at_weight(compiled.lm_head_weight);
    if (lm_head_operator.bfloat16
        && compiled.final_norm_weight != invalid_tensor_handle)
    {
        (void)lm_head_operator.bfloat16->prepare_rms_norm(compiled.weights.at(compiled.final_norm_weight),
                                                          compiled.descriptor.norm_epsilon,
                                                          compiled.descriptor.norm_weight_offset);
    }
    return {};
}

bool support_vulkan_experts(const WeightStore& weights,
                            const MoeBlockPlan& moe,
                            uint64_t optimization_flags) noexcept
{
    for (const ExpertPlan& expert : moe.experts)
    {
        const TensorHandle gate = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight;
        if (gate == invalid_tensor_handle || expert.down_weight == invalid_tensor_handle)
            continue;
        const TensorData* up = expert.gate_up_weight == invalid_tensor_handle && expert.up_weight != invalid_tensor_handle
                                   ? &weights.at(expert.up_weight)
                                   : nullptr;
        if (support_vulkan_expert(expert, weights.at(gate), weights.at(expert.down_weight), optimization_flags, up))
            return true;
    }
    return false;
}

bool support_vulkan_ffn(const CompiledOperatorTable& operators, const ExpertPlan& ffn) noexcept
{
    const CompiledOperator& fused = operators.at(ffn.vulkan_operator);
    if (fused.qnk_expert)
        return true;
    const auto prepared = [](const CompiledOperator& op) { return op.bfloat16 || op.float8 || op.qnk || op.linear; };
    if (fused.bfloat16 && prepared(operators.at_weight(ffn.down_weight)))
        return true;
    return ffn.gate_weight != invalid_tensor_handle && ffn.up_weight != invalid_tensor_handle
           && ffn.down_weight != invalid_tensor_handle
           && prepared(operators.at_weight(ffn.gate_weight))
           && prepared(operators.at_weight(ffn.up_weight))
           && prepared(operators.at_weight(ffn.down_weight));
}

bool support_vulkan_dense_ffn(const CompiledOperatorTable& operators, const ExpertPlan& ffn) noexcept
{
    if (support_vulkan_ffn(operators, ffn))
        return true;
    for (TensorHandle handle : {ffn.gate_weight, ffn.up_weight, ffn.down_weight})
    {
        const CompiledOperator& op = operators.at_weight(handle);
        if (op.bfloat16 || op.float8 || op.qnk || op.linear)
            return true;
    }
    return false;
}

bool support_vulkan_shared_experts(const CompiledOperatorTable& operators,
                                   const MoeBlockPlan& moe) noexcept
{
    if (moe.fused_shared_input_bfloat16_operator != invalid_compiled_operator_handle
        && operators.at(moe.fused_shared_input_bfloat16_operator).bfloat16)
        return true;
    const ExpertPlan& shared = moe.shared_expert;
    if (support_vulkan_ffn(operators, shared))
        return true;
    return moe.shared_expert_gate_weight == invalid_tensor_handle
           && shared.activation == ExpertActivation::DeepSeekSwiGlu
           && operators.at_weight(shared.gate_weight).float8
           && operators.at_weight(shared.up_weight).float8
           && operators.at_weight(shared.down_weight).float8;
}

static uint64_t saturating_add_u64(uint64_t left, uint64_t right) noexcept
{
    return right > std::numeric_limits<uint64_t>::max() - left
               ? std::numeric_limits<uint64_t>::max()
               : left + right;
}

static uint64_t saturating_multiply_u64(uint64_t left, uint64_t right) noexcept
{
    return left != 0 && right > std::numeric_limits<uint64_t>::max() / left
               ? std::numeric_limits<uint64_t>::max()
               : left * right;
}

static uint64_t gated_delta_vulkan_working_set_size(const CompiledModel& compiled,
                                                    const CompiledLayerPlan& layer)
{
    const MoeModelDescriptor& descriptor = compiled.descriptor;
    const AttentionDescriptor& attention = descriptor.layers[layer.layer_id].attention;
    if (attention.kind != AttentionKind::GatedDeltaNet
        || descriptor.activation_dtype != DType::BFloat16
        || attention.head_count == 0
        || attention.kv_head_count == 0
        || attention.head_dimension == 0
        || attention.value_head_dimension == 0
        || attention.convolution_kernel_size == 0
        || attention.head_count % attention.kv_head_count != 0)
    {
        return 0;
    }

    const uint64_t key_size = saturating_multiply_u64(attention.kv_head_count, attention.head_dimension);
    const uint64_t value_size = saturating_multiply_u64(attention.head_count, attention.value_head_dimension);
    const uint64_t convolution_size = saturating_add_u64(saturating_multiply_u64(key_size, 2), value_size);
    const uint64_t fused_columns = saturating_add_u64(saturating_add_u64(convolution_size, value_size),
                                                      saturating_multiply_u64(attention.head_count, 2));

    // Bfloat16Linear_vulkan stores BF16 matrix weights and allocates
    // FP32 bias/input-output metadata for each projection.  The GDN operator
    // additionally uploads convolution and per-head constants as FP32.
    const uint64_t projection_elements = saturating_multiply_u64(saturating_add_u64(fused_columns, value_size), descriptor.hidden_size);
    uint64_t projection_size = saturating_multiply_u64(projection_elements, sizeof(uint16_t));
    uint64_t projection_metadata_size = saturating_multiply_u64(saturating_add_u64(saturating_add_u64(saturating_multiply_u64(descriptor.hidden_size, 2),
                                                                                                      fused_columns),
                                                                                   value_size),
                                                                sizeof(float));
    const TensorHandle handles[] = {layer.attention.delta_qkv_weight, layer.attention.delta_z_weight,
                                    layer.attention.delta_beta_weight, layer.attention.delta_alpha_weight,
                                    layer.attention.output_weight};
    const bool quantized = std::any_of(std::begin(handles), std::end(handles), [&](TensorHandle handle) {
        const DType dtype = compiled.weights.at(handle).dtype;
        return dtype == DType::Int4 || is_qnk_dtype(dtype);
    });
    if (quantized)
    {
#if NCNN_MOE_WITH_VULKAN
        auto context = VulkanContext::acquire(layer.vulkan_device_index, compiled.vulkan_runtime,
                                              compiled.opt.optimization_flags);
        if (!context)
            return 0;
        const uint64_t alignment = std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
        projection_size = 0;
        projection_metadata_size = 0;
        for (TensorHandle handle : handles)
            projection_size = saturating_add_u64(projection_size, QnkLinear_vulkan::storage_bytes(compiled.weights.at(handle), alignment));
#else
        return 0;
#endif
    }
    const uint64_t recurrent_constant_elements = saturating_add_u64(saturating_multiply_u64(convolution_size, attention.convolution_kernel_size),
                                                                    saturating_add_u64(saturating_multiply_u64(attention.head_count, 2),
                                                                                       attention.value_head_dimension));
    const uint64_t recurrent_constant_size = saturating_multiply_u64(recurrent_constant_elements, sizeof(float));
    const uint64_t allocated_size = saturating_add_u64(saturating_add_u64(projection_size, projection_metadata_size),
                                                       recurrent_constant_size);

    // Weight allocators and the first dispatch need transient workspace in
    // addition to the raw payload.  A two-times safety margin is deliberately
    // conservative and keeps this policy independent of a model's layer id or
    // tensor naming convention.
    return saturating_multiply_u64(allocated_size, 2);
}

static bool uses_vulkan_dense_operator(const CompiledOperator& executable) noexcept
{
    return executable.bfloat16
           || executable.qnk
           || executable.float8
           || static_cast<bool>(executable.linear);
}

bool support_vulkan_attention(const CompiledOperatorTable& operators,
                              const AttentionBlockPlan& attention) noexcept
{
    if (attention.kind == AttentionKind::GatedDeltaNet)
    {
        // Recurrent attention needs its dedicated Vulkan operator, not just Vulkan projections.
        if (attention.gated_delta_vulkan_operator == invalid_compiled_operator_handle)
            return false;
        return static_cast<bool>(operators.at(attention.gated_delta_vulkan_operator).gated_delta);
    }
    if (attention.kind != AttentionKind::Standard
        && attention.kind != AttentionKind::MultiHeadLatent)
    {
        return false;
    }

    if (attention.kind == AttentionKind::Standard)
    {
        // Projections alone still leave normalization, RoPE, KV append, and
        // SDPA on the CPU; admit only the complete attention operator.
        return attention.vulkan_attention_operator != invalid_compiled_operator_handle
               && (operators.at(attention.vulkan_attention_operator).attention
                   || operators.at(attention.vulkan_attention_operator).qsa_attention);
    }

    if (attention.vulkan_attention_operator != invalid_compiled_operator_handle
        && operators.at(attention.vulkan_attention_operator).qsa_attention)
        return true;

    if (attention.vulkan_attention_operator != invalid_compiled_operator_handle
        && operators.at(attention.vulkan_attention_operator).attention)
    {
        return true;
    }
    if (uses_vulkan_dense_operator(operators.at(attention.fused_qkv_operator))
        || uses_vulkan_dense_operator(operators.at(attention.fused_qkv_gate_bfloat16_operator)))
        return true;

    const TensorHandle dense_handles[] = {
        attention.query_weight,
        attention.key_weight,
        attention.value_weight,
        attention.output_weight,
        attention.output_gate_weight,
        attention.query_a_weight,
        attention.query_b_weight,
        attention.key_value_weight,
        attention.output_a_weight,
        attention.output_b_weight,
    };
    for (TensorHandle handle : dense_handles)
    {
        if (handle != invalid_tensor_handle
            && uses_vulkan_dense_operator(operators.at_weight(handle)))
        {
            return true;
        }
    }
    return false;
}

static bool dense_qwen_bfloat16_projection_supported(const CompiledModel& compiled,
                                                     const TensorData& matrix,
                                                     uint64_t alignment,
                                                     uint32_t device_index);

static bool standard_attention_requires_native(const CompiledModel& compiled,
                                               const AttentionBlockPlan& attention,
                                               uint32_t device_index)
{
    if (attention.kind != AttentionKind::Standard
        || has_flag(attention.flags, AttentionBlockQsa)
        || has_flag(attention.flags, AttentionBlockExternalResidual))
    {
        return false;
    }

    const TensorHandle projection_handles[] = {
        attention.query_weight,
        attention.key_weight,
        attention.value_weight,
        attention.output_weight,
        attention.output_gate_weight,
    };
    bool has_native_quantized_projection = false;
    bool all_bfloat16 = true;
    for (TensorHandle handle : projection_handles)
    {
        if (handle == invalid_tensor_handle)
            continue;
        const DType dtype = compiled.weights.at(handle).dtype;
        has_native_quantized_projection = has_native_quantized_projection
                                          || dtype == DType::Int4
                                          || dtype == DType::Float8E4M3
                                          || is_qnk_dtype(dtype);
        all_bfloat16 = all_bfloat16 && dtype == DType::BFloat16;
    }

    const bool has_query_key_norm = has_flag(attention.flags, AttentionBlockQueryKeyNorm);
    const bool has_output_gate = has_flag(attention.flags, AttentionBlockOutputGate);
    // Preserve the established fused BF16 implementation for the paired
    // QK-norm/output-gate form; the native QnK operator covers independent
    // flags and partial RoPE for quantized, BF16, and FP32 projections.
    const bool legacy_bfloat16_qk_gate = has_query_key_norm && has_output_gate && all_bfloat16
                                         && dense_qwen_bfloat16_projection_supported(compiled,
                                                                                     compiled.weights.at(attention.output_weight),
                                                                                     4, device_index);
    const bool generalized_semantics = ((has_query_key_norm || has_output_gate)
                                        && !legacy_bfloat16_qk_gate)
                                       || ((attention.rope_head_dimension != 0
                                            && attention.rope_head_dimension != attention.head_dimension)
                                           && !legacy_bfloat16_qk_gate)
                                       || (attention.norm_weight_offset != 0.0f
                                           && !legacy_bfloat16_qk_gate)
                                       || (attention.sliding_window != 0
                                           && !has_flag(attention.flags, AttentionBlockSink));
    return has_native_quantized_projection || generalized_semantics;
}

static bool standard_attention_native_compatible(const CompiledModel& compiled,
                                                 const AttentionBlockPlan& attention) noexcept
{
    const TensorHandle projection_handles[] = {
        attention.query_weight,
        attention.key_weight,
        attention.value_weight,
        attention.output_weight,
        attention.output_gate_weight,
    };
    for (TensorHandle handle : projection_handles)
    {
        if (handle == invalid_tensor_handle)
            continue;
        const DType dtype = compiled.weights.at(handle).dtype;
        if (dtype != DType::Int4 && !is_qnk_dtype(dtype)
            && dtype != DType::BFloat16 && dtype != DType::Float32
            && dtype != DType::Float8E4M3)
        {
            return false;
        }
    }
    return true;
}

static uint64_t release_tensor_host_storage(TensorData& tensor)
{
    uint64_t released_size = 0;
    if (tensor.dtype == DType::Float32)
    {
        released_size = tensor.float32_data.size() * sizeof(float);
        std::vector<float>().swap(tensor.float32_data);
    }
    else if (tensor.dtype == DType::BFloat16)
    {
        released_size = tensor.bfloat16_data.size() * sizeof(uint16_t);
        std::vector<uint16_t>().swap(tensor.bfloat16_data);
    }
    else if (tensor.dtype == DType::Float8E4M3)
    {
        released_size = tensor.quantized_data.size() + tensor.quantization_scales.size() * sizeof(float);
        std::vector<uint8_t>().swap(tensor.quantized_data);
        std::vector<float>().swap(tensor.quantization_scales);
    }
    else if (tensor.dtype == DType::Q8_0)
    {
        released_size = tensor.quantized_data.size();
        std::vector<uint8_t>().swap(tensor.quantized_data);
    }
    else
    {
        return 0;
    }
    tensor.mapped_data.reset();
    tensor.mapped_size = 0;
    return released_size;
}

static void release_vulkan_dense_handle(CompiledModel& compiled, TensorHandle handle)
{
    if (handle == invalid_tensor_handle)
        return;
    compiled.released_dense_host_size = saturating_add_u64(compiled.released_dense_host_size,
                                                           release_tensor_host_storage(compiled.weights.at_mutable(handle)));
}

void release_vulkan_dense_host_copies(CompiledModel& compiled)
{
    using MappedOwners = std::map<std::shared_ptr<const uint8_t>, uint64_t, std::owner_less<std::shared_ptr<const uint8_t>>>;
    const auto mapped_owners = [&compiled] {
        MappedOwners owners;
        for (TensorHandle handle = 0; handle < compiled.weights.size(); ++handle)
        {
            const TensorData& tensor = compiled.weights.at(handle);
            if (tensor.mapped_data)
                owners[tensor.mapped_data] = std::max(owners[tensor.mapped_data], tensor.mapped_size);
        }
        return owners;
    };
    const MappedOwners previous_owners = mapped_owners();
    // FP8 Shared Experts keep their CPU fallback despite resident Vulkan projections.
    std::vector<TensorHandle> shared_fallback_handles;
    const auto retain_shared = [&](const CompiledLayerPlan& layer) {
        if (!layer.moe.has_shared_expert)
            return;
        const ExpertPlan& shared = layer.moe.shared_expert;
        if (shared.gate_weight != invalid_tensor_handle
            && compiled.weights.at(shared.gate_weight).dtype == DType::Float8E4M3)
        {
            shared_fallback_handles.push_back(shared.gate_weight);
            shared_fallback_handles.push_back(shared.up_weight);
            shared_fallback_handles.push_back(shared.down_weight);
        }
    };
    for (const CompiledLayerPlan& layer : compiled.graph.layer_plans)
        retain_shared(layer);
    for (const CompiledLayerPlan& layer : compiled.speculative.graph.layer_plans)
        retain_shared(layer);
    for (TensorHandle handle = 0; handle < compiled.weights.size(); ++handle)
    {
        TensorData& tensor = compiled.weights.at_mutable(handle);
        if (uses_vulkan_dense_operator(compiled.operators.at_weight(handle))
            && std::find(shared_fallback_handles.begin(), shared_fallback_handles.end(), handle) == shared_fallback_handles.end())
            compiled.released_dense_host_size = saturating_add_u64(compiled.released_dense_host_size,
                                                                   release_tensor_host_storage(tensor));
    }

    const auto release_fused_layer_handles = [&compiled](CompiledLayerPlan& layer) {
        AttentionBlockPlan& attention = layer.attention;
        const bool qkv_fused = attention.fused_qkv_operator != invalid_compiled_operator_handle
                               || attention.fused_qkv_gate_bfloat16_operator != invalid_compiled_operator_handle;
        if (qkv_fused)
        {
            release_vulkan_dense_handle(compiled, attention.query_weight);
            release_vulkan_dense_handle(compiled, attention.key_weight);
            release_vulkan_dense_handle(compiled, attention.value_weight);
        }
        if (attention.fused_qkv_gate_bfloat16_operator != invalid_compiled_operator_handle)
            release_vulkan_dense_handle(compiled, attention.output_gate_weight);

        if (attention.fused_delta_input_operator != invalid_compiled_operator_handle
            && attention.gated_delta_vulkan_operator != invalid_compiled_operator_handle
            && compiled.operators.at(attention.gated_delta_vulkan_operator).gated_delta
            && uses_vulkan_dense_operator(compiled.operators.at(attention.fused_delta_input_operator)))
        {
            release_vulkan_dense_handle(compiled, attention.delta_qkv_weight);
            release_vulkan_dense_handle(compiled, attention.delta_z_weight);
            release_vulkan_dense_handle(compiled, attention.delta_beta_weight);
            release_vulkan_dense_handle(compiled, attention.delta_alpha_weight);
        }

        if (attention.vulkan_attention_operator != invalid_compiled_operator_handle)
        {
            release_vulkan_dense_handle(compiled, attention.pre_attention_norm_weight);
            release_vulkan_dense_handle(compiled, attention.sinks);
            if (has_flag(attention.flags, AttentionBlockQueryKeyNorm))
            {
                release_vulkan_dense_handle(compiled, attention.query_norm_weight);
                release_vulkan_dense_handle(compiled, attention.key_norm_weight);
            }
        }

        if (layer.moe.fused_shared_input_bfloat16_operator != invalid_compiled_operator_handle)
        {
            release_vulkan_dense_handle(compiled, layer.moe.shared_expert.gate_weight);
            release_vulkan_dense_handle(compiled, layer.moe.shared_expert.up_weight);
            release_vulkan_dense_handle(compiled, layer.moe.shared_expert_gate_weight);
        }
        const ExpertPlan& shared = layer.moe.shared_expert;
        if (layer.moe.has_shared_expert && shared.vulkan_operator != invalid_compiled_operator_handle
            && compiled.operators.at(shared.vulkan_operator).qnk_expert
            && compiled.weights.at(shared.gate_weight).dtype == DType::Q8_0
            && compiled.weights.at(shared.up_weight).dtype == DType::Q8_0
            && compiled.weights.at(shared.down_weight).dtype == DType::Q8_0)
        {
            release_vulkan_dense_handle(compiled, shared.gate_weight);
            release_vulkan_dense_handle(compiled, shared.up_weight);
            release_vulkan_dense_handle(compiled, shared.down_weight);
        }
    };

    for (CompiledLayerPlan& layer : compiled.graph.layer_plans)
        release_fused_layer_handles(layer);
    for (CompiledLayerPlan& layer : compiled.speculative.graph.layer_plans)
        release_fused_layer_handles(layer);

    const MappedOwners retained_owners = mapped_owners();
    for (const auto& [owner, bytes] : previous_owners)
    {
        // Tied embeddings and aliased ranges keep their mapping resident until
        // the last model reference is gone. Partial owner release grants no credit.
        if (!retained_owners.contains(owner))
            compiled.released_dense_host_size = saturating_add_u64(compiled.released_dense_host_size, bytes);
    }
}

static Result<void> prepare_ffn_operators(WeightStore& weights,
                                          CompiledOperatorTable& operators,
                                          ExpertPlan& ffn,
                                          uint32_t vulkan_device_index,
                                          const VulkanRuntimePtr& vulkan_runtime,
                                          uint64_t optimization_flags,
                                          bool allow_float32_fallback = true)
{
    const TensorData& gate = weights.at(ffn.gate_weight);
    const TensorData& up = weights.at(ffn.up_weight);
    const TensorData& down = weights.at(ffn.down_weight);
    if (support_vulkan_expert(ffn, gate, down, optimization_flags, &up))
    {
        auto chain = QnkExpert_vulkan::create_separate(gate, nullptr, up, nullptr, down, nullptr,
                                                       ffn.activation_limit, vulkan_device_index,
                                                       ffn.activation, vulkan_runtime, optimization_flags);
        if (chain)
        {
            ffn.vulkan_operator = operators.allocate();
            operators.at_mutable(ffn.vulkan_operator).qnk_expert = std::move(chain);
            return {};
        }
    }
    if (gate.dtype == DType::BFloat16 && up.dtype == DType::BFloat16
        && down.dtype == DType::BFloat16)
    {
        auto fused = Bfloat16Linear_vulkan::create_fused({&gate, &up}, {nullptr, nullptr},
                                                         vulkan_device_index, vulkan_runtime, optimization_flags);
        if (fused)
        {
            ffn.vulkan_operator = operators.allocate();
            operators.at_mutable(ffn.vulkan_operator).bfloat16 = std::move(fused);
            return prepare_vulkan_linear_operator(weights, operators, ffn.down_weight, invalid_tensor_handle,
                                                  vulkan_device_index, vulkan_runtime, optimization_flags, 1, true, allow_float32_fallback);
        }
    }
    if (!allow_float32_fallback)
        return Error{ErrorCode::InternalError, "failed to create fused Vulkan BF16 FFN operator"};
    for (TensorHandle handle : {ffn.gate_weight, ffn.up_weight, ffn.down_weight})
    {
        auto prepared = prepare_vulkan_linear_operator(weights, operators, handle, invalid_tensor_handle,
                                                       vulkan_device_index, vulkan_runtime, optimization_flags, 1, true, allow_float32_fallback);
        if (!prepared)
            return prepared.error();
    }
    return {};
}

static Result<void> prepare_shared_expert_operators(WeightStore& weights,
                                                    CompiledOperatorTable& operators,
                                                    MoeBlockPlan& moe,
                                                    uint32_t vulkan_device_index,
                                                    const VulkanRuntimePtr& vulkan_runtime,
                                                    uint64_t optimization_flags)
{
    ExpertPlan& shared = moe.shared_expert;
    const TensorData& gate = weights.at(shared.gate_weight);
    const TensorData& up = weights.at(shared.up_weight);
    const bool has_router_gate = moe.shared_expert_gate_weight != invalid_tensor_handle;
    const auto quantized = [](DType dtype) { return dtype == DType::Int4 || is_qnk_dtype(dtype); };
    if (quantized(gate.dtype) || quantized(up.dtype) || quantized(weights.at(shared.down_weight).dtype))
    {
        auto prepared = prepare_ffn_operators(weights, operators, shared, vulkan_device_index,
                                              vulkan_runtime, optimization_flags);
        if (!prepared || !has_router_gate)
            return prepared;
        return prepare_vulkan_linear_operator(weights, operators, moe.shared_expert_gate_weight, invalid_tensor_handle,
                                              vulkan_device_index, vulkan_runtime, optimization_flags);
    }
    if (!has_router_gate && shared.activation == ExpertActivation::DeepSeekSwiGlu
        && gate.dtype == DType::Float8E4M3 && up.dtype == DType::Float8E4M3
        && weights.at(shared.down_weight).dtype == DType::Float8E4M3)
    {
        const TensorHandle handles[] = {shared.gate_weight, shared.up_weight, shared.down_weight};
        std::shared_ptr<Float8Linear_vulkan> prepared[3];
        // Shared GPU execution is optional. Keep all creation private until the
        // whole chain is available, including on otherwise valid CPU shapes
        // unsupported by the GPU kernel or when GPU allocation fails.
        for (size_t index = 0; index < 3; ++index)
        {
            prepared[index] = Float8Linear_vulkan::create(weights.at(handles[index]), nullptr, 1,
                                                          vulkan_device_index, vulkan_runtime, optimization_flags);
            if (!prepared[index])
                return {};
        }
        for (size_t index = 0; index < 3; ++index)
            operators.at_weight_mutable(handles[index]).float8 = std::move(prepared[index]);
        return {};
    }
    if (gate.dtype != DType::BFloat16
        || up.dtype != DType::BFloat16
        || (has_router_gate
            && weights.at(moe.shared_expert_gate_weight).dtype != DType::BFloat16))
        return {};

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
        matrices.push_back(&weights.at(moe.shared_expert_gate_weight));
        biases.push_back(nullptr);
    }
    std::shared_ptr<Bfloat16Linear_vulkan> fused_operator = Bfloat16Linear_vulkan::create_fused(matrices,
                                                                                                biases,
                                                                                                vulkan_device_index,
                                                                                                vulkan_runtime,
                                                                                                optimization_flags);
    if (!fused_operator)
        return {};

    const CompiledOperatorHandle fused_handle = operators.allocate();
    operators.at_mutable(fused_handle).bfloat16 = std::move(fused_operator);
    moe.fused_shared_input_bfloat16_operator = fused_handle;
    return prepare_vulkan_linear_operator(weights,
                                          operators,
                                          shared.down_weight,
                                          invalid_tensor_handle,
                                          vulkan_device_index,
                                          vulkan_runtime,
                                          optimization_flags);
}

static Result<void> prepare_vulkan_qkv_operator(CompiledModel& compiled,
                                                CompiledLayerPlan& layer_plan,
                                                bool use_bfloat16_fusion,
                                                const char* failure_message,
                                                bool allow_float32_fallback = true)
{
    AttentionBlockPlan& plan = layer_plan.attention;
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
    if (std::any_of(qkv_matrices.begin(), qkv_matrices.end(), [](const TensorData* matrix) {
            return matrix->dtype == DType::Int4 || matrix->dtype == DType::Float8E4M3 || is_qnk_dtype(matrix->dtype);
        }))
    {
        const TensorHandle handles[] = {plan.query_weight, plan.key_weight, plan.value_weight};
        const TensorHandle biases[] = {plan.query_bias, plan.key_bias, plan.value_bias};
        for (size_t index = 0; index < 3; ++index)
        {
            auto prepared = prepare_vulkan_linear_operator(compiled.weights, compiled.operators,
                                                           handles[index], biases[index],
                                                           layer_plan.vulkan_device_index, compiled.vulkan_runtime,
                                                           compiled.opt.optimization_flags);
            if (!prepared)
                return prepared.error();
        }
        return {};
    }
    if (use_bfloat16_fusion
        && qkv_matrices.front()->dtype
               == DType::BFloat16
        && plan.output_gate_weight
               != invalid_tensor_handle
        && !query_bias_data
        && !key_bias_data
        && !value_bias_data)
    {
        const TensorData* output_gate = &compiled.weights.at(plan.output_gate_weight);
        auto fused_operator = Bfloat16Linear_vulkan::create_fused({qkv_matrices[0], qkv_matrices[1], qkv_matrices[2], output_gate},
                                                                  {nullptr, nullptr, nullptr, nullptr},
                                                                  layer_plan.vulkan_device_index,
                                                                  compiled.vulkan_runtime,
                                                                  compiled.opt.optimization_flags);
        if (fused_operator)
        {
            const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
            compiled.operators.at_mutable(fused_handle).bfloat16 = std::move(fused_operator);
            plan.fused_qkv_gate_bfloat16_operator = fused_handle;
            return {};
        }
        if (!allow_float32_fallback)
            return Error{ErrorCode::InternalError, failure_message};
    }
    if (use_bfloat16_fusion
        && qkv_matrices.front()->dtype
               == DType::BFloat16)
    {
        auto fused_operator = Bfloat16Linear_vulkan::create_fused(qkv_matrices,
                                                                  qkv_biases,
                                                                  layer_plan.vulkan_device_index,
                                                                  compiled.vulkan_runtime,
                                                                  compiled.opt.optimization_flags);
        if (fused_operator)
        {
            const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
            compiled.operators.at_mutable(fused_handle).bfloat16 = std::move(fused_operator);
            plan.fused_qkv_operator = fused_handle;
            return {};
        }
    }
    if (!allow_float32_fallback)
        return Error{ErrorCode::InternalError, failure_message};
    auto fused_operator = Linear::create_fused(qkv_matrices,
                                               qkv_biases,
                                               layer_plan.vulkan_device_index,
                                               compiled.vulkan_runtime,
                                               compiled.opt.optimization_flags);
    if (fused_operator)
    {
        const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
        compiled.operators.at_mutable(fused_handle).linear = std::move(fused_operator);
        plan.fused_qkv_operator = fused_handle;
        return {};
    }
    return Error{ErrorCode::InternalError, failure_message};
}

static Result<void> prepare_latent_attention_operators(CompiledModel& compiled,
                                                       CompiledLayerPlan& layer_plan,
                                                       const char* diagnostic_prefix = "",
                                                       bool attention_core_enabled = true)
{
    AttentionBlockPlan& plan = layer_plan.attention;
    Result<void> prepared;
    const TensorHandle latent_linear_handles[] = {
        plan.query_a_weight,
        plan.query_b_weight,
        plan.key_value_weight,
        plan.output_b_weight,
        plan.indexer_query_weight,
        plan.indexer_weights_weight,
    };
    for (TensorHandle handle : latent_linear_handles)
    {
        if (handle == invalid_tensor_handle)
            continue;
        prepared = prepare_vulkan_linear_operator(compiled.weights, compiled.operators, handle,
                                                  invalid_tensor_handle,
                                                  layer_plan.vulkan_device_index,
                                                  compiled.vulkan_runtime,
                                                  compiled.opt.optimization_flags);
        if (!prepared)
            return prepared.error();
    }
    if ((plan.compression_ratio == 4 || plan.compression_ratio == 128)
        && has_flag(compiled.opt.optimization_flags, OptimizationVulkanLatentCompressor))
    {
        const TensorHandle compressor_linear_handles[] = {
            plan.compressor_key_value_weight,
            plan.compressor_gate_weight,
            plan.indexer_compressor_key_value_weight,
            plan.indexer_compressor_gate_weight,
        };
        for (TensorHandle handle : compressor_linear_handles)
        {
            if (handle == invalid_tensor_handle)
                continue;
            prepared = prepare_vulkan_linear_operator(compiled.weights, compiled.operators, handle,
                                                      invalid_tensor_handle,
                                                      layer_plan.vulkan_device_index,
                                                      compiled.vulkan_runtime,
                                                      compiled.opt.optimization_flags);
            if (!prepared)
                return prepared.error();
        }
    }
    prepared = prepare_vulkan_linear_operator(compiled.weights, compiled.operators, plan.output_a_weight,
                                              invalid_tensor_handle,
                                              layer_plan.vulkan_device_index,
                                              compiled.vulkan_runtime,
                                              compiled.opt.optimization_flags,
                                              plan.output_group_count);
    if (!prepared)
        return prepared.error();
    const CompiledOperator& query_a = compiled.operators.at_weight(plan.query_a_weight);
    const CompiledOperator& query_b = compiled.operators.at_weight(plan.query_b_weight);
    if (query_a.float8 && query_b.float8)
    {
        if (!query_a.float8->prepare_rms_norm(compiled.weights.at(plan.query_norm_weight),
                                              plan.norm_epsilon))
            return Error{ErrorCode::InternalError, "failed to prepare " + std::string(diagnostic_prefix) + "Vulkan FP8 query RMSNorm chain"};
        if (!query_a.float8->prepare_input_rms_norm(compiled.weights.at(plan.pre_attention_norm_weight),
                                                    plan.norm_epsilon))
            return Error{ErrorCode::InternalError, "failed to prepare " + std::string(diagnostic_prefix) + "Vulkan FP8 latent input RMSNorm chain"};
    }
    if (query_a.bfloat16 && query_b.bfloat16)
    {
        if (!query_a.bfloat16->prepare_output_rms_norm(compiled.weights.at(plan.query_norm_weight),
                                                       plan.norm_epsilon))
            return Error{ErrorCode::InternalError, "failed to prepare " + std::string(diagnostic_prefix) + "Vulkan BF16 query RMSNorm chain"};
        if (!query_a.bfloat16->prepare_input_rms_norm(compiled.weights.at(plan.pre_attention_norm_weight),
                                                      plan.norm_epsilon))
            return Error{ErrorCode::InternalError, "failed to prepare " + std::string(diagnostic_prefix) + "Vulkan BF16 latent input RMSNorm chain"};
    }
    const CompiledOperator& output_a = compiled.operators.at_weight(plan.output_a_weight);
    const CompiledOperator& output_b = compiled.operators.at_weight(plan.output_b_weight);
    if (attention_core_enabled
        && has_flag(compiled.opt.optimization_flags, OptimizationVulkanAttention))
    {
        compiled.operators.at_weight_mutable(plan.query_a_weight).latent_attention = LatentAttention_vulkan::create(plan, compiled.weights.at(plan.sinks),
                                                                                                                    output_a, output_b);
        const auto& latent = compiled.operators.at_weight(plan.query_a_weight).latent_attention;
        if (latent)
            (void)latent->prepare(compiled.weights, compiled.operators, compiled.opt.optimization_flags);
    }
    return {};
}

static void prepare_latent_layer_operators(CompiledModel& compiled, CompiledLayerPlan& layer)
{
#if NCNN_MOE_WITH_VULKAN
    if (compiled.descriptor.hyper_connection_kind != HyperConnectionKind::Sinkhorn
        || layer.attention.kind != AttentionKind::MultiHeadLatent
        || has_flag(compiled.opt.flags, OptionRouterPrediction))
        return;
    auto& op = compiled.operators.at_weight_mutable(layer.attention.query_a_weight);
    if (!op.latent_attention || !op.latent_attention->can_record())
        return;
    const auto create_hyper = [&](TensorHandle function, TensorHandle scale, TensorHandle base) {
        return HyperConnection_vulkan::create(compiled.weights.at(function), compiled.weights.at(scale), compiled.weights.at(base),
                                              compiled.descriptor.hyper_connection_multiplier,
                                              compiled.descriptor.hyper_connection_iterations,
                                              compiled.descriptor.norm_epsilon, compiled.descriptor.hyper_connection_epsilon,
                                              layer.vulkan_device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    };
    auto attention_hyper = create_hyper(layer.hyper_connection.attention_function, layer.hyper_connection.attention_scale,
                                        layer.hyper_connection.attention_base);
    auto ffn_hyper = create_hyper(layer.hyper_connection.ffn_function, layer.hyper_connection.ffn_scale,
                                  layer.hyper_connection.ffn_base);
    if (!attention_hyper || !ffn_hyper)
        return;
    auto norm = RmsNorm_vulkan::create(compiled.weights.at(layer.moe.pre_ffn_norm_weight), compiled.descriptor.norm_epsilon,
                                       compiled.descriptor.norm_weight_offset, ffn_hyper->vulkan_context(), ffn_hyper->option());
    const TensorData* bias = layer.moe.router_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(layer.moe.router_bias);
    const TensorData* selection_bias = layer.moe.router_selection_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(layer.moe.router_selection_bias);
    auto router = Router_vulkan::create(compiled.weights.at(layer.moe.router_weight), bias, selection_bias,
                                        layer.vulkan_device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    op.latent_layer = LatentLayer_vulkan::create(op.latent_attention, std::move(attention_hyper), std::move(ffn_hyper),
                                                 std::move(norm), std::move(router));
#else
    (void)compiled;
    (void)layer;
#endif
}

static void prepare_gated_delta_attention_operators(CompiledModel& compiled,
                                                    CompiledLayerPlan& layer_plan)
{
    AttentionBlockPlan& plan = layer_plan.attention;
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
    const TensorData& output_matrix = compiled.weights.at(plan.output_weight);
    const auto native_projection = [](const TensorData& weight) { return weight.dtype == DType::Float32 || weight.dtype == DType::Int4 || is_qnk_dtype(weight.dtype); };
    if (native_projection(output_matrix)
        || std::any_of(delta_input_matrices.begin(), delta_input_matrices.end(), [&](const TensorData* weight) { return native_projection(*weight); }))
    {
        if (!has_flag(compiled.opt.optimization_flags, OptimizationVulkanQnK)
            && (is_qnk_dtype(output_matrix.dtype)
                || std::any_of(delta_input_matrices.begin(), delta_input_matrices.end(), [](const TensorData* weight) { return is_qnk_dtype(weight->dtype); })))
        {
            return;
        }
        std::vector<std::shared_ptr<QnkLinear_vulkan>> inputs;
        for (const TensorData* weight : delta_input_matrices)
        {
            auto projection = QnkLinear_vulkan::create(*weight, nullptr, layer_plan.vulkan_device_index,
                                                       compiled.vulkan_runtime, compiled.opt.optimization_flags);
            if (!projection)
                return;
            inputs.push_back(std::move(projection));
        }
        auto output = QnkLinear_vulkan::create(output_matrix, nullptr, layer_plan.vulkan_device_index,
                                               compiled.vulkan_runtime, compiled.opt.optimization_flags);
        if (!output)
            return;
        auto gated = GatedDeltaNet_vulkan::create_quantized(inputs,
                                                            compiled.weights.at(plan.delta_convolution_weight),
                                                            compiled.weights.at(plan.delta_time_bias),
                                                            compiled.weights.at(plan.delta_decay_log),
                                                            compiled.weights.at(plan.delta_norm_weight), output,
                                                            plan.head_count, plan.kv_head_count, plan.head_dimension,
                                                            plan.value_head_dimension, plan.convolution_kernel_size,
                                                            plan.norm_epsilon, has_flag(plan.flags, AttentionBlockSigmoidGate),
                                                            layer_plan.vulkan_device_index, compiled.vulkan_runtime,
                                                            compiled.opt.optimization_flags);
        if (!gated)
            return;
        const CompiledOperatorHandle handle = compiled.operators.allocate();
        compiled.operators.at_mutable(handle).gated_delta = std::move(gated);
        plan.gated_delta_vulkan_operator = handle;
        const TensorHandle input_handles[] = {plan.delta_qkv_weight, plan.delta_z_weight, plan.delta_beta_weight, plan.delta_alpha_weight};
        for (size_t index = 0; index < inputs.size(); ++index)
            compiled.operators.at_weight_mutable(input_handles[index]).qnk = std::move(inputs[index]);
        compiled.operators.at_weight_mutable(plan.output_weight).qnk = std::move(output);
        return;
    }
    if (delta_input_matrices.front()->dtype != DType::BFloat16
        || output_matrix.dtype != DType::BFloat16)
        return;

    std::shared_ptr<Bfloat16Linear_vulkan> fused_input_operator = Bfloat16Linear_vulkan::create_fused(delta_input_matrices,
                                                                                                      delta_input_biases,
                                                                                                      layer_plan.vulkan_device_index,
                                                                                                      compiled.vulkan_runtime,
                                                                                                      compiled.opt.optimization_flags);
    if (!fused_input_operator)
        return;

    // Fuse pre-attention RMSNorm with the DeltaNet input projection when the
    // optional chain is available; the dedicated operator remains valid when
    // the chain itself cannot be prepared.
    if (plan.pre_attention_norm_weight != invalid_tensor_handle)
    {
        (void)fused_input_operator->prepare_rms_norm(compiled.weights.at(plan.pre_attention_norm_weight),
                                                     plan.norm_epsilon,
                                                     plan.norm_weight_offset);
    }

    std::shared_ptr<Bfloat16Linear_vulkan> output_operator = Bfloat16Linear_vulkan::create(output_matrix,
                                                                                           nullptr,
                                                                                           layer_plan.vulkan_device_index,
                                                                                           compiled.vulkan_runtime,
                                                                                           compiled.opt.optimization_flags);
    if (!output_operator)
        return;

    std::shared_ptr<GatedDeltaNet_vulkan> gated_operator = GatedDeltaNet_vulkan::create(fused_input_operator,
                                                                                        compiled.weights.at(plan.delta_convolution_weight),
                                                                                        compiled.weights.at(plan.delta_time_bias),
                                                                                        compiled.weights.at(plan.delta_decay_log),
                                                                                        compiled.weights.at(plan.delta_norm_weight),
                                                                                        output_operator,
                                                                                        plan.head_count,
                                                                                        plan.kv_head_count,
                                                                                        plan.head_dimension,
                                                                                        plan.value_head_dimension,
                                                                                        plan.convolution_kernel_size,
                                                                                        plan.norm_epsilon,
                                                                                        has_flag(plan.flags, AttentionBlockSigmoidGate),
                                                                                        layer_plan.vulkan_device_index,
                                                                                        compiled.vulkan_runtime,
                                                                                        compiled.opt.optimization_flags);
    if (!gated_operator)
        return;

    // Publish handles only after every optional Vulkan product is complete.
    // The output weight entry is assigned last because the handle allocations
    // above may relocate the operator table.
    const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
    compiled.operators.at_mutable(fused_handle).bfloat16 = std::move(fused_input_operator);
    plan.fused_delta_input_operator = fused_handle;

    const CompiledOperatorHandle gated_handle = compiled.operators.allocate();
    compiled.operators.at_mutable(gated_handle).gated_delta = std::move(gated_operator);
    plan.gated_delta_vulkan_operator = gated_handle;

    compiled.operators.at_weight_mutable(plan.output_weight).bfloat16 = std::move(output_operator);
}

static AttentionConfig_vulkan standard_attention_vulkan_config(const CompiledModel& compiled,
                                                               const AttentionBlockPlan& plan)
{
    AttentionConfig_vulkan config;
    config.hidden_size = compiled.descriptor.hidden_size;
    config.head_count = plan.head_count;
    config.kv_head_count = plan.kv_head_count;
    config.head_dimension = plan.head_dimension;
    config.rope_head_dimension = plan.rope_head_dimension;
    config.sliding_window = plan.sliding_window;
    config.initial_context_length = plan.initial_context_length;
    config.norm_epsilon = plan.norm_epsilon;
    config.norm_weight_offset = plan.norm_weight_offset;
    config.rope_theta = plan.rope_theta;
    config.rope_scaling_factor = plan.rope_scaling_factor;
    config.rope_ntk_alpha = plan.rope_ntk_alpha;
    config.rope_ntk_beta = plan.rope_ntk_beta;
    config.activation_dtype = compiled.descriptor.activation_dtype;
    config.kv_cache_dtype = plan.kv_cache_dtype;
    config.optimization_flags = compiled.opt.optimization_flags;
    if (has_flag(plan.flags, AttentionBlockSink))
        config.flags |= AttentionSink;
    if (has_flag(plan.flags, AttentionBlockQueryKeyNorm))
        config.flags |= AttentionQueryKeyNorm;
    if (has_flag(plan.flags, AttentionBlockOutputGate))
        config.flags |= AttentionOutputGate;
    return config;
}

static Result<void> prepare_native_attention(CompiledModel& compiled,
                                             CompiledLayerPlan& layer_plan)
{
    AttentionBlockPlan& plan = layer_plan.attention;
    if (plan.pre_attention_norm_weight == invalid_tensor_handle)
        return Error{ErrorCode::UnsupportedModel, "native Vulkan Standard Attention requires a pre-attention RMSNorm"};

    const TensorHandle weights[] = {
        plan.query_weight,
        plan.key_weight,
        plan.value_weight,
        plan.output_weight,
        plan.output_gate_weight,
    };
    const TensorHandle biases[] = {
        plan.query_bias,
        plan.key_bias,
        plan.value_bias,
        plan.output_bias,
        invalid_tensor_handle,
    };
    std::array<CompiledOperator, 5> projections;
    for (size_t index = 0; index < projections.size(); ++index)
    {
        const TensorHandle handle = weights[index];
        if (handle == invalid_tensor_handle)
            continue;

        CompiledOperator& projection_operator = compiled.operators.at_weight_mutable(handle);
        if (projection_operator.qnk || projection_operator.float8)
        {
            projections[index] = projection_operator;
            continue;
        }

        const TensorData* bias = biases[index] == invalid_tensor_handle
                                     ? nullptr
                                     : &compiled.weights.at(biases[index]);
        CompiledOperator projection;
        if (compiled.weights.at(handle).dtype == DType::Float8E4M3)
        {
            projection.float8 = Float8Linear_vulkan::create(compiled.weights.at(handle), bias, 1,
                                                            layer_plan.vulkan_device_index,
                                                            compiled.vulkan_runtime,
                                                            compiled.opt.optimization_flags);
        }
        else
        {
            projection.qnk = QnkLinear_vulkan::create(compiled.weights.at(handle), bias,
                                                      layer_plan.vulkan_device_index,
                                                      compiled.vulkan_runtime,
                                                      compiled.opt.optimization_flags);
        }
        if (!projection.qnk && !projection.float8)
            return Error{ErrorCode::InternalError,
                         "failed to create native Vulkan Standard Attention projection at layer "
                             + std::to_string(layer_plan.layer_id)};
        // This path owns one native operator per matrix. Do not retain an
        // older FP32/BF16 upload for the same weight alongside it.
        projection_operator = projection;
        projections[index] = std::move(projection);
    }

    const TensorData* query_norm = has_flag(plan.flags, AttentionBlockQueryKeyNorm)
                                       ? &compiled.weights.at(plan.query_norm_weight)
                                       : nullptr;
    const TensorData* key_norm = has_flag(plan.flags, AttentionBlockQueryKeyNorm)
                                     ? &compiled.weights.at(plan.key_norm_weight)
                                     : nullptr;
    const TensorData* sinks = plan.sinks == invalid_tensor_handle
                                  ? nullptr
                                  : &compiled.weights.at(plan.sinks);
    auto attention_operator = Attention_vulkan::create_native(compiled.weights.at(plan.pre_attention_norm_weight),
                                                              query_norm,
                                                              key_norm,
                                                              sinks,
                                                              projections,
                                                              standard_attention_vulkan_config(compiled, plan));
    if (!attention_operator)
        return Error{ErrorCode::InternalError,
                     "failed to create native Vulkan Standard Attention at layer "
                         + std::to_string(layer_plan.layer_id)};

    const CompiledOperatorHandle attention_handle = compiled.operators.allocate();
    compiled.operators.at_mutable(attention_handle).attention = std::move(attention_operator);
    plan.vulkan_attention_operator = attention_handle;
    return {};
}

static Result<void> prepare_standard_attention_operators(CompiledModel& compiled,
                                                         CompiledLayerPlan& layer_plan,
                                                         bool allow_float32_fallback = true)
{
    AttentionBlockPlan& plan = layer_plan.attention;
    if (!has_flag(compiled.opt.optimization_flags, OptimizationVulkanAttention))
        return {};
    if (has_flag(plan.flags, AttentionBlockQsa | AttentionBlockExternalResidual))
    {
        return Error{ErrorCode::UnsupportedModel,
                     "Vulkan QSA and external residual attention require the complete Qwen device pipeline"};
    }

    const bool requires_native_attention = standard_attention_requires_native(compiled, plan, layer_plan.vulkan_device_index);
    if (requires_native_attention)
    {
        // Some Standard layouts need the native QnK path for their complete
        // attention semantics. If that optimization is explicitly disabled,
        // leave the graph on CPU; otherwise an unsupported native dtype must
        // fail model preparation rather than silently becoming CPU attention.
        if (!has_flag(compiled.opt.optimization_flags, OptimizationVulkanQnK))
            return {};
        if (!standard_attention_native_compatible(compiled, plan))
            return Error{ErrorCode::UnsupportedModel,
                         "Vulkan Standard Attention projection dtype is unsupported at layer "
                             + std::to_string(layer_plan.layer_id)};
        return prepare_native_attention(compiled, layer_plan);
    }

    const bool fused_vulkan_attention_eligible = !has_flag(plan.flags, AttentionBlockQueryKeyNorm)
                                                 && !has_flag(plan.flags, AttentionBlockOutputGate)
                                                 && (plan.rope_head_dimension == 0
                                                     || plan.rope_head_dimension == plan.head_dimension)
                                                 && plan.norm_weight_offset == 0.0f;
    Result<void> prepared;
    prepared = prepare_vulkan_qkv_operator(compiled, layer_plan, !fused_vulkan_attention_eligible,
                                           "failed to create fused Vulkan QKV operator", allow_float32_fallback);
    if (!prepared)
        return prepared.error();
    prepared = prepare_vulkan_linear_operator(compiled.weights, compiled.operators, plan.output_weight,
                                              plan.output_bias,
                                              layer_plan.vulkan_device_index,
                                              compiled.vulkan_runtime,
                                              compiled.opt.optimization_flags,
                                              1,
                                              !fused_vulkan_attention_eligible,
                                              allow_float32_fallback);
    if (!prepared)
        return prepared.error();
    if (plan.output_gate_weight != invalid_tensor_handle)
    {
        if (plan.fused_qkv_gate_bfloat16_operator == invalid_compiled_operator_handle)
        {
            prepared = prepare_vulkan_linear_operator(compiled.weights, compiled.operators, plan.output_gate_weight,
                                                      invalid_tensor_handle,
                                                      layer_plan.vulkan_device_index,
                                                      compiled.vulkan_runtime,
                                                      compiled.opt.optimization_flags, 1, true, allow_float32_fallback);
            if (!prepared)
                return prepared.error();
        }
    }
    const bool legacy_qk_gate_eligible = has_flag(plan.flags, AttentionBlockQueryKeyNorm)
                                         && has_flag(plan.flags, AttentionBlockOutputGate)
                                         && plan.fused_qkv_gate_bfloat16_operator != invalid_compiled_operator_handle
                                         && plan.query_norm_weight != invalid_tensor_handle
                                         && plan.key_norm_weight != invalid_tensor_handle
                                         && compiled.operators.at_weight(plan.output_weight).bfloat16;
    const bool expected_vulkan_attention = fused_vulkan_attention_eligible || legacy_qk_gate_eligible;
    if (expected_vulkan_attention)
    {
        const AttentionConfig_vulkan attention_config = standard_attention_vulkan_config(compiled, plan);
        std::shared_ptr<Attention_vulkan> attention_operator;
        if (fused_vulkan_attention_eligible)
        {
            if (plan.fused_qkv_operator == invalid_compiled_operator_handle
                || !compiled.operators.at(plan.fused_qkv_operator).linear
                || !compiled.operators.at_weight(plan.output_weight).linear)
            {
                return Error{ErrorCode::UnsupportedModel,
                             "Vulkan Standard Attention has no compatible fused projection operator at layer "
                                 + std::to_string(layer_plan.layer_id)};
            }
            const CompiledOperator& fused_operator = compiled.operators.at(plan.fused_qkv_operator);
            attention_operator = Attention_vulkan::create(compiled.weights.at(plan.pre_attention_norm_weight),
                                                          plan.sinks == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.sinks),
                                                          fused_operator.linear,
                                                          compiled.operators.at_weight(plan.output_weight).linear,
                                                          attention_config);
        }
        else
        {
            if (plan.fused_qkv_gate_bfloat16_operator == invalid_compiled_operator_handle
                || !compiled.operators.at(plan.fused_qkv_gate_bfloat16_operator).bfloat16
                || !compiled.operators.at_weight(plan.output_weight).bfloat16)
            {
                return Error{ErrorCode::UnsupportedModel,
                             "Vulkan Standard Attention has no compatible BF16 QK-norm/gate operator at layer "
                                 + std::to_string(layer_plan.layer_id)};
            }
            const CompiledOperator& fused_operator = compiled.operators.at(plan.fused_qkv_gate_bfloat16_operator);
            attention_operator = Attention_vulkan::create(compiled.weights.at(plan.pre_attention_norm_weight),
                                                          compiled.weights.at(plan.query_norm_weight),
                                                          compiled.weights.at(plan.key_norm_weight),
                                                          plan.sinks == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.sinks),
                                                          fused_operator.bfloat16,
                                                          compiled.operators.at_weight(plan.output_weight).bfloat16,
                                                          attention_config);
        }
        if (attention_operator)
        {
            const CompiledOperatorHandle attention_handle = compiled.operators.allocate();
            compiled.operators.at_mutable(attention_handle).attention = std::move(attention_operator);
            plan.vulkan_attention_operator = attention_handle;
        }
        else
        {
            return Error{ErrorCode::InternalError,
                         "failed to create Vulkan Standard Attention at layer "
                             + std::to_string(layer_plan.layer_id)};
        }
    }
    if (!support_vulkan_attention(compiled.operators, plan))
    {
        return Error{ErrorCode::UnsupportedModel,
                     "Vulkan Standard Attention has no complete operator at layer "
                         + std::to_string(layer_plan.layer_id)};
    }
    return {};
}

static uint64_t gated_delta_vulkan_budget_size(const CompilerOption& opt,
                                               bool use_vulkan_dense,
                                               bool protects_file_backed_experts) noexcept
{
    if (!use_vulkan_dense || opt.gpu_heap_budget == 0)
        return 0;

    // Explicit file-backed Expert caches retain their smaller attention
    // allowance. Automatic caches are sized from the live free heap after
    // these operators are allocated, so they do not need this extra reserve.
    const uint64_t heap_divisor = protects_file_backed_experts ? 64 : 8;
    const uint64_t concurrency = std::max(1u, opt.num_concurrent_sessions);
    return opt.gpu_heap_budget / (heap_divisor * concurrency);
}

static Result<void> prepare_mtp_operators(CompiledModel& compiled, bool use_vulkan_attention)
{
    if (compiled.speculative.graph.layer_plans.empty())
        return {};

    CompiledLayerPlan& layer_plan = compiled.speculative.graph.layer_plans.back();
    if (use_vulkan_attention)
    {
        auto prepared = prepare_standard_attention_operators(compiled, layer_plan);
        if (!prepared)
            return prepared.error();
    }

    return prepare_shared_expert_operators(compiled.weights,
                                           compiled.operators,
                                           layer_plan.moe,
                                           layer_plan.vulkan_device_index,
                                           compiled.vulkan_runtime,
                                           compiled.opt.optimization_flags);
}

static Result<void> prepare_dspark_operators(CompiledModel& compiled)
{
    for (CompiledLayerPlan& layer_plan : compiled.speculative.graph.layer_plans)
    {
        auto prepared = prepare_latent_attention_operators(compiled,
                                                           layer_plan,
                                                           "speculative ");
        if (!prepared)
            return prepared.error();

        if (layer_plan.moe.has_shared_expert)
        {
            prepared = prepare_shared_expert_operators(compiled.weights,
                                                       compiled.operators,
                                                       layer_plan.moe,
                                                       layer_plan.vulkan_device_index,
                                                       compiled.vulkan_runtime,
                                                       compiled.opt.optimization_flags);
            if (!prepared)
                return prepared.error();
        }
    }
    return {};
}

static std::vector<TensorHandle> dense_qwen_layer_weights(const CompiledLayerPlan& layer)
{
    const AttentionBlockPlan& attention = layer.attention;
    return {attention.pre_attention_norm_weight, attention.query_weight, attention.query_bias,
            attention.key_weight, attention.key_bias, attention.value_weight, attention.value_bias,
            attention.output_weight, attention.output_bias, attention.output_gate_weight,
            attention.query_norm_weight, attention.key_norm_weight, attention.sinks,
            attention.delta_qkv_weight, attention.delta_z_weight, attention.delta_beta_weight,
            attention.delta_alpha_weight, attention.delta_convolution_weight, attention.delta_time_bias,
            attention.delta_decay_log, attention.delta_norm_weight, layer.dense_pre_ffn_norm_weight,
            layer.dense_ffn.gate_weight, layer.dense_ffn.up_weight, layer.dense_ffn.down_weight};
}

static bool use_bounded_dense_qwen_pipeline(const CompiledModel& compiled)
{
    if (compiled.descriptor.model_type != "qwen3_5"
        || compiled.descriptor.expert_count != 0
        || compiled.weights.at(compiled.lm_head_weight).dtype != DType::BFloat16)
        return false;
    for (const CompiledLayerPlan& layer : compiled.graph.layer_plans)
    {
        if (layer.dense_ffn.down_weight == invalid_tensor_handle
            || (layer.attention.kind != AttentionKind::Standard
                && layer.attention.kind != AttentionKind::GatedDeltaNet))
            return false;
        for (TensorHandle handle : dense_qwen_layer_weights(layer))
        {
            if (handle == invalid_tensor_handle)
                continue;
            const TensorData& tensor = compiled.weights.at(handle);
            if (tensor.shape.size() == 2 && tensor.dtype != DType::BFloat16)
                return false;
        }
    }
    return true;
}

static bool dense_qwen_bfloat16_projection_supported(const CompiledModel& compiled,
                                                     const TensorData& matrix,
                                                     uint64_t alignment,
                                                     uint32_t device_index)
{
#if NCNN_MOE_WITH_VULKAN
    if (Bfloat16Linear_vulkan::storage_bytes(matrix, alignment) == 0)
        return false;
    auto context = VulkanContext::acquire(device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    if (!context || context->device()->info.subgroup_size() != 32)
        return false;
    const uint64_t bytes = (matrix.element_count() + 1) / 2 * sizeof(uint32_t);
    return bytes <= context->device()->info.physicalDeviceProperties().limits.maxStorageBufferRange;
#else
    (void)compiled;
    (void)matrix;
    (void)alignment;
    (void)device_index;
    return false;
#endif
}

uint64_t dense_vulkan_weight_storage_size(const CompiledModel& compiled,
                                          const TensorData& matrix,
                                          uint64_t alignment,
                                          uint32_t device_index,
                                          bool has_bias) noexcept
{
    if (dense_qwen_bfloat16_projection_supported(compiled, matrix, alignment, device_index))
        return Bfloat16Linear_vulkan::storage_bytes(matrix, alignment);
#if NCNN_MOE_WITH_VULKAN
    if (matrix.shape.size() != 2 || matrix.shape[0] == 0 || matrix.shape[1] == 0
        || (matrix.dtype != DType::BFloat16 && matrix.dtype != DType::Float32)
        || matrix.element_count() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return 0;
    auto context = VulkanContext::acquire(device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    const uint64_t bytes = matrix.element_count() * sizeof(float);
    const uint64_t bias_bytes = has_bias ? static_cast<uint64_t>(matrix.shape[0]) * sizeof(float) : 0;
    if (!context || bytes > context->device()->info.physicalDeviceProperties().limits.maxStorageBufferRange
        || bias_bytes > context->device()->info.physicalDeviceProperties().limits.maxStorageBufferRange)
        return 0;
    alignment = std::max<uint64_t>(4, alignment);
    const uint64_t padding = bytes % alignment == 0 ? 0 : alignment - bytes % alignment;
    // Generic InnerProduct stores FP32 and owns the default 8 MiB allocator.
    uint64_t size = std::max<uint64_t>(UINT64_C(8) * 1024 * 1024, saturating_add_u64(bytes, padding));
    if (has_bias)
    {
        const uint64_t bias_padding = bias_bytes % alignment == 0 ? 0 : alignment - bias_bytes % alignment;
        size = saturating_add_u64(size, std::max<uint64_t>(UINT64_C(8) * 1024 * 1024, saturating_add_u64(bias_bytes, bias_padding)));
    }
    return size;
#else
    (void)has_bias;
    return 0;
#endif
}

static uint64_t dense_qwen_fused_storage_size(const CompiledModel& compiled,
                                              std::initializer_list<TensorHandle> handles,
                                              uint64_t alignment,
                                              uint32_t device_index) noexcept
{
    uint64_t rows = 0;
    uint32_t columns = 0;
    for (TensorHandle handle : handles)
    {
        if (handle == invalid_tensor_handle)
            continue;
        const TensorData& tensor = compiled.weights.at(handle);
        if (tensor.dtype != DType::BFloat16 || tensor.shape.size() != 2
            || (columns != 0 && columns != tensor.shape[1]))
            return 0;
        columns = tensor.shape[1];
        rows += tensor.shape[0];
    }
    if (rows > std::numeric_limits<uint32_t>::max())
        return 0;
    TensorData fused;
    fused.dtype = DType::BFloat16;
    fused.shape = {static_cast<uint32_t>(rows), columns};
    if (!dense_qwen_bfloat16_projection_supported(compiled, fused, alignment, device_index))
        return 0;
    return Bfloat16Linear_vulkan::storage_bytes(fused, alignment);
}

static uint64_t dense_qwen_constant_storage_size(std::initializer_list<uint64_t> buffers,
                                                 uint64_t alignment) noexcept
{
    alignment = std::max<uint64_t>(4, alignment);
    uint64_t size = 0;
    uint64_t remaining = 0;
    for (uint64_t bytes : buffers)
    {
        const uint64_t padding = bytes % alignment == 0 ? 0 : alignment - bytes % alignment;
        bytes = saturating_add_u64(bytes, padding);
        if (bytes > remaining)
        {
            remaining = std::max<uint64_t>(vulkan_weight_block_size, bytes);
            const uint64_t block_padding = remaining % alignment == 0 ? 0 : alignment - remaining % alignment;
            remaining = saturating_add_u64(remaining, block_padding);
            size = saturating_add_u64(size, remaining);
        }
        remaining -= bytes;
    }
    return size;
}

static uint64_t dense_qwen_gated_delta_state_size(const CompiledModel& compiled,
                                                  const AttentionBlockPlan& attention) noexcept
{
    const uint64_t convolution_columns = saturating_add_u64(saturating_multiply_u64(saturating_multiply_u64(attention.kv_head_count, attention.head_dimension), 2),
                                                            saturating_multiply_u64(attention.head_count, attention.value_head_dimension));
    const uint64_t convolution = saturating_multiply_u64(saturating_multiply_u64(convolution_columns, attention.convolution_kernel_size), sizeof(float));
    const uint64_t recurrent = saturating_multiply_u64(saturating_multiply_u64(attention.head_count, attention.head_dimension),
                                                       static_cast<uint64_t>(attention.value_head_dimension) * sizeof(float));
    // Cached decode keeps the live state and an initial transaction copy.
    return saturating_multiply_u64(saturating_add_u64(convolution, recurrent),
                                   static_cast<uint64_t>(std::max(1u, compiled.opt.num_concurrent_sessions)) * 2);
}

static uint64_t dense_standard_attention_initial_state_size(const CompiledModel& compiled,
                                                            const AttentionBlockPlan& attention) noexcept
{
    const uint64_t key_columns = saturating_multiply_u64(attention.kv_head_count,
                                                         attention.head_dimension);
    uint64_t required_tokens = std::max<uint64_t>(1, SessionOptions{}.prefill_chunk_size);
    if (has_flag(attention.flags, AttentionBlockSink))
        required_tokens = saturating_add_u64(required_tokens, 1);
    uint64_t capacity = 16;
    while (capacity < required_tokens
           && capacity <= static_cast<uint64_t>(std::numeric_limits<int>::max()) / 2)
    {
        capacity *= 2;
    }
    if (capacity < required_tokens)
        capacity = required_tokens;
    // Attention_vulkan inherits the QnK projection option, which currently
    // disables BF16 storage. Its device KV rings therefore use FP32 even when
    // the model declares a BF16 cache; reserve the actual upper-bound size.
    const uint64_t element_size = sizeof(float);
    // The Vulkan KV ring stores two copies of each key/value row so wrapped
    // cache views stay contiguous. Count every concurrent Session's initial
    // prefill capacity; later growth remains checked at allocation time.
    const uint64_t ring = saturating_multiply_u64(saturating_multiply_u64(capacity,
                                                                          key_columns),
                                                  element_size * 4);
    return saturating_multiply_u64(ring,
                                   std::max(1u, compiled.opt.num_concurrent_sessions));
}

uint64_t dense_vulkan_attention_storage_size(const CompiledModel& compiled,
                                             const CompiledLayerPlan& layer,
                                             uint64_t alignment,
                                             uint32_t device_index) noexcept
{
    const AttentionBlockPlan& attention = layer.attention;
    uint64_t input = 0;
    uint64_t constants = 0;
    uint64_t state = 0;
    if (attention.kind == AttentionKind::Standard)
    {
        if (standard_attention_requires_native(compiled, attention, device_index))
        {
            if (!standard_attention_native_compatible(compiled, attention))
                return 0;
            const TensorHandle weights[] = {
                attention.query_weight,
                attention.key_weight,
                attention.value_weight,
                attention.output_weight,
                attention.output_gate_weight,
            };
            for (TensorHandle handle : weights)
            {
                if (handle == invalid_tensor_handle)
                    continue;
                const TensorData& matrix = compiled.weights.at(handle);
                const uint64_t projection = matrix.dtype == DType::Float8E4M3
                                                ? dense_qwen_constant_storage_size({matrix.element_count(),
                                                                                    matrix.quantization_scales.size() * sizeof(float),
                                                                                    static_cast<uint64_t>(matrix.shape[0]) * sizeof(float)},
                                                                                   alignment)
                                                : QnkLinear_vulkan::storage_bytes(matrix, alignment);
                if (projection == 0)
                    return 0;
                input = saturating_add_u64(input, projection);
            }
            constants = dense_qwen_constant_storage_size({static_cast<uint64_t>(compiled.descriptor.hidden_size) * sizeof(float),
                                                          static_cast<uint64_t>(attention.head_dimension) * sizeof(float),
                                                          static_cast<uint64_t>(attention.head_dimension) * sizeof(float),
                                                          static_cast<uint64_t>(attention.head_dimension / 2) * sizeof(float),
                                                          static_cast<uint64_t>(attention.head_count) * sizeof(float)},
                                                         alignment);
            state = dense_standard_attention_initial_state_size(compiled, attention);
            return saturating_add_u64(saturating_add_u64(input, constants), state);
        }
        state = dense_standard_attention_initial_state_size(compiled, attention);
        input = dense_qwen_fused_storage_size(compiled,
                                              {attention.query_weight, attention.key_weight, attention.value_weight, attention.output_gate_weight}, alignment, device_index);
        // QKV biases prevent the combined query/key/value/gate projection.
        // Account its separate projections instead of charging one fusion.
        if (attention.query_bias != invalid_tensor_handle || attention.key_bias != invalid_tensor_handle
            || attention.value_bias != invalid_tensor_handle)
            input = 0;
        constants = dense_qwen_constant_storage_size({static_cast<uint64_t>(compiled.descriptor.hidden_size) * sizeof(float),
                                                      static_cast<uint64_t>(attention.head_dimension) * sizeof(float),
                                                      static_cast<uint64_t>(attention.head_dimension) * sizeof(float),
                                                      static_cast<uint64_t>(attention.head_dimension / 2) * sizeof(float),
                                                      static_cast<uint64_t>(attention.head_count) * sizeof(float)},
                                                     alignment);
    }
    else if (attention.kind == AttentionKind::GatedDeltaNet)
    {
        input = dense_qwen_fused_storage_size(compiled,
                                              {attention.delta_qkv_weight, attention.delta_z_weight, attention.delta_beta_weight, attention.delta_alpha_weight}, alignment, device_index);
        const uint64_t convolution = compiled.weights.at(attention.delta_convolution_weight).element_count() * sizeof(float);
        constants = dense_qwen_constant_storage_size({convolution,
                                                      static_cast<uint64_t>(attention.head_count) * sizeof(float),
                                                      static_cast<uint64_t>(attention.head_count) * sizeof(float),
                                                      static_cast<uint64_t>(attention.value_head_dimension) * sizeof(float)},
                                                     alignment);
        state = dense_qwen_gated_delta_state_size(compiled, attention);
    }
    else
        return 0;
    const TensorData& output_weight = compiled.weights.at(attention.output_weight);
    const uint64_t output = dense_qwen_bfloat16_projection_supported(compiled, output_weight, alignment, device_index)
                                ? Bfloat16Linear_vulkan::storage_bytes(output_weight, alignment)
                                : 0;
    if (input == 0 || output == 0)
    {
        if (attention.kind != AttentionKind::Standard)
            return 0;
        uint64_t projections = 0;
        const TensorHandle weights[] = {attention.query_weight, attention.key_weight, attention.value_weight,
                                        attention.output_weight, attention.output_gate_weight};
        const TensorHandle biases[] = {attention.query_bias, attention.key_bias, attention.value_bias,
                                       attention.output_bias, invalid_tensor_handle};
        for (size_t i = 0; i < 5; ++i)
        {
            if (weights[i] == invalid_tensor_handle)
                continue;
            const uint64_t size = dense_vulkan_weight_storage_size(compiled, compiled.weights.at(weights[i]), alignment, device_index,
                                                                   biases[i] != invalid_tensor_handle);
            if (size == 0)
                return 0;
            projections = saturating_add_u64(projections, size);
        }
        return projections;
    }
    return saturating_add_u64(saturating_add_u64(input, output), saturating_add_u64(constants, state));
}

uint64_t dense_vulkan_pending_state_size(const CompiledModel& compiled,
                                         uint32_t device_index) noexcept
{
    uint64_t size = 0;
    for (const ExecutionGraph* graph : {&compiled.graph, &compiled.speculative.graph})
    {
        for (const CompiledLayerPlan& layer : graph->layer_plans)
        {
            const AttentionBlockPlan& attention = layer.attention;
            if (layer.vulkan_device_index != device_index)
                continue;
            if (attention.kind == AttentionKind::GatedDeltaNet
                && compiled.operators.at(attention.gated_delta_vulkan_operator).gated_delta)
            {
                size = saturating_add_u64(size, dense_qwen_gated_delta_state_size(compiled, attention));
            }
            else if (attention.kind == AttentionKind::Standard
                     && !has_flag(attention.flags, AttentionBlockQsa)
                     && !has_flag(attention.flags, AttentionBlockExternalResidual)
                     && attention.vulkan_attention_operator != invalid_compiled_operator_handle
                     && compiled.operators.at(attention.vulkan_attention_operator).attention)
            {
                size = saturating_add_u64(size, dense_standard_attention_initial_state_size(compiled, attention));
            }
        }
    }
    return size;
}

uint64_t dense_vulkan_workspace_reserve(uint64_t heap_budget) noexcept
{
    return std::min(heap_budget / 4, std::max<uint64_t>(UINT64_C(256) * 1024 * 1024, heap_budget / 16));
}

static uint64_t dense_qwen_dynamic_cache_reserve(const CompiledModel& compiled,
                                                 const CompilerOption& opt) noexcept
{
    uint64_t largest = 0;
    for (const CompiledLayerPlan& layer : compiled.graph.layer_plans)
    {
        if (layer.dense_ffn.use_gpu_cache)
            largest = std::max(largest, layer.dense_ffn.weight_size);
    }
    if (largest == 0)
        return 0;
    if (opt.gpu_expert_cache_reserve != 0)
        return std::max(largest, opt.gpu_expert_cache_reserve);
    // Automatic caches retain at least 85% of the remaining heap. Preserve
    // enough free space for one complete FFN after applying that policy.
    return saturating_add_u64(saturating_multiply_u64(largest / 85, 100),
                              (largest % 85 * 100 + 84) / 85);
}

static uint64_t dense_qwen_gpu_alignment(const CompiledModel& compiled,
                                         uint32_t device_index)
{
#if NCNN_MOE_WITH_VULKAN
    auto context = VulkanContext::acquire(device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    if (context)
        return std::max<uint64_t>(4, context->device()->info.buffer_offset_alignment());
#else
    (void)compiled;
    (void)device_index;
#endif
    return 0;
}

static bool dense_qwen_can_admit(const CompiledModel& compiled, const CompilerOption& opt,
                                 uint32_t device_index, uint64_t size, uint64_t planned)
{
    const uint64_t reserve = saturating_add_u64(dense_vulkan_workspace_reserve(opt.gpu_heap_budget),
                                                dense_qwen_dynamic_cache_reserve(compiled, opt));
    if (reserve >= opt.gpu_heap_budget)
        return false;
    const uint64_t capacity = opt.gpu_heap_budget - reserve;
    if (size == 0 || size > capacity || planned > capacity - size)
        return false;
#if NCNN_MOE_WITH_VULKAN
    auto context = VulkanContext::acquire(device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    if (!context)
        return false;
    const uint64_t budget = static_cast<uint64_t>(context->device()->get_heap_budget()) * 1024 * 1024;
    const uint64_t usage = static_cast<uint64_t>(context->device()->get_heap_usage()) * 1024 * 1024;
    // State is allocated on the first forward, so the live heap does not yet
    // include it. Standard KV rings are reserved across the model; a candidate
    // Gated DeltaNet group's state is included in its own size.
    const uint64_t live_reserve = saturating_add_u64(saturating_add_u64(dense_vulkan_workspace_reserve(budget), dense_qwen_dynamic_cache_reserve(compiled, opt)),
                                                     dense_vulkan_pending_state_size(compiled, device_index));
    return usage < budget && live_reserve < budget - usage && size <= budget - usage - live_reserve;
#else
    (void)compiled;
    (void)device_index;
    return false;
#endif
}

static void clear_dense_qwen_group_operators(CompiledModel& compiled,
                                             std::initializer_list<TensorHandle> weights,
                                             size_t first_operator)
{
    for (TensorHandle handle : weights)
    {
        if (handle != invalid_tensor_handle)
            compiled.operators.at_weight_mutable(handle) = {};
    }
    for (size_t handle = first_operator; handle < compiled.operators.size(); ++handle)
        compiled.operators.at_mutable(static_cast<CompiledOperatorHandle>(handle)) = {};
}

static void clear_dense_qwen_attention_operators(CompiledModel& compiled,
                                                 AttentionBlockPlan& attention)
{
    for (TensorHandle handle : {attention.query_weight, attention.key_weight, attention.value_weight, attention.output_weight,
                                attention.output_gate_weight, attention.delta_qkv_weight, attention.delta_z_weight,
                                attention.delta_beta_weight, attention.delta_alpha_weight})
    {
        if (handle != invalid_tensor_handle)
            compiled.operators.at_weight_mutable(handle) = {};
    }
    for (CompiledOperatorHandle handle : {attention.vulkan_attention_operator, attention.fused_qkv_operator,
                                          attention.fused_qkv_gate_bfloat16_operator, attention.fused_delta_input_operator,
                                          attention.gated_delta_vulkan_operator})
    {
        if (handle != invalid_compiled_operator_handle)
            compiled.operators.at_mutable(handle) = {};
    }
    attention.vulkan_attention_operator = invalid_compiled_operator_handle;
    attention.fused_qkv_operator = invalid_compiled_operator_handle;
    attention.fused_qkv_gate_bfloat16_operator = invalid_compiled_operator_handle;
    attention.fused_delta_input_operator = invalid_compiled_operator_handle;
    attention.gated_delta_vulkan_operator = invalid_compiled_operator_handle;
}

static bool dense_qwen_dynamic_cache_fits(const CompiledModel& compiled,
                                          const CompilerOption& opt,
                                          uint32_t device_index)
{
#if NCNN_MOE_WITH_VULKAN
    const uint64_t cache_reserve = dense_qwen_dynamic_cache_reserve(compiled, opt);
    if (cache_reserve == 0)
        return true;
    auto context = VulkanContext::acquire(device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    if (!context)
        return true;
    const uint64_t budget = static_cast<uint64_t>(context->device()->get_heap_budget()) * 1024 * 1024;
    const uint64_t reserve = saturating_add_u64(dense_vulkan_workspace_reserve(budget), cache_reserve);
    // An entry that cannot fit even on an empty heap must not evict every
    // useful small projection in pursuit of an impossible cache.
    if (reserve >= std::min(budget, opt.gpu_heap_budget))
        return true;
    const uint64_t usage = static_cast<uint64_t>(context->device()->get_heap_usage()) * 1024 * 1024;
    const uint64_t required = saturating_add_u64(reserve, dense_vulkan_pending_state_size(compiled, device_index));
    return usage < budget && required <= budget - usage;
#else
    (void)compiled;
    (void)opt;
    (void)device_index;
    return true;
#endif
}

static Result<void> preserve_dense_qwen_dynamic_cache(CompiledModel& compiled,
                                                      const CompilerOption& opt)
{
    std::vector<uint32_t> devices;
    for (const CompiledLayerPlan& layer : compiled.graph.layer_plans)
    {
        if (layer.dense_ffn.use_gpu_cache
            && std::find(devices.begin(), devices.end(), layer.vulkan_device_index) == devices.end())
            devices.push_back(layer.vulkan_device_index);
    }
    for (uint32_t device_index : devices)
    {
        if (dense_qwen_dynamic_cache_fits(compiled, opt, device_index))
            continue;
        const bool has_required_standard_attention = std::any_of(compiled.graph.layer_plans.begin(),
                                                                 compiled.graph.layer_plans.end(),
                                                                 [&](const CompiledLayerPlan& layer) {
                                                                     return layer.vulkan_device_index == device_index
                                                                            && layer.attention.kind == AttentionKind::Standard
                                                                            && support_vulkan_attention(compiled.operators,
                                                                                                        layer.attention);
                                                                 });
        // Driver allocations and heap-budget changes can exceed the planned
        // storage estimate. Recheck actual usage after releasing each group.
        for (auto layer = compiled.graph.layer_plans.rbegin(); layer != compiled.graph.layer_plans.rend(); ++layer)
        {
            if (layer->vulkan_device_index != device_index
                || layer->attention.kind == AttentionKind::Standard
                || !support_vulkan_attention(compiled.operators, layer->attention))
                continue;
            clear_dense_qwen_attention_operators(compiled, layer->attention);
            if (dense_qwen_dynamic_cache_fits(compiled, opt, device_index))
                break;
        }
        if (!dense_qwen_dynamic_cache_fits(compiled, opt, device_index)
            && compiled.opt.vulkan_device_index == device_index)
            compiled.operators.at_weight_mutable(compiled.lm_head_weight) = {};
        if (!dense_qwen_dynamic_cache_fits(compiled, opt, device_index))
        {
            if (!has_required_standard_attention)
                continue;
            return Error{ErrorCode::InvalidArgument,
                         "GPU heap budget cannot retain required Vulkan Standard Attention state and the dense FFN cache on device "
                             + std::to_string(device_index)};
        }
    }
    return {};
}

static Result<void> prepare_dense_qwen_projection(CompiledModel& compiled,
                                                  const CompilerOption& opt,
                                                  TensorHandle weight, TensorHandle bias,
                                                  uint32_t device_index, uint64_t alignment,
                                                  uint64_t& planned)
{
    const TensorData& matrix = compiled.weights.at(weight);
    const uint64_t size = dense_vulkan_weight_storage_size(compiled, matrix, alignment, device_index, bias != invalid_tensor_handle);
    if (!dense_qwen_can_admit(compiled, opt, device_index, size, planned))
        return {};
    auto ret = prepare_vulkan_linear_operator(compiled.weights, compiled.operators, weight, bias,
                                              device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags, 1,
                                              dense_qwen_bfloat16_projection_supported(compiled, matrix, alignment, device_index), false);
    if (!ret)
    {
        compiled.operators.at_weight_mutable(weight) = {};
        if (ret.error().code != ErrorCode::InternalError)
            return ret.error();
        return {};
    }
    planned = saturating_add_u64(planned, size);
    return {};
}

static Result<void> prepare_bounded_dense_qwen_pipeline(CompiledModel& compiled,
                                                        const CompilerOption& opt)
{
    if (opt.gpu_heap_budget == 0)
        return {};
    uint64_t planned = 0;
    const uint64_t head_alignment = dense_qwen_gpu_alignment(compiled, compiled.opt.vulkan_device_index);
    const TensorData& head = compiled.weights.at(compiled.lm_head_weight);
    const uint64_t head_size = dense_vulkan_weight_storage_size(compiled, head, head_alignment, compiled.opt.vulkan_device_index);
    if (head_alignment != 0 && dense_qwen_can_admit(compiled, opt, compiled.opt.vulkan_device_index, head_size, planned))
    {
        auto ret = prepare_lm_head_operator(compiled, false,
                                            dense_qwen_bfloat16_projection_supported(compiled, head, head_alignment, compiled.opt.vulkan_device_index));
        if (!ret)
        {
            compiled.operators.at_weight_mutable(compiled.lm_head_weight) = {};
            if (ret.error().code != ErrorCode::InternalError)
                return ret.error();
        }
        else
            planned += head_size;
    }

    // Reserve attention groups across the model before admitting large FFNs.
    if (has_flag(opt.flags, BackendVulkanAttention)
        && has_flag(compiled.opt.optimization_flags, OptimizationVulkanAttention))
    {
        for (CompiledLayerPlan& layer : compiled.graph.layer_plans)
        {
            AttentionBlockPlan& attention = layer.attention;
            if (attention.kind == AttentionKind::Standard
                && (has_flag(attention.flags, AttentionBlockQsa)
                    || has_flag(attention.flags, AttentionBlockExternalResidual)))
            {
                // These modes use the dedicated Qwen path or CPU execution;
                // the generic Standard Attention operator does not implement
                // their indexing/residual semantics.
                continue;
            }
            if (standard_attention_requires_native(compiled, attention, layer.vulkan_device_index)
                && !has_flag(compiled.opt.optimization_flags, OptimizationVulkanQnK))
            {
                continue;
            }
            const uint64_t alignment = dense_qwen_gpu_alignment(compiled, layer.vulkan_device_index);
            if (alignment == 0)
                continue;
            const uint64_t size = dense_vulkan_attention_storage_size(compiled, layer, alignment, layer.vulkan_device_index);
            if (attention.kind == AttentionKind::Standard)
            {
                if (size == 0)
                {
                    return Error{ErrorCode::UnsupportedModel,
                                 "Vulkan Standard Attention has no supported storage plan at layer "
                                     + std::to_string(layer.layer_id)};
                }
                if (!dense_qwen_can_admit(compiled, opt, layer.vulkan_device_index, size, planned))
                {
                    return Error{ErrorCode::InvalidArgument,
                                 "GPU heap budget cannot admit required Vulkan Standard Attention at layer "
                                     + std::to_string(layer.layer_id)};
                }
                Result<void> ret;
                ret = prepare_standard_attention_operators(compiled, layer, false);
                if (ret && !support_vulkan_attention(compiled.operators, attention))
                {
                    ret = Error{ErrorCode::InternalError,
                                "failed to create required Vulkan Standard Attention at layer "
                                    + std::to_string(layer.layer_id)};
                }
                if (!ret)
                {
                    clear_dense_qwen_attention_operators(compiled, attention);
                    return ret.error();
                }
                planned = saturating_add_u64(planned, size);
                continue;
            }
            if (attention.kind == AttentionKind::GatedDeltaNet
                && size != 0
                && dense_qwen_can_admit(compiled, opt, layer.vulkan_device_index, size, planned))
            {
                Result<void> ret;
                prepare_gated_delta_attention_operators(compiled, layer);
                if (!support_vulkan_attention(compiled.operators, attention))
                    ret = Error{ErrorCode::InternalError, "failed to create Vulkan dense DeltaNet operator"};
                if (ret)
                {
                    planned = saturating_add_u64(planned, size);
                    continue;
                }
                clear_dense_qwen_attention_operators(compiled, attention);
                if (ret.error().code != ErrorCode::InternalError)
                    return ret.error();
            }
        }
    }

    for (CompiledLayerPlan& layer : compiled.graph.layer_plans)
    {
        if (layer.dense_ffn.use_gpu_cache)
            continue;
        const uint64_t alignment = dense_qwen_gpu_alignment(compiled, layer.vulkan_device_index);
        if (alignment == 0)
            continue;
        ExpertPlan& ffn = layer.dense_ffn;
        const uint64_t input_size = dense_qwen_fused_storage_size(compiled, {ffn.gate_weight, ffn.up_weight}, alignment, layer.vulkan_device_index);
        const TensorData& down = compiled.weights.at(ffn.down_weight);
        const uint64_t down_size = dense_qwen_bfloat16_projection_supported(compiled, down, alignment, layer.vulkan_device_index)
                                       ? Bfloat16Linear_vulkan::storage_bytes(down, alignment)
                                       : 0;
        const uint64_t size = input_size == 0 || down_size == 0 ? 0 : saturating_add_u64(input_size, down_size);
        if (dense_qwen_can_admit(compiled, opt, layer.vulkan_device_index, size, planned))
        {
            const size_t first_operator = compiled.operators.size();
            auto ret = prepare_ffn_operators(compiled.weights, compiled.operators, ffn,
                                             layer.vulkan_device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags, false);
            if (ret)
            {
                planned = saturating_add_u64(planned, size);
                continue;
            }
            clear_dense_qwen_group_operators(compiled, {ffn.gate_weight, ffn.up_weight, ffn.down_weight}, first_operator);
            ffn.vulkan_operator = invalid_compiled_operator_handle;
            if (ret.error().code != ErrorCode::InternalError)
                return ret.error();
        }
        for (TensorHandle handle : {ffn.gate_weight, ffn.up_weight, ffn.down_weight})
        {
            auto ret = prepare_dense_qwen_projection(compiled, opt, handle, invalid_tensor_handle,
                                                     layer.vulkan_device_index, alignment, planned);
            if (!ret)
                return ret.error();
        }
    }
    auto cache_status = preserve_dense_qwen_dynamic_cache(compiled, opt);
    if (!cache_status)
        return cache_status.error();
    return {};
}

static bool use_qwen_device_pipeline(const CompiledModel& compiled, const CompilerOption& opt) noexcept
{
    return compiled.opt.hybrid_mode == HybridMode::HybridExperts
           && compiled.descriptor.hyper_connection_kind == HyperConnectionKind::GatedResidual
           && has_flag(opt.flags, BackendVulkanDense)
           && has_flag(opt.flags, BackendVulkanAttention)
           && has_flag(opt.flags, BackendVulkanExperts)
           && has_flag(opt.optimization_flags, OptimizationVulkanAttention)
           && has_flag(opt.optimization_flags, OptimizationVulkanQnK)
           && has_flag(opt.optimization_flags, OptimizationVulkanExpertGpuPriority)
           && has_flag(opt.optimization_flags, OptimizationVulkanCommandGraph);
}

static uint64_t qwen_pending_reserve(const CompiledModel& compiled, uint32_t device_index) noexcept
{
    const auto& descriptor = compiled.descriptor;
    const uint64_t hidden = descriptor.hidden_size;
    const uint64_t copies = descriptor.hyper_connection_multiplier;
    const uint64_t expanded = saturating_multiply_u64(hidden, copies);
    const uint64_t rank = descriptor.hyper_connection_low_rank;
    const uint64_t concurrency = std::max(1u, compiled.opt.num_concurrent_sessions);
    const uint64_t rows = std::max<uint64_t>(1, SessionOptions{}.prefill_chunk_size);
    uint64_t capacity = 1;
    while (capacity < rows)
        capacity = saturating_multiply_u64(capacity, 2);

    uint64_t peak_layer_elements = 0;
    uint64_t persistent_states = 0;
    uint64_t peak_initial_state = 0;
    for (const CompiledLayerPlan& layer : compiled.graph.layer_plans)
    {
        if (layer.vulkan_device_index != device_index)
            continue;
        const auto& attention = layer.attention;
        // Both GatedResidual pre calls retain normalized/mixed/projected/
        // activated buffers until submission completes: 4E + 4R + 2H + 4M.
        uint64_t elements = saturating_add_u64(saturating_multiply_u64(expanded, 4),
                                               saturating_add_u64(saturating_multiply_u64(rank, 4),
                                                                  saturating_add_u64(saturating_multiply_u64(hidden, 2), saturating_multiply_u64(copies, 4))));
        // Previous combine outputs/residual, the original and device-cloned
        // layer input, this layer's residual, and retained normalized/injection
        // vectors overlap until the command completes.
        elements = saturating_add_u64(elements,
                                      saturating_add_u64(saturating_multiply_u64(expanded, 4),
                                                         saturating_add_u64(saturating_multiply_u64(hidden, static_cast<uint64_t>(layer.moe.top_k) + 3), copies)));
        const uint64_t expert_intermediate = saturating_multiply_u64(layer.moe.top_k,
                                                                     saturating_multiply_u64(descriptor.intermediate_size, 2));
        // Router projection/logits/selection-bias and compact ids/weights/status.
        const uint64_t router = saturating_add_u64(saturating_multiply_u64(layer.moe.experts.size(), 3),
                                                   saturating_add_u64(saturating_multiply_u64(layer.moe.top_k, 3), 1));
        elements = saturating_add_u64(elements, saturating_add_u64(expert_intermediate, router));
        if (attention.kind == AttentionKind::GatedDeltaNet)
        {
            const uint64_t key = saturating_multiply_u64(attention.kv_head_count, attention.head_dimension);
            const uint64_t value = saturating_multiply_u64(attention.head_count, attention.value_head_dimension);
            // Fused input, recurrent output, and output projection are live
            // simultaneously while the GDN work ticket is pending.
            elements = saturating_add_u64(elements,
                                          saturating_add_u64(saturating_multiply_u64(key, 2),
                                                             saturating_add_u64(saturating_multiply_u64(value, 3),
                                                                                saturating_add_u64(saturating_multiply_u64(attention.head_count, 2), hidden))));
            const uint64_t convolution = saturating_multiply_u64(saturating_add_u64(saturating_multiply_u64(key, 2), value),
                                                                 attention.convolution_kernel_size);
            const uint64_t recurrent = saturating_multiply_u64(saturating_multiply_u64(attention.head_count, attention.head_dimension),
                                                               attention.value_head_dimension);
            const uint64_t state = saturating_multiply_u64(saturating_add_u64(convolution, recurrent), sizeof(float));
            persistent_states = saturating_add_u64(persistent_states, state);
            // Fresh initialization additionally retains its zero/migration
            // upload until the first command completes; layers execute in order.
            peak_initial_state = std::max(peak_initial_state, state);
        }
        else if (attention.kind == AttentionKind::Standard)
        {
            const uint64_t query = saturating_multiply_u64(attention.head_count, attention.head_dimension);
            const uint64_t key = saturating_multiply_u64(attention.kv_head_count, attention.head_dimension);
            const uint64_t index = saturating_multiply_u64(static_cast<uint64_t>(attention.index_head_count) + 1,
                                                           attention.index_head_dimension);
            const uint64_t ratio = std::max<uint64_t>(1, attention.compression_ratio);
            const uint64_t blocks = capacity / ratio + (capacity % ratio != 0);
            elements = saturating_add_u64(elements,
                                          saturating_add_u64(saturating_multiply_u64(query, 4),
                                                             saturating_add_u64(saturating_multiply_u64(key, 3),
                                                                                saturating_add_u64(saturating_multiply_u64(index, 2), saturating_add_u64(hidden, blocks + 2)))));
            const uint64_t kv_element_size = attention.kv_cache_dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
            const uint64_t kv = saturating_multiply_u64(saturating_multiply_u64(capacity, key), kv_element_size * 2);
            const uint64_t index_state = saturating_multiply_u64(saturating_multiply_u64(capacity + std::max<uint64_t>(1, blocks),
                                                                                         attention.index_head_dimension),
                                                                 sizeof(float));
            uint64_t state = saturating_add_u64(kv, index_state);
            if (attention.vulkan_attention_operator != invalid_compiled_operator_handle)
            {
                const auto& qsa = compiled.operators.at(attention.vulkan_attention_operator).qsa_attention;
                if (qsa)
                    state = std::max(state, qsa->initial_state_bytes());
            }
            // A growing QSA cache keeps its old prefix and replacement buffers
            // alive together through the fence. Reserve two default chunks,
            // rather than the model's maximum context length.
            persistent_states = saturating_add_u64(persistent_states, saturating_multiply_u64(state, 2));
        }
        if (layer.ple.enabled())
        {
            // Packed selected rows (bounded by FP32 embedding size), decoded
            // embeddings, key/value and six expanded norm/gate/conv vectors.
            elements = saturating_add_u64(elements,
                                          saturating_add_u64(saturating_multiply_u64(expanded, 6),
                                                             saturating_add_u64(saturating_multiply_u64(layer.ple.embedding_dimension, 2), hidden)));
            const uint64_t history = saturating_multiply_u64(layer.ple.convolution_kernel_size - 1, layer.ple.ngram_size);
            persistent_states = saturating_add_u64(persistent_states,
                                                   saturating_multiply_u64(saturating_multiply_u64(history, expanded), sizeof(float)));
        }
        peak_layer_elements = std::max(peak_layer_elements, elements);
    }
    uint64_t workspace = saturating_multiply_u64(saturating_multiply_u64(peak_layer_elements, rows), sizeof(float));
    if (device_index == compiled.opt.vulkan_device_index)
    {
        // Ordinary prefill requests only the final row's logits. Its head
        // temporaries can coexist with the final layer's retained activations.
        const uint64_t head = saturating_add_u64(descriptor.vocabulary_size,
                                                 saturating_add_u64(saturating_multiply_u64(expanded, 2),
                                                                    saturating_add_u64(saturating_multiply_u64(rank, 2), saturating_multiply_u64(hidden, 2))));
        workspace = saturating_add_u64(workspace, saturating_multiply_u64(head, sizeof(float)));
    }
    // Each Expert allocator enforces a physical allocation ceiling at logical
    // cache capacity plus at most 64 MiB. Fragmentation can exhaust that ceiling
    // before the logical cache fills; it cannot consume the execution reserve.
    const uint64_t expert_allocator_slack = (compiled.opt.expert_gpu_victim_cache_size != 0 ? 2 : 1) * UINT64_C(64) * 1024 * 1024;
    return saturating_add_u64(expert_allocator_slack,
                              saturating_multiply_u64(saturating_add_u64(persistent_states,
                                                                         saturating_add_u64(peak_initial_state, workspace)),
                                                      concurrency));
}

uint64_t vulkan_pending_reserve(const CompiledModel& compiled,
                                uint32_t device_index,
                                uint64_t heap_budget) noexcept
{
    if (compiled.qwen_device_chain)
        return qwen_pending_reserve(compiled, device_index);

    bool has_attention_operator = false;
    for (const ExecutionGraph* graph : {&compiled.graph, &compiled.speculative.graph})
    {
        for (const CompiledLayerPlan& layer : graph->layer_plans)
        {
            if (layer.vulkan_device_index != device_index)
                continue;
            if (layer.attention.kind == AttentionKind::Standard
                && layer.attention.vulkan_attention_operator != invalid_compiled_operator_handle
                && compiled.operators.at(layer.attention.vulkan_attention_operator).attention)
            {
                has_attention_operator = true;
                break;
            }
        }
    }
    if (!has_attention_operator)
        return 0;

    return saturating_add_u64(dense_vulkan_workspace_reserve(heap_budget),
                              dense_vulkan_pending_state_size(compiled, device_index));
}

static uint64_t qwen_minimum_expert_reserve(const CompiledModel& compiled, uint32_t device_index)
{
    uint64_t required = 0;
    for (const CompiledLayerPlan& layer : compiled.graph.layer_plans)
    {
        if (layer.vulkan_device_index != device_index)
            continue;
        uint64_t largest = 0;
        for (const ExpertPlan& expert : layer.moe.experts)
        {
            const TensorHandle gate_handle = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight;
            if (gate_handle == invalid_tensor_handle || expert.down_weight == invalid_tensor_handle)
                continue;
            const TensorData* up = expert.gate_up_weight == invalid_tensor_handle && expert.up_weight != invalid_tensor_handle
                                       ? &compiled.weights.at(expert.up_weight)
                                       : nullptr;
            uint64_t bytes = native_quantized_expert_storage_bytes(compiled.weights.at(gate_handle), up,
                                                                   compiled.weights.at(expert.down_weight), device_index,
                                                                   compiled.vulkan_runtime, compiled.opt.optimization_flags);
            // Non-native Expert backends still reserve their declared complete
            // pair, rather than discounting a CPU/file-backed representation.
            if (bytes == 0)
                bytes = expert.weight_size;
            largest = std::max(largest, bytes);
        }
        required = std::max(required, saturating_multiply_u64(largest, layer.moe.top_k));
    }
    return saturating_multiply_u64(required, std::max(1u, compiled.opt.num_concurrent_sessions));
}

static Result<void> check_qwen_device_budget(const CompiledModel& compiled, const CompilerOption& opt,
                                             uint32_t device_index, uint64_t minimum_experts)
{
#if NCNN_MOE_WITH_VULKAN
    auto context = VulkanContext::acquire(device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    if (!context)
        return Error{ErrorCode::UnsupportedModel, "Qwen device execution requires an available Vulkan context"};
    const uint64_t live_budget = static_cast<uint64_t>(context->device()->get_heap_budget()) * 1024 * 1024;
    const uint64_t usage = static_cast<uint64_t>(context->device()->get_heap_usage()) * 1024 * 1024;
    const uint64_t budget = opt.gpu_heap_budget == 0 ? live_budget : std::min(live_budget, opt.gpu_heap_budget);
    // The live query includes all persistent operator allocations, including
    // allocator block/slack bytes. Only state/work and the still-unallocated
    // executable Expert cache must be added; do not charge weights twice.
    const uint64_t pending = qwen_pending_reserve(compiled, device_index);
    const uint64_t experts = std::max(minimum_experts, opt.gpu_expert_cache_reserve);
    const uint64_t reserve = saturating_add_u64(pending, experts);
    if (budget == 0 || usage >= budget || reserve > budget - usage)
        return Error{ErrorCode::InvalidArgument, "Qwen Vulkan persistent weights and minimum working set exceed the GPU heap budget on device "
                                                     + std::to_string(device_index) + " (usage=" + std::to_string(usage)
                                                     + ", pending=" + std::to_string(pending) + ", experts=" + std::to_string(experts)
                                                     + ", budget=" + std::to_string(budget) + ")"};
    return {};
#else
    (void)compiled;
    (void)opt;
    (void)device_index;
    (void)minimum_experts;
    return Error{ErrorCode::UnsupportedModel, "Qwen device execution requires Vulkan"};
#endif
}

static Result<void> prepare_qwen_device_pipeline(CompiledModel& compiled, const CompilerOption& opt)
{
    for (TensorHandle handle = 0; handle < compiled.weights.size(); ++handle)
        if (compiled.weights.at(handle).dtype == DType::Float8E4M3)
            return Error{ErrorCode::UnsupportedModel, "continuous Qwen Vulkan execution does not support FP8 model weights"};
    for (const auto& layer : compiled.graph.layer_plans)
    {
        if (layer.vulkan_device_index != compiled.opt.vulkan_device_index)
            return Error{ErrorCode::UnsupportedModel, "Qwen device execution requires all layers and the output head on one Vulkan device"};
    }
    std::map<uint32_t, uint64_t> expert_reserves;
    expert_reserves.emplace(compiled.opt.vulkan_device_index, 0);
    for (const auto& layer : compiled.graph.layer_plans)
        expert_reserves.emplace(layer.vulkan_device_index, 0);
    for (auto& [device, reserve] : expert_reserves)
    {
        reserve = qwen_minimum_expert_reserve(compiled, device);
        auto checked = check_qwen_device_budget(compiled, opt, device, reserve);
        if (!checked)
            return checked.error();
    }
    auto head = QnkLinear_vulkan::create(compiled.weights.at(compiled.lm_head_weight), nullptr,
                                         compiled.opt.vulkan_device_index, compiled.vulkan_runtime,
                                         compiled.opt.optimization_flags);
    if (!head)
        return Error{ErrorCode::InternalError, "failed to create the required native Vulkan Qwen LM head"};
    compiled.operators.at_weight_mutable(compiled.lm_head_weight).qnk = std::move(head);
    compiled.qwen_device_head = QwenLayer_vulkan::create_head(compiled);
    if (!compiled.qwen_device_head)
        return Error{ErrorCode::InternalError, "failed to create the required Vulkan Qwen output head"};
    for (CompiledLayerPlan& layer : compiled.graph.layer_plans)
    {
        auto& attention = layer.attention;
        if (attention.kind == AttentionKind::GatedDeltaNet)
        {
            prepare_gated_delta_attention_operators(compiled, layer);
            if (!support_vulkan_attention(compiled.operators, attention))
                return Error{ErrorCode::InternalError, "failed to create required Vulkan Gated DeltaNet at layer " + std::to_string(layer.layer_id)};
        }
        else if (attention.kind == AttentionKind::Standard)
        {
            auto qsa = QsaAttention_vulkan::create(compiled.weights, attention, layer.vulkan_device_index,
                                                   compiled.vulkan_runtime, compiled.opt.optimization_flags);
            if (!qsa)
                return Error{ErrorCode::UnsupportedModel, "failed to create required Vulkan QSA at layer " + std::to_string(layer.layer_id)};
            const auto handle = compiled.operators.allocate();
            compiled.operators.at_mutable(handle).qsa_attention = std::move(qsa);
            attention.vulkan_attention_operator = handle;
        }
        else
            return Error{ErrorCode::UnsupportedModel, "Qwen device execution requires Gated DeltaNet or QSA attention"};
        if (layer.moe.has_shared_expert)
        {
            auto& plan = layer.moe.shared_expert;
            auto shared = QnkExpert_vulkan::create_separate(compiled.weights.at(plan.gate_weight), nullptr,
                                                            compiled.weights.at(plan.up_weight), nullptr,
                                                            compiled.weights.at(plan.down_weight), nullptr,
                                                            plan.activation_limit, layer.vulkan_device_index, plan.activation,
                                                            compiled.vulkan_runtime, compiled.opt.optimization_flags);
            if (!shared || (layer.moe.shared_expert_gate_weight != invalid_tensor_handle && !shared->prepare_router_gate(compiled.weights.at(layer.moe.shared_expert_gate_weight))))
                return Error{ErrorCode::InternalError, "failed to create required Vulkan Qwen shared Expert at layer " + std::to_string(layer.layer_id)};
            const auto handle = compiled.operators.allocate();
            compiled.operators.at_mutable(handle).qnk_expert = std::move(shared);
            plan.vulkan_operator = handle;
        }
        auto operation = QwenLayer_vulkan::create(compiled, layer);
        if (!operation)
            return Error{ErrorCode::InternalError, "failed to create required Vulkan Qwen device layer " + std::to_string(layer.layer_id)};
        compiled.operators.at_weight_mutable(attention.output_weight).qwen_layer = std::move(operation);
        auto checked = check_qwen_device_budget(compiled, opt, layer.vulkan_device_index, expert_reserves.at(layer.vulkan_device_index));
        if (!checked)
            return checked.error();
    }
    for (const auto& [device, reserve] : expert_reserves)
    {
        auto checked = check_qwen_device_budget(compiled, opt, device, reserve);
        if (!checked)
            return checked.error();
    }
    compiled.qwen_device_chain = true;
    return {};
}

Result<void> prepare_model_pipeline(CompiledModel& compiled,
                                    const CompilerOption& opt)
{
    // Descriptor validation and tensor-plan binding are complete before this
    // backend pass, so malformed-model errors precede GPU object creation errors.
    compiled.operators.bind_weight_count(compiled.weights.size());
    if (compiled.opt.hybrid_mode == HybridMode::CpuOnly)
        return {};
    if (use_qwen_device_pipeline(compiled, opt))
        return prepare_qwen_device_pipeline(compiled, opt);
    if (use_bounded_dense_qwen_pipeline(compiled))
        return prepare_bounded_dense_qwen_pipeline(compiled, opt);

    Result<void> ret = prepare_lm_head_operator(compiled);
    if (!ret)
        return ret.error();

    if (compiled.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn
        && compiled.descriptor.final_norm == NormType::RmsNorm
        && compiled.descriptor.norm_weight_offset == 0.0f
        && !has_flag(compiled.opt.flags, OptionRouterPrediction))
    {
        auto head = LayerHead_vulkan::create(compiled.weights.at(compiled.hyper_head_function),
                                             compiled.weights.at(compiled.hyper_head_scale),
                                             compiled.weights.at(compiled.hyper_head_base), compiled.weights.at(compiled.final_norm_weight),
                                             compiled.operators.at_weight(compiled.lm_head_weight), compiled.descriptor.hyper_connection_multiplier,
                                             compiled.descriptor.norm_epsilon, compiled.descriptor.hyper_connection_epsilon,
                                             compiled.opt.vulkan_device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
        compiled.operators.at_weight_mutable(compiled.hyper_head_function).layer_head = std::move(head);
    }

    uint64_t planned_gated_delta_gpu_size = 0;

    for (size_t layer_id = 0; layer_id < compiled.graph.layer_plans.size(); ++layer_id)
    {
        CompiledLayerPlan& layer_plan = compiled.graph.layer_plans[layer_id];
        const LayerDescriptor& layer = compiled.descriptor.layers[layer_id];
        const AttentionDescriptor& attention = layer.attention;
        if (attention.kind != AttentionKind::None)
        {
            if (attention.kind == AttentionKind::Standard
                && has_flag(opt.flags, BackendVulkanAttention))
            {
                ret = prepare_standard_attention_operators(compiled,
                                                           layer_plan);
            }
            else if (attention.kind == AttentionKind::MultiHeadLatent)
            {
                ret = prepare_latent_attention_operators(compiled,
                                                         layer_plan, "", has_flag(opt.flags, BackendVulkanAttention));
            }
            else if (attention.kind == AttentionKind::GatedDeltaNet)
            {
                const bool vulkan_delta_fusion_available = has_flag(opt.flags, BackendVulkanAttention)
                                                           && has_flag(compiled.opt.optimization_flags,
                                                                       OptimizationVulkanAttention);
                const bool protect_file_backed_experts = has_flag(opt.flags, BackendVulkanExperts)
                                                         && has_flag(opt.flags, BackendFileBackedExperts)
                                                         && opt.explicit_expert_gpu_cache;
                const uint64_t gated_delta_gpu_budget = gated_delta_vulkan_budget_size(opt,
                                                                                       vulkan_delta_fusion_available,
                                                                                       protect_file_backed_experts);
                const uint64_t gated_delta_layer_size = gated_delta_vulkan_working_set_size(compiled, layer_plan);
                if (vulkan_delta_fusion_available
                    && gated_delta_layer_size != 0
                    && gated_delta_layer_size <= gated_delta_gpu_budget
                    && planned_gated_delta_gpu_size
                           <= gated_delta_gpu_budget - gated_delta_layer_size)
                {
                    prepare_gated_delta_attention_operators(compiled,
                                                            layer_plan);
                    const CompiledOperatorHandle gated_handle = layer_plan.attention.gated_delta_vulkan_operator;
                    if (gated_handle != invalid_compiled_operator_handle
                        && compiled.operators.at(gated_handle).gated_delta)
                    {
                        planned_gated_delta_gpu_size += gated_delta_layer_size;
                    }
                }
            }
            if (!ret)
                return ret.error();
            prepare_latent_layer_operators(compiled, layer_plan);
        }
        if (layer_plan.dense_ffn.down_weight != invalid_tensor_handle
            && !layer_plan.dense_ffn.use_gpu_cache)
        {
            ret = prepare_ffn_operators(compiled.weights, compiled.operators, layer_plan.dense_ffn,
                                        layer_plan.vulkan_device_index, compiled.vulkan_runtime,
                                        compiled.opt.optimization_flags);
            if (!ret)
                return ret.error();
        }
        if (layer_plan.moe.has_shared_expert)
        {
            ret = prepare_shared_expert_operators(compiled.weights,
                                                  compiled.operators,
                                                  layer_plan.moe,
                                                  layer_plan.vulkan_device_index,
                                                  compiled.vulkan_runtime,
                                                  compiled.opt.optimization_flags);
            if (!ret)
                return ret.error();
        }
    }

    if (compiled.speculative.kind == SpeculativeModelKind::Mtp)
    {
        ret = prepare_mtp_operators(compiled, has_flag(opt.flags, BackendVulkanAttention));
    }
    else if (compiled.speculative.kind == SpeculativeModelKind::DSpark)
    {
        ret = prepare_dspark_operators(compiled);
    }
    if (!ret)
        return ret.error();
    return {};
}

} // namespace moe
} // namespace ncnn
