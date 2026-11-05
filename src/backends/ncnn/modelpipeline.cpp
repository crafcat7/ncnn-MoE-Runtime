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
#include "gatedresiduallayer_vulkan.h"
#include "gpusegment_vulkan.h"
#include "expertbackend_vulkan.h"
#include "expertbank_vulkan.h"
#include "graph/compiler.h"
#include "engine/expert.h"
#include "graph/compiledmodel.h"
#include "graph/resourceplan.h"
#include "ncnn/moe/session.h"
#include "storage/weightstore.h"
#include "vulkancontext.h"

#if NCNN_MOE_WITH_VULKAN
#include <gpu.h>
#endif

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static Result<void> prepare_linear_vk(WeightStore& weights,
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
    if (matrix.dtype == DType::Int4 || matrix.dtype == DType::MxFp4 || is_qnk_dtype(matrix.dtype)
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
    auto prepared = prepare_linear_vk(compiled.weights, compiled.operators, compiled.lm_head_weight, compiled.lm_head_bias,
                                      compiled.opt.vulkan_device_index, compiled.vulkan_runtime,
                                      compiled.opt.optimization_flags, 1, prefer_bfloat16_vulkan, allow_float32_fallback);
    if (!prepared)
        return prepared.error();
    const CompiledOperator& lm_head_operator = compiled.operators.at_weight(compiled.lm_head_weight);
    if (lm_head_operator.bfloat16
        && compiled.final_norm_weight != invalid_tensor_handle
        && compiled.descriptor.final_norm == NormType::RmsNorm)
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
    if (moe.experts.empty()) return false;
    for (const ExpertPlan& expert : moe.experts)
    {
        const TensorHandle gate = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight;
        if (gate == invalid_tensor_handle || expert.down_weight == invalid_tensor_handle)
            return false;
        const TensorData* up = expert.gate_up_weight == invalid_tensor_handle && expert.up_weight != invalid_tensor_handle
                                   ? &weights.at(expert.up_weight)
                                   : nullptr;
        if (!support_vulkan_expert(expert, weights.at(gate), weights.at(expert.down_weight), optimization_flags, up))
            return false;
    }
    // A scheduled ExpertGroup promises device execution for every routable
    // Expert; one supported bank cannot make a mixed unsupported group GPU.
    return true;
}

bool support_vulkan_ffn(const CompiledOperatorTable& operators, const ExpertPlan& ffn) noexcept
{
    const CompiledOperator& fused = operators.at(ffn.vulkan_operator);
    if (fused.qnk_expert)
        return true;
    if (fused.bfloat16 && operators.at_weight(ffn.down_weight).bfloat16)
        return true;
    return fused.float8 && operators.at_weight(ffn.up_weight).float8
           && operators.at_weight(ffn.down_weight).float8
           && (ffn.activation == ExpertActivation::Silu || ffn.activation == ExpertActivation::DeepSeekSwiGlu);
}

bool support_vulkan_dense_ffn(const CompiledOperatorTable& operators, const ExpertPlan& ffn) noexcept
{
    return support_vulkan_ffn(operators, ffn);
}

bool support_vulkan_shared_experts(const CompiledOperatorTable& operators,
                                   const MoeBlockPlan& moe) noexcept
{
    if (moe.fused_shared_input_bfloat16_operator != invalid_compiled_operator_handle
        && operators.at(moe.fused_shared_input_bfloat16_operator).bfloat16)
        return static_cast<bool>(operators.at_weight(moe.shared_expert.down_weight).bfloat16);
    const ExpertPlan& shared = moe.shared_expert;
    const auto& qnk = operators.at(shared.vulkan_operator).qnk_expert;
    if (qnk)
        return moe.shared_expert_gate_weight == invalid_tensor_handle || qnk->has_router_gate();
    return moe.shared_expert_gate_weight == invalid_tensor_handle
           && shared.activation == ExpertActivation::DeepSeekSwiGlu
           && operators.at_weight(shared.gate_weight).float8
           && operators.at_weight(shared.up_weight).float8
           && operators.at_weight(shared.down_weight).float8;
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

    // Segmented MLA does not need the optional complete command recorder, but
    // its Attention core and every projection must be executable on the GPU.
    if (!operators.at_weight(attention.query_a_weight).latent_attention)
        return false;
    const TensorHandle dense_handles[] = {
        attention.query_a_weight,
        attention.query_b_weight,
        attention.key_value_weight,
        attention.output_a_weight,
        attention.output_b_weight,
    };
    for (TensorHandle handle : dense_handles)
    {
        if (handle == invalid_tensor_handle || !uses_vulkan_dense_operator(operators.at_weight(handle)))
            return false;
    }
    if (has_flag(attention.flags, AttentionBlockLatentValue)
        && (attention.query_b_absorbed_weight == invalid_tensor_handle
            || !uses_vulkan_dense_operator(operators.at_weight(attention.query_b_absorbed_weight))
            || attention.output_gate_weight == invalid_tensor_handle
            || !uses_vulkan_dense_operator(operators.at_weight(attention.output_gate_weight))))
        return false;
    if (attention.compression_ratio != 0)
        for (TensorHandle handle : {attention.compressor_key_value_weight, attention.compressor_gate_weight})
            if (handle == invalid_tensor_handle || !uses_vulkan_dense_operator(operators.at_weight(handle)))
                return false;
    if (attention.compression_ratio == 4)
        for (TensorHandle handle : {attention.indexer_compressor_key_value_weight, attention.indexer_compressor_gate_weight,
                                    attention.indexer_query_weight, attention.indexer_weights_weight})
            if (handle == invalid_tensor_handle || !uses_vulkan_dense_operator(operators.at_weight(handle)))
                return false;
    return true;
}

static bool bf16_projection_supported(const CompiledModel& compiled,
                                      const TensorData& matrix,
                                      uint64_t alignment,
                                      uint32_t device_index);

static bool attn_bf16_fusion(const CompiledModel& compiled,
                             const AttentionBlockPlan& attention,
                             uint64_t alignment,
                             uint32_t device_index)
{
    const bool output_gate = has_flag(attention.flags, AttentionBlockOutputGate);
    if (output_gate && (attention.query_bias != invalid_tensor_handle || attention.key_bias != invalid_tensor_handle || attention.value_bias != invalid_tensor_handle))
        return false;
    uint64_t rows = 0;
    uint32_t columns = 0;
    for (TensorHandle handle : {attention.query_weight, attention.key_weight, attention.value_weight,
                                output_gate ? attention.output_gate_weight : invalid_tensor_handle})
    {
        if (handle == invalid_tensor_handle)
            continue;
        const TensorData& matrix = compiled.weights.at(handle);
        if (matrix.dtype != DType::BFloat16 || matrix.shape.size() != 2
            || (columns != 0 && columns != matrix.shape[1]))
            return false;
        columns = matrix.shape[1];
        rows += matrix.shape[0];
    }
    if (rows == 0 || rows > std::numeric_limits<uint32_t>::max())
        return false;
    TensorData fused;
    fused.dtype = DType::BFloat16;
    fused.shape = {static_cast<uint32_t>(rows), columns};
    return bf16_projection_supported(compiled, fused, alignment, device_index)
           && bf16_projection_supported(compiled, compiled.weights.at(attention.output_weight), alignment, device_index);
}

static bool attn_needs_native(const CompiledModel& compiled,
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
                                          || dtype == DType::MxFp4
                                          || is_qnk_dtype(dtype);
        all_bfloat16 = all_bfloat16 && dtype == DType::BFloat16;
    }

    const bool has_query_key_norm = has_flag(attention.flags, AttentionBlockQueryKeyNorm);
    const bool has_output_gate = has_flag(attention.flags, AttentionBlockOutputGate);
    // Preserve the established fused BF16 implementation for the paired
    // QK-norm/output-gate form; the native QnK operator covers independent
    // flags and partial RoPE for quantized, BF16, and FP32 projections.
    const bool legacy_bfloat16_qk_gate = has_query_key_norm && has_output_gate && all_bfloat16
                                         && attn_bf16_fusion(compiled, attention, 4, device_index);
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

static bool attn_native(const CompiledModel& compiled,
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
        if (dtype != DType::Int4 && dtype != DType::MxFp4 && !is_qnk_dtype(dtype)
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

void release_host_copies(CompiledModel& compiled)
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
        const auto& shared_chain = compiled.operators.at(shared.vulkan_operator).qnk_expert;
        if (layer.moe.has_shared_expert && shared_chain && shared_chain->has_router_gate())
            release_vulkan_dense_handle(compiled, layer.moe.shared_expert_gate_weight);
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
        const ExpertPlan& dense = layer.dense_ffn;
        if (dense.down_weight != invalid_tensor_handle && !dense.use_gpu_cache
            && support_vulkan_dense_ffn(compiled.operators, dense))
        {
            release_vulkan_dense_handle(compiled, dense.gate_weight);
            release_vulkan_dense_handle(compiled, dense.up_weight);
            release_vulkan_dense_handle(compiled, dense.down_weight);
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
        if (!chain)
            return Error{ErrorCode::InternalError, "failed to create the selected Vulkan FFN chain"};
        ffn.vulkan_operator = operators.allocate();
        operators.at_mutable(ffn.vulkan_operator).qnk_expert = std::move(chain);
        return {};
    }
    if (gate.dtype == DType::Float8E4M3 && up.dtype == DType::Float8E4M3 && down.dtype == DType::Float8E4M3
        && (ffn.activation == ExpertActivation::Silu || ffn.activation == ExpertActivation::DeepSeekSwiGlu)
        && up.shape.size() == 2 && up.shape[0] % 128 == 0)
    {
        for (TensorHandle handle : {ffn.gate_weight, ffn.up_weight, ffn.down_weight})
        {
            auto prepared = prepare_linear_vk(weights, operators, handle, invalid_tensor_handle,
                                              vulkan_device_index, vulkan_runtime, optimization_flags);
            if (!prepared) return prepared.error();
        }
        ffn.vulkan_operator = operators.handle_for_weight(ffn.gate_weight);
        return {};
    }
    if (gate.dtype == DType::BFloat16 && up.dtype == DType::BFloat16
        && down.dtype == DType::BFloat16
        && (ffn.activation == ExpertActivation::Silu
            || ffn.activation == ExpertActivation::DeepSeekSwiGlu)
        && up.shape.size() == 2
        && up.shape[0] % 128 == 0)
    {
        auto fused = Bfloat16Linear_vulkan::create_fused({&gate, &up}, {nullptr, nullptr},
                                                         vulkan_device_index, vulkan_runtime, optimization_flags);
        if (fused)
        {
            ffn.vulkan_operator = operators.allocate();
            operators.at_mutable(ffn.vulkan_operator).bfloat16 = std::move(fused);
            return prepare_linear_vk(weights, operators, ffn.down_weight, invalid_tensor_handle,
                                     vulkan_device_index, vulkan_runtime, optimization_flags, 1, true, allow_float32_fallback);
        }
    }
    if (!allow_float32_fallback)
        return Error{ErrorCode::InternalError, "failed to create fused Vulkan BF16 FFN operator"};
    for (TensorHandle handle : {ffn.gate_weight, ffn.up_weight, ffn.down_weight})
    {
        auto prepared = prepare_linear_vk(weights, operators, handle, invalid_tensor_handle,
                                          vulkan_device_index, vulkan_runtime, optimization_flags, 1, true, allow_float32_fallback);
        if (!prepared)
            return prepared.error();
    }
    return {};
}

static Result<void> prepare_shared_ops(WeightStore& weights,
                                       CompiledOperatorTable& operators,
                                       MoeBlockPlan& moe,
                                       uint32_t vulkan_device_index,
                                       const VulkanRuntimePtr& vulkan_runtime,
                                       uint64_t optimization_flags)
{
    ExpertPlan& shared = moe.shared_expert;
    const TensorData& gate = weights.at(shared.gate_weight);
    const TensorData& up = weights.at(shared.up_weight);
    const TensorData& down = weights.at(shared.down_weight);
    const bool has_router_gate = moe.shared_expert_gate_weight != invalid_tensor_handle;
    const auto quantized = [](DType dtype) { return dtype == DType::Int4 || dtype == DType::MxFp4 || is_qnk_dtype(dtype); };
    const bool unsupported_bfloat16_chain = gate.dtype == DType::BFloat16
                                            && up.dtype == DType::BFloat16
                                            && (down.dtype != DType::BFloat16
                                                || (shared.activation != ExpertActivation::Silu
                                                    && shared.activation != ExpertActivation::DeepSeekSwiGlu)
                                                || up.shape.size() != 2
                                                || up.shape[0] % 128 != 0);
    if (quantized(gate.dtype) || quantized(up.dtype) || quantized(down.dtype)
        || unsupported_bfloat16_chain)
    {
        auto prepared = prepare_ffn_operators(weights, operators, shared, vulkan_device_index,
                                              vulkan_runtime, optimization_flags);
        if (!prepared || !has_router_gate)
            return prepared;
        const auto& chain = operators.at(shared.vulkan_operator).qnk_expert;
        if (chain)
        {
            if (!chain->prepare_router_gate(weights.at(moe.shared_expert_gate_weight)))
                return Error{ErrorCode::InternalError, "failed to prepare the selected Vulkan Shared Expert router gate"};
            return {};
        }
        return prepare_linear_vk(weights, operators, moe.shared_expert_gate_weight, invalid_tensor_handle,
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
    return prepare_linear_vk(weights,
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
            return matrix->dtype == DType::Int4 || matrix->dtype == DType::MxFp4 || matrix->dtype == DType::Float8E4M3 || is_qnk_dtype(matrix->dtype);
        }))
    {
        const TensorHandle handles[] = {plan.query_weight, plan.key_weight, plan.value_weight};
        const TensorHandle biases[] = {plan.query_bias, plan.key_bias, plan.value_bias};
        for (size_t index = 0; index < 3; ++index)
        {
            auto prepared = prepare_linear_vk(compiled.weights, compiled.operators,
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

static Result<void> prepare_latent_ops(CompiledModel& compiled,
                                       CompiledLayerPlan& layer_plan,
                                       const char* diagnostic_prefix = "",
                                       bool attention_core_enabled = true)
{
    AttentionBlockPlan& plan = layer_plan.attention;
    Result<void> prepared;
    const TensorHandle latent_linear_handles[] = {
        plan.query_a_weight,
        plan.query_b_weight,
        plan.query_b_absorbed_weight,
        plan.key_value_weight,
        plan.output_gate_weight,
        plan.output_b_weight,
        plan.indexer_query_weight,
        plan.indexer_weights_weight,
    };
    for (TensorHandle handle : latent_linear_handles)
    {
        if (handle == invalid_tensor_handle)
            continue;
        prepared = prepare_linear_vk(compiled.weights, compiled.operators, handle,
                                     invalid_tensor_handle,
                                     layer_plan.vulkan_device_index,
                                     compiled.vulkan_runtime,
                                     compiled.opt.optimization_flags);
        if (!prepared)
            return prepared.error();
    }
    // The flag controls fusion, not device placement. Unfused execution of a
    // selected Vulkan Attention still needs every compressor projection.
    if (plan.compression_ratio == 4 || plan.compression_ratio == 128)
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
            prepared = prepare_linear_vk(compiled.weights, compiled.operators, handle,
                                         invalid_tensor_handle,
                                         layer_plan.vulkan_device_index,
                                         compiled.vulkan_runtime,
                                         compiled.opt.optimization_flags);
            if (!prepared)
                return prepared.error();
        }
    }
    prepared = prepare_linear_vk(compiled.weights, compiled.operators, plan.output_a_weight,
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
        compiled.operators.at_weight_mutable(plan.query_a_weight).latent_attention = LatentAttention_vulkan::create(plan, plan.sinks == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.sinks),
                                                                                                                    output_a, output_b);
        const auto& latent = compiled.operators.at_weight(plan.query_a_weight).latent_attention;
        if (!latent)
            return Error{ErrorCode::InternalError, "failed to prepare " + std::string(diagnostic_prefix) + "Vulkan latent Attention core"};
        const CompiledOperator& key_value = compiled.operators.at_weight(plan.key_value_weight);
        const bool recording_types = (query_a.float8 && query_b.float8 && key_value.float8)
                                     || (query_a.bfloat16 && query_b.bfloat16 && key_value.bfloat16)
                                     || (query_a.qnk && query_b.qnk && key_value.qnk);
        const bool recording_shape = plan.compression_ratio == 0 || plan.compression_ratio == 128
                                     || (plan.compression_ratio == 4 && plan.index_head_dimension != 0
                                         && plan.index_head_dimension <= 512
                                         && (plan.index_head_dimension & (plan.index_head_dimension - 1)) == 0
                                         && plan.rope_head_dimension <= plan.index_head_dimension && plan.index_head_count != 0);
        // Disabling a fusion keeps the complete segmented GPU implementation.
        // Once a supported recorder is selected, allocation/preparation failure
        // must be reported instead of changing the execution path at runtime.
        const bool prepare_recording = !has_flag(plan.flags, AttentionBlockLatentValue)
                                       && recording_types && recording_shape
                                       && has_flag(compiled.opt.optimization_flags, OptimizationVulkanLatentInputRmsNorm)
                                       && (plan.compression_ratio == 0
                                           || has_flag(compiled.opt.optimization_flags, OptimizationVulkanLatentCompressor));
        if (has_flag(plan.flags, AttentionBlockLatentValue))
        {
            if (!latent->prepare_value(compiled.weights, compiled.operators))
                return Error{ErrorCode::InternalError, "failed to prepare " + std::string(diagnostic_prefix) + "Vulkan expanded-value latent Attention"};
        }
        else if (prepare_recording && !latent->prepare(compiled.weights, compiled.operators, compiled.opt.optimization_flags))
            return Error{ErrorCode::InternalError, "failed to prepare " + std::string(diagnostic_prefix) + "Vulkan latent Attention recorder"};
    }
    return {};
}

static void prepare_latent_layer_ops(CompiledModel& compiled, CompiledLayerPlan& layer)
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

static void prepare_gdn_ops(CompiledModel& compiled,
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
    const auto native_projection = [](const TensorData& weight) { return weight.dtype == DType::Float32 || weight.dtype == DType::Int4 || weight.dtype == DType::MxFp4 || is_qnk_dtype(weight.dtype); };
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
                                                            plan.delta_gate_channels, plan.delta_lower_bound,
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
                                                                                        plan.delta_gate_channels,
                                                                                        plan.delta_lower_bound,
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

static AttentionConfig_vulkan attn_vk_config(const CompiledModel& compiled,
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
    const std::array<const TensorData*, 3> qkv = {
        &compiled.weights.at(plan.query_weight), &compiled.weights.at(plan.key_weight), &compiled.weights.at(plan.value_weight)};
    const std::array<const TensorData*, 3> qkv_biases = {
        plan.query_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.query_bias),
        plan.key_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.key_bias),
        plan.value_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.value_bias)};
    std::shared_ptr<QnkLinear_vulkan> fused_qkv;
#if NCNN_MOE_WITH_VULKAN
    const auto context = VulkanContext::acquire(layer_plan.vulkan_device_index, compiled.vulkan_runtime,
                                                compiled.opt.optimization_flags);
    if (context && QnkLinear_vulkan::can_fuse(qkv, context->device()->info.buffer_offset_alignment(), context->device()->info.physicalDeviceProperties().limits.maxStorageBufferRange))
    {
        fused_qkv = QnkLinear_vulkan::create_fused(qkv, qkv_biases, layer_plan.vulkan_device_index,
                                                   compiled.vulkan_runtime, compiled.opt.optimization_flags);
        if (!fused_qkv)
            return Error{ErrorCode::InternalError, "failed to create fused native Vulkan projections"};
        const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
        compiled.operators.at_mutable(fused_handle).qnk = fused_qkv;
        plan.fused_qkv_operator = fused_handle;
    }
#endif
    std::array<CompiledOperator, 5> projections;
    for (size_t index = 0; index < projections.size(); ++index)
    {
        const TensorHandle handle = weights[index];
        if (handle == invalid_tensor_handle)
            continue;
        if (fused_qkv && index < 3)
        {
            compiled.operators.at_weight_mutable(handle) = {};
            continue;
        }

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
    const TensorData* norm_bias = plan.pre_attention_norm_bias == invalid_tensor_handle
                                      ? nullptr
                                      : &compiled.weights.at(plan.pre_attention_norm_bias);
    auto attention_operator = Attention_vulkan::create_native(compiled.weights.at(plan.pre_attention_norm_weight),
                                                              query_norm,
                                                              key_norm,
                                                              sinks,
                                                              projections,
                                                              attn_vk_config(compiled, plan), fused_qkv, norm_bias);
    if (!attention_operator)
        return Error{ErrorCode::InternalError,
                     "failed to create native Vulkan Standard Attention at layer "
                         + std::to_string(layer_plan.layer_id)};

    const CompiledOperatorHandle attention_handle = compiled.operators.allocate();
    compiled.operators.at_mutable(attention_handle).attention = std::move(attention_operator);
    plan.vulkan_attention_operator = attention_handle;
    return {};
}

static Result<void> prepare_attn_ops(CompiledModel& compiled,
                                     CompiledLayerPlan& layer_plan,
                                     bool allow_float32_fallback = true)
{
    AttentionBlockPlan& plan = layer_plan.attention;
    if (!has_flag(compiled.opt.optimization_flags, OptimizationVulkanAttention))
        return {};
    if (has_flag(plan.flags, AttentionBlockQsa | AttentionBlockExternalResidual))
    {
        return Error{ErrorCode::UnsupportedModel,
                     "Vulkan QSA and external residual attention require the complete gated residual pipeline"};
    }

    const bool requires_native_attention = attn_needs_native(compiled, plan, layer_plan.vulkan_device_index);
    if (requires_native_attention)
    {
        // Some Standard layouts need the native QnK path for their complete
        // attention semantics. If that optimization is explicitly disabled,
        // leave the graph on CPU; otherwise an unsupported native dtype must
        // fail model preparation rather than silently becoming CPU attention.
        if (!has_flag(compiled.opt.optimization_flags, OptimizationVulkanQnK))
            return {};
        if (!attn_native(compiled, plan))
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
    const bool use_bfloat16_fusion = attn_bf16_fusion(compiled, plan, 4, layer_plan.vulkan_device_index);
    // Representation is selected from metadata and device capability before
    // allocation. A failure preparing selected BF16 storage is terminal.
    prepared = prepare_vulkan_qkv_operator(compiled, layer_plan, use_bfloat16_fusion,
                                           "failed to create fused Vulkan QKV operator", !use_bfloat16_fusion && allow_float32_fallback);
    if (!prepared)
        return prepared.error();
    prepared = prepare_linear_vk(compiled.weights, compiled.operators, plan.output_weight,
                                 plan.output_bias,
                                 layer_plan.vulkan_device_index,
                                 compiled.vulkan_runtime,
                                 compiled.opt.optimization_flags,
                                 1,
                                 use_bfloat16_fusion,
                                 !use_bfloat16_fusion && allow_float32_fallback);
    if (!prepared)
        return prepared.error();
    if (plan.output_gate_weight != invalid_tensor_handle)
    {
        if (plan.fused_qkv_gate_bfloat16_operator == invalid_compiled_operator_handle)
        {
            prepared = prepare_linear_vk(compiled.weights, compiled.operators, plan.output_gate_weight,
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
        const AttentionConfig_vulkan attention_config = attn_vk_config(compiled, plan);
        std::shared_ptr<Attention_vulkan> attention_operator;
        if (fused_vulkan_attention_eligible)
        {
            if (plan.fused_qkv_operator == invalid_compiled_operator_handle
                || (use_bfloat16_fusion
                        ? (!compiled.operators.at(plan.fused_qkv_operator).bfloat16
                           || !compiled.operators.at_weight(plan.output_weight).bfloat16)
                        : (!compiled.operators.at(plan.fused_qkv_operator).linear
                           || !compiled.operators.at_weight(plan.output_weight).linear)))
            {
                return Error{ErrorCode::UnsupportedModel,
                             "Vulkan Standard Attention has no compatible fused projection operator at layer "
                                 + std::to_string(layer_plan.layer_id)};
            }
            const CompiledOperator& fused_operator = compiled.operators.at(plan.fused_qkv_operator);
            const TensorData* sinks = plan.sinks == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.sinks);
            const TensorData* norm_bias = plan.pre_attention_norm_bias == invalid_tensor_handle
                                              ? nullptr
                                              : &compiled.weights.at(plan.pre_attention_norm_bias);
            if (use_bfloat16_fusion)
                attention_operator = Attention_vulkan::create(compiled.weights.at(plan.pre_attention_norm_weight), sinks,
                                                              fused_operator.bfloat16,
                                                              compiled.operators.at_weight(plan.output_weight).bfloat16,
                                                              attention_config, norm_bias);
            else
                attention_operator = Attention_vulkan::create(compiled.weights.at(plan.pre_attention_norm_weight), sinks,
                                                              fused_operator.linear,
                                                              compiled.operators.at_weight(plan.output_weight).linear,
                                                              attention_config, norm_bias);
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
            const TensorData* norm_bias = plan.pre_attention_norm_bias == invalid_tensor_handle
                                              ? nullptr
                                              : &compiled.weights.at(plan.pre_attention_norm_bias);
            attention_operator = Attention_vulkan::create(compiled.weights.at(plan.pre_attention_norm_weight),
                                                          compiled.weights.at(plan.query_norm_weight),
                                                          compiled.weights.at(plan.key_norm_weight),
                                                          plan.sinks == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.sinks),
                                                          fused_operator.bfloat16,
                                                          compiled.operators.at_weight(plan.output_weight).bfloat16,
                                                          attention_config, norm_bias);
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

static bool bf16_projection_supported(const CompiledModel& compiled,
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

uint64_t dense_weight_size(const CompiledModel& compiled,
                           const TensorData& matrix,
                           uint64_t alignment,
                           uint32_t device_index,
                           bool has_bias) noexcept
{
    if (matrix.dtype == DType::Int4 || matrix.dtype == DType::MxFp4 || is_qnk_dtype(matrix.dtype))
    {
        if (is_qnk_dtype(matrix.dtype) && !has_flag(compiled.opt.optimization_flags, OptimizationVulkanQnK))
            return 0;
        return QnkLinear_vulkan::projection_storage_bytes(matrix, alignment);
    }
#if NCNN_MOE_WITH_VULKAN
    if (matrix.dtype == DType::Float8E4M3)
    {
        auto context = VulkanContext::acquire(device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
        if (!context || matrix.shape.size() != 2 || matrix.shape[1] % 128 != 0)
            return 0;
        const uint64_t payload = matrix.element_count();
        const uint64_t scales = saturating_multiply_u64(matrix.quantization_scales.size(), sizeof(float));
        const uint64_t bias = saturating_multiply_u64(matrix.shape[0], sizeof(float));
        const uint64_t limit = context->device()->info.physicalDeviceProperties().limits.maxStorageBufferRange;
        if (payload > limit || scales > limit || bias > limit)
            return 0;
        const uint64_t padding = saturating_multiply_u64(std::max<uint64_t>(4, alignment), 3);
        return saturating_add_u64(saturating_add_u64(payload, scales), saturating_add_u64(bias, padding));
    }
#endif
    if (bf16_projection_supported(compiled, matrix, alignment, device_index))
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

static uint64_t bf16_fused_storage_size(const CompiledModel& compiled,
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
    if (!bf16_projection_supported(compiled, fused, alignment, device_index))
        return 0;
    return Bfloat16Linear_vulkan::storage_bytes(fused, alignment);
}

static uint64_t const_buf_size(std::initializer_list<uint64_t> buffers,
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

static uint64_t gated_delta_state_size(const CompiledModel& compiled,
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

static uint64_t attn_init_size(const CompiledModel& compiled,
                               const AttentionBlockPlan& attention,
                               uint32_t device_index,
                               uint32_t context_tokens = 0) noexcept
{
    const uint64_t key_columns = saturating_multiply_u64(attention.kv_head_count,
                                                         attention.head_dimension);
    uint64_t required_tokens = std::max<uint64_t>(1, context_tokens == 0 ? SessionOptions{}.prefill_chunk_size : context_tokens);
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
    const uint64_t element_size = vulkan_kv_size(attention.kv_cache_dtype, device_index);
    if (element_size == UINT64_MAX) return UINT64_MAX;
    // The Vulkan KV ring stores two copies of each key/value row so wrapped
    // cache views stay contiguous. Count every concurrent Session's initial
    // prefill capacity; later growth remains checked at allocation time.
    const uint64_t ring = saturating_multiply_u64(saturating_multiply_u64(capacity,
                                                                          key_columns),
                                                  element_size * 4);
    return saturating_multiply_u64(ring,
                                   std::max(1u, compiled.opt.num_concurrent_sessions));
}

uint64_t attn_vk_size(const CompiledModel& compiled,
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
        if (attn_needs_native(compiled, attention, device_index))
        {
            if (!attn_native(compiled, attention))
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
                                                ? const_buf_size({matrix.element_count(),
                                                                  matrix.quantization_scales.size() * sizeof(float),
                                                                  static_cast<uint64_t>(matrix.shape[0]) * sizeof(float)},
                                                                 alignment)
                                                : QnkLinear_vulkan::projection_storage_bytes(matrix, alignment);
                if (projection == 0)
                    return 0;
                input = saturating_add_u64(input, projection);
            }
            constants = const_buf_size({static_cast<uint64_t>(compiled.descriptor.hidden_size) * sizeof(float),
                                        static_cast<uint64_t>(attention.head_dimension) * sizeof(float),
                                        static_cast<uint64_t>(attention.head_dimension) * sizeof(float),
                                        static_cast<uint64_t>(attention.head_dimension / 2) * sizeof(float),
                                        static_cast<uint64_t>(attention.head_count) * sizeof(float)},
                                       alignment);
            state = attn_init_size(compiled, attention, device_index);
            return saturating_add_u64(saturating_add_u64(input, constants), state);
        }
        state = attn_init_size(compiled, attention, device_index);
        const bool bfloat16 = attn_bf16_fusion(compiled, attention, alignment, device_index);
        TensorData fused;
        fused.dtype = bfloat16 ? DType::BFloat16 : DType::Float32;
        uint64_t rows = 0;
        uint32_t columns = 0;
        for (TensorHandle handle : {attention.query_weight, attention.key_weight, attention.value_weight,
                                    has_flag(attention.flags, AttentionBlockOutputGate) ? attention.output_gate_weight : invalid_tensor_handle})
        {
            if (handle == invalid_tensor_handle)
                continue;
            const TensorData& matrix = compiled.weights.at(handle);
            if (matrix.shape.size() != 2 || (columns != 0 && columns != matrix.shape[1]))
                return 0;
            rows += matrix.shape[0];
            columns = matrix.shape[1];
        }
        if (rows > std::numeric_limits<uint32_t>::max())
            return 0;
        fused.shape = {static_cast<uint32_t>(rows), columns};
        const bool input_bias = attention.query_bias != invalid_tensor_handle || attention.key_bias != invalid_tensor_handle
                                || attention.value_bias != invalid_tensor_handle;
        input = dense_weight_size(compiled, fused, alignment, device_index, input_bias);
        // Generic InnerProduct expands even BF16 source weights to FP32;
        // estimate the prepared representation rather than the source dtype.
        TensorData output;
        output.dtype = fused.dtype;
        output.shape = compiled.weights.at(attention.output_weight).shape;
        const uint64_t output_size = dense_weight_size(compiled, output, alignment, device_index,
                                                       attention.output_bias != invalid_tensor_handle);
        if (input == 0 || output_size == 0)
            return 0;
        constants = const_buf_size({static_cast<uint64_t>(compiled.descriptor.hidden_size) * sizeof(float),
                                    static_cast<uint64_t>(attention.head_dimension) * sizeof(float),
                                    static_cast<uint64_t>(attention.head_dimension) * sizeof(float),
                                    static_cast<uint64_t>(attention.head_dimension / 2) * sizeof(float),
                                    static_cast<uint64_t>(attention.head_count) * sizeof(float)},
                                   alignment);
        return saturating_add_u64(saturating_add_u64(input, output_size), saturating_add_u64(constants, state));
    }
    else if (attention.kind == AttentionKind::GatedDeltaNet)
    {
        input = bf16_fused_storage_size(compiled,
                                        {attention.delta_qkv_weight, attention.delta_z_weight, attention.delta_beta_weight, attention.delta_alpha_weight}, alignment, device_index);
        const uint64_t convolution = compiled.weights.at(attention.delta_convolution_weight).element_count() * sizeof(float);
        constants = const_buf_size({convolution,
                                    static_cast<uint64_t>(attention.head_count) * sizeof(float),
                                    static_cast<uint64_t>(attention.head_count) * sizeof(float),
                                    static_cast<uint64_t>(attention.value_head_dimension) * sizeof(float)},
                                   alignment);
        state = gated_delta_state_size(compiled, attention);
    }
    else
        return 0;
    const TensorData& output_weight = compiled.weights.at(attention.output_weight);
    const uint64_t output = bf16_projection_supported(compiled, output_weight, alignment, device_index)
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
            const uint64_t size = dense_weight_size(compiled, compiled.weights.at(weights[i]), alignment, device_index,
                                                    biases[i] != invalid_tensor_handle);
            if (size == 0)
                return 0;
            projections = saturating_add_u64(projections, size);
        }
        return saturating_add_u64(projections, saturating_add_u64(constants, state));
    }
    return saturating_add_u64(saturating_add_u64(input, output), saturating_add_u64(constants, state));
}

static uint64_t attention_session_state_size(const CompiledModel& compiled,
                                             const AttentionBlockPlan& attention,
                                             uint32_t device_index,
                                             uint32_t context_tokens = 0) noexcept
{
    const uint64_t sessions = std::max(1u, compiled.opt.num_concurrent_sessions);
    if (attention.kind == AttentionKind::GatedDeltaNet)
        return gated_delta_state_size(compiled, attention) / sessions;
    if (attention.kind == AttentionKind::Standard)
    {
        if (has_flag(attention.flags, AttentionBlockQsa))
        {
            uint64_t capacity = 16;
            const uint64_t tokens = context_tokens == 0 ? SessionOptions{}.prefill_chunk_size : context_tokens;
            while (capacity < tokens) capacity = saturating_multiply_u64(capacity, 2);
            const uint64_t key = saturating_multiply_u64(attention.kv_head_count, attention.head_dimension);
            const uint64_t kv = saturating_multiply_u64(saturating_multiply_u64(capacity, key),
                                                        attention.kv_cache_dtype == DType::BFloat16 ? sizeof(uint16_t) * 2 : sizeof(float) * 2);
            const uint64_t blocks = std::max<uint64_t>(1, capacity / std::max(1u, attention.compression_ratio));
            const uint64_t index = saturating_multiply_u64(saturating_multiply_u64(capacity + blocks, attention.index_head_dimension), sizeof(float));
            return saturating_add_u64(kv, index);
        }
        const uint64_t state = attn_init_size(compiled, attention, device_index,
                                              context_tokens);
        return state == UINT64_MAX ? UINT64_MAX : state / sessions;
    }
    if (attention.kind != AttentionKind::MultiHeadLatent)
        return 0;
    if (has_flag(attention.flags, AttentionBlockLatentValue))
    {
        // Expanded-value MLA grows a linear latent cache; plan for the
        // requested context (or the default prefill chunk) plus growth copy.
        const uint64_t tokens = context_tokens == 0 ? SessionOptions{}.prefill_chunk_size : context_tokens;
        uint64_t capacity = 128;
        while (capacity < tokens)
            capacity = saturating_multiply_u64(capacity, 2);
        const uint64_t elements = saturating_multiply_u64(capacity, attention.head_dimension);
        if (capacity == UINT64_MAX || elements == UINT64_MAX)
            return UINT64_MAX;
        return saturating_multiply_u64(elements, sizeof(float) * 2);
    }
    const uint64_t window = saturating_multiply_u64(attention.sliding_window, attention.head_dimension);
    uint64_t elements = window;
    if (attention.compression_ratio != 0)
    {
        const uint64_t ratio = attention.compression_ratio;
        const uint64_t tokens = context_tokens == 0 ? SessionOptions{}.prefill_chunk_size : context_tokens;
        const uint64_t required_chunks = (tokens + ratio - 1) / ratio;
        uint64_t chunks = 32;
        while (chunks < required_chunks) chunks = saturating_multiply_u64(chunks, 2);
        const uint64_t dimension = saturating_add_u64(attention.head_dimension,
                                                      ratio == 4 ? attention.index_head_dimension : 0);
        // A ratio-four compressor retains two pending banks and a previous
        // bank, each with values and scores. Other ratios retain one bank.
        const uint64_t compressor = saturating_multiply_u64(ratio, ratio == 4 ? 6 : 2);
        elements = saturating_add_u64(elements, saturating_multiply_u64(chunks + compressor, dimension));
    }
    // Cache growth/transaction publication keeps the old and new state alive.
    return saturating_multiply_u64(elements, sizeof(float) * 2);
}

uint64_t dense_pending_size(const CompiledModel& compiled,
                            uint32_t device_index) noexcept
{
    uint64_t size = 0;
    for (const ExecutionGraph* graph : {&compiled.graph, &compiled.speculative.graph})
    {
        for (const CompiledLayerPlan& layer : graph->layer_plans)
        {
            if (layer.vulkan_device_index == device_index
                && support_vulkan_attention(compiled.operators, layer.attention))
                size = saturating_add_u64(size, attention_session_state_size(compiled, layer.attention, device_index));
        }
    }
    return saturating_multiply_u64(size, std::max(1u, compiled.opt.num_concurrent_sessions));
}

static uint64_t gpu_buffer_alignment(const CompiledModel& compiled,
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

static bool use_gated_residual_pipeline(const CompiledModel& compiled, const CompilerOption& opt) noexcept
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

static GpuResourceRequest gated_residual_resources(const CompiledModel& compiled, uint32_t device_index) noexcept
{
    const auto& descriptor = compiled.descriptor;
    const uint64_t hidden = descriptor.hidden_size;
    const uint64_t copies = descriptor.hyper_connection_multiplier;
    const uint64_t expanded = saturating_multiply_u64(hidden, copies);
    const uint64_t rank = descriptor.hyper_connection_low_rank;
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
    return {0, persistent_states, saturating_add_u64(peak_initial_state, workspace)};
}

uint64_t vulkan_pending_reserve(const CompiledModel& compiled,
                                uint32_t device_index) noexcept
{
    for (const GpuResourcePlan& plan : compiled.memory_plan.gpu_devices)
        if (plan.initial.device_index == device_index)
            return gpu_exec_reserve(plan);
    if (compiled.gated_residual_chain)
    {
        const GpuResourceRequest request = gated_residual_resources(compiled, device_index);
        return saturating_multiply_u64(saturating_add_u64(request.session_state_size, request.workspace_size),
                                       std::max(1u, compiled.opt.num_concurrent_sessions));
    }
    bool has_gpu_operator = false;
    for (size_t handle = 0; handle < compiled.operators.size(); ++handle)
    {
        const CompiledOperator& op = compiled.operators.at(static_cast<CompiledOperatorHandle>(handle));
        if (uses_vulkan_dense_operator(op) || op.gated_delta || op.qnk_expert)
        {
            has_gpu_operator = true;
            break;
        }
    }
    return has_gpu_operator ? dense_pending_size(compiled, device_index) : 0;
}

static uint64_t minimum_gpu_expert_reserve(const CompiledModel& compiled, uint32_t device_index)
{
    uint64_t required = 0;
    for (const ExecutionGraph* graph : {&compiled.graph, &compiled.speculative.graph})
    {
        for (const CompiledLayerPlan& layer : graph->layer_plans)
        {
            if (layer.vulkan_device_index != device_index) continue;
            uint64_t largest = 0;
            for (const ExpertPlan& expert : layer.moe.experts)
            {
                const TensorHandle gate = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight;
                if (gate == invalid_tensor_handle || expert.down_weight == invalid_tensor_handle) continue;
                const TensorData* up = expert.gate_up_weight == invalid_tensor_handle && expert.up_weight != invalid_tensor_handle
                                           ? &compiled.weights.at(expert.up_weight)
                                           : nullptr;
                const TensorHandle gate_bias = up ? expert.gate_bias : expert.gate_up_bias;
                const uint64_t bytes = quant_expert_bytes(compiled.weights.at(gate), up,
                                                          compiled.weights.at(expert.down_weight), device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags,
                                                          gate_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(gate_bias),
                                                          expert.up_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(expert.up_bias),
                                                          expert.down_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(expert.down_bias),
                                                          expert.activation, expert.activation_limit);
                largest = std::max(largest, bytes);
            }
            required = std::max(required, saturating_multiply_u64(largest, layer.moe.top_k));
            if (layer.dense_ffn.use_gpu_cache)
            {
                const ExpertPlan& ffn = layer.dense_ffn;
                const uint64_t bytes = quant_expert_bytes(compiled.weights.at(ffn.gate_weight),
                                                          &compiled.weights.at(ffn.up_weight), compiled.weights.at(ffn.down_weight), device_index,
                                                          compiled.vulkan_runtime, compiled.opt.optimization_flags, nullptr, nullptr, nullptr,
                                                          ffn.activation, ffn.activation_limit);
                required = std::max(required, bytes);
            }
        }
    }
    return saturating_multiply_u64(required, std::max(1u, compiled.opt.num_concurrent_sessions));
}

static Result<void> prepare_gated_residual_pipeline(CompiledModel& compiled)
{
    for (TensorHandle handle = 0; handle < compiled.weights.size(); ++handle)
        if (compiled.weights.at(handle).dtype == DType::Float8E4M3)
            return Error{ErrorCode::UnsupportedModel, "continuous gated residual Vulkan execution does not support FP8 model weights"};
    for (const auto& layer : compiled.graph.layer_plans)
    {
        if (layer.vulkan_device_index != compiled.opt.vulkan_device_index)
            return Error{ErrorCode::UnsupportedModel, "gated residual device execution requires all layers and the output head on one Vulkan device"};
    }
    auto head = QnkLinear_vulkan::create(compiled.weights.at(compiled.lm_head_weight), nullptr,
                                         compiled.opt.vulkan_device_index, compiled.vulkan_runtime,
                                         compiled.opt.optimization_flags);
    if (!head)
        return Error{ErrorCode::InternalError, "failed to create the required native Vulkan LM head for gated residual execution"};
    compiled.operators.at_weight_mutable(compiled.lm_head_weight).qnk = std::move(head);
    compiled.gated_residual_device_head = GatedResidualLayer_vulkan::create_head(compiled);
    if (!compiled.gated_residual_device_head)
        return Error{ErrorCode::InternalError, "failed to create the required Vulkan LM head for gated residual execution"};
    for (CompiledLayerPlan& layer : compiled.graph.layer_plans)
    {
        auto& attention = layer.attention;
        if (attention.kind == AttentionKind::GatedDeltaNet)
        {
            prepare_gdn_ops(compiled, layer);
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
            return Error{ErrorCode::UnsupportedModel, "gated residual device execution requires Gated DeltaNet or QSA attention"};
        if (layer.moe.has_shared_expert)
        {
            auto& plan = layer.moe.shared_expert;
            auto shared = QnkExpert_vulkan::create_separate(compiled.weights.at(plan.gate_weight), nullptr,
                                                            compiled.weights.at(plan.up_weight), nullptr,
                                                            compiled.weights.at(plan.down_weight), nullptr,
                                                            plan.activation_limit, layer.vulkan_device_index, plan.activation,
                                                            compiled.vulkan_runtime, compiled.opt.optimization_flags);
            if (!shared || (layer.moe.shared_expert_gate_weight != invalid_tensor_handle && !shared->prepare_router_gate(compiled.weights.at(layer.moe.shared_expert_gate_weight))))
                return Error{ErrorCode::InternalError, "failed to create required Vulkan shared Expert for gated residual layer " + std::to_string(layer.layer_id)};
            const auto handle = compiled.operators.allocate();
            compiled.operators.at_mutable(handle).qnk_expert = std::move(shared);
            plan.vulkan_operator = handle;
        }
        auto operation = GatedResidualLayer_vulkan::create(compiled, layer);
        if (!operation)
            return Error{ErrorCode::InternalError, "failed to create required Vulkan gated residual layer " + std::to_string(layer.layer_id)};
        compiled.operators.at_weight_mutable(attention.output_weight).gated_residual_layer = std::move(operation);
    }
    compiled.gated_residual_chain = true;
    return {};
}

static GpuResourceSnapshot live_gpu_resources(const CompiledModel& compiled, uint32_t device_index)
{
    GpuResourceSnapshot snapshot{device_index, 0, 0};
#if NCNN_MOE_WITH_VULKAN
    auto context = VulkanContext::acquire(device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    if (context)
    {
        snapshot.heap_budget = static_cast<uint64_t>(context->device()->get_heap_budget()) * 1024 * 1024;
        snapshot.heap_usage = static_cast<uint64_t>(context->device()->get_heap_usage()) * 1024 * 1024;
    }
#endif
    return snapshot;
}

static uint64_t expert_wave_workspace_size(const CompiledModel& model, uint32_t device)
{
    uint64_t workspace = 0;
    for (const ExecutionGraph* graph : {&model.graph, &model.speculative.graph})
    {
        for (const CompiledLayerPlan& layer : graph->layer_plans)
        {
            if (layer.vulkan_device_index != device) continue;
            const auto inspect = [&](const ExpertPlan& expert, uint32_t top_k) {
                if (expert.down_weight == invalid_tensor_handle) return;
                const TensorData& down = model.weights.at(expert.down_weight);
                if (down.shape.size() != 2) return;
                const TensorHandle gate = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight;
                if (gate == invalid_tensor_handle) return;
                const TensorData& input = model.weights.at(gate);
                if (input.dtype != DType::BFloat16 && input.dtype != DType::Float32
                    && input.dtype != DType::Int4 && input.dtype != DType::MxFp4 && !is_qnk_dtype(input.dtype)) return;
                const TensorData* up = expert.gate_up_weight == invalid_tensor_handle && expert.up_weight != invalid_tensor_handle
                                           ? &model.weights.at(expert.up_weight)
                                           : nullptr;
                const TensorHandle gate_bias = up ? expert.gate_bias : expert.gate_up_bias;
                if (quant_expert_bytes(input, up, down, device, model.vulkan_runtime, model.opt.optimization_flags,
                                       gate_bias == invalid_tensor_handle ? nullptr : &model.weights.at(gate_bias),
                                       expert.up_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.up_bias),
                                       expert.down_bias == invalid_tensor_handle ? nullptr : &model.weights.at(expert.down_bias),
                                       expert.activation, expert.activation_limit)
                    == 0) return;
                const uint64_t tokens = SessionOptions{}.prefill_chunk_size;
                const uint64_t routes = saturating_multiply_u64(tokens, std::max(1u, top_k));
                // An eight-Expert wave can contain all routes from the chunk.
                // Gathered input/output and gate/up projections coexist; route
                // aggregation also retains one token-sized output matrix.
                const uint64_t columns = saturating_add_u64(saturating_multiply_u64(down.shape[0], 2),
                                                            saturating_add_u64(saturating_multiply_u64(down.shape[1], 2), 3));
                const uint64_t elements = saturating_add_u64(saturating_multiply_u64(routes, columns),
                                                             saturating_multiply_u64(tokens, down.shape[0]));
                workspace = std::max(workspace, saturating_multiply_u64(elements, sizeof(float)));
            };
            for (const ExpertPlan& expert : layer.moe.experts) inspect(expert, layer.moe.top_k);
            if (layer.dense_ffn.use_gpu_cache) inspect(layer.dense_ffn, 1);
        }
    }
    return workspace;
}

// One model-wide ledger shared by persistent preparation and Expert caching.
// Shape-specific operators supply requests; policy is independent of family.
class ResourcePlanner
{
public:
    ResourcePlanner(CompiledModel& model, const CompilerOption& opt);
    [[nodiscard]] bool can_admit(uint32_t device, GpuResourceRequest request) const;
    [[nodiscard]] bool allocation_fits(uint32_t device, GpuResourceRequest request) const;
    [[nodiscard]] bool explicit_cache_reserve(uint32_t device) const;
    void refresh_expert_reserve(uint32_t device);
    void commit(uint32_t device, GpuResourceRequest request);

private:
    [[nodiscard]] const GpuResourcePlan* find(uint32_t device) const;
    CompiledModel& model;
    std::map<uint32_t, uint64_t> usage_baselines;
    std::map<uint32_t, uint64_t> explicit_cache_reserves;
};

ResourcePlanner::ResourcePlanner(CompiledModel& model, const CompilerOption& opt) : model(model)
{
    std::vector<uint32_t> devices = opt.device_indices;
    if (devices.empty()) devices.push_back(model.opt.vulkan_device_index);
    for (const ExecutionGraph* graph : {&model.graph, &model.speculative.graph})
        for (const auto& layer : graph->layer_plans)
            if (std::find(devices.begin(), devices.end(), layer.vulkan_device_index) == devices.end())
                devices.push_back(layer.vulkan_device_index);
    model.memory_plan.gpu_devices.clear();
    for (uint32_t device : devices)
    {
        GpuResourceSnapshot initial = live_gpu_resources(model, device);
        const bool direct_allowance = opt.gpu_resources.empty() && opt.gpu_heap_budget != 0;
        if (direct_allowance)
        {
            usage_baselines[device] = initial.heap_usage;
            initial.heap_budget = initial.heap_budget > initial.heap_usage ? initial.heap_budget - initial.heap_usage : 0;
            initial.heap_budget = std::min(initial.heap_budget, opt.gpu_heap_budget);
            initial.heap_usage = 0;
        }
        uint64_t reserve = opt.gpu_expert_cache_reserve;
        for (size_t i = 0; i < opt.gpu_resources.size(); ++i)
        {
            if (opt.gpu_resources[i].device_index != device) continue;
            initial = opt.gpu_resources[i];
            reserve = i < opt.gpu_expert_cache_reserves.size() ? opt.gpu_expert_cache_reserves[i] : reserve;
            break;
        }
        explicit_cache_reserves[device] = reserve;
        if (has_flag(opt.flags, BackendVulkanExperts)
            && !has_flag(model.opt.flags, OptionDisableGpuExpertExecution))
            reserve = std::max(reserve, minimum_gpu_expert_reserve(model, device));
        GpuResourcePlan plan = make_gpu_resource_plan(initial, model.opt.num_concurrent_sessions, reserve);
        if (has_flag(opt.flags, BackendVulkanExperts) && !has_flag(model.opt.flags, OptionDisableGpuExpertExecution))
            commit_gpu_resources(plan, {0, 0, expert_wave_workspace_size(model, device)});
        model.memory_plan.gpu_devices.push_back(plan);
    }
}

bool ResourcePlanner::can_admit(uint32_t device, GpuResourceRequest request) const
{
    const GpuResourcePlan* plan = find(device);
    GpuResourceSnapshot live = live_gpu_resources(model, device);
    const auto baseline = usage_baselines.find(device);
    if (baseline != usage_baselines.end())
    {
        live.heap_budget = live.heap_budget > baseline->second ? live.heap_budget - baseline->second : 0;
        live.heap_usage = live.heap_usage > baseline->second ? live.heap_usage - baseline->second : 0;
    }
    return plan && can_admit_gpu_resources(*plan, live, request);
}

// Allocation has already happened. Only the additional unallocated state
// and workspace belong in the live check; the preflight checked weights.
bool ResourcePlanner::allocation_fits(uint32_t device, GpuResourceRequest request) const
{
    request.persistent_size = 0;
    return can_admit(device, request);
}

bool ResourcePlanner::explicit_cache_reserve(uint32_t device) const
{
    const auto reserve = explicit_cache_reserves.find(device);
    return reserve != explicit_cache_reserves.end() && reserve->second != 0;
}

void ResourcePlanner::refresh_expert_reserve(uint32_t device)
{
    for (GpuResourcePlan& plan : model.memory_plan.gpu_devices)
        if (plan.initial.device_index == device)
            plan.expert_cache_reserve = std::max(explicit_cache_reserves.at(device), minimum_gpu_expert_reserve(model, device));
}

void ResourcePlanner::commit(uint32_t device, GpuResourceRequest request)
{
    for (GpuResourcePlan& plan : model.memory_plan.gpu_devices)
        if (plan.initial.device_index == device) commit_gpu_resources(plan, request);
}

const GpuResourcePlan* ResourcePlanner::find(uint32_t device) const
{
    for (const GpuResourcePlan& plan : model.memory_plan.gpu_devices)
        if (plan.initial.device_index == device) return &plan;
    return nullptr;
}

// Backend products are published as a complete group. A failed creation or a
// driver-budget change must preserve both the CPU weights and earlier groups.
static Result<bool> prepare_resource_group(CompiledModel& model, ResourcePlanner& resources,
                                           uint32_t device, GpuResourceRequest request,
                                           CompiledLayerPlan* layer, const std::string& operation,
                                           const std::function<Result<void>()>& prepare,
                                           const std::function<bool()>& supported,
                                           bool allow_budget_fallback = false)
{
    // Zero denotes a capability decision made before selecting a GPU product.
    // Creation failures are errors; optional residency can retain its cache
    // execution path when the budget cannot hold persistent allocations.
    if (request.persistent_size == 0) return false;
    const std::string location = operation + " on device " + std::to_string(device);
    if (!resources.can_admit(device, request))
    {
        if (allow_budget_fallback) return false;
        return Error{ErrorCode::InvalidArgument, "Vulkan resource plan cannot allocate " + location
                                                     + " (persistent=" + std::to_string(request.persistent_size)
                                                     + ", session_state=" + std::to_string(request.session_state_size)
                                                     + ", workspace=" + std::to_string(request.workspace_size) + ")"};
    }
    CompiledOperatorTable previous = model.operators;
    CompiledLayerPlan previous_layer;
    if (layer) previous_layer = *layer;
    const auto prepared = prepare();
    const bool executable = prepared && supported();
    const bool allocation_fits = executable && resources.allocation_fits(device, request);
    if (!prepared || !executable || !allocation_fits)
    {
        model.operators = std::move(previous);
        if (layer) *layer = std::move(previous_layer);
        if (!prepared)
            return Error{prepared.error().code, "Vulkan preparation failed for " + location + ": " + prepared.error().message};
        if (!executable)
            return Error{ErrorCode::InternalError, "Vulkan preparation produced no executable product for " + location};
        if (allow_budget_fallback) return false;
        return Error{ErrorCode::InvalidArgument, "Vulkan resource allocation exceeded the live budget for " + location};
    }
    resources.commit(device, request);
    return true;
}

static uint64_t projection_storage(const CompiledModel& model, TensorHandle weight,
                                   uint64_t alignment, uint32_t device, TensorHandle bias = invalid_tensor_handle)
{
    return weight == invalid_tensor_handle ? 0
                                           : dense_weight_size(model, model.weights.at(weight), alignment, device,
                                                               bias != invalid_tensor_handle);
}

static uint64_t constant_storage(const CompiledModel& model, std::initializer_list<TensorHandle> handles,
                                 uint64_t alignment)
{
    uint64_t bytes = 0;
    for (TensorHandle handle : handles)
        if (handle != invalid_tensor_handle)
            bytes = saturating_add_u64(bytes, saturating_multiply_u64(model.weights.at(handle).element_count(), sizeof(float)));
    return bytes == 0 ? 0 : const_buf_size({bytes}, alignment);
}

static Result<GpuResourceRequest> attention_resource_request(const CompiledModel& model, const CompiledLayerPlan& layer)
{
    const AttentionBlockPlan& attention = layer.attention;
    const uint32_t device = layer.vulkan_device_index;
    GpuResourceRequest request;
    const uint64_t session_state_size = attention_session_state_size(model, attention, device);
    if (session_state_size == UINT64_MAX)
        return Error{ErrorCode::InvalidArgument, "Vulkan Attention state size exceeds the allocation range"};
    const uint64_t alignment = gpu_buffer_alignment(model, device);
    if (alignment == 0) return request;
    if (attention.kind == AttentionKind::Standard
        && attn_needs_native(model, attention, device)
        && !has_flag(model.opt.optimization_flags, OptimizationVulkanQnK)) return request;
    request.session_state_size = session_state_size;
    const uint64_t rows = std::max(1u, SessionOptions{}.prefill_chunk_size);
    const uint64_t query = saturating_multiply_u64(attention.head_count, attention.head_dimension);
    const uint64_t value = saturating_multiply_u64(attention.head_count, attention.value_head_dimension);
    // Uploaded input, normalized input, Q/K/V projections, attended values
    // and output projection can coexist until the submission completes.
    const uint64_t key = saturating_multiply_u64(attention.kv_head_count, attention.head_dimension);
    uint64_t columns = saturating_add_u64(saturating_multiply_u64(model.descriptor.hidden_size, 3),
                                          saturating_add_u64(query, saturating_add_u64(saturating_multiply_u64(key, 2), value)));
    if (attention.kind == AttentionKind::GatedDeltaNet)
    {
        // record_forward owns fused(q,k,v,z,beta,alpha), recurrent output and
        // projected output, alongside the uploaded layer input.
        columns = saturating_add_u64(saturating_multiply_u64(model.descriptor.hidden_size, 2),
                                     saturating_add_u64(saturating_multiply_u64(key, 2),
                                                        saturating_add_u64(saturating_multiply_u64(value, 3),
                                                                           saturating_multiply_u64(attention.head_count, 2))));
    }
    request.workspace_size = saturating_multiply_u64(saturating_multiply_u64(rows, columns), sizeof(float));
    if (attention.kind == AttentionKind::Standard || attention.kind == AttentionKind::GatedDeltaNet)
    {
        const uint64_t total = attn_vk_size(model, layer, alignment, device);
        if (total == UINT64_MAX)
            return Error{ErrorCode::InvalidArgument, "Vulkan Attention storage size exceeds the allocation range"};
        const uint64_t state = saturating_multiply_u64(request.session_state_size, std::max(1u, model.opt.num_concurrent_sessions));
        if (state == UINT64_MAX)
            return Error{ErrorCode::InvalidArgument, "Vulkan Attention state size exceeds the allocation range"};
        if (total >= state && total != 0) request.persistent_size = total - state;
        if (attention.kind == AttentionKind::GatedDeltaNet && request.persistent_size == 0)
        {
            const auto native = [&](TensorHandle handle) {
                const DType dtype = model.weights.at(handle).dtype;
                return dtype == DType::Float32 || dtype == DType::Int4 || dtype == DType::MxFp4 || is_qnk_dtype(dtype);
            };
            if (!native(attention.delta_qkv_weight) && !native(attention.delta_z_weight)
                && !native(attention.delta_beta_weight) && !native(attention.delta_alpha_weight)
                && !native(attention.output_weight)) return request;
            // Native quantized projections cannot use the BF16 fused estimate.
            for (TensorHandle handle : {attention.delta_qkv_weight, attention.delta_z_weight, attention.delta_beta_weight,
                                        attention.delta_alpha_weight, attention.output_weight})
            {
                const uint64_t size = projection_storage(model, handle, alignment, device);
                if (size == 0)
                {
                    request.persistent_size = 0;
                    return request;
                }
                request.persistent_size = saturating_add_u64(request.persistent_size, size);
            }
            request.persistent_size = saturating_add_u64(request.persistent_size,
                                                         constant_storage(model, {attention.delta_convolution_weight, attention.delta_time_bias, attention.delta_decay_log, attention.delta_norm_weight}, alignment));
        }
        return request;
    }
    if (attention.kind != AttentionKind::MultiHeadLatent) return GpuResourceRequest{};
    for (TensorHandle handle : {attention.query_a_weight, attention.query_b_weight,
                                attention.query_b_absorbed_weight, attention.key_value_weight,
                                attention.output_a_weight, attention.output_b_weight, attention.indexer_query_weight,
                                attention.indexer_weights_weight, attention.compressor_key_value_weight,
                                attention.compressor_gate_weight, attention.indexer_compressor_key_value_weight,
                                attention.indexer_compressor_gate_weight})
    {
        if (handle == invalid_tensor_handle) continue;
        const uint64_t size = projection_storage(model, handle, alignment, device);
        if (size == 0) return GpuResourceRequest{};
        request.persistent_size = saturating_add_u64(request.persistent_size, size);
    }
    request.persistent_size = saturating_add_u64(request.persistent_size,
                                                 constant_storage(model, {attention.pre_attention_norm_weight, attention.pre_attention_norm_bias, attention.query_norm_weight, attention.key_value_norm_weight, attention.sinks, attention.compressor_position, attention.compressor_norm_weight, attention.indexer_compressor_position, attention.indexer_compressor_norm_weight}, alignment));
    // The optional contiguous MLA layer additionally owns these hyper weights.
    if (model.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
        request.persistent_size = saturating_add_u64(request.persistent_size,
                                                     constant_storage(model, {layer.hyper_connection.attention_function, layer.hyper_connection.attention_scale, layer.hyper_connection.attention_base, layer.hyper_connection.ffn_function, layer.hyper_connection.ffn_scale, layer.hyper_connection.ffn_base, layer.moe.pre_ffn_norm_weight, layer.moe.pre_ffn_norm_bias}, alignment));
    return request;
}

static GpuResourceRequest ffn_resource_request(const CompiledModel& model, const CompiledLayerPlan& layer, bool shared)
{
    const ExpertPlan& ffn = shared ? layer.moe.shared_expert : layer.dense_ffn;
    const uint32_t device = layer.vulkan_device_index;
    const uint64_t alignment = gpu_buffer_alignment(model, device);
    if (alignment == 0 || ffn.gate_weight == invalid_tensor_handle || ffn.up_weight == invalid_tensor_handle
        || ffn.down_weight == invalid_tensor_handle) return {};
    if (shared)
    {
        const TensorData& gate_weight = model.weights.at(ffn.gate_weight);
        const TensorData& up_weight = model.weights.at(ffn.up_weight);
        const TensorData& down_weight = model.weights.at(ffn.down_weight);
        const bool router_gate = layer.moe.shared_expert_gate_weight != invalid_tensor_handle;
        const auto quantized = [](DType dtype) { return dtype == DType::Int4 || dtype == DType::MxFp4 || is_qnk_dtype(dtype); };
        const bool projection_chain = quantized(gate_weight.dtype) || quantized(up_weight.dtype) || quantized(down_weight.dtype)
                                      || (gate_weight.dtype == DType::BFloat16 && up_weight.dtype == DType::BFloat16
                                          && (down_weight.dtype != DType::BFloat16
                                              || (ffn.activation != ExpertActivation::Silu && ffn.activation != ExpertActivation::DeepSeekSwiGlu)
                                              || up_weight.shape.size() != 2 || up_weight.shape[0] % 128 != 0));
        const bool float8_chain = !router_gate && ffn.activation == ExpertActivation::DeepSeekSwiGlu
                                  && gate_weight.dtype == DType::Float8E4M3 && up_weight.dtype == DType::Float8E4M3
                                  && down_weight.dtype == DType::Float8E4M3;
        const bool bfloat16_chain = gate_weight.dtype == DType::BFloat16 && up_weight.dtype == DType::BFloat16
                                    && down_weight.dtype == DType::BFloat16
                                    && (!router_gate || model.weights.at(layer.moe.shared_expert_gate_weight).dtype == DType::BFloat16)
                                    && bf16_fused_storage_size(model,
                                                               {ffn.gate_weight, ffn.up_weight, layer.moe.shared_expert_gate_weight}, alignment, device)
                                           != 0;
        // Shared Expert kernels support a narrower activation/router contract
        // than their individual projections (notably FP8 DeepSeek SwiGLU).
        // An unsupported chain is a CPU placement decision before allocation.
        if (!projection_chain && !float8_chain && !bfloat16_chain) return {};
    }
    uint64_t size = 0;
    if (!shared)
    {
        // Native FFNs own their complete projection chain; its physical
        // allocations can differ from three independently prepared linears.
        size = quant_expert_bytes(model.weights.at(ffn.gate_weight),
                                  &model.weights.at(ffn.up_weight), model.weights.at(ffn.down_weight),
                                  device, model.vulkan_runtime, model.opt.optimization_flags,
                                  nullptr, nullptr, nullptr, ffn.activation, ffn.activation_limit);
    }
    if (size == 0)
    {
        const uint64_t gate = projection_storage(model, ffn.gate_weight, alignment, device);
        const uint64_t up = projection_storage(model, ffn.up_weight, alignment, device);
        const uint64_t down = projection_storage(model, ffn.down_weight, alignment, device);
        if (gate == 0 || up == 0 || down == 0) return {};
        uint64_t input = bf16_fused_storage_size(model, {ffn.gate_weight, ffn.up_weight, shared ? layer.moe.shared_expert_gate_weight : invalid_tensor_handle}, alignment, device);
        if (input == 0) input = saturating_add_u64(gate, up);
        size = saturating_add_u64(input, down);
    }
    if (shared && layer.moe.shared_expert_gate_weight != invalid_tensor_handle)
        size = saturating_add_u64(size, projection_storage(model, layer.moe.shared_expert_gate_weight, alignment, device));
    const uint64_t rows = SessionOptions{}.prefill_chunk_size;
    const uint64_t columns = saturating_add_u64(saturating_multiply_u64(model.descriptor.hidden_size, 2),
                                                saturating_multiply_u64(model.weights.at(ffn.up_weight).shape[0], 3));
    return {size, 0, saturating_multiply_u64(saturating_multiply_u64(rows, columns), sizeof(float))};
}

static bool ffn_kernel_ready(const CompiledModel& model, const CompiledLayerPlan& layer)
{
    const ExpertPlan& ffn = layer.dense_ffn;
    const TensorData& gate = model.weights.at(ffn.gate_weight);
    const TensorData& up = model.weights.at(ffn.up_weight);
    const TensorData& down = model.weights.at(ffn.down_weight);
    if (gate.dtype == DType::Float8E4M3 && up.dtype == DType::Float8E4M3 && down.dtype == DType::Float8E4M3
        && (ffn.activation == ExpertActivation::Silu || ffn.activation == ExpertActivation::DeepSeekSwiGlu)
        && up.shape.size() == 2 && up.shape[0] % 128 == 0)
        return true;
    if (quant_expert_bytes(gate, &up, down, layer.vulkan_device_index,
                           model.vulkan_runtime, model.opt.optimization_flags, nullptr, nullptr, nullptr, ffn.activation, ffn.activation_limit)
        != 0)
        return true;
    const uint64_t alignment = gpu_buffer_alignment(model, layer.vulkan_device_index);
    return (ffn.activation == ExpertActivation::Silu || ffn.activation == ExpertActivation::DeepSeekSwiGlu)
           && up.shape.size() == 2 && up.shape[0] % 128 == 0
           && bf16_fused_storage_size(model, {ffn.gate_weight, ffn.up_weight}, alignment, layer.vulkan_device_index) != 0
           && bf16_projection_supported(model, down, alignment, layer.vulkan_device_index);
}

Result<void> prepare_model_pipeline(CompiledModel& compiled, const CompilerOption& opt)
{
    compiled.operators.bind_weight_count(compiled.weights.size());
    if (compiled.opt.hybrid_mode == HybridMode::CpuOnly) return {};
    ResourcePlanner resources(compiled, opt);
    if (use_gated_residual_pipeline(compiled, opt))
    {
        const uint32_t device = compiled.opt.vulkan_device_index;
        const uint64_t alignment = gpu_buffer_alignment(compiled, device);
        GpuResourceRequest request;
        request.persistent_size = projection_storage(compiled, compiled.lm_head_weight, alignment, device, compiled.lm_head_bias);
        // This semantic chain owns more state than independent operators. The
        // shape estimate is retained; admission uses the common device ledger.
        const GpuResourceRequest execution = gated_residual_resources(compiled, device);
        request.session_state_size = execution.session_state_size;
        request.workspace_size = execution.workspace_size;
        for (const CompiledLayerPlan& layer : compiled.graph.layer_plans)
        {
            auto attention_result = attention_resource_request(compiled, layer);
            if (!attention_result) return attention_result.error();
            GpuResourceRequest attention = std::move(attention_result).value();
            if (attention.persistent_size == 0 && layer.attention.kind == AttentionKind::Standard)
            {
                for (TensorHandle weight : {layer.attention.query_weight, layer.attention.key_weight,
                                            layer.attention.value_weight, layer.attention.output_weight,
                                            layer.attention.output_gate_weight, layer.attention.qsa_query_key_weight})
                    attention.persistent_size = saturating_add_u64(attention.persistent_size, projection_storage(compiled, weight, alignment, device));
            }
            request.persistent_size = saturating_add_u64(request.persistent_size, attention.persistent_size);
            if (layer.moe.has_shared_expert)
                request.persistent_size = saturating_add_u64(request.persistent_size, ffn_resource_request(compiled, layer, true).persistent_size);
            request.persistent_size = saturating_add_u64(request.persistent_size, projection_storage(compiled, layer.moe.router_weight, alignment, device));
            request.persistent_size = saturating_add_u64(request.persistent_size,
                                                         constant_storage(compiled, {layer.attention_gated_residual.norm_weight, layer.attention_gated_residual.mix_down_weight, layer.attention_gated_residual.mix_up_weight, layer.attention_gated_residual.inject_weight, layer.ffn_gated_residual.norm_weight, layer.ffn_gated_residual.mix_down_weight, layer.ffn_gated_residual.mix_up_weight, layer.ffn_gated_residual.inject_weight}, alignment));
        }
        if (!resources.can_admit(device, request))
            return Error{ErrorCode::InvalidArgument, "gated residual Vulkan persistent weights and minimum working set exceed the GPU heap budget on device " + std::to_string(device)};
        const auto previous_operators = compiled.operators;
        const auto previous_layers = compiled.graph.layer_plans;
        auto prepared = prepare_gated_residual_pipeline(compiled);
        if (!prepared || !resources.allocation_fits(device, request))
        {
            compiled.gated_residual_device_head.reset();
            compiled.gated_residual_chain = false;
            compiled.operators = previous_operators;
            compiled.graph.layer_plans = previous_layers;
            if (!prepared) return prepared.error();
            return Error{ErrorCode::InvalidArgument, "gated residual Vulkan persistent weights and minimum working set exceed the GPU heap budget on device " + std::to_string(device)};
        }
        resources.commit(device, request);
        return {};
    }

    // Complete attention groups have priority over independently useful
    // projections. Main and speculative graphs share the same device budget.
    for (ExecutionGraph* graph : {&compiled.graph, &compiled.speculative.graph})
    {
        for (CompiledLayerPlan& layer : graph->layer_plans)
        {
            const auto kind = layer.attention.kind;
            const bool attention_enabled = has_flag(opt.flags, BackendVulkanAttention)
                                           && has_flag(compiled.opt.optimization_flags, OptimizationVulkanAttention);
            if (kind == AttentionKind::None || !attention_enabled) continue;
            if (kind == AttentionKind::Standard
                && (has_flag(layer.attention.flags, AttentionBlockQsa) || has_flag(layer.attention.flags, AttentionBlockExternalResidual))) continue;
            auto request_result = attention_resource_request(compiled, layer);
            if (!request_result) return request_result.error();
            const GpuResourceRequest request = std::move(request_result).value();
            auto prepared = prepare_resource_group(compiled, resources, layer.vulkan_device_index, request, &layer, "attention at layer " + std::to_string(layer.layer_id), [&]() -> Result<void> {
                    if (kind == AttentionKind::Standard) return prepare_attn_ops(compiled, layer);
                    if (kind == AttentionKind::GatedDeltaNet) { prepare_gdn_ops(compiled, layer); return {}; }
                    auto prepared = prepare_latent_ops(compiled, layer, "", attention_enabled);
                    if (prepared) prepare_latent_layer_ops(compiled, layer);
                    return prepared; }, [&] { return support_vulkan_attention(compiled.operators, layer.attention); });
            if (!prepared) return prepared.error();
        }
    }
    for (ExecutionGraph* graph : {&compiled.graph, &compiled.speculative.graph})
    {
        for (CompiledLayerPlan& layer : graph->layer_plans)
        {
            if (layer.moe.has_shared_expert)
            {
                auto prepared = prepare_resource_group(compiled, resources, layer.vulkan_device_index, ffn_resource_request(compiled, layer, true), &layer, "Shared Expert at layer " + std::to_string(layer.layer_id), [&] { return prepare_shared_ops(compiled.weights, compiled.operators, layer.moe,
                                                                                                                                                                                                                                                        layer.vulkan_device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags); }, [&] { return support_vulkan_shared_experts(compiled.operators, layer.moe); });
                if (!prepared) return prepared.error();
            }
        }
    }

    const uint32_t device = compiled.opt.vulkan_device_index;
    const uint64_t alignment = gpu_buffer_alignment(compiled, device);
    uint64_t head_size = projection_storage(compiled, compiled.lm_head_weight, alignment, device, compiled.lm_head_bias);
    if (head_size != 0)
    {
        head_size = saturating_add_u64(head_size, constant_storage(compiled, {compiled.final_norm_weight, compiled.final_norm_bias}, alignment));
        if (compiled.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn)
            head_size = saturating_add_u64(head_size, constant_storage(compiled,
                                                                       {compiled.hyper_head_function, compiled.hyper_head_scale, compiled.hyper_head_base}, alignment));
    }
    auto head_prepared = prepare_resource_group(compiled, resources, device, {head_size, 0, saturating_multiply_u64(compiled.descriptor.vocabulary_size, sizeof(float))}, nullptr, "LM head", [&]() -> Result<void> {
            auto prepared = prepare_lm_head_operator(compiled, false,
                bf16_projection_supported(compiled, compiled.weights.at(compiled.lm_head_weight), alignment, device));
            if (!prepared) return prepared;
            if (compiled.descriptor.hyper_connection_kind == HyperConnectionKind::Sinkhorn
                && compiled.descriptor.final_norm == NormType::RmsNorm && compiled.descriptor.norm_weight_offset == 0.0f
                && !has_flag(compiled.opt.flags, OptionRouterPrediction))
                compiled.operators.at_weight_mutable(compiled.hyper_head_function).layer_head = LayerHead_vulkan::create(compiled.weights.at(compiled.hyper_head_function), compiled.weights.at(compiled.hyper_head_scale),
                    compiled.weights.at(compiled.hyper_head_base), compiled.weights.at(compiled.final_norm_weight),
                    compiled.operators.at_weight(compiled.lm_head_weight), compiled.descriptor.hyper_connection_multiplier,
                    compiled.descriptor.norm_epsilon, compiled.descriptor.hyper_connection_epsilon,
                    device, compiled.vulkan_runtime, compiled.opt.optimization_flags);
            return {}; }, [&] { return uses_vulkan_dense_operator(compiled.operators.at_weight(compiled.lm_head_weight)); });
    if (!head_prepared) return head_prepared.error();

    // Automatic dense placement keeps frequently reused FFNs resident after
    // attention and the output head. Explicit cache capacity selects the
    // existing demand-cache policy, including its bounded-memory behavior.
    for (ExecutionGraph* graph : {&compiled.graph, &compiled.speculative.graph})
    {
        for (CompiledLayerPlan& layer : graph->layer_plans)
        {
            ExpertPlan& ffn = layer.dense_ffn;
            if (ffn.down_weight == invalid_tensor_handle
                || (ffn.use_gpu_cache && resources.explicit_cache_reserve(layer.vulkan_device_index))
                || !ffn_kernel_ready(compiled, layer)) continue;
            const bool cached = ffn.use_gpu_cache;
            const GpuResourceRequest request = ffn_resource_request(compiled, layer, false);
            auto prepared = prepare_resource_group(compiled, resources, layer.vulkan_device_index, request, &layer, "dense FFN at layer " + std::to_string(layer.layer_id), [&] { return prepare_ffn_operators(compiled.weights, compiled.operators, layer.dense_ffn,
                                                                                                                                                                                                               layer.vulkan_device_index, compiled.vulkan_runtime,
                                                                                                                                                                                                               compiled.opt.optimization_flags, false); }, [&] { return support_vulkan_ffn(compiled.operators, layer.dense_ffn); }, cached);
            if (!prepared) return prepared.error();
            if (prepared.value() && cached)
            {
                layer.dense_ffn.use_gpu_cache = false;
                resources.refresh_expert_reserve(layer.vulkan_device_index);
            }
        }
    }

    compiled.gpu_segments.resize(compiled.graph.layer_plans.size());
    if (has_flag(opt.flags, BackendVulkanExperts)
        && !has_flag(compiled.opt.flags, OptionDisableGpuExpertExecution))
    {
        const bool dense_chain = !compiled.graph.layer_plans.empty() && GpuSegment_vulkan::supports_head(compiled)
                                 && std::all_of(compiled.graph.layer_plans.begin(), compiled.graph.layer_plans.end(), [&](const auto& layer) {
                                        return layer.dense_ffn.down_weight != invalid_tensor_handle
                                               && layer.vulkan_device_index == device && GpuSegment_vulkan::supports(compiled, layer);
                                    });
        if (dense_chain)
        {
            GpuResourceRequest request;
            request.persistent_size = constant_storage(compiled, {compiled.final_norm_weight, compiled.final_norm_bias}, alignment);
            for (const auto& layer : compiled.graph.layer_plans)
                request.persistent_size = saturating_add_u64(request.persistent_size,
                                                             saturating_add_u64(constant_storage(compiled, {layer.dense_pre_ffn_norm_weight, layer.dense_pre_ffn_norm_bias}, alignment), alignment * 8));
            request.workspace_size = saturating_multiply_u64(SessionOptions{}.prefill_chunk_size,
                                                             saturating_multiply_u64(compiled.descriptor.hidden_size, 4 * sizeof(float)));
            if (resources.can_admit(device, request))
            {
                for (size_t index = 0; index < compiled.graph.layer_plans.size(); ++index)
                {
                    compiled.gpu_segments[index] = GpuSegment_vulkan::create(compiled, compiled.graph.layer_plans[index]);
                    if (!compiled.gpu_segments[index])
                        return Error{ErrorCode::InternalError, "cannot prepare selected resident dense segment at layer " + std::to_string(index)};
                }
                compiled.gpu_head = GpuSegment_vulkan::create_head(compiled);
                if (!compiled.gpu_head)
                    return Error{ErrorCode::InternalError, "cannot prepare selected resident dense head"};
                if (!resources.allocation_fits(device, request))
                    return Error{ErrorCode::InvalidArgument, "resident dense graph exceeds the live device budget"};
                resources.commit(device, request);
                compiled.resident_submission_rows = SessionOptions{}.prefill_chunk_size;
            }
        }
        // A mixed graph may contain a dense FFN beside routed layers. The
        // dense layer still needs a segment so the routed chain can continue
        // through it (and so a resident Expert bank chain can be selected).
        for (size_t index = 0; index < compiled.graph.layer_plans.size(); ++index)
        {
            CompiledLayerPlan& layer = compiled.graph.layer_plans[index];
            if (layer.dense_ffn.down_weight == invalid_tensor_handle || compiled.gpu_segments[index]) continue;
            if (!GpuSegment_vulkan::supports(compiled, layer)) continue;
            const uint32_t dense_device = layer.vulkan_device_index;
            const uint64_t dense_alignment = gpu_buffer_alignment(compiled, dense_device);
            GpuResourceRequest request;
            request.persistent_size = constant_storage(compiled, {layer.dense_pre_ffn_norm_weight, layer.dense_pre_ffn_norm_bias}, dense_alignment);
            request.persistent_size = saturating_add_u64(request.persistent_size, saturating_multiply_u64(dense_alignment, 8));
            request.workspace_size = saturating_multiply_u64(SessionOptions{}.prefill_chunk_size,
                                                             saturating_multiply_u64(compiled.descriptor.hidden_size, 4 * sizeof(float)));
            std::shared_ptr<GpuSegment_vulkan> segment;
            auto prepared = prepare_resource_group(compiled, resources, dense_device, request, &layer, "dense segment at layer " + std::to_string(layer.layer_id), [&]() -> Result<void> {
                                                       segment = GpuSegment_vulkan::create(compiled, layer);
                                                       if (!segment)
                                                           return Error{ErrorCode::InternalError, "cannot prepare selected dense Vulkan segment at layer " + std::to_string(layer.layer_id)};
                                                       return {}; }, [&] { return GpuSegment_vulkan::supports(compiled, layer); }, true);
            if (!prepared) return prepared.error();
            if (prepared.value())
                compiled.gpu_segments[index] = std::move(segment);
        }

        for (size_t index = 0; index < compiled.graph.layer_plans.size(); ++index)
        {
            const auto& layer = compiled.graph.layer_plans[index];
            if (layer.dense_ffn.down_weight != invalid_tensor_handle) continue;
            if (!GpuSegment_vulkan::supports(compiled, layer)
                || !support_vulkan_experts(compiled.weights, layer.moe, compiled.opt.optimization_flags)) continue;
            const uint32_t segment_device = layer.vulkan_device_index;
            const uint64_t segment_alignment = gpu_buffer_alignment(compiled, segment_device);
            // Router's InnerProduct owns FP32 weights; the two Norm recorders
            // own FP32 constants. Account those products before creating them.
            const auto& router = compiled.weights.at(layer.moe.router_weight);
            GpuResourceRequest request;
            request.persistent_size = saturating_add_u64(saturating_multiply_u64(router.element_count(), sizeof(float)),
                                                         constant_storage(compiled, {layer.moe.pre_ffn_norm_weight, layer.moe.pre_ffn_norm_bias, layer.attention.pre_attention_norm_weight, layer.attention.pre_attention_norm_bias, layer.moe.router_bias, layer.moe.router_selection_bias}, segment_alignment));
            request.persistent_size = saturating_add_u64(request.persistent_size, saturating_multiply_u64(segment_alignment, 8));
            const uint64_t columns = saturating_add_u64(saturating_multiply_u64(compiled.descriptor.hidden_size, 4),
                                                        saturating_add_u64(layer.moe.experts.size(), Router_vulkan::selected_columns(layer.moe.top_k)));
            request.workspace_size = saturating_multiply_u64(saturating_multiply_u64(SessionOptions{}.prefill_chunk_size, columns), sizeof(float));
            if (!resources.can_admit(segment_device, request))
                return Error{ErrorCode::InvalidArgument, "Vulkan resource plan cannot allocate resident graph segment at layer " + std::to_string(layer.layer_id)};
            auto segment = GpuSegment_vulkan::create(compiled, layer);
            if (!segment)
                return Error{ErrorCode::InternalError, "cannot prepare selected Vulkan graph segment at layer " + std::to_string(layer.layer_id)};
            if (!resources.allocation_fits(segment_device, request))
                return Error{ErrorCode::InvalidArgument, "Vulkan graph segment exceeds the live device budget"};
            compiled.gpu_segments[index] = std::move(segment);
            resources.commit(segment_device, request);
        }
        if (!compiled.gpu_head && !compiled.gpu_segments.empty() && compiled.gpu_segments.back()
            && GpuSegment_vulkan::supports_head(compiled))
        {
            GpuResourceRequest request{constant_storage(compiled, {compiled.final_norm_weight, compiled.final_norm_bias}, alignment), 0, 0};
            if (!resources.can_admit(device, request))
                return Error{ErrorCode::InvalidArgument, "Vulkan resource plan cannot allocate resident graph head"};
            compiled.gpu_head = GpuSegment_vulkan::create_head(compiled);
            if (!compiled.gpu_head)
                return Error{ErrorCode::InternalError, "cannot prepare selected Vulkan graph head"};
            if (!resources.allocation_fits(device, request))
                return Error{ErrorCode::InvalidArgument, "Vulkan graph head exceeds the live device budget"};
            resources.commit(device, request);
        }
    }
    return {};
}

static uint32_t graph_node_device(const CompiledModel& model, const ExecutionGraph& graph,
                                  const Layer& node)
{
    return node.plan_index < graph.layer_plans.size()
               ? graph.layer_plans[node.plan_index].vulkan_device_index
               : model.opt.vulkan_device_index;
}

static bool graph_has_resident_chain(const CompiledModel& model, const ExecutionGraph& graph)
{
    return &graph == &model.graph && model.resident_submission_rows != 0 && !graph.layer_plans.empty();
}

static bool graph_has_expert_banks(const CompiledModel& model, const ExecutionGraph& graph)
{
    return graph_has_resident_chain(model, graph) && model.resident_bank_layers != 0
           && model.gpu_expert_bank_plans.size() == graph.layer_plans.size();
}

static uint64_t graph_tensor_elements(const CompiledModel& model, const ExecutionGraph& graph,
                                      const ExecutionTensor& tensor, uint32_t rows, uint32_t logits_rows)
{
    if (tensor.producer >= graph.layers.size()) return UINT64_MAX;
    const Layer& producer = graph.layers[tensor.producer];
    // Full-bank recording folds routed aggregation and the following Combine
    // into one output. The ExpertGroup tensor is not materialized there.
    if (producer.kind == LayerKind::ExpertGroup && graph_has_expert_banks(model, graph)) return 0;
    // The cache output documents the state dependency; the cache has a
    // Session lifetime and is not an activation in the reusable graph arena.
    if (producer.kind == LayerKind::Attention && producer.outputs.size() > 1
        && producer.outputs[1] == tensor.id) return 0;
    uint64_t elements = 1;
    for (size_t dimension = 0; dimension < tensor.shape.size(); ++dimension)
    {
        const uint64_t extent = dimension == 0 && tensor.shape[dimension] == 0
                                    ? (producer.kind == LayerKind::LmHead ? logits_rows : rows)
                                    : tensor.shape[dimension];
        elements = saturating_multiply_u64(elements, extent);
    }
    if (producer.kind == LayerKind::ExpertGroup && producer.plan_index < graph.layer_plans.size())
        elements = saturating_multiply_u64(elements, graph.layer_plans[producer.plan_index].moe.top_k);
    if (model.descriptor.hyper_connection_kind != HyperConnectionKind::None
        && (producer.kind == LayerKind::TokenEmbedding || producer.kind == LayerKind::Attention
            || producer.kind == LayerKind::Combine || producer.kind == LayerKind::DenseFfn))
        elements = saturating_multiply_u64(elements, model.descriptor.hyper_connection_multiplier);
    return elements;
}

static Result<void> append_graph_buffer(std::vector<GraphMemoryBuffer>& buffers, uint32_t device,
                                        uint64_t elements, uint64_t alignment, uint32_t first, uint32_t last,
                                        uint64_t element_size = sizeof(float))
{
    if (elements == 0) return {};
    const uint64_t bytes = saturating_multiply_u64(elements, element_size);
    if (bytes == UINT64_MAX || buffers.size() >= UINT32_MAX)
        return Error{ErrorCode::InvalidArgument, "Vulkan graph memory shape exceeds the allocation range"};
    buffers.push_back({static_cast<uint32_t>(buffers.size()), device, bytes, alignment, first, last});
    return {};
}

static uint64_t proj_columns(const CompiledModel& model, TensorHandle weight)
{
    if (weight == invalid_tensor_handle) return 0;
    const TensorData& matrix = model.weights.at(weight);
    return matrix.shape.size() == 2 ? matrix.shape[1] : 0;
}

static bool graph_has_gpu_segment(const CompiledModel& model, const ExecutionGraph& graph, size_t index)
{
    return &graph == &model.graph && index < model.gpu_segments.size() && model.gpu_segments[index];
}

uint32_t resident_chunk_layers(const CompiledModel& model, uint32_t rows) noexcept
{
    if (model.resident_submission_rows == 0 || model.graph.layer_plans.empty()) return 0;
    return static_cast<uint32_t>(std::min<size_t>(model.graph.layer_plans.size(),
                                                  std::max(1u, model.resident_submission_rows / std::max(1u, rows))));
}

// Opaque recorders retain their VkMat operands through the submission fence.
// These are their private intermediates only: graph inputs, graph outputs,
// persistent weights/state and host-visible transfer-slot storage are omitted.
static Result<void> append_graph_node_scratch(const CompiledModel& model, const ExecutionGraph& graph,
                                              const Layer& node, uint32_t rows, uint32_t context_tokens,
                                              uint32_t sequence_count,
                                              uint64_t alignment, uint32_t first, uint32_t last,
                                              std::vector<GraphMemoryBuffer>& buffers)
{
    const uint32_t device = graph_node_device(model, graph, node);
    const uint64_t hidden = model.descriptor.hidden_size;
    const auto add = [&](uint64_t columns, uint64_t count = 0) {
        return append_graph_buffer(buffers, device, saturating_multiply_u64(columns, count == 0 ? rows : count), alignment, first, last);
    };
    const auto add_projection = [&](TensorHandle weight, uint64_t count = 0) -> Result<void> {
        if (weight == invalid_tensor_handle || model.weights.at(weight).dtype != DType::Float8E4M3) return {};
        const uint64_t width = proj_columns(model, weight);
        if (width > UINT32_MAX)
            return Error{ErrorCode::InvalidArgument, "Vulkan projection input exceeds the allocation range"};
        for (uint64_t elements : Float8Linear_vulkan::scratch_elements(static_cast<uint32_t>(width), count == 0 ? rows : count))
        {
            auto ret = add(elements, 1);
            if (!ret) return ret.error();
        }
        return {};
    };
    if (node.kind == LayerKind::LmHead)
    {
        const uint64_t copies = model.descriptor.hyper_connection_multiplier;
        const uint64_t expanded = saturating_multiply_u64(hidden, copies);
        std::vector<uint64_t> head_columns;
        if (model.gated_residual_device_head)
        {
            const uint64_t rank = model.weights.at(model.gated_residual_head.mix_down_weight).shape[0];
            head_columns = {expanded, expanded, rank, rank, expanded, hidden};
        }
        else if (model.operators.at_weight(model.hyper_head_function).layer_head)
            head_columns = {expanded, expanded, copies, copies, hidden};
        for (uint64_t columns : head_columns)
        {
            auto ret = add(columns);
            if (!ret) return ret.error();
        }
        return add_projection(model.lm_head_weight);
    }
    if (node.plan_index >= graph.layer_plans.size()) return {};
    const CompiledLayerPlan& layer = graph.layer_plans[node.plan_index];
    const AttentionBlockPlan& attention = layer.attention;
    if (node.kind == LayerKind::Attention)
    {
        std::vector<uint64_t> elements;
        if (attention.kind == AttentionKind::GatedDeltaNet)
        {
            elements = GatedDeltaNet_vulkan::scratch_elements(attention.head_count, attention.kv_head_count,
                                                              attention.head_dimension, attention.value_head_dimension,
                                                              attention.delta_gate_channels, rows);
        }
        else if (attention.kind == AttentionKind::Standard && !has_flag(attention.flags, AttentionBlockQsa))
        {
            bool direct_ring = false;
#if NCNN_MOE_WITH_VULKAN
            const auto& op = model.operators.at(attention.vulkan_attention_operator).attention;
            direct_ring = op && op->uses_qkv_ring(rows);
            if (rows == 1 || sequence_count > 1)
            {
                uint64_t workspace_bytes = 0;
                if (op && !op->decode_workspace_bytes(context_tokens, workspace_bytes))
                    return Error{ErrorCode::InternalError, "failed to plan Vulkan Attention decode workspace"};
                // Each independent Session decode retains its own workspace.
                for (uint32_t index = 0; workspace_bytes != 0 && index < std::min(rows, sequence_count); ++index)
                {
                    auto ret = add(workspace_bytes / sizeof(float), 1);
                    if (!ret) return ret.error();
                }
            }
#endif
            const uint64_t kv_size = vulkan_kv_size(attention.kv_cache_dtype, device);
            if (kv_size == UINT64_MAX)
                return Error{ErrorCode::InvalidArgument, "Vulkan Attention KV device is invalid"};
            elements = Attention_vulkan::scratch_elements(attn_vk_config(model, attention),
                                                          rows, context_tokens, sequence_count, kv_size, direct_ring);
            for (TensorHandle weight : {attention.query_weight, attention.key_weight, attention.value_weight,
                                        attention.output_gate_weight, attention.output_weight})
            {
                auto ret = add_projection(weight);
                if (!ret) return ret.error();
            }
        }
        else if (attention.kind == AttentionKind::Standard)
        {
            elements = QsaAttention_vulkan::scratch_elements(attention, model.descriptor.hidden_size, rows, context_tokens);
        }
        else if (attention.kind == AttentionKind::MultiHeadLatent)
        {
            elements = LatentAttention_vulkan::scratch_elements(attention, model.descriptor.hidden_size, rows, context_tokens);
            for (TensorHandle weight : {attention.query_a_weight, attention.query_b_weight, attention.key_value_weight,
                                        attention.output_a_weight, attention.output_b_weight, attention.indexer_query_weight,
                                        attention.indexer_weights_weight, attention.compressor_key_value_weight, attention.compressor_gate_weight,
                                        attention.indexer_compressor_key_value_weight, attention.indexer_compressor_gate_weight})
            {
                auto ret = add_projection(weight);
                if (!ret) return ret.error();
            }
        }
        for (uint64_t count : elements)
        {
            auto ret = add(count, 1);
            if (!ret) return ret.error();
        }
        return {};
    }
    if (node.kind == LayerKind::DenseFfn || node.kind == LayerKind::SharedExpertGroup
        || node.kind == LayerKind::ExpertGroup)
    {
        const bool routed = node.kind == LayerKind::ExpertGroup;
        const bool shared = node.kind == LayerKind::SharedExpertGroup;
        if (routed && graph_has_expert_banks(model, graph)
            && node.plan_index < model.resident_bank_layers)
        {
            const ExpertBankPlan_vulkan& bank = model.gpu_expert_bank_plans[node.plan_index];
            for (uint64_t elements : ExpertBank_vulkan::scratch_elements(bank, rows, layer.moe.top_k))
            {
                auto ret = add(elements, 1);
                if (!ret) return ret.error();
            }
            return {};
        }
        const bool segment_input = graph_has_gpu_segment(model, graph, node.plan_index);
        if (node.kind == LayerKind::DenseFfn && graph_has_resident_chain(model, graph))
        {
            auto branch = add(hidden);
            if (!branch) return branch.error();
        }
        const bool resident_input = segment_input || model.gated_residual_chain
                                    || (&graph == &model.graph && !has_flag(model.opt.flags, OptionRouterPrediction)
                                        && model.operators.at_weight(layer.attention.query_a_weight).latent_layer);
        const auto& experts = layer.moe.experts;
        if (routed && experts.empty())
            return Error{ErrorCode::InvalidModel, "Vulkan Expert graph node has no Experts"};
        const ExpertPlan& expert = routed ? experts.front() : shared ? layer.moe.shared_expert
                                                                     : layer.dense_ffn;
        const TensorHandle gate = expert.gate_up_weight != invalid_tensor_handle ? expert.gate_up_weight : expert.gate_weight;
        uint64_t intermediate = proj_columns(model, expert.down_weight);
        if (intermediate == 0) intermediate = model.descriptor.intermediate_size;
        const uint64_t count = saturating_multiply_u64(rows, routed ? layer.moe.top_k : 1);
        const DType dtype = gate == invalid_tensor_handle ? DType::Float32 : model.weights.at(gate).dtype;
        const bool fused_bfloat16 = model.operators.at(expert.vulkan_operator).bfloat16
                                    || (shared && model.operators.at(layer.moe.fused_shared_input_bfloat16_operator).bfloat16)
                                    || (routed && expert.gate_up_weight != invalid_tensor_handle && dtype == DType::BFloat16
                                        && model.weights.at(expert.down_weight).dtype == DType::BFloat16);
        const bool qnk = model.operators.at(expert.vulkan_operator).qnk_expert
                         || dtype == DType::Int4 || dtype == DType::MxFp4 || is_qnk_dtype(dtype) || dtype == DType::Float32
                         || (dtype == DType::BFloat16 && !fused_bfloat16);
        if (intermediate > UINT32_MAX / 2)
            return Error{ErrorCode::InvalidArgument, "Vulkan FFN intermediate exceeds the allocation range"};
        const uint32_t width = static_cast<uint32_t>(intermediate);
        const bool router_gate = shared && layer.moe.shared_expert_gate_weight != invalid_tensor_handle;
        const bool fp8_chain = !routed && (model.operators.at(expert.vulkan_operator).float8 || (shared && !router_gate && expert.activation == ExpertActivation::DeepSeekSwiGlu && model.operators.at_weight(expert.gate_weight).float8 && model.operators.at_weight(expert.up_weight).float8 && model.operators.at_weight(expert.down_weight).float8));
        std::vector<uint64_t> scratch;
        if (qnk)
            scratch = QnkExpert_vulkan::scratch_elements(width, model.descriptor.hidden_size,
                                                         expert.gate_up_weight == invalid_tensor_handle, router_gate, count);
        else if (fp8_chain)
            scratch = Float8Linear_vulkan::swiglu_scratch_elements(model.descriptor.hidden_size, width, count, shared && resident_input);
        else if (fused_bfloat16 && !routed)
            scratch = Bfloat16Linear_vulkan::swiglu_scratch_elements(width, router_gate, count);
        else if (fused_bfloat16)
            scratch = Bfloat16Expert_vulkan::scratch_elements(width * 2, count);
        else
            // Separate projection outputs connect the unfused FFN operators.
            scratch = {saturating_multiply_u64(width, count), saturating_multiply_u64(width, count)};
        for (uint64_t elements : scratch)
        {
            auto ret = add(elements, 1);
            if (!ret) return ret.error();
        }
        if (routed)
        {
            if (!resident_input || segment_input)
            {
                // Generic routed FFN keeps the full normalized upload while
                // waves gather their route-major input (the graph copy below).
                // The last wave also owns one token-major reduced output until
                // its download completes, alongside all route-major results.
                for (unsigned int operand = 0; operand < (segment_input ? 0u : 2u); ++operand)
                {
                    auto ret = add(hidden);
                    if (!ret) return ret.error();
                }
                // Each active Expert owns one CSR buffer: T+1 offsets and
                // two words per route. Include each buffer's word-granular
                // allocator rounding before representing their total here.
                // At most min(T * top_k, Expert count) Experts can be active.
                const uint64_t active_experts = std::min<uint64_t>(count, experts.size());
                const uint64_t per_expert_offsets = saturating_add_u64(rows, alignment / sizeof(uint32_t));
                const uint64_t csr_words = saturating_add_u64(saturating_multiply_u64(per_expert_offsets, active_experts),
                                                              saturating_multiply_u64(count, 2));
                auto metadata = add(csr_words, 1);
                if (!metadata) return metadata.error();
            }
            // Gathered route input and compact metadata. The graph's Expert
            // output instance is route-major and survives through Combine;
            // counting it here too would duplicate the same device storage.
            std::vector<uint64_t> route_columns = {1};
            if (resident_input) route_columns.insert(route_columns.end(), {1, 1, hidden});
            // Misaligned per-Expert row views need independent input/output
            // allocations in the actual batched kernels. Ordinary aligned
            // hidden widths do not allocate these copies.
            if (saturating_multiply_u64(hidden, sizeof(float)) % alignment != 0
                && std::min<uint64_t>(count, layer.moe.experts.size()) > 1)
                route_columns.insert(route_columns.end(), {hidden, hidden});
            for (uint64_t columns : route_columns)
            {
                auto ret = add(columns, count);
                if (!ret) return ret.error();
            }
        }
        if (router_gate && !qnk && !fused_bfloat16)
        {
            for (uint64_t columns : {hidden, UINT64_C(1)})
            {
                auto ret = add(columns);
                if (!ret) return ret.error();
            }
        }
        if (shared && segment_input && !graph_has_expert_banks(model, graph))
        {
            // Non-bank recorded tails retain aggregate + shared before
            // adding the residual. Bank recording folds both additions into
            // the final hidden output.
            auto sum = add(hidden);
            if (!sum) return sum.error();
        }
        for (TensorHandle weight : {gate, expert.up_weight, expert.down_weight})
        {
            // The fused FP8 owner already accounts for its input clone.
            if (fp8_chain) continue;
            auto ret = add_projection(weight, count);
            if (!ret) return ret.error();
        }
    }
    return {};
}

static Result<void> append_graph_region_memory(const CompiledModel& model, const ExecutionGraph& graph,
                                               const ExecutionSchedule& schedule, uint32_t rows, uint32_t logits_rows,
                                               uint32_t context_tokens, uint32_t sequence_count, uint32_t node_offset,
                                               std::vector<GraphMemoryBuffer>& buffers,
                                               std::vector<GraphMemorySubmission>& submissions)
{
    if (graph.layers.empty()) return {};
    auto valid = schedule.validate(graph);
    if (!valid) return valid.error();
    std::vector<uint32_t> positions(graph.layers.size());
    for (uint32_t position = 0; position < schedule.node_order.size(); ++position)
        positions[schedule.node_order[position]] = node_offset + position;
    const bool resident_chain = graph_has_resident_chain(model, graph);
    if (resident_chain)
    {
        const uint32_t chunk = resident_chunk_layers(model, rows);
        std::vector<ExecutionNodeId> attention(graph.layer_plans.size(), invalid_execution_node_id);
        std::vector<ExecutionNodeId> combine(graph.layer_plans.size(), invalid_execution_node_id);
        ExecutionNodeId head = invalid_execution_node_id;
        for (const Layer& node : graph.layers)
        {
            if (node.kind == LayerKind::LmHead) head = node.id;
            if (node.plan_index >= graph.layer_plans.size()) continue;
            if (node.kind == LayerKind::Attention) attention[node.plan_index] = node.id;
            if (node.kind == LayerKind::Combine || node.kind == LayerKind::DenseFfn) combine[node.plan_index] = node.id;
        }
        for (uint32_t first = 0; first < graph.layer_plans.size(); first += chunk)
        {
            const uint32_t last = std::min<uint32_t>(static_cast<uint32_t>(graph.layer_plans.size()), first + chunk) - 1;
            const ExecutionNodeId finish = last + 1 == graph.layer_plans.size() ? head : combine[last];
            if (attention[first] == invalid_execution_node_id || finish == invalid_execution_node_id)
                return Error{ErrorCode::InvalidModel, "resident graph has no recording boundary"};
            const uint32_t device = graph_node_device(model, graph, graph.layers[attention[first]]);
            submissions.push_back({device, positions[attention[first]], positions[finish]});
            // One zeroed validation word is shared by the entire chunk. Its
            // blob allocation is rounded to the real device alignment; host
            // upload/download slots use the separate staging allocator.
            if (graph_has_expert_banks(model, graph))
            {
                const uint64_t alignment = std::max<uint64_t>(4, gpu_buffer_alignment(model, device));
                auto flag = append_graph_buffer(buffers, device, alignment, alignment,
                                                positions[attention[first]], positions[finish], 1);
                if (!flag) return flag.error();
            }
        }
    }
    else if (&graph == &model.graph)
        for (ExecutionNodeId node_id : schedule.node_order)
        {
            const Layer& source = graph.layers[node_id];
            if (source.kind != LayerKind::ExpertGroup) continue;
            uint32_t fence_id = invalid_execution_node_id;
            const auto resolved = GpuSegment_vulkan::continuation_target(model, node_id, fence_id);
            if (!resolved) return resolved.error();
            const uint32_t target_id = resolved.value();
            if (target_id == invalid_execution_node_id) continue;
            if (target_id >= graph.layers.size() || fence_id >= graph.layers.size()
                || positions[target_id] <= positions[node_id] || positions[fence_id] <= positions[target_id])
                return Error{ErrorCode::InvalidModel, "GPU continuation memory target or fence does not follow its Expert group"};
            const uint32_t device = graph_node_device(model, graph, source);
            submissions.push_back({device, positions[node_id], positions[fence_id]});
        }
    std::vector<uint32_t> starts = positions;
    std::vector<uint32_t> ends = positions;
    std::vector<uint32_t> layer_finish(graph.layer_plans.size(), 0);
    std::vector<bool> fused_layers(graph.layer_plans.size(), false);
    for (size_t index = 0; index < graph.layer_plans.size(); ++index)
    {
        const CompiledLayerPlan& layer = graph.layer_plans[index];
        fused_layers[index] = graph_has_gpu_segment(model, graph, index) || model.gated_residual_chain
                              || (&graph == &model.graph && !has_flag(model.opt.flags, OptionRouterPrediction)
                                  && model.operators.at_weight(layer.attention.query_a_weight).latent_layer);
    }
    if (has_flag(model.opt.optimization_flags, OptimizationVulkanSharedExpertOverlap))
        for (const Layer& shared : graph.layers)
        {
            if (shared.kind != LayerKind::SharedExpertGroup || shared.backend != ExecutionBackend::Vulkan) continue;
            for (const Layer& expert : graph.layers)
                if (expert.kind == LayerKind::ExpertGroup && expert.backend == ExecutionBackend::Vulkan
                    && expert.plan_index == shared.plan_index)
                    starts[shared.id] = std::min(starts[shared.id], positions[expert.id]);
        }
    for (size_t index = 0; index < schedule.node_order.size(); ++index)
    {
        const Layer& node = graph.layers[schedule.node_order[index]];
        if ((node.kind != LayerKind::Combine && !(resident_chain && node.kind == LayerKind::DenseFfn)) || node.plan_index >= fused_layers.size()
            || !fused_layers[node.plan_index]) continue;
        // Fused layers defer the combine and FFN post until the next device
        // Attention, materialization, or final head submission actually runs.
        uint32_t finish = positions[node.id];
        if (!resident_chain && index + 1 < schedule.node_order.size())
        {
            finish = positions[schedule.node_order[index + 1]];
            if ((model.gpu_head || model.gated_residual_device_head || model.operators.at_weight(model.hyper_head_function).layer_head)
                && graph.layers[schedule.node_order[index + 1]].kind == LayerKind::FinalNorm
                && index + 2 < schedule.node_order.size())
                finish = positions[schedule.node_order[index + 2]];
        }
        layer_finish[node.plan_index] = finish;
    }
    // The logical Shared Expert and residual nodes are recorded in the next
    // device segment. Their operands coexist with that segment's Attention.
    for (const Layer& node : graph.layers)
    {
        if (!graph_has_gpu_segment(model, graph, node.plan_index)) continue;
        if (!resident_chain && (node.kind == LayerKind::SharedExpertGroup || node.kind == LayerKind::Combine))
            starts[node.id] = ends[node.id] = layer_finish[node.plan_index];
        if (node.kind == LayerKind::Router)
            for (const Layer& attention : graph.layers)
                if (attention.kind == LayerKind::Attention && attention.plan_index == node.plan_index)
                    starts[node.id] = positions[attention.id];
    }
    for (const ExecutionTensor& tensor : graph.tensors)
    {
        if (tensor.producer >= graph.layers.size())
            return Error{ErrorCode::InvalidModel, "graph memory tensor has no scheduled producer"};
        const Layer& producer = graph.layers[tensor.producer];
        const uint64_t elements = graph_tensor_elements(model, graph, tensor, rows, logits_rows);
        if (elements == 0) continue;
        const uint32_t producer_device = graph_node_device(model, graph, producer);
        uint32_t last = ends[tensor.producer];
        std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t>> copies;
        std::map<std::pair<uint32_t, uint32_t>, uint64_t> copy_elements;
        for (ExecutionNodeId consumer_id : tensor.consumers)
        {
            if (consumer_id >= graph.layers.size() || positions[consumer_id] <= positions[tensor.producer])
                return Error{ErrorCode::InvalidModel, "graph memory tensor consumer precedes its producer"};
            const Layer& consumer = graph.layers[consumer_id];
            last = std::max(last, ends[consumer_id]);
            if (consumer.backend != ExecutionBackend::Vulkan) continue;
            // Assignments are control edges. The Expert recorder uploads its
            // own compact route buffers, accounted with its opaque operands.
            if (consumer.kind == LayerKind::ExpertGroup && consumer.inputs.size() > 1
                && consumer.inputs[1] == tensor.id) continue;
            const uint32_t device = graph_node_device(model, graph, consumer);
            const bool host_normalized = (consumer.kind == LayerKind::ExpertGroup || consumer.kind == LayerKind::SharedExpertGroup
                                          || consumer.kind == LayerKind::DenseFfn)
                                         && (consumer.plan_index >= fused_layers.size() || !fused_layers[consumer.plan_index]);
            if (producer.backend == ExecutionBackend::Vulkan && device == producer_device && !host_normalized) continue;
            // Generic routed FFN gathers route-major input on the device from
            // one normalized upload. Its gathered allocation is counted here;
            // the retained upload and final aggregate are opaque node scratch.
            // Shared and dense FFN own independent host-normalized operands.
            const std::pair<uint32_t, uint32_t> key{device, host_normalized ? consumer.id : invalid_execution_node_id};
            auto [it, inserted] = copies.try_emplace(key, starts[consumer_id], positions[consumer_id]);
            if (!inserted) it->second.second = std::max(it->second.second, positions[consumer_id]);
            uint64_t consumed = elements;
            if (consumer.kind == LayerKind::LmHead)
                consumed = saturating_multiply_u64(logits_rows, model.descriptor.hidden_size);
            else if (consumer.kind == LayerKind::ExpertGroup || consumer.kind == LayerKind::SharedExpertGroup
                     || consumer.kind == LayerKind::DenseFfn)
                consumed = saturating_multiply_u64(rows, model.descriptor.hidden_size);
            if (host_normalized && consumer.kind == LayerKind::ExpertGroup)
                consumed = saturating_multiply_u64(consumed, graph.layer_plans[consumer.plan_index].moe.top_k);
            copy_elements[key] = std::max(copy_elements[key], consumed);
        }
        if (producer.plan_index < layer_finish.size()
            && (producer.kind == LayerKind::Attention || producer.kind == LayerKind::ExpertGroup
                || producer.kind == LayerKind::SharedExpertGroup))
            last = std::max(last, layer_finish[producer.plan_index]);
        // Separate submissions release route-major slots at this Expert node.
        // Selected continuation ranges retain them through the shared fence.
        if (producer.kind == LayerKind::ExpertGroup && graph_has_gpu_segment(model, graph, producer.plan_index))
            last = positions[producer.id];
        if (producer.backend == ExecutionBackend::Vulkan)
        {
            auto ret = append_graph_buffer(buffers, producer_device, elements,
                                           std::max<uint64_t>(4, gpu_buffer_alignment(model, producer_device)),
                                           starts[tensor.producer], last);
            if (!ret) return ret.error();
        }
        for (const auto& [key, lifetime] : copies)
        {
            const uint32_t device = key.first;
            auto ret = append_graph_buffer(buffers, device, copy_elements[key],
                                           std::max<uint64_t>(4, gpu_buffer_alignment(model, device)), lifetime.first, lifetime.second);
            if (!ret) return ret.error();
        }
    }
    for (ExecutionNodeId node_id : schedule.node_order)
    {
        const Layer& node = graph.layers[node_id];
        if (node.backend != ExecutionBackend::Vulkan) continue;
        const uint32_t device = graph_node_device(model, graph, node);
        const uint32_t position = positions[node_id];
        const uint64_t alignment = std::max<uint64_t>(4, gpu_buffer_alignment(model, device));
        if (!resident_chain && node.kind == LayerKind::ExpertGroup && graph_has_gpu_segment(model, graph, node.plan_index))
        {
            // One token-major aggregate survives to the next segment. The
            // submission ranges also retain final-wave route-major operands
            // when that segment is recorded before the same fence.
            auto aggregate = append_graph_buffer(buffers, device, saturating_multiply_u64(model.descriptor.hidden_size, rows), alignment,
                                                 position, std::max(position, layer_finish[node.plan_index]));
            if (!aggregate) return aggregate.error();
        }
        if (node.kind == LayerKind::Attention && node.plan_index < fused_layers.size()
            && fused_layers[node.plan_index])
        {
            const CompiledLayerPlan& layer = graph.layer_plans[node.plan_index];
            const uint64_t hidden = model.descriptor.hidden_size;
            const uint64_t copies = model.descriptor.hyper_connection_multiplier;
            const uint64_t expanded = saturating_multiply_u64(hidden, copies);
            std::vector<uint64_t> private_columns;
            const bool resident_segment = graph_has_gpu_segment(model, graph, node.plan_index);
            if (resident_segment)
            {
                // Compact router selection and the residual/Norm temporaries
                // recorded together with Attention. Hidden itself is a graph
                // output; normalized is retained separately below.
                const bool dense = layer.dense_ffn.down_weight != invalid_tensor_handle;
                private_columns = dense ? std::vector<uint64_t>{hidden}
                                        : std::vector<uint64_t>{hidden, Router_vulkan::selected_columns(layer.moe.top_k)};
                if (layer.attention.kind == AttentionKind::GatedDeltaNet) private_columns.push_back(hidden);
                // InnerProduct can pack groups of four rows. The Router's
                // pack1 logits then coexist with the projection allocation.
                if (!dense && rows % 4 == 0) private_columns.push_back(layer.moe.experts.size());
            }
            else if (model.gated_residual_chain)
            {
                for (const GatedResidualPlan* pre : {&layer.attention_gated_residual, &layer.ffn_gated_residual})
                {
                    const TensorData& down = model.weights.at(pre->mix_down_weight);
                    const uint64_t rank = down.shape[0];
                    private_columns.insert(private_columns.end(), {expanded, rank, rank, expanded});
                    if (pre == &layer.attention_gated_residual) private_columns.push_back(hidden);
                    if (pre->inject_weight != invalid_tensor_handle)
                    {
                        private_columns.push_back(copies);
                        if (pre == &layer.attention_gated_residual) private_columns.push_back(copies);
                    }
                }
                // Raw Attention branch before expanded residual publication,
                // and the delayed token-major Expert sum before FFN post.
                private_columns.insert(private_columns.end(), {hidden, hidden});
                if (layer.attention.kind == AttentionKind::Standard) private_columns.push_back(expanded);
                if (layer.ple.enabled())
                    private_columns.insert(private_columns.end(), {layer.ple.embedding_dimension, layer.ple.embedding_dimension,
                                                                   expanded, expanded, expanded, expanded, expanded, expanded, expanded});
            }
            else
            {
                // Two Sinkhorn pre calls: normalized/projection/pre operands,
                // reduced branch, post gates and the copy-mixing matrix.
                const uint64_t projection = saturating_multiply_u64(copies, copies + 2);
                for (uint32_t pre = 0; pre < 2; ++pre)
                {
                    private_columns.insert(private_columns.end(), {expanded, projection, copies});
                    if (pre == 0)
                        private_columns.insert(private_columns.end(), {hidden, copies, saturating_multiply_u64(copies, copies)});
                }
                private_columns.insert(private_columns.end(), {hidden, hidden});
            }
            // Fused GPU router projection and compact selected ids/weights.
            if (!resident_segment)
                private_columns.insert(private_columns.end(), {layer.moe.experts.size(), Router_vulkan::selected_columns(layer.moe.top_k)});
            if (layer.moe.token_experts != invalid_tensor_handle) private_columns.push_back(layer.moe.top_k);
            for (uint64_t columns : private_columns)
            {
                auto extra = append_graph_buffer(buffers, device, saturating_multiply_u64(columns, rows), alignment,
                                                 position, resident_segment ? position : layer_finish[node.plan_index] > position ? layer_finish[node.plan_index] - 1
                                                                                                                                  : position);
                if (!extra) return extra.error();
            }
            // Only the next FFN post's normalized input and mixing operands
            // survive reset of the old command's private recording storage.
            std::vector<uint64_t> retained = resident_segment ? std::vector<uint64_t>{hidden} : std::vector<uint64_t>{hidden, copies};
            if (!resident_segment && !model.gated_residual_chain) retained.push_back(saturating_multiply_u64(copies, copies));
            for (uint64_t columns : retained)
            {
                auto extra = append_graph_buffer(buffers, device, saturating_multiply_u64(columns, rows), alignment,
                                                 position, std::max(position, layer_finish[node.plan_index]));
                if (!extra) return extra.error();
            }
        }
        auto ret = append_graph_node_scratch(model, graph, node, node.kind == LayerKind::LmHead ? logits_rows : rows,
                                             context_tokens, sequence_count, alignment, starts[node_id], ends[node_id], buffers);
        if (!ret) return ret.error();
    }
    return {};
}

Result<ModelGraphMemoryPlan> plan_model_graph_memory(const CompiledModel& compiled, uint32_t prefill_rows,
                                                     uint32_t sequence_count, uint32_t context_tokens)
{
    if (prefill_rows == 0 || sequence_count == 0)
        return Error{ErrorCode::InvalidArgument, "graph memory planning requires nonzero prefill rows and sequence count"};
    ModelGraphMemoryPlan result;
    result.prefill_rows = prefill_rows;
    result.sequence_count = sequence_count;
    result.context_tokens = std::max(prefill_rows, context_tokens);
    uint32_t host_output_rows = 0;
    for (const ExecutionGraph* graph : {&compiled.graph, &compiled.speculative.graph})
        if (std::any_of(graph->layers.begin(), graph->layers.end(), [](const Layer& node) {
                return node.kind == LayerKind::LmHead && node.backend == ExecutionBackend::Vulkan;
            }))
        {
            host_output_rows = std::max({host_output_rows, prefill_rows, sequence_count});
            if (graph == &compiled.speculative.graph)
                host_output_rows = std::max(host_output_rows, compiled.speculative.block_size);
        }
    if (host_output_rows != 0)
    {
        const auto slot_size = plan_host_output_slot_size(host_output_rows, compiled.descriptor.vocabulary_size);
        if (!slot_size) return slot_size.error();
        if (slot_size.value() > UINT64_MAX / 2)
            return Error{ErrorCode::InvalidArgument, "Vulkan host output workspace exceeds the allocation range"};
        result.host_output_slot_size = slot_size.value();
        result.host_output_workspace_size = slot_size.value() * 2;
    }
    const uint64_t total_nodes = static_cast<uint64_t>(compiled.graph.layers.size()) + compiled.speculative.graph.layers.size();
    if (total_nodes > UINT32_MAX)
        return Error{ErrorCode::InvalidModel, "graph memory schedule exceeds the node position range"};
    const auto shape_plan = [&](uint32_t rows) -> Result<GraphMemoryPlan> {
        std::vector<GraphMemoryBuffer> buffers;
        std::vector<GraphMemorySubmission> submissions;
        auto ret = append_graph_region_memory(compiled, compiled.graph, compiled.schedule, rows, rows,
                                              result.context_tokens, sequence_count, 0, buffers, submissions);
        if (!ret) return ret.error();
        // Target and draft graphs execute serially, sharing the same arena.
        // Draft verification can record one full speculative block at once.
        const uint32_t draft_rows = std::max(rows, compiled.speculative.block_size);
        ret = append_graph_region_memory(compiled, compiled.speculative.graph, compiled.speculative.schedule,
                                         draft_rows, draft_rows,
                                         result.context_tokens, sequence_count, static_cast<uint32_t>(compiled.graph.layers.size()), buffers, submissions);
        if (!ret) return ret.error();
        auto plan = plan_graph_memory(buffers, submissions);
        if (!plan) return plan.error();
        return std::move(plan).value();
    };
    for (bool decode : {false, true})
    {
        auto plan = shape_plan(decode ? sequence_count : prefill_rows);
        if (!plan) return plan.error();
        (decode ? result.decode : result.prefill) = std::move(plan).value();
    }
    if (graph_has_resident_chain(compiled, compiled.graph))
    {
        // Smaller row counts record more layers before a fence. A reservation
        // for P rows therefore needs the peak envelope of every 1..P shape,
        // not just the largest matrix. Above the recording row budget the
        // chunk is one layer; cover its largest packed and unpacked shapes.
        std::vector<uint32_t> shapes;
        const uint32_t submission_rows = compiled.resident_submission_rows;
        const uint32_t enumerated = std::min(prefill_rows, submission_rows);
        for (uint32_t rows = 1; rows <= enumerated; ++rows) shapes.push_back(rows);
        if (prefill_rows > enumerated)
        {
            shapes.push_back(prefill_rows);
            const uint32_t packed = prefill_rows - prefill_rows % 4;
            if (packed > enumerated) shapes.push_back(packed);
            if (prefill_rows % 4 == 0 && prefill_rows - 1 > enumerated) shapes.push_back(prefill_rows - 1);
        }
        for (uint32_t rows : shapes)
        {
            if (rows == prefill_rows) continue;
            auto plan = shape_plan(rows);
            if (!plan) return plan.error();
            for (const auto& device : plan.value().devices)
            {
                auto target = std::find_if(result.prefill.devices.begin(), result.prefill.devices.end(), [&](const auto& candidate) {
                    return candidate.device_index == device.device_index;
                });
                if (target == result.prefill.devices.end())
                    result.prefill.devices.push_back(device);
                else
                    target->peak_size = std::max(target->peak_size, device.peak_size);
            }
        }
    }
    std::map<uint32_t, uint64_t> states;
    std::map<uint32_t, uint64_t> growth;
    const bool resident_chain = graph_has_resident_chain(compiled, compiled.graph);
    std::vector<uint64_t> resident_growth(compiled.graph.layer_plans.size(), 0);
    for (const ExecutionGraph* graph : {&compiled.graph, &compiled.speculative.graph})
        for (const Layer& node : graph->layers)
        {
            if (node.kind != LayerKind::Attention || node.backend != ExecutionBackend::Vulkan
                || node.plan_index >= graph->layer_plans.size()) continue;
            const CompiledLayerPlan& layer = graph->layer_plans[node.plan_index];
            uint64_t bytes = attention_session_state_size(compiled, layer.attention, layer.vulkan_device_index,
                                                          result.context_tokens);
            if (layer.attention.kind == AttentionKind::Standard)
            {
                // Doubling a ring retains its previous half-capacity buffer
                // until publication after the fence. This is allocation
                // overlap, rather than spare heap headroom.
                const uint64_t previous = result.context_tokens > 16
                                              ? attention_session_state_size(compiled, layer.attention, layer.vulkan_device_index,
                                                                             std::max(1u, result.context_tokens / 2))
                                              : 0;
                if (resident_chain && graph == &compiled.graph)
                    resident_growth[node.plan_index] = previous;
                else
                    growth[layer.vulkan_device_index] = std::max(growth[layer.vulkan_device_index], previous);
            }
            else if (layer.attention.kind == AttentionKind::GatedDeltaNet)
            {
                // A verification transaction stores live + initial + one
                // snapshot for each nonfinal recorded row across all layers.
                const uint64_t copies = compiled.speculative.enabled() ? std::max<uint64_t>(2, static_cast<uint64_t>(compiled.speculative.block_size) + 1) : 1;
                bytes = saturating_multiply_u64(bytes / 2, copies);
            }
            else if (layer.attention.kind == AttentionKind::MultiHeadLatent)
            {
                bytes /= 2;
                const uint64_t previous = attention_session_state_size(compiled, layer.attention, layer.vulkan_device_index,
                                                                       std::max(1u, result.context_tokens / 2))
                                          / 2;
                growth[layer.vulkan_device_index] = std::max(growth[layer.vulkan_device_index], previous);
            }
            uint64_t& total = states[layer.vulkan_device_index];
            total = saturating_add_u64(total, bytes);
            if (total == UINT64_MAX)
                return Error{ErrorCode::InvalidArgument, "Vulkan graph Session state exceeds the allocation range"};
        }
    if (resident_chain)
    {
        // Every growing KV ring in a recorded chunk retains its old backing
        // until the common fence. Cover each row-dependent chunk partition;
        // one-layer-at-a-time growth is insufficient for a resident decode.
        std::vector<uint32_t> chunks{1};
        const uint32_t submission_rows = compiled.resident_submission_rows;
        for (uint32_t rows = 1; rows <= submission_rows; ++rows)
        {
            const uint32_t chunk = resident_chunk_layers(compiled, rows);
            if (std::find(chunks.begin(), chunks.end(), chunk) == chunks.end()) chunks.push_back(chunk);
        }
        for (uint32_t chunk : chunks)
            for (size_t first = 0; first < resident_growth.size(); first += chunk)
            {
                uint64_t previous = 0;
                for (size_t index = first; index < std::min(resident_growth.size(), first + chunk); ++index)
                    previous = saturating_add_u64(previous, resident_growth[index]);
                const uint32_t device = compiled.graph.layer_plans[first].vulkan_device_index;
                growth[device] = std::max(growth[device], previous);
            }
    }
    for (const auto& [device, bytes] : states)
    {
        const uint64_t peak = saturating_add_u64(bytes, growth[device]);
        if (peak == UINT64_MAX)
            return Error{ErrorCode::InvalidArgument, "Vulkan graph Session state exceeds the allocation range"};
        result.session_state.push_back({device, peak});
    }
    return result;
}

} // namespace moe
} // namespace ncnn
