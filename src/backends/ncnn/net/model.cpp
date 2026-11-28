#include "backends/ncnn/net/model.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/expert/backend.h"
#include "backends/ncnn/layer/attention_vulkan.h"
#include "backends/ncnn/layer/gateddeltanet_vulkan.h"
#include "backends/ncnn/layer/hyperconnection_vulkan.h"
#include "backends/ncnn/layer/latentattention_vulkan.h"
#include "backends/ncnn/layer/qsaattention_vulkan.h"
#include "backends/ncnn/layer/rmsnorm_vulkan.h"
#include "backends/ncnn/layer/router_vulkan.h"
#include "backends/ncnn/linear/linear.h"
#include "backends/ncnn/net/latent_layer.h"
#include "backends/ncnn/net/layer_head.h"
#include "backends/ncnn/net/residual_layer.h"
#include "engine/expert.h"
#include "graph/compiledmodel.h"
#include "graph/compiler.h"
#include "graph/resourceplan.h"
#include "kernels/bfloat16.h"
#include "kernels/qnk.h"
#include "storage/weightstore.h"

#include <gpu.h>

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
    if (moe.experts.empty())
        return false;
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

bool support_vulkan_attention(const CompiledOperatorTable& operators,
                              const AttentionBlockPlan& attention) noexcept
{
    return attention_vulkan_supported(operators, attention);
}

static uint64_t release_storage(TensorData& tensor)
{
    uint64_t released_size = 0;
    if (tensor.dtype == DType::Float32)
    {
        released_size = tensor.float32_data.size() * sizeof(float);
        std::vector<float>().swap(tensor.float32_data);
    }
    else if (tensor.dtype == DType::BFloat16)
    {
        released_size = tensor.bfloat16_data.size() * sizeof(uint16_t)
                        + tensor.quantized_data.size();
        std::vector<uint16_t>().swap(tensor.bfloat16_data);
        std::vector<uint8_t>().swap(tensor.quantized_data);
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

static void release_dense(CompiledModel& compiled, TensorHandle handle)
{
    if (handle == invalid_tensor_handle)
        return;
    const uint64_t released = release_storage(compiled.weights.at_mutable(handle));
    if (released != 0)
    {
        compiled.released_dense_host_size = saturating_add_u64(compiled.released_dense_host_size,
                                                               released);
        CompiledOperator& op = compiled.operators.at_weight_mutable(handle);
        op.cpu.residency = WeightResidencyPolicy::DeviceOnly;
        op.cpu.release_host_packs();
    }
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
    for (const CompiledLayerPlan& layer : compiled.net.layer_plans)
        retain_shared(layer);
    for (const CompiledLayerPlan& layer : compiled.speculative.net.layer_plans)
        retain_shared(layer);
    for (TensorHandle handle : shared_fallback_handles)
    {
        if (handle != invalid_tensor_handle)
            compiled.operators.at_weight_mutable(handle).cpu.residency = WeightResidencyPolicy::HybridDual;
    }
    for (TensorHandle handle = 0; handle < compiled.weights.size(); ++handle)
    {
        if (uses_vulkan_dense_operator(compiled.operators.at_weight(handle))
            && std::find(shared_fallback_handles.begin(), shared_fallback_handles.end(), handle) == shared_fallback_handles.end())
        {
            release_dense(compiled, handle);
        }
    }

    const auto release_fused_layer_handles = [&compiled](CompiledLayerPlan& layer) {
        AttentionBlockPlan& attention = layer.attention;
        const bool qkv_fused = attention.fused_qkv_operator != invalid_compiled_operator_handle
                               || attention.fused_qkv_gate_bfloat16_operator != invalid_compiled_operator_handle;
        if (qkv_fused)
        {
            release_dense(compiled, attention.query_weight);
            release_dense(compiled, attention.key_weight);
            release_dense(compiled, attention.value_weight);
        }
        if (attention.fused_qkv_gate_bfloat16_operator != invalid_compiled_operator_handle)
            release_dense(compiled, attention.output_gate_weight);

        if (attention.fused_delta_input_operator != invalid_compiled_operator_handle
            && attention.gated_delta_vulkan_operator != invalid_compiled_operator_handle
            && compiled.operators.at(attention.gated_delta_vulkan_operator).gated_delta
            && uses_vulkan_dense_operator(compiled.operators.at(attention.fused_delta_input_operator)))
        {
            release_dense(compiled, attention.delta_qkv_weight);
            release_dense(compiled, attention.delta_z_weight);
            release_dense(compiled, attention.delta_beta_weight);
            release_dense(compiled, attention.delta_alpha_weight);
        }

        if (attention.vulkan_attention_operator != invalid_compiled_operator_handle)
        {
            release_dense(compiled, attention.pre_attention_norm_weight);
            release_dense(compiled, attention.sinks);
            if (has_flag(attention.flags, AttentionBlockPostNorm))
                release_dense(compiled, attention.attn_post_norm);
            if (has_flag(attention.flags, AttentionBlockQueryKeyNorm))
            {
                release_dense(compiled, attention.query_norm_weight);
                release_dense(compiled, attention.key_norm_weight);
            }
        }

        if (layer.moe.fused_shared_input_bfloat16_operator != invalid_compiled_operator_handle)
        {
            release_dense(compiled, layer.moe.shared_expert.gate_weight);
            release_dense(compiled, layer.moe.shared_expert.up_weight);
            release_dense(compiled, layer.moe.shared_expert_gate_weight);
        }
        const ExpertPlan& shared = layer.moe.shared_expert;
        const auto& shared_chain = compiled.operators.at(shared.vulkan_operator).qnk_expert;
        if (layer.moe.has_shared_expert && shared_chain && shared_chain->has_router_gate())
            release_dense(compiled, layer.moe.shared_expert_gate_weight);
        if (layer.moe.has_shared_expert && shared.vulkan_operator != invalid_compiled_operator_handle
            && compiled.operators.at(shared.vulkan_operator).qnk_expert
            && compiled.weights.at(shared.gate_weight).dtype == DType::Q8_0
            && compiled.weights.at(shared.up_weight).dtype == DType::Q8_0
            && compiled.weights.at(shared.down_weight).dtype == DType::Q8_0)
        {
            release_dense(compiled, shared.gate_weight);
            release_dense(compiled, shared.up_weight);
            release_dense(compiled, shared.down_weight);
        }
        const ExpertPlan& dense = layer.dense_ffn;
        if (dense.down_weight != invalid_tensor_handle && !dense.use_gpu_cache
            && support_vulkan_ffn(compiled.operators, dense))
        {
            release_dense(compiled, dense.gate_weight);
            release_dense(compiled, dense.up_weight);
            release_dense(compiled, dense.down_weight);
        }
    };

    for (CompiledLayerPlan& layer : compiled.net.layer_plans)
        release_fused_layer_handles(layer);
    for (CompiledLayerPlan& layer : compiled.speculative.net.layer_plans)
        release_fused_layer_handles(layer);

    // The vocabulary matrix is the largest dense tensor; release it once the
    // device head owns the weights. A tied embedding keeps the shared mapping
    if (compiled.lm_head_weight != invalid_tensor_handle
        && compiled.lm_head_weight != compiled.token_embedding
        && std::any_of(compiled.net.layers.begin(), compiled.net.layers.end(), [](const Layer& node) {
               return node.kind == LayerKind::LmHead && node.backend == ExecutionBackend::Vulkan;
           }))
    {
        const CompiledOperator& head = compiled.operators.at_weight(compiled.lm_head_weight);
        if (head.qnk || head.bfloat16 || head.float8 || head.linear)
            release_dense(compiled, compiled.lm_head_weight);
    }

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
            if (!prepared)
                return prepared.error();
        }
        const CompiledOperatorHandle handle = operators.allocate();
        operators.at_mutable(handle).float8 = operators.at_weight(ffn.gate_weight).float8;
        ffn.vulkan_operator = handle;
        return {};
    }
    if (gate.dtype == DType::BFloat16 && up.dtype == DType::BFloat16
        && down.dtype == DType::BFloat16
        && (ffn.activation == ExpertActivation::Silu
            || ffn.activation == ExpertActivation::DeepSeekSwiGlu
            || ffn.activation == ExpertActivation::GeluTanh)
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
        && qkv_matrices.front()->dtype == DType::BFloat16
        && plan.output_gate_weight != invalid_tensor_handle
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
        && qkv_matrices.front()->dtype == DType::BFloat16)
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
                                       compiled.descriptor.norm_weight_offset, ffn_hyper->context(), ffn_hyper->option());
    const TensorData* bias = layer.moe.router_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(layer.moe.router_bias);
    const TensorData* selection_bias = layer.moe.router_selection_bias == invalid_tensor_handle ? nullptr : &compiled.weights.at(layer.moe.router_selection_bias);
    auto router = Router_vulkan::create(compiled.weights.at(layer.moe.router_weight), bias, selection_bias,
                                        layer.vulkan_device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags);
    op.latent_layer = LatentLayer_vulkan::create(op.latent_attention, std::move(attention_hyper), std::move(ffn_hyper),
                                                 std::move(norm), std::move(router));
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
        // The cooperative-matrix prefill path cannot fold the norm into its tile
        // loads, so it needs the normalization as its own pass. Keep the fused
        // chain above as the fallback.
        (void)fused_input_operator->prepare_input_rms_norm(compiled.weights.at(plan.pre_attention_norm_weight),
                                                           plan.norm_epsilon, plan.norm_weight_offset);
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
    const auto make_qnk = [&](const TensorData& matrix, const TensorData* bias) {
        return QnkLinear_vulkan::create(matrix, bias, layer_plan.vulkan_device_index,
                                        compiled.vulkan_runtime, compiled.opt.optimization_flags);
    };
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
            projection.qnk = make_qnk(compiled.weights.at(handle), bias);
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

    // Decode override: rebuild the projections from the retained block-
    // quantized payload. One-row decode then streams half the bytes while the
    // BF16 set above keeps serving multi-row prefill.
    std::array<CompiledOperator, 5> projections_q8;
    std::shared_ptr<QnkLinear_vulkan> fused_qkv_q8;
    std::vector<TensorData> shadows;
    bool ready = false;
    if (has_flag(compiled.opt.optimization_flags, OptimizationVulkanQnK))
    {
        shadows.reserve(5);
        bool complete = true;
        for (size_t index = 0; index < std::size(weights) && complete; ++index)
        {
            const TensorHandle handle = weights[index];
            if (handle == invalid_tensor_handle)
            {
                complete = index == 4;
                continue;
            }
            complete = is_qnk_dtype(compiled.weights.at(handle).qnk_dtype);
            if (complete)
                shadows.push_back(qnk_view(compiled.weights.at(handle)));
        }
        if (complete)
        {
            const std::array<const TensorData*, 3> qkv = {&shadows[0], &shadows[1], &shadows[2]};
            const std::array<const TensorData*, 3> qkv_bias = {
                biases[0] == invalid_tensor_handle ? nullptr : &compiled.weights.at(biases[0]),
                biases[1] == invalid_tensor_handle ? nullptr : &compiled.weights.at(biases[1]),
                biases[2] == invalid_tensor_handle ? nullptr : &compiled.weights.at(biases[2])};
            if (QnkLinear_vulkan::can_fuse(qkv))
                fused_qkv_q8 = QnkLinear_vulkan::create_fused(qkv, qkv_bias, layer_plan.vulkan_device_index,
                                                              compiled.vulkan_runtime,
                                                              compiled.opt.optimization_flags);
            if (!fused_qkv_q8)
            {
                // A biased or non-fusable QKV layout keeps one Q8 operator per
                // matrix; record_native_qkv already handles the unfused shape.
                for (size_t index = 0; index < 3; ++index)
                    projections_q8[index].qnk = make_qnk(shadows[index], qkv_bias[index]);
            }
            projections_q8[3].qnk = make_qnk(shadows[3], biases[3] == invalid_tensor_handle ? nullptr : &compiled.weights.at(biases[3]));
            if (weights[4] != invalid_tensor_handle)
                projections_q8[4].qnk = make_qnk(shadows[4], nullptr);
            const bool qkv_ready = fused_qkv_q8
                                   || (projections_q8[0].qnk && projections_q8[1].qnk && projections_q8[2].qnk);
            ready = qkv_ready && projections_q8[3].qnk != nullptr
                    && (weights[4] == invalid_tensor_handle || projections_q8[4].qnk != nullptr);
            if (!ready)
            {
                projections_q8 = {};
                fused_qkv_q8.reset();
            }
        }
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
    const TensorData* post_attention_norm = plan.attn_post_norm == invalid_tensor_handle
                                                ? nullptr
                                                : &compiled.weights.at(plan.attn_post_norm);
    auto attention_operator = Attention_vulkan::create_native(compiled.weights.at(plan.pre_attention_norm_weight),
                                                              query_norm,
                                                              key_norm,
                                                              sinks,
                                                              projections,
                                                              attn_vk_config(compiled, plan, layer_plan.vulkan_device_index), fused_qkv, norm_bias,
                                                              post_attention_norm,
                                                              ready ? &projections_q8 : nullptr,
                                                              std::move(fused_qkv_q8));
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
        const AttentionConfig_vulkan attention_config = attn_vk_config(compiled, plan, layer_plan.vulkan_device_index);
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

static Result<void> prepare_gated_residual_pipeline(CompiledModel& compiled)
{
    for (TensorHandle handle = 0; handle < compiled.weights.size(); ++handle)
        if (compiled.weights.at(handle).dtype == DType::Float8E4M3)
            return Error{ErrorCode::UnsupportedModel, "continuous gated residual Vulkan execution does not support FP8 model weights"};
    for (const auto& layer : compiled.net.layer_plans)
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
    for (CompiledLayerPlan& layer : compiled.net.layer_plans)
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
    return (ffn.activation == ExpertActivation::Silu || ffn.activation == ExpertActivation::DeepSeekSwiGlu
            || ffn.activation == ExpertActivation::GeluTanh)
           && up.shape.size() == 2 && up.shape[0] % 128 == 0
           && bf16_fused_storage_size(model, {ffn.gate_weight, ffn.up_weight}, alignment, layer.vulkan_device_index) != 0
           && bf16_projection_supported(model, down, alignment, layer.vulkan_device_index);
}

Result<void> prepare_model_pipeline(CompiledModel& compiled, const CompilerOption& opt)
{
    compiled.operators.bind_weight_count(compiled.weights.size());
    if (compiled.opt.hybrid_mode == HybridMode::CpuOnly)
        return {};
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
        for (const CompiledLayerPlan& layer : compiled.net.layer_plans)
        {
            auto attention_result = attention_resource_request(compiled, layer);
            if (!attention_result)
                return attention_result.error();
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
        const auto previous_layers = compiled.net.layer_plans;
        auto prepared = prepare_gated_residual_pipeline(compiled);
        if (!prepared || !resources.allocation_fits(device, request))
        {
            compiled.gated_residual_device_head.reset();
            compiled.gated_residual_chain = false;
            compiled.operators = previous_operators;
            compiled.net.layer_plans = previous_layers;
            if (!prepared)
                return prepared.error();
            return Error{ErrorCode::InvalidArgument, "gated residual Vulkan persistent weights and minimum working set exceed the GPU heap budget on device " + std::to_string(device)};
        }
        resources.commit(device, request);
        return {};
    }

    // Complete attention groups have priority over independently useful
    // projections. Main and speculative graphs share the same device budget.
    for (Net* graph : {&compiled.net, &compiled.speculative.net})
    {
        for (CompiledLayerPlan& layer : graph->layer_plans)
        {
            const auto kind = layer.attention.kind;
            const bool attention_enabled = has_flag(opt.flags, BackendVulkanAttention)
                                           && has_flag(compiled.opt.optimization_flags, OptimizationVulkanAttention);
            if (kind == AttentionKind::None || !attention_enabled)
                continue;
            if (kind == AttentionKind::Standard
                && (has_flag(layer.attention.flags, AttentionBlockQsa) || has_flag(layer.attention.flags, AttentionBlockExternalResidual)))
                continue;
            auto request_result = attention_resource_request(compiled, layer);
            if (!request_result)
                return request_result.error();
            const GpuResourceRequest request = std::move(request_result).value();
            auto prepared = prepare_resource_group(compiled, resources, layer.vulkan_device_index, request, &layer, "attention at layer " + std::to_string(layer.layer_id), [&]() -> Result<void> {
                    if (kind == AttentionKind::Standard) return prepare_attn_ops(compiled, layer);
                    if (kind == AttentionKind::GatedDeltaNet) { prepare_gdn_ops(compiled, layer); return {}; }
                    auto prepared = prepare_latent_ops(compiled, layer, "", attention_enabled);
                    if (prepared) prepare_latent_layer_ops(compiled, layer);
                    return prepared; }, [&] { return support_vulkan_attention(compiled.operators, layer.attention); });
            if (!prepared)
                return prepared.error();
        }
    }
    for (Net* graph : {&compiled.net, &compiled.speculative.net})
    {
        for (CompiledLayerPlan& layer : graph->layer_plans)
        {
            if (layer.moe.has_shared_expert)
            {
                auto prepared = prepare_resource_group(compiled, resources, layer.vulkan_device_index, ffn_resource_request(compiled, layer, true), &layer, "Shared Expert at layer " + std::to_string(layer.layer_id), [&] { return prepare_shared_ops(compiled.weights, compiled.operators, layer.moe,
                                                                                                                                                                                                                                                        layer.vulkan_device_index, compiled.vulkan_runtime, compiled.opt.optimization_flags); }, [&] { return support_vulkan_shared_experts(compiled.operators, layer.moe); });
                if (!prepared)
                    return prepared.error();
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
    if (!head_prepared)
        return head_prepared.error();

    // Automatic dense placement keeps frequently reused FFNs resident after
    // attention and the output head. Explicit cache capacity selects the
    // existing demand-cache policy, including its bounded-memory behavior.
    for (Net* graph : {&compiled.net, &compiled.speculative.net})
    {
        for (CompiledLayerPlan& layer : graph->layer_plans)
        {
            ExpertPlan& ffn = layer.dense_ffn;
            if (ffn.down_weight == invalid_tensor_handle
                || (ffn.use_gpu_cache && resources.explicit_cache_reserve(layer.vulkan_device_index))
                || !ffn_kernel_ready(compiled, layer))
                continue;
            const bool cached = ffn.use_gpu_cache;
            const GpuResourceRequest request = ffn_resource_request(compiled, layer, false);
            auto prepared = prepare_resource_group(compiled, resources, layer.vulkan_device_index, request, &layer, "dense FFN at layer " + std::to_string(layer.layer_id), [&] { return prepare_ffn_operators(compiled.weights, compiled.operators, layer.dense_ffn,
                                                                                                                                                                                                               layer.vulkan_device_index, compiled.vulkan_runtime,
                                                                                                                                                                                                               compiled.opt.optimization_flags, false); }, [&] { return support_vulkan_ffn(compiled.operators, layer.dense_ffn); }, cached);
            if (!prepared)
                return prepared.error();
            if (prepared.value() && cached)
            {
                layer.dense_ffn.use_gpu_cache = false;
                resources.refresh_expert_reserve(layer.vulkan_device_index);
            }
        }
    }

    compiled.gpu_segments.resize(compiled.net.layer_plans.size());
    if (has_flag(opt.flags, BackendVulkanExperts)
        && !has_flag(compiled.opt.flags, OptionDisableGpuExpertExecution))
    {
        const bool dense_chain = !compiled.net.layer_plans.empty() && GpuSegment_vulkan::supports_head(compiled)
                                 && std::all_of(compiled.net.layer_plans.begin(), compiled.net.layer_plans.end(), [&](const auto& layer) {
                                        // The resident dense chain models pure dense
                                        // models; parallel FFN+MoE layers keep the
                                        // per-node hybrid path.
                                        return layer.moe.experts.empty()
                                               && layer.dense_ffn.down_weight != invalid_tensor_handle
                                               && layer.vulkan_device_index == device && GpuSegment_vulkan::supports(compiled, layer);
                                    });
        if (dense_chain)
        {
            GpuResourceRequest request;
            request.persistent_size = constant_storage(compiled, {compiled.final_norm_weight, compiled.final_norm_bias}, alignment);
            for (const auto& layer : compiled.net.layer_plans)
                request.persistent_size = saturating_add_u64(request.persistent_size,
                                                             saturating_add_u64(constant_storage(compiled, {layer.dense_pre_ffn_norm_weight, layer.dense_pre_ffn_norm_bias}, alignment), alignment * 8));
            request.workspace_size = saturating_multiply_u64(default_prefill_rows,
                                                             saturating_multiply_u64(compiled.descriptor.hidden_size, 4 * sizeof(float)));
            if (resources.can_admit(device, request))
            {
                for (size_t index = 0; index < compiled.net.layer_plans.size(); ++index)
                {
                    compiled.gpu_segments[index] = GpuSegment_vulkan::create(compiled, compiled.net.layer_plans[index]);
                    if (!compiled.gpu_segments[index])
                        return Error{ErrorCode::InternalError, "cannot prepare selected resident dense segment at layer " + std::to_string(index)};
                }
                compiled.gpu_head = GpuSegment_vulkan::create_head(compiled);
                if (!compiled.gpu_head)
                    return Error{ErrorCode::InternalError, "cannot prepare selected resident dense head"};
                if (!resources.allocation_fits(device, request))
                    return Error{ErrorCode::InvalidArgument, "resident dense graph exceeds the live device budget"};
                resources.commit(device, request);
                const uint32_t max_layers = static_cast<uint32_t>(std::max<size_t>(1, compiled.net.layer_plans.size()));
                compiled.resident_submission_rows = default_prefill_rows * std::min<uint32_t>(32u, max_layers);
            }
        }
        // A mixed graph may contain a dense FFN beside routed layers. The
        // dense layer still needs a segment so the routed chain can continue
        // through it (and so a resident Expert bank chain can be selected).
        for (size_t index = 0; index < compiled.net.layer_plans.size(); ++index)
        {
            CompiledLayerPlan& layer = compiled.net.layer_plans[index];
            if (layer.dense_ffn.down_weight == invalid_tensor_handle || compiled.gpu_segments[index])
                continue;
            // Parallel dense+MoE layers run as independent per-node segments and
            // must not take the dense-only segment path.
            if (!layer.moe.experts.empty())
                continue;
            if (!GpuSegment_vulkan::supports(compiled, layer))
                continue;
            const uint32_t dense_device = layer.vulkan_device_index;
            const uint64_t dense_alignment = gpu_buffer_alignment(compiled, dense_device);
            GpuResourceRequest request;
            request.persistent_size = constant_storage(compiled, {layer.dense_pre_ffn_norm_weight, layer.dense_pre_ffn_norm_bias}, dense_alignment);
            request.persistent_size = saturating_add_u64(request.persistent_size, saturating_multiply_u64(dense_alignment, 8));
            request.workspace_size = saturating_multiply_u64(default_prefill_rows,
                                                             saturating_multiply_u64(compiled.descriptor.hidden_size, 4 * sizeof(float)));
            std::shared_ptr<GpuSegment_vulkan> segment;
            auto prepared = prepare_resource_group(compiled, resources, dense_device, request, &layer, "dense segment at layer " + std::to_string(layer.layer_id), [&]() -> Result<void> {
                                                       segment = GpuSegment_vulkan::create(compiled, layer);
                                                       if (!segment)
                                                           return Error{ErrorCode::InternalError, "cannot prepare selected dense Vulkan segment at layer " + std::to_string(layer.layer_id)};
                                                       return {}; }, [&] { return GpuSegment_vulkan::supports(compiled, layer); }, true);
            if (!prepared)
                return prepared.error();
            if (prepared.value())
                compiled.gpu_segments[index] = std::move(segment);
        }

        for (size_t index = 0; index < compiled.net.layer_plans.size(); ++index)
        {
            const auto& layer = compiled.net.layer_plans[index];
            if (layer.dense_ffn.down_weight != invalid_tensor_handle)
                continue;
            if (!GpuSegment_vulkan::supports(compiled, layer)
                || !support_vulkan_experts(compiled.weights, layer.moe, compiled.opt.optimization_flags))
                continue;
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
            request.workspace_size = saturating_multiply_u64(saturating_multiply_u64(default_prefill_rows, columns), sizeof(float));
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

} // namespace moe
} // namespace ncnn
