#include "modelpipeline.h"

#include "kernels/qnk.h"
#include "linear.h"
#include "attention.h"
#include "graph/compiledmodel.h"
#include "storage/weightstore.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

Result<void> prepare_linear_operator(
    WeightStore& weights,
    CompiledOperatorTable& operators,
    TensorHandle matrix_handle,
    TensorHandle bias_handle,
    NcnnLinearDevice device,
    bool retain_cpu_dense_copy,
    uint32_t vulkan_device_index,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags,
    uint32_t input_group_count,
    bool prefer_bfloat16_vulkan)
{
    TensorData& matrix = weights.at_mutable(matrix_handle);
    CompiledOperator& compiled_operator = operators.at_weight_mutable(matrix_handle);
    const TensorData* bias = bias_handle == invalid_tensor_handle ? nullptr : &weights.at(bias_handle);
    if (matrix.dtype == DType::Float8E4M3)
    {
        if (device == NcnnLinearDevice::Vulkan)
        {
            compiled_operator.float8 = NcnnVulkanFloat8Operator::create(
                matrix,
                bias,
                input_group_count,
                vulkan_device_index,
                context_instance,
                optimization_flags);
            if (!compiled_operator.float8)
                return Error{ErrorCode::InternalError, "failed to create Vulkan FP8 Linear operator"};
        }
        return {};
    }
    if (is_qnk_dtype(matrix.dtype))
    {
        if (device == NcnnLinearDevice::Vulkan
            && has_flag(
                optimization_flags,
                OptimizationVulkanQnK))
        {
            compiled_operator.qnk = NcnnVulkanQnkOperator::create(
                matrix,
                bias,
                vulkan_device_index,
                context_instance,
                optimization_flags);
            if (!compiled_operator.qnk)
                return Error{ErrorCode::InternalError, "failed to create Vulkan Qn_K Linear operator"};
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
        compiled_operator.bfloat16 = NcnnVulkanBfloat16Operator::create(
            matrix,
            bias,
            vulkan_device_index,
            context_instance,
            optimization_flags);
        if (compiled_operator.bfloat16)
            return {};
    }
    compiled_operator.linear = NcnnLinearOperator::create(
        matrix,
        bias,
        device,
        vulkan_device_index,
        context_instance,
        optimization_flags);
    if (device == NcnnLinearDevice::Vulkan && !compiled_operator.linear)
        return Error{ErrorCode::InternalError, "failed to create Vulkan InnerProduct operator"};
    return {};
}

Result<void> prepare_lm_head_operator(
    CompiledModel& compiled,
    NcnnLinearDevice dense_device,
    bool retain_cpu_dense_copies)
{
    auto prepared = prepare_linear_operator(compiled.weights, compiled.operators, compiled.lm_head_weight, invalid_tensor_handle, dense_device, retain_cpu_dense_copies,
                                            compiled.opt.vulkan_device_index, compiled.vulkan_context_instance,
                                            compiled.opt.optimization_flags);
    if (!prepared)
        return prepared.error();
    const CompiledOperator& lm_head_operator = compiled.operators.at_weight(compiled.lm_head_weight);
    if (lm_head_operator.bfloat16
        && compiled.final_norm_weight != invalid_tensor_handle)
    {
        (void)lm_head_operator.bfloat16->prepare_rms_norm(
            compiled.weights.at(compiled.final_norm_weight),
            compiled.descriptor.norm_epsilon,
            compiled.descriptor.norm_weight_offset);
    }
    return {};
}

static bool expert_activation_supported_by_vulkan(ExpertActivation activation) noexcept
{
    return activation == ExpertActivation::Silu
           || activation == ExpertActivation::GptOssSwiGlu
           || activation == ExpertActivation::DeepSeekSwiGlu;
}

bool support_vulkan_experts(
    const WeightStore& weights,
    const MoeBlockPlan& moe,
    uint64_t optimization_flags) noexcept
{
    if (moe.experts.empty())
        return false;

    DType qnk_dtype = DType::Float32;
    bool has_qnk_dtype = false;
    for (const ExpertPlan& expert : moe.experts)
    {
        if (expert.gate_up_weight == invalid_tensor_handle
            || expert.down_weight == invalid_tensor_handle
            || !expert_activation_supported_by_vulkan(expert.activation))
        {
            return false;
        }

        const TensorData& gate_up = weights.at(expert.gate_up_weight);
        const TensorData& down = weights.at(expert.down_weight);
        if (gate_up.shape.size() != 2
            || down.shape.size() != 2
            || gate_up.shape[0] == 0
            || gate_up.shape[0] % 2 != 0
            || down.shape[0] == 0
            || down.shape[1] != gate_up.shape[0] / 2)
        {
            return false;
        }

        if (gate_up.dtype == DType::MxFp4 && down.dtype == DType::MxFp4)
        {
            if (gate_up.shape[1] == 0
                || down.shape[1] == 0
                || gate_up.shape[1] % 32 != 0
                || down.shape[1] % 32 != 0)
            {
                return false;
            }
            continue;
        }

        if (gate_up.dtype == DType::BFloat16 && down.dtype == DType::BFloat16)
        {
            if (expert.activation != ExpertActivation::Silu
                || gate_up.shape[1] == 0
                || down.shape[1] == 0
                || gate_up.shape[0] / 2 % 128 != 0
                || gate_up.bfloat16_values().size() != gate_up.element_count()
                || down.bfloat16_values().size() != down.element_count())
            {
                return false;
            }
            continue;
        }

        if (!has_flag(optimization_flags, OptimizationVulkanQnK)
            || !is_qnk_dtype(gate_up.dtype)
            || gate_up.dtype != down.dtype
            || !qnk_shape_supported(gate_up.dtype, gate_up.shape[0], gate_up.shape[1])
            || !qnk_shape_supported(down.dtype, down.shape[0], down.shape[1]))
        {
            return false;
        }

        if (!has_qnk_dtype)
        {
            qnk_dtype = gate_up.dtype;
            has_qnk_dtype = true;
        }
        else if (qnk_dtype != gate_up.dtype)
        {
            return false;
        }
    }

    return true;
}

bool support_vulkan_shared_experts(
    const CompiledOperatorTable& operators,
    const MoeBlockPlan& moe) noexcept
{
    return moe.fused_shared_input_bfloat16_operator != invalid_compiled_operator_handle
           && static_cast<bool>(operators.at(moe.fused_shared_input_bfloat16_operator).bfloat16);
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

uint64_t gated_delta_vulkan_working_set_size(
    const AttentionDescriptor& attention,
    const MoeModelDescriptor& descriptor) noexcept
{
    // The fused Vulkan implementation currently consumes BF16 projection
    // matrices and expands the small recurrent constants to FP32.  Returning
    // zero for other dtypes keeps the admission policy capability-driven
    // instead of selecting a path that cannot create the fused operator.
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

    const uint64_t key_size = saturating_multiply_u64(
        attention.kv_head_count, attention.head_dimension);
    const uint64_t value_size = saturating_multiply_u64(
        attention.head_count, attention.value_head_dimension);
    const uint64_t convolution_size = saturating_add_u64(
        saturating_multiply_u64(key_size, 2), value_size);
    const uint64_t fused_columns = saturating_add_u64(
        saturating_add_u64(convolution_size, value_size),
        saturating_multiply_u64(attention.head_count, 2));

    // NcnnVulkanBfloat16Operator stores BF16 matrix weights and allocates
    // FP32 bias/input-output metadata for each projection.  The GDN operator
    // additionally uploads convolution and per-head constants as FP32.
    const uint64_t projection_elements = saturating_multiply_u64(
        saturating_add_u64(fused_columns, value_size), descriptor.hidden_size);
    const uint64_t projection_size = saturating_multiply_u64(
        projection_elements, sizeof(uint16_t));
    const uint64_t projection_metadata_size = saturating_multiply_u64(
        saturating_add_u64(
            saturating_add_u64(
                saturating_multiply_u64(descriptor.hidden_size, 2),
                fused_columns),
            value_size),
        sizeof(float));
    const uint64_t recurrent_constant_elements = saturating_add_u64(
        saturating_multiply_u64(convolution_size, attention.convolution_kernel_size),
        saturating_add_u64(
            saturating_multiply_u64(attention.head_count, 2),
            attention.value_head_dimension));
    const uint64_t recurrent_constant_size = saturating_multiply_u64(
        recurrent_constant_elements, sizeof(float));
    const uint64_t allocated_size = saturating_add_u64(
        saturating_add_u64(projection_size, projection_metadata_size),
        recurrent_constant_size);

    // Weight allocators and the first dispatch need transient workspace in
    // addition to the raw payload.  A two-times safety margin is deliberately
    // conservative and keeps this policy independent of a model's layer id or
    // tensor naming convention.
    return saturating_multiply_u64(allocated_size, 2);
}

bool uses_vulkan_dense_operator(const CompiledOperator& executable) noexcept
{
    return executable.bfloat16
           || executable.float8
           || (executable.linear && executable.linear->uses_vulkan());
}

bool support_vulkan_attention(
    const CompiledOperatorTable& operators,
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

    if (attention.vulkan_attention_operator != invalid_compiled_operator_handle
        && operators.at(attention.vulkan_attention_operator).attention)
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
            && uses_vulkan_dense_operator(operators.at_weight(handle)))
        {
            return true;
        }
    }
    return false;
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
    tensor.mapped_size = 0;
}

static void release_vulkan_dense_handle(CompiledModel& compiled, TensorHandle handle)
{
    if (handle == invalid_tensor_handle)
        return;
    release_tensor_host_storage(compiled.weights.at_mutable(handle));
}

void release_vulkan_dense_host_copies(CompiledModel& compiled)
{
    for (TensorHandle handle = 0; handle < compiled.weights.size(); ++handle)
    {
        TensorData& tensor = compiled.weights.at_mutable(handle);
        if (uses_vulkan_dense_operator(compiled.operators.at_weight(handle)))
            release_tensor_host_storage(tensor);
    }

    const auto release_fused_layer_handles = [&compiled](CompiledLayerPlan& layer) {
        AttentionBlockPlan& attention = layer.attention;
        const bool qkv_fused = attention.fused_qkv_operator != invalid_compiled_operator_handle
                               || attention.fused_qkv_bfloat16_operator != invalid_compiled_operator_handle
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
            || attention.fused_delta_input_bfloat16_operator != invalid_compiled_operator_handle)
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
    };

    for (CompiledLayerPlan& layer : compiled.graph.layer_plans)
        release_fused_layer_handles(layer);
    for (CompiledLayerPlan& layer : compiled.speculative.graph.layer_plans)
        release_fused_layer_handles(layer);
}

Result<void> prepare_shared_expert_operators(
    WeightStore& weights,
    CompiledOperatorTable& operators,
    MoeBlockPlan& moe,
    NcnnLinearDevice device,
    bool retain_cpu_dense_copy,
    uint32_t vulkan_device_index,
    const NcnnVulkanContextInstancePtr& context_instance,
    uint64_t optimization_flags)
{
    ExpertPlan& shared = moe.shared_expert;
    const TensorData& gate = weights.at(shared.gate_weight);
    const TensorData& up = weights.at(shared.up_weight);
    const bool has_router_gate = moe.shared_expert_gate_weight != invalid_tensor_handle;
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
        const CompiledOperatorHandle fused_handle = operators.allocate();
        operators.at_mutable(fused_handle).bfloat16 = NcnnVulkanBfloat16Operator::create_fused(
            matrices,
            biases,
            vulkan_device_index,
            context_instance,
            optimization_flags);
        if (operators.at(fused_handle).bfloat16)
            moe.fused_shared_input_bfloat16_operator = fused_handle;
    }

    if (moe.fused_shared_input_bfloat16_operator == invalid_compiled_operator_handle)
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
                operators,
                handle,
                invalid_tensor_handle,
                device,
                retain_cpu_dense_copy,
                vulkan_device_index,
                context_instance,
                optimization_flags);
            if (!prepared)
                return prepared.error();
        }
    }
    return prepare_linear_operator(
        weights,
        operators,
        shared.down_weight,
        invalid_tensor_handle,
        device,
        retain_cpu_dense_copy,
        vulkan_device_index,
        context_instance,
        optimization_flags);
}

Result<void> prepare_vulkan_qkv_operator(
    CompiledModel& compiled,
    CompiledLayerPlan& layer_plan,
    bool use_bfloat16_fusion,
    const char* failure_message)
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
        const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
        compiled.operators.at_mutable(fused_handle).bfloat16 = NcnnVulkanBfloat16Operator::create_fused(
            {qkv_matrices[0], qkv_matrices[1], qkv_matrices[2], output_gate},
            {nullptr, nullptr, nullptr, nullptr},
            layer_plan.vulkan_device_index,
            compiled.vulkan_context_instance,
            compiled.opt.optimization_flags);
        if (compiled.operators.at(fused_handle).bfloat16)
            plan.fused_qkv_gate_bfloat16_operator = fused_handle;
    }
    if (plan.fused_qkv_gate_bfloat16_operator == invalid_compiled_operator_handle
        && use_bfloat16_fusion
        && qkv_matrices.front()->dtype
               == DType::BFloat16)
    {
        const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
        compiled.operators.at_mutable(fused_handle).bfloat16 = NcnnVulkanBfloat16Operator::create_fused(
            qkv_matrices,
            qkv_biases,
            layer_plan.vulkan_device_index,
            compiled.vulkan_context_instance,
            compiled.opt.optimization_flags);
        if (compiled.operators.at(fused_handle).bfloat16)
            plan.fused_qkv_bfloat16_operator = fused_handle;
    }
    if (plan.fused_qkv_gate_bfloat16_operator == invalid_compiled_operator_handle
        && plan.fused_qkv_bfloat16_operator == invalid_compiled_operator_handle)
    {
        const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
        compiled.operators.at_mutable(fused_handle).linear = NcnnLinearOperator::create_fused(
            qkv_matrices,
            qkv_biases,
            NcnnLinearDevice::Vulkan,
            layer_plan.vulkan_device_index,
            compiled.vulkan_context_instance,
            compiled.opt.optimization_flags);
        if (compiled.operators.at(fused_handle).linear)
            plan.fused_qkv_operator = fused_handle;
    }
    if (plan.fused_qkv_gate_bfloat16_operator == invalid_compiled_operator_handle
        && plan.fused_qkv_bfloat16_operator == invalid_compiled_operator_handle
        && plan.fused_qkv_operator == invalid_compiled_operator_handle)
        return Error{ErrorCode::InternalError, failure_message};
    return {};
}

Result<void> prepare_latent_attention_operators(
    CompiledModel& compiled,
    CompiledLayerPlan& layer_plan,
    const AttentionDescriptor& attention,
    NcnnLinearDevice attention_device,
    bool retain_cpu_dense_copies,
    const char* diagnostic_prefix)
{
    AttentionBlockPlan& plan = layer_plan.attention;
    Result<void> prepared;
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
        prepared = prepare_linear_operator(compiled.weights, compiled.operators, handle,
                                           invalid_tensor_handle,
                                           attention_device,
                                           retain_cpu_dense_copies,
                                           layer_plan.vulkan_device_index,
                                           compiled.vulkan_context_instance,
                                           compiled.opt.optimization_flags);
        if (!prepared)
            return prepared.error();
    }
    if (attention_device == NcnnLinearDevice::Vulkan
        && attention.compression_ratio == 4
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
            prepared = prepare_linear_operator(compiled.weights, compiled.operators, handle,
                                               invalid_tensor_handle,
                                               attention_device,
                                               retain_cpu_dense_copies,
                                               layer_plan.vulkan_device_index,
                                               compiled.vulkan_context_instance,
                                               compiled.opt.optimization_flags);
            if (!prepared)
                return prepared.error();
        }
    }
    prepared = prepare_linear_operator(compiled.weights, compiled.operators, plan.output_a_weight,
                                       invalid_tensor_handle,
                                       attention_device,
                                       retain_cpu_dense_copies,
                                       layer_plan.vulkan_device_index,
                                       compiled.vulkan_context_instance,
                                       compiled.opt.optimization_flags,
                                       plan.output_group_count);
    if (!prepared)
        return prepared.error();
    if (attention_device == NcnnLinearDevice::Vulkan)
    {
        const CompiledOperator& query_a = compiled.operators.at_weight(plan.query_a_weight);
        const CompiledOperator& query_b = compiled.operators.at_weight(plan.query_b_weight);
        if (query_a.float8 && query_b.float8)
        {
            if (!query_a.float8->prepare_rms_norm(
                    compiled.weights.at(plan.query_norm_weight),
                    compiled.descriptor.norm_epsilon))
                return Error{ErrorCode::InternalError, "failed to prepare " + std::string(diagnostic_prefix) + "Vulkan FP8 query RMSNorm chain"};
            if (!query_a.float8->prepare_input_rms_norm(
                    compiled.weights.at(plan.pre_attention_norm_weight),
                    compiled.descriptor.norm_epsilon))
                return Error{ErrorCode::InternalError, "failed to prepare " + std::string(diagnostic_prefix) + "Vulkan FP8 latent input RMSNorm chain"};
        }
    }
    return {};
}

Result<void> prepare_gated_delta_attention_operators(
    CompiledModel& compiled,
    CompiledLayerPlan& layer_plan,
    NcnnLinearDevice attention_device,
    bool retain_cpu_dense_copies)
{
    AttentionBlockPlan& plan = layer_plan.attention;
    Result<void> prepared;
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
        const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
        compiled.operators.at_mutable(fused_handle).bfloat16 = NcnnVulkanBfloat16Operator::create_fused(
            delta_input_matrices,
            delta_input_biases,
            layer_plan.vulkan_device_index,
            compiled.vulkan_context_instance,
            compiled.opt.optimization_flags);
        if (compiled.operators.at(fused_handle).bfloat16)
            plan.fused_delta_input_bfloat16_operator = fused_handle;
    }
    else
    {
        const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
        compiled.operators.at_mutable(fused_handle).linear = NcnnLinearOperator::create_fused(
            delta_input_matrices,
            delta_input_biases,
            attention_device,
            layer_plan.vulkan_device_index,
            compiled.vulkan_context_instance,
            compiled.opt.optimization_flags);
        if (compiled.operators.at(fused_handle).linear)
            plan.fused_delta_input_operator = fused_handle;
    }
    if (attention_device == NcnnLinearDevice::Vulkan
        && plan.fused_delta_input_bfloat16_operator == invalid_compiled_operator_handle
        && plan.fused_delta_input_operator == invalid_compiled_operator_handle)
    {
        const CompiledOperatorHandle fused_handle = compiled.operators.allocate();
        compiled.operators.at_mutable(fused_handle).linear = NcnnLinearOperator::create_fused(
            delta_input_matrices,
            delta_input_biases,
            attention_device,
            layer_plan.vulkan_device_index,
            compiled.vulkan_context_instance,
            compiled.opt.optimization_flags);
        if (compiled.operators.at(fused_handle).linear)
            plan.fused_delta_input_operator = fused_handle;
    }
    if (attention_device == NcnnLinearDevice::Vulkan
        && plan.fused_delta_input_bfloat16_operator == invalid_compiled_operator_handle
        && plan.fused_delta_input_operator == invalid_compiled_operator_handle)
    {
        return Error{
            ErrorCode::InternalError,
            "failed to create fused Vulkan Gated DeltaNet input operator"};
    }
    if (plan.fused_delta_input_bfloat16_operator == invalid_compiled_operator_handle
        && plan.fused_delta_input_operator == invalid_compiled_operator_handle)
    {
        const TensorHandle delta_input_handles[] = {
            plan.delta_qkv_weight,
            plan.delta_z_weight,
            plan.delta_beta_weight,
            plan.delta_alpha_weight,
        };
        for (TensorHandle handle : delta_input_handles)
        {
            prepared = prepare_linear_operator(compiled.weights, compiled.operators, handle,
                                               invalid_tensor_handle,
                                               attention_device,
                                               retain_cpu_dense_copies,
                                               layer_plan.vulkan_device_index,
                                               compiled.vulkan_context_instance,
                                               compiled.opt.optimization_flags);
            if (!prepared)
                return prepared.error();
        }
    }
    prepared = prepare_linear_operator(compiled.weights, compiled.operators, plan.output_weight,
                                       invalid_tensor_handle,
                                       attention_device,
                                       retain_cpu_dense_copies,
                                       layer_plan.vulkan_device_index,
                                       compiled.vulkan_context_instance,
                                       compiled.opt.optimization_flags);
    if (!prepared)
        return prepared.error();
    if (attention_device == NcnnLinearDevice::Vulkan
        && has_flag(
            compiled.opt.optimization_flags,
            OptimizationVulkanAttention)
        && plan.fused_delta_input_bfloat16_operator != invalid_compiled_operator_handle)
    {
        // Operator-table allocation can relocate entries; keep the operators themselves alive.
        const std::shared_ptr<NcnnVulkanBfloat16Operator> output_operator = compiled.operators.at_weight(plan.output_weight).bfloat16;
        if (output_operator)
        {
            const std::shared_ptr<NcnnVulkanBfloat16Operator> fused_operator = compiled.operators.at(plan.fused_delta_input_bfloat16_operator).bfloat16;
            // Fuse pre-attention RMSNorm with the DeltaNet input projection.
            if (fused_operator
                && plan.pre_attention_norm_weight != invalid_tensor_handle)
            {
                (void)fused_operator->prepare_rms_norm(
                    compiled.weights.at(plan.pre_attention_norm_weight),
                    compiled.descriptor.norm_epsilon,
                    plan.norm_weight_offset);
            }
            const CompiledOperatorHandle gated_handle = compiled.operators.allocate();
            compiled.operators.at_mutable(gated_handle).gated_delta = NcnnVulkanGatedDeltaNetOperator::create(
                fused_operator,
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
                compiled.descriptor.norm_epsilon,
                has_flag(plan.flags, AttentionBlockSigmoidGate),
                layer_plan.vulkan_device_index,
                compiled.vulkan_context_instance,
                compiled.opt.optimization_flags);
            if (compiled.operators.at(gated_handle).gated_delta)
                plan.gated_delta_vulkan_operator = gated_handle;
        }
    }
    return {};
}

Result<void> prepare_standard_attention_operators(
    CompiledModel& compiled,
    CompiledLayerPlan& layer_plan,
    NcnnLinearDevice attention_device,
    bool retain_cpu_dense_copies)
{
    AttentionBlockPlan& plan = layer_plan.attention;
    const bool fused_vulkan_attention_eligible = attention_device == NcnnLinearDevice::Vulkan
                                                 && !has_flag(plan.flags, AttentionBlockQueryKeyNorm)
                                                 && !has_flag(plan.flags, AttentionBlockOutputGate)
                                                 && (plan.rope_head_dimension == 0
                                                     || plan.rope_head_dimension == plan.head_dimension)
                                                 && plan.norm_weight_offset == 0.0f;
    Result<void> prepared;
    if (attention_device == NcnnLinearDevice::Vulkan)
    {
        prepared = prepare_vulkan_qkv_operator(
            compiled, layer_plan, !fused_vulkan_attention_eligible,
            "failed to create fused Vulkan QKV operator");
        if (!prepared)
            return prepared.error();
    }
    else
    {
        prepared = prepare_linear_operator(compiled.weights, compiled.operators, plan.query_weight, plan.query_bias, attention_device, retain_cpu_dense_copies,
                                           layer_plan.vulkan_device_index, compiled.vulkan_context_instance,
                                           compiled.opt.optimization_flags);
        if (!prepared)
            return prepared.error();
        prepared = prepare_linear_operator(compiled.weights, compiled.operators, plan.key_weight, plan.key_bias, attention_device, retain_cpu_dense_copies,
                                           layer_plan.vulkan_device_index, compiled.vulkan_context_instance,
                                           compiled.opt.optimization_flags);
        if (!prepared)
            return prepared.error();
        prepared = prepare_linear_operator(compiled.weights, compiled.operators, plan.value_weight, plan.value_bias, attention_device, retain_cpu_dense_copies,
                                           layer_plan.vulkan_device_index, compiled.vulkan_context_instance,
                                           compiled.opt.optimization_flags);
        if (!prepared)
            return prepared.error();
    }
    prepared = prepare_linear_operator(compiled.weights, compiled.operators, plan.output_weight,
                                       plan.output_bias,
                                       attention_device,
                                       retain_cpu_dense_copies,
                                       layer_plan.vulkan_device_index,
                                       compiled.vulkan_context_instance,
                                       compiled.opt.optimization_flags,
                                       1,
                                       !fused_vulkan_attention_eligible);
    if (!prepared)
        return prepared.error();
    if (plan.output_gate_weight != invalid_tensor_handle)
    {
        if (plan.fused_qkv_gate_bfloat16_operator == invalid_compiled_operator_handle)
        {
            prepared = prepare_linear_operator(compiled.weights, compiled.operators, plan.output_gate_weight,
                                               invalid_tensor_handle,
                                               attention_device,
                                               retain_cpu_dense_copies,
                                               layer_plan.vulkan_device_index,
                                               compiled.vulkan_context_instance,
                                               compiled.opt.optimization_flags);
            if (!prepared)
                return prepared.error();
        }
    }
    if (plan.qsa_query_key_weight != invalid_tensor_handle)
    {
        prepared = prepare_linear_operator(
            compiled.weights, compiled.operators,
            plan.qsa_query_key_weight, invalid_tensor_handle,
            NcnnLinearDevice::Cpu, retain_cpu_dense_copies,
            layer_plan.vulkan_device_index,
            compiled.vulkan_context_instance,
            compiled.opt.optimization_flags);
        if (!prepared)
            return prepared.error();
    }
    if (fused_vulkan_attention_eligible
        || (attention_device == NcnnLinearDevice::Vulkan
            && has_flag(plan.flags, AttentionBlockQueryKeyNorm)
            && has_flag(plan.flags, AttentionBlockOutputGate)
            && plan.fused_qkv_gate_bfloat16_operator != invalid_compiled_operator_handle
            && plan.query_norm_weight != invalid_tensor_handle
            && plan.key_norm_weight != invalid_tensor_handle
            && compiled.operators.at_weight(plan.output_weight).bfloat16))
    {
        NcnnVulkanAttentionConfig attention_config;
        attention_config.hidden_size = compiled.descriptor.hidden_size;
        attention_config.head_count = plan.head_count;
        attention_config.kv_head_count = plan.kv_head_count;
        attention_config.head_dimension = plan.head_dimension;
        attention_config.rope_head_dimension = plan.rope_head_dimension;
        attention_config.sliding_window = plan.sliding_window;
        attention_config.initial_context_length = plan.initial_context_length;
        attention_config.norm_epsilon = compiled.descriptor.norm_epsilon;
        attention_config.norm_weight_offset = plan.norm_weight_offset;
        attention_config.rope_theta = plan.rope_theta;
        attention_config.rope_scaling_factor = plan.rope_scaling_factor;
        attention_config.rope_ntk_alpha = plan.rope_ntk_alpha;
        attention_config.rope_ntk_beta = plan.rope_ntk_beta;
        attention_config.activation_dtype = compiled.descriptor.activation_dtype;
        attention_config.kv_cache_dtype = compiled.descriptor.kv_cache_dtype;
        attention_config.optimization_flags = compiled.opt.optimization_flags;
        if (!fused_vulkan_attention_eligible)
            attention_config.flags |= NcnnAttentionQueryKeyNorm | NcnnAttentionOutputGate;
        if (has_flag(plan.flags, AttentionBlockSink))
            attention_config.flags |= NcnnAttentionSink;
        const CompiledOperatorHandle attention_handle = compiled.operators.allocate();
        if (fused_vulkan_attention_eligible)
        {
            const CompiledOperator& fused_operator = compiled.operators.at(plan.fused_qkv_operator);
            compiled.operators.at_mutable(attention_handle).attention = NcnnVulkanAttentionOperator::create(
                compiled.weights.at(plan.pre_attention_norm_weight),
                plan.sinks == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.sinks),
                fused_operator.linear,
                compiled.operators.at_weight(plan.output_weight).linear,
                attention_config);
        }
        else
        {
            const CompiledOperator& fused_operator = compiled.operators.at(plan.fused_qkv_gate_bfloat16_operator);
            compiled.operators.at_mutable(attention_handle).attention = NcnnVulkanAttentionOperator::create_with_query_key_norm_and_gate(
                compiled.weights.at(plan.pre_attention_norm_weight),
                compiled.weights.at(plan.query_norm_weight),
                compiled.weights.at(plan.key_norm_weight),
                plan.sinks == invalid_tensor_handle ? nullptr : &compiled.weights.at(plan.sinks),
                fused_operator.bfloat16,
                compiled.operators.at_weight(plan.output_weight).bfloat16,
                attention_config);
        }
        if (compiled.operators.at(attention_handle).attention)
            plan.vulkan_attention_operator = attention_handle;
    }
    return {};
}

} // namespace moe
} // namespace ncnn
