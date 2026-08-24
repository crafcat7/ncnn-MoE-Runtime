#include "cpu_gated_delta_net.h"

#include "cpu_fast_math.h"
#include "cpu_bfloat16.h"
#include "cpu_ops.h"
#include "cpu_vector.h"
#include "cpu_state_cache.h"
#include "backends/ncnn/ncnn_linear.h"
#include "engine/cpu_session_state.h"
#include "ncnn/moe/runtime_config.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace ncnn {
namespace moe {

static float gated_delta_tensor_value(const TensorData& tensor, size_t index)
{
    if (tensor.dtype == DType::Float32)
        return tensor.float32_values()[index];
    return bfloat16_to_float(tensor.bfloat16_values()[index]);
}

static float gated_delta_sigmoid(float value)
{
    if (value >= 0.0f)
        return 1.0f / (1.0f + float_approximate_exp(-value));
    const float exponential = float_approximate_exp(value);
    return exponential / (1.0f + exponential);
}

static float gated_delta_softplus(float value)
{
    if (value > 20.0f)
        return value;
    if (value < -20.0f)
        return float_approximate_exp(value);
    return std::log1p(float_approximate_exp(value));
}

static bool gated_delta_simd_enabled(uint64_t optimization_flags) noexcept
{
    return runtime_optimization_enabled(optimization_flags, RuntimeOptimizationCpuGatedDeltaSimd);
}

static void configure_gated_delta_cache(CpuLayerCache& cache, const AttentionBlockPlan& plan)
{
    const uint32_t key_size = plan.kv_head_count * plan.head_dimension;
    const uint32_t value_size = plan.head_count * plan.value_head_dimension;
    const uint32_t convolution_size = key_size * 2 + value_size;
    const size_t convolution_elements = static_cast<size_t>(convolution_size) * plan.convolution_kernel_size;
    const size_t recurrent_elements = static_cast<size_t>(plan.head_count) * plan.head_dimension * plan.value_head_dimension;
    if (cache.gated_delta_convolution.size() == convolution_elements
        && cache.gated_delta_recurrent.size() == recurrent_elements)
    {
        return;
    }
    cache.gated_delta_convolution.assign(convolution_elements, 0.0f);
    cache.gated_delta_recurrent.assign(recurrent_elements, 0.0f);
    cache.gated_delta_token_count = 0;
}

static void execute_depthwise_convolution_row(
    const TensorData& weight,
    uint32_t kernel_size,
    std::vector<float>& state,
    float* values,
    uint32_t columns,
    bool vectorized)
{
    assert(weight.shape.size() == 3);
    assert(weight.shape[0] == columns);
    assert(weight.shape[1] == 1);
    assert(weight.shape[2] == kernel_size);
    if (kernel_size == 4 && weight.dtype == DType::Float32)
    {
        const std::span<const float> filter = weight.float32_values();
        for (uint32_t channel = 0; channel < columns; ++channel)
        {
            float* history = state.data() + static_cast<size_t>(channel) * 4;
            history[0] = history[1];
            history[1] = history[2];
            history[2] = history[3];
            history[3] = values[channel];
            const float* taps = filter.data() + static_cast<size_t>(channel) * 4;
            float sum = history[0] * taps[0];
            sum += history[1] * taps[1];
            sum += history[2] * taps[2];
            sum += history[3] * taps[3];
            values[channel] = sum;
        }
        if (vectorized)
            float_silu_inplace(values, columns);
        else
            for (uint32_t channel = 0; channel < columns; ++channel)
                values[channel] *= gated_delta_sigmoid(values[channel]);
        return;
    }
    if (kernel_size == 4 && weight.dtype == DType::BFloat16)
    {
        const std::span<const uint16_t> filter = weight.bfloat16_values();
        for (uint32_t channel = 0; channel < columns; ++channel)
        {
            float* history = state.data() + static_cast<size_t>(channel) * 4;
            history[0] = history[1];
            history[1] = history[2];
            history[2] = history[3];
            history[3] = values[channel];
            const size_t offset = static_cast<size_t>(channel) * 4;
            float sum = history[0] * bfloat16_to_float(filter[offset]);
            sum += history[1] * bfloat16_to_float(filter[offset + 1]);
            sum += history[2] * bfloat16_to_float(filter[offset + 2]);
            sum += history[3] * bfloat16_to_float(filter[offset + 3]);
            values[channel] = sum;
        }
        if (vectorized)
            float_silu_inplace(values, columns);
        else
            for (uint32_t channel = 0; channel < columns; ++channel)
                values[channel] *= gated_delta_sigmoid(values[channel]);
        return;
    }
    for (uint32_t channel = 0; channel < columns; ++channel)
    {
        float* history = state.data() + static_cast<size_t>(channel) * kernel_size;
        std::move(history + 1, history + kernel_size, history);
        history[kernel_size - 1] = values[channel];
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < kernel_size; ++tap)
        {
            sum += history[tap]
                   * gated_delta_tensor_value(
                       weight,
                       static_cast<size_t>(channel) * kernel_size
                           + tap);
        }
        values[channel] = sum;
    }
    if (vectorized)
        float_silu_inplace(values, columns);
    else
        for (uint32_t channel = 0; channel < columns; ++channel)
            values[channel] *= gated_delta_sigmoid(values[channel]);
}

static void execute_gated_delta_recurrence_row(
    const WeightStore& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    CpuLayerCache& cache,
    float* qkv,
    const float* z,
    const float* beta_values,
    const float* alpha_values,
    std::vector<float>& memory,
    std::vector<float>& delta,
    float* recurrent_output,
    uint64_t optimization_flags)
{
    const uint32_t key_size = plan.kv_head_count * plan.head_dimension;
    const uint32_t head_ratio = plan.head_count / plan.kv_head_count;
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(plan.head_dimension));
    const TensorData& time_bias = weights.at(plan.delta_time_bias);
    const TensorData& decay_log = weights.at(plan.delta_decay_log);
    const TensorData& norm_weight = weights.at(plan.delta_norm_weight);
    const bool vectorized = gated_delta_simd_enabled(optimization_flags);

    // Every value head in a KV group shares the same query/key head.  Normalize
    // each pair once instead of recomputing both norms and their inverse
    // multiplies for every replicated value head.
    for (uint32_t key_head = 0; key_head < plan.kv_head_count; ++key_head)
    {
        float* query = qkv + static_cast<size_t>(key_head) * plan.head_dimension;
        float* key = qkv + key_size
                     + static_cast<size_t>(key_head) * plan.head_dimension;
        if (vectorized)
        {
            float_l2_scale_inplace(query, 1e-6f, plan.head_dimension);
            float_l2_scale_inplace(key, 1e-6f, plan.head_dimension);
        }
        else
        {
            float query_square_sum = 0.0f;
            float key_square_sum = 0.0f;
            for (uint32_t column = 0; column < plan.head_dimension; ++column)
            {
                query_square_sum += query[column] * query[column];
                key_square_sum += key[column] * key[column];
            }
            const float query_inverse_norm = 1.0f / std::sqrt(query_square_sum + 1e-6f);
            const float key_inverse_norm = 1.0f / std::sqrt(key_square_sum + 1e-6f);
            for (uint32_t column = 0; column < plan.head_dimension; ++column)
            {
                query[column] *= query_inverse_norm;
                key[column] *= key_inverse_norm;
            }
        }
    }

    if (vectorized)
    {
        memory.resize(plan.value_head_dimension);
        delta.resize(plan.value_head_dimension);
    }

    for (uint32_t value_head = 0; value_head < plan.head_count; ++value_head)
    {
        const uint32_t key_head = value_head / head_ratio;
        const float* query = qkv + static_cast<size_t>(key_head) * plan.head_dimension;
        const float* key = qkv + key_size
                           + static_cast<size_t>(key_head)
                                 * plan.head_dimension;
        const float* value = qkv + key_size * 2
                             + static_cast<size_t>(value_head)
                                   * plan.value_head_dimension;
        const float beta = gated_delta_sigmoid(beta_values[value_head]);
        const float decay = float_approximate_exp(
            -float_approximate_exp(gated_delta_tensor_value(decay_log, value_head))
            * gated_delta_softplus(
                alpha_values[value_head]
                + gated_delta_tensor_value(time_bias, value_head)));
        float* recurrent = cache.gated_delta_recurrent.data()
                           + static_cast<size_t>(value_head) * plan.head_dimension
                                 * plan.value_head_dimension;
        float* head_output = recurrent_output
                             + static_cast<size_t>(value_head)
                                   * plan.value_head_dimension;
        if (vectorized)
        {
            // Keep the state in its public [key][value] layout, but traverse it
            // row-wise.  Decay and the first matrix-vector product share one
            // load/store pass: the state is updated in place and the decayed
            // row is accumulated into memory before moving to the next key.
            std::fill(memory.begin(), memory.end(), 0.0f);
            for (uint32_t key_column = 0;
                 key_column < plan.head_dimension;
                 ++key_column)
            {
                float_scale_inplace_and_scaled_add(
                    recurrent
                        + static_cast<size_t>(key_column)
                              * plan.value_head_dimension,
                    decay,
                    memory.data(),
                    key[key_column],
                    plan.value_head_dimension);
            }
            for (uint32_t value_column = 0;
                 value_column < plan.value_head_dimension;
                 ++value_column)
            {
                delta[value_column] = (value[value_column] - memory[value_column]) * beta;
            }
            for (uint32_t key_column = 0;
                 key_column < plan.head_dimension;
                 ++key_column)
            {
                float_scaled_add(
                    recurrent
                        + static_cast<size_t>(key_column)
                              * plan.value_head_dimension,
                    delta.data(),
                    key[key_column],
                    plan.value_head_dimension);
            }
            std::fill(head_output,
                      head_output + plan.value_head_dimension,
                      0.0f);
            for (uint32_t key_column = 0;
                 key_column < plan.head_dimension;
                 ++key_column)
            {
                float_scaled_add(
                    head_output,
                    recurrent
                        + static_cast<size_t>(key_column)
                              * plan.value_head_dimension,
                    query[key_column],
                    plan.value_head_dimension);
            }
            float_scale_inplace(
                head_output,
                query_scale,
                plan.value_head_dimension);
        }
        else
        {
            for (uint32_t key_column = 0;
                 key_column < plan.head_dimension;
                 ++key_column)
            {
                float* recurrent_row = recurrent
                                       + static_cast<size_t>(key_column)
                                             * plan.value_head_dimension;
                for (uint32_t value_column = 0;
                     value_column < plan.value_head_dimension;
                     ++value_column)
                {
                    recurrent_row[value_column] *= decay;
                }
            }
            for (uint32_t value_column = 0;
                 value_column < plan.value_head_dimension;
                 ++value_column)
            {
                float memory_value = 0.0f;
                for (uint32_t key_column = 0;
                     key_column < plan.head_dimension;
                     ++key_column)
                {
                    memory_value += recurrent[static_cast<size_t>(key_column)
                                                  * plan.value_head_dimension
                                              + value_column]
                                    * key[key_column];
                }
                const float value_delta = (value[value_column] - memory_value) * beta;
                for (uint32_t key_column = 0;
                     key_column < plan.head_dimension;
                     ++key_column)
                {
                    recurrent[static_cast<size_t>(key_column)
                                  * plan.value_head_dimension
                              + value_column] += key[key_column] * value_delta;
                }
                float output_value = 0.0f;
                for (uint32_t key_column = 0;
                     key_column < plan.head_dimension;
                     ++key_column)
                {
                    output_value += recurrent[static_cast<size_t>(key_column)
                                                  * plan.value_head_dimension
                                              + value_column]
                                    * query[key_column];
                }
                head_output[value_column] = output_value * query_scale;
            }
        }

        const bool vectorized_norm = vectorized
                                     && (norm_weight.dtype == DType::Float32
                                         || norm_weight.dtype == DType::BFloat16);
        if (vectorized_norm && norm_weight.dtype == DType::Float32)
        {
            float_rms_norm(
                head_output,
                head_output,
                norm_weight.float32_values().data(),
                norm_epsilon,
                0.0f,
                plan.value_head_dimension);
        }
        else if (vectorized_norm)
        {
            bfloat16_rms_norm(
                head_output,
                head_output,
                norm_weight.bfloat16_values().data(),
                norm_epsilon,
                0.0f,
                plan.value_head_dimension);
        }
        if (vectorized_norm)
        {
            const float* head_gate = z + static_cast<size_t>(value_head) * plan.value_head_dimension;
            float_silu_mul(
                head_output,
                head_gate,
                head_output,
                1.0f,
                0.0f,
                plan.value_head_dimension);
        }
        else
        {
            const float square_sum = vectorized
                                         ? float_dot(
                                               head_output,
                                               head_output,
                                               plan.value_head_dimension)
                                         : [&]() {
                                               float sum = 0.0f;
                                               for (uint32_t value_column = 0;
                                                    value_column < plan.value_head_dimension;
                                                    ++value_column)
                                                   sum += head_output[value_column] * head_output[value_column];
                                               return sum;
                                           }();
            const float inverse_rms = 1.0f
                                      / std::sqrt(
                                          square_sum
                                              / static_cast<float>(plan.value_head_dimension)
                                          + norm_epsilon);
            const float* head_gate = z + static_cast<size_t>(value_head) * plan.value_head_dimension;
            for (uint32_t value_column = 0;
                 value_column < plan.value_head_dimension;
                 ++value_column)
            {
                head_output[value_column] *= inverse_rms
                                             * gated_delta_tensor_value(norm_weight, value_column)
                                             * (head_gate[value_column]
                                                * gated_delta_sigmoid(head_gate[value_column]));
            }
        }
    }
}

Result<void> execute_gated_delta_net_into(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    CpuLayerCache& cache,
    CpuGatedDeltaExecutionScratch& scratch,
    const CpuBatch& hidden,
    CpuBatch& output,
    uint64_t optimization_flags)
{
    const CompiledOperator& gated_delta_operator = operators.at(plan.gated_delta_vulkan_operator);
    const bool device_state_available = backend == ExecutionBackend::Vulkan
                                        && gated_delta_operator.gated_delta
                                        && (cache.gated_delta_device_state
                                            || hidden.rows() == 1);
    bool normalized_ready = false;
    if (device_state_available)
    {
        const bool device_input_rms_norm = gated_delta_operator.gated_delta->has_input_rms_norm();
        if (!device_input_rms_norm)
        {
            rms_norm_batch_into(
                hidden,
                weights.at(plan.pre_attention_norm_weight),
                norm_epsilon,
                scratch.normalized,
                plan.norm_weight_offset,
                optimization_flags);
            normalized_ready = true;
        }
        const bool device_executed = device_input_rms_norm
                                         ? gated_delta_operator.gated_delta->forward_input_rms_norm(
                                               hidden,
                                               cache,
                                               scratch.projected)
                                         : gated_delta_operator.gated_delta->forward(
                                               scratch.normalized,
                                               cache,
                                               scratch.projected);
        if (device_executed)
        {
            output = hidden;
            add_batch_inplace(output, scratch.projected);
            cache.gated_delta_convolution.clear();
            cache.gated_delta_recurrent.clear();
            for (size_t row = 0; row < hidden.rows(); ++row)
                record_gated_delta_cache_transaction_row(cache);
            cache.gated_delta_token_count += hidden.rows();
            if (cache.gated_delta_device_state)
                cache.device_allocated_bytes = cache.gated_delta_device_state->allocated_bytes();
            return {};
        }

        if (cache.transaction.active
            && cache.gated_delta_device_state)
        {
            return Error{
                ErrorCode::InternalError,
                "Vulkan Gated DeltaNet failed while its transactional state was authoritative"};
        }

        // A failed device dispatch should not strand the Session on an
        // opaque state.  Download once, then continue on the established CPU
        // implementation.  This is a failure-path synchronization, not a
        // per-token boundary.
        if (cache.gated_delta_device_state)
        {
            std::vector<float> convolution;
            std::vector<float> recurrent;
            if (!cache.gated_delta_device_state->download(
                    convolution,
                    recurrent))
            {
                return Error{
                    ErrorCode::InternalError,
                    "failed to recover Vulkan Gated DeltaNet state"};
            }
            cache.gated_delta_convolution = std::move(convolution);
            cache.gated_delta_recurrent = std::move(recurrent);
            cache.gated_delta_device_state.reset();
            cache.device_allocated_bytes = 0;
        }
    }

    if (!normalized_ready)
    {
        rms_norm_batch_into(
            hidden,
            weights.at(plan.pre_attention_norm_weight),
            norm_epsilon,
            scratch.normalized,
            plan.norm_weight_offset,
            optimization_flags);
        normalized_ready = true;
    }

    configure_gated_delta_cache(cache, plan);
    const uint32_t key_size = plan.kv_head_count * plan.head_dimension;
    const uint32_t value_size = plan.head_count * plan.value_head_dimension;
    const uint32_t convolution_size = key_size * 2 + value_size;
    const uint32_t fused_columns = convolution_size + value_size + plan.head_count * 2;
    const CompiledOperator& fused_delta_bfloat16_operator = operators.at(plan.fused_delta_input_bfloat16_operator);
    const CompiledOperator& fused_delta_linear_operator = operators.at(plan.fused_delta_input_operator);
    bool fused_input = (backend == ExecutionBackend::Vulkan
                        && fused_delta_bfloat16_operator.bfloat16
                        && fused_delta_bfloat16_operator.bfloat16->forward(
                            scratch.normalized,
                            scratch.fused_input))
                       || (backend == ExecutionBackend::Vulkan
                           && fused_delta_linear_operator.linear
                           && fused_delta_linear_operator.linear->forward(
                               scratch.normalized,
                               scratch.fused_input));
    if (fused_input)
    {
        fused_input = scratch.fused_input.rows() == hidden.rows()
                      && scratch.fused_input.columns() == fused_columns;
    }
    if (!fused_input)
    {
        linear_batch_into(
            weights.at(plan.delta_qkv_weight),
            scratch.normalized,
            scratch.qkv,
            optimization_flags,
            &operators.at_weight(plan.delta_qkv_weight));
        linear_batch_into(
            weights.at(plan.delta_z_weight),
            scratch.normalized,
            scratch.z,
            optimization_flags,
            &operators.at_weight(plan.delta_z_weight));
        linear_batch_into(
            weights.at(plan.delta_beta_weight),
            scratch.normalized,
            scratch.beta,
            optimization_flags,
            &operators.at_weight(plan.delta_beta_weight));
        linear_batch_into(
            weights.at(plan.delta_alpha_weight),
            scratch.normalized,
            scratch.alpha,
            optimization_flags,
            &operators.at_weight(plan.delta_alpha_weight));
    }
    scratch.recurrent_output.reset(
        hidden.rows(),
        value_size,
        false);
    for (size_t token_index = 0; token_index < hidden.rows(); ++token_index)
    {
        float* qkv = fused_input
                         ? scratch.fused_input.row(token_index)
                         : scratch.qkv.row(token_index);
        const float* z = fused_input
                             ? qkv + convolution_size
                             : scratch.z.row(token_index);
        const float* beta = fused_input
                                ? z + value_size
                                : scratch.beta.row(token_index);
        const float* alpha = fused_input
                                 ? beta + plan.head_count
                                 : scratch.alpha.row(token_index);
        execute_depthwise_convolution_row(
            weights.at(plan.delta_convolution_weight),
            plan.convolution_kernel_size,
            cache.gated_delta_convolution,
            qkv,
            convolution_size,
            gated_delta_simd_enabled(optimization_flags));
        execute_gated_delta_recurrence_row(
            weights,
            plan,
            norm_epsilon,
            cache,
            qkv,
            z,
            beta,
            alpha,
            scratch.recurrent_memory,
            scratch.recurrent_delta,
            scratch.recurrent_output.row(token_index),
            optimization_flags);
        record_gated_delta_cache_transaction_row(cache);
    }
    linear_batch_into(weights.at(plan.output_weight), scratch.recurrent_output, scratch.projected, optimization_flags, &operators.at_weight(plan.output_weight));
    output = hidden;
    add_batch_inplace(output, scratch.projected);
    cache.gated_delta_token_count += hidden.rows();
    return {};
}

bool execute_gated_delta_net_batch_into(
    const WeightStore& weights,
    const CompiledOperatorTable& operators,
    const AttentionBlockPlan& plan,
    ExecutionBackend backend,
    float norm_epsilon,
    std::span<CpuGatedDeltaBatchEntry> entries,
    CpuGatedDeltaBatchScratch& scratch,
    uint64_t optimization_flags)
{
    if (entries.empty())
        return true;
    const CompiledOperator& gated_delta_operator = operators.at(plan.gated_delta_vulkan_operator);
    if (backend != ExecutionBackend::Vulkan
        || !gated_delta_operator.gated_delta
        || entries.size() == 1)
    {
        for (CpuGatedDeltaBatchEntry& entry : entries)
        {
            if (!entry.hidden || !entry.scratch || !entry.cache || !entry.output)
                return false;
            auto executed = execute_gated_delta_net_into(
                weights,
                operators,
                plan,
                backend,
                norm_epsilon,
                *entry.cache,
                *entry.scratch,
                *entry.hidden,
                *entry.output,
                optimization_flags);
            if (!executed)
                return false;
        }
        return true;
    }

    std::vector<NcnnVulkanGatedDeltaBatchEntry>& device_entries = scratch.device_entries;
    device_entries.clear();
    device_entries.reserve(entries.size());
    for (CpuGatedDeltaBatchEntry& entry : entries)
    {
        if (!entry.hidden || !entry.scratch || !entry.cache || !entry.output
            || entry.hidden->rows() != 1)
        {
            device_entries.clear();
            break;
        }
        rms_norm_batch_into(
            *entry.hidden,
            weights.at(plan.pre_attention_norm_weight),
            norm_epsilon,
            entry.scratch->normalized,
            plan.norm_weight_offset,
            optimization_flags);
        device_entries.push_back({&entry.scratch->normalized,
                                  entry.cache,
                                  &entry.scratch->projected});
    }

    NcnnVulkanGatedDeltaBatchResult batch_result = NcnnVulkanGatedDeltaBatchResult::NotExecuted;
    if (!device_entries.empty()
        && device_entries.size() == entries.size())
    {
        batch_result = gated_delta_operator.gated_delta->forward_batch(device_entries);
    }
    if (batch_result == NcnnVulkanGatedDeltaBatchResult::Executed)
    {
        for (CpuGatedDeltaBatchEntry& entry : entries)
        {
            entry.output->reset(
                entry.hidden->rows(),
                entry.hidden->columns(),
                false);
            std::copy_n(
                entry.hidden->row(0),
                entry.hidden->columns(),
                entry.output->row(0));
            add_batch_inplace(*entry.output, entry.scratch->projected);
            entry.cache->gated_delta_convolution.clear();
            entry.cache->gated_delta_recurrent.clear();
            record_gated_delta_cache_transaction_row(*entry.cache);
        }
        return true;
    }
    if (batch_result == NcnnVulkanGatedDeltaBatchResult::Failed)
        return false;

    // The batch path is an optimization for independent decode rows.  If a
    // device allocation or dispatch is unavailable, preserve the established
    // per-Session implementation and its failure-path state handoff.
    for (CpuGatedDeltaBatchEntry& entry : entries)
    {
        if (!entry.hidden || !entry.scratch || !entry.cache || !entry.output)
            return false;
        auto executed = execute_gated_delta_net_into(
            weights,
            operators,
            plan,
            backend,
            norm_epsilon,
            *entry.cache,
            *entry.scratch,
            *entry.hidden,
            *entry.output,
            optimization_flags);
        if (!executed)
            return false;
    }
    return true;
}

} // namespace moe
} // namespace ncnn
