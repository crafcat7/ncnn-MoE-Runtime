#include "gateddeltanet.h"

#include "fastmath.h"
#include "bfloat16.h"
#include "ops.h"
#include "vector.h"
#include "statecache.h"
#include "backends/ncnn/linear.h"
#include "ncnn/moe/option.h"

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

static void configure_gated_delta_cache(LayerCache& cache, const AttentionBlockPlan& plan)
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

static void execute_depthwise_convolution_row(const TensorData& weight,
                                              uint32_t kernel_size,
                                              std::vector<float>& state,
                                              float* values,
                                              uint32_t columns)
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
        float_silu_inplace(values, columns);
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
        float_silu_inplace(values, columns);
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
                   * gated_delta_tensor_value(weight,
                                              static_cast<size_t>(channel) * kernel_size
                                                  + tap);
        }
        values[channel] = sum;
    }
    float_silu_inplace(values, columns);
}

static void forward_gated_delta_recurrence(const WeightStore& weights,
                                           const AttentionBlockPlan& plan,
                                           LayerCache& cache,
                                           GatedDeltaScratch& scratch,
                                           bool fused_input,
                                           size_t first_row,
                                           size_t row_count)
{
    const uint32_t key_size = plan.kv_head_count * plan.head_dimension;
    const uint32_t value_size = plan.head_count * plan.value_head_dimension;
    const uint32_t convolution_size = key_size * 2 + value_size;
    const uint32_t head_ratio = plan.head_count / plan.kv_head_count;
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(plan.head_dimension));
    const TensorData& time_bias = weights.at(plan.delta_time_bias);
    const TensorData& decay_log = weights.at(plan.delta_decay_log);
    const TensorData& norm_weight = weights.at(plan.delta_norm_weight);
    const size_t qkv_stride = fused_input ? scratch.fused_input.columns() : scratch.qkv.columns();
    const size_t z_stride = fused_input ? scratch.fused_input.columns() : scratch.z.columns();
    const size_t beta_stride = fused_input ? scratch.fused_input.columns() : scratch.beta.columns();
    const size_t alpha_stride = fused_input ? scratch.fused_input.columns() : scratch.alpha.columns();
    float* qkv = fused_input ? scratch.fused_input.row(first_row) : scratch.qkv.row(first_row);
    const float* z = fused_input ? qkv + convolution_size : scratch.z.row(first_row);
    const float* beta_values = fused_input ? z + value_size : scratch.beta.row(first_row);
    const float* alpha_values = fused_input ? beta_values + plan.head_count : scratch.alpha.row(first_row);
    float* recurrent_output = scratch.recurrent_output.row(first_row);
    std::vector<float>& memory = scratch.recurrent_memory;
    std::vector<float>& delta = scratch.recurrent_delta;

    // Every value head in a KV group shares the same query/key head.  Normalize
    // each pair once instead of recomputing both norms and their inverse
    // multiplies for every replicated value head.
    for (size_t token_index = 0; token_index < row_count; ++token_index)
    {
        float* token_qkv = qkv + token_index * qkv_stride;
        for (uint32_t key_head = 0; key_head < plan.kv_head_count; ++key_head)
        {
            float* query = token_qkv + static_cast<size_t>(key_head) * plan.head_dimension;
            float* key = token_qkv + key_size
                         + static_cast<size_t>(key_head) * plan.head_dimension;
            float_l2_scale_inplace(query, 1e-6f, plan.head_dimension);
            float_l2_scale_inplace(key, 1e-6f, plan.head_dimension);
        }
    }

    // The recurrence performs several state/value vector sweeps per element;
    // this factor estimates that work for the shared linear team policy.
    const uint64_t operation_count = static_cast<uint64_t>(row_count)
                                     * static_cast<uint64_t>(plan.head_count)
                                     * static_cast<uint64_t>(plan.head_dimension)
                                     * static_cast<uint64_t>(plan.value_head_dimension)
                                     * 8;
    const uint32_t head_team_size = std::min(plan.head_count,
                                             static_cast<uint32_t>(cpu_linear_team_size(operation_count,
                                                                                        DType::Float32)));
    const bool parallelize_value_heads = head_team_size > 1;
    const size_t scratch_head_count = head_team_size > 1
                                          ? plan.head_count
                                          : 1;
    memory.resize(static_cast<size_t>(scratch_head_count)
                  * plan.value_head_dimension);
    delta.resize(static_cast<size_t>(scratch_head_count)
                 * plan.value_head_dimension);

#pragma omp parallel for schedule(static) num_threads(head_team_size) if (parallelize_value_heads)
    for (int64_t value_head_index = 0;
         value_head_index < static_cast<int64_t>(plan.head_count);
         ++value_head_index)
    {
        const uint32_t value_head = static_cast<uint32_t>(value_head_index);
        const uint32_t key_head = value_head / head_ratio;
        float* recurrent = cache.gated_delta_recurrent.data()
                           + static_cast<size_t>(value_head) * plan.head_dimension
                                 * plan.value_head_dimension;
        float* memory_values = memory.data() + static_cast<size_t>(parallelize_value_heads ? value_head : 0) * plan.value_head_dimension;
        float* delta_values = delta.data() + static_cast<size_t>(parallelize_value_heads ? value_head : 0) * plan.value_head_dimension;
        const float decay_scale = -float_approximate_exp(gated_delta_tensor_value(decay_log, value_head));
        const float time_bias_value = gated_delta_tensor_value(time_bias, value_head);
        for (size_t token_index = 0; token_index < row_count; ++token_index)
        {
            const float* token_qkv = qkv + token_index * qkv_stride;
            const float* query = token_qkv + static_cast<size_t>(key_head) * plan.head_dimension;
            const float* key = token_qkv + key_size
                               + static_cast<size_t>(key_head) * plan.head_dimension;
            const float* value = token_qkv + key_size * 2
                                 + static_cast<size_t>(value_head) * plan.value_head_dimension;
            const float* token_beta = beta_values + token_index * beta_stride;
            const float* token_alpha = alpha_values + token_index * alpha_stride;
            const float beta = gated_delta_sigmoid(token_beta[value_head]);
            const float decay = float_approximate_exp(decay_scale
                                                      * gated_delta_softplus(token_alpha[value_head]
                                                                             + time_bias_value));
            float* head_output = recurrent_output
                                 + token_index * scratch.recurrent_output.columns()
                                 + static_cast<size_t>(value_head) * plan.value_head_dimension;
            // Keep the state in its public [key][value] layout, but traverse it
            // row-wise.  Decay and the first matrix-vector product share one
            // load/store pass: the state is updated in place and the decayed
            // row is accumulated into memory before moving to the next key.
            std::fill(memory_values,
                      memory_values + plan.value_head_dimension,
                      0.0f);
            for (uint32_t key_column = 0;
                 key_column < plan.head_dimension;
                 ++key_column)
            {
                float_scale_inplace_and_scaled_add(recurrent
                                                       + static_cast<size_t>(key_column)
                                                             * plan.value_head_dimension,
                                                   decay,
                                                   memory_values,
                                                   key[key_column],
                                                   plan.value_head_dimension);
            }
            for (uint32_t value_column = 0;
                 value_column < plan.value_head_dimension;
                 ++value_column)
            {
                delta_values[value_column] = (value[value_column] - memory_values[value_column]) * beta;
            }
            std::fill(head_output,
                      head_output + plan.value_head_dimension,
                      0.0f);
            for (uint32_t key_column = 0;
                 key_column < plan.head_dimension;
                 ++key_column)
            {
                float_scale_inplace_and_scaled_add_and_accumulate(recurrent
                                                                      + static_cast<size_t>(key_column)
                                                                            * plan.value_head_dimension,
                                                                  1.0f,
                                                                  delta_values,
                                                                  key[key_column],
                                                                  head_output,
                                                                  query[key_column],
                                                                  plan.value_head_dimension);
            }
            float_scale_inplace(head_output,
                                query_scale,
                                plan.value_head_dimension);

            const bool fused_norm = norm_weight.dtype == DType::Float32 || norm_weight.dtype == DType::BFloat16;
            if (fused_norm && norm_weight.dtype == DType::Float32)
            {
                float_rms_norm(head_output,
                               head_output,
                               norm_weight.float32_values().data(),
                               plan.norm_epsilon,
                               0.0f,
                               plan.value_head_dimension);
            }
            else if (fused_norm)
            {
                bfloat16_rms_norm(head_output,
                                  head_output,
                                  norm_weight.bfloat16_values().data(),
                                  plan.norm_epsilon,
                                  0.0f,
                                  plan.value_head_dimension);
            }
            if (fused_norm)
            {
                const float* head_gate = z + token_index * z_stride
                                         + static_cast<size_t>(value_head) * plan.value_head_dimension;
                if (has_flag(plan.flags, AttentionBlockSigmoidGate))
                {
                    float_sigmoid_mul(head_output, head_gate, head_output,
                                      plan.value_head_dimension);
                }
                else
                {
                    float_silu_mul(head_output,
                                   head_gate,
                                   head_output,
                                   1.0f,
                                   0.0f,
                                   plan.value_head_dimension);
                }
            }
            else
            {
                const float square_sum = float_dot(head_output, head_output, plan.value_head_dimension);
                const float inverse_rms = 1.0f
                                          / std::sqrt(square_sum
                                                          / static_cast<float>(plan.value_head_dimension)
                                                      + plan.norm_epsilon);
                const float* head_gate = z + token_index * z_stride
                                         + static_cast<size_t>(value_head) * plan.value_head_dimension;
                for (uint32_t value_column = 0;
                     value_column < plan.value_head_dimension;
                     ++value_column)
                {
                    head_output[value_column] *= inverse_rms
                                                 * gated_delta_tensor_value(norm_weight, value_column)
                                                 * (has_flag(plan.flags, AttentionBlockSigmoidGate)
                                                        ? gated_delta_sigmoid(head_gate[value_column])
                                                        : head_gate[value_column]
                                                              * gated_delta_sigmoid(head_gate[value_column]));
                }
            }
        }
    }
}

Result<void> forward_gated_delta(const WeightStore& weights,
                                 const CompiledOperatorTable& operators,
                                 const AttentionBlockPlan& plan,
                                 ExecutionBackend backend,
                                 LayerCache& cache,
                                 GatedDeltaScratch& scratch,
                                 const ActivationBuffer& hidden,
                                 ActivationBuffer& output,
                                 uint64_t optimization_flags)
{
    const CompiledOperator& gated_delta_operator = operators.at(plan.gated_delta_vulkan_operator);
    const bool device_state_available = backend == ExecutionBackend::Vulkan
                                        && gated_delta_operator.gated_delta
                                        && (cache.gated_delta_device_state
                                            || hidden.rows() == 1);
    const ActivationBuffer* normalized = &hidden;
    bool normalized_ready = false;
    if (device_state_available)
    {
        const bool device_input_rms_norm = gated_delta_operator.gated_delta->has_input_rms_norm();
        if (!device_input_rms_norm)
        {
            if (plan.pre_attention_norm_weight != invalid_tensor_handle)
            {
                rms_norm_batch_into(hidden,
                                    weights.at(plan.pre_attention_norm_weight),
                                    plan.norm_epsilon,
                                    scratch.normalized,
                                    plan.norm_weight_offset);
                normalized = &scratch.normalized;
            }
            normalized_ready = true;
        }
        const bool device_executed = device_input_rms_norm
                                         ? gated_delta_operator.gated_delta->forward_input_rms_norm(hidden,
                                                                                                    cache,
                                                                                                    scratch.projected)
                                         : gated_delta_operator.gated_delta->forward(*normalized,
                                                                                     cache,
                                                                                     scratch.projected);
        if (device_executed)
        {
            const size_t hidden_rows = hidden.rows();
            if (has_flag(plan.flags, AttentionBlockExternalResidual))
                output.swap(scratch.projected);
            else
            {
                add_batch_inplace(scratch.projected, hidden);
                output.swap(scratch.projected);
            }
            cache.gated_delta_convolution.clear();
            cache.gated_delta_recurrent.clear();
            for (size_t row = 0; row < hidden_rows; ++row)
                record_gated_delta_cache_transaction_row(cache);
            cache.gated_delta_token_count += hidden_rows;
            if (cache.gated_delta_device_state)
                cache.device_allocated_size = cache.gated_delta_device_state->allocated_bytes();
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
            if (!cache.gated_delta_device_state->download(convolution,
                                                          recurrent))
            {
                return Error{
                    ErrorCode::InternalError,
                    "failed to recover Vulkan Gated DeltaNet state"};
            }
            cache.gated_delta_convolution = std::move(convolution);
            cache.gated_delta_recurrent = std::move(recurrent);
            cache.gated_delta_device_state.reset();
            cache.device_allocated_size = 0;
        }
    }

    if (!normalized_ready)
    {
        if (plan.pre_attention_norm_weight != invalid_tensor_handle)
        {
            rms_norm_batch_into(hidden,
                                weights.at(plan.pre_attention_norm_weight),
                                plan.norm_epsilon,
                                scratch.normalized,
                                plan.norm_weight_offset);
            normalized = &scratch.normalized;
        }
        normalized_ready = true;
    }

    configure_gated_delta_cache(cache, plan);
    const uint32_t key_size = plan.kv_head_count * plan.head_dimension;
    const uint32_t value_size = plan.head_count * plan.value_head_dimension;
    const uint32_t convolution_size = key_size * 2 + value_size;
    const uint32_t fused_columns = convolution_size + value_size + plan.head_count * 2;
    const CompiledOperator& fused_delta_operator = operators.at(plan.fused_delta_input_operator);
    bool fused_input = (backend == ExecutionBackend::Vulkan
                        && fused_delta_operator.bfloat16
                        && fused_delta_operator.bfloat16->forward(*normalized,
                                                                  scratch.fused_input))
                       || (backend == ExecutionBackend::Vulkan
                           && fused_delta_operator.linear
                           && fused_delta_operator.linear->forward(*normalized,
                                                                   scratch.fused_input));
    if (fused_input)
    {
        fused_input = scratch.fused_input.rows() == hidden.rows()
                      && scratch.fused_input.columns() == fused_columns;
    }
    if (!fused_input)
    {
        linear_batch_into(weights.at(plan.delta_qkv_weight),
                          *normalized,
                          scratch.qkv,
                          optimization_flags,
                          operators.find_weight(plan.delta_qkv_weight));
        linear_batch_into(weights.at(plan.delta_z_weight),
                          *normalized,
                          scratch.z,
                          optimization_flags,
                          operators.find_weight(plan.delta_z_weight));
        linear_batch_into(weights.at(plan.delta_beta_weight),
                          *normalized,
                          scratch.beta,
                          optimization_flags,
                          operators.find_weight(plan.delta_beta_weight));
        linear_batch_into(weights.at(plan.delta_alpha_weight),
                          *normalized,
                          scratch.alpha,
                          optimization_flags,
                          operators.find_weight(plan.delta_alpha_weight));
    }
    scratch.recurrent_output.reset(hidden.rows(),
                                   value_size,
                                   false);
    const size_t hidden_rows = hidden.rows();
    // Transactions retain each token's state; ordinary prefill shares one
    // head team across all rows without changing their recurrent order.
    const size_t rows = cache.transaction.active ? 1 : hidden_rows;
    for (size_t first = 0; first < hidden_rows; first += rows)
    {
        for (size_t token_index = first; token_index < first + rows; ++token_index)
        {
            float* qkv = fused_input
                             ? scratch.fused_input.row(token_index)
                             : scratch.qkv.row(token_index);
            execute_depthwise_convolution_row(weights.at(plan.delta_convolution_weight),
                                              plan.convolution_kernel_size,
                                              cache.gated_delta_convolution,
                                              qkv,
                                              convolution_size);
        }
        forward_gated_delta_recurrence(weights,
                                       plan,
                                       cache,
                                       scratch,
                                       fused_input,
                                       first,
                                       rows);
        record_gated_delta_cache_transaction_row(cache);
    }
    ActivationBuffer& projected = (&output == &hidden || &output == &scratch.recurrent_output)
                                      ? scratch.projected
                                      : output;
    linear_batch_into(weights.at(plan.output_weight),
                      scratch.recurrent_output,
                      projected,
                      optimization_flags,
                      operators.find_weight(plan.output_weight));
    if (!has_flag(plan.flags, AttentionBlockExternalResidual))
        add_batch_inplace(projected, hidden);
    if (&projected != &output)
        output.swap(projected);
    cache.gated_delta_token_count += hidden_rows;
    return {};
}

bool forward_gated_delta_batch(const WeightStore& weights,
                               const CompiledOperatorTable& operators,
                               const AttentionBlockPlan& plan,
                               ExecutionBackend backend,
                               std::span<GatedDeltaBatchEntry> entries,
                               std::vector<GatedDeltaBatchEntry_vulkan>& device_entries,
                               uint64_t optimization_flags)
{
    if (entries.empty())
        return true;
    const CompiledOperator& gated_delta_operator = operators.at(plan.gated_delta_vulkan_operator);
    if (backend != ExecutionBackend::Vulkan
        || !gated_delta_operator.gated_delta
        || entries.size() == 1)
    {
        for (GatedDeltaBatchEntry& entry : entries)
        {
            if (!entry.hidden || !entry.scratch || !entry.cache || !entry.output)
                return false;
            auto executed = forward_gated_delta(weights,
                                                operators,
                                                plan,
                                                backend,
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

    device_entries.clear();
    device_entries.reserve(entries.size());
    for (GatedDeltaBatchEntry& entry : entries)
    {
        if (!entry.hidden || !entry.scratch || !entry.cache || !entry.output
            || entry.hidden->rows() != 1)
        {
            device_entries.clear();
            break;
        }
        const ActivationBuffer* normalized = entry.hidden;
        if (plan.pre_attention_norm_weight != invalid_tensor_handle)
        {
            rms_norm_batch_into(*entry.hidden,
                                weights.at(plan.pre_attention_norm_weight),
                                plan.norm_epsilon,
                                entry.scratch->normalized,
                                plan.norm_weight_offset);
            normalized = &entry.scratch->normalized;
        }
        device_entries.push_back({normalized,
                                  entry.cache,
                                  &entry.scratch->projected});
    }

    GatedDeltaBatchResult_vulkan batch_result = GatedDeltaBatchResult_vulkan::NotExecuted;
    if (!device_entries.empty()
        && device_entries.size() == entries.size())
    {
        batch_result = gated_delta_operator.gated_delta->forward_batch(device_entries);
    }
    if (batch_result == GatedDeltaBatchResult_vulkan::Executed)
    {
        for (GatedDeltaBatchEntry& entry : entries)
        {
            if (has_flag(plan.flags, AttentionBlockExternalResidual))
                entry.output->swap(entry.scratch->projected);
            else
            {
                add_batch_inplace(entry.scratch->projected, *entry.hidden);
                entry.output->swap(entry.scratch->projected);
            }
            entry.cache->gated_delta_convolution.clear();
            entry.cache->gated_delta_recurrent.clear();
            record_gated_delta_cache_transaction_row(*entry.cache);
        }
        return true;
    }
    if (batch_result == GatedDeltaBatchResult_vulkan::Failed)
        return false;

    // The batch path is an optimization for independent decode rows.  If a
    // device allocation or dispatch is unavailable, preserve the established
    // per-Session implementation and its failure-path state handoff.
    for (GatedDeltaBatchEntry& entry : entries)
    {
        if (!entry.hidden || !entry.scratch || !entry.cache || !entry.output)
            return false;
        auto executed = forward_gated_delta(weights,
                                            operators,
                                            plan,
                                            backend,
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
