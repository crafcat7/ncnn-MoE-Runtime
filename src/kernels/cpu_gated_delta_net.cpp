#include "cpu_gated_delta_net.h"

#include "cpu_bfloat16.h"
#include "cpu_ops.h"
#include "cpu_state_cache.h"
#include "backends/ncnn/ncnn_linear.h"
#include "engine/cpu_session_state.h"

#include <algorithm>
#include <cassert>
#include <cmath>

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
        return 1.0f / (1.0f + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

static float gated_delta_softplus(float value)
{
    if (value > 20.0f)
        return value;
    if (value < -20.0f)
        return std::exp(value);
    return std::log1p(std::exp(value));
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
    uint32_t columns)
{
    assert(weight.shape.size() == 3);
    assert(weight.shape[0] == columns);
    assert(weight.shape[1] == 1);
    assert(weight.shape[2] == kernel_size);
    for (uint32_t channel = 0; channel < columns; ++channel)
    {
        float* history =
            state.data() + static_cast<size_t>(channel) * kernel_size;
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
        values[channel] = sum * gated_delta_sigmoid(sum);
    }
}

static void execute_gated_delta_recurrence_row(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    CpuLayerCache& cache,
    const float* qkv,
    const float* z,
    const float* beta_values,
    const float* alpha_values,
    float* recurrent_output)
{
    const uint32_t key_size = plan.kv_head_count * plan.head_dimension;
    const uint32_t head_ratio = plan.head_count / plan.kv_head_count;
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(plan.head_dimension));
    const TensorData& time_bias = weights.at(plan.delta_time_bias);
    const TensorData& decay_log = weights.at(plan.delta_decay_log);
    const TensorData& norm_weight = weights.at(plan.delta_norm_weight);

    for (uint32_t value_head = 0; value_head < plan.head_count; ++value_head)
    {
        const uint32_t key_head = value_head / head_ratio;
        const float* query =
            qkv + static_cast<size_t>(key_head) * plan.head_dimension;
        const float* key = qkv + key_size
                           + static_cast<size_t>(key_head)
                                 * plan.head_dimension;
        const float* value = qkv + key_size * 2
                             + static_cast<size_t>(value_head)
                                   * plan.value_head_dimension;
        float query_square_sum = 0.0f;
        float key_square_sum = 0.0f;
        for (uint32_t column = 0; column < plan.head_dimension; ++column)
        {
            query_square_sum += query[column] * query[column];
            key_square_sum += key[column] * key[column];
        }
        const float query_inverse_norm =
            1.0f / std::sqrt(query_square_sum + 1e-6f);
        const float key_inverse_norm =
            1.0f / std::sqrt(key_square_sum + 1e-6f);
        const float beta = gated_delta_sigmoid(beta_values[value_head]);
        const float decay = std::exp(
            -std::exp(gated_delta_tensor_value(decay_log, value_head))
            * gated_delta_softplus(
                alpha_values[value_head]
                + gated_delta_tensor_value(time_bias, value_head)));
        float* recurrent =
            cache.gated_delta_recurrent.data()
            + static_cast<size_t>(value_head) * plan.head_dimension
                  * plan.value_head_dimension;
        for (uint32_t key_column = 0;
             key_column < plan.head_dimension;
             ++key_column)
        {
            float* recurrent_row =
                recurrent
                + static_cast<size_t>(key_column)
                      * plan.value_head_dimension;
            for (uint32_t value_column = 0;
                 value_column < plan.value_head_dimension;
                 ++value_column)
            {
                recurrent_row[value_column] *= decay;
            }
        }
        float* head_output =
            recurrent_output
            + static_cast<size_t>(value_head)
                  * plan.value_head_dimension;
        for (uint32_t value_column = 0;
             value_column < plan.value_head_dimension;
             ++value_column)
        {
            float memory_value = 0.0f;
            for (uint32_t key_column = 0;
                 key_column < plan.head_dimension;
                 ++key_column)
            {
                memory_value +=
                    recurrent[static_cast<size_t>(key_column)
                                  * plan.value_head_dimension
                              + value_column]
                    * key[key_column] * key_inverse_norm;
            }
            const float delta =
                (value[value_column] - memory_value) * beta;
            for (uint32_t key_column = 0;
                 key_column < plan.head_dimension;
                 ++key_column)
            {
                recurrent[static_cast<size_t>(key_column)
                              * plan.value_head_dimension
                          + value_column] +=
                    key[key_column] * key_inverse_norm * delta;
            }
            float output_value = 0.0f;
            for (uint32_t key_column = 0;
                 key_column < plan.head_dimension;
                 ++key_column)
            {
                output_value +=
                    recurrent[static_cast<size_t>(key_column)
                                  * plan.value_head_dimension
                              + value_column]
                    * query[key_column] * query_inverse_norm;
            }
            head_output[value_column] = output_value * query_scale;
        }

        float square_sum = 0.0f;
        for (uint32_t value_column = 0;
             value_column < plan.value_head_dimension;
             ++value_column)
        {
            square_sum +=
                head_output[value_column] * head_output[value_column];
        }
        const float inverse_rms =
            1.0f
            / std::sqrt(
                square_sum
                    / static_cast<float>(plan.value_head_dimension)
                + norm_epsilon);
        const float* head_gate =
            z + static_cast<size_t>(value_head)
                    * plan.value_head_dimension;
        for (uint32_t value_column = 0;
             value_column < plan.value_head_dimension;
             ++value_column)
        {
            head_output[value_column] *=
                inverse_rms
                * gated_delta_tensor_value(norm_weight, value_column)
                * (head_gate[value_column]
                   * gated_delta_sigmoid(head_gate[value_column]));
        }
    }
}

void execute_gated_delta_net_into(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    CpuLayerCache& cache,
    CpuGatedDeltaExecutionScratch& scratch,
    const CpuBatch& hidden,
    CpuBatch& output)
{
    configure_gated_delta_cache(cache, plan);
    rms_norm_batch_into(
        hidden,
        weights.at(plan.pre_attention_norm_weight),
        norm_epsilon,
        scratch.normalized,
        plan.norm_weight_offset);
    const uint32_t key_size = plan.kv_head_count * plan.head_dimension;
    const uint32_t value_size =
        plan.head_count * plan.value_head_dimension;
    const uint32_t convolution_size = key_size * 2 + value_size;
    const uint32_t fused_columns =
        convolution_size + value_size + plan.head_count * 2;
    bool fused_input =
        (plan.fused_delta_input_bfloat16_operator
         && plan.fused_delta_input_bfloat16_operator->forward(
             scratch.normalized,
             scratch.fused_input))
        || (plan.fused_delta_input_operator
            && plan.fused_delta_input_operator->forward(
                scratch.normalized,
                scratch.fused_input));
    if (fused_input)
    {
        fused_input =
            scratch.fused_input.rows() == hidden.rows()
            && scratch.fused_input.columns() == fused_columns;
    }
    if (!fused_input)
    {
        linear_batch_into(
            weights.at(plan.delta_qkv_weight),
            scratch.normalized,
            scratch.qkv);
        linear_batch_into(
            weights.at(plan.delta_z_weight),
            scratch.normalized,
            scratch.z);
        linear_batch_into(
            weights.at(plan.delta_beta_weight),
            scratch.normalized,
            scratch.beta);
        linear_batch_into(
            weights.at(plan.delta_alpha_weight),
            scratch.normalized,
            scratch.alpha);
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
            convolution_size);
        execute_gated_delta_recurrence_row(
            weights,
            plan,
            norm_epsilon,
            cache,
            qkv,
            z,
            beta,
            alpha,
            scratch.recurrent_output.row(token_index));
        record_gated_delta_cache_transaction_row(cache);
    }
    linear_batch_into(weights.at(plan.output_weight), scratch.recurrent_output, scratch.projected);
    output = hidden;
    add_batch_inplace(output, scratch.projected);
    cache.gated_delta_token_count += hidden.rows();
}

} // namespace moe
} // namespace ncnn
