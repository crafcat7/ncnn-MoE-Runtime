#include "cpu_executor.h"

#include "cpu_attention.h"
#include "cpu_batch.h"
#include "cpu_ops.h"
#include "cpu_session_state.h"

#include "ncnn/moe/session.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ncnn {
namespace moe {

struct RouteItem
{
    uint32_t token_index = 0;
    float weight = 0.0f;
};

static std::vector<float> softmax(const float* logits, uint32_t size)
{
    const float maximum = *std::max_element(logits, logits + size);
    std::vector<float> probabilities(size);
    float sum = 0.0f;
    for (uint32_t index = 0; index < size; ++index) {
        probabilities[index] = std::exp(logits[index] - maximum);
        sum += probabilities[index];
    }
    for (float& probability : probabilities)
        probability /= sum;
    return probabilities;
}

static float activate(float value, ExpertActivation activation, float limit)
{
    switch (activation) {
    case ExpertActivation::Relu:
        return std::max(0.0f, value);
    case ExpertActivation::Silu:
        return value / (1.0f + std::exp(-value));
    case ExpertActivation::Gelu:
        return 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
    case ExpertActivation::ClampedSilu:
    {
        const float clamped = limit > 0.0f ? std::clamp(value, -limit, limit) : value;
        return clamped / (1.0f + std::exp(-clamped));
    }
    case ExpertActivation::GptOssSwiGlu:
        return value;
    }
    return value;
}

static CpuBatch execute_expert_batch(const WeightTable& weights, const ExpertPlan& expert, const CpuBatch& input)
{
    if (expert.gate_up_weight != invalid_tensor_handle) {
        CpuBatch gate_up = expert.gate_up_bias == invalid_tensor_handle
                               ? linear_batch(weights.at(expert.gate_up_weight), input)
                               : linear_batch(weights.at(expert.gate_up_weight), weights.at(expert.gate_up_bias), input);
        CpuBatch activated(gate_up.rows(), gate_up.columns() / 2);
        for (size_t token_index = 0; token_index < gate_up.rows(); ++token_index) {
            const float* source = gate_up.row(token_index);
            float* destination = activated.row(token_index);
            for (uint32_t column = 0; column < activated.columns(); ++column) {
                const float gate = expert.activation_limit > 0.0f
                                       ? std::min(source[column * 2], expert.activation_limit)
                                       : source[column * 2];
                const float linear = expert.activation_limit > 0.0f
                                         ? std::clamp(source[column * 2 + 1], -expert.activation_limit, expert.activation_limit)
                                         : source[column * 2 + 1];
                const float silu = gate / (1.0f + std::exp(-1.702f * gate));
                destination[column] = silu * (linear + 1.0f);
            }
        }
        return expert.down_bias == invalid_tensor_handle
                   ? linear_batch(weights.at(expert.down_weight), activated)
                   : linear_batch(weights.at(expert.down_weight), weights.at(expert.down_bias), activated);
    }

    CpuBatch up = linear_batch(weights.at(expert.up_weight), input);
    if (expert.gated) {
        const CpuBatch gate = linear_batch(weights.at(expert.gate_weight), input);
        for (size_t token_index = 0; token_index < up.rows(); ++token_index) {
            float* up_row = up.row(token_index);
            const float* gate_row = gate.row(token_index);
            for (uint32_t column = 0; column < up.columns(); ++column)
                up_row[column] *= activate(gate_row[column], expert.activation, expert.activation_limit);
        }
    }
    else {
        for (size_t token_index = 0; token_index < up.rows(); ++token_index) {
            float* token = up.row(token_index);
            for (uint32_t column = 0; column < up.columns(); ++column)
                token[column] = activate(token[column], expert.activation, expert.activation_limit);
        }
    }
    return linear_batch(weights.at(expert.down_weight), up);
}

static CpuBatch gather_tokens(const CpuBatch& source, const std::vector<RouteItem>& routes)
{
    CpuBatch gathered(routes.size(), source.columns());
    for (size_t route_index = 0; route_index < routes.size(); ++route_index) {
        std::copy_n(
            source.row(routes[route_index].token_index),
            source.columns(),
            gathered.row(route_index));
    }
    return gathered;
}

Result<std::vector<std::vector<float> > > CpuExecutor::execute(
    const CompiledModel& model,
    std::span<const int32_t> input_ids,
    SessionStatistics& statistics,
    CpuSessionState& state,
    uint64_t position_offset) const
{
    for (int32_t token_id : input_ids) {
        if (token_id < 0 || static_cast<uint32_t>(token_id) >= model.descriptor.vocabulary_size)
            return Error{ErrorCode::InvalidArgument, "token id is outside the model vocabulary"};
    }

    const uint32_t hidden_size = model.descriptor.hidden_size;
    CpuBatch hidden = embedding_batch(model.weights.at(model.token_embedding), input_ids);

    if (statistics.expert_token_counts.size() < model.descriptor.expert_count)
        statistics.expert_token_counts.resize(model.descriptor.expert_count, 0);

    if (state.layers.size() != model.layers.size())
        state.layers.resize(model.layers.size());

    for (const CompiledLayerPlan& layer : model.layers) {
        if (layer.use_attention) {
            hidden = execute_attention_block(
                model.weights,
                layer.attention,
                model.descriptor.norm_epsilon,
                position_offset,
                state.layers[layer.layer_id],
                hidden);
        }

        const MoeBlockPlan& moe = layer.moe;
        const TensorData& norm_weight = model.weights.at(moe.pre_ffn_norm_weight);
        const TensorData& router_weight = model.weights.at(moe.router_weight);
        const CpuBatch normalized = rms_norm_batch(hidden, norm_weight, model.descriptor.norm_epsilon);
        CpuBatch router_logits = linear_batch(router_weight, normalized);

        if (moe.router_bias != invalid_tensor_handle) {
            const TensorData& bias = model.weights.at(moe.router_bias);
            add_bias_inplace(router_logits, bias);
        }

        std::vector<std::vector<RouteItem> > groups(moe.experts.size());
        for (uint32_t token_index = 0; token_index < router_logits.rows(); ++token_index) {
            const std::vector<float> probabilities = softmax(router_logits.row(token_index), router_logits.columns());
            std::vector<uint32_t> expert_ids(probabilities.size());
            std::iota(expert_ids.begin(), expert_ids.end(), 0u);
            std::stable_sort(expert_ids.begin(), expert_ids.end(), [&](uint32_t left, uint32_t right) {
                if (probabilities[left] == probabilities[right])
                    return left < right;
                return probabilities[left] > probabilities[right];
            });
            expert_ids.resize(moe.top_k);

            float selected_sum = 0.0f;
            for (uint32_t expert_id : expert_ids)
                selected_sum += probabilities[expert_id];
            const bool renormalize = moe.normalize_topk_weights
                                     || moe.normalization == RouterNormalization::SelectedExperts;

            for (uint32_t expert_id : expert_ids) {
                const float routing_weight = renormalize
                                                 ? probabilities[expert_id] / selected_sum
                                                 : probabilities[expert_id];
                groups[expert_id].push_back(RouteItem{token_index, routing_weight});
                ++statistics.expert_assignments;
                ++statistics.expert_token_counts[expert_id];
            }
        }

        CpuBatch moe_output(hidden.rows(), hidden_size);
        for (uint32_t expert_id = 0; expert_id < groups.size(); ++expert_id) {
            const std::vector<RouteItem>& group = groups[expert_id];
            if (group.empty())
                continue;

            const CpuBatch expert_input = gather_tokens(normalized, group);
            const CpuBatch expert_output = execute_expert_batch(
                model.weights, moe.experts[expert_id], expert_input);
            ++statistics.expert_batches;

            for (size_t batch_index = 0; batch_index < group.size(); ++batch_index) {
                const RouteItem& route = group[batch_index];
                float* destination = moe_output.row(route.token_index);
                const float* source = expert_output.row(batch_index);
                for (uint32_t column = 0; column < hidden_size; ++column)
                    destination[column] += route.weight * source[column];
            }
        }

        for (size_t token_index = 0; token_index < hidden.rows(); ++token_index) {
            float* hidden_row = hidden.row(token_index);
            const float* output_row = moe_output.row(token_index);
            for (uint32_t column = 0; column < hidden_size; ++column)
                hidden_row[column] += output_row[column];
        }
    }

    const TensorData& final_norm = model.weights.at(model.final_norm_weight);
    const TensorData& lm_head = model.weights.at(model.lm_head_weight);
    const CpuBatch normalized = rms_norm_batch(hidden, final_norm, model.descriptor.norm_epsilon);
    return batch_to_vectors(linear_batch(lm_head, normalized));
}

} // namespace moe
} // namespace ncnn
