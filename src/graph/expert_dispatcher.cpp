#include "ncnn/moe/expert_dispatcher.h"

#include "kernels/cpu_fast_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace ncnn {
namespace moe {

struct RouteCandidate
{
    uint32_t expert_id = 0;
    float score = 0.0f;
    uint32_t rank = 0;
};

static bool route_precedes(const RouteCandidate& left, const RouteCandidate& right)
{
    return left.score > right.score || (left.score == right.score && left.expert_id < right.expert_id);
}

static float stable_softplus(float value)
{
    if (value > 20.0f)
        return value;
    if (value < -20.0f)
        return float_approximate_exp(value);
    return std::log1p(float_approximate_exp(value));
}

static float score_router_logit(float logit, RouterScoreFunction score_function)
{
    if (score_function == RouterScoreFunction::Sigmoid)
        return 1.0f / (1.0f + float_approximate_exp(-logit));
    return std::sqrt(stable_softplus(logit));
}

static void router_scores(const float* logits, uint32_t size, RouterScoreFunction score_function, std::vector<float>& scores)
{
    scores.resize(size);
    if (score_function == RouterScoreFunction::Softmax)
    {
        const float maximum = *std::max_element(logits, logits + size);
        float sum = 0.0f;
        for (uint32_t index = 0; index < size; ++index)
        {
            scores[index] = float_approximate_exp(logits[index] - maximum);
            sum += scores[index];
        }
        for (float& score : scores)
            score /= sum;
        return;
    }

    for (uint32_t index = 0; index < size; ++index)
        scores[index] = score_router_logit(logits[index], score_function);
}

static void select_topk_routes(const float* scores, std::span<const float> selection_bias, uint32_t expert_count, uint32_t top_k, std::vector<RouteCandidate>& selected)
{
    selected.clear();
    for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
    {
        const RouteCandidate candidate{
            expert_id,
            scores[expert_id] + (selection_bias.empty() ? 0.0f : selection_bias[expert_id]),
        };
        const auto insertion = std::lower_bound(selected.begin(), selected.end(), candidate, route_precedes);
        if (selected.size() == top_k && insertion == selected.end())
            continue;
        selected.insert(insertion, candidate);
        if (selected.size() > top_k)
            selected.pop_back();
    }
}

Result<ExpertDispatchPlan> ExpertDispatcher::dispatch(std::span<const float> router_logits, uint32_t token_count, const ExpertDispatchOptions& options) const
{
    if (token_count == 0)
        return Error{ErrorCode::InvalidArgument, "expert dispatch requires at least one token"};
    if (options.expert_count == 0 || options.top_k == 0 || options.top_k > options.expert_count)
    {
        return Error{ErrorCode::InvalidArgument, "expert dispatch requires 0 < top_k <= expert_count"};
    }
    if (router_logits.size() / options.expert_count != token_count || router_logits.size() % options.expert_count != 0)
    {
        return Error{ErrorCode::InvalidArgument, "router logits shape does not match token and expert counts"};
    }
    if (!options.selection_bias.empty() && options.selection_bias.size() != options.expert_count)
        return Error{ErrorCode::InvalidArgument, "router selection bias must match expert_count"};
    if (!options.explicit_expert_ids.empty() && options.explicit_expert_ids.size() != static_cast<size_t>(token_count) * options.top_k)
        return Error{ErrorCode::InvalidArgument, "explicit expert ids must match token_count * top_k"};
    if (!std::isfinite(options.routed_scaling_factor) || options.routed_scaling_factor <= 0.0f)
        return Error{ErrorCode::InvalidArgument, "routed scaling factor must be finite and positive"};
    for (float logit : router_logits)
    {
        if (!std::isfinite(logit))
            return Error{ErrorCode::InvalidArgument, "router logits must be finite"};
    }

    ExpertDispatchPlan result;
    result.assignment_count = static_cast<size_t>(token_count) * options.top_k;
    result.batches.resize(options.expert_count);
    for (uint32_t expert_id = 0; expert_id < options.expert_count; ++expert_id)
        result.batches[expert_id].expert_id = expert_id;

    const bool renormalize = has_flag(options.flags, ExpertDispatchNormalizeTopKWeights) || options.normalization == RouterNormalization::SelectedExperts;
    std::vector<float> scores;
    scores.reserve(options.expert_count);
    std::vector<RouteCandidate> selected;
    selected.reserve(options.top_k);
    for (uint32_t token_index = 0; token_index < token_count; ++token_index)
    {
        const float* logits = router_logits.data() + static_cast<size_t>(token_index) * options.expert_count;
        router_scores(logits, options.expert_count, options.score_function, scores);
        if (options.explicit_expert_ids.empty())
        {
            select_topk_routes(scores.data(), options.selection_bias, options.expert_count, options.top_k, selected);
        }
        else
        {
            selected.clear();
            for (uint32_t route_index = 0; route_index < options.top_k; ++route_index)
            {
                const uint32_t expert_id = options.explicit_expert_ids[static_cast<size_t>(token_index) * options.top_k + route_index];
                if (expert_id >= options.expert_count)
                    return Error{ErrorCode::InvalidArgument, "explicit expert id is out of range"};
                selected.push_back({expert_id, scores[expert_id]});
            }
        }
        float selected_sum = 0.0f;
        for (const RouteCandidate& candidate : selected)
            selected_sum += scores[candidate.expert_id];
        if (renormalize && selected_sum <= 0.0f)
            return Error{ErrorCode::InvalidModel, "selected router weights have a non-positive sum"};
        for (uint32_t rank = 0; rank < selected.size(); ++rank)
        {
            const RouteCandidate& candidate = selected[rank];
            const float score = scores[candidate.expert_id];
            const float weight = (renormalize ? score / selected_sum : score) * options.routed_scaling_factor;
            result.batches[candidate.expert_id].routes.push_back({token_index, rank, weight});
        }
    }

    size_t batch_count = 0;
    for (size_t index = 0; index < result.batches.size(); ++index)
    {
        if (result.batches[index].routes.empty())
            continue;
        if (batch_count != index)
            result.batches[batch_count] = std::move(result.batches[index]);
        ++batch_count;
    }
    result.batches.resize(batch_count);
    return result;
}

Result<void> ExpertDispatcher::dispatch_into(std::span<const float> router_logits, uint32_t token_count, const ExpertDispatchOptions& options, ExpertDispatchPlan& result) const
{
    static constexpr uint32_t stack_top_k = 16;
    if (token_count != 1
        || options.top_k > stack_top_k)
    {
        auto dispatched = dispatch(router_logits, token_count, options);
        if (!dispatched)
            return dispatched.error();
        result = std::move(dispatched).value();
        return {};
    }
    if (options.expert_count == 0 || options.top_k == 0 || options.top_k > options.expert_count)
    {
        return Error{ErrorCode::InvalidArgument, "expert dispatch requires 0 < top_k <= expert_count"};
    }
    if (router_logits.size() != options.expert_count)
    {
        return Error{ErrorCode::InvalidArgument, "router logits shape does not match token and expert counts"};
    }
    if (!options.selection_bias.empty() && options.selection_bias.size() != options.expert_count)
        return Error{ErrorCode::InvalidArgument, "router selection bias must match expert_count"};
    if (!options.explicit_expert_ids.empty() && options.explicit_expert_ids.size() != options.top_k)
        return Error{ErrorCode::InvalidArgument, "explicit expert ids must match token_count * top_k"};
    if (!std::isfinite(options.routed_scaling_factor) || options.routed_scaling_factor <= 0.0f)
        return Error{ErrorCode::InvalidArgument, "routed scaling factor must be finite and positive"};

    std::array<RouteCandidate, stack_top_k> selected{};
    uint32_t selected_count = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    for (float logit : router_logits)
    {
        if (!std::isfinite(logit))
        {
            return Error{ErrorCode::InvalidArgument, "router logits must be finite"};
        }
        maximum = std::max(maximum, logit);
    }

    float softmax_denominator = 1.0f;
    if (options.score_function == RouterScoreFunction::Softmax)
    {
        softmax_denominator = 0.0f;
        for (float logit : router_logits)
            softmax_denominator += float_approximate_exp(logit - maximum);
    }
    const auto score_for_expert = [&](uint32_t expert_id) {
        const float logit = router_logits[expert_id];
        return options.score_function == RouterScoreFunction::Softmax
                   ? float_approximate_exp(logit - maximum) / softmax_denominator
                   : score_router_logit(logit, options.score_function);
    };

    if (!options.explicit_expert_ids.empty())
    {
        for (uint32_t route_index = 0; route_index < options.top_k; ++route_index)
        {
            const uint32_t expert_id = options.explicit_expert_ids[route_index];
            if (expert_id >= options.expert_count)
                return Error{ErrorCode::InvalidArgument, "explicit expert id is out of range"};
            for (uint32_t previous = 0; previous < route_index; ++previous)
            {
                if (options.explicit_expert_ids[previous] == expert_id)
                {
                    auto dispatched = dispatch(router_logits, token_count, options);
                    if (!dispatched)
                        return dispatched.error();
                    result = std::move(dispatched).value();
                    return {};
                }
            }
            selected[selected_count++] = {
                expert_id,
                score_for_expert(expert_id),
            };
        }
    }
    else
    {
        for (uint32_t expert_id = 0; expert_id < options.expert_count; ++expert_id)
        {
            const float score = score_for_expert(expert_id);
            const RouteCandidate candidate{
                expert_id,
                score + (options.selection_bias.empty() ? 0.0f : options.selection_bias[expert_id]),
            };
            uint32_t insertion = 0;
            while (insertion < selected_count && route_precedes(selected[insertion], candidate))
            {
                ++insertion;
            }
            if (insertion >= options.top_k)
                continue;
            const uint32_t last = std::min(selected_count, options.top_k - 1);
            for (uint32_t index = last; index > insertion; --index)
            {
                selected[index] = selected[index - 1];
            }
            selected[insertion] = candidate;
            selected_count = std::min(options.top_k, selected_count + 1);
        }
    }

    const bool renormalize = has_flag(options.flags, ExpertDispatchNormalizeTopKWeights) || options.normalization == RouterNormalization::SelectedExperts;
    float denominator = 0.0f;
    if (renormalize)
    {
        for (uint32_t index = 0; index < selected_count; ++index)
            denominator += score_for_expert(selected[index].expert_id);
        if (denominator <= 0.0f)
            return Error{ErrorCode::InvalidModel, "selected router weights have a non-positive sum"};
    }
    for (uint32_t index = 0; index < selected_count; ++index)
        selected[index].rank = index;
    for (uint32_t index = 1; index < selected_count; ++index)
    {
        const RouteCandidate candidate = selected[index];
        uint32_t insertion = index;
        while (insertion > 0 && candidate.expert_id < selected[insertion - 1].expert_id)
        {
            selected[insertion] = selected[insertion - 1];
            --insertion;
        }
        selected[insertion] = candidate;
    }

    result.assignment_count = selected_count;
    result.batches.resize(selected_count);
    for (uint32_t index = 0; index < selected_count; ++index)
    {
        ExpertBatch& batch = result.batches[index];
        batch.expert_id = selected[index].expert_id;
        batch.routes.resize(1);
        batch.routes.front() = {
            0,
            selected[index].rank,
            (renormalize ? score_for_expert(selected[index].expert_id) / denominator : score_for_expert(selected[index].expert_id))
                * options.routed_scaling_factor,
        };
    }
    return {};
}

} // namespace moe
} // namespace ncnn
