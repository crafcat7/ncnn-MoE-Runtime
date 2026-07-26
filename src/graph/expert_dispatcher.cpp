#include "ncnn/moe/expert_dispatcher.h"

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
};

static bool route_precedes(const RouteCandidate& left, const RouteCandidate& right)
{
    return left.score > right.score || (left.score == right.score && left.expert_id < right.expert_id);
}

static void softmax(const float* logits, uint32_t size, std::vector<float>& probabilities)
{
    probabilities.resize(size);
    const float maximum = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (uint32_t index = 0; index < size; ++index)
    {
        probabilities[index] = std::exp(logits[index] - maximum);
        sum += probabilities[index];
    }
    for (float& probability : probabilities)
        probability /= sum;
}

static void select_topk_routes(const float* probabilities, uint32_t expert_count, uint32_t top_k, std::vector<RouteCandidate>& selected)
{
    selected.clear();
    for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id)
    {
        const RouteCandidate candidate{
            expert_id,
            probabilities[expert_id],
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
    std::vector<float> probabilities;
    probabilities.reserve(options.expert_count);
    std::vector<RouteCandidate> selected;
    selected.reserve(options.top_k);
    for (uint32_t token_index = 0; token_index < token_count; ++token_index)
    {
        const float* logits = router_logits.data() + static_cast<size_t>(token_index) * options.expert_count;
        softmax(logits, options.expert_count, probabilities);
        select_topk_routes(probabilities.data(), options.expert_count, options.top_k, selected);
        float selected_sum = 0.0f;
        for (const RouteCandidate& candidate : selected)
            selected_sum += candidate.score;
        for (const RouteCandidate& candidate : selected)
        {
            const float weight = renormalize ? candidate.score / selected_sum : candidate.score;
            result.batches[candidate.expert_id].routes.push_back({token_index, weight});
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
    if (token_count != 1 || options.top_k > stack_top_k)
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

    std::array<RouteCandidate, stack_top_k> selected{};
    uint32_t selected_count = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t expert_id = 0; expert_id < options.expert_count; ++expert_id)
    {
        const float score = router_logits[expert_id];
        if (!std::isfinite(score))
        {
            return Error{ErrorCode::InvalidArgument, "router logits must be finite"};
        }
        maximum = std::max(maximum, score);
        const RouteCandidate candidate{
            expert_id,
            score,
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

    const bool renormalize = has_flag(options.flags, ExpertDispatchNormalizeTopKWeights) || options.normalization == RouterNormalization::SelectedExperts;
    float denominator = 0.0f;
    if (renormalize)
    {
        for (uint32_t index = 0; index < selected_count; ++index)
        {
            denominator += std::exp(selected[index].score - maximum);
        }
    }
    else
    {
        for (float score : router_logits)
            denominator += std::exp(score - maximum);
    }
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
            std::exp(selected[index].score - maximum) / denominator,
        };
    }
    return {};
}

} // namespace moe
} // namespace ncnn
