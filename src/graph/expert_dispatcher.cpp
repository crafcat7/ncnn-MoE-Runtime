#include "ncnn/moe/expert_dispatcher.h"

#include <algorithm>
#include <cmath>

namespace ncnn {
namespace moe {

struct RouteCandidate
{
    uint32_t expert_id = 0;
    float score = 0.0f;
};

static bool route_precedes(
    const RouteCandidate& left,
    const RouteCandidate& right)
{
    return left.score > right.score
           || (left.score == right.score && left.expert_id < right.expert_id);
}

static void softmax(
    const float* logits,
    uint32_t size,
    std::vector<float>& probabilities)
{
    probabilities.resize(size);
    const float maximum = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (uint32_t index = 0; index < size; ++index) {
        probabilities[index] = std::exp(logits[index] - maximum);
        sum += probabilities[index];
    }
    for (float& probability : probabilities)
        probability /= sum;
}

static void select_topk_routes(
    const float* probabilities,
    uint32_t expert_count,
    uint32_t top_k,
    std::vector<RouteCandidate>& selected)
{
    selected.clear();
    for (uint32_t expert_id = 0; expert_id < expert_count; ++expert_id) {
        const RouteCandidate candidate{
            expert_id,
            probabilities[expert_id],
        };
        const auto insertion = std::lower_bound(
            selected.begin(),
            selected.end(),
            candidate,
            route_precedes);
        if (selected.size() == top_k && insertion == selected.end())
            continue;
        selected.insert(insertion, candidate);
        if (selected.size() > top_k)
            selected.pop_back();
    }
}

Result<ExpertDispatchPlan> ExpertDispatcher::dispatch(
    std::span<const float> router_logits,
    uint32_t token_count,
    const ExpertDispatchOptions& options) const
{
    if (token_count == 0)
        return Error{ErrorCode::InvalidArgument, "expert dispatch requires at least one token"};
    if (options.expert_count == 0
        || options.top_k == 0
        || options.top_k > options.expert_count) {
        return Error{
            ErrorCode::InvalidArgument,
            "expert dispatch requires 0 < top_k <= expert_count"};
    }
    if (router_logits.size() / options.expert_count != token_count
        || router_logits.size() % options.expert_count != 0) {
        return Error{
            ErrorCode::InvalidArgument,
            "router logits shape does not match token and expert counts"};
    }
    for (float logit : router_logits) {
        if (!std::isfinite(logit))
            return Error{ErrorCode::InvalidArgument, "router logits must be finite"};
    }

    ExpertDispatchPlan result;
    result.assignment_count = static_cast<size_t>(token_count) * options.top_k;
    result.batches.resize(options.expert_count);
    for (uint32_t expert_id = 0; expert_id < options.expert_count; ++expert_id)
        result.batches[expert_id].expert_id = expert_id;

    const bool renormalize
        = has_flag(options.flags, ExpertDispatchNormalizeTopKWeights)
          || options.normalization == RouterNormalization::SelectedExperts;
    std::vector<float> probabilities;
    probabilities.reserve(options.expert_count);
    std::vector<RouteCandidate> selected;
    selected.reserve(options.top_k);
    for (uint32_t token_index = 0; token_index < token_count; ++token_index) {
        const float* logits
            = router_logits.data() + static_cast<size_t>(token_index) * options.expert_count;
        softmax(logits, options.expert_count, probabilities);
        select_topk_routes(
            probabilities.data(),
            options.expert_count,
            options.top_k,
            selected);
        float selected_sum = 0.0f;
        for (const RouteCandidate& candidate : selected)
            selected_sum += candidate.score;
        for (const RouteCandidate& candidate : selected) {
            const float weight = renormalize
                                     ? candidate.score / selected_sum
                                     : candidate.score;
            result.batches[candidate.expert_id].routes.push_back({
                token_index,
                weight,
            });
        }
    }

    result.batches.erase(
        std::remove_if(
            result.batches.begin(),
            result.batches.end(),
            [](const ExpertBatch& batch) {
                return batch.routes.empty();
            }),
        result.batches.end());
    return result;
}

} // namespace moe
} // namespace ncnn
