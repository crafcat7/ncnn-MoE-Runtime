#ifndef NCNN_MOE_EXPERT_DISPATCHER_H
#define NCNN_MOE_EXPERT_DISPATCHER_H

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

struct ExpertRoute
{
    uint32_t token_index = 0;
    uint32_t rank = 0;
    float weight = 0.0f;
};

struct ExpertBatch
{
    uint32_t expert_id = 0;
    std::vector<ExpertRoute> routes;
};

#define NCNN_MOE_DISPATCH_NORMALIZE_TOPK_BIT 0

enum ExpertDispatchOptionFlag : uint32_t
{
    ExpertDispatchNormalizeTopKWeights = UINT32_C(1) << NCNN_MOE_DISPATCH_NORMALIZE_TOPK_BIT
};

struct ExpertDispatchOptions
{
    uint32_t expert_count = 0;
    uint32_t top_k = 0;
    RouterScoreFunction score_function = RouterScoreFunction::Softmax;
    RouterNormalization normalization = RouterNormalization::SelectedExperts;
    float routed_scaling_factor = 1.0f;
    std::span<const float> selection_bias;
    std::span<const uint32_t> explicit_expert_ids;
    uint32_t flags = ExpertDispatchNormalizeTopKWeights;
};

struct ExpertDispatchPlan
{
    std::vector<ExpertBatch> batches;
    size_t assignment_count = 0;
};

class ExpertDispatcher
{
public:
    // Input is token-major; output is stable and ordered by Expert id.
    [[nodiscard]] Result<ExpertDispatchPlan> dispatch(std::span<const float> router_logits, uint32_t token_count, const ExpertDispatchOptions& options) const;

    // Reuses caller storage across decode steps.
    [[nodiscard]] Result<void> dispatch_into(std::span<const float> router_logits, uint32_t token_count, const ExpertDispatchOptions& options, ExpertDispatchPlan& result) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_DISPATCHER_H
