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
    float weight = 0.0f;
};

struct ExpertBatch
{
    uint32_t expert_id = 0;
    std::vector<ExpertRoute> routes;
};

enum ExpertDispatchOptionFlag : uint32_t
{
    ExpertDispatchNormalizeTopKWeights = 1u << 0
};

struct ExpertDispatchOptions
{
    uint32_t expert_count = 0;
    uint32_t top_k = 0;
    RouterNormalization normalization = RouterNormalization::SelectedExperts;
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
    // router_logits is a contiguous token_count x expert_count row-major
    // matrix. The result contains only active Experts, ordered by expert id,
    // with token order preserved inside each batch.
    [[nodiscard]] Result<ExpertDispatchPlan> dispatch(
        std::span<const float> router_logits,
        uint32_t token_count,
        const ExpertDispatchOptions& options) const;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXPERT_DISPATCHER_H
