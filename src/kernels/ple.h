#ifndef NCNN_MOE_PLE_H
#define NCNN_MOE_PLE_H

#include "activationbuffer.h"

#include "graph/layerplan.h"
#include "ncnn/moe/result.h"
#include "storage/weightstore.h"

#include <cstdint>
#include <span>

namespace ncnn {
namespace moe {

struct LayerCache;
struct AttentionScratch;

// Hidden must not alias the attention scratch buffers.
[[nodiscard]] Result<void> forward_ple(const WeightStore& weights,
                                       const PleBlockPlan& plan,
                                       uint32_t multiplier,
                                       uint32_t hidden_size,
                                       float norm_epsilon,
                                       float norm_weight_offset,
                                       std::span<const int32_t> input_ids,
                                       LayerCache& cache,
                                       AttentionScratch& scratch,
                                       ActivationBuffer& hidden,
                                       uint64_t flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_PLE_H
