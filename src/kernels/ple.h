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

[[nodiscard]] Result<void> execute_ple_into(const WeightStore& weights,
                                            const PleBlockPlan& plan,
                                            uint32_t multiplier,
                                            uint32_t hidden_size,
                                            float norm_epsilon,
                                            float norm_weight_offset,
                                            std::span<const int32_t> input_ids,
                                            LayerCache& cache,
                                            ActivationBuffer& hidden,
                                            uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_PLE_H
