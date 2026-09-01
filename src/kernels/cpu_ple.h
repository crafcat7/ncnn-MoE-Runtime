#ifndef NCNN_MOE_CPU_PLE_H
#define NCNN_MOE_CPU_PLE_H

#include "cpu_batch.h"

#include "ncnn/moe/compiled_layer_plan.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/weight_store.h"

#include <cstdint>
#include <span>

namespace ncnn {
namespace moe {

struct CpuLayerCache;

[[nodiscard]] Result<void> execute_ple_into(
    const WeightStore& weights,
    const PleBlockPlan& plan,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    std::span<const int32_t> input_ids,
    CpuLayerCache& cache,
    CpuBatch& hidden,
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_PLE_H
