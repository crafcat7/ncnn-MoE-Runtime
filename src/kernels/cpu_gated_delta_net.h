#ifndef NCNN_MOE_CPU_GATED_DELTA_NET_H
#define NCNN_MOE_CPU_GATED_DELTA_NET_H

#include "ncnn/moe/execution_plan.h"

namespace ncnn {
namespace moe {

class CpuBatch;
struct CpuGatedDeltaExecutionScratch;
struct CpuLayerCache;

void execute_gated_delta_net_into(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    CpuLayerCache& cache,
    CpuGatedDeltaExecutionScratch& scratch,
    const CpuBatch& hidden,
    CpuBatch& output);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_GATED_DELTA_NET_H
