#ifndef NCNN_MOE_CPU_ATTENTION_H
#define NCNN_MOE_CPU_ATTENTION_H

#include "cpu_batch.h"
#include "cpu_session_state.h"

#include "ncnn/moe/execution_plan.h"

#include <cstdint>

namespace ncnn {
namespace moe {

[[nodiscard]] CpuBatch execute_attention_block(
    const WeightTable& weights,
    const AttentionBlockPlan& plan,
    float norm_epsilon,
    DType kv_cache_dtype,
    uint64_t position_offset,
    CpuLayerCache& cache,
    const CpuBatch& hidden);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_ATTENTION_H
