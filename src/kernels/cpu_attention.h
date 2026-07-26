#ifndef NCNN_MOE_CPU_ATTENTION_H
#define NCNN_MOE_CPU_ATTENTION_H

#include "cpu_batch.h"
#include "engine/cpu_session_state.h"

#include "ncnn/moe/execution_plan.h"

#include <cstdint>

namespace ncnn {
namespace moe {

void execute_attention_block_into(const WeightTable& weights, const AttentionBlockPlan& plan, float norm_epsilon, DType kv_cache_dtype,
                                  uint64_t position_offset, CpuLayerCache& cache, CpuAttentionExecutionScratch& scratch, const CpuBatch& hidden,
                                  CpuBatch& output);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_ATTENTION_H
