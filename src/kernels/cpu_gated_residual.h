#ifndef NCNN_MOE_CPU_GATED_RESIDUAL_H
#define NCNN_MOE_CPU_GATED_RESIDUAL_H

#include "cpu_hyper_connection.h"

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>

namespace ncnn {
namespace moe {

[[nodiscard]] Result<CpuHyperConnectionMix> gated_residual_pre(
    const CpuBatch& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    const TensorData& inject_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    uint64_t optimization_flags);

[[nodiscard]] Result<CpuBatch> gated_residual_post(
    const CpuBatch& branch,
    const CpuBatch& residual,
    const CpuHyperConnectionMix& mix,
    uint32_t multiplier);

[[nodiscard]] Result<CpuBatch> gated_residual_head(
    const CpuBatch& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_GATED_RESIDUAL_H
