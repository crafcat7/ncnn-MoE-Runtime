#ifndef NCNN_MOE_GATEDRESIDUAL_H
#define NCNN_MOE_GATEDRESIDUAL_H

#include "hyperconnection.h"

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>

namespace ncnn {
namespace moe {

[[nodiscard]] Result<void> gated_residual_pre(
    const ActivationBuffer& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    const TensorData& inject_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    HyperConnectionMix& result,
    HyperConnectionScratch& scratch,
    uint64_t optimization_flags);

// Output may alias residual for the element-wise in-place combine, but not branch.
[[nodiscard]] Result<void> gated_residual_post(
    const ActivationBuffer& branch,
    const ActivationBuffer& residual,
    const HyperConnectionMix& mix,
    uint32_t multiplier,
    ActivationBuffer& output);

[[nodiscard]] Result<void> gated_residual_head(
    const ActivationBuffer& input,
    const TensorData& norm_weight,
    const TensorData& mix_down_weight,
    const TensorData& mix_up_weight,
    uint32_t multiplier,
    uint32_t hidden_size,
    float norm_epsilon,
    float norm_weight_offset,
    ActivationBuffer& output,
    HyperConnectionScratch& scratch,
    uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GATEDRESIDUAL_H
