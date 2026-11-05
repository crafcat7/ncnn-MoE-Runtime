#ifndef NCNN_MOE_GATEDRESIDUAL_H
#define NCNN_MOE_GATEDRESIDUAL_H

#include "hyperconnection.h"

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>

namespace ncnn {
namespace moe {

class CompiledOperatorTable;
class WeightStore;
struct GatedResidualPlan;
struct MoeModelDescriptor;

[[nodiscard]] Result<void> gr_pre(const ActivationBuffer& input,
                                  const WeightStore& weights,
                                  const CompiledOperatorTable& operators,
                                  const GatedResidualPlan& plan,
                                  const MoeModelDescriptor& descriptor,
                                  HyperConnectionMix& result,
                                  HyperConnectionScratch& scratch,
                                  uint64_t optimization_flags);

// Output may alias residual for the element-wise in-place combine, but not branch.
[[nodiscard]] Result<void> gr_post(const ActivationBuffer& branch,
                                   const ActivationBuffer& residual,
                                   const HyperConnectionMix& mix,
                                   uint32_t multiplier,
                                   ActivationBuffer& output);

[[nodiscard]] Result<void> gr_head(const ActivationBuffer& input,
                                   const WeightStore& weights,
                                   const CompiledOperatorTable& operators,
                                   const GatedResidualPlan& plan,
                                   const MoeModelDescriptor& descriptor,
                                   ActivationBuffer& output,
                                   HyperConnectionScratch& scratch,
                                   uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_GATEDRESIDUAL_H
