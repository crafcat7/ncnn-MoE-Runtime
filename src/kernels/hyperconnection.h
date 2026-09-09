#ifndef NCNN_MOE_HYPERCONNECTION_H
#define NCNN_MOE_HYPERCONNECTION_H

#include "activationbuffer.h"

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

struct HyperConnectionMix
{
    ActivationBuffer reduced;
    std::vector<float> post;
    std::vector<float> combine;
};

struct HyperConnectionScratch
{
    // Sequential Float32 workspace; inputs and outputs must not alias
    // normalized, projection, or auxiliary.
    HyperConnectionMix transient_mix;
    ActivationBuffer normalized;
    ActivationBuffer projection;
    ActivationBuffer auxiliary;
    std::vector<float> sums;
};

void hyper_connection_expand(ActivationBuffer& hidden, uint32_t multiplier, ActivationBuffer& scratch);

// Fill caller-owned result and retain scratch capacity for the next call.
[[nodiscard]] Result<void> hyper_connection_pre(const ActivationBuffer& input, const TensorData& function, const TensorData& scale,
                                                const TensorData& base, uint32_t multiplier, uint32_t sinkhorn_iterations,
                                                float norm_epsilon, float hyper_epsilon, HyperConnectionMix& result,
                                                HyperConnectionScratch& scratch, uint64_t optimization_flags);

// Both inputs stay live for the mix; output must not alias either input.
[[nodiscard]] Result<void> hyper_connection_post(const ActivationBuffer& branch, const ActivationBuffer& residual, const HyperConnectionMix& mix,
                                                 uint32_t multiplier, ActivationBuffer& output);

// Fill caller-owned Float32 output; output must not alias input.
[[nodiscard]] Result<void> hyper_connection_head(const ActivationBuffer& input, const TensorData& function, const TensorData& scale, const TensorData& base,
                                                 uint32_t multiplier, float norm_epsilon, float hyper_epsilon, ActivationBuffer& output,
                                                 HyperConnectionScratch& scratch, uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_HYPERCONNECTION_H
