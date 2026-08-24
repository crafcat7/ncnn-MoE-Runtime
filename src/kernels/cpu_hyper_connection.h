#ifndef NCNN_MOE_CPU_HYPER_CONNECTION_H
#define NCNN_MOE_CPU_HYPER_CONNECTION_H

#include "cpu_batch.h"

#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>
#include <vector>

namespace ncnn {
namespace moe {

struct CpuHyperConnectionMix
{
    CpuBatch reduced;
    std::vector<float> post;
    std::vector<float> combine;
};

[[nodiscard]] Result<CpuHyperConnectionMix> hyper_connection_pre(const CpuBatch& input, const TensorData& function, const TensorData& scale,
                                                                 const TensorData& base, uint32_t multiplier, uint32_t sinkhorn_iterations,
                                                                 float norm_epsilon, float hyper_epsilon, uint64_t optimization_flags);

[[nodiscard]] Result<CpuBatch> hyper_connection_post(const CpuBatch& branch, const CpuBatch& residual, const CpuHyperConnectionMix& mix, uint32_t multiplier);

[[nodiscard]] Result<CpuBatch> hyper_connection_head(const CpuBatch& input, const TensorData& function, const TensorData& scale, const TensorData& base,
                                                     uint32_t multiplier, float norm_epsilon, float hyper_epsilon, uint64_t optimization_flags);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_HYPER_CONNECTION_H
