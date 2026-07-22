#ifndef NCNN_MOE_CPU_OPS_H
#define NCNN_MOE_CPU_OPS_H

#include "cpu_batch.h"

#include "ncnn/moe/types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

[[nodiscard]] float bfloat16_to_float(uint16_t value) noexcept;
[[nodiscard]] uint16_t float_to_bfloat16(float value) noexcept;
[[nodiscard]] CpuBatch embedding_batch(const TensorData& embedding, std::span<const int32_t> input_ids);
[[nodiscard]] CpuBatch linear_batch(const TensorData& matrix, const CpuBatch& input);
[[nodiscard]] CpuBatch linear_batch(const TensorData& matrix, const TensorData& bias, const CpuBatch& input);
[[nodiscard]] CpuBatch rms_norm_batch(const CpuBatch& input, const TensorData& weight, float epsilon);
void add_bias_inplace(CpuBatch& destination, const TensorData& bias);
void add_batch_inplace(CpuBatch& destination, const CpuBatch& source);
[[nodiscard]] std::vector<std::vector<float> > batch_to_vectors(const CpuBatch& batch);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_CPU_OPS_H
