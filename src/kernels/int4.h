#ifndef NCNN_MOE_INT4_H
#define NCNN_MOE_INT4_H

#include "activationbuffer.h"
#include "ncnn/moe/result.h"
#include "ncnn/moe/types.h"

#include <cstdint>

namespace ncnn {
namespace moe {

// Rows contain ceil(K / 2) bytes, low nibble first. Weights are consumed
// directly as (code - zero[row, group]) * scale[row, group].
[[nodiscard]] Result<void> validate_int4_tensor(const TensorData& tensor);
[[nodiscard]] Result<void> validate_input_column_indices(const TensorData& tensor);
[[nodiscard]] uint64_t int4_storage_bytes(const TensorData& tensor) noexcept;

// Decodes only the requested row into logical column order.
[[nodiscard]] bool int4_dequantize_row(const TensorData& matrix, uint32_t row, float* output);

// Weights must pass validate_int4_tensor before execution and remain immutable.
// Input columns are in stored weight order; forward_linear gathers logical inputs.
[[nodiscard]] bool forward_linear_int4(const TensorData& matrix,
                                       const ActivationBuffer& input,
                                       ActivationBuffer& output);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_INT4_H
