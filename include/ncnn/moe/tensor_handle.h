#ifndef NCNN_MOE_TENSOR_HANDLE_H
#define NCNN_MOE_TENSOR_HANDLE_H

#include <cstdint>
#include <limits>

namespace ncnn {
namespace moe {

using TensorHandle = uint32_t;
inline constexpr TensorHandle invalid_tensor_handle = std::numeric_limits<TensorHandle>::max();

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_TENSOR_HANDLE_H
