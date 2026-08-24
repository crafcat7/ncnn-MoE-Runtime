#ifndef NCNN_MOE_SYSTEM_MEMORY_H
#define NCNN_MOE_SYSTEM_MEMORY_H

#include <cstdint>

namespace ncnn {
namespace moe {

[[nodiscard]] uint64_t physical_memory_bytes() noexcept;
[[nodiscard]] uint64_t available_memory_bytes() noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SYSTEM_MEMORY_H
