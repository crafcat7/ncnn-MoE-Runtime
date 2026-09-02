#ifndef NCNN_MOE_SYSTEM_MEMORY_H
#define NCNN_MOE_SYSTEM_MEMORY_H

#include <cstdint>

namespace ncnn {
namespace moe {

[[nodiscard]] uint64_t physical_memory_size() noexcept;
[[nodiscard]] uint64_t available_memory_size() noexcept;

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_SYSTEM_MEMORY_H
