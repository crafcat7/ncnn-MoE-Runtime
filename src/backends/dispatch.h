#ifndef NCNN_MOE_BACKENDS_DISPATCH_H
#define NCNN_MOE_BACKENDS_DISPATCH_H

namespace ncnn {
namespace moe {

class ActivationBuffer;
enum class ExecutionBackend;
struct CompiledOperator;

[[nodiscard]] bool has_vulkan_linear_operator(const CompiledOperator* op) noexcept;
[[nodiscard]] bool has_vulkan_float8_or_linear_operator(const CompiledOperator* op) noexcept;

[[nodiscard]] bool dispatch_vulkan_linear(const CompiledOperator* op,
                                          const ActivationBuffer& input,
                                          ActivationBuffer& output,
                                          ExecutionBackend backend);

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_BACKENDS_DISPATCH_H
