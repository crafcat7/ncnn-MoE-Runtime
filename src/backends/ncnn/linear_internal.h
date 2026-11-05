#ifndef NCNN_MOE_LINEAR_INTERNAL_H
#define NCNN_MOE_LINEAR_INTERNAL_H

#include "linear_core.h"

#include <cstdint>
#include <memory>
#include <span>

namespace ncnn {
class Option;
class Pipeline;
class VkCompute;
class VkMat;
} // namespace ncnn

namespace ncnn {
namespace moe {

#if NCNN_MOE_USE_NCNN && NCNN_MOE_WITH_VULKAN
[[nodiscard]] bool matvec_supported(const std::shared_ptr<VulkanContext>& context,
                                    uint32_t input_columns, uint32_t output_columns,
                                    bool bfloat16);
[[nodiscard]] bool create_matvec_pipeline(const std::shared_ptr<VulkanContext>& context,
                                          const ncnn::Option& option, bool bfloat16,
                                          uint32_t input_columns, uint32_t output_columns,
                                          std::shared_ptr<ncnn::Pipeline>& output);
[[nodiscard]] bool single_row_input(const ncnn::VkMat& input, uint32_t columns);
void record_matvec(const ncnn::Pipeline* pipeline, bool bfloat16,
                   uint32_t input_columns, uint32_t output_columns, bool has_bias,
                   const ncnn::VkMat& input, const ncnn::VkMat& weights,
                   const ncnn::VkMat& bias, ncnn::VkMat& output, ncnn::VkCompute& command);
#endif

#if NCNN_MOE_WITH_VULKAN
[[nodiscard]] ncnn::VkMat aligned_row_view(const ncnn::VkMat& source,
                                           size_t first_row, size_t rows,
                                           ncnn::VkCompute& command,
                                           const ncnn::Option& option,
                                           const std::shared_ptr<VulkanContext>& context,
                                           std::vector<ncnn::VkMat>& retained,
                                           size_t reserve_size = 0);
[[nodiscard]] bool record_row_copy(const ncnn::VkMat& source, ncnn::VkMat& destination,
                                   ncnn::VkCompute& command, const ncnn::Option& option);
[[nodiscard]] ncnn::VkMat bind_host_out(ncnn::VkMat& staging);

[[nodiscard]] bool create_gather_pipeline(const std::shared_ptr<VulkanContext>& context,
                                          const ncnn::Option& option,
                                          std::shared_ptr<ncnn::Pipeline>& destination);
[[nodiscard]] bool can_share_row(size_t available_rows, size_t request_count,
                                 std::span<const uint32_t> input_rows);
[[nodiscard]] bool create_route_agg_pipeline(const std::shared_ptr<VulkanContext>& context,
                                             const ncnn::Option& option,
                                             std::shared_ptr<ncnn::Pipeline>& destination);
[[nodiscard]] bool create_indexed_pipeline(const std::shared_ptr<VulkanContext>& context,
                                           const ncnn::Option& option,
                                           std::shared_ptr<ncnn::Pipeline>& destination);

inline constexpr uint32_t indexed_max_experts = 8;
#endif

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LINEAR_INTERNAL_H
