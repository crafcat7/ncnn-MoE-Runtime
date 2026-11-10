#ifndef NCNN_MOE_LINEAR_INTERNAL_H
#define NCNN_MOE_LINEAR_INTERNAL_H

#include "linear_core.h"

#include <array>
#include <cstdint>
#include <layer.h>
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

// Tiled prefill projection: BF16 weights x FP32 activations, FP16 shared tiles.
// Two geometries exist because narrow output tiles would otherwise idle most of
// their last wave; both are specializations of one kernel body.
struct TileGemm
{
    std::shared_ptr<ncnn::Pipeline> pipeline;
    uint32_t bm = 0; // tile rows, in tokens
    uint32_t bn = 0; // tile columns, in outputs
    uint32_t wg = 0; // workgroup size
};

// Builds ncnn's cooperative-matrix Gemm for plain BF16 rows: it packs the weights
// itself and runs FP16 storage with FP32 accumulate, which is the default prefill
// path. Returns false when the device or the shape cannot use it, in which case
// callers keep the tiled shader.
[[nodiscard]] bool create_coop_gemm(const std::shared_ptr<VulkanContext>& context,
                                    const ncnn::Option& base,
                                    uint32_t input_columns,
                                    uint32_t output_columns,
                                    std::span<const uint16_t> weights,
                                    std::shared_ptr<ncnn::Layer>& destination,
                                    ncnn::Option& destination_option);

struct TileShape
{
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t stride = 0;
    uint32_t offset = 0;
    uint32_t groups = 1;
};

// Fills both geometries; true when at least one compiled.
[[nodiscard]] bool create_tile_gemm(const std::shared_ptr<VulkanContext>& context,
                                    const ncnn::Option& option,
                                    TileGemm (&pipes)[2]);

// out[rows][stride] = act[rows][k * groups] x w[n][k] + bias, with the optional
// row/column scales folding an RMSNorm into the activation tile load. Returns
// false, leaving the command untouched, when the shape is unsupported.
[[nodiscard]] bool record_tile_gemm(const TileGemm (&pipes)[2],
                                    const ncnn::VkMat& in,
                                    const ncnn::VkMat& w,
                                    const ncnn::VkMat& bias,
                                    const ncnn::VkMat& row_scale,
                                    const ncnn::VkMat& col_scale,
                                    ncnn::VkMat& out,
                                    const TileShape& shape,
                                    ncnn::VkCompute& cmd);

#endif

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LINEAR_INTERNAL_H
