#ifndef NCNN_MOE_LINEAR_INTERNAL_H
#define NCNN_MOE_LINEAR_INTERNAL_H

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/layer/rmsnorm_vulkan.h"
#include "backends/ncnn/linear/bf16.h"
#include "backends/ncnn/linear/core.h"
#include "backends/ncnn/linear/float8.h"
#include "backends/ncnn/linear/mxfp4.h"
#include "backends/ncnn/linear/qnk.h"

#include <allocator.h>
#include <layer.h>
#include <mat.h>
#include <option.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ncnn {
namespace moe {

// Packed BF16 / Q8_0 / FP32 matrix-vector helpers.
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

[[nodiscard]] ncnn::VkMat aligned_row_view(const ncnn::VkMat& source,
                                           size_t first_row, size_t rows,
                                           ncnn::VkCompute& command,
                                           const ncnn::Option& option,
                                           const std::shared_ptr<VulkanContext>& context,
                                           std::vector<ncnn::VkMat>& retained,
                                           size_t reserve_size = 0);
[[nodiscard]] bool record_row_copy(const ncnn::VkMat& source, ncnn::VkMat& destination,
                                   ncnn::VkCompute& command, const ncnn::Option& option);

inline constexpr uint32_t indexed_max_experts = 8;

struct TileShape
{
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t stride = 0;
    uint32_t offset = 0;
    uint32_t groups = 1;
};

// Tiled prefill projection: BF16 weights x FP32 activations, FP16 shared tiles.
struct TileGemm
{
    std::shared_ptr<ncnn::Pipeline> pipeline;
    uint32_t bm = 0;
    uint32_t bn = 0;
    uint32_t wg = 0;
};

// Native NV cooperative_matrix2 BF16 GEMM geometries.
struct Cm2Gemm
{
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> cast_pipeline;
    uint32_t bm = 0;
    uint32_t bn = 0;
    uint32_t bk = 0;
    uint32_t wg = 0;
};

[[nodiscard]] bool create_cm2_gemm(const std::shared_ptr<VulkanContext>& context,
                                   const ncnn::Option& option,
                                   Cm2Gemm (&pipes)[4],
                                   bool bf16_out = false);
[[nodiscard]] bool record_cm2_gemm(const Cm2Gemm (&pipes)[4],
                                   const ncnn::VkMat& in,
                                   const ncnn::VkMat& w,
                                   const ncnn::VkMat& bias,
                                   bool has_bias,
                                   ncnn::VkMat& out,
                                   const TileShape& shape,
                                   VulkanContext* context,
                                   ncnn::VkCompute& cmd,
                                   std::vector<ncnn::VkMat>& bindings,
                                   std::vector<ncnn::vk_constant_type>& constants,
                                   const ncnn::VkMat* res = nullptr,
                                   bool* res_fused = nullptr,
                                   bool out_bf16 = false);

[[nodiscard]] bool create_qnk_swiglu_pipe(const std::shared_ptr<VulkanContext>& context,
                                          const ncnn::Option& option,
                                          std::shared_ptr<ncnn::Pipeline>& destination,
                                          bool bf16_out = false,
                                          bool bf16_in = false);

[[nodiscard]] bool create_coop_gemm(const std::shared_ptr<VulkanContext>& context,
                                    const ncnn::Option& base,
                                    uint32_t input_columns,
                                    uint32_t output_columns,
                                    std::span<const uint16_t> weights,
                                    std::span<const float> bias,
                                    std::shared_ptr<ncnn::Layer>& destination,
                                    ncnn::Option& destination_option);

[[nodiscard]] bool create_tile_gemm(const std::shared_ptr<VulkanContext>& context,
                                    const ncnn::Option& option,
                                    TileGemm (&pipes)[2]);
[[nodiscard]] bool record_tile_gemm(const TileGemm (&pipes)[2],
                                    const ncnn::VkMat& in,
                                    const ncnn::VkMat& w,
                                    const ncnn::VkMat& bias,
                                    const ncnn::VkMat& row_scale,
                                    const ncnn::VkMat& col_scale,
                                    ncnn::VkMat& out,
                                    const TileShape& shape,
                                    ncnn::VkCompute& cmd);

class Bfloat16Linear_vulkan::Implementation
{
public:
    // vulkan_context precedes pipelines so pipelines release before the context lock resets.
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> single_row_pipeline;
    std::shared_ptr<ncnn::Pipeline> single_row_bf16_pipeline;
    std::shared_ptr<ncnn::Pipeline> cooperative_pipeline;
    std::shared_ptr<ncnn::Layer> coop_gemm;
    ncnn::Option coop_opt = {};
    TileGemm tile_gemm[2];
    Cm2Gemm gemm_cm2[4];
    bool cm2_ok = false;
    bool has_nonzero_bias = false;
    std::shared_ptr<ncnn::Pipeline> row_rms_scale_pipeline;
    mutable std::shared_ptr<ncnn::Pipeline> swiglu_down_pipeline;
    mutable std::shared_ptr<ncnn::Pipeline> swiglu_pipe;
    std::shared_ptr<ncnn::Pipeline> rms_norm_projection_pipeline;
    ncnn::VkMat packed;
    ncnn::VkMat bias;
    ncnn::VkMat rms_norm_weight;
    std::shared_ptr<RmsNorm_vulkan> input_rms_norm;
    std::shared_ptr<RmsNorm_vulkan> output_rms_norm;
    ncnn::Option option;
    float rms_norm_epsilon = 0.0f;
    uint32_t rms_norm_output_subgroups = 0;
    uint32_t cooperative_tile_m = 0;
    uint32_t cooperative_tile_n = 0;
    uint32_t cooperative_tile_k = 0;
    uint32_t cooperative_subgroup_size = 0;
    uint64_t optimization_flags = OptimizationDefaultFlags;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;

    [[nodiscard]] bool record_projection(const ncnn::VkMat& input,
                                         ncnn::VkMat& output,
                                         uint32_t token_count,
                                         ncnn::VkCompute& command,
                                         std::vector<ncnn::VkMat>& bindings,
                                         std::vector<ncnn::vk_constant_type>& constants) const;
    [[nodiscard]] bool record_tile(const ncnn::VkMat& input,
                                   ncnn::VkMat& output,
                                   uint32_t rows,
                                   const ncnn::VkMat& row_scale,
                                   const ncnn::VkMat& column_scale,
                                   ncnn::VkCompute& command) const;
};

class Bfloat16Expert_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> vulkan_context;
    std::shared_ptr<Bfloat16Linear_vulkan> gate_up;
    std::shared_ptr<Bfloat16Linear_vulkan> down;
    ncnn::Option option;
    uint32_t input_columns = 0;
    uint32_t intermediate_columns = 0;
    uint32_t output_columns = 0;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::Silu;
};

class Float8Linear_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> quantize_pipeline;
    std::shared_ptr<ncnn::Pipeline> swiglu_quantize_pipeline;
    std::shared_ptr<ncnn::Pipeline> rms_norm_quantize_pipeline;
    ncnn::VkMat packed;
    ncnn::VkMat scales;
    ncnn::VkMat bias;
    ncnn::VkMat rms_norm_weight;
    ncnn::VkMat input_rms_norm_weight;
    std::shared_ptr<RmsNorm_vulkan> rms_norm;
    std::shared_ptr<RmsNorm_vulkan> input_rms_norm;
    ncnn::Option option;
    uint64_t optimization_flags = OptimizationDefaultFlags;
    uint32_t matrix_input_columns = 0;
    uint32_t logical_input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t output_columns_per_group = 0;
    uint32_t block_count = 0;
    uint32_t input_group_count = 1;
    uint32_t scale_row_size = 128;
    float rms_norm_epsilon = 0.0f;
    float input_rms_norm_epsilon = 0.0f;

    void record_projection(const ncnn::VkMat& input,
                           const ncnn::VkMat& output,
                           uint32_t token_count,
                           ncnn::VkCompute& command,
                           uint32_t output_stride = 0,
                           uint32_t output_offset = 0) const;
};

class Mxfp4Linear_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    // Packs the read-only MXFP4 projection inputs into one device buffer.
    ncnn::VkMat storage;
    ncnn::VkMat packed;
    ncnn::VkMat scales;
    ncnn::VkMat bias;
    ncnn::Option option;
    bool indexed_storage = false;
    uint32_t indexed_scales_word_offset = 0;
    uint32_t indexed_bias_word_offset = 0;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
};

class Mxfp4Expert_vulkan::Implementation
{
public:
    std::shared_ptr<Mxfp4Linear_vulkan> gate_up;
    std::shared_ptr<Mxfp4Linear_vulkan> down;
    std::shared_ptr<ncnn::Pipeline> gate_up_pipeline;
    std::shared_ptr<ncnn::Pipeline> row_tile_pipeline;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
};

class QnkLinear_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> prefill_pipeline;
    std::shared_ptr<ncnn::Pipeline> decode_q8_pipeline;
    std::shared_ptr<ncnn::Pipeline> decode_bf16_pipeline;
    TileGemm tile_gemm[2];
    Cm2Gemm gemm_cm2[4];
    Cm2Gemm gemm_cm2_bf16[4];
    bool cm2_ok = false;
    bool cm2_bf16_ok = false;
    bool has_nonzero_bias = false;
    std::shared_ptr<ncnn::Layer> coop_gemm;
    ncnn::Option coop_opt = {};
    ncnn::VkMat storage;
    ncnn::VkMat quantized;
    ncnn::VkMat bias;
    ncnn::Option option;
    uint64_t optimization_flags = OptimizationDefaultFlags;
    DType dtype = DType::Q4K;
    uint32_t input_columns = 0;
    uint32_t input_group_count = 1;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
    uint32_t block_size = 0;
    uint32_t group_size = 0;
    uint32_t group_count = 0;
    uint32_t scales_offset = 0;
    uint32_t zeros_offset = 0;
    uint32_t groups_offset = 0;
    uint32_t columns_offset = 0;
    uint64_t device_size = 0;
};

class QnkExpert_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> vulkan_context;
    std::shared_ptr<QnkLinear_vulkan> gate_up;
    std::shared_ptr<QnkLinear_vulkan> up;
    std::shared_ptr<QnkLinear_vulkan> down;
    std::shared_ptr<QnkLinear_vulkan> router_gate;
    std::shared_ptr<ncnn::Pipeline> swiglu_pipe[3]; // 0: f32, 1: bf16_out, 2: bf16_io
    ncnn::Option option;
    uint32_t intermediate_columns = 0;
    uint32_t output_columns = 0;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
    bool interleave_gate_up_rows = true;
    uint32_t device_index = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LINEAR_INTERNAL_H
