#ifndef NCNN_MOE_LINEAR_IMPL_H
#define NCNN_MOE_LINEAR_IMPL_H

#include "linear_bf16.h"
#include "linear_float8.h"
#include "linear_internal.h"
#include "linear_mxfp4.h"
#include "linear_qnk.h"
#include "rmsnorm_vulkan.h"
#include "vulkancontext.h"

#if NCNN_MOE_USE_NCNN
#include <allocator.h>
#include <mat.h>
#include <option.h>
#endif

namespace ncnn {
namespace moe {

class Bfloat16Linear_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> single_row_pipeline;
    std::shared_ptr<ncnn::Pipeline> cooperative_pipeline;
    // Tiled FP16-staged FP32-accumulate projection geometries, pipe[0] wide.
    TileGemm tile_gemm[2];
    std::shared_ptr<ncnn::Pipeline> row_rms_scale_pipeline;
    mutable std::shared_ptr<ncnn::Pipeline> swiglu_down_pipeline;
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

    [[nodiscard]] bool record_projection(const ncnn::VkMat& input,
                                         ncnn::VkMat& output,
                                         uint32_t token_count,
                                         ncnn::VkCompute& command,
                                         std::vector<ncnn::VkMat>& bindings,
                                         std::vector<ncnn::vk_constant_type>& constants) const;
    // Tiled prefill projection over the packed BF16 weights; the row and column
    // scales fold an RMSNorm into the activation tile load.
    [[nodiscard]] bool record_tile(const ncnn::VkMat& input,
                                   ncnn::VkMat& output,
                                   uint32_t rows,
                                   const ncnn::VkMat& row_scale,
                                   const ncnn::VkMat& column_scale,
                                   ncnn::VkCompute& command) const;
#endif
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
};

class Bfloat16Expert_vulkan::Implementation
{
public:
    std::shared_ptr<Bfloat16Linear_vulkan> gate_up;
    std::shared_ptr<Bfloat16Linear_vulkan> down;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    ncnn::Option option;
#endif
    uint32_t input_columns = 0;
    uint32_t intermediate_columns = 0;
    uint32_t output_columns = 0;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::Silu;
};

class Float8Linear_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
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

    void record_projection(const ncnn::VkMat& input,
                           const ncnn::VkMat& output,
                           uint32_t token_count,
                           ncnn::VkCompute& command,
                           uint32_t output_stride = 0,
                           uint32_t output_offset = 0) const;
#endif
    uint32_t matrix_input_columns = 0;
    uint32_t logical_input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t output_columns_per_group = 0;
    uint32_t block_count = 0;
    uint32_t input_group_count = 1;
    uint32_t scale_row_size = 128;
    float rms_norm_epsilon = 0.0f;
    float input_rms_norm_epsilon = 0.0f;
};

class Mxfp4Linear_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    // Pack the read-only MXFP4 projection inputs into one device buffer.
    ncnn::VkMat storage;
    ncnn::VkMat packed;
    ncnn::VkMat scales;
    ncnn::VkMat bias;
    ncnn::Option option;
    bool indexed_storage = false;
    uint32_t indexed_scales_word_offset = 0;
    uint32_t indexed_bias_word_offset = 0;
#endif
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    uint32_t block_count = 0;
};

class Mxfp4Expert_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<ncnn::Pipeline> gate_up_pipeline;
    std::shared_ptr<ncnn::Pipeline> row_tile_pipeline;
#endif
    std::shared_ptr<Mxfp4Linear_vulkan> gate_up;
    std::shared_ptr<Mxfp4Linear_vulkan> down;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
};

class QnkLinear_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    std::shared_ptr<ncnn::Pipeline> pipeline;
    std::shared_ptr<ncnn::Pipeline> prefill_pipeline;
    // Tiled FP16-staged FP32-accumulate projection geometries, pipe[0] wide.
    TileGemm tile_gemm[2];
    ncnn::VkMat storage;
    ncnn::VkMat quantized;
    ncnn::VkMat bias;
    ncnn::Option option;
    uint64_t optimization_flags = OptimizationDefaultFlags;
#endif
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
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<QnkLinear_vulkan> gate_up;
    std::shared_ptr<QnkLinear_vulkan> up;
    std::shared_ptr<QnkLinear_vulkan> down;
    std::shared_ptr<QnkLinear_vulkan> router_gate;
    std::shared_ptr<VulkanContext> vulkan_context;
    std::shared_ptr<ncnn::Pipeline> swiglu_pipeline;
    ncnn::Option option;
#endif
    uint32_t intermediate_columns = 0;
    uint32_t output_columns = 0;
    float activation_limit = 0.0f;
    ExpertActivation activation = ExpertActivation::GptOssSwiGlu;
    bool interleave_gate_up_rows = true;
    uint32_t device_index = 0;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_LINEAR_IMPL_H
