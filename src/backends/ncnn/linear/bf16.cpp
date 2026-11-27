#include "backends/ncnn/core/context.h"
#include "backends/ncnn/core/memory.h"
#include "backends/ncnn/core/tensor.h"
#include "backends/ncnn/core/transfer.h"
#include "backends/ncnn/layer/rmsnorm_vulkan.h"
#include "backends/ncnn/linear/bf16.h"
#include "backends/ncnn/linear/internal.h"
#include "kernels/ops.h"
#include "kernels/vulkan/bfloat16_cooperative_projection.comp.hex.h"
#include "kernels/vulkan/bfloat16_gemm_cm2.comp.hex.h"
#include "kernels/vulkan/bfloat16_projection.comp.hex.h"
#include "kernels/vulkan/bfloat16_rms_norm_projection.comp.hex.h"
#include "kernels/vulkan/bfloat16_row_rms.comp.hex.h"
#include "kernels/vulkan/bfloat16_swiglu_down.comp.hex.h"
#include "kernels/vulkan/bfloat16_tile_32x64.comp.hex.h"
#include "kernels/vulkan/bfloat16_tile_64x64.comp.hex.h"
#include "kernels/vulkan/fp32_to_bf16.comp.hex.h"

#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <layer.h>
#include <layer/gemm.h>
#include <layer_type.h>
#include <mat.h>
#include <paramdict.h>
#include <pipeline.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

// --- Pipeline & upload helpers -----------------------------------------------

static bool create_bf16_pipeline(const std::shared_ptr<VulkanContext>& context,
                                 const ncnn::Option& option,
                                 const char* src, int len, int local_size_x,
                                 std::shared_ptr<ncnn::Pipeline>& destination)
{
    const uint64_t variant = (option.use_subgroup_ops ? 1u : 0u) * 3
                             + activation_variant(option);
    ncnn::Option compile_opt = option;
    compile_opt.use_fp16_packed = false;
    compile_opt.use_fp16_arithmetic = false;
    compile_opt.use_bf16_packed = false;
    return create_pipeline(context, src, len, compile_opt, variant, variant,
                           local_size_x, destination, {}, 32);
}

static bool create_bfloat16_cooperative_projection_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                            const ncnn::Option& option,
                                                            uint32_t input_columns,
                                                            uint32_t output_columns,
                                                            std::shared_ptr<ncnn::Pipeline>& destination,
                                                            uint32_t& tile_m,
                                                            uint32_t& tile_n,
                                                            uint32_t& tile_k,
                                                            uint32_t& subgroup_size)
{
    ncnn::VulkanDevice* device = context->device();
    // The cooperative shader uses the BF16 storage macro family for its
    // matrix tiles. Activation storage must use the scalar/subgroup path until
    // activation and weight storage types are independently parameterized.
    if (activation_variant(option) != 0)
        return false;
    if (!device->info.support_VK_KHR_cooperative_matrix()
        || !device->info.support_bf16_cooperative_matrix()
        || !device->info.support_bf16_storage())
    {
        return false;
    }

    int selected_m = 0;
    int selected_n = 0;
    int selected_k = 0;
    int selected_subgroup_size = 0;
    device->info.get_optimal_cooperative_matrix_mnk(16,
                                                    static_cast<int>(output_columns),
                                                    static_cast<int>(input_columns),
                                                    VK_COMPONENT_TYPE_BFLOAT16_KHR,
                                                    VK_COMPONENT_TYPE_FLOAT32_KHR,
                                                    VK_SCOPE_SUBGROUP_KHR,
                                                    selected_m,
                                                    selected_n,
                                                    selected_k,
                                                    selected_subgroup_size);
    if (selected_m <= 0 || selected_n <= 0 || selected_k <= 0
        || selected_subgroup_size <= 0
        || selected_m % 4 != 0 || selected_n % 4 != 0 || selected_k % 4 != 0)
    {
        return false;
    }

    const uint64_t shared_size = (static_cast<uint64_t>(selected_m) * selected_k
                                  + static_cast<uint64_t>(selected_k) * selected_n)
                                     * sizeof(uint16_t)
                                 + static_cast<uint64_t>(selected_m) * selected_n
                                       * sizeof(float);
    if (shared_size > device->info.max_shared_memory_size()
        || selected_m > std::numeric_limits<uint16_t>::max()
        || selected_n > std::numeric_limits<uint16_t>::max()
        || selected_k > std::numeric_limits<uint16_t>::max()
        || selected_subgroup_size > std::numeric_limits<uint16_t>::max())
    {
        return false;
    }

    ncnn::Option compile_option = option;
    compile_option.vulkan_device_index = device->info.device_index();
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_storage = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    compile_option.use_bf16_storage = true;

    const uint64_t pipeline_key = static_cast<uint64_t>(selected_m)
                                  | static_cast<uint64_t>(selected_n) << 16
                                  | static_cast<uint64_t>(selected_k) << 32
                                  | static_cast<uint64_t>(selected_subgroup_size) << 48;
    std::vector<ncnn::vk_specialization_type> specializations(3);
    specializations[0].u32 = static_cast<uint32_t>(selected_m);
    specializations[1].u32 = static_cast<uint32_t>(selected_n);
    specializations[2].u32 = static_cast<uint32_t>(selected_k);
    if (!create_pipeline(context,
                         bfloat16_cooperative_projection_shader,
                         static_cast<int>(sizeof(bfloat16_cooperative_projection_shader) - 1),
                         compile_option, 0, pipeline_key,
                         selected_subgroup_size, destination,
                         specializations, selected_subgroup_size))
    {
        return false;
    }

    tile_m = static_cast<uint32_t>(selected_m);
    tile_n = static_cast<uint32_t>(selected_n);
    tile_k = static_cast<uint32_t>(selected_k);
    subgroup_size = static_cast<uint32_t>(selected_subgroup_size);
    return true;
}

static bool create_bfloat16_swiglu_down_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                 const ncnn::Option& option,
                                                 std::shared_ptr<ncnn::Pipeline>& destination)
{
    return create_bf16_pipeline(context, option, bfloat16_swiglu_down_shader,
                                static_cast<int>(sizeof(bfloat16_swiglu_down_shader) - 1),
                                32, destination);
}

static bool supports_gemm_cm2_shape(const ncnn::GpuInfo& info, uint32_t wg,
                                    uint32_t bm, uint32_t bn, uint32_t bk) noexcept
{
    if (!info.support_VK_KHR_cooperative_matrix() || !info.support_bf16_storage()
        || !info.support_VK_KHR_shader_bfloat16() || !info.support_bf16_cooperative_matrix()
        || !info.support_VK_NV_cooperative_matrix2())
    {
        return false;
    }

    const auto& features = info.queryCooperativeMatrix2FeaturesNV();
    const auto& properties = info.queryCooperativeMatrix2PropertiesNV();
    if (!features.cooperativeMatrixWorkgroupScope
        || !features.cooperativeMatrixFlexibleDimensions
        || !features.cooperativeMatrixConversions
        || !features.cooperativeMatrixPerElementOperations
        || !features.cooperativeMatrixTensorAddressing
        || properties.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize < wg
        || properties.cooperativeMatrixWorkgroupScopeReservedSharedMemory > info.max_shared_memory_size()
        || properties.cooperativeMatrixFlexibleDimensionsMaxDimension < std::max({bm, bn, bk})
        || info.max_workgroup_invocations() < wg
        || info.max_workgroup_size_x() < wg
        || (info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_BASIC_BIT) == 0)
    {
        return false;
    }

    for (const auto& property : info.queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV())
    {
        if (property.scope == VK_SCOPE_WORKGROUP_KHR
            && property.workgroupInvocations == wg
            && property.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR
            && property.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR
            && property.CType == VK_COMPONENT_TYPE_FLOAT32_KHR
            && property.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
            && property.MGranularity != 0 && property.NGranularity != 0 && property.KGranularity != 0
            && bm % property.MGranularity == 0 && bn % property.NGranularity == 0
            && bk % property.KGranularity == 0)
        {
            return true;
        }
    }
    return false;
}

static bool prepare_bfloat16_upload(std::span<const uint16_t> source,
                                    ncnn::Mat& destination)
{
    if (source.empty()
        || source.size() > static_cast<size_t>(std::numeric_limits<int>::max()) * 2)
    {
        return false;
    }

    const size_t word_count = (source.size() + 1) / 2;
    if (word_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;

    destination.create(static_cast<int>(word_count), sizeof(uint32_t));
    if (destination.empty())
        return false;

    uint32_t* words = static_cast<uint32_t*>(destination.data);
    for (size_t index = 0; index < word_count; ++index)
    {
        const size_t first = index * 2;
        const uint32_t low = source[first];
        const uint32_t high = first + 1 < source.size() ? source[first + 1] : 0;
        words[index] = low | high << 16;
    }
    return true;
}

static void record_swiglu_down(const ncnn::Pipeline* pipeline,
                               const ncnn::VkMat& fused,
                               const Bfloat16Linear_vulkan::Implementation& down,
                               const ncnn::VkMat& output,
                               uint32_t intermediate_columns,
                               uint32_t fused_columns,
                               uint32_t rows,
                               ExpertActivation activation,
                               float activation_limit,
                               bool apply_router_gate,
                               ncnn::VkCompute& command,
                               std::vector<ncnn::VkMat>& bindings,
                               std::vector<ncnn::vk_constant_type>& constants)
{
    bindings = {fused, down.packed, down.bias, output};
    constants.resize(7);
    constants[0].u32 = intermediate_columns;
    constants[1].u32 = fused_columns;
    constants[2].u32 = down.output_columns;
    constants[3].u32 = rows;
    constants[4].u32 = apply_router_gate ? 1u : 0u;
    constants[5].f = activation == ExpertActivation::DeepSeekSwiGlu ? activation_limit : 0.0f;
    // Gated-down shader activation code: 0 = SiLU, 1 = GELU-tanh.
    constants[6].u32 = activation == ExpertActivation::GeluTanh ? 1u : 0u;

    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(down.output_columns * 32);
    dispatcher.h = static_cast<int>(rows);
    dispatcher.c = 1;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
}

// --- Shared CM2, cooperative matrix & tile GEMM ------------------------------

bool create_cm2_gemm(const std::shared_ptr<VulkanContext>& context,
                     const ncnn::Option& option,
                     Cm2Gemm (&pipes)[4],
                     bool bf16_out)
{
    if (!context)
        return false;
    ncnn::VulkanDevice* device = context->device();
    if (device->info.subgroup_size() != 32)
        return false;

    struct Geometry
    {
        uint32_t bm;
        uint32_t bn;
        uint32_t bk;
        uint32_t wg;
    };
    const Geometry geometries[4] = {
        {128, 128, 64, 256},
        {64, 128, 64, 128},
        {64, 64, 64, 128},
        {32, 64, 64, 128},
    };

    ncnn::Option cast_opt = option;
    cast_opt.use_fp16_packed = false;
    cast_opt.use_fp16_storage = false;
    cast_opt.use_fp16_arithmetic = false;
    cast_opt.use_bf16_packed = false;
    cast_opt.use_bf16_storage = false;
    std::shared_ptr<ncnn::Pipeline> cast_pipeline;

    ncnn::Option compile_option = cast_opt;
    compile_option.use_bf16_storage = true;
    static const std::string bf16_out_shader = prefix_shader(bfloat16_gemm_cm2_shader,
                                                             static_cast<int>(sizeof(bfloat16_gemm_cm2_shader) - 1),
                                                             "#define NCNN_GEMM_CM2_BF16_OUT\n");
    const char* shader_source = bf16_out ? bf16_out_shader.c_str() : bfloat16_gemm_cm2_shader;
    const int shader_length = bf16_out ? static_cast<int>(bf16_out_shader.size())
                                       : static_cast<int>(sizeof(bfloat16_gemm_cm2_shader) - 1);
    const uint64_t source_variant = bf16_out ? UINT64_C(0x1000000000000000) : UINT64_C(0);

    bool any = false;
    for (size_t index = 0; index < 4; ++index)
    {
        Geometry item = geometries[index];
        if (!supports_gemm_cm2_shape(device->info, item.wg, item.bm, item.bn, item.bk))
        {
            if (supports_gemm_cm2_shape(device->info, item.wg, item.bm, item.bn, 32u))
                item.bk = 32u;
            else
                continue;
        }
        if (!cast_pipeline && !create_pipeline(context, fp32_to_bf16_shader, static_cast<int>(sizeof(fp32_to_bf16_shader) - 1), cast_opt, 0, 0, 256, cast_pipeline))
        {
            return false;
        }

        const uint64_t variant = static_cast<uint64_t>(item.bm)
                                 | (static_cast<uint64_t>(item.bn) << 16)
                                 | (static_cast<uint64_t>(item.bk) << 32)
                                 | (static_cast<uint64_t>(item.wg) << 48)
                                 | source_variant;
        std::vector<ncnn::vk_specialization_type> specs(3);
        specs[0].u32 = item.bm;
        specs[1].u32 = item.bn;
        specs[2].u32 = item.bk;
        if (!create_pipeline(context, shader_source, shader_length, compile_option,
                             0, variant, static_cast<int>(item.wg), pipes[index].pipeline, specs, 32))
        {
            continue;
        }
        pipes[index].cast_pipeline = cast_pipeline;
        pipes[index].bm = item.bm;
        pipes[index].bn = item.bn;
        pipes[index].bk = item.bk;
        pipes[index].wg = item.wg;
        any = true;
    }
    return any;
}

bool record_cm2_gemm(const Cm2Gemm (&pipes)[4],
                     const ncnn::VkMat& in,
                     const ncnn::VkMat& w,
                     const ncnn::VkMat& bias,
                     bool has_bias,
                     ncnn::VkMat& out,
                     const TileShape& shape,
                     VulkanContext* context,
                     ncnn::VkCompute& command,
                     std::vector<ncnn::VkMat>& bindings,
                     std::vector<ncnn::vk_constant_type>& constants,
                     const ncnn::VkMat* res,
                     bool* res_fused,
                     bool out_bf16)
{
    const uint32_t stride = shape.stride != 0 ? shape.stride : shape.n;
    const bool input_bf16 = in.elemsize == sizeof(uint16_t);
    const bool input_f32 = in.elemsize == sizeof(float);
    if (shape.rows < 8 || shape.k == 0 || shape.n == 0
        || (shape.k & 1u) != 0 || shape.groups != 1
        || stride < static_cast<uint64_t>(shape.offset) + shape.n
        || in.empty() || in.elempack != 1 || (!input_bf16 && !input_f32)
        || out.empty() || out.elempack != 1
        || w.empty() || bias.empty()
        || w.buffer_capacity() < static_cast<uint64_t>(shape.k) * shape.n * sizeof(uint16_t)
        || bias.buffer_capacity() < static_cast<uint64_t>(shape.n) * sizeof(float)
        || out.buffer_capacity() < static_cast<uint64_t>(stride) * shape.rows * out.elemsize)
    {
        return false;
    }

    const Cm2Gemm* selected = nullptr;
    if (shape.rows < 48 && pipes[3].pipeline)
    {
        selected = &pipes[3];
    }
    else
    {
        if (shape.rows >= 128 && pipes[0].pipeline)
        {
            const uint64_t bm0 = (shape.rows + pipes[0].bm - 1) / pipes[0].bm;
            const uint64_t bn0 = (shape.n + pipes[0].bn - 1) / pipes[0].bn;
            if (bm0 * bn0 >= 210)
                selected = &pipes[0];
        }
        if (!selected && pipes[1].pipeline)
        {
            const uint64_t bm1 = (shape.rows + pipes[1].bm - 1) / pipes[1].bm;
            const uint64_t bn1 = (shape.n + pipes[1].bn - 1) / pipes[1].bn;
            if (bm1 * bn1 >= 140)
                selected = &pipes[1];
        }
        if (!selected && pipes[2].pipeline)
            selected = &pipes[2];
        if (!selected && pipes[1].pipeline)
            selected = &pipes[1];
        if (!selected && pipes[0].pipeline)
            selected = &pipes[0];
        if (!selected && pipes[3].pipeline)
            selected = &pipes[3];
    }
    if (!selected || !selected->pipeline)
        return false;

    const uint64_t blocks_m = (shape.rows + selected->bm - 1) / selected->bm;
    const uint64_t blocks_n = (shape.n + selected->bn - 1) / selected->bn;
    if (blocks_m * selected->wg > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || blocks_n > 65535u)
    {
        return false;
    }

    ncnn::VkMat mat_a_bf16 = in;
    if (input_f32)
    {
        if (!selected->cast_pipeline || !context)
            return false;
        const uint64_t pair_count = (static_cast<uint64_t>(shape.rows) * shape.k) >> 1;
        if (pair_count == 0 || pair_count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
            return false;
        ncnn::VkMat converted = context->cm2_scratch(static_cast<int>(shape.k),
                                                     static_cast<int>(shape.rows),
                                                     sizeof(uint16_t));
        if (converted.empty())
            return false;

        bindings.resize(2);
        bindings[0] = in;
        bindings[1] = converted;
        constants.resize(1);
        constants[0].u32 = static_cast<uint32_t>(pair_count);
        ncnn::VkMat cast_dispatcher;
        cast_dispatcher.w = static_cast<int>(pair_count);
        cast_dispatcher.h = 1;
        cast_dispatcher.c = 1;
        command.record_pipeline_readonly(selected->cast_pipeline.get(), bindings, {1, 0},
                                         constants, cast_dispatcher);
        mat_a_bf16 = std::move(converted);
    }

    const bool has_res = !out_bf16 && res != nullptr && !res->empty()
                         && res->dims == 2 && res->elempack == 1
                         && res->elemsize == sizeof(float)
                         && res->w == static_cast<int>(stride)
                         && res->h == static_cast<int>(shape.rows);
    bindings.resize(5);
    bindings[0] = mat_a_bf16;
    bindings[1] = w;
    bindings[2] = bias;
    bindings[3] = out;
    bindings[4] = has_res ? *res : bias;
    constants.resize(6);
    constants[0].u32 = shape.rows;
    constants[1].u32 = shape.n;
    constants[2].u32 = shape.k;
    constants[3].u32 = stride;
    constants[4].u32 = shape.offset;
    constants[5].u32 = (has_bias ? 1u : 0u) | (has_res ? 2u : 0u);

    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(blocks_m * selected->wg);
    dispatcher.h = static_cast<int>(blocks_n);
    dispatcher.c = 1;
    command.record_pipeline_readonly(selected->pipeline.get(), bindings,
                                     {1, 0, 0, 0, has_res ? static_cast<unsigned char>(1) : static_cast<unsigned char>(0)},
                                     constants, dispatcher);
    if (res_fused && has_res)
        *res_fused = true;
    return true;
}

bool create_tile_gemm(const std::shared_ptr<VulkanContext>& context,
                      const ncnn::Option& option,
                      TileGemm (&pipes)[2])
{
    // One kernel body, two compile-time geometries: a narrow output tile would
    // otherwise idle most of the last wave.
    struct Geometry
    {
        const char* source;
        int length;
        uint32_t bm;
        uint32_t bn;
        uint32_t wg;
    };
    const Geometry geometries[2] = {
        {bfloat16_tile_64x64_shader, static_cast<int>(sizeof(bfloat16_tile_64x64_shader) - 1), 64, 64, 256},
        {bfloat16_tile_32x64_shader, static_cast<int>(sizeof(bfloat16_tile_32x64_shader) - 1), 32, 64, 128},
    };

    bool any = false;
    for (size_t index = 0; index < 2; ++index)
    {
        const Geometry& item = geometries[index];
        if (!create_bf16_pipeline(context, option, item.source, item.length,
                                  static_cast<int>(item.wg), pipes[index].pipeline))
        {
            continue;
        }
        pipes[index].bm = item.bm;
        pipes[index].bn = item.bn;
        pipes[index].wg = item.wg;
        any = true;
    }
    return any;
}

bool record_tile_gemm(const TileGemm (&pipes)[2],
                      const ncnn::VkMat& in,
                      const ncnn::VkMat& w,
                      const ncnn::VkMat& bias,
                      const ncnn::VkMat& row_scale,
                      const ncnn::VkMat& col_scale,
                      ncnn::VkMat& out,
                      const TileShape& shape,
                      ncnn::VkCompute& command)
{
    constexpr uint32_t tile_k = 16;
    const uint32_t stride = shape.stride != 0 ? shape.stride : shape.n;
    const uint32_t group_columns = shape.groups > 1 ? shape.n / shape.groups : shape.n;
    if (shape.rows < 8 || shape.k == 0 || shape.n == 0
        || shape.k % tile_k != 0 || shape.n % 4 != 0 || shape.groups == 0
        || static_cast<uint64_t>(group_columns) * shape.groups != shape.n
        || stride < static_cast<uint64_t>(shape.offset) + shape.n
        || (!row_scale.empty() && row_scale.elempack != 1)
        || (!col_scale.empty() && col_scale.elempack != 1)
        || in.empty() || in.elempack != 1 || in.elemsize != sizeof(float)
        // vec4 activation loads: the binding must be 16 B aligned and the rows contiguous.
        || (in.offset & 15u) != 0 || in.cstep != static_cast<size_t>(in.w) * in.h
        || out.empty() || out.elempack != 1 || out.elemsize != sizeof(float)
        || w.empty() || bias.empty()
        || w.buffer_capacity() < static_cast<uint64_t>(shape.k) * shape.n * sizeof(uint16_t)
        || bias.buffer_capacity() < static_cast<uint64_t>(shape.n) * sizeof(float)
        || out.buffer_capacity() < static_cast<uint64_t>(stride) * shape.rows * sizeof(float))
    {
        return false;
    }

    // Take the first geometry whose grid still fills the device. 160 workgroups
    // is four waves on the 40-core class of device this runtime targets; the
    // 32x64 geometry doubles the count for narrow output tiles, which otherwise
    // idle most of their last wave.
    constexpr uint64_t minimum_workgroups = 160;
    const TileGemm* selected = nullptr;
    const TileGemm* fallback = nullptr;
    for (const TileGemm& candidate : pipes)
    {
        if (!candidate.pipeline || candidate.bm == 0 || candidate.bn == 0)
            continue;
        // A workgroup derives its activation group from the tile origin, so a
        // grouped operator must split on tile boundaries.
        if (shape.groups > 1 && group_columns % candidate.bn != 0)
            continue;
        const uint64_t cols = (shape.n + candidate.bn - 1) / candidate.bn;
        const uint64_t rows = (shape.rows + candidate.bm - 1) / candidate.bm;
        const uint64_t tiles = cols * rows;
        if (tiles * candidate.wg > static_cast<uint64_t>(std::numeric_limits<int>::max()))
            continue;
        fallback = &candidate;
        if (tiles >= minimum_workgroups)
        {
            selected = &candidate;
            break;
        }
    }
    if (!selected)
        selected = fallback;
    if (!selected)
        return false;

    const uint64_t cols = (shape.n + selected->bn - 1) / selected->bn;
    const uint64_t rows = (shape.rows + selected->bm - 1) / selected->bm;
    const std::vector<ncnn::VkMat> bindings = {in, w, bias, row_scale, col_scale, out};
    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].u32 = shape.k;
    constants[1].u32 = shape.n;
    constants[2].u32 = shape.rows;
    constants[3].u32 = row_scale.empty() ? 0u : 1u;
    constants[4].u32 = col_scale.empty() ? 0u : 1u;
    constants[5].u32 = stride;
    constants[6].u32 = shape.offset;
    constants[7].u32 = shape.groups;
    constants[8].u32 = shape.groups > 1 ? group_columns : 0u;

    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(cols * selected->wg);
    dispatcher.h = static_cast<int>(rows);
    dispatcher.c = 1;
    command.record_pipeline_readonly(selected->pipeline.get(),
                                     bindings,
                                     {1, 0, 0, 1, 1, 0},
                                     constants,
                                     dispatcher);
    return true;
}

bool create_coop_gemm(const std::shared_ptr<VulkanContext>& context,
                      const ncnn::Option& base,
                      uint32_t input_columns,
                      uint32_t output_columns,
                      std::span<const uint16_t> weights,
                      std::span<const float> bias,
                      std::shared_ptr<ncnn::Layer>& destination,
                      ncnn::Option& destination_option)
{
    if (!context || context->graph_arena_sealed())
        return false;
    ncnn::VulkanDevice* device = context->device();
    if (!device->info.support_cooperative_matrix()
        || weights.size() != static_cast<size_t>(input_columns) * output_columns
        || (!bias.empty() && bias.size() != output_columns))
    {
        return false;
    }

    // A zero bias keeps the previous constant-free Gemm; a real bias rides along
    // as the broadcast C operand so the cooperative path cannot silently drop it.
    bool has_bias = false;
    for (float value : bias)
    {
        if (value != 0.0f)
        {
            has_bias = true;
            break;
        }
    }

    ncnn::Layer* raw = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
    if (!raw)
        return false;

    ncnn::Option option = make_vk_opt(*context);
    option.use_packing_layout = true;
    option.use_fp16_storage = true;
    option.use_cooperative_matrix = true;
    option.num_threads = base.num_threads;
    option.blob_vkallocator = base.blob_vkallocator;
    option.workspace_vkallocator = base.workspace_vkallocator;
    option.staging_vkallocator = base.staging_vkallocator;

    std::shared_ptr<ncnn::Layer> layer = own_vk_layer(context, raw, option);
    raw->vkdev = device;

    ncnn::ParamDict pd;
    pd.set(0, 1.f);                  // alpha
    pd.set(1, has_bias ? 1.f : 0.f); // beta: keep the broadcast bias
    pd.set(2, 0);                    // transA
    pd.set(3, 1);                    // transB: B rows are outputs
    pd.set(4, 0);                    // constantA
    pd.set(5, 1);                    // constantB
    pd.set(6, has_bias ? 1 : 0);     // constantC
    pd.set(8, static_cast<int>(output_columns));
    pd.set(9, static_cast<int>(input_columns));
    pd.set(10, 4); // constantC is one value per output column

    ncnn::Mat host_weights(static_cast<int>(input_columns), static_cast<int>(output_columns));
    if (raw->load_param(pd) != 0 || host_weights.empty())
        return false;
    for (uint32_t output = 0; output < output_columns; ++output)
    {
        float* row = host_weights.row<float>(static_cast<int>(output));
        const uint16_t* source = weights.data() + static_cast<size_t>(output) * input_columns;
        for (uint32_t column = 0; column < input_columns; ++column)
            row[column] = bfloat16_to_float(source[column]);
    }
    static_cast<ncnn::Gemm*>(raw)->B_data = host_weights;

    if (has_bias)
    {
        ncnn::Mat host_bias(static_cast<int>(output_columns), 1, sizeof(float));
        if (host_bias.empty())
            return false;
        std::memcpy(host_bias.data, bias.data(), bias.size_bytes());
        static_cast<ncnn::Gemm*>(raw)->C_data = host_bias;
    }

    ncnn::VkTransfer upload(device);
    if (raw->create_pipeline(option) != 0 || raw->upload_model(upload, option) != 0
        || upload.submit_and_wait() != 0)
    {
        return false;
    }

    destination = std::move(layer);
    destination_option = option;
    return true;
}

// --- Bfloat16Linear_vulkan ---------------------------------------------------

Bfloat16Linear_vulkan::Bfloat16Linear_vulkan()
    : d(new Implementation)
{
}

Bfloat16Linear_vulkan::~Bfloat16Linear_vulkan() = default;

uint64_t Bfloat16Linear_vulkan::storage_bytes(const TensorData& matrix, uint64_t alignment) noexcept
{
    if (matrix.dtype != DType::BFloat16 || matrix.shape.size() != 2
        || matrix.shape[0] == 0 || matrix.shape[1] == 0 || matrix.shape[1] % 4 != 0
        || matrix.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 32)
        || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return 0;
    }

    const uint64_t elements = static_cast<uint64_t>(matrix.shape[0]) * matrix.shape[1];
    if (elements > static_cast<uint64_t>(std::numeric_limits<int>::max()) * 2
        || elements > std::numeric_limits<size_t>::max() / sizeof(uint16_t))
    {
        return 0;
    }

    alignment = std::max<uint64_t>(4, alignment);
    const uint64_t block_padding = vulkan_weight_block_size % alignment == 0
                                       ? 0
                                       : alignment - vulkan_weight_block_size % alignment;
    if (vulkan_weight_block_size > std::numeric_limits<uint64_t>::max() - block_padding)
        return 0;
    const uint64_t block_size = vulkan_weight_block_size + block_padding;
    const uint64_t segments[] = {
        elements * sizeof(uint16_t),
        static_cast<uint64_t>(matrix.shape[0]) * sizeof(float),
        static_cast<uint64_t>(matrix.shape[1]) * sizeof(float),
    };
    uint64_t total = 0;
    for (uint64_t bytes : segments)
    {
        const uint64_t padding = bytes % alignment == 0 ? 0 : alignment - bytes % alignment;
        if (bytes > std::numeric_limits<uint64_t>::max() - padding)
            return 0;
        const uint64_t allocation = std::max(block_size, bytes + padding);
        if (total > std::numeric_limits<uint64_t>::max() - allocation)
            return 0;
        total += allocation;
    }
    return total;
}

std::shared_ptr<Bfloat16Linear_vulkan>
Bfloat16Linear_vulkan::create(const TensorData& matrix,
                              const TensorData* bias,
                              uint32_t vulkan_device_index,
                              const VulkanRuntimePtr& vulkan_runtime,
                              uint64_t optimization_flags)
{
    return create_with_allocator(matrix, bias, vulkan_device_index, nullptr,
                                 vulkan_runtime, optimization_flags);
}

std::shared_ptr<Bfloat16Linear_vulkan>
Bfloat16Linear_vulkan::create_with_allocator(const TensorData& matrix,
                                             const TensorData* bias,
                                             uint32_t vulkan_device_index,
                                             ncnn::VkAllocator* weight_allocator,
                                             const VulkanRuntimePtr& vulkan_runtime,
                                             uint64_t optimization_flags,
                                             VkWeightBatch* upload_batch)
{
    if (matrix.dtype != DType::BFloat16 || matrix.shape.size() != 2
        || matrix.shape[0] == 0 || matrix.shape[1] == 0 || matrix.shape[1] % 4 != 0
        || matrix.shape[0] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 32)
        || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }

    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const std::span<const uint16_t> weights = matrix.bfloat16_values();
    if (weights.size() != matrix.element_count())
        return {};
    if (bias && (bias->shape.size() != 1 || bias->shape[0] != output_columns || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    std::shared_ptr<Bfloat16Linear_vulkan> result(new Bfloat16Linear_vulkan);
    if (upload_batch)
        upload_batch->retain_owner(result);

    Implementation& implementation = *result->d;
    implementation.input_columns = input_columns;
    implementation.output_columns = output_columns;
    implementation.block_count = (input_columns + 127) / 128;
    implementation.optimization_flags = optimization_flags;
    implementation.vulkan_context = VulkanContext::acquire(vulkan_device_index,
                                                           vulkan_runtime,
                                                           optimization_flags);
    if (!implementation.vulkan_context)
        return {};

    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    if (device->info.subgroup_size() != 32
        || storage_bytes(matrix, device->info.buffer_offset_alignment()) == 0)
    {
        return {};
    }

    implementation.option = make_vk_opt(*implementation.vulkan_context);
    implementation.option.vulkan_device_index = device->info.device_index();
    implementation.option.use_cooperative_matrix = device->info.support_cooperative_matrix();

    if (!weight_allocator)
    {
        // Large weights may use dedicated allocations. Keep their small bias
        // and RMSNorm uploads from each reserving another weight-sized block.
        implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(device,
                                                                          static_cast<size_t>(vulkan_weight_block_size)));
        weight_allocator = implementation.weight_allocator.get();
    }
    if (!upload_batch)
        implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));

    ncnn::Mat packed;
    ncnn::Mat biases(static_cast<int>(output_columns), sizeof(float));
    if (!prepare_bfloat16_upload(weights, packed) || biases.empty())
        return {};

    float* bias_values = static_cast<float*>(biases.data);
    if (!bias)
    {
        std::fill_n(bias_values, output_columns, 0.0f);
    }
    else if (bias->dtype == DType::Float32)
    {
        const std::span<const float> values = bias->float32_values();
        if (values.size() != output_columns)
            return {};
        std::copy(values.begin(), values.end(), bias_values);
    }
    else
    {
        const std::span<const uint16_t> values = bias->bfloat16_values();
        if (values.size() != output_columns)
            return {};
        for (uint32_t index = 0; index < output_columns; ++index)
            bias_values[index] = bfloat16_to_float(values[index]);
    }

    implementation.has_nonzero_bias = false;
    for (uint32_t index = 0; index < output_columns; ++index)
    {
        if (bias_values[index] != 0.0f)
        {
            implementation.has_nonzero_bias = true;
            break;
        }
    }

    bool cm2_ready = false;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_bf16_pipeline(implementation.vulkan_context,
                                  implementation.option,
                                  bfloat16_projection_shader,
                                  static_cast<int>(sizeof(bfloat16_projection_shader) - 1),
                                  32,
                                  implementation.pipeline))
        {
            return {};
        }

        implementation.cm2_ok = create_cm2_gemm(implementation.vulkan_context,
                                                implementation.option,
                                                implementation.gemm_cm2);
        cm2_ready = implementation.cm2_ok;
        (void)create_tile_gemm(implementation.vulkan_context,
                               implementation.option,
                               implementation.tile_gemm);
        (void)create_pipeline(implementation.vulkan_context,
                              bfloat16_row_rms_shader,
                              static_cast<int>(sizeof(bfloat16_row_rms_shader) - 1),
                              implementation.option,
                              0, 0, 256,
                              implementation.row_rms_scale_pipeline,
                              {}, 32);
        if (has_flag(implementation.optimization_flags,
                     OptimizationVulkanBfloat16CoopMatrix))
        {
            (void)create_bfloat16_cooperative_projection_pipeline(implementation.vulkan_context,
                                                                  implementation.option,
                                                                  input_columns,
                                                                  output_columns,
                                                                  implementation.cooperative_pipeline,
                                                                  implementation.cooperative_tile_m,
                                                                  implementation.cooperative_tile_n,
                                                                  implementation.cooperative_tile_k,
                                                                  implementation.cooperative_subgroup_size);
        }
    }

    if (matvec_supported(implementation.vulkan_context, input_columns, output_columns, true)
        && !create_matvec_pipeline(implementation.vulkan_context, implementation.option, true,
                                   input_columns, output_columns, implementation.single_row_pipeline))
    {
        return {};
    }

    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = weight_allocator;
    upload_option.workspace_vkallocator = weight_allocator;
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();

    bool uploaded = false;
    if (upload_batch)
    {
        if (!dynamic_cast<VkWeightAllocator*>(weight_allocator))
            return {};
        uploaded = upload_batch->record(packed, implementation.packed, upload_option, weight_allocator)
                   && upload_batch->record(biases, implementation.bias, upload_option, weight_allocator);
    }
    else if (auto* independent_allocator = dynamic_cast<VkWeightAllocator*>(weight_allocator))
    {
        VkWeightTransfer command(implementation.vulkan_context, implementation.weight_staging_allocator.get());
        uploaded = command.record(packed, implementation.packed, independent_allocator)
                   && command.record(biases, implementation.bias, independent_allocator)
                   && command.submit_and_wait();
    }
    else
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(device);
        command.record_upload(packed, implementation.packed, upload_option);
        command.record_upload(biases, implementation.bias, upload_option);
        uploaded = !implementation.packed.empty()
                   && !implementation.bias.empty()
                   && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return {};

    if (!cm2_ready)
    {
        // Pipeline creation and the weight upload both submit through the shared
        // context; the QNK path takes this same lock.
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        (void)create_coop_gemm(implementation.vulkan_context, implementation.option,
                               input_columns, output_columns, weights,
                               std::span<const float>(bias_values, static_cast<size_t>(output_columns)),
                               implementation.coop_gemm, implementation.coop_opt);
    }
    return result;
}

std::shared_ptr<Bfloat16Linear_vulkan>
Bfloat16Linear_vulkan::create_fused(const std::vector<const TensorData*>& matrices,
                                    const std::vector<const TensorData*>& biases,
                                    uint32_t vulkan_device_index,
                                    const VulkanRuntimePtr& vulkan_runtime,
                                    uint64_t optimization_flags)
{
    if (matrices.empty() || matrices.size() != biases.size() || !matrices.front()
        || matrices.front()->dtype != DType::BFloat16 || matrices.front()->shape.size() != 2)
    {
        return {};
    }

    const uint32_t input_columns = matrices.front()->shape[1];
    const bool has_bias = biases.front() != nullptr;
    uint64_t output_columns = 0;
    uint64_t element_count = 0;
    for (size_t index = 0; index < matrices.size(); ++index)
    {
        const TensorData* matrix = matrices[index];
        const TensorData* bias = biases[index];
        if (!matrix || matrix->dtype != DType::BFloat16 || matrix->shape.size() != 2
            || matrix->shape[1] != input_columns || (bias != nullptr) != has_bias
            || (bias && (bias->shape.size() != 1 || bias->shape[0] != matrix->shape[0] || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16))))
        {
            return {};
        }
        output_columns += matrix->shape[0];
        element_count += matrix->element_count();
    }
    if (output_columns > std::numeric_limits<uint32_t>::max()
        || element_count > std::numeric_limits<size_t>::max())
    {
        return {};
    }

    TensorData fused_matrix;
    fused_matrix.dtype = DType::BFloat16;
    fused_matrix.shape = {static_cast<uint32_t>(output_columns), input_columns};
    fused_matrix.bfloat16_data.reserve(static_cast<size_t>(element_count));

    TensorData fused_bias;
    fused_bias.dtype = DType::Float32;
    fused_bias.shape = {static_cast<uint32_t>(output_columns)};
    if (has_bias)
        fused_bias.float32_data.reserve(static_cast<size_t>(output_columns));

    for (size_t index = 0; index < matrices.size(); ++index)
    {
        const std::span<const uint16_t> matrix_values = matrices[index]->bfloat16_values();
        if (matrix_values.size() != matrices[index]->element_count())
            return {};
        fused_matrix.bfloat16_data.insert(fused_matrix.bfloat16_data.end(),
                                          matrix_values.begin(),
                                          matrix_values.end());
        if (!has_bias)
            continue;

        const TensorData& bias = *biases[index];
        if (bias.dtype == DType::Float32)
        {
            const std::span<const float> bias_values = bias.float32_values();
            fused_bias.float32_data.insert(fused_bias.float32_data.end(),
                                           bias_values.begin(),
                                           bias_values.end());
        }
        else
        {
            for (uint16_t value : bias.bfloat16_values())
                fused_bias.float32_data.push_back(bfloat16_to_float(value));
        }
    }

    return create(fused_matrix, has_bias ? &fused_bias : nullptr,
                  vulkan_device_index, vulkan_runtime, optimization_flags);
}

std::vector<uint64_t> Bfloat16Linear_vulkan::swiglu_scratch_elements(uint32_t intermediate_columns,
                                                                     bool router_gate, uint64_t rows)
{
    if (intermediate_columns == 0 || rows == 0)
        return {};
    const uint64_t fused_columns = saturating_add_u64(saturating_multiply_u64(intermediate_columns, 2),
                                                      router_gate ? 1 : 0);
    if (fused_columns == UINT64_MAX
        || saturating_multiply_u64(fused_columns, rows) == UINT64_MAX)
    {
        return {UINT64_MAX};
    }
    if (rows >= 16)
    {
        return {saturating_multiply_u64(fused_columns, rows),
                saturating_multiply_u64(intermediate_columns, rows)};
    }
    return {saturating_multiply_u64(fused_columns, rows)};
}

bool Bfloat16Linear_vulkan::prepare_rms_norm(const TensorData& weight,
                                             float epsilon,
                                             float weight_offset)
{
    Implementation& implementation = *d;

    ncnn::Mat values(static_cast<int>(implementation.input_columns), sizeof(float));
    if (values.empty())
        return false;
    float* destination = static_cast<float*>(values.data);
    if (weight.dtype == DType::Float32)
    {
        const std::span<const float> source = weight.float32_values();
        if (source.size() != implementation.input_columns)
            return false;
        for (uint32_t index = 0; index < implementation.input_columns; ++index)
            destination[index] = source[index] + weight_offset;
    }
    else
    {
        const std::span<const uint16_t> source = weight.bfloat16_values();
        if (source.size() != implementation.input_columns)
            return false;
        for (uint32_t index = 0; index < implementation.input_columns; ++index)
            destination[index] = bfloat16_to_float(source[index]) + weight_offset;
    }

    const ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    const uint32_t local_size = device->info.max_workgroup_invocations() >= 256
                                        && device->info.max_workgroup_size_x() >= 256
                                    ? 256
                                    : 128;
    implementation.rms_norm_output_subgroups = local_size / 32;
    if (!create_bf16_pipeline(implementation.vulkan_context,
                              implementation.option,
                              bfloat16_rms_norm_projection_shader,
                              static_cast<int>(sizeof(bfloat16_rms_norm_projection_shader) - 1),
                              static_cast<int>(local_size),
                              implementation.rms_norm_projection_pipeline))
    {
        return false;
    }

    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(implementation.vulkan_context->device()));
    ncnn::VkAllocator* weight_alloc = implementation.weight_allocator
                                          ? implementation.weight_allocator.get()
                                          : implementation.vulkan_context->blob_allocator();
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = weight_alloc;
    upload_option.workspace_vkallocator = weight_alloc;
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();

    bool uploaded = false;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(implementation.vulkan_context->device());
        command.record_upload(values, implementation.rms_norm_weight, upload_option);
        uploaded = !implementation.rms_norm_weight.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
    {
        implementation.rms_norm_projection_pipeline.reset();
        return false;
    }

    implementation.rms_norm_epsilon = epsilon;
    return true;
}

bool Bfloat16Linear_vulkan::has_rms_norm_chain() const noexcept
{
    return d->rms_norm_projection_pipeline != nullptr && !d->rms_norm_weight.empty();
}

bool Bfloat16Linear_vulkan::prepare_input_rms_norm(const TensorData& weight,
                                                   float epsilon,
                                                   float weight_offset)
{
    d->input_rms_norm = RmsNorm_vulkan::create(weight, epsilon, weight_offset,
                                               d->vulkan_context, d->option);
    return static_cast<bool>(d->input_rms_norm);
}

bool Bfloat16Linear_vulkan::prepare_output_rms_norm(const TensorData& weight, float epsilon)
{
    d->output_rms_norm = RmsNorm_vulkan::create(weight, epsilon, 0.0f,
                                                d->vulkan_context, d->option);
    return static_cast<bool>(d->output_rms_norm);
}

bool Bfloat16Linear_vulkan::record_rms_norm_chain(const ncnn::VkMat& input,
                                                  ncnn::VkMat& output,
                                                  ncnn::VkCompute& command) const
{
    const Implementation& implementation = *d;
    // Keep a distinct destination even if a caller supplies another view of
    // input. The fused shader still needs the original input while writing.
    ncnn::VkMat destination(static_cast<int>(implementation.output_columns), input.h,
                            sizeof(float), implementation.vulkan_context->blob_allocator());
    if (destination.empty())
        return false;
    record_rms_norm_projection(input, destination, command);
    output = std::move(destination);
    return true;
}

bool Bfloat16Linear_vulkan::record_input_normalization(const ncnn::VkMat& input,
                                                       ncnn::VkMat& output,
                                                       ncnn::VkCompute& cmd) const
{
    return d->input_rms_norm && d->input_rms_norm->record(input, output, cmd);
}

bool Bfloat16Linear_vulkan::record_rms_norm_chain_parallel(const ncnn::VkMat& input,
                                                           const Bfloat16Linear_vulkan& next,
                                                           const Bfloat16Linear_vulkan& parallel,
                                                           ncnn::VkMat& query,
                                                           ncnn::VkMat& kv,
                                                           ncnn::VkMat& query_rank,
                                                           ncnn::VkCompute& cmd,
                                                           std::vector<ncnn::VkMat>& workspace) const
{
    if (!d->output_rms_norm)
        return false;

    ncnn::VkMat intermediate;
    if (forward(input, intermediate, cmd, d->option) != 0
        || !d->output_rms_norm->record(intermediate, query_rank, cmd)
        || next.forward(query_rank, query, cmd, next.d->option) != 0
        || parallel.forward(input, kv, cmd, parallel.d->option) != 0)
    {
        return false;
    }

    workspace.push_back(std::move(intermediate));
    return true;
}

bool Bfloat16Linear_vulkan::Implementation::record_projection(const ncnn::VkMat& input,
                                                              ncnn::VkMat& output,
                                                              uint32_t token_count,
                                                              ncnn::VkCompute& command,
                                                              std::vector<ncnn::VkMat>& bindings,
                                                              std::vector<ncnn::vk_constant_type>& constants) const
{
    if (single_row_pipeline && token_count == 1 && single_row_input(input, input_columns)
        && output.elempack == 1 && output.elemsize == sizeof(float))
    {
        record_matvec(single_row_pipeline.get(), true, input_columns, output_columns, true,
                      input, packed, bias, output, command);
        return false;
    }
    if (cm2_ok && activation_variant(option) == 0
        && record_cm2_gemm(gemm_cm2, input, packed, bias, has_nonzero_bias, output,
                           {token_count, input_columns, output_columns, output_columns, 0, 1},
                           vulkan_context.get(), command, bindings, constants))
    {
        return true;
    }

    bindings = {input, packed, bias, output};
    constants.resize(4);
    const uint64_t cooperative_output_work = static_cast<uint64_t>(token_count) * output_columns;
    if (cooperative_pipeline
        && token_count >= cooperative_tile_m
        && input_columns >= cooperative_tile_k
        && output_columns >= cooperative_tile_n
        && cooperative_output_work >= UINT64_C(65536))
    {
        const uint64_t token_tile_count = (token_count + cooperative_tile_m - 1)
                                          / cooperative_tile_m;
        const uint64_t output_tile_count = (output_columns + cooperative_tile_n - 1)
                                           / cooperative_tile_n;
        const uint64_t invocation_count = token_tile_count * output_tile_count
                                          * cooperative_subgroup_size;
        if (invocation_count <= static_cast<uint64_t>(std::numeric_limits<int>::max()))
        {
            constants[0].u32 = input_columns;
            constants[1].u32 = output_columns;
            constants[2].u32 = token_count;
            constants[3].u32 = static_cast<uint32_t>(token_tile_count);

            ncnn::VkMat dispatcher;
            dispatcher.w = static_cast<int>(invocation_count);
            dispatcher.h = 1;
            dispatcher.c = 1;
            command.record_pipeline_readonly(cooperative_pipeline.get(),
                                             bindings, {1, 0, 0, 0},
                                             constants, dispatcher);
            return true;
        }
    }

    if (record_tile(input, output, token_count, ncnn::VkMat(), ncnn::VkMat(), command))
        return false;

    constants[0].u32 = input_columns;
    constants[1].u32 = output_columns;
    constants[2].u32 = block_count;
    constants[3].u32 = token_count;

    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(output_columns * 32);
    dispatcher.h = static_cast<int>(token_count);
    dispatcher.c = 1;
    command.record_pipeline_readonly(pipeline.get(),
                                     bindings, {1, 0, 0, 0},
                                     constants, dispatcher);
    return false;
}

bool Bfloat16Linear_vulkan::Implementation::record_tile(const ncnn::VkMat& input,
                                                        ncnn::VkMat& output,
                                                        uint32_t rows,
                                                        const ncnn::VkMat& row_scale,
                                                        const ncnn::VkMat& column_scale,
                                                        ncnn::VkCompute& command) const
{
    if ((!tile_gemm[0].pipeline && !tile_gemm[1].pipeline) || activation_variant(option) != 0)
        return false;
    const TileShape shape{rows, input_columns, output_columns, output_columns, 0, 1};
    return record_tile_gemm(tile_gemm, input, packed, bias, row_scale, column_scale,
                            output, shape, command);
}

bool Bfloat16Linear_vulkan::record_coop(const ncnn::VkMat& input,
                                        ncnn::VkMat& output,
                                        ncnn::VkCompute& cmd,
                                        bool normalize) const
{
    const Implementation& implementation = *d;
    // The Gemm path cannot fold the norm into its tile loads, so the prepared
    // normalization is materialized first when the caller asks for it.
    ncnn::VkMat source = input;
    if (normalize)
    {
        ncnn::VkMat normalized;
        if (!record_input_normalization(input, normalized, cmd))
            return false;
        source = std::move(normalized);
    }

    const ncnn::Option& gemm_option = implementation.coop_opt;
    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    ncnn::VkMat input_fp16;
    device->convert_packing(source, input_fp16, 1, 2, cmd, gemm_option);
    if (input_fp16.empty())
        return false;

    std::vector<ncnn::VkMat> tops(1);
    if (implementation.coop_gemm->forward({input_fp16}, tops, cmd, gemm_option) != 0 || tops[0].empty())
        return false;

    ncnn::VkMat projected;
    device->convert_packing(tops[0], projected, 1, 1, cmd, gemm_option);
    if (projected.empty())
        return false;

    output = std::move(projected);
    return true;
}

void Bfloat16Linear_vulkan::record_scalar_projection(const ncnn::VkMat& input,
                                                     ncnn::VkMat& output,
                                                     ncnn::VkCompute& cmd) const
{
    const Implementation& implementation = *d;
    if (implementation.single_row_pipeline && single_row_input(input, implementation.input_columns)
        && output.elempack == 1 && output.elemsize == sizeof(float))
    {
        record_matvec(implementation.single_row_pipeline.get(), true, implementation.input_columns,
                      implementation.output_columns, true, input, implementation.packed,
                      implementation.bias, output, cmd);
        return;
    }
    if (implementation.cm2_ok && activation_variant(implementation.option) == 0)
    {
        std::vector<ncnn::VkMat> cm2_bindings;
        std::vector<ncnn::vk_constant_type> cm2_constants;
        if (record_cm2_gemm(implementation.gemm_cm2, input, implementation.packed,
                            implementation.bias, implementation.has_nonzero_bias, output,
                            {static_cast<uint32_t>(input.h), implementation.input_columns,
                             implementation.output_columns, implementation.output_columns, 0, 1},
                            implementation.vulkan_context.get(), cmd, cm2_bindings, cm2_constants))
        {
            return;
        }
    }
    if (implementation.coop_gemm && record_coop(input, output, cmd))
        return;
    if (implementation.record_tile(input, output, static_cast<uint32_t>(input.h),
                                   ncnn::VkMat(), ncnn::VkMat(), cmd))
    {
        return;
    }

    const std::vector<ncnn::VkMat> bindings = {input, implementation.packed, implementation.bias, output};
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.h);

    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(implementation.output_columns * 32);
    dispatcher.h = input.h;
    dispatcher.c = 1;
    cmd.record_pipeline_readonly(implementation.pipeline.get(),
                                 bindings, {1, 0, 0, 0},
                                 constants, dispatcher);
}

bool Bfloat16Linear_vulkan::use_cm2() const noexcept
{
    return d->cm2_ok && activation_variant(d->option) == 0;
}

void Bfloat16Linear_vulkan::record_norm_projection(const ncnn::VkMat& input,
                                                   ncnn::VkMat& output,
                                                   ncnn::VkCompute& cmd,
                                                   ncnn::VkMat* row_scale_workspace) const
{
    const Implementation& implementation = *d;
    if (row_scale_workspace && input.h >= 8 && implementation.cm2_ok
        && activation_variant(implementation.option) == 0 && implementation.input_rms_norm)
    {
        ncnn::VkMat normalized;
        const bool norm_ok = implementation.input_rms_norm->record(input, normalized, cmd, nullptr, nullptr, true)
                             || record_input_normalization(input, normalized, cmd);
        if (norm_ok)
        {
            *row_scale_workspace = normalized;
            std::vector<ncnn::VkMat> cm2_bindings;
            std::vector<ncnn::vk_constant_type> cm2_constants;
            if (record_cm2_gemm(implementation.gemm_cm2, normalized, implementation.packed,
                                implementation.bias, implementation.has_nonzero_bias, output,
                                {static_cast<uint32_t>(input.h), implementation.input_columns,
                                 implementation.output_columns, implementation.output_columns, 0, 1},
                                implementation.vulkan_context.get(), cmd, cm2_bindings, cm2_constants))
            {
                return;
            }
        }
    }
    if (implementation.coop_gemm && record_coop(input, output, cmd, true))
        return;
    record_rms_norm_projection(input, output, cmd, row_scale_workspace);
}

void Bfloat16Linear_vulkan::record_rms_norm_projection(const ncnn::VkMat& input,
                                                       ncnn::VkMat& output,
                                                       ncnn::VkCompute& cmd,
                                                       ncnn::VkMat* row_scale_workspace) const
{
    const Implementation& implementation = *d;
    if (row_scale_workspace
        && implementation.row_rms_scale_pipeline
        && !implementation.rms_norm_weight.empty()
        && input.h >= 8
        && implementation.input_columns % 16 == 0)
    {
        // Exactly `token_count` floats of graph-arena scratch, retained by the
        // caller until the submission completes. The per-token byte count is far
        // below the arena slack, so this allocation never grows the plan.
        row_scale_workspace->create(input.h, sizeof(float),
                                    implementation.vulkan_context->blob_allocator());
        if (!row_scale_workspace->empty())
        {
            std::vector<ncnn::vk_constant_type> scale_constants(2);
            scale_constants[0].u32 = implementation.input_columns;
            scale_constants[1].f = implementation.rms_norm_epsilon;

            ncnn::VkMat scale_dispatcher;
            scale_dispatcher.w = 256 * input.h;
            scale_dispatcher.h = 1;
            scale_dispatcher.c = 1;
            cmd.record_pipeline_readonly(implementation.row_rms_scale_pipeline.get(),
                                         {input, *row_scale_workspace},
                                         {1, 0},
                                         scale_constants,
                                         scale_dispatcher);
            if (implementation.record_tile(input, output, static_cast<uint32_t>(input.h),
                                           *row_scale_workspace, implementation.rms_norm_weight, cmd))
            {
                return;
            }
        }
    }

    const std::vector<ncnn::VkMat> bindings = {
        input,
        implementation.packed,
        implementation.bias,
        implementation.rms_norm_weight,
        output};
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.h);
    constants[4].f = implementation.rms_norm_epsilon;
    constants[5].u32 = implementation.rms_norm_output_subgroups;

    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(((static_cast<uint64_t>(implementation.output_columns)
                                      + implementation.rms_norm_output_subgroups - 1)
                                     / implementation.rms_norm_output_subgroups)
                                    * implementation.rms_norm_output_subgroups * 32);
    dispatcher.h = input.h;
    dispatcher.c = 1;
    cmd.record_pipeline_readonly(implementation.rms_norm_projection_pipeline.get(),
                                 bindings, {1, 0, 0, 0, 0},
                                 constants, dispatcher);
}

int Bfloat16Linear_vulkan::forward(const ncnn::VkMat& input,
                                   ncnn::VkMat& output,
                                   ncnn::VkCompute& cmd,
                                   const ncnn::Option& opt) const
{
    output.create(static_cast<int>(d->output_columns), input.h, sizeof(float), opt.blob_vkallocator);
    if (output.empty())
        return -100;
    record_scalar_projection(input, output, cmd);
    return 0;
}

const std::shared_ptr<VulkanContext>& Bfloat16Linear_vulkan::context() const noexcept
{
    return d->vulkan_context;
}

const ncnn::Option& Bfloat16Linear_vulkan::option() const noexcept
{
    return d->option;
}

uint32_t Bfloat16Linear_vulkan::input_columns() const noexcept
{
    return d->input_columns;
}

uint32_t Bfloat16Linear_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

bool Bfloat16Linear_vulkan::forward(const ActivationBuffer& input,
                                    ActivationBuffer& output) const
{
    const Implementation& implementation = *d;
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();

    const bool direct_host_input = activation_variant(implementation.option) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                    * implementation.input_columns * sizeof(float),
                                                                                                input.dtype());
    const bool direct_host_output = activation_variant(implementation.option) == 0
                                    && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                     * implementation.output_columns * sizeof(float),
                                                                                                 output.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(),
                                  implementation.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }

    const DType output_dtype = output.dtype();
    output.reset(input.rows(), implementation.output_columns, false);

    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used && command.reset() != 0)
        return false;
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (direct_host_input)
    {
        input_gpu = bind_host_in(transfer_slot.upload);
    }
    else if (!record_mapped_act_upload(transfer_slot.upload, input_gpu, command,
                                       implementation.vulkan_context->device(),
                                       implementation.option, input.dtype()))
    {
        return false;
    }

    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = bind_host_out(transfer_slot.download);
    else
        output_gpu.create(static_cast<int>(implementation.output_columns),
                          static_cast<int>(input.rows()),
                          act_element_size(implementation.option),
                          implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings;
    std::vector<ncnn::vk_constant_type> constants;
    (void)implementation.record_projection(input_gpu, output_gpu,
                                           static_cast<uint32_t>(input.rows()),
                                           command, bindings, constants);
    if ((!direct_host_output
         && !record_staging_download(output_gpu, input.rows(), implementation.output_columns,
                                     transfer_slot.download, command,
                                     implementation.vulkan_context->device(),
                                     implementation.option, output_dtype))
        || submit_wait(command, implementation.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }

    ++runtime_state.dispatches;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
}

bool Bfloat16Linear_vulkan::forward_rms_norm_chain(const ActivationBuffer& input,
                                                   ActivationBuffer& output) const
{
    const Implementation& implementation = *d;
    if (!implementation.rms_norm_projection_pipeline
        || implementation.rms_norm_weight.empty()
        || implementation.rms_norm_output_subgroups == 0)
    {
        return false;
    }

    const uint64_t dispatch_width = ((static_cast<uint64_t>(implementation.output_columns)
                                      + implementation.rms_norm_output_subgroups - 1)
                                     / implementation.rms_norm_output_subgroups)
                                    * implementation.rms_norm_output_subgroups * 32;

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();

    const bool direct_host_input = activation_variant(implementation.option) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                    * implementation.input_columns * sizeof(float),
                                                                                                input.dtype());
    const bool direct_host_output = activation_variant(implementation.option) == 0
                                    && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                     * implementation.output_columns * sizeof(float),
                                                                                                 output.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(),
                                  implementation.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }

    const DType output_dtype = output.dtype();
    output.reset(input.rows(), implementation.output_columns, false);

    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used && command.reset() != 0)
        return false;
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (direct_host_input)
    {
        input_gpu = bind_host_in(transfer_slot.upload);
    }
    else if (!record_mapped_act_upload(transfer_slot.upload, input_gpu, command,
                                       implementation.vulkan_context->device(),
                                       implementation.option, input.dtype()))
    {
        return false;
    }

    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = bind_host_out(transfer_slot.download);
    else
        output_gpu.create(static_cast<int>(implementation.output_columns),
                          static_cast<int>(input.rows()),
                          act_element_size(implementation.option),
                          implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    record_rms_norm_projection(input_gpu, output_gpu, command);
    if ((!direct_host_output
         && !record_staging_download(output_gpu, input.rows(), implementation.output_columns,
                                     transfer_slot.download, command,
                                     implementation.vulkan_context->device(),
                                     implementation.option, output_dtype))
        || submit_wait(command, implementation.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }

    ++runtime_state.dispatches;
    ++runtime_state.rms_norm_linear_fusions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
}

bool Bfloat16Linear_vulkan::forward_parallel(const ActivationBuffer& input,
                                             const Bfloat16Linear_vulkan& parallel_operator,
                                             ActivationBuffer& output,
                                             ActivationBuffer& parallel_output) const
{
    const Implementation& first = *d;
    const Implementation& parallel = *parallel_operator.d;
    if (first.vulkan_context.get() != parallel.vulkan_context.get()
        || activation_variant(first.option) != activation_variant(parallel.option)
        || input.rows() == 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    ncnn::VkMat parallel_download;

    const bool direct_host_input = activation_variant(first.option) == 0
                                   && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                           * first.input_columns * sizeof(float),
                                                                                       input.dtype());
    const bool direct_host_output = activation_variant(first.option) == 0
                                    && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                            * first.output_columns * sizeof(float),
                                                                                        output.dtype());
    const bool parallel_direct_host_output = activation_variant(parallel.option) == 0
                                             && parallel.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                        * parallel.output_columns * sizeof(float),
                                                                                                    parallel_output.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(),
                                  first.output_columns, transfer_slot.staging_allocator)
        || !prepare_staging_batch(parallel_download, input.rows(),
                                  parallel.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }

    const DType output_dtype = output.dtype();
    const DType parallel_output_dtype = parallel_output.dtype();
    output.reset(input.rows(), first.output_columns, false);
    parallel_output.reset(input.rows(), parallel.output_columns, false);

    const std::lock_guard<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used && command.reset() != 0)
        return false;
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (direct_host_input)
    {
        input_gpu = bind_host_in(transfer_slot.upload);
    }
    else if (!record_mapped_act_upload(transfer_slot.upload, input_gpu, command,
                                       first.vulkan_context->device(),
                                       first.option, input.dtype()))
    {
        return false;
    }

    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = bind_host_out(transfer_slot.download);
    else
        output_gpu.create(static_cast<int>(first.output_columns),
                          static_cast<int>(input.rows()),
                          act_element_size(first.option),
                          first.vulkan_context->blob_allocator());

    ncnn::VkMat parallel_output_gpu;
    if (parallel_direct_host_output)
        parallel_output_gpu = bind_host_out(parallel_download);
    else
        parallel_output_gpu.create(static_cast<int>(parallel.output_columns),
                                   static_cast<int>(input.rows()),
                                   act_element_size(parallel.option),
                                   parallel.vulkan_context->blob_allocator());
    if (output_gpu.empty() || parallel_output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings;
    std::vector<ncnn::vk_constant_type> constants;
    (void)first.record_projection(input_gpu, output_gpu,
                                  static_cast<uint32_t>(input.rows()),
                                  command, bindings, constants);
    (void)parallel.record_projection(input_gpu, parallel_output_gpu,
                                     static_cast<uint32_t>(input.rows()),
                                     command, bindings, constants);

    if ((!direct_host_output
         && !record_staging_download(output_gpu, input.rows(), first.output_columns,
                                     transfer_slot.download, command,
                                     first.vulkan_context->device(),
                                     first.option, output_dtype))
        || (!parallel_direct_host_output
            && !record_staging_download(parallel_output_gpu, input.rows(), parallel.output_columns,
                                        parallel_download, command,
                                        parallel.vulkan_context->device(),
                                        parallel.option, parallel_output_dtype))
        || submit_wait(command, first.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output)
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }

    runtime_state.dispatches += 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads += 2;
    return true;
}

bool Bfloat16Linear_vulkan::forward_swiglu_chain(const ActivationBuffer& input,
                                                 const Bfloat16Linear_vulkan& down_operator,
                                                 uint32_t intermediate_columns,
                                                 ExpertActivation activation,
                                                 float activation_limit,
                                                 bool apply_router_gate,
                                                 ActivationBuffer& output) const
{
    const Implementation& first = *d;
    const Implementation& down = *down_operator.d;
    const uint64_t expected_fused_columns = static_cast<uint64_t>(intermediate_columns) * 2
                                            + (apply_router_gate ? 1u : 0u);
    if ((activation != ExpertActivation::Silu && activation != ExpertActivation::DeepSeekSwiGlu
         && activation != ExpertActivation::GeluTanh)
        || first.vulkan_context.get() != down.vulkan_context.get()
        || activation_variant(first.option) != activation_variant(down.option)
        || first.output_columns != expected_fused_columns
        || down.input_columns != intermediate_columns
        || intermediate_columns == 0
        || intermediate_columns % 128 != 0
        || input.rows() == 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();

    const bool direct_host_input = activation_variant(first.option) == 0
                                   && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                           * first.input_columns * sizeof(float),
                                                                                       input.dtype());
    const bool direct_host_output = activation_variant(first.option) == 0
                                    && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                            * down.output_columns * sizeof(float),
                                                                                        output.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(),
                                  down.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }

    const DType output_dtype = output.dtype();
    output.reset(input.rows(), down.output_columns, false);

    const std::lock_guard<std::mutex> lock(first.vulkan_context->command_mutex());
    if (!first.swiglu_down_pipeline
        && !create_bfloat16_swiglu_down_pipeline(first.vulkan_context,
                                                 first.option,
                                                 first.swiglu_down_pipeline))
    {
        return false;
    }

    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used && command.reset() != 0)
        return false;
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (direct_host_input)
    {
        input_gpu = bind_host_in(transfer_slot.upload);
    }
    else if (!record_mapped_act_upload(transfer_slot.upload, input_gpu, command,
                                       first.vulkan_context->device(),
                                       first.option, input.dtype()))
    {
        return false;
    }

    ncnn::VkMat fused_gpu(static_cast<int>(first.output_columns),
                          static_cast<int>(input.rows()),
                          act_element_size(first.option),
                          first.vulkan_context->blob_allocator());
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = bind_host_out(transfer_slot.download);
    else
        output_gpu.create(static_cast<int>(down.output_columns),
                          static_cast<int>(input.rows()),
                          act_element_size(first.option),
                          first.vulkan_context->blob_allocator());
    if (fused_gpu.empty() || output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> bindings(4);
    std::vector<ncnn::vk_constant_type> constants(5);
    (void)first.record_projection(input_gpu, fused_gpu,
                                  static_cast<uint32_t>(input.rows()),
                                  command, bindings, constants);
    record_swiglu_down(first.swiglu_down_pipeline.get(), fused_gpu, down, output_gpu,
                       intermediate_columns, first.output_columns,
                       static_cast<uint32_t>(input.rows()),
                       activation, activation_limit, apply_router_gate,
                       command, bindings, constants);

    if ((!direct_host_output
         && !record_staging_download(output_gpu, input.rows(), down.output_columns,
                                     transfer_slot.download, command,
                                     first.vulkan_context->device(),
                                     first.option, output_dtype))
        || submit_wait(command, first.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }

    runtime_state.dispatches += 2;
    ++runtime_state.shared_expert_swiglu_fusions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
}

bool Bfloat16Linear_vulkan::record_swiglu_chain(const ncnn::VkMat& input,
                                                const Bfloat16Linear_vulkan& down_operator,
                                                uint32_t intermediate_columns,
                                                ExpertActivation activation,
                                                float activation_limit,
                                                bool apply_router_gate,
                                                ncnn::VkMat& output,
                                                ncnn::VkCompute& command,
                                                std::vector<ncnn::VkMat>& workspace,
                                                FfnRecordStatistics_vulkan& statistics,
                                                const ncnn::VkMat* res,
                                                bool* res_fused) const
{
    if (res_fused)
        *res_fused = false;

    const Implementation& first = *d;
    const Implementation& down = *down_operator.d;
    const bool use_cm2_prefill = input.h >= 16
                                 && down.cm2_ok
                                 && activation_variant(first.option) == 0
                                 && (!apply_router_gate || !down.has_nonzero_bias)
                                 && (first.swiglu_pipe
                                     || create_qnk_swiglu_pipe(first.vulkan_context,
                                                               first.option,
                                                               first.swiglu_pipe,
                                                               true));
    if (use_cm2_prefill)
    {
        workspace.reserve(workspace.size() + 3);
        ncnn::VkMat fused(static_cast<int>(first.output_columns), input.h,
                          sizeof(float), first.vulkan_context->blob_allocator());
        ncnn::VkMat activated(static_cast<int>(intermediate_columns), input.h,
                              sizeof(uint16_t), 1, first.vulkan_context->blob_allocator());
        ncnn::VkMat result(static_cast<int>(down.output_columns), input.h,
                           sizeof(float), first.vulkan_context->blob_allocator());
        if (fused.empty() || activated.empty() || result.empty())
            return false;
        workspace.push_back(fused);
        workspace.push_back(activated);
        workspace.push_back(result);

        std::vector<ncnn::VkMat> bindings;
        std::vector<ncnn::vk_constant_type> constants;
        (void)first.record_projection(input, fused, static_cast<uint32_t>(input.h),
                                      command, bindings, constants);

        bindings = {fused, activated, fused};
        constants.resize(5);
        constants[0].u32 = intermediate_columns;
        constants[1].u32 = static_cast<uint32_t>(input.h);
        constants[2].u32 = activation == ExpertActivation::GeluTanh ? 3u : 1u;
        constants[3].f = activation == ExpertActivation::DeepSeekSwiGlu ? activation_limit : 0.0f;
        constants[4].u32 = apply_router_gate ? 3u : 1u;

        ncnn::VkMat swiglu_dispatcher;
        swiglu_dispatcher.w = static_cast<int>(intermediate_columns >> 1);
        swiglu_dispatcher.h = input.h;
        swiglu_dispatcher.c = 1;
        command.record_pipeline_readonly(first.swiglu_pipe.get(), bindings, {1, 0, 1},
                                         constants, swiglu_dispatcher);

        if (!record_cm2_gemm(down.gemm_cm2, activated, down.packed, down.bias,
                             down.has_nonzero_bias, result,
                             {static_cast<uint32_t>(input.h), down.input_columns,
                              down.output_columns, down.output_columns, 0, 1},
                             first.vulkan_context.get(), command, bindings, constants,
                             res, res_fused))
        {
            return false;
        }

        output = std::move(result);
        statistics.dispatches += 3;
        return true;
    }

    if (!first.swiglu_down_pipeline
        && !create_bfloat16_swiglu_down_pipeline(first.vulkan_context, first.option,
                                                 first.swiglu_down_pipeline))
    {
        return false;
    }

    workspace.reserve(workspace.size() + 2);
    ncnn::VkMat fused(static_cast<int>(first.output_columns), input.h,
                      sizeof(float), first.vulkan_context->blob_allocator());
    ncnn::VkMat result(static_cast<int>(down.output_columns), input.h,
                       sizeof(float), first.vulkan_context->blob_allocator());
    if (fused.empty() || result.empty())
        return false;
    workspace.push_back(fused);
    workspace.push_back(result);

    std::vector<ncnn::VkMat> bindings;
    std::vector<ncnn::vk_constant_type> constants;
    (void)first.record_projection(input, fused, static_cast<uint32_t>(input.h),
                                  command, bindings, constants);
    record_swiglu_down(first.swiglu_down_pipeline.get(), fused, down, result,
                       intermediate_columns, first.output_columns,
                       static_cast<uint32_t>(input.h),
                       activation, activation_limit, apply_router_gate,
                       command, bindings, constants);

    output = std::move(result);
    statistics.dispatches += 2;
    ++statistics.shared_expert_swiglu_fusions;
    return true;
}

bool Bfloat16Linear_vulkan::forward_swiglu_chain_device(const DeviceTensor_vulkan& input,
                                                        const Bfloat16Linear_vulkan& down_operator,
                                                        uint32_t intermediate_columns,
                                                        ExpertActivation activation,
                                                        float activation_limit,
                                                        bool apply_router_gate,
                                                        DeviceTensor_vulkan& output) const
{
    const std::shared_ptr<VulkanContext>& context = d->vulkan_context;
    const ncnn::VkMat* source = input.value();
    if (!source || input.context() != context)
        return false;

    const ncnn::VkMat retained_input = *source;
    VkTransferLease lease = context->acquire_transfer_slot();
    VkTransferSlot& slot = lease.slot();

    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if (slot.command_used && slot.command->reset() != 0)
        return false;
    slot.command_used = true;

    ncnn::VkCompute& command = *slot.command;
    bool successful = false;
    const auto cleanup = [&](void*) {
        if (!successful)
            (void)command.reset();
    };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);

    input.prepare_read();
    ncnn::VkMat input_gpu = retained_input;
    if (input_gpu.elempack != 1)
    {
        context->device()->convert_packing(retained_input, input_gpu, 1, 1, command, d->option);
        if (input_gpu.empty())
            return false;
    }

    ncnn::VkMat output_gpu;
    std::vector<ncnn::VkMat> workspace;
    FfnRecordStatistics_vulkan statistics;
    if (!record_swiglu_chain(input_gpu, down_operator, intermediate_columns, activation,
                             activation_limit, apply_router_gate, output_gpu,
                             command, workspace, statistics)
        || submit_wait(command, context->device()) != 0)
    {
        return false;
    }

    DeviceTensor_vulkan completed;
    if (!completed.assign_completed(output_gpu, context))
        return false;
    output = std::move(completed);

    VulkanRuntimeState& state = context->runtime_state();
    state.dispatches += statistics.dispatches;
    state.int4_dispatches += statistics.int4_dispatches;
    state.shared_expert_swiglu_fusions += statistics.shared_expert_swiglu_fusions;
    ++state.compute_submissions;
    successful = true;
    return true;
}

bool Bfloat16Linear_vulkan::materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output) const
{
    return input.materialize(output);
}

// --- Bfloat16Expert_vulkan ---------------------------------------------------

Bfloat16Expert_vulkan::Bfloat16Expert_vulkan()
    : d(new Implementation)
{
}

Bfloat16Expert_vulkan::~Bfloat16Expert_vulkan() = default;

uint32_t Bfloat16Expert_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

std::shared_ptr<Bfloat16Expert_vulkan>
Bfloat16Expert_vulkan::create(const TensorData& gate_up,
                              const TensorData* gate_up_bias,
                              const TensorData& down,
                              const TensorData* down_bias,
                              float activation_limit,
                              uint32_t vulkan_device_index,
                              ExpertActivation activation,
                              const VulkanRuntimePtr& vulkan_runtime,
                              uint64_t optimization_flags)
{
    return create_with_allocator(gate_up, gate_up_bias, down, down_bias, activation_limit,
                                 vulkan_device_index, nullptr, activation,
                                 vulkan_runtime, optimization_flags);
}

std::shared_ptr<Bfloat16Expert_vulkan>
Bfloat16Expert_vulkan::create_with_allocator(const TensorData& gate_up,
                                             const TensorData* gate_up_bias,
                                             const TensorData& down,
                                             const TensorData* down_bias,
                                             float activation_limit,
                                             uint32_t vulkan_device_index,
                                             ncnn::VkAllocator* weight_allocator,
                                             ExpertActivation activation,
                                             const VulkanRuntimePtr& vulkan_runtime,
                                             uint64_t optimization_flags,
                                             VkWeightBatch* upload_batch)
{
    if (activation != ExpertActivation::Silu
        || gate_up.dtype != DType::BFloat16 || down.dtype != DType::BFloat16
        || gate_up.shape.size() != 2 || down.shape.size() != 2
        || gate_up.shape[0] == 0 || gate_up.shape[0] % 2 != 0
        || down.shape[0] == 0 || down.shape[1] != gate_up.shape[0] / 2
        || (gate_up.shape[0] / 2) % 128 != 0
        || activation_limit < 0.0f)
    {
        return {};
    }

    std::shared_ptr<Bfloat16Expert_vulkan> result(new Bfloat16Expert_vulkan);
    if (upload_batch)
        upload_batch->retain_owner(result);

    Implementation& implementation = *result->d;
    implementation.gate_up = Bfloat16Linear_vulkan::create_with_allocator(gate_up, gate_up_bias,
                                                                          vulkan_device_index,
                                                                          weight_allocator,
                                                                          vulkan_runtime,
                                                                          optimization_flags,
                                                                          upload_batch);
    implementation.down = Bfloat16Linear_vulkan::create_with_allocator(down, down_bias,
                                                                       vulkan_device_index,
                                                                       weight_allocator,
                                                                       vulkan_runtime,
                                                                       optimization_flags,
                                                                       upload_batch);
    if (!implementation.gate_up || !implementation.down)
        return {};

    const Bfloat16Linear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const Bfloat16Linear_vulkan::Implementation& down_projection = *implementation.down->d;
    if (gate.vulkan_context != down_projection.vulkan_context
        || gate.output_columns != gate_up.shape[0]
        || down_projection.input_columns != gate_up.shape[0] / 2
        || down_projection.output_columns != down.shape[0])
    {
        return {};
    }

    implementation.vulkan_context = gate.vulkan_context;
    implementation.option = gate.option;
    implementation.input_columns = gate.input_columns;
    implementation.intermediate_columns = gate.output_columns / 2;
    implementation.output_columns = down_projection.output_columns;
    implementation.activation_limit = activation_limit;
    implementation.activation = activation;

    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    if (!create_bfloat16_swiglu_down_pipeline(implementation.vulkan_context,
                                              gate.option,
                                              gate.swiglu_down_pipeline))
    {
        return {};
    }
    return result;
}

bool Bfloat16Expert_vulkan::forward(const ActivationBuffer& input,
                                    ActivationBuffer& output) const
{
    const Implementation& implementation = *d;
    return implementation.gate_up->forward_swiglu_chain(input,
                                                        *implementation.down,
                                                        implementation.intermediate_columns,
                                                        implementation.activation,
                                                        implementation.activation_limit,
                                                        false,
                                                        output);
}

std::vector<uint64_t> Bfloat16Expert_vulkan::scratch_elements(uint32_t gate_output_columns,
                                                              uint64_t rows)
{
    if (gate_output_columns == 0 || rows == 0)
        return {};
    return {saturating_multiply_u64(gate_output_columns, rows)};
}

bool Bfloat16Expert_vulkan::record(const ncnn::VkMat& input, ncnn::VkMat& output,
                                   ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace,
                                   std::vector<ncnn::VkMat>& bindings,
                                   std::vector<ncnn::vk_constant_type>& constants) const
{
    const Implementation& expert = *d;
    const Bfloat16Linear_vulkan::Implementation& gate = *expert.gate_up->d;
    const Bfloat16Linear_vulkan::Implementation& down = *expert.down->d;

    ncnn::VkMat fused(static_cast<int>(gate.output_columns), input.h,
                      sizeof(float), expert.vulkan_context->blob_allocator());
    if (fused.empty())
        return false;
    workspace.push_back(fused);

    (void)gate.record_projection(input, fused, static_cast<uint32_t>(input.h),
                                 command, bindings, constants);
    record_swiglu_down(gate.swiglu_down_pipeline.get(), fused, down, output,
                       expert.intermediate_columns, gate.output_columns,
                       static_cast<uint32_t>(input.h),
                       expert.activation, expert.activation_limit, false,
                       command, bindings, constants);
    return true;
}

bool Bfloat16Expert_vulkan::forward_batch(std::span<const Bfloat16Expert_vulkan*> experts,
                                          std::span<const ActivationBuffer*> inputs,
                                          std::span<ActivationBuffer*> outputs)
{
    const Implementation& first_expert = *experts.front()->d;
    const Bfloat16Linear_vulkan::Implementation& first_gate = *first_expert.gate_up->d;
    const Bfloat16Linear_vulkan::Implementation& first_down = *first_expert.down->d;
    const uint32_t input_columns = first_gate.input_columns;
    const uint32_t output_columns = first_down.output_columns;

    size_t total_rows = 0;
    bool shared_input = true;
    for (size_t index = 0; index < experts.size(); ++index)
    {
        const ActivationBuffer* input = inputs[index];
        total_rows += input->rows();
        shared_input = shared_input && input == inputs.front();
    }

    const size_t upload_rows = shared_input ? inputs.front()->rows() : total_rows;
    VulkanRuntimeState& runtime_state = first_expert.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = first_expert.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    if (!prepare_staging_batch(transfer_slot.upload, upload_rows, input_columns,
                               transfer_slot.staging_allocator, sizeof(float))
        || !prepare_staging_batch(transfer_slot.download, total_rows, output_columns,
                                  transfer_slot.staging_allocator, sizeof(float)))
    {
        return false;
    }

    ncnn::Mat mapped_input = transfer_slot.upload.mapped();
    if (mapped_input.empty())
        return false;
    auto* mapped_input_data = static_cast<std::byte*>(mapped_input.data);
    size_t row_offset = 0;
    for (const ActivationBuffer* input : inputs)
    {
        const size_t input_size = input->rows() * static_cast<size_t>(input_columns) * sizeof(float);
        std::memcpy(mapped_input_data + row_offset * static_cast<size_t>(input_columns) * sizeof(float),
                    input->bytes().data(),
                    input_size);
        if (shared_input)
            break;
        row_offset += input->rows();
    }
    (void)bind_host_in(transfer_slot.upload);

    std::unique_lock<std::mutex> lock(first_expert.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used && command.reset() != 0)
        return false;
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (!record_mapped_act_upload(transfer_slot.upload, input_gpu, command,
                                  first_expert.vulkan_context->device(),
                                  first_gate.option, DType::Float32))
    {
        return false;
    }

    ncnn::VkMat output_gpu(static_cast<int>(output_columns),
                           static_cast<int>(total_rows),
                           act_element_size(first_gate.option),
                           first_expert.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    const size_t descriptor_alignment = std::max<size_t>(4, first_expert.vulkan_context->device()->info.buffer_offset_alignment());
    const bool align_output_rows = experts.size() > 1
                                   && (static_cast<size_t>(output_columns) * output_gpu.elemsize) % descriptor_alignment != 0;
    std::vector<ncnn::VkMat> aligned_inputs;
    std::vector<ncnn::VkMat> aligned_outputs(align_output_rows ? experts.size() : 0);
    std::vector<ncnn::VkMat> intermediates;
    intermediates.reserve(experts.size());
    std::vector<ncnn::VkMat> bindings(4);
    std::vector<ncnn::vk_constant_type> constants(5);

    row_offset = 0;
    for (size_t index = 0; index < experts.size(); ++index)
    {
        const Implementation& expert = *experts[index]->d;
        const Bfloat16Linear_vulkan::Implementation& gate = *expert.gate_up->d;
        const Bfloat16Linear_vulkan::Implementation& down = *expert.down->d;
        const size_t rows = inputs[index]->rows();

        ncnn::VkMat input_view = aligned_row_view(input_gpu, shared_input ? 0 : row_offset, rows,
                                                  command, first_gate.option,
                                                  first_expert.vulkan_context, aligned_inputs, experts.size());
        ncnn::VkMat output_view = row_view(output_gpu, row_offset, rows);
        if (input_view.empty() || output_view.empty())
            return false;
        if (align_output_rows)
        {
            aligned_outputs[index].create(output_view.w, output_view.h, output_view.elemsize,
                                          first_expert.vulkan_context->blob_allocator());
            output_view = aligned_outputs[index];
            if (output_view.empty())
                return false;
        }

        ncnn::VkMat& fused_gpu = intermediates.emplace_back(static_cast<int>(gate.output_columns),
                                                            static_cast<int>(rows),
                                                            act_element_size(first_gate.option),
                                                            first_expert.vulkan_context->blob_allocator());
        if (fused_gpu.empty())
            return false;

        (void)gate.record_projection(input_view, fused_gpu, static_cast<uint32_t>(rows),
                                     command, bindings, constants);
        record_swiglu_down(gate.swiglu_down_pipeline.get(), fused_gpu, down, output_view,
                           expert.intermediate_columns, gate.output_columns,
                           static_cast<uint32_t>(rows),
                           expert.activation, expert.activation_limit, false,
                           command, bindings, constants);
        if (align_output_rows)
        {
            ncnn::VkMat destination = row_view(output_gpu, row_offset, rows);
            if (!record_row_copy(output_view, destination, command, first_gate.option))
                return false;
        }
        row_offset += rows;
    }

    if (!record_staging_download(output_gpu, total_rows, output_columns,
                                 transfer_slot.download, command,
                                 first_expert.vulkan_context->device(),
                                 first_down.option, DType::Float32)
        || submit_wait(command, first_expert.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batches(transfer_slot.download, inputs, outputs, output_columns))
    {
        return false;
    }

    lock.unlock();
    runtime_state.dispatches += static_cast<uint64_t>(experts.size()) * 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
}

} // namespace moe
} // namespace ncnn
