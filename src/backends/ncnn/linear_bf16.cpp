#include "linear.h"
#include <command.h>
#include <layer.h>
#include <layer/gemm.h>
#include <layer_type.h>
#include "sharedffn_vulkan.h"
#include "mxfp4_weightpacking.h"
#include "rmsnorm_vulkan.h"
#include "vulkancontext.h"
#include "experttransfer_vulkan.h"
#include "expertbackend_vulkan.h"
#include "innerproductweights_vulkan.h"
#include "moecombine_vulkan.h"
#include "residentffn_vulkan.h"

#include "ncnn/moe/runtime.h"
#include "engine/expertbackend.h"
#include "kernels/float8.h"
#include "kernels/int4.h"
#include "kernels/mxfp4.h"
#include "kernels/ops.h"
#include "kernels/qnk.h"

#if NCNN_MOE_WITH_VULKAN
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#if NCNN_MOE_USE_NCNN
#include <layer.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif
#endif

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/projection_matvec.comp.hex.h"
#include "kernels/vulkan/bfloat16_cooperative_projection.comp.hex.h"
#include "kernels/vulkan/bfloat16_projection.comp.hex.h"
#include "kernels/vulkan/bfloat16_rms_norm_projection.comp.hex.h"
#include "kernels/vulkan/bfloat16_row_rms.comp.hex.h"
#include "kernels/vulkan/bfloat16_swiglu_down.comp.hex.h"
#include "kernels/vulkan/bfloat16_tile_64x64.comp.hex.h"
#include "kernels/vulkan/bfloat16_tile_32x64.comp.hex.h"
#endif
#include "linear_impl.h"
#include "linear_internal.h"

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN

static bool create_bfloat16_projection_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                const ncnn::Option& option,
                                                std::shared_ptr<ncnn::Pipeline>& destination)
{
    const size_t shader_variant = (option.use_subgroup_ops ? 1u : 0u) * 3
                                  + activation_variant(option);
    ncnn::Option compile_option = option;
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(bfloat16_projection_shader,
                                                                                      static_cast<int>(sizeof(bfloat16_projection_shader) - 1),
                                                                                      compile_option,
                                                                                      shader_variant);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(bfloat16_projection_shader,
                                         shader_variant);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context = context](ncnn::Pipeline* value) mutable {
                                                      {
                                                          const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                          delete value;
                                                      }
                                                      // Weak cache entries retain the deleter control block after disposal.
                                                      context.reset();
                                                  });
    context->cache_pipeline(bfloat16_projection_shader,
                            shader_variant,
                            destination);
    return true;
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
    if (selected_m <= 0
        || selected_n <= 0
        || selected_k <= 0
        || selected_subgroup_size <= 0
        || selected_m % 4 != 0
        || selected_n % 4 != 0
        || selected_k % 4 != 0)
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
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(bfloat16_cooperative_projection_shader,
                                                                                      static_cast<int>(sizeof(bfloat16_cooperative_projection_shader) - 1),
                                                                                      compile_option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    const uint64_t pipeline_key = static_cast<uint64_t>(selected_m)
                                  | static_cast<uint64_t>(selected_n) << 16
                                  | static_cast<uint64_t>(selected_k) << 32
                                  | static_cast<uint64_t>(selected_subgroup_size) << 48;
    destination = context->find_pipeline(bfloat16_cooperative_projection_shader,
                                         pipeline_key);
    if (destination)
    {
        tile_m = static_cast<uint32_t>(selected_m);
        tile_n = static_cast<uint32_t>(selected_n);
        tile_k = static_cast<uint32_t>(selected_k);
        subgroup_size = static_cast<uint32_t>(selected_subgroup_size);
        return true;
    }

    std::vector<ncnn::vk_specialization_type> specializations(3);
    specializations[0].u32 = static_cast<uint32_t>(selected_m);
    specializations[1].u32 = static_cast<uint32_t>(selected_n);
    specializations[2].u32 = static_cast<uint32_t>(selected_k);
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(selected_subgroup_size);
    pipeline->set_local_size_xyz(selected_subgroup_size, 1, 1);
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context = context](ncnn::Pipeline* value) mutable {
                                                      {
                                                          const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                          delete value;
                                                      }
                                                      // Weak cache entries retain the deleter control block after disposal.
                                                      context.reset();
                                                  });
    context->cache_pipeline(bfloat16_cooperative_projection_shader,
                            pipeline_key,
                            destination);
    tile_m = static_cast<uint32_t>(selected_m);
    tile_n = static_cast<uint32_t>(selected_n);
    tile_k = static_cast<uint32_t>(selected_k);
    subgroup_size = static_cast<uint32_t>(selected_subgroup_size);
    return true;
}

static bool create_bfloat16_rms_norm_projection_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                         const ncnn::Option& option,
                                                         std::shared_ptr<ncnn::Pipeline>& destination,
                                                         uint32_t& output_subgroups)
{
    const size_t shader_variant = (option.use_subgroup_ops ? 1u : 0u) * 3
                                  + activation_variant(option);
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(bfloat16_rms_norm_projection_shader,
                                                                                      static_cast<int>(sizeof(bfloat16_rms_norm_projection_shader) - 1),
                                                                                      option,
                                                                                      shader_variant);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    const uint32_t local_size = device->info.max_workgroup_invocations() >= 256
                                        && device->info.max_workgroup_size_x() >= 256
                                    ? 256
                                    : 128;
    output_subgroups = local_size / 32;
    destination = context->find_pipeline(bfloat16_rms_norm_projection_shader,
                                         shader_variant);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(static_cast<int>(local_size), 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context = context](ncnn::Pipeline* value) mutable {
                                                      {
                                                          const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                          delete value;
                                                      }
                                                      // Weak cache entries retain the deleter control block after disposal.
                                                      context.reset();
                                                  });
    context->cache_pipeline(bfloat16_rms_norm_projection_shader,
                            shader_variant,
                            destination);
    return true;
}

static bool create_bfloat16_swiglu_down_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                 const ncnn::Option& option,
                                                 std::shared_ptr<ncnn::Pipeline>& destination)
{
    const size_t shader_variant = (option.use_subgroup_ops ? 1u : 0u) * 3
                                  + activation_variant(option);
    ncnn::Option compile_option = option;
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(bfloat16_swiglu_down_shader,
                                                                                      static_cast<int>(sizeof(bfloat16_swiglu_down_shader) - 1),
                                                                                      compile_option,
                                                                                      shader_variant);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(bfloat16_swiglu_down_shader,
                                         shader_variant);
    if (destination)
    {
        return true;
    }
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_subgroup_size(32);
    pipeline->set_local_size_xyz(32, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(),
                         spirv->size() * sizeof(uint32_t),
                         specializations)
        != 0)
    {
        return false;
    }
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(),
                                                  [context = context](ncnn::Pipeline* value) mutable {
                                                      {
                                                          const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                          delete value;
                                                      }
                                                      // Weak cache entries retain the deleter control block after disposal.
                                                      context.reset();
                                                  });
    context->cache_pipeline(bfloat16_swiglu_down_shader,
                            shader_variant,
                            destination);
    return true;
}

#if NCNN_MOE_WITH_VULKAN
bool create_tile_gemm(const std::shared_ptr<VulkanContext>& context,
                      const ncnn::Option& option,
                      TileGemm (&pipes)[2])
{
    // One kernel body, two compile-time geometries: a narrow output tile would
    // otherwise idle most of the last wave.
    struct geometry
    {
        const char* source;
        int length;
        uint32_t bm;
        uint32_t bn;
        uint32_t wg;
    };
    const geometry geometries[2] = {
        {bfloat16_tile_64x64_shader, static_cast<int>(sizeof(bfloat16_tile_64x64_shader) - 1), 64, 64, 256},
        {bfloat16_tile_32x64_shader, static_cast<int>(sizeof(bfloat16_tile_32x64_shader) - 1), 32, 64, 128},
    };

    ncnn::Option compile_option = option;
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    const size_t variant = (option.use_subgroup_ops ? 1u : 0u) * 3 + activation_variant(option);
    ncnn::VulkanDevice* device = context->device();
    bool any = false;
    for (size_t index = 0; index < 2; ++index)
    {
        const geometry& item = geometries[index];
        const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(item.source, item.length, compile_option, variant);
        if (!spirv || spirv->empty())
            continue;
        std::shared_ptr<ncnn::Pipeline> pipeline = context->find_pipeline(item.source, variant);
        if (!pipeline)
        {
            std::unique_ptr<ncnn::Pipeline> created(new ncnn::Pipeline(device));
            created->set_subgroup_size(32);
            created->set_local_size_xyz(static_cast<int>(item.wg), 1, 1);
            const std::vector<ncnn::vk_specialization_type> specializations;
            if (created->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
                continue;
            pipeline = std::shared_ptr<ncnn::Pipeline>(created.release(),
                                                       [context = context](ncnn::Pipeline* value) mutable {
                                                           {
                                                               const std::lock_guard<std::mutex> lock(context->command_mutex());
                                                               delete value;
                                                           }
                                                           context.reset();
                                                       });
            context->cache_pipeline(item.source, variant, pipeline);
        }
        pipes[index].pipeline = std::move(pipeline);
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
        || (!row_scale.empty() && (row_scale.w < static_cast<int>(shape.rows) || row_scale.elempack != 1))
        || (!col_scale.empty() && (col_scale.w < static_cast<int>(shape.k) || col_scale.elempack != 1))
        || in.empty() || in.dims != 2 || in.elempack != 1 || in.elemsize != sizeof(float)
        || in.w != static_cast<int>(static_cast<uint64_t>(shape.k) * shape.groups)
        || in.h != static_cast<int>(shape.rows)
        // vec4 activation loads: the binding must be 16 B aligned and the rows contiguous.
        || (in.offset & 15u) != 0 || in.cstep != static_cast<size_t>(in.w) * in.h
        || out.empty() || out.dims != 2 || out.elempack != 1 || out.elemsize != sizeof(float)
        || out.w != static_cast<int>(stride) || out.h != static_cast<int>(shape.rows)
        || w.empty() || bias.empty()
        || w.buffer_capacity() < static_cast<uint64_t>(shape.k) * shape.n * sizeof(uint16_t)
        || bias.buffer_capacity() < static_cast<uint64_t>(shape.n) * sizeof(float)
        || out.buffer_capacity() < static_cast<uint64_t>(stride) * shape.rows * sizeof(float))
        return false;

    // Take the first geometry whose grid still fills the device. 160 workgroups
    // is four waves on the 40-core class of device this runtime targets; the
    // 32x64 geometry doubles the count for narrow output tiles, which otherwise
    // idle most of their last wave. A device-derived threshold needs the shared
    // parallelism query that attentiondecode_vulkan.cpp keeps private.
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
#endif

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
#endif

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
        return 0;
    const uint64_t elements = static_cast<uint64_t>(matrix.shape[0]) * matrix.shape[1];
    if (elements > static_cast<uint64_t>(std::numeric_limits<int>::max()) * 2
        || elements > std::numeric_limits<size_t>::max() / sizeof(uint16_t))
        return 0;

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

bool Bfloat16Linear_vulkan::prepare_rms_norm(const TensorData& weight,
                                             float epsilon,
                                             float weight_offset)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || implementation.rms_norm_projection_pipeline
        || epsilon <= 0.0f
        || weight.shape.size() != 1 || weight.shape[0] != implementation.input_columns
        || (weight.dtype != DType::Float32
            && weight.dtype != DType::BFloat16))
    {
        return false;
    }

    ncnn::Mat values;
    values.create(static_cast<int>(implementation.input_columns), sizeof(float));
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

    if (!create_bfloat16_rms_norm_projection_pipeline(implementation.vulkan_context,
                                                      implementation.option,
                                                      implementation.rms_norm_projection_pipeline,
                                                      implementation.rms_norm_output_subgroups))
        return false;

    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(implementation.vulkan_context->device()));
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    bool uploaded = false;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(implementation.vulkan_context->device());
        command.record_upload(values,
                              implementation.rms_norm_weight,
                              upload_option);
        uploaded = !implementation.rms_norm_weight.empty()
                   && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
    {
        implementation.rms_norm_projection_pipeline.reset();
        return false;
    }
    implementation.rms_norm_epsilon = epsilon;
    return true;
#else
    (void)weight;
    (void)epsilon;
    (void)weight_offset;
    return false;
#endif
}

bool Bfloat16Linear_vulkan::has_rms_norm_chain() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->rms_norm_projection_pipeline != nullptr
           && !d->rms_norm_weight.empty();
#else
    return false;
#endif
}

bool Bfloat16Linear_vulkan::prepare_input_rms_norm(const TensorData& weight, float epsilon, float weight_offset)
{
#if NCNN_MOE_WITH_VULKAN
    if (!d->vulkan_context || weight.shape.size() != 1 || weight.shape[0] != d->input_columns || d->input_rms_norm)
        return false;
    d->input_rms_norm = RmsNorm_vulkan::create(weight, epsilon, weight_offset, d->vulkan_context, d->option);
    return static_cast<bool>(d->input_rms_norm);
#else
    (void)weight;
    (void)epsilon;
    return false;
#endif
}

bool Bfloat16Linear_vulkan::prepare_output_rms_norm(const TensorData& weight, float epsilon)
{
#if NCNN_MOE_WITH_VULKAN
    if (!d->vulkan_context || weight.shape.size() != 1 || weight.shape[0] != d->output_columns || d->output_rms_norm)
        return false;
    d->output_rms_norm = RmsNorm_vulkan::create(weight, epsilon, 0.0f, d->vulkan_context, d->option);
    return static_cast<bool>(d->output_rms_norm);
#else
    (void)weight;
    (void)epsilon;
    return false;
#endif
}

#if NCNN_MOE_WITH_VULKAN
bool Bfloat16Linear_vulkan::record_rms_norm_chain(const ncnn::VkMat& input,
                                                  ncnn::VkMat& output,
                                                  ncnn::VkCompute& command) const
{
    const Implementation& implementation = *d;
    if (&input == &output || !implementation.vulkan_context || !implementation.pipeline
        || !has_rms_norm_chain() || implementation.rms_norm_output_subgroups == 0
        || input.empty() || input.dims != 2 || input.w != static_cast<int>(implementation.input_columns)
        || input.h <= 0 || input.elempack != 1 || input.elemsize != sizeof(float)
        || !input.allocator || input.allocator->vkdev != implementation.vulkan_context->device()
        || activation_variant(implementation.option) != 0)
        return false;
    const uint64_t dispatch_width = ((static_cast<uint64_t>(implementation.output_columns)
                                      + implementation.rms_norm_output_subgroups - 1)
                                     / implementation.rms_norm_output_subgroups)
                                    * implementation.rms_norm_output_subgroups * 32;
    if (dispatch_width > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;

    // Keep a distinct destination even if a caller supplies another view of
    // input. The fused shader still needs the original input while writing.
    ncnn::VkMat destination;
    destination.create(static_cast<int>(implementation.output_columns), input.h, sizeof(float),
                       implementation.vulkan_context->blob_allocator());
    if (destination.empty()) return false;
    record_rms_norm_projection(input, destination, command);
    output = std::move(destination);
    return true;
}

bool Bfloat16Linear_vulkan::record_input_normalization(const ncnn::VkMat& input, ncnn::VkMat& output, ncnn::VkCompute& cmd) const
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
    if (!d->output_rms_norm || d->vulkan_context != next.d->vulkan_context
        || d->vulkan_context != parallel.d->vulkan_context
        || activation_variant(d->option) != 0
        || activation_variant(next.d->option) != 0
        || activation_variant(parallel.d->option) != 0)
        return false;
    ncnn::VkMat intermediate;
    if (forward(input, intermediate, cmd, d->option) != 0
        || !d->output_rms_norm->record(intermediate, query_rank, cmd)
        || next.forward(query_rank, query, cmd, next.d->option) != 0
        || parallel.forward(input, kv, cmd, parallel.d->option) != 0)
        return false;
    workspace.push_back(std::move(intermediate));
    return true;
}
#endif

bool Bfloat16Linear_vulkan::forward_rms_norm_chain(const ActivationBuffer& input,
                                                   ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || !implementation.rms_norm_projection_pipeline
        || implementation.rms_norm_weight.empty()
        || implementation.rms_norm_output_subgroups == 0
        || input.rows() == 0
        || input.columns() != implementation.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();

    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = activation_variant(implementation.option) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                    * implementation.input_columns * sizeof(float),
                                                                                                input.dtype());
    const bool direct_host_output = activation_variant(implementation.option) == 0
                                    && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                     * implementation.output_columns * sizeof(float),
                                                                                                 output.dtype());
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  implementation.output_columns,
                                  transfer_slot.staging_allocator))
    {
        return false;
    }
    output.reset(input.rows(), implementation.output_columns, false);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
        {
            return false;
        }
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload);
    else if (!record_mapped_act_upload(transfer_slot.upload,
                                       input_gpu,
                                       command,
                                       implementation.vulkan_context->device(),
                                       implementation.option,
                                       input.dtype()))
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
    const std::vector<ncnn::VkMat> bindings = {
        input_gpu,
        implementation.packed,
        implementation.bias,
        implementation.rms_norm_weight,
        output_gpu};
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.rows());
    constants[4].f = implementation.rms_norm_epsilon;
    constants[5].u32 = implementation.rms_norm_output_subgroups;
    const uint64_t dispatch_width = ((static_cast<uint64_t>(implementation.output_columns)
                                      + implementation.rms_norm_output_subgroups - 1)
                                     / implementation.rms_norm_output_subgroups)
                                    * implementation.rms_norm_output_subgroups * 32;
    if (dispatch_width > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(dispatch_width);
    dispatcher.h = static_cast<int>(input.rows());
    dispatcher.c = 1;
    command.record_pipeline(implementation.rms_norm_projection_pipeline.get(),
                            bindings,
                            constants,
                            dispatcher);
    if ((!direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     implementation.output_columns,
                                     transfer_slot.download,
                                     command,
                                     implementation.vulkan_context->device(),
                                     implementation.option,
                                     output.dtype()))
        || submit_compute_and_wait(command, implementation.vulkan_context->device()) != 0
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
#else
    (void)input;
    (void)output;
    return false;
#endif
}

std::shared_ptr<Bfloat16Linear_vulkan>
Bfloat16Linear_vulkan::create(const TensorData& matrix,
                              const TensorData* bias,
                              uint32_t vulkan_device_index,
                              const VulkanRuntimePtr& vulkan_runtime,
                              uint64_t optimization_flags)
{
    return create_with_allocator(matrix,
                                 bias,
                                 vulkan_device_index,
                                 nullptr,
                                 vulkan_runtime,
                                 optimization_flags);
}

bool create_coop_gemm(const std::shared_ptr<VulkanContext>& context,
                      const ncnn::Option& base,
                      uint32_t input_columns,
                      uint32_t output_columns,
                      std::span<const uint16_t> weights,
                      std::shared_ptr<ncnn::Layer>& destination,
                      ncnn::Option& destination_option)
{
#if NCNN_MOE_WITH_VULKAN
    if (!context)
        return false;
    ncnn::VulkanDevice* device = context->device();
    if (!device->info.support_cooperative_matrix()
        || weights.size() != static_cast<size_t>(input_columns) * output_columns)
        return false;
    ncnn::Layer* raw = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
    if (!raw)
        return false;

    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_fp16_storage = true;
    option.use_fp16_packed = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_storage = false;
    option.use_cooperative_matrix = true;
    option.use_subgroup_ops = device->info.support_subgroup_ops();
    option.num_threads = base.num_threads;
    option.blob_vkallocator = base.blob_vkallocator;
    option.workspace_vkallocator = base.workspace_vkallocator;
    option.staging_vkallocator = base.staging_vkallocator;

    std::shared_ptr<ncnn::Layer> layer(raw, [option](ncnn::Layer* value) {
        if (!value) return;
        value->destroy_pipeline(option);
        delete value;
    });
    raw->vkdev = device;

    ncnn::ParamDict pd;
    pd.set(0, 1.f); // alpha
    pd.set(1, 0.f); // beta
    pd.set(2, 0);   // transA
    pd.set(3, 1);   // transB: B rows are outputs
    pd.set(4, 0);   // constantA
    pd.set(5, 1);   // constantB
    pd.set(6, 0);   // constantC
    pd.set(8, static_cast<int>(output_columns));
    pd.set(9, static_cast<int>(input_columns));
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
    ncnn::VkTransfer upload(device);
    if (raw->create_pipeline(option) != 0 || raw->upload_model(upload, option) != 0
        || upload.submit_and_wait() != 0)
        return false;
    destination = layer;
    destination_option = option;
    return true;
#else
    (void)context;
    (void)base;
    (void)input_columns;
    (void)output_columns;
    (void)weights;
    (void)destination;
    (void)destination_option;
    return false;
#endif
}

std::shared_ptr<Bfloat16Linear_vulkan>
Bfloat16Linear_vulkan::create_with_allocator(const TensorData& matrix,
                                             const TensorData* bias,
                                             uint32_t vulkan_device_index,
                                             ncnn::VkAllocator* weight_allocator,
                                             const VulkanRuntimePtr& vulkan_runtime,
                                             uint64_t optimization_flags,
                                             VulkanWeightUploadBatch* upload_batch)
{
#if NCNN_MOE_WITH_VULKAN
    if (matrix.dtype != DType::BFloat16
        || matrix.shape.size() != 2
        || matrix.shape[0] == 0
        || matrix.shape[1] == 0
        || matrix.shape[1] % 4 != 0
        || matrix.shape[0]
               > static_cast<uint32_t>(std::numeric_limits<int>::max() / 32)
        || matrix.shape[1]
               > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const std::span<const uint16_t> weights = matrix.bfloat16_values();
    if (weights.size() != matrix.element_count())
        return {};
    if (bias
        && (bias->shape.size() != 1
            || bias->shape[0] != output_columns
            || (bias->dtype != DType::Float32
                && bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    std::shared_ptr<Bfloat16Linear_vulkan> result(new Bfloat16Linear_vulkan);
    if (upload_batch) upload_batch->retain_owner(result);
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
    if (device->info.subgroup_size() != 32)
        return {};
    implementation.option.vulkan_device_index = device->info.device_index();
    implementation.option.use_vulkan_compute = true;
    implementation.option.use_fp16_packed = false;
    implementation.option.use_fp16_storage = false;
    implementation.option.use_fp16_arithmetic = false;
    implementation.option.use_bf16_packed = false;
    implementation.option.use_bf16_storage = false;
    implementation.option.blob_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.workspace_vkallocator = implementation.vulkan_context->blob_allocator();
    implementation.option.staging_vkallocator = implementation.vulkan_context->staging_allocator();
    implementation.option.use_cooperative_matrix = device->info.support_cooperative_matrix();
    implementation.option.use_subgroup_ops = device->info.support_subgroup_ops();

    if (storage_bytes(matrix, device->info.buffer_offset_alignment()) == 0)
    {
        return {};
    }
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
    ncnn::Mat biases;
    if (!prepare_bfloat16_upload(weights, packed))
        return {};
    biases.create(static_cast<int>(output_columns), sizeof(float));
    if (biases.empty())
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

    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_bfloat16_projection_pipeline(implementation.vulkan_context,
                                                 implementation.option,
                                                 implementation.pipeline))
        {
            return {};
        }
        create_tile_gemm(implementation.vulkan_context, implementation.option, implementation.tile_gemm);
        if (!implementation.row_rms_scale_pipeline)
        {
            constexpr size_t row_rms_variant = 0;
            const std::shared_ptr<const std::vector<uint32_t>> rms_spirv = implementation.vulkan_context->shader_binary(bfloat16_row_rms_shader,
                                                                                                                        static_cast<int>(sizeof(bfloat16_row_rms_shader) - 1),
                                                                                                                        implementation.option,
                                                                                                                        row_rms_variant);
            if (rms_spirv && !rms_spirv->empty())
            {
                std::unique_ptr<ncnn::Pipeline> rms_pipeline(new ncnn::Pipeline(device));
                rms_pipeline->set_subgroup_size(32);
                rms_pipeline->set_local_size_xyz(256, 1, 1);
                const std::vector<ncnn::vk_specialization_type> rms_specializations;
                if (rms_pipeline->create(rms_spirv->data(),
                                         rms_spirv->size() * sizeof(uint32_t),
                                         rms_specializations)
                    == 0)
                {
                    implementation.row_rms_scale_pipeline = std::shared_ptr<ncnn::Pipeline>(rms_pipeline.release(),
                                                                                            [ctx = implementation.vulkan_context](ncnn::Pipeline* value) mutable {
                                                                                                {
                                                                                                    const std::lock_guard<std::mutex> lock(ctx->command_mutex());
                                                                                                    delete value;
                                                                                                }
                                                                                                ctx.reset();
                                                                                            });
                }
            }
        }
        if (has_flag(implementation.optimization_flags,
                     OptimizationVulkanBfloat16CoopMatrix))
        {
            create_bfloat16_cooperative_projection_pipeline(implementation.vulkan_context,
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
                                   input_columns, output_columns, implementation.single_row_pipeline)) return {};
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = weight_allocator;
    upload_option.workspace_vkallocator = weight_allocator;
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    ncnn::Option packed_upload_option = upload_option;
    packed_upload_option.use_fp16_storage = false;
    packed_upload_option.use_bf16_storage = false;
    bool uploaded = false;
    if (upload_batch)
    {
        if (!dynamic_cast<VulkanExpertWeightAllocator*>(weight_allocator)) return {};
        uploaded = upload_batch->record(packed, implementation.packed, packed_upload_option, weight_allocator)
                   && upload_batch->record(biases, implementation.bias, upload_option, weight_allocator);
    }
    else if (auto* independent_allocator = dynamic_cast<VulkanExpertWeightAllocator*>(weight_allocator))
    {
        VulkanIndependentWeightTransfer command(implementation.vulkan_context, implementation.weight_staging_allocator.get());
        uploaded = command.record(packed, implementation.packed, independent_allocator)
                   && command.record(biases, implementation.bias, independent_allocator)
                   && command.submit_and_wait();
    }
    else
    {
        ncnn::VkTransfer command(device);
        command.record_upload(packed, implementation.packed, packed_upload_option);
        command.record_upload(biases, implementation.bias, upload_option);
        uploaded = !implementation.packed.empty() && !implementation.bias.empty() && command.submit_and_wait() == 0;
    }
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return {};
    {
        // Pipeline creation and the weight upload both submit through the shared
        // context; the QNK path takes this same lock.
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        (void)create_coop_gemm(implementation.vulkan_context, implementation.option, input_columns, output_columns,
                               weights, implementation.coop_gemm, implementation.coop_opt);
    }
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)vulkan_runtime;
    (void)optimization_flags;
    (void)upload_batch;
    return {};
#endif
}

std::shared_ptr<Bfloat16Linear_vulkan>
Bfloat16Linear_vulkan::create_fused(const std::vector<const TensorData*>& matrices,
                                    const std::vector<const TensorData*>& biases,
                                    uint32_t vulkan_device_index,
                                    const VulkanRuntimePtr& vulkan_runtime,
                                    uint64_t optimization_flags)
{
    if (matrices.empty()
        || matrices.size() != biases.size()
        || !matrices.front()
        || matrices.front()->dtype != DType::BFloat16
        || matrices.front()->shape.size() != 2)
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
        if (!matrix
            || matrix->dtype != DType::BFloat16
            || matrix->shape.size() != 2
            || matrix->shape[1] != input_columns
            || (bias != nullptr) != has_bias
            || (bias
                && (bias->shape.size() != 1
                    || bias->shape[0] != matrix->shape[0]
                    || (bias->dtype != DType::Float32
                        && bias->dtype != DType::BFloat16))))
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
    fused_matrix.shape = {
        static_cast<uint32_t>(output_columns),
        input_columns};
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
            const std::span<const uint16_t> bias_values = bias.bfloat16_values();
            for (uint16_t value : bias_values)
                fused_bias.float32_data.push_back(bfloat16_to_float(value));
        }
    }
    return create(fused_matrix,
                  has_bias ? &fused_bias : nullptr,
                  vulkan_device_index,
                  vulkan_runtime,
                  optimization_flags);
}

#if NCNN_MOE_WITH_VULKAN
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
    bindings.resize(4);
    bindings[0] = input;
    bindings[1] = packed;
    bindings[2] = bias;
    bindings[3] = output;
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
        if (invocation_count
            <= static_cast<uint64_t>(std::numeric_limits<int>::max()))
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
                                             bindings,
                                             {1, 0, 0, 0},
                                             constants,
                                             dispatcher);
            return true;
        }
    }

    constants[0].u32 = input_columns;
    constants[1].u32 = output_columns;
    constants[2].u32 = block_count;
    constants[3].u32 = token_count;
    if (record_tile(input, output, token_count, ncnn::VkMat(), ncnn::VkMat(), command))
        return false;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(output_columns * 32);
    dispatcher.h = static_cast<int>(token_count);
    dispatcher.c = 1;
    command.record_pipeline_readonly(pipeline.get(),
                                     bindings,
                                     {1, 0, 0, 0},
                                     constants,
                                     dispatcher);
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
    return record_tile_gemm(tile_gemm, input, packed, bias, row_scale, column_scale, output, shape, command);
}
#endif

bool Bfloat16Linear_vulkan::record_coop(const ncnn::VkMat& input,
                                        ncnn::VkMat& output,
                                        ncnn::VkCompute& cmd,
                                        bool normalize) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.coop_gemm || !implementation.vulkan_context
        || (normalize && !implementation.input_rms_norm)
        || input.dims != 2 || input.elemsize != sizeof(float) || input.elempack != 1
        || input.h < 32 || input.w != static_cast<int>(implementation.input_columns)
        || output.dims != 2 || output.elemsize != sizeof(float) || output.elempack != 1
        || output.w != static_cast<int>(implementation.output_columns)
        || output.h != input.h)
    {
        return false;
    }
    // The Gemm path cannot fold the norm into its tile loads, so the prepared
    // normalization is materialized first when the caller asks for it.
    ncnn::VkMat source = input;
    if (normalize)
    {
        ncnn::VkMat normalized;
        if (!record_input_normalization(input, normalized, cmd) || normalized.empty())
            return false;
        source = normalized;
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
    output = projected;
    return true;
#else
    (void)input;
    (void)output;
    (void)cmd;
    return false;
#endif
}

void Bfloat16Linear_vulkan::record_scalar_projection(const ncnn::VkMat& input,
                                                     ncnn::VkMat& output,
                                                     ncnn::VkCompute& cmd) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (implementation.single_row_pipeline && single_row_input(input, implementation.input_columns)
        && output.elempack == 1 && output.elemsize == sizeof(float))
    {
        record_matvec(implementation.single_row_pipeline.get(), true, implementation.input_columns,
                      implementation.output_columns, true, input, implementation.packed,
                      implementation.bias, output, cmd);
        return;
    }
    if (record_coop(input, output, cmd))
        return;
    const std::vector<ncnn::VkMat> bindings = {
        input,
        implementation.packed,
        implementation.bias,
        output};
    if (implementation.record_tile(input, output, static_cast<uint32_t>(input.h),
                                   ncnn::VkMat(), ncnn::VkMat(), cmd))
        return;
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
                                 bindings,
                                 {1, 0, 0, 0},
                                 constants,
                                 dispatcher);
#else
    (void)input;
    (void)output;
    (void)cmd;
#endif
}

void Bfloat16Linear_vulkan::record_rms_norm_projection(const ncnn::VkMat& input,
                                                       ncnn::VkMat& output,
                                                       ncnn::VkCompute& cmd,
                                                       ncnn::VkMat* row_scale_workspace) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (row_scale_workspace
        && implementation.row_rms_scale_pipeline
        && implementation.vulkan_context
        && !implementation.rms_norm_weight.empty()
        && input.h >= 8
        && implementation.input_columns % 16 == 0)
    {
        // Exactly `token_count` floats of graph-arena scratch, retained by the
        // caller until the submission completes (GatedDeltaWork keeps it). The
        // per-token byte count is far below the arena slack reserved for the
        // workspace peak, so this allocation never grows the plan.
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
                                 bindings,
                                 {1, 0, 0, 0, 0},
                                 constants,
                                 dispatcher);
#else
    (void)input;
    (void)output;
    (void)cmd;
#endif
}

int Bfloat16Linear_vulkan::forward(const ncnn::VkMat& input,
                                   ncnn::VkMat& output,
                                   ncnn::VkCompute& cmd,
                                   const ncnn::Option& opt) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || input.empty()
        || input.dims != 2
        || input.w != static_cast<int>(implementation.input_columns)
        || input.h <= 0
        || input.elempack != 1
        || input.elemsize != sizeof(float))
    {
        return -1;
    }
#if NCNN_BATCH
    if (input.n != 1) return -1;
#endif
    if (input.buffer_capacity() < static_cast<uint64_t>(input.w) * input.h * sizeof(float)) return -1;
    if (&input == &output) return -1;

    output.create(static_cast<int>(implementation.output_columns),
                  input.h,
                  sizeof(float),
                  opt.blob_vkallocator);
    if (output.empty())
        return -100;
    if (output.data == input.data || output.buffer_capacity() < static_cast<uint64_t>(implementation.output_columns) * input.h * sizeof(float)) return -1;

    if (implementation.single_row_pipeline && single_row_input(input, implementation.input_columns))
    {
        record_matvec(implementation.single_row_pipeline.get(), true, implementation.input_columns,
                      implementation.output_columns, true, input, implementation.packed,
                      implementation.bias, output, cmd);
        return 0;
    }
    // Prefill takes the cooperative-matrix Gemm when the weights and shape fit;
    // this is the same reuse record_scalar_projection already applies.
    if (record_coop(input, output, cmd))
        return 0;
    if (implementation.record_tile(input, output, static_cast<uint32_t>(input.h),
                                   ncnn::VkMat(), ncnn::VkMat(), cmd))
        return 0;
    const std::vector<ncnn::VkMat> bindings = {
        input,
        implementation.packed,
        implementation.bias,
        output};
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = static_cast<uint32_t>(input.h);
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(implementation.output_columns * 32);
    dispatcher.h = input.h;
    dispatcher.c = 1;
    cmd.record_pipeline(implementation.pipeline.get(),
                        bindings,
                        constants,
                        dispatcher);
    return 0;
#else
    (void)input;
    (void)output;
    (void)cmd;
    (void)opt;
    return -1;
#endif
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& Bfloat16Linear_vulkan::vulkan_context() const noexcept
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
#endif

bool Bfloat16Linear_vulkan::forward(const ActivationBuffer& input,
                                    ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || input.rows() == 0
        || input.columns() != implementation.input_columns
        || input.rows()
               > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = activation_variant(implementation.option) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                    * implementation.input_columns * sizeof(float),
                                                                                                input.dtype());
    const bool direct_host_output = activation_variant(implementation.option) == 0
                                    && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows())
                                                                                                     * implementation.output_columns * sizeof(float),
                                                                                                 output.dtype());
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  implementation.output_columns,
                                  transfer_slot.staging_allocator))
    {
        return false;
    }
    output.reset(input.rows(), implementation.output_columns, false);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
        {
            return false;
        }
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload);
    else if (!record_mapped_act_upload(transfer_slot.upload,
                                       input_gpu,
                                       command,
                                       implementation.vulkan_context->device(),
                                       implementation.option,
                                       input.dtype()))
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
    const bool used_cooperative_matrix = implementation.record_projection(input_gpu,
                                                                          output_gpu,
                                                                          static_cast<uint32_t>(input.rows()),
                                                                          command,
                                                                          bindings,
                                                                          constants);
    if ((!direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     implementation.output_columns,
                                     transfer_slot.download,
                                     command,
                                     implementation.vulkan_context->device(),
                                     implementation.option,
                                     output.dtype()))
        || submit_compute_and_wait(command, implementation.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download,
                                      output))
    {
        return false;
    }
    ++runtime_state.dispatches;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    if (used_cooperative_matrix)
    {
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    }
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool Bfloat16Linear_vulkan::forward_parallel(const ActivationBuffer& input,
                                             const Bfloat16Linear_vulkan& parallel_operator,
                                             ActivationBuffer& output,
                                             ActivationBuffer& parallel_output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *d;
    const Implementation& parallel = *parallel_operator.d;
    if (!first.vulkan_context
        || !parallel.vulkan_context
        || first.vulkan_context.get() != parallel.vulkan_context.get()
        || !first.pipeline
        || !parallel.pipeline
        || activation_variant(first.option)
               != activation_variant(parallel.option)
        || input.rows() == 0
        || input.columns() != first.input_columns
        || input.columns() != parallel.input_columns
        || input.rows()
               > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    ncnn::VkMat parallel_download;
    const bool direct_host_input = activation_variant(first.option) == 0
                                   && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * first.input_columns
                                                                                           * sizeof(float),
                                                                                       input.dtype());
    const bool direct_host_output = activation_variant(first.option) == 0
                                    && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * first.output_columns
                                                                                            * sizeof(float),
                                                                                        output.dtype());
    const bool parallel_direct_host_output = activation_variant(parallel.option) == 0
                                             && parallel.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * parallel.output_columns
                                                                                                        * sizeof(float),
                                                                                                    parallel_output.dtype());
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  first.output_columns,
                                  transfer_slot.staging_allocator)
        || !prepare_staging_batch(parallel_download,
                                  input.rows(),
                                  parallel.output_columns,
                                  transfer_slot.staging_allocator))
    {
        return false;
    }
    output.reset(input.rows(), first.output_columns, false);
    parallel_output.reset(input.rows(), parallel.output_columns, false);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload);
    else if (!record_mapped_act_upload(transfer_slot.upload,
                                       input_gpu,
                                       command,
                                       first.vulkan_context->device(),
                                       first.option,
                                       input.dtype()))
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
    const bool first_used_cooperative_matrix = first.record_projection(input_gpu,
                                                                       output_gpu,
                                                                       static_cast<uint32_t>(input.rows()),
                                                                       command,
                                                                       bindings,
                                                                       constants);
    const bool parallel_used_cooperative_matrix = parallel.record_projection(input_gpu,
                                                                             parallel_output_gpu,
                                                                             static_cast<uint32_t>(input.rows()),
                                                                             command,
                                                                             bindings,
                                                                             constants);
    if ((!direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     first.output_columns,
                                     transfer_slot.download,
                                     command,
                                     first.vulkan_context->device(),
                                     first.option,
                                     output.dtype()))
        || (!parallel_direct_host_output
            && !record_staging_download(parallel_output_gpu,
                                        input.rows(),
                                        parallel.output_columns,
                                        parallel_download,
                                        command,
                                        parallel.vulkan_context->device(),
                                        parallel.option,
                                        parallel_output.dtype()))
        || submit_compute_and_wait(command, first.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output)
        || !copy_staging_to_cpu_batch(parallel_download, parallel_output))
    {
        return false;
    }
    runtime_state.dispatches += 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    runtime_state.batch_downloads += 2;
    if (first_used_cooperative_matrix)
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    if (parallel_used_cooperative_matrix)
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    return true;
#else
    (void)input;
    (void)parallel_operator;
    (void)output;
    (void)parallel_output;
    return false;
#endif
}

bool Bfloat16Linear_vulkan::forward_swiglu_chain(const ActivationBuffer& input,
                                                 const Bfloat16Linear_vulkan& down_operator,
                                                 uint32_t intermediate_columns,
                                                 ExpertActivation activation,
                                                 float activation_limit,
                                                 bool apply_router_gate,
                                                 ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& first = *d;
    const Implementation& down = *down_operator.d;
    const uint64_t expected_fused_columns = static_cast<uint64_t>(intermediate_columns) * 2
                                            + (apply_router_gate ? 1u : 0u);
    if ((activation != ExpertActivation::Silu && activation != ExpertActivation::DeepSeekSwiGlu)
        || !first.vulkan_context
        || !down.vulkan_context
        || first.vulkan_context.get() != down.vulkan_context.get()
        || activation_variant(first.option)
               != activation_variant(down.option)
        || !first.pipeline
        || first.output_columns != expected_fused_columns
        || down.input_columns != intermediate_columns
        || intermediate_columns == 0
        || intermediate_columns % 128 != 0
        || input.rows() == 0
        || input.columns() != first.input_columns
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
        || down.packed.empty()
        || down.bias.empty())
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = first.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = first.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = activation_variant(first.option) == 0
                                   && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * first.input_columns
                                                                                           * sizeof(float),
                                                                                       input.dtype());
    const bool direct_host_output = activation_variant(first.option) == 0
                                    && first.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * down.output_columns
                                                                                            * sizeof(float),
                                                                                        output.dtype());
    if (!fill_staging_upload(input,
                             transfer_slot.upload,
                             transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  down.output_columns,
                                  transfer_slot.staging_allocator))
    {
        return false;
    }
    output.reset(input.rows(), down.output_columns, false);

    std::unique_lock<std::mutex> lock(first.vulkan_context->command_mutex());
    if (!first.swiglu_down_pipeline
        && !create_bfloat16_swiglu_down_pipeline(first.vulkan_context,
                                                 first.option,
                                                 first.swiglu_down_pipeline))
    {
        return false;
    }
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload);
    else if (!record_mapped_act_upload(transfer_slot.upload,
                                       input_gpu,
                                       command,
                                       first.vulkan_context->device(),
                                       first.option,
                                       input.dtype()))
    {
        return false;
    }
    ncnn::VkMat fused_gpu;
    fused_gpu.create(static_cast<int>(first.output_columns),
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
    const bool used_cooperative_matrix = first.record_projection(input_gpu,
                                                                 fused_gpu,
                                                                 static_cast<uint32_t>(input.rows()),
                                                                 command,
                                                                 bindings,
                                                                 constants);

    bindings = {
        fused_gpu,
        down.packed,
        down.bias,
        output_gpu};
    constants.resize(6);
    constants[0].u32 = intermediate_columns;
    constants[1].u32 = first.output_columns;
    constants[2].u32 = down.output_columns;
    constants[3].u32 = static_cast<uint32_t>(input.rows());
    constants[4].u32 = apply_router_gate ? 1u : 0u;
    constants[5].f = activation == ExpertActivation::DeepSeekSwiGlu ? activation_limit : 0.0f;
    ncnn::VkMat swiglu_dispatcher;
    swiglu_dispatcher.w = static_cast<int>(down.output_columns * 32);
    swiglu_dispatcher.h = static_cast<int>(input.rows());
    swiglu_dispatcher.c = 1;
    command.record_pipeline(first.swiglu_down_pipeline.get(),
                            bindings,
                            constants,
                            swiglu_dispatcher);

    if ((!direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     down.output_columns,
                                     transfer_slot.download,
                                     command,
                                     first.vulkan_context->device(),
                                     first.option,
                                     output.dtype()))
        || submit_compute_and_wait(command, first.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    runtime_state.dispatches += 2;
    ++runtime_state.shared_expert_swiglu_fusions;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    if (used_cooperative_matrix)
    {
        ++runtime_state.bfloat16_cooperative_matrix_dispatches;
    }
    return true;
#else
    (void)input;
    (void)down_operator;
    (void)intermediate_columns;
    (void)activation;
    (void)activation_limit;
    (void)apply_router_gate;
    (void)output;
    return false;
#endif
}

static bool swiglu_shape(uint32_t intermediate_columns, bool router_gate,
                         uint64_t rows, uint64_t& fused_columns) noexcept
{
    if (intermediate_columns == 0 || rows == 0)
        return false;
    fused_columns = saturating_add_u64(saturating_multiply_u64(intermediate_columns, 2), router_gate ? 1 : 0);
    return fused_columns != UINT64_MAX
           && saturating_multiply_u64(fused_columns, rows) != UINT64_MAX;
}

std::vector<uint64_t> Bfloat16Linear_vulkan::swiglu_scratch_elements(uint32_t intermediate_columns,
                                                                     bool router_gate, uint64_t rows)
{
    uint64_t fused_columns = 0;
    if (!swiglu_shape(intermediate_columns, router_gate, rows, fused_columns))
        return intermediate_columns == 0 || rows == 0 ? std::vector<uint64_t>{}
                                                      : std::vector<uint64_t>{UINT64_MAX};
    return {saturating_multiply_u64(fused_columns, rows)};
}

#if NCNN_MOE_WITH_VULKAN
bool Bfloat16Linear_vulkan::record_swiglu_chain(const ncnn::VkMat& input,
                                                const Bfloat16Linear_vulkan& down_operator,
                                                uint32_t intermediate_columns, ExpertActivation activation,
                                                float activation_limit, bool apply_router_gate,
                                                ncnn::VkMat& output, ncnn::VkCompute& command,
                                                std::vector<ncnn::VkMat>& workspace,
                                                FfnRecordStatistics_vulkan& statistics) const
{
    const auto& first = *d;
    const auto& down = *down_operator.d;
    uint64_t fused_columns = 0;
    if (&input == &output || (activation != ExpertActivation::Silu && activation != ExpertActivation::DeepSeekSwiGlu)
        || !first.vulkan_context || first.vulkan_context != down.vulkan_context || !first.pipeline
        || input.empty() || input.dims != 2 || input.elempack != 1 || input.elemsize != sizeof(float)
        || input.h <= 0 || input.w != static_cast<int>(first.input_columns)
        || !input.allocator || input.allocator->vkdev != first.vulkan_context->device()
        || static_cast<uint32_t>(input.h) > first.vulkan_context->device()->info.max_workgroup_count_y()
        || activation_variant(first.option) != 0 || activation_variant(down.option) != 0
        || !swiglu_shape(intermediate_columns, apply_router_gate,
                         static_cast<uint64_t>(input.h), fused_columns)
        || first.output_columns != fused_columns || down.input_columns != intermediate_columns
        || intermediate_columns % 128 != 0 || down.packed.empty() || down.bias.empty())
        return false;
    if (!first.swiglu_down_pipeline
        && !create_bfloat16_swiglu_down_pipeline(first.vulkan_context, first.option, first.swiglu_down_pipeline))
        return false;
    workspace.reserve(workspace.size() + 2);
    ncnn::VkMat fused, result;
    fused.create(static_cast<int>(first.output_columns), input.h, sizeof(float), first.vulkan_context->blob_allocator());
    result.create(static_cast<int>(down.output_columns), input.h, sizeof(float), first.vulkan_context->blob_allocator());
    if (fused.empty() || result.empty()) return false;
    workspace.push_back(fused);
    workspace.push_back(result);
    std::vector<ncnn::VkMat> bindings;
    std::vector<ncnn::vk_constant_type> constants;
    const bool cooperative = first.record_projection(input, fused, static_cast<uint32_t>(input.h), command, bindings, constants);
    bindings = {fused, down.packed, down.bias, result};
    constants.resize(6);
    constants[0].u32 = intermediate_columns;
    constants[1].u32 = first.output_columns;
    constants[2].u32 = down.output_columns;
    constants[3].u32 = static_cast<uint32_t>(input.h);
    constants[4].u32 = apply_router_gate ? 1u : 0u;
    constants[5].f = activation == ExpertActivation::DeepSeekSwiGlu ? activation_limit : 0.0f;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(down.output_columns * 32);
    dispatcher.h = input.h;
    dispatcher.c = 1;
    command.record_pipeline(first.swiglu_down_pipeline.get(), bindings, constants, dispatcher);
    output = result;
    statistics.dispatches += 2;
    ++statistics.shared_expert_swiglu_fusions;
    statistics.bfloat16_cooperative_matrix_dispatches += cooperative ? 1 : 0;
    return true;
}
#endif

bool Bfloat16Linear_vulkan::forward_swiglu_chain_device(const DeviceTensor_vulkan& input,
                                                        const Bfloat16Linear_vulkan& down_operator,
                                                        uint32_t intermediate_columns,
                                                        ExpertActivation activation,
                                                        float activation_limit,
                                                        bool apply_router_gate,
                                                        DeviceTensor_vulkan& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const auto& context = d->vulkan_context;
    const auto* source = input.value();
    if (&input == &output || !context || !source || source->dims != 2
        || source->elempack <= 0 || source->elemsize != sizeof(float) * static_cast<size_t>(source->elempack)
        || input.vulkan_context() != context)
        return false;
    const ncnn::VkMat retained_input = *source;
    auto lease = context->acquire_transfer_slot();
    auto& slot = lease.slot();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    if (slot.command_used && slot.command->reset() != 0) return false;
    slot.command_used = true;
    auto& command = *slot.command;
    ncnn::VkMat input_gpu = retained_input;
    ncnn::VkMat output_gpu;
    std::vector<ncnn::VkMat> workspace;
    FfnRecordStatistics_vulkan statistics;
    bool successful = false;
    const auto cleanup = [&](void*) {
        if (!successful) (void)command.reset();
    };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);
    input.prepare_read();
    if (input_gpu.elempack != 1)
    {
        context->device()->convert_packing(retained_input, input_gpu, 1, 1, command, d->option);
        if (input_gpu.empty()) return false;
    }
    if (!record_swiglu_chain(input_gpu, down_operator, intermediate_columns, activation, activation_limit, apply_router_gate,
                             output_gpu, command, workspace, statistics)
        || submit_compute_and_wait(command, context->device()) != 0)
        return false;
    DeviceTensor_vulkan completed;
    if (!completed.assign_completed(output_gpu, context)) return false;
    output = std::move(completed);
    auto& state = context->runtime_state();
    state.dispatches += statistics.dispatches;
    state.int4_dispatches += statistics.int4_dispatches;
    state.shared_expert_swiglu_fusions += statistics.shared_expert_swiglu_fusions;
    state.bfloat16_cooperative_matrix_dispatches += statistics.bfloat16_cooperative_matrix_dispatches;
    ++state.compute_submissions;
    successful = true;
    return true;
#else
    (void)input;
    (void)down_operator;
    (void)intermediate_columns;
    (void)activation;
    (void)activation_limit;
    (void)apply_router_gate;
    (void)output;
    return false;
#endif
}

bool Bfloat16Linear_vulkan::materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const ncnn::VkMat* value = input.value();
    if (!value || !d->vulkan_context || input.vulkan_context() != d->vulkan_context)
        return false;
    auto lease = d->vulkan_context->acquire_transfer_slot();
    auto& slot = lease.slot();
    if (!prepare_staging_batch(slot.download, input.rows(), input.columns(), slot.staging_allocator, output.element_size()))
        return false;
    const DType dtype = output.dtype();
    output.reset(input.rows(), input.columns(), false);
    const std::lock_guard<std::mutex> lock(d->vulkan_context->command_mutex());
    input.prepare_read();
    if (slot.command_used && slot.command->reset() != 0)
        return false;
    slot.command_used = true;
    if (!record_staging_download(*value, input.rows(), input.columns(), slot.download,
                                 *slot.command, d->vulkan_context->device(), d->option, dtype)
        || submit_compute_and_wait(*slot.command, d->vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(slot.download, output))
        return false;
    ++d->vulkan_context->runtime_state().compute_submissions;
    ++d->vulkan_context->runtime_state().batch_downloads;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

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
    return create_with_allocator(gate_up,
                                 gate_up_bias,
                                 down,
                                 down_bias,
                                 activation_limit,
                                 vulkan_device_index,
                                 nullptr,
                                 activation,
                                 vulkan_runtime,
                                 optimization_flags);
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
                                             VulkanWeightUploadBatch* upload_batch)
{
#if NCNN_MOE_WITH_VULKAN
    if (activation != ExpertActivation::Silu
        || gate_up.dtype != DType::BFloat16
        || down.dtype != DType::BFloat16
        || gate_up.shape.size() != 2
        || down.shape.size() != 2
        || gate_up.shape[0] == 0
        || gate_up.shape[0] % 2 != 0
        || down.shape[0] == 0
        || down.shape[1] != gate_up.shape[0] / 2
        || (gate_up.shape[0] / 2) % 128 != 0
        || activation_limit < 0.0f)
    {
        return {};
    }

    std::shared_ptr<Bfloat16Expert_vulkan> result(new Bfloat16Expert_vulkan);
    if (upload_batch) upload_batch->retain_owner(result);
    Implementation& implementation = *result->d;
    implementation.gate_up = Bfloat16Linear_vulkan::create_with_allocator(gate_up,
                                                                          gate_up_bias,
                                                                          vulkan_device_index,
                                                                          weight_allocator,
                                                                          vulkan_runtime,
                                                                          optimization_flags, upload_batch);
    implementation.down = Bfloat16Linear_vulkan::create_with_allocator(down,
                                                                       down_bias,
                                                                       vulkan_device_index,
                                                                       weight_allocator,
                                                                       vulkan_runtime,
                                                                       optimization_flags, upload_batch);
    if (!implementation.gate_up || !implementation.down)
        return {};

    const Bfloat16Linear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const Bfloat16Linear_vulkan::Implementation& down_projection = *implementation.down->d;
    if (!gate.vulkan_context
        || gate.vulkan_context != down_projection.vulkan_context
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
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_bfloat16_swiglu_down_pipeline(implementation.vulkan_context,
                                                  gate.option,
                                                  gate.swiglu_down_pipeline))
        {
            return {};
        }
    }
    return result;
#else
    (void)gate_up;
    (void)gate_up_bias;
    (void)down;
    (void)down_bias;
    (void)activation_limit;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)activation;
    (void)vulkan_runtime;
    (void)optimization_flags;
    (void)upload_batch;
    return {};
#endif
}

bool Bfloat16Expert_vulkan::forward(const ActivationBuffer& input,
                                    ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    return implementation.gate_up
           && implementation.down
           && implementation.gate_up->forward_swiglu_chain(input,
                                                           *implementation.down,
                                                           implementation.intermediate_columns,
                                                           implementation.activation,
                                                           implementation.activation_limit,
                                                           false,
                                                           output);
#else
    (void)input;
    (void)output;
    return false;
#endif
}

static bool expert_shape(uint32_t gate_output_columns, uint64_t rows,
                         uint32_t& fused_columns) noexcept
{
    fused_columns = gate_output_columns;
    return gate_output_columns != 0 && rows != 0
           && saturating_multiply_u64(gate_output_columns, rows) != UINT64_MAX;
}

std::vector<uint64_t> Bfloat16Expert_vulkan::scratch_elements(uint32_t gate_output_columns,
                                                              uint64_t rows)
{
    uint32_t fused_columns = 0;
    if (!expert_shape(gate_output_columns, rows, fused_columns))
        return gate_output_columns == 0 || rows == 0 ? std::vector<uint64_t>{}
                                                     : std::vector<uint64_t>{UINT64_MAX};
    return {saturating_multiply_u64(fused_columns, rows)};
}

bool Bfloat16Expert_vulkan::record(const ncnn::VkMat& input, ncnn::VkMat& output,
                                   ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace,
                                   std::vector<ncnn::VkMat>& bindings,
                                   std::vector<ncnn::vk_constant_type>& constants) const
{
#if NCNN_MOE_WITH_VULKAN
    const auto& expert = *d;
    const auto& gate = *expert.gate_up->d;
    const auto& down = *expert.down->d;
    uint32_t fused_columns = 0;
    if (input.empty() || input.dims != 2 || input.elempack != 1 || input.elemsize != sizeof(float)
        || input.w != static_cast<int>(expert.input_columns) || input.h <= 0
        || !gate.swiglu_down_pipeline || activation_variant(gate.option) != 0
        || !expert_shape(gate.output_columns, static_cast<uint64_t>(input.h), fused_columns))
        return false;
    ncnn::VkMat fused;
    fused.create(static_cast<int>(fused_columns), input.h, sizeof(float), expert.vulkan_context->blob_allocator());
    if (fused.empty()) return false;
    workspace.push_back(fused);
    if (gate.record_projection(input, fused, static_cast<uint32_t>(input.h), command, bindings, constants))
        ++expert.vulkan_context->runtime_state().bfloat16_cooperative_matrix_dispatches;
    bindings = {fused, down.packed, down.bias, output};
    constants.resize(6);
    constants[0].u32 = expert.intermediate_columns;
    constants[1].u32 = gate.output_columns;
    constants[2].u32 = down.output_columns;
    constants[3].u32 = static_cast<uint32_t>(input.h);
    constants[4].u32 = 0;
    constants[5].f = expert.activation == ExpertActivation::DeepSeekSwiGlu ? expert.activation_limit : 0.0f;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(down.output_columns * 32);
    dispatcher.h = input.h;
    dispatcher.c = 1;
    command.record_pipeline(gate.swiglu_down_pipeline.get(), bindings, constants, dispatcher);
    return true;
#else
    (void)input;
    (void)output;
    (void)command;
    (void)workspace;
    (void)bindings;
    (void)constants;
    return false;
#endif
}

bool Bfloat16Expert_vulkan::forward_batch(std::span<const Bfloat16Expert_vulkan*> experts,
                                          std::span<const ActivationBuffer*> inputs,
                                          std::span<ActivationBuffer*> outputs)
{
#if NCNN_MOE_WITH_VULKAN
    if (experts.empty()
        || experts.size() != inputs.size()
        || experts.size() != outputs.size())
    {
        return false;
    }

    const Bfloat16Expert_vulkan* first_operator = experts.front();
    if (!first_operator || !first_operator->d)
        return false;
    const Implementation& first_expert = *first_operator->d;
    if (!first_expert.gate_up
        || !first_expert.down
        || !first_expert.vulkan_context)
    {
        return false;
    }
    const Bfloat16Linear_vulkan::Implementation& first_gate = *first_expert.gate_up->d;
    const Bfloat16Linear_vulkan::Implementation& first_down = *first_expert.down->d;
    if (!first_gate.pipeline
        || !first_gate.swiglu_down_pipeline
        || first_down.packed.empty()
        || first_down.bias.empty()
        || first_gate.output_columns % 2 != 0
        || first_down.input_columns != first_expert.intermediate_columns
        || first_down.output_columns != first_expert.output_columns)
    {
        return false;
    }

    const uint32_t input_columns = first_gate.input_columns;
    const uint32_t output_columns = first_down.output_columns;
    if (input_columns == 0
        || output_columns == 0
        || first_expert.intermediate_columns == 0
        || first_expert.intermediate_columns % 128 != 0)
    {
        return false;
    }

    size_t total_rows = 0;
    bool shared_input = true;
    for (size_t index = 0; index < experts.size(); ++index)
    {
        const Bfloat16Expert_vulkan* operator_instance = experts[index];
        const ActivationBuffer* input = inputs[index];
        const ActivationBuffer* output = outputs[index];
        if (!operator_instance
            || !operator_instance->d
            || !input
            || !output
            || input->rows() == 0
            || input->rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
            || input->rows() > static_cast<size_t>(std::numeric_limits<int>::max())
            || input->dtype() != DType::Float32
            || input->columns() != input_columns
            || output->dtype() != DType::Float32)
        {
            return false;
        }
        const Implementation& expert = *operator_instance->d;
        if (!expert.gate_up
            || !expert.down
            || !expert.vulkan_context
            || expert.vulkan_context != first_expert.vulkan_context
            || expert.intermediate_columns != first_expert.intermediate_columns
            || expert.output_columns != output_columns
            || expert.activation != first_expert.activation
            || expert.activation_limit != first_expert.activation_limit)
        {
            return false;
        }
        const Bfloat16Linear_vulkan::Implementation& gate = *expert.gate_up->d;
        const Bfloat16Linear_vulkan::Implementation& down = *expert.down->d;
        if (!gate.pipeline
            || !gate.swiglu_down_pipeline
            || down.packed.empty()
            || down.bias.empty()
            || gate.vulkan_context != first_expert.vulkan_context
            || down.vulkan_context != first_expert.vulkan_context
            || gate.input_columns != input_columns
            || gate.output_columns != first_gate.output_columns
            || down.input_columns != first_down.input_columns
            || down.output_columns != output_columns
            || activation_variant(gate.option)
                   != activation_variant(first_gate.option))
        {
            return false;
        }
        if (total_rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
                             - input->rows()
            || total_rows > static_cast<size_t>(std::numeric_limits<int>::max())
                                - input->rows())
        {
            return false;
        }
        total_rows += input->rows();
        shared_input = shared_input && input == inputs.front();
    }
    if (total_rows == 0
        || total_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    const size_t upload_rows = shared_input ? inputs.front()->rows() : total_rows;
    VulkanRuntimeState& runtime_state = first_expert.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = first_expert.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!prepare_staging_batch(transfer_slot.upload,
                               upload_rows,
                               input_columns,
                               transfer_slot.staging_allocator,
                               sizeof(float))
        || !prepare_staging_batch(transfer_slot.download,
                                  total_rows,
                                  output_columns,
                                  transfer_slot.staging_allocator,
                                  sizeof(float)))
    {
        return false;
    }

    ncnn::Mat mapped_input = transfer_slot.upload.mapped();
    if (mapped_input.empty()
        || mapped_input.dims != 2
        || mapped_input.w != static_cast<int>(input_columns)
        || mapped_input.h != static_cast<int>(upload_rows)
        || mapped_input.elemsize != sizeof(float)
        || mapped_input.elempack != 1)
    {
        return false;
    }
    auto* mapped_input_data = static_cast<std::byte*>(mapped_input.data);
    size_t row_offset = 0;
    for (const ActivationBuffer* input : inputs)
    {
        const size_t input_size = input->rows() * static_cast<size_t>(input_columns) * sizeof(float);
        if (input->bytes().size() != input_size)
            return false;
        std::memcpy(mapped_input_data
                        + row_offset * static_cast<size_t>(input_columns) * sizeof(float),
                    input->bytes().data(),
                    input_size);
        if (shared_input) break;
        row_offset += input->rows();
    }
    transfer_slot.upload.allocator->flush(transfer_slot.upload.data);
    transfer_slot.upload.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    transfer_slot.upload.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    std::unique_lock<std::mutex> lock(first_expert.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
    }
    transfer_slot.command_used = true;

    ncnn::VkMat input_gpu;
    if (!record_mapped_act_upload(transfer_slot.upload,
                                  input_gpu,
                                  command,
                                  first_expert.vulkan_context->device(),
                                  first_gate.option,
                                  DType::Float32))
    {
        return false;
    }
    ncnn::VkMat output_gpu;
    output_gpu.create(static_cast<int>(output_columns),
                      static_cast<int>(total_rows),
                      act_element_size(first_gate.option),
                      first_expert.vulkan_context->blob_allocator());
    if (input_gpu.empty() || output_gpu.empty())
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
        ncnn::VkMat input_view = aligned_row_view(input_gpu, shared_input ? 0 : row_offset, rows, command, first_gate.option,
                                                  first_expert.vulkan_context, aligned_inputs, experts.size());
        ncnn::VkMat output_view = row_view(output_gpu, row_offset, rows);
        if (input_view.empty() || output_view.empty())
            return false;
        if (align_output_rows)
        {
            aligned_outputs[index].create(output_view.w, output_view.h, output_view.elemsize,
                                          first_expert.vulkan_context->blob_allocator());
            output_view = aligned_outputs[index];
            if (output_view.empty()) return false;
        }

        ncnn::VkMat& fused_gpu = intermediates.emplace_back();
        fused_gpu.create(static_cast<int>(gate.output_columns),
                         static_cast<int>(rows),
                         act_element_size(first_gate.option),
                         first_expert.vulkan_context->blob_allocator());
        if (fused_gpu.empty())
            return false;

        const bool used_cooperative_matrix = gate.record_projection(input_view,
                                                                    fused_gpu,
                                                                    static_cast<uint32_t>(rows),
                                                                    command,
                                                                    bindings,
                                                                    constants);
        if (used_cooperative_matrix)
            ++runtime_state.bfloat16_cooperative_matrix_dispatches;

        bindings[0] = fused_gpu;
        bindings[1] = down.packed;
        bindings[2] = down.bias;
        bindings[3] = output_view;
        constants.resize(6);
        constants[0].u32 = expert.intermediate_columns;
        constants[1].u32 = gate.output_columns;
        constants[2].u32 = down.output_columns;
        constants[3].u32 = static_cast<uint32_t>(rows);
        constants[4].u32 = 0;
        constants[5].f = expert.activation == ExpertActivation::DeepSeekSwiGlu ? expert.activation_limit : 0.0f;
        ncnn::VkMat swiglu_dispatcher;
        swiglu_dispatcher.w = static_cast<int>(down.output_columns * 32);
        swiglu_dispatcher.h = static_cast<int>(rows);
        swiglu_dispatcher.c = 1;
        command.record_pipeline(gate.swiglu_down_pipeline.get(),
                                bindings,
                                constants,
                                swiglu_dispatcher);
        if (align_output_rows)
        {
            ncnn::VkMat destination = row_view(output_gpu, row_offset, rows);
            if (!record_row_copy(output_view, destination, command, first_gate.option)) return false;
        }
        row_offset += rows;
    }

    if (!record_staging_download(output_gpu,
                                 total_rows,
                                 output_columns,
                                 transfer_slot.download,
                                 command,
                                 first_expert.vulkan_context->device(),
                                 first_down.option,
                                 DType::Float32)
        || submit_compute_and_wait(command, first_expert.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batches(transfer_slot.download,
                                        inputs,
                                        outputs,
                                        output_columns))
    {
        return false;
    }
    lock.unlock();
    runtime_state.dispatches += static_cast<uint64_t>(experts.size()) * 2;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)experts;
    (void)inputs;
    (void)outputs;
    return false;
#endif
}

} // namespace moe
} // namespace ncnn
