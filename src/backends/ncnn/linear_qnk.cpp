#include "linear.h"
#include "sharedffn_vulkan.h"
#include "mxfp4_weightpacking.h"
#include "rmsnorm_vulkan.h"
#include <command.h>
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
#include "kernels/vulkan/qnk_projection.comp.hex.h"
#include "kernels/vulkan/qnk_swiglu.comp.hex.h"
#endif
#include "linear_impl.h"
#include "linear_internal.h"

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN

static size_t qnk_shader_type(DType dtype) noexcept;

static bool create_qnk_projection_pipeline(const std::shared_ptr<VulkanContext>& context,
                                           const ncnn::Option& option,
                                           DType dtype, bool permuted_input,
                                           uint32_t tile_rows,
                                           std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(qnk_projection_shader,
                                                                                      static_cast<int>(sizeof(qnk_projection_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    const size_t pipeline_key = (qnk_shader_type(dtype) << 3u) | (permuted_input ? 4u : 0u)
                                | (tile_rows == 4 ? 2u : 0u)
                                | (option.use_subgroup_ops ? 1u : 0u);
    destination = context->find_pipeline(qnk_projection_shader, pipeline_key);
    if (destination)
        return true;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_local_size_xyz(128, 1, 1);
    std::vector<ncnn::vk_specialization_type> specializations(3);
    specializations[0].u32 = static_cast<uint32_t>(qnk_shader_type(dtype));
    specializations[1].u32 = permuted_input ? 1u : 0u;
    specializations[2].u32 = tile_rows;
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
    context->cache_pipeline(qnk_projection_shader, pipeline_key, destination);
    return true;
}

static bool create_qnk_swiglu_pipeline(const std::shared_ptr<VulkanContext>& context,
                                       const ncnn::Option& option,
                                       std::shared_ptr<ncnn::Pipeline>& destination)
{
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(qnk_swiglu_shader,
                                                                                      static_cast<int>(sizeof(qnk_swiglu_shader) - 1),
                                                                                      option,
                                                                                      0);
    if (!spirv || spirv->empty())
        return false;

    ncnn::VulkanDevice* device = context->device();
    destination = context->find_pipeline(qnk_swiglu_shader, 0);
    if (destination)
        return true;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(device));
    pipeline->set_optimal_local_size_xyz(128, 1, 1);
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
    context->cache_pipeline(qnk_swiglu_shader, 0, destination);
    return true;
}

static size_t qnk_shader_type(DType dtype) noexcept
{
    if (dtype == DType::Int4)
        return 6;
    if (dtype == DType::BFloat16)
        return 7;
    if (dtype == DType::Float32)
        return 8;
    if (dtype == DType::Q8_0)
        return 9;
    if (dtype == DType::IQ4NL)
        return 10;
    if (dtype == DType::Q5_1)
        return 11;
    if (dtype == DType::Q4_0)
        return 12;
    if (dtype == DType::Q4_1)
        return 13;
    if (dtype == DType::Q5_0)
        return 14;
    if (dtype == DType::IQ4XS)
        return 15;
    if (dtype == DType::NvFp4)
        return 16;
    if (dtype == DType::MxFp4)
        return 17;
    return static_cast<size_t>(dtype) - static_cast<size_t>(DType::Q2K);
}

static bool qnk_has_metadata(const TensorData& matrix) noexcept
{
    return !matrix.quantization_scales.empty()
           || matrix.quantization_group_size != 0
           || !matrix.quantization_zero_points.empty()
           || !matrix.quantization_group_indices.empty()
           || !matrix.input_column_indices.empty();
}

#endif

QnkLinear_vulkan::QnkLinear_vulkan()
    : d(new Implementation)
{
}

QnkLinear_vulkan::~QnkLinear_vulkan() = default;

static std::span<const uint8_t> qnk_projection_values(const TensorData& matrix) noexcept
{
    if (matrix.dtype == DType::Int4)
        return matrix.int4_values();
    if (matrix.dtype == DType::BFloat16)
    {
        const auto values = matrix.bfloat16_values();
        return {reinterpret_cast<const uint8_t*>(values.data()), values.size_bytes()};
    }
    if (matrix.dtype == DType::Float32)
    {
        const auto values = matrix.float32_values();
        return {reinterpret_cast<const uint8_t*>(values.data()), values.size_bytes()};
    }
    return matrix.qnk_values();
}

static uint64_t qnk_mxfp4_native_size(const TensorData& matrix) noexcept
{
    if (matrix.dtype != DType::MxFp4 || matrix.shape.size() != 2
        || matrix.shape[0] == 0 || matrix.shape[1] == 0 || matrix.shape[1] % 32 != 0
        || matrix.mxfp4_file_storage || !matrix.quantization_scales.empty()
        || !matrix.float32_data.empty() || !matrix.bfloat16_data.empty()
        || !matrix.int8_data.empty() || !matrix.int64_data.empty()
        || !matrix.quantization_zero_points.empty() || matrix.quantization_group_size != 0
        || !matrix.quantization_group_indices.empty() || !matrix.quantized_data.empty()
        || matrix.mapped_data)
    {
        return 0;
    }
    const uint64_t blocks_per_row = matrix.shape[1] / 32;
    if (matrix.shape[0] > std::numeric_limits<uint64_t>::max() / blocks_per_row)
        return 0;
    const uint64_t block_count = static_cast<uint64_t>(matrix.shape[0]) * blocks_per_row;
    if (block_count > std::numeric_limits<uint64_t>::max() / 17
        || block_count > std::numeric_limits<size_t>::max() / 16
        || block_count > std::numeric_limits<size_t>::max() / 17)
    {
        return 0;
    }
    if (matrix.mxfp4_blocks.size() != static_cast<size_t>(block_count * 16)
        || matrix.mxfp4_scales.size() != static_cast<size_t>(block_count))
    {
        return 0;
    }
    return block_count * 17;
}

uint64_t QnkLinear_vulkan::storage_bytes(const TensorData& matrix, uint64_t alignment) noexcept
{
    const bool mxfp4 = matrix.dtype == DType::MxFp4;
    if ((matrix.dtype != DType::Int4 && !is_qnk_dtype(matrix.dtype) && matrix.dtype != DType::BFloat16 && matrix.dtype != DType::Float32 && !mxfp4) || matrix.shape.size() != 2
        || matrix.shape[0] == 0 || matrix.shape[1] == 0)
    {
        return 0;
    }
    alignment = std::max<uint64_t>(4, alignment);
    uint64_t raw = 0;
    if (mxfp4)
    {
        raw = qnk_mxfp4_native_size(matrix);
        if (raw == 0)
            return 0;
    }
    else
    {
        raw = qnk_projection_values(matrix).size();
    }
    if (matrix.mxfp4_file_storage)
    {
        const MxFp4FileStorage& file = *matrix.mxfp4_file_storage;
        if (mxfp4 || !is_qnk_dtype(matrix.dtype) || raw != 0
            || file.layout != MxFp4FileLayout::Separate || file.interleave_rows
            || file.blocks_path.empty() || file.blocks_size == 0
            || file.blocks_offset > std::numeric_limits<uint64_t>::max() - file.blocks_size
            || !file.scales_path.empty() || file.scales_offset != 0 || file.scales_size != 0
            || !file.secondary_scales_path.empty() || file.secondary_scales_offset != 0 || file.secondary_scales_size != 0
            || (file.secondary_blocks_size == 0
                    ? !file.secondary_blocks_path.empty() || file.secondary_blocks_offset != 0
                    : file.secondary_blocks_path.empty() || file.secondary_blocks_offset > std::numeric_limits<uint64_t>::max() - file.secondary_blocks_size)
            || file.blocks_size > std::numeric_limits<uint64_t>::max() - file.secondary_blocks_size)
        {
            return 0;
        }
        raw = qnk_storage_bytes(matrix.dtype, matrix.shape[0], matrix.shape[1]);
        if (raw != file.blocks_size + file.secondary_blocks_size)
            return 0;
    }
    else if (is_qnk_dtype(matrix.dtype)
             && raw != qnk_storage_bytes(matrix.dtype, matrix.shape[0], matrix.shape[1]))
    {
        return 0;
    }
    if (raw == 0)
        return 0;
    const uint64_t segments[] = {
        raw,
        static_cast<uint64_t>(matrix.shape[0]) * sizeof(float),
        static_cast<uint64_t>(matrix.quantization_scales.size()) * sizeof(float),
        matrix.quantization_zero_points.size(),
        static_cast<uint64_t>(matrix.quantization_group_indices.size()) * sizeof(uint32_t),
        static_cast<uint64_t>(matrix.input_column_indices.size()) * sizeof(uint32_t),
    };
    uint64_t total = 0;
    for (uint64_t bytes : segments)
    {
        const uint64_t padding = bytes % alignment == 0 ? 0 : alignment - bytes % alignment;
        if (bytes > std::numeric_limits<uint64_t>::max() - padding || total > std::numeric_limits<uint64_t>::max() - bytes - padding)
            return 0;
        total += bytes + padding;
    }
    return total;
}

// Unpack codes once without rounding weights or retaining a second GPU copy.
static DType qnk_projection_storage_type(DType dtype) noexcept
{
    return dtype == DType::Q4_0 || dtype == DType::Q5_0 || dtype == DType::IQ4NL
               ? DType::Q8_0
               : dtype;
}

uint64_t QnkLinear_vulkan::projection_storage_bytes(const TensorData& matrix, uint64_t alignment) noexcept
{
    const uint64_t native = storage_bytes(matrix, alignment);
    const DType dtype = qnk_projection_storage_type(matrix.dtype);
    if (native == 0 || dtype == matrix.dtype) return native;
    alignment = std::max<uint64_t>(4, alignment);
    const uint64_t raw = qnk_storage_bytes(matrix.dtype, matrix.shape[0], matrix.shape[1]);
    const uint64_t packed = qnk_storage_bytes(dtype, matrix.shape[0], matrix.shape[1]);
    const uint64_t raw_aligned = raw + (alignment - raw % alignment) % alignment;
    const uint64_t packed_padding = (alignment - packed % alignment) % alignment;
    if (packed == 0 || packed > UINT64_MAX - packed_padding
        || native - raw_aligned > UINT64_MAX - packed - packed_padding) return 0;
    return native - raw_aligned + packed + packed_padding;
}

#if NCNN_MOE_WITH_VULKAN
static bool qnk_prepare_projection(const TensorData& matrix, TensorData& prepared)
{
    const DType dtype = qnk_projection_storage_type(matrix.dtype);
    if (dtype == matrix.dtype || matrix.shape.size() != 2
        || !matrix.quantization_scales.empty() || matrix.quantization_group_size != 0
        || !matrix.quantization_zero_points.empty() || !matrix.quantization_group_indices.empty()) return false;
    const auto raw = matrix.qnk_values();
    const uint64_t packed_size = qnk_storage_bytes(dtype, matrix.shape[0], matrix.shape[1]);
    if (packed_size == 0 || packed_size > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || raw.size() != qnk_storage_bytes(matrix.dtype, matrix.shape[0], matrix.shape[1])) return false;
    prepared.dtype = dtype;
    prepared.shape = matrix.shape;
    prepared.input_column_indices = matrix.input_column_indices;
    prepared.quantized_data.resize(static_cast<size_t>(packed_size));
    const size_t block_bytes = qnk_block_bytes(matrix.dtype);
    std::array<uint8_t, 22> unit_block{};
    std::array<float, qnk_block_elements> codes;
    for (size_t source = 0, destination = 0; source < raw.size(); source += block_bytes, destination += 34)
    {
        std::copy_n(raw.data() + source, block_bytes, unit_block.data());
        unit_block[0] = 0;
        unit_block[1] = 0x3c; // FP16 1.0: reuse the decoder for exact integer codes.
        qnk_dequantize_block(matrix.dtype, unit_block.data(), codes.data());
        uint8_t* output = prepared.quantized_data.data() + destination;
        std::copy_n(raw.data() + source, 2, output);
        for (size_t index = 0; index < 32; ++index)
            output[2 + index] = static_cast<uint8_t>(static_cast<int8_t>(codes[index]));
    }
    return true;
}
#endif

bool QnkLinear_vulkan::can_fuse(std::span<const TensorData* const> matrices,
                                uint64_t alignment, uint64_t storage_limit) noexcept
{
    if (matrices.empty() || !matrices.front()) return false;
    const TensorData& first = *matrices.front();
    if (first.shape.size() != 2 || !is_qnk_dtype(first.dtype)
        || !check_input_columns(first)) return false;
    uint64_t rows = 0;
    uint64_t bytes = 0;
    for (const TensorData* matrix : matrices)
    {
        if (!matrix || matrix->shape.size() != 2 || matrix->shape[1] != first.shape[1]
            || qnk_projection_storage_type(matrix->dtype) != qnk_projection_storage_type(first.dtype)
            || !matrix->quantization_scales.empty() || matrix->quantization_group_size != 0
            || !matrix->quantization_zero_points.empty() || !matrix->quantization_group_indices.empty()
            || matrix->input_column_indices != first.input_column_indices
            || !qnk_shape_supported(matrix->dtype, matrix->shape[0], matrix->shape[1])) return false;
        rows += matrix->shape[0];
        const uint64_t size = projection_storage_bytes(*matrix, alignment);
        if (size == 0) return false;
        bytes += size;
        if (rows > static_cast<uint64_t>(std::numeric_limits<int>::max() / 32)
            || bytes > std::min<uint64_t>(storage_limit, std::numeric_limits<int>::max())) return false;
    }
    return rows != 0;
}

std::shared_ptr<QnkLinear_vulkan> QnkLinear_vulkan::create_fused(std::span<const TensorData* const> matrices,
                                                                 std::span<const TensorData* const> biases,
                                                                 uint32_t vulkan_device_index,
                                                                 const VulkanRuntimePtr& vulkan_runtime,
                                                                 uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    if (!can_fuse(matrices) || matrices.size() != biases.size()) return {};
    const auto context = VulkanContext::acquire(vulkan_device_index, vulkan_runtime, optimization_flags);
    if (!context || !can_fuse(matrices, context->device()->info.buffer_offset_alignment(), context->device()->info.physicalDeviceProperties().limits.maxStorageBufferRange)) return {};
    TensorData combined, combined_bias;
    combined.dtype = qnk_projection_storage_type(matrices.front()->dtype);
    combined.shape = {0, matrices.front()->shape[1]};
    combined.input_column_indices = matrices.front()->input_column_indices;
    size_t rows = 0;
    for (const TensorData* matrix : matrices) rows += matrix->shape[0];
    combined.quantized_data.reserve(static_cast<size_t>(qnk_storage_bytes(combined.dtype, rows, combined.shape[1])));
    combined_bias.float32_data.reserve(rows);
    for (size_t index = 0; index < matrices.size(); ++index)
    {
        const TensorData& matrix = *matrices[index];
        TensorData prepared;
        const TensorData* storage = &matrix;
        if (qnk_projection_storage_type(matrix.dtype) != matrix.dtype)
        {
            if (!qnk_prepare_projection(matrix, prepared)) return {};
            storage = &prepared;
        }
        const auto raw = storage->qnk_values();
        if (raw.size() != qnk_storage_bytes(storage->dtype, storage->shape[0], storage->shape[1])) return {};
        combined.quantized_data.insert(combined.quantized_data.end(), raw.begin(), raw.end());
        const size_t offset = combined.shape[0];
        combined.shape[0] += matrix.shape[0];
        combined_bias.float32_data.resize(combined.shape[0], 0.0f);
        if (const TensorData* bias = biases[index])
        {
            if (bias->shape != std::vector<uint32_t>{matrix.shape[0]}) return {};
            if (bias->dtype == DType::Float32 && bias->float32_values().size() == matrix.shape[0])
                std::copy(bias->float32_values().begin(), bias->float32_values().end(), combined_bias.float32_data.begin() + offset);
            else if (bias->dtype == DType::BFloat16 && bias->bfloat16_values().size() == matrix.shape[0])
                for (size_t row = 0; row < matrix.shape[0]; ++row)
                    combined_bias.float32_data[offset + row] = bfloat16_to_float(bias->bfloat16_values()[row]);
            else
                return {};
        }
    }
    combined_bias.dtype = DType::Float32;
    combined_bias.shape = {combined.shape[0]};
    return create_with_allocator(combined, &combined_bias, vulkan_device_index, nullptr, nullptr,
                                 vulkan_runtime, optimization_flags, 1);
#else
    (void)matrices;
    (void)biases;
    (void)vulkan_device_index;
    (void)vulkan_runtime;
    (void)optimization_flags;
    return {};
#endif
}

std::shared_ptr<QnkLinear_vulkan> QnkLinear_vulkan::create(const TensorData& matrix,
                                                           const TensorData* bias,
                                                           uint32_t vulkan_device_index,
                                                           const VulkanRuntimePtr& vulkan_runtime,
                                                           uint64_t optimization_flags,
                                                           uint32_t input_group_count)
{
#if NCNN_MOE_WITH_VULKAN
    if (qnk_projection_storage_type(matrix.dtype) != matrix.dtype)
    {
        TensorData prepared;
        if (!qnk_prepare_projection(matrix, prepared)) return {};
        auto result = create_with_allocator(prepared, bias, vulkan_device_index, nullptr, nullptr,
                                            vulkan_runtime, optimization_flags, input_group_count);
        if (result) result->d->dtype = matrix.dtype;
        return result;
    }
#endif
    return create_with_allocator(matrix,
                                 bias,
                                 vulkan_device_index,
                                 nullptr,
                                 nullptr,
                                 vulkan_runtime,
                                 optimization_flags,
                                 input_group_count);
}

std::shared_ptr<QnkLinear_vulkan> QnkLinear_vulkan::create_with_allocator(const TensorData& matrix,
                                                                          const TensorData* bias,
                                                                          uint32_t vulkan_device_index,
                                                                          ncnn::VkAllocator* weight_allocator,
                                                                          ncnn::VkAllocator* staging_allocator,
                                                                          const VulkanRuntimePtr& vulkan_runtime,
                                                                          uint64_t optimization_flags,
                                                                          uint32_t input_group_count,
                                                                          const TensorData* extra,
                                                                          VulkanWeightUploadBatch* upload_batch)
{
#if NCNN_MOE_WITH_VULKAN
    const bool int4 = matrix.dtype == DType::Int4;
    const bool dense = matrix.dtype == DType::BFloat16 || matrix.dtype == DType::Float32;
    const bool mxfp4 = matrix.dtype == DType::MxFp4;
    const uint64_t mxfp4_size = mxfp4 ? qnk_mxfp4_native_size(matrix) : 0;
    if ((!int4 && !dense && !mxfp4 && !is_qnk_dtype(matrix.dtype))
        || matrix.shape.size() != 2
        || input_group_count == 0
        || matrix.shape[0] % input_group_count != 0
        || static_cast<uint64_t>(matrix.shape[1]) * input_group_count > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || matrix.shape[0] == 0
        || matrix.shape[1] == 0
        || matrix.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || (int4 && !validate_int4_tensor(matrix))
        || (mxfp4 && mxfp4_size == 0)
        || (!int4 && !dense && !mxfp4 && !qnk_shape_supported(matrix.dtype, matrix.shape[0], matrix.shape[1]))
        || (!int4 && !check_input_columns(matrix)))
    {
        return {};
    }
    if (extra
        && (!(is_qnk_dtype(matrix.dtype) || mxfp4)
            || extra->dtype != matrix.dtype
            || extra->shape != matrix.shape
            || input_group_count != 1
            || bias
            || qnk_has_metadata(matrix)
            || qnk_has_metadata(*extra)
            || (mxfp4 && qnk_mxfp4_native_size(*extra) != mxfp4_size)))
    {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(matrix.shape[0])
                          + (extra ? matrix.shape[0] : 0u);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int>::max() / 32))
    {
        return {};
    }
    const std::span<const uint8_t> raw_values = mxfp4 ? std::span<const uint8_t>{}
                                                      : qnk_projection_values(matrix);
    const uint64_t first_size = int4    ? static_cast<uint64_t>(matrix.shape[0]) * (matrix.shape[1] / 2 + matrix.shape[1] % 2)
                                : dense ? static_cast<uint64_t>(matrix.shape[0]) * matrix.shape[1] * (matrix.dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float))
                                : mxfp4 ? mxfp4_size
                                        : qnk_storage_bytes(matrix.dtype, matrix.shape[0], matrix.shape[1]);
    const std::span<const uint8_t> extra_values = extra && !mxfp4 ? qnk_projection_values(*extra)
                                                                  : std::span<const uint8_t>{};
    const uint64_t extra_size = extra ? (mxfp4 ? mxfp4_size : qnk_storage_bytes(extra->dtype, matrix.shape[0], matrix.shape[1])) : 0;
    if (first_size == 0
        || (!mxfp4 && first_size != raw_values.size())
        || (extra && (extra_size == 0 || (!mxfp4 && extra_size != extra_values.size())))
        || first_size > std::numeric_limits<uint64_t>::max() - extra_size)
    {
        return {};
    }
    const uint64_t expected_size = first_size + extra_size;
    if (expected_size > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return {};
    const uint32_t output_columns = static_cast<uint32_t>(rows);
    const uint32_t input_columns = matrix.shape[1];
    if (bias
        && (bias->shape.size() != 1
            || bias->shape[0] != output_columns
            || (bias->dtype != DType::Float32 && bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    std::shared_ptr<QnkLinear_vulkan> result(new QnkLinear_vulkan);
    if (upload_batch) upload_batch->retain_owner(result);
    Implementation& implementation = *result->d;
    implementation.dtype = matrix.dtype;
    implementation.input_columns = input_columns;
    implementation.input_group_count = input_group_count;
    implementation.output_columns = output_columns;
    implementation.block_count = int4 || dense ? 0 : input_columns / qnk_block_size(matrix.dtype);
    implementation.block_size = int4    ? input_columns / 2 + input_columns % 2
                                : dense ? input_columns * (matrix.dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float))
                                        : static_cast<uint32_t>(qnk_block_bytes(matrix.dtype));
    implementation.group_size = matrix.quantization_group_size;
    implementation.group_count = int4 ? input_columns / matrix.quantization_group_size + (input_columns % matrix.quantization_group_size != 0) : 0;
    implementation.optimization_flags = optimization_flags;
    implementation.vulkan_context = VulkanContext::acquire(vulkan_device_index,
                                                           vulkan_runtime,
                                                           optimization_flags);
    if (!implementation.vulkan_context)
        return {};

    ncnn::VulkanDevice* device = implementation.vulkan_context->device();
    if (staging_allocator && staging_allocator->vkdev != device)
        return {};
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

    const size_t raw_size = static_cast<size_t>(expected_size);
    const size_t bias_size = static_cast<size_t>(output_columns) * sizeof(float);
    const size_t storage_alignment = std::max<size_t>(4, device->info.buffer_offset_alignment());
    const auto aligned_segment = [storage_alignment](size_t bytes) -> size_t {
        const size_t remainder = bytes % storage_alignment;
        if (remainder == 0)
            return bytes;
        const size_t padding = storage_alignment - remainder;
        if (bytes > std::numeric_limits<size_t>::max() - padding)
            return 0;
        return bytes + padding;
    };
    const size_t raw_segment_size = aligned_segment(raw_size);
    const size_t bias_offset = raw_segment_size;
    const size_t bias_segment_size = aligned_segment(bias_size);
    if (raw_segment_size == 0
        || bias_segment_size == 0
        || raw_segment_size > std::numeric_limits<size_t>::max() - bias_segment_size)
    {
        return {};
    }
    uint64_t total_bytes = storage_bytes(matrix, storage_alignment);
    if (extra)
    {
        const uint64_t extra_bytes = storage_bytes(*extra, storage_alignment);
        if (extra_bytes == 0 || total_bytes > std::numeric_limits<uint64_t>::max() - extra_bytes)
            return {};
        total_bytes += extra_bytes;
    }
    if (total_bytes == 0
        || total_bytes > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || static_cast<uint64_t>(raw_segment_size + bias_segment_size) > total_bytes
        || (extra
            && total_bytes > device->info.physicalDeviceProperties().limits.maxStorageBufferRange))
        return {};
    const size_t storage_size = static_cast<size_t>(total_bytes);
    if (storage_size == 0 || storage_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        return {};
    implementation.device_size = storage_size;
    size_t metadata_offset = raw_segment_size + bias_segment_size;
    const auto segment_offset = [&](size_t bytes) -> uint32_t {
        if (bytes == 0)
            return 0;
        const uint32_t offset = static_cast<uint32_t>(metadata_offset);
        metadata_offset += aligned_segment(bytes);
        return offset;
    };
    implementation.scales_offset = segment_offset(matrix.quantization_scales.size() * sizeof(float));
    implementation.zeros_offset = segment_offset(matrix.quantization_zero_points.size());
    implementation.groups_offset = segment_offset(matrix.quantization_group_indices.size() * sizeof(uint32_t));
    implementation.columns_offset = segment_offset(matrix.input_column_indices.size() * sizeof(uint32_t));
    if (!weight_allocator)
    {
        implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(device, storage_size));
        weight_allocator = implementation.weight_allocator.get();
    }
    if (!staging_allocator && !upload_batch)
    {
        implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(device));
        staging_allocator = implementation.weight_staging_allocator.get();
    }

    auto* independent_allocator = dynamic_cast<VulkanExpertWeightAllocator*>(weight_allocator);
    std::optional<VulkanIndependentWeightTransfer> transfer;
    ncnn::Mat combined;
    std::span<uint8_t> packed_storage;
    if (upload_batch)
    {
        if (!independent_allocator) return {};
        packed_storage = upload_batch->prepare_storage(storage_size, independent_allocator);
    }
    else if (independent_allocator)
    {
        transfer.emplace(implementation.vulkan_context, staging_allocator);
        packed_storage = transfer->prepare_storage(storage_size, independent_allocator);
    }
    else
    {
        combined.create(static_cast<int>(storage_size), sizeof(uint8_t));
        if (!combined.empty())
            packed_storage = {static_cast<uint8_t*>(combined.data), storage_size};
    }
    if (packed_storage.size() != storage_size)
        return {};
    std::memset(packed_storage.data() + raw_size, 0, storage_size - raw_size);
    if (mxfp4)
    {
        if (!pack_gguf_mxfp4({matrix.mxfp4_blocks.data(), matrix.mxfp4_blocks.size()},
                             {matrix.mxfp4_scales.data(), matrix.mxfp4_scales.size()},
                             {packed_storage.data(), static_cast<size_t>(first_size)}))
            return {};
    }
    else
        std::memcpy(packed_storage.data(), raw_values.data(), static_cast<size_t>(first_size));
    if (extra)
    {
        if (mxfp4)
        {
            if (!pack_gguf_mxfp4({extra->mxfp4_blocks.data(), extra->mxfp4_blocks.size()},
                                 {extra->mxfp4_scales.data(), extra->mxfp4_scales.size()},
                                 {packed_storage.data() + first_size, static_cast<size_t>(extra_size)}))
                return {};
        }
        else
            std::memcpy(packed_storage.data() + first_size,
                        extra_values.data(),
                        static_cast<size_t>(extra_size));
    }
    if (bias)
    {
        float* bias_values = reinterpret_cast<float*>(packed_storage.data() + bias_offset);
        if (bias->dtype == DType::Float32)
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
    }
    std::vector<float> coop_bias;
    if (matrix.dtype == DType::BFloat16 && bias)
    {
        // Keep the f32 bias for the cooperative Gemm; the packed storage below
        // cannot be read back once the weight upload owns it.
        coop_bias.resize(output_columns);
        if (bias->dtype == DType::Float32)
        {
            const std::span<const float> values = bias->float32_values();
            if (values.size() != output_columns)
                return {};
            std::copy(values.begin(), values.end(), coop_bias.begin());
        }
        else
        {
            const std::span<const uint16_t> values = bias->bfloat16_values();
            if (values.size() != output_columns)
                return {};
            for (uint32_t index = 0; index < output_columns; ++index)
                coop_bias[index] = bfloat16_to_float(values[index]);
        }
    }
    const auto copy_metadata = [&](uint32_t offset, const void* data, size_t bytes) {
        if (bytes != 0)
            std::memcpy(packed_storage.data() + offset, data, bytes);
    };
    copy_metadata(implementation.scales_offset, matrix.quantization_scales.data(), matrix.quantization_scales.size() * sizeof(float));
    copy_metadata(implementation.zeros_offset, matrix.quantization_zero_points.data(), matrix.quantization_zero_points.size());
    copy_metadata(implementation.groups_offset, matrix.quantization_group_indices.data(), matrix.quantization_group_indices.size() * sizeof(uint32_t));
    copy_metadata(implementation.columns_offset, matrix.input_column_indices.data(), matrix.input_column_indices.size() * sizeof(uint32_t));

    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_qnk_projection_pipeline(implementation.vulkan_context, implementation.option, matrix.dtype,
                                            implementation.columns_offset != 0, 1, implementation.pipeline)
            || !create_qnk_projection_pipeline(implementation.vulkan_context, implementation.option, matrix.dtype,
                                               implementation.columns_offset != 0, 4, implementation.prefill_pipeline))
            return {};
        if (matrix.dtype == DType::BFloat16)
        {
            create_tile_gemm(implementation.vulkan_context, implementation.option, implementation.tile_gemm);
            // Plain BF16 rows are the only layout ncnn's Gemm can pack as-is.
            if (implementation.block_count == 0 && implementation.input_group_count == 1
                && implementation.block_size == static_cast<uint64_t>(implementation.input_columns) * sizeof(uint16_t))
                (void)create_coop_gemm(implementation.vulkan_context, implementation.option,
                                       implementation.input_columns, implementation.output_columns,
                                       matrix.bfloat16_values(),
                                       std::span<const float>(coop_bias),
                                       implementation.coop_gemm,
                                       implementation.coop_opt);
        }
    }
    if ((output_columns + 3u) / 4u > static_cast<uint32_t>(std::numeric_limits<int>::max()) / implementation.pipeline->local_size_x())
        return {};
    bool uploaded = false;
    if (upload_batch)
        uploaded = upload_batch->record_prepared_storage(implementation.storage, weight_allocator);
    else if (transfer)
        uploaded = transfer->record_prepared_storage(implementation.storage, independent_allocator) && transfer->submit_and_wait();
    else
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::Option upload_option = implementation.option;
        upload_option.blob_vkallocator = weight_allocator;
        upload_option.workspace_vkallocator = weight_allocator;
        upload_option.staging_vkallocator = staging_allocator;
        ncnn::VkTransfer command(device);
        command.record_upload(combined, implementation.storage, upload_option);
        uploaded = !implementation.storage.empty() && command.submit_and_wait() == 0;
    }
    transfer.reset();
    implementation.weight_staging_allocator.reset();
    if (!uploaded)
        return {};

    // Weights and bias become views of one allocation, so a projection binds one
    // allocation under several bindings. ncnn tracks a single access state per
    // allocation and cannot honor a readonly claim on one view while another
    // view of it is bound, so recorders keep the prepared operands writable and
    // mark only activations readonly.
    const auto storage_view = [&implementation](size_t offset, size_t bytes) {
        ncnn::VkMat view = implementation.storage;
        view.dims = 1;
        view.w = static_cast<int>(bytes);
        view.h = 1;
        view.d = 1;
        view.c = 1;
        view.elemsize = sizeof(uint8_t);
        view.elempack = 1;
        view.cstep = bytes;
#if NCNN_BATCH
        view.n = 1;
        view.nstep = bytes;
#endif
        view.offset += offset;
        return view;
    };
    implementation.quantized = storage_view(0, storage_size);
    implementation.bias = storage_view(bias_offset, bias_size);
    return result;
#else
    (void)matrix;
    (void)bias;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)vulkan_runtime;
    (void)optimization_flags;
    (void)input_group_count;
    (void)extra;
    (void)staging_allocator;
    (void)upload_batch;
    return {};
#endif
}

bool QnkLinear_vulkan::record_forward(const ncnn::VkMat& input, ncnn::VkMat& output,
                                      ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace) const
{
#if NCNN_MOE_WITH_VULKAN
    (void)workspace;
    if (!d->vulkan_context || !d->pipeline || input.empty() || input.dims != 2
        || input.elempack != 1 || input.elemsize != sizeof(float) || input.h <= 0
        || input.w != static_cast<int>(d->input_columns * d->input_group_count) || activation_variant(d->option) != 0)
        return false;
    output.create(static_cast<int>(d->output_columns), input.h, sizeof(float), d->vulkan_context->blob_allocator());
    if (output.empty())
        return false;
    record(input, output, command);
    return true;
#else
    (void)input;
    (void)output;
    (void)command;
    (void)workspace;
    return false;
#endif
}

void QnkLinear_vulkan::record(const ncnn::VkMat& input, ncnn::VkMat& output, ncnn::VkCompute& command, uint32_t output_column_offset) const
{
#if NCNN_MOE_WITH_VULKAN
    std::vector<ncnn::VkMat> bindings;
    std::vector<ncnn::vk_constant_type> constants;
    record(input, output, command, output_column_offset, bindings, constants);
#else
    (void)input;
    (void)output;
    (void)command;
    (void)output_column_offset;
#endif
}

void QnkLinear_vulkan::record(const ncnn::VkMat& input,
                              ncnn::VkMat& output,
                              ncnn::VkCompute& command,
                              uint32_t output_column_offset,
                              std::vector<ncnn::VkMat>& bindings,
                              std::vector<ncnn::vk_constant_type>& constants) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    bindings.resize(4);
    bindings[0] = input;
    bindings[1] = implementation.quantized;
    bindings[2] = implementation.bias;
    bindings[3] = output;
    constants.resize(14);
    constants[0].u32 = implementation.input_columns;
    constants[1].u32 = implementation.output_columns;
    constants[2].u32 = implementation.block_count;
    constants[3].u32 = input.dims == 1 ? 1u : static_cast<uint32_t>(input.h);
    constants[4].u32 = implementation.block_size;
    constants[5].u32 = implementation.group_size;
    constants[6].u32 = implementation.group_count;
    constants[7].u32 = implementation.scales_offset;
    constants[8].u32 = implementation.zeros_offset;
    constants[9].u32 = implementation.groups_offset;
    constants[10].u32 = implementation.columns_offset;
    constants[11].u32 = static_cast<uint32_t>(output.w);
    constants[12].u32 = output_column_offset;
    constants[13].u32 = implementation.input_group_count;
    if (implementation.coop_gemm && implementation.block_count == 0
        && implementation.input_group_count == 1 && output_column_offset == 0
        && output.elemsize == sizeof(float) && output.elempack == 1
        && static_cast<uint32_t>(output.w) == implementation.output_columns
        && input.elemsize == sizeof(float) && input.elempack == 1
        && input.dims == 2 && input.h >= 32)
    {
        const auto& gemm_option = implementation.coop_opt;
        ncnn::VulkanDevice* device = implementation.vulkan_context->device();
        ncnn::VkMat input_fp16;
        std::vector<ncnn::VkMat> tops(1);
        ncnn::VkMat projected;
        device->convert_packing(input, input_fp16, 1, 2, command, gemm_option);
        if (!input_fp16.empty()
            && implementation.coop_gemm->forward({input_fp16}, tops, command, gemm_option) == 0
            && !tops[0].empty())
        {
            device->convert_packing(tops[0], projected, 1, 1, command, gemm_option);
            if (!projected.empty())
            {
                output = projected;
                return;
            }
        }
    }
    if ((implementation.tile_gemm[0].pipeline || implementation.tile_gemm[1].pipeline)
        && activation_variant(implementation.option) == 0
        && implementation.block_count == 0
        && implementation.block_size == static_cast<uint64_t>(implementation.input_columns) * sizeof(uint16_t))
    {
        const TileShape shape{constants[3].u32,
                              implementation.input_columns,
                              implementation.output_columns,
                              static_cast<uint32_t>(output.w),
                              output_column_offset,
                              implementation.input_group_count};
        if (record_tile_gemm(implementation.tile_gemm, input, implementation.quantized,
                             implementation.bias, ncnn::VkMat(), ncnn::VkMat(), output, shape, command))
            return;
    }
    const uint32_t tile_rows = constants[3].u32 > 1 ? 4u : 1u;
    const auto& pipeline = tile_rows == 1 ? implementation.pipeline : implementation.prefill_pipeline;
    ncnn::VkMat dispatcher;
    // Vocabulary-sized projections can exceed the guaranteed 65535 workgroups
    // along X. Tile over Z independently of the native block encoding.
    const uint32_t output_groups = (implementation.output_columns + 3u) / 4u;
    const uint32_t column_groups = std::min(output_groups,
                                            std::min(65535u, implementation.vulkan_context->device()->info.max_workgroup_count_x()));
    dispatcher.w = static_cast<int>(column_groups * pipeline->local_size_x());
    dispatcher.h = static_cast<int>((constants[3].u32 + tile_rows - 1) / tile_rows);
    dispatcher.c = static_cast<int>((output_groups + column_groups - 1) / column_groups);
    command.record_pipeline_readonly(pipeline.get(), bindings, {1, 0, 0, 0}, constants, dispatcher);
#else
    (void)input;
    (void)output;
    (void)command;
    (void)output_column_offset;
    (void)bindings;
    (void)constants;
#endif
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& QnkLinear_vulkan::vulkan_context() const noexcept
{
    return d->vulkan_context;
}

const ncnn::Option& QnkLinear_vulkan::option() const noexcept
{
    return d->option;
}
#endif

bool QnkLinear_vulkan::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.vulkan_context
        || !implementation.pipeline
        || input.dtype() != DType::Float32
        || input.rows() == 0
        || input.columns() != implementation.input_columns * implementation.input_group_count
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * input.columns() * sizeof(float),
                                                                                             input.dtype());
    const bool direct_host_output = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * implementation.output_columns * sizeof(float),
                                                                                              output.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download, input.rows(), implementation.output_columns, transfer_slot.staging_allocator))
    {
        return false;
    }
    const DType output_dtype = output.dtype();
    output.reset(input.rows(), implementation.output_columns, false);

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
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
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, implementation.option))
        return false;
    ncnn::VkMat output_gpu;
    if (direct_host_output)
        output_gpu = bind_host_out(transfer_slot.download);
    else
        output_gpu.create(static_cast<int>(implementation.output_columns), static_cast<int>(input.rows()), sizeof(float), implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    record(input_gpu, output_gpu, command);
    if ((!direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     implementation.output_columns,
                                     transfer_slot.download,
                                     command,
                                     implementation.vulkan_context->device(),
                                     implementation.option,
                                     output_dtype))
        || submit_compute_and_wait(command, implementation.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    ++runtime_state.dispatches;
    if (implementation.dtype == DType::Int4)
        ++runtime_state.int4_dispatches;
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

DType QnkLinear_vulkan::dtype() const noexcept
{
    return d->dtype;
}

uint32_t QnkLinear_vulkan::input_columns() const noexcept
{
    return d->input_columns * d->input_group_count;
}

uint32_t QnkLinear_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

uint64_t QnkLinear_vulkan::device_bytes() const noexcept
{
    return d->device_size;
}

QnkExpert_vulkan::QnkExpert_vulkan()
    : d(new Implementation)
{
}

QnkExpert_vulkan::~QnkExpert_vulkan() = default;

uint32_t QnkExpert_vulkan::output_columns() const noexcept
{
    return d->output_columns;
}

uint64_t QnkExpert_vulkan::device_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return (d->gate_up ? d->gate_up->device_bytes() : 0)
           + (d->up ? d->up->device_bytes() : 0)
           + (d->down ? d->down->device_bytes() : 0)
           + (d->router_gate ? d->router_gate->device_bytes() : 0);
#else
    return 0;
#endif
}

std::shared_ptr<QnkExpert_vulkan> QnkExpert_vulkan::create(const TensorData& gate_up,
                                                           const TensorData* gate_up_bias,
                                                           const TensorData& down,
                                                           const TensorData* down_bias,
                                                           float activation_limit,
                                                           uint32_t vulkan_device_index,
                                                           ExpertActivation activation,
                                                           const VulkanRuntimePtr& vulkan_runtime,
                                                           uint64_t optimization_flags)
{
    return create_impl(gate_up, gate_up_bias, nullptr, nullptr, down, down_bias,
                       activation_limit, vulkan_device_index, nullptr, nullptr, activation,
                       vulkan_runtime, optimization_flags);
}

std::shared_ptr<QnkExpert_vulkan> QnkExpert_vulkan::create_separate(const TensorData& gate,
                                                                    const TensorData* gate_bias,
                                                                    const TensorData& up,
                                                                    const TensorData* up_bias,
                                                                    const TensorData& down,
                                                                    const TensorData* down_bias,
                                                                    float activation_limit,
                                                                    uint32_t vulkan_device_index,
                                                                    ExpertActivation activation,
                                                                    const VulkanRuntimePtr& vulkan_runtime,
                                                                    uint64_t optimization_flags)
{
    return create_impl(gate, gate_bias, &up, up_bias, down, down_bias,
                       activation_limit, vulkan_device_index, nullptr, nullptr, activation,
                       vulkan_runtime, optimization_flags);
}

std::shared_ptr<QnkExpert_vulkan> QnkExpert_vulkan::create_impl(const TensorData& gate_up,
                                                                const TensorData* gate_up_bias,
                                                                const TensorData* up,
                                                                const TensorData* up_bias,
                                                                const TensorData& down,
                                                                const TensorData* down_bias,
                                                                float activation_limit,
                                                                uint32_t vulkan_device_index,
                                                                ncnn::VkAllocator* weight_allocator,
                                                                ncnn::VkAllocator* staging_allocator,
                                                                ExpertActivation activation,
                                                                const VulkanRuntimePtr& vulkan_runtime,
                                                                uint64_t optimization_flags,
                                                                VulkanWeightUploadBatch* upload_batch)
{
#if NCNN_MOE_WITH_VULKAN
    const auto supported_dtype = [](DType dtype) {
        return dtype == DType::Int4 || is_qnk_dtype(dtype) || dtype == DType::MxFp4
               || dtype == DType::BFloat16 || dtype == DType::Float32;
    };
    if ((activation != ExpertActivation::Silu
         && activation != ExpertActivation::GptOssSwiGlu
         && activation != ExpertActivation::DeepSeekSwiGlu)
        || !supported_dtype(gate_up.dtype)
        || !supported_dtype(down.dtype)
        || gate_up.shape.size() != 2
        || down.shape.size() != 2
        || gate_up.shape[0] == 0
        || (!up && gate_up.shape[0] % 2 != 0)
        || down.shape[0] == 0
        || down.shape[1] != (up ? gate_up.shape[0] : gate_up.shape[0] / 2)
        || down.shape[1] > static_cast<uint32_t>(std::numeric_limits<int>::max() / 128)
        || (up && (up->shape != gate_up.shape || !supported_dtype(up->dtype)))
        || activation_limit < 0.0f)
    {
        return {};
    }
    if (gate_up_bias
        && (gate_up_bias->shape.size() != 1
            || gate_up_bias->shape[0] != gate_up.shape[0]
            || (gate_up_bias->dtype != DType::Float32 && gate_up_bias->dtype != DType::BFloat16)))
    {
        return {};
    }
    if (down_bias
        && (down_bias->shape.size() != 1
            || down_bias->shape[0] != down.shape[0]
            || (down_bias->dtype != DType::Float32 && down_bias->dtype != DType::BFloat16)))
    {
        return {};
    }

    bool fuse_gate_up = up
                        && has_flag(optimization_flags, OptimizationVulkanQnK)
                        && (is_qnk_dtype(gate_up.dtype) || gate_up.dtype == DType::MxFp4)
                        && up->dtype == gate_up.dtype
                        && down.dtype == gate_up.dtype
                        && !gate_up_bias
                        && !up_bias
                        && !down_bias
                        && !qnk_has_metadata(gate_up)
                        && !qnk_has_metadata(*up)
                        && !qnk_has_metadata(down);

    std::shared_ptr<QnkExpert_vulkan> result(new QnkExpert_vulkan);
    if (upload_batch) upload_batch->retain_owner(result);
    Implementation& implementation = *result->d;
    implementation.device_index = vulkan_device_index;
    if (fuse_gate_up)
    {
        implementation.gate_up = QnkLinear_vulkan::create_with_allocator(gate_up,
                                                                         nullptr,
                                                                         vulkan_device_index,
                                                                         weight_allocator,
                                                                         staging_allocator,
                                                                         vulkan_runtime,
                                                                         optimization_flags,
                                                                         1,
                                                                         up, upload_batch);
        if (upload_batch && !implementation.gate_up) return {};
        fuse_gate_up = static_cast<bool>(implementation.gate_up);
    }
    if (!fuse_gate_up)
        implementation.gate_up = QnkLinear_vulkan::create_with_allocator(gate_up,
                                                                         gate_up_bias,
                                                                         vulkan_device_index,
                                                                         weight_allocator,
                                                                         staging_allocator,
                                                                         vulkan_runtime,
                                                                         optimization_flags, 1, nullptr, upload_batch);
    implementation.down = QnkLinear_vulkan::create_with_allocator(down,
                                                                  down_bias,
                                                                  vulkan_device_index,
                                                                  weight_allocator,
                                                                  staging_allocator,
                                                                  vulkan_runtime,
                                                                  optimization_flags, 1, nullptr, upload_batch);
    if (up && !fuse_gate_up)
    {
        implementation.up = QnkLinear_vulkan::create_with_allocator(*up, up_bias, vulkan_device_index,
                                                                    weight_allocator, staging_allocator,
                                                                    vulkan_runtime, optimization_flags, 1, nullptr, upload_batch);
        if (!implementation.up)
            return {};
    }
    if (!implementation.gate_up || !implementation.down)
        return {};

    const QnkLinear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const QnkLinear_vulkan::Implementation& down_projection = *implementation.down->d;
    if (!gate.vulkan_context || gate.vulkan_context != down_projection.vulkan_context
        || (implementation.up && implementation.up->d->vulkan_context != gate.vulkan_context))
        return {};
    implementation.vulkan_context = gate.vulkan_context;
    implementation.option = gate.option;
    implementation.intermediate_columns = implementation.up ? gate.output_columns : gate.output_columns / 2;
    implementation.output_columns = implementation.down->output_columns();
    implementation.activation_limit = activation == ExpertActivation::Silu ? 0.0f : activation_limit;
    implementation.activation = activation;
    implementation.interleave_gate_up_rows = fuse_gate_up ? false : gate_up.qnk_interleave_rows;
    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        if (!create_qnk_swiglu_pipeline(implementation.vulkan_context,
                                        implementation.option,
                                        implementation.swiglu_pipeline))
        {
            return {};
        }
    }
    return result;
#else
    (void)gate_up;
    (void)gate_up_bias;
    (void)up;
    (void)up_bias;
    (void)down;
    (void)down_bias;
    (void)activation_limit;
    (void)vulkan_device_index;
    (void)weight_allocator;
    (void)staging_allocator;
    (void)activation;
    (void)vulkan_runtime;
    (void)optimization_flags;
    (void)upload_batch;
    return {};
#endif
}

bool QnkExpert_vulkan::prepare_router_gate(const TensorData& weight)
{
#if NCNN_MOE_WITH_VULKAN
    if (!d->vulkan_context || !d->gate_up || weight.shape.size() != 2
        || weight.shape[0] != 1 || weight.shape[1] != d->gate_up->input_columns())
        return false;
    auto prepared = QnkLinear_vulkan::create(weight, nullptr, d->device_index,
                                             d->vulkan_context->runtime(), d->vulkan_context->optimization_flags());
    if (!prepared || prepared->vulkan_context() != d->vulkan_context)
        return false;
    d->router_gate = std::move(prepared);
    return true;
#else
    (void)weight;
    return false;
#endif
}

bool QnkExpert_vulkan::has_router_gate() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return static_cast<bool>(d->router_gate);
#else
    return false;
#endif
}

uint32_t QnkExpert_vulkan::input_columns() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->gate_up ? d->gate_up->input_columns() : 0;
#else
    return 0;
#endif
}

const std::shared_ptr<VulkanContext>& QnkExpert_vulkan::vulkan_context() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->vulkan_context;
#else
    static const std::shared_ptr<VulkanContext> empty;
    return empty;
#endif
}

uint32_t QnkExpert_vulkan::int4_operations() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return (d->gate_up && d->gate_up->dtype() == DType::Int4 ? 1u : 0u)
           + (d->up && d->up->dtype() == DType::Int4 ? 1u : 0u)
           + (d->down && d->down->dtype() == DType::Int4 ? 1u : 0u)
           + (d->router_gate && d->router_gate->dtype() == DType::Int4 ? 1u : 0u);
#else
    return 0;
#endif
}

uint32_t QnkExpert_vulkan::record_dispatches() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return (d->up ? 4u : 3u) + (d->router_gate ? 2u : 0u);
#else
    return 0;
#endif
}

static size_t qnk_expert_scratch_shapes(uint32_t intermediate_columns, uint32_t output_columns,
                                        bool separate_up, bool router_gate, uint64_t columns[5]) noexcept
{
    if (intermediate_columns == 0 || output_columns == 0)
        return 0;
    size_t count = 0;
    columns[count++] = separate_up ? intermediate_columns
                                   : saturating_multiply_u64(intermediate_columns, 2);
    if (separate_up)
        columns[count++] = intermediate_columns;
    columns[count++] = intermediate_columns;
    if (router_gate)
    {
        columns[count++] = output_columns;
        columns[count++] = 1;
    }
    return count;
}

std::vector<uint64_t> QnkExpert_vulkan::scratch_elements(uint32_t intermediate_columns,
                                                         uint32_t output_columns,
                                                         bool separate_up,
                                                         bool router_gate,
                                                         uint64_t rows)
{
    if (intermediate_columns == 0 || output_columns == 0 || rows == 0)
        return {};
    uint64_t columns[5]{};
    const size_t count = qnk_expert_scratch_shapes(intermediate_columns, output_columns,
                                                   separate_up, router_gate, columns);
    std::vector<uint64_t> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index)
    {
        const uint64_t elements = saturating_multiply_u64(columns[index], rows);
        if (elements == UINT64_MAX) return {UINT64_MAX};
        if (elements != 0) result.push_back(elements);
    }
    return result;
}

bool QnkExpert_vulkan::record(const ncnn::VkMat& input, ncnn::VkMat& output,
                              ncnn::VkCompute& command,
                              std::vector<ncnn::VkMat>& workspace,
                              std::vector<ncnn::VkMat>& bindings,
                              std::vector<ncnn::vk_constant_type>& constants) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (&input == &output || !implementation.vulkan_context || !implementation.gate_up || !implementation.down
        || input.empty() || input.dims != 2 || input.elempack != 1 || input.elemsize != sizeof(float)
        || input.h <= 0 || input.w != static_cast<int>(input_columns())
        || !input.allocator || input.allocator->vkdev != implementation.vulkan_context->device()
        || output.empty() || output.dims != 2 || output.elempack != 1 || output.elemsize != sizeof(float)
        || output.h != input.h || output.w != static_cast<int>(implementation.output_columns)
        || !output.allocator || output.allocator->vkdev != implementation.vulkan_context->device())
        return false;
    const QnkLinear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const uint32_t rows = input.dims == 1 ? 1u : static_cast<uint32_t>(input.h);
    uint64_t scratch_columns[5]{};
    const size_t scratch_count = qnk_expert_scratch_shapes(implementation.intermediate_columns,
                                                           implementation.output_columns,
                                                           static_cast<bool>(implementation.up),
                                                           static_cast<bool>(implementation.router_gate),
                                                           scratch_columns);
    if (scratch_count != (implementation.up ? (implementation.router_gate ? 5u : 3u) : (implementation.router_gate ? 4u : 2u))
        || gate.output_columns != scratch_columns[0])
        return false;
    size_t scratch_index = 0;
    const auto create_scratch = [&](ncnn::VkMat& value, uint64_t columns) {
        if (columns == 0 || columns > static_cast<uint64_t>(std::numeric_limits<int>::max())) return false;
        value.create(static_cast<int>(columns), static_cast<int>(rows), sizeof(float), implementation.vulkan_context->blob_allocator());
        return !value.empty();
    };
    ncnn::VkMat gate_gpu;
    if (!create_scratch(gate_gpu, scratch_columns[scratch_index++])) return false;
    ncnn::VkMat up_gpu;
    if (implementation.up)
        if (!create_scratch(up_gpu, scratch_columns[scratch_index++])) return false;
    ncnn::VkMat intermediate_gpu;
    if (!create_scratch(intermediate_gpu, scratch_columns[scratch_index++])) return false;
    workspace.push_back(gate_gpu);
    workspace.push_back(intermediate_gpu);
    if (implementation.up)
        workspace.push_back(up_gpu);
    implementation.gate_up->record(input, gate_gpu, command, 0, bindings, constants);
    if (implementation.up)
        implementation.up->record(input, up_gpu, command, 0, bindings, constants);
    bindings.resize(3);
    bindings[0] = gate_gpu;
    bindings[1] = intermediate_gpu;
    bindings[2] = implementation.up ? up_gpu : gate_gpu;
    constants.resize(5);
    constants[0].u32 = implementation.intermediate_columns;
    constants[1].u32 = rows;
    constants[2].u32 = implementation.activation == ExpertActivation::GptOssSwiGlu ? 0u : 1u;
    constants[3].f = implementation.activation_limit;
    constants[4].u32 = implementation.up ? 2u : implementation.interleave_gate_up_rows ? 0u
                                                                                       : 1u;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(implementation.intermediate_columns);
    dispatcher.h = static_cast<int>(rows);
    dispatcher.c = 1;
    command.record_pipeline(implementation.swiglu_pipeline.get(), bindings, constants, dispatcher);
    ncnn::VkMat unscaled_output = output;
    ncnn::VkMat router_logits;
    if (implementation.router_gate)
    {
        if (!create_scratch(unscaled_output, scratch_columns[scratch_index++])
            || !create_scratch(router_logits, scratch_columns[scratch_index++])) return false;
        workspace.push_back(unscaled_output);
        workspace.push_back(router_logits);
    }
    implementation.down->record(intermediate_gpu, unscaled_output, command, 0, bindings, constants);
    if (implementation.router_gate)
    {
        implementation.router_gate->record(input, router_logits, command, 0, bindings, constants);
        bindings = {unscaled_output, output, router_logits};
        constants.resize(5);
        constants[0].u32 = implementation.output_columns;
        constants[1].u32 = rows;
        constants[2].u32 = 2;
        constants[3].f = 0.f;
        constants[4].u32 = 2;
        dispatcher.w = static_cast<int>(implementation.output_columns);
        command.record_pipeline(implementation.swiglu_pipeline.get(), bindings, constants, dispatcher);
    }
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

bool QnkExpert_vulkan::forward_batch(std::span<const QnkExpert_vulkan*> experts,
                                     std::span<const ActivationBuffer*> inputs,
                                     std::span<ActivationBuffer*> outputs)
{
#if NCNN_MOE_WITH_VULKAN
    if (experts.empty() || experts.size() != inputs.size() || experts.size() != outputs.size())
        return false;

    const Implementation* first_expert_ptr = nullptr;
    const QnkLinear_vulkan::Implementation* first_gate_ptr = nullptr;
    const QnkLinear_vulkan::Implementation* first_down_ptr = nullptr;
    uint32_t input_columns = 0;
    uint32_t output_columns = 0;
    size_t total_rows = 0;
    bool shared_input = true;
    bool batch_compatible = true;
    uint64_t int4_ops = 0;
    for (size_t index = 0; index < experts.size(); ++index)
    {
        const QnkExpert_vulkan* operator_instance = experts[index];
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
            || output->element_size() == 0)
        {
            return false;
        }
        if (output->dtype() != DType::Float32)
            batch_compatible = false;
        const Implementation& expert = *operator_instance->d;
        if (!expert.gate_up
            || !expert.down
            || !expert.vulkan_context
            || !expert.swiglu_pipeline
            || !expert.gate_up->d
            || !expert.down->d
            || (expert.up && !expert.up->d))
        {
            return false;
        }
        const QnkLinear_vulkan::Implementation& gate = *expert.gate_up->d;
        const QnkLinear_vulkan::Implementation& down = *expert.down->d;
        if (!gate.pipeline
            || !down.pipeline
            || gate.vulkan_context != expert.vulkan_context
            || down.vulkan_context != expert.vulkan_context
            || gate.input_columns == 0
            || gate.output_columns == 0
            || expert.intermediate_columns == 0
            || expert.output_columns == 0
            || down.input_columns != expert.intermediate_columns
            || down.output_columns != expert.output_columns)
        {
            return false;
        }
        if (expert.up)
        {
            const QnkLinear_vulkan::Implementation& up = *expert.up->d;
            if (!up.pipeline
                || up.vulkan_context != expert.vulkan_context
                || gate.output_columns != expert.intermediate_columns
                || up.input_columns != gate.input_columns
                || up.output_columns != expert.intermediate_columns)
            {
                return false;
            }
        }
        else if (gate.output_columns % 2 != 0
                 || gate.output_columns / 2 != expert.intermediate_columns)
        {
            return false;
        }

        if (input->columns() != gate.input_columns
            || static_cast<size_t>(gate.input_columns) > std::numeric_limits<size_t>::max() / sizeof(float))
        {
            return false;
        }
        const size_t input_row_bytes = static_cast<size_t>(gate.input_columns) * sizeof(float);
        if (input->rows() > std::numeric_limits<size_t>::max() / input_row_bytes
            || input->bytes().size() != input->rows() * input_row_bytes)
        {
            return false;
        }

        if (!first_expert_ptr)
        {
            first_expert_ptr = &expert;
            first_gate_ptr = &gate;
            first_down_ptr = &down;
            input_columns = gate.input_columns;
            output_columns = down.output_columns;
        }
        else if (expert.vulkan_context != first_expert_ptr->vulkan_context
                 || expert.intermediate_columns != first_expert_ptr->intermediate_columns
                 || expert.output_columns != first_expert_ptr->output_columns
                 || expert.activation != first_expert_ptr->activation
                 || expert.activation_limit != first_expert_ptr->activation_limit
                 || static_cast<bool>(expert.up) != static_cast<bool>(first_expert_ptr->up)
                 || expert.interleave_gate_up_rows != first_expert_ptr->interleave_gate_up_rows
                 || gate.input_columns != first_gate_ptr->input_columns
                 || gate.output_columns != first_gate_ptr->output_columns
                 || down.input_columns != first_down_ptr->input_columns
                 || down.output_columns != first_down_ptr->output_columns)
        {
            // These Experts are individually valid, but cannot share the
            // single upload/output geometry used by the batched path.
            batch_compatible = false;
        }

        int4_ops += gate.dtype == DType::Int4 ? 1 : 0;
        int4_ops += down.dtype == DType::Int4 ? 1 : 0;
        if (expert.up && expert.up->d->dtype == DType::Int4)
            ++int4_ops;
        if (batch_compatible)
        {
            const size_t uint32_max = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
            if (input->rows() > uint32_max - total_rows)
                batch_compatible = false;
            else
                total_rows += input->rows();
        }
        shared_input = shared_input && input == inputs.front();
    }
    if (batch_compatible && total_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        batch_compatible = false;
    if (!batch_compatible)
    {
        // Complete the inexpensive validation pass before allowing any
        // per-Expert execution, then fall back once for static incompatibility.
        for (size_t index = 0; index < experts.size(); ++index)
            if (!experts[index]->forward(*inputs[index], *outputs[index]))
                return false;
        return true;
    }

    const Implementation& first_expert = *first_expert_ptr;
    const QnkLinear_vulkan::Implementation& first_gate = *first_gate_ptr;
    const QnkLinear_vulkan::Implementation& first_down = *first_down_ptr;
    const DType output_dtype = DType::Float32;

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
        std::memcpy(mapped_input_data + row_offset * static_cast<size_t>(input_columns) * sizeof(float),
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
    if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, first_gate.option))
        return false;
    ncnn::VkMat output_gpu;
    output_gpu.create(static_cast<int>(output_columns),
                      static_cast<int>(total_rows),
                      sizeof(float),
                      first_expert.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;

    const size_t descriptor_alignment = std::max<size_t>(4, first_expert.vulkan_context->device()->info.buffer_offset_alignment());
    const bool align_output_rows = experts.size() > 1
                                   && (static_cast<size_t>(output_columns) * output_gpu.elemsize) % descriptor_alignment != 0;
    const size_t workspace_count = first_expert.up ? 4u : 3u;
    std::vector<ncnn::VkMat> workspace;
    if (experts.size() > workspace.max_size() / workspace_count)
        return false;
    workspace.reserve(experts.size() * workspace_count);
    std::vector<ncnn::VkMat> aligned_inputs;
    std::vector<ncnn::VkMat> bindings;
    bindings.reserve(4);
    std::vector<ncnn::vk_constant_type> constants;
    constants.reserve(15);
    row_offset = 0;
    for (size_t index = 0; index < experts.size(); ++index)
    {
        const size_t rows = inputs[index]->rows();
        ncnn::VkMat input_view = aligned_row_view(input_gpu, shared_input ? 0 : row_offset, rows, command, first_gate.option,
                                                  first_expert.vulkan_context, aligned_inputs, experts.size());
        ncnn::VkMat output_view = row_view(output_gpu, row_offset, rows);
        if (input_view.empty() || output_view.empty())
            return false;
        if (align_output_rows)
        {
            ncnn::VkMat aligned_output;
            aligned_output.create(output_view.w, output_view.h, output_view.elemsize,
                                  first_expert.vulkan_context->blob_allocator());
            if (aligned_output.empty()) return false;
            output_view = aligned_output;
            workspace.push_back(aligned_output);
        }

        if (!experts[index]->record(input_view, output_view, command, workspace, bindings, constants))
            return false;
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
                                 output_dtype)
        || submit_compute_and_wait(command, first_expert.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batches(transfer_slot.download,
                                        inputs,
                                        outputs,
                                        output_columns))
    {
        return false;
    }
    bindings.clear();
    constants.clear();
    workspace.clear();
    aligned_inputs.clear();
    lock.unlock();
    runtime_state.dispatches += static_cast<uint64_t>(experts.size()) * (first_expert.up ? 4u : 3u);
    if (int4_ops != 0)
        runtime_state.int4_dispatches += int4_ops;
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

bool QnkExpert_vulkan::forward(const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    if (!implementation.gate_up
        || !implementation.down
        || !implementation.vulkan_context
        || !implementation.swiglu_pipeline
        || input.rows() == 0
        || input.rows() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())
        || input.columns() != implementation.gate_up->input_columns())
    {
        return false;
    }
    const QnkLinear_vulkan::Implementation& gate = *implementation.gate_up->d;
    const QnkLinear_vulkan::Implementation& down = *implementation.down->d;
    if ((!implementation.up && gate.output_columns % 2 != 0)
        || down.input_columns != implementation.intermediate_columns
        || down.output_columns != implementation.output_columns)
    {
        return false;
    }

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * gate.input_columns * sizeof(float),
                                                                                             input.dtype());
    const bool direct_host_output = implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(input.rows()) * down.output_columns * sizeof(float),
                                                                                              output.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator)
        || !prepare_staging_batch(transfer_slot.download,
                                  input.rows(),
                                  down.output_columns,
                                  transfer_slot.staging_allocator))
    {
        return false;
    }

    const DType output_dtype = output.dtype();
    output.reset(input.rows(), down.output_columns, false);
    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
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
    else if (!record_mapped_upload(transfer_slot.upload, input_gpu, command, gate.option))
        return false;

    ncnn::VkMat output_gpu;
    if (direct_host_output)
    {
        output_gpu = bind_host_out(transfer_slot.download);
    }
    else
    {
        output_gpu.create(static_cast<int>(down.output_columns),
                          static_cast<int>(input.rows()),
                          sizeof(float),
                          implementation.vulkan_context->blob_allocator());
    }
    if (output_gpu.empty())
        return false;

    std::vector<ncnn::VkMat> workspace;
    workspace.reserve(implementation.up ? 3u : 2u);
    std::vector<ncnn::VkMat> bindings;
    bindings.reserve(4);
    std::vector<ncnn::vk_constant_type> constants;
    constants.reserve(15);
    if (!record(input_gpu, output_gpu, command, workspace, bindings, constants))
        return false;

    if ((!direct_host_output
         && !record_staging_download(output_gpu,
                                     input.rows(),
                                     down.output_columns,
                                     transfer_slot.download,
                                     command,
                                     implementation.vulkan_context->device(),
                                     down.option,
                                     output_dtype))
        || submit_compute_and_wait(command, implementation.vulkan_context->device()) != 0
        || !copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        return false;
    }
    uint64_t int4_ops = 0;
    int4_ops += implementation.gate_up->dtype() == DType::Int4 ? 1 : 0;
    int4_ops += implementation.down->dtype() == DType::Int4 ? 1 : 0;
    if (implementation.up && implementation.up->dtype() == DType::Int4)
        ++int4_ops;
    runtime_state.dispatches += (implementation.up ? 4 : 3) + (implementation.router_gate ? 2 : 0);
    if (int4_ops != 0)
        runtime_state.int4_dispatches += int4_ops;
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

bool QnkExpert_vulkan::forward_device(const DeviceTensor_vulkan& input, DeviceTensor_vulkan& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    const ncnn::VkMat* source = input.value();
    if (&input == &output || !implementation.gate_up || !implementation.down
        || !implementation.vulkan_context || !implementation.swiglu_pipeline
        || !source || source->dims != 2 || source->elemsize / source->elempack != sizeof(float)
        || input.vulkan_context() != implementation.vulkan_context
        || input.rows() == 0 || input.columns() != implementation.gate_up->input_columns()
        || input.rows() > implementation.vulkan_context->device()->info.max_workgroup_count_y()
        || activation_variant(implementation.option) != 0)
        return false;
    const ncnn::VkMat retained_input = *source;
    auto lease = implementation.vulkan_context->acquire_transfer_slot();
    auto& slot = lease.slot();
    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    if (slot.command_used && slot.command->reset() != 0)
        return false;
    slot.command_used = true;
    auto& command = *slot.command;
    input.prepare_read();
    ncnn::VkMat input_gpu = retained_input;
    if (input_gpu.elempack != 1)
    {
        implementation.vulkan_context->device()->convert_packing(retained_input, input_gpu, 1, 1, command, implementation.option);
        if (input_gpu.empty())
            return false;
    }
    ncnn::VkMat output_gpu;
    output_gpu.create(static_cast<int>(implementation.output_columns), input_gpu.h, sizeof(float), implementation.vulkan_context->blob_allocator());
    if (output_gpu.empty())
        return false;
    std::vector<ncnn::VkMat> workspace;
    std::vector<ncnn::VkMat> bindings;
    std::vector<ncnn::vk_constant_type> constants;
    if (!record(input_gpu, output_gpu, command, workspace, bindings, constants)
        || submit_compute_and_wait(command, implementation.vulkan_context->device()) != 0)
        return false;
    DeviceTensor_vulkan completed;
    if (!completed.assign_completed(output_gpu, implementation.vulkan_context))
        return false;
    output = std::move(completed);
    auto& state = implementation.vulkan_context->runtime_state();
    state.dispatches += (implementation.up ? 4 : 3) + (implementation.router_gate ? 2 : 0);
    state.int4_dispatches += implementation.gate_up->dtype() == DType::Int4 ? 1 : 0;
    state.int4_dispatches += implementation.down->dtype() == DType::Int4 ? 1 : 0;
    state.int4_dispatches += implementation.up && implementation.up->dtype() == DType::Int4 ? 1 : 0;
    ++state.compute_submissions;
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
}

bool QnkExpert_vulkan::materialize(const DeviceTensor_vulkan& input, ActivationBuffer& output) const
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

} // namespace moe
} // namespace ncnn
