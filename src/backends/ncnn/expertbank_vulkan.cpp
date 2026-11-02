#include "expertbank_vulkan.h"
#include "experttransfer_vulkan.h"
#include "vulkancontext.h"
#include "kernels/ops.h"
#include "ncnn/moe/option.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/expertbank_gate_up.comp.hex.h"
#include "kernels/vulkan/expertbank_down.comp.hex.h"
#include "kernels/vulkan/expertbank_combine.comp.hex.h"
#endif

namespace ncnn {
namespace moe {

class ExpertBank_vulkan::Implementation
{
public:
    std::shared_ptr<VulkanContext> context;
    ExpertBankPlan_vulkan plan;
#if NCNN_MOE_WITH_VULKAN
    std::unique_ptr<VulkanExpertWeightAllocator> allocator;
    ncnn::VkMat weights;
    std::shared_ptr<ncnn::Pipeline> gate_up;
    std::shared_ptr<ncnn::Pipeline> down;
    std::shared_ptr<ncnn::Pipeline> combine;
#endif
};

ExpertBank_vulkan::ExpertBank_vulkan() : d(new Implementation)
{
}

ExpertBank_vulkan::~ExpertBank_vulkan()
{
#if NCNN_MOE_WITH_VULKAN
    if (d->context)
    {
        const auto context = d->context;
        // The final pipeline deleter takes this mutex itself. Retain all
        // pipeline owners until the allocator lock below has been released.
        const auto gate_up = d->gate_up;
        const auto down = d->down;
        const auto combine = d->combine;
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        d.reset();
    }
#endif
}

const std::shared_ptr<VulkanContext>& ExpertBank_vulkan::context() const noexcept
{
    return d->context;
}

const ExpertBankPlan_vulkan& ExpertBank_vulkan::description() const noexcept
{
    return d->plan;
}

uint64_t ExpertBank_vulkan::physical_size() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->allocator ? d->allocator->physical_bytes() : 0;
#else
    return 0;
#endif
}

static uint64_t bank_aligned(uint64_t bytes, uint64_t alignment) noexcept
{
    if (alignment == 0 || bytes > std::numeric_limits<uint64_t>::max() - alignment + 1) return 0;
    return ((bytes + alignment - 1) / alignment) * alignment;
}

static uint64_t bank_scratch_shapes(const ExpertBankPlan_vulkan& plan, uint64_t rows, uint32_t top_k,
                                   uint32_t columns[3], uint64_t heights[3]) noexcept
{
    const uint64_t route_rows = saturating_multiply_u64(rows, top_k);
    columns[0] = plan.intermediate_columns;
    columns[1] = plan.output_columns;
    columns[2] = plan.output_columns;
    heights[0] = route_rows;
    heights[1] = route_rows;
    heights[2] = rows;
    return route_rows;
}

std::vector<uint64_t> ExpertBank_vulkan::scratch_elements(const ExpertBankPlan_vulkan& plan,
                                                          uint64_t rows, uint32_t top_k)
{
    if (rows == 0)
        return {};
    if (top_k == 0 || top_k > 64 || top_k > plan.expert_count
        || plan.intermediate_columns == 0 || plan.output_columns == 0)
        return {UINT64_MAX};
    uint32_t columns[3];
    uint64_t heights[3];
    if (bank_scratch_shapes(plan, rows, top_k, columns, heights) == UINT64_MAX)
        return {UINT64_MAX};
    std::vector<uint64_t> result;
    result.reserve(2);
    for (size_t index = 0; index < 2; ++index)
    {
        const uint64_t elements = saturating_multiply_u64(columns[index], heights[index]);
        if (elements == UINT64_MAX) return {UINT64_MAX};
        if (elements != 0) result.push_back(elements);
    }
    return result;
}

#if NCNN_MOE_WITH_VULKAN
static bool bank_bias_metadata(const TensorData* bias, uint32_t columns)
{
    return !bias || (bias->shape.size() == 1 && bias->shape[0] == columns && (bias->dtype == DType::Float32 || bias->dtype == DType::BFloat16));
}

static bool bank_metadata(const ExpertBankWeights_vulkan& weights, ExpertBankPlan_vulkan& plan)
{
    if (!weights.gate_up || !weights.down || weights.gate_up->shape.size() != 2 || weights.down->shape.size() != 2
        || (weights.gate_up->dtype != DType::MxFp4 && weights.gate_up->dtype != DType::BFloat16 && weights.gate_up->dtype != DType::Float32)
        || weights.down->dtype != weights.gate_up->dtype || !weights.gate_up->input_column_indices.empty()
        || !weights.down->input_column_indices.empty() || !std::isfinite(weights.activation_limit)
        || (weights.activation != ExpertActivation::Silu && weights.activation != ExpertActivation::GptOssSwiGlu
            && !(weights.activation == ExpertActivation::DeepSeekSwiGlu && weights.activation_limit <= 0.0f))) return false;
    const bool separate = weights.layout == ExpertLayout::GateUpDown;
    if (!separate && weights.layout != ExpertLayout::PackedGateUpDown && weights.layout != ExpertLayout::InterleavedGateUpDown) return false;
    if ((weights.gate_up->mxfp4_file_storage && weights.gate_up->mxfp4_file_storage->interleave_rows
         && weights.layout != ExpertLayout::InterleavedGateUpDown)
        || (weights.up && weights.up->mxfp4_file_storage && weights.up->mxfp4_file_storage->interleave_rows)
        || (weights.down->mxfp4_file_storage && weights.down->mxfp4_file_storage->interleave_rows)) return false;
    if ((separate && (!weights.up || weights.up->dtype != weights.gate_up->dtype || weights.up->shape != weights.gate_up->shape || !weights.up->input_column_indices.empty()))
        || (!separate && (weights.up || weights.up_bias || weights.gate_up->shape[0] % 2 != 0))) return false;
    plan.dtype = weights.gate_up->dtype;
    plan.layout = weights.layout;
    plan.activation = weights.activation;
    plan.activation_limit = weights.activation_limit;
    plan.input_columns = weights.gate_up->shape[1];
    plan.intermediate_columns = weights.gate_up->shape[0] / (separate ? 1u : 2u);
    plan.output_columns = weights.down->shape[0];
    if (plan.input_columns == 0 || plan.intermediate_columns == 0 || plan.output_columns == 0
        || plan.intermediate_columns > std::numeric_limits<uint32_t>::max() / 2
        || weights.down->shape[1] != plan.intermediate_columns
        || !bank_bias_metadata(weights.gate_up_bias, weights.gate_up->shape[0])
        || !bank_bias_metadata(weights.up_bias, plan.intermediate_columns)
        || !bank_bias_metadata(weights.down_bias, plan.output_columns)) return false;
    return plan.dtype != DType::MxFp4 || (plan.input_columns % 32 == 0 && plan.intermediate_columns % 32 == 0);
}

static bool same_bank_shape(const ExpertBankPlan_vulkan& a, const ExpertBankPlan_vulkan& b)
{
    return a.dtype == b.dtype && a.layout == b.layout && a.activation == b.activation && a.activation_limit == b.activation_limit
           && a.input_columns == b.input_columns && a.intermediate_columns == b.intermediate_columns && a.output_columns == b.output_columns;
}

static bool bank_storage_layout(ExpertBankPlan_vulkan& plan)
{
    const uint64_t gate_elements = static_cast<uint64_t>(plan.intermediate_columns) * 2 * plan.input_columns;
    const uint64_t down_elements = static_cast<uint64_t>(plan.output_columns) * plan.intermediate_columns;
    const uint64_t scalar_size = plan.dtype == DType::Float32 ? 4 : 2;
    const uint64_t maximum = static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (plan.dtype != DType::MxFp4 && (gate_elements > maximum / scalar_size || down_elements > maximum / scalar_size)) return false;
    const uint64_t gate_bytes = plan.dtype == DType::MxFp4 ? gate_elements / 2 : gate_elements * scalar_size;
    const uint64_t down_bytes = plan.dtype == DType::MxFp4 ? down_elements / 2 : down_elements * scalar_size;
    const uint64_t gate_scales = plan.dtype == DType::MxFp4 ? gate_elements / 32 : 0;
    const uint64_t down_scales = plan.dtype == DType::MxFp4 ? down_elements / 32 : 0;
    if (gate_bytes > maximum || down_bytes > maximum) return false;
    plan.gate_scales = bank_aligned(gate_bytes, 4);
    plan.gate_bias = bank_aligned(plan.gate_scales + gate_scales, 4);
    plan.gate_stride = bank_aligned(plan.gate_bias + static_cast<uint64_t>(plan.intermediate_columns) * 2 * sizeof(float), 4);
    plan.down_scales = bank_aligned(down_bytes, 4);
    plan.down_bias = bank_aligned(plan.down_scales + down_scales, 4);
    plan.down_stride = bank_aligned(plan.down_bias + static_cast<uint64_t>(plan.output_columns) * sizeof(float), 4);
    if (!plan.gate_stride || !plan.down_stride || plan.gate_stride > maximum / plan.expert_count
        || plan.down_stride > maximum / plan.expert_count) return false;
    plan.down_weights = plan.gate_stride * plan.expert_count;
    const uint64_t down_bank = plan.down_stride * plan.expert_count;
    if (plan.down_weights > maximum - down_bank) return false;
    plan.storage_size = bank_aligned(plan.down_weights + down_bank, plan.alignment);
    return plan.storage_size != 0 && plan.storage_size <= maximum;
}

static Result<uint64_t> bank_physical_size(const ExpertBankPlan_vulkan& plan, const std::shared_ptr<VulkanContext>& context)
{
    // Match the exact single-block allocator below, including its queue-family
    // sharing flags. This queries requirements without allocating device memory.
    const auto* device = context->device();
    const uint32_t families[] = {device->info.compute_queue_family_index(), device->info.transfer_queue_family_index()};
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = plan.storage_size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = families[0] == families[1] ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT;
    info.queueFamilyIndexCount = families[0] == families[1] ? 0 : 2;
    info.pQueueFamilyIndices = families[0] == families[1] ? nullptr : families;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device->vkdevice(), &info, nullptr, &buffer) != VK_SUCCESS)
        return Error{ErrorCode::InternalError, "cannot query full-resident Expert bank allocation requirements"};
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device->vkdevice(), buffer, &requirements);
    vkDestroyBuffer(device->vkdevice(), buffer, nullptr);
    if (requirements.size == 0) return Error{ErrorCode::InternalError, "Expert bank allocation requirements are empty"};
    return static_cast<uint64_t>(requirements.size);
}

static bool bank_payload(const TensorData& matrix)
{
    if (matrix.dtype == DType::Float32) return matrix.float32_values().size() == matrix.element_count();
    if (matrix.dtype == DType::BFloat16) return matrix.bfloat16_values().size() == matrix.element_count();
    return matrix.dtype == DType::MxFp4 && matrix.mxfp4_blocks.size() == matrix.element_count() / 2
           && matrix.mxfp4_scales.size() == matrix.element_count() / 32;
}

static bool bank_bias_payload(const TensorData* bias)
{
    return !bias || (bias->dtype == DType::Float32 ? bias->float32_values().size() == bias->element_count() : bias->bfloat16_values().size() == bias->element_count());
}

static float bank_bias_value(const TensorData* bias, size_t index)
{
    if (!bias) return 0.0f;
    return bias->dtype == DType::Float32 ? bias->float32_values()[index] : bfloat16_to_float(bias->bfloat16_values()[index]);
}

static const uint8_t* bank_native_values(const TensorData& matrix)
{
    return matrix.dtype == DType::Float32 ? reinterpret_cast<const uint8_t*>(matrix.float32_values().data())
                                          : reinterpret_cast<const uint8_t*>(matrix.bfloat16_values().data());
}

static bool pack_bank_expert(const ExpertBankWeights_vulkan& weights, const ExpertBankPlan_vulkan& plan,
                             uint32_t expert, std::span<uint8_t> storage)
{
    ExpertBankPlan_vulkan actual;
    if (!bank_metadata(weights, actual) || !same_bank_shape(plan, actual) || !bank_payload(*weights.gate_up)
        || !bank_payload(*weights.down) || (weights.up && !bank_payload(*weights.up))
        || !bank_bias_payload(weights.gate_up_bias) || !bank_bias_payload(weights.up_bias) || !bank_bias_payload(weights.down_bias)) return false;
    uint8_t* gate = storage.data() + expert * plan.gate_stride;
    uint8_t* down = storage.data() + plan.down_weights + expert * plan.down_stride;
    const bool separate = plan.layout == ExpertLayout::GateUpDown;
    const bool interleaved = plan.layout == ExpertLayout::InterleavedGateUpDown;
    const size_t row_bytes = plan.dtype == DType::MxFp4 ? plan.input_columns / 2
                                                        : static_cast<size_t>(plan.input_columns) * (plan.dtype == DType::Float32 ? 4 : 2);
    const size_t row_scales = plan.dtype == DType::MxFp4 ? plan.input_columns / 32 : 0;
    for (uint32_t column = 0; column < plan.intermediate_columns; ++column)
    {
        const size_t gate_row = separate ? column : interleaved ? 2 * column
                                                                : column;
        const size_t up_row = separate ? column : interleaved ? 2 * column + 1
                                                              : column + plan.intermediate_columns;
        const TensorData& up = separate ? *weights.up : *weights.gate_up;
        if (plan.dtype == DType::MxFp4)
        {
            std::memcpy(gate + 2 * column * row_bytes, weights.gate_up->mxfp4_blocks.data() + gate_row * row_bytes, row_bytes);
            std::memcpy(gate + (2 * column + 1) * row_bytes, up.mxfp4_blocks.data() + up_row * row_bytes, row_bytes);
            std::memcpy(gate + plan.gate_scales + 2 * column * row_scales, weights.gate_up->mxfp4_scales.data() + gate_row * row_scales, row_scales);
            std::memcpy(gate + plan.gate_scales + (2 * column + 1) * row_scales, up.mxfp4_scales.data() + up_row * row_scales, row_scales);
        }
        else
        {
            std::memcpy(gate + 2 * column * row_bytes, bank_native_values(*weights.gate_up) + gate_row * row_bytes, row_bytes);
            std::memcpy(gate + (2 * column + 1) * row_bytes, bank_native_values(up) + up_row * row_bytes, row_bytes);
        }
        float* bias = reinterpret_cast<float*>(gate + plan.gate_bias);
        bias[2 * column] = bank_bias_value(weights.gate_up_bias, gate_row);
        bias[2 * column + 1] = bank_bias_value(separate ? weights.up_bias : weights.gate_up_bias, up_row);
    }
    const size_t gate_bytes = 2 * plan.intermediate_columns * row_bytes;
    std::memset(gate + gate_bytes, 0, static_cast<size_t>(plan.gate_scales) - gate_bytes);
    const size_t gate_scale_end = static_cast<size_t>(plan.gate_scales) + 2 * plan.intermediate_columns * row_scales;
    std::memset(gate + gate_scale_end, 0, static_cast<size_t>(plan.gate_bias) - gate_scale_end);
    const size_t gate_bias_end = static_cast<size_t>(plan.gate_bias) + 2 * plan.intermediate_columns * sizeof(float);
    std::memset(gate + gate_bias_end, 0, static_cast<size_t>(plan.gate_stride) - gate_bias_end);
    const size_t down_bytes = plan.dtype == DType::MxFp4 ? weights.down->mxfp4_blocks.size()
                                                         : weights.down->element_count() * (plan.dtype == DType::Float32 ? 4 : 2);
    std::memcpy(down, plan.dtype == DType::MxFp4 ? weights.down->mxfp4_blocks.data() : bank_native_values(*weights.down), down_bytes);
    std::memset(down + down_bytes, 0, static_cast<size_t>(plan.down_scales) - down_bytes);
    const size_t down_scale_bytes = plan.dtype == DType::MxFp4 ? weights.down->mxfp4_scales.size() : 0;
    if (down_scale_bytes) std::memcpy(down + plan.down_scales, weights.down->mxfp4_scales.data(), down_scale_bytes);
    const size_t down_scale_end = static_cast<size_t>(plan.down_scales) + down_scale_bytes;
    std::memset(down + down_scale_end, 0, static_cast<size_t>(plan.down_bias) - down_scale_end);
    float* down_bias = reinterpret_cast<float*>(down + plan.down_bias);
    for (uint32_t column = 0; column < plan.output_columns; ++column) down_bias[column] = bank_bias_value(weights.down_bias, column);
    const size_t down_bias_end = static_cast<size_t>(plan.down_bias) + plan.output_columns * sizeof(float);
    std::memset(down + down_bias_end, 0, static_cast<size_t>(plan.down_stride) - down_bias_end);
    return true;
}

static bool create_bank_pipeline(const std::shared_ptr<VulkanContext>& context, const ncnn::Option& option,
                                 const char* shader, size_t size, uint32_t group_size, uint32_t weight_format,
                                 bool specialize_format, std::shared_ptr<ncnn::Pipeline>& output)
{
    const auto binary = context->shader_binary(shader, static_cast<int>(size), option, 0);
    if (!binary || binary->empty()) return false;
    // Construct the control block outside the command lock: allocation failure
    // invokes its deleter, which must also be safe for an unprepared pipeline.
    auto pipeline = std::shared_ptr<ncnn::Pipeline>(new ncnn::Pipeline(context->device()), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        context.reset();
    });
    const uint64_t key = weight_format | (static_cast<uint64_t>(option.use_subgroup_ops) << 8)
                         | (static_cast<uint64_t>(group_size) << 16);
    std::vector<ncnn::vk_specialization_type> specializations;
    if (specialize_format)
    {
        specializations.resize(1);
        specializations[0].u32 = weight_format;
    }
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    output = context->find_pipeline(shader, key);
    if (output) return true;
    pipeline->set_local_size_xyz(group_size, 1, 1);
    if (pipeline->create(binary->data(), binary->size() * sizeof(uint32_t), specializations) != 0) return false;
    output = pipeline;
    context->cache_pipeline(shader, key, output);
    return true;
}
#endif

Result<ExpertBankPlan_vulkan> ExpertBank_vulkan::plan(std::span<const ExpertBankWeights_vulkan> experts,
                                                      const std::shared_ptr<VulkanContext>& context)
{
#if NCNN_MOE_WITH_VULKAN
    if (!context || experts.empty() || experts.size() > 2048
        || !has_flag(context->optimization_flags(), OptimizationVulkanIndexedExperts))
        return Error{ErrorCode::UnsupportedModel, "full-resident Expert bank capability is unavailable"};
    ExpertBankPlan_vulkan plan;
    if (!bank_metadata(experts.front(), plan)) return Error{ErrorCode::UnsupportedModel, "unsupported full-resident Expert layout, dtype, or activation"};
    for (const auto& expert : experts)
    {
        ExpertBankPlan_vulkan current;
        if (!bank_metadata(expert, current) || !same_bank_shape(plan, current))
            return Error{ErrorCode::UnsupportedModel, "full-resident Expert bank requires homogeneous layer metadata"};
    }
    plan.expert_count = static_cast<uint32_t>(experts.size());
    plan.alignment = std::max<size_t>(4, context->device()->info.buffer_offset_alignment());
    const auto& limits = context->device()->info.physicalDeviceProperties().limits;
    if (limits.maxPerStageDescriptorStorageBuffers < 6 || limits.maxDescriptorSetStorageBuffers < 6
        || limits.maxPushConstantsSize < 64 || limits.maxComputeWorkGroupInvocations < 128
        || limits.maxComputeWorkGroupSize[0] < 128 || limits.maxComputeWorkGroupCount[0] < (static_cast<uint64_t>(std::max(plan.intermediate_columns, plan.output_columns)) + 3) / 4
        || std::max(plan.intermediate_columns, plan.output_columns) > static_cast<uint32_t>((std::numeric_limits<int>::max() / 128) * 4)
        || !bank_storage_layout(plan) || plan.storage_size > limits.maxStorageBufferRange)
        return Error{ErrorCode::UnsupportedModel, "full-resident Expert bank exceeds device buffer or dispatch capability"};
    bool found_device = false;
    for (int index = 0; index < ncnn::get_gpu_count(); ++index)
        if (ncnn::get_gpu_device(index) == context->device())
        {
            plan.device_index = static_cast<uint32_t>(index);
            found_device = true;
            break;
        }
    if (!found_device) return Error{ErrorCode::InternalError, "full-resident Expert bank Context has no device index"};
    auto physical = bank_physical_size(plan, context);
    if (!physical) return physical.error();
    plan.physical_size = physical.value();
    return plan;
#else
    (void)experts;
    (void)context;
    return Error{ErrorCode::UnsupportedModel, "Vulkan Expert banks are unavailable in this build"};
#endif
}

Result<std::shared_ptr<ExpertBank_vulkan>> ExpertBank_vulkan::create(const ExpertBankPlan_vulkan& plan,
                                                                     const std::shared_ptr<VulkanContext>& context,
                                                                     const ExpertBankLoader_vulkan& loader)
{
#if NCNN_MOE_WITH_VULKAN
    if (!context || !loader || plan.expert_count == 0 || plan.physical_size == 0 || plan.storage_size == 0
        || plan.device_index >= static_cast<uint32_t>(ncnn::get_gpu_count())
        || ncnn::get_gpu_device(static_cast<int>(plan.device_index)) != context->device())
        return Error{ErrorCode::InvalidArgument, "invalid selected full-resident Expert bank plan"};
    auto first = loader(0);
    if (!first) return first.error();
    auto description = std::move(first).value();
    std::vector<ExpertBankWeights_vulkan> metadata(plan.expert_count, description);
    auto checked = ExpertBank_vulkan::plan(metadata, context);
    if (!checked) return checked.error();
    const auto& expected = checked.value();
    if (!same_bank_shape(plan, expected) || plan.storage_size != expected.storage_size || plan.physical_size != expected.physical_size
        || plan.alignment != expected.alignment || plan.gate_stride != expected.gate_stride || plan.down_stride != expected.down_stride
        || plan.gate_scales != expected.gate_scales || plan.gate_bias != expected.gate_bias || plan.down_weights != expected.down_weights
        || plan.down_scales != expected.down_scales || plan.down_bias != expected.down_bias)
        return Error{ErrorCode::InvalidArgument, "selected full-resident Expert bank layout no longer matches its plan"};
    metadata.clear();
    auto result = std::shared_ptr<ExpertBank_vulkan>(new ExpertBank_vulkan);
    auto& d = *result->d;
    d.context = context;
    d.plan = plan;
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.use_subgroup_ops = context->device()->info.support_subgroup_ops();
    option.blob_vkallocator = context->blob_allocator();
    option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    d.allocator = std::make_unique<VulkanExpertWeightAllocator>(context->device(), static_cast<size_t>(plan.storage_size), plan.physical_size);
    const uint32_t weight_format = plan.dtype == DType::Float32 ? 0 : plan.dtype == DType::BFloat16 ? 1
                                                                                                    : 2;
    if (!create_bank_pipeline(context, option, expertbank_gate_up_shader, sizeof(expertbank_gate_up_shader) - 1, 128, weight_format, true, d.gate_up)
        || !create_bank_pipeline(context, option, expertbank_down_shader, sizeof(expertbank_down_shader) - 1, 128, weight_format, true, d.down)
        || !create_bank_pipeline(context, option, expertbank_combine_shader, sizeof(expertbank_combine_shader) - 1, 128,
                                 weight_format, false, d.combine))
        return Error{ErrorCode::InternalError, "selected full-resident Expert bank pipeline preparation failed"};
    VulkanWeightUploadBatch upload(context);
    const auto storage = upload.prepare_storage(static_cast<size_t>(plan.storage_size), d.allocator.get());
    if (storage.size() != plan.storage_size) return Error{ErrorCode::InternalError, "cannot allocate selected Expert bank upload storage"};
    for (uint32_t expert = 0; expert < plan.expert_count; ++expert)
    {
        if (expert != 0)
        {
            auto loaded = loader(expert);
            if (!loaded) return loaded.error();
            description = std::move(loaded).value();
        }
        if (!pack_bank_expert(description, plan, expert, storage))
            return Error{ErrorCode::InvalidModel, "selected Expert bank payload does not match metadata for Expert " + std::to_string(expert)};
        description = {};
    }
    const uint64_t end = plan.down_weights + plan.down_stride * plan.expert_count;
    std::memset(storage.data() + end, 0, static_cast<size_t>(plan.storage_size - end));
    if (!upload.record_prepared_storage(d.weights, d.allocator.get()) || !upload.submit()
        || d.weights.empty() || d.allocator->physical_bytes() != plan.physical_size)
        return Error{ErrorCode::InternalError, "selected full-resident Expert bank upload or physical allocation failed"};
    return result;
#else
    (void)plan;
    (void)context;
    (void)loader;
    return Error{ErrorCode::UnsupportedModel, "Vulkan Expert banks are unavailable in this build"};
#endif
}

#if NCNN_MOE_WITH_VULKAN
bool ExpertBank_vulkan::record(const ncnn::VkMat& normalized, const ncnn::VkMat& selected,
                               uint32_t top_k, ncnn::VkMat& combined, ncnn::VkMat& validation,
                               ncnn::VkCompute& command, std::vector<ncnn::VkMat>& workspace,
                               const ncnn::VkMat* shared, const ncnn::VkMat* residual) const
{
    const auto& plan = d->plan;
    const auto valid_tensor = [&](const ncnn::VkMat& tensor) {
        return !tensor.empty() && tensor.elempack == 1 && tensor.elemsize == sizeof(float) && tensor.allocator
               && tensor.allocator->vkdev == d->context->device();
    };
    if (!d->context || !d->gate_up || !d->down || !d->combine || top_k == 0 || top_k > 64 || top_k > plan.expert_count
        || &combined == &normalized || &combined == &selected || &combined == &validation
        || !valid_tensor(normalized) || normalized.dims != 2 || normalized.w != static_cast<int>(plan.input_columns) || normalized.h <= 0
        || !valid_tensor(selected) || selected.dims != 2 || selected.w != static_cast<int>(2 * top_k + 1) || selected.h != normalized.h
        || !valid_tensor(validation) || validation.w != 1
        || (validation.dims != 1 && !(validation.dims == 2 && validation.h == 1))
        || validation.data == normalized.data || validation.data == selected.data
        || (shared && (shared == &combined || shared == &validation || !valid_tensor(*shared) || shared->dims != 2 || shared->w != static_cast<int>(plan.output_columns) || shared->h != normalized.h || shared->data == validation.data))
        || (residual && (residual == &combined || residual == &validation || !valid_tensor(*residual) || residual->dims != 2 || residual->w != static_cast<int>(plan.output_columns) || residual->h != normalized.h || residual->data == validation.data))) return false;
    const uint32_t rows = static_cast<uint32_t>(normalized.h);
    const auto& limits = d->context->device()->info.physicalDeviceProperties().limits;
    uint32_t scratch_columns[3];
    uint64_t scratch_heights[3];
    const uint64_t route_rows = bank_scratch_shapes(plan, rows, top_k, scratch_columns, scratch_heights);
    if (route_rows == UINT64_MAX) return false;
    if (route_rows > static_cast<uint64_t>(std::numeric_limits<int>::max()) || rows > limits.maxComputeWorkGroupCount[2]
        || top_k > limits.maxComputeWorkGroupCount[1] || rows > limits.maxComputeWorkGroupCount[1]
        || route_rows > std::numeric_limits<uint32_t>::max() / std::max(plan.input_columns, std::max(plan.intermediate_columns, plan.output_columns))
        || static_cast<uint64_t>(rows) * (2 * top_k + 1) > std::numeric_limits<uint32_t>::max()) return false;
    if (route_rows * std::max(plan.intermediate_columns, plan.output_columns) * sizeof(float) > limits.maxStorageBufferRange
        || static_cast<uint64_t>(rows) * plan.input_columns * sizeof(float) > limits.maxStorageBufferRange
        || static_cast<uint64_t>(rows) * (2 * top_k + 1) * sizeof(float) > limits.maxStorageBufferRange) return false;
    try
    {
        ncnn::VkMat intermediate(static_cast<int>(scratch_columns[0]), static_cast<int>(scratch_heights[0]), sizeof(float), d->context->blob_allocator());
        ncnn::VkMat outputs(static_cast<int>(scratch_columns[1]), static_cast<int>(scratch_heights[1]), sizeof(float), d->context->blob_allocator());
        ncnn::VkMat destination(static_cast<int>(scratch_columns[2]), static_cast<int>(scratch_heights[2]), sizeof(float), d->context->blob_allocator());
        if (intermediate.empty() || outputs.empty() || destination.empty()) return false;
        workspace.push_back(intermediate);
        workspace.push_back(outputs);
        workspace.push_back(destination);
        std::vector<ncnn::vk_constant_type> constants(16);
        constants[0].u32 = plan.input_columns;
        constants[1].u32 = plan.intermediate_columns;
        constants[2].u32 = plan.output_columns;
        constants[3].u32 = plan.expert_count;
        constants[4].u32 = top_k;
        constants[5].u32 = rows;
        constants[6].u32 = plan.dtype == DType::Float32 ? 0 : plan.dtype == DType::BFloat16 ? 1
                                                                                            : 2;
        constants[7].u32 = static_cast<uint32_t>(plan.gate_stride / 4);
        constants[8].u32 = static_cast<uint32_t>(plan.gate_scales / 4);
        constants[9].u32 = static_cast<uint32_t>(plan.gate_bias / 4);
        constants[10].u32 = static_cast<uint32_t>(plan.down_weights / 4);
        constants[11].u32 = static_cast<uint32_t>(plan.down_stride / 4);
        constants[12].u32 = static_cast<uint32_t>(plan.down_scales / 4);
        constants[13].u32 = static_cast<uint32_t>(plan.down_bias / 4);
        constants[14].u32 = plan.activation == ExpertActivation::GptOssSwiGlu ? 0 : 1;
        constants[15].f = plan.activation_limit;
        ncnn::VkMat dispatch;
        dispatch.w = static_cast<int>(((static_cast<uint64_t>(plan.intermediate_columns) + 3) / 4) * 128);
        dispatch.h = static_cast<int>(top_k);
        dispatch.c = normalized.h;
        command.record_pipeline_readonly(d->gate_up.get(), {normalized, d->weights, selected, intermediate}, {1, 1, 1, 0}, constants, dispatch);
        dispatch.w = static_cast<int>(((static_cast<uint64_t>(plan.output_columns) + 3) / 4) * 128);
        command.record_pipeline_readonly(d->down.get(), {intermediate, d->weights, selected, outputs}, {1, 1, 1, 0}, constants, dispatch);
        constants.resize(6);
        constants[0].u32 = plan.output_columns;
        constants[1].u32 = rows;
        constants[2].u32 = top_k;
        constants[3].u32 = plan.expert_count;
        constants[4].u32 = shared != nullptr;
        constants[5].u32 = residual != nullptr;
        dispatch.w = static_cast<int>(plan.output_columns);
        dispatch.h = normalized.h;
        dispatch.c = 1;
        command.record_pipeline_readonly(d->combine.get(), {outputs, selected, destination, validation, shared ? *shared : outputs, residual ? *residual : outputs},
                                         {1, 1, 0, 0, 1, 1}, constants, dispatch);
        combined = destination;
        return true;
    }
    catch (...)
    {
        return false;
    }
}
#endif

} // namespace moe
} // namespace ncnn
