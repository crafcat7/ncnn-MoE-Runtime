#include "attentiondecode_vulkan.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/attention_gqa_matrix.comp.hex.h"
#include "kernels/vulkan/attention_gqa_reduce.comp.hex.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace ncnn {
namespace moe {

static bool has_extension(const ncnn::GpuInfo& info, const char* name)
{
    const auto& extensions = info.deviceExtensionProperties();
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

static uint32_t query_device_parallelism(const ncnn::GpuInfo& info)
{
    // ncnn's minimal Vulkan headers omit these property-only extensions.
    // Keep the registry layouts local; querying them requires no device feature.
    struct ShaderSmProperties
    {
        VkStructureType sType = static_cast<VkStructureType>(1000154001);
        void* pNext = nullptr;
        uint32_t shaderSMCount = 0;
        uint32_t shaderWarpsPerSM = 0;
    } sm;
    struct ShaderCoreProperties
    {
        VkStructureType sType = static_cast<VkStructureType>(1000227000);
        void* pNext = nullptr;
        VkFlags shaderCoreFeatures = 0;
        uint32_t activeComputeUnitCount = 0;
    } cores;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    if (has_extension(info, "VK_NV_shader_sm_builtins"))
    {
        sm.pNext = properties.pNext;
        properties.pNext = &sm;
    }
    if (has_extension(info, "VK_AMD_shader_core_properties2"))
    {
        cores.pNext = properties.pNext;
        properties.pNext = &cores;
    }
    if (properties.pNext && ncnn::vkGetPhysicalDeviceProperties2KHR)
        ncnn::vkGetPhysicalDeviceProperties2KHR(info.physicalDevice(), &properties);
    if (sm.shaderSMCount != 0)
        return sm.shaderSMCount;
    if (cores.activeComputeUnitCount != 0)
        return cores.activeComputeUnitCount;
    // A portable occupancy target when no core-count property is available.
    // This is a scheduling hint, never a capacity or allocation limit.
    return 32;
}

static bool supports_matrix_shape(const ncnn::GpuInfo& info, uint32_t m, uint32_t n, uint32_t k)
{
    for (const auto& property : info.queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV())
    {
        if (property.scope == VK_SCOPE_WORKGROUP_KHR && property.workgroupInvocations == 256
            && property.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR
            && property.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR
            && property.CType == VK_COMPONENT_TYPE_FLOAT32_KHR
            && property.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
            && property.MGranularity != 0 && property.NGranularity != 0 && property.KGranularity != 0
            && m % property.MGranularity == 0 && n % property.NGranularity == 0 && k % property.KGranularity == 0)
            return true;
    }
    return false;
}

static bool create_pipeline(const std::shared_ptr<VulkanContext>& context,
                            const ncnn::Option& option,
                            const char* shader,
                            int shader_size,
                            uint64_t variant,
                            uint32_t workgroup_size,
                            uint32_t subgroup_size,
                            const std::vector<ncnn::vk_specialization_type>& specializations,
                            std::shared_ptr<ncnn::Pipeline>& destination)
{
    destination = context->find_pipeline(shader, variant);
    if (destination)
        return true;
    const auto binary = context->shader_binary(shader, shader_size, option, 0);
    if (!binary || binary->empty())
        return false;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(context->device()));
    pipeline->set_local_size_xyz(workgroup_size, 1, 1);
    if (subgroup_size != 0)
        pipeline->set_subgroup_size(subgroup_size);
    if (pipeline->create(binary->data(), binary->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        context.reset();
    });
    context->cache_pipeline(shader, variant, destination);
    return true;
}

static bool same_pipeline_shape(const AttentionDecodePlan& left, const AttentionDecodePlan& right)
{
    return left.shape.head_dimension == right.shape.head_dimension
           && left.shape.kv_storage_bfloat16 == right.shape.kv_storage_bfloat16
           && left.query_tile == right.query_tile && left.kv_tile == right.kv_tile
           && left.workgroup_size == right.workgroup_size && left.subgroup_size == right.subgroup_size;
}

static bool valid_matrix_plan(const AttentionDecodePlan& plan)
{
    const auto& shape = plan.shape;
    if (plan.path != AttentionDecodePath::Matrix || shape.query_heads == 0 || shape.kv_heads == 0
        || shape.query_heads % shape.kv_heads != 0 || shape.query_heads / shape.kv_heads < 2
        || shape.head_dimension == 0 || shape.head_dimension > 128 || shape.head_dimension % 16 != 0
        || !shape.bfloat16_kv || shape.kv_tokens == 0 || plan.query_tile != 32 || plan.kv_tile != 32
        || plan.workgroup_size != 256 || plan.subgroup_size == 0 || 256 % plan.subgroup_size != 0
        || plan.group_size != shape.query_heads / shape.kv_heads
        || plan.query_groups != (static_cast<uint64_t>(plan.group_size) + 31) / 32
        || plan.split_count == 0 || plan.split_count > plan.scratch_split_capacity
        || plan.split_tokens == 0 || plan.split_tokens % 32 != 0)
        return false;
    const uint64_t row_elements = static_cast<uint64_t>(shape.query_heads) * (shape.head_dimension + 2);
    if (row_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || plan.scratch_split_capacity > static_cast<uint64_t>(std::numeric_limits<int>::max()) / row_elements)
        return false;
    const uint64_t coverage = static_cast<uint64_t>(plan.split_count) * plan.split_tokens;
    const uint64_t elements = static_cast<uint64_t>(plan.scratch_split_capacity)
                              * shape.query_heads * (shape.head_dimension + 2);
    return coverage >= shape.kv_tokens && coverage <= std::numeric_limits<uint32_t>::max()
           && elements <= static_cast<uint64_t>(std::numeric_limits<int>::max())
           && plan.scratch_elements == elements && plan.scratch_bytes == elements * sizeof(float)
           && static_cast<uint64_t>(shape.query_heads) * shape.head_dimension
                  <= static_cast<uint64_t>(std::numeric_limits<int>::max());
}

static bool valid_buffer(const ncnn::VkMat& tensor, uint64_t required_bytes, const VkPhysicalDeviceLimits& limits)
{
    if (!tensor.data || tensor.elemsize == 0 || tensor.offset > tensor.data->capacity
        || tensor.buffer_offset() < tensor.data->offset)
        return false;
#if NCNN_BATCH
    if (tensor.n != 1 || tensor.nstep > std::numeric_limits<size_t>::max() / tensor.elemsize)
        return false;
#endif
    const size_t descriptor_bytes = tensor.buffer_capacity();
    return descriptor_bytes >= required_bytes && descriptor_bytes <= limits.maxStorageBufferRange
           && descriptor_bytes <= tensor.data->capacity - tensor.offset
           && tensor.buffer_offset() % std::max<VkDeviceSize>(1, limits.minStorageBufferOffsetAlignment) == 0;
}

static bool valid_tensor(const ncnn::VkMat& tensor, uint32_t columns, uint32_t rows, uint32_t heads,
                         size_t element_size, const VkPhysicalDeviceLimits& limits, bool permit_sink_row = false)
{
    const bool valid_rows = tensor.h == rows
                            || (permit_sink_row && static_cast<uint64_t>(tensor.h) == static_cast<uint64_t>(rows) + 1);
    const uint64_t row_elements = static_cast<uint64_t>(columns) * static_cast<uint32_t>(tensor.h);
    return !tensor.empty() && tensor.dims == 3 && tensor.w == columns && valid_rows && tensor.c == heads
           && tensor.elempack == 1 && tensor.elemsize == element_size
           && tensor.cstep >= row_elements
           && tensor.cstep <= std::numeric_limits<uint32_t>::max() / heads
           && (tensor.cstep * element_size) % 16 == 0 && tensor.buffer_offset() % 16 == 0
           && valid_buffer(tensor, ((heads - 1) * tensor.cstep + row_elements) * element_size, limits);
}

AttentionDecodeCapabilities query_attention_decode_capabilities(const ncnn::GpuInfo& info)
{
    AttentionDecodeCapabilities capabilities;
    capabilities.device_parallelism = query_device_parallelism(info);
    capabilities.max_workgroups_x = info.max_workgroup_count_x();
    capabilities.max_workgroups_y = info.max_workgroup_count_y();
    capabilities.max_workgroups_z = info.max_workgroup_count_z();
    capabilities.subgroup_size = info.subgroup_size();
    capabilities.bfloat16_matrix = info.support_VK_KHR_cooperative_matrix()
                                   && info.support_VK_KHR_shader_bfloat16()
                                   && info.support_bf16_storage() && info.support_bf16_cooperative_matrix();
    if (!capabilities.bfloat16_matrix || !info.support_VK_NV_cooperative_matrix2())
        return capabilities;
    const auto& features = info.queryCooperativeMatrix2FeaturesNV();
    const auto& properties = info.queryCooperativeMatrix2PropertiesNV();
    capabilities.workgroup_matrix = features.cooperativeMatrixWorkgroupScope
                                    && features.cooperativeMatrixFlexibleDimensions
                                    && features.cooperativeMatrixReductions
                                    && features.cooperativeMatrixConversions
                                    && features.cooperativeMatrixPerElementOperations
                                    && features.cooperativeMatrixTensorAddressing
                                    && properties.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize >= 256
                                    && properties.cooperativeMatrixWorkgroupScopeReservedSharedMemory <= info.max_shared_memory_size()
                                    && info.max_workgroup_invocations() >= 256 && info.max_workgroup_size_x() >= 256
                                    && (info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
    if (!capabilities.workgroup_matrix || !supports_matrix_shape(info, 32, 32, 32))
        return capabilities;
    for (uint32_t dimension = 16; dimension <= 128; dimension += 16)
    {
        if (properties.cooperativeMatrixFlexibleDimensionsMaxDimension >= std::max(32u, dimension)
            && supports_matrix_shape(info, 32, 32, dimension) && supports_matrix_shape(info, 32, dimension, 32)
            && supports_matrix_shape(info, dimension, 32, 32))
            capabilities.supported_head_dimensions |= 1u << (dimension / 16 - 1);
    }
    return capabilities;
}

bool AttentionDecodeVulkan::create(const std::shared_ptr<VulkanContext>& requested_context,
                                   const ncnn::Option& option,
                                   const AttentionDecodePlan& plan)
{
    if (!requested_context || !requested_context->device() || !valid_matrix_plan(plan))
        return false;
    if (context && (context != requested_context || !same_pipeline_shape(plan, pipeline_plan)))
        return false;
    const auto capabilities = query_attention_decode_capabilities(requested_context->device()->info);
    AttentionDecodePlan supported_plan;
    if (!make_attention_decode_plan(capabilities, plan.shape, supported_plan)
        || supported_plan.path != AttentionDecodePath::Matrix || !same_pipeline_shape(plan, supported_plan))
        return false;
    context = requested_context;
    pipeline_plan = plan;
    ncnn::Option compile_option = option;
    compile_option.vulkan_device_index = context->device()->info.device_index();
    compile_option.use_fp16_packed = false;
    compile_option.use_fp16_storage = false;
    compile_option.use_fp16_arithmetic = false;
    compile_option.use_bf16_packed = false;
    compile_option.use_bf16_storage = true;
    const uint64_t variant = static_cast<uint64_t>(plan.shape.head_dimension)
                             | static_cast<uint64_t>(plan.query_tile) << 8
                             | static_cast<uint64_t>(plan.kv_tile) << 16
                             | static_cast<uint64_t>(plan.subgroup_size) << 24
                             | static_cast<uint64_t>(plan.workgroup_size) << 40
                             | static_cast<uint64_t>(plan.shape.kv_storage_bfloat16) << 56;
    std::vector<ncnn::vk_specialization_type> specializations(4);
    specializations[0].u32 = plan.query_tile;
    specializations[1].u32 = plan.kv_tile;
    specializations[2].u32 = plan.shape.head_dimension;
    specializations[3].u32 = plan.shape.kv_storage_bfloat16 ? 1 : 0;
    if (!matrix_pipeline
        && !create_pipeline(context, compile_option, attention_gqa_matrix_shader,
                            static_cast<int>(sizeof(attention_gqa_matrix_shader) - 1), variant,
                            plan.workgroup_size, plan.subgroup_size, specializations, matrix_pipeline))
        return false;
    compile_option.use_bf16_storage = false;
    return reduce_pipeline
           || create_pipeline(context, compile_option, attention_gqa_reduce_shader,
                              static_cast<int>(sizeof(attention_gqa_reduce_shader) - 1), 0, 128, 0, {}, reduce_pipeline);
}

bool AttentionDecodeVulkan::record(const AttentionDecodePlan& plan,
                                   const ncnn::VkMat& query,
                                   const ncnn::VkMat& key,
                                   const ncnn::VkMat& value,
                                   const ncnn::VkMat& sinks,
                                   bool attention_sink,
                                   ncnn::VkMat& output,
                                   ncnn::VkCompute& command,
                                   ncnn::VkAllocator* allocator,
                                   std::vector<ncnn::VkMat>& workspace) const
{
    const auto& shape = plan.shape;
    if (!context || !matrix_pipeline || !reduce_pipeline || !allocator)
        return false;
    const auto& limits = context->device()->info.physicalDeviceProperties().limits;
    const size_t kv_element_size = shape.kv_storage_bfloat16 ? sizeof(uint16_t) : sizeof(float);
    if (!valid_matrix_plan(plan) || !same_pipeline_shape(plan, pipeline_plan)
        || !valid_tensor(query, shape.head_dimension, 1, shape.query_heads, sizeof(float), limits)
        || !valid_tensor(key, shape.head_dimension, shape.kv_tokens, shape.kv_heads, kv_element_size, limits, attention_sink)
        || !valid_tensor(value, shape.head_dimension, shape.kv_tokens, shape.kv_heads, kv_element_size, limits, attention_sink)
        || sinks.empty() || sinks.dims != 1 || sinks.w != shape.query_heads || sinks.elempack != 1 || sinks.elemsize != sizeof(float)
        || !valid_buffer(sinks, static_cast<uint64_t>(shape.query_heads) * sizeof(float), limits)
        || plan.scratch_bytes > limits.maxStorageBufferRange
        || static_cast<uint64_t>(shape.query_heads) * shape.head_dimension * sizeof(float) > limits.maxStorageBufferRange
        || plan.split_count > context->device()->info.max_workgroup_count_x()
        || plan.query_groups > context->device()->info.max_workgroup_count_y()
        || shape.kv_heads > context->device()->info.max_workgroup_count_z()
        || shape.query_heads > context->device()->info.max_workgroup_count_x()
        || shape.query_heads > static_cast<uint32_t>(std::numeric_limits<int>::max()) / 128
        || plan.split_count > static_cast<uint32_t>(std::numeric_limits<int>::max()) / plan.workgroup_size)
        return false;

    ncnn::VkMat partial;
    partial.create(static_cast<int>(plan.scratch_elements), sizeof(float), 1, allocator);
    output.create(static_cast<int>(static_cast<uint64_t>(shape.query_heads) * shape.head_dimension), 1, sizeof(float), 1, allocator);
    if (partial.empty() || output.empty())
        return false;
    workspace.push_back(partial);
    std::vector<ncnn::vk_constant_type> constants(11);
    constants[0].u32 = shape.head_dimension;
    constants[1].u32 = shape.kv_tokens;
    constants[2].u32 = shape.query_heads;
    constants[3].u32 = shape.kv_heads;
    constants[4].u32 = static_cast<uint32_t>(query.cstep);
    constants[5].u32 = static_cast<uint32_t>(key.cstep);
    constants[6].u32 = static_cast<uint32_t>(value.cstep);
    constants[7].u32 = attention_sink ? 1 : 0;
    constants[8].f = 1.0f / std::sqrt(static_cast<float>(shape.head_dimension));
    constants[9].u32 = plan.split_count;
    constants[10].u32 = plan.split_tokens;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(plan.split_count * plan.workgroup_size);
    dispatcher.h = static_cast<int>(plan.query_groups);
    dispatcher.c = static_cast<int>(shape.kv_heads);
    command.record_pipeline_readonly(matrix_pipeline.get(), {query, key, value, sinks, partial}, {1, 1, 1, 1, 0}, constants, dispatcher);
    constants.resize(3);
    constants[0].u32 = shape.head_dimension;
    constants[1].u32 = shape.query_heads;
    constants[2].u32 = plan.split_count;
    dispatcher.w = static_cast<int>(shape.query_heads * 128);
    dispatcher.h = 1;
    dispatcher.c = 1;
    command.record_pipeline_readonly(reduce_pipeline.get(), {partial, output}, {1, 0}, constants, dispatcher);
    return true;
}

} // namespace moe
} // namespace ncnn
#endif
