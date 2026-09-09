#include "attention_vulkan.h"
#include "linear.h"
#include "vulkancontext.h"

#include "kernels/ops.h"
#include "kernels/statecache.h"

#if NCNN_MOE_USE_NCNN
#include <layer.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif
#endif

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/attention_decode_sdpa.comp.hex.h"
#include "kernels/vulkan/attention_output_gate.comp.hex.h"
#include "kernels/vulkan/attention_qkv_norm_rope.comp.hex.h"
#include "kernels/vulkan/attention_qkv_rope.comp.hex.h"
#include "kernels/vulkan/attention_ring_append.comp.hex.h"
#include "kernels/vulkan/attention_ring_zero.comp.hex.h"
#endif

namespace ncnn {
namespace moe {

class Attention_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    ~Implementation()
    {
        if (vulkan_context)
        {
            qkv_rope_pipeline.reset();
            qkv_norm_rope_pipeline.reset();
            output_gate_pipeline.reset();
            decode_sdpa_pipeline.reset();
            ring_append_pipeline.reset();
            ring_zero_pipeline.reset();
            const std::lock_guard<std::mutex> lock(vulkan_context->command_mutex());
            for (ncnn::Layer* layer : layers)
            {
                layer->destroy_pipeline(option);
                delete layer;
            }
        }
        layers.clear();
        attention_sinks = ncnn::VkMat();
        rope_inverse_frequencies_gpu = ncnn::VkMat();
        query_norm_weight = ncnn::VkMat();
        key_norm_weight = ncnn::VkMat();
        weight_staging_allocator.reset();
        weight_allocator.reset();
    }

    ncnn::Layer* norm = nullptr;
    ncnn::Layer* slice_qkv = nullptr;
    ncnn::Layer* reshape_query = nullptr;
    ncnn::Layer* reshape_key_value = nullptr;
    ncnn::Layer* permute_heads_tokens = nullptr;
    ncnn::Layer* rotary = nullptr;
    ncnn::Layer* sdpa = nullptr;
    ncnn::Layer* reshape_attention = nullptr;
    ncnn::Layer* add = nullptr;
    std::shared_ptr<ncnn::Pipeline> qkv_rope_pipeline;
    std::shared_ptr<ncnn::Pipeline> qkv_norm_rope_pipeline;
    std::shared_ptr<ncnn::Pipeline> output_gate_pipeline;
    std::shared_ptr<ncnn::Pipeline> decode_sdpa_pipeline;
    std::shared_ptr<ncnn::Pipeline> ring_append_pipeline;
    std::shared_ptr<ncnn::Pipeline> ring_zero_pipeline;
    std::vector<ncnn::Layer*> layers;
    ncnn::Option option;
    ncnn::Option kv_option;
    std::shared_ptr<VulkanContext> vulkan_context;
    std::unique_ptr<ncnn::VkWeightAllocator> weight_allocator;
    std::unique_ptr<ncnn::VkWeightStagingAllocator> weight_staging_allocator;
    ncnn::VkMat attention_sinks;
    ncnn::VkMat rope_inverse_frequencies_gpu;
#endif
    std::shared_ptr<Linear> fused_qkv;
    std::shared_ptr<Bfloat16Linear_vulkan> fused_qkv_gate;
    std::shared_ptr<Linear> output_projection;
    std::shared_ptr<Bfloat16Linear_vulkan> output_projection_bfloat16;
    AttentionConfig_vulkan config;
    std::vector<float> sinks;
    std::vector<float> rope_inverse_frequencies;
#if NCNN_MOE_WITH_VULKAN
    ncnn::VkMat query_norm_weight;
    ncnn::VkMat key_norm_weight;
#endif
    float rope_concentration = 1.0f;
};

Attention_vulkan::Attention_vulkan()
    : d(new Implementation)
{
}

Attention_vulkan::~Attention_vulkan() = default;

#if NCNN_MOE_WITH_VULKAN
static bool tensor_to_float_vector(const TensorData& tensor, std::vector<float>& values)
{
    if (tensor.dtype != DType::Float32 && tensor.dtype != DType::BFloat16)
        return false;
    values.resize(tensor.element_count());
    if (tensor.dtype == DType::Float32)
    {
        const std::span<const float> tensor_values = tensor.float32_values();
        if (tensor_values.size() != values.size())
            return false;
        std::copy(tensor_values.begin(), tensor_values.end(), values.begin());
    }
    else
    {
        const std::span<const uint16_t> tensor_values = tensor.bfloat16_values();
        if (tensor_values.size() != values.size())
            return false;
        for (size_t index = 0; index < values.size(); ++index)
            values[index] = bfloat16_to_float(tensor_values[index]);
    }
    return true;
}

static bool fill_rope_staging_pair(ncnn::VkMat& cosine_staging, ncnn::VkMat& sine_staging, size_t token_count, uint64_t position_offset,
                                   const std::vector<float>& inverse_frequencies, float concentration, bool bfloat16_storage, ncnn::VkAllocator* allocator,
                                   VulkanRuntimeState& runtime_state)
{
    if (token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || inverse_frequencies.size() > static_cast<size_t>(std::numeric_limits<int>::max())
        || !prepare_staging_batch(
            cosine_staging,
            token_count,
            static_cast<uint32_t>(inverse_frequencies.size()),
            allocator,
            runtime_state,
            bfloat16_storage ? sizeof(uint16_t) : sizeof(float))
        || !prepare_staging_batch(
            sine_staging,
            token_count,
            static_cast<uint32_t>(inverse_frequencies.size()),
            allocator,
            runtime_state,
            bfloat16_storage ? sizeof(uint16_t) : sizeof(float)))
        return false;

    ncnn::Mat cosine_mapped = cosine_staging.mapped();
    ncnn::Mat sine_mapped = sine_staging.mapped();
    if (cosine_mapped.empty() || sine_mapped.empty())
        return false;
    for (size_t token_index = 0; token_index < token_count; ++token_index)
    {
        float* cosine_float_row = bfloat16_storage ? nullptr : cosine_mapped.row<float>(static_cast<int>(token_index));
        float* sine_float_row = bfloat16_storage ? nullptr : sine_mapped.row<float>(static_cast<int>(token_index));
        uint16_t* cosine_bfloat16_row = bfloat16_storage ? cosine_mapped.row<uint16_t>(static_cast<int>(token_index)) : nullptr;
        uint16_t* sine_bfloat16_row = bfloat16_storage ? sine_mapped.row<uint16_t>(static_cast<int>(token_index)) : nullptr;
        for (size_t index = 0; index < inverse_frequencies.size(); ++index)
        {
            const float angle = static_cast<float>(position_offset + token_index) * inverse_frequencies[index];
            const float cosine = std::cos(angle) * concentration;
            const float sine = std::sin(angle) * concentration;
            if (bfloat16_storage)
            {
                cosine_bfloat16_row[index] = float_to_bfloat16(cosine);
                sine_bfloat16_row[index] = float_to_bfloat16(sine);
            }
            else
            {
                cosine_float_row[index] = cosine;
                sine_float_row[index] = sine;
            }
        }
    }
    return true;
}

static bool fill_attention_mask_staging(ncnn::VkMat& staging, size_t token_count, uint64_t destination_count, uint64_t position_offset,
                                        const LayerCache& cache, const AttentionConfig_vulkan& config, const std::vector<float>& sinks,
                                        bool bfloat16_storage, ncnn::VkAllocator* allocator,
                                        VulkanRuntimeState& runtime_state)
{
    if (destination_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) || token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || !prepare_staging_tensor(staging, static_cast<int>(destination_count), static_cast<int>(token_count), static_cast<int>(config.head_count),
                                   bfloat16_storage ? sizeof(uint16_t) : sizeof(float), allocator, runtime_state))
        return false;

    // The finite sentinel avoids BF16 NaNs and still underflows after softmax.
    constexpr float masked_logit = -10000.0f;
    ncnn::Mat mapped = staging.mapped();
    if (mapped.empty())
        return false;
    const uint64_t actual_end = cache.token_count + token_count;
    const bool use_attention_sink = has_flag(config.flags, AttentionSink);
    ncnn::Mat first_head = mapped.channel(0);
    for (size_t query_index = 0; query_index < token_count; ++query_index)
    {
        const uint64_t query_position = position_offset + query_index;
        if (bfloat16_storage)
        {
            uint16_t* row = first_head.row<uint16_t>(static_cast<int>(query_index));
            const uint16_t masked_value = float_to_bfloat16(masked_logit);
            for (uint64_t key_index = 0; key_index < actual_end; ++key_index)
            {
                const uint64_t key_position = key_index < cache.token_count ? cache.start_position + key_index : position_offset + key_index - cache.token_count;
                const bool future = key_position > query_position;
                const bool too_old = config.sliding_window > 0 && key_position + config.sliding_window <= query_position;
                row[key_index] = future || too_old ? masked_value : 0;
            }
            if (use_attention_sink)
                row[actual_end] = float_to_bfloat16(sinks[0]);
            std::fill(row + actual_end + (use_attention_sink ? 1 : 0), row + destination_count, masked_value);
        }
        else
        {
            float* row = first_head.row<float>(static_cast<int>(query_index));
            for (uint64_t key_index = 0; key_index < actual_end; ++key_index)
            {
                const uint64_t key_position = key_index < cache.token_count ? cache.start_position + key_index : position_offset + key_index - cache.token_count;
                const bool future = key_position > query_position;
                const bool too_old = config.sliding_window > 0 && key_position + config.sliding_window <= query_position;
                row[key_index] = future || too_old ? masked_logit : 0.0f;
            }
            if (use_attention_sink)
                row[actual_end] = sinks[0];
            std::fill(row + actual_end + (use_attention_sink ? 1 : 0), row + destination_count, masked_logit);
        }
    }
    for (uint32_t head = 1; head < config.head_count; ++head)
    {
        ncnn::Mat head_mask = mapped.channel(static_cast<int>(head));
        for (size_t query_index = 0; query_index < token_count; ++query_index)
        {
            if (bfloat16_storage)
            {
                const uint16_t* source = first_head.row<uint16_t>(static_cast<int>(query_index));
                uint16_t* row = head_mask.row<uint16_t>(static_cast<int>(query_index));
                std::copy_n(source, destination_count, row);
                if (use_attention_sink)
                    row[actual_end] = float_to_bfloat16(sinks[head]);
            }
            else
            {
                const float* source = first_head.row<float>(static_cast<int>(query_index));
                float* row = head_mask.row<float>(static_cast<int>(query_index));
                std::copy_n(source, destination_count, row);
                if (use_attention_sink)
                    row[actual_end] = sinks[head];
            }
        }
    }
    return true;
}

static bool fill_attention_cache_promotion_staging(
    ncnn::VkMat& key_staging,
    ncnn::VkMat& value_staging,
    const LayerCache& cache,
    const AttentionConfig_vulkan& config,
    ncnn::VkAllocator* allocator,
    VulkanRuntimeState& runtime_state)
{
    const uint32_t columns = config.kv_head_count * config.head_dimension;
    if (cache.token_count == 0
        || columns == 0
        || cache.token_count
               > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || config.kv_head_count
               > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || cache.capacity_tokens == 0
        || cache.first_slot >= cache.capacity_tokens
        || cache.token_count > cache.capacity_tokens
        || cache.columns != columns
        || (cache.dtype != DType::Float32
            && cache.dtype != DType::BFloat16)
        || cache.capacity_tokens
               > static_cast<uint64_t>(
                   std::numeric_limits<size_t>::max() / columns))
    {
        return false;
    }

    const size_t capacity_elements = static_cast<size_t>(cache.capacity_tokens) * columns;
    const bool bfloat16 = cache.dtype == DType::BFloat16;
    if ((bfloat16
         && (cache.bfloat16_keys.size() < capacity_elements
             || cache.bfloat16_values.size() < capacity_elements))
        || (!bfloat16
            && (cache.keys.size() < capacity_elements
                || cache.values.size() < capacity_elements))
        || !prepare_staging_tensor(
            key_staging,
            static_cast<int>(config.head_dimension),
            static_cast<int>(cache.token_count),
            static_cast<int>(config.kv_head_count),
            sizeof(float),
            allocator,
            runtime_state)
        || !prepare_staging_tensor(
            value_staging,
            static_cast<int>(config.head_dimension),
            static_cast<int>(cache.token_count),
            static_cast<int>(config.kv_head_count),
            sizeof(float),
            allocator,
            runtime_state))
    {
        return false;
    }

    ncnn::Mat key_mapped = key_staging.mapped();
    ncnn::Mat value_mapped = value_staging.mapped();
    if (key_mapped.empty() || value_mapped.empty())
        return false;

    for (uint32_t head = 0; head < config.kv_head_count; ++head)
    {
        ncnn::Mat key_channel = key_mapped.channel(static_cast<int>(head));
        ncnn::Mat value_channel = value_mapped.channel(static_cast<int>(head));
        const size_t head_offset = static_cast<size_t>(head) * config.head_dimension;
        for (uint64_t token = 0; token < cache.token_count; ++token)
        {
            const uint64_t slot = (cache.first_slot + token) % cache.capacity_tokens;
            const size_t source_offset = static_cast<size_t>(slot) * columns + head_offset;
            float* key_row = key_channel.row<float>(static_cast<int>(token));
            float* value_row = value_channel.row<float>(static_cast<int>(token));
            if (bfloat16)
            {
                for (uint32_t column = 0;
                     column < config.head_dimension;
                     ++column)
                {
                    key_row[column] = bfloat16_to_float(
                        cache.bfloat16_keys[source_offset + column]);
                    value_row[column] = bfloat16_to_float(
                        cache.bfloat16_values[source_offset + column]);
                }
            }
            else
            {
                std::copy_n(
                    cache.keys.data() + source_offset,
                    config.head_dimension,
                    key_row);
                std::copy_n(
                    cache.values.data() + source_offset,
                    config.head_dimension,
                    value_row);
            }
        }
    }
    return true;
}

static bool create_attention_pipeline(
    const std::shared_ptr<VulkanContext>& context,
    const ncnn::Option& option,
    const char* shader,
    int shader_size,
    std::shared_ptr<ncnn::Pipeline>& destination)
{
    const size_t storage_variant = vulkan_activation_storage_variant(option);
    const std::shared_ptr<const std::vector<uint32_t>> spirv = context->shader_binary(shader, shader_size, option, storage_variant);
    if (!spirv || spirv->empty())
        return false;

    destination = context->find_pipeline(shader, storage_variant);
    if (destination)
        return true;

    ncnn::VulkanDevice* vkdev = context->device();
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(vkdev));
    if (shader == attention_qkv_norm_rope_shader)
        pipeline->set_local_size_xyz(32, 1, 1);
    else if (shader == attention_output_gate_shader)
        pipeline->set_optimal_local_size_xyz(128, 1, 1);
    else if (shader == attention_qkv_rope_shader)
        pipeline->set_optimal_local_size_xyz(64, 1, 1);
    else if (shader == attention_decode_sdpa_shader)
        pipeline->set_local_size_xyz(128, 1, 1);
    else
        pipeline->set_optimal_local_size_xyz(8, 8, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(
            spirv->data(),
            spirv->size() * sizeof(uint32_t),
            specializations)
        != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(
        pipeline.release(),
        [context](ncnn::Pipeline* value) {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        });
    context->cache_pipeline(shader, storage_variant, destination);
    return true;
}

bool Attention_vulkan::support_qkv_rope(size_t token_count) const noexcept
{
    const Implementation& implementation = *d;
    const AttentionConfig_vulkan& config = implementation.config;
    const uint64_t head_count = config.head_count;
    const uint64_t kv_head_count = config.kv_head_count;
    const uint64_t head_dimension = config.head_dimension;
    const uint64_t int_max = static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (token_count == 0 || head_count == 0 || kv_head_count == 0 || head_dimension == 0)
        return false;

    const uint64_t query_columns = head_count * head_dimension;
    const uint64_t key_value_columns = kv_head_count * head_dimension;
    if (query_columns > int_max || key_value_columns > int_max)
        return false;
    const bool gated = implementation.fused_qkv_gate != nullptr;
    const uint64_t total_columns = query_columns + key_value_columns * 2 + (gated ? query_columns : 0);
    if (token_count > static_cast<size_t>(int_max) || total_columns > int_max)
        return false;

    const uint64_t heads = head_count + kv_head_count;
    const uint64_t half_dimension = head_dimension / 2;
    const uint64_t work_items_per_token = gated
                                              ? heads
                                              : heads * half_dimension + key_value_columns;
    if (work_items_per_token == 0)
        return false;

    const uint64_t work_limit = gated ? int_max / 32 : int_max;
    return static_cast<uint64_t>(token_count) <= work_limit / work_items_per_token;
}

bool Attention_vulkan::record_qkv_rope(
    const ncnn::VkMat& fused_qkv,
    const ncnn::VkMat& cosine,
    const ncnn::VkMat& sine,
    size_t token_count,
    uint64_t position_offset,
    bool device_rope,
    const AttentionCache_vulkan* ring,
    uint64_t ring_capacity,
    uint64_t destination_start,
    ncnn::VkMat& query,
    ncnn::VkMat& key,
    ncnn::VkMat& value,
    ncnn::VkCompute& cmd) const
{
    const Implementation& implementation = *d;
    const ncnn::Pipeline* pipeline = implementation.qkv_rope_pipeline.get();
    const AttentionConfig_vulkan& config = implementation.config;
    const float rope_concentration = implementation.rope_concentration;
    const size_t key_value_element_size = vulkan_activation_element_size(implementation.kv_option);
    ncnn::VkAllocator* allocator = implementation.option.blob_vkallocator;
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    const uint64_t total_columns = static_cast<uint64_t>(query_columns) + static_cast<uint64_t>(key_value_columns) * 2;
    const uint64_t half_dimension = config.head_dimension / 2;
    const uint64_t work_items = static_cast<uint64_t>(token_count)
                                * (static_cast<uint64_t>(config.head_count) * half_dimension
                                   + static_cast<uint64_t>(config.kv_head_count) * half_dimension + key_value_columns);
    const bool direct_ring = ring != nullptr;
    const bool valid_rope_source = device_rope
                                       ? token_count != 0 && cosine.dims == 1 && cosine.w >= static_cast<int>(half_dimension)
                                             && position_offset <= std::numeric_limits<uint32_t>::max()
                                             && token_count - 1 <= std::numeric_limits<uint32_t>::max() - position_offset
                                       : cosine.dims == 2 && sine.dims == 2
                                             && cosine.w >= static_cast<int>(half_dimension)
                                             && sine.w >= static_cast<int>(half_dimension)
                                             && cosine.h >= static_cast<int>(token_count)
                                             && sine.h >= static_cast<int>(token_count);
    if (!pipeline || fused_qkv.empty() || cosine.empty() || sine.empty())
        return false;
    if (fused_qkv.elempack != 1
        || fused_qkv.elemsize != sizeof(float)
        || (key_value_element_size != sizeof(float)
            && key_value_element_size != sizeof(uint16_t))
        || fused_qkv.dims != 2 || fused_qkv.w != static_cast<int>(total_columns)
        || fused_qkv.h != static_cast<int>(token_count))
        return false;
    if (cosine.elempack != 1 || cosine.elemsize != sizeof(float)
        || sine.elempack != 1 || sine.elemsize != sizeof(float)
        || token_count == 0 || !valid_rope_source)
        return false;
    if (token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || total_columns > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || work_items > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;
    if (direct_ring
        && (ring_capacity == 0
            || ring_capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            || destination_start >= ring_capacity
            || token_count > ring_capacity))
        return false;

    query.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count), static_cast<int>(config.head_count), sizeof(float), 1, allocator);
    if (direct_ring)
    {
        key = ring->key;
        value = ring->value;
    }
    else
    {
        key.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count), static_cast<int>(config.kv_head_count), key_value_element_size, 1, allocator);
        value.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count), static_cast<int>(config.kv_head_count), key_value_element_size, 1,
                     allocator);
    }
    if (query.empty() || key.empty() || value.empty()
        || (direct_ring
            && (key.dims != 3 || value.dims != 3 || key.w != static_cast<int>(config.head_dimension) || value.w != key.w
                || key.h != static_cast<int>(ring_capacity * 2) || value.h != key.h || key.c != static_cast<int>(config.kv_head_count) || value.c != key.c
                || key.elemsize != key_value_element_size || value.elemsize != key_value_element_size || key.elempack != 1 || value.elempack != 1))
        || query.cstep > std::numeric_limits<uint32_t>::max() || key.cstep > std::numeric_limits<uint32_t>::max()
        || value.cstep > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }

    const std::vector<ncnn::VkMat> bindings = {
        fused_qkv,
        cosine,
        sine,
        query,
        key,
        value,
    };
    std::vector<ncnn::vk_constant_type> constants(15);
    constants[0].u32 = static_cast<uint32_t>(total_columns);
    constants[1].u32 = query_columns;
    constants[2].u32 = key_value_columns;
    constants[3].u32 = config.head_dimension;
    constants[4].u32 = static_cast<uint32_t>(token_count);
    constants[5].u32 = static_cast<uint32_t>(query.cstep);
    constants[6].u32 = static_cast<uint32_t>(key.cstep);
    constants[7].u32 = static_cast<uint32_t>(value.cstep);
    constants[8].u32 = static_cast<uint32_t>(work_items);
    constants[9].u32 = direct_ring ? 1 : 0;
    constants[10].u32 = direct_ring ? static_cast<uint32_t>(ring_capacity) : 0;
    constants[11].u32 = direct_ring ? static_cast<uint32_t>(destination_start) : 0;
    constants[12].u32 = device_rope ? 1 : 0;
    constants[13].u32 = device_rope ? static_cast<uint32_t>(position_offset) : 0;
    constants[14].f = rope_concentration;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(work_items);
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

bool Attention_vulkan::record_qkv_norm_rope(
    const ncnn::VkMat& fused_qkv,
    const ncnn::VkMat& cosine,
    const ncnn::VkMat& sine,
    size_t token_count,
    uint64_t position_offset,
    bool device_rope,
    const AttentionCache_vulkan* ring,
    uint64_t ring_capacity,
    uint64_t destination_start,
    ncnn::VkMat& query,
    ncnn::VkMat& key,
    ncnn::VkMat& value,
    ncnn::VkMat& gate,
    ncnn::VkCompute& cmd) const
{
    const Implementation& implementation = *d;
    const ncnn::Pipeline* pipeline = implementation.qkv_norm_rope_pipeline.get();
    const ncnn::VkMat& query_norm = implementation.query_norm_weight;
    const ncnn::VkMat& key_norm = implementation.key_norm_weight;
    const AttentionConfig_vulkan& config = implementation.config;
    const float rope_concentration = implementation.rope_concentration;
    const size_t key_value_element_size = vulkan_activation_element_size(implementation.kv_option);
    ncnn::VkAllocator* allocator = implementation.option.blob_vkallocator;
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    const uint64_t total_columns = static_cast<uint64_t>(query_columns)
                                   + static_cast<uint64_t>(key_value_columns) * 2
                                   + query_columns;
    const uint32_t rope_dimension = config.rope_head_dimension == 0
                                        ? config.head_dimension
                                        : config.rope_head_dimension;
    const uint64_t work_items = static_cast<uint64_t>(token_count)
                                * (static_cast<uint64_t>(config.head_count)
                                   + config.kv_head_count);
    const bool direct_ring = ring != nullptr;
    const bool valid_rope_source = device_rope
                                       ? token_count != 0 && cosine.dims == 1 && cosine.w >= static_cast<int>(rope_dimension / 2)
                                             && position_offset <= std::numeric_limits<uint32_t>::max()
                                             && token_count - 1 <= std::numeric_limits<uint32_t>::max() - position_offset
                                       : cosine.dims == 2 && sine.dims == 2
                                             && cosine.w >= static_cast<int>(rope_dimension / 2)
                                             && sine.w >= static_cast<int>(rope_dimension / 2)
                                             && cosine.h >= static_cast<int>(token_count)
                                             && sine.h >= static_cast<int>(token_count);
    if (!pipeline || fused_qkv.empty() || cosine.empty() || sine.empty()
        || query_norm.empty() || key_norm.empty())
        return false;
    if (token_count == 0
        || token_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || total_columns > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || work_items > static_cast<uint64_t>(std::numeric_limits<int>::max() / 32)
        || config.head_count == 0 || config.kv_head_count == 0
        || config.head_count % config.kv_head_count != 0)
        return false;
    if (fused_qkv.dims != 2 || fused_qkv.elempack != 1
        || fused_qkv.elemsize != sizeof(float)
        || (key_value_element_size != sizeof(float)
            && key_value_element_size != sizeof(uint16_t))
        || fused_qkv.w != static_cast<int>(total_columns)
        || fused_qkv.h != static_cast<int>(token_count))
        return false;
    if (cosine.elempack != 1 || sine.elempack != 1
        || cosine.elemsize != sizeof(float)
        || sine.elemsize != sizeof(float)
        || (rope_dimension != 0
            && (rope_dimension > config.head_dimension
                || (rope_dimension & 1) != 0))
        || !valid_rope_source)
        return false;
    if (query_norm.dims != 1 || key_norm.dims != 1
        || query_norm.w != static_cast<int>(config.head_dimension)
        || key_norm.w != static_cast<int>(config.head_dimension)
        || query_norm.elempack != 1 || key_norm.elempack != 1
        || query_norm.elemsize != sizeof(float)
        || key_norm.elemsize != sizeof(float))
        return false;
    if (direct_ring
        && (ring_capacity == 0
            || ring_capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
            || destination_start >= ring_capacity
            || token_count > ring_capacity))
        return false;

    query.create(
        static_cast<int>(config.head_dimension),
        static_cast<int>(token_count),
        static_cast<int>(config.head_count),
        sizeof(float),
        1,
        allocator);
    gate.create(
        static_cast<int>(query_columns),
        static_cast<int>(token_count),
        sizeof(float),
        1,
        allocator);
    if (direct_ring)
    {
        key = ring->key;
        value = ring->value;
    }
    else
    {
        key.create(
            static_cast<int>(config.head_dimension),
            static_cast<int>(token_count),
            static_cast<int>(config.kv_head_count),
            key_value_element_size,
            1,
            allocator);
        value.create(
            static_cast<int>(config.head_dimension),
            static_cast<int>(token_count),
            static_cast<int>(config.kv_head_count),
            key_value_element_size,
            1,
            allocator);
    }
    if (query.empty() || key.empty() || value.empty() || gate.empty()
        || query.cstep > std::numeric_limits<uint32_t>::max()
        || key.cstep > std::numeric_limits<uint32_t>::max()
        || value.cstep > std::numeric_limits<uint32_t>::max()
        || (direct_ring
            && (key.dims != 3 || value.dims != 3
                || key.w != static_cast<int>(config.head_dimension)
                || value.w != key.w
                || key.h != static_cast<int>(ring_capacity * 2)
                || value.h != key.h
                || key.c != static_cast<int>(config.kv_head_count)
                || value.c != key.c
                || key.elemsize != key_value_element_size
                || value.elemsize != key_value_element_size
                || key.elempack != 1 || value.elempack != 1)))
    {
        return false;
    }

    const std::vector<ncnn::VkMat> bindings = {
        fused_qkv,
        cosine,
        sine,
        query_norm,
        key_norm,
        query,
        key,
        value,
        gate,
    };
    std::vector<ncnn::vk_constant_type> constants(19);
    constants[0].u32 = static_cast<uint32_t>(total_columns);
    constants[1].u32 = query_columns;
    constants[2].u32 = key_value_columns;
    constants[3].u32 = config.head_dimension;
    constants[4].u32 = config.head_count;
    constants[5].u32 = config.kv_head_count;
    constants[6].u32 = static_cast<uint32_t>(token_count);
    constants[7].u32 = static_cast<uint32_t>(query.cstep);
    constants[8].u32 = static_cast<uint32_t>(key.cstep);
    constants[9].u32 = static_cast<uint32_t>(value.cstep);
    constants[10].u32 = static_cast<uint32_t>(work_items);
    constants[11].u32 = direct_ring ? 1 : 0;
    constants[12].u32 = direct_ring ? static_cast<uint32_t>(ring_capacity) : 0;
    constants[13].u32 = direct_ring ? static_cast<uint32_t>(destination_start) : 0;
    constants[14].u32 = rope_dimension;
    constants[15].f = config.norm_epsilon;
    constants[16].u32 = device_rope ? 1 : 0;
    constants[17].u32 = device_rope ? static_cast<uint32_t>(position_offset) : 0;
    constants[18].f = rope_concentration;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(work_items * 32);
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static bool record_attention_output_gate(
    const ncnn::Pipeline* pipeline,
    ncnn::VkMat& attention,
    const ncnn::VkMat& gate,
    size_t token_count,
    uint32_t columns,
    ncnn::VkCompute& command)
{
    const uint64_t elements = static_cast<uint64_t>(token_count) * columns;
    if (!pipeline || attention.empty() || gate.empty()
        || elements == 0
        || elements > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || gate.dims != 2 || gate.w != static_cast<int>(columns)
        || gate.h != static_cast<int>(token_count)
        || gate.elempack != 1 || gate.elemsize != sizeof(float)
        || attention.elempack != 1
        || attention.elemsize != sizeof(float)
        || attention.total() * attention.elempack < elements)
    {
        return false;
    }
    const std::vector<ncnn::VkMat> bindings = {attention, gate};
    std::vector<ncnn::vk_constant_type> constants(1);
    constants[0].u32 = static_cast<uint32_t>(elements);
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(elements);
    dispatcher.h = 1;
    dispatcher.c = 1;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static bool record_attention_decode_sdpa(const ncnn::Pipeline* pipeline, const ncnn::VkMat& query, const ncnn::VkMat& key, const ncnn::VkMat& value,
                                         const ncnn::VkMat& sinks, const AttentionConfig_vulkan& config, uint64_t destination_count, ncnn::VkMat& output,
                                         ncnn::VkCompute& command, ncnn::VkAllocator* allocator)
{
    const uint64_t query_columns = static_cast<uint64_t>(config.head_count) * config.head_dimension;
    if (!pipeline || query.empty() || key.empty() || value.empty() || sinks.empty() || query.dims != 3 || query.w != static_cast<int>(config.head_dimension)
        || query.h != 1 || query.c != static_cast<int>(config.head_count) || key.dims != 3 || value.dims != 3
        || key.w != static_cast<int>(config.head_dimension) || value.w != key.w || key.h != static_cast<int>(destination_count) || value.h != key.h
        || key.c != static_cast<int>(config.kv_head_count) || value.c != key.c || sinks.dims != 1 || sinks.w != static_cast<int>(config.head_count)
        || query.elempack != 1 || key.elempack != 1 || value.elempack != 1 || sinks.elempack != 1
        || query.elemsize != sizeof(float)
        || (key.elemsize != sizeof(float)
            && key.elemsize != sizeof(uint16_t))
        || value.elemsize != key.elemsize
        || sinks.elemsize != sizeof(float) || config.head_dimension == 0
        || config.head_dimension > 128 || config.head_count == 0 || config.kv_head_count == 0 || config.head_count % config.kv_head_count != 0
        || destination_count == 0 || destination_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
        || query_columns > static_cast<uint64_t>(std::numeric_limits<int>::max()) || query.cstep > std::numeric_limits<uint32_t>::max()
        || key.cstep > std::numeric_limits<uint32_t>::max() || value.cstep > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }

    output.create(static_cast<int>(query_columns), 1, sizeof(float), 1, allocator);
    if (output.empty())
        return false;

    const std::vector<ncnn::VkMat> bindings = {
        query,
        key,
        value,
        sinks,
        output,
    };
    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].u32 = config.head_dimension;
    constants[1].u32 = static_cast<uint32_t>(destination_count);
    constants[2].u32 = config.head_count;
    constants[3].u32 = config.kv_head_count;
    constants[4].u32 = static_cast<uint32_t>(query.cstep);
    constants[5].u32 = static_cast<uint32_t>(key.cstep);
    constants[6].u32 = static_cast<uint32_t>(value.cstep);
    constants[7].u32 = has_flag(config.flags, AttentionSink) ? 1 : 0;
    constants[8].f = 1.0f / std::sqrt(static_cast<float>(config.head_dimension));
    ncnn::VkMat dispatcher;
    dispatcher.w = 128;
    dispatcher.h = 1;
    dispatcher.c = static_cast<int>(config.head_count);
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static uint64_t next_attention_ring_capacity(uint64_t current, uint64_t required)
{
    uint64_t capacity = current == 0 ? 16 : current;
    while (capacity < required)
    {
        if (capacity > static_cast<uint64_t>(std::numeric_limits<int>::max()) / 4)
            return required;
        capacity *= 2;
    }
    return capacity;
}

static bool checked_multiply_u64(
    uint64_t left,
    uint64_t right,
    uint64_t& product)
{
    if (left != 0
        && right > std::numeric_limits<uint64_t>::max() / left)
    {
        return false;
    }
    product = left * right;
    return true;
}

static bool attention_promotion_within_budget(
    const VulkanContext& context,
    uint64_t ring_capacity,
    uint64_t kv_columns,
    size_t element_size,
    uint64_t transfer_bytes)
{
    uint64_t ring_elements = 0;
    uint64_t ring_bytes = 0;
    if (!checked_multiply_u64(
            ring_capacity,
            kv_columns,
            ring_elements)
        || !checked_multiply_u64(
            ring_elements,
            element_size * 4,
            ring_bytes)
        || transfer_bytes
               > std::numeric_limits<uint64_t>::max() - ring_bytes)
    {
        return false;
    }

    const ncnn::VulkanDevice* device = context.device();
    if (!device)
        return false;
    const uint64_t heap_budget = static_cast<uint64_t>(device->get_heap_budget()) * 1024 * 1024;
    // Promotion temporarily owns the FP32 staging payload and a double-written
    // key/value ring. Keep this opportunistic, model-neutral path to a small
    // per-layer share of the device budget; larger caches remain on the CPU.
    constexpr uint64_t maximum_per_layer_working_set = 32ull * 1024 * 1024;
    const uint64_t admission_bytes = std::min(
        maximum_per_layer_working_set,
        heap_budget / 256);
    return admission_bytes != 0
           && ring_bytes + transfer_bytes <= admission_bytes;
}

static bool create_attention_ring_storage(
    AttentionCache_vulkan& cache,
    uint32_t width,
    uint32_t channels,
    uint64_t capacity,
    size_t element_size,
    ncnn::VkAllocator* allocator)
{
    if (capacity == 0
        || capacity > static_cast<uint64_t>(std::numeric_limits<int>::max()) / 2
        || (element_size != sizeof(float)
            && element_size != sizeof(uint16_t)))
        return false;
    cache.key.create(static_cast<int>(width), static_cast<int>(capacity * 2), static_cast<int>(channels), element_size, 1, allocator);
    cache.value.create(static_cast<int>(width), static_cast<int>(capacity * 2), static_cast<int>(channels), element_size, 1, allocator);
    return !cache.key.empty() && !cache.value.empty();
}

static ncnn::VkMat attention_ring_view(const ncnn::VkMat& storage, uint64_t first_slot, uint64_t rows)
{
#if NCNN_BATCH
    if (storage.empty() || rows == 0 || first_slot + rows > static_cast<uint64_t>(storage.h))
        return {};
    ncnn::VkMat view = storage;
    view.h = static_cast<int>(rows);
    view.offset += static_cast<size_t>(first_slot) * storage.w * storage.elemsize;
    view.n = 1;
    view.nstep = static_cast<size_t>(view.c - 1) * view.cstep + static_cast<size_t>(view.h) * view.w;
    return view;
#else
    (void)storage;
    (void)first_slot;
    (void)rows;
    return {};
#endif
}

static bool record_attention_ring_append(const ncnn::Pipeline* pipeline, const ncnn::VkMat& source_key, const ncnn::VkMat& source_value,
                                         ncnn::VkMat& destination_key, ncnn::VkMat& destination_value, uint64_t capacity, uint64_t destination_start,
                                         ncnn::VkCompute& command)
{
    if (!pipeline || source_key.empty() || source_value.empty() || destination_key.empty() || destination_value.empty() || source_key.dims != 3
        || source_key.elempack != 1
        || (source_key.elemsize != sizeof(float)
            && source_key.elemsize != sizeof(uint16_t))
        || source_key.elemsize != source_value.elemsize
        || source_key.w != source_value.w || source_key.h != source_value.h
        || source_key.c != source_value.c || destination_key.w != source_key.w || destination_key.c != source_key.c || destination_value.w != source_key.w
        || destination_value.c != source_key.c
        || destination_key.elemsize != source_key.elemsize
        || destination_value.elemsize != source_key.elemsize
        || capacity == 0 || destination_start >= capacity || source_key.cstep > std::numeric_limits<uint32_t>::max()
        || destination_key.cstep > std::numeric_limits<uint32_t>::max())
        return false;

    const std::vector<ncnn::VkMat> bindings = {
        source_key,
        source_value,
        destination_key,
        destination_value,
    };
    std::vector<ncnn::vk_constant_type> constants(7);
    constants[0].u32 = static_cast<uint32_t>(source_key.w);
    constants[1].u32 = static_cast<uint32_t>(source_key.h);
    constants[2].u32 = static_cast<uint32_t>(source_key.c);
    constants[3].u32 = static_cast<uint32_t>(source_key.cstep);
    constants[4].u32 = static_cast<uint32_t>(destination_key.cstep);
    constants[5].u32 = static_cast<uint32_t>(capacity);
    constants[6].u32 = static_cast<uint32_t>(destination_start);
    ncnn::VkMat dispatcher;
    dispatcher.w = source_key.w;
    dispatcher.h = source_key.h;
    dispatcher.c = source_key.c;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static bool record_attention_ring_zero(const ncnn::Pipeline* pipeline, ncnn::VkMat& destination_key, ncnn::VkMat& destination_value, uint64_t destination_row,
                                       ncnn::VkCompute& command)
{
    if (!pipeline || destination_key.empty() || destination_value.empty() || destination_key.w != destination_value.w
        || destination_key.c != destination_value.c || destination_key.cstep != destination_value.cstep
        || (destination_key.elemsize != sizeof(float)
            && destination_key.elemsize != sizeof(uint16_t))
        || destination_value.elemsize != destination_key.elemsize
        || destination_row >= static_cast<uint64_t>(destination_key.h) || destination_key.cstep > std::numeric_limits<uint32_t>::max())
        return false;

    const std::vector<ncnn::VkMat> bindings = {
        destination_key,
        destination_value,
    };
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = static_cast<uint32_t>(destination_key.w);
    constants[1].u32 = static_cast<uint32_t>(destination_key.c);
    constants[2].u32 = static_cast<uint32_t>(destination_key.cstep);
    constants[3].u32 = static_cast<uint32_t>(destination_row);
    ncnn::VkMat dispatcher;
    dispatcher.w = destination_key.w;
    dispatcher.h = 1;
    dispatcher.c = destination_key.c;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}
#endif

#if NCNN_MOE_WITH_VULKAN
static bool create_vulkan_layer(int type, const ncnn::ParamDict& parameters, ncnn::VulkanDevice* device, const ncnn::Option& option,
                                std::vector<ncnn::Layer*>& layers, ncnn::Layer*& destination)
{
    ncnn::Layer* layer = ncnn::create_layer_vulkan(type);
    if (!layer)
        return false;
    layer->vkdev = device;
    if (layer->load_param(parameters) != 0 || layer->create_pipeline(option) != 0)
    {
        delete layer;
        return false;
    }
    layers.push_back(layer);
    destination = layer;
    return true;
}
#endif

std::shared_ptr<Attention_vulkan> Attention_vulkan::create(const TensorData& norm_weight, const TensorData* sinks,
                                                           std::shared_ptr<Linear> fused_qkv,
                                                           std::shared_ptr<Linear> output_projection,
                                                           const AttentionConfig_vulkan& config)
{
#if NCNN_MOE_WITH_VULKAN
#if !NCNN_BATCH
    // Wrapped KV rings require VkMat offset views.
    return {};
#endif
    if (!fused_qkv || !output_projection || !fused_qkv->uses_vulkan() || !output_projection->uses_vulkan() || config.hidden_size == 0 || config.head_count == 0
        || config.kv_head_count == 0 || config.head_dimension == 0 || config.head_dimension % 2 != 0 || config.head_count % config.kv_head_count != 0
        || config.activation_dtype != config.kv_cache_dtype || (!has_flag(config.flags, AttentionSink) && config.sliding_window > 0)
        || norm_weight.shape != std::vector<uint32_t>{config.hidden_size}
        || (has_flag(config.flags, AttentionSink) && (!sinks || sinks->shape != std::vector<uint32_t>{config.head_count})))
        return {};

    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    if (fused_qkv->input_columns() != config.hidden_size
        || fused_qkv->output_columns() != query_columns + 2 * key_value_columns
        || output_projection->input_columns() != query_columns
        || output_projection->output_columns() != config.hidden_size
        || fused_qkv->vulkan_context() != output_projection->vulkan_context())
        return {};

    std::shared_ptr<Attention_vulkan> attention(new Attention_vulkan);
    Implementation& implementation = *attention->d;
    implementation.fused_qkv = std::move(fused_qkv);
    implementation.output_projection = std::move(output_projection);
    implementation.config = config;
    if (has_flag(config.flags, AttentionSink) && !tensor_to_float_vector(*sinks, implementation.sinks))
        return {};
    implementation.vulkan_context = implementation.fused_qkv->vulkan_context();
    implementation.option = implementation.fused_qkv->option();
    // SDPA uses unpacked [head, token, dimension] tensors.
    implementation.option.use_packing_layout = false;
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    implementation.kv_option = implementation.option;

    const uint32_t rotary_dimension = config.rope_head_dimension == 0
                                          ? config.head_dimension
                                          : config.rope_head_dimension;
    const uint32_t half_dimension = config.head_dimension / 2;
    const uint32_t rotary_half_dimension = rotary_dimension / 2;
    implementation.rope_inverse_frequencies.resize(half_dimension);
    float rope_low = 0.0f;
    float rope_high = 0.0f;
    if (config.rope_scaling_factor > 1.0f)
    {
        implementation.rope_concentration = 0.1f * std::log(config.rope_scaling_factor) + 1.0f;
        const float half = static_cast<float>(rotary_half_dimension);
        rope_low = half * std::log(static_cast<float>(config.initial_context_length) / (config.rope_ntk_beta * 2.0f * std::acos(-1.0f)))
                   / std::log(config.rope_theta);
        rope_high = half * std::log(static_cast<float>(config.initial_context_length) / (config.rope_ntk_alpha * 2.0f * std::acos(-1.0f)))
                    / std::log(config.rope_theta);
    }
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const float frequency = std::pow(config.rope_theta, static_cast<float>(2 * index) / static_cast<float>(config.head_dimension));
        float inverse_frequency = 1.0f / frequency;
        if (config.rope_scaling_factor > 1.0f)
        {
            const float ramp = std::clamp((static_cast<float>(index) - rope_low) / (rope_high - rope_low), 0.0f, 1.0f);
            const float mask = 1.0f - ramp;
            const float interpolation = 1.0f / (config.rope_scaling_factor * frequency);
            inverse_frequency = interpolation * (1.0f - mask) + inverse_frequency * mask;
        }
        implementation.rope_inverse_frequencies[index] = inverse_frequency;
    }

    std::vector<float> norm_values;
    if (!tensor_to_float_vector(norm_weight, norm_values))
        return {};
    for (float& value : norm_values)
        value += config.norm_weight_offset;
    ncnn::Layer* norm = ncnn::create_layer_vulkan(ncnn::LayerType::RMSNorm);
    if (!norm)
        return {};
    norm->vkdev = vkdev;
    ncnn::ParamDict norm_parameters;
    norm_parameters.set(0, static_cast<int>(config.hidden_size));
    norm_parameters.set(1, config.norm_epsilon);
    norm_parameters.set(2, 1);
    ncnn::Mat norm_model[1] = {ncnn::Mat(static_cast<int>(norm_values.size()), norm_values.data(), sizeof(float))};
    if (norm->load_param(norm_parameters) != 0 || norm->load_model(ncnn::ModelBinFromMatArray(norm_model)) != 0
        || norm->create_pipeline(implementation.option) != 0)
    {
        delete norm;
        return {};
    }
    implementation.layers.push_back(norm);
    implementation.norm = norm;

    ncnn::ParamDict slice_parameters;
    ncnn::Mat slice_sizes(3, sizeof(int));
    int* slice_values = static_cast<int*>(slice_sizes.data);
    slice_values[0] = static_cast<int>(query_columns);
    slice_values[1] = static_cast<int>(key_value_columns);
    slice_values[2] = static_cast<int>(key_value_columns);
    slice_parameters.set(0, slice_sizes);
    slice_parameters.set(1, 1);
    if (!create_vulkan_layer(ncnn::LayerType::Slice, slice_parameters, vkdev, implementation.option, implementation.layers, implementation.slice_qkv))
        return {};

    ncnn::ParamDict reshape_query_parameters;
    reshape_query_parameters.set(0, static_cast<int>(config.head_dimension));
    reshape_query_parameters.set(1, static_cast<int>(config.head_count));
    reshape_query_parameters.set(2, -1);
    if (!create_vulkan_layer(ncnn::LayerType::Reshape, reshape_query_parameters, vkdev, implementation.option, implementation.layers,
                             implementation.reshape_query))
        return {};

    ncnn::ParamDict reshape_key_value_parameters;
    reshape_key_value_parameters.set(0, static_cast<int>(config.head_dimension));
    reshape_key_value_parameters.set(1, static_cast<int>(config.kv_head_count));
    reshape_key_value_parameters.set(2, -1);
    if (!create_vulkan_layer(ncnn::LayerType::Reshape, reshape_key_value_parameters, vkdev, implementation.option, implementation.layers,
                             implementation.reshape_key_value))
        return {};

    ncnn::ParamDict permute_parameters;
    permute_parameters.set(0, 2);
    if (!create_vulkan_layer(ncnn::LayerType::Permute, permute_parameters, vkdev, implementation.option, implementation.layers,
                             implementation.permute_heads_tokens))
        return {};

    ncnn::ParamDict rotary_parameters;
    rotary_parameters.set(0, 0);
    if (!create_vulkan_layer(ncnn::LayerType::RotaryEmbed, rotary_parameters, vkdev, implementation.option, implementation.layers, implementation.rotary))
        return {};

    ncnn::ParamDict sdpa_parameters;
    sdpa_parameters.set(5, 1);
    sdpa_parameters.set(6, 1.0f / std::sqrt(static_cast<float>(config.head_dimension)));
    sdpa_parameters.set(7, 0);
    if (!create_vulkan_layer(ncnn::LayerType::SDPA, sdpa_parameters, vkdev, implementation.option, implementation.layers, implementation.sdpa))
        return {};

    ncnn::ParamDict reshape_attention_parameters;
    reshape_attention_parameters.set(0, static_cast<int>(query_columns));
    reshape_attention_parameters.set(1, -1);
    if (!create_vulkan_layer(ncnn::LayerType::Reshape, reshape_attention_parameters, vkdev, implementation.option, implementation.layers,
                             implementation.reshape_attention))
        return {};

    ncnn::ParamDict add_parameters;
    add_parameters.set(0, 0);
    if (!create_vulkan_layer(ncnn::LayerType::BinaryOp, add_parameters, vkdev, implementation.option, implementation.layers, implementation.add))
        return {};

    implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(vkdev));
    const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
    if (!create_attention_pipeline(implementation.vulkan_context, implementation.kv_option, attention_qkv_rope_shader, static_cast<int>(sizeof(attention_qkv_rope_shader) - 1),
                                   implementation.qkv_rope_pipeline)
        || !create_attention_pipeline(implementation.vulkan_context, implementation.kv_option, attention_decode_sdpa_shader, static_cast<int>(sizeof(attention_decode_sdpa_shader) - 1),
                                      implementation.decode_sdpa_pipeline)
        || !create_attention_pipeline(implementation.vulkan_context, implementation.kv_option, attention_ring_append_shader, static_cast<int>(sizeof(attention_ring_append_shader) - 1),
                                      implementation.ring_append_pipeline)
        || !create_attention_pipeline(implementation.vulkan_context, implementation.kv_option, attention_ring_zero_shader, static_cast<int>(sizeof(attention_ring_zero_shader) - 1),
                                      implementation.ring_zero_pipeline))
        return {};
    ncnn::VkTransfer command(vkdev);
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    ncnn::Mat rope_inverse_model(
        static_cast<int>(implementation.rope_inverse_frequencies.size()),
        sizeof(float));
    ncnn::Mat sink_model(static_cast<int>(config.head_count), sizeof(float));
    if (rope_inverse_model.empty() || sink_model.empty())
        return {};
    std::copy(
        implementation.rope_inverse_frequencies.begin(),
        implementation.rope_inverse_frequencies.end(),
        static_cast<float*>(rope_inverse_model.data));
    float* sink_values = static_cast<float*>(sink_model.data);
    std::fill_n(sink_values, config.head_count, 0.0f);
    if (has_flag(config.flags, AttentionSink))
    {
        std::copy(implementation.sinks.begin(), implementation.sinks.end(), sink_values);
    }
    command.record_upload(
        rope_inverse_model,
        implementation.rope_inverse_frequencies_gpu,
        upload_option);
    command.record_upload(sink_model, implementation.attention_sinks, upload_option);
    if (implementation.norm->upload_model(command, upload_option) != 0
        || implementation.rope_inverse_frequencies_gpu.empty()
        || implementation.attention_sinks.empty()
        || command.submit_and_wait() != 0)
        return {};
    return attention;
#else
    (void)norm_weight;
    (void)sinks;
    (void)fused_qkv;
    (void)output_projection;
    (void)config;
    return {};
#endif
}

std::shared_ptr<Attention_vulkan>
Attention_vulkan::create(
    const TensorData& norm_weight,
    const TensorData& query_norm_weight,
    const TensorData& key_norm_weight,
    const TensorData* sinks,
    std::shared_ptr<Bfloat16Linear_vulkan> fused_qkv_gate,
    std::shared_ptr<Bfloat16Linear_vulkan> output_projection,
    const AttentionConfig_vulkan& config)
{
#if NCNN_MOE_WITH_VULKAN
#if !NCNN_BATCH
    return {};
#endif
    if (!fused_qkv_gate || !output_projection
        || config.hidden_size == 0 || config.head_count == 0
        || config.kv_head_count == 0 || config.head_dimension == 0
        || (config.head_dimension & 1) != 0
        || config.head_count % config.kv_head_count != 0
        || config.activation_dtype != config.kv_cache_dtype
        || !has_flag(config.flags, AttentionQueryKeyNorm)
        || !has_flag(config.flags, AttentionOutputGate)
        || (!has_flag(config.flags, AttentionSink)
            && config.sliding_window > 0)
        || norm_weight.shape != std::vector<uint32_t>{config.hidden_size}
        || query_norm_weight.shape
               != std::vector<uint32_t>{config.head_dimension}
        || key_norm_weight.shape
               != std::vector<uint32_t>{config.head_dimension}
        || (has_flag(config.flags, AttentionSink)
            && (!sinks
                || sinks->shape
                       != std::vector<uint32_t>{config.head_count})))
    {
        return {};
    }

    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    const uint64_t expected_fused_columns = static_cast<uint64_t>(query_columns)
                                            + static_cast<uint64_t>(key_value_columns) * 2
                                            + query_columns;
    if (fused_qkv_gate->input_columns() != config.hidden_size
        || fused_qkv_gate->output_columns() != expected_fused_columns
        || output_projection->input_columns() != query_columns
        || output_projection->output_columns() != config.hidden_size
        || fused_qkv_gate->vulkan_context() != output_projection->vulkan_context())
    {
        return {};
    }

    const uint32_t rope_dimension = config.rope_head_dimension == 0
                                        ? config.head_dimension
                                        : config.rope_head_dimension;
    if (rope_dimension > config.head_dimension
        || (rope_dimension & 1) != 0)
    {
        return {};
    }

    std::shared_ptr<Attention_vulkan> attention(
        new Attention_vulkan);
    Implementation& implementation = *attention->d;
    implementation.fused_qkv_gate = std::move(fused_qkv_gate);
    implementation.output_projection_bfloat16 = std::move(output_projection);
    implementation.config = config;
    if (has_flag(config.flags, AttentionSink)
        && !tensor_to_float_vector(*sinks, implementation.sinks))
    {
        return {};
    }
    implementation.vulkan_context = implementation.fused_qkv_gate->vulkan_context();
    implementation.option = implementation.fused_qkv_gate->option();
    implementation.option.use_packing_layout = false;
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    implementation.kv_option = implementation.option;

    const uint32_t rotary_dimension = config.rope_head_dimension == 0
                                          ? config.head_dimension
                                          : config.rope_head_dimension;
    const uint32_t half_dimension = config.head_dimension / 2;
    const uint32_t rotary_half_dimension = rotary_dimension / 2;
    implementation.rope_inverse_frequencies.resize(half_dimension);
    float rope_low = 0.0f;
    float rope_high = 0.0f;
    if (config.rope_scaling_factor > 1.0f)
    {
        implementation.rope_concentration = 0.1f * std::log(config.rope_scaling_factor) + 1.0f;
        const float half = static_cast<float>(rotary_half_dimension);
        rope_low = half
                   * std::log(static_cast<float>(config.initial_context_length)
                              / (config.rope_ntk_beta * 2.0f
                                 * std::acos(-1.0f)))
                   / std::log(config.rope_theta);
        rope_high = half
                    * std::log(static_cast<float>(config.initial_context_length)
                               / (config.rope_ntk_alpha * 2.0f
                                  * std::acos(-1.0f)))
                    / std::log(config.rope_theta);
    }
    for (uint32_t index = 0; index < half_dimension; ++index)
    {
        const uint32_t frequency_dimension = index < rotary_half_dimension
                                                 ? rotary_dimension
                                                 : config.head_dimension;
        const float frequency = std::pow(
            config.rope_theta,
            static_cast<float>(2 * index)
                / static_cast<float>(frequency_dimension));
        float inverse_frequency = 1.0f / frequency;
        if (config.rope_scaling_factor > 1.0f)
        {
            const float ramp = std::clamp(
                (static_cast<float>(index) - rope_low)
                    / (rope_high - rope_low),
                0.0f,
                1.0f);
            const float mask = 1.0f - ramp;
            const float interpolation = 1.0f / (config.rope_scaling_factor * frequency);
            inverse_frequency = interpolation * (1.0f - mask) + inverse_frequency * mask;
        }
        implementation.rope_inverse_frequencies[index] = inverse_frequency;
    }

    std::vector<float> norm_values;
    if (!tensor_to_float_vector(norm_weight, norm_values))
        return {};
    for (float& value : norm_values)
        value += config.norm_weight_offset;
    ncnn::Layer* norm = ncnn::create_layer_vulkan(ncnn::LayerType::RMSNorm);
    if (!norm)
        return {};
    norm->vkdev = vkdev;
    ncnn::ParamDict norm_parameters;
    norm_parameters.set(0, static_cast<int>(config.hidden_size));
    norm_parameters.set(1, config.norm_epsilon);
    norm_parameters.set(2, 1);
    ncnn::Mat norm_model[1] = {
        ncnn::Mat(static_cast<int>(norm_values.size()), norm_values.data(), sizeof(float))};
    if (norm->load_param(norm_parameters) != 0
        || norm->load_model(ncnn::ModelBinFromMatArray(norm_model)) != 0
        || norm->create_pipeline(implementation.option) != 0)
    {
        delete norm;
        return {};
    }
    implementation.layers.push_back(norm);
    implementation.norm = norm;

    ncnn::ParamDict permute_parameters;
    permute_parameters.set(0, 2);
    if (!create_vulkan_layer(
            ncnn::LayerType::Permute,
            permute_parameters,
            vkdev,
            implementation.kv_option,
            implementation.layers,
            implementation.permute_heads_tokens))
    {
        return {};
    }

    ncnn::ParamDict sdpa_parameters;
    sdpa_parameters.set(5, 1);
    sdpa_parameters.set(
        6, 1.0f / std::sqrt(static_cast<float>(config.head_dimension)));
    sdpa_parameters.set(7, 0);
    if (!create_vulkan_layer(
            ncnn::LayerType::SDPA,
            sdpa_parameters,
            vkdev,
            implementation.option,
            implementation.layers,
            implementation.sdpa))
    {
        return {};
    }

    ncnn::ParamDict reshape_attention_parameters;
    reshape_attention_parameters.set(0, static_cast<int>(query_columns));
    reshape_attention_parameters.set(1, -1);
    if (!create_vulkan_layer(
            ncnn::LayerType::Reshape,
            reshape_attention_parameters,
            vkdev,
            implementation.kv_option,
            implementation.layers,
            implementation.reshape_attention))
    {
        return {};
    }

    ncnn::ParamDict add_parameters;
    add_parameters.set(0, 0);
    if (!create_vulkan_layer(
            ncnn::LayerType::BinaryOp,
            add_parameters,
            vkdev,
            implementation.kv_option,
            implementation.layers,
            implementation.add))
    {
        return {};
    }

    implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev));
    implementation.weight_staging_allocator.reset(
        new ncnn::VkWeightStagingAllocator(vkdev));
    const std::lock_guard<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    if (!create_attention_pipeline(
            implementation.vulkan_context,
            implementation.kv_option,
            attention_qkv_norm_rope_shader,
            static_cast<int>(sizeof(attention_qkv_norm_rope_shader) - 1),
            implementation.qkv_norm_rope_pipeline)
        || !create_attention_pipeline(
            implementation.vulkan_context,
            implementation.option,
            attention_output_gate_shader,
            static_cast<int>(sizeof(attention_output_gate_shader) - 1),
            implementation.output_gate_pipeline)
        || !create_attention_pipeline(
            implementation.vulkan_context,
            implementation.kv_option,
            attention_decode_sdpa_shader,
            static_cast<int>(sizeof(attention_decode_sdpa_shader) - 1),
            implementation.decode_sdpa_pipeline)
        || !create_attention_pipeline(
            implementation.vulkan_context,
            implementation.kv_option,
            attention_ring_append_shader,
            static_cast<int>(sizeof(attention_ring_append_shader) - 1),
            implementation.ring_append_pipeline)
        || !create_attention_pipeline(
            implementation.vulkan_context,
            implementation.kv_option,
            attention_ring_zero_shader,
            static_cast<int>(sizeof(attention_ring_zero_shader) - 1),
            implementation.ring_zero_pipeline))
    {
        return {};
    }

    ncnn::Mat query_norm_model;
    ncnn::Mat key_norm_model;
    if (!prepare_float_tensor_upload(query_norm_weight, query_norm_model)
        || !prepare_float_tensor_upload(key_norm_weight, key_norm_model))
    {
        return {};
    }
    if (config.norm_weight_offset != 0.0f)
    {
        float* query_values = static_cast<float*>(query_norm_model.data);
        float* key_values = static_cast<float*>(key_norm_model.data);
        for (uint32_t index = 0; index < config.head_dimension; ++index)
        {
            query_values[index] += config.norm_weight_offset;
            key_values[index] += config.norm_weight_offset;
        }
    }
    ncnn::Mat sink_model(static_cast<int>(config.head_count), sizeof(float));
    if (sink_model.empty())
        return {};
    float* sink_values = static_cast<float*>(sink_model.data);
    std::fill_n(sink_values, config.head_count, 0.0f);
    if (has_flag(config.flags, AttentionSink))
    {
        std::copy(implementation.sinks.begin(), implementation.sinks.end(), sink_values);
    }
    ncnn::VkTransfer command(vkdev);
    ncnn::Option upload_option = implementation.option;
    upload_option.blob_vkallocator = implementation.weight_allocator.get();
    upload_option.workspace_vkallocator = implementation.weight_allocator.get();
    upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
    ncnn::Mat rope_inverse_model(
        static_cast<int>(implementation.rope_inverse_frequencies.size()),
        sizeof(float));
    if (rope_inverse_model.empty())
        return {};
    std::copy(
        implementation.rope_inverse_frequencies.begin(),
        implementation.rope_inverse_frequencies.end(),
        static_cast<float*>(rope_inverse_model.data));
    command.record_upload(
        query_norm_model,
        implementation.query_norm_weight,
        upload_option);
    command.record_upload(
        key_norm_model,
        implementation.key_norm_weight,
        upload_option);
    command.record_upload(
        rope_inverse_model,
        implementation.rope_inverse_frequencies_gpu,
        upload_option);
    command.record_upload(sink_model, implementation.attention_sinks, upload_option);
    if (implementation.norm->upload_model(command, upload_option) != 0
        || implementation.query_norm_weight.empty()
        || implementation.key_norm_weight.empty()
        || implementation.rope_inverse_frequencies_gpu.empty()
        || implementation.attention_sinks.empty()
        || command.submit_and_wait() != 0)
    {
        return {};
    }
    implementation.weight_staging_allocator.reset();
    return attention;
#else
    (void)norm_weight;
    (void)query_norm_weight;
    (void)key_norm_weight;
    (void)sinks;
    (void)fused_qkv_gate;
    (void)output_projection;
    (void)config;
    return {};
#endif
}

bool Attention_vulkan::forward(uint64_t position_offset, LayerCache& cache, const ActivationBuffer& input, ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    const AttentionConfig_vulkan& config = implementation.config;
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    const size_t activation_element_size = vulkan_activation_element_size(implementation.kv_option);
    const bool low_precision_kv = vulkan_activation_storage_variant(implementation.kv_option) != 0;
    const bool has_device_cache = cache.vulkan_attention_cache != nullptr;
    const bool promote_host_cache = cache.token_count != 0 && !has_device_cache;
    if (!has_flag(config.optimization_flags, OptimizationVulkanAttention)
        || cache.vulkan_attention_state_unknown
        || input.rows() == 0 || input.columns() != config.hidden_size || input.rows() > static_cast<size_t>(std::numeric_limits<int>::max())
        || (cache.dtype != config.kv_cache_dtype && cache.token_count != 0)
        || (cache.token_count != 0
            && (cache.capacity_tokens == 0
                || cache.first_slot >= cache.capacity_tokens
                || cache.token_count > cache.capacity_tokens))
        || (cache.transaction.active && !has_device_cache)
        || (promote_host_cache
            && (cache.vulkan_attention_promotion_disabled
                || !has_flag(config.optimization_flags, OptimizationVulkanKvPromotion)))
        || (has_device_cache
            && (cache.vulkan_attention_cache->key.empty()
                || cache.vulkan_attention_cache->value.empty()
                || cache.vulkan_attention_cache->key.dims != 3
                || cache.vulkan_attention_cache->value.dims != 3
                || cache.vulkan_attention_cache->key.w
                       != static_cast<int>(config.head_dimension)
                || cache.vulkan_attention_cache->value.w
                       != static_cast<int>(config.head_dimension)
                || cache.vulkan_attention_cache->key.c
                       != static_cast<int>(config.kv_head_count)
                || cache.vulkan_attention_cache->value.c
                       != static_cast<int>(config.kv_head_count)
                || static_cast<uint64_t>(
                       cache.vulkan_attention_cache->key.h)
                       != cache.capacity_tokens * 2
                || static_cast<uint64_t>(
                       cache.vulkan_attention_cache->value.h)
                       != cache.capacity_tokens * 2
                || cache.vulkan_attention_cache->key.elemsize
                       != activation_element_size
                || cache.vulkan_attention_cache->value.elemsize
                       != activation_element_size
                || cache.vulkan_attention_cache->key.elempack != 1
                || cache.vulkan_attention_cache->value.elempack != 1)))
    {
        return false;
    }

    const bool bfloat16_storage = config.activation_dtype == DType::BFloat16 && implementation.option.use_bf16_storage;
    const bool device_rope = has_flag(config.optimization_flags, OptimizationVulkanDeviceRope)
                             && !implementation.rope_inverse_frequencies_gpu.empty()
                             && position_offset <= std::numeric_limits<uint32_t>::max()
                             && input.rows() - 1 <= std::numeric_limits<uint32_t>::max() - position_offset;
    const bool query_key_norm_and_gate = implementation.fused_qkv_gate != nullptr;
    const bool use_qkv_rope = support_qkv_rope(input.rows());
    const bool use_qkv_ring = use_qkv_rope
                              && input.rows() == 1
                              && has_flag(config.optimization_flags, OptimizationVulkanQkvRing);
    if (!use_qkv_rope && (device_rope || query_key_norm_and_gate))
        return false;
    const uint64_t query_columns_u64 = static_cast<uint64_t>(config.head_count)
                                       * config.head_dimension;
    const uint64_t kv_columns = static_cast<uint64_t>(config.kv_head_count)
                                * config.head_dimension;
    if (query_columns_u64 > std::numeric_limits<uint32_t>::max()
        || kv_columns > std::numeric_limits<uint32_t>::max())
        return false;
    const uint32_t query_columns = static_cast<uint32_t>(query_columns_u64);
    uint64_t promotion_elements = 0;
    uint64_t promotion_transfer_bytes = 0;
    if (promote_host_cache
        && (!checked_multiply_u64(
                cache.token_count,
                kv_columns,
                promotion_elements)
            || !checked_multiply_u64(
                promotion_elements,
                sizeof(float) * 2,
                promotion_transfer_bytes)))
    {
        cache.vulkan_attention_promotion_disabled = true;
        return false;
    }
    const uint64_t actual_token_count = cache.token_count + input.rows();
    const uint64_t sink_token_count = has_flag(config.flags, AttentionSink) ? 1 : 0;
    const uint64_t destination_count = actual_token_count + sink_token_count;
    if (actual_token_count < cache.token_count || destination_count < actual_token_count
        || destination_count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    if (promote_host_cache
        && !attention_promotion_within_budget(
            *implementation.vulkan_context,
            next_attention_ring_capacity(0, actual_token_count),
            kv_columns,
            activation_element_size,
            promotion_transfer_bytes))
    {
        cache.vulkan_attention_promotion_disabled = true;
        return false;
    }
    if (promote_host_cache)
        cache.vulkan_attention_promotion_disabled = true;

    const bool use_decode_sdpa = has_flag(config.optimization_flags, OptimizationVulkanDecodeSdpa)
                                 && input.rows() == 1
                                 && config.head_dimension <= 128
                                 && destination_count <= 4096;

    const Linear* fused = implementation.fused_qkv.get();
    const Bfloat16Linear_vulkan* fused_gate = implementation.fused_qkv_gate.get();
    const Linear* projection = implementation.output_projection.get();
    const Bfloat16Linear_vulkan* projection_bfloat16 = implementation.output_projection_bfloat16.get();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = input.rows() == 1
                                   && vulkan_activation_storage_variant(implementation.option) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(
                                       static_cast<size_t>(config.hidden_size)
                                           * sizeof(float),
                                       input.dtype());
    if (!fill_staging_upload(input, transfer_slot.upload, transfer_slot.staging_allocator, runtime_state)
        || (!device_rope
            && !fill_rope_staging_pair(transfer_slot.rope_cosine, transfer_slot.rope_sine, input.rows(), position_offset,
                                       implementation.rope_inverse_frequencies, implementation.rope_concentration, bfloat16_storage,
                                       transfer_slot.staging_allocator, runtime_state))
        || (!use_decode_sdpa
            && !fill_attention_mask_staging(transfer_slot.attention_mask, input.rows(), destination_count, position_offset, cache, config, implementation.sinks,
                                            bfloat16_storage, transfer_slot.staging_allocator, runtime_state))
        || (promote_host_cache
            && !fill_attention_cache_promotion_staging(
                transfer_slot.attention_cache_key,
                transfer_slot.attention_cache_value,
                cache,
                config,
                transfer_slot.staging_allocator, runtime_state))
        || !prepare_staging_batch(transfer_slot.download, input.rows(), config.hidden_size, transfer_slot.staging_allocator, runtime_state))
    {
        return false;
    }

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
        {
            return false;
        }
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    ncnn::VkMat input_gpu;
    ncnn::VkMat cosine_gpu = device_rope
                                 ? implementation.rope_inverse_frequencies_gpu
                                 : ncnn::VkMat();
    ncnn::VkMat sine_gpu = device_rope
                               ? implementation.rope_inverse_frequencies_gpu
                               : ncnn::VkMat();
    ncnn::VkMat mask_gpu;
    ncnn::VkMat promoted_key_gpu;
    ncnn::VkMat promoted_value_gpu;
    if (direct_host_input)
        input_gpu = bind_direct_host_input(transfer_slot.upload, runtime_state);
    else if (!record_prepared_staging_upload(transfer_slot.upload, input.rows(), input_gpu, command, vkdev, implementation.option, input.dtype()))
    {
        return false;
    }
    if (!device_rope
        && (!record_mapped_upload(transfer_slot.rope_cosine, cosine_gpu, command, implementation.option)
            || !record_mapped_upload(transfer_slot.rope_sine, sine_gpu, command, implementation.option)))
    {
        return false;
    }
    if (promote_host_cache
        && (!record_mapped_activation_upload(
                transfer_slot.attention_cache_key,
                promoted_key_gpu,
                command,
                vkdev,
                implementation.kv_option)
            || !record_mapped_activation_upload(
                transfer_slot.attention_cache_value,
                promoted_value_gpu,
                command,
                vkdev,
                implementation.kv_option)))
    {
        return false;
    }
    if (!use_decode_sdpa)
    {
        if (!record_mapped_upload(transfer_slot.attention_mask, mask_gpu, command, implementation.option))
        {
            return false;
        }
    }
    ncnn::VkMat normalized_gpu;
    if (implementation.norm->forward(input_gpu, normalized_gpu, command, implementation.option) != 0)
    {
        return false;
    }

    ncnn::VkMat fused_gpu;
    if (query_key_norm_and_gate)
    {
        ncnn::VkMat normalized_unpacked = normalized_gpu;
        if (normalized_unpacked.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(
                normalized_unpacked,
                unpacked,
                1,
                command,
                implementation.option);
            normalized_unpacked = unpacked;
        }
        if (!fused_gate
            || normalized_unpacked.empty()
            || normalized_unpacked.dims != 2
            || normalized_unpacked.h != static_cast<int>(input.rows())
            || fused_gate->forward(
                   normalized_unpacked,
                   fused_gpu,
                   command,
                   implementation.option)
                   != 0)
        {
            return false;
        }
    }
    else
    {
        if (!fused
            || fused->forward(
                   normalized_gpu,
                   fused_gpu,
                   command,
                   implementation.option)
                   != 0)
        {
            return false;
        }
    }

    uint64_t ring_capacity = cache.capacity_tokens;
    uint64_t ring_first_slot = cache.first_slot;
    const bool ring_resized = !promote_host_cache && actual_token_count > ring_capacity;
    const bool allocate_ring = !has_device_cache || ring_resized;
    std::shared_ptr<AttentionCache_vulkan> next_cache = allocate_ring
                                                            ? std::make_shared<AttentionCache_vulkan>()
                                                            : cache.vulkan_attention_cache;
    if (allocate_ring)
    {
        ring_capacity = next_attention_ring_capacity(
            has_device_cache ? ring_capacity : 0,
            actual_token_count);
        if (!create_attention_ring_storage(
                *next_cache,
                config.head_dimension,
                config.kv_head_count,
                ring_capacity,
                activation_element_size,
                implementation.option.blob_vkallocator))
        {
            return false;
        }
        if (cache.token_count != 0)
        {
            const ncnn::VkMat previous_key = promote_host_cache
                                                 ? promoted_key_gpu
                                                 : attention_ring_view(
                                                       cache.vulkan_attention_cache->key,
                                                       cache.first_slot,
                                                       cache.token_count);
            const ncnn::VkMat previous_value = promote_host_cache
                                                   ? promoted_value_gpu
                                                   : attention_ring_view(
                                                         cache.vulkan_attention_cache->value,
                                                         cache.first_slot,
                                                         cache.token_count);
            if (!record_attention_ring_append(implementation.ring_append_pipeline.get(), previous_key, previous_value, next_cache->key, next_cache->value,
                                              ring_capacity, 0, command))
            {
                return false;
            }
        }
        ring_first_slot = 0;
    }
    const uint64_t append_slot = (ring_first_slot + cache.token_count) % ring_capacity;

    ncnn::VkMat query_rope;
    ncnn::VkMat key_rope;
    ncnn::VkMat value_heads;
    ncnn::VkMat output_gate;
    ncnn::VkMat fused_qkv_unpacked = fused_gpu;
    if (fused_qkv_unpacked.elempack != 1)
    {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(fused_qkv_unpacked, unpacked, 1, command, implementation.option);
        fused_qkv_unpacked = unpacked;
    }
    if (use_qkv_rope)
    {
        const AttentionCache_vulkan* ring = use_qkv_ring ? next_cache.get() : nullptr;
        if (query_key_norm_and_gate)
        {
            if (!record_qkv_norm_rope(
                    fused_qkv_unpacked, cosine_gpu, sine_gpu, input.rows(),
                    position_offset, device_rope, ring, ring_capacity,
                    append_slot, query_rope, key_rope, value_heads,
                    output_gate, command))
                return false;
        }
        else if (!record_qkv_rope(
                     fused_qkv_unpacked, cosine_gpu, sine_gpu, input.rows(),
                     position_offset, device_rope, ring, ring_capacity,
                     append_slot, query_rope, key_rope, value_heads, command))
        {
            return false;
        }
    }
    else
    {
        std::vector<ncnn::VkMat> qkv_input(1, fused_gpu);
        std::vector<ncnn::VkMat> qkv(3);
        if (implementation.slice_qkv->forward(qkv_input, qkv, command, implementation.option) != 0)
        {
            return false;
        }

        ncnn::VkMat query_shaped;
        ncnn::VkMat key_shaped;
        ncnn::VkMat value_shaped;
        if (implementation.reshape_query->forward(qkv[0], query_shaped, command, implementation.option) != 0
            || implementation.reshape_key_value->forward(qkv[1], key_shaped, command, implementation.option) != 0
            || implementation.reshape_key_value->forward(qkv[2], value_shaped, command, implementation.option) != 0)
        {
            return false;
        }

        ncnn::VkMat query_heads;
        ncnn::VkMat key_heads;
        if (implementation.permute_heads_tokens->forward(query_shaped, query_heads, command, implementation.option) != 0
            || implementation.permute_heads_tokens->forward(key_shaped, key_heads, command, implementation.option) != 0
            || implementation.permute_heads_tokens->forward(value_shaped, value_heads, command, implementation.option) != 0)
        {
            return false;
        }
        if (query_heads.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(query_heads, unpacked, 1, command, implementation.option);
            query_heads = unpacked;
        }
        if (key_heads.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(key_heads, unpacked, 1, command, implementation.option);
            key_heads = unpacked;
        }
        if (value_heads.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(value_heads, unpacked, 1, command, implementation.option);
            value_heads = unpacked;
        }

        std::vector<ncnn::VkMat> query_rope_input = {
            query_heads,
            cosine_gpu,
            sine_gpu,
        };
        std::vector<ncnn::VkMat> key_rope_input = {
            key_heads,
            cosine_gpu,
            sine_gpu,
        };
        std::vector<ncnn::VkMat> query_rope_output(1);
        std::vector<ncnn::VkMat> key_rope_output(1);
        if (implementation.rotary->forward(query_rope_input, query_rope_output, command, implementation.option) != 0
            || implementation.rotary->forward(key_rope_input, key_rope_output, command, implementation.option) != 0)
        {
            return false;
        }
        query_rope = query_rope_output[0];
        key_rope = key_rope_output[0];
    }

    if (query_rope.elempack != 1 || query_rope.dims != 3 || query_rope.w != static_cast<int>(config.head_dimension)
        || query_rope.h != static_cast<int>(input.rows()) || query_rope.c != static_cast<int>(config.head_count)
        || (!use_qkv_ring
            && (key_rope.elempack != 1 || value_heads.elempack != 1 || key_rope.dims != 3 || key_rope.w != static_cast<int>(config.head_dimension)
                || key_rope.h != static_cast<int>(input.rows()) || key_rope.c != static_cast<int>(config.kv_head_count) || value_heads.dims != 3
                || value_heads.w != static_cast<int>(config.head_dimension) || value_heads.h != static_cast<int>(input.rows())
                || value_heads.c != static_cast<int>(config.kv_head_count) || key_rope.elemsize != activation_element_size
                || value_heads.elemsize != activation_element_size)))
    {
        return false;
    }

    if (!use_qkv_ring
        && !record_attention_ring_append(implementation.ring_append_pipeline.get(), key_rope, value_heads, next_cache->key, next_cache->value, ring_capacity,
                                         append_slot, command))
    {
        return false;
    }

    if (sink_token_count != 0 && !use_qkv_ring)
    {
        const uint64_t sink_row = ring_first_slot + actual_token_count;
        if (!record_attention_ring_zero(implementation.ring_zero_pipeline.get(), next_cache->key, next_cache->value, sink_row, command))
        {
            return false;
        }
    }

    ncnn::VkMat combined_key = attention_ring_view(next_cache->key, ring_first_slot, destination_count);
    ncnn::VkMat combined_value = attention_ring_view(next_cache->value, ring_first_slot, destination_count);
    if (combined_key.empty() || combined_value.empty())
    {
        return false;
    }
    ncnn::VkMat attention_matrix;
    if (use_decode_sdpa)
    {
        if (!record_attention_decode_sdpa(implementation.decode_sdpa_pipeline.get(), query_rope, combined_key, combined_value, implementation.attention_sinks, config,
                                          destination_count, attention_matrix, command, implementation.option.blob_vkallocator))
        {
            return false;
        }
    }
    else
    {
        if (sink_token_count != 0 && use_qkv_ring)
        {
            const uint64_t sink_row = ring_first_slot + actual_token_count;
            if (!record_attention_ring_zero(implementation.ring_zero_pipeline.get(), next_cache->key, next_cache->value, sink_row, command))
            {
                return false;
            }
        }
        ncnn::VkMat sdpa_key = combined_key;
        ncnn::VkMat sdpa_value = combined_value;
        if (low_precision_kv)
        {
            vkdev->convert_packing(
                combined_key,
                sdpa_key,
                1,
                1,
                command,
                implementation.kv_option);
            vkdev->convert_packing(
                combined_value,
                sdpa_value,
                1,
                1,
                command,
                implementation.kv_option);
            if (sdpa_key.empty() || sdpa_value.empty()
                || sdpa_key.elemsize != sizeof(float)
                || sdpa_value.elemsize != sizeof(float))
            {
                return false;
            }
        }
        std::vector<ncnn::VkMat> sdpa_input = {
            query_rope,
            sdpa_key,
            sdpa_value,
            mask_gpu,
        };
        std::vector<ncnn::VkMat> sdpa_output(1);
        if (implementation.sdpa->forward(sdpa_input, sdpa_output, command, implementation.option) != 0)
        {
            return false;
        }

        ncnn::VkMat attention_token_major;
        if (implementation.permute_heads_tokens->forward(sdpa_output[0], attention_token_major, command, implementation.option) != 0
            || implementation.reshape_attention->forward(attention_token_major, attention_matrix, command, implementation.option) != 0)
        {
            return false;
        }
    }

    if (query_key_norm_and_gate)
    {
        if (attention_matrix.elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(
                attention_matrix,
                unpacked,
                1,
                command,
                implementation.option);
            attention_matrix = unpacked;
        }
        if (!record_attention_output_gate(
                implementation.output_gate_pipeline.get(),
                attention_matrix,
                output_gate,
                input.rows(),
                query_columns,
                command))
        {
            return false;
        }
    }

    ncnn::VkMat projected_gpu;
    if (query_key_norm_and_gate)
    {
        if (!projection_bfloat16
            || attention_matrix.empty()
            || attention_matrix.dims != 2
            || attention_matrix.h != static_cast<int>(input.rows()))
        {
            return false;
        }
        const int projection_result = projection_bfloat16->forward(
            attention_matrix,
            projected_gpu,
            command,
            implementation.option);
        if (projection_result != 0)
            return false;
    }
    else
    {
        if (!projection
            || projection->forward(
                   attention_matrix,
                   projected_gpu,
                   command,
                   implementation.option)
                   != 0)
        {
            return false;
        }
    }
    std::vector<ncnn::VkMat> add_input = {input_gpu, projected_gpu};
    std::vector<ncnn::VkMat> add_output(1);
    if (implementation.add->forward(add_input, add_output, command, implementation.option) != 0)
    {
        return false;
    }

    ncnn::VkMat download_gpu = add_output[0];
    if (download_gpu.elempack != 1)
    {
        ncnn::VkMat unpacked;
        vkdev->convert_packing(download_gpu, unpacked, 1, command, implementation.option);
        download_gpu = unpacked;
    }
    if (!record_prepared_activation_staging_download(
            download_gpu,
            input.rows(),
            config.hidden_size,
            transfer_slot.download,
            command,
            vkdev,
            implementation.option,
            output.dtype()))
    {
        return false;
    }

    const uint64_t total_actual_tokens = cache.token_count + input.rows();
    const uint64_t retained_tokens = config.sliding_window == 0 ? total_actual_tokens
                                                                : std::min<uint64_t>(total_actual_tokens, config.sliding_window > 1 ? config.sliding_window - 1 : 0);
    const uint64_t dropped_tokens = total_actual_tokens - retained_tokens;
    const uint64_t next_first_slot = retained_tokens == 0 ? 0 : (ring_first_slot + dropped_tokens) % ring_capacity;
    const uint64_t allocated_cache_size = retained_tokens == 0
                                              ? 0
                                              : static_cast<uint64_t>(next_cache->key.cstep) * next_cache->key.c * next_cache->key.elemsize
                                                    + static_cast<uint64_t>(next_cache->value.cstep) * next_cache->value.c * next_cache->value.elemsize;

    output = ActivationBuffer(input.rows(), config.hidden_size);
    const int submit_result = submit_compute_and_wait(command, runtime_state);
    if (submit_result != 0)
    {
        if (has_device_cache && !ring_resized)
            cache.vulkan_attention_state_unknown = true;
        return false;
    }
    if (!copy_staging_to_cpu_batch(transfer_slot.download, output))
    {
        if (has_device_cache && !ring_resized)
            cache.vulkan_attention_state_unknown = true;
        return false;
    }
    for (size_t row_index = 0; row_index < output.rows(); ++row_index)
    {
        for (uint32_t column = 0; column < output.columns(); ++column)
        {
            if (!std::isfinite(output.row(row_index)[column]))
            {
                if (has_device_cache && !ring_resized)
                    cache.vulkan_attention_state_unknown = true;
                return false;
            }
        }
    }

    const uint64_t previous_start = cache.token_count == 0 ? position_offset : cache.start_position;
    std::vector<float>{}.swap(cache.keys);
    std::vector<float>{}.swap(cache.values);
    std::vector<uint16_t>{}.swap(cache.bfloat16_keys);
    std::vector<uint16_t>{}.swap(cache.bfloat16_values);
    cache.start_position = previous_start + dropped_tokens;
    cache.token_count = retained_tokens;
    cache.first_slot = next_first_slot;
    cache.capacity_tokens = retained_tokens == 0 ? 0 : ring_capacity;
    cache.columns = config.kv_head_count * config.head_dimension;
    cache.dtype = config.kv_cache_dtype;
    if (retained_tokens == 0)
        cache.vulkan_attention_cache.reset();
    else if (allocate_ring)
        cache.vulkan_attention_cache = std::move(next_cache);
    cache.device_allocated_size = allocated_cache_size;
    cache.vulkan_attention_state_unknown = false;
    record_standard_cache_transaction_rows(cache, input.rows());
    runtime_state.dispatches += 2;
    ++runtime_state.attention_blocks;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    runtime_state.auxiliary_uploads += (!use_decode_sdpa ? 1 : 0) + (device_rope ? 0 : 2)
                                       + (promote_host_cache ? 2 : 0);
    if (use_qkv_rope)
        ++runtime_state.attention_qkv_rope_fusions;
    if (device_rope)
        ++runtime_state.attention_device_rope_fusions;
    if (use_qkv_ring)
        ++runtime_state.attention_qkv_ring_fusions;
    if (use_decode_sdpa)
        ++runtime_state.attention_decode_sdpa_fusions;
    ++runtime_state.kv_ring_appends;
    if (ring_resized)
        ++runtime_state.kv_ring_resizes;
    if (ring_first_slot + destination_count > ring_capacity)
        ++runtime_state.kv_ring_wrapped_views;
    if (promote_host_cache)
    {
        ++runtime_state.kv_cache_promotions;
        runtime_state.kv_cache_promotion_bytes += promotion_transfer_bytes;
    }
    runtime_state.auxiliary_upload_bytes += (device_rope
                                                 ? 0
                                                 : transfer_slot.rope_cosine.buffer_capacity()
                                                       + transfer_slot.rope_sine.buffer_capacity())
                                            + (!use_decode_sdpa ? transfer_slot.attention_mask.buffer_capacity() : 0)
                                            + (promote_host_cache
                                                   ? transfer_slot.attention_cache_key.buffer_capacity()
                                                         + transfer_slot.attention_cache_value.buffer_capacity()
                                                   : 0);
    if (promote_host_cache)
        cache.vulkan_attention_promotion_disabled = false;
    return true;
#else
    (void)position_offset;
    (void)cache;
    (void)input;
    (void)output;
    return false;
#endif
}

bool Attention_vulkan::materialize_device_cache(
    LayerCache& cache) const
{
#if NCNN_MOE_WITH_VULKAN
    const Implementation& implementation = *d;
    const AttentionConfig_vulkan& config = implementation.config;
    if (!implementation.vulkan_context
        || cache.vulkan_attention_state_unknown
        || !cache.vulkan_attention_cache
        || cache.token_count == 0
        || cache.capacity_tokens == 0
        || cache.first_slot >= cache.capacity_tokens
        || cache.token_count > cache.capacity_tokens
        || cache.dtype != config.kv_cache_dtype
        || config.head_dimension == 0
        || config.kv_head_count == 0
        || cache.token_count
               > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }

    const size_t element_size = vulkan_activation_element_size(implementation.kv_option);
    const AttentionCache_vulkan& device_cache = *cache.vulkan_attention_cache;
    if (device_cache.key.empty()
        || device_cache.value.empty()
        || device_cache.key.dims != 3
        || device_cache.value.dims != 3
        || device_cache.key.w
               != static_cast<int>(config.head_dimension)
        || device_cache.value.w
               != static_cast<int>(config.head_dimension)
        || device_cache.key.c
               != static_cast<int>(config.kv_head_count)
        || device_cache.value.c
               != static_cast<int>(config.kv_head_count)
        || static_cast<uint64_t>(device_cache.key.h)
               != cache.capacity_tokens * 2
        || static_cast<uint64_t>(device_cache.value.h)
               != cache.capacity_tokens * 2
        || device_cache.key.elemsize != element_size
        || device_cache.value.elemsize != element_size
        || device_cache.key.elempack != 1
        || device_cache.value.elempack != 1)
    {
        return false;
    }

    const ncnn::VkMat source_key = attention_ring_view(
        device_cache.key,
        cache.first_slot,
        cache.token_count);
    const ncnn::VkMat source_value = attention_ring_view(
        device_cache.value,
        cache.first_slot,
        cache.token_count);
    if (source_key.empty() || source_value.empty())
        return false;

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    if (!prepare_staging_tensor(
            transfer_slot.attention_cache_key,
            static_cast<int>(config.head_dimension),
            static_cast<int>(cache.token_count),
            static_cast<int>(config.kv_head_count),
            element_size,
            transfer_slot.staging_allocator,
            runtime_state)
        || !prepare_staging_tensor(
            transfer_slot.attention_cache_value,
            static_cast<int>(config.head_dimension),
            static_cast<int>(cache.token_count),
            static_cast<int>(config.kv_head_count),
            element_size,
            transfer_slot.staging_allocator,
            runtime_state))
    {
        return false;
    }

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    ncnn::Option copy_option = implementation.option;
    copy_option.blob_vkallocator = transfer_slot.staging_allocator;
    copy_option.workspace_vkallocator = transfer_slot.staging_allocator;
    copy_option.staging_vkallocator = transfer_slot.staging_allocator;
    ncnn::VkMat& key_staging = transfer_slot.attention_cache_key;
    ncnn::VkMat& value_staging = transfer_slot.attention_cache_value;
    command.record_clone(source_key, key_staging, copy_option);
    command.record_clone(source_value, value_staging, copy_option);
    if (key_staging.empty() || value_staging.empty())
        return false;

    if (submit_compute_and_wait(command, runtime_state) != 0)
        return false;
    key_staging.allocator->invalidate(key_staging.data);
    value_staging.allocator->invalidate(value_staging.data);
    const ncnn::Mat key_mapped = key_staging.mapped();
    const ncnn::Mat value_mapped = value_staging.mapped();
    if (key_mapped.empty()
        || value_mapped.empty()
        || key_mapped.dims != 3
        || value_mapped.dims != 3
        || key_mapped.w != static_cast<int>(config.head_dimension)
        || value_mapped.w != static_cast<int>(config.head_dimension)
        || key_mapped.h != static_cast<int>(cache.token_count)
        || value_mapped.h != static_cast<int>(cache.token_count)
        || key_mapped.c != static_cast<int>(config.kv_head_count)
        || value_mapped.c != static_cast<int>(config.kv_head_count)
        || key_mapped.elemsize != element_size
        || value_mapped.elemsize != element_size
        || key_mapped.elempack != 1
        || value_mapped.elempack != 1)
    {
        return false;
    }

    const size_t storage_variant = vulkan_activation_storage_variant(implementation.kv_option);
    ncnn::Mat key_float_storage;
    ncnn::Mat value_float_storage;
    const ncnn::Mat* key_float_source = &key_mapped;
    const ncnn::Mat* value_float_source = &value_mapped;
    if (storage_variant == 1
        || (storage_variant == 2 && cache.dtype == DType::Float32))
    {
        ncnn::Option cast_option = implementation.option;
        if (storage_variant == 1)
        {
            ncnn::cast_float16_to_float32(
                key_mapped,
                key_float_storage,
                cast_option);
            ncnn::cast_float16_to_float32(
                value_mapped,
                value_float_storage,
                cast_option);
        }
        else
        {
            ncnn::cast_bfloat16_to_float32(
                key_mapped,
                key_float_storage,
                cast_option);
            ncnn::cast_bfloat16_to_float32(
                value_mapped,
                value_float_storage,
                cast_option);
        }
        if (key_float_storage.empty() || value_float_storage.empty())
            return false;
        key_float_source = &key_float_storage;
        value_float_source = &value_float_storage;
    }

    const uint64_t columns_u64 = static_cast<uint64_t>(config.kv_head_count)
                                 * config.head_dimension;
    if (columns_u64 == 0
        || columns_u64 > std::numeric_limits<uint32_t>::max()
        || cache.token_count
               > static_cast<uint64_t>(
                   std::numeric_limits<size_t>::max() / columns_u64))
    {
        return false;
    }
    const uint32_t columns = static_cast<uint32_t>(columns_u64);
    const size_t element_count = static_cast<size_t>(cache.token_count) * columns;
    if (cache.dtype == DType::BFloat16)
    {
        std::vector<uint16_t> keys(element_count);
        std::vector<uint16_t> values(element_count);
        for (uint32_t head = 0; head < config.kv_head_count; ++head)
        {
            const ncnn::Mat key_channel = key_mapped.channel(head);
            const ncnn::Mat value_channel = value_mapped.channel(head);
            const ncnn::Mat key_float_channel = key_float_source->channel(head);
            const ncnn::Mat value_float_channel = value_float_source->channel(head);
            for (uint64_t token = 0; token < cache.token_count; ++token)
            {
                uint16_t* key_destination = keys.data()
                                            + static_cast<size_t>(token) * columns
                                            + static_cast<size_t>(head) * config.head_dimension;
                uint16_t* value_destination = values.data()
                                              + static_cast<size_t>(token) * columns
                                              + static_cast<size_t>(head) * config.head_dimension;
                if (storage_variant == 2)
                {
                    std::copy_n(
                        key_channel.row<uint16_t>(static_cast<int>(token)),
                        config.head_dimension,
                        key_destination);
                    std::copy_n(
                        value_channel.row<uint16_t>(static_cast<int>(token)),
                        config.head_dimension,
                        value_destination);
                }
                else
                {
                    const float* key_source = key_float_channel.row<float>(static_cast<int>(token));
                    const float* value_source = value_float_channel.row<float>(static_cast<int>(token));
                    for (uint32_t column = 0;
                         column < config.head_dimension;
                         ++column)
                    {
                        key_destination[column] = float_to_bfloat16(key_source[column]);
                        value_destination[column] = float_to_bfloat16(value_source[column]);
                    }
                }
            }
        }
        cache.bfloat16_keys = std::move(keys);
        cache.bfloat16_values = std::move(values);
        std::vector<float>{}.swap(cache.keys);
        std::vector<float>{}.swap(cache.values);
    }
    else if (cache.dtype == DType::Float32)
    {
        std::vector<float> keys(element_count);
        std::vector<float> values(element_count);
        for (uint32_t head = 0; head < config.kv_head_count; ++head)
        {
            const ncnn::Mat key_channel = key_float_source->channel(head);
            const ncnn::Mat value_channel = value_float_source->channel(head);
            for (uint64_t token = 0; token < cache.token_count; ++token)
            {
                std::copy_n(
                    key_channel.row<float>(static_cast<int>(token)),
                    config.head_dimension,
                    keys.data()
                        + static_cast<size_t>(token) * columns
                        + static_cast<size_t>(head) * config.head_dimension);
                std::copy_n(
                    value_channel.row<float>(static_cast<int>(token)),
                    config.head_dimension,
                    values.data()
                        + static_cast<size_t>(token) * columns
                        + static_cast<size_t>(head) * config.head_dimension);
            }
        }
        cache.keys = std::move(keys);
        cache.values = std::move(values);
        std::vector<uint16_t>{}.swap(cache.bfloat16_keys);
        std::vector<uint16_t>{}.swap(cache.bfloat16_values);
    }
    else
    {
        return false;
    }

    cache.first_slot = 0;
    cache.capacity_tokens = cache.token_count;
    cache.vulkan_attention_cache.reset();
    cache.device_allocated_size = 0;
    cache.vulkan_attention_state_unknown = false;
    // The failed device path has now been converted to a valid CPU path.  Do
    // not immediately re-promote the same cache and repeat the failure.
    cache.vulkan_attention_promotion_disabled = true;
    ++runtime_state.attention_cache_materializations;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_downloads;
    return true;
#else
    (void)cache;
    return false;
#endif
}

void Attention_vulkan::record_cpu_fallback() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    if (d && d->vulkan_context)
        ++d->vulkan_context->runtime_state().attention_cpu_fallbacks;
#endif
}

AttentionBatchResult_vulkan
Attention_vulkan::forward_batch(
    std::span<const AttentionBatchEntry_vulkan> entries) const
{
#if NCNN_MOE_WITH_VULKAN
    const uint64_t optimization_flags = d->config.optimization_flags;
    if (entries.size() < 2
        || !has_flag(optimization_flags, OptimizationVulkanAttention)
        || !has_flag(optimization_flags, OptimizationVulkanAttentionBatch))
        return AttentionBatchResult_vulkan::NotExecuted;

    const Implementation& implementation = *d;
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    const AttentionConfig_vulkan& config = implementation.config;
    const size_t activation_element_size = vulkan_activation_element_size(implementation.kv_option);
    const bool low_precision_kv = vulkan_activation_storage_variant(implementation.kv_option) != 0;
    const bool bfloat16_storage = config.activation_dtype == DType::BFloat16
                                  && implementation.option.use_bf16_storage;
    const uint64_t query_columns_u64 = static_cast<uint64_t>(config.head_count)
                                       * config.head_dimension;
    if (query_columns_u64 > std::numeric_limits<uint32_t>::max())
        return AttentionBatchResult_vulkan::NotExecuted;
    const uint32_t query_columns = static_cast<uint32_t>(query_columns_u64);
    const uint64_t sink_token_count = has_flag(config.flags, AttentionSink) ? 1 : 0;
    const bool query_key_norm_and_gate = implementation.fused_qkv_gate != nullptr;
    const Linear* fused = implementation.fused_qkv.get();
    const Bfloat16Linear_vulkan* fused_gate = implementation.fused_qkv_gate.get();
    const Linear* projection = implementation.output_projection.get();
    const Bfloat16Linear_vulkan* projection_bfloat16 = implementation.output_projection_bfloat16.get();
    const bool use_qkv_rope = support_qkv_rope(1);
    const bool use_qkv_ring = use_qkv_rope
                              && has_flag(config.optimization_flags, OptimizationVulkanQkvRing);
    if (!use_qkv_rope && query_key_norm_and_gate)
        return AttentionBatchResult_vulkan::NotExecuted;

    VulkanTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    // Host dtype is checked per entry before direct binding.
    const bool direct_host_input = vulkan_activation_storage_variant(implementation.option) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(
                                       static_cast<size_t>(config.hidden_size) * sizeof(float),
                                       DType::Float32);

    struct PreparedAttentionEntry
    {
        const AttentionBatchEntry_vulkan* entry = nullptr;
        ncnn::VkMat upload;
        ncnn::VkMat download;
        ncnn::VkMat rope_cosine;
        ncnn::VkMat rope_sine;
        ncnn::VkMat attention_mask;
        ncnn::VkMat input_gpu;
        ncnn::VkMat cosine_gpu;
        ncnn::VkMat sine_gpu;
        ncnn::VkMat mask_gpu;
        ncnn::VkMat normalized_gpu;
        ncnn::VkMat normalized_unpacked_gpu;
        ncnn::VkMat fused_gpu;
        ncnn::VkMat fused_qkv_unpacked_gpu;
        std::vector<ncnn::VkMat> qkv;
        ncnn::VkMat query_shaped_gpu;
        ncnn::VkMat key_shaped_gpu;
        ncnn::VkMat value_shaped_gpu;
        ncnn::VkMat query_heads_gpu;
        ncnn::VkMat key_heads_gpu;
        ncnn::VkMat query_rope;
        ncnn::VkMat key_rope;
        ncnn::VkMat value_heads;
        ncnn::VkMat output_gate;
        std::vector<ncnn::VkMat> query_rope_output;
        std::vector<ncnn::VkMat> key_rope_output;
        ncnn::VkMat combined_key;
        ncnn::VkMat combined_value;
        ncnn::VkMat attention_matrix;
        std::vector<ncnn::VkMat> sdpa_output;
        ncnn::VkMat attention_token_major;
        ncnn::VkMat projected_gpu;
        std::vector<ncnn::VkMat> add_output;
        ncnn::VkMat download_gpu;
        std::vector<ncnn::VkMat> retained_gpu;
        std::shared_ptr<AttentionCache_vulkan> next_cache;
        uint64_t actual_token_count = 0;
        uint64_t destination_count = 0;
        uint64_t retained_tokens = 0;
        uint64_t dropped_tokens = 0;
        uint64_t ring_first_slot = 0;
        uint64_t next_first_slot = 0;
        uint64_t allocated_cache_size = 0;
        bool use_decode_sdpa = false;
        bool device_rope = false;
    };

    std::vector<PreparedAttentionEntry> prepared;
    prepared.reserve(entries.size());
    for (const AttentionBatchEntry_vulkan& entry : entries)
    {
        if (!entry.cache || !entry.input || !entry.output)
            return AttentionBatchResult_vulkan::NotExecuted;
        LayerCache& cache = *entry.cache;
        const ActivationBuffer& input = *entry.input;
        if (cache.vulkan_attention_state_unknown)
            return AttentionBatchResult_vulkan::Failed;
        if (input.rows() != 1
            || input.columns() != config.hidden_size
            || cache.transaction.active
            || cache.token_count == 0
            || cache.dtype != config.kv_cache_dtype
            || cache.capacity_tokens == 0
            || cache.first_slot >= cache.capacity_tokens
            || cache.token_count > cache.capacity_tokens
            || !cache.vulkan_attention_cache
            || cache.vulkan_attention_cache->key.empty()
            || cache.vulkan_attention_cache->value.empty()
            || cache.vulkan_attention_cache->key.dims != 3
            || cache.vulkan_attention_cache->value.dims != 3
            || cache.vulkan_attention_cache->key.w
                   != static_cast<int>(config.head_dimension)
            || cache.vulkan_attention_cache->value.w
                   != static_cast<int>(config.head_dimension)
            || cache.vulkan_attention_cache->key.c
                   != static_cast<int>(config.kv_head_count)
            || cache.vulkan_attention_cache->value.c
                   != static_cast<int>(config.kv_head_count)
            || static_cast<uint64_t>(
                   cache.vulkan_attention_cache->key.h)
                   != cache.capacity_tokens * 2
            || static_cast<uint64_t>(
                   cache.vulkan_attention_cache->value.h)
                   != cache.capacity_tokens * 2
            || cache.vulkan_attention_cache->key.elemsize
                   != activation_element_size
            || cache.vulkan_attention_cache->value.elemsize
                   != activation_element_size
            || cache.vulkan_attention_cache->key.elempack != 1
            || cache.vulkan_attention_cache->value.elempack != 1)
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }

        const uint64_t actual_token_count = cache.token_count + 1;
        const uint64_t destination_count = actual_token_count + sink_token_count;
        if (actual_token_count <= cache.token_count
            || actual_token_count > cache.capacity_tokens
            || destination_count < actual_token_count
            || destination_count
                   > static_cast<uint64_t>(
                       std::numeric_limits<int>::max()))
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }

        prepared.emplace_back();
        PreparedAttentionEntry& work = prepared.back();
        work.entry = &entry;
        work.device_rope = has_flag(config.optimization_flags, OptimizationVulkanDeviceRope)
                           && !implementation.rope_inverse_frequencies_gpu.empty()
                           && entry.position_offset
                                  <= std::numeric_limits<uint32_t>::max();
        if (!use_qkv_rope && work.device_rope)
            return AttentionBatchResult_vulkan::NotExecuted;
        work.actual_token_count = actual_token_count;
        work.destination_count = destination_count;
        work.use_decode_sdpa = has_flag(config.optimization_flags, OptimizationVulkanDecodeSdpa)
                               && config.head_dimension <= 128
                               && destination_count <= 4096;
        work.next_cache = cache.vulkan_attention_cache;
        work.retained_tokens = config.sliding_window == 0
                                   ? actual_token_count
                                   : std::min<uint64_t>(
                                         actual_token_count,
                                         config.sliding_window > 1
                                             ? config.sliding_window - 1
                                             : 0);
        work.dropped_tokens = actual_token_count - work.retained_tokens;
        work.ring_first_slot = cache.first_slot;
        work.next_first_slot = work.retained_tokens == 0
                                   ? 0
                                   : (cache.first_slot + work.dropped_tokens)
                                         % cache.capacity_tokens;
        work.allocated_cache_size = work.retained_tokens == 0
                                        ? 0
                                        : static_cast<uint64_t>(work.next_cache->key.cstep)
                                                  * work.next_cache->key.c
                                                  * work.next_cache->key.elemsize
                                              + static_cast<uint64_t>(work.next_cache->value.cstep)
                                                    * work.next_cache->value.c
                                                    * work.next_cache->value.elemsize;

        if (!fill_staging_upload(
                input,
                work.upload,
                transfer_slot.staging_allocator,
                runtime_state)
            || (!work.device_rope
                && !fill_rope_staging_pair(
                    work.rope_cosine,
                    work.rope_sine,
                    1,
                    entry.position_offset,
                    implementation.rope_inverse_frequencies,
                    implementation.rope_concentration,
                    bfloat16_storage,
                    transfer_slot.staging_allocator,
                    runtime_state))
            || (!work.use_decode_sdpa
                && !fill_attention_mask_staging(
                    work.attention_mask,
                    1,
                    destination_count,
                    entry.position_offset,
                    cache,
                    config,
                    implementation.sinks,
                    bfloat16_storage,
                    transfer_slot.staging_allocator,
                    runtime_state))
            || !prepare_staging_batch(
                work.download,
                1,
                config.hidden_size,
                transfer_slot.staging_allocator,
                runtime_state))
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }
        entry.output->reset(1, config.hidden_size, false);
    }

    std::unique_lock<std::mutex> lock(
        implementation.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (command.reset() != 0)
            return AttentionBatchResult_vulkan::NotExecuted;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;

    for (PreparedAttentionEntry& work : prepared)
    {
        LayerCache& cache = *work.entry->cache;
        if (work.device_rope)
        {
            work.cosine_gpu = implementation.rope_inverse_frequencies_gpu;
            work.sine_gpu = implementation.rope_inverse_frequencies_gpu;
        }
        if (direct_host_input
            && work.entry->input->dtype() == DType::Float32)
            work.input_gpu = bind_direct_host_input(work.upload, runtime_state);
        else if (!record_prepared_staging_upload(
                     work.upload,
                     1,
                     work.input_gpu,
                     command,
                     vkdev,
                     implementation.option,
                     work.entry->input->dtype()))
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }
        if (!work.device_rope
            && (!record_mapped_activation_upload(
                    work.rope_cosine,
                    work.cosine_gpu,
                    command,
                    vkdev,
                    implementation.option)
                || !record_mapped_activation_upload(
                    work.rope_sine,
                    work.sine_gpu,
                    command,
                    vkdev,
                    implementation.option)))
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }
        if (!work.use_decode_sdpa)
        {
            if (!record_mapped_activation_upload(
                    work.attention_mask,
                    work.mask_gpu,
                    command,
                    vkdev,
                    implementation.option))
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
        }
        if (implementation.norm->forward(
                work.input_gpu,
                work.normalized_gpu,
                command,
                implementation.option)
            != 0)
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }

        if (query_key_norm_and_gate)
        {
            work.normalized_unpacked_gpu = work.normalized_gpu;
            if (work.normalized_unpacked_gpu.elempack != 1)
            {
                work.retained_gpu.push_back(
                    work.normalized_unpacked_gpu);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.normalized_unpacked_gpu,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.normalized_unpacked_gpu = unpacked;
            }
            if (!fused_gate
                || work.normalized_unpacked_gpu.empty()
                || work.normalized_unpacked_gpu.dims != 2
                || work.normalized_unpacked_gpu.h != 1
                || fused_gate->forward(
                       work.normalized_unpacked_gpu,
                       work.fused_gpu,
                       command,
                       implementation.option)
                       != 0)
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
        }
        else if (!fused
                 || fused->forward(
                        work.normalized_gpu,
                        work.fused_gpu,
                        command,
                        implementation.option)
                        != 0)
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }

        const uint64_t ring_capacity = cache.capacity_tokens;
        const uint64_t ring_first_slot = cache.first_slot;
        const uint64_t append_slot = (ring_first_slot + cache.token_count) % ring_capacity;
        work.fused_qkv_unpacked_gpu = work.fused_gpu;
        if (work.fused_qkv_unpacked_gpu.elempack != 1)
        {
            work.retained_gpu.push_back(work.fused_qkv_unpacked_gpu);
            ncnn::VkMat unpacked;
            vkdev->convert_packing(
                work.fused_qkv_unpacked_gpu,
                unpacked,
                1,
                command,
                implementation.option);
            work.fused_qkv_unpacked_gpu = unpacked;
        }
        if (use_qkv_rope)
        {
            const AttentionCache_vulkan* ring = use_qkv_ring ? work.next_cache.get() : nullptr;
            if (query_key_norm_and_gate)
            {
                if (!record_qkv_norm_rope(
                        work.fused_qkv_unpacked_gpu, work.cosine_gpu,
                        work.sine_gpu, 1, work.entry->position_offset,
                        work.device_rope, ring, ring_capacity, append_slot,
                        work.query_rope, work.key_rope, work.value_heads,
                        work.output_gate, command))
                    return AttentionBatchResult_vulkan::NotExecuted;
            }
            else if (!record_qkv_rope(
                         work.fused_qkv_unpacked_gpu, work.cosine_gpu,
                         work.sine_gpu, 1, work.entry->position_offset,
                         work.device_rope, ring, ring_capacity, append_slot,
                         work.query_rope, work.key_rope, work.value_heads,
                         command))
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
        }
        else
        {
            std::vector<ncnn::VkMat> qkv_input(1, work.fused_gpu);
            work.qkv.resize(3);
            if (implementation.slice_qkv->forward(
                    qkv_input,
                    work.qkv,
                    command,
                    implementation.option)
                != 0)
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
            if (implementation.reshape_query->forward(
                    work.qkv[0],
                    work.query_shaped_gpu,
                    command,
                    implementation.option)
                    != 0
                || implementation.reshape_key_value->forward(
                       work.qkv[1],
                       work.key_shaped_gpu,
                       command,
                       implementation.option)
                       != 0
                || implementation.reshape_key_value->forward(
                       work.qkv[2],
                       work.value_shaped_gpu,
                       command,
                       implementation.option)
                       != 0
                || implementation.permute_heads_tokens->forward(
                       work.query_shaped_gpu,
                       work.query_heads_gpu,
                       command,
                       implementation.option)
                       != 0
                || implementation.permute_heads_tokens->forward(
                       work.key_shaped_gpu,
                       work.key_heads_gpu,
                       command,
                       implementation.option)
                       != 0
                || implementation.permute_heads_tokens->forward(
                       work.value_shaped_gpu,
                       work.value_heads,
                       command,
                       implementation.option)
                       != 0)
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
            if (work.query_heads_gpu.elempack != 1)
            {
                work.retained_gpu.push_back(work.query_heads_gpu);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.query_heads_gpu,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.query_heads_gpu = unpacked;
            }
            if (work.key_heads_gpu.elempack != 1)
            {
                work.retained_gpu.push_back(work.key_heads_gpu);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.key_heads_gpu,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.key_heads_gpu = unpacked;
            }
            if (work.value_heads.elempack != 1)
            {
                work.retained_gpu.push_back(work.value_heads);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.value_heads,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.value_heads = unpacked;
            }
            std::vector<ncnn::VkMat> query_rope_input = {
                work.query_heads_gpu,
                work.cosine_gpu,
                work.sine_gpu};
            std::vector<ncnn::VkMat> key_rope_input = {
                work.key_heads_gpu,
                work.cosine_gpu,
                work.sine_gpu};
            work.query_rope_output.resize(1);
            work.key_rope_output.resize(1);
            if (implementation.rotary->forward(
                    query_rope_input,
                    work.query_rope_output,
                    command,
                    implementation.option)
                    != 0
                || implementation.rotary->forward(
                       key_rope_input,
                       work.key_rope_output,
                       command,
                       implementation.option)
                       != 0)
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
            work.query_rope = work.query_rope_output[0];
            work.key_rope = work.key_rope_output[0];
        }

        if (work.query_rope.elempack != 1
            || work.query_rope.dims != 3
            || work.query_rope.w
                   != static_cast<int>(config.head_dimension)
            || work.query_rope.h != 1
            || work.query_rope.c
                   != static_cast<int>(config.head_count)
            || (!use_qkv_ring
                && (work.key_rope.elempack != 1
                    || work.value_heads.elempack != 1
                    || work.key_rope.dims != 3
                    || work.key_rope.w
                           != static_cast<int>(config.head_dimension)
                    || work.key_rope.h != 1
                    || work.key_rope.c
                           != static_cast<int>(config.kv_head_count)
                    || work.value_heads.dims != 3
                    || work.value_heads.w
                           != static_cast<int>(config.head_dimension)
                    || work.value_heads.h != 1
                    || work.value_heads.c
                           != static_cast<int>(config.kv_head_count)
                    || work.key_rope.elemsize != activation_element_size
                    || work.value_heads.elemsize != activation_element_size)))
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }
        if (!use_qkv_ring
            && !record_attention_ring_append(
                implementation.ring_append_pipeline.get(),
                work.key_rope,
                work.value_heads,
                work.next_cache->key,
                work.next_cache->value,
                ring_capacity,
                append_slot,
                command))
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }

        if (sink_token_count != 0 && !use_qkv_ring)
        {
            const uint64_t sink_row = ring_first_slot + work.actual_token_count;
            if (!record_attention_ring_zero(
                    implementation.ring_zero_pipeline.get(),
                    work.next_cache->key,
                    work.next_cache->value,
                    sink_row,
                    command))
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
        }
        work.combined_key = attention_ring_view(
            work.next_cache->key,
            ring_first_slot,
            work.destination_count);
        work.combined_value = attention_ring_view(
            work.next_cache->value,
            ring_first_slot,
            work.destination_count);
        if (work.combined_key.empty() || work.combined_value.empty())
            return AttentionBatchResult_vulkan::NotExecuted;

        if (work.use_decode_sdpa)
        {
            if (!record_attention_decode_sdpa(
                    implementation.decode_sdpa_pipeline.get(),
                    work.query_rope,
                    work.combined_key,
                    work.combined_value,
                    implementation.attention_sinks,
                    config,
                    work.destination_count,
                    work.attention_matrix,
                    command,
                    implementation.option.blob_vkallocator))
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
        }
        else
        {
            if (sink_token_count != 0 && use_qkv_ring)
            {
                const uint64_t sink_row = ring_first_slot + work.actual_token_count;
                if (!record_attention_ring_zero(
                        implementation.ring_zero_pipeline.get(),
                        work.next_cache->key,
                        work.next_cache->value,
                        sink_row,
                        command))
                {
                    return AttentionBatchResult_vulkan::NotExecuted;
                }
            }
            ncnn::VkMat sdpa_key = work.combined_key;
            ncnn::VkMat sdpa_value = work.combined_value;
            if (low_precision_kv)
            {
                ncnn::VkMat converted_key;
                ncnn::VkMat converted_value;
                vkdev->convert_packing(
                    work.combined_key,
                    converted_key,
                    1,
                    1,
                    command,
                    implementation.kv_option);
                vkdev->convert_packing(
                    work.combined_value,
                    converted_value,
                    1,
                    1,
                    command,
                    implementation.kv_option);
                if (converted_key.empty() || converted_value.empty()
                    || converted_key.elemsize != sizeof(float)
                    || converted_value.elemsize != sizeof(float))
                {
                    return AttentionBatchResult_vulkan::NotExecuted;
                }
                work.retained_gpu.push_back(converted_key);
                work.retained_gpu.push_back(converted_value);
                sdpa_key = converted_key;
                sdpa_value = converted_value;
            }
            std::vector<ncnn::VkMat> sdpa_input = {
                work.query_rope,
                sdpa_key,
                sdpa_value,
                work.mask_gpu};
            work.sdpa_output.resize(1);
            if (implementation.sdpa->forward(
                    sdpa_input,
                    work.sdpa_output,
                    command,
                    implementation.option)
                    != 0
                || implementation.permute_heads_tokens->forward(
                       work.sdpa_output[0],
                       work.attention_token_major,
                       command,
                       implementation.option)
                       != 0
                || implementation.reshape_attention->forward(
                       work.attention_token_major,
                       work.attention_matrix,
                       command,
                       implementation.option)
                       != 0)
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
        }

        if (query_key_norm_and_gate)
        {
            if (work.attention_matrix.elempack != 1)
            {
                work.retained_gpu.push_back(work.attention_matrix);
                ncnn::VkMat unpacked;
                vkdev->convert_packing(
                    work.attention_matrix,
                    unpacked,
                    1,
                    command,
                    implementation.option);
                work.attention_matrix = unpacked;
            }
            if (!record_attention_output_gate(
                    implementation.output_gate_pipeline.get(),
                    work.attention_matrix,
                    work.output_gate,
                    1,
                    query_columns,
                    command))
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
        }

        if (query_key_norm_and_gate)
        {
            if (!projection_bfloat16
                || work.attention_matrix.empty()
                || work.attention_matrix.dims != 2
                || work.attention_matrix.h != 1
                || projection_bfloat16->forward(
                       work.attention_matrix,
                       work.projected_gpu,
                       command,
                       implementation.option)
                       != 0)
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
        }
        else if (!projection
                 || projection->forward(
                        work.attention_matrix,
                        work.projected_gpu,
                        command,
                        implementation.option)
                        != 0)
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }

        std::vector<ncnn::VkMat> add_input = {
            work.input_gpu,
            work.projected_gpu};
        work.add_output.resize(1);
        if (implementation.add->forward(
                add_input,
                work.add_output,
                command,
                implementation.option)
            != 0)
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }
        work.download_gpu = work.add_output[0];
        if (work.download_gpu.elempack != 1)
        {
            work.retained_gpu.push_back(work.download_gpu);
            ncnn::VkMat unpacked;
            vkdev->convert_packing(
                work.download_gpu,
                unpacked,
                1,
                command,
                implementation.option);
            work.download_gpu = unpacked;
        }
        if (!record_prepared_activation_staging_download(
                work.download_gpu,
                1,
                config.hidden_size,
                work.download,
                command,
                vkdev,
                implementation.option,
                work.entry->output->dtype()))
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }
    }

    const auto mark_device_states_unknown = [&]() noexcept {
        for (const PreparedAttentionEntry& work : prepared)
            work.entry->cache->vulkan_attention_state_unknown = true;
    };
    if (submit_compute_and_wait(command, runtime_state) != 0)
    {
        mark_device_states_unknown();
        return AttentionBatchResult_vulkan::Failed;
    }
    for (PreparedAttentionEntry& work : prepared)
    {
        if (!copy_staging_to_cpu_batch(
                work.download,
                *work.entry->output))
        {
            mark_device_states_unknown();
            return AttentionBatchResult_vulkan::Failed;
        }
        for (uint32_t column = 0;
             column < work.entry->output->columns();
             ++column)
        {
            if (!std::isfinite(work.entry->output->row(0)[column]))
            {
                mark_device_states_unknown();
                return AttentionBatchResult_vulkan::Failed;
            }
        }
    }

    for (PreparedAttentionEntry& work : prepared)
    {
        LayerCache& cache = *work.entry->cache;
        const uint64_t previous_start = cache.start_position;
        cache.start_position = previous_start + work.dropped_tokens;
        cache.token_count = work.retained_tokens;
        cache.first_slot = work.next_first_slot;
        cache.capacity_tokens = work.retained_tokens == 0 ? 0 : cache.capacity_tokens;
        cache.columns = config.kv_head_count * config.head_dimension;
        cache.dtype = config.kv_cache_dtype;
        cache.vulkan_attention_cache = work.retained_tokens == 0 ? nullptr : work.next_cache;
        cache.device_allocated_size = work.allocated_cache_size;
        cache.vulkan_attention_state_unknown = false;
    }
    const PreparedAttentionEntry& representative = prepared.front();
    runtime_state.dispatches += 2;
    ++runtime_state.attention_blocks;
    if (use_qkv_rope)
        ++runtime_state.attention_qkv_rope_fusions;
    if (representative.device_rope)
        ++runtime_state.attention_device_rope_fusions;
    if (use_qkv_ring)
        ++runtime_state.attention_qkv_ring_fusions;
    if (representative.use_decode_sdpa)
        ++runtime_state.attention_decode_sdpa_fusions;
    ++runtime_state.kv_ring_appends;
    if (representative.ring_first_slot
            + representative.destination_count
        > representative.entry->cache->capacity_tokens)
    {
        ++runtime_state.kv_ring_wrapped_views;
    }
    runtime_state.auxiliary_uploads += (!representative.use_decode_sdpa ? 1 : 0)
                                       + (representative.device_rope ? 0 : 2);
    runtime_state.auxiliary_upload_bytes += (representative.device_rope
                                                 ? 0
                                                 : representative.rope_cosine.buffer_capacity()
                                                       + representative.rope_sine.buffer_capacity())
                                            + (!representative.use_decode_sdpa
                                                   ? representative.attention_mask.buffer_capacity()
                                                   : 0);
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return AttentionBatchResult_vulkan::Executed;
#else
    (void)entries;
    return AttentionBatchResult_vulkan::NotExecuted;
#endif
}

} // namespace moe
} // namespace ncnn
