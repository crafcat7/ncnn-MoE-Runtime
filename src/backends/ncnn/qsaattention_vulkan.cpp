#include "qsaattention_vulkan.h"

#include "linear.h"
#include "rmsnorm_vulkan.h"
#include "vulkancontext.h"
#include "kernels/statecache.h"
#include "storage/weightstore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <numbers>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/qsa_attention.comp.hex.h"
#include "kernels/vulkan/qsa_cache_write.comp.hex.h"
#include "kernels/vulkan/qsa_copy.comp.hex.h"
#include "kernels/vulkan/qsa_norm_rope.comp.hex.h"
#include "kernels/vulkan/qsa_pool.comp.hex.h"
#include "kernels/vulkan/qsa_scores.comp.hex.h"
#include "kernels/vulkan/qsa_select.comp.hex.h"
#endif

namespace ncnn {
namespace moe {

class QsaAttentionWork_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    LayerCache* cache = nullptr;
    std::shared_ptr<QsaCache_vulkan> original;
    std::shared_ptr<QsaCache_vulkan> next;
    uint64_t previous_tokens = 0;
    std::vector<ncnn::VkMat> storage;
    uint64_t dispatches = 0;
#endif
    bool committed = false;
};

class QsaAttention_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    AttentionBlockPlan plan;
    uint32_t hidden = 0;
    uint64_t weights_bytes = 0;
    float concentration = 1.0f;
    std::shared_ptr<QnkLinear_vulkan> query;
    std::shared_ptr<QnkLinear_vulkan> key;
    std::shared_ptr<QnkLinear_vulkan> value;
    std::shared_ptr<QnkLinear_vulkan> gate;
    std::shared_ptr<QnkLinear_vulkan> index;
    std::shared_ptr<QnkLinear_vulkan> output;
    std::shared_ptr<RmsNorm_vulkan> norm;
    ncnn::VkMat query_norm;
    ncnn::VkMat key_norm;
    ncnn::VkMat index_query_norm;
    ncnn::VkMat index_key_norm;
    ncnn::VkMat frequency;
    std::shared_ptr<ncnn::Pipeline> norm_rope_pipeline;
    std::shared_ptr<ncnn::Pipeline> cache_write_pipeline;
    std::shared_ptr<ncnn::Pipeline> pool_pipeline;
    std::shared_ptr<ncnn::Pipeline> scores_pipeline;
    std::shared_ptr<ncnn::Pipeline> select_pipeline;
    std::shared_ptr<ncnn::Pipeline> attention_pipeline;
    std::shared_ptr<ncnn::Pipeline> copy_pipeline;
#endif
};

QsaAttention_vulkan::QsaAttention_vulkan()
    : d(new Implementation)
{
}

QsaAttention_vulkan::~QsaAttention_vulkan() = default;

QsaAttentionWork_vulkan::QsaAttentionWork_vulkan()
    : d(new Implementation)
{
}

QsaAttentionWork_vulkan::~QsaAttentionWork_vulkan() = default;

uint64_t QsaCache_vulkan::allocated_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return static_cast<uint64_t>(keys.total()) * keys.elemsize
           + static_cast<uint64_t>(values.total()) * values.elemsize
           + static_cast<uint64_t>(index_raw.total()) * index_raw.elemsize
           + static_cast<uint64_t>(pooled_keys.total()) * pooled_keys.elemsize;
#else
    return 0;
#endif
}

bool QsaAttentionWork_vulkan::commit()
{
#if NCNN_MOE_WITH_VULKAN
    if (d->committed || !d->cache || !d->next
        || d->cache->qsa_device_state != d->original
        || d->cache->token_count != d->previous_tokens
        || d->cache->transaction.active)
        return false;
    LayerCache& cache = *d->cache;
    const uint64_t previous_bytes = d->original ? d->original->allocated_bytes() : 0;
    cache.device_allocated_size = cache.device_allocated_size - std::min(cache.device_allocated_size, previous_bytes)
                                  + d->next->allocated_bytes();
    cache.qsa_device_state = d->next;
    cache.start_position = d->next->start_position;
    cache.token_count = d->next->token_count;
    cache.capacity_tokens = d->next->capacity;
    cache.first_slot = 0;
    cache.columns = d->next->columns;
    cache.dtype = d->next->dtype;
    // The device is authoritative; there is no duplicate CPU prefix.
    cache.keys.clear();
    cache.values.clear();
    cache.bfloat16_keys.clear();
    cache.bfloat16_values.clear();
    cache.qsa_block_keys.clear();
    cache.qsa_index_key_tail.clear();
    auto& counters = d->next->context->runtime_state();
    counters.dispatches += d->dispatches;
    ++counters.attention_blocks;
    ++counters.attention_device_rope_fusions;
    ++counters.kv_ring_appends;
    if (!d->original || d->original->capacity != d->next->capacity)
        ++counters.kv_ring_resizes;
    d->committed = true;
    return true;
#else
    return false;
#endif
}

#if NCNN_MOE_WITH_VULKAN
static std::shared_ptr<ncnn::Pipeline> qsa_pipeline(const std::shared_ptr<VulkanContext>& context,
                                                    const ncnn::Option& opt,
                                                    const char* shader,
                                                    size_t shader_size)
{
    if (auto cached = context->find_pipeline(shader, 0))
        return cached;
    auto binary = context->shader_binary(shader, shader_size, opt, 0);
    if (!binary || binary->empty())
        return {};
    auto pipeline = std::make_unique<ncnn::Pipeline>(context->device());
    pipeline->set_optimal_local_size_xyz(32, 1, 1);
    if (pipeline->create(binary->data(), binary->size() * sizeof(uint32_t), {}) != 0)
        return {};
    auto result = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = std::shared_ptr<VulkanContext>(context)](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        context.reset();
    });
    context->cache_pipeline(shader, 0, result);
    return result;
}

static bool qsa_shape(const WeightStore& weights, TensorHandle handle, uint32_t rows, uint32_t columns)
{
    return handle < weights.size() && weights.at(handle).shape == std::vector<uint32_t>{rows, columns};
}

static bool qsa_vector(const WeightStore& weights, TensorHandle handle, uint32_t columns)
{
    return handle < weights.size() && weights.at(handle).shape == std::vector<uint32_t>{columns};
}

static bool qsa_matrix(ncnn::VkMat& target, uint32_t columns, uint32_t rows, VulkanContext& context)
{
    // Shader offsets use uint32_t. Keep allocation and shader products checked
    // before recording any command; VkMat uses signed dimensions.
    if (columns == 0 || rows == 0 || columns > INT32_MAX || rows > INT32_MAX
        || static_cast<uint64_t>(columns) * rows > INT32_MAX)
        return false;
    target.create(static_cast<int>(columns), static_cast<int>(rows), sizeof(float), context.blob_allocator());
    return !target.empty();
}

#endif

std::shared_ptr<QsaAttention_vulkan> QsaAttention_vulkan::create(const WeightStore& weights,
                                                                 const AttentionBlockPlan& plan,
                                                                 uint32_t device_index,
                                                                 const VulkanRuntimePtr& runtime,
                                                                 uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    const uint64_t query_columns64 = static_cast<uint64_t>(plan.head_count) * plan.head_dimension;
    const uint64_t kv_columns64 = static_cast<uint64_t>(plan.kv_head_count) * plan.head_dimension;
    const uint64_t index_columns64 = static_cast<uint64_t>(plan.index_head_count + 1) * plan.index_head_dimension;
    const uint32_t rope_dimension = plan.rope_head_dimension == 0 ? plan.head_dimension : plan.rope_head_dimension;
    if (plan.kind != AttentionKind::Standard || !has_flag(plan.flags, AttentionBlockQsa)
        || !has_flag(plan.flags, AttentionBlockQueryKeyNorm)
        || !has_flag(plan.flags, AttentionBlockExternalResidual)
        || plan.sliding_window != 0 || plan.sinks != invalid_tensor_handle
        || plan.head_count == 0 || plan.kv_head_count == 0 || plan.head_count % plan.kv_head_count != 0
        || plan.head_dimension == 0 || plan.head_dimension > 256 || (plan.head_dimension & 1) != 0
        || plan.index_head_count == 0 || plan.index_head_dimension == 0 || plan.index_head_dimension > 512
        || plan.compression_ratio == 0 || plan.index_top_k == 0
        || rope_dimension == 0 || (rope_dimension & 1) != 0
        || rope_dimension > plan.head_dimension || rope_dimension > plan.index_head_dimension
        || query_columns64 > INT32_MAX || kv_columns64 > INT32_MAX || index_columns64 > INT32_MAX
        || plan.norm_epsilon <= 0.0f || plan.rope_theta <= 1.0f
        || (plan.kv_cache_dtype != DType::Float32 && plan.kv_cache_dtype != DType::BFloat16)
        || plan.query_weight >= weights.size() || weights.at(plan.query_weight).shape.size() != 2)
        return {};
    const uint32_t hidden = weights.at(plan.query_weight).shape[1];
    const uint32_t query_columns = static_cast<uint32_t>(query_columns64);
    const uint32_t kv_columns = static_cast<uint32_t>(kv_columns64);
    const uint32_t index_columns = static_cast<uint32_t>(index_columns64);
    if (hidden == 0 || !qsa_shape(weights, plan.query_weight, query_columns, hidden)
        || !qsa_shape(weights, plan.key_weight, kv_columns, hidden)
        || !qsa_shape(weights, plan.value_weight, kv_columns, hidden)
        || !qsa_shape(weights, plan.output_weight, hidden, query_columns)
        || !qsa_shape(weights, plan.qsa_query_key_weight, index_columns, hidden)
        || !qsa_vector(weights, plan.query_norm_weight, plan.head_dimension)
        || !qsa_vector(weights, plan.key_norm_weight, plan.head_dimension)
        || !qsa_vector(weights, plan.qsa_query_norm_weight, plan.index_head_dimension)
        || !qsa_vector(weights, plan.qsa_key_norm_weight, plan.index_head_dimension)
        || (has_flag(plan.flags, AttentionBlockOutputGate) && !qsa_shape(weights, plan.output_gate_weight, query_columns, hidden))
        || (plan.pre_attention_norm_weight != invalid_tensor_handle && !qsa_vector(weights, plan.pre_attention_norm_weight, hidden)))
        return {};
    auto result = std::shared_ptr<QsaAttention_vulkan>(new QsaAttention_vulkan);
    auto& impl = *result->d;
    impl.plan = plan;
    impl.hidden = hidden;
    const auto projection = [&](TensorHandle weight, TensorHandle bias) {
        return QnkLinear_vulkan::create(weights.at(weight), bias < weights.size() ? &weights.at(bias) : nullptr,
                                        device_index, runtime, optimization_flags);
    };
    impl.query = projection(plan.query_weight, plan.query_bias);
    impl.key = projection(plan.key_weight, plan.key_bias);
    impl.value = projection(plan.value_weight, plan.value_bias);
    impl.index = projection(plan.qsa_query_key_weight, invalid_tensor_handle);
    impl.output = projection(plan.output_weight, plan.output_bias);
    if (has_flag(plan.flags, AttentionBlockOutputGate))
        impl.gate = projection(plan.output_gate_weight, invalid_tensor_handle);
    if (!impl.query || !impl.key || !impl.value || !impl.index || !impl.output
        || (has_flag(plan.flags, AttentionBlockOutputGate) && !impl.gate))
        return {};
    impl.context = impl.query->vulkan_context();
    impl.opt = impl.query->option();
    if (!impl.context || vulkan_activation_storage_variant(impl.opt) != 0)
        return {};
    if (plan.pre_attention_norm_weight != invalid_tensor_handle)
    {
        impl.norm = RmsNorm_vulkan::create(weights.at(plan.pre_attention_norm_weight), plan.norm_epsilon,
                                           plan.norm_weight_offset, impl.context, impl.opt);
        if (!impl.norm)
            return {};
    }
    std::array<ncnn::Mat, 5> upload;
    if (!prepare_float_tensor_upload(weights.at(plan.query_norm_weight), upload[0])
        || !prepare_float_tensor_upload(weights.at(plan.key_norm_weight), upload[1])
        || !prepare_float_tensor_upload(weights.at(plan.qsa_query_norm_weight), upload[2])
        || !prepare_float_tensor_upload(weights.at(plan.qsa_key_norm_weight), upload[3]))
        return {};
    upload[4].create(static_cast<int>(rope_dimension / 2), sizeof(float));
    if (upload[4].empty())
        return {};
    float* frequencies = upload[4];
    const float factor = plan.rope_scaling_factor;
    const float half = static_cast<float>(rope_dimension / 2);
    float low = 0.0f;
    float high = 0.0f;
    if (factor > 1.0f)
    {
        if (plan.initial_context_length == 0 || plan.rope_ntk_alpha <= 0.0f || plan.rope_ntk_beta <= 0.0f)
            return {};
        impl.concentration = 0.1f * std::log(factor) + 1.0f;
        low = half * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_beta * 2.0f * std::numbers::pi_v<float>)) / std::log(plan.rope_theta);
        high = half * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_alpha * 2.0f * std::numbers::pi_v<float>)) / std::log(plan.rope_theta);
        if (low == high)
            return {};
    }
    for (uint32_t index = 0; index < rope_dimension / 2; ++index)
    {
        const float frequency = std::pow(plan.rope_theta, static_cast<float>(2 * index) / static_cast<float>(rope_dimension));
        float inverse = 1.0f / frequency;
        if (factor > 1.0f)
        {
            const float ramp = std::clamp((static_cast<float>(index) - low) / (high - low), 0.0f, 1.0f);
            inverse = (1.0f / (factor * frequency)) * ramp + inverse * (1.0f - ramp);
        }
        frequencies[index] = inverse;
    }
    const std::lock_guard<std::mutex> lock(impl.context->command_mutex());
    impl.norm_rope_pipeline = qsa_pipeline(impl.context, impl.opt, qsa_norm_rope_shader, sizeof(qsa_norm_rope_shader) - 1);
    impl.cache_write_pipeline = qsa_pipeline(impl.context, impl.opt, qsa_cache_write_shader, sizeof(qsa_cache_write_shader) - 1);
    impl.pool_pipeline = qsa_pipeline(impl.context, impl.opt, qsa_pool_shader, sizeof(qsa_pool_shader) - 1);
    impl.scores_pipeline = qsa_pipeline(impl.context, impl.opt, qsa_scores_shader, sizeof(qsa_scores_shader) - 1);
    impl.select_pipeline = qsa_pipeline(impl.context, impl.opt, qsa_select_shader, sizeof(qsa_select_shader) - 1);
    impl.attention_pipeline = qsa_pipeline(impl.context, impl.opt, qsa_attention_shader, sizeof(qsa_attention_shader) - 1);
    impl.copy_pipeline = qsa_pipeline(impl.context, impl.opt, qsa_copy_shader, sizeof(qsa_copy_shader) - 1);
    if (!impl.norm_rope_pipeline || !impl.cache_write_pipeline || !impl.pool_pipeline || !impl.scores_pipeline
        || !impl.select_pipeline || !impl.attention_pipeline || !impl.copy_pipeline)
        return {};
    ncnn::VkWeightStagingAllocator staging(impl.context->device());
    ncnn::Option upload_opt = impl.opt;
    upload_opt.blob_vkallocator = impl.context->blob_allocator();
    upload_opt.workspace_vkallocator = impl.context->blob_allocator();
    upload_opt.staging_vkallocator = &staging;
    ncnn::VkTransfer command(impl.context->device());
    std::array<ncnn::VkMat*, 5> destination = {&impl.query_norm, &impl.key_norm, &impl.index_query_norm, &impl.index_key_norm, &impl.frequency};
    for (size_t index = 0; index < upload.size(); ++index)
        command.record_upload(upload[index], *destination[index], upload_opt);
    if (command.submit_and_wait() != 0 || std::any_of(destination.begin(), destination.end(), [](const ncnn::VkMat* value) { return value->empty(); }))
        return {};
    impl.weights_bytes = impl.query->device_bytes() + impl.key->device_bytes() + impl.value->device_bytes()
                         + impl.index->device_bytes() + impl.output->device_bytes() + (impl.gate ? impl.gate->device_bytes() : 0);
    for (const ncnn::VkMat* value : destination)
        impl.weights_bytes += static_cast<uint64_t>(value->total()) * value->elemsize;
    if (impl.norm)
        impl.weights_bytes += static_cast<uint64_t>(hidden) * sizeof(float);
    return result;
#else
    (void)weights;
    (void)plan;
    (void)device_index;
    (void)runtime;
    (void)optimization_flags;
    return {};
#endif
}

uint64_t QsaAttention_vulkan::device_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return d->weights_bytes;
#else
    return 0;
#endif
}

uint64_t QsaAttention_vulkan::initial_state_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    constexpr uint64_t capacity = 16;
    const uint64_t columns = static_cast<uint64_t>(d->plan.kv_head_count) * d->plan.head_dimension;
    const uint64_t element_size = d->plan.kv_cache_dtype == DType::BFloat16 ? sizeof(uint16_t) : sizeof(float);
    return capacity * columns * element_size * 2
           + (capacity + std::max<uint64_t>(1, capacity / d->plan.compression_ratio))
                 * d->plan.index_head_dimension * sizeof(float);
#else
    return 0;
#endif
}

#if NCNN_MOE_WITH_VULKAN
const std::shared_ptr<VulkanContext>& QsaAttention_vulkan::vulkan_context() const noexcept
{
    return d->context;
}

const ncnn::Option& QsaAttention_vulkan::option() const noexcept
{
    return d->opt;
}

bool QsaAttention_vulkan::record_forward(const ncnn::VkMat& input,
                                         LayerCache& cache,
                                         uint64_t position_offset,
                                         ncnn::VkMat& output,
                                         ncnn::VkCompute& command,
                                         QsaAttentionWork_vulkan& work) const
{
    const auto& impl = *d;
    const auto& plan = impl.plan;
    if (!impl.context || work.d->cache || input.empty() || input.dims != 2 || input.elemsize != sizeof(float)
        || input.elempack != 1 || input.w != static_cast<int>(impl.hidden) || input.h <= 0
        || cache.transaction.active || cache.vulkan_attention_state_unknown
        || position_offset > UINT32_MAX || static_cast<uint64_t>(input.h) - 1 > UINT32_MAX - position_offset
        || cache.token_count > UINT32_MAX - static_cast<uint32_t>(input.h)
        || (cache.token_count != 0 && (!cache.qsa_device_state || cache.start_position + cache.token_count != position_offset)))
        return false;
    const uint32_t previous = static_cast<uint32_t>(cache.token_count);
    const uint32_t rows = static_cast<uint32_t>(input.h);
    const uint32_t total = previous + rows;
    const uint32_t columns = plan.kv_head_count * plan.head_dimension;
    const uint32_t query_columns = plan.head_count * plan.head_dimension;
    const uint32_t index_columns = (plan.index_head_count + 1) * plan.index_head_dimension;
    const uint32_t cache_columns = plan.kv_cache_dtype == DType::BFloat16 ? columns / 2 : columns;
    const uint32_t rope_dimension = plan.rope_head_dimension == 0 ? plan.head_dimension : plan.rope_head_dimension;
    auto original = cache.qsa_device_state;
    if (original && (original->context != impl.context || original->token_count != previous || original->columns != columns || original->index_dimension != plan.index_head_dimension || original->compression_ratio != plan.compression_ratio || original->dtype != plan.kv_cache_dtype || original->start_position != cache.start_position))
        return false;
    // Allocate persistent destinations and nonprojection scratch first. Once
    // recording begins, a false return requires the caller to discard the
    // command. Published cache metadata is only changed by work.commit().
    auto next = original ? std::make_shared<QsaCache_vulkan>(*original) : std::make_shared<QsaCache_vulkan>();
    next->context = impl.context;
    next->columns = columns;
    next->index_dimension = plan.index_head_dimension;
    next->compression_ratio = plan.compression_ratio;
    next->dtype = plan.kv_cache_dtype;
    next->start_position = previous == 0 ? position_offset : cache.start_position;
    next->token_count = total;
    const bool grow = !original || total > original->capacity;
    if (grow)
    {
        uint64_t capacity = original ? original->capacity : 16;
        while (capacity < total)
            capacity *= 2;
        if (capacity > INT32_MAX)
            return false;
        next->capacity = static_cast<uint32_t>(capacity);
        if (!qsa_matrix(next->keys, cache_columns, next->capacity, *impl.context)
            || !qsa_matrix(next->values, cache_columns, next->capacity, *impl.context)
            || !qsa_matrix(next->index_raw, plan.index_head_dimension, next->capacity, *impl.context)
            || !qsa_matrix(next->pooled_keys, plan.index_head_dimension, std::max(1u, next->capacity / plan.compression_ratio), *impl.context))
            return false;
    }
    const uint32_t blocks = total / plan.compression_ratio;
    ncnn::VkMat normalized = input;
    ncnn::VkMat query;
    ncnn::VkMat key;
    ncnn::VkMat value;
    ncnn::VkMat index;
    ncnn::VkMat gate;
    ncnn::VkMat rotated_query;
    ncnn::VkMat rotated_key;
    ncnn::VkMat rotated_index;
    ncnn::VkMat scores;
    ncnn::VkMat cutoff;
    ncnn::VkMat attended;
    if (!qsa_matrix(rotated_query, query_columns, rows, *impl.context)
        || !qsa_matrix(rotated_key, columns, rows, *impl.context)
        || !qsa_matrix(rotated_index, plan.index_head_count * plan.index_head_dimension, rows, *impl.context)
        || !qsa_matrix(scores, std::max(1u, blocks), rows, *impl.context)
        || !qsa_matrix(cutoff, 2, rows, *impl.context)
        || !qsa_matrix(attended, query_columns, rows, *impl.context))
        return false;
    // The recorder owns all referenced allocations until the caller's wait.
    auto& ticket = *work.d;
    ticket.cache = &cache;
    ticket.original = original;
    ticket.next = next;
    ticket.previous_tokens = previous;
    auto& storage = ticket.storage;
    std::vector<ncnn::VkMat> bindings;
    bindings.reserve(7);
    std::vector<unsigned char> readonly;
    readonly.reserve(7);
    std::vector<ncnn::vk_constant_type> constants;
    constants.reserve(10);
    const auto uints = [&](std::initializer_list<uint32_t> values) -> std::vector<ncnn::vk_constant_type>& {
        constants.resize(values.size());
        size_t index = 0;
        for (uint32_t value : values)
            constants[index++].u32 = value;
        return constants;
    };
    const auto record = [&](const std::shared_ptr<ncnn::Pipeline>& pipeline,
                            std::initializer_list<ncnn::VkMat> binding_values,
                            std::initializer_list<unsigned char> readonly_values,
                            const std::vector<ncnn::vk_constant_type>& push_constants,
                            uint32_t groups_x, uint32_t groups_y = 1) {
        bindings.assign(binding_values.begin(), binding_values.end());
        readonly.assign(readonly_values.begin(), readonly_values.end());
        ncnn::VkMat dispatcher;
        // Flatten work into X/Z so full context blocks and large prefill chunks
        // also fit Vulkan's guaranteed 65535 workgroups per dimension.
        const uint64_t groups = static_cast<uint64_t>(groups_x) * groups_y;
        const uint32_t width = static_cast<uint32_t>(std::min<uint64_t>(groups, 65535));
        const uint32_t depth = static_cast<uint32_t>((groups + width - 1) / width);
        dispatcher.w = static_cast<int>(width * pipeline->local_size_x());
        dispatcher.h = 1;
        dispatcher.c = static_cast<int>(depth * pipeline->local_size_z());
        command.record_pipeline_readonly(pipeline.get(), bindings, readonly, push_constants, dispatcher);
    };
    if (grow && original && previous != 0)
    {
        const auto copy = [&](const ncnn::VkMat& source, const ncnn::VkMat& target, uint32_t words) {
            if (words != 0)
            {
                record(impl.copy_pipeline, {source, target}, {1, 0},
                       uints({words, impl.copy_pipeline->local_size_x()}), (words + 31) / 32);
                ++ticket.dispatches;
            }
        };
        copy(original->keys, next->keys, previous * cache_columns);
        copy(original->values, next->values, previous * cache_columns);
        copy(original->index_raw, next->index_raw, previous * plan.index_head_dimension);
        copy(original->pooled_keys, next->pooled_keys, (previous / plan.compression_ratio) * plan.index_head_dimension);
    }
    if ((impl.norm && !impl.norm->record(input, normalized, command))
        || !impl.query->record_forward(normalized, query, command, storage)
        || !impl.key->record_forward(normalized, key, command, storage)
        || !impl.value->record_forward(normalized, value, command, storage)
        || !impl.index->record_forward(normalized, index, command, storage)
        || (impl.gate && !impl.gate->record_forward(normalized, gate, command, storage)))
        return false;
    ticket.dispatches += 4 + (impl.norm ? 1 : 0) + (impl.gate ? 1 : 0);
    if (!impl.gate)
        gate = query;
    const auto rotate = [&](const ncnn::VkMat& source, const ncnn::VkMat& norm,
                            const ncnn::VkMat& target, uint32_t heads, uint32_t dimension, uint32_t source_columns) {
        auto& constants = uints({rows, heads, dimension, source_columns, rope_dimension, static_cast<uint32_t>(position_offset), 0, 0, 0});
        constants[6].f = plan.norm_epsilon;
        constants[7].f = plan.norm_weight_offset;
        constants[8].f = impl.concentration;
        record(impl.norm_rope_pipeline, {source, norm, impl.frequency, target}, {1, 1, 1, 0}, constants, rows * heads);
        ++ticket.dispatches;
    };
    rotate(query, impl.query_norm, rotated_query, plan.head_count, plan.head_dimension, query_columns);
    rotate(key, impl.key_norm, rotated_key, plan.kv_head_count, plan.head_dimension, columns);
    rotate(index, impl.index_query_norm, rotated_index, plan.index_head_count, plan.index_head_dimension, index_columns);
    record(impl.cache_write_pipeline, {rotated_key, value, index, next->keys, next->values, next->index_raw},
           {1, 1, 1, 0, 0, 0},
           uints({rows, columns, index_columns, plan.index_head_dimension,
                  plan.index_head_count * plan.index_head_dimension, previous, plan.kv_cache_dtype == DType::BFloat16 ? 1u : 0u,
                  impl.cache_write_pipeline->local_size_x()}),
           (std::max(rows * cache_columns, rows * plan.index_head_dimension) + 31) / 32);
    ++ticket.dispatches;
    const uint32_t first_block = previous / plan.compression_ratio;
    if (blocks > first_block)
    {
        auto& constants = uints({first_block, blocks - first_block, plan.index_head_dimension, plan.compression_ratio,
                                 rope_dimension, static_cast<uint32_t>(next->start_position), 0, 0, 0});
        constants[6].f = plan.norm_epsilon;
        constants[7].f = plan.norm_weight_offset;
        constants[8].f = impl.concentration;
        record(impl.pool_pipeline, {next->index_raw, impl.index_key_norm, impl.frequency, next->pooled_keys},
               {1, 1, 1, 0}, constants, blocks - first_block);
        ++ticket.dispatches;
    }
    if (blocks != 0)
    {
        record(impl.scores_pipeline, {rotated_index, next->pooled_keys, scores}, {1, 1, 0},
               uints({rows, blocks, plan.index_head_count, plan.index_head_dimension}), blocks, rows);
        ++ticket.dispatches;
    }
    record(impl.select_pipeline, {scores, cutoff}, {1, 0},
           uints({rows, blocks, previous, plan.compression_ratio, plan.index_top_k}), rows);
    record(impl.attention_pipeline, {rotated_query, next->keys, next->values, cutoff, scores, gate, attended},
           {1, 1, 1, 1, 1, 1, 0},
           uints({rows, plan.head_count, plan.kv_head_count, plan.head_dimension, previous, plan.compression_ratio,
                  blocks, plan.index_top_k, plan.kv_cache_dtype == DType::BFloat16 ? 1u : 0u, impl.gate ? 1u : 0u}),
           plan.head_count, rows);
    ticket.dispatches += 2;
    if (!impl.output->record_forward(attended, output, command, storage))
        return false;
    ++ticket.dispatches;
    storage.insert(storage.end(), {input, normalized, query, key, value, index, gate, rotated_query, rotated_key,
                                   rotated_index, scores, cutoff, attended, output});
    return true;
}
#endif

} // namespace moe
} // namespace ncnn
