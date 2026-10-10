#include "latentattention_vulkan.h"
#include "linear.h"
#include "vulkancontext.h"

#include "kernels/attention.h"
#include "kernels/statecache.h"
#include "kernels/latentattention.h"
#include "graph/compiledoperator.h"
#include "storage/weightstore.h"
#include "kernels/ops.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/latent_attention_query.comp.hex.h"
#include "kernels/vulkan/latent_attention_cache_append.comp.hex.h"
#include "kernels/vulkan/latent_attention_core.comp.hex.h"
#include "kernels/vulkan/latent_attention_score.comp.hex.h"
#include "kernels/vulkan/latent_attention_scored_core.comp.hex.h"
#include "kernels/vulkan/latent_attention_transform.comp.hex.h"
#include "kernels/vulkan/latent_attention_compress.comp.hex.h"
#include "kernels/vulkan/latent_attention_index_score.comp.hex.h"
#include "kernels/vulkan/latent_attention_index_topk.comp.hex.h"
#include "kernels/vulkan/latent_attention_index_sort.comp.hex.h"
#include "kernels/vulkan/latent_attention_index_merge.comp.hex.h"

#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static bool create_latent_pipeline(const std::shared_ptr<VulkanContext>& context,
                                   const ncnn::Option& opt,
                                   const char* shader,
                                   int shader_size,
                                   std::shared_ptr<ncnn::Pipeline>& destination)
{
    const auto spirv = context->shader_binary(shader, shader_size, opt, 0);
    if (!spirv || spirv->empty())
        return false;
    destination = context->find_pipeline(shader, 0);
    if (destination)
        return true;
    std::unique_ptr<ncnn::Pipeline> pipeline(new ncnn::Pipeline(context->device()));
    pipeline->set_local_size_xyz(128, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    if (pipeline->create(spirv->data(), spirv->size() * sizeof(uint32_t), specializations) != 0)
        return false;
    destination = std::shared_ptr<ncnn::Pipeline>(pipeline.release(), [context = context](ncnn::Pipeline* value) mutable {
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            delete value;
        }
        // Weak cache entries retain the deleter control block after disposal.
        context.reset();
    });
    context->cache_pipeline(shader, 0, destination);
    return true;
}

static void prepare_latent_rope(const AttentionBlockPlan& plan,
                                uint64_t position,
                                std::vector<float>& coefficients)
{
    const uint32_t dimension = plan.rope_head_dimension;
    coefficients.assign(std::max(2u, dimension), 0.0f);
    if (dimension == 0)
        return;
    const bool yarn = plan.compression_ratio != 0;
    const float base = yarn ? plan.compressed_rope_theta : plan.rope_theta;
    int correction_low = 0;
    int correction_high = 0;
    if (yarn)
    {
        const float denominator = 2.0f * std::log(base);
        const float rotations_low = static_cast<float>(dimension) * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_beta * 2.0f * 3.14159265358979323846f)) / denominator;
        const float rotations_high = static_cast<float>(dimension) * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_alpha * 2.0f * 3.14159265358979323846f)) / denominator;
        correction_low = std::max(0, static_cast<int>(std::floor(rotations_low)));
        correction_high = std::min(static_cast<int>(dimension) - 1, static_cast<int>(std::ceil(rotations_high)));
    }
    for (uint32_t pair = 0; pair < dimension / 2; ++pair)
    {
        float frequency = 1.0f / std::pow(base, static_cast<float>(pair * 2) / static_cast<float>(dimension));
        if (yarn)
        {
            const float denominator = correction_low == correction_high ? 0.001f : static_cast<float>(correction_high - correction_low);
            const float ramp = std::clamp((static_cast<float>(pair) - static_cast<float>(correction_low)) / denominator, 0.0f, 1.0f);
            const float smooth = 1.0f - ramp;
            frequency = frequency / plan.rope_scaling_factor * (1.0f - smooth) + frequency * smooth;
        }
        const float angle = static_cast<float>(position) * frequency;
        coefficients[pair * 2] = std::cos(angle);
        coefficients[pair * 2 + 1] = std::sin(angle);
    }
}

static bool allocate_latent_buffer(ncnn::VkMat& destination,
                                   uint64_t count,
                                   const std::shared_ptr<VulkanContext>& context)
{
    if (count == 0 || count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;
    // Reserved allocator blocks can satisfy this request even when the driver
    // reports no free heap budget. Let the allocator try its reusable ranges.
    destination.create(static_cast<int>(count), sizeof(float), context->blob_allocator());
    return !destination.empty();
}

static bool record_latent_parallel_topk(const std::shared_ptr<ncnn::Pipeline>& sort_pipeline,
                                        const std::shared_ptr<ncnn::Pipeline>& merge_pipeline,
                                        const ncnn::VkMat& scores,
                                        uint32_t count,
                                        uint32_t top_k,
                                        ncnn::VkMat& indices,
                                        const std::shared_ptr<VulkanContext>& context,
                                        std::vector<ncnn::VkMat>& storage,
                                        ncnn::VkCompute& cmd)
{
    if (!sort_pipeline || !merge_pipeline || top_k == 0 || top_k > 1024 || count < top_k)
        return false;
    uint32_t runs = (count + 1023u) / 1024u;
    ncnn::VkMat temporary[2];
    // Allocate all merge scratch before recording. Failure can still use the
    // GPU heap without leaving a partial sorting operation in the command.
    if (runs > 1
        && (!allocate_latent_buffer(temporary[0], static_cast<uint64_t>(runs) * top_k, context)
            || !allocate_latent_buffer(temporary[1], static_cast<uint64_t>((runs + 1) / 2) * top_k, context)))
        return false;
    ncnn::VkMat sorted = runs == 1 ? indices : temporary[0];
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].u32 = count;
    constants[1].u32 = top_k;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(std::min(runs, 65535u) * 128u);
    dispatcher.h = static_cast<int>((runs + 65534u) / 65535u);
    dispatcher.c = 1;
    cmd.record_pipeline(sort_pipeline.get(), {scores, sorted}, constants, dispatcher);
    uint32_t span = 1024;
    uint32_t next_buffer = 1;
    while (runs > 1)
    {
        ncnn::VkMat merged = runs <= 2 ? indices : temporary[next_buffer];
        constants.resize(4);
        constants[0].u32 = count;
        constants[1].u32 = top_k;
        constants[2].u32 = runs;
        constants[3].u32 = span;
        const uint32_t candidates = runs * top_k;
        constexpr uint32_t row_width = 65535u * 128u;
        dispatcher.w = static_cast<int>(std::min(candidates, row_width));
        dispatcher.h = static_cast<int>((candidates + row_width - 1u) / row_width);
        cmd.record_pipeline(merge_pipeline.get(), {scores, sorted, merged}, constants, dispatcher);
        sorted = merged;
        runs = (runs + 1u) / 2u;
        span *= 2u;
        next_buffer ^= 1u;
    }
    if (!temporary[0].empty())
        storage.push_back(std::move(temporary[0]));
    if (!temporary[1].empty())
        storage.push_back(std::move(temporary[1]));
    return true;
}

static bool record_latent_copy(const ncnn::Pipeline* pipeline,
                               const ncnn::VkMat& source,
                               ncnn::VkMat& destination,
                               uint64_t count,
                               uint64_t destination_offset,
                               ncnn::VkCompute& cmd)
{
    if (count == 0)
        return true;
    if (!pipeline || source.empty() || destination.empty()
        || count > std::numeric_limits<uint32_t>::max()
        || destination_offset > std::numeric_limits<uint32_t>::max())
        return false;
    const std::vector<ncnn::VkMat> bindings = {source, destination};
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].u32 = static_cast<uint32_t>(count);
    constants[1].u32 = static_cast<uint32_t>(destination_offset);
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(count);
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}
#endif

class LatentAttention_vulkan::Implementation
{
public:
    AttentionBlockPlan plan;
    std::shared_ptr<Float8Linear_vulkan> output_a;
    std::shared_ptr<Float8Linear_vulkan> output_b;
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    ncnn::Option opt;
    std::vector<float> host_sinks;
    ncnn::VkMat sinks;
    std::shared_ptr<ncnn::Pipeline> query_pipeline;
    std::shared_ptr<ncnn::Pipeline> append_pipeline;
    std::shared_ptr<ncnn::Pipeline> attention_pipeline;
    std::shared_ptr<ncnn::Pipeline> score_pipeline;
    std::shared_ptr<ncnn::Pipeline> scored_attention_pipeline;
    std::shared_ptr<ncnn::Pipeline> transform_pipeline;
    std::shared_ptr<ncnn::Pipeline> compressor_pipeline;
    std::shared_ptr<ncnn::Pipeline> index_score_pipeline;
    std::shared_ptr<ncnn::Pipeline> index_topk_pipeline;
    std::shared_ptr<ncnn::Pipeline> index_sort_pipeline;
    std::shared_ptr<ncnn::Pipeline> index_merge_pipeline;
    std::shared_ptr<Float8Linear_vulkan> query_a;
    std::shared_ptr<Float8Linear_vulkan> query_b;
    std::shared_ptr<Float8Linear_vulkan> key_value;
    CompiledOperator index_query;
    CompiledOperator index_weights;
    std::shared_ptr<Bfloat16Linear_vulkan> compressor_value;
    std::shared_ptr<Bfloat16Linear_vulkan> compressor_gate;
    std::shared_ptr<Bfloat16Linear_vulkan> index_compressor_value;
    std::shared_ptr<Bfloat16Linear_vulkan> index_compressor_gate;
    ncnn::VkMat key_norm;
    ncnn::VkMat compressor_norm;
    ncnn::VkMat compressor_position;
    ncnn::VkMat index_compressor_norm;
    ncnn::VkMat index_compressor_position;
    bool prepared = false;
#endif
};

uint64_t LatentCache_vulkan::allocated_bytes() const noexcept
{
#if NCNN_MOE_WITH_VULKAN
    return static_cast<uint64_t>(window.total()) * window.elemsize
           + static_cast<uint64_t>(compressed.total()) * compressed.elemsize
           + static_cast<uint64_t>(compressor_state.total()) * compressor_state.elemsize
           + static_cast<uint64_t>(index_compressor_state.total()) * index_compressor_state.elemsize
           + static_cast<uint64_t>(index_compressed.total()) * index_compressed.elemsize;
#else
    return 0;
#endif
}

LatentAttention_vulkan::LatentAttention_vulkan()
    : d(new Implementation)
{
}

LatentAttention_vulkan::~LatentAttention_vulkan() = default;

std::shared_ptr<LatentAttention_vulkan> LatentAttention_vulkan::create(const AttentionBlockPlan& plan,
                                                                       const TensorData& sinks,
                                                                       std::shared_ptr<Float8Linear_vulkan> output_a,
                                                                       std::shared_ptr<Float8Linear_vulkan> output_b)
{
#if NCNN_MOE_WITH_VULKAN
    const uint64_t query_columns = static_cast<uint64_t>(plan.head_count) * plan.head_dimension;
    if (plan.kind != AttentionKind::MultiHeadLatent || plan.head_dimension == 0 || plan.head_dimension > 512
        || plan.head_count == 0 || plan.sliding_window == 0 || plan.norm_epsilon <= 0.0f
        || plan.rope_head_dimension > plan.head_dimension || plan.rope_head_dimension % 2 != 0
        || query_columns > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || sinks.dtype != DType::Float32 || sinks.float32_values().size() != plan.head_count
        || !output_a || !output_b || !output_a->vulkan_context()
        || output_a->vulkan_context() != output_b->vulkan_context()
        || output_a->input_columns() != query_columns
        || output_a->output_columns() != output_b->input_columns())
        return {};
    auto result = std::shared_ptr<LatentAttention_vulkan>(new LatentAttention_vulkan);
    Implementation& implementation = *result->d;
    implementation.plan = plan;
    implementation.output_a = std::move(output_a);
    implementation.output_b = std::move(output_b);
    implementation.context = implementation.output_a->vulkan_context();
    implementation.opt = implementation.output_a->option();
    implementation.opt.use_packing_layout = false;
    implementation.opt.use_fp16_packed = false;
    implementation.opt.use_fp16_storage = false;
    implementation.opt.use_fp16_arithmetic = false;
    implementation.opt.use_bf16_packed = false;
    implementation.opt.use_bf16_storage = false;
    implementation.opt.blob_vkallocator = implementation.context->blob_allocator();
    implementation.opt.workspace_vkallocator = implementation.context->blob_allocator();
    implementation.host_sinks.assign(sinks.float32_values().begin(), sinks.float32_values().end());
    const std::lock_guard<std::mutex> lock(implementation.context->command_mutex());
    if (!create_latent_pipeline(implementation.context, implementation.opt,
                                latent_attention_query_shader, static_cast<int>(sizeof(latent_attention_query_shader) - 1), implementation.query_pipeline)
        || !create_latent_pipeline(implementation.context, implementation.opt,
                                   latent_attention_cache_append_shader, static_cast<int>(sizeof(latent_attention_cache_append_shader) - 1), implementation.append_pipeline)
        || !create_latent_pipeline(implementation.context, implementation.opt,
                                   latent_attention_core_shader, static_cast<int>(sizeof(latent_attention_core_shader) - 1), implementation.attention_pipeline))
        return {};
    // Parallel scoring is optional; preparation failures retain the GPU core.
    (void)create_latent_pipeline(implementation.context, implementation.opt,
                                 latent_attention_score_shader, static_cast<int>(sizeof(latent_attention_score_shader) - 1), implementation.score_pipeline);
    (void)create_latent_pipeline(implementation.context, implementation.opt,
                                 latent_attention_scored_core_shader, static_cast<int>(sizeof(latent_attention_scored_core_shader) - 1), implementation.scored_attention_pipeline);
    return result;
#else
    (void)plan;
    (void)sinks;
    (void)output_a;
    (void)output_b;
    return {};
#endif
}

bool LatentAttention_vulkan::forward_batch(std::span<const uint64_t> positions,
                                           std::span<LayerCache* const> caches,
                                           std::span<const LatentAttentionRowContext> contexts,
                                           const ActivationBuffer& query,
                                           const DeviceTensor_vulkan* device_query,
                                           ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    const AttentionBlockPlan& plan = implementation.plan;
    const size_t rows = positions.size();
    const uint32_t query_columns = plan.head_count * plan.head_dimension;
    if (rows == 0 || rows != caches.size() || rows != contexts.size()
        || rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        return false;
    const ncnn::VkMat* retained_query = device_query ? device_query->value() : nullptr;
    if (retained_query)
    {
        if (device_query->vulkan_context() != implementation.context
            || retained_query->dims != 2 || retained_query->w != static_cast<int>(query_columns)
            || retained_query->h != static_cast<int>(rows) || retained_query->elempack != 1
            || retained_query->elemsize != sizeof(float))
            return false;
    }
    else if (query.rows() != rows || query.columns() != query_columns || query.dtype() != DType::Float32)
        return false;
    const uint64_t window_elements = static_cast<uint64_t>(plan.sliding_window) * plan.head_dimension;
    if (window_elements > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;
    for (size_t row = 0; row < rows; ++row)
    {
        const LayerCache* cache = caches[row];
        const LatentAttentionRowContext& context = contexts[row];
        if (!cache || positions[row] == std::numeric_limits<uint64_t>::max()
            || cache->latent_token_count != positions[row] + 1
            || cache->latent_window.size() != window_elements
            || cache->latent_compressed.size() % plan.head_dimension != 0
            || cache->latent_compressed.size() / plan.head_dimension != context.compressed_count
            || context.window_count == 0 || context.window_count > plan.sliding_window
            || context.window_begin > positions[row]
            || positions[row] + 1 - context.window_begin != context.window_count)
            return false;
        for (size_t previous = 0; previous < row; ++previous)
            if (caches[previous] == cache)
                return false;
        if (context.selected_compressed_indices)
        {
            if (context.compressed_indices.size() > std::numeric_limits<uint32_t>::max())
                return false;
            for (uint32_t index : context.compressed_indices)
                if (index >= context.compressed_count)
                    return false;
        }
    }
    struct RowWork
    {
        std::shared_ptr<LatentCache_vulkan> state;
        ncnn::VkMat window_staging;
        ncnn::VkMat compressed_staging;
        ncnn::VkMat index_staging;
        ncnn::VkMat rope_staging;
        ncnn::VkMat uploaded_window;
        ncnn::VkMat uploaded_compressed;
        ncnn::VkMat indices;
        ncnn::VkMat rope;
        ncnn::VkMat previous_compressed;
        std::vector<float> coefficients;
    };
    std::vector<RowWork> work(rows);
    VulkanRuntimeState& runtime_state = implementation.context->runtime_state();
    struct MirrorGuard
    {
        std::span<LayerCache* const> caches;
        VulkanRuntimeState& runtime_state;
        bool committed = false;
        ~MirrorGuard()
        {
            if (!committed)
            {
                runtime_state.attention_cpu_fallbacks += caches.size();
                for (LayerCache* cache : caches)
                {
                    cache->latent_device_state.reset();
                    cache->device_allocated_size = 0;
                }
            }
        }
    } mirror_guard{caches, runtime_state};
    VulkanTransferLease transfer_lease = implementation.context->acquire_transfer_slot();
    VulkanTransferSlot& transfer_slot = transfer_lease.slot();
    const std::lock_guard<std::mutex> lock(implementation.context->command_mutex());
    ncnn::VkCompute& cmd = *transfer_slot.command;
    if (transfer_slot.command_used)
    {
        if (cmd.reset() != 0)
            return false;
        ++runtime_state.command_buffer_reuses;
    }
    transfer_slot.command_used = true;
    uint64_t uploads = 0;
    ncnn::VkMat sink_staging;
    ncnn::VkMat next_sinks = implementation.sinks;
    if (next_sinks.empty())
    {
        if (!fill_staging_values(implementation.host_sinks.data(), implementation.host_sinks.size(), sizeof(float), sink_staging, transfer_slot.staging_allocator)
            || !record_mapped_upload(sink_staging, next_sinks, cmd, implementation.opt))
            return false;
        ++uploads;
    }
    ncnn::VkMat raw_query;
    if (retained_query)
    {
        device_query->prepare_read();
        raw_query = *retained_query;
    }
    else
    {
        if (!fill_staging_upload(query, transfer_slot.upload, transfer_slot.staging_allocator)
            || !record_mapped_upload(transfer_slot.upload, raw_query, cmd, implementation.opt))
            return false;
        ++uploads;
    }
    ncnn::VkMat normalized_query;
    ncnn::VkMat attention;
    normalized_query.create(static_cast<int>(query_columns), static_cast<int>(rows), sizeof(float), implementation.context->blob_allocator());
    attention.create(static_cast<int>(query_columns), static_cast<int>(rows), sizeof(float), implementation.context->blob_allocator());
    if (normalized_query.empty() || attention.empty())
        return false;
    for (size_t row = 0; row < rows; ++row)
    {
        RowWork& current = work[row];
        LayerCache& cache = *caches[row];
        const LatentAttentionRowContext& context = contexts[row];
        const bool shared_state = cache.latent_device_state && cache.latent_device_state.use_count() != 1;
        std::shared_ptr<LatentCache_vulkan> state = cache.latent_device_state;
        const bool refresh = shared_state || !state || state->context != implementation.context
                             || state->dimension != plan.head_dimension || state->window_capacity != plan.sliding_window
                             || state->token_count != positions[row] || state->compressed_count > context.compressed_count;
        if (refresh)
        {
            state = std::make_shared<LatentCache_vulkan>();
            state->context = implementation.context;
            state->dimension = plan.head_dimension;
            state->window_capacity = plan.sliding_window;
            if (!allocate_latent_buffer(state->window, window_elements, implementation.context))
                return false;
        }
        current.state = state;
        cache.latent_device_state = state;
        // Before the ring fills, only slots 0 through position can be read.
        const uint64_t window_count = refresh
                                          ? std::min<uint64_t>(positions[row] + 1, plan.sliding_window) * plan.head_dimension
                                          : plan.head_dimension;
        const uint64_t window_offset = refresh ? 0 : positions[row] % plan.sliding_window * plan.head_dimension;
        if (!fill_staging_values(cache.latent_window.data() + window_offset, static_cast<size_t>(window_count), sizeof(float), current.window_staging, transfer_slot.staging_allocator)
            || !record_mapped_upload(current.window_staging, current.uploaded_window, cmd, implementation.opt)
            || !record_latent_copy(implementation.append_pipeline.get(), current.uploaded_window, state->window, window_count, window_offset, cmd))
            return false;
        ++uploads;
        if (context.compressed_count > state->compressed_capacity)
        {
            const uint64_t doubled = std::max<uint64_t>(32, static_cast<uint64_t>(state->compressed_capacity) * 2);
            const uint64_t capacity = std::max<uint64_t>(context.compressed_count, doubled);
            if (capacity > std::numeric_limits<uint32_t>::max())
                return false;
            ncnn::VkMat next_compressed;
            if (!allocate_latent_buffer(next_compressed, capacity * plan.head_dimension, implementation.context))
                return false;
            if (!refresh && state->compressed_count != 0)
            {
                current.previous_compressed = state->compressed;
                if (!record_latent_copy(implementation.append_pipeline.get(), current.previous_compressed, next_compressed,
                                        static_cast<uint64_t>(state->compressed_count) * plan.head_dimension, 0, cmd))
                    return false;
            }
            state->compressed = std::move(next_compressed);
            state->compressed_capacity = static_cast<uint32_t>(capacity);
        }
        const uint32_t previous_count = refresh ? 0 : state->compressed_count;
        if (context.compressed_count > previous_count)
        {
            const uint64_t offset = static_cast<uint64_t>(previous_count) * plan.head_dimension;
            const uint64_t count = static_cast<uint64_t>(context.compressed_count - previous_count) * plan.head_dimension;
            if (!fill_staging_values(cache.latent_compressed.data() + offset, static_cast<size_t>(count), sizeof(float), current.compressed_staging, transfer_slot.staging_allocator)
                || !record_mapped_upload(current.compressed_staging, current.uploaded_compressed, cmd, implementation.opt)
                || !record_latent_copy(implementation.append_pipeline.get(), current.uploaded_compressed, state->compressed, count, offset, cmd))
                return false;
            ++uploads;
        }
        if (context.selected_compressed_indices && !context.compressed_indices.empty())
        {
            if (!fill_staging_values(context.compressed_indices.data(), context.compressed_indices.size(), sizeof(uint32_t), current.index_staging, transfer_slot.staging_allocator)
                || !record_mapped_upload(current.index_staging, current.indices, cmd, implementation.opt))
                return false;
            ++uploads;
        }
        else
            current.indices = state->window;
        prepare_latent_rope(plan, positions[row], current.coefficients);
        if (!fill_staging_values(current.coefficients.data(), current.coefficients.size(), sizeof(float), current.rope_staging, transfer_slot.staging_allocator)
            || !record_mapped_upload(current.rope_staging, current.rope, cmd, implementation.opt))
            return false;
        ++uploads;
        {
            const std::vector<ncnn::VkMat> bindings = {raw_query, current.rope, normalized_query};
            std::vector<ncnn::vk_constant_type> constants(5);
            constants[0].u32 = plan.head_dimension;
            constants[1].u32 = plan.head_count;
            constants[2].u32 = plan.rope_head_dimension;
            constants[3].u32 = static_cast<uint32_t>(row);
            constants[4].f = plan.norm_epsilon;
            ncnn::VkMat dispatcher;
            dispatcher.w = 128;
            dispatcher.h = 1;
            dispatcher.c = static_cast<int>(plan.head_count);
            cmd.record_pipeline(implementation.query_pipeline.get(), bindings, constants, dispatcher);
        }
        {
            const std::vector<ncnn::VkMat> bindings = {normalized_query, state->window, state->compressed.empty() ? state->window : state->compressed, current.indices, next_sinks, current.rope, attention};
            std::vector<ncnn::vk_constant_type> constants(10);
            constants[0].u32 = plan.head_dimension;
            constants[1].u32 = plan.head_count;
            constants[2].u32 = plan.rope_head_dimension;
            constants[3].u32 = static_cast<uint32_t>(row);
            constants[4].u32 = plan.sliding_window;
            constants[5].u32 = static_cast<uint32_t>(context.window_begin % plan.sliding_window);
            constants[6].u32 = context.window_count;
            constants[7].u32 = context.selected_compressed_indices ? static_cast<uint32_t>(context.compressed_indices.size()) : context.compressed_count;
            constants[8].u32 = context.selected_compressed_indices ? 1 : 0;
            constants[9].f = 1.0f / std::sqrt(static_cast<float>(plan.head_dimension));
            ncnn::VkMat dispatcher;
            dispatcher.w = 128;
            dispatcher.h = 1;
            dispatcher.c = static_cast<int>(plan.head_count);
            cmd.record_pipeline(implementation.attention_pipeline.get(), bindings, constants, dispatcher);
        }
    }
    ncnn::VkMat output_rank;
    ncnn::VkMat final_output;
    std::vector<ncnn::VkMat> projection_storage;
    const uint32_t output_columns = implementation.output_b->output_columns();
    const DType output_dtype = output.dtype();
    if (!implementation.output_a->record_forward(attention, output_rank, cmd, projection_storage)
        || !implementation.output_b->record_forward(output_rank, final_output, cmd, projection_storage)
        || !prepare_staging_batch(transfer_slot.download, rows, output_columns, transfer_slot.staging_allocator, output.element_size())
        || !record_prepared_activation_staging_download(final_output, rows, output_columns, transfer_slot.download, cmd,
                                                        implementation.context->device(), implementation.opt, output_dtype))
        return false;
    if (submit_compute_and_wait(cmd, implementation.context->device()) != 0)
        return false;
    ActivationBuffer completed(rows, output_columns, output_dtype);
    if (!copy_staging_to_cpu_batch(transfer_slot.download, completed))
        return false;
    output.swap(completed);
    implementation.sinks = std::move(next_sinks);
    for (size_t row = 0; row < rows; ++row)
    {
        work[row].state->token_count = positions[row] + 1;
        work[row].state->compressed_count = contexts[row].compressed_count;
        // This compatibility path did not update compressor pending state.
        work[row].state->complete_state = false;
        caches[row]->device_allocated_size = work[row].state->allocated_bytes();
    }
    mirror_guard.committed = true;
    ++runtime_state.compute_submissions;
    runtime_state.batch_uploads += uploads;
    ++runtime_state.batch_downloads;
    runtime_state.attention_blocks += rows;
    return true;
#else
    (void)positions;
    (void)caches;
    (void)contexts;
    (void)query;
    (void)device_query;
    (void)output;
    return false;
#endif
}

#if NCNN_MOE_WITH_VULKAN
struct LatentRecordedRow_vulkan
{
    LayerCache* cache = nullptr;
    uint64_t position = 0;
    uint32_t selected_count = 0;
    std::shared_ptr<LatentCache_vulkan> state;
    ncnn::VkMat window_shadow;
    ncnn::VkMat compressor_shadow;
    ncnn::VkMat index_compressor_shadow;
    ncnn::VkMat compressed_shadow;
    ncnn::VkMat index_compressed_shadow;
    ncnn::VkMat selected_shadow;
    std::vector<float> window_values;
    std::vector<float> compressor_values;
    std::vector<float> index_compressor_values;
    std::vector<float> compressed_values;
    std::vector<float> index_compressed_values;
    std::vector<uint32_t> selected_indices;
};

static bool read_latent_shadow(ncnn::VkMat& staging, std::vector<float>& destination)
{
    if (staging.empty())
        return true;
    staging.allocator->invalidate(staging.data);
    const ncnn::Mat mapped = staging.mapped();
    if (mapped.empty() || mapped.elemsize != sizeof(float) || mapped.elempack != 1)
        return false;
    const float* values = static_cast<const float*>(mapped.data);
    // mapped.total() includes padding; packed shadows carry logical counts.
    destination.assign(values, values + static_cast<size_t>(staging.w) * staging.h);
    staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
    staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    return true;
}

static void commit_compressor_shadow(const AttentionBlockPlan& plan,
                                     uint64_t position,
                                     uint32_t dimension,
                                     const std::vector<float>& shadow,
                                     std::vector<float>& pending_values,
                                     std::vector<float>& pending_scores,
                                     std::vector<float>& previous_values,
                                     std::vector<float>& previous_scores)
{
    const uint32_t multiplier = plan.compression_ratio == 4 ? 2 : 1;
    const size_t columns = static_cast<size_t>(multiplier) * dimension;
    const size_t pending_size = static_cast<size_t>(plan.compression_ratio) * columns;
    if (pending_values.size() != pending_size)
    {
        pending_values.assign(pending_size, 0.0f);
        pending_scores.assign(pending_size, 0.0f);
    }
    const size_t slot = position % plan.compression_ratio;
    std::copy_n(shadow.data(), columns, pending_values.data() + slot * columns);
    std::copy_n(shadow.data() + columns, columns, pending_scores.data() + slot * columns);
    if (multiplier == 2 && slot + 1 == plan.compression_ratio)
    {
        const size_t previous_size = static_cast<size_t>(plan.compression_ratio) * dimension;
        previous_values.resize(previous_size);
        previous_scores.resize(previous_size);
        for (uint32_t token = 0; token < plan.compression_ratio; ++token)
        {
            std::copy_n(pending_values.data() + token * columns, dimension, previous_values.data() + static_cast<size_t>(token) * dimension);
            std::copy_n(pending_scores.data() + token * columns, dimension, previous_scores.data() + static_cast<size_t>(token) * dimension);
        }
    }
}

static ncnn::VkMat latent_flat_view(const ncnn::VkMat& source, uint64_t offset, uint32_t count)
{
    ncnn::VkMat result = source;
    result.dims = 2;
    result.w = static_cast<int>(count);
    result.h = 1;
    result.d = 1;
    result.c = 1;
    result.cstep = count;
    result.offset += static_cast<size_t>(offset) * sizeof(float);
    return result;
}

static bool record_latent_shadow(const ncnn::VkMat& source,
                                 uint32_t columns,
                                 ncnn::VkMat& staging,
                                 const std::shared_ptr<VulkanContext>& context,
                                 const ncnn::Option& opt,
                                 ncnn::VkCompute& cmd)
{
    (void)context;
    (void)opt;
    (void)cmd;
    if (source.empty() || source.total() < columns)
        return false;
    // Keep the device result live; all shadows are copied and downloaded once
    // after the caller's attention projections have been recorded.
    staging = latent_flat_view(source, 0, columns);
    return true;
}

static bool record_latent_transform(const std::shared_ptr<ncnn::Pipeline>& pipeline,
                                    const ncnn::VkMat& input,
                                    const ncnn::VkMat& norm,
                                    const ncnn::VkMat& rope,
                                    ncnn::VkMat& output,
                                    uint32_t dimension,
                                    uint32_t rope_dimension,
                                    uint32_t input_offset,
                                    uint32_t output_offset,
                                    uint32_t flags,
                                    uint32_t head_count,
                                    float epsilon,
                                    ncnn::VkCompute& cmd)
{
    if (!pipeline || input.empty() || output.empty())
        return false;
    const std::vector<ncnn::VkMat> bindings = {input, norm, rope, output};
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = dimension;
    constants[1].u32 = rope_dimension;
    constants[2].u32 = input_offset;
    constants[3].u32 = output_offset;
    constants[4].u32 = flags;
    constants[5].f = epsilon;
    ncnn::VkMat dispatcher;
    dispatcher.w = 128;
    dispatcher.h = 1;
    dispatcher.c = static_cast<int>(head_count);
    cmd.record_pipeline(pipeline.get(), bindings, constants, dispatcher);
    return true;
}

static bool tensor_latent_values(const TensorData& tensor, uint64_t count, std::vector<float>& values)
{
    if (tensor.element_count() != count || count == 0)
        return false;
    values.resize(static_cast<size_t>(count));
    if (tensor.dtype == DType::Float32)
        std::copy(tensor.float32_values().begin(), tensor.float32_values().end(), values.begin());
    else if (tensor.dtype == DType::BFloat16)
    {
        const auto source = tensor.bfloat16_values();
        for (size_t index = 0; index < values.size(); ++index)
            values[index] = bfloat16_to_float(source[index]);
    }
    else
        return false;
    return true;
}
#endif

class LatentAttentionWork_vulkan::Implementation
{
public:
#if NCNN_MOE_WITH_VULKAN
    std::shared_ptr<VulkanContext> context;
    AttentionBlockPlan plan;
    std::vector<LatentRecordedRow_vulkan> rows;
    std::vector<ncnn::VkMat> storage;
    ncnn::VkMat shadow_staging;
    uint64_t uploads = 0;
    uint64_t downloads = 0;
    bool committed = false;
    ~Implementation()
    {
        if (!committed)
        {
            for (LatentRecordedRow_vulkan& row : rows)
            {
                row.cache->latent_device_state.reset();
                row.cache->device_allocated_size = 0;
            }
        }
    }
#endif
};

LatentAttentionWork_vulkan::LatentAttentionWork_vulkan()
    : d(new Implementation)
{
}

LatentAttentionWork_vulkan::~LatentAttentionWork_vulkan() = default;

bool LatentAttentionWork_vulkan::commit()
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& work = *d;
    if (!work.context || work.committed || work.rows.empty())
        return false;
    for (LatentRecordedRow_vulkan& row : work.rows)
    {
        if (!read_latent_shadow(row.window_shadow, row.window_values)
            || !read_latent_shadow(row.compressor_shadow, row.compressor_values)
            || !read_latent_shadow(row.index_compressor_shadow, row.index_compressor_values)
            || !read_latent_shadow(row.compressed_shadow, row.compressed_values)
            || !read_latent_shadow(row.index_compressed_shadow, row.index_compressed_values))
            return false;
        if (!row.selected_shadow.empty())
        {
            row.selected_shadow.allocator->invalidate(row.selected_shadow.data);
            const ncnn::Mat mapped = row.selected_shadow.mapped();
            if (mapped.empty() || mapped.w != static_cast<int>(row.selected_count) || mapped.h != 1)
                return false;
            const uint32_t* values = static_cast<const uint32_t*>(mapped.data);
            row.selected_indices.assign(values, values + row.selected_count);
            row.selected_shadow.data->access_flags = VK_ACCESS_HOST_READ_BIT;
            row.selected_shadow.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        }
    }
    const AttentionBlockPlan& plan = work.plan;
    for (LatentRecordedRow_vulkan& row : work.rows)
    {
        LayerCache& cache = *row.cache;
        record_latent_cache_transaction_row(cache, plan, row.position);
        cache.columns = plan.head_dimension;
        cache.capacity_tokens = plan.sliding_window;
        cache.latent_cache = true;
        const size_t window_size = static_cast<size_t>(plan.sliding_window) * plan.head_dimension;
        if (cache.latent_window.size() != window_size)
            cache.latent_window.assign(window_size, 0.0f);
        std::copy_n(row.window_values.data(), plan.head_dimension,
                    cache.latent_window.data() + static_cast<size_t>(row.position % plan.sliding_window) * plan.head_dimension);
        if (plan.compression_ratio != 0)
        {
            commit_compressor_shadow(plan, row.position, plan.head_dimension, row.compressor_values,
                                     cache.compressor_pending_values, cache.compressor_pending_scores,
                                     cache.compressor_previous_values, cache.compressor_previous_scores);
            cache.latent_compressed.insert(cache.latent_compressed.end(), row.compressed_values.begin(), row.compressed_values.end());
            if (plan.compression_ratio == 4)
            {
                commit_compressor_shadow(plan, row.position, plan.index_head_dimension, row.index_compressor_values,
                                         cache.index_compressor_pending_values, cache.index_compressor_pending_scores,
                                         cache.index_compressor_previous_values, cache.index_compressor_previous_scores);
                cache.latent_index_compressed.insert(cache.latent_index_compressed.end(), row.index_compressed_values.begin(), row.index_compressed_values.end());
            }
        }
        cache.latent_selected_indices = row.selected_indices;
        cache.latent_token_count = row.position + 1;
        cache.latent_device_state = row.state;
        cache.device_allocated_size = row.state->allocated_bytes();
    }
    work.context->runtime_state().batch_uploads += work.uploads;
    work.context->runtime_state().batch_downloads += work.downloads;
    work.context->runtime_state().attention_blocks += work.rows.size();
    work.committed = true;
    return true;
#else
    return false;
#endif
}

bool LatentAttention_vulkan::forward_projected_batch(const ActivationBuffer& input,
                                                     std::span<const uint64_t> positions,
                                                     std::span<LayerCache* const> caches,
                                                     ActivationBuffer& output) const
{
#if NCNN_MOE_WITH_VULKAN
    if (!d->prepared || input.rows() == 0 || input.rows() != positions.size() || input.rows() != caches.size())
        return false;
    const std::lock_guard<std::mutex> lock(d->context->command_mutex());
    ncnn::VkCompute cmd(d->context->device(), d->context->command_optimization_flags());
    ncnn::VkMat input_staging;
    ncnn::VkMat device_input;
    ncnn::VkMat device_output;
    ncnn::VkMat output_staging;
    LatentAttentionWork_vulkan work;
    VulkanRuntimeState& state = d->context->runtime_state();
    if (!fill_staging_upload(input, input_staging, d->context->staging_allocator())
        || !record_mapped_activation_upload(input_staging, device_input, cmd, d->context->device(), d->opt, input.dtype())
        || !record_batch(device_input, positions, caches, device_output, cmd, work)
        || !prepare_staging_batch(output_staging, input.rows(), static_cast<uint32_t>(device_output.w), d->context->staging_allocator(), output.element_size())
        || !record_prepared_activation_staging_download(device_output, input.rows(), static_cast<uint32_t>(device_output.w), output_staging,
                                                        cmd, d->context->device(), d->opt, output.dtype()))
        return false;
    if (submit_compute_and_wait(cmd, d->context->device()) != 0)
        return false;
    ActivationBuffer completed(input.rows(), static_cast<uint32_t>(device_output.w), output.dtype());
    if (!copy_staging_to_cpu_batch(output_staging, completed) || !work.commit())
        return false;
    output.swap(completed);
    ++state.compute_submissions;
    ++state.batch_uploads;
    ++state.batch_downloads;
    return true;
#else
    (void)input;
    (void)positions;
    (void)caches;
    (void)output;
    return false;
#endif
}

bool LatentAttention_vulkan::prepare(const WeightStore& weights, const CompiledOperatorTable& operators, uint64_t optimization_flags)
{
#if NCNN_MOE_WITH_VULKAN
    Implementation& implementation = *d;
    const AttentionBlockPlan& plan = implementation.plan;
    const uint64_t flags = optimization_flags;
    if (!has_flag(flags, OptimizationVulkanAttention)
        || !has_flag(flags, OptimizationVulkanLatentInputRmsNorm)
        || (plan.compression_ratio != 0 && !has_flag(flags, OptimizationVulkanLatentCompressor))
        || (plan.compression_ratio != 0 && plan.compression_ratio != 4 && plan.compression_ratio != 128)
        || (plan.compression_ratio == 4
            && (plan.index_head_dimension == 0 || plan.index_head_dimension > 512
                || (plan.index_head_dimension & (plan.index_head_dimension - 1)) != 0
                || plan.rope_head_dimension > plan.index_head_dimension || plan.index_head_count == 0)))
        return false;
    const auto& query_a = operators.at_weight(plan.query_a_weight).float8;
    const auto& query_b = operators.at_weight(plan.query_b_weight).float8;
    const auto& key_value = operators.at_weight(plan.key_value_weight).float8;
    if (!query_a || !query_b || !key_value || query_a->vulkan_context() != implementation.context
        || query_b->vulkan_context() != implementation.context || key_value->vulkan_context() != implementation.context
        || query_a->input_columns() != key_value->input_columns()
        || query_a->output_columns() != query_b->input_columns()
        || query_b->output_columns() != plan.head_count * plan.head_dimension
        || key_value->output_columns() != plan.head_dimension)
        return false;
    implementation.query_a = query_a;
    implementation.query_b = query_b;
    implementation.key_value = key_value;
    if (plan.compression_ratio != 0)
    {
        implementation.compressor_value = operators.at_weight(plan.compressor_key_value_weight).bfloat16;
        implementation.compressor_gate = operators.at_weight(plan.compressor_gate_weight).bfloat16;
        const uint32_t multiplier = plan.compression_ratio == 4 ? 2 : 1;
        const auto valid_projection = [&](const std::shared_ptr<Bfloat16Linear_vulkan>& value, uint32_t columns) {
            return value && value->vulkan_context() == implementation.context
                   && value->input_columns() == query_a->input_columns()
                   && value->output_columns() == columns;
        };
        if (!valid_projection(implementation.compressor_value, multiplier * plan.head_dimension)
            || !valid_projection(implementation.compressor_gate, multiplier * plan.head_dimension))
            return false;
        if (plan.compression_ratio == 4)
        {
            implementation.index_compressor_value = operators.at_weight(plan.indexer_compressor_key_value_weight).bfloat16;
            implementation.index_compressor_gate = operators.at_weight(plan.indexer_compressor_gate_weight).bfloat16;
            if (!valid_projection(implementation.index_compressor_value, 2 * plan.index_head_dimension)
                || !valid_projection(implementation.index_compressor_gate, 2 * plan.index_head_dimension))
                return false;
            implementation.index_query = operators.at_weight(plan.indexer_query_weight);
            implementation.index_weights = operators.at_weight(plan.indexer_weights_weight);
            const auto valid_index = [&](const CompiledOperator& value, uint32_t input_columns, uint32_t output_columns) {
                if (value.bfloat16)
                    return value.bfloat16->vulkan_context() == implementation.context
                           && value.bfloat16->input_columns() == input_columns && value.bfloat16->output_columns() == output_columns;
                if (value.float8)
                    return value.float8->vulkan_context() == implementation.context
                           && value.float8->input_columns() == input_columns && value.float8->output_columns() == output_columns;
                return value.linear && value.linear->vulkan_context() == implementation.context
                       && value.linear->input_columns() == input_columns && value.linear->output_columns() == output_columns;
            };
            if (!valid_index(implementation.index_query, query_a->output_columns(), plan.index_head_count * plan.index_head_dimension)
                || !valid_index(implementation.index_weights, query_a->input_columns(), plan.index_head_count))
                return false;
        }
    }
    const std::lock_guard<std::mutex> lock(implementation.context->command_mutex());
    if (!create_latent_pipeline(implementation.context, implementation.opt, latent_attention_transform_shader,
                                static_cast<int>(sizeof(latent_attention_transform_shader) - 1), implementation.transform_pipeline)
        || !create_latent_pipeline(implementation.context, implementation.opt, latent_attention_compress_shader,
                                   static_cast<int>(sizeof(latent_attention_compress_shader) - 1), implementation.compressor_pipeline)
        || !create_latent_pipeline(implementation.context, implementation.opt, latent_attention_index_score_shader,
                                   static_cast<int>(sizeof(latent_attention_index_score_shader) - 1), implementation.index_score_pipeline)
        || !create_latent_pipeline(implementation.context, implementation.opt, latent_attention_index_topk_shader,
                                   static_cast<int>(sizeof(latent_attention_index_topk_shader) - 1), implementation.index_topk_pipeline))
        return false;
    // Parallel selection is optional: unusual K and unavailable scratch retain
    // the existing GPU heap path instead of moving selection onto the CPU.
    if (plan.index_top_k > 0 && plan.index_top_k <= 1024
        && implementation.context->device()->info.max_shared_memory_size() >= 8192)
    {
        (void)create_latent_pipeline(implementation.context, implementation.opt, latent_attention_index_sort_shader,
                                     static_cast<int>(sizeof(latent_attention_index_sort_shader) - 1), implementation.index_sort_pipeline);
        (void)create_latent_pipeline(implementation.context, implementation.opt, latent_attention_index_merge_shader,
                                     static_cast<int>(sizeof(latent_attention_index_merge_shader) - 1), implementation.index_merge_pipeline);
    }
    ncnn::VkCompute cmd(implementation.context->device(), implementation.context->command_optimization_flags());
    std::vector<ncnn::VkMat> staging;
    const auto upload_constant = [&](const TensorData& tensor, uint64_t count, ncnn::VkMat& destination) {
        std::vector<float> values;
        if (!tensor_latent_values(tensor, count, values))
            return false;
        staging.emplace_back();
        return fill_staging_values(values.data(), values.size(), sizeof(float), staging.back(), implementation.context->staging_allocator())
               && record_mapped_upload(staging.back(), destination, cmd, implementation.opt);
    };
    if (!upload_constant(weights.at(plan.key_value_norm_weight), plan.head_dimension, implementation.key_norm)
        || !upload_constant(weights.at(plan.sinks), plan.head_count, implementation.sinks))
        return false;
    if (plan.compression_ratio != 0)
    {
        const uint32_t multiplier = plan.compression_ratio == 4 ? 2 : 1;
        if (!upload_constant(weights.at(plan.compressor_norm_weight), plan.head_dimension, implementation.compressor_norm)
            || !upload_constant(weights.at(plan.compressor_position), static_cast<uint64_t>(plan.compression_ratio) * multiplier * plan.head_dimension, implementation.compressor_position))
            return false;
        if (plan.compression_ratio == 4
            && (!upload_constant(weights.at(plan.indexer_compressor_norm_weight), plan.index_head_dimension, implementation.index_compressor_norm)
                || !upload_constant(weights.at(plan.indexer_compressor_position), static_cast<uint64_t>(plan.compression_ratio) * 2 * plan.index_head_dimension, implementation.index_compressor_position)))
            return false;
    }
    if (submit_compute_and_wait(cmd, implementation.context->device()) != 0)
        return false;
    implementation.context->runtime_state().compute_submissions += 1;
    implementation.context->runtime_state().batch_uploads += staging.size();
    implementation.prepared = true;
    return true;
#else
    (void)weights;
    (void)operators;
    (void)optimization_flags;
    return false;
#endif
}

#if NCNN_MOE_WITH_VULKAN
bool LatentAttention_vulkan::can_record() const noexcept
{
    return d->prepared;
}

const std::shared_ptr<VulkanContext>& LatentAttention_vulkan::vulkan_context() const noexcept
{
    return d->context;
}

const ncnn::Option& LatentAttention_vulkan::option() const noexcept
{
    return d->opt;
}

bool LatentAttention_vulkan::record_batch(const ncnn::VkMat& input,
                                          std::span<const uint64_t> positions,
                                          std::span<LayerCache* const> caches,
                                          ncnn::VkMat& output,
                                          ncnn::VkCompute& cmd,
                                          LatentAttentionWork_vulkan& work) const
{
    Implementation& implementation = *d;
    const AttentionBlockPlan& plan = implementation.plan;
    LatentAttentionWork_vulkan::Implementation& recording = *work.d;
    const size_t rows = positions.size();
    if (!implementation.prepared || recording.context || input.empty() || input.dims != 2
        || input.elemsize != sizeof(float) || input.elempack != 1
        || input.w != static_cast<int>(implementation.query_a->input_columns())
        || rows == 0 || rows != caches.size() || rows != static_cast<size_t>(input.h))
        return false;
    std::vector<std::pair<LayerCache*, uint64_t>> next_positions;
    for (size_t row = 0; row < rows; ++row)
    {
        if (!caches[row] || positions[row] == std::numeric_limits<uint64_t>::max())
            return false;
        auto next = std::find_if(next_positions.begin(), next_positions.end(), [&](const auto& item) { return item.first == caches[row]; });
        const uint64_t expected = next == next_positions.end() ? caches[row]->latent_token_count : next->second;
        if (positions[row] != expected || caches[row]->latent_compressed.size() % plan.head_dimension != 0
            || (plan.compression_ratio == 4 && caches[row]->latent_index_compressed.size() / plan.index_head_dimension != caches[row]->latent_compressed.size() / plan.head_dimension))
            return false;
        if (next == next_positions.end())
            next_positions.push_back({caches[row], positions[row] + 1});
        else
            next->second = positions[row] + 1;
    }
    for (ncnn::VkMat* value : {&implementation.sinks, &implementation.key_norm, &implementation.compressor_norm,
                               &implementation.compressor_position, &implementation.index_compressor_norm, &implementation.index_compressor_position})
    {
        if (!value->empty())
        {
            value->data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            value->data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
    }
    recording.context = implementation.context;
    recording.plan = plan;
    recording.rows.reserve(rows);
    recording.storage.reserve(rows * 24 + 16);
    const auto upload_values = [&](const std::vector<float>& values, ncnn::VkMat& destination) {
        if (values.empty())
            return true;
        recording.storage.emplace_back();
        if (!fill_staging_values(values.data(), values.size(), sizeof(float), recording.storage.back(), implementation.context->staging_allocator())
            || !record_mapped_upload(recording.storage.back(), destination, cmd, implementation.opt))
            return false;
        ++recording.uploads;
        return true;
    };
    const auto allocate = [&](ncnn::VkMat& destination, uint64_t count) {
        return allocate_latent_buffer(destination, count, implementation.context);
    };
    const auto record_index_projection = [&](const CompiledOperator& op, const ncnn::VkMat& source, ncnn::VkMat& destination) {
        if (op.bfloat16)
            return op.bfloat16->forward(source, destination, cmd, implementation.opt) == 0;
        if (op.float8)
            return op.float8->record_forward(source, destination, cmd, recording.storage);
        return op.linear && op.linear->forward(source, destination, cmd, implementation.opt) == 0;
    };
    ncnn::VkMat normalized;
    ncnn::VkMat query;
    ncnn::VkMat key_value;
    ncnn::VkMat query_rank;
    if (!implementation.query_a->record_input_normalization(input, normalized, cmd)
        || !implementation.query_a->record_rms_norm_chain_parallel(normalized, *implementation.query_b, *implementation.key_value,
                                                                   query, key_value, query_rank, cmd, recording.storage))
        return false;
    recording.storage.push_back(normalized);
    recording.storage.push_back(query);
    recording.storage.push_back(key_value);
    recording.storage.push_back(query_rank);
    ncnn::VkMat compressor_values;
    ncnn::VkMat compressor_scores;
    ncnn::VkMat index_compressor_values;
    ncnn::VkMat index_compressor_scores;
    ncnn::VkMat index_query;
    ncnn::VkMat index_weights;
    bool index_projection_ready = false;
    if (plan.compression_ratio != 0)
    {
        if (implementation.compressor_value->forward(normalized, compressor_values, cmd, implementation.opt) != 0
            || implementation.compressor_gate->forward(normalized, compressor_scores, cmd, implementation.opt) != 0)
            return false;
        recording.storage.push_back(compressor_values);
        recording.storage.push_back(compressor_scores);
        if (plan.compression_ratio == 4)
        {
            if (implementation.index_compressor_value->forward(normalized, index_compressor_values, cmd, implementation.opt) != 0
                || implementation.index_compressor_gate->forward(normalized, index_compressor_scores, cmd, implementation.opt) != 0)
                return false;
            recording.storage.push_back(index_compressor_values);
            recording.storage.push_back(index_compressor_scores);
        }
    }
    ncnn::VkMat normalized_query;
    ncnn::VkMat attention;
    const uint32_t query_columns = plan.head_count * plan.head_dimension;
    normalized_query.create(static_cast<int>(query_columns), static_cast<int>(rows), sizeof(float), implementation.context->blob_allocator());
    attention.create(static_cast<int>(query_columns), static_cast<int>(rows), sizeof(float), implementation.context->blob_allocator());
    if (normalized_query.empty() || attention.empty())
        return false;
    recording.storage.push_back(normalized_query);
    recording.storage.push_back(attention);
    for (size_t row = 0; row < rows; ++row)
    {
        LayerCache& cache = *caches[row];
        const uint64_t position = positions[row];
        recording.rows.emplace_back();
        LatentRecordedRow_vulkan& current = recording.rows.back();
        current.cache = &cache;
        current.position = position;
        auto earlier = std::find_if(recording.rows.begin(), recording.rows.end() - 1, [&](const LatentRecordedRow_vulkan& item) { return item.cache == &cache; });
        std::shared_ptr<LatentCache_vulkan> state;
        if (earlier != recording.rows.end() - 1)
            state = earlier->state;
        else
        {
            const bool shared_state = cache.latent_device_state && cache.latent_device_state.use_count() != 1;
            state = cache.latent_device_state;
            const uint32_t count = static_cast<uint32_t>(cache.latent_compressed.size() / plan.head_dimension);
            if (shared_state || !state || !state->complete_state || state->context != implementation.context
                || state->dimension != plan.head_dimension || state->window_capacity != plan.sliding_window
                || state->token_count != position || state->compressed_count != count)
            {
                state = std::make_shared<LatentCache_vulkan>();
                state->context = implementation.context;
                state->dimension = plan.head_dimension;
                state->window_capacity = plan.sliding_window;
                state->token_count = position;
                state->compressed_count = count;
                state->complete_state = true;
                const uint64_t window_elements = static_cast<uint64_t>(plan.sliding_window) * plan.head_dimension;
                if (!allocate(state->window, window_elements))
                    return false;
                if (position != 0)
                {
                    const size_t valid = static_cast<size_t>(std::min<uint64_t>(position, plan.sliding_window)) * plan.head_dimension;
                    if (cache.latent_window.size() < valid)
                        return false;
                    std::vector<float> values(cache.latent_window.begin(), cache.latent_window.begin() + valid);
                    ncnn::VkMat uploaded;
                    if (!upload_values(values, uploaded)
                        || !record_latent_copy(implementation.append_pipeline.get(), uploaded, state->window, valid, 0, cmd))
                        return false;
                    recording.storage.push_back(uploaded);
                }
                if (plan.compression_ratio != 0)
                {
                    const auto restore_compressor = [&](uint32_t dimension,
                                                        const std::vector<float>& pending_values,
                                                        const std::vector<float>& pending_scores,
                                                        const std::vector<float>& previous_values,
                                                        const std::vector<float>& previous_scores,
                                                        ncnn::VkMat& destination) {
                        const uint32_t multiplier = plan.compression_ratio == 4 ? 2 : 1;
                        const size_t pending = static_cast<size_t>(plan.compression_ratio) * multiplier * dimension;
                        const size_t previous = multiplier == 2 ? static_cast<size_t>(plan.compression_ratio) * dimension : 0;
                        std::vector<float> values(pending * 2 + previous * 2, 0.0f);
                        if (!pending_values.empty())
                        {
                            if (pending_values.size() != pending || pending_scores.size() != pending)
                                return false;
                            std::copy(pending_values.begin(), pending_values.end(), values.begin());
                            std::copy(pending_scores.begin(), pending_scores.end(), values.begin() + pending);
                        }
                        if (!previous_values.empty())
                        {
                            if (previous_values.size() != previous || previous_scores.size() != previous)
                                return false;
                            std::copy(previous_values.begin(), previous_values.end(), values.begin() + pending * 2);
                            std::copy(previous_scores.begin(), previous_scores.end(), values.begin() + pending * 2 + previous);
                        }
                        return upload_values(values, destination);
                    };
                    if (!restore_compressor(plan.head_dimension, cache.compressor_pending_values, cache.compressor_pending_scores,
                                            cache.compressor_previous_values, cache.compressor_previous_scores, state->compressor_state))
                        return false;
                    if (plan.compression_ratio == 4
                        && !restore_compressor(plan.index_head_dimension, cache.index_compressor_pending_values, cache.index_compressor_pending_scores,
                                               cache.index_compressor_previous_values, cache.index_compressor_previous_scores, state->index_compressor_state))
                        return false;
                }
                if (count != 0)
                {
                    state->compressed_capacity = std::max(32u, count);
                    if (!allocate(state->compressed, static_cast<uint64_t>(state->compressed_capacity) * plan.head_dimension))
                        return false;
                    ncnn::VkMat uploaded;
                    if (!upload_values(cache.latent_compressed, uploaded)
                        || !record_latent_copy(implementation.append_pipeline.get(), uploaded, state->compressed, cache.latent_compressed.size(), 0, cmd))
                        return false;
                    recording.storage.push_back(uploaded);
                    if (plan.compression_ratio == 4)
                    {
                        if (!allocate(state->index_compressed, static_cast<uint64_t>(state->compressed_capacity) * plan.index_head_dimension)
                            || !upload_values(cache.latent_index_compressed, uploaded)
                            || !record_latent_copy(implementation.append_pipeline.get(), uploaded, state->index_compressed, cache.latent_index_compressed.size(), 0, cmd))
                            return false;
                        recording.storage.push_back(uploaded);
                    }
                }
            }
        }
        current.state = state;
        cache.latent_device_state = state;
        // Re-arm persistent buffers after any previously abandoned recorder.
        for (ncnn::VkMat* value : {&state->window, &state->compressed, &state->compressor_state, &state->index_compressor_state, &state->index_compressed})
        {
            if (!value->empty())
            {
                value->data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                value->data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            }
        }
        std::vector<float> coefficients;
        prepare_latent_rope(plan, position, coefficients);
        ncnn::VkMat rope;
        if (!upload_values(coefficients, rope))
            return false;
        recording.storage.push_back(rope);
        ncnn::VkMat window_update;
        if (!allocate(window_update, plan.head_dimension)
            || !record_latent_transform(implementation.transform_pipeline, key_value, implementation.key_norm, rope, window_update,
                                        plan.head_dimension, plan.rope_head_dimension, static_cast<uint32_t>(row) * plan.head_dimension,
                                        0, 1 | 4, 1, plan.norm_epsilon, cmd)
            || !record_latent_copy(implementation.append_pipeline.get(), window_update, state->window, plan.head_dimension,
                                   position % plan.sliding_window * plan.head_dimension, cmd)
            || !record_latent_shadow(window_update, plan.head_dimension, current.window_shadow, implementation.context, implementation.opt, cmd))
            return false;
        recording.storage.push_back(window_update);
        const bool completes = plan.compression_ratio != 0 && position % plan.compression_ratio + 1 == plan.compression_ratio;
        const uint32_t next_count = state->compressed_count + (completes ? 1 : 0);
        if (next_count > state->compressed_capacity)
        {
            const uint32_t capacity = std::max(next_count, std::max(32u, state->compressed_capacity * 2));
            const auto grow = [&](ncnn::VkMat& value, uint32_t dimension) {
                ncnn::VkMat replacement;
                if (!allocate(replacement, static_cast<uint64_t>(capacity) * dimension))
                    return false;
                if (state->compressed_count != 0
                    && !record_latent_copy(implementation.append_pipeline.get(), value, replacement,
                                           static_cast<uint64_t>(state->compressed_count) * dimension, 0, cmd))
                    return false;
                recording.storage.push_back(value);
                value = std::move(replacement);
                return true;
            };
            if (!grow(state->compressed, plan.head_dimension)
                || (plan.compression_ratio == 4 && !grow(state->index_compressed, plan.index_head_dimension)))
                return false;
            state->compressed_capacity = capacity;
        }
        if (plan.compression_ratio != 0)
        {
            std::vector<float> compressed_coefficients;
            if (completes)
                prepare_latent_rope(plan, position + 1 - plan.compression_ratio, compressed_coefficients);
            // Upload into distinct storage: a same-shape VkMat upload reuses
            // its allocation even when it aliases the current token's RoPE.
            ncnn::VkMat compressed_rope;
            if (completes)
            {
                if (!upload_values(compressed_coefficients, compressed_rope))
                    return false;
            }
            else
                compressed_rope = rope;
            recording.storage.push_back(compressed_rope);
            const auto compress = [&](bool indexer,
                                      const ncnn::VkMat& values,
                                      const ncnn::VkMat& scores,
                                      ncnn::VkMat& compressor_state,
                                      ncnn::VkMat& compressed,
                                      ncnn::VkMat& shadow,
                                      ncnn::VkMat& compressed_shadow) {
                const uint32_t dimension = indexer ? plan.index_head_dimension : plan.head_dimension;
                const uint32_t multiplier = plan.compression_ratio == 4 ? 2 : 1;
                ncnn::VkMat pooled;
                ncnn::VkMat pending_update;
                if (!allocate(pooled, dimension) || !allocate(pending_update, 2 * multiplier * dimension))
                    return false;
                const std::vector<ncnn::VkMat> bindings = {values, scores, indexer ? implementation.index_compressor_position : implementation.compressor_position,
                                                           compressor_state, pooled, pending_update};
                std::vector<ncnn::vk_constant_type> constants(5);
                constants[0].u32 = dimension;
                constants[1].u32 = plan.compression_ratio;
                constants[2].u32 = static_cast<uint32_t>(row);
                constants[3].u32 = static_cast<uint32_t>(position % plan.compression_ratio);
                constants[4].u32 = plan.compression_ratio == 4 && state->compressed_count != 0 ? 1 : 0;
                ncnn::VkMat dispatcher;
                dispatcher.w = 128;
                dispatcher.h = 1;
                dispatcher.c = 1;
                cmd.record_pipeline(implementation.compressor_pipeline.get(), bindings, constants, dispatcher);
                if (!record_latent_shadow(pending_update, 2 * multiplier * dimension, shadow, implementation.context, implementation.opt, cmd))
                    return false;
                recording.storage.push_back(pooled);
                recording.storage.push_back(pending_update);
                if (completes)
                {
                    ncnn::VkMat completed;
                    if (!allocate(completed, dimension)
                        || !record_latent_transform(implementation.transform_pipeline, pooled,
                                                    indexer ? implementation.index_compressor_norm : implementation.compressor_norm,
                                                    compressed_rope, completed, dimension, plan.rope_head_dimension, 0, 0,
                                                    indexer ? 1 | 2 | 8 : 1 | 4, 1, plan.norm_epsilon, cmd)
                        || !record_latent_copy(implementation.append_pipeline.get(), completed, compressed, dimension,
                                               static_cast<uint64_t>(state->compressed_count) * dimension, cmd)
                        || !record_latent_shadow(completed, dimension, compressed_shadow, implementation.context, implementation.opt, cmd))
                        return false;
                    recording.storage.push_back(completed);
                }
                return true;
            };
            if (!compress(false, compressor_values, compressor_scores, state->compressor_state, state->compressed,
                          current.compressor_shadow, current.compressed_shadow)
                || (plan.compression_ratio == 4
                    && !compress(true, index_compressor_values, index_compressor_scores, state->index_compressor_state, state->index_compressed,
                                 current.index_compressor_shadow, current.index_compressed_shadow)))
                return false;
        }
        ncnn::VkMat indices;
        const bool selected = plan.compression_ratio == 4 && next_count > plan.index_top_k;
        if (!selected)
            indices = state->window;
        if (selected)
        {
            if (!index_projection_ready)
            {
                if (!record_index_projection(implementation.index_query, query_rank, index_query)
                    || !record_index_projection(implementation.index_weights, normalized, index_weights))
                    return false;
                recording.storage.push_back(index_query);
                recording.storage.push_back(index_weights);
                index_projection_ready = true;
            }
            ncnn::VkMat transformed_query;
            ncnn::VkMat scores;
            if (!allocate(transformed_query, static_cast<uint64_t>(plan.index_head_count) * plan.index_head_dimension)
                || !allocate(scores, next_count) || !allocate(indices, std::max(1u, plan.index_top_k))
                || !record_latent_transform(implementation.transform_pipeline, index_query, implementation.key_norm, rope, transformed_query,
                                            plan.index_head_dimension, plan.rope_head_dimension,
                                            static_cast<uint32_t>(row) * plan.index_head_count * plan.index_head_dimension,
                                            0, 2 | 8, plan.index_head_count, plan.norm_epsilon, cmd))
                return false;
            {
                const std::vector<ncnn::VkMat> bindings = {transformed_query, state->index_compressed, index_weights, scores};
                std::vector<ncnn::vk_constant_type> constants(6);
                constants[0].u32 = plan.index_head_dimension;
                constants[1].u32 = plan.index_head_count;
                constants[2].u32 = next_count;
                constants[3].u32 = 0;
                constants[4].u32 = static_cast<uint32_t>(row) * plan.index_head_count;
                constants[5].f = 1.0f / std::sqrt(static_cast<float>(plan.index_head_dimension * plan.index_head_count));
                ncnn::VkMat dispatcher;
                dispatcher.w = static_cast<int>(std::min(next_count, 65535u) * 128);
                dispatcher.h = static_cast<int>((next_count + 65534) / 65535);
                dispatcher.c = 1;
                cmd.record_pipeline(implementation.index_score_pipeline.get(), bindings, constants, dispatcher);
            }
            const bool parallel_selection = record_latent_parallel_topk(implementation.index_sort_pipeline,
                                                                        implementation.index_merge_pipeline,
                                                                        scores, next_count, plan.index_top_k, indices,
                                                                        implementation.context, recording.storage, cmd);
            if (!parallel_selection)
            {
                const std::vector<ncnn::VkMat> bindings = {scores, indices};
                std::vector<ncnn::vk_constant_type> constants(2);
                constants[0].u32 = next_count;
                constants[1].u32 = plan.index_top_k;
                ncnn::VkMat dispatcher;
                dispatcher.w = 128;
                dispatcher.h = 1;
                dispatcher.c = 1;
                cmd.record_pipeline(implementation.index_topk_pipeline.get(), bindings, constants, dispatcher);
            }
            current.selected_count = plan.index_top_k;
            if (current.selected_count != 0)
            {
                if (!record_latent_shadow(indices, current.selected_count, current.selected_shadow, implementation.context, implementation.opt, cmd))
                    return false;
            }
            recording.storage.push_back(transformed_query);
            recording.storage.push_back(scores);
            recording.storage.push_back(indices);
        }
        {
            const std::vector<ncnn::VkMat> bindings = {query, rope, normalized_query};
            std::vector<ncnn::vk_constant_type> constants(5);
            constants[0].u32 = plan.head_dimension;
            constants[1].u32 = plan.head_count;
            constants[2].u32 = plan.rope_head_dimension;
            constants[3].u32 = static_cast<uint32_t>(row);
            constants[4].f = plan.norm_epsilon;
            ncnn::VkMat dispatcher;
            dispatcher.w = 128;
            dispatcher.h = 1;
            dispatcher.c = static_cast<int>(plan.head_count);
            cmd.record_pipeline(implementation.query_pipeline.get(), bindings, constants, dispatcher);
        }
        {
            const std::vector<ncnn::VkMat> bindings = {normalized_query, state->window, state->compressed.empty() ? state->window : state->compressed,
                                                       indices, implementation.sinks, rope, attention};
            std::vector<ncnn::vk_constant_type> constants(10);
            constants[0].u32 = plan.head_dimension;
            constants[1].u32 = plan.head_count;
            constants[2].u32 = plan.rope_head_dimension;
            constants[3].u32 = static_cast<uint32_t>(row);
            constants[4].u32 = plan.sliding_window;
            const uint64_t window_begin = position + 1 > plan.sliding_window ? position + 1 - plan.sliding_window : 0;
            constants[5].u32 = static_cast<uint32_t>(window_begin % plan.sliding_window);
            constants[6].u32 = static_cast<uint32_t>(position + 1 - window_begin);
            constants[7].u32 = selected ? current.selected_count : next_count;
            constants[8].u32 = selected ? 1 : 0;
            constants[9].f = 1.0f / std::sqrt(static_cast<float>(plan.head_dimension));
            ncnn::VkMat dispatcher;
            dispatcher.w = 128;
            dispatcher.h = 1;
            dispatcher.c = static_cast<int>(plan.head_count);
            // Parallelize the independent dot products, then retain the
            // canonical candidate order for online softmax/value accumulation.
            // Small candidate sets use the original core; workspace allocation
            // failure also keeps Attention on GPU without an added buffer.
            ncnn::VkMat scores;
            const uint32_t candidate_count = constants[6].u32 + constants[7].u32;
            if (candidate_count >= 8 && implementation.score_pipeline && implementation.scored_attention_pipeline
                && allocate(scores, static_cast<uint64_t>(plan.head_count) * candidate_count))
            {
                const std::vector<ncnn::VkMat> score_bindings = {bindings[0], bindings[1], bindings[2], bindings[3], scores};
                ncnn::VkMat score_dispatcher;
                score_dispatcher.w = static_cast<int>(std::min(candidate_count, 65535u) * 128);
                score_dispatcher.h = static_cast<int>((candidate_count + 65534) / 65535);
                score_dispatcher.c = static_cast<int>(plan.head_count);
                cmd.record_pipeline(implementation.score_pipeline.get(), score_bindings, constants, score_dispatcher);
                std::vector<ncnn::VkMat> scored_bindings = bindings;
                scored_bindings[0] = scores;
                cmd.record_pipeline(implementation.scored_attention_pipeline.get(), scored_bindings, constants, dispatcher);
                recording.storage.push_back(scores);
            }
            else
            {
                cmd.record_pipeline(implementation.attention_pipeline.get(), bindings, constants, dispatcher);
            }
        }
        state->token_count = position + 1;
        state->compressed_count = next_count;
    }
    ncnn::VkMat output_rank;
    if (!implementation.output_a->record_forward(attention, output_rank, cmd, recording.storage)
        || !implementation.output_b->record_forward(output_rank, output, cmd, recording.storage))
        return false;
    recording.storage.push_back(output_rank);
    recording.storage.push_back(output);
    struct ShadowView
    {
        ncnn::VkMat* value;
        uint64_t offset;
        uint32_t count;
    };
    std::vector<ShadowView> shadow_views;
    uint64_t shadow_count = 0;
    for (LatentRecordedRow_vulkan& row : recording.rows)
    {
        for (ncnn::VkMat* value : {&row.window_shadow, &row.compressor_shadow, &row.index_compressor_shadow,
                                   &row.compressed_shadow, &row.index_compressed_shadow, &row.selected_shadow})
        {
            if (value->empty())
                continue;
            const uint64_t count = value->total();
            if (count > std::numeric_limits<uint32_t>::max()
                || shadow_count + count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                return false;
            shadow_views.push_back({value, shadow_count, static_cast<uint32_t>(count)});
            shadow_count += count;
        }
    }
    ncnn::VkMat packed_shadow;
    if (!allocate(packed_shadow, shadow_count))
        return false;
    for (const ShadowView& view : shadow_views)
        if (!record_latent_copy(implementation.append_pipeline.get(), *view.value, packed_shadow, view.count, view.offset, cmd))
            return false;
    if (!prepare_staging_batch(recording.shadow_staging, 1, static_cast<uint32_t>(shadow_count), implementation.context->staging_allocator())
        || !record_prepared_activation_staging_download(latent_flat_view(packed_shadow, 0, static_cast<uint32_t>(shadow_count)),
                                                        1, static_cast<uint32_t>(shadow_count), recording.shadow_staging, cmd,
                                                        implementation.context->device(), implementation.opt))
        return false;
    for (const ShadowView& view : shadow_views)
        *view.value = latent_flat_view(recording.shadow_staging, view.offset, view.count);
    recording.storage.push_back(packed_shadow);
    ++recording.downloads;
    return true;
}
#endif

} // namespace moe
} // namespace ncnn
