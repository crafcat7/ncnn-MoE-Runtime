#include "backends/ncnn/layer/attention_vulkan.h"

#include "backends/ncnn/core/context.h"
#include "backends/ncnn/layer/attentiondecode_vulkan.h"
#include "backends/ncnn/layer/layernorm_vulkan.h"
#include "backends/ncnn/layer/rmsnorm_vulkan.h"
#include "backends/ncnn/linear/linear.h"
#include "kernels/ops.h"
#include "kernels/statecache.h"

#include "kernels/vulkan/attention_decode_sdpa.comp.hex.h"
#include "kernels/vulkan/attention_gqa_subgroup.comp.hex.h"
#include "kernels/vulkan/attention_output_gate.comp.hex.h"
#include "kernels/vulkan/attention_qkv_norm_rope.comp.hex.h"
#include "kernels/vulkan/attention_qkv_rope.comp.hex.h"
#include "kernels/vulkan/attention_ring_append.comp.hex.h"
#include "kernels/vulkan/attention_ring_zero.comp.hex.h"
#include "kernels/vulkan/attn_decode_gqa.comp.hex.h"
#include "kernels/vulkan/attn_decode_gqa_share.comp.hex.h"
#include "kernels/vulkan/attn_decode_gqa_vec.comp.hex.h"
#include "kernels/vulkan/attn_decode_merge.comp.hex.h"
#include "kernels/vulkan/attn_decode_split.comp.hex.h"
#include "kernels/vulkan/attn_prefill.comp.hex.h"
#include "kernels/vulkan/attn_prefill_cm2.comp.hex.h"
#include "kernels/vulkan/attn_prefill_cm2_merge.comp.hex.h"

#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <layer.h>
#include <layer_type.h>
#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>
#include <pipeline.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

// Longest cache the single-workgroup mask-free decode kernel scans without a
// split. Above split_min_tokens record_decode can split the scan across
// workgroups; neither path needs the additive mask plane.
static constexpr uint64_t decode_single_workgroup_tokens = 4096;

// KV rows per split workgroup and the widest split. Splitting starts well
// below the 4096-row single-workgroup ceiling so one-row decode keeps multiple
// resident workgroups per SM to hide KV latency at long context.
static constexpr uint64_t split_tokens = 64;
static constexpr uint64_t split_max = 256;
// Shared-KV scan carries several query heads per warp, so it needs more splits.
static constexpr uint64_t share_split_max = 1024;
static constexpr uint64_t split_min_tokens = 256;

AttnSplitRange split_range(uint64_t destination_count, bool available) noexcept
{
    AttnSplitRange range;
    if (!available || destination_count < split_min_tokens)
        return range;
    const uint64_t hint = std::min((destination_count + split_tokens - 1) / split_tokens, split_max);
    if (hint < 2)
        return range;
    const uint64_t per_split = (destination_count + hint - 1) / hint;
    range.tokens = (per_split + 15) / 16 * 16;
    range.splits = (destination_count + range.tokens - 1) / range.tokens;
    if (range.splits < 2)
        range = AttnSplitRange{};
    return range;
}

static AttnSplitRange compute_split_range(uint64_t destination_count, uint64_t min_tokens,
                                          uint64_t target_groups, uint64_t group_divisor,
                                          uint64_t max_splits, uint64_t token_align,
                                          bool require_two_splits) noexcept
{
    AttnSplitRange range;
    if (destination_count < min_tokens || group_divisor == 0)
        return range;
    const uint64_t splits = std::min<uint64_t>(max_splits,
                                               std::max<uint64_t>(2, target_groups / group_divisor));
    uint64_t per_split = (destination_count + splits - 1) / splits;
    per_split = (per_split + token_align - 1) / token_align * token_align;
    range.tokens = per_split;
    range.splits = (destination_count + per_split - 1) / per_split;
    if (require_two_splits && range.splits < 2)
        range = AttnSplitRange{};
    return range;
}

std::vector<uint64_t> Attention_vulkan::scratch_elements(const AttentionConfig_vulkan& config, uint32_t rows,
                                                         uint32_t context_tokens, uint32_t sequence_count,
                                                         uint64_t kv_element_size, bool direct_ring,
                                                         bool flash)
{
    const uint64_t query = saturating_multiply_u64(config.head_count, config.head_dimension);
    const uint64_t key = saturating_multiply_u64(config.kv_head_count, config.head_dimension);
    const bool gate = has_flag(config.flags, AttentionOutputGate);
    // Normalization, fused QKV[/gate], rotated query, attended values and the
    // projection before residual Add. Add's result belongs to the graph.
    std::vector<uint64_t> columns = {config.hidden_size,
                                     saturating_add_u64(query, saturating_add_u64(saturating_multiply_u64(key, 2), gate ? query : 0)),
                                     query, query, config.hidden_size, 1, 1}; // RMSNorm mean and coefficient.
    if (has_flag(config.flags, AttentionPostNorm))
        columns.push_back(config.hidden_size);
    if (!direct_ring)
        columns.insert(columns.end(), {key, key});
    if (gate)
        columns.push_back(query);
    if (rows >= 8)
    {
        // BF16 activation staging for CM2 GEMM (QKV input + O input, 2B/elem = half FP32).
        columns.push_back((config.hidden_size + 1u) / 2u);
        columns.push_back((query + 1u) / 2u);
    }
    const bool decode_sdpa = rows == 1
                             && (config.head_dimension <= 128
                                 || (static_cast<uint64_t>(context_tokens) + (has_flag(config.flags, AttentionSink) ? 1 : 0)
                                         > decode_single_workgroup_tokens
                                     && config.head_dimension <= 256))
                             && (config.sliding_window == 0 || context_tokens <= config.sliding_window)
                             && has_flag(config.optimization_flags, OptimizationVulkanDecodeSdpa);
    std::vector<uint64_t> elements;
    if (!decode_sdpa)
    {
        const uint64_t destinations = saturating_add_u64(context_tokens, has_flag(config.flags, AttentionSink) ? 1 : 0);
        if (!flash && kv_element_size == sizeof(uint16_t))
        {
            // General SDPA expands two KV operands per Session, with VkMat's
            // 16-byte channel alignment. Ring views themselves alias state.
            const uint64_t head_elements = saturating_multiply_u64(config.head_dimension, destinations);
            const uint64_t kv_elements = saturating_multiply_u64(saturating_add_u64(head_elements, 3) / 4 * 4,
                                                                 config.kv_head_count);
            for (uint64_t operand = 0; operand < static_cast<uint64_t>(std::min(rows, sequence_count)) * 2; ++operand)
                elements.push_back(kv_elements);
        }
        if (!flash)
        {
            const uint64_t plane = saturating_multiply_u64(destinations, config.head_count);
            columns.insert(columns.end(), {plane, plane, query}); // qk, mask, head-to-token permute.
            if (rows % 4 == 0)
                columns.push_back(query); // packed permute needs pack1 reshape.
#if !NCNN_BATCH
            else
                columns.push_back(query);
#endif
        }
    }
    else
    {
        // Split decode keeps one online-softmax state per split and one merged
        // output, both allocated from this arena. Reserve the widest split
        // because decode destinations outgrow the planned context before replanning.
        const uint64_t destinations = saturating_add_u64(context_tokens, has_flag(config.flags, AttentionSink) ? 1 : 0);
        if (split_range(destinations, true).splits != 0)
        {
            const uint64_t per_head_reserve = std::min<uint64_t>(split_max, std::max<uint64_t>(2, destinations / 8));
            const uint64_t fused_reserve = destinations >= 4096
                                               ? std::min<uint64_t>(share_split_max, (destinations + 3) / 4)
                                               : 0;
            const uint64_t reserve = std::max(per_head_reserve, fused_reserve);
            elements.push_back(saturating_multiply_u64(saturating_multiply_u64(reserve, config.head_count),
                                                       saturating_add_u64(config.head_dimension, 2)));
            elements.push_back(query);
        }
    }
    if (!has_flag(config.optimization_flags, OptimizationVulkanDeviceRope))
        columns.insert(columns.end(), {config.head_dimension, config.head_dimension});
    for (uint64_t columns_per_row : columns)
        elements.push_back(saturating_multiply_u64(columns_per_row, rows));
    return elements;
}

static bool create_staging(ncnn::VkMat& m, int w, int h, int c,
                           size_t elemsize, ncnn::VkAllocator* allocator)
{
    m.create(w, h, c, elemsize, allocator);
    return !m.empty() && m.mapped_ptr() != nullptr;
}

class Attention_vulkan::Implementation
{
public:
    ~Implementation()
    {
        if (vulkan_context)
        {
            norm.reset();
            layer_norm.reset();
            post_attention_norm.reset();
            qkv_rope_pipeline.reset();
            qkv_norm_rope_pipeline.reset();
            output_gate_pipeline.reset();
            output_gate_bf16_pipeline.reset();
            decode_sdpa_pipeline.reset();
            split_pipeline.reset();
            merge_pipeline.reset();
            gqa_pipeline.reset();
            gqa_vector_pipeline.reset();
            gqa_share_pipeline.reset();
            subgroup_pipeline.reset();
            decode_matrix.reset();
            flash_pipeline.reset();
            cm2_pipe[0].reset();
            cm2_pipe[1].reset();
            cm2_merge[0].reset();
            cm2_merge[1].reset();
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

    std::shared_ptr<RmsNorm_vulkan> norm;
    std::shared_ptr<LayerNorm_vulkan> layer_norm;
    std::shared_ptr<RmsNorm_vulkan> post_attention_norm;
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
    std::shared_ptr<ncnn::Pipeline> output_gate_bf16_pipeline;
    std::shared_ptr<ncnn::Pipeline> decode_sdpa_pipeline;
    // Split decode: one KV range per workgroup plus a merge pass.
    std::shared_ptr<ncnn::Pipeline> split_pipeline;
    std::shared_ptr<ncnn::Pipeline> merge_pipeline;
    // GQA split decode: one warp per query head, sharing partial layout/merge.
    std::shared_ptr<ncnn::Pipeline> gqa_pipeline;
    std::shared_ptr<ncnn::Pipeline> gqa_vector_pipeline;
    // Shared-KV GQA split decode: one warp scans one KV head across the group.
    std::shared_ptr<ncnn::Pipeline> gqa_share_pipeline;
    // Subgroup-scope cooperative-matrix decode (16-row tensor tiles).
    std::shared_ptr<ncnn::Pipeline> subgroup_pipeline;
    uint32_t device_parallelism = 0;
    // Multi-row flash attention with fp16 shared K/V tiles.
    std::shared_ptr<ncnn::Pipeline> flash_pipeline;
    // Workgroup-scope NV cooperative_matrix2 flash prefill (0: fp32 out, 1: bf16 out).
    std::shared_ptr<ncnn::Pipeline> cm2_pipe[2];
    std::shared_ptr<ncnn::Pipeline> cm2_merge[2];
    uint32_t cm2_br = 0;
    uint32_t cm2_bc = 0;
    bool cm2_ok = false;
    std::unique_ptr<AttentionDecodeVulkan> decode_matrix;
    AttentionDecodeCapabilities decode_capabilities;
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
    std::shared_ptr<Linear> fused_qkv;
    std::shared_ptr<QnkLinear_vulkan> fused_native_qkv;
    std::shared_ptr<Bfloat16Linear_vulkan> fused_qkv_gate;
    std::shared_ptr<Linear> output_projection;
    std::shared_ptr<Bfloat16Linear_vulkan> output_projection_bfloat16;
    std::array<CompiledOperator, 5> native_projections;
    // Decode override: the same projections fed from the retained QnK payload.
    std::array<CompiledOperator, 5> projections_q8;
    std::shared_ptr<QnkLinear_vulkan> fused_qkv_q8;
    bool q8 = false;
    bool flash_ok = false;
    AttentionConfig_vulkan config;
    bool query_key_norm = false;
    bool output_gate = false;
    std::vector<float> sinks;
    std::vector<float> rope_inverse_frequencies;
    ncnn::VkMat query_norm_weight;
    ncnn::VkMat key_norm_weight;
    float rope_concentration = 1.0f;
};

Attention_vulkan::Attention_vulkan()
    : d(new Implementation)
{
}

Attention_vulkan::~Attention_vulkan() = default;

static bool fill_rope_staging_pair(ncnn::VkMat& cosine_staging, ncnn::VkMat& sine_staging,
                                   size_t token_count, uint64_t position_offset,
                                   const std::vector<float>& inverse_frequencies,
                                   float concentration, bool bfloat16_storage,
                                   ncnn::VkAllocator* allocator)
{
    if (!prepare_staging_batch(cosine_staging, token_count,
                               static_cast<uint32_t>(inverse_frequencies.size()),
                               allocator, bfloat16_storage ? sizeof(uint16_t) : sizeof(float))
        || !prepare_staging_batch(sine_staging, token_count,
                                  static_cast<uint32_t>(inverse_frequencies.size()),
                                  allocator, bfloat16_storage ? sizeof(uint16_t) : sizeof(float)))
        return false;

    ncnn::Mat cosine_mapped = cosine_staging.mapped();
    ncnn::Mat sine_mapped = sine_staging.mapped();
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

static bool fill_attention_mask_staging(ncnn::VkMat& staging, size_t token_count,
                                        uint64_t destination_count, uint64_t position_offset,
                                        const LayerCache& cache, const AttentionConfig_vulkan& config,
                                        const std::vector<float>& sinks, bool bfloat16_storage,
                                        ncnn::VkAllocator* allocator)
{
    // Attention sinks are per head; without sinks a single channel broadcasts across all heads.
    const bool per_head_mask = has_flag(config.flags, AttentionSink);
    const int mask_channels = per_head_mask ? static_cast<int>(config.head_count) : 1;
    if (!create_staging(staging, static_cast<int>(destination_count), static_cast<int>(token_count),
                        mask_channels, bfloat16_storage ? sizeof(uint16_t) : sizeof(float), allocator))
        return false;

    // Finite sentinel avoids BF16 NaNs and still underflows after softmax.
    constexpr float masked_logit = -10000.0f;
    ncnn::Mat mapped = staging.mapped();
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
                const uint64_t key_position = key_index < cache.token_count
                                                  ? cache.start_position + key_index
                                                  : position_offset + key_index - cache.token_count;
                const bool future = key_position > query_position;
                const bool too_old = config.sliding_window > 0
                                     && key_position + config.sliding_window <= query_position;
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
                const uint64_t key_position = key_index < cache.token_count
                                                  ? cache.start_position + key_index
                                                  : position_offset + key_index - cache.token_count;
                const bool future = key_position > query_position;
                const bool too_old = config.sliding_window > 0
                                     && key_position + config.sliding_window <= query_position;
                row[key_index] = future || too_old ? masked_logit : 0.0f;
            }
            if (use_attention_sink)
                row[actual_end] = sinks[0];
            std::fill(row + actual_end + (use_attention_sink ? 1 : 0), row + destination_count, masked_logit);
        }
    }
    for (uint32_t head = 1; per_head_mask && head < config.head_count; ++head)
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

static bool attention_decode_window_is_visible(const LayerCache& cache, uint64_t query_position,
                                               uint64_t actual_token_count, uint32_t sliding_window) noexcept
{
    if (sliding_window == 0)
        return true;
    if (actual_token_count > sliding_window)
        return false;
    if (cache.token_count == 0)
        return true;
    if (cache.start_position > std::numeric_limits<uint64_t>::max() - cache.token_count)
        return false;
    // Decode SDPA skips the position mask; require contiguous prior history.
    return query_position == cache.start_position + cache.token_count;
}

static bool fill_promotion_stage(ncnn::VkMat& key_staging, ncnn::VkMat& value_staging,
                                 const LayerCache& cache, const AttentionConfig_vulkan& config,
                                 ncnn::VkAllocator* allocator)
{
    const uint32_t columns = config.kv_head_count * config.head_dimension;
    if (cache.token_count == 0
        || columns == 0
        || cache.token_count > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || config.kv_head_count > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || cache.capacity_tokens == 0
        || cache.first_slot >= cache.capacity_tokens
        || cache.token_count > cache.capacity_tokens
        || cache.columns != columns
        || (cache.dtype != DType::Float32 && cache.dtype != DType::BFloat16)
        || cache.capacity_tokens > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / columns))
        return false;

    const size_t capacity_elements = static_cast<size_t>(cache.capacity_tokens) * columns;
    const bool bfloat16 = cache.dtype == DType::BFloat16;
    if ((bfloat16 && (cache.bfloat16_keys.size() < capacity_elements || cache.bfloat16_values.size() < capacity_elements))
        || (!bfloat16 && (cache.keys.size() < capacity_elements || cache.values.size() < capacity_elements))
        || !create_staging(key_staging, static_cast<int>(config.head_dimension),
                           static_cast<int>(cache.token_count), static_cast<int>(config.kv_head_count),
                           sizeof(float), allocator)
        || !create_staging(value_staging, static_cast<int>(config.head_dimension),
                           static_cast<int>(cache.token_count), static_cast<int>(config.kv_head_count),
                           sizeof(float), allocator))
        return false;

    ncnn::Mat key_mapped = key_staging.mapped();
    ncnn::Mat value_mapped = value_staging.mapped();

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
                for (uint32_t column = 0; column < config.head_dimension; ++column)
                {
                    key_row[column] = bfloat16_to_float(cache.bfloat16_keys[source_offset + column]);
                    value_row[column] = bfloat16_to_float(cache.bfloat16_values[source_offset + column]);
                }
            }
            else
            {
                std::copy_n(cache.keys.data() + source_offset, config.head_dimension, key_row);
                std::copy_n(cache.values.data() + source_offset, config.head_dimension, value_row);
            }
        }
    }
    return true;
}

static uint64_t pipeline_opt_variant(const ncnn::Option& opt) noexcept
{
    return static_cast<uint64_t>(activation_variant(opt) + (opt.use_subgroup_ops ? 3u : 0u));
}

static const char* prefill_defines(uint32_t head_dimension) noexcept
{
    switch (head_dimension)
    {
    case 64u:
        return "#define ATTN_D 64u\n#define ATTN_M 64u\n#define ATTN_N 64u\n#define ATTN_ROWS 8u\n";
    case 128u:
        return "#define ATTN_D 128u\n#define ATTN_M 32u\n#define ATTN_N 32u\n#define ATTN_ROWS 4u\n";
    case 256u:
        return "#define ATTN_D 256u\n#define ATTN_M 16u\n#define ATTN_N 32u\n#define ATTN_ROWS 2u\n";
    default:
        return nullptr;
    }
}

static const char* gqa_defines(uint32_t head_dimension) noexcept
{
    switch (head_dimension)
    {
    case 64u:
        return "#define ATTN_HEAD_DIM 64u\n#define ATTN_TOKENS 8u\n";
    case 128u:
        return "#define ATTN_HEAD_DIM 128u\n#define ATTN_TOKENS 8u\n";
    case 256u:
        return "#define ATTN_HEAD_DIM 256u\n#define ATTN_TOKENS 4u\n";
    default:
        return nullptr;
    }
}

static const char* gqa_vec_defines(uint32_t head_dimension) noexcept
{
    switch (head_dimension)
    {
    case 128u:
        return "#define ATTN_HEAD_DIM 128u\n#define ATTN_TOKENS 8u\n";
    case 256u:
        return "#define ATTN_HEAD_DIM 256u\n#define ATTN_TOKENS 4u\n";
    default:
        return nullptr;
    }
}

static const char* gqa_share_defines(uint32_t head_dimension, uint32_t ratio) noexcept
{
    if (head_dimension != 64u)
        return nullptr;
    switch (ratio)
    {
    case 4u:
        return "#define HEAD_DIM 64\n#define RATIO 4\n#define Q_DIMS 8\n";
    case 7u:
        return "#define HEAD_DIM 64\n#define RATIO 7\n#define Q_DIMS 8\n";
    case 8u:
        return "#define HEAD_DIM 64\n#define RATIO 8\n#define Q_DIMS 8\n";
    default:
        return nullptr;
    }
}

// Subgroup-scope cooperative matrices use 16-row tiles (unlike 32-row workgroup tiles).
static bool supports_subgroup_matrix(const ncnn::GpuInfo& info, uint32_t m, uint32_t n, uint32_t k) noexcept
{
    if (!info.support_VK_KHR_cooperative_matrix() || !info.support_bf16_storage()
        || !info.support_VK_KHR_shader_bfloat16() || !info.support_VK_NV_cooperative_matrix2())
        return false;
    const auto& features = info.queryCooperativeMatrix2FeaturesNV();
    if (!features.cooperativeMatrixFlexibleDimensions || !features.cooperativeMatrixReductions
        || !features.cooperativeMatrixConversions || !features.cooperativeMatrixPerElementOperations
        || !features.cooperativeMatrixTensorAddressing)
        return false;
    for (const auto& property : info.queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV())
    {
        if (property.scope == VK_SCOPE_SUBGROUP_KHR
            && property.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR
            && property.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR
            && property.CType == VK_COMPONENT_TYPE_FLOAT32_KHR
            && property.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
            && property.MGranularity != 0 && property.NGranularity != 0 && property.KGranularity != 0
            && m % property.MGranularity == 0 && n % property.NGranularity == 0
            && k % property.KGranularity == 0)
            return true;
    }
    return false;
}

static bool supports_workgroup_cm2_matrix(const ncnn::GpuInfo& info, uint32_t workgroup_size,
                                          uint32_t m, uint32_t n, uint32_t k) noexcept
{
    if (!info.support_VK_KHR_cooperative_matrix() || !info.support_bf16_storage()
        || !info.support_VK_KHR_shader_bfloat16() || !info.support_bf16_cooperative_matrix()
        || !info.support_VK_NV_cooperative_matrix2())
        return false;
    const auto& features = info.queryCooperativeMatrix2FeaturesNV();
    const auto& properties = info.queryCooperativeMatrix2PropertiesNV();
    if (!features.cooperativeMatrixWorkgroupScope
        || !features.cooperativeMatrixFlexibleDimensions
        || !features.cooperativeMatrixReductions
        || !features.cooperativeMatrixConversions
        || !features.cooperativeMatrixPerElementOperations
        || !features.cooperativeMatrixTensorAddressing
        || properties.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize < workgroup_size
        || properties.cooperativeMatrixWorkgroupScopeReservedSharedMemory > info.max_shared_memory_size()
        || properties.cooperativeMatrixFlexibleDimensionsMaxDimension < std::max({m, n, k})
        || info.max_workgroup_invocations() < workgroup_size
        || info.max_workgroup_size_x() < workgroup_size
        || (info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_BASIC_BIT) == 0)
        return false;
    for (const auto& property : info.queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV())
    {
        if (property.scope == VK_SCOPE_WORKGROUP_KHR
            && property.workgroupInvocations == workgroup_size
            && property.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR
            && property.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR
            && property.CType == VK_COMPONENT_TYPE_FLOAT32_KHR
            && property.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
            && property.MGranularity != 0 && property.NGranularity != 0 && property.KGranularity != 0
            && m % property.MGranularity == 0 && n % property.NGranularity == 0
            && k % property.KGranularity == 0)
            return true;
    }
    return false;
}

static bool create_prefill_cm2_pipeline(const std::shared_ptr<VulkanContext>& context,
                                        uint32_t br, uint32_t bc, uint32_t head_dimension,
                                        bool kv_bfloat16, bool has_window,
                                        std::shared_ptr<ncnn::Pipeline>& destination,
                                        bool out_bfloat16 = false)
{
    const uint64_t variant = static_cast<uint64_t>(head_dimension)
                             | (static_cast<uint64_t>(br) << 12)
                             | (static_cast<uint64_t>(bc) << 24)
                             | (static_cast<uint64_t>(kv_bfloat16 ? 1u : 0u) << 36)
                             | (static_cast<uint64_t>(out_bfloat16 ? 1u : 0u) << 37)
                             | (static_cast<uint64_t>(has_window ? 1u : 0u) << 38);
    ncnn::Option compile_option = make_vk_opt(*context);
    compile_option.use_bf16_storage = true;
    std::vector<ncnn::vk_specialization_type> specs(5);
    specs[0].u32 = br;
    specs[1].u32 = bc;
    specs[2].u32 = head_dimension;
    specs[3].u32 = kv_bfloat16 ? 1u : 0u;
    specs[4].u32 = has_window ? 1u : 0u;
    const int base_len = static_cast<int>(sizeof(attn_prefill_cm2_shader) - 1);
    const int subgroup_size = static_cast<int>(context->device()->info.subgroup_size());
    if (out_bfloat16)
    {
        return create_pipeline_prefixed(context, attn_prefill_cm2_shader, base_len,
                                        "#define NCNN_ATTN_CM2_BF16_OUT\n", compile_option,
                                        0, variant, 128, destination, specs, subgroup_size);
    }
    return create_pipeline(context, attn_prefill_cm2_shader, base_len, compile_option,
                           0, variant, 128, destination, specs, subgroup_size);
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
    const bool gated = implementation.output_gate;
    const uint64_t total_columns = query_columns + key_value_columns * 2 + (gated ? query_columns : 0);
    if (token_count > static_cast<size_t>(int_max) || total_columns > int_max)
        return false;

    const uint64_t heads = head_count + kv_head_count;
    const uint64_t half_dimension = head_dimension / 2;
    const uint32_t rope_dimension = config.rope_head_dimension == 0
                                        ? config.head_dimension
                                        : config.rope_head_dimension;
    const bool norm_rope = implementation.query_key_norm
                           || implementation.output_gate
                           || rope_dimension != config.head_dimension;
    const uint64_t work_items_per_token = norm_rope
                                              ? heads
                                              : heads * half_dimension + key_value_columns;
    if (work_items_per_token == 0)
        return false;

    uint64_t work_limit = int_max;
    if (norm_rope)
    {
        const ncnn::Pipeline* pipeline = implementation.qkv_norm_rope_pipeline.get();
        if (!pipeline || pipeline->local_size_x() == 0)
            return false;
        work_limit /= pipeline->local_size_x();
    }
    return static_cast<uint64_t>(token_count) <= work_limit / work_items_per_token;
}

bool Attention_vulkan::record_native_qkv(const ncnn::VkMat& input,
                                         ncnn::VkMat& output,
                                         ncnn::VkCompute& command,
                                         std::vector<ncnn::VkMat>& workspace) const
{
    const Implementation& implementation = *d;
    const AttentionConfig_vulkan& config = implementation.config;
    // One-row decode streams block-quantized weights; multi-row prefill uses BF16 tile GEMM.
    const bool decode_q8 = input.h == 1 && implementation.q8;
    const auto& fused_qkv = decode_q8 ? implementation.fused_qkv_q8 : implementation.fused_native_qkv;
    const auto& projections = decode_q8 ? implementation.projections_q8 : implementation.native_projections;
    const uint64_t query_columns = static_cast<uint64_t>(config.head_count) * config.head_dimension;
    const uint64_t key_value_columns = static_cast<uint64_t>(config.kv_head_count) * config.head_dimension;
    const uint64_t total_columns = query_columns + key_value_columns * 2
                                   + (implementation.output_gate ? query_columns : 0);
    output.create(static_cast<int>(total_columns), input.h, sizeof(float), 1,
                  implementation.option.blob_vkallocator);
    if (output.empty())
        return false;

    const uint32_t query_offset = static_cast<uint32_t>(query_columns);
    const uint32_t value_offset = static_cast<uint32_t>(query_columns + key_value_columns);
    std::vector<ncnn::VkMat> projection_bindings;
    std::vector<ncnn::vk_constant_type> projection_constants;
    const auto record_projection = [&](const CompiledOperator& projection, uint32_t offset) {
        if (projection.qnk)
        {
            projection.qnk->record(input, output, command, offset, projection_bindings,
                                   projection_constants);
            return true;
        }
        return projection.float8
               && projection.float8->record_forward(input, output, command, workspace,
                                                    offset, static_cast<uint32_t>(total_columns));
    };
    if (fused_qkv)
        fused_qkv->record(input, output, command, 0u, projection_bindings, projection_constants);
    else if (!record_projection(projections[0], 0)
             || !record_projection(projections[1], query_offset)
             || !record_projection(projections[2], value_offset))
        return false;
    if (implementation.output_gate)
    {
        if (!record_projection(projections[4],
                               static_cast<uint32_t>(query_columns + key_value_columns * 2)))
            return false;
    }
    return true;
}

bool Attention_vulkan::record_qkv_rope(const ncnn::VkMat& fused_qkv,
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
    const size_t key_value_element_size = act_element_size(implementation.kv_option);
    ncnn::VkAllocator* allocator = implementation.option.blob_vkallocator;
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    const uint64_t total_columns = static_cast<uint64_t>(query_columns) + static_cast<uint64_t>(key_value_columns) * 2;
    const uint64_t half_dimension = config.head_dimension / 2;
    const uint64_t work_items = static_cast<uint64_t>(token_count)
                                * (static_cast<uint64_t>(config.head_count) * half_dimension
                                   + static_cast<uint64_t>(config.kv_head_count) * half_dimension + key_value_columns);
    const bool direct_ring = ring != nullptr;
    if (!pipeline || token_count == 0)
        return false;
    if (ring && (ring_capacity == 0 || destination_start >= ring_capacity || token_count > ring_capacity))
        return false;

    query.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count),
                 static_cast<int>(config.head_count), sizeof(float), 1, allocator);
    if (direct_ring)
    {
        key = ring->key;
        value = ring->value;
    }
    else
    {
        key.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count),
                   static_cast<int>(config.kv_head_count), key_value_element_size, 1, allocator);
        value.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count),
                     static_cast<int>(config.kv_head_count), key_value_element_size, 1, allocator);
    }
    if (query.empty() || key.empty() || value.empty())
        return false;

    const std::vector<ncnn::VkMat> bindings = {
        fused_qkv,
        cosine,
        sine,
        query,
        key,
        value,
    };
    std::vector<ncnn::vk_constant_type> constants(17);
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
    constants[15].u32 = config.kv_dtype == DType::BFloat16 ? 1 : 0;
    constants[16].u32 = config.sliding_window != 0 ? 1 : 0;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(work_items);
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

bool Attention_vulkan::record_qkv_norm_rope(const ncnn::VkMat& fused_qkv,
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
    const ncnn::VkMat& query_norm = implementation.query_norm_weight.empty()
                                        ? fused_qkv
                                        : implementation.query_norm_weight;
    const ncnn::VkMat& key_norm = implementation.key_norm_weight.empty()
                                      ? fused_qkv
                                      : implementation.key_norm_weight;
    const AttentionConfig_vulkan& config = implementation.config;
    const float rope_concentration = implementation.rope_concentration;
    const size_t key_value_element_size = act_element_size(implementation.kv_option);
    ncnn::VkAllocator* allocator = implementation.option.blob_vkallocator;
    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t key_value_columns = config.kv_head_count * config.head_dimension;
    const uint64_t total_columns = static_cast<uint64_t>(query_columns)
                                   + static_cast<uint64_t>(key_value_columns) * 2
                                   + (implementation.output_gate ? query_columns : 0);
    const uint32_t rope_dimension = config.rope_head_dimension == 0
                                        ? config.head_dimension
                                        : config.rope_head_dimension;
    const uint64_t work_items = static_cast<uint64_t>(token_count)
                                * (static_cast<uint64_t>(config.head_count)
                                   + config.kv_head_count);
    const bool direct_ring = ring != nullptr;
    if (!pipeline || token_count == 0)
        return false;
    if (ring && (ring_capacity == 0 || destination_start >= ring_capacity || token_count > ring_capacity))
        return false;

    query.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count),
                 static_cast<int>(config.head_count), sizeof(float), 1, allocator);
    if (implementation.output_gate)
    {
        gate.create(static_cast<int>(query_columns), static_cast<int>(token_count),
                    sizeof(float), 1, allocator);
    }
    else
    {
        gate = query;
    }
    if (direct_ring)
    {
        key = ring->key;
        value = ring->value;
    }
    else
    {
        key.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count),
                   static_cast<int>(config.kv_head_count), key_value_element_size, 1, allocator);
        value.create(static_cast<int>(config.head_dimension), static_cast<int>(token_count),
                     static_cast<int>(config.kv_head_count), key_value_element_size, 1, allocator);
    }
    if (query.empty() || key.empty() || value.empty() || gate.empty())
        return false;

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
    std::vector<ncnn::vk_constant_type> constants(25);
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
    constants[19].u32 = implementation.query_key_norm ? 1 : 0;
    constants[20].u32 = implementation.output_gate ? 1 : 0;
    constants[21].u32 = config.kv_dtype == DType::BFloat16 ? 1 : 0;
    constants[22].u32 = pipeline->local_size_x();
    constants[23].u32 = has_flag(config.flags, AttentionValueNorm) ? 1u : 0u;
    constants[24].u32 = config.sliding_window != 0 ? 1 : 0;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(work_items * pipeline->local_size_x());
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static bool record_attention_output_gate(const ncnn::Pipeline* pipeline,
                                         ncnn::VkMat& attention,
                                         const ncnn::VkMat& gate,
                                         size_t token_count,
                                         uint32_t columns,
                                         ncnn::VkCompute& command)
{
    const uint64_t elements = static_cast<uint64_t>(token_count) * columns;
    if (!pipeline || attention.empty() || gate.empty()
        || elements == 0
        || gate.dims != 2 || gate.w != static_cast<int>(columns)
        || gate.h != static_cast<int>(token_count)
        || gate.elempack != 1 || gate.elemsize != sizeof(float)
        || attention.elempack != 1 || attention.elemsize != sizeof(float)
        || attention.total() * attention.elempack < elements)
        return false;

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

// Multi-row flash attention. K/V tiles stage in fp16 shared memory and output
// is written directly in token-major (query_columns, rows) layout.
static bool record_flash(const ncnn::Pipeline* pipeline, const ncnn::VkMat& query,
                         const ncnn::VkMat& key, const ncnn::VkMat& value,
                         const AttentionConfig_vulkan& config,
                         uint64_t query_count, uint64_t destination_count,
                         uint64_t position_offset, uint64_t cache_start_position,
                         ncnn::VkMat& output, ncnn::VkCompute& command,
                         ncnn::VkAllocator* allocator)
{
    const uint64_t head_dimension = config.head_dimension;
    const uint32_t query_tile = head_dimension == 256u ? 16u : (head_dimension == 128u ? 32u : 64u);
    const uint64_t query_columns = static_cast<uint64_t>(config.head_count) * head_dimension;

    output.create(static_cast<int>(query_columns), static_cast<int>(query_count), sizeof(float), 1, allocator);
    if (output.empty())
        return false;

    const std::vector<ncnn::VkMat> bindings = {query, key, value, output};
    std::vector<ncnn::vk_constant_type> constants(12);
    constants[0].u32 = static_cast<uint32_t>(query_count);
    constants[1].u32 = static_cast<uint32_t>(destination_count);
    constants[2].u32 = config.head_count;
    constants[3].u32 = config.kv_head_count;
    constants[4].u32 = static_cast<uint32_t>(query.cstep);
    constants[5].u32 = static_cast<uint32_t>(key.cstep);
    constants[6].u32 = static_cast<uint32_t>(value.cstep);
    constants[7].u32 = static_cast<uint32_t>(query_columns);
    constants[8].u32 = static_cast<uint32_t>(position_offset);
    constants[9].u32 = static_cast<uint32_t>(cache_start_position);
    constants[10].u32 = config.sliding_window;
    constants[11].f = config.attn_scale;

    ncnn::VkMat dispatcher;
    constexpr int workgroup = 256;
    dispatcher.w = static_cast<int>((query_count + query_tile - 1u) / query_tile) * workgroup;
    dispatcher.h = static_cast<int>(config.head_count);
    dispatcher.c = 1;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

// Workgroup-scope NV cooperative_matrix2 flash prefill with optional split-K.
static bool record_prefill_cm2(const ncnn::Pipeline* pipeline,
                               const ncnn::Pipeline* merge_pipeline,
                               uint32_t br, uint32_t bc, uint32_t device_parallelism,
                               VulkanContext* context,
                               const ncnn::VkMat& query, const ncnn::VkMat& key, const ncnn::VkMat& value,
                               const AttentionConfig_vulkan& config,
                               uint64_t query_count, uint64_t destination_count,
                               uint64_t position_offset, uint64_t cache_start_position,
                               ncnn::VkMat& output, ncnn::VkCompute& command,
                               ncnn::VkAllocator* allocator, bool out_bfloat16 = false)
{
    if (br == 0 || bc == 0 || query_count < 8 || destination_count == 0
        || config.kv_head_count == 0 || config.head_count % config.kv_head_count != 0)
        return false;
    if (config.kv_dtype == DType::BFloat16 && ((key.cstep & 1u) != 0 || (value.cstep & 1u) != 0))
        return false;
    const uint64_t query_columns = static_cast<uint64_t>(config.head_count) * config.head_dimension;

    output.create(static_cast<int>(query_columns), static_cast<int>(query_count),
                  out_bfloat16 ? sizeof(uint16_t) : sizeof(float), 1, allocator);
    if (output.empty())
        return false;

    const uint32_t group_size = config.head_count / config.kv_head_count;
    const uint64_t query_tiles = (query_count + br - 1u) / br;
    const uint64_t base_wgs = static_cast<uint64_t>(config.head_count) * query_tiles;

    uint32_t split_count = 1u;
    if (merge_pipeline && context && destination_count >= 4096u)
    {
        const uint64_t target_wgs = std::max<uint64_t>(140u, static_cast<uint64_t>(device_parallelism) * 4u)
                                    * (destination_count >= 16384u ? 2u : 1u);
        if (base_wgs < target_wgs)
        {
            uint32_t candidate = 1u;
            while (candidate < 8u && base_wgs * candidate < target_wgs
                   && destination_count / (candidate * 2u) >= 1024u)
                candidate *= 2u;
            split_count = candidate;
        }
    }

    ncnn::VkMat partial = output;
    if (split_count > 1u)
    {
        const uint64_t partial_elements = static_cast<uint64_t>(split_count)
                                          * config.head_count
                                          * query_count
                                          * (config.head_dimension + 2u);
        partial = context->cm2_scratch(static_cast<int>(partial_elements), 0, sizeof(float));
        if (partial.empty())
        {
            split_count = 1u;
            partial = output;
        }
    }

    std::vector<ncnn::vk_constant_type> constants(13);
    constants[0].u32 = static_cast<uint32_t>(query_count);
    constants[1].u32 = static_cast<uint32_t>(destination_count);
    constants[2].u32 = config.head_count;
    constants[3].u32 = config.kv_head_count;
    constants[4].u32 = static_cast<uint32_t>(query.cstep);
    constants[5].u32 = static_cast<uint32_t>(key.cstep);
    constants[6].u32 = static_cast<uint32_t>(value.cstep);
    constants[7].u32 = static_cast<uint32_t>(query_columns);
    constants[8].u32 = static_cast<uint32_t>(position_offset);
    constants[9].u32 = static_cast<uint32_t>(cache_start_position);
    constants[10].u32 = config.sliding_window;
    constants[11].f = config.attn_scale;
    constants[12].u32 = split_count;

    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(group_size * split_count * 128u);
    dispatcher.h = static_cast<int>(query_tiles);
    dispatcher.c = static_cast<int>(config.kv_head_count);
    command.record_pipeline_readonly(pipeline, {query, key, value, output, partial},
                                     {1, 1, 1, 0, 0}, constants, dispatcher);

    if (split_count > 1u)
    {
        constants.resize(5);
        constants[0].u32 = static_cast<uint32_t>(query_count);
        constants[1].u32 = config.head_count;
        constants[2].u32 = config.head_dimension;
        constants[3].u32 = static_cast<uint32_t>(query_columns);
        constants[4].u32 = split_count;
        ncnn::VkMat merge_dispatcher;
        merge_dispatcher.w = static_cast<int>(query_count * 128u);
        merge_dispatcher.h = static_cast<int>(config.head_count);
        merge_dispatcher.c = 1;
        command.record_pipeline_readonly(merge_pipeline, {partial, output}, {1, 0},
                                         constants, merge_dispatcher);
    }
    return true;
}

static bool record_attention_decode_sdpa(const ncnn::Pipeline* pipeline, const ncnn::VkMat& query,
                                         const ncnn::VkMat& key, const ncnn::VkMat& value,
                                         const ncnn::VkMat& sinks, const AttentionConfig_vulkan& config,
                                         uint64_t destination_count, ncnn::VkMat& output,
                                         ncnn::VkCompute& command, ncnn::VkAllocator* allocator)
{
    const uint64_t query_columns = static_cast<uint64_t>(config.head_count) * config.head_dimension;
    if (!pipeline)
        return false;

    output.create(static_cast<int>(query_columns), 1, sizeof(float), 1, allocator);
    if (output.empty())
        return false;

    const std::vector<ncnn::VkMat> bindings = {query, key, value, sinks, output};
    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].u32 = config.head_dimension;
    constants[1].u32 = static_cast<uint32_t>(destination_count);
    constants[2].u32 = config.head_count;
    constants[3].u32 = config.kv_head_count;
    constants[4].u32 = static_cast<uint32_t>(query.cstep);
    constants[5].u32 = static_cast<uint32_t>(key.cstep);
    constants[6].u32 = static_cast<uint32_t>(value.cstep);
    constants[7].u32 = has_flag(config.flags, AttentionSink) ? 1 : 0;
    constants[8].f = config.attn_scale;
    ncnn::VkMat dispatcher;
    dispatcher.w = 256;
    dispatcher.h = 1;
    dispatcher.c = static_cast<int>(config.head_count);
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static bool record_split_decode(const ncnn::Pipeline* split_pipeline, const ncnn::Pipeline* merge_pipeline,
                                const ncnn::VkMat& query, const ncnn::VkMat& key, const ncnn::VkMat& value,
                                const ncnn::VkMat& sinks, const AttentionConfig_vulkan& config,
                                uint64_t destination_count, const AttnSplitRange& range,
                                int dispatch_w, int dispatch_h, int dispatch_c,
                                ncnn::VkMat& output, ncnn::VkCompute& command,
                                ncnn::VkAllocator* allocator)
{
    if (!split_pipeline || !merge_pipeline || range.splits < 2)
        return false;

    const uint64_t row_elements = config.head_dimension + 2;
    const uint64_t partial_elements = range.splits * config.head_count * row_elements;
    ncnn::VkMat partial;
    partial.create(static_cast<int>(partial_elements), sizeof(float), 1, allocator);
    output.create(static_cast<int>(static_cast<uint64_t>(config.head_count) * config.head_dimension),
                  1, sizeof(float), 1, allocator);
    if (partial.empty() || output.empty())
        return false;

    std::vector<ncnn::vk_constant_type> constants(11);
    constants[0].u32 = config.head_dimension;
    constants[1].u32 = static_cast<uint32_t>(destination_count);
    constants[2].u32 = config.head_count;
    constants[3].u32 = config.kv_head_count;
    constants[4].u32 = static_cast<uint32_t>(query.cstep);
    constants[5].u32 = static_cast<uint32_t>(key.cstep);
    constants[6].u32 = static_cast<uint32_t>(value.cstep);
    constants[7].u32 = has_flag(config.flags, AttentionSink) ? 1u : 0u;
    constants[8].f = config.attn_scale;
    constants[9].u32 = static_cast<uint32_t>(range.splits);
    constants[10].u32 = static_cast<uint32_t>(range.tokens);
    ncnn::VkMat dispatcher;
    dispatcher.w = dispatch_w;
    dispatcher.h = dispatch_h;
    dispatcher.c = dispatch_c;
    command.record_pipeline(split_pipeline, {query, key, value, sinks, partial}, constants, dispatcher);

    std::vector<ncnn::vk_constant_type> merge_constants(3);
    merge_constants[0].u32 = config.head_dimension;
    merge_constants[1].u32 = config.head_count;
    merge_constants[2].u32 = static_cast<uint32_t>(range.splits);
    dispatcher.w = static_cast<int>(config.head_count * 256);
    dispatcher.h = 1;
    dispatcher.c = 1;
    command.record_pipeline(merge_pipeline, {partial, output}, merge_constants, dispatcher);
    return true;
}

static uint64_t next_attention_ring_capacity(uint64_t current, uint64_t required) noexcept
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

static bool checked_multiply_u64(uint64_t left, uint64_t right, uint64_t& product) noexcept
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
        return false;
    product = left * right;
    return true;
}

static bool attention_promotion_within_budget(const VulkanContext& context, uint64_t ring_capacity,
                                              uint64_t kv_columns, size_t element_size,
                                              uint64_t transfer_bytes) noexcept
{
    uint64_t ring_elements = 0;
    uint64_t ring_bytes = 0;
    if (!checked_multiply_u64(ring_capacity, kv_columns, ring_elements)
        || !checked_multiply_u64(ring_elements, element_size * 4, ring_bytes)
        || transfer_bytes > std::numeric_limits<uint64_t>::max() - ring_bytes)
        return false;

    const ncnn::VulkanDevice* device = context.device();
    if (!device)
        return false;
    const uint64_t heap_budget = static_cast<uint64_t>(device->get_heap_budget()) * 1024 * 1024;
    // Keep opportunistic promotion to a small per-layer share; larger caches stay on CPU.
    constexpr uint64_t maximum_per_layer_working_set = 32ull * 1024 * 1024;
    const uint64_t admission_bytes = std::min(maximum_per_layer_working_set, heap_budget / 256);
    return admission_bytes != 0 && ring_bytes + transfer_bytes <= admission_bytes;
}

static bool create_attention_ring_storage(AttentionCache_vulkan& cache, uint32_t width,
                                          uint32_t channels, uint64_t capacity,
                                          size_t element_size, bool double_write,
                                          ncnn::VkAllocator* allocator)
{
    if (capacity == 0
        || capacity > static_cast<uint64_t>(std::numeric_limits<int>::max()) / 2
        || (element_size != sizeof(float) && element_size != sizeof(uint16_t)))
        return false;
    const int rows = static_cast<int>(capacity * (double_write ? 2 : 1));
    cache.key.create(static_cast<int>(width), rows, static_cast<int>(channels), element_size, 1, allocator);
    cache.value.create(static_cast<int>(width), rows, static_cast<int>(channels), element_size, 1, allocator);
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

// Copy strided KV rows; append mode also writes duplicate ring rows.
static bool record_attention_copy(const ncnn::Pipeline* pipeline, const ncnn::VkMat& source_key,
                                  const ncnn::VkMat& source_value, ncnn::VkMat& destination_key,
                                  ncnn::VkMat& destination_value, uint64_t capacity,
                                  uint64_t destination_start, bool round_kv, uint64_t source_start,
                                  bool copy_only, bool double_write, ncnn::VkCompute& command)
{
    const bool source_fp32 = source_key.elemsize == sizeof(float) && destination_key.elemsize == sizeof(uint16_t);
    const bool destination_fp32 = copy_only && source_key.elemsize == sizeof(uint16_t)
                                  && destination_key.elemsize == sizeof(float);
    if (!pipeline || capacity == 0 || destination_start >= capacity)
        return false;

    const std::vector<ncnn::VkMat> bindings = {
        source_key,
        source_value,
        destination_fp32 ? source_key : destination_key,
        destination_fp32 ? source_value : destination_value,
        destination_fp32 ? destination_key : source_key,
        destination_fp32 ? destination_value : source_value,
    };
    const uint32_t rows = static_cast<uint32_t>(copy_only ? destination_key.h : source_key.h);
    std::vector<ncnn::vk_constant_type> constants(13);
    constants[0].u32 = static_cast<uint32_t>(source_key.w);
    constants[1].u32 = rows;
    constants[2].u32 = static_cast<uint32_t>(source_key.c);
    constants[3].u32 = static_cast<uint32_t>(source_key.cstep);
    constants[4].u32 = static_cast<uint32_t>(destination_key.cstep);
    constants[5].u32 = static_cast<uint32_t>(capacity);
    constants[6].u32 = static_cast<uint32_t>(destination_start);
    constants[7].u32 = round_kv ? 1 : 0;
    constants[8].u32 = source_fp32 ? 1 : 0;
    constants[9].u32 = static_cast<uint32_t>(source_start);
    constants[10].u32 = copy_only ? 1 : 0;
    constants[11].u32 = destination_fp32 ? 1 : 0;
    constants[12].u32 = double_write ? 1 : 0;
    ncnn::VkMat dispatcher;
    dispatcher.w = source_key.w;
    dispatcher.h = static_cast<int>(rows);
    dispatcher.c = source_key.c;
    command.record_pipeline(pipeline, bindings, constants, dispatcher);
    return true;
}

static bool record_attention_ring_zero(const ncnn::Pipeline* pipeline, ncnn::VkMat& destination_key,
                                       ncnn::VkMat& destination_value, uint64_t destination_row,
                                       ncnn::VkCompute& command)
{
    if (!pipeline || destination_key.empty() || destination_value.empty()
        || (destination_key.elemsize != sizeof(float) && destination_key.elemsize != sizeof(uint16_t))
        || destination_value.elemsize != destination_key.elemsize
        || destination_row >= static_cast<uint64_t>(destination_key.h))
        return false;

    const std::vector<ncnn::VkMat> bindings = {destination_key, destination_value};
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

static bool create_vulkan_layer(int type, const ncnn::ParamDict& parameters, ncnn::VulkanDevice* device,
                                const ncnn::Option& option, std::vector<ncnn::Layer*>& layers,
                                ncnn::Layer*& destination)
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

std::shared_ptr<Attention_vulkan> Attention_vulkan::create(const TensorData& norm_weight,
                                                           const TensorData* sinks,
                                                           std::shared_ptr<Linear> fused_qkv,
                                                           std::shared_ptr<Linear> output_projection,
                                                           const AttentionConfig_vulkan& config,
                                                           const TensorData* norm_bias)
{
    return create_impl(norm_weight, nullptr, nullptr, sinks, std::move(fused_qkv),
                       std::move(output_projection), {}, {}, {}, config, {}, norm_bias);
}

std::shared_ptr<Attention_vulkan> Attention_vulkan::create(const TensorData& norm_weight,
                                                           const TensorData* sinks,
                                                           std::shared_ptr<Bfloat16Linear_vulkan> fused_qkv,
                                                           std::shared_ptr<Bfloat16Linear_vulkan> output_projection,
                                                           const AttentionConfig_vulkan& config,
                                                           const TensorData* norm_bias)
{
    return create_impl(norm_weight, nullptr, nullptr, sinks, {}, {}, std::move(fused_qkv),
                       std::move(output_projection), {}, config, {}, norm_bias);
}

std::shared_ptr<Attention_vulkan> Attention_vulkan::create(const TensorData& norm_weight,
                                                           const TensorData& query_norm_weight,
                                                           const TensorData& key_norm_weight,
                                                           const TensorData* sinks,
                                                           std::shared_ptr<Bfloat16Linear_vulkan> fused_qkv_gate,
                                                           std::shared_ptr<Bfloat16Linear_vulkan> output_projection,
                                                           const AttentionConfig_vulkan& config,
                                                           const TensorData* norm_bias)
{
    return create_impl(norm_weight, &query_norm_weight, &key_norm_weight, sinks, {}, {},
                       std::move(fused_qkv_gate), std::move(output_projection), {}, config, {}, norm_bias);
}

std::shared_ptr<Attention_vulkan> Attention_vulkan::create_native(const TensorData& norm_weight,
                                                                  const TensorData* query_norm_weight,
                                                                  const TensorData* key_norm_weight,
                                                                  const TensorData* sinks,
                                                                  const std::array<CompiledOperator, 5>& projections,
                                                                  const AttentionConfig_vulkan& config,
                                                                  std::shared_ptr<QnkLinear_vulkan> fused_native_qkv,
                                                                  const TensorData* norm_bias,
                                                                  const TensorData* attn_post_norm,
                                                                  const std::array<CompiledOperator, 5>* projections_q8,
                                                                  std::shared_ptr<QnkLinear_vulkan> fused_qkv_q8)
{
    return create_impl(norm_weight, query_norm_weight, key_norm_weight, sinks, {}, {}, {}, {},
                       projections, config, std::move(fused_native_qkv), norm_bias, attn_post_norm,
                       projections_q8, std::move(fused_qkv_q8));
}

std::shared_ptr<Attention_vulkan> Attention_vulkan::create_impl(const TensorData& norm_weight,
                                                                const TensorData* query_norm_weight,
                                                                const TensorData* key_norm_weight,
                                                                const TensorData* sinks,
                                                                std::shared_ptr<Linear> fused_qkv,
                                                                std::shared_ptr<Linear> output_projection,
                                                                std::shared_ptr<Bfloat16Linear_vulkan> fused_qkv_gate,
                                                                std::shared_ptr<Bfloat16Linear_vulkan> output_projection_bfloat16,
                                                                std::array<CompiledOperator, 5> native_projections,
                                                                const AttentionConfig_vulkan& config,
                                                                std::shared_ptr<QnkLinear_vulkan> fused_native_qkv,
                                                                const TensorData* norm_bias,
                                                                const TensorData* attn_post_norm,
                                                                const std::array<CompiledOperator, 5>* projections_q8,
                                                                std::shared_ptr<QnkLinear_vulkan> fused_qkv_q8)
{
#if !NCNN_BATCH
    // Wrapped KV rings require VkMat offset views.
    return {};
#endif
    const bool use_native = fused_native_qkv
                            || std::any_of(native_projections.begin(), native_projections.end(),
                                           [](const auto& projection) { return projection.qnk || projection.float8; });
    const bool use_bfloat16 = fused_qkv_gate || output_projection_bfloat16;
    const bool query_key_norm = query_norm_weight != nullptr;
    const bool output_gate = has_flag(config.flags, AttentionOutputGate);
    const uint32_t rope_dimension = config.rope_head_dimension == 0
                                        ? config.head_dimension
                                        : config.rope_head_dimension;
    if ((query_key_norm && !key_norm_weight)
        || (has_flag(config.flags, AttentionSink) && (!sinks || sinks->element_count() != config.head_count))
        || config.head_dimension == 0 || rope_dimension > config.head_dimension)
        return {};

    const uint64_t query_columns64 = static_cast<uint64_t>(config.head_count) * config.head_dimension;
    const uint64_t key_value_columns64 = static_cast<uint64_t>(config.kv_head_count) * config.head_dimension;
    const uint32_t query_columns = static_cast<uint32_t>(query_columns64);
    const uint32_t key_value_columns = static_cast<uint32_t>(key_value_columns64);

    std::shared_ptr<VulkanContext> context;
    if (use_native)
    {
        const auto& first_projection = native_projections[0];
        context = fused_native_qkv       ? fused_native_qkv->context()
                  : first_projection.qnk ? first_projection.qnk->context()
                                         : first_projection.float8->context();
    }
    else if (use_bfloat16)
    {
        context = fused_qkv_gate ? fused_qkv_gate->context() : output_projection_bfloat16->context();
    }
    else
    {
        context = fused_qkv ? fused_qkv->context() : output_projection->context();
    }
    if (!context)
        return {};

    std::shared_ptr<Attention_vulkan> attention(new Attention_vulkan);
    Implementation& implementation = *attention->d;
    implementation.fused_qkv = std::move(fused_qkv);
    implementation.output_projection = std::move(output_projection);
    implementation.fused_qkv_gate = std::move(fused_qkv_gate);
    implementation.output_projection_bfloat16 = std::move(output_projection_bfloat16);
    implementation.native_projections = std::move(native_projections);
    implementation.fused_native_qkv = std::move(fused_native_qkv);
    if (projections_q8)
    {
        implementation.projections_q8 = *projections_q8;
        implementation.fused_qkv_q8 = std::move(fused_qkv_q8);
        implementation.q8 = true;
    }
    implementation.config = config;
    implementation.query_key_norm = query_key_norm;
    implementation.output_gate = output_gate;
    if (has_flag(config.flags, AttentionSink))
    {
        implementation.sinks.resize(sinks->element_count());
        if (sinks->dtype == DType::Float32)
            std::copy(sinks->float32_values().begin(), sinks->float32_values().end(), implementation.sinks.begin());
        else
            for (size_t index = 0; index < implementation.sinks.size(); ++index)
                implementation.sinks[index] = bfloat16_to_float(sinks->bfloat16_values()[index]);
    }

    implementation.vulkan_context = std::move(context);
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    implementation.option = make_vk_opt(*implementation.vulkan_context);
    implementation.option.use_cooperative_matrix = vkdev->info.support_cooperative_matrix();
    implementation.device_parallelism = query_attention_device_parallelism(vkdev->info);
    implementation.kv_option = implementation.option;
    const DType kv_dtype = config.kv_dtype;
    if (vulkan_kv_size(kv_dtype, vkdev->info.device_index()) == UINT64_MAX)
        return {};
    implementation.config.kv_dtype = kv_dtype;
    implementation.kv_option.use_fp16_storage = kv_dtype == DType::Float16;
    implementation.kv_option.use_bf16_storage = kv_dtype == DType::BFloat16;

    if (has_flag(config.optimization_flags, OptimizationVulkanDecodeSdpa)
        && has_flag(config.optimization_flags, OptimizationVulkanBfloat16CoopMatrix))
    {
        implementation.decode_capabilities = query_attention_decode_capabilities(vkdev->info);
        AttentionDecodePlan plan;
        const AttentionDecodeShape shape{config.head_count, config.kv_head_count, config.head_dimension, 1,
                                         config.kv_cache_dtype == DType::BFloat16, kv_dtype == DType::BFloat16};
        if (!make_attention_decode_plan(implementation.decode_capabilities, shape, plan))
            return {};
        if (plan.path == AttentionDecodePath::Matrix)
        {
            implementation.decode_matrix = std::make_unique<AttentionDecodeVulkan>();
            if (!implementation.decode_matrix->create(implementation.vulkan_context, implementation.option, plan))
                return {};
        }
    }

    const uint32_t half_dimension = config.head_dimension / 2;
    if (config.rope_scaling_factor > 1.0f)
        implementation.rope_concentration = 0.1f * std::log(config.rope_scaling_factor) + 1.0f;
    implementation.rope_inverse_frequencies.assign(half_dimension, 0.0f);
    compute_yarn_inv_freq(rope_dimension, config.rope_active, config.rope_theta,
                          config.rope_scaling_factor, config.rope_ntk_alpha,
                          config.rope_ntk_beta, config.initial_context_length,
                          implementation.rope_inverse_frequencies.data());

    ncnn::ParamDict permute_parameters;
    permute_parameters.set(0, 2);
    if (!create_vulkan_layer(ncnn::LayerType::Permute, permute_parameters, vkdev,
                             implementation.option, implementation.layers,
                             implementation.permute_heads_tokens))
        return {};

    if (!query_key_norm && !output_gate && rope_dimension == config.head_dimension)
    {
        ncnn::Mat slice_sizes(3, sizeof(int));
        if (slice_sizes.empty())
            return {};
        int* slice_values = static_cast<int*>(slice_sizes.data);
        slice_values[0] = static_cast<int>(query_columns);
        slice_values[1] = static_cast<int>(key_value_columns);
        slice_values[2] = static_cast<int>(key_value_columns);
        ncnn::ParamDict slice_parameters;
        slice_parameters.set(0, slice_sizes);
        slice_parameters.set(1, 1);
        if (!create_vulkan_layer(ncnn::LayerType::Slice, slice_parameters, vkdev,
                                 implementation.option, implementation.layers,
                                 implementation.slice_qkv))
            return {};

        ncnn::ParamDict reshape_query_parameters;
        reshape_query_parameters.set(0, static_cast<int>(config.head_dimension));
        reshape_query_parameters.set(1, static_cast<int>(config.head_count));
        reshape_query_parameters.set(2, -1);
        if (!create_vulkan_layer(ncnn::LayerType::Reshape, reshape_query_parameters, vkdev,
                                 implementation.option, implementation.layers,
                                 implementation.reshape_query))
            return {};

        ncnn::ParamDict reshape_key_value_parameters;
        reshape_key_value_parameters.set(0, static_cast<int>(config.head_dimension));
        reshape_key_value_parameters.set(1, static_cast<int>(config.kv_head_count));
        reshape_key_value_parameters.set(2, -1);
        if (!create_vulkan_layer(ncnn::LayerType::Reshape, reshape_key_value_parameters, vkdev,
                                 implementation.option, implementation.layers,
                                 implementation.reshape_key_value))
            return {};

        ncnn::ParamDict rotary_parameters;
        rotary_parameters.set(0, 0);
        if (!create_vulkan_layer(ncnn::LayerType::RotaryEmbed, rotary_parameters, vkdev,
                                 implementation.option, implementation.layers,
                                 implementation.rotary))
            return {};
    }

    ncnn::ParamDict sdpa_parameters;
    sdpa_parameters.set(5, 1);
    sdpa_parameters.set(6, config.attn_scale);
    sdpa_parameters.set(7, 0);
    if (!create_vulkan_layer(ncnn::LayerType::SDPA, sdpa_parameters, vkdev,
                             implementation.option, implementation.layers, implementation.sdpa))
        return {};

    // Workgroup-scope NV cooperative_matrix2 flash attention for multi-row prefill.
    if ((config.kv_dtype == DType::BFloat16 || config.kv_dtype == DType::Float32)
        && (config.head_dimension == 64u || config.head_dimension == 128u || config.head_dimension == 256u)
        && config.head_count != 0 && config.kv_head_count != 0
        && config.head_count % config.kv_head_count == 0
        && !has_flag(config.flags, AttentionSink)
        && vkdev->info.subgroup_size() == 32)
    {
        const uint32_t br = config.head_dimension >= 256u ? 32u : 64u;
        const uint32_t bc = br;
        const bool has_win = config.sliding_window != 0u;
        if (supports_workgroup_cm2_matrix(vkdev->info, 128u, br, bc, config.head_dimension)
            && supports_workgroup_cm2_matrix(vkdev->info, 128u, br, config.head_dimension, bc)
            && supports_workgroup_cm2_matrix(vkdev->info, 128u, br, bc, bc)
            && create_prefill_cm2_pipeline(implementation.vulkan_context, br, bc, config.head_dimension,
                                           config.kv_dtype == DType::BFloat16, has_win,
                                           implementation.cm2_pipe[0], false))
        {
            implementation.cm2_br = br;
            implementation.cm2_bc = bc;
            implementation.cm2_ok = true;
            const uint64_t merge_variant = pipeline_opt_variant(implementation.option);
            const int merge_len = static_cast<int>(sizeof(attn_prefill_cm2_merge_shader) - 1);
            (void)create_pipeline(implementation.vulkan_context, attn_prefill_cm2_merge_shader, merge_len,
                                  implementation.option, merge_variant, merge_variant, 128,
                                  implementation.cm2_merge[0]);
            const bool o_uses_cm2 = !output_gate && ((implementation.output_projection_bfloat16 && implementation.output_projection_bfloat16->use_cm2()) || (implementation.native_projections[3].qnk && implementation.native_projections[3].qnk->use_cm2()));
            if (o_uses_cm2 && (config.head_dimension & 1u) == 0u
                && create_prefill_cm2_pipeline(implementation.vulkan_context, br, bc, config.head_dimension,
                                               config.kv_dtype == DType::BFloat16, has_win,
                                               implementation.cm2_pipe[1], true))
            {
                (void)create_pipeline_prefixed(implementation.vulkan_context, attn_prefill_cm2_merge_shader,
                                               merge_len, "#define NCNN_ATTN_CM2_BF16_OUT\n",
                                               implementation.option, 0, merge_variant | 0x100u, 128,
                                               implementation.cm2_merge[1]);
            }
        }
    }

    ncnn::ParamDict reshape_attention_parameters;
    reshape_attention_parameters.set(0, static_cast<int>(query_columns));
    reshape_attention_parameters.set(1, -1);
    if (!create_vulkan_layer(ncnn::LayerType::Reshape, reshape_attention_parameters, vkdev,
                             implementation.option, implementation.layers,
                             implementation.reshape_attention))
        return {};

    ncnn::ParamDict add_parameters;
    add_parameters.set(0, 0);
    if (!create_vulkan_layer(ncnn::LayerType::BinaryOp, add_parameters, vkdev,
                             implementation.option, implementation.layers, implementation.add))
        return {};

    implementation.weight_allocator.reset(new ncnn::VkWeightAllocator(vkdev,
                                                                      static_cast<size_t>(vulkan_weight_block_size)));
    implementation.weight_staging_allocator.reset(new ncnn::VkWeightStagingAllocator(vkdev));
    // Keep normalization weights in the same persistent arena as RoPE/sinks.
    if (norm_bias)
    {
        implementation.layer_norm = LayerNorm_vulkan::create(norm_weight, *norm_bias, config.norm_epsilon,
                                                             implementation.vulkan_context, implementation.option,
                                                             implementation.weight_allocator.get());
        if (!implementation.layer_norm)
            return {};
    }
    else
    {
        implementation.norm = RmsNorm_vulkan::create(norm_weight, config.norm_epsilon, config.norm_weight_offset,
                                                     implementation.vulkan_context, implementation.option,
                                                     implementation.weight_allocator.get());
        if (!implementation.norm)
            return {};
    }
    if (attn_post_norm)
    {
        implementation.post_attention_norm = RmsNorm_vulkan::create(*attn_post_norm, config.norm_epsilon, 0.0f,
                                                                    implementation.vulkan_context,
                                                                    implementation.option,
                                                                    implementation.weight_allocator.get());
        if (!implementation.post_attention_norm)
            return {};
    }

    const uint64_t opt_variant = pipeline_opt_variant(implementation.option);
    const uint64_t kv_variant = pipeline_opt_variant(implementation.kv_option);
    const bool needs_norm_rope = query_key_norm || output_gate || rope_dimension != config.head_dimension;

    std::vector<ncnn::vk_specialization_type> decode_sdpa_specs(1);
    {
        const uint32_t subgroup = vkdev->info.subgroup_size();
        const uint32_t required_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        const bool subgroup_reductions = implementation.kv_option.use_subgroup_ops
                                         && (vkdev->info.support_subgroup_ops() & required_ops) == required_ops
                                         && subgroup >= 16 && subgroup <= 128 && 128 % subgroup == 0;
        decode_sdpa_specs[0].u32 = subgroup_reductions ? 1u : 0u;
    }

    if ((!needs_norm_rope
         && !create_pipeline(implementation.vulkan_context, attention_qkv_rope_shader,
                             static_cast<int>(sizeof(attention_qkv_rope_shader) - 1),
                             implementation.kv_option, kv_variant, kv_variant, 64,
                             implementation.qkv_rope_pipeline))
        || (needs_norm_rope
            && !create_pipeline(implementation.vulkan_context, attention_qkv_norm_rope_shader,
                                static_cast<int>(sizeof(attention_qkv_norm_rope_shader) - 1),
                                implementation.kv_option, kv_variant, kv_variant, 32,
                                implementation.qkv_norm_rope_pipeline))
        || (output_gate
            && !create_pipeline(implementation.vulkan_context, attention_output_gate_shader,
                                static_cast<int>(sizeof(attention_output_gate_shader) - 1),
                                implementation.option, opt_variant, opt_variant, 128,
                                implementation.output_gate_pipeline))
        || !create_pipeline(implementation.vulkan_context, attention_decode_sdpa_shader,
                            static_cast<int>(sizeof(attention_decode_sdpa_shader) - 1),
                            implementation.kv_option, kv_variant, kv_variant, 256,
                            implementation.decode_sdpa_pipeline, decode_sdpa_specs)
        || !create_pipeline(implementation.vulkan_context, attn_decode_merge_shader,
                            static_cast<int>(sizeof(attn_decode_merge_shader) - 1),
                            implementation.kv_option, kv_variant, kv_variant, 256,
                            implementation.merge_pipeline)
        || !create_pipeline(implementation.vulkan_context, attention_ring_append_shader,
                            static_cast<int>(sizeof(attention_ring_append_shader) - 1),
                            implementation.kv_option, kv_variant, kv_variant, 8,
                            implementation.ring_append_pipeline, {}, 0, 8, 1)
        || !create_pipeline(implementation.vulkan_context, attention_ring_zero_shader,
                            static_cast<int>(sizeof(attention_ring_zero_shader) - 1),
                            implementation.kv_option, kv_variant, kv_variant, 8,
                            implementation.ring_zero_pipeline, {}, 0, 8, 1))
    {
        return {};
    }

    if (output_gate && (query_columns & 1u) == 0u
        && ((implementation.output_projection_bfloat16 && implementation.output_projection_bfloat16->use_cm2())
            || (implementation.native_projections[3].qnk && implementation.native_projections[3].qnk->use_cm2())))
    {
        (void)create_pipeline_prefixed(implementation.vulkan_context, attention_output_gate_shader,
                                       static_cast<int>(sizeof(attention_output_gate_shader) - 1),
                                       "#define NCNN_ATTN_GATE_BF16_OUT\n",
                                       implementation.option, 0, opt_variant | 0x100u, 128,
                                       implementation.output_gate_bf16_pipeline);
    }

    ncnn::Option subgroup_option = implementation.kv_option;
    subgroup_option.use_subgroup_ops = true;
    const uint64_t subgroup_variant = pipeline_opt_variant(subgroup_option);

    if (implementation.merge_pipeline && vkdev->info.subgroup_size() == 32)
    {
        const uint32_t shuffle_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        if ((vkdev->info.support_subgroup_ops() & shuffle_ops) == shuffle_ops)
        {
            if (!create_pipeline(implementation.vulkan_context, attn_decode_split_shader,
                                 static_cast<int>(sizeof(attn_decode_split_shader) - 1),
                                 subgroup_option, subgroup_variant, subgroup_variant, 256,
                                 implementation.split_pipeline))
                implementation.split_pipeline.reset();
        }

        const char* gqa_defs = gqa_defines(config.head_dimension);
        const uint32_t arith_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        if (gqa_defs != nullptr && (vkdev->info.support_subgroup_ops() & arith_ops) == arith_ops)
        {
            const uint64_t pipe_variant = subgroup_variant | (static_cast<uint64_t>(config.head_dimension) << 8);
            if (!create_pipeline_prefixed(implementation.vulkan_context, attn_decode_gqa_shader,
                                          static_cast<int>(sizeof(attn_decode_gqa_shader) - 1),
                                          gqa_defs, subgroup_option, 0, pipe_variant, 256,
                                          implementation.gqa_pipeline))
                implementation.gqa_pipeline.reset();
        }

        if (implementation.gqa_pipeline && act_element_size(implementation.kv_option) == sizeof(uint16_t))
        {
            const char* vec_defs = gqa_vec_defines(config.head_dimension);
            if (vec_defs != nullptr)
            {
                const uint64_t pipe_variant = subgroup_variant | (static_cast<uint64_t>(config.head_dimension) << 8);
                if (!create_pipeline_prefixed(implementation.vulkan_context, attn_decode_gqa_vec_shader,
                                              static_cast<int>(sizeof(attn_decode_gqa_vec_shader) - 1),
                                              vec_defs, subgroup_option, 0, pipe_variant, 256,
                                              implementation.gqa_vector_pipeline))
                    implementation.gqa_vector_pipeline.reset();
            }
        }

        if (config.kv_head_count != 0 && config.head_dimension == 64u
            && config.head_count % config.kv_head_count == 0
            && act_element_size(implementation.kv_option) == sizeof(uint16_t))
        {
            const uint32_t ratio = config.head_count / config.kv_head_count;
            const char* share_defs = ratio >= 4u ? gqa_share_defines(config.head_dimension, ratio) : nullptr;
            const uint32_t share_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT
                                       | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
            if (share_defs != nullptr && (vkdev->info.support_subgroup_ops() & share_ops) == share_ops)
            {
                const uint64_t pipe_variant = subgroup_variant | (static_cast<uint64_t>(ratio) << 8);
                if (!create_pipeline_prefixed(implementation.vulkan_context, attn_decode_gqa_share_shader,
                                              static_cast<int>(sizeof(attn_decode_gqa_share_shader) - 1),
                                              share_defs, subgroup_option, 0, pipe_variant, 128,
                                              implementation.gqa_share_pipeline))
                    implementation.gqa_share_pipeline.reset();
            }
        }

        if (config.kv_head_count != 0 && (config.head_dimension == 64u || config.head_dimension == 128u)
            && config.head_count % config.kv_head_count == 0
            && !has_flag(config.flags, AttentionSink)
            && act_element_size(implementation.kv_option) == sizeof(uint16_t)
            && supports_subgroup_matrix(vkdev->info, 16, 16, 16))
        {
            const uint32_t group_size = config.head_count / config.kv_head_count;
            if (group_size >= 4 && group_size <= 16)
            {
                std::vector<ncnn::vk_specialization_type> specializations(4);
                specializations[0].u32 = config.head_dimension;
                specializations[1].u32 = 16;
                specializations[2].u32 = 16;
                specializations[3].u32 = 16;
                const uint64_t pipe_variant = kv_variant | (static_cast<uint64_t>(config.head_dimension) << 8);
                if (!create_pipeline(implementation.vulkan_context, attention_gqa_subgroup_shader,
                                     static_cast<int>(sizeof(attention_gqa_subgroup_shader) - 1),
                                     implementation.kv_option, kv_variant, pipe_variant, 128,
                                     implementation.subgroup_pipeline, specializations))
                    implementation.subgroup_pipeline.reset();
            }
        }
    }

    const char* flash_defs = prefill_defines(config.head_dimension);
    if (flash_defs != nullptr && !has_flag(config.flags, AttentionSink))
    {
        const uint32_t arith_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        if (vkdev->info.subgroup_size() == 32
            && (vkdev->info.support_subgroup_ops() & arith_ops) == arith_ops)
        {
            const uint64_t pipe_variant = subgroup_variant | (static_cast<uint64_t>(config.head_dimension) << 8);
            if (!create_pipeline_prefixed(implementation.vulkan_context, attn_prefill_shader,
                                          static_cast<int>(sizeof(attn_prefill_shader) - 1),
                                          flash_defs, subgroup_option, 0, pipe_variant, 256,
                                          implementation.flash_pipeline))
                implementation.flash_pipeline.reset();
        }
    }
    implementation.flash_ok = implementation.flash_pipeline != nullptr
                              && !has_flag(config.flags, AttentionSink)
                              && config.sliding_window < std::numeric_limits<uint32_t>::max();

    ncnn::Mat query_norm_model;
    ncnn::Mat key_norm_model;
    if (query_key_norm
        && (!prepare_float_upload(*query_norm_weight, query_norm_model)
            || !prepare_float_upload(*key_norm_weight, key_norm_model)))
        return {};
    if (query_key_norm && config.norm_weight_offset != 0.0f)
    {
        float* query_values = static_cast<float*>(query_norm_model.data);
        float* key_values = static_cast<float*>(key_norm_model.data);
        for (uint32_t index = 0; index < config.head_dimension; ++index)
        {
            query_values[index] += config.norm_weight_offset;
            key_values[index] += config.norm_weight_offset;
        }
    }

    ncnn::Mat rope_inverse_model(static_cast<int>(implementation.rope_inverse_frequencies.size()), sizeof(float));
    ncnn::Mat sink_model(static_cast<int>(config.head_count), sizeof(float));
    if (rope_inverse_model.empty() || sink_model.empty())
        return {};
    std::copy(implementation.rope_inverse_frequencies.begin(),
              implementation.rope_inverse_frequencies.end(),
              static_cast<float*>(rope_inverse_model.data));
    float* sink_values = static_cast<float*>(sink_model.data);
    std::fill_n(sink_values, config.head_count, 0.0f);
    if (has_flag(config.flags, AttentionSink))
        std::copy(implementation.sinks.begin(), implementation.sinks.end(), sink_values);

    {
        const std::lock_guard<std::mutex> lock(implementation.vulkan_context->command_mutex());
        ncnn::VkTransfer command(vkdev);
        ncnn::Option upload_option = implementation.option;
        upload_option.blob_vkallocator = implementation.weight_allocator.get();
        upload_option.workspace_vkallocator = implementation.weight_allocator.get();
        upload_option.staging_vkallocator = implementation.weight_staging_allocator.get();
        if (query_key_norm)
        {
            command.record_upload(query_norm_model, implementation.query_norm_weight, upload_option);
            command.record_upload(key_norm_model, implementation.key_norm_weight, upload_option);
        }
        command.record_upload(rope_inverse_model, implementation.rope_inverse_frequencies_gpu, upload_option);
        command.record_upload(sink_model, implementation.attention_sinks, upload_option);
        if ((query_key_norm
             && (implementation.query_norm_weight.empty() || implementation.key_norm_weight.empty()))
            || implementation.rope_inverse_frequencies_gpu.empty()
            || implementation.attention_sinks.empty()
            || command.submit_and_wait() != 0)
            return {};
        rearm_read(implementation.query_norm_weight);
        rearm_read(implementation.key_norm_weight);
        rearm_read(implementation.rope_inverse_frequencies_gpu);
        rearm_read(implementation.attention_sinks);
    }
    implementation.weight_staging_allocator.reset();
    return attention;
}

class AttentionWork_vulkan::Implementation
{
public:
    LayerCache* cache = nullptr;
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<AttentionCache_vulkan> original;
    std::shared_ptr<AttentionCache_vulkan> next_cache;
    std::deque<ncnn::VkMat> buffers;
    std::deque<std::vector<ncnn::VkMat>> lists;
    size_t buffers_used = 0;
    size_t lists_used = 0;
    ncnn::VkMat rope_cosine;
    ncnn::VkMat rope_sine;
    ncnn::VkMat attention_mask;
    ncnn::VkMat attention_cache_key;
    ncnn::VkMat attention_cache_value;
    AttentionConfig_vulkan config;
    uint64_t previous_tokens = 0;
    uint64_t previous_start = 0;
    uint64_t retained_tokens = 0;
    uint64_t dropped_tokens = 0;
    uint64_t next_first_slot = 0;
    uint64_t ring_capacity = 0;
    uint64_t allocated_cache_size = 0;
    size_t rows = 0;
    bool allocate_ring = false;
    bool ring_resized = false;
    bool mutated_original = false;
    bool qkv_rope = false;
    bool device_rope = false;
    bool qkv_ring = false;
    bool decode_sdpa = false;
    bool decode_matrix = false;
    bool wrapped = false;
    bool promoted = false;
    bool recorded = false;
    bool failed = false;
    bool committed = false;

    ncnn::VkMat& keep_buffer(const ncnn::VkMat& value = {})
    {
        if (buffers_used == buffers.size())
            buffers.emplace_back();
        ncnn::VkMat& result = buffers[buffers_used++];
        result = value;
        return result;
    }

    std::vector<ncnn::VkMat>& keep_list(size_t count = 0)
    {
        if (lists_used == lists.size())
            lists.emplace_back();
        std::vector<ncnn::VkMat>& result = lists[lists_used++];
        result.resize(count);
        return result;
    }

    std::vector<ncnn::VkMat>& keep_list(std::initializer_list<ncnn::VkMat> values)
    {
        if (lists_used == lists.size())
            lists.emplace_back();
        std::vector<ncnn::VkMat>& result = lists[lists_used++];
        result.assign(values);
        return result;
    }
};

AttentionWork_vulkan::AttentionWork_vulkan()
    : d(new Implementation)
{
}

AttentionWork_vulkan::~AttentionWork_vulkan() = default;

void AttentionWork_vulkan::retire() noexcept
{
    // Slot objects remain stable for record_forward's references, while no VkMat
    // survives retirement. Device allocation lifetimes are unchanged.
    for (size_t i = 0; i < d->buffers_used; ++i)
        d->buffers[i].release();
    for (size_t i = 0; i < d->lists_used; ++i)
        d->lists[i].clear();
    d->buffers_used = d->lists_used = 0;
    d->rope_cosine.release();
    d->rope_sine.release();
    d->attention_mask.release();
    d->attention_cache_key.release();
    d->attention_cache_value.release();
    d->original.reset();
    d->next_cache.reset();
    d->context.reset();
    d->cache = nullptr;
    d->config = {};
    d->previous_tokens = d->previous_start = 0;
    d->retained_tokens = d->dropped_tokens = d->next_first_slot = d->ring_capacity = 0;
    d->allocated_cache_size = 0;
    d->rows = 0;
    d->allocate_ring = d->ring_resized = d->mutated_original = false;
    d->qkv_rope = d->device_rope = d->qkv_ring = d->decode_sdpa = false;
    d->decode_matrix = d->wrapped = d->promoted = false;
    d->recorded = d->failed = d->committed = false;
}

void AttentionWork_vulkan::mark_submit_failed() noexcept
{
    d->failed = true;
    if (d->cache && d->mutated_original)
        d->cache->vulkan_attention_state_unknown = true;
}

bool AttentionWork_vulkan::commit() noexcept
{
    if (!d->recorded || d->failed || d->committed || !d->cache
        || d->cache->token_count != d->previous_tokens || d->cache->vulkan_attention_cache != d->original)
        return false;
    auto& cache = *d->cache;
    std::vector<float>{}.swap(cache.keys);
    std::vector<float>{}.swap(cache.values);
    std::vector<uint16_t>{}.swap(cache.bfloat16_keys);
    std::vector<uint16_t>{}.swap(cache.bfloat16_values);
    cache.start_position = d->previous_start + d->dropped_tokens;
    cache.token_count = d->retained_tokens;
    cache.first_slot = d->next_first_slot;
    cache.capacity_tokens = d->retained_tokens == 0 ? 0 : d->ring_capacity;
    cache.columns = d->config.kv_head_count * d->config.head_dimension;
    cache.dtype = d->config.kv_cache_dtype;
    if (d->retained_tokens == 0)
        cache.vulkan_attention_cache.reset();
    else if (d->allocate_ring)
        cache.vulkan_attention_cache = d->next_cache;
    cache.device_allocated_size = d->allocated_cache_size;
    cache.vulkan_attention_state_unknown = false;
    record_standard_cache_transaction_rows(cache, d->rows);
    auto& state = d->context->runtime_state();
    state.dispatches += 2;
    ++state.attention_blocks;
    state.attention_qkv_rope_fusions += d->qkv_rope;
    state.attention_device_rope_fusions += d->device_rope;
    state.attention_qkv_ring_fusions += d->qkv_ring;
    state.attention_decode_sdpa_fusions += d->decode_sdpa;
    ++state.kv_ring_appends;
    state.kv_ring_resizes += d->ring_resized;
    state.kv_ring_wrapped_views += d->wrapped;
    if (d->promoted)
        cache.vulkan_attention_promotion_disabled = false;
    d->committed = true;
    return true;
}

const std::shared_ptr<VulkanContext>& Attention_vulkan::context() const noexcept
{
    return d->vulkan_context;
}

const ncnn::Option& Attention_vulkan::option() const noexcept
{
    return d->option;
}

bool Attention_vulkan::uses_qkv_ring(size_t token_count) const noexcept
{
    return token_count == 1 && has_flag(d->config.optimization_flags, OptimizationVulkanQkvRing)
           && support_qkv_rope(token_count);
}

bool Attention_vulkan::decode_workspace_bytes(uint32_t context_tokens, uint64_t& bytes) const noexcept
{
    bytes = 0;
    if (!d->decode_matrix)
        return true;
    AttentionDecodePlan plan;
    const AttentionDecodeShape shape{d->config.head_count, d->config.kv_head_count, d->config.head_dimension,
                                     std::max(1u, context_tokens), d->config.kv_cache_dtype == DType::BFloat16,
                                     d->config.kv_dtype == DType::BFloat16};
    if (!make_attention_decode_plan(d->decode_capabilities, shape, plan)
        || plan.path != AttentionDecodePath::Matrix)
        return false;
    bytes = plan.scratch_bytes;
    return true;
}

bool Attention_vulkan::flash_ok() const noexcept
{
    return d && d->flash_ok;
}

bool Attention_vulkan::record_decode(const ncnn::VkMat& query, const ncnn::VkMat& key,
                                     const ncnn::VkMat& value, uint64_t destination_count,
                                     ncnn::VkMat& output, ncnn::VkCompute& command,
                                     std::vector<ncnn::VkMat>& workspace) const
{
    const Implementation& implementation = *d;
    const AttentionConfig_vulkan& config = implementation.config;
    ncnn::VkAllocator* allocator = implementation.option.blob_vkallocator;

    const auto record_matrix_decode = [&]() -> bool {
        if (!implementation.decode_matrix)
            return false;
        const uint32_t sink_rows = has_flag(config.flags, AttentionSink) ? 1 : 0;
        if (destination_count <= sink_rows || destination_count > std::numeric_limits<uint32_t>::max())
            return false;
        AttentionDecodePlan plan;
        const AttentionDecodeShape shape{config.head_count, config.kv_head_count, config.head_dimension,
                                         static_cast<uint32_t>(destination_count - sink_rows),
                                         config.kv_cache_dtype == DType::BFloat16,
                                         config.kv_dtype == DType::BFloat16};
        if (!make_attention_decode_plan(implementation.decode_capabilities, shape, plan)
            || plan.path != AttentionDecodePath::Matrix)
            return false;
        // Matrix decode handles the sink denominator without reading the synthetic KV row.
        ncnn::VkMat real_key = key;
        ncnn::VkMat real_value = value;
        real_key.h = real_value.h = static_cast<int>(shape.kv_tokens);
        return implementation.decode_matrix->record(plan, query, real_key, real_value,
                                                    implementation.attention_sinks,
                                                    has_flag(config.flags, AttentionSink),
                                                    config.attn_scale, output, command, allocator, workspace);
    };

    if (implementation.subgroup_pipeline)
    {
        const AttnSplitRange subgroup_range = compute_split_range(destination_count, 4096u,
                                                                  std::max<uint64_t>(1, implementation.device_parallelism) * 8u,
                                                                  config.kv_head_count, 1024u, 16u, true);
        if (subgroup_range.splits != 0)
        {
            const uint64_t tasks = static_cast<uint64_t>(config.kv_head_count) * subgroup_range.splits;
            if (config.kv_head_count != 0
                && record_split_decode(implementation.subgroup_pipeline.get(), implementation.merge_pipeline.get(),
                                       query, key, value, implementation.attention_sinks, config,
                                       destination_count, subgroup_range,
                                       static_cast<int>(((tasks + 3u) / 4u) * 128u), 1, 1,
                                       output, command, allocator))
                return true;
        }
    }

    if (implementation.gqa_share_pipeline)
    {
        const uint32_t ratio = config.kv_head_count == 0 ? 0 : config.head_count / config.kv_head_count;
        const AttnSplitRange share_range = compute_split_range(destination_count, 4096u,
                                                               std::max<uint64_t>(1, implementation.device_parallelism) * 16u,
                                                               config.kv_head_count, share_split_max, 4u, true);
        if (share_range.splits != 0 && ratio >= 2)
        {
            const uint64_t tasks = static_cast<uint64_t>(config.kv_head_count) * share_range.splits;
            if (record_split_decode(implementation.gqa_share_pipeline.get(), implementation.merge_pipeline.get(),
                                    query, key, value, implementation.attention_sinks, config,
                                    destination_count, share_range,
                                    static_cast<int>(((tasks + 3u) / 4u) * 128u), 1, 1,
                                    output, command, allocator))
                return true;
        }
    }

    const uint64_t head_blocks = (config.head_count + 7u) / 8u;
    const uint64_t gqa_scale = head_blocks == 1u ? 2u : 4u;
    const AttnSplitRange gqa_range = compute_split_range(destination_count, split_min_tokens,
                                                         std::max<uint64_t>(1, implementation.device_parallelism) * gqa_scale,
                                                         head_blocks, 256u, 8u, false);
    if (gqa_range.splits != 0
        && record_split_decode(implementation.gqa_vector_pipeline
                                   ? implementation.gqa_vector_pipeline.get()
                                   : implementation.gqa_pipeline.get(),
                               implementation.merge_pipeline.get(),
                               query, key, value, implementation.attention_sinks, config,
                               destination_count, gqa_range, 256, static_cast<int>(gqa_range.splits),
                               static_cast<int>(head_blocks), output, command, allocator))
        return true;

    if (record_matrix_decode())
        return true;

    const bool split_available = implementation.merge_pipeline && implementation.split_pipeline;
    const AttnSplitRange range = split_range(destination_count, split_available);
    if (range.splits != 0
        && record_split_decode(implementation.split_pipeline.get(), implementation.merge_pipeline.get(),
                               query, key, value, implementation.attention_sinks, config,
                               destination_count, range, 256, static_cast<int>(range.splits),
                               static_cast<int>(config.head_count), output, command, allocator))
        return true;

    return record_attention_decode_sdpa(implementation.decode_sdpa_pipeline.get(), query, key, value,
                                        implementation.attention_sinks, config, destination_count,
                                        output, command, allocator);
}

bool Attention_vulkan::record_forward(uint64_t position_offset, LayerCache& cache,
                                      const ncnn::VkMat& input, ncnn::VkMat& output,
                                      ncnn::VkCompute& command, AttentionWork_vulkan& work,
                                      bool add_residual) const
{
    auto& ticket = *work.d;
    ticket.cache = &cache;
    ticket.original = cache.vulkan_attention_cache;
    ticket.previous_tokens = cache.token_count;
    ticket.previous_start = cache.token_count == 0 ? position_offset : cache.start_position;
    ticket.rows = static_cast<size_t>(input.h);
    const size_t token_count = ticket.rows;
    const Implementation& implementation = *d;
    const AttentionConfig_vulkan& config = implementation.config;
    const size_t activation_element_size = act_element_size(implementation.kv_option);
    const bool low_precision_kv = activation_variant(implementation.kv_option) != 0;
    const bool has_device_cache = cache.vulkan_attention_cache != nullptr;
    const bool promote_host_cache = cache.token_count != 0 && !has_device_cache;
    const bool bfloat16_storage = config.activation_dtype == DType::BFloat16 && implementation.option.use_bf16_storage;
    const bool device_rope = has_flag(config.optimization_flags, OptimizationVulkanDeviceRope)
                             && !implementation.rope_inverse_frequencies_gpu.empty();
    const bool use_norm_rope = implementation.query_key_norm || implementation.output_gate
                               || (config.rope_head_dimension != 0
                                   && config.rope_head_dimension < config.head_dimension);
    const bool use_native = static_cast<bool>(implementation.fused_native_qkv
                                              || implementation.native_projections[0].qnk
                                              || implementation.native_projections[0].float8);
    const bool use_qkv_rope = support_qkv_rope(token_count);
    const bool use_qkv_ring = uses_qkv_ring(token_count);
    if (!use_qkv_rope && (device_rope || use_norm_rope))
        return false;

    const uint32_t query_columns = config.head_count * config.head_dimension;
    const uint32_t kv_columns = config.kv_head_count * config.head_dimension;
    uint64_t promotion_elements = 0;
    uint64_t promotion_transfer_bytes = 0;
    if (promote_host_cache
        && (!checked_multiply_u64(cache.token_count, kv_columns, promotion_elements)
            || !checked_multiply_u64(promotion_elements, sizeof(float) * 2, promotion_transfer_bytes)))
    {
        cache.vulkan_attention_promotion_disabled = true;
        return false;
    }
    const uint64_t actual_token_count = cache.token_count + token_count;
    const uint64_t sink_token_count = has_flag(config.flags, AttentionSink) ? 1 : 0;
    const uint64_t destination_count = actual_token_count + sink_token_count;
    if (promote_host_cache
        && !attention_promotion_within_budget(*implementation.vulkan_context,
                                              next_attention_ring_capacity(0, actual_token_count),
                                              kv_columns, activation_element_size,
                                              promotion_transfer_bytes))
    {
        cache.vulkan_attention_promotion_disabled = true;
        return false;
    }
    if (promote_host_cache)
        cache.vulkan_attention_promotion_disabled = true;

    const bool decode_split_available = implementation.merge_pipeline && implementation.split_pipeline;
    const bool use_decode_sdpa = has_flag(config.optimization_flags, OptimizationVulkanDecodeSdpa)
                                 && token_count == 1
                                 && attention_decode_window_is_visible(cache, position_offset,
                                                                       actual_token_count, config.sliding_window)
                                 && ((destination_count <= decode_single_workgroup_tokens
                                      && config.head_dimension <= 128)
                                     || (config.head_dimension <= 256
                                         && split_range(destination_count, decode_split_available).splits != 0));
    const bool flash = implementation.flash_ok
                       && !use_decode_sdpa
                       && token_count > 1
                       && position_offset == cache.start_position + cache.token_count;

    const Linear* fused = implementation.fused_qkv.get();
    const Bfloat16Linear_vulkan* fused_gate = implementation.fused_qkv_gate.get();
    const Linear* projection = implementation.output_projection.get();
    const Bfloat16Linear_vulkan* projection_bfloat16 = implementation.output_projection_bfloat16.get();
    if ((!device_rope
         && !fill_rope_staging_pair(ticket.rope_cosine, ticket.rope_sine, token_count, position_offset,
                                    implementation.rope_inverse_frequencies, implementation.rope_concentration,
                                    bfloat16_storage, implementation.vulkan_context->staging_allocator()))
        || (!use_decode_sdpa && !flash
            && !fill_attention_mask_staging(ticket.attention_mask, token_count, destination_count,
                                            position_offset, cache, config, implementation.sinks,
                                            bfloat16_storage, implementation.vulkan_context->staging_allocator()))
        || (promote_host_cache
            && !fill_promotion_stage(ticket.attention_cache_key, ticket.attention_cache_value,
                                     cache, config, implementation.vulkan_context->staging_allocator())))
        return false;

    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    auto& input_gpu = ticket.keep_buffer(input);
    auto& cosine_gpu = ticket.keep_buffer(device_rope ? implementation.rope_inverse_frequencies_gpu : ncnn::VkMat());
    auto& sine_gpu = ticket.keep_buffer(device_rope ? implementation.rope_inverse_frequencies_gpu : ncnn::VkMat());
    auto& mask_gpu = ticket.keep_buffer();
    auto& promoted_key_gpu = ticket.keep_buffer();
    auto& promoted_value_gpu = ticket.keep_buffer();
    if (!device_rope
        && (!record_mapped_upload(ticket.rope_cosine, cosine_gpu, command, implementation.option)
            || !record_mapped_upload(ticket.rope_sine, sine_gpu, command, implementation.option)))
        return false;
    if (promote_host_cache
        && (!record_mapped_act_upload(ticket.attention_cache_key, promoted_key_gpu, command, vkdev,
                                      implementation.kv_option)
            || !record_mapped_act_upload(ticket.attention_cache_value, promoted_value_gpu, command, vkdev,
                                         implementation.kv_option)))
        return false;
    if (!use_decode_sdpa && !flash
        && !record_mapped_upload(ticket.attention_mask, mask_gpu, command, implementation.option))
        return false;

    auto& normalized_gpu = ticket.keep_buffer();
    const bool qkv_wants_bf16 = token_count >= 8 && !implementation.layer_norm && implementation.norm
                                && ((use_native && implementation.fused_native_qkv
                                     && implementation.fused_native_qkv->use_cm2())
                                    || (!use_native && fused_gate && fused_gate->use_cm2()));
    const bool normalized = implementation.layer_norm
                                ? implementation.layer_norm->record(input_gpu, normalized_gpu, command)
                                : implementation.norm->record(input_gpu, normalized_gpu, command,
                                                              nullptr, nullptr, qkv_wants_bf16);
    if (!normalized)
        return false;

    auto& fused_gpu = ticket.keep_buffer();
    auto& normalized_unpacked_gpu = ticket.keep_buffer(normalized_gpu);
    auto& projection_workspace = ticket.keep_list();
    if ((use_native || use_norm_rope || fused_gate) && normalized_unpacked_gpu.elempack != 1)
    {
        auto& unpacked = ticket.keep_buffer();
        vkdev->convert_packing(normalized_unpacked_gpu, unpacked, 1, command, implementation.option);
        normalized_unpacked_gpu = unpacked;
    }
    if (use_native)
    {
        if (!record_native_qkv(normalized_unpacked_gpu, fused_gpu, command, projection_workspace))
            return false;
    }
    else if (fused_gate)
    {
        if (fused_gate->forward(normalized_unpacked_gpu, fused_gpu, command, implementation.option) != 0)
            return false;
    }
    else if (!fused || fused->forward(normalized_gpu, fused_gpu, command, implementation.option) != 0)
    {
        return false;
    }

    uint64_t ring_capacity = cache.capacity_tokens;
    uint64_t ring_first_slot = cache.first_slot;
    const bool ring_resized = !promote_host_cache && actual_token_count > ring_capacity;
    const bool allocate_ring = !has_device_cache || ring_resized;
    auto& next_cache = ticket.next_cache;
    next_cache = allocate_ring ? std::make_shared<AttentionCache_vulkan>() : cache.vulkan_attention_cache;
    if (allocate_ring)
    {
        if (config.sliding_window != 0)
            ring_capacity = next_attention_ring_capacity(has_device_cache ? ring_capacity : 0, actual_token_count);
        else
            ring_capacity = kv_capacity(false, std::max<uint64_t>(actual_token_count, cache.reserved_tokens));
        // Long-lived KV rings use device-local storage so decode does not stream over BAR.
        if (!create_attention_ring_storage(*next_cache, config.head_dimension, config.kv_head_count,
                                           ring_capacity, activation_element_size,
                                           config.sliding_window != 0,
                                           implementation.vulkan_context->kv_allocator()))
            return false;
        if (cache.token_count != 0)
        {
            const ncnn::VkMat previous_key = promote_host_cache
                                                 ? promoted_key_gpu
                                                 : attention_ring_view(cache.vulkan_attention_cache->key,
                                                                       cache.first_slot, cache.token_count);
            const ncnn::VkMat previous_value = promote_host_cache
                                                   ? promoted_value_gpu
                                                   : attention_ring_view(cache.vulkan_attention_cache->value,
                                                                         cache.first_slot, cache.token_count);
            if (!record_attention_copy(implementation.ring_append_pipeline.get(), previous_key, previous_value,
                                       next_cache->key, next_cache->value, ring_capacity, 0,
                                       config.kv_dtype == DType::BFloat16, 0, false,
                                       config.sliding_window != 0, command))
                return false;
        }
        ring_first_slot = 0;
    }
    ticket.mutated_original = has_device_cache && !ring_resized;
    const uint64_t append_slot = (ring_first_slot + cache.token_count) % ring_capacity;

    auto& query_rope = ticket.keep_buffer();
    auto& key_rope = ticket.keep_buffer();
    auto& value_heads = ticket.keep_buffer();
    auto& output_gate = ticket.keep_buffer();
    auto& fused_qkv_unpacked = ticket.keep_buffer(fused_gpu);
    if (fused_qkv_unpacked.elempack != 1)
    {
        auto& unpacked = ticket.keep_buffer();
        vkdev->convert_packing(fused_qkv_unpacked, unpacked, 1, command, implementation.option);
        fused_qkv_unpacked = unpacked;
    }
    if (use_qkv_rope)
    {
        const AttentionCache_vulkan* ring = use_qkv_ring ? next_cache.get() : nullptr;
        const bool recorded_rope = use_norm_rope
                                       ? record_qkv_norm_rope(fused_qkv_unpacked, cosine_gpu, sine_gpu,
                                                              token_count, position_offset, device_rope,
                                                              ring, ring_capacity, append_slot, query_rope,
                                                              key_rope, value_heads, output_gate, command)
                                       : record_qkv_rope(fused_qkv_unpacked, cosine_gpu, sine_gpu,
                                                         token_count, position_offset, device_rope,
                                                         ring, ring_capacity, append_slot, query_rope,
                                                         key_rope, value_heads, command);
        if (!recorded_rope)
            return false;
    }
    else
    {
        auto& qkv_input = ticket.keep_list({fused_gpu});
        auto& qkv = ticket.keep_list(3);
        if (implementation.slice_qkv->forward(qkv_input, qkv, command, implementation.option) != 0)
            return false;

        auto& query_shaped = ticket.keep_buffer();
        auto& key_shaped = ticket.keep_buffer();
        auto& value_shaped = ticket.keep_buffer();
        if (implementation.reshape_query->forward(qkv[0], query_shaped, command, implementation.option) != 0
            || implementation.reshape_key_value->forward(qkv[1], key_shaped, command, implementation.option) != 0
            || implementation.reshape_key_value->forward(qkv[2], value_shaped, command, implementation.option) != 0)
            return false;

        auto& query_heads = ticket.keep_buffer();
        auto& key_heads = ticket.keep_buffer();
        if (implementation.permute_heads_tokens->forward(query_shaped, query_heads, command, implementation.option) != 0
            || implementation.permute_heads_tokens->forward(key_shaped, key_heads, command, implementation.option) != 0
            || implementation.permute_heads_tokens->forward(value_shaped, value_heads, command, implementation.option) != 0)
            return false;

        for (ncnn::VkMat* tensor : {&query_heads, &key_heads, &value_heads})
        {
            if (tensor->elempack != 1)
            {
                auto& unpacked = ticket.keep_buffer();
                vkdev->convert_packing(*tensor, unpacked, 1, command, implementation.option);
                *tensor = unpacked;
            }
        }

        auto& query_rope_input = ticket.keep_list({query_heads, cosine_gpu, sine_gpu});
        auto& key_rope_input = ticket.keep_list({key_heads, cosine_gpu, sine_gpu});
        auto& query_rope_output = ticket.keep_list(1);
        auto& key_rope_output = ticket.keep_list(1);
        if (implementation.rotary->forward(query_rope_input, query_rope_output, command, implementation.option) != 0
            || implementation.rotary->forward(key_rope_input, key_rope_output, command, implementation.option) != 0)
            return false;
        query_rope = query_rope_output[0];
        key_rope = key_rope_output[0];
    }

    if (!use_qkv_ring
        && !record_attention_copy(implementation.ring_append_pipeline.get(), key_rope, value_heads,
                                  next_cache->key, next_cache->value, ring_capacity, append_slot,
                                  config.kv_dtype == DType::BFloat16, 0, false,
                                  config.sliding_window != 0, command))
        return false;

    if (sink_token_count != 0 && !use_qkv_ring)
    {
        const uint64_t sink_row = ring_first_slot + actual_token_count;
        if (!record_attention_ring_zero(implementation.ring_zero_pipeline.get(),
                                        next_cache->key, next_cache->value, sink_row, command))
            return false;
    }

    auto& combined_key = ticket.keep_buffer(attention_ring_view(next_cache->key, ring_first_slot, destination_count));
    auto& combined_value = ticket.keep_buffer(attention_ring_view(next_cache->value, ring_first_slot, destination_count));
    if (combined_key.empty() || combined_value.empty())
        return false;

    auto& attention_matrix = ticket.keep_buffer();
    if (use_decode_sdpa)
    {
        if (!record_decode(query_rope, combined_key, combined_value, destination_count,
                           attention_matrix, command, projection_workspace))
            return false;
    }
    else
    {
        bool flash_recorded = false;
        if (flash && implementation.cm2_ok)
        {
            const bool bf16_out = !implementation.output_gate
                                  && implementation.cm2_pipe[1] != nullptr
                                  && implementation.cm2_merge[1] != nullptr;
            const size_t idx = bf16_out ? 1u : 0u;
            flash_recorded = record_prefill_cm2(implementation.cm2_pipe[idx].get(),
                                                implementation.cm2_merge[idx].get(),
                                                implementation.cm2_br, implementation.cm2_bc,
                                                implementation.device_parallelism,
                                                implementation.vulkan_context.get(),
                                                query_rope, combined_key, combined_value, config,
                                                token_count, destination_count, position_offset,
                                                cache.start_position, attention_matrix, command,
                                                implementation.option.blob_vkallocator, bf16_out);
        }
        if (!flash_recorded && flash)
        {
            flash_recorded = record_flash(implementation.flash_pipeline.get(),
                                          query_rope, combined_key, combined_value, config,
                                          token_count, destination_count, position_offset,
                                          cache.start_position, attention_matrix, command,
                                          implementation.option.blob_vkallocator);
        }
        if (!flash_recorded)
        {
            // Flash skipped additive mask staging; build and upload lazily on SDPA fallback.
            if (flash
                && (!fill_attention_mask_staging(ticket.attention_mask, token_count, destination_count,
                                                 position_offset, cache, config, implementation.sinks,
                                                 bfloat16_storage,
                                                 implementation.vulkan_context->staging_allocator())
                    || !record_mapped_upload(ticket.attention_mask, mask_gpu, command, implementation.option)))
                return false;
            if (sink_token_count != 0 && use_qkv_ring)
            {
                const uint64_t sink_row = ring_first_slot + actual_token_count;
                if (!record_attention_ring_zero(implementation.ring_zero_pipeline.get(),
                                                next_cache->key, next_cache->value, sink_row, command))
                    return false;
            }
            auto& sdpa_key = ticket.keep_buffer();
            auto& sdpa_value = ticket.keep_buffer();
            if (low_precision_kv)
            {
                sdpa_key.create(combined_key.w, combined_key.h, combined_key.c, sizeof(float),
                                implementation.option.blob_vkallocator);
                sdpa_value.create(combined_value.w, combined_value.h, combined_value.c, sizeof(float),
                                  implementation.option.blob_vkallocator);
                if (sdpa_key.empty() || sdpa_value.empty()
                    || !record_attention_copy(implementation.ring_append_pipeline.get(),
                                              next_cache->key, next_cache->value,
                                              sdpa_key, sdpa_value, ring_capacity, 0,
                                              false, ring_first_slot, true, false, command))
                    return false;
            }
            else
            {
                sdpa_key = combined_key;
                sdpa_value = combined_value;
            }
            auto& sdpa_input = ticket.keep_list({query_rope, sdpa_key, sdpa_value, mask_gpu});
            auto& sdpa_output = ticket.keep_list(1);
            if (implementation.sdpa->forward(sdpa_input, sdpa_output, command, implementation.option) != 0)
                return false;

            auto& attention_token_major = ticket.keep_buffer();
            if (implementation.permute_heads_tokens->forward(sdpa_output[0], attention_token_major,
                                                             command, implementation.option)
                    != 0
                || implementation.reshape_attention->forward(attention_token_major, attention_matrix,
                                                             command, implementation.option)
                       != 0)
                return false;
        }
    }

    if ((use_native || projection_bfloat16 || implementation.output_gate) && attention_matrix.elempack != 1)
    {
        auto& unpacked = ticket.keep_buffer();
        vkdev->convert_packing(attention_matrix, unpacked, 1, command, implementation.option);
        attention_matrix = unpacked;
    }

    if (implementation.output_gate)
    {
        if (token_count >= 8 && implementation.output_gate_bf16_pipeline != nullptr
            && attention_matrix.elemsize == sizeof(float) && attention_matrix.elempack == 1)
        {
            auto& gated_bf16 = ticket.keep_buffer();
            gated_bf16.create(static_cast<int>(query_columns), static_cast<int>(token_count),
                              sizeof(uint16_t), 1, implementation.option.blob_vkallocator);
            if (gated_bf16.empty())
                return false;
            const uint64_t elements = static_cast<uint64_t>(token_count) * query_columns;
            std::vector<ncnn::vk_constant_type> gate_constants(1);
            gate_constants[0].u32 = static_cast<uint32_t>(elements);
            ncnn::VkMat gate_dispatcher;
            gate_dispatcher.w = static_cast<int>(elements >> 1u);
            gate_dispatcher.h = 1;
            gate_dispatcher.c = 1;
            command.record_pipeline_readonly(implementation.output_gate_bf16_pipeline.get(),
                                             {attention_matrix, output_gate, gated_bf16},
                                             {1, 1, 0}, gate_constants, gate_dispatcher);
            attention_matrix = gated_bf16;
        }
        else if (!record_attention_output_gate(implementation.output_gate_pipeline.get(),
                                               attention_matrix, output_gate, token_count,
                                               query_columns, command))
        {
            return false;
        }
    }

    auto& projected_gpu = ticket.keep_buffer();
    if (projection_bfloat16)
    {
        if (attention_matrix.empty()
            || projection_bfloat16->forward(attention_matrix, projected_gpu, command, implementation.option) != 0)
            return false;
    }
    else if (use_native)
    {
        const CompiledOperator& output_operator = token_count == 1 && implementation.q8
                                                      ? implementation.projections_q8[3]
                                                      : implementation.native_projections[3];
        const bool recorded = output_operator.qnk
                                  ? output_operator.qnk->record_forward(attention_matrix, projected_gpu, command)
                                  : output_operator.float8
                                        && output_operator.float8->record_forward(attention_matrix, projected_gpu,
                                                                                  command, projection_workspace);
        if (!recorded)
            return false;
    }
    else if (!projection
             || projection->forward(attention_matrix, projected_gpu, command, implementation.option) != 0)
    {
        return false;
    }

    if (implementation.post_attention_norm)
    {
        // Post-attention RMSNorm normalizes the output projection before residual add.
        if (projected_gpu.empty() || projected_gpu.elempack != 1 || projected_gpu.elemsize != sizeof(float))
            return false;
        auto& post_attention_normalized = ticket.keep_buffer();
        if (!implementation.post_attention_norm->record(projected_gpu, post_attention_normalized, command))
            return false;
        projected_gpu = post_attention_normalized;
    }

    if (add_residual)
    {
        auto& residual_input_gpu = ticket.keep_buffer(input_gpu);
        if (use_native && (residual_input_gpu.elempack != 1 || residual_input_gpu.elemsize != sizeof(float)))
        {
            auto& unpacked = ticket.keep_buffer();
            vkdev->convert_packing(residual_input_gpu, unpacked, 1, 0, command, implementation.option);
            residual_input_gpu = unpacked;
        }
        if (use_native && residual_input_gpu.empty())
            return false;
        if (projected_gpu.elempack != residual_input_gpu.elempack)
        {
            auto& packed = ticket.keep_buffer();
            vkdev->convert_packing(projected_gpu, packed, residual_input_gpu.elempack,
                                   command, implementation.option);
            if (packed.empty())
                return false;
            projected_gpu = packed;
        }
        auto& add_input = ticket.keep_list({residual_input_gpu, projected_gpu});
        auto& add_output = ticket.keep_list(1);
        if (implementation.add->forward(add_input, add_output, command, implementation.option) != 0)
            return false;
        output = add_output[0];
    }
    else
    {
        output = projected_gpu;
    }

    auto& download_gpu = ticket.keep_buffer(output);
    if (download_gpu.elempack != 1)
    {
        auto& unpacked = ticket.keep_buffer();
        vkdev->convert_packing(download_gpu, unpacked, 1, command, implementation.option);
        download_gpu = unpacked;
    }
    if (download_gpu.empty())
        return false;
    output = download_gpu;

    const uint64_t total_actual_tokens = cache.token_count + token_count;
    const uint64_t retained_tokens = config.sliding_window == 0
                                         ? total_actual_tokens
                                         : std::min<uint64_t>(total_actual_tokens,
                                                              config.sliding_window > 1 ? config.sliding_window - 1 : 0);
    const uint64_t dropped_tokens = total_actual_tokens - retained_tokens;
    const uint64_t next_first_slot = retained_tokens == 0 ? 0 : (ring_first_slot + dropped_tokens) % ring_capacity;
    const uint64_t allocated_cache_size = retained_tokens == 0
                                              ? 0
                                              : static_cast<uint64_t>(next_cache->key.cstep)
                                                        * next_cache->key.c * next_cache->key.elemsize
                                                    + static_cast<uint64_t>(next_cache->value.cstep)
                                                          * next_cache->value.c * next_cache->value.elemsize;

    ticket.context = implementation.vulkan_context;
    ticket.config = config;
    ticket.next_cache = next_cache;
    ticket.retained_tokens = retained_tokens;
    ticket.dropped_tokens = dropped_tokens;
    ticket.next_first_slot = next_first_slot;
    ticket.ring_capacity = ring_capacity;
    ticket.allocated_cache_size = allocated_cache_size;
    ticket.allocate_ring = allocate_ring;
    ticket.ring_resized = ring_resized;
    ticket.qkv_rope = use_qkv_rope;
    ticket.device_rope = device_rope;
    ticket.qkv_ring = use_qkv_ring;
    ticket.decode_sdpa = use_decode_sdpa;
    ticket.decode_matrix = use_decode_sdpa && implementation.decode_matrix != nullptr;
    ticket.wrapped = ring_first_slot + destination_count > ring_capacity;
    ticket.promoted = promote_host_cache;
    ticket.recorded = true;
    return true;
}

bool Attention_vulkan::forward(uint64_t position_offset, LayerCache& cache,
                               const ActivationBuffer& input, ActivationBuffer& output) const
{
    const auto& implementation = *d;
    const auto& context = implementation.vulkan_context;
    VkTransferLease lease = context->acquire_transfer_slot();
    VkTransferSlot& slot = lease.slot();
    if (!fill_staging_upload(input, slot.upload, slot.staging_allocator)
        || !prepare_staging_batch(slot.download, input.rows(), input.columns(), slot.staging_allocator))
        return false;

    const std::lock_guard<std::mutex> lock(context->command_mutex());
    auto& command = *slot.command;
    if (slot.command_used && command.reset() != 0)
        return false;
    slot.command_used = true;

    AttentionWork_vulkan work;
    ncnn::VkMat input_gpu;
    ncnn::VkMat packed_input;
    ncnn::VkMat output_gpu;
    bool attempted = false;
    bool successful = false;
    const auto cleanup = [&](void*) {
        if (!successful)
        {
            if (attempted)
                work.mark_submit_failed();
            (void)command.reset();
        }
    };
    std::unique_ptr<void, decltype(cleanup)> guard(reinterpret_cast<void*>(1), cleanup);

    const bool direct_host_input = input.rows() == 1 && activation_variant(implementation.option) == 0
                                   && context->support_direct_host_buffer(static_cast<size_t>(input.columns()) * sizeof(float),
                                                                          input.dtype());
    if (direct_host_input)
        input_gpu = bind_host_in(slot.upload);
    else if (!record_mapped_act_upload(slot.upload, input_gpu, command, context->device(),
                                       implementation.option, input.dtype()))
        return false;

    // Public recorder expects unpacked FP32; legacy host upload may pick pack4.
    if (input_gpu.elempack != 1)
    {
        packed_input = input_gpu;
        context->device()->convert_packing(packed_input, input_gpu, 1, command, implementation.option);
        if (input_gpu.empty())
            return false;
    }
    if (!record_forward(position_offset, cache, input_gpu, output_gpu, command, work)
        || !record_staging_download(output_gpu, input.rows(), input.columns(), slot.download,
                                    command, context->device(), implementation.option))
        return false;
    attempted = true;
    if (submit_wait(command, context->device()) != 0)
        return false;

    ActivationBuffer completed(input.rows(), input.columns());
    if (!copy_staging_to_cpu_batch(slot.download, completed)
        || std::any_of(completed.values().begin(), completed.values().end(),
                       [](float value) { return !std::isfinite(value); })
        || !work.commit())
        return false;

    output.swap(completed);
    auto& counters = context->runtime_state();
    ++counters.compute_submissions;
    ++counters.batch_uploads;
    ++counters.batch_downloads;
    successful = true;
    return true;
}

bool Attention_vulkan::materialize_device_cache(LayerCache& cache) const
{
    const Implementation& implementation = *d;
    const AttentionConfig_vulkan& config = implementation.config;
    if (!implementation.vulkan_context || !cache.vulkan_attention_cache)
        return false;
    const AttentionCache_vulkan& device_cache = *cache.vulkan_attention_cache;
    if (device_cache.key.empty() || device_cache.value.empty())
        return false;
    const size_t element_size = act_element_size(implementation.kv_option);

    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    VkTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    if (!create_staging(transfer_slot.attention_cache_key, static_cast<int>(config.head_dimension),
                        static_cast<int>(cache.token_count), static_cast<int>(config.kv_head_count),
                        element_size, transfer_slot.staging_allocator)
        || !create_staging(transfer_slot.attention_cache_value, static_cast<int>(config.head_dimension),
                           static_cast<int>(cache.token_count), static_cast<int>(config.kv_head_count),
                           element_size, transfer_slot.staging_allocator))
        return false;

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used && command.reset() != 0)
        return false;
    transfer_slot.command_used = true;

    ncnn::VkMat& key_staging = transfer_slot.attention_cache_key;
    ncnn::VkMat& value_staging = transfer_slot.attention_cache_value;
    if (!record_attention_copy(implementation.ring_append_pipeline.get(), device_cache.key,
                               device_cache.value, key_staging, value_staging,
                               cache.capacity_tokens, 0, false, cache.first_slot, true, false, command)
        || key_staging.empty() || value_staging.empty())
        return false;

    if (submit_wait(command, implementation.vulkan_context->device()) != 0)
        return false;
    key_staging.allocator->invalidate(key_staging.data);
    value_staging.allocator->invalidate(value_staging.data);
    const ncnn::Mat key_mapped = key_staging.mapped();
    const ncnn::Mat value_mapped = value_staging.mapped();

    const size_t storage_variant = activation_variant(implementation.kv_option);
    ncnn::Mat key_float_storage;
    ncnn::Mat value_float_storage;
    const ncnn::Mat* key_float_source = &key_mapped;
    const ncnn::Mat* value_float_source = &value_mapped;
    if (storage_variant == 1 || (storage_variant == 2 && cache.dtype == DType::Float32))
    {
        ncnn::Option cast_option = implementation.option;
        if (storage_variant == 1)
        {
            // Cast is not part of the trimmed layer set; widen fp16 directly.
            key_float_storage.create(key_mapped.w, key_mapped.h, key_mapped.c, sizeof(float));
            value_float_storage.create(value_mapped.w, value_mapped.h, value_mapped.c, sizeof(float));
            if (key_float_storage.empty() || value_float_storage.empty())
                return false;
            for (int channel = 0; channel < key_mapped.c; ++channel)
            {
                const ncnn::Mat key_channel = key_mapped.channel(channel);
                const ncnn::Mat value_channel = value_mapped.channel(channel);
                ncnn::Mat key_float_channel = key_float_storage.channel(channel);
                ncnn::Mat value_float_channel = value_float_storage.channel(channel);
                for (int row = 0; row < key_mapped.h; ++row)
                {
                    const uint16_t* key_source = key_channel.row<uint16_t>(row);
                    const uint16_t* value_source = value_channel.row<uint16_t>(row);
                    float* key_destination = key_float_channel.row<float>(row);
                    float* value_destination = value_float_channel.row<float>(row);
                    for (int column = 0; column < key_mapped.w; ++column)
                    {
                        key_destination[column] = ncnn::float16_to_float32(key_source[column]);
                        value_destination[column] = ncnn::float16_to_float32(value_source[column]);
                    }
                }
            }
        }
        else
        {
            ncnn::cast_bfloat16_to_float32(key_mapped, key_float_storage, cast_option);
            ncnn::cast_bfloat16_to_float32(value_mapped, value_float_storage, cast_option);
        }
        if (key_float_storage.empty() || value_float_storage.empty())
            return false;
        key_float_source = &key_float_storage;
        value_float_source = &value_float_storage;
    }

    const uint64_t columns_u64 = static_cast<uint64_t>(config.kv_head_count) * config.head_dimension;
    if (columns_u64 == 0
        || cache.token_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / columns_u64))
        return false;

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
                uint16_t* key_destination = keys.data() + static_cast<size_t>(token) * columns
                                            + static_cast<size_t>(head) * config.head_dimension;
                uint16_t* value_destination = values.data() + static_cast<size_t>(token) * columns
                                              + static_cast<size_t>(head) * config.head_dimension;
                if (storage_variant == 2)
                {
                    std::copy_n(key_channel.row<uint16_t>(static_cast<int>(token)),
                                config.head_dimension, key_destination);
                    std::copy_n(value_channel.row<uint16_t>(static_cast<int>(token)),
                                config.head_dimension, value_destination);
                }
                else
                {
                    const float* key_source = key_float_channel.row<float>(static_cast<int>(token));
                    const float* value_source = value_float_channel.row<float>(static_cast<int>(token));
                    for (uint32_t column = 0; column < config.head_dimension; ++column)
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
                std::copy_n(key_channel.row<float>(static_cast<int>(token)), config.head_dimension,
                            keys.data() + static_cast<size_t>(token) * columns
                                + static_cast<size_t>(head) * config.head_dimension);
                std::copy_n(value_channel.row<float>(static_cast<int>(token)), config.head_dimension,
                            values.data() + static_cast<size_t>(token) * columns
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
    // Restore a valid host shadow and avoid repeating the same promotion next call.
    cache.vulkan_attention_promotion_disabled = true;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_downloads;
    return true;
}

AttentionBatchResult_vulkan Attention_vulkan::forward_batch(std::span<const AttentionBatchEntry_vulkan> entries) const
{
    const uint64_t optimization_flags = d->config.optimization_flags;
    if (entries.size() < 2
        || !has_flag(optimization_flags, OptimizationVulkanAttention)
        || !has_flag(optimization_flags, OptimizationVulkanAttentionBatch))
        return AttentionBatchResult_vulkan::NotExecuted;
    // Batched recorder has no post-attention norm stage; executor uses per-session path.
    if (d->post_attention_norm)
        return AttentionBatchResult_vulkan::NotExecuted;

    const Implementation& implementation = *d;
    VulkanRuntimeState& runtime_state = implementation.vulkan_context->runtime_state();
    const AttentionConfig_vulkan& config = implementation.config;
    const bool low_precision_kv = activation_variant(implementation.kv_option) != 0;
    const bool bfloat16_storage = config.activation_dtype == DType::BFloat16 && implementation.option.use_bf16_storage;
    const uint64_t query_columns_u64 = static_cast<uint64_t>(config.head_count) * config.head_dimension;
    const uint32_t query_columns = static_cast<uint32_t>(query_columns_u64);
    const uint64_t sink_token_count = has_flag(config.flags, AttentionSink) ? 1 : 0;
    const bool use_norm_rope = implementation.query_key_norm || implementation.output_gate
                               || (config.rope_head_dimension != 0
                                   && config.rope_head_dimension < config.head_dimension);
    const bool use_native = static_cast<bool>(implementation.fused_native_qkv
                                              || implementation.native_projections[0].qnk
                                              || implementation.native_projections[0].float8);
    const Linear* fused = implementation.fused_qkv.get();
    const Bfloat16Linear_vulkan* fused_gate = implementation.fused_qkv_gate.get();
    const Linear* projection = implementation.output_projection.get();
    const Bfloat16Linear_vulkan* projection_bfloat16 = implementation.output_projection_bfloat16.get();
    const bool use_qkv_rope = support_qkv_rope(1);
    const bool use_qkv_ring = uses_qkv_ring(1);
    if (!use_qkv_rope && use_norm_rope)
        return AttentionBatchResult_vulkan::NotExecuted;

    VkTransferLease transfer_lease = implementation.vulkan_context->acquire_transfer_slot();
    VkTransferSlot& transfer_slot = transfer_lease.slot();
    const bool direct_host_input = activation_variant(implementation.option) == 0
                                   && implementation.vulkan_context->support_direct_host_buffer(static_cast<size_t>(config.hidden_size) * sizeof(float),
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
        std::vector<ncnn::VkMat> projection_workspace;
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
        LayerCache& cache = *entry.cache;
        const ActivationBuffer& input = *entry.input;
        const uint64_t actual_token_count = cache.token_count + 1;
        const uint64_t destination_count = actual_token_count + sink_token_count;
        if (actual_token_count <= cache.token_count
            || actual_token_count > cache.capacity_tokens
            || destination_count < actual_token_count)
            return AttentionBatchResult_vulkan::NotExecuted;

        prepared.emplace_back();
        PreparedAttentionEntry& work = prepared.back();
        work.entry = &entry;
        work.device_rope = has_flag(config.optimization_flags, OptimizationVulkanDeviceRope)
                           && !implementation.rope_inverse_frequencies_gpu.empty()
                           && entry.position_offset <= std::numeric_limits<uint32_t>::max();
        if (!use_qkv_rope && work.device_rope)
            return AttentionBatchResult_vulkan::NotExecuted;
        work.actual_token_count = actual_token_count;
        work.destination_count = destination_count;
        const bool decode_split_available = implementation.merge_pipeline && implementation.split_pipeline;
        work.use_decode_sdpa = has_flag(config.optimization_flags, OptimizationVulkanDecodeSdpa)
                               && attention_decode_window_is_visible(cache, entry.position_offset,
                                                                     actual_token_count, config.sliding_window)
                               && ((destination_count <= decode_single_workgroup_tokens
                                    && config.head_dimension <= 128)
                                   || (config.head_dimension <= 256
                                       && split_range(destination_count, decode_split_available).splits != 0));
        work.next_cache = cache.vulkan_attention_cache;
        work.retained_tokens = config.sliding_window == 0
                                   ? actual_token_count
                                   : std::min<uint64_t>(actual_token_count,
                                                        config.sliding_window > 1 ? config.sliding_window - 1 : 0);
        work.dropped_tokens = actual_token_count - work.retained_tokens;
        work.ring_first_slot = cache.first_slot;
        work.next_first_slot = work.retained_tokens == 0
                                   ? 0
                                   : (cache.first_slot + work.dropped_tokens) % cache.capacity_tokens;
        work.allocated_cache_size = work.retained_tokens == 0
                                        ? 0
                                        : static_cast<uint64_t>(work.next_cache->key.cstep)
                                                  * work.next_cache->key.c * work.next_cache->key.elemsize
                                              + static_cast<uint64_t>(work.next_cache->value.cstep)
                                                    * work.next_cache->value.c * work.next_cache->value.elemsize;

        if (!fill_staging_upload(input, work.upload, transfer_slot.staging_allocator)
            || (!work.device_rope
                && !fill_rope_staging_pair(work.rope_cosine, work.rope_sine, 1, entry.position_offset,
                                           implementation.rope_inverse_frequencies,
                                           implementation.rope_concentration, bfloat16_storage,
                                           transfer_slot.staging_allocator))
            || (!work.use_decode_sdpa
                && !fill_attention_mask_staging(work.attention_mask, 1, destination_count,
                                                entry.position_offset, cache, config, implementation.sinks,
                                                bfloat16_storage, transfer_slot.staging_allocator))
            || !prepare_staging_batch(work.download, 1, config.hidden_size, transfer_slot.staging_allocator))
            return AttentionBatchResult_vulkan::NotExecuted;
        entry.output->reset(1, config.hidden_size, false);
    }

    std::unique_lock<std::mutex> lock(implementation.vulkan_context->command_mutex());
    ncnn::VulkanDevice* vkdev = implementation.vulkan_context->device();
    ncnn::VkCompute& command = *transfer_slot.command;
    if (transfer_slot.command_used && command.reset() != 0)
        return AttentionBatchResult_vulkan::NotExecuted;
    transfer_slot.command_used = true;

    const auto unpack_in_place = [&](std::vector<ncnn::VkMat>& retained, ncnn::VkMat& mat) {
        if (mat.elempack == 1)
            return;
        retained.push_back(mat);
        ncnn::VkMat unpacked;
        vkdev->convert_packing(mat, unpacked, 1, command, implementation.option);
        mat = unpacked;
    };

    for (PreparedAttentionEntry& work : prepared)
    {
        LayerCache& cache = *work.entry->cache;
        if (work.device_rope)
        {
            work.cosine_gpu = implementation.rope_inverse_frequencies_gpu;
            work.sine_gpu = implementation.rope_inverse_frequencies_gpu;
        }
        if (direct_host_input && work.entry->input->dtype() == DType::Float32)
            work.input_gpu = bind_host_in(work.upload);
        else if (!record_prepared_staging_upload(work.upload, 1, work.input_gpu, command, vkdev,
                                                 implementation.option, work.entry->input->dtype()))
            return AttentionBatchResult_vulkan::NotExecuted;

        if (!work.device_rope
            && (!record_mapped_act_upload(work.rope_cosine, work.cosine_gpu, command, vkdev, implementation.option)
                || !record_mapped_act_upload(work.rope_sine, work.sine_gpu, command, vkdev, implementation.option)))
            return AttentionBatchResult_vulkan::NotExecuted;
        if (!work.use_decode_sdpa
            && !record_mapped_act_upload(work.attention_mask, work.mask_gpu, command, vkdev, implementation.option))
            return AttentionBatchResult_vulkan::NotExecuted;

        const bool normalized = implementation.layer_norm
                                    ? implementation.layer_norm->record(work.input_gpu, work.normalized_gpu, command)
                                    : implementation.norm->record(work.input_gpu, work.normalized_gpu, command);
        if (!normalized)
            return AttentionBatchResult_vulkan::NotExecuted;

        if (use_native || use_norm_rope || fused_gate)
        {
            work.normalized_unpacked_gpu = work.normalized_gpu;
            unpack_in_place(work.retained_gpu, work.normalized_unpacked_gpu);
            if (use_native)
            {
                if (!record_native_qkv(work.normalized_unpacked_gpu, work.fused_gpu,
                                       command, work.projection_workspace))
                    return AttentionBatchResult_vulkan::NotExecuted;
            }
            else if (!fused_gate || work.normalized_unpacked_gpu.empty()
                     || work.normalized_unpacked_gpu.dims != 2 || work.normalized_unpacked_gpu.h != 1
                     || fused_gate->forward(work.normalized_unpacked_gpu, work.fused_gpu,
                                            command, implementation.option)
                            != 0)
            {
                return AttentionBatchResult_vulkan::NotExecuted;
            }
        }
        else if (!fused || fused->forward(work.normalized_gpu, work.fused_gpu, command, implementation.option) != 0)
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }

        const uint64_t ring_capacity = cache.capacity_tokens;
        const uint64_t ring_first_slot = cache.first_slot;
        const uint64_t append_slot = (ring_first_slot + cache.token_count) % ring_capacity;
        work.fused_qkv_unpacked_gpu = work.fused_gpu;
        unpack_in_place(work.retained_gpu, work.fused_qkv_unpacked_gpu);
        if (use_qkv_rope)
        {
            const AttentionCache_vulkan* ring = use_qkv_ring ? work.next_cache.get() : nullptr;
            const bool recorded_rope = use_norm_rope
                                           ? record_qkv_norm_rope(work.fused_qkv_unpacked_gpu, work.cosine_gpu,
                                                                  work.sine_gpu, 1, work.entry->position_offset,
                                                                  work.device_rope, ring, ring_capacity,
                                                                  append_slot, work.query_rope, work.key_rope,
                                                                  work.value_heads, work.output_gate, command)
                                           : record_qkv_rope(work.fused_qkv_unpacked_gpu, work.cosine_gpu,
                                                             work.sine_gpu, 1, work.entry->position_offset,
                                                             work.device_rope, ring, ring_capacity,
                                                             append_slot, work.query_rope, work.key_rope,
                                                             work.value_heads, command);
            if (!recorded_rope)
                return AttentionBatchResult_vulkan::NotExecuted;
        }
        else
        {
            std::vector<ncnn::VkMat> qkv_input(1, work.fused_gpu);
            work.qkv.resize(3);
            if (implementation.slice_qkv->forward(qkv_input, work.qkv, command, implementation.option) != 0
                || implementation.reshape_query->forward(work.qkv[0], work.query_shaped_gpu,
                                                         command, implementation.option)
                       != 0
                || implementation.reshape_key_value->forward(work.qkv[1], work.key_shaped_gpu,
                                                             command, implementation.option)
                       != 0
                || implementation.reshape_key_value->forward(work.qkv[2], work.value_shaped_gpu,
                                                             command, implementation.option)
                       != 0
                || implementation.permute_heads_tokens->forward(work.query_shaped_gpu, work.query_heads_gpu,
                                                                command, implementation.option)
                       != 0
                || implementation.permute_heads_tokens->forward(work.key_shaped_gpu, work.key_heads_gpu,
                                                                command, implementation.option)
                       != 0
                || implementation.permute_heads_tokens->forward(work.value_shaped_gpu, work.value_heads,
                                                                command, implementation.option)
                       != 0)
                return AttentionBatchResult_vulkan::NotExecuted;

            unpack_in_place(work.retained_gpu, work.query_heads_gpu);
            unpack_in_place(work.retained_gpu, work.key_heads_gpu);
            unpack_in_place(work.retained_gpu, work.value_heads);

            std::vector<ncnn::VkMat> query_rope_input = {work.query_heads_gpu, work.cosine_gpu, work.sine_gpu};
            std::vector<ncnn::VkMat> key_rope_input = {work.key_heads_gpu, work.cosine_gpu, work.sine_gpu};
            work.query_rope_output.resize(1);
            work.key_rope_output.resize(1);
            if (implementation.rotary->forward(query_rope_input, work.query_rope_output,
                                               command, implementation.option)
                    != 0
                || implementation.rotary->forward(key_rope_input, work.key_rope_output,
                                                  command, implementation.option)
                       != 0)
                return AttentionBatchResult_vulkan::NotExecuted;
            work.query_rope = work.query_rope_output[0];
            work.key_rope = work.key_rope_output[0];
        }

        if (!use_qkv_ring
            && !record_attention_copy(implementation.ring_append_pipeline.get(), work.key_rope,
                                      work.value_heads, work.next_cache->key, work.next_cache->value,
                                      ring_capacity, append_slot, config.kv_dtype == DType::BFloat16,
                                      0, false, config.sliding_window != 0, command))
            return AttentionBatchResult_vulkan::NotExecuted;

        if (sink_token_count != 0 && !use_qkv_ring)
        {
            const uint64_t sink_row = ring_first_slot + work.actual_token_count;
            if (!record_attention_ring_zero(implementation.ring_zero_pipeline.get(),
                                            work.next_cache->key, work.next_cache->value, sink_row, command))
                return AttentionBatchResult_vulkan::NotExecuted;
        }
        work.combined_key = attention_ring_view(work.next_cache->key, ring_first_slot, work.destination_count);
        work.combined_value = attention_ring_view(work.next_cache->value, ring_first_slot, work.destination_count);
        if (work.combined_key.empty() || work.combined_value.empty())
            return AttentionBatchResult_vulkan::NotExecuted;

        if (work.use_decode_sdpa)
        {
            if (!record_decode(work.query_rope, work.combined_key, work.combined_value,
                               work.destination_count, work.attention_matrix, command,
                               work.projection_workspace))
                return AttentionBatchResult_vulkan::Failed;
        }
        else
        {
            if (sink_token_count != 0 && use_qkv_ring)
            {
                const uint64_t sink_row = ring_first_slot + work.actual_token_count;
                if (!record_attention_ring_zero(implementation.ring_zero_pipeline.get(),
                                                work.next_cache->key, work.next_cache->value, sink_row, command))
                    return AttentionBatchResult_vulkan::NotExecuted;
            }
            ncnn::VkMat sdpa_key;
            ncnn::VkMat sdpa_value;
            if (low_precision_kv)
            {
                ncnn::VkMat converted_key;
                ncnn::VkMat converted_value;
                converted_key.create(work.combined_key.w, work.combined_key.h, work.combined_key.c,
                                     sizeof(float), implementation.option.blob_vkallocator);
                converted_value.create(work.combined_value.w, work.combined_value.h, work.combined_value.c,
                                       sizeof(float), implementation.option.blob_vkallocator);
                if (converted_key.empty() || converted_value.empty()
                    || !record_attention_copy(implementation.ring_append_pipeline.get(),
                                              work.next_cache->key, work.next_cache->value,
                                              converted_key, converted_value, ring_capacity, 0,
                                              false, ring_first_slot, true, false, command))
                    return AttentionBatchResult_vulkan::NotExecuted;
                work.retained_gpu.push_back(converted_key);
                work.retained_gpu.push_back(converted_value);
                sdpa_key = converted_key;
                sdpa_value = converted_value;
            }
            else
            {
                sdpa_key = work.combined_key;
                sdpa_value = work.combined_value;
            }
            std::vector<ncnn::VkMat> sdpa_input = {work.query_rope, sdpa_key, sdpa_value, work.mask_gpu};
            work.sdpa_output.resize(1);
            if (implementation.sdpa->forward(sdpa_input, work.sdpa_output, command, implementation.option) != 0
                || implementation.permute_heads_tokens->forward(work.sdpa_output[0], work.attention_token_major,
                                                                command, implementation.option)
                       != 0
                || implementation.reshape_attention->forward(work.attention_token_major, work.attention_matrix,
                                                             command, implementation.option)
                       != 0)
                return AttentionBatchResult_vulkan::NotExecuted;
        }

        if (implementation.output_gate || use_native || projection_bfloat16)
        {
            unpack_in_place(work.retained_gpu, work.attention_matrix);
            if (implementation.output_gate
                && !record_attention_output_gate(implementation.output_gate_pipeline.get(),
                                                 work.attention_matrix, work.output_gate, 1,
                                                 query_columns, command))
                return AttentionBatchResult_vulkan::NotExecuted;
        }

        if (projection_bfloat16)
        {
            if (work.attention_matrix.empty()
                || projection_bfloat16->forward(work.attention_matrix, work.projected_gpu,
                                                command, implementation.option)
                       != 0)
                return AttentionBatchResult_vulkan::NotExecuted;
        }
        else if (use_native)
        {
            const CompiledOperator& output_operator = implementation.native_projections[3];
            const bool recorded = output_operator.qnk
                                      ? output_operator.qnk->record_forward(work.attention_matrix,
                                                                            work.projected_gpu, command)
                                      : output_operator.float8
                                            && output_operator.float8->record_forward(work.attention_matrix,
                                                                                      work.projected_gpu,
                                                                                      command,
                                                                                      work.projection_workspace);
            if (!recorded)
                return AttentionBatchResult_vulkan::NotExecuted;
        }
        else if (!projection
                 || projection->forward(work.attention_matrix, work.projected_gpu,
                                        command, implementation.option)
                        != 0)
        {
            return AttentionBatchResult_vulkan::NotExecuted;
        }

        std::vector<ncnn::VkMat> add_input = {work.input_gpu, work.projected_gpu};
        work.add_output.resize(1);
        if (implementation.add->forward(add_input, work.add_output, command, implementation.option) != 0)
            return AttentionBatchResult_vulkan::NotExecuted;
        work.download_gpu = work.add_output[0];
        unpack_in_place(work.retained_gpu, work.download_gpu);
        if (!record_staging_download(work.download_gpu, 1, config.hidden_size, work.download,
                                     command, vkdev, implementation.option, work.entry->output->dtype()))
            return AttentionBatchResult_vulkan::NotExecuted;
    }

    const auto mark_device_states_unknown = [&]() noexcept {
        for (const PreparedAttentionEntry& work : prepared)
            work.entry->cache->vulkan_attention_state_unknown = true;
    };
    if (submit_wait(command, implementation.vulkan_context->device()) != 0)
    {
        mark_device_states_unknown();
        return AttentionBatchResult_vulkan::Failed;
    }
    for (PreparedAttentionEntry& work : prepared)
    {
        if (!copy_staging_to_cpu_batch(work.download, *work.entry->output))
        {
            mark_device_states_unknown();
            return AttentionBatchResult_vulkan::Failed;
        }
        for (uint32_t column = 0; column < work.entry->output->columns(); ++column)
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
    if (representative.use_decode_sdpa && implementation.decode_matrix)
        ++runtime_state.kv_ring_appends;
    if (representative.ring_first_slot + representative.destination_count
        > representative.entry->cache->capacity_tokens)
        ++runtime_state.kv_ring_wrapped_views;
    ++runtime_state.compute_submissions;
    ++runtime_state.batch_uploads;
    ++runtime_state.batch_downloads;
    return AttentionBatchResult_vulkan::Executed;
}

} // namespace moe
} // namespace ncnn
