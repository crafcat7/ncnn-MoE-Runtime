#include "backends/ncnn/latentattention_vulkan.h"
#include "backends/ncnn/linear.h"
#include "backends/ncnn/vulkan.h"
#include "backends/ncnn/vulkancontext.h"
#include "graph/compiledoperator.h"
#include "kernels/attention.h"
#include "kernels/float8.h"
#include "kernels/latentattention.h"
#include "kernels/ops.h"
#include "kernels/statecache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void latent_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void latent_check_output(const ActivationBuffer& actual,
                                const ActivationBuffer& expected,
                                const std::string& label,
                                float absolute_tolerance = 0.003f,
                                float relative_tolerance = 0.003f)
{
    latent_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + ": output shape");
    float maximum_reference = 0.0f;
    for (float value : expected.values())
        maximum_reference = std::max(maximum_reference, std::abs(value));
    latent_check(maximum_reference > 0.0001f, label + ": oracle must have nonzero output");
    for (size_t index = 0; index < actual.values().size(); ++index)
    {
        const float observed = actual.values()[index];
        const float reference = expected.values()[index];
        const float tolerance = absolute_tolerance + relative_tolerance * std::abs(reference);
        latent_check(std::isfinite(observed) && std::abs(observed - reference) <= tolerance,
                     label + ": element " + std::to_string(index) + " actual=" + std::to_string(observed)
                         + " expected=" + std::to_string(reference) + " tolerance=" + std::to_string(tolerance));
    }
}

static TensorData latent_float_weight(std::vector<uint32_t> shape, float value)
{
    TensorData result;
    result.dtype = DType::Float32;
    result.shape = std::move(shape);
    result.float32_data.assign(static_cast<size_t>(result.element_count()), value);
    return result;
}

static TensorData latent_fp8_weight(uint32_t rows, uint32_t columns, uint32_t seed, float scale)
{
    TensorData result;
    result.dtype = DType::Float8E4M3;
    result.shape = {rows, columns};
    const size_t count = static_cast<size_t>(result.element_count());
    std::shared_ptr<uint8_t[]> storage(new uint8_t[count], std::default_delete<uint8_t[]>());
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const int sample = static_cast<int>((row * 17 + column * 13 + column / 7 + seed * 11) % 31) - 15;
            storage[static_cast<size_t>(row) * columns + column] = float_to_float8_e4m3(static_cast<float>(sample) * 0.0625f);
        }
    }
    result.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
    result.mapped_size = count;
    const uint32_t column_blocks = (columns + 127) / 128;
    const uint32_t row_blocks = (rows + 127) / 128;
    result.quantization_scales.resize(static_cast<size_t>(row_blocks) * column_blocks);
    for (size_t index = 0; index < result.quantization_scales.size(); ++index)
        result.quantization_scales[index] = scale * (1.0f + static_cast<float>(index % 3) * 0.5f);
    return result;
}

static TensorData latent_bfloat16_weight(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData result;
    result.dtype = DType::BFloat16;
    result.shape = {rows, columns};
    result.bfloat16_data.resize(static_cast<size_t>(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            const int sample = static_cast<int>((row * 19 + column * 7 + seed * 5) % 37) - 18;
            result.bfloat16_data[static_cast<size_t>(row) * columns + column] = float_to_bfloat16(static_cast<float>(sample) * 0.0078125f);
        }
    }
    return result;
}

static AttentionBlockPlan latent_test_plan(uint32_t compression_ratio)
{
    AttentionBlockPlan plan;
    plan.kind = AttentionKind::MultiHeadLatent;
    plan.head_count = 4;
    plan.kv_head_count = 1;
    plan.head_dimension = 512;
    plan.rope_head_dimension = 64;
    plan.sliding_window = 5;
    plan.query_lora_rank = 128;
    plan.output_lora_rank = 64;
    plan.output_group_count = 2;
    plan.initial_context_length = 4096;
    plan.compression_ratio = compression_ratio;
    plan.rope_theta = 10000.0f;
    plan.compressed_rope_theta = 1000000.0f;
    plan.rope_scaling_factor = 8.0f;
    plan.rope_ntk_alpha = 1.0f;
    plan.rope_ntk_beta = 32.0f;
    plan.norm_epsilon = 1e-6f;
    plan.index_head_count = 2;
    plan.index_head_dimension = 128;
    plan.index_top_k = 1;
    return plan;
}

static void latent_oracle_rope(float* values, uint64_t position, const AttentionBlockPlan& plan, bool inverse)
{
    const uint32_t dimension = plan.rope_head_dimension;
    const bool yarn = plan.compression_ratio != 0;
    const float base = yarn ? plan.compressed_rope_theta : plan.rope_theta;
    int low = 0;
    int high = 0;
    if (yarn)
    {
        const float denominator = 2.0f * std::log(base);
        const float low_rotations = static_cast<float>(dimension)
                                    * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_beta * 2.0f * 3.14159265358979323846f))
                                    / denominator;
        const float high_rotations = static_cast<float>(dimension)
                                     * std::log(static_cast<float>(plan.initial_context_length) / (plan.rope_ntk_alpha * 2.0f * 3.14159265358979323846f))
                                     / denominator;
        low = std::max(0, static_cast<int>(std::floor(low_rotations)));
        high = std::min(static_cast<int>(dimension) - 1, static_cast<int>(std::ceil(high_rotations)));
    }
    for (uint32_t pair = 0; pair < dimension / 2; ++pair)
    {
        float frequency = 1.0f / std::pow(base, static_cast<float>(pair * 2) / static_cast<float>(dimension));
        if (yarn)
        {
            const float denominator = low == high ? 0.001f : static_cast<float>(high - low);
            const float smooth = 1.0f - std::clamp((static_cast<float>(pair) - static_cast<float>(low)) / denominator, 0.0f, 1.0f);
            frequency = frequency / plan.rope_scaling_factor * (1.0f - smooth) + frequency * smooth;
        }
        const float angle = (inverse ? -1.0f : 1.0f) * static_cast<float>(position) * frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float real = values[pair * 2];
        const float imaginary = values[pair * 2 + 1];
        values[pair * 2] = real * cosine - imaginary * sine;
        values[pair * 2 + 1] = real * sine + imaginary * cosine;
    }
}

static ActivationBuffer latent_oracle(const AttentionBlockPlan& plan,
                                      const TensorData& sinks,
                                      const TensorData& output_a,
                                      const TensorData& output_b,
                                      const Float8Linear_vulkan& projection_a,
                                      const Float8Linear_vulkan& projection_b,
                                      std::span<const uint64_t> positions,
                                      std::span<LayerCache* const> caches,
                                      std::span<const LatentAttentionRowContext> contexts,
                                      const ActivationBuffer& raw_query,
                                      DType output_dtype = DType::Float32)
{
    ActivationBuffer attention(raw_query.rows(), raw_query.columns());
    for (size_t row = 0; row < raw_query.rows(); ++row)
    {
        const LayerCache& cache = *caches[row];
        const LatentAttentionRowContext& context = contexts[row];
        for (uint32_t head = 0; head < plan.head_count; ++head)
        {
            std::vector<float> query(plan.head_dimension);
            const float* source = raw_query.row(row) + static_cast<size_t>(head) * plan.head_dimension;
            float square_sum = 0.0f;
            for (uint32_t column = 0; column < plan.head_dimension; ++column)
                square_sum += source[column] * source[column];
            const float normalization = 1.0f / std::sqrt(square_sum / static_cast<float>(plan.head_dimension) + plan.norm_epsilon);
            for (uint32_t column = 0; column < plan.head_dimension; ++column)
                query[column] = source[column] * normalization;
            latent_oracle_rope(query.data() + plan.head_dimension - plan.rope_head_dimension, positions[row], plan, false);

            std::vector<const float*> keys;
            for (uint32_t candidate = 0; candidate < context.window_count; ++candidate)
            {
                const uint64_t position = context.window_begin + candidate;
                keys.push_back(cache.latent_window.data() + static_cast<size_t>(position % plan.sliding_window) * plan.head_dimension);
            }
            const uint32_t selected_count = context.selected_compressed_indices
                                                ? static_cast<uint32_t>(context.compressed_indices.size())
                                                : context.compressed_count;
            for (uint32_t candidate = 0; candidate < selected_count; ++candidate)
            {
                const uint32_t index = context.selected_compressed_indices ? context.compressed_indices[candidate] : candidate;
                keys.push_back(cache.latent_compressed.data() + static_cast<size_t>(index) * plan.head_dimension);
            }
            std::vector<float> logits(keys.size());
            float maximum = sinks.float32_values()[head];
            for (size_t candidate = 0; candidate < keys.size(); ++candidate)
            {
                float dot = 0.0f;
                for (uint32_t column = 0; column < plan.head_dimension; ++column)
                    dot += query[column] * keys[candidate][column];
                logits[candidate] = dot / std::sqrt(static_cast<float>(plan.head_dimension));
                maximum = std::max(maximum, logits[candidate]);
            }
            float denominator = std::exp(sinks.float32_values()[head] - maximum);
            float* destination = attention.row(row) + static_cast<size_t>(head) * plan.head_dimension;
            for (size_t candidate = 0; candidate < keys.size(); ++candidate)
            {
                const float probability = std::exp(logits[candidate] - maximum);
                denominator += probability;
                for (uint32_t column = 0; column < plan.head_dimension; ++column)
                    destination[column] += probability * keys[candidate][column];
            }
            for (uint32_t column = 0; column < plan.head_dimension; ++column)
                destination[column] /= denominator;
            latent_oracle_rope(destination + plan.head_dimension - plan.rope_head_dimension, positions[row], plan, true);
        }
    }
    const uint32_t group_columns = plan.head_count / plan.output_group_count * plan.head_dimension;
    const uint32_t blocks = (group_columns + 127) / 128;
    ActivationBuffer rank(raw_query.rows(), plan.output_group_count * plan.output_lora_rank);
    for (size_t row = 0; row < raw_query.rows(); ++row)
    {
        for (uint32_t group = 0; group < plan.output_group_count; ++group)
        {
            const float* input = attention.row(row) + static_cast<size_t>(group) * group_columns;
            for (uint32_t component = 0; component < plan.output_lora_rank; ++component)
            {
                const uint32_t matrix_row = group * plan.output_lora_rank + component;
                rank.row(row)[matrix_row] = float8_e4m3_block_dot(output_a.float8_values().data() + static_cast<size_t>(matrix_row) * group_columns,
                                                                  output_a.quantization_scales.data() + static_cast<size_t>(matrix_row / 128) * blocks,
                                                                  input, group_columns, 128);
            }
        }
    }
    const ActivationBuffer cpu_projection = forward_linear(output_b, rank, 0);
    latent_check(std::any_of(cpu_projection.values().begin(), cpu_projection.values().end(),
                             [](float value) { return std::abs(value) > 0.0001f; }),
                 "CPU attention and grouped output projection oracle must be nonzero");
    // Keep the existing Hybrid output projection's activation quantization.
    // The independently calculated attention is the input to the existing chain.
    ActivationBuffer expected(0, 0, output_dtype);
    latent_check(projection_a.forward_chain(attention, projection_b, expected), "project CPU attention oracle using existing Hybrid FP8 chain");
    return expected;
}

static ActivationBuffer latent_query(size_t rows, const AttentionBlockPlan& plan, uint32_t seed)
{
    ActivationBuffer result(rows, plan.head_count * plan.head_dimension);
    for (size_t row = 0; row < rows; ++row)
    {
        for (uint32_t column = 0; column < result.columns(); ++column)
        {
            const float phase = static_cast<float>(column * 7 + seed * 19 + row * 31) * 0.017f;
            result.row(row)[column] = column < plan.head_dimension && seed == 0 ? 0.0f : std::sin(phase) + 0.23f * std::cos(phase * 0.37f);
        }
    }
    return result;
}

static void latent_append_cache(LayerCache& cache, const AttentionBlockPlan& plan, uint64_t position, uint32_t seed)
{
    cache.latent_cache = true;
    cache.columns = plan.head_dimension;
    cache.capacity_tokens = plan.sliding_window;
    cache.latent_window.resize(static_cast<size_t>(plan.sliding_window) * plan.head_dimension);
    float* destination = cache.latent_window.data() + static_cast<size_t>(position % plan.sliding_window) * plan.head_dimension;
    for (uint32_t column = 0; column < plan.head_dimension; ++column)
    {
        const float phase = static_cast<float>(column * 3 + position * 17 + seed * 13) * 0.023f;
        destination[column] = 0.45f * std::sin(phase) + 0.13f * std::cos(phase * 0.29f);
    }
    cache.latent_token_count = position + 1;
    if (plan.compression_ratio != 0 && (position + 1) % plan.compression_ratio == 0)
    {
        const size_t first = cache.latent_compressed.size();
        const uint32_t index = static_cast<uint32_t>(first / plan.head_dimension);
        cache.latent_compressed.resize(first + plan.head_dimension);
        for (uint32_t column = 0; column < plan.head_dimension; ++column)
        {
            const float phase = static_cast<float>(column * 11 + index * 43 + seed * 5) * 0.019f;
            cache.latent_compressed[first + column] = 0.67f * std::cos(phase) - 0.21f * std::sin(phase * 0.43f);
        }
    }
}

static LatentAttentionRowContext latent_context(const AttentionBlockPlan& plan, const LayerCache& cache, uint64_t position, bool select)
{
    LatentAttentionRowContext context;
    context.window_begin = position + 1 > plan.sliding_window ? position + 1 - plan.sliding_window : 0;
    context.window_count = static_cast<uint32_t>(position + 1 - context.window_begin);
    context.compressed_count = static_cast<uint32_t>(cache.latent_compressed.size() / plan.head_dimension);
    context.selected_compressed_indices = select;
    if (select)
        context.compressed_indices = cache.latent_selected_indices;
    return context;
}

struct LatentVulkanFixture
{
    AttentionBlockPlan plan;
    TensorData sinks;
    TensorData output_a;
    TensorData output_b;
    VulkanRuntimePtr runtime;
    std::shared_ptr<Float8Linear_vulkan> projection_a;
    std::shared_ptr<Float8Linear_vulkan> projection_b;
    std::shared_ptr<LatentAttention_vulkan> core;

    explicit LatentVulkanFixture(uint32_t compression_ratio)
        : plan(latent_test_plan(compression_ratio)),
          sinks(latent_float_weight({4}, 0.0f)),
          output_a(latent_fp8_weight(128, 1024, 3, 0.03125f)),
          output_b(latent_fp8_weight(128, 128, 11, 0.125f)),
          runtime(create_vulkan_runtime())
    {
        sinks.float32_data = {-1.5f, 0.0f, 3.0f, 12.0f};
        projection_a = Float8Linear_vulkan::create(output_a, nullptr, 2, get_default_gpu_index(), runtime, 0);
        projection_b = Float8Linear_vulkan::create(output_b, nullptr, 1, get_default_gpu_index(), runtime, 0);
        latent_check(projection_a && projection_b, "create nonzero FP8 output projections");
        core = LatentAttention_vulkan::create(plan, sinks, projection_a, projection_b);
        latent_check(static_cast<bool>(core), "create 512-dimension latent Vulkan core");
    }
};

static void latent_test_incremental(uint32_t ratio)
{
    LatentVulkanFixture fixture(ratio);
    LayerCache cache;
    std::array<LayerCache*, 1> caches = {&cache};
    const uint64_t last_position = ratio == 128 ? 258 : 13;
    for (uint64_t position = 0; position <= last_position; ++position)
    {
        latent_append_cache(cache, fixture.plan, position, 5);
        const uint32_t compressed_count = static_cast<uint32_t>(cache.latent_compressed.size() / fixture.plan.head_dimension);
        const bool selected = ratio == 4 && compressed_count > 1;
        if (selected)
            cache.latent_selected_indices = {compressed_count - 1, 0};
        std::array<uint64_t, 1> positions = {position};
        std::array<LatentAttentionRowContext, 1> contexts = {latent_context(fixture.plan, cache, position, selected)};
        ActivationBuffer query = latent_query(1, fixture.plan, static_cast<uint32_t>(position));
        const ActivationBuffer expected = latent_oracle(fixture.plan, fixture.sinks, fixture.output_a, fixture.output_b, *fixture.projection_a, *fixture.projection_b, positions, caches, contexts, query);
        ActivationBuffer actual;
        const VulkanStatistics before = get_vulkan_statistics(fixture.runtime);
        const LatentCache_vulkan* previous_mirror = cache.latent_device_state.get();
        latent_check(fixture.core->forward_batch(positions, caches, contexts, query, nullptr, actual), "incremental GPU MLA forward");
        const VulkanStatistics after = get_vulkan_statistics(fixture.runtime);
        latent_check(after.attention_blocks == before.attention_blocks + 1, "incremental GPU attention counter");
        latent_check(after.compute_submissions == before.compute_submissions + 1, "one submission for RMS/RoPE/SDPA/output projection");
        latent_check(after.batch_downloads == before.batch_downloads + 1, "only final output download");
        if (position != 0)
            latent_check(cache.latent_device_state.get() == previous_mirror, "warm decode reuses the GPU mirror");
        if (ratio == 0 && position != 0)
            latent_check(after.batch_uploads == before.batch_uploads + 3, "warm MLA uploads only query, current key and RoPE coefficients");
        latent_check_output(actual, expected, "ratio " + std::to_string(ratio) + " position " + std::to_string(position));
    }

    LayerCache branch = cache;
    const LatentCache_vulkan* original_mirror = cache.latent_device_state.get();
    latent_append_cache(branch, fixture.plan, last_position + 1, 83);
    branch.latent_selected_indices.clear();
    std::array<LayerCache*, 1> branch_caches = {&branch};
    std::array<uint64_t, 1> branch_positions = {last_position + 1};
    std::array<LatentAttentionRowContext, 1> branch_contexts = {latent_context(fixture.plan, branch, last_position + 1, false)};
    ActivationBuffer branch_query = latent_query(1, fixture.plan, 83);
    const ActivationBuffer branch_expected = latent_oracle(fixture.plan, fixture.sinks, fixture.output_a, fixture.output_b, *fixture.projection_a, *fixture.projection_b, branch_positions, branch_caches, branch_contexts, branch_query);
    ActivationBuffer branch_actual;
    latent_check(fixture.core->forward_batch(branch_positions, branch_caches, branch_contexts, branch_query, nullptr, branch_actual), "copied-cache branch GPU MLA forward");
    latent_check(branch.latent_device_state.get() != original_mirror, "copied cache detaches its shared GPU mirror");
    latent_check(cache.latent_device_state.get() == original_mirror, "copied-cache branch keeps the original GPU mirror");
    latent_check_output(branch_actual, branch_expected, "copied-cache branch");

    // A rewind keeps the same mirror object; the changed token count must force a refresh.
    cache.latent_compressed.clear();
    cache.latent_selected_indices.clear();
    for (uint64_t position = 0; position <= 2; ++position)
        latent_append_cache(cache, fixture.plan, position, 97);
    std::array<uint64_t, 1> positions = {2};
    std::array<LatentAttentionRowContext, 1> contexts = {latent_context(fixture.plan, cache, 2, false)};
    ActivationBuffer query = latent_query(1, fixture.plan, 91);
    ActivationBuffer expected = latent_oracle(fixture.plan, fixture.sinks, fixture.output_a, fixture.output_b, *fixture.projection_a, *fixture.projection_b, positions, caches, contexts, query);
    ActivationBuffer actual;
    latent_check(fixture.core->forward_batch(positions, caches, contexts, query, nullptr, actual), "rewound cache GPU MLA forward");
    latent_check_output(actual, expected, "rewound mirror refresh");

    cache = LayerCache{};
    latent_append_cache(cache, fixture.plan, 0, 113);
    positions[0] = 0;
    contexts[0] = latent_context(fixture.plan, cache, 0, false);
    expected = latent_oracle(fixture.plan, fixture.sinks, fixture.output_a, fixture.output_b, *fixture.projection_a, *fixture.projection_b, positions, caches, contexts, query);
    latent_check(fixture.core->forward_batch(positions, caches, contexts, query, nullptr, actual), "reset cache GPU MLA forward");
    latent_check_output(actual, expected, "reset mirror promotion");
}

static void latent_test_independent_batch()
{
    LatentVulkanFixture fixture(4);
    std::array<LayerCache, 3> cache_storage;
    std::array<LayerCache*, 3> caches = {&cache_storage[0], &cache_storage[1], &cache_storage[2]};
    std::array<uint64_t, 3> positions = {12, 5, 9};
    std::array<LatentAttentionRowContext, 3> contexts;
    for (size_t row = 0; row < caches.size(); ++row)
    {
        for (uint64_t position = 0; position <= positions[row]; ++position)
            latent_append_cache(*caches[row], fixture.plan, position, static_cast<uint32_t>(row) * 7);
        if (row != 1)
        {
            const uint32_t count = static_cast<uint32_t>(caches[row]->latent_compressed.size() / fixture.plan.head_dimension);
            caches[row]->latent_selected_indices = {count - 1, 0};
        }
        contexts[row] = latent_context(fixture.plan, *caches[row], positions[row], row != 1);
    }
    ActivationBuffer query = latent_query(caches.size(), fixture.plan, 29);
    const ActivationBuffer expected = latent_oracle(fixture.plan, fixture.sinks, fixture.output_a, fixture.output_b, *fixture.projection_a, *fixture.projection_b, positions, caches, contexts, query);
    ActivationBuffer actual;
    latent_check(fixture.core->forward_batch(positions, caches, contexts, query, nullptr, actual), "independent-cache GPU MLA batch");
    latent_check_output(actual, expected, "independent-cache batch with different window and compressed lengths");

    std::array<LayerCache*, 3> repeated = {caches[0], caches[0], caches[2]};
    positions[1] = positions[0];
    contexts[1] = contexts[0];
    latent_check(!fixture.core->forward_batch(positions, repeated, contexts, query, nullptr, actual), "same-cache multirow core must fall back");
}

static void latent_test_bfloat16_output()
{
    LatentVulkanFixture fixture(4);
    LayerCache cache;
    for (uint64_t position = 0; position <= 9; ++position)
        latent_append_cache(cache, fixture.plan, position, 37);
    cache.latent_selected_indices = {1};
    std::array<LayerCache*, 1> caches = {&cache};
    std::array<uint64_t, 1> positions = {9};
    std::array<LatentAttentionRowContext, 1> contexts = {latent_context(fixture.plan, cache, 9, true)};
    ActivationBuffer query = latent_query(1, fixture.plan, 47);
    const ActivationBuffer expected = latent_oracle(fixture.plan, fixture.sinks, fixture.output_a, fixture.output_b,
                                                    *fixture.projection_a, *fixture.projection_b, positions, caches, contexts, query,
                                                    DType::BFloat16);
    ActivationBuffer actual(0, 0, DType::BFloat16);
    latent_check(actual.dtype() == DType::BFloat16 && expected.dtype() == DType::BFloat16, "BF16 output and existing projection oracle dtype before forward");
    const VulkanStatistics before = get_vulkan_statistics(fixture.runtime);
    latent_check(fixture.core->forward_batch(positions, caches, contexts, query, nullptr, actual), "BF16 output direct-core GPU MLA forward");
    const VulkanStatistics after = get_vulkan_statistics(fixture.runtime);
    latent_check(actual.dtype() == DType::BFloat16, "GPU MLA preserves requested BF16 output dtype");
    latent_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), "BF16 output shape matches existing GPU projection oracle");
    latent_check(actual.bytes().size() == actual.rows() * actual.columns() * sizeof(uint16_t), "GPU MLA uses BF16 output storage");
    latent_check(after.batch_downloads == before.batch_downloads + 1, "BF16 output direct core downloads only final output");
    float maximum_reference = 0.0f;
    for (size_t index = 0; index < actual.rows() * actual.columns(); ++index)
    {
        uint16_t actual_bits = 0;
        uint16_t expected_bits = 0;
        std::memcpy(&actual_bits, actual.bytes().data() + index * sizeof(uint16_t), sizeof(uint16_t));
        std::memcpy(&expected_bits, expected.bytes().data() + index * sizeof(uint16_t), sizeof(uint16_t));
        const float observed = bfloat16_to_float(actual_bits);
        const float reference = bfloat16_to_float(expected_bits);
        maximum_reference = std::max(maximum_reference, std::abs(reference));
        const float tolerance = 0.003f + 0.008f * std::abs(reference);
        latent_check(std::isfinite(observed) && std::abs(observed - reference) <= tolerance,
                     "BF16 GPU MLA element " + std::to_string(index) + " actual=" + std::to_string(observed)
                         + " expected=" + std::to_string(reference));
    }
    latent_check(maximum_reference > 0.0001f, "BF16 GPU MLA oracle must have nonzero output");

    const std::vector<float> initial_window = cache.latent_window;
    const std::vector<float> initial_compressed = cache.latent_compressed;
    const std::vector<uint32_t> initial_selected_indices = cache.latent_selected_indices;
    const uint64_t initial_token_count = cache.latent_token_count;
    const uint64_t initial_device_bytes = cache.device_allocated_size;
    const LatentCache_vulkan* initial_mirror = cache.latent_device_state.get();
    const std::vector<std::byte> initial_output(actual.bytes().begin(), actual.bytes().end());
    const std::array<uint32_t, 1> invalid_indices = {contexts[0].compressed_count};
    contexts[0].compressed_indices = invalid_indices;
    latent_check(!fixture.core->forward_batch(positions, caches, contexts, query, nullptr, actual), "out-of-range compressed index rejects GPU MLA");
    contexts[0] = latent_context(fixture.plan, cache, positions[0], true);
    contexts[0].window_count = 0;
    latent_check(!fixture.core->forward_batch(positions, caches, contexts, query, nullptr, actual), "empty canonical window rejects GPU MLA");
    latent_check(cache.latent_window == initial_window && cache.latent_compressed == initial_compressed
                     && cache.latent_selected_indices == initial_selected_indices && cache.latent_token_count == initial_token_count,
                 "GPU MLA input rejection leaves canonical CPU cache unchanged");
    latent_check(cache.latent_device_state.get() == initial_mirror && cache.device_allocated_size == initial_device_bytes,
                 "GPU MLA input rejection leaves existing device mirror unchanged");
    latent_check(actual.dtype() == DType::BFloat16 && actual.bytes().size() == initial_output.size()
                     && std::equal(actual.bytes().begin(), actual.bytes().end(), initial_output.begin()),
                 "GPU MLA input rejection leaves BF16 output unchanged");
    latent_check(get_vulkan_statistics(fixture.runtime).compute_submissions == after.compute_submissions,
                 "GPU MLA input rejection submits no GPU work");
}

static TensorHandle latent_add_weight(WeightStore& weights, TensorData tensor)
{
    auto added = weights.add("latent_vulkan_" + std::to_string(weights.size()), std::move(tensor));
    latent_check(static_cast<bool>(added), "add synthetic MLA weight");
    return added.value();
}

#if NCNN_MOE_WITH_VULKAN
static bool latent_forward_record(const std::shared_ptr<LatentAttention_vulkan>& core,
                                  const ActivationBuffer& input,
                                  std::span<const uint64_t> positions,
                                  std::span<LayerCache* const> caches,
                                  ActivationBuffer& output,
                                  bool cancel = false)
{
    const auto& context = core->vulkan_context();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    ncnn::VkCompute cmd(context->device(), context->command_optimization_flags());
    ncnn::VkMat input_staging;
    ncnn::VkMat device_input;
    ncnn::VkMat device_output;
    ncnn::VkMat output_staging;
    LatentAttentionWork_vulkan work;
    if (!fill_staging_upload(input, input_staging, context->staging_allocator())
        || !record_mapped_upload(input_staging, device_input, cmd, core->option())
        || !core->record_batch(device_input, positions, caches, device_output, cmd, work))
        return false;
    if (cancel)
        return true;
    if (!prepare_staging_batch(output_staging, input.rows(), static_cast<uint32_t>(device_output.w), context->staging_allocator(), output.element_size())
        || !record_prepared_activation_staging_download(device_output, input.rows(), static_cast<uint32_t>(device_output.w), output_staging,
                                                        cmd, context->device(), core->option(), output.dtype()))
        return false;
    if (submit_compute_and_wait(cmd, context->device()) != 0)
        return false;
    ActivationBuffer completed(input.rows(), static_cast<uint32_t>(device_output.w), output.dtype());
    if (!copy_staging_to_cpu_batch(output_staging, completed) || !work.commit())
        return false;
    output.swap(completed);
    ++context->runtime_state().compute_submissions;
    ++context->runtime_state().batch_uploads;
    ++context->runtime_state().batch_downloads;
    return true;
}

static void latent_check_cache_values(const std::vector<float>& actual,
                                      const std::vector<float>& expected,
                                      const std::string& label)
{
    latent_check(actual.size() == expected.size(), label + ": canonical shape");
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const float tolerance = 0.015f + 0.015f * std::abs(expected[index]);
        latent_check(std::isfinite(actual[index]) && std::abs(actual[index] - expected[index]) <= tolerance,
                     label + ": canonical element " + std::to_string(index) + " actual=" + std::to_string(actual[index])
                         + " expected=" + std::to_string(expected[index]));
    }
}
#endif

static void latent_test_end_to_end(uint32_t ratio, bool record_path = false)
{
    const uint64_t flags = static_cast<uint64_t>(OptimizationVulkanAttention)
                           | static_cast<uint64_t>(OptimizationVulkanLatentInputRmsNorm)
                           | (record_path ? static_cast<uint64_t>(OptimizationVulkanLatentCompressor) : 0);
    WeightStore weights;
    AttentionBlockPlan plan = latent_test_plan(ratio);
    plan.pre_attention_norm_weight = latent_add_weight(weights, latent_float_weight({128}, 1.0f));
    plan.query_a_weight = latent_add_weight(weights, latent_fp8_weight(128, 128, 1, 0.25f));
    plan.query_norm_weight = latent_add_weight(weights, latent_float_weight({128}, 1.0f));
    plan.query_b_weight = latent_add_weight(weights, latent_fp8_weight(2048, 128, 5, 0.125f));
    plan.key_value_weight = latent_add_weight(weights, latent_fp8_weight(512, 128, 7, 0.125f));
    plan.key_value_norm_weight = latent_add_weight(weights, latent_float_weight({512}, 1.0f));
    TensorData sinks = latent_float_weight({4}, 0.0f);
    sinks.float32_data = {-1.5f, 0.0f, 3.0f, 12.0f};
    plan.sinks = latent_add_weight(weights, std::move(sinks));
    plan.output_a_weight = latent_add_weight(weights, latent_fp8_weight(128, 1024, 3, 0.03125f));
    plan.output_b_weight = latent_add_weight(weights, latent_fp8_weight(128, 128, 11, 0.125f));
    if (ratio != 0)
    {
        const uint32_t multiplier = ratio == 4 ? 2 : 1;
        plan.compressor_key_value_weight = latent_add_weight(weights, latent_bfloat16_weight(multiplier * 512, 128, 13));
        plan.compressor_gate_weight = latent_add_weight(weights, latent_bfloat16_weight(multiplier * 512, 128, 17));
        plan.compressor_position = latent_add_weight(weights, latent_float_weight({ratio, multiplier * 512}, 0.0f));
        plan.compressor_norm_weight = latent_add_weight(weights, latent_float_weight({512}, 1.0f));
        if (ratio == 4)
        {
            plan.indexer_compressor_key_value_weight = latent_add_weight(weights, latent_bfloat16_weight(256, 128, 19));
            plan.indexer_compressor_gate_weight = latent_add_weight(weights, latent_bfloat16_weight(256, 128, 23));
            plan.indexer_compressor_position = latent_add_weight(weights, latent_float_weight({ratio, 256}, 0.0f));
            plan.indexer_compressor_norm_weight = latent_add_weight(weights, latent_float_weight({128}, 1.0f));
            plan.indexer_query_weight = latent_add_weight(weights, latent_bfloat16_weight(256, 128, 29));
            plan.indexer_weights_weight = latent_add_weight(weights, latent_bfloat16_weight(2, 128, 31));
        }
    }
    CompiledOperatorTable operators;
    operators.bind_weight_count(weights.size());
    const VulkanRuntimePtr runtime = create_vulkan_runtime();
    const std::array<TensorHandle, 5> fp8_handles = {plan.query_a_weight, plan.query_b_weight, plan.key_value_weight, plan.output_a_weight, plan.output_b_weight};
    for (TensorHandle handle : fp8_handles)
    {
        const uint32_t groups = handle == plan.output_a_weight ? plan.output_group_count : 1;
        operators.at_weight_mutable(handle).float8 = Float8Linear_vulkan::create(weights.at(handle), nullptr, groups, get_default_gpu_index(), runtime, flags);
        latent_check(static_cast<bool>(operators.at_weight(handle).float8), "compile FP8 MLA projection");
    }
    latent_check(operators.at_weight(plan.query_a_weight).float8->prepare_rms_norm(weights.at(plan.query_norm_weight), plan.norm_epsilon), "prepare query rank normalization");
    latent_check(operators.at_weight(plan.query_a_weight).float8->prepare_input_rms_norm(weights.at(plan.pre_attention_norm_weight), plan.norm_epsilon), "prepare GPU input normalization");
    operators.at_weight_mutable(plan.query_a_weight).latent_attention = LatentAttention_vulkan::create(plan, weights.at(plan.sinks), operators.at_weight(plan.output_a_weight).float8, operators.at_weight(plan.output_b_weight).float8);
    latent_check(static_cast<bool>(operators.at_weight(plan.query_a_weight).latent_attention), "compile end-to-end GPU MLA core");
#if NCNN_MOE_WITH_VULKAN
    if (record_path)
    {
        const TensorHandle bf16_handles[] = {plan.compressor_key_value_weight, plan.compressor_gate_weight,
                                             plan.indexer_compressor_key_value_weight, plan.indexer_compressor_gate_weight,
                                             plan.indexer_query_weight, plan.indexer_weights_weight};
        for (TensorHandle handle : bf16_handles)
        {
            if (handle == invalid_tensor_handle)
                continue;
            operators.at_weight_mutable(handle).bfloat16 = Bfloat16Linear_vulkan::create(weights.at(handle), nullptr, get_default_gpu_index(), runtime, flags);
            latent_check(static_cast<bool>(operators.at_weight(handle).bfloat16), "compile BF16 compressor/index projection");
        }
        latent_check(operators.at_weight(plan.query_a_weight).latent_attention->prepare(weights, operators, flags), "prepare continuous GPU MLA segment");
        latent_check(operators.at_weight(plan.query_a_weight).latent_attention->can_record(), "continuous GPU MLA capability");
    }
#else
    (void)record_path;
#endif
    CompiledOperatorTable prior_operators = operators;
    prior_operators.at_weight_mutable(plan.query_a_weight).latent_attention.reset();

    LayerCache cpu_cache;
    LayerCache gpu_cache;
    LayerCache prior_cache;
    AttentionScratch cpu_scratch;
    AttentionScratch gpu_scratch;
    AttentionScratch prior_scratch;
    const auto forward_gpu = [&](uint64_t position, LayerCache& cache, const ActivationBuffer& input, ActivationBuffer& output) {
#if NCNN_MOE_WITH_VULKAN
        if (record_path)
        {
            const std::array<uint64_t, 1> positions = {position};
            const std::array<LayerCache*, 1> caches = {&cache};
            return latent_forward_record(operators.at_weight(plan.query_a_weight).latent_attention, input, positions, caches, output);
        }
#endif
        return static_cast<bool>(forward_latent_attention(weights, operators, plan, ExecutionBackend::Vulkan, position, cache, gpu_scratch, input, output, flags));
    };
    const uint64_t count = ratio == 128 ? 129 : 9;
    for (uint64_t position = 0; position < count; ++position)
    {
        ActivationBuffer input(1, 128);
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(0)[column] = std::sin(static_cast<float>(column * 13 + position * 23) * 0.031f) + 0.17f;
        ActivationBuffer expected;
        ActivationBuffer actual;
        ActivationBuffer prior_output;
        latent_check(static_cast<bool>(forward_latent_attention(weights, operators, plan, ExecutionBackend::Cpu, position, cpu_cache, cpu_scratch, input, expected, flags)), "end-to-end CPU MLA forward");
        latent_check(static_cast<bool>(forward_latent_attention(weights, prior_operators, plan, ExecutionBackend::Vulkan, position, prior_cache, prior_scratch, input, prior_output, flags)), "prior Hybrid MLA CPU-attention path");
        const VulkanStatistics before = get_vulkan_statistics(runtime);
        latent_check(forward_gpu(position, gpu_cache, input, actual), "end-to-end GPU MLA forward ratio=" + std::to_string(ratio) + " position=" + std::to_string(position));
        const VulkanStatistics after = get_vulkan_statistics(runtime);
        latent_check(after.attention_blocks == before.attention_blocks + 1, "end-to-end MLA must execute GPU SDPA");
        const bool query_stays_on_device = ratio != 4 || (position + 1) / ratio <= plan.index_top_k;
        if (record_path)
        {
            latent_check(after.compute_submissions == before.compute_submissions + 1, "continuous GPU MLA uses one projection/cache/attention submission");
            latent_check(after.batch_downloads == before.batch_downloads + 2, "continuous GPU MLA downloads one packed canonical shadow and final output");
        }
        else if (query_stays_on_device)
        {
            latent_check(after.compute_submissions == before.compute_submissions + 2, "end-to-end projection and MLA core share two submissions");
            latent_check(after.batch_downloads == before.batch_downloads + 2, "end-to-end downloads only CPU KV projection and final output");
            latent_check(gpu_scratch.query.rows() == 0, "raw query projection stays on GPU");
        }
        const std::string label = "end-to-end ratio " + std::to_string(ratio) + " position " + std::to_string(position);
#if NCNN_MOE_WITH_VULKAN
        if (record_path)
        {
            latent_check_cache_values(gpu_cache.latent_window, prior_cache.latent_window, label + " GPU window");
            latent_check_cache_values(gpu_cache.latent_compressed, prior_cache.latent_compressed, label + " GPU compressor");
            latent_check_cache_values(gpu_cache.latent_index_compressed, prior_cache.latent_index_compressed, label + " GPU index compressor");
            latent_check_cache_values(gpu_cache.compressor_pending_values, prior_cache.compressor_pending_values, label + " GPU compressor pending values");
            latent_check_cache_values(gpu_cache.compressor_pending_scores, prior_cache.compressor_pending_scores, label + " GPU compressor pending scores");
            latent_check(gpu_cache.latent_selected_indices == prior_cache.latent_selected_indices, "GPU compressed index selection parity");
        }
        else
#endif
        {
            latent_check(cpu_cache.latent_compressed == gpu_cache.latent_compressed, "CPU compressor canonical state parity");
            latent_check(cpu_cache.latent_selected_indices == gpu_cache.latent_selected_indices, "CPU compressed index selection parity");
        }
        latent_check_output(actual, prior_output, label + " existing Hybrid parity", record_path ? 0.015f : 0.005f, record_path ? 0.015f : 0.005f);
        // CPU grouped outA uses unquantized activations; Hybrid has always
        // quantized those activations before the FP8 projection.
        latent_check_output(actual, expected, label + " CPU parity including existing output quantization", 0.05f, 0.08f);
    }

    const std::vector<float> initial_window = gpu_cache.latent_window;
    const std::vector<float> initial_compressed = gpu_cache.latent_compressed;
    begin_latent_cache_transaction(std::span<LayerCache>(&gpu_cache, 1));
    for (uint64_t position = count; position < count + 2; ++position)
    {
        ActivationBuffer input(1, 128);
        for (uint32_t column = 0; column < input.columns(); ++column)
            input.row(0)[column] = std::cos(static_cast<float>(column + position * 17) * 0.049f);
        ActivationBuffer output;
        latent_check(forward_gpu(position, gpu_cache, input, output), "speculative GPU MLA forward");
    }
    latent_check(static_cast<bool>(finish_latent_cache_transaction(std::span<LayerCache>(&gpu_cache, 1), 0)), "roll back speculative GPU MLA rows");
    latent_check(!gpu_cache.latent_device_state, "rollback invalidates GPU MLA mirror");
    latent_check(gpu_cache.latent_token_count == cpu_cache.latent_token_count, "rollback restores latent token count");
    latent_check(gpu_cache.latent_window == initial_window, "rollback restores canonical window");
    latent_check(gpu_cache.latent_compressed == initial_compressed, "rollback restores canonical compressed rows");
    ActivationBuffer branch_input(1, 128);
    for (uint32_t column = 0; column < branch_input.columns(); ++column)
        branch_input.row(0)[column] = std::sin(static_cast<float>(column * 3 + count * 11) * 0.029f);
    ActivationBuffer expected;
    ActivationBuffer actual;
    ActivationBuffer prior_output;
    latent_check(static_cast<bool>(forward_latent_attention(weights, operators, plan, ExecutionBackend::Cpu, count, cpu_cache, cpu_scratch, branch_input, expected, flags)), "CPU replacement branch after rollback");
    latent_check(static_cast<bool>(forward_latent_attention(weights, prior_operators, plan, ExecutionBackend::Vulkan, count, prior_cache, prior_scratch, branch_input, prior_output, flags)), "prior Hybrid replacement branch after rollback");
    latent_check(forward_gpu(count, gpu_cache, branch_input, actual), "GPU replacement branch after rollback");
    latent_check_output(actual, prior_output, "GPU MLA replacement branch existing Hybrid parity", record_path ? 0.015f : 0.005f, record_path ? 0.015f : 0.005f);
    latent_check_output(actual, expected, "GPU MLA replacement branch CPU parity including existing output quantization", 0.05f, 0.08f);
#if NCNN_MOE_WITH_VULKAN
    if (record_path)
    {
        const auto& core = operators.at_weight(plan.query_a_weight).latent_attention;
        const std::array<uint64_t, 1> cancel_positions = {count + 1};
        const std::array<LayerCache*, 1> cancel_caches = {&gpu_cache};
        const LayerCache before_cancel = gpu_cache;
        latent_check(latent_forward_record(core, branch_input, cancel_positions, cancel_caches, actual, true), "record then abandon GPU MLA segment");
        latent_check(gpu_cache.latent_window == before_cancel.latent_window && gpu_cache.latent_compressed == before_cancel.latent_compressed
                         && gpu_cache.compressor_pending_values == before_cancel.compressor_pending_values && gpu_cache.latent_token_count == before_cancel.latent_token_count,
                     "abandoned GPU MLA recorder preserves canonical state");
        latent_check(!gpu_cache.latent_device_state, "abandoned GPU MLA recorder invalidates device state");
        LayerCache reference_branch = prior_cache;
        latent_check(static_cast<bool>(forward_latent_attention(weights, prior_operators, plan, ExecutionBackend::Vulkan, count + 1, reference_branch, prior_scratch, branch_input, prior_output, flags)), "prior Hybrid after abandoned recorder");
        latent_check(forward_gpu(count + 1, gpu_cache, branch_input, actual), "GPU MLA reconstructs cache after abandoned recorder");
        latent_check_output(actual, prior_output, "GPU MLA after canceled recorder", 0.015f, 0.015f);

        LayerCache prefill_cache;
        LayerCache prefill_reference;
        ActivationBuffer prefill_input(9, 128);
        for (size_t row = 0; row < prefill_input.rows(); ++row)
            for (uint32_t column = 0; column < prefill_input.columns(); ++column)
                prefill_input.row(row)[column] = std::sin(static_cast<float>(column * 7 + row * 11) * 0.043f);
        const auto prefill_before = get_vulkan_statistics(runtime);
        std::array<uint64_t, 9> prefill_positions = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        std::array<LayerCache*, 9> prefill_caches;
        prefill_caches.fill(&prefill_cache);
        latent_check(latent_forward_record(core, prefill_input, prefill_positions, prefill_caches, actual), "same-cache GPU MLA prefill");
        const auto prefill_after = get_vulkan_statistics(runtime);
        latent_check(prefill_after.attention_blocks == prefill_before.attention_blocks + 9, "all prefill rows execute GPU MLA");
        latent_check(prefill_after.compute_submissions == prefill_before.compute_submissions + 1, "prefill shares one projection/cache/attention submission");
        ActivationBuffer prefill_expected(9, 128);
        for (size_t row = 0; row < prefill_input.rows(); ++row)
        {
            ActivationBuffer row_input(1, 128);
            std::copy_n(prefill_input.row(row), 128, row_input.row(0));
            ActivationBuffer row_output;
            latent_check(static_cast<bool>(forward_latent_attention(weights, prior_operators, plan, ExecutionBackend::Vulkan, row, prefill_reference, prior_scratch, row_input, row_output, flags)), "prefill CPU canonical oracle");
            std::copy_n(row_output.row(0), 128, prefill_expected.row(row));
        }
        latent_check_output(actual, prefill_expected, "GPU MLA prefill row ordering", 0.02f, 0.02f);
        latent_check_cache_values(prefill_cache.latent_window, prefill_reference.latent_window, "prefill canonical window");
        latent_check_cache_values(prefill_cache.latent_compressed, prefill_reference.latent_compressed, "prefill canonical compressed");

        LayerCache independent = gpu_cache;
        const LatentCache_vulkan* shared = gpu_cache.latent_device_state.get();
        ActivationBuffer alternate = branch_input;
        for (uint32_t column = 0; column < alternate.columns(); ++column)
            alternate.row(0)[column] += 0.125f;
        latent_check(forward_gpu(count + 2, independent, alternate, actual), "GPU MLA cloned cache branch");
        latent_check(independent.latent_device_state.get() != shared && gpu_cache.latent_device_state.get() == shared,
                     "GPU MLA recording detaches copied cache state");
        latent_check(gpu_cache.latent_token_count == count + 2, "GPU MLA cloned branch preserves original token count");
        if (ratio == 4)
        {
            for (uint32_t top_k : {plan.sliding_window * plan.head_dimension, 512u})
            {
                AttentionBlockPlan alias_plan = plan;
                alias_plan.index_top_k = top_k;
                CompiledOperatorTable alias_operators = operators;
                CompiledOperator& index_weights_operator = alias_operators.at_weight_mutable(plan.indexer_weights_weight);
                index_weights_operator.bfloat16.reset();
                index_weights_operator.linear = Linear::create(weights.at(plan.indexer_weights_weight), nullptr,
                                                               LinearDevice::Vulkan, get_default_gpu_index(), runtime, flags);
                latent_check(static_cast<bool>(index_weights_operator.linear), "compile generic GPU index weight projection");
                auto alias_core = LatentAttention_vulkan::create(alias_plan, weights.at(plan.sinks),
                                                                 operators.at_weight(plan.output_a_weight).float8,
                                                                 operators.at_weight(plan.output_b_weight).float8);
                latent_check(alias_core && alias_core->prepare(weights, alias_operators, flags), "prepare selected-index allocation regression");
                LayerCache alias_cache;
                alias_cache.latent_cache = true;
                alias_cache.columns = plan.head_dimension;
                alias_cache.capacity_tokens = plan.sliding_window;
                const uint32_t compressed_rows = top_k <= 1024 ? 3001 : top_k + 1;
                const uint64_t alias_position = static_cast<uint64_t>(compressed_rows) * ratio;
                alias_cache.latent_token_count = alias_position;
                alias_cache.latent_window.resize(static_cast<size_t>(plan.sliding_window) * plan.head_dimension);
                for (size_t index = 0; index < alias_cache.latent_window.size(); ++index)
                    alias_cache.latent_window[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.0625f;
                alias_cache.latent_compressed.resize(static_cast<size_t>(compressed_rows) * plan.head_dimension);
                for (size_t index = 0; index < alias_cache.latent_compressed.size(); ++index)
                    alias_cache.latent_compressed[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0625f;
                alias_cache.latent_index_compressed.assign(static_cast<size_t>(compressed_rows) * plan.index_head_dimension, 0.5f);
                LayerCache alias_reference = alias_cache;
                ActivationBuffer alias_expected;
                AttentionScratch alias_scratch;
                latent_check(static_cast<bool>(forward_latent_attention(weights, prior_operators, alias_plan, ExecutionBackend::Vulkan,
                                                                        alias_position, alias_reference, alias_scratch, branch_input, alias_expected, flags)),
                             "selected-index allocation CPU canonical oracle");
                const std::array<uint64_t, 1> alias_positions = {alias_position};
                const std::array<LayerCache*, 1> alias_caches = {&alias_cache};
                const auto alias_before = get_vulkan_statistics(runtime);
                latent_check(latent_forward_record(alias_core, branch_input, alias_positions, alias_caches, actual),
                             "GPU selected-index allocation with window-sized K");
                const auto alias_after = get_vulkan_statistics(runtime);
                latent_check(alias_after.compute_submissions == alias_before.compute_submissions + 1
                                 && alias_after.batch_downloads == alias_before.batch_downloads + 2,
                             "window-sized index selection stays in one GPU segment");
                latent_check(alias_cache.latent_selected_indices == alias_reference.latent_selected_indices,
                             "window-sized K keeps deterministic selected indices");
                latent_check_cache_values(alias_cache.latent_window, alias_reference.latent_window, "window-sized K canonical window");
                latent_check_output(actual, alias_expected, "window-sized K GPU/CPU attention parity", 0.015f, 0.015f);
            }
        }
    }
#endif
}

} // namespace moe
} // namespace ncnn

int main()
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::moe::get_gpu_count() == 0)
    {
        std::cout << "SKIP: Vulkan GPU unavailable\n";
        return 77;
    }
    try
    {
        ncnn::moe::latent_test_incremental(0);
        ncnn::moe::latent_test_incremental(4);
        ncnn::moe::latent_test_incremental(128);
        ncnn::moe::latent_test_independent_batch();
        ncnn::moe::latent_test_bfloat16_output();
        ncnn::moe::latent_test_end_to_end(0);
        ncnn::moe::latent_test_end_to_end(4);
        ncnn::moe::latent_test_end_to_end(128);
        ncnn::moe::latent_test_end_to_end(0, true);
        ncnn::moe::latent_test_end_to_end(4, true);
        ncnn::moe::latent_test_end_to_end(128, true);
        std::cout << "Vulkan MLA tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Vulkan MLA test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
