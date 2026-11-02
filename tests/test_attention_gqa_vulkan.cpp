#include "backends/ncnn/attentiondecode_vulkan.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void gqa_check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

#if NCNN_MOE_WITH_VULKAN
// Independent scalar conversions, dot products and softmax: this oracle does
// not call a runtime Attention operator or reuse its reduction implementation.
static uint16_t bf16_bits(float value)
{
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static double bf16_value(uint16_t value)
{
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

static double bf16_round(double value)
{
    return bf16_value(bf16_bits(static_cast<float>(value)));
}

static std::array<double, 3> bf16_components(float value)
{
    const float high = static_cast<float>(bf16_round(value));
    const float first_residual = value - high;
    const float middle = static_cast<float>(bf16_round(first_residual));
    const float second_residual = first_residual - middle;
    return {high, middle, bf16_round(second_residual)};
}

static double bf16_compensated(double value)
{
    const auto components = bf16_components(static_cast<float>(value));
    return components[0] + components[1] + components[2];
}

static double gamma_bound(uint64_t operations)
{
    const double product = operations * static_cast<double>(std::numeric_limits<float>::epsilon());
    return product / (1.0 - product);
}

struct GqaInputs
{
    uint32_t query_step = 0, key_step = 0, value_step = 0;
    size_t query_prefix = 0, key_prefix = 0, value_prefix = 0;
    std::vector<float> query, sinks;
    std::vector<uint16_t> key, value;
};

static GqaInputs make_inputs(const AttentionDecodeShape& shape, size_t alignment, bool extreme)
{
    GqaInputs input;
    input.query_step = shape.head_dimension + 16;
    input.key_step = shape.kv_tokens * shape.head_dimension + 32;
    input.value_step = shape.kv_tokens * shape.head_dimension + 64;
#if NCNN_BATCH
    const size_t kv_element_size = shape.kv_storage_bfloat16 ? sizeof(uint16_t) : sizeof(float);
    input.query_prefix = alignment / sizeof(float);
    input.key_prefix = alignment / kv_element_size;
    input.value_prefix = 2 * alignment / kv_element_size;
#else
    (void)alignment;
#endif
    input.query.assign(input.query_prefix + input.query_step * shape.query_heads, 9999.0f);
    input.key.assign(input.key_prefix + input.key_step * shape.kv_heads, bf16_bits(8192.0f));
    input.value.assign(input.value_prefix + input.value_step * shape.kv_heads, bf16_bits(-8192.0f));
    input.sinks.resize(shape.query_heads);
    for (uint32_t head = 0; head < shape.query_heads; ++head)
    {
        for (uint32_t column = 0; column < shape.head_dimension; ++column)
            input.query[input.query_prefix + head * input.query_step + column] = extreme
                                                                                     ? 4.0f + static_cast<float>((head + column) % 3) / 16.0f
                                                                                     : static_cast<float>(static_cast<int>((head * 13 + column * 7) % 41) - 20) / 19.0f;
        input.sinks[head] = extreme
                                ? (head % 3 == 0 ? 1200.0f : head % 3 == 1 ? -1200.0f
                                                                           : -std::numeric_limits<float>::infinity())
                                : static_cast<float>(static_cast<int>(head % 11) - 5) * 0.375f;
    }
    for (uint32_t head = 0; head < shape.kv_heads; ++head)
        for (uint32_t token = 0; token < shape.kv_tokens; ++token)
            for (uint32_t column = 0; column < shape.head_dimension; ++column)
            {
                const float key = extreme
                                      ? static_cast<float>(static_cast<int>((head + token) % 9) - 4) * 8.0f
                                            + static_cast<float>(column % 3) / 16.0f
                                      : static_cast<float>(static_cast<int>((head * 17 + token * 11 + column * 3) % 47) - 23) / 16.0f;
                const float value = static_cast<float>(static_cast<int>((head * 19 + token * 7 + column * 13) % 53) - 26) / 16.0f;
                input.key[input.key_prefix + head * input.key_step + token * shape.head_dimension + column] = bf16_bits(key);
                input.value[input.value_prefix + head * input.value_step + token * shape.head_dimension + column] = bf16_bits(value);
            }
    return input;
}

struct GqaOracle
{
    std::vector<double> full, rounded, numerical_bound, full_bound;
    std::vector<double> maxima, sums, maximum_bound;
};

static GqaOracle reference(const AttentionDecodePlan& plan, const GqaInputs& input, bool sink)
{
    const auto& shape = plan.shape;
    const uint32_t dimension = shape.head_dimension;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dimension));
    constexpr double epsilon = std::numeric_limits<float>::epsilon();
    constexpr double bf16_unit_roundoff = 1.0 / 256.0;
    GqaOracle result;
    result.full.resize(shape.query_heads * dimension);
    result.rounded.resize(result.full.size());
    result.numerical_bound.resize(shape.query_heads);
    result.full_bound.resize(shape.query_heads);
    result.maximum_bound.resize(shape.query_heads);
    result.maxima.resize(plan.split_count * shape.query_heads);
    result.sums.resize(result.maxima.size());
    for (uint32_t head = 0; head < shape.query_heads; ++head)
    {
        const uint32_t kv_head = head / plan.group_size;
        std::vector<double> original_scores(shape.kv_tokens), rounded_scores(shape.kv_tokens);
        double score_perturbation = 0.0, dot_magnitude = 0.0, maximum_value = 0.0;
        for (uint32_t token = 0; token < shape.kv_tokens; ++token)
        {
            double original = 0.0, rounded = 0.0, magnitude = 0.0;
            for (uint32_t column = 0; column < dimension; ++column)
            {
                const float query = input.query[input.query_prefix + head * input.query_step + column];
                const double key = bf16_value(input.key[input.key_prefix + kv_head * input.key_step + token * dimension + column]);
                const auto query_parts = bf16_components(query);
                const double term = (query_parts[0] + query_parts[1] + query_parts[2]) * key;
                original += static_cast<double>(query) * key;
                rounded += term;
                magnitude += (std::abs(query_parts[0]) + std::abs(query_parts[1]) + std::abs(query_parts[2])) * std::abs(key);
                maximum_value = std::max(maximum_value,
                                         std::abs(bf16_value(input.value[input.value_prefix + kv_head * input.value_step + token * dimension + column])));
            }
            original_scores[token] = original * scale;
            rounded_scores[token] = rounded * scale;
            dot_magnitude = std::max(dot_magnitude, magnitude * scale);
            score_perturbation = std::max(score_perturbation, std::abs(original_scores[token] - rounded_scores[token]));
        }
        double full_maximum = *std::max_element(original_scores.begin(), original_scores.end());
        if (sink) full_maximum = std::max(full_maximum, static_cast<double>(input.sinks[head]));
        double full_sum = sink ? std::exp(input.sinks[head] - full_maximum) : 0.0;
        for (double& score : original_scores)
        {
            score = std::exp(score - full_maximum);
            full_sum += score;
        }
        for (uint32_t column = 0; column < dimension; ++column)
        {
            double numerator = 0.0;
            for (uint32_t token = 0; token < shape.kv_tokens; ++token)
                numerator += original_scores[token]
                             * bf16_value(input.value[input.value_prefix + kv_head * input.value_step + token * dimension + column]);
            result.full[head * dimension + column] = numerator / full_sum;
        }

        // Emulate only precision boundaries in FP64: unscaled Q and each
        // tile's P use three RNE BF16 terms; scale follows the QK dot product.
        // The denominator uses unrounded P.
        // FP32 operation ordering remains independent of the GPU implementation.
        std::vector<double> numerators(plan.split_count * dimension, 0.0), uncertain_mass(plan.split_count, 0.0);
        const double score_error = gamma_bound(3ull * dimension + 1) * dot_magnitude;
        result.maximum_bound[head] = score_error;
        const double probability_relative_error = std::expm1(2.0 * score_error + 8.0 * epsilon);
        for (uint32_t split = 0; split < plan.split_count; ++split)
        {
            const uint32_t begin = split * plan.split_tokens;
            const uint32_t end = std::min(shape.kv_tokens, begin + plan.split_tokens);
            double maximum = -static_cast<double>(std::numeric_limits<float>::max()) / 2.0;
            double sum = 0.0;
            // The physical cooperative tile is 32 columns, but only 16
            // probability columns participate in each online softmax update.
            constexpr uint32_t probability_tile = 16;
            for (uint32_t first = begin; first < end; first += probability_tile)
            {
                const uint32_t last = std::min(end, first + probability_tile);
                const double next_maximum = std::max(maximum,
                                                     *std::max_element(rounded_scores.begin() + first, rounded_scores.begin() + last));
                const double rescale = std::exp(maximum - next_maximum);
                double tile_sum = 0.0, tile_uncertain_mass = 0.0;
                std::vector<double> tile_numerator(dimension, 0.0);
                for (uint32_t token = first; token < last; ++token)
                {
                    const double probability = std::exp(rounded_scores[token] - next_maximum);
                    const double rounded = bf16_compensated(probability);
                    // A tiny FP32 score error can cross a residual BF16 midpoint.
                    // Bound those data-dependent discontinuities using both
                    // residual terms instead of a fixed BF16 error allowance.
                    const double low = bf16_compensated(std::max(0.0, probability * (1.0 - probability_relative_error)));
                    const double high = bf16_compensated(probability * (1.0 + probability_relative_error));
                    tile_uncertain_mass += std::max(std::abs(high - rounded), std::abs(low - rounded));
                    tile_sum += probability;
                    for (uint32_t column = 0; column < dimension; ++column)
                        tile_numerator[column] += rounded
                                                  * bf16_value(input.value[input.value_prefix + kv_head * input.value_step + token * dimension + column]);
                }
                sum = sum * rescale + tile_sum;
                uncertain_mass[split] = uncertain_mass[split] * rescale + tile_uncertain_mass;
                for (uint32_t column = 0; column < dimension; ++column)
                    numerators[split * dimension + column] = numerators[split * dimension + column] * rescale + tile_numerator[column];
                maximum = next_maximum;
            }
            // Merge sinks after BF16 probability rounding, so their magnitude
            // cannot alter the tile probability quantization.
            if (sink && split == 0 && std::isfinite(input.sinks[head]))
            {
                const double next_maximum = std::max(maximum, static_cast<double>(input.sinks[head]));
                const double rescale = std::exp(maximum - next_maximum);
                sum = sum * rescale + std::exp(input.sinks[head] - next_maximum);
                uncertain_mass[split] *= rescale;
                for (uint32_t column = 0; column < dimension; ++column)
                    numerators[split * dimension + column] *= rescale;
                maximum = next_maximum;
            }
            result.maxima[split * shape.query_heads + head] = maximum;
            result.sums[split * shape.query_heads + head] = sum;
        }
        double merged_maximum = -std::numeric_limits<double>::infinity();
        for (uint32_t split = 0; split < plan.split_count; ++split)
            if (result.sums[split * shape.query_heads + head] > 0.0)
                merged_maximum = std::max(merged_maximum, result.maxima[split * shape.query_heads + head]);
        double denominator = 0.0, merged_uncertain_mass = 0.0;
        for (uint32_t split = 0; split < plan.split_count; ++split)
        {
            const size_t index = split * shape.query_heads + head;
            if (result.sums[index] == 0.0) continue;
            const double factor = std::exp(result.maxima[index] - merged_maximum);
            denominator += result.sums[index] * factor;
            merged_uncertain_mass += uncertain_mass[split] * factor;
            for (uint32_t column = 0; column < dimension; ++column)
                result.rounded[head * dimension + column] += numerators[split * dimension + column] * factor;
        }
        for (uint32_t column = 0; column < dimension; ++column)
            result.rounded[head * dimension + column] /= denominator;
        // gamma(n) bounds FP32 dot/reduction/merge rounding; the explicit
        // uncertain mass above covers residual-P rounding-bin changes. Three
        // QK/PV MMAs triple their accumulation counts; denominator remains FP32.
        result.numerical_bound[head] = maximum_value
                                       * (4.0 * score_error + merged_uncertain_mass / denominator
                                          + gamma_bound(6ull * dimension + 6ull * shape.kv_tokens + 16ull * plan.split_count + 1) + 16.0 * epsilon);
        const double softmax_bound = score_perturbation < 0.5 ? std::expm1(2.0 * score_perturbation) : 2.0;
        // Two successive BF16 residuals bound reconstruction error by u_bf16^3.
        // The FP32 cast of reference P contributes at most epsilon. Unlike the
        // former single-term path, there is no BF16 denominator bias to allow.
        const double precision_bound = maximum_value
                                       * (std::min(2.0, softmax_bound)
                                          + bf16_unit_roundoff * bf16_unit_roundoff * bf16_unit_roundoff * (1.0 + epsilon) + epsilon);
        result.full_bound[head] = precision_bound + result.numerical_bound[head];
        for (uint32_t column = 0; column < dimension; ++column)
        {
            const size_t index = head * dimension + column;
            gqa_check(std::abs(result.rounded[index] - result.full[index]) <= precision_bound,
                      "compensated precision reference exceeded its BF16 residual error bound");
        }
    }
    return result;
}

static ncnn::VkMat head_view(ncnn::VkMat source, uint32_t dimension, uint32_t tokens,
                             uint32_t heads, uint32_t step, size_t prefix)
{
    source.dims = 3;
    source.w = static_cast<int>(dimension);
    source.h = static_cast<int>(tokens);
    source.d = 1;
    source.c = static_cast<int>(heads);
    source.cstep = step;
#if NCNN_BATCH
    source.n = 1;
    source.nstep = static_cast<size_t>(step) * heads;
    source.offset += prefix * source.elemsize;
#else
    (void)prefix;
#endif
    return source;
}

static void rejected_record_cases(const AttentionDecodeVulkan& kernel, const AttentionDecodePlan& plan,
                                  const std::array<ncnn::VkMat, 4>& gpu,
                                  const std::shared_ptr<VulkanContext>& context, ncnn::VkCompute& command)
{
    const auto before = get_vulkan_statistics(context->runtime());
    for (uint32_t invalid_case = 0; invalid_case < 12; ++invalid_case)
    {
        auto invalid = gpu;
        auto invalid_plan = plan;
        switch (invalid_case)
        {
        case 0: invalid[0].cstep = plan.shape.head_dimension - 1; break;
        case 1: invalid[1].cstep = plan.shape.head_dimension * plan.shape.kv_tokens - 1; break;
        case 2: invalid[2].elemsize = plan.shape.kv_storage_bfloat16 ? sizeof(float) : sizeof(uint16_t); break;
        case 3: --invalid[1].h; break;
        case 4: --invalid[3].w; break;
        case 5: invalid[3].release(); break;
        case 6:
            invalid_plan.split_count = 1;
            invalid_plan.split_tokens = 32;
            break;
        case 7: --invalid_plan.scratch_elements; break;
        case 8: invalid_plan.split_count = invalid_plan.scratch_split_capacity + 1; break;
        case 9: invalid[2].cstep *= 4; break;
#if NCNN_BATCH
        case 10: invalid[0].nstep = 0; break;
        case 11: ++invalid[2].offset; break;
#else
        case 10: invalid[0].dims = 2; break;
        case 11: invalid[2].elempack = 4; break;
#endif
        }
        ncnn::VkMat output;
        output.create(3, sizeof(float), 1, context->blob_allocator());
        const auto storage = output.data;
        std::vector<ncnn::VkMat> workspace;
        gqa_check(!kernel.record(invalid_plan, invalid[0], invalid[1], invalid[2], invalid[3], true,
                                 output, command, context->blob_allocator(), workspace)
                      && output.data == storage && output.w == 3 && workspace.empty(),
                  "invalid matrix descriptor/plan published output or scratch: case " + std::to_string(invalid_case));
    }
    const auto after = get_vulkan_statistics(context->runtime());
    gqa_check(after.compute_submissions == before.compute_submissions && after.dispatches == before.dispatches
                  && after.bfloat16_cooperative_matrix_dispatches == before.bfloat16_cooperative_matrix_dispatches,
              "rejected matrix record submitted work or published dispatch counters");
}

static void execute_case(AttentionDecodeVulkan& kernel, const std::shared_ptr<VulkanContext>& context, const ncnn::Option& option,
                         AttentionDecodeCapabilities capabilities, const AttentionDecodeShape& shape,
                         bool sink, bool extreme, bool empty_trailing, double& largest_error)
{
    AttentionDecodePlan plan;
    gqa_check(make_attention_decode_plan(capabilities, shape, plan) && plan.path == AttentionDecodePath::Matrix,
              "supported GPU case did not select matrix Attention");
    if (empty_trailing)
    {
        plan.split_count = plan.scratch_split_capacity;
        gqa_check(static_cast<uint64_t>(plan.split_count - 1) * plan.split_tokens >= shape.kv_tokens,
                  "empty-split fixture does not contain an empty trailing split");
    }
    const auto input = make_inputs(shape, std::max(size_t{16}, context->device()->info.buffer_offset_alignment()), extreme);
    const auto expected = reference(plan, input, sink);
    const std::string label = "D=" + std::to_string(shape.head_dimension) + " GQA=" + std::to_string(plan.group_size)
                              + " KV=" + std::to_string(shape.kv_tokens) + " sink=" + std::to_string(sink)
                              + " storage=" + (shape.kv_storage_bfloat16 ? "BF16" : "FP32")
                              + " splits=" + std::to_string(plan.split_count) + " extreme=" + std::to_string(extreme);
    // Pipeline owners outlive this lock: their deleters take command_mutex.
    // Retain one owner per dimension/storage pair, avoiding repeated shader compilation.
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    gqa_check(kernel.create(context, option, plan), "matrix pipeline creation failed: " + label);
    ncnn::VkCompute command(context->device());
    std::array<ncnn::VkMat, 4> staging, gpu;
    std::vector<float> float_key, float_value;
    const void* key_data = input.key.data();
    const void* value_data = input.value.data();
    const size_t kv_element_size = shape.kv_storage_bfloat16 ? sizeof(uint16_t) : sizeof(float);
    if (!shape.kv_storage_bfloat16)
    {
        // These are the identical BF16 values in a physical FP32 KV ring.
        // The oracle remains unchanged, isolating storage-layout correctness.
        float_key.reserve(input.key.size());
        float_value.reserve(input.value.size());
        for (uint16_t value : input.key) float_key.push_back(static_cast<float>(bf16_value(value)));
        for (uint16_t value : input.value) float_value.push_back(static_cast<float>(bf16_value(value)));
        key_data = float_key.data();
        value_data = float_value.data();
    }
    gqa_check(fill_staging_values(input.query.data(), input.query.size(), sizeof(float), staging[0], option.staging_vkallocator)
                  && fill_staging_values(key_data, input.key.size(), kv_element_size, staging[1], option.staging_vkallocator)
                  && fill_staging_values(value_data, input.value.size(), kv_element_size, staging[2], option.staging_vkallocator)
                  && fill_staging_values(input.sinks.data(), input.sinks.size(), sizeof(float), staging[3], option.staging_vkallocator),
              "Q/K/V staging failed: " + label);
    for (size_t index = 0; index < gpu.size(); ++index)
        gqa_check(record_mapped_upload(staging[index], gpu[index], command, option), "Q/K/V upload recording failed: " + label);
    gpu[0] = head_view(gpu[0], shape.head_dimension, 1, shape.query_heads, input.query_step, input.query_prefix);
    gpu[1] = head_view(gpu[1], shape.head_dimension, shape.kv_tokens, shape.kv_heads, input.key_step, input.key_prefix);
    gpu[2] = head_view(gpu[2], shape.head_dimension, shape.kv_tokens, shape.kv_heads, input.value_step, input.value_prefix);
    gpu[3].dims = 1;
    if (shape.head_dimension == 64 && plan.group_size == 20 && shape.kv_tokens == 129 && sink && !extreme && !empty_trailing)
        rejected_record_cases(kernel, plan, gpu, context, command);
    ncnn::VkMat output;
    std::vector<ncnn::VkMat> workspace;
    const auto before = get_vulkan_statistics(context->runtime());
    gqa_check(kernel.record(plan, gpu[0], gpu[1], gpu[2], gpu[3], sink, output, command,
                            option.blob_vkallocator, workspace),
              "matrix recording failed: " + label);
    const auto recorded = get_vulkan_statistics(context->runtime());
    gqa_check(recorded.compute_submissions == before.compute_submissions
                  && recorded.batch_uploads == before.batch_uploads && recorded.batch_downloads == before.batch_downloads
                  && recorded.attention_cpu_fallbacks == before.attention_cpu_fallbacks
                  && recorded.dispatches == before.dispatches
                  && recorded.bfloat16_cooperative_matrix_dispatches == before.bfloat16_cooperative_matrix_dispatches,
              "record-only Attention submitted or performed host transfers: " + label);
    ncnn::Mat observed;
    command.record_download(output, observed, option);
    gqa_check(submit_compute_and_wait(command, context->device()) == 0 && !observed.empty()
                  && observed.total() == static_cast<size_t>(shape.query_heads) * shape.head_dimension,
              "matrix execution/download failed: " + label);
    const float* actual = static_cast<const float*>(observed.data);
    for (uint32_t head = 0; head < shape.query_heads; ++head)
        for (uint32_t column = 0; column < shape.head_dimension; ++column)
        {
            const size_t index = head * shape.head_dimension + column;
            const double precision_error = std::abs(actual[index] - expected.rounded[index]);
            const double full_error = std::abs(actual[index] - expected.full[index]);
            largest_error = std::max(largest_error, full_error);
            gqa_check(std::isfinite(actual[index]) && precision_error <= expected.numerical_bound[head]
                          && full_error <= expected.full_bound[head],
                      "FP64 oracle mismatch: " + label + " index=" + std::to_string(index)
                          + " actual=" + std::to_string(actual[index]) + " rounded=" + std::to_string(expected.rounded[index])
                          + " full=" + std::to_string(expected.full[index]) + " precision_bound=" + std::to_string(expected.numerical_bound[head]));
        }
}

static int run(uint32_t device)
{
    const auto context = VulkanContext::acquire(device, create_vulkan_runtime(), OptimizationDefaultFlags);
    if (!context) return 77;
    auto capabilities = query_attention_decode_capabilities(context->device()->info);
    const auto device_capabilities = capabilities;
    AttentionDecodePlan probe;
    gqa_check(make_attention_decode_plan(capabilities, {16, 2, 64, 129, true}, probe), "capability probe failed");
    if (probe.path != AttentionDecodePath::Matrix)
    {
        gqa_check(probe.path == AttentionDecodePath::Scalar && probe.scratch_bytes == 0,
                  "unsupported device did not choose scalar before allocation");
        std::cout << "SKIP: device " << device << " selects scalar Attention; BF16 workgroup matrix support is unavailable\n";
        return 77;
    }
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.vulkan_device_index = device;
    option.use_packing_layout = false;
    option.use_fp16_packed = option.use_fp16_storage = option.use_fp16_arithmetic = false;
    option.use_bf16_packed = option.use_bf16_storage = false;
    option.blob_vkallocator = option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    std::array<std::array<AttentionDecodeVulkan, 8>, 2> kernels;
    double largest_error = 0.0;
    uint32_t cases = 0, skips = 0;
    const auto execute = [&](uint32_t dimension, uint32_t kv_heads, uint32_t ratio, uint32_t tokens,
                             bool storage_bfloat16, bool sink, bool extreme, bool empty_trailing = false) {
        auto selected_capabilities = device_capabilities;
        if (empty_trailing) selected_capabilities.device_parallelism = 8;
        const AttentionDecodeShape shape{kv_heads * ratio, kv_heads, dimension, tokens, true, storage_bfloat16};
        AttentionDecodePlan plan;
        gqa_check(make_attention_decode_plan(selected_capabilities, shape, plan), "shape failed planning");
        if (plan.path != AttentionDecodePath::Matrix)
        {
            ++skips;
            std::cout << "SKIP: device " << device << " has no matrix capability for D=" << dimension << '\n';
            return;
        }
        execute_case(kernels[storage_bfloat16 ? 1 : 0][dimension / 16 - 1], context, option,
                     selected_capabilities, shape, sink, extreme, empty_trailing, largest_error);
        ++cases;
    };
    struct Case
    {
        uint32_t dimension, kv_heads, ratio, tokens;
        bool sink;
    };
    // Cover every matrix dimension and query-tile/KV tails without crossing
    // all independent axes. Both physical KV storage layouts use this set.
    const std::array<Case, 15> cases_to_run = {{{32, 2, 2, 1, false}, {64, 2, 8, 15, true}, {64, 2, 8, 16, false}, {64, 2, 8, 17, true}, {32, 2, 4, 31, false}, {128, 2, 16, 32, true}, {128, 2, 40, 33, false}, {64, 2, 20, 129, true}, {16, 3, 3, 33, true}, {48, 3, 5, 129, false}, {80, 3, 3, 33, true}, {96, 3, 5, 129, true}, {112, 3, 3, 33, false}, {64, 1, 40, 513, false}, {128, 8, 8, 1025, true}}};
    for (bool storage_bfloat16 : {true, false})
    {
        for (const auto& test : cases_to_run)
            execute(test.dimension, test.kv_heads, test.ratio, test.tokens, storage_bfloat16, test.sink, false);
        execute(64, 3, 5, 1025, storage_bfloat16, true, true);
        execute(64, 2, 20, 129, storage_bfloat16, false, false, true);
        execute(64, 2, 20, 129, storage_bfloat16, true, true, true);
    }
    std::cout << "GQA matrix/split-KV FP64 and precision-aware oracle passed " << cases
              << " cases (" << skips << " capability skips), max full-FP64 abs error " << largest_error
              << "; BF16/FP32 physical KV, record-only, independent strides, view offsets, sinks and empty splits checked\n";
    return 0;
}
#endif

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::moe::get_gpu_count() == 0) return 77;
    try
    {
        uint32_t device = ncnn::moe::get_default_gpu_index();
        if (argc != 1)
        {
            ncnn::moe::gqa_check(argc == 3 && std::string(argv[1]) == "--device",
                                 "usage: attention_gqa_vulkan [--device N]");
            size_t consumed = 0;
            const unsigned long requested = std::stoul(argv[2], &consumed);
            ncnn::moe::gqa_check(consumed == std::string(argv[2]).size() && requested < ncnn::moe::get_gpu_count(),
                                 "Vulkan device index is out of range");
            device = static_cast<uint32_t>(requested);
        }
        return ncnn::moe::run(device);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
#else
    (void)argc;
    (void)argv;
    return 77;
#endif
}
