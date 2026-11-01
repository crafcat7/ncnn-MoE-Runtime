#include "gpu_segment_fixture.h"

#include "backends/ncnn/attention_vulkan.h"
#include "backends/ncnn/modelpipeline.h"

#if NCNN_MOE_WITH_VULKAN
#include "kernels/vulkan/attention_decode_sdpa.comp.hex.h"
#include <pipeline.h>
#endif

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
struct DecodeCase
{
    uint32_t head_dimension;
    DType weight_dtype;
    DType cache_dtype;
    bool sinks;
    uint32_t window;
    uint32_t last_position;
};

static void decode_kernel_cases(const std::shared_ptr<VulkanContext>& context, bool subgroup_reductions)
{
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.vulkan_device_index = context->device()->info.device_index();
    option.use_subgroup_ops = subgroup_reductions;
    option.use_packing_layout = false;
    option.use_fp16_packed = option.use_fp16_storage = option.use_fp16_arithmetic = false;
    option.use_bf16_packed = option.use_bf16_storage = false;
    option.blob_vkallocator = option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const auto binary = context->shader_binary(attention_decode_sdpa_shader,
                                               static_cast<int>(sizeof(attention_decode_sdpa_shader) - 1), option,
                                               subgroup_reductions ? 1 : 0);
    segment_check(binary && !binary->empty(), "independent blocked SDPA shader compilation failed");
    ncnn::Pipeline pipeline(context->device());
    pipeline.set_local_size_xyz(128, 1, 1);
    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = subgroup_reductions ? 1 : 0;
    segment_check(pipeline.create(binary->data(), binary->size() * sizeof(uint32_t), specializations) == 0,
                  "independent blocked SDPA pipeline creation failed");

    uint32_t cases = 0;
    float largest_error = 0.0f;
    // Bind deterministic Q/K/V directly, bypassing normalization, projection,
    // RoPE and the general prefill operator. The reference uses FP64 dot
    // products, one complete softmax and one complete weighted-value sum.
    struct KernelCase
    {
        uint32_t dimension, tokens;
        bool sink;
    };
    const std::array<KernelCase, 12> cases_to_run = {{{10, 1, false}, {64, 15, true}, {128, 16, false}, {64, 17, true}, {10, 31, false}, {128, 32, true}, {64, 33, false}, {10, 63, true}, {128, 64, false}, {64, 65, true}, {128, 129, false}, {64, 513, true}}};
    for (const auto& test : cases_to_run)
    {
        const uint32_t dimension = test.dimension, tokens = test.tokens;
        const bool sink = test.sink;
        constexpr uint32_t heads = 8, kv_heads = 2;
        const uint32_t destination = tokens + static_cast<uint32_t>(sink);
        const uint32_t query_step = dimension + 12;
        const uint32_t kv_step = destination * dimension + 20;
        std::array<ActivationBuffer, 4> inputs = {ActivationBuffer(1, query_step * heads),
                                                  ActivationBuffer(1, kv_step * kv_heads),
                                                  ActivationBuffer(1, kv_step * kv_heads), ActivationBuffer(1, heads)};
        for (auto& input : inputs)
            std::fill_n(input.row(0), input.columns(), 10000.0f);
        for (uint32_t head = 0; head < heads; ++head)
        {
            for (uint32_t column = 0; column < dimension; ++column)
                inputs[0].row(0)[head * query_step + column] = static_cast<float>(static_cast<int>((head * 13 + column * 7) % 41) - 20) / 16.0f;
            inputs[3].row(0)[head] = static_cast<float>(static_cast<int>(head) - 3) * 0.75f;
        }
        for (uint32_t head = 0; head < kv_heads; ++head)
            for (uint32_t token = 0; token < tokens; ++token)
                for (uint32_t column = 0; column < dimension; ++column)
                {
                    const size_t index = head * kv_step + token * dimension + column;
                    inputs[1].row(0)[index] = static_cast<float>(static_cast<int>((head * 17 + token * 11 + column * 3) % 47) - 23) / 16.0f;
                    inputs[2].row(0)[index] = static_cast<float>(static_cast<int>((head * 19 + token * 7 + column * 13) % 53) - 26) / 16.0f;
                }
        const float scale = 1.0f / std::sqrt(static_cast<float>(dimension));
        std::vector<float> expected(heads * dimension);
        for (uint32_t head = 0; head < heads; ++head)
        {
            const uint32_t kv_head = head / (heads / kv_heads);
            std::vector<double> scores(destination);
            for (uint32_t token = 0; token < tokens; ++token)
            {
                double dot = 0.0;
                for (uint32_t column = 0; column < dimension; ++column)
                    dot += static_cast<double>(inputs[0].row(0)[head * query_step + column])
                           * inputs[1].row(0)[kv_head * kv_step + token * dimension + column];
                scores[token] = dot * scale;
            }
            if (sink) scores[tokens] = inputs[3].row(0)[head];
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double denominator = 0.0;
            for (double& score : scores)
            {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (uint32_t column = 0; column < dimension; ++column)
            {
                double numerator = 0.0;
                for (uint32_t token = 0; token < tokens; ++token)
                    numerator += scores[token] * inputs[2].row(0)[kv_head * kv_step + token * dimension + column];
                expected[head * dimension + column] = static_cast<float>(numerator / denominator);
            }
        }

        ncnn::VkCompute command(context->device());
        std::array<ncnn::VkMat, 4> staging, gpu;
        for (size_t index = 0; index < inputs.size(); ++index)
            segment_check(fill_staging_upload(inputs[index], staging[index], option.staging_vkallocator)
                              && record_mapped_upload(staging[index], gpu[index], command, option),
                          "independent blocked SDPA input upload failed");
        ncnn::VkMat output;
        output.create(static_cast<int>(heads * dimension), 1, sizeof(float), 1, option.blob_vkallocator);
        segment_check(!output.empty(), "independent blocked SDPA output allocation failed");
        std::vector<ncnn::vk_constant_type> constants(9);
        constants[0].u32 = dimension;
        constants[1].u32 = destination;
        constants[2].u32 = heads;
        constants[3].u32 = kv_heads;
        constants[4].u32 = query_step;
        constants[5].u32 = constants[6].u32 = kv_step;
        constants[7].u32 = static_cast<uint32_t>(sink);
        constants[8].f = scale;
        ncnn::VkMat dispatcher;
        dispatcher.w = 128;
        dispatcher.h = 1;
        dispatcher.c = heads;
        command.record_pipeline(&pipeline, {gpu[0], gpu[1], gpu[2], gpu[3], output}, constants, dispatcher);
        ncnn::Mat observed;
        command.record_download(output, observed, option);
        segment_check(submit_compute_and_wait(command, context->device()) == 0 && !observed.empty()
                          && observed.elempack == 1 && observed.elemsize == sizeof(float),
                      "independent blocked SDPA submit/download failed");
        const auto* actual = static_cast<const float*>(observed.data);
        for (size_t index = 0; index < expected.size(); ++index)
        {
            const float error = std::abs(actual[index] - expected[index]);
            largest_error = std::max(largest_error, error);
            const float tolerance = 3e-5f + 3e-5f * std::abs(expected[index]);
            segment_check(std::isfinite(actual[index]) && error <= tolerance,
                          "independent blocked SDPA FP64 mismatch: head_dimension=" + std::to_string(dimension)
                              + " tokens=" + std::to_string(tokens) + " sink=" + std::to_string(sink)
                              + " index=" + std::to_string(index) + " actual=" + std::to_string(actual[index])
                              + " expected=" + std::to_string(expected[index]));
        }
        ++cases;
    }
    std::cout << "Independent blocked SDPA FP64 oracle passed " << cases << " cases, path "
              << (subgroup_reductions ? "subgroup" : "shared") << ", subgroup "
              << context->device()->info.subgroup_size() << ", max_abs " << largest_error << std::endl;
}

static void as_float32(TensorData& tensor)
{
    for (uint16_t value : tensor.bfloat16_data)
        tensor.float32_data.push_back(bfloat16_to_float(value));
    tensor.bfloat16_data.clear();
    tensor.dtype = DType::Float32;
}

static CompiledModel decode_model(const DecodeCase& spec, bool gpu, uint32_t device)
{
    auto descriptor = segment_descriptor(false);
    descriptor.layers.resize(1);
    descriptor.activation_dtype = descriptor.kv_cache_dtype = spec.cache_dtype;
    descriptor.head_dimension = spec.head_dimension;
    descriptor.attention_head_count = 4;
    descriptor.kv_head_count = 1;
    auto& attention = descriptor.layers[0].attention;
    attention.head_count = 4;
    attention.kv_head_count = 1;
    attention.head_dimension = attention.qk_rope_head_dimension = spec.head_dimension;
    attention.sliding_window = spec.window;
    attention.projection_weight_dtype = spec.weight_dtype;
    attention.flags = AttentionDescriptorBias | (spec.sinks ? AttentionDescriptorSinks : 0);
    auto mapping = segment_mapping(false);
    const auto add = [&](const char* suffix, std::vector<uint32_t> shape, uint32_t seed) {
        mapping[std::string("layers.0.attention.") + suffix] = segment_weight(std::move(shape), seed);
    };
    const uint32_t query = 4 * spec.head_dimension;
    add("query.weight", {query, segment_hidden}, 101);
    add("key.weight", {spec.head_dimension, segment_hidden}, 102);
    add("value.weight", {spec.head_dimension, segment_hidden}, 103);
    add("output.weight", {segment_hidden, query}, 104);
    add("query.bias", {query}, 105);
    add("key.bias", {spec.head_dimension}, 106);
    add("value.bias", {spec.head_dimension}, 107);
    add("output.bias", {segment_hidden}, 108);
    if (spec.sinks) add("sinks", {4}, 109);
    for (auto& [name, tensor] : mapping)
    {
        const bool projection = name.starts_with("layers.0.attention.") && name.ends_with(".weight");
        if ((projection && spec.weight_dtype == DType::Float32)
            || (!projection && spec.cache_dtype == DType::Float32))
            as_float32(tensor);
    }
    CompilerOption option;
    option.optimization_flags = segment_flags;
    option.num_concurrent_sessions = 2;
    if (gpu)
    {
        option.flags |= BackendVulkanDense | BackendVulkanAttention;
        option.device_index = device;
        option.vulkan_runtime = create_vulkan_runtime();
    }
    CompiledModel result;
    const auto compiled = compile_model(descriptor, mapping, result, gpu ? HybridMode::HybridExperts : HybridMode::CpuOnly, option);
    segment_check(static_cast<bool>(compiled), compiled ? "" : compiled.error().message);
    return result;
}

static ActivationBuffer decode_input(uint32_t rows, uint64_t position)
{
    ActivationBuffer input(rows, segment_hidden);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < segment_hidden; ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((position + row) * 17 + column * 11) % 71 - 35) / 32.0f;
    return input;
}

static void decode_cpu(const CompiledModel& cpu, uint64_t position, LayerCache& cache, AttentionScratch& scratch,
                       const ActivationBuffer& input, ActivationBuffer& output)
{
    const auto result = forward_attention(cpu.weights, cpu.operators, cpu.graph.layer_plans[0].attention,
                                          ExecutionBackend::Cpu, position, cache, scratch, input, output, cpu.opt.optimization_flags);
    segment_check(static_cast<bool>(result), result ? "" : result.error().message);
}

static std::shared_ptr<Attention_vulkan> general_sdpa(const DecodeCase& spec, const CompiledModel& cpu, const CompiledModel& gpu)
{
    const auto& plan = gpu.graph.layer_plans[0].attention;
    const auto& input = gpu.operators.at(plan.fused_qkv_operator);
    const auto& output = gpu.operators.at_weight(plan.output_weight);
    AttentionConfig_vulkan config;
    config.hidden_size = segment_hidden;
    config.head_count = 4;
    config.kv_head_count = 1;
    config.head_dimension = config.rope_head_dimension = spec.head_dimension;
    config.sliding_window = spec.window;
    config.initial_context_length = 16;
    config.norm_epsilon = cpu.descriptor.norm_epsilon;
    config.activation_dtype = config.kv_cache_dtype = spec.cache_dtype;
    config.optimization_flags = segment_flags & ~OptimizationVulkanDecodeSdpa;
    config.flags = spec.sinks ? AttentionSink : 0;
    const auto& cpu_plan = cpu.graph.layer_plans[0].attention;
    const auto& norm = cpu.weights.at(cpu_plan.pre_attention_norm_weight);
    const auto* sinks = spec.sinks ? &cpu.weights.at(cpu_plan.sinks) : nullptr;
    return spec.weight_dtype == DType::BFloat16
               ? Attention_vulkan::create(norm, sinks, input.bfloat16, output.bfloat16, config)
               : Attention_vulkan::create(norm, sinks, input.linear, output.linear, config);
}

static void decode_storage(const DecodeCase& spec, const CompiledModel& gpu)
{
    const auto& layer = gpu.graph.layer_plans[0];
    const auto& plan = layer.attention;
    const auto& qkv = gpu.operators.at(plan.fused_qkv_operator);
    const auto& output = gpu.operators.at_weight(plan.output_weight);
    segment_check(spec.weight_dtype == DType::BFloat16 ? qkv.bfloat16 && output.bfloat16 && !qkv.linear && !output.linear
                                                       : qkv.linear && output.linear && !qkv.bfloat16 && !output.bfloat16,
                  "plain Attention did not preserve its selected BF16/FP32 representation");
}

static void decode_sequence(const DecodeCase& spec, const CompiledModel& cpu, const CompiledModel& gpu)
{
    const auto& plan = gpu.graph.layer_plans[0].attention;
    const auto& operation = gpu.operators.at(plan.vulkan_attention_operator).attention;
    const auto general = general_sdpa(spec, cpu, gpu);
    segment_check(operation && general, "plain BF16/FP32 Attention could not prepare both GPU SDPA capabilities");
    LayerCache cpu_cache, gpu_cache, general_cache;
    AttentionScratch scratch;
    uint64_t position = 0;
    for (uint32_t rows = 14; position <= spec.last_position; rows = 1)
    {
        const auto input = decode_input(rows, position);
        ActivationBuffer expected, actual, old_gpu;
        decode_cpu(cpu, position, cpu_cache, scratch, input, expected);
        const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
        segment_check(operation->forward(position, gpu_cache, input, actual), "blocked decode Attention failed at " + std::to_string(position));
        const auto after = get_vulkan_statistics(gpu.vulkan_runtime);
        segment_check(after.compute_submissions > before.compute_submissions
                          && after.attention_cpu_fallbacks == before.attention_cpu_fallbacks,
                      "Attention did not execute on GPU");
        segment_check(general->forward(position, general_cache, input, old_gpu), "general GPU SDPA reference failed");
        segment_check(actual.values().size() == expected.values().size() && old_gpu.values().size() == expected.values().size(),
                      "Attention comparison shape mismatch");
        const auto max_difference = [](std::span<const float> first, std::span<const float> second) {
            float difference = 0.0f;
            for (size_t index = 0; index < first.size(); ++index)
                difference = std::max(difference, std::abs(first[index] - second[index]));
            return difference;
        };
        const float new_cpu = max_difference(actual.values(), expected.values());
        const float old_cpu = max_difference(old_gpu.values(), expected.values());
        const float new_old = max_difference(actual.values(), old_gpu.values());
        if (position == 0 || new_cpu > 0.002f || old_cpu > 0.002f || new_old > 0.002f)
            std::cout << "Attention position " << position << ", rows " << rows
                      << ", blocked_dispatch " << (after.attention_decode_sdpa_fusions - before.attention_decode_sdpa_fusions)
                      << ", max_abs new/CPU " << new_cpu << ", general/CPU " << old_cpu
                      << ", new/general " << new_old << std::endl;
        const std::string location = " at position " + std::to_string(position) + ", rows " + std::to_string(rows);
        segment_compare(actual.values(), old_gpu.values(), "blocked decode / general GPU SDPA" + location);
        segment_compare(old_gpu.values(), expected.values(), "general GPU SDPA / independent CPU Attention" + location);
        segment_compare(actual.values(), expected.values(), "blocked decode / independent CPU Attention" + location);
        segment_check(gpu_cache.token_count == cpu_cache.token_count && gpu_cache.start_position == cpu_cache.start_position,
                      "blocked Attention changed sliding-window cache semantics");
        position += rows;
    }
    if (spec.window != 0)
        segment_check(get_vulkan_statistics(gpu.vulkan_runtime).kv_ring_wrapped_views > 0,
                      "sliding-window test did not exercise a wrapped GPU ring");

    const auto context = operation->vulkan_context();
    const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device());
        ncnn::VkMat wrong, output;
        wrong.create(segment_hidden - 4, 1, sizeof(float), context->blob_allocator());
        output.create(3, 1, sizeof(float), context->blob_allocator());
        const auto storage = output.data;
        AttentionWork_vulkan work;
        const auto count = gpu_cache.token_count;
        segment_check(!operation->record_forward(position, gpu_cache, wrong, output, command, work)
                          && output.data == storage && output.w == 3 && gpu_cache.token_count == count && !work.commit(),
                      "invalid record-only Attention published output or cache metadata");
    }
    const auto after = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(after.compute_submissions == before.compute_submissions && after.dispatches == before.dispatches,
                  "rejected record-only Attention published execution counters");
}

static void decode_staged(const CompiledModel& cpu, const CompiledModel& gpu)
{
    const auto operation = gpu.operators.at(gpu.graph.layer_plans[0].attention.vulkan_attention_operator).attention;
    std::array<LayerCache, 2> expected_caches, actual_caches;
    std::array<AttentionScratch, 2> scratches;
    // forward_batch requires established rings with room for the next row.
    // Ring growth is exercised separately by decode_sequence.
    const std::array<uint64_t, 2> positions = {15, 31};
    std::array<ActivationBuffer, 2> input, expected, actual;
    for (size_t index = 0; index < positions.size(); ++index)
    {
        const auto prefix = decode_input(static_cast<uint32_t>(positions[index]), 0);
        ActivationBuffer ignored;
        decode_cpu(cpu, 0, expected_caches[index], scratches[index], prefix, ignored);
        segment_check(operation->forward(0, actual_caches[index], prefix, ignored), "staged Attention prefix failed");
        segment_check(actual_caches[index].token_count + 1 <= actual_caches[index].capacity_tokens,
                      "staged Attention fixture did not reserve room for its decode row");
        input[index] = decode_input(1, positions[index]);
        decode_cpu(cpu, positions[index], expected_caches[index], scratches[index], input[index], expected[index]);
    }
    const std::array<AttentionBatchEntry_vulkan, 2> entries = {{{positions[0], &actual_caches[0], &input[0], &actual[0]},
                                                                {positions[1], &actual_caches[1], &input[1], &actual[1]}}};
    const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(operation->forward_batch(entries) == AttentionBatchResult_vulkan::Executed,
                  "plain BF16/FP32 staged Attention did not execute on GPU");
    const auto after = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(after.compute_submissions == before.compute_submissions + 1
                      && after.attention_cpu_fallbacks == before.attention_cpu_fallbacks,
                  "staged Attention did not record both rows in one GPU submission");
    for (size_t index = 0; index < entries.size(); ++index)
        segment_compare(actual[index].values(), expected[index].values(), "staged blocked decode / independent CPU Attention");
}
#endif

static int run(uint32_t device, bool kernel_only)
{
#if NCNN_MOE_WITH_VULKAN
    if (get_gpu_count() == 0) return 77;
    const auto context = VulkanContext::acquire(device, create_vulkan_runtime(), segment_flags);
    if (!context) return 77;
    try
    {
        decode_kernel_cases(context, false);
        const uint32_t subgroup = context->device()->info.subgroup_size();
        const uint32_t required_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        if ((context->device()->info.support_subgroup_ops() & required_ops) == required_ops
            && subgroup >= 16 && subgroup <= 128 && 128 % subgroup == 0)
            decode_kernel_cases(context, true);
        else
            std::cout << "SKIP: subgroup decode requires shuffle and complete groups of 16 to 128 lanes\n";
        if (kernel_only)
        {
            std::cout << "Kernel-only validation completed; full Attention and prefill were not exercised\n";
            return 0;
        }
        for (const auto& spec : {DecodeCase{64, DType::BFloat16, DType::BFloat16, true, 0, 129},
                                 DecodeCase{128, DType::BFloat16, DType::Float32, false, 0, 33},
                                 DecodeCase{64, DType::Float32, DType::Float32, true, 0, 129},
                                 DecodeCase{10, DType::BFloat16, DType::BFloat16, true, 17, 65}})
        {
            if (spec.weight_dtype == DType::BFloat16 && context->device()->info.subgroup_size() != 32)
            {
                std::cout << "SKIP: native BF16 projection requires subgroup32; device " << device
                          << ", head_dimension " << spec.head_dimension << '\n';
                continue;
            }
            std::cout << "Attention device " << device << ", head_dimension " << spec.head_dimension
                      << ", weights " << (spec.weight_dtype == DType::BFloat16 ? "BF16" : "FP32")
                      << ", window " << spec.window << std::endl;
            const auto cpu = decode_model(spec, false, device);
            const auto gpu = decode_model(spec, true, device);
            decode_storage(spec, gpu);
            decode_sequence(spec, cpu, gpu);
            decode_staged(cpu, gpu);
        }
        std::cout << "plain BF16/FP32 Attention storage, blocked SDPA tails/window/ring, staged rows, and rejection checks passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
#else
    (void)device;
    (void)kernel_only;
    return 77;
#endif
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
#if NCNN_MOE_WITH_VULKAN
    if (ncnn::moe::get_gpu_count() == 0) return 77;
    try
    {
        uint32_t device = ncnn::moe::get_default_gpu_index();
        bool kernel_only = false;
        for (int index = 1; index < argc; ++index)
        {
            if (std::string(argv[index]) == "--kernel-only")
            {
                kernel_only = true;
                continue;
            }
            if (std::string(argv[index]) != "--device" || ++index == argc)
                throw std::invalid_argument("usage: attention_bf16_decode_vulkan [--device N] [--kernel-only]");
            size_t consumed = 0;
            const unsigned long requested = std::stoul(argv[index], &consumed);
            if (consumed != std::string(argv[index]).size() || requested >= ncnn::moe::get_gpu_count())
                throw std::invalid_argument("Vulkan device index is out of range");
            device = static_cast<uint32_t>(requested);
        }
        return ncnn::moe::run(device, kernel_only);
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
