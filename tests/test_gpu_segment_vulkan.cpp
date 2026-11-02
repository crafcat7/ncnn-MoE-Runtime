#include "gpu_segment_fixture.h"

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
struct SegmentHeadCase
{
    const char* name;
    std::shared_ptr<GpuSegment_vulkan> operation;
};

static std::array<SegmentHeadCase, 3> segment_head_cases(const CompiledModel& cpu, const CompiledModel& gpu)
{
    const auto& fused = gpu.operators.at_weight(gpu.lm_head_weight).bfloat16;
    segment_check(fused && fused->has_rms_norm_chain(), "resident head fixture did not prepare its configured BF16 RMSNorm chain");
    const auto& weight = cpu.weights.at(cpu.lm_head_weight);
    const auto plain = Bfloat16Linear_vulkan::create(weight, nullptr, gpu.opt.vulkan_device_index,
                                                     gpu.vulkan_runtime, gpu.opt.optimization_flags);
    segment_check(plain && !plain->has_rms_norm_chain(), "unprepared BF16 head unexpectedly owns a fused Norm");
    CompiledModel definition = gpu;
    auto& operation = definition.operators.at_weight_mutable(definition.lm_head_weight);
    operation = {};
    operation.bfloat16 = plain;
    const auto unfused = GpuSegment_vulkan::create_head(definition);
    segment_check(static_cast<bool>(unfused), "cannot prepare an unfused BF16 resident head");

    TensorData float32_weight;
    float32_weight.dtype = DType::Float32;
    float32_weight.shape = weight.shape;
    for (const uint16_t value : weight.bfloat16_values())
        float32_weight.float32_data.push_back(bfloat16_to_float(value));
    operation = {};
    operation.linear = Linear::create(float32_weight, nullptr, gpu.opt.vulkan_device_index,
                                      gpu.vulkan_runtime, gpu.opt.optimization_flags);
    segment_check(static_cast<bool>(operation.linear), "cannot prepare the exact FP32 resident head projection");
    const auto non_bfloat16 = GpuSegment_vulkan::create_head(definition);
    segment_check(static_cast<bool>(non_bfloat16), "cannot prepare a non-BF16 resident head");

    const auto context = gpu.gpu_head->context();
    const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device());
        ncnn::VkMat input, wrong_width, output;
        input.create(segment_hidden, 2, sizeof(float), context->blob_allocator());
        wrong_width.create(segment_hidden - 4, 2, sizeof(float), context->blob_allocator());
        output.create(3, 1, sizeof(float), context->blob_allocator());
        segment_check(!input.empty() && !wrong_width.empty() && !output.empty(), "cannot allocate head rejection fixtures");
        const auto output_storage = output.data;
        segment_check(!plain->record_rms_norm_chain(input, output, command)
                          && !fused->record_rms_norm_chain(wrong_width, output, command)
                          && output.data == output_storage && output.w == 3 && output.h == 1,
                      "unprepared or invalid fused head recording modified the caller's output");
        const auto input_storage = input.data;
        segment_check(!fused->record_rms_norm_chain(input, input, command) && input.data == input_storage,
                      "fused head recording accepted input/output object aliasing");
    }
    const auto after = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(after.compute_submissions == before.compute_submissions && after.dispatches == before.dispatches
                      && after.rms_norm_linear_fusions == before.rms_norm_linear_fusions
                      && after.batch_uploads == before.batch_uploads && after.batch_downloads == before.batch_downloads,
                  "rejected record-only head work submitted a command or published success counters");
    return {{{"BF16 fused Norm/projection", gpu.gpu_head},
             {"BF16 without a prepared Norm chain", unfused},
             {"FP32 projection", non_bfloat16}}};
}

static void segment_counters(const VulkanStatistics& before, const VulkanStatistics& after, size_t rows)
{
    segment_check(after.resident_graph_activation_upload_bytes - before.resident_graph_activation_upload_bytes
                      == rows * segment_hidden * sizeof(float),
                  "resident graph must upload hidden activations only at its entrance; observed bytes "
                      + std::to_string(after.resident_graph_activation_upload_bytes - before.resident_graph_activation_upload_bytes));
    segment_check(after.resident_graph_activation_download_bytes == before.resident_graph_activation_download_bytes,
                  "resident graph materialized an intermediate hidden activation");
    segment_check(after.resident_graph_route_download_bytes - before.resident_graph_route_download_bytes
                      == rows * segment_layers * Router_vulkan::selected_columns(segment_top_k) * sizeof(float),
                  "resident graph route transfer is not the compact per-layer selection; observed bytes "
                      + std::to_string(after.resident_graph_route_download_bytes - before.resident_graph_route_download_bytes));
    segment_check(after.attention_cpu_fallbacks == before.attention_cpu_fallbacks,
                  "resident graph retried Attention on CPU");
}

static void segment_engine(const CompiledModel& cpu, const CompiledModel& gpu)
{
    for (const auto output : {LogitsOutput::All, LogitsOutput::Last})
    {
        SessionState expected_state;
        SessionState actual_state;
        SessionStatistics expected_statistics;
        SessionStatistics actual_statistics;
        uint64_t position = 0;
        for (const std::vector<int32_t> ids : {std::vector<int32_t>{1, 4, 2}, std::vector<int32_t>{7}})
        {
            const auto reference = forward_model(cpu, ids, expected_statistics, expected_state, position, output);
            segment_check(static_cast<bool>(reference), reference ? "" : reference.error().message);
            const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
            const auto executed = forward_model(gpu, ids, actual_statistics, actual_state, position, output);
            segment_check(static_cast<bool>(executed), executed ? "" : executed.error().message);
            segment_compare(actual_state.logits.values(), expected_state.logits.values(), "cross-layer resident prefill/decode");
            segment_counters(before, get_vulkan_statistics(gpu.vulkan_runtime), ids.size());
            segment_gpu_routes(actual_statistics);
            position += ids.size();
            segment_check(actual_state.layers[0].token_count == position && actual_state.layers[1].gated_delta_token_count == position,
                          "resident Standard/GDN cache counts did not advance exactly once");
        }
        segment_check(!actual_state.gpu_segment.active(), "final head retained an unfinished resident segment");
    }
}

static void segment_staged(const CompiledModel& cpu, const CompiledModel& gpu)
{
    std::array<SessionState, 2> expected_states;
    std::array<SessionState, 2> actual_states;
    std::array<SessionStatistics, 2> expected_statistics;
    std::array<SessionStatistics, 2> actual_statistics;
    const std::array<std::vector<int32_t>, 2> prefixes = {std::vector<int32_t>{2}, std::vector<int32_t>{5, 1, 4}};
    for (size_t index = 0; index < prefixes.size(); ++index)
    {
        const auto reference = forward_model(cpu, prefixes[index], expected_statistics[index], expected_states[index], 0, LogitsOutput::Last);
        const auto executed = forward_model(gpu, prefixes[index], actual_statistics[index], actual_states[index], 0, LogitsOutput::Last);
        segment_check(static_cast<bool>(reference), reference ? "" : reference.error().message);
        segment_check(static_cast<bool>(executed), executed ? "" : executed.error().message);
        segment_compare(actual_states[index].logits.values(), expected_states[index].logits.values(), "staged prefix");
    }
    BatchWorkspace workspace;
    for (uint64_t step = 0; step < 2; ++step)
    {
        std::array<std::vector<float>, 2> logits;
        std::array<DecodeBatchEntry, 2> entries;
        for (size_t index = 0; index < entries.size(); ++index)
        {
            const int32_t token = static_cast<int32_t>(9 - index * 3 - step);
            const uint64_t position = prefixes[index].size() + step;
            const std::array<int32_t, 1> ids = {token};
            const auto reference = forward_model(cpu, ids, expected_statistics[index], expected_states[index], position, LogitsOutput::All);
            segment_check(static_cast<bool>(reference), reference ? "" : reference.error().message);
            entries[index] = {token, &actual_statistics[index], &actual_states[index], position, &logits[index]};
        }
        const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
        const auto executed = forward_decode_batch(gpu, entries, workspace);
        segment_check(static_cast<bool>(executed), executed ? "" : executed.error().message);
        segment_counters(before, get_vulkan_statistics(gpu.vulkan_runtime), entries.size());
        for (size_t index = 0; index < entries.size(); ++index)
        {
            segment_compare(logits[index], expected_states[index].logits.values(), "staged resident independent cache rows");
            segment_gpu_routes(actual_statistics[index]);
            const uint64_t expected_tokens = prefixes[index].size() + step + 1;
            segment_check(actual_states[index].layers[0].token_count == expected_tokens
                              && actual_states[index].layers[1].gated_delta_token_count == expected_tokens,
                          "staged resident rows advanced another session's state");
        }
    }
    segment_check(!workspace.gpu_segment.active(), "staged final head retained an unfinished resident segment");
}

static void segment_unknown_state(const CompiledModel& gpu)
{
    // Exercise the terminal state produced by a failed stateful GPU submit;
    // this does not inject or pretend to roll back an actual device failure.
    for (size_t layer = 0; layer < segment_layers; ++layer)
    {
        SessionState state;
        SessionStatistics statistics;
        state.layers.resize(segment_layers);
        state.layers[layer].vulkan_attention_state_unknown = true;
        state.layers[layer].token_count = 7;
        state.layers[layer].gated_delta_token_count = 7;
        state.logits.reset(1, 1, false);
        state.logits.row(0)[0] = -91.0f;
        const std::array<int32_t, 1> ids = {3};
        const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
        const auto executed = forward_model(gpu, ids, statistics, state, 7, LogitsOutput::Last);
        segment_check(!executed && executed.error().code == ErrorCode::InternalError
                          && state.layers[layer].vulkan_attention_state_unknown
                          && state.layers[layer].token_count == 7 && state.layers[layer].gated_delta_token_count == 7
                          && state.logits.rows() == 1 && state.logits.columns() == 1 && state.logits.row(0)[0] == -91.0f
                          && statistics.expert_cpu_routes == 0 && statistics.expert_gpu_routes == 0
                          && statistics.expert_cpu_fallbacks == 0 && !state.gpu_segment.active(),
                      "unknown Standard/GDN device state was replayed or published a new result");
        const auto after = get_vulkan_statistics(gpu.vulkan_runtime);
        segment_check(after.compute_submissions == before.compute_submissions
                          && after.dispatches == before.dispatches
                          && after.rms_norm_linear_fusions == before.rms_norm_linear_fusions
                          && after.resident_graph_activation_upload_bytes == before.resident_graph_activation_upload_bytes
                          && after.resident_graph_activation_download_bytes == before.resident_graph_activation_download_bytes,
                      "unknown state rejection recorded or submitted replacement work");
    }
}

static void segment_boundary_and_rejection(const CompiledModel& cpu, const CompiledModel& gpu)
{
    const auto& layer = gpu.graph.layer_plans[0];
    const auto& cpu_layer = cpu.graph.layer_plans[0];
    const auto& segment = gpu.gpu_segments[0];
    const std::array<int32_t, 2> ids = {2, 6};
    ActivationBuffer input;
    forward_embedding(cpu.weights.at(cpu.token_embedding), ids, input);
    LayerCache cache;
    GpuSegmentWorkspace_vulkan workspace;
    ExpertDispatchOptions options;
    options.expert_count = segment_experts;
    options.top_k = segment_top_k;
    ExpertDispatchPlan routes;
    routes.assignment_count = 97;
    routes.batches = {{17, {{3, 1, 0.25f}}}};
    const std::array<uint64_t, 2> positions = {0, 1};
    const std::array<LayerCache*, 2> caches = {&cache, &cache};
    std::string error;
    const auto before_rejection = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(!segment->forward(input, std::span(positions).first(1), caches, options, routes, workspace, error)
                      && !error.empty() && !workspace.active() && !workspace.device_input()
                      && cache.token_count == 0 && !cache.vulkan_attention_state_unknown
                      && routes.assignment_count == 97 && routes.batches.size() == 1 && routes.batches[0].expert_id == 17,
                  "an invalid unsubmitted segment changed cache metadata or published routes/device output");
    const auto after_rejection = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(after_rejection.compute_submissions == before_rejection.compute_submissions
                      && after_rejection.resident_graph_submissions == before_rejection.resident_graph_submissions,
                  "invalid segment input submitted GPU work");

    LayerCache expected_cache;
    AttentionScratch scratch;
    ActivationBuffer expected;
    const auto reference = forward_attention(cpu.weights, cpu.operators, cpu_layer.attention, ExecutionBackend::Cpu,
                                             0, expected_cache, scratch, input, expected, cpu.opt.optimization_flags);
    segment_check(static_cast<bool>(reference), reference ? "" : reference.error().message);
    const auto executed = segment->forward(input, positions, caches, options, routes, workspace, error);
    segment_check(executed, "segment forward failed after rejected input: " + error);
    segment_check(workspace.active() && workspace.device_input() && cache.token_count == ids.size(),
                  "successful segment did not publish completed normalized input and cache metadata");
    const auto normalized = forward_rms_norm(expected, cpu.weights.at(cpu_layer.moe.pre_ffn_norm_weight),
                                             cpu.descriptor.norm_epsilon, cpu.descriptor.norm_weight_offset);
    const auto router_logits = forward_linear(cpu.weights.at(cpu_layer.moe.router_weight), normalized, cpu.opt.optimization_flags);
    ExpertDispatchPlan expected_routes;
    const auto routed = forward_router(router_logits.values(), static_cast<uint32_t>(ids.size()), options, expected_routes);
    segment_check(static_cast<bool>(routed), routed ? "" : routed.error().message);
    segment_check(routes.assignment_count == expected_routes.assignment_count && routes.batches.size() == expected_routes.batches.size(),
                  "compact device routes changed assignments");
    for (size_t batch = 0; batch < routes.batches.size(); ++batch)
    {
        const auto& actual = routes.batches[batch];
        const auto& reference_batch = expected_routes.batches[batch];
        segment_check(actual.expert_id == reference_batch.expert_id && actual.routes.size() == reference_batch.routes.size(),
                      "compact device routes changed canonical Expert order");
        for (size_t row = 0; row < actual.routes.size(); ++row)
            segment_check(actual.routes[row].token_index == reference_batch.routes[row].token_index
                              && actual.routes[row].rank == reference_batch.routes[row].rank
                              && std::abs(actual.routes[row].weight - reference_batch.routes[row].weight) < 2e-4f,
                          "compact device route differs from CPU selection");
    }
    ActivationBuffer hidden(1, 1);
    hidden.row(0)[0] = -83.0f;
    const auto before_unready = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(!GpuSegment_vulkan::materialize_hidden(workspace, hidden, error)
                      && workspace.active() && hidden.rows() == 1 && hidden.columns() == 1 && hidden.row(0)[0] == -83.0f,
                  "an unrecordable missing aggregate changed the CPU output or discarded resident state");
    segment_check(get_vulkan_statistics(gpu.vulkan_runtime).compute_submissions == before_unready.compute_submissions,
                  "an incomplete FFN aggregate submitted GPU work");
    ActivationBuffer logits(1, 1);
    logits.row(0)[0] = -79.0f;
    segment_check(!gpu.gpu_head->finish(true, true, hidden, logits, workspace, error)
                      && workspace.active() && hidden.row(0)[0] == -83.0f
                      && logits.rows() == 1 && logits.columns() == 1 && logits.row(0)[0] == -79.0f,
                  "a head with no completed aggregate published logits or consumed its resident workspace");
    const auto after_unready = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(after_unready.compute_submissions == before_unready.compute_submissions
                      && after_unready.resident_graph_submissions == before_unready.resident_graph_submissions
                      && after_unready.dispatches == before_unready.dispatches
                      && after_unready.rms_norm_linear_fusions == before_unready.rms_norm_linear_fusions
                      && after_unready.batch_downloads == before_unready.batch_downloads,
                  "a rejected resident head published submission, fusion, or download success counters");
    ActivationBuffer aggregate(ids.size(), segment_hidden);
    for (size_t row = 0; row < aggregate.rows(); ++row)
        for (uint32_t column = 0; column < segment_hidden; ++column)
            aggregate.row(row)[column] = static_cast<float>(static_cast<int>((row * 7 + column * 3) % 17) - 8) * 0.0078125f;
    const auto device_aggregate = segment_upload(aggregate, gpu);
    const auto before_defer = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(GpuSegment_vulkan::defer_combine(device_aggregate, workspace)
                      && !GpuSegment_vulkan::defer_combine(device_aggregate, workspace),
                  "resident aggregate must be accepted exactly once");
    segment_check(get_vulkan_statistics(gpu.vulkan_runtime).compute_submissions == before_defer.compute_submissions,
                  "deferring an aggregate submitted GPU work");
    ActivationBuffer shared;
    ExpertWorkspace shared_workspace;
    forward_shared_expert(cpu, cpu_layer.moe, normalized, shared, shared_workspace);
    add_batch_inplace(aggregate, shared);
    add_batch_inplace(expected, aggregate);
    const auto before_boundary = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(GpuSegment_vulkan::materialize_hidden(workspace, hidden, error), "explicit CPU boundary failed: " + error);
    segment_compare(hidden.values(), expected.values(), "explicit CPU boundary");
    const auto after_boundary = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(!workspace.active()
                      && after_boundary.compute_submissions == before_boundary.compute_submissions + 1
                      && after_boundary.resident_graph_activation_upload_bytes == before_boundary.resident_graph_activation_upload_bytes
                      && after_boundary.resident_graph_activation_download_bytes - before_boundary.resident_graph_activation_download_bytes
                             == ids.size() * segment_hidden * sizeof(float),
                  "explicit CPU boundary must perform exactly one activation download and finish the segment");
    workspace.reset();
    segment_check(!workspace.active() && !workspace.device_input(), "reset retained a published normalized input");
    segment_check(GpuSegment_vulkan::supports(gpu, layer), "rejected input mutated prepared segment capabilities");
}

static void segment_standalone_head_capacity(const CompiledModel& cpu, const CompiledModel& gpu)
{
    const auto planned = plan_host_output_slot_size(2, gpu.descriptor.vocabulary_size);
    segment_check(static_cast<bool>(planned), "standalone head output planning failed");
    GpuSegmentWorkspace_vulkan workspace;
    workspace.reserve_host_output(1);
    ActivationBuffer logits(1, 1);
    logits.row(0)[0] = -101.0f;
    for (size_t step = 0; step < 2; ++step)
    {
        const std::vector<int32_t> ids = step == 0 ? std::vector<int32_t>{2, 6}
                                                   : std::vector<int32_t>{static_cast<int32_t>(step + 3)};
        ActivationBuffer input;
        forward_embedding(cpu.weights.at(cpu.token_embedding), ids, input);
        LayerCache cache, expected_cache;
        std::vector<uint64_t> positions(ids.size());
        std::vector<LayerCache*> caches(ids.size(), &cache);
        for (size_t row = 0; row < positions.size(); ++row) positions[row] = row;
        ExpertDispatchOptions options;
        options.expert_count = segment_experts;
        options.top_k = segment_top_k;
        ExpertDispatchPlan routes;
        std::string error;
        segment_check(gpu.gpu_segments[0]->forward(input, positions, caches, options, routes, workspace, error),
                      "standalone head input segment failed: " + error);
        ActivationBuffer aggregate(ids.size(), segment_hidden);
        const auto device_aggregate = segment_upload(aggregate, gpu);
        segment_check(GpuSegment_vulkan::defer_combine(device_aggregate, workspace),
                      "standalone head could not retain its completed aggregate");

        AttentionScratch scratch;
        ActivationBuffer expected;
        const auto& layer = cpu.graph.layer_plans[0];
        const auto attention = forward_attention(cpu.weights, cpu.operators, layer.attention, ExecutionBackend::Cpu,
                                                 0, expected_cache, scratch, input, expected, cpu.opt.optimization_flags);
        segment_check(static_cast<bool>(attention), attention ? "" : attention.error().message);
        const auto normalized = forward_rms_norm(expected, cpu.weights.at(layer.moe.pre_ffn_norm_weight),
                                                 cpu.descriptor.norm_epsilon, cpu.descriptor.norm_weight_offset);
        ActivationBuffer shared;
        ExpertWorkspace shared_workspace;
        forward_shared_expert(cpu, layer.moe, normalized, shared, shared_workspace);
        add_batch_inplace(expected, shared);
        const auto final = forward_rms_norm(expected, cpu.weights.at(cpu.final_norm_weight),
                                            cpu.descriptor.norm_epsilon, cpu.descriptor.norm_weight_offset);
        const auto reference = forward_linear(cpu.weights.at(cpu.lm_head_weight), final, cpu.opt.optimization_flags);
        ActivationBuffer hidden;
        if (step == 0)
        {
            const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
            segment_check(!gpu.gpu_head->finish(true, false, hidden, logits, workspace, error)
                              && workspace.active() && logits.rows() == 1 && logits.columns() == 1 && logits.row(0)[0] == -101.0f
                              && get_vulkan_statistics(gpu.vulkan_runtime).compute_submissions == before.compute_submissions,
                          "standalone head ignored its explicit output bound or published a failed recording");
            workspace.reserve_host_output(planned.value());
        }
        segment_check(gpu.gpu_head->finish(true, false, hidden, logits, workspace, error),
                      "standalone head failed after planned output admission: " + error);
        segment_compare(logits.values(), reference.values(), "standalone head complete mapped logits");
        segment_check(workspace.allocated_bytes() <= 2 * planned.value(),
                      "standalone head exceeded its planned output capacity");
        workspace.reset();
    }
}
#endif

static int run()
{
#if NCNN_MOE_WITH_VULKAN
    if (get_gpu_count() == 0) return 77;
    try
    {
        const auto runtime = create_vulkan_runtime();
        const auto context = VulkanContext::acquire(get_default_gpu_index(), runtime, segment_flags);
        segment_check(static_cast<bool>(context), "cannot create generic segment Vulkan context");
        if (context->device()->info.subgroup_size() != 32)
        {
            std::cout << "SKIP: native BF16 GDN/Experts require subgroup size 32\n";
            return 77;
        }
        for (const bool native_delta : {false, true})
        {
            std::cout << "generic resident GDN projection: " << (native_delta ? "native FP32 with separate RMSNorm" : "BF16 fused RMSNorm") << '\n';
            const auto cpu = segment_compile(false, native_delta);
            auto gpu = segment_compile(true, native_delta);
            const auto heads = segment_head_cases(cpu, gpu);
            {
                const CompiledModel enabled = gpu;
                CompiledModel disabled = gpu;
                disabled.opt.optimization_flags &= ~OptimizationVulkanCommandGraph;
                for (size_t layer = 0; layer < segment_layers; ++layer)
                    segment_check(GpuSegment_vulkan::supports(enabled, enabled.graph.layer_plans[layer])
                                      && !GpuSegment_vulkan::supports(disabled, disabled.graph.layer_plans[layer]),
                                  "explicitly disabled CommandGraph still admits a resident layer segment");
                segment_check(GpuSegment_vulkan::supports_head(enabled) && !GpuSegment_vulkan::supports_head(disabled),
                              "explicitly disabled CommandGraph still admits a resident head");
            }
            // Fixture-only products are prepared before sealing the shared arena.
            segment_boundary_and_rejection(cpu, gpu);
            segment_standalone_head_capacity(cpu, gpu);
            const auto reserved = prepare_model_graph_resources(gpu, 4);
            segment_check(static_cast<bool>(reserved), reserved ? "" : reserved.error().message);
            segment_check(gpu.graph_resources && gpu.graph_resources->plan.sequence_count == 2,
                          "resident graph did not reserve its two-session lifetime plan");
            for (const auto& head : heads)
            {
                std::cout << "generic resident head: " << head.name << '\n';
                // Head formats and GDN projection formats need not form a Cartesian product.
                if (native_delta && head.operation != heads.front().operation) continue;
                gpu.gpu_head = head.operation;
                segment_engine(cpu, gpu);
            }
            gpu.gpu_head = heads.front().operation;
            segment_staged(cpu, gpu);
            segment_unknown_state(gpu);
        }
        std::cout << "generic resident Standard/GDN graph, staged rows, counters, CPU boundary, and publication checks passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
#else
    return 77;
#endif
}

} // namespace moe
} // namespace ncnn

int main()
{
    return ncnn::moe::run();
}
