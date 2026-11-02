#include "gpu_segment_fixture.h"

#include <algorithm>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static ncnn::Option continuation_option(const VulkanContext& context)
{
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_packing_layout = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.blob_vkallocator = context.blob_allocator();
    option.workspace_vkallocator = context.blob_allocator();
    option.staging_vkallocator = context.staging_allocator();
    return option;
}

static ExpertDispatchOptions continuation_options()
{
    ExpertDispatchOptions options;
    options.expert_count = segment_experts;
    options.top_k = segment_top_k;
    return options;
}

static ncnn::VkMat continuation_upload(const CompiledModel& gpu, const ActivationBuffer& source)
{
    const auto context = gpu.gpu_segments.front()->context();
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const auto option = continuation_option(*context);
    ncnn::VkCompute command(context->device());
    ncnn::VkMat staging, output;
    segment_check(fill_staging_upload(source, staging, option.staging_vkallocator)
                      && record_mapped_upload(staging, output, command, option)
                      && submit_compute_and_wait(command, context->device()) == 0,
                  "cannot upload a continuation operand fixture");
    ++context->runtime_state().compute_submissions;
    ++context->runtime_state().batch_uploads;
    return output;
}

static void continuation_compare_routes(const ExpertDispatchPlan& actual, const ExpertDispatchPlan& expected)
{
    segment_check(actual.assignment_count == expected.assignment_count && actual.batches.size() == expected.batches.size(),
                  "continued routes changed the assignment shape");
    for (size_t index = 0; index < actual.batches.size(); ++index)
    {
        const auto& first = actual.batches[index];
        const auto& second = expected.batches[index];
        segment_check(first.expert_id == second.expert_id && first.routes.size() == second.routes.size(),
                      "continued routes changed canonical Expert order");
        for (size_t route = 0; route < first.routes.size(); ++route)
            segment_check(first.routes[route].token_index == second.routes[route].token_index
                              && first.routes[route].rank == second.routes[route].rank
                              && std::abs(first.routes[route].weight - second.routes[route].weight) < 2e-4f,
                          "continued compact routes differ from separately submitted GPU routes");
    }
}

struct ContinuationOutputs
{
    std::vector<ncnn::VkMat> values;
    std::vector<std::vector<ExpertRoute>> routes;
};

static ContinuationOutputs continuation_outputs(const CompiledModel& gpu, size_t rows, size_t experts)
{
    ContinuationOutputs outputs;
    outputs.routes.resize(experts);
    for (size_t expert = 0; expert < experts; ++expert)
    {
        ActivationBuffer value(rows, segment_hidden);
        for (size_t row = 0; row < rows; ++row)
        {
            outputs.routes[expert].push_back({static_cast<uint32_t>(row), static_cast<uint32_t>(expert),
                                              0.5f / static_cast<float>(expert + 1)});
            for (uint32_t column = 0; column < segment_hidden; ++column)
                value.row(row)[column] = static_cast<float>(static_cast<int>((expert * 11 + row * 3 + column) % 23) - 11) * 0.00390625f;
        }
        outputs.values.push_back(continuation_upload(gpu, value));
    }
    return outputs;
}

static std::shared_ptr<ResidentFfnRegion_vulkan> continuation_wave(const CompiledModel& gpu,
                                                                   GpuSegmentWorkspace_vulkan& workspace,
                                                                   const ContinuationOutputs& outputs, bool two_waves,
                                                                   const std::shared_ptr<GpuSegmentContinuation_vulkan>& pending)
{
    const auto context = gpu.gpu_segments.front()->context();
    auto region = ResidentFfnRegion_vulkan::create(workspace.device_input(), outputs.values.size(), segment_hidden);
    const auto combiner = MoeCombine_vulkan::create(context);
    segment_check(region && combiner && region->bind_context(context, combiner), "cannot bind continuation Expert region");
    if (pending) segment_check(region->set_continuation(pending), "cannot attach a prepared continuation");
    const size_t wave_count = two_waves ? 2 : 1;
    segment_check(!two_waves || outputs.values.size() >= 2, "two-wave fixture requires multiple Experts");
    for (size_t wave = 0; wave < wave_count; ++wave)
    {
        const size_t first = two_waves && wave == 0 ? outputs.values.size() / 2 : 0;
        const size_t last = two_waves && wave == 1 ? outputs.values.size() / 2 : outputs.values.size();
        std::vector<ResidentFfnRegion_vulkan::Output> selected;
        for (size_t index = first; index < last; ++index)
            selected.push_back({index, outputs.values[index], outputs.routes[index]});
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        const auto option = continuation_option(*context);
        ncnn::VkCompute command(context->device());
        ncnn::VkMat input;
        bool uploaded = true;
        bool downloaded = true;
        segment_check(region->record_input(input, command, option, uploaded) && !uploaded
                          && region->record_outputs(selected, command, option, downloaded) && !downloaded,
                      "a continued device region added an activation transfer");
        region->mark_submit_attempted();
        segment_check(submit_compute_and_wait(command, context->device()) == 0 && region->complete_wave(),
                      "the Expert-owned continuation command failed");
        ++context->runtime_state().compute_submissions;
        context->runtime_state().dispatches += region->completed_combine_dispatches();
    }
    return region;
}

struct ContinuationResult
{
    ActivationBuffer logits;
    std::array<ExpertDispatchPlan, segment_layers> routes;
    VulkanStatistics before;
    VulkanStatistics after;
};

static ContinuationResult continuation_direct(const CompiledModel& gpu, const ActivationBuffer& input,
                                              const ContinuationOutputs& outputs, bool staged, bool continued,
                                              bool two_waves, LogitsOutput output)
{
    std::array<std::vector<LayerCache>, segment_layers> caches;
    for (auto& layer : caches) layer.resize(staged ? input.rows() : 1);
    std::vector<uint64_t> positions(input.rows());
    std::vector<LayerCache*> pointers(input.rows());
    const auto prepare_rows = [&](size_t layer) {
        for (size_t row = 0; row < input.rows(); ++row)
        {
            positions[row] = staged ? 0 : row;
            pointers[row] = &caches[layer][staged ? row : 0];
        }
    };
    ContinuationResult result;
    result.logits.reset(1, 1, false);
    result.logits.row(0)[0] = -71.0f;
    GpuSegmentWorkspace_vulkan workspace;
    const auto options = continuation_options();
    std::string error;
    prepare_rows(0);
    result.before = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(gpu.gpu_segments[0]->forward(input, positions, pointers, options, result.routes[0], workspace, error),
                  "first standalone segment failed: " + error);
    for (size_t layer = 0; layer < segment_layers; ++layer)
    {
        std::shared_ptr<GpuSegmentContinuation_vulkan> pending;
        if (continued)
        {
            if (layer + 1 < segment_layers)
            {
                prepare_rows(layer + 1);
                pending = gpu.gpu_segments[layer + 1]->prepare_continuation(positions, pointers, options, workspace, error);
            }
            else
                pending = gpu.gpu_head->prepare_head_continuation(output != LogitsOutput::None, output == LogitsOutput::Last, workspace, error);
            segment_check(static_cast<bool>(pending), "cannot prepare direct continuation: " + error);
            ExpertDispatchPlan sentinel;
            sentinel.assignment_count = 91;
            const auto original_input = workspace.device_input();
            const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
            segment_check(!pending->publish(sentinel, workspace, error) && sentinel.assignment_count == 91
                              && workspace.device_input() == original_input,
                          "an unrecorded continuation published routes or consumed the workspace");
            segment_check(get_vulkan_statistics(gpu.vulkan_runtime).compute_submissions == before.compute_submissions,
                          "unsubmitted continuation publication submitted replacement GPU work");
        }
        auto region = continuation_wave(gpu, workspace, outputs, two_waves, pending);
        std::shared_ptr<const DeviceTensor_vulkan> aggregate;
        segment_check(region->take_device_output(aggregate), "the completed Expert wave did not publish its device aggregate");
        if (continued)
        {
            if (layer + 1 < segment_layers)
            {
                segment_check(pending->publish(result.routes[layer + 1], workspace, error), "continued layer publication failed: " + error);
                const auto published_input = workspace.device_input();
                ExpertDispatchPlan sentinel;
                sentinel.assignment_count = 93;
                segment_check(!pending->publish(sentinel, workspace, error) && sentinel.assignment_count == 93
                                  && workspace.device_input() == published_input,
                              "a published layer continuation was committed twice");
            }
            else
            {
                segment_check(pending->publish_head(result.logits, workspace, error), "continued head publication failed: " + error);
                ActivationBuffer sentinel(1, 1);
                sentinel.row(0)[0] = -73.0f;
                segment_check(!pending->publish_head(sentinel, workspace, error) && sentinel.row(0)[0] == -73.0f,
                              "a published head continuation was committed twice");
            }
            pending->abort(); // Published transactions must remain valid.
            region.reset();
        }
        else
        {
            segment_check(GpuSegment_vulkan::defer_combine(aggregate, workspace), "cannot defer standalone Expert aggregate");
            if (layer + 1 < segment_layers)
            {
                prepare_rows(layer + 1);
                segment_check(gpu.gpu_segments[layer + 1]->forward(input, positions, pointers, options, result.routes[layer + 1], workspace, error),
                              "standalone next segment failed: " + error);
            }
            else
            {
                ActivationBuffer hidden;
                segment_check(gpu.gpu_head->finish(output != LogitsOutput::None, output == LogitsOutput::Last,
                                                   hidden, result.logits, workspace, error),
                              "standalone head failed: " + error);
            }
        }
    }
    segment_check(!workspace.active(), "the final continuation retained active device state");
    for (size_t row = 0; row < (staged ? input.rows() : 1); ++row)
    {
        const uint64_t tokens = staged ? 1 : input.rows();
        segment_check(caches[0][row].token_count == tokens && caches[1][row].gated_delta_token_count == tokens,
                      "continuation cache metadata was not committed exactly once");
    }
    if (output == LogitsOutput::None)
        segment_check(result.logits.rows() == 1 && result.logits.columns() == 1 && result.logits.row(0)[0] == -71.0f,
                      "a logits-free continuation materialized or replaced its output");
    result.after = get_vulkan_statistics(gpu.vulkan_runtime);
    return result;
}

static void continuation_compare(const ContinuationResult& actual, const ContinuationResult& reference,
                                 size_t rows)
{
    segment_compare(actual.logits.values(), reference.logits.values(), "continued versus standalone GPU graph");
    for (size_t layer = 0; layer < segment_layers; ++layer)
        continuation_compare_routes(actual.routes[layer], reference.routes[layer]);
    for (const auto* result : {&actual, &reference})
    {
        segment_check(result->after.resident_graph_activation_upload_bytes - result->before.resident_graph_activation_upload_bytes
                              == rows * segment_hidden * sizeof(float)
                          && result->after.resident_graph_activation_download_bytes == result->before.resident_graph_activation_download_bytes
                          && result->after.resident_graph_route_download_bytes - result->before.resident_graph_route_download_bytes
                                 == rows * segment_layers * Router_vulkan::selected_columns(segment_top_k) * sizeof(float)
                          && result->after.attention_cpu_fallbacks == result->before.attention_cpu_fallbacks,
                      "continuation changed the activation/compact-route transfer contract or retried Attention on CPU");
    }
}

static void continuation_abort(const CompiledModel& gpu, size_t target_layer)
{
    const std::array<int32_t, 1> ids = {3};
    ActivationBuffer input;
    forward_embedding(gpu.weights.at(gpu.token_embedding), ids, input);
    const auto aggregate = continuation_upload(gpu, input);
    const auto context = gpu.gpu_segments.front()->context();
    const auto options = continuation_options();
    const std::array<uint64_t, 1> zero = {0};
    const std::array<uint64_t, 1> one = {1};
    LayerCache source_cache, target_cache;
    const std::array<LayerCache*, 1> source = {&source_cache}, target = {&target_cache};
    GpuSegmentWorkspace_vulkan workspace, initial_target;
    ExpertDispatchPlan routes;
    std::string error;
    segment_check(gpu.gpu_segments[target_layer]->forward(input, zero, target, options, routes, initial_target, error),
                  "cannot initialize a continuation abort cache: " + error);
    initial_target.reset();
    segment_check(gpu.gpu_segments[0]->forward(input, zero, source, options, routes, workspace, error),
                  "cannot initialize continuation abort source: " + error);
    const auto original_input = workspace.device_input();
    for (const bool submitted : {false, true})
    {
        auto pending = gpu.gpu_segments[target_layer]->prepare_continuation(one, target, options, workspace, error);
        segment_check(static_cast<bool>(pending), "cannot prepare the continuation abort fixture: " + error);
        const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            ncnn::VkCompute command(context->device());
            segment_check(pending->record(aggregate, command, continuation_option(*context)),
                          "an unsubmitted aborted state could not be recorded again");
            ExpertDispatchPlan sentinel;
            sentinel.assignment_count = 95;
            segment_check(!pending->publish(sentinel, workspace, error) && sentinel.assignment_count == 95
                              && workspace.device_input() == original_input && target_cache.token_count == (target_layer == 0 ? 1 : 0)
                              && target_cache.gated_delta_token_count == (target_layer == 1 ? 1 : 0),
                          "recording before a fence published routes, workspace, or KV metadata");
            if (submitted)
            {
                pending->mark_submit_attempted();
                segment_check(submit_compute_and_wait(command, context->device()) == 0, "discarded continuation submit failed");
                pending->mark_completed();
                // This deliberately discards a successfully completed command;
                // it is not a device-failure injection or a rollback claim.
            }
            else
                segment_check(command.reset() == 0, "cannot reset unsubmitted continuation recording");
        }
        pending->abort();
        pending->abort();
        pending.reset();
        segment_check(workspace.device_input() == original_input
                          && target_cache.token_count == (target_layer == 0 ? 1 : 0)
                          && target_cache.gated_delta_token_count == (target_layer == 1 ? 1 : 0),
                      "discarding a continuation published state");
        const auto after = get_vulkan_statistics(gpu.vulkan_runtime);
        segment_check(after.dispatches == before.dispatches && after.resident_graph_submissions == before.resident_graph_submissions,
                      "discarded continuation published success statistics");
        if (target_layer == 0)
            segment_check(target_cache.vulkan_attention_state_unknown == submitted,
                          "discarded Standard Attention did not distinguish recorded from submitted in-place KV writes");
    }
    auto retry = gpu.gpu_segments[target_layer]->prepare_continuation(one, target, options, workspace, error);
    segment_check(static_cast<bool>(retry), "cannot prepare terminal state retry probe");
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device());
        segment_check(!retry->record(aggregate, command, continuation_option(*context)),
                      "submitted but unpublished Standard/GDN state was silently reused");
        segment_check(command.reset() == 0, "cannot reset rejected terminal state recording");
    }
    retry->abort();
}

#endif

static int continuation_run()
{
#if NCNN_MOE_WITH_VULKAN
    if (get_gpu_count() == 0) return 77;
    try
    {
        const auto runtime = create_vulkan_runtime();
        const auto context = VulkanContext::acquire(get_default_gpu_index(), runtime, segment_flags);
        segment_check(static_cast<bool>(context), "cannot acquire continuation fixture context");
        if (context->device()->info.subgroup_size() != 32) return 77;
        for (const bool native_delta : {false, true})
        {
            auto gpu = segment_compile(true, native_delta);
            if (!native_delta)
            {
                const auto scheduled_node = [](const CompiledModel& model, ExecutionNodeType type, uint32_t layer) {
                    const auto found = std::find_if(model.schedule.node_order.begin(), model.schedule.node_order.end(),
                                                    [&](ExecutionNodeId id) {
                                                        const auto& node = model.graph.nodes[id];
                                                        return node.type == type && node.layer_plan_index == layer;
                                                    });
                    return found == model.schedule.node_order.end() ? invalid_execution_node_id : *found;
                };
                const uint32_t first_expert = scheduled_node(gpu, ExecutionNodeType::ExpertGroup, 0);
                const uint32_t next_attention = scheduled_node(gpu, ExecutionNodeType::Attention, 1);
                const uint32_t next_router = scheduled_node(gpu, ExecutionNodeType::Router, 1);
                const uint32_t last_expert = scheduled_node(gpu, ExecutionNodeType::ExpertGroup, 1);
                const uint32_t final_norm = scheduled_node(gpu, ExecutionNodeType::FinalNorm, invalid_execution_layer_id);
                const uint32_t lm_head = scheduled_node(gpu, ExecutionNodeType::LmHead, invalid_execution_layer_id);
                const auto rejects = [](const CompiledModel& model, uint32_t source) {
                    uint32_t resolved_fence = invalid_execution_node_id;
                    const auto resolved = GpuSegment_vulkan::continuation_target(model, source, resolved_fence);
                    return !resolved && resolved.error().code == ErrorCode::InvalidModel
                           && resolved_fence == invalid_execution_node_id;
                };
                segment_check(first_expert != invalid_execution_node_id && next_attention != invalid_execution_node_id
                                  && next_router != invalid_execution_node_id && last_expert != invalid_execution_node_id
                                  && final_norm != invalid_execution_node_id && lm_head != invalid_execution_node_id,
                              "continuation boundary fixture is incomplete");

                uint32_t fence = invalid_execution_node_id;
                const auto layer_target = GpuSegment_vulkan::continuation_target(gpu, first_expert, fence);
                segment_check(layer_target && layer_target.value() == next_attention && fence == next_router,
                              "layer continuation target and Router fence were not resolved together");
                const auto head_target = GpuSegment_vulkan::continuation_target(gpu, last_expert, fence);
                segment_check(head_target && head_target.value() == final_norm && fence == lm_head,
                              "head continuation target and LM-head fence were not resolved together");

                CompiledModel no_target = gpu;
                no_target.gpu_segments[1].reset();
                fence = invalid_execution_node_id;
                const auto unmatched = GpuSegment_vulkan::continuation_target(no_target, first_expert, fence);
                segment_check(unmatched && unmatched.value() == invalid_execution_node_id
                                  && fence == invalid_execution_node_id,
                              "a legal boundary without a selected target segment must remain not applicable");

                CompiledModel cpu_router = gpu;
                cpu_router.graph.nodes[next_router].backend = ExecutionBackend::Cpu;
                segment_check(rejects(cpu_router, first_expert),
                              "a selected continuation with a CPU Router fence must be rejected");

                CompiledModel missing_router = gpu;
                auto& no_router_order = missing_router.schedule.node_order;
                const auto router_position = std::find(no_router_order.begin(), no_router_order.end(), next_router);
                segment_check(router_position != no_router_order.end(), "Router fence is missing from the fixture schedule");
                no_router_order.erase(router_position);
                segment_check(rejects(missing_router, first_expert),
                              "a selected layer continuation without its Router fence must be rejected");

                CompiledModel missing_head = gpu;
                auto& no_head_order = missing_head.schedule.node_order;
                const auto head_position = std::find(no_head_order.begin(), no_head_order.end(), lm_head);
                segment_check(head_position != no_head_order.end(), "LM-head fence is missing from the fixture schedule");
                no_head_order.erase(head_position);
                segment_check(rejects(missing_head, last_expert),
                              "a selected head continuation without its LM-head fence must be rejected");

                CompiledModel blocked = gpu;
                auto& blocked_order = blocked.schedule.node_order;
                const auto target_position = std::find(blocked_order.begin(), blocked_order.end(), next_attention);
                segment_check(target_position != blocked_order.end(), "Attention target is missing from the fixture schedule");
                blocked_order.insert(target_position + 1, first_expert);
                segment_check(rejects(blocked, first_expert),
                              "a continuation fence must not be searched past another layer boundary");

                CompiledModel cross_device = gpu;
                const uint32_t layer_device = cross_device.graph.layer_plans[1].vulkan_device_index;
                cross_device.opt.vulkan_device_index = layer_device == 0 ? 1 : layer_device - 1;
                segment_check(rejects(cross_device, last_expert),
                              "a continuation whose head fence crosses device ownership must be rejected");
            }
            const std::array<int32_t, 2> ids = {2, 6};
            ActivationBuffer input;
            forward_embedding(gpu.weights.at(gpu.token_embedding), ids, input);
            struct Case
            {
                size_t experts;
                bool staged, two_waves;
                LogitsOutput output;
            };
            const std::array<Case, 4> cases = {{{1, false, false, LogitsOutput::All},
                                                {4, false, true, LogitsOutput::Last},
                                                {4, true, false, LogitsOutput::All},
                                                {4, true, true, LogitsOutput::None}}};
            for (const auto& test : cases)
            {
                const auto outputs = continuation_outputs(gpu, input.rows(), test.experts);
                const auto reference = continuation_direct(gpu, input, outputs, test.staged, false, test.two_waves, test.output);
                const auto actual = continuation_direct(gpu, input, outputs, test.staged, true, test.two_waves, test.output);
                continuation_compare(actual, reference, input.rows());
            }
            if (!native_delta)
            {
                continuation_abort(gpu, 0);
                continuation_abort(gpu, 1);
            }
        }
        std::cout << "generic continuation, standalone GPU parity, waves, staged Sessions, and publication boundaries passed\n";
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
    return ncnn::moe::continuation_run();
}
