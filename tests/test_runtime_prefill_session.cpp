#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

void test_prefill_decode_and_reset()
{
    BatchWorkspace batch_workspace;
    TemporaryModelPackage package;
    TestRuntime runtime;
    check(static_cast<bool>(has_flag(runtime.info().flags, RuntimeVulkanAttention) == has_flag(runtime.info().flags, RuntimeVulkanCpu)));
    check(static_cast<bool>(runtime.info().cpu_linear_num_threads >= 1));
    check(static_cast<bool>(runtime.info().float8_linear_row_group_size == 1 || runtime.info().float8_linear_row_group_size == 2
                            || runtime.info().float8_linear_row_group_size == 4
                            || runtime.info().float8_linear_row_group_size == 8));
    // This test clones and reschedules the model for CPU workspace checks;
    // load CPU storage rather than converting a device-only Hybrid copy.
    Option cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), cpu_options);
    check(static_cast<bool>(model));
    check(static_cast<bool>(model.value()->descriptor().model_type == "test_moe"));
    check(static_cast<bool>(model.value()->descriptor().layers.size() == 1));

    const CompiledModel& compiled = model_compiled(*model.value());
    SessionState reused_state;
    SessionStatistics reused_statistics;
    const std::array<int32_t, 3> direct_prompt = {0, 1, 2};
    auto all_status = forward_model(compiled,
                                    direct_prompt,
                                    reused_statistics,
                                    reused_state,
                                    0,
                                    LogitsOutput::All);
    check(static_cast<bool>(all_status));
    const auto all_output = batch_to_vectors(reused_state.logits);
    check(all_output.size() == direct_prompt.size());
    LayerState& reused_layer = reused_state.execution_state;
    check(reused_state.expert_scratch.backend_aggregated_output.allocated_bytes() == 0);
    ExpertScratch aggregate_scratch;
    aggregate_scratch.backend_aggregated_output.reset(2, 3, false);
    std::fill_n(aggregate_scratch.backend_aggregated_output.row(0), 6, 7.0f);
    aggregate_scratch.backend_aggregated_output_valid = true;
    ActivationBuffer accumulator(2, 3);
    const std::byte* aggregate_data = aggregate_scratch.backend_aggregated_output.bytes().data();
    const std::byte* accumulator_data = accumulator.bytes().data();
    check(init_moe_output(aggregate_scratch, 2, 3, accumulator));
    check(accumulator.bytes().data() == aggregate_data);
    check(aggregate_scratch.backend_aggregated_output.bytes().data() == accumulator_data);
    check(!aggregate_scratch.backend_aggregated_output_valid);
    check(std::all_of(accumulator.values().begin(), accumulator.values().end(), [](float value) { return value == 7.0f; }));
    check(!init_moe_output(aggregate_scratch, 2, 3, accumulator));
    check(std::all_of(accumulator.values().begin(), accumulator.values().end(), [](float value) { return value == 0.0f; }));
    aggregate_scratch.backend_aggregated_output_valid = true;
    check(!init_moe_output(aggregate_scratch, 1, 3, accumulator));
    check(!aggregate_scratch.backend_aggregated_output_valid);
    check(reused_layer.normalized.rows() == direct_prompt.size());
    LayerState retained_experts;
    retained_experts.resize_experts(3);
    retained_experts.active_experts()[2].input.reset(4, 8, false);
    retained_experts.active_experts()[2].output.reset(4, 8, false);
    const std::byte* retained_input = retained_experts.active_experts()[2].input.bytes().data();
    const std::byte* retained_output = retained_experts.active_experts()[2].output.bytes().data();
    retained_experts.resize_experts(1);
    check(retained_experts.active_experts().size() == 1);
    check(retained_experts.expert_slots.size() == 3);
    retained_experts.reset();
    retained_experts.resize_experts(3);
    check(retained_experts.active_experts()[2].input.bytes().data() == retained_input);
    check(retained_experts.active_experts()[2].output.bytes().data() == retained_output);
    retained_experts.resize_experts(0);
    check(retained_experts.active_experts().empty());
    check(reused_layer.router_logits.rows() == direct_prompt.size());
    const std::byte* normalized_data = reused_layer.normalized.bytes().data();
    const std::byte* router_data = reused_layer.router_logits.bytes().data();
    // Reused scratch must be fully overwritten, without clearing its storage.
    std::fill_n(reused_layer.normalized.row(0), reused_layer.normalized.values().size(),
                std::numeric_limits<float>::quiet_NaN());
    std::fill_n(reused_layer.router_logits.row(0), reused_layer.router_logits.values().size(),
                std::numeric_limits<float>::quiet_NaN());
    auto last_status = forward_model(compiled,
                                     direct_prompt,
                                     reused_statistics,
                                     reused_state,
                                     0,
                                     LogitsOutput::Last);
    check(static_cast<bool>(last_status));
    const auto last_output = batch_to_vectors(reused_state.logits);
    check(last_output.size() == 1);
    check(reused_layer.normalized.bytes().data() == normalized_data);
    check(reused_layer.router_logits.bytes().data() == router_data);
    check(last_output.front().size() == all_output.back().size());
    for (size_t column = 0; column < last_output.front().size(); ++column)
        check_near(last_output.front()[column], all_output.back()[column], 1e-5f);
    auto no_output_status = forward_model(compiled,
                                          direct_prompt,
                                          reused_statistics,
                                          reused_state,
                                          0,
                                          LogitsOutput::None);
    check(static_cast<bool>(no_output_status));
    check(reused_state.logits.rows() == 0);
    auto reused_last_status = forward_model(compiled,
                                            direct_prompt,
                                            reused_statistics,
                                            reused_state,
                                            0,
                                            LogitsOutput::Last);
    check(static_cast<bool>(reused_last_status));
    const auto reused_last_output = batch_to_vectors(reused_state.logits);
    check(reused_last_output.size() == 1);
    check(reused_last_output.front().size() == all_output.back().size());
    for (size_t column = 0; column < reused_last_output.front().size(); ++column)
        check_near(reused_last_output.front()[column], all_output.back()[column], 1e-5f);
    auto reused_all_status = forward_model(compiled,
                                           direct_prompt,
                                           reused_statistics,
                                           reused_state,
                                           0,
                                           LogitsOutput::All);
    check(static_cast<bool>(reused_all_status));
    const auto owned_logits = batch_to_vectors(reused_state.logits);
    check(owned_logits.size() == all_output.size());
    for (size_t row = 0; row < all_output.size(); ++row)
    {
        check(owned_logits[row].size() == all_output[row].size());
        for (size_t column = 0; column < all_output[row].size(); ++column)
            check_near(owned_logits[row][column], all_output[row][column], 1e-5f);
    }

    // A copied logits snapshot must survive later execution and FFN scratch
    // use without being overwritten.
    auto retained_status = forward_model(compiled, direct_prompt, reused_statistics,
                                         reused_state, 0, LogitsOutput::Last);
    check(static_cast<bool>(retained_status));
    check(reused_state.logits.rows() == 1);
    for (size_t column = 0; column < owned_logits.back().size(); ++column)
        check_near(reused_state.logits.row(0)[column], owned_logits.back()[column], 1e-5f);
    const std::vector<float> retained_logits(reused_state.logits.values().begin(), reused_state.logits.values().end());
    reused_state.expert_scratch.staged_output.reset(3, compiled.descriptor.vocabulary_size, true);
    reused_state.expert_scratch.staged_merged.reset(3, compiled.descriptor.hidden_size, true);
    check(std::equal(retained_logits.begin(), retained_logits.end(), reused_state.logits.values().begin()));
    check(owned_logits == all_output);

    CompiledModel missing_logits_model = compiled;
    check(missing_logits_model.net.layers[missing_logits_model.net.order.back()].kind == LayerKind::LmHead);
    missing_logits_model.net.order.pop_back();
    auto missing_logits = forward_model(missing_logits_model, direct_prompt, reused_statistics,
                                        reused_state, 0, LogitsOutput::Last);
    check(!missing_logits);
    check(missing_logits.error().message == "execution graph did not produce logits");

    // Sequential layers share FFN storage; KV/history state remains per layer.
    CompiledModel layered_model = compiled;
    layered_model.opt.hybrid_mode = HybridMode::CpuOnly;
    layered_model.descriptor.layers.push_back(compiled.descriptor.layers.front());
    layered_model.net.layer_plans.push_back(compiled.net.layer_plans.front());
    layered_model.net.layer_plans.back().layer_id = 1;
    check(static_cast<bool>(build_graph(layered_model, false)));
    auto layered_status = forward_model(layered_model, direct_prompt, reused_statistics, reused_state, 0, LogitsOutput::All);
    check(static_cast<bool>(layered_status));
    const auto layered_output = batch_to_vectors(reused_state.logits);
    check(reused_state.layers.size() == 2);
    check(reused_layer.normalized.bytes().data() == normalized_data);
    check(reused_layer.router_logits.bytes().data() == router_data);
    check(!reused_layer.experts_executed);
    const std::array<int32_t, 1> layered_decode = {1};
    check(static_cast<bool>(forward_model(layered_model, layered_decode, reused_statistics, reused_state, direct_prompt.size())));
    check(reused_layer.normalized.bytes().data() == normalized_data);
    check(reused_layer.router_logits.bytes().data() == router_data);
    auto layered_repeat_status = forward_model(layered_model, direct_prompt, reused_statistics, reused_state, 0, LogitsOutput::All);
    check(static_cast<bool>(layered_repeat_status));
    const auto layered_repeat = batch_to_vectors(reused_state.logits);
    check(layered_repeat.size() == layered_output.size());
    for (size_t row = 0; row < direct_prompt.size(); ++row)
    {
        check(layered_repeat[row].size() == layered_output[row].size());
        for (size_t column = 0; column < layered_output[row].size(); ++column)
            check_near(layered_repeat[row][column], layered_output[row][column], 1e-5f);
    }
    std::array<SessionState, 3> layered_states;
    std::array<SessionStatistics, 3> layered_statistics;
    std::array<std::vector<float>, 3> layered_logits;
    const std::array<DecodeBatchEntry, 3> layered_entries = {{
        {0, &layered_statistics[0], &layered_states[0], 0, &layered_logits[0]},
        {1, &layered_statistics[1], &layered_states[1], 0, &layered_logits[1]},
        {2, &layered_statistics[2], &layered_states[2], 0, &layered_logits[2]},
    }};
    auto layered_staged = forward_decode_batch(layered_model, layered_entries, batch_workspace);
    check(static_cast<bool>(layered_staged));
    for (size_t row = 0; row < direct_prompt.size(); ++row)
    {
        check(layered_logits[row].size() == layered_output[row].size());
        for (size_t column = 0; column < layered_output[row].size(); ++column)
            check_near(layered_logits[row][column], layered_output[row][column], 1e-5f);
        check(layered_states[row].layers.size() == 2);
        check(!layered_states[row].execution_state.experts_executed);
        check(layered_states[row].expert_scratch.backend_aggregated_output.allocated_bytes() == 0);
    }

    // A schedule must still finish the live FFN workspace before starting
    // another layer's router.
    Net overlapping_net = layered_model.net;
    const auto router = std::find_if(overlapping_net.order.begin(),
                                     overlapping_net.order.end(),
                                     [&](LayerId id) {
                                         return layered_model.net.layers[id].kind == LayerKind::Router
                                                && layered_model.net.layers[id].plan_index == 1;
                                     });
    check(router != overlapping_net.order.end());
    const LayerId router_id = *router;
    overlapping_net.order.erase(router);
    const auto router_zero = std::find_if(overlapping_net.order.begin(),
                                          overlapping_net.order.end(),
                                          [&](LayerId id) {
                                              return layered_model.net.layers[id].kind == LayerKind::Router
                                                     && layered_model.net.layers[id].plan_index == 0;
                                          });
    check(router_zero != overlapping_net.order.end());
    overlapping_net.order.insert(router_zero + 1, router_id);
    auto overlapping_status = overlapping_net.validate_order();
    check(!overlapping_status);
    check(overlapping_status.error().message == "execution schedule overlaps an unfinished FFN layer");

    // Exercise target-layer capture only; no speculative draft is executed.
    CompiledModel capture_model = compiled;
    capture_model.speculative.kind = SpeculativeModelKind::DSpark;
    capture_model.speculative.block_size = 1;
    capture_model.speculative.target_layer_ids = {0};
    capture_model.speculative.net.layer_plans = compiled.net.layer_plans;
    SessionState disabled_state;
    disabled_state.use_speculative_context = false;
    SessionStatistics disabled_statistics;
    auto disabled_status = forward_model(capture_model, direct_prompt, disabled_statistics, disabled_state, 0, LogitsOutput::Last);
    check(static_cast<bool>(disabled_status));
    const auto disabled_output = batch_to_vectors(disabled_state.logits);
    check(disabled_output.size() == 1);
    check(disabled_state.speculative_main_hidden.bytes().empty());
    check(disabled_output.front().size() == all_output.back().size());
    for (size_t column = 0; column < disabled_output.front().size(); ++column)
        check_near(disabled_output.front()[column], all_output.back()[column], 1e-5f);

    std::array<SessionState, 2> capture_states;
    std::array<SessionStatistics, 2> capture_statistics;
    std::array<std::vector<float>, 2> capture_logits;
    capture_states[1].use_speculative_context = false;
    const std::array<DecodeBatchEntry, 2> capture_entries = {{
        {0, &capture_statistics[0], &capture_states[0], 0, &capture_logits[0]},
        {0, &capture_statistics[1], &capture_states[1], 0, &capture_logits[1]},
    }};
    auto capture_output = forward_decode_batch(capture_model, capture_entries, batch_workspace);
    check(static_cast<bool>(capture_output));
    check(capture_states[0].speculative_main_hidden.rows() == 1);
    check(std::any_of(capture_states[0].speculative_main_hidden.values().begin(),
                      capture_states[0].speculative_main_hidden.values().end(),
                      [](float value) { return value != 0.0f; }));
    check(capture_states[1].speculative_main_hidden.bytes().empty());
    check(capture_logits[0].size() == capture_logits[1].size());
    for (size_t column = 0; column < capture_logits[0].size(); ++column)
        check_near(capture_logits[0][column], capture_logits[1][column], 1e-5f);

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));

    const std::vector<int32_t> prompt = {0, 1, 2};
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill));
    check(static_cast<bool>(prefill.value().processed_tokens == 3));
    check(static_cast<bool>(prefill.value().logits.size() == 4));

    const float normalized_equal = (1.0f + 1.0f / std::sqrt(1.0f + 1e-5f));
    const float final_value = normalized_equal / std::sqrt(normalized_equal * normalized_equal + 1e-5f);
    check_near(prefill.value().logits[0], final_value, 1e-5f);
    check_near(prefill.value().logits[1], final_value, 1e-5f);
    check_near(prefill.value().logits[2], 2.0f * final_value, 1e-5f);
    check_near(prefill.value().logits[3], -final_value, 1e-5f);

    check(static_cast<bool>(session.value()->sequence_length() == 3));
    check(static_cast<bool>(session.value()->statistics().prefill_tokens == 3));
    check(static_cast<bool>(session.value()->statistics().decode_tokens == 0));
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 3));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 2));
    check(static_cast<bool>(model.value()->descriptor().experts_per_token == 1));
    const SessionMetrics prefill_metrics = session.value()->metrics();
    check(prefill_metrics.generation.expert_assignments == 0);
    check(prefill_metrics.cumulative.expert_assignments == 3);
    check(prefill_metrics.generation.kv_cache_logical_size
          == prefill_metrics.cumulative.kv_cache_logical_size);

    auto decode = session.value()->decode(1);
    check(static_cast<bool>(decode));
    check(static_cast<bool>(decode.value().sequence_length == 4));
    check(static_cast<bool>(session.value()->statistics().decode_tokens == 1));
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 4));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 3));
    const SessionMetrics direct_metrics = session.value()->metrics();
    check(direct_metrics.generation.expert_assignments == 0);
    check(direct_metrics.cumulative.expert_assignments == 4);
    check(direct_metrics.generation.expert_cache_resident_size
          == direct_metrics.cumulative.expert_cache_resident_size);
    check(direct_metrics.generation.kv_cache_logical_size
          == direct_metrics.cumulative.kv_cache_logical_size);
    check(direct_metrics.generation.kv_cache_allocated_size
          == direct_metrics.cumulative.kv_cache_allocated_size);

    const float pre_norm = 1.0f / std::sqrt(0.5f + 1e-5f);
    const float expert_one_value = 1.0f + 2.0f * pre_norm;
    const float final_expert_one = expert_one_value / std::sqrt(expert_one_value * expert_one_value / 2.0f + 1e-5f);
    check_near(decode.value().logits[0], 0.0f, 1e-5f);
    check_near(decode.value().logits[1], final_expert_one, 2e-5f);
    check_near(decode.value().logits[2], final_expert_one, 2e-5f);
    check_near(decode.value().logits[3], 0.0f, 1e-5f);

    check(static_cast<bool>(session.value()->reset()));
    check(static_cast<bool>(session.value()->sequence_length() == 0));
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 0));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 0));

    TemporaryModelPackage token_experts_package(true);
    Option token_experts_options;
    token_experts_options.hybrid_mode = HybridMode::CpuOnly;
    auto token_experts_model = runtime.load_model(token_experts_package.path(), token_experts_options);
    check(static_cast<bool>(token_experts_model));
    check(static_cast<bool>(token_experts_model.value()->descriptor().hash_routing_layer_count == 1));
    auto token_experts_session = runtime.create_session(token_experts_model.value());
    check(static_cast<bool>(token_experts_session));

    const float route_epsilon = 1e-5f;
    const std::array<int32_t, 3> explicit_prompt = {0, 1, 2};
    auto explicit_prefill = token_experts_session.value()->prefill(explicit_prompt);
    check(static_cast<bool>(explicit_prefill));
    check(static_cast<bool>(token_experts_session.value()->statistics().expert_assignments == 3));
    // The table routes the final prompt token to Expert 1, whose down
    // projection scales the normalized coordinates by 2 and 3.
    const float two_axis_norm = 1.0f / std::sqrt(1.0f + route_epsilon);
    const float explicit_prefill_x = 1.0f + 2.0f * two_axis_norm;
    const float explicit_prefill_y = 1.0f + 3.0f * two_axis_norm;
    const float explicit_prefill_rms = std::sqrt((explicit_prefill_x * explicit_prefill_x
                                                  + explicit_prefill_y * explicit_prefill_y)
                                                     / 2.0f
                                                 + route_epsilon);
    const float explicit_prefill_x_norm = explicit_prefill_x / explicit_prefill_rms;
    const float explicit_prefill_y_norm = explicit_prefill_y / explicit_prefill_rms;
    check_near(explicit_prefill.value().logits[0], explicit_prefill_x_norm, 2e-5f);
    check_near(explicit_prefill.value().logits[1], explicit_prefill_y_norm, 2e-5f);
    check_near(explicit_prefill.value().logits[2], explicit_prefill_x_norm + explicit_prefill_y_norm, 2e-5f);
    check_near(explicit_prefill.value().logits[3], -explicit_prefill_x_norm, 2e-5f);

    auto explicit_decode = token_experts_session.value()->decode(1);
    check(static_cast<bool>(explicit_decode));
    check(static_cast<bool>(token_experts_session.value()->statistics().expert_assignments == 4));
    // Token 1 maps to Expert 0, whose down projection is the identity.
    const float one_axis_norm = 1.0f / std::sqrt(0.5f + route_epsilon);
    const float explicit_decode_value = 1.0f + one_axis_norm;
    const float explicit_decode_final = explicit_decode_value
                                        / std::sqrt(explicit_decode_value * explicit_decode_value / 2.0f + route_epsilon);
    check_near(explicit_decode.value().logits[0], 0.0f, 1e-5f);
    check_near(explicit_decode.value().logits[1], explicit_decode_final, 2e-5f);
    check_near(explicit_decode.value().logits[2], explicit_decode_final, 2e-5f);
    check_near(explicit_decode.value().logits[3], 0.0f, 1e-5f);

    check(static_cast<bool>(token_experts_session.value()->reset()));
    check(static_cast<bool>(token_experts_session.value()->statistics().expert_assignments == 0));
    const std::array<int32_t, 2> reused_explicit_prompt = {3, 0};
    auto reused_explicit_prefill = token_experts_session.value()->prefill(reused_explicit_prompt);
    check(static_cast<bool>(reused_explicit_prefill));
    check(static_cast<bool>(token_experts_session.value()->statistics().expert_assignments == 2));
    // After reset, the final token 0 maps to Expert 1 again.
    const float explicit_reuse_value = 1.0f + 2.0f * one_axis_norm;
    const float explicit_reuse_final = explicit_reuse_value
                                       / std::sqrt(explicit_reuse_value * explicit_reuse_value / 2.0f + route_epsilon);
    check_near(reused_explicit_prefill.value().logits[0], explicit_reuse_final, 2e-5f);
    check_near(reused_explicit_prefill.value().logits[1], 0.0f, 1e-5f);
    check_near(reused_explicit_prefill.value().logits[2], explicit_reuse_final, 2e-5f);
    check_near(reused_explicit_prefill.value().logits[3], -explicit_reuse_final, 2e-5f);
}
void test_cross_session_batch_scheduler()
{
    BatchWorkspace batch_workspace;
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    auto first = runtime.create_session(model.value());
    auto second = runtime.create_session(model.value());
    check(static_cast<bool>(first));
    check(static_cast<bool>(second));

    SchedulerOptions options;
    options.num_threads = 2;
    auto scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(scheduler));
    auto prefill_first = runtime.create_session(model.value());
    auto prefill_second = runtime.create_session(model.value());
    auto prefill_reference_first = runtime.create_session(model.value());
    auto prefill_reference_second = runtime.create_session(model.value());
    check(static_cast<bool>(prefill_first));
    check(static_cast<bool>(prefill_second));
    check(static_cast<bool>(prefill_reference_first));
    check(static_cast<bool>(prefill_reference_second));
    const std::vector<int32_t> first_prompt = {0, 1};
    const std::vector<int32_t> second_prompt = {1};
    auto prefill_future = scheduler.value()->submit_prefill({
        {prefill_first.value(), first_prompt},
        {prefill_second.value(), second_prompt},
    });
    auto prefill_results = prefill_future.get();
    auto first_reference = prefill_reference_first.value()->prefill(first_prompt);
    auto second_reference = prefill_reference_second.value()->prefill(second_prompt);
    check(static_cast<bool>(prefill_results.size() == 2));
    check(static_cast<bool>(prefill_results[0]));
    check(static_cast<bool>(prefill_results[1]));
    check(static_cast<bool>(first_reference));
    check(static_cast<bool>(second_reference));
    check(static_cast<bool>(prefill_first.value()->sequence_length() == 2));
    check(static_cast<bool>(prefill_second.value()->sequence_length() == 1));
    for (size_t index = 0;
         index < first_reference.value().logits.size();
         ++index)
    {
        check_near(prefill_results[0].value().logits[index],
                   first_reference.value().logits[index],
                   1e-5f);
    }
    for (size_t index = 0;
         index < second_reference.value().logits.size();
         ++index)
    {
        check_near(prefill_results[1].value().logits[index],
                   second_reference.value().logits[index],
                   1e-5f);
    }
    auto future = scheduler.value()->submit_decode({
        {first.value(), 0},
        {second.value(), 0},
    });
    std::vector<Result<DecodeResult>> results = future.get();
    check(static_cast<bool>(results.size() == 2));
    check(static_cast<bool>(results[0]));
    check(static_cast<bool>(results[1]));
    check(static_cast<bool>(first.value()->sequence_length() == 1));
    check(static_cast<bool>(second.value()->sequence_length() == 1));

    auto ordered_session = runtime.create_session(model.value());
    auto reference_session = runtime.create_session(model.value());
    check(static_cast<bool>(ordered_session));
    check(static_cast<bool>(reference_session));
    auto ordered_first = scheduler.value()->submit_decode({
        {ordered_session.value(), 0},
    });
    auto ordered_second = scheduler.value()->submit_decode({
        {ordered_session.value(), 1},
    });
    auto reference_first = reference_session.value()->decode(0);
    auto reference_second = reference_session.value()->decode(1);
    check(static_cast<bool>(reference_first));
    check(static_cast<bool>(reference_second));
    std::vector<Result<DecodeResult>> ordered_first_result = ordered_first.get();
    std::vector<Result<DecodeResult>> ordered_second_result = ordered_second.get();
    check(static_cast<bool>(ordered_first_result[0]));
    check(static_cast<bool>(ordered_second_result[0]));
    check(static_cast<bool>(ordered_first_result[0].value().sequence_length == 1));
    check(static_cast<bool>(ordered_second_result[0].value().sequence_length == 2));
    for (size_t index = 0; index < reference_second.value().logits.size(); ++index)
    {
        check_near(ordered_second_result[0].value().logits[index], reference_second.value().logits[index], 1e-5f);
    }

    auto duplicate_future = scheduler.value()->submit_decode({
        {first.value(), 0},
        {first.value(), 1},
    });
    std::vector<Result<DecodeResult>> duplicate_results = duplicate_future.get();
    check(static_cast<bool>(!duplicate_results[0]));
    check(static_cast<bool>(!duplicate_results[1]));
    check(static_cast<bool>(first.value()->sequence_length() == 1));
    const SchedulerStatistics statistics = scheduler.value()->statistics();
    check(static_cast<bool>(statistics.num_threads >= 1));
    check(static_cast<bool>(statistics.num_threads <= options.num_threads));
    check(static_cast<bool>(statistics.prefill_batches == 1));
    check(static_cast<bool>(statistics.staged_prefill_batches == 1));
    check(static_cast<bool>(statistics.decode_batches == 4));
    check(static_cast<bool>(statistics.staged_decode_batches == 1));

    check(static_cast<bool>(prefill_first.value()->reset()));
    check(static_cast<bool>(prefill_second.value()->reset()));
    check(static_cast<bool>(prefill_reference_first.value()->reset()));
    check(static_cast<bool>(prefill_reference_second.value()->reset()));
    auto staggered_third = runtime.create_session(model.value());
    auto staggered_reference_third = runtime.create_session(model.value());
    check(static_cast<bool>(staggered_third));
    check(static_cast<bool>(staggered_reference_third));

    auto staggered_scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(staggered_scheduler));
    std::array<SessionPtr, 3> staggered_sessions = {
        prefill_first.value(),
        prefill_second.value(),
        staggered_third.value(),
    };
    std::array<SessionPtr, 3> staggered_reference_sessions = {
        prefill_reference_first.value(),
        prefill_reference_second.value(),
        staggered_reference_third.value(),
    };
    std::array<std::vector<int32_t>, 3> staggered_prompts = {
        std::vector<int32_t>{0},
        std::vector<int32_t>{1, 0, 1},
        std::vector<int32_t>{0, 1},
    };
    std::array<size_t, 3> staggered_order = {0, 1, 2};
    std::sort(staggered_order.begin(),
              staggered_order.end(),
              [&](size_t left, size_t right) {
                  return std::less<Session*>()(staggered_sessions[right].get(),
                                               staggered_sessions[left].get());
              });
    const auto original_staggered_sessions = staggered_sessions;
    const auto original_staggered_reference_sessions = staggered_reference_sessions;
    const auto original_staggered_prompts = staggered_prompts;
    for (size_t index = 0; index < staggered_order.size(); ++index)
    {
        staggered_sessions[index] = original_staggered_sessions[staggered_order[index]];
        staggered_reference_sessions[index] = original_staggered_reference_sessions[staggered_order[index]];
        staggered_prompts[index] = original_staggered_prompts[staggered_order[index]];
    }
    check(std::less<Session*>()(staggered_sessions[1].get(), staggered_sessions[0].get()));
    std::vector<PrefillBatchRequest> staggered_requests;
    staggered_requests.reserve(staggered_sessions.size());
    for (size_t index = 0; index < staggered_sessions.size(); ++index)
        staggered_requests.push_back({staggered_sessions[index], staggered_prompts[index]});
    const std::vector<Result<PrefillResult>> staggered_results = staggered_scheduler.value()->submit_prefill(std::move(staggered_requests)).get();

    check(staggered_results.size() == staggered_sessions.size());
    for (size_t index = 0; index < staggered_sessions.size(); ++index)
    {
        auto expected = staggered_reference_sessions[index]->prefill(staggered_prompts[index]);
        check(static_cast<bool>(expected));
        check(static_cast<bool>(staggered_results[index]));
        check(staggered_results[index].value().processed_tokens == staggered_prompts[index].size());
        check(staggered_sessions[index]->sequence_length() == staggered_prompts[index].size());
        check(staggered_results[index].value().logits.size() == expected.value().logits.size());
        for (size_t logit = 0; logit < expected.value().logits.size(); ++logit)
            check_near(staggered_results[index].value().logits[logit], expected.value().logits[logit], 1e-5f);
    }

    const CompiledModel& compiled = model_compiled(*model.value());
    std::array<SessionState, 3> mixed_states;
    std::array<SessionStatistics, 3> mixed_statistics;
    std::array<std::vector<float>, 3> mixed_logits;
    const std::array<DecodeBatchEntry, 3> mixed_entries = {{
        {0, &mixed_statistics[0], &mixed_states[0], 0, &mixed_logits[0]},
        {1, &mixed_statistics[1], &mixed_states[1], 0},
        {2, &mixed_statistics[2], &mixed_states[2], 0, &mixed_logits[2]},
    }};
    auto mixed_status = forward_decode_batch(compiled, mixed_entries, batch_workspace);
    check(static_cast<bool>(mixed_status));

    std::array<const std::byte*, 3> hidden_data;
    // Each Session keeps its activation storage across staged batches.
    for (size_t index = 0; index < mixed_states.size(); ++index)
    {
        ActivationBuffer& hidden = mixed_states[index].hidden;
        check(hidden.rows() == 1);
        hidden_data[index] = hidden.bytes().data();
        std::fill_n(hidden.row(0), hidden.values().size(),
                    std::numeric_limits<float>::quiet_NaN());
    }
    mixed_status = forward_decode_batch(compiled, mixed_entries, batch_workspace);
    check(static_cast<bool>(mixed_status));
    for (size_t index = 0; index < mixed_states.size(); ++index)
    {
        check(mixed_states[index].hidden.bytes().data() == hidden_data[index]);
    }

    std::array<SessionState, 3> all_states;
    std::array<SessionStatistics, 3> all_statistics;
    std::array<std::vector<float>, 3> all_output_logits;
    const std::array<DecodeBatchEntry, 3> all_entries = {{
        {0, &all_statistics[0], &all_states[0], 0, &all_output_logits[0]},
        {1, &all_statistics[1], &all_states[1], 0, &all_output_logits[1]},
        {2, &all_statistics[2], &all_states[2], 0, &all_output_logits[2]},
    }};
    auto all_status = forward_decode_batch(compiled, all_entries, batch_workspace);
    check(static_cast<bool>(all_status));
    for (size_t index = 0; index < mixed_entries.size(); ++index)
    {
        check(mixed_states[index].hidden.rows() == all_states[index].hidden.rows());
        check(mixed_states[index].hidden.columns() == all_states[index].hidden.columns());
        for (size_t row = 0; row < mixed_states[index].hidden.rows(); ++row)
        {
            for (uint32_t column = 0; column < mixed_states[index].hidden.columns(); ++column)
            {
                check_near(mixed_states[index].hidden.row(row)[column],
                           all_states[index].hidden.row(row)[column],
                           1e-5f);
            }
        }
        if (!mixed_entries[index].logits)
        {
            check(mixed_logits[index].empty());
            continue;
        }
        check(mixed_logits[index].size() == all_output_logits[index].size());
        for (size_t column = 0; column < mixed_logits[index].size(); ++column)
            check_near(mixed_logits[index][column], all_output_logits[index][column], 1e-5f);
    }

    auto different_input_first = runtime.create_session(model.value());
    auto different_input_second = runtime.create_session(model.value());
    auto different_reference_first_session = runtime.create_session(model.value());
    auto different_reference_second_session = runtime.create_session(model.value());
    check(static_cast<bool>(different_input_first));
    check(static_cast<bool>(different_input_second));
    check(static_cast<bool>(different_reference_first_session));
    check(static_cast<bool>(different_reference_second_session));
    std::array<SessionPtr, 2> different_input_sessions = {
        different_input_first.value(),
        different_input_second.value(),
    };
    std::array<SessionPtr, 2> different_input_references = {
        different_reference_first_session.value(),
        different_reference_second_session.value(),
    };
    if (std::less<Session*>()(different_input_sessions[0].get(), different_input_sessions[1].get()))
    {
        std::swap(different_input_sessions[0], different_input_sessions[1]);
        std::swap(different_input_references[0], different_input_references[1]);
    }
    check(std::less<Session*>()(different_input_sessions[1].get(), different_input_sessions[0].get()));
    const SchedulerStatistics before_different_inputs = scheduler.value()->statistics();
    auto different_input_future = scheduler.value()->submit_decode({
        {different_input_sessions[0], 0},
        {different_input_sessions[1], 1},
    });
    const std::vector<Result<DecodeResult>> different_input_results = different_input_future.get();
    check(static_cast<bool>(different_input_results.size() == 2));
    check(static_cast<bool>(different_input_results[0]));
    check(static_cast<bool>(different_input_results[1]));
    auto different_first_result = different_input_references[0]->decode(0);
    auto different_second_result = different_input_references[1]->decode(1);
    check(static_cast<bool>(different_first_result));
    check(static_cast<bool>(different_second_result));
    check(different_input_results[0].value().logits == different_first_result.value().logits);
    check(different_input_results[1].value().logits == different_second_result.value().logits);
    const SchedulerStatistics after_different_inputs = scheduler.value()->statistics();
    check(static_cast<bool>(after_different_inputs.staged_decode_batches
                            == before_different_inputs.staged_decode_batches + 1));

    SchedulerOptions large_scheduler_options;
    large_scheduler_options.num_threads = 1025;
    auto large_scheduler = runtime.create_scheduler(large_scheduler_options);
    check(static_cast<bool>(large_scheduler));
    if (large_scheduler)
    {
        const SchedulerStatistics statistics = large_scheduler.value()->statistics();
        check(static_cast<bool>(statistics.num_threads >= 1));
        check(static_cast<bool>(statistics.num_threads <= large_scheduler_options.num_threads));
    }

    if (has_usable_vulkan_device(runtime.info(), RuntimeVulkanCpu))
    {
        AttentionPackage attention_package;
        Option hybrid_options;
        hybrid_options.hybrid_mode = HybridMode::HybridExperts;
        Option cpu_options;
        cpu_options.hybrid_mode = HybridMode::CpuOnly;
        auto hybrid_model = runtime.load_model(attention_package.path(), hybrid_options);
        auto cpu_model = runtime.load_model(attention_package.path(), cpu_options);
        check(static_cast<bool>(hybrid_model));
        check(static_cast<bool>(cpu_model));

        SchedulerOptions pipeline_options;
        pipeline_options.num_threads = 4;
        auto pipeline_scheduler = runtime.create_scheduler(pipeline_options);
        check(static_cast<bool>(pipeline_scheduler));
        std::vector<SessionPtr> hybrid_sessions;
        std::vector<SessionPtr> cpu_sessions;
        for (uint32_t index = 0; index < 4; ++index)
        {
            auto hybrid_session = runtime.create_session(hybrid_model.value());
            auto cpu_session = runtime.create_session(cpu_model.value());
            check(static_cast<bool>(hybrid_session));
            check(static_cast<bool>(cpu_session));
            hybrid_sessions.push_back(hybrid_session.value());
            cpu_sessions.push_back(cpu_session.value());
        }

        constexpr uint32_t pipeline_rounds = 6;
        for (uint32_t round = 0; round < pipeline_rounds; ++round)
        {
            std::vector<DecodeBatchRequest> requests;
            for (uint32_t session_index = 0; session_index < hybrid_sessions.size(); ++session_index)
            {
                requests.push_back({
                    hybrid_sessions[session_index],
                    static_cast<int32_t>((round + session_index) % 2),
                });
            }
            std::vector<Result<DecodeResult>> pipeline_results = pipeline_scheduler.value()->submit_decode(std::move(requests)).get();
            check(static_cast<bool>(pipeline_results.size() == hybrid_sessions.size()));
            for (uint32_t session_index = 0; session_index < hybrid_sessions.size(); ++session_index)
            {
                auto cpu_result = cpu_sessions[session_index]->decode(static_cast<int32_t>((round + session_index) % 2));
                check(static_cast<bool>(pipeline_results[session_index]));
                check(static_cast<bool>(cpu_result));
                check(static_cast<bool>(pipeline_results[session_index].value().sequence_length == round + 1));
                check(static_cast<bool>(pipeline_results[session_index].value().logits.size() == cpu_result.value().logits.size()));
                for (size_t logit = 0; logit < cpu_result.value().logits.size(); ++logit)
                {
                    check_near(pipeline_results[session_index].value().logits[logit], cpu_result.value().logits[logit], 1e-4f);
                }
            }
        }
        for (const SessionPtr& session : hybrid_sessions)
        {
            const SessionStatistics session_statistics = session->statistics();
            check(static_cast<bool>(session_statistics.vulkan_attention_blocks == pipeline_rounds));
            check(static_cast<bool>(session_statistics.vulkan_compute_submissions == pipeline_rounds * 2));
        }
        const SchedulerStatistics pipeline_statistics = pipeline_scheduler.value()->statistics();
        check(static_cast<bool>(pipeline_statistics.decode_batches == pipeline_rounds));
        check(static_cast<bool>(pipeline_statistics.staged_decode_batches == pipeline_rounds));
    }
}

void test_batch_scheduler_submission_contract()
{
    AttentionPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    auto other_model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    check(static_cast<bool>(other_model));
    auto first = runtime.create_session(model.value());
    auto second = runtime.create_session(model.value());
    auto other = runtime.create_session(other_model.value());
    check(static_cast<bool>(first));
    check(static_cast<bool>(second));
    check(static_cast<bool>(other));
    check(static_cast<bool>(first.value()->prefill(std::vector<int32_t>{0})));
    check(static_cast<bool>(second.value()->prefill(std::vector<int32_t>{1})));
    const SessionStatistics first_before = first.value()->statistics();
    const SessionStatistics second_before = second.value()->statistics();

    SchedulerOptions options;
    options.num_threads = 2;
    auto validation_scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(validation_scheduler));

    // Empty submissions are no-ops; malformed batches must not enqueue a valid prefix.
    check(validation_scheduler.value()->submit_prefill({}).get().empty());
    check(validation_scheduler.value()->submit_decode({}).get().empty());
    const auto reject_prefill = [&](std::vector<PrefillBatchRequest> requests) {
        const size_t count = requests.size();
        const std::vector<Result<PrefillResult>> results = validation_scheduler.value()->submit_prefill(std::move(requests)).get();
        check(results.size() == count);
        for (const Result<PrefillResult>& result : results)
        {
            check(!result);
            check(result.error().code == ErrorCode::InvalidArgument);
        }
    };
    const auto reject_decode = [&](std::vector<DecodeBatchRequest> requests) {
        const size_t count = requests.size();
        const std::vector<Result<DecodeResult>> results = validation_scheduler.value()->submit_decode(std::move(requests)).get();
        check(results.size() == count);
        for (const Result<DecodeResult>& result : results)
        {
            check(!result);
            check(result.error().code == ErrorCode::InvalidArgument);
        }
    };
    reject_prefill({{first.value(), {1}}, {nullptr, {0}}});
    reject_prefill({{first.value(), {1}}, {second.value(), {}}});
    reject_prefill({{first.value(), {1}}, {first.value(), {0}}});
    reject_prefill({{first.value(), {1}}, {other.value(), {0}}});
    reject_decode({{first.value(), 1}, {nullptr, 0}});
    reject_decode({{first.value(), 1}, {first.value(), 0}});
    reject_decode({{first.value(), 1}, {second.value(), 2}});
    validation_scheduler.value().reset();

    const auto check_unchanged = [](const SessionPtr& session, const SessionStatistics& before) {
        const SessionStatistics after = session->statistics();
        check(session->sequence_length() == 1);
        check(after.prefill_tokens == before.prefill_tokens);
        check(after.decode_tokens == before.decode_tokens);
        check(after.expert_assignments == before.expert_assignments);
        check(after.expert_batches == before.expert_batches);
        check(after.expert_cpu_routes == before.expert_cpu_routes);
        check(after.expert_gpu_routes == before.expert_gpu_routes);
        check(after.kv_cache_logical_size == before.kv_cache_logical_size);
        check(after.kv_cache_allocated_size == before.kv_cache_allocated_size);
    };
    check_unchanged(first.value(), first_before);
    check_unchanged(second.value(), second_before);
    check(other.value()->sequence_length() == 0);
    check(other.value()->statistics().prefill_tokens == 0);
    check(other.value()->statistics().decode_tokens == 0);

    auto ordered = runtime.create_session(model.value());
    auto reference = runtime.create_session(model.value());
    check(static_cast<bool>(ordered));
    check(static_cast<bool>(reference));
    auto draining_scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(draining_scheduler));
    const std::vector<int32_t> prompt = {0, 1};
    auto queued_prefill = draining_scheduler.value()->submit_prefill({{ordered.value(), prompt}});
    constexpr std::array<int32_t, 6> tokens = {1, 0, 1, 1, 0, 0};
    std::vector<std::future<std::vector<Result<DecodeResult>>>> queued_decodes;
    queued_decodes.reserve(tokens.size());
    for (int32_t token : tokens)
        queued_decodes.push_back(draining_scheduler.value()->submit_decode({{ordered.value(), token}}));

    // Destruction must complete all accepted work without requiring future.get().
    draining_scheduler.value().reset();
    check(queued_prefill.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
    for (std::future<std::vector<Result<DecodeResult>>>& future : queued_decodes)
        check(future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
    const std::vector<Result<PrefillResult>> prefilled = queued_prefill.get();
    check(prefilled.size() == 1);
    check(static_cast<bool>(prefilled[0]));
    check(prefilled[0].value().processed_tokens == prompt.size());
    auto reference_prefill = reference.value()->prefill(prompt);
    check(static_cast<bool>(reference_prefill));
    check(prefilled[0].value().logits.size() == reference_prefill.value().logits.size());
    for (size_t logit = 0; logit < reference_prefill.value().logits.size(); ++logit)
        check_near(prefilled[0].value().logits[logit], reference_prefill.value().logits[logit], 1e-5f);
    for (size_t index = 0; index < tokens.size(); ++index)
    {
        const std::vector<Result<DecodeResult>> results = queued_decodes[index].get();
        auto expected = reference.value()->decode(tokens[index]);
        check(results.size() == 1);
        check(static_cast<bool>(results[0]));
        check(static_cast<bool>(expected));
        check(results[0].value().sequence_length == prompt.size() + index + 1);
        check(results[0].value().logits.size() == expected.value().logits.size());
        for (size_t logit = 0; logit < expected.value().logits.size(); ++logit)
            check_near(results[0].value().logits[logit], expected.value().logits[logit], 1e-5f);
    }
    check(ordered.value()->sequence_length() == prompt.size() + tokens.size());
    check(ordered.value()->statistics().prefill_tokens == prompt.size());
    check(ordered.value()->statistics().decode_tokens == tokens.size());

    auto staged_first = runtime.create_session(model.value());
    auto staged_second = runtime.create_session(model.value());
    auto reference_first = runtime.create_session(model.value());
    auto reference_second = runtime.create_session(model.value());
    check(static_cast<bool>(staged_first));
    check(static_cast<bool>(staged_second));
    check(static_cast<bool>(reference_first));
    check(static_cast<bool>(reference_second));
    auto resubmit_scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(resubmit_scheduler));
    const std::vector<int32_t> first_prompt = {0, 1};
    const std::vector<int32_t> second_prompt = {1};
    const std::vector<PrefillBatchRequest> requests = {
        {staged_first.value(), first_prompt},
        {staged_second.value(), second_prompt},
    };
    constexpr uint32_t rounds = 4;
    auto pending = resubmit_scheduler.value()->submit_prefill(requests);
    for (uint32_t round = 0; round < rounds; ++round)
    {
        const std::vector<Result<PrefillResult>> results = pending.get();
        // A ready future must mean the staged Session reservations are already released.
        if (round + 1 < rounds)
            pending = resubmit_scheduler.value()->submit_prefill(requests);
        check(results.size() == 2);
        check(static_cast<bool>(results[0]));
        check(static_cast<bool>(results[1]));
        check(results[0].value().processed_tokens == first_prompt.size());
        check(results[1].value().processed_tokens == second_prompt.size());
        auto expected_first = reference_first.value()->prefill(first_prompt);
        auto expected_second = reference_second.value()->prefill(second_prompt);
        check(static_cast<bool>(expected_first));
        check(static_cast<bool>(expected_second));
        check(results[0].value().logits.size() == expected_first.value().logits.size());
        check(results[1].value().logits.size() == expected_second.value().logits.size());
        for (size_t logit = 0; logit < expected_first.value().logits.size(); ++logit)
            check_near(results[0].value().logits[logit], expected_first.value().logits[logit], 1e-5f);
        for (size_t logit = 0; logit < expected_second.value().logits.size(); ++logit)
            check_near(results[1].value().logits[logit], expected_second.value().logits[logit], 1e-5f);
    }
    check(staged_first.value()->sequence_length() == rounds * first_prompt.size());
    check(staged_second.value()->sequence_length() == rounds * second_prompt.size());
    check(staged_first.value()->statistics().prefill_tokens == rounds * first_prompt.size());
    check(staged_second.value()->statistics().prefill_tokens == rounds * second_prompt.size());
    check(resubmit_scheduler.value()->statistics().staged_prefill_batches == rounds);
}

void test_independent_batch_scheduler()
{
    AttentionPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    std::array<SessionPtr, 2> sessions;
    std::array<SessionPtr, 2> references;
    for (size_t i = 0; i < sessions.size(); i++)
    {
        auto session = runtime.create_session(model.value());
        auto reference = runtime.create_session(model.value());
        check(static_cast<bool>(session));
        check(static_cast<bool>(reference));
        sessions[i] = session.value();
        references[i] = reference.value();
    }

    SchedulerOptions options;
    options.num_threads = 2;
    options.use_staged_decode = false;
    auto scheduler = runtime.create_scheduler(options);
    check(static_cast<bool>(scheduler));
    constexpr size_t rounds = 4;
    std::vector<std::future<std::vector<Result<DecodeResult>>>> pending;
    for (size_t round = 0; round < rounds; round++)
    {
        std::vector<DecodeBatchRequest> requests;
        for (size_t i = 0; i < sessions.size(); i++)
            requests.push_back({sessions[(round + i) % sessions.size()], static_cast<int32_t>(i)});
        pending.push_back(scheduler.value()->submit_decode(std::move(requests)));
    }
    auto mixed = scheduler.value()->submit_decode({{sessions[0], 2}, {sessions[1], 0}});
    check(scheduler.value()->statistics().decode_batches == rounds + 1);
    check(scheduler.value()->statistics().staged_decode_batches == 0);
    scheduler.value().reset();

    // Request order and per-Session FIFO must hold even when batch order alternates.
    for (size_t round = 0; round < rounds; round++)
    {
        check(pending[round].wait_for(std::chrono::seconds(0)) == std::future_status::ready);
        const auto results = pending[round].get();
        check(results.size() == sessions.size());
        for (size_t i = 0; i < sessions.size(); i++)
        {
            auto expected = references[(round + i) % sessions.size()]->decode(static_cast<int32_t>(i));
            check(static_cast<bool>(expected));
            check(static_cast<bool>(results[i]));
            check(results[i].value().sequence_length == round + 1);
            check(results[i].value().logits.size() == expected.value().logits.size());
            for (size_t j = 0; j < expected.value().logits.size(); j++)
                check_near(results[i].value().logits[j], expected.value().logits[j], 1e-5f);
        }
    }
    const auto mixed_results = mixed.get();
    check(mixed_results.size() == 2);
    check(!mixed_results[0]);
    check(mixed_results[0].error().code == ErrorCode::InvalidArgument);
    check(static_cast<bool>(mixed_results[1]));
    check(mixed_results[1].value().sequence_length == rounds + 1);
    check(sessions[0]->sequence_length() == rounds);
    check(sessions[1]->sequence_length() == rounds + 1);
}

void test_staged_bfloat16_dispatch_telemetry()
{
    Bfloat16StagedBatchPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    SchedulerOptions scheduler_options;
    scheduler_options.num_threads = 2;
    auto scheduler = runtime.create_scheduler(scheduler_options);
    check(static_cast<bool>(scheduler));

    auto run_batch = [&](bool batched_enabled) {
        Option batch_options = opt;
        const uint64_t batch_flags = OptimizationCpuBfloat16Batched;
        if (batched_enabled)
            batch_options.optimization_flags |= batch_flags;
        else
            batch_options.optimization_flags &= ~batch_flags;
        auto batch_model = runtime.load_model(package.path(), batch_options);
        check(static_cast<bool>(batch_model));
        check(model_compiled(*batch_model.value()).opt.optimization_flags
              == batch_options.optimization_flags);
        std::vector<SessionPtr> sessions;
        std::vector<DecodeBatchRequest> requests;
        sessions.reserve(4);
        requests.reserve(4);
        for (uint32_t index = 0; index < 4; ++index)
        {
            auto session = runtime.create_session(batch_model.value());
            check(static_cast<bool>(session));
            sessions.push_back(session.value());
            requests.push_back({session.value(), static_cast<int32_t>(index)});
        }
        const std::vector<Result<DecodeResult>> results = scheduler.value()->submit_decode(std::move(requests)).get();
        check(static_cast<bool>(results.size() == sessions.size()));
        for (const Result<DecodeResult>& result : results)
            check(static_cast<bool>(result));
        return sessions;
    };

    run_batch(true);
    const bool kernel_available = std::string(bfloat16_batched_linear_kernel_name(opt.optimization_flags
                                                                                  | OptimizationCpuBfloat16Batched))
                                  != "unavailable";
    run_batch(false);

    if (kernel_available)
    {
        std::vector<DecodeBatchRequest> first_requests;
        std::vector<DecodeBatchRequest> second_requests;
        Option enabled_options = opt;
        enabled_options.optimization_flags |= OptimizationCpuBfloat16Batched;
        auto enabled_model = runtime.load_model(package.path(), enabled_options);
        check(static_cast<bool>(enabled_model));
        check(model_compiled(*enabled_model.value()).opt.optimization_flags
              == enabled_options.optimization_flags);
        for (uint32_t index = 0; index < 4; ++index)
        {
            auto first = runtime.create_session(enabled_model.value());
            auto second = runtime.create_session(enabled_model.value());
            check(static_cast<bool>(first));
            check(static_cast<bool>(second));
            first_requests.push_back({first.value(), static_cast<int32_t>(index)});
            second_requests.push_back({second.value(), static_cast<int32_t>(3 - index)});
        }
        auto first_future = scheduler.value()->submit_decode(std::move(first_requests));
        auto second_future = scheduler.value()->submit_decode(std::move(second_requests));
        const std::vector<Result<DecodeResult>> first_results = first_future.get();
        const std::vector<Result<DecodeResult>> second_results = second_future.get();
        check(static_cast<bool>(first_results.size() == 4));
        check(static_cast<bool>(second_results.size() == 4));
        for (const Result<DecodeResult>& result : first_results)
            check(static_cast<bool>(result));
        for (const Result<DecodeResult>& result : second_results)
            check(static_cast<bool>(result));
    }
}

void test_chunked_prefill_statistics_commit_on_success()
{
    Bfloat16StagedBatchPackage package;
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    opt.optimization_flags |= OptimizationCpuBfloat16Batched;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));

    SessionOptions session_options;
    session_options.prefill_chunk_size = 4;
    auto control = runtime.create_session(model.value(), session_options);
    auto session = runtime.create_session(model.value(), session_options);
    check(static_cast<bool>(control));
    check(static_cast<bool>(session));

    const std::array<int32_t, 4> prompt = {0, 1, 2, 3};
    for (const SessionPtr& current : {control.value(), session.value()})
    {
        auto prefilled = current->prefill(prompt);
        check(static_cast<bool>(prefilled));
        check(prefilled.value().processed_tokens == prompt.size());
        check(current->sequence_length() == prompt.size());
        const SessionStatistics statistics = current->statistics();
        check(statistics.prefill_tokens == prompt.size());
        check(statistics.expert_assignments == prompt.size());
        check(statistics.expert_batches == 1);
    }

    const SessionStatistics before = session.value()->statistics();
    const uint64_t sequence_length_before = session.value()->sequence_length();
    // A token in a later chunk must be rejected before execution starts.
    const std::array<int32_t, 5> invalid_prompt = {0, 1, 2, 3, 4};
    auto failed = session.value()->prefill(invalid_prompt);
    check(!failed);
    check(failed.error().code == ErrorCode::InvalidArgument);
    check(failed.error().message == "token id is outside the model vocabulary");
    const SessionStatistics after = session.value()->statistics();
    check(session.value()->sequence_length() == sequence_length_before);
    check(after.prefill_tokens == before.prefill_tokens);
    check(after.decode_tokens == before.decode_tokens);
    check(after.expert_assignments == before.expert_assignments);
    check(after.expert_batches == before.expert_batches);
    check(after.expert_cpu_routes == before.expert_cpu_routes);
    check(after.expert_gpu_routes == before.expert_gpu_routes);

    check(static_cast<bool>(session.value()->reset()));
    check(session.value()->sequence_length() == 0);
}

void test_chunked_prefill_invalid_token_preserves_attention_state()
{
    for (uint32_t sliding_window : {0u, 2u})
    {
        AttentionPackage package(false, sliding_window, false, false);
        TestRuntime runtime;
        Option opt;
        opt.hybrid_mode = HybridMode::CpuOnly;
        auto model = runtime.load_model(package.path(), opt);
        check(static_cast<bool>(model));
        SessionOptions options;
        options.prefill_chunk_size = 1;
        options.use_speculative_context = false;
        auto session = runtime.create_session(model.value(), options).value();
        auto control = runtime.create_session(model.value(), options).value();
        for (int32_t invalid_token : {-1, 2})
        {
            const SessionStatistics before = session->statistics();
            const uint64_t position = session->sequence_length();
            const std::array<int32_t, 3> invalid_prompt = {0, 1, invalid_token};
            auto failed = session->prefill(invalid_prompt);
            check(!failed);
            check(failed.error().code == ErrorCode::InvalidArgument);
            check(failed.error().message == "token id is outside the model vocabulary");
            check(session->sequence_length() == position);
            check(session->statistics().prefill_tokens == before.prefill_tokens);
            check(session->statistics().kv_cache_logical_size == before.kv_cache_logical_size);
            auto actual = session->decode(1);
            auto expected = control->decode(1);
            check(static_cast<bool>(actual));
            check(static_cast<bool>(expected));
            check(actual.value().logits.size() == expected.value().logits.size());
            for (size_t index = 0; index < expected.value().logits.size(); ++index)
                check_near(actual.value().logits[index], expected.value().logits[index], 1e-6f);
        }
    }
}

void test_staged_invalid_token_preserves_attention_state()
{
    AttentionPackage package(false, 2, false, false);
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    SchedulerOptions options;
    options.num_threads = 1;
    auto scheduler = runtime.create_scheduler(options).value();
    std::array<SessionPtr, 2> sessions;
    std::array<SessionPtr, 2> controls;
    for (size_t index = 0; index < sessions.size(); ++index)
    {
        sessions[index] = runtime.create_session(model.value()).value();
        controls[index] = runtime.create_session(model.value()).value();
    }
    auto pending = scheduler->submit_prefill({
        {sessions[0], {0}},
        {sessions[1], {1, 0, 2}},
    });
    const auto failed_prefill = pending.get();
    check(failed_prefill.size() == sessions.size());
    for (const auto& result : failed_prefill)
    {
        check(!result);
        check(result.error().code == ErrorCode::InvalidArgument);
    }
    for (size_t round = 0; round < 2; ++round)
    {
        if (round != 0)
        {
            auto pending = scheduler->submit_decode({
                {sessions[0], 0},
                {sessions[1], -1},
            });
            const auto failed_decode = pending.get();
            check(failed_decode.size() == sessions.size());
            for (const auto& result : failed_decode)
            {
                check(!result);
                check(result.error().code == ErrorCode::InvalidArgument);
            }
        }
        for (size_t index = 0; index < sessions.size(); ++index)
        {
            check(sessions[index]->sequence_length() == round);
            check(sessions[index]->statistics().prefill_tokens == 0);
            check(sessions[index]->statistics().decode_tokens == round);
            auto actual = sessions[index]->decode(1);
            auto expected = controls[index]->decode(1);
            check(static_cast<bool>(actual));
            check(static_cast<bool>(expected));
            check(actual.value().logits.size() == expected.value().logits.size());
            for (size_t logit = 0; logit < expected.value().logits.size(); ++logit)
                check_near(actual.value().logits[logit], expected.value().logits[logit], 1e-6f);
        }
    }
}

void test_session_execution_failure_requires_reset()
{
    AttentionPackage package(false, 0, false, false);
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    SessionOptions options;
    options.prefill_chunk_size = 1;
    for (bool prefill : {false, true})
    {
        for (bool exception : {false, true})
        {
            auto session = runtime.create_session(model.value(), options).value();
            auto control = runtime.create_session(model.value(), options).value();
            bool caught = false;
            {
                ScopedLmHeadFailure failure(*model.value(), exception);
                try
                {
                    if (prefill)
                    {
                        const std::array<int32_t, 2> input = {0, 1};
                        auto result = session->prefill(input);
                        check(!result);
                        check(result.error().message == "LM head node has an invalid weight binding");
                    }
                    else
                    {
                        auto result = session->decode(0);
                        check(!result);
                        check(result.error().message == "LM head node has an invalid weight binding");
                    }
                }
                catch (const std::runtime_error& error)
                {
                    check(exception);
                    check(std::string(error.what()).find("host storage was released") != std::string::npos);
                    caught = true;
                }
            }
            check(caught == exception);
            check_session_requires_reset(session);
            check(static_cast<bool>(session->reset()));
            auto actual = session->decode(1);
            auto expected = control->decode(1);
            check(static_cast<bool>(actual));
            check(static_cast<bool>(expected));
            check(actual.value().logits == expected.value().logits);
        }
    }
}

void test_staged_execution_failure_requires_reset()
{
    AttentionPackage package(false, 0, false, false);
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    SchedulerOptions options;
    options.num_threads = 1;
    auto scheduler = runtime.create_scheduler(options).value();
    for (bool prefill : {false, true})
    {
        for (bool exception : {false, true})
        {
            const std::array<SessionPtr, 2> sessions = {
                runtime.create_session(model.value()).value(),
                runtime.create_session(model.value()).value(),
            };
            {
                ScopedLmHeadFailure failure(*model.value(), exception);
                if (prefill)
                {
                    auto pending = scheduler->submit_prefill({
                        {sessions[0], {0}},
                        {sessions[1], {1, 0}},
                    });
                    const auto results = pending.get();
                    check(results.size() == sessions.size());
                    for (const auto& result : results)
                    {
                        check(!result);
                        check(result.error().code == ErrorCode::InternalError);
                    }
                }
                else
                {
                    auto pending = scheduler->submit_decode({
                        {sessions[0], 0},
                        {sessions[1], 1},
                    });
                    const auto results = pending.get();
                    check(results.size() == sessions.size());
                    for (const auto& result : results)
                    {
                        check(!result);
                        check(result.error().code == ErrorCode::InternalError);
                    }
                }
            }
            for (const SessionPtr& session : sessions)
                check_session_requires_reset(session);

            // Reject a poisoned member before advancing an otherwise healthy batch member.
            auto healthy = runtime.create_session(model.value()).value();
            auto control = runtime.create_session(model.value()).value();
            auto pending = scheduler->submit_decode({
                {healthy, 0},
                {sessions[1], 0},
            });
            const auto rejected = pending.get();
            check(rejected.size() == 2);
            for (const auto& result : rejected)
                check(!result);
            auto expected = control->decode(1);
            auto actual = healthy->decode(1);
            check(static_cast<bool>(expected));
            check(static_cast<bool>(actual));
            check(actual.value().logits == expected.value().logits);
            for (const SessionPtr& session : sessions)
            {
                check(static_cast<bool>(session->reset()));
                auto recovered = session->decode(1);
                check(static_cast<bool>(recovered));
                check(recovered.value().logits == expected.value().logits);
            }
        }
    }
}

void test_speculative_generation()
{
    AttentionPackage package(false, 0, false, false);
    TestRuntime runtime;
    Option opt;
    opt.hybrid_mode = HybridMode::CpuOnly;
    auto model = runtime.load_model(package.path(), opt);
    check(static_cast<bool>(model));
    CompiledModel& compiled = const_cast<CompiledModel&>(model_compiled(*model.value()));
    TensorData projection;
    projection.dtype = DType::Float32;
    projection.shape = {2, 4};
    projection.float32_data = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    auto projection_handle = compiled.weights.add("test_mtp_projection", std::move(projection));
    check(static_cast<bool>(projection_handle));
    compiled.speculative.kind = SpeculativeModelKind::Mtp;
    compiled.speculative.block_size = 2;
    compiled.speculative.net.layer_plans = compiled.net.layer_plans;
    compiled.speculative.mtp_embedding_norm_weight = compiled.final_norm_weight;
    compiled.speculative.mtp_hidden_norm_weight = compiled.final_norm_weight;
    compiled.speculative.final_norm_weight = compiled.final_norm_weight;
    compiled.speculative.mtp_input_projection_weight = projection_handle.value();
    check(static_cast<bool>(build_graph(compiled, false)));
    check(compiled.speculative.enabled());

    const auto check_mtp_norm_capture = [](const SessionState& state, size_t rows) {
        check(state.speculative_main_hidden.rows() == rows);
        check(state.speculative_main_hidden.columns() == state.hidden.columns());
        check(state.hidden.rows() == rows);
        check(state.hidden.columns() == state.speculative_main_hidden.columns());
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t column = 0; column < state.hidden.columns(); ++column)
                check_near(state.speculative_main_hidden.row(row)[column], state.hidden.row(row)[column], 1e-5f);
        }
    };
    const std::array<int32_t, 2> mtp_prompt = {0, 1};
    SessionStatistics mtp_all_statistics;
    SessionState mtp_all_state;
    auto mtp_all_status = forward_model(compiled,
                                        mtp_prompt,
                                        mtp_all_statistics,
                                        mtp_all_state,
                                        0,
                                        LogitsOutput::All);
    check(static_cast<bool>(mtp_all_status));
    const auto mtp_all_logits = batch_to_vectors(mtp_all_state.logits);
    check(mtp_all_logits.size() == mtp_prompt.size());
    check_mtp_norm_capture(mtp_all_state, mtp_prompt.size());

    SessionStatistics mtp_last_statistics;
    SessionState mtp_last_state;
    auto mtp_last_status = forward_model(compiled,
                                         mtp_prompt,
                                         mtp_last_statistics,
                                         mtp_last_state,
                                         0,
                                         LogitsOutput::Last);
    check(static_cast<bool>(mtp_last_status));
    const auto mtp_last_logits = batch_to_vectors(mtp_last_state.logits);
    check(mtp_last_logits.size() == 1);
    check_mtp_norm_capture(mtp_last_state, mtp_prompt.size());
    check(mtp_last_logits.front().size() == mtp_all_logits.back().size());
    for (size_t column = 0; column < mtp_last_logits.front().size(); ++column)
        check_near(mtp_last_logits.front()[column], mtp_all_logits.back()[column], 1e-5f);

    SessionStatistics mtp_none_statistics;
    SessionState mtp_none_state;
    auto mtp_none_status = forward_model(compiled,
                                         mtp_prompt,
                                         mtp_none_statistics,
                                         mtp_none_state,
                                         0,
                                         LogitsOutput::None);
    check(static_cast<bool>(mtp_none_status));
    check(mtp_none_state.logits.rows() == 0);
    check_mtp_norm_capture(mtp_none_state, mtp_prompt.size());

    SessionStatistics mtp_reused_statistics;
    SessionState mtp_reused_state;
    auto mtp_reused_none = forward_model(compiled,
                                         mtp_prompt,
                                         mtp_reused_statistics,
                                         mtp_reused_state,
                                         0,
                                         LogitsOutput::None);
    check(static_cast<bool>(mtp_reused_none));
    check(mtp_reused_state.logits.rows() == 0);
    auto mtp_reused_last = forward_model(compiled,
                                         mtp_prompt,
                                         mtp_reused_statistics,
                                         mtp_reused_state,
                                         0,
                                         LogitsOutput::Last);
    check(static_cast<bool>(mtp_reused_last));
    const auto mtp_reused_logits = batch_to_vectors(mtp_reused_state.logits);
    check(mtp_reused_logits.size() == 1);
    check_mtp_norm_capture(mtp_reused_state, mtp_prompt.size());
    for (size_t column = 0; column < mtp_reused_logits.front().size(); ++column)
        check_near(mtp_reused_logits.front()[column], mtp_all_logits.back()[column], 1e-5f);

    const std::array<int32_t, 3> chunked_prompt = {0, 1, 0};
    SessionStatistics chunked_statistics;
    SessionState chunked_state;
    auto chunked_first = forward_model(compiled,
                                       std::span<const int32_t>(chunked_prompt.data(), 1),
                                       chunked_statistics,
                                       chunked_state,
                                       0,
                                       LogitsOutput::None);
    check(static_cast<bool>(chunked_first));
    check(chunked_state.logits.rows() == 0);
    auto chunked_second = forward_model(compiled,
                                        std::span<const int32_t>(chunked_prompt.data() + 1, 2),
                                        chunked_statistics,
                                        chunked_state,
                                        1,
                                        LogitsOutput::Last);
    check(static_cast<bool>(chunked_second));
    check_mtp_norm_capture(chunked_state, 2);
    const auto chunked_prefill_logits = batch_to_vectors(chunked_state.logits);
    check(chunked_prefill_logits.size() == 1);
    const int32_t next_token = 1;
    auto chunked_decode = forward_model(compiled,
                                        std::span<const int32_t>(&next_token, 1),
                                        chunked_statistics,
                                        chunked_state,
                                        chunked_prompt.size(),
                                        LogitsOutput::Last);
    check(static_cast<bool>(chunked_decode));
    check_mtp_norm_capture(chunked_state, 1);
    const auto chunked_decode_logits = batch_to_vectors(chunked_state.logits);

    SessionStatistics full_statistics;
    SessionState full_state;
    auto full_prefill = forward_model(compiled,
                                      chunked_prompt,
                                      full_statistics,
                                      full_state,
                                      0,
                                      LogitsOutput::Last);
    check(static_cast<bool>(full_prefill));
    check_mtp_norm_capture(full_state, chunked_prompt.size());
    auto full_decode = forward_model(compiled,
                                     std::span<const int32_t>(&next_token, 1),
                                     full_statistics,
                                     full_state,
                                     chunked_prompt.size(),
                                     LogitsOutput::Last);
    check(static_cast<bool>(full_decode));
    check_mtp_norm_capture(full_state, 1);
    const auto full_decode_logits = batch_to_vectors(full_state.logits);
    check(chunked_decode_logits.size() == full_decode_logits.size());
    check(chunked_decode_logits.front().size() == full_decode_logits.front().size());
    for (size_t column = 0; column < full_decode_logits.front().size(); ++column)
        check_near(chunked_decode_logits.front()[column], full_decode_logits.front()[column], 1e-5f);

    GenerationOptions options;
    options.max_new_tokens = 4;
    options.sampling.temperature = 0.0f;
    options.speculative_confidence_threshold = 0.0f;
    const std::array<int32_t, 1> input = {0};
    const auto check_shared_head_host_release = [&] {
#if NCNN_MOE_WITH_VULKAN
        if (get_gpu_count() == 0)
            return;
        const TensorData original_head = compiled.weights.at(compiled.lm_head_weight);
        const CompiledOperator original_operator = compiled.operators.at_weight(compiled.lm_head_weight);
        const HybridMode original_mode = compiled.opt.hybrid_mode;
        const VulkanRuntimePtr original_runtime = compiled.vulkan_runtime;
        const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
        for (DType dtype : {DType::Float32, DType::BFloat16})
        {
            TensorData head = original_head;
            if (dtype == DType::BFloat16)
            {
                head.dtype = dtype;
                head.bfloat16_data.reserve(head.float32_data.size());
                for (float value : head.float32_data)
                    head.bfloat16_data.push_back(float_to_bfloat16(value));
                head.float32_data.clear();
            }
            compiled.weights.at_mutable(compiled.lm_head_weight) = std::move(head);
            CompiledOperator& head_operator = compiled.operators.at_weight_mutable(compiled.lm_head_weight);
            head_operator = {};
            // The small fixture uses the generic Vulkan Linear implementation
            // for both dtypes; native BF16 projections require wider inputs.
            head_operator.linear = Linear::create(compiled.weights.at(compiled.lm_head_weight),
                                                  nullptr,
                                                  automatic_vulkan_device_index,
                                                  vulkan_runtime,
                                                  compiled.opt.optimization_flags);
            check(static_cast<bool>(head_operator.linear));
            compiled.vulkan_runtime = vulkan_runtime;
            compiled.opt.hybrid_mode = HybridMode::CpuOnly;
            check(static_cast<bool>(build_graph(compiled, false)));
            const auto check_head_backend = [&](ExecutionBackend backend) {
                bool found = false;
                for (const Layer& node : compiled.net.layers)
                {
                    if (node.kind == LayerKind::LmHead)
                    {
                        check(node.backend == backend);
                        found = true;
                    }
                }
                check(found);
            };
            check_head_backend(ExecutionBackend::Cpu);

            // An available Vulkan operator must not override a CPU graph's
            // placement, including when the speculative head is shared.
            auto reference = runtime.create_session(model.value()).value();
            auto expected = reference->generate(input, options);
            check(static_cast<bool>(expected));
            check(reference->statistics().speculative_proposals > 0);
            check(reference->statistics().vulkan_linear_dispatches == 0);
            auto expected_decode = reference->decode(0);
            check(static_cast<bool>(expected_decode));

            compiled.opt.hybrid_mode = HybridMode::HybridExperts;
            check(static_cast<bool>(build_graph(compiled, false)));
            check_head_backend(ExecutionBackend::Vulkan);
            release_host_copies(compiled);
            check(compiled.weights.at(compiled.lm_head_weight).float32_values().empty());
            check(compiled.weights.at(compiled.lm_head_weight).bfloat16_values().empty());

            // Execute proposals after the real host allocation is gone, from
            // independent Sessions sharing the same device-only head.
            for (size_t session_index = 0; session_index < 2; ++session_index)
            {
                auto session = runtime.create_session(model.value()).value();
                auto actual = session->generate(input, options);
                check(static_cast<bool>(actual));
                check(session->statistics().speculative_proposals > 0);
                check(session->statistics().vulkan_linear_dispatches > 0);
                check(actual.value().tokens.size() == expected.value().tokens.size());
                for (size_t index = 0; index < expected.value().tokens.size(); ++index)
                {
                    check(actual.value().tokens[index].token_id == expected.value().tokens[index].token_id);
                    check_near(actual.value().tokens[index].probability,
                               expected.value().tokens[index].probability,
                               1e-5f);
                }
                auto actual_decode = session->decode(0);
                check(static_cast<bool>(actual_decode));
                check(actual_decode.value().logits.size() == expected_decode.value().logits.size());
                for (size_t index = 0; index < expected_decode.value().logits.size(); ++index)
                    check_near(actual_decode.value().logits[index], expected_decode.value().logits[index], 1e-5f);
            }
        }
        compiled.weights.at_mutable(compiled.lm_head_weight) = original_head;
        compiled.operators.at_weight_mutable(compiled.lm_head_weight) = original_operator;
        compiled.opt.hybrid_mode = original_mode;
        compiled.vulkan_runtime = original_runtime;
        check(static_cast<bool>(build_graph(compiled, false)));
#endif
    };
    for (int speculative_kind = 0; speculative_kind < 2; ++speculative_kind)
    {
        if (speculative_kind == 0)
        {
            const std::vector<float> original_lm_head = compiled.weights.at(compiled.lm_head_weight).float32_data;
            const auto original_mtp_final_norm = compiled.speculative.final_norm_weight;
            // This fixture's target hidden has positive sum, so this head selects token 1; zeroing only the MTP final norm ties draft logits and selects token 0.
            // temperature > 0 with top_k=1 makes these disjoint one-hot distributions exercise rejection and residual sampling deterministically.
            TensorData zero_mtp_final_norm;
            zero_mtp_final_norm.dtype = DType::Float32;
            zero_mtp_final_norm.shape = {2};
            zero_mtp_final_norm.float32_data = {0.0f, 0.0f};
            auto zero_mtp_final_norm_handle = compiled.weights.add("test_mtp_zero_final_norm", std::move(zero_mtp_final_norm));
            check(static_cast<bool>(zero_mtp_final_norm_handle));
            compiled.speculative.final_norm_weight = zero_mtp_final_norm_handle.value();
            compiled.weights.at_mutable(compiled.lm_head_weight).float32_data = {-1.0f, -1.0f, 1.0f, 1.0f};
            compiled.operators.bind_weight_count(compiled.weights.size());
            check(static_cast<bool>(build_graph(compiled, false)));

            GenerationOptions residual_options = options;
            residual_options.max_new_tokens = 3;
            residual_options.sampling.temperature = 1.0f;
            residual_options.sampling.top_k = 1;
            residual_options.speculative_confidence_threshold = 0.0f;
            SessionOptions residual_session_options;
            residual_session_options.sampling_seed = 1234;
            auto residual_session = runtime.create_session(model.value(), residual_session_options).value();
            auto residual_result = residual_session->generate(input, residual_options);
            check(static_cast<bool>(residual_result));
            check(residual_result.value().tokens.size() == residual_options.max_new_tokens);
            check(residual_session->statistics().speculative_proposals == 1);
            check(residual_session->statistics().speculative_accepted_tokens == 0);
            for (const StreamToken& token : residual_result.value().tokens)
            {
                check(token.token_id == 1);
                check(token.probability == 1.0f);
            }

            compiled.weights.at_mutable(compiled.lm_head_weight).float32_data = original_lm_head;
            compiled.speculative.final_norm_weight = original_mtp_final_norm;
            compiled.operators.bind_weight_count(compiled.weights.size());
            check(static_cast<bool>(build_graph(compiled, false)));

            GenerationOptions mtp_options = options;
            mtp_options.max_new_tokens = 12;
            for (uint32_t block_size : {2u, 4u, 8u})
            {
                compiled.speculative.block_size = block_size;
                auto mtp_session = runtime.create_session(model.value()).value();
                auto mtp_result = mtp_session->generate(input, mtp_options);
                check(static_cast<bool>(mtp_result));
                check(mtp_result.value().tokens.size() == mtp_options.max_new_tokens);
                check(mtp_session->statistics().speculative_proposals > 0);
                check(mtp_session->statistics().speculative_accepted_tokens > 0);

                GenerationOptions ordinary_options = mtp_options;
                ordinary_options.use_speculative = false;
                auto ordinary_session = runtime.create_session(model.value()).value();
                auto ordinary_result = ordinary_session->generate(input, ordinary_options);
                check(static_cast<bool>(ordinary_result));
                check(ordinary_result.value().tokens.size() == mtp_result.value().tokens.size());
                for (size_t index = 0; index < mtp_result.value().tokens.size(); ++index)
                {
                    check(mtp_result.value().tokens[index].token_id
                          == ordinary_result.value().tokens[index].token_id);
                    check(mtp_result.value().tokens[index].probability
                          == ordinary_result.value().tokens[index].probability);
                }
            }
            compiled.speculative.block_size = 2;
        }
        if (speculative_kind == 1)
        {
            auto add_float = [&compiled](const char* name,
                                         std::initializer_list<uint32_t> shape,
                                         std::initializer_list<float> values) {
                TensorData tensor;
                tensor.dtype = DType::Float32;
                tensor.shape.assign(shape.begin(), shape.end());
                tensor.float32_data.assign(values.begin(), values.end());
                auto added = compiled.weights.add(name, std::move(tensor));
                check(static_cast<bool>(added));
                return added.value();
            };

            AttentionBlockPlan latent;
            latent.kind = AttentionKind::MultiHeadLatent;
            latent.head_count = 1;
            latent.kv_head_count = 1;
            latent.head_dimension = 2;
            latent.sliding_window = 4;
            latent.initial_context_length = 16;
            latent.query_lora_rank = 1;
            latent.rope_head_dimension = 2;
            latent.output_lora_rank = 1;
            latent.output_group_count = 1;
            latent.norm_epsilon = compiled.descriptor.norm_epsilon;
            latent.kv_cache_dtype = compiled.descriptor.kv_cache_dtype;
            latent.pre_attention_norm_weight = add_float("test_dspark_pre_attention_norm", {2}, {1.0f, 1.0f});
            latent.query_a_weight = add_float("test_dspark_query_a", {1, 2}, {0.5f, -0.25f});
            latent.query_norm_weight = add_float("test_dspark_query_norm", {1}, {1.0f});
            latent.query_b_weight = add_float("test_dspark_query_b", {2, 1}, {1.0f, 0.5f});
            latent.key_value_weight = add_float("test_dspark_key_value", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
            latent.key_value_norm_weight = add_float("test_dspark_key_value_norm", {2}, {1.0f, 1.0f});
            latent.sinks = add_float("test_dspark_sinks", {1}, {0.0f});

            TensorData output_a;
            output_a.dtype = DType::Float8E4M3;
            output_a.shape = {1, 2};
            std::shared_ptr<uint8_t[]> output_a_storage(new uint8_t[2], std::default_delete<uint8_t[]>());
            output_a_storage[0] = float_to_float8_e4m3(1.0f);
            output_a_storage[1] = float_to_float8_e4m3(0.5f);
            output_a.mapped_data = std::shared_ptr<const uint8_t>(output_a_storage, output_a_storage.get());
            output_a.mapped_size = 2;
            output_a.quantization_scales = {1.0f};
            auto output_a_handle = compiled.weights.add("test_dspark_output_a", std::move(output_a));
            check(static_cast<bool>(output_a_handle));
            latent.output_a_weight = output_a_handle.value();
            latent.output_b_weight = add_float("test_dspark_output_b", {2, 1}, {0.0f, 0.0f});
            for (const ExpertPlan& expert : compiled.net.layer_plans.front().moe.experts)
            {
                TensorData& down = compiled.weights.at_mutable(expert.down_weight);
                std::fill(down.float32_data.begin(), down.float32_data.end(), 0.0f);
            }
            compiled.net.layer_plans.front().attention = latent;

            AttentionDescriptor& descriptor_attention = compiled.descriptor.layers.front().attention;
            descriptor_attention.kind = AttentionKind::MultiHeadLatent;
            descriptor_attention.head_count = 1;
            descriptor_attention.kv_head_count = 1;
            descriptor_attention.head_dimension = 2;
            descriptor_attention.sliding_window = 4;
            descriptor_attention.initial_context_length = 16;
            descriptor_attention.query_lora_rank = 1;
            descriptor_attention.kv_lora_rank = 2;
            descriptor_attention.qk_rope_head_dimension = 2;
            descriptor_attention.output_lora_rank = 1;
            descriptor_attention.output_group_count = 1;

            compiled.speculative.kind = SpeculativeModelKind::DSpark;
            compiled.speculative.block_size = 2;
            compiled.speculative.noise_token_id = 0;
            compiled.speculative.markov_rank = 1;
            compiled.speculative.target_layer_ids = {0};
            compiled.speculative.net.layer_plans = compiled.net.layer_plans;
            compiled.speculative.main_projection_weight = latent.key_value_weight;
            compiled.speculative.main_norm_weight = latent.key_value_norm_weight;
            compiled.speculative.final_norm_weight = compiled.final_norm_weight;
            compiled.speculative.hyper_head_function = add_float("test_dspark_hyper_head_function", {1, 2}, {0.0f, 0.0f});
            compiled.speculative.hyper_head_base = add_float("test_dspark_hyper_head_base", {1}, {1.0f});
            compiled.speculative.hyper_head_scale = add_float("test_dspark_hyper_head_scale", {1}, {1.0f});
            compiled.speculative.markov_embedding_weight = add_float("test_dspark_markov_embedding", {2, 1}, {1.0f, -1.0f});
            compiled.speculative.markov_head_weight = add_float("test_dspark_markov_head", {2, 1}, {-1.0f, 1.0f});

            TensorData confidence;
            confidence.dtype = DType::Int4;
            confidence.shape = {1, 3};
            confidence.quantization_group_size = 3;
            confidence.quantization_scales = {1.0f};
            confidence.quantization_zero_points = {8};
            confidence.quantized_data = {0x88, 0x08};
            confidence.input_column_indices = {2, 1, 0};
            auto confidence_handle = compiled.weights.add("test_dspark_confidence", std::move(confidence));
            check(static_cast<bool>(confidence_handle));
            compiled.speculative.confidence_weight = confidence_handle.value();

            compiled.operators.bind_weight_count(compiled.weights.size());
            const std::vector<float> original_lm_head = compiled.weights.at(compiled.lm_head_weight).float32_data;
            const std::vector<float> dspark_lm_head = {0.0f, 1.0f, 1.0f, 0.0f};
            compiled.weights.at_mutable(compiled.lm_head_weight).float32_data = dspark_lm_head;
            check(static_cast<bool>(build_graph(compiled, false)));
            check(compiled.speculative.enabled());

#if NCNN_MOE_WITH_VULKAN
            if (get_gpu_count() > 0)
            {
                const VulkanRuntimePtr dspark_vulkan_runtime = create_vulkan_runtime();
                check(static_cast<bool>(dspark_vulkan_runtime));
                CompilerOption dspark_pipeline_options;
                dspark_pipeline_options.flags = BackendVulkanDense;
                dspark_pipeline_options.optimization_flags = g_test_optimization_flags
                                                             & ~static_cast<uint64_t>(OptimizationVulkanAttention);
                dspark_pipeline_options.vulkan_runtime = dspark_vulkan_runtime;
                const auto prepare_hybrid_dspark = [&](CompiledModel& candidate) {
                    // The tiny mapped FP8 projection is a CPU fixture. Use its
                    // equivalent FP32 matrix to isolate Shared Expert preparation.
                    TensorData& output_a = candidate.weights.at_mutable(latent.output_a_weight);
                    output_a = TensorData();
                    output_a.dtype = DType::Float32;
                    output_a.shape = {1, 2};
                    output_a.float32_data = {0.5f, 1.0f};
                    candidate.opt.hybrid_mode = HybridMode::HybridExperts;
                    candidate.opt.optimization_flags = dspark_pipeline_options.optimization_flags;
                    candidate.vulkan_runtime = dspark_vulkan_runtime;
                    auto prepared = prepare_model_pipeline(candidate, dspark_pipeline_options);
                    if (!prepared)
                        throw std::runtime_error("DSpark Hybrid preparation failed: " + prepared.error().message);
                    check(static_cast<bool>(build_graph(candidate, false)));
                };
                const auto speculative_shared_node = [](const CompiledModel& candidate) -> const Layer& {
                    const LayerId id = candidate.speculative.layer_nodes.front().shared_expert_group;
                    check(id < candidate.speculative.net.layers.size());
                    return candidate.speculative.net.layers[id];
                };
                CompiledModel no_shared_dspark = compiled;
                prepare_hybrid_dspark(no_shared_dspark);

                const auto add_dspark_shared = [](CompiledModel& candidate, DType dtype) {
                    const uint32_t hidden = candidate.descriptor.hidden_size;
                    const uint32_t intermediate = candidate.descriptor.layers.front().moe.intermediate_size;
                    const auto add_weight = [&](const std::string& name, bool down) {
                        TensorData weight;
                        weight.dtype = dtype;
                        weight.shape = down ? std::vector<uint32_t>{hidden, intermediate}
                                            : std::vector<uint32_t>{intermediate, hidden};
                        if (dtype == DType::Int4)
                        {
                            weight.quantization_group_size = weight.shape[1];
                            weight.quantization_scales.assign(static_cast<size_t>(weight.shape[0]), 0.125f);
                            weight.quantization_zero_points.assign(static_cast<size_t>(weight.shape[0]), 8);
                            weight.quantized_data.assign(static_cast<size_t>(weight.shape[0]) * ((weight.shape[1] + 1) / 2), 0x99);
                        }
                        else
                            weight.float32_data.assign(static_cast<size_t>(weight.shape[0]) * weight.shape[1], 0.125f);
                        const auto added = candidate.weights.add(name, std::move(weight));
                        check(static_cast<bool>(added));
                        return added.value();
                    };
                    const auto configure = [&](MoeBlockPlan& moe, const std::string& prefix) {
                        moe.has_shared_expert = true;
                        moe.shared_expert.activation = ExpertActivation::Silu;
                        moe.shared_expert.gate_weight = add_weight(prefix + "gate", false);
                        moe.shared_expert.up_weight = add_weight(prefix + "up", false);
                        moe.shared_expert.down_weight = add_weight(prefix + "down", true);
                    };
                    configure(candidate.net.layer_plans.front().moe, "test_target_shared_");
                    configure(candidate.speculative.net.layer_plans.front().moe, "test_dspark_shared_");
                    candidate.descriptor.layers.front().moe.shared_expert_count = 1;
                    candidate.descriptor.layers.front().moe.shared_expert_weight_dtype = dtype;
                    candidate.operators.bind_weight_count(candidate.weights.size());
                };
                CompiledModel cpu_int4_dspark = compiled;
                add_dspark_shared(cpu_int4_dspark, DType::Int4);
                CompiledModel gpu_int4_dspark = cpu_int4_dspark;
                prepare_hybrid_dspark(gpu_int4_dspark);
                const MoeBlockPlan& gpu_shared_plan = gpu_int4_dspark.speculative.net.layer_plans.front().moe;
                check(support_vulkan_shared_experts(gpu_int4_dspark.operators, gpu_shared_plan));
                check(speculative_shared_node(gpu_int4_dspark).backend == ExecutionBackend::Vulkan);

                ActivationBuffer shared_input(2, gpu_int4_dspark.descriptor.hidden_size);
                shared_input.row(0)[0] = 1.0f;
                shared_input.row(0)[1] = -0.25f;
                shared_input.row(1)[0] = 0.5f;
                shared_input.row(1)[1] = 0.75f;
                ActivationBuffer cpu_shared_output;
                ActivationBuffer gpu_shared_output;
                ExpertWorkspace cpu_shared_workspace;
                ExpertWorkspace gpu_shared_workspace;
                forward_shared_expert(cpu_int4_dspark, cpu_int4_dspark.speculative.net.layer_plans.front().moe,
                                      shared_input, cpu_shared_output, cpu_shared_workspace);
                const VulkanStatistics shared_before = get_vulkan_statistics(dspark_vulkan_runtime);
                forward_shared_expert(gpu_int4_dspark, gpu_shared_plan, shared_input, gpu_shared_output, gpu_shared_workspace);
                const VulkanStatistics shared_after = get_vulkan_statistics(dspark_vulkan_runtime);
                check(shared_after.compute_submissions > shared_before.compute_submissions);
                check(shared_after.int4_dispatches - shared_before.int4_dispatches == 3);
                check(gpu_shared_output.rows() == cpu_shared_output.rows());
                check(gpu_shared_output.columns() == cpu_shared_output.columns());
                check(std::any_of(gpu_shared_output.values().begin(), gpu_shared_output.values().end(),
                                  [](float value) { return std::isfinite(value) && value != 0.0f; }));
                for (size_t index = 0; index < cpu_shared_output.values().size(); ++index)
                {
                    check(std::isfinite(gpu_shared_output.values()[index]));
                    check_near(gpu_shared_output.values()[index], cpu_shared_output.values()[index], 1e-5f);
                }

                CompiledModel unsupported_dspark = compiled;
                add_dspark_shared(unsupported_dspark, DType::Float32);
                prepare_hybrid_dspark(unsupported_dspark);
                check(!support_vulkan_shared_experts(unsupported_dspark.operators,
                                                     unsupported_dspark.speculative.net.layer_plans.front().moe));
                check(speculative_shared_node(unsupported_dspark).backend == ExecutionBackend::Cpu);
            }
#endif

            // The two-token target alternates, so the final verification row
            // must supply the next token after each fully accepted block.
            GenerationOptions dspark_options = options;
            dspark_options.max_new_tokens = 7;
            auto dspark_session = runtime.create_session(model.value()).value();
            auto dspark_result = dspark_session->generate(input, dspark_options);
            check(static_cast<bool>(dspark_result));
            check(dspark_result.value().tokens.size() == dspark_options.max_new_tokens);
            check(dspark_session->statistics().speculative_proposals >= 2);
            check(dspark_session->statistics().speculative_accepted_tokens >= 2);
            for (size_t index = 0; index < dspark_result.value().tokens.size(); ++index)
                check(dspark_result.value().tokens[index].token_id == (index % 2 == 0 ? 1 : 0));

            SessionState proposal_state;
            proposal_state.speculative_layers.resize(compiled.speculative.layer_nodes.size());
            for (size_t layer_index = 0;
                 layer_index < proposal_state.speculative_layers.size();
                 ++layer_index)
            {
                const AttentionBlockPlan& attention = compiled.speculative.net.layer_plans[layer_index].attention;
                proposal_state.speculative_layers[layer_index].latent_window.assign(static_cast<size_t>(attention.sliding_window) * attention.head_dimension,
                                                                                    0.0f);
            }
            SessionStatistics proposal_statistics;
            std::vector<std::span<const float>> sampled_logits;
            const SpeculativeSampler fixed_sampler = [&sampled_logits](std::span<const float> logits) -> Result<int32_t> {
                sampled_logits.push_back(logits);
                return 1;
            };
            SpeculativeProposal confidence_proposal;
            auto confidence_status = propose_speculative(compiled, 0, proposal_statistics,
                                                         proposal_state, 0, confidence_proposal, fixed_sampler);
            if (!confidence_status)
                throw std::runtime_error("DSpark confidence proposal failed: "
                                         + confidence_status.error().message);
            check(confidence_proposal.confidence_logits.size() == compiled.speculative.block_size);
            check(sampled_logits.size() == confidence_proposal.logits.size());
            for (size_t row = 0; row < sampled_logits.size(); ++row)
            {
                const std::span<const float> retained = confidence_proposal.logits[row].values();
                check(sampled_logits[row].data() == retained.data());
                check(sampled_logits[row].size() == retained.size());
                check(std::equal(sampled_logits[row].begin(), sampled_logits[row].end(), retained.begin()));
            }
            const std::array<const std::byte*, 3> confidence_storage = {
                proposal_state.hyper_connection_scratch.auxiliary.bytes().data(),
                proposal_state.hyper_connection_scratch.normalized.bytes().data(),
                proposal_state.hyper_connection_scratch.projection.bytes().data(),
            };
            const std::array<uint64_t, 3> confidence_capacity = {
                proposal_state.hyper_connection_scratch.auxiliary.allocated_bytes(),
                proposal_state.hyper_connection_scratch.normalized.allocated_bytes(),
                proposal_state.hyper_connection_scratch.projection.allocated_bytes(),
            };
            check(confidence_capacity[2] >= 3 * sizeof(float));
            check(confidence_storage[0] != confidence_storage[1]);
            check(confidence_storage[0] != confidence_storage[2]);
            check(confidence_storage[1] != confidence_storage[2]);
            check(proposal_state.expert_scratch.staged_output.bytes().data()
                  != proposal_state.expert_scratch.staged_merged.bytes().data());
            check(confidence_storage[0] != proposal_state.expert_scratch.staged_output.bytes().data());
            check(confidence_storage[0] != proposal_state.expert_scratch.staged_merged.bytes().data());
            check(confidence_storage[1] != proposal_state.expert_scratch.staged_output.bytes().data());
            check(confidence_storage[1] != proposal_state.expert_scratch.staged_merged.bytes().data());
            check(confidence_storage[2] != proposal_state.expert_scratch.staged_output.bytes().data());
            check(confidence_storage[2] != proposal_state.expert_scratch.staged_merged.bytes().data());
            const SpeculativeProposal expected_confidence = confidence_proposal;
            const auto proposal_storage = sampled_logits;
            const auto* token_storage = confidence_proposal.token_ids.data();
            const auto* logits_storage = confidence_proposal.logits.data();
            const auto* confidence_data = confidence_proposal.confidence_logits.data();
            sampled_logits.clear();
            auto repeated_confidence_proposal = propose_speculative(compiled, 0, proposal_statistics,
                                                                    proposal_state, 0, confidence_proposal, fixed_sampler);
            if (!repeated_confidence_proposal)
                throw std::runtime_error("repeated DSpark confidence proposal failed: "
                                         + repeated_confidence_proposal.error().message);
            check(confidence_proposal.token_ids == expected_confidence.token_ids);
            check(sampled_logits.size() == confidence_proposal.logits.size());
            check(confidence_proposal.logits.size() == expected_confidence.logits.size());
            check(confidence_proposal.token_ids.data() == token_storage);
            check(confidence_proposal.logits.data() == logits_storage);
            check(confidence_proposal.confidence_logits.data() == confidence_data);
            for (size_t row = 0; row < sampled_logits.size(); ++row)
            {
                const std::span<const float> retained = confidence_proposal.logits[row].values();
                const std::span<const float> previous = expected_confidence.logits[row].values();
                check(retained.data() == proposal_storage[row].data());
                check(sampled_logits[row].data() == retained.data());
                check(sampled_logits[row].size() == retained.size());
                check(std::equal(sampled_logits[row].begin(), sampled_logits[row].end(), retained.begin()));
                check(retained.size() == previous.size());
                check(std::equal(retained.begin(), retained.end(), previous.begin()));
            }
            check(confidence_proposal.confidence_logits == expected_confidence.confidence_logits);
            check(proposal_state.hyper_connection_scratch.auxiliary.bytes().data() == confidence_storage[0]);
            check(proposal_state.hyper_connection_scratch.normalized.bytes().data() == confidence_storage[1]);
            check(proposal_state.hyper_connection_scratch.projection.bytes().data() == confidence_storage[2]);
            check(proposal_state.hyper_connection_scratch.auxiliary.allocated_bytes() == confidence_capacity[0]);
            check(proposal_state.hyper_connection_scratch.normalized.allocated_bytes() == confidence_capacity[1]);
            check(proposal_state.hyper_connection_scratch.projection.allocated_bytes() == confidence_capacity[2]);

            const auto saved_kind = compiled.speculative.kind;
            compiled.speculative.kind = SpeculativeModelKind::None;
            auto disabled_proposal = propose_speculative(compiled, 0, proposal_statistics,
                                                         proposal_state, 0, confidence_proposal, fixed_sampler);
            compiled.speculative.kind = saved_kind;
            check(!disabled_proposal && disabled_proposal.error().code == ErrorCode::UnsupportedModel);
            check(confidence_proposal.token_ids.empty() && confidence_proposal.confidence_logits.empty());
            check(confidence_proposal.committed_context_rows == 0);
            check(confidence_proposal.logits.data() == logits_storage);
            for (size_t row = 0; row < proposal_storage.size(); ++row)
                check(confidence_proposal.logits[row].values().data() == proposal_storage[row].data());

            GenerationOptions ordinary_options = dspark_options;
            ordinary_options.use_speculative = false;
            auto ordinary_session = runtime.create_session(model.value()).value();
            auto ordinary_result = ordinary_session->generate(input, ordinary_options);
            check(static_cast<bool>(ordinary_result));
            check(ordinary_result.value().tokens.size() == dspark_result.value().tokens.size());
            for (size_t index = 0; index < dspark_result.value().tokens.size(); ++index)
                check(dspark_result.value().tokens[index].token_id == ordinary_result.value().tokens[index].token_id);

            GenerationOptions empty_options = dspark_options;
            empty_options.speculative_confidence_threshold = 1.0f;
            auto empty_session = runtime.create_session(model.value()).value();
            auto empty_result = empty_session->generate(input, empty_options);
            check(static_cast<bool>(empty_result));
            check(empty_session->statistics().speculative_proposals == 1);
            check(empty_session->statistics().speculative_accepted_tokens == 0);
            check(empty_result.value().tokens.size() == ordinary_result.value().tokens.size());
            for (size_t index = 0; index < empty_result.value().tokens.size(); ++index)
                check(empty_result.value().tokens[index].token_id == ordinary_result.value().tokens[index].token_id);

            compiled.weights.at_mutable(latent.output_b_weight).float32_data = {1.0f, -0.25f};
            compiled.weights.at_mutable(compiled.speculative.markov_head_weight).float32_data = {0.0f, 0.0f};
            compiled.weights.at_mutable(compiled.lm_head_weight).float32_data = original_lm_head;
            auto dspark_decode = dspark_session->decode(0);
            auto ordinary_decode = ordinary_session->decode(0);
            check(static_cast<bool>(dspark_decode));
            check(static_cast<bool>(ordinary_decode));
            check(dspark_decode.value().logits.size() == ordinary_decode.value().logits.size());
            for (size_t column = 0; column < dspark_decode.value().logits.size(); ++column)
                check_near(dspark_decode.value().logits[column], ordinary_decode.value().logits[column], 1e-5f);
            compiled.weights.at_mutable(latent.output_b_weight).float32_data = {0.0f, 0.0f};
            compiled.weights.at_mutable(compiled.lm_head_weight).float32_data = dspark_lm_head;
            compiled.weights.at_mutable(compiled.speculative.markov_head_weight).float32_data = {-1.0f, 1.0f};

            compiled.weights.at_mutable(compiled.speculative.markov_head_weight).float32_data = {0.0f, -8.0f};
            GenerationOptions rejection_options = options;
            rejection_options.max_new_tokens = 5;
            auto rejection_session = runtime.create_session(model.value()).value();
            auto rejection_result = rejection_session->generate(input, rejection_options);
            check(static_cast<bool>(rejection_result));
            check(rejection_result.value().tokens.size() == rejection_options.max_new_tokens);
            check(rejection_session->statistics().speculative_proposals == 1);
            check(rejection_session->statistics().speculative_accepted_tokens == 0);
            for (size_t index = 0; index < rejection_result.value().tokens.size(); ++index)
                check(rejection_result.value().tokens[index].token_id == (index % 2 == 0 ? 1 : 0));

            ordinary_options.max_new_tokens = rejection_options.max_new_tokens;
            auto rejection_control = runtime.create_session(model.value()).value();
            auto rejection_control_result = rejection_control->generate(input, ordinary_options);
            check(static_cast<bool>(rejection_control_result));
            check(rejection_control_result.value().tokens.size() == rejection_result.value().tokens.size());
            for (size_t index = 0; index < rejection_result.value().tokens.size(); ++index)
                check(rejection_result.value().tokens[index].token_id == rejection_control_result.value().tokens[index].token_id);

            compiled.weights.at_mutable(latent.output_b_weight).float32_data = {1.0f, -0.25f};
            compiled.weights.at_mutable(compiled.speculative.markov_head_weight).float32_data = {0.0f, 0.0f};
            compiled.weights.at_mutable(compiled.lm_head_weight).float32_data = original_lm_head;
            auto rejection_decode = rejection_session->decode(0);
            auto rejection_control_decode = rejection_control->decode(0);
            check(static_cast<bool>(rejection_decode));
            check(static_cast<bool>(rejection_control_decode));
            check(rejection_decode.value().logits.size() == rejection_control_decode.value().logits.size());
            for (size_t column = 0; column < rejection_decode.value().logits.size(); ++column)
            {
                check_near(rejection_decode.value().logits[column],
                           rejection_control_decode.value().logits[column],
                           1e-5f);
            }
            compiled.weights.at_mutable(latent.output_b_weight).float32_data = {0.0f, 0.0f};
            compiled.weights.at_mutable(compiled.lm_head_weight).float32_data = dspark_lm_head;
            compiled.weights.at_mutable(compiled.speculative.markov_head_weight).float32_data = {-1.0f, 1.0f};
        }

        check_shared_head_host_release();

        for (int failure = 0; failure < 3; ++failure)
        {
            auto session = runtime.create_session(model.value()).value();
            const auto saved_bindings = compiled.speculative.layer_nodes;
            bool callback_reached = false;
            bool caught = false;
            try
            {
                auto result = session->generate(input, options, [&](const StreamToken& token) {
                    if (token.index == 1)
                    {
                        callback_reached = true;
                        if (failure == 0)
                        {
                            // Verification has completed; fail the context update after commit.
                            compiled.speculative.layer_nodes.clear();
                        }
                        else if (failure == 1)
                        {
                            throw std::runtime_error("speculative callback failed");
                        }
                        else
                        {
                            check(static_cast<bool>(session->reset()));
                        }
                    }
                    return true;
                });
                check(!result);
                check(result.error().message == (failure == 0 ? "speculative execution graph has an incomplete layer binding" : "generation callback modified the Session"));
            }
            catch (const std::runtime_error& error)
            {
                check(failure == 1);
                check(std::string(error.what()) == "speculative callback failed");
                caught = true;
            }
            compiled.speculative.layer_nodes = saved_bindings;
            check(callback_reached);
            check(caught == (failure == 1));
            if (failure == 0)
            {
                check_session_requires_reset(session, 1);
                check(static_cast<bool>(session->reset()));
            }
            auto control = runtime.create_session(model.value()).value();
            if (failure == 1)
                check(static_cast<bool>(control->prefill(input)));
            auto actual = session->decode(0);
            auto expected = control->decode(0);
            check(static_cast<bool>(actual));
            check(static_cast<bool>(expected));
            check(actual.value().logits == expected.value().logits);
        }
    }
}

void test_invalid_token_is_transactional()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));

    auto invalid = session.value()->decode(4);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));
    check(static_cast<bool>(session.value()->sequence_length() == 0));
    check(static_cast<bool>(session.value()->statistics().decode_tokens == 0));
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 0));

    const std::vector<int32_t> empty;
    auto empty_prefill = session.value()->prefill(empty);
    check(static_cast<bool>(!empty_prefill));
    check(static_cast<bool>(empty_prefill.error().code == ErrorCode::InvalidArgument));
}

void test_chunked_prefill_matches_single_batch()
{
    TemporaryModelPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));

    SessionOptions single_batch_options;
    single_batch_options.prefill_chunk_size = 0;
    auto single_batch_session = runtime.create_session(model.value(), single_batch_options);
    check(static_cast<bool>(single_batch_session));

    SessionOptions chunked_options;
    chunked_options.prefill_chunk_size = 2;
    auto chunked_session = runtime.create_session(model.value(), chunked_options);
    check(static_cast<bool>(chunked_session));

    const std::vector<int32_t> prompt = {0, 1, 2};
    auto single_batch = single_batch_session.value()->prefill(prompt);
    auto chunked = chunked_session.value()->prefill(prompt);
    check(static_cast<bool>(single_batch));
    check(static_cast<bool>(chunked));
    check(static_cast<bool>(chunked.value().processed_tokens == prompt.size()));
    check(static_cast<bool>(chunked_session.value()->sequence_length() == prompt.size()));
    check(static_cast<bool>(chunked_session.value()->statistics().prefill_tokens == prompt.size()));
    check(static_cast<bool>(chunked.value().logits.size() == single_batch.value().logits.size()));
    for (size_t index = 0; index < single_batch.value().logits.size(); ++index)
    {
        check_near(chunked.value().logits[index], single_batch.value().logits[index], 1e-5f);
    }
}

void test_topk_selected_weight_normalization_and_combine()
{
    WeightedTopKPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));

    auto result = session.value()->decode(0);
    check(static_cast<bool>(result));

    const float normalized_input = 1.0f / std::sqrt(0.5f + 1e-5f);
    const float expert_zero_weight = std::exp(normalized_input) / (std::exp(normalized_input) + 1.0f);
    const float expert_one_weight = 1.0f - expert_zero_weight;
    const float hidden_x = 1.0f + expert_zero_weight * normalized_input;
    const float hidden_y = expert_one_weight * normalized_input;
    const float final_scale = std::sqrt((hidden_x * hidden_x + hidden_y * hidden_y) / 2.0f + 1e-5f);

    check_near(result.value().logits[0], hidden_x / final_scale, 1e-5f);
    check_near(result.value().logits[1], hidden_y / final_scale, 1e-5f);
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 2));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 2));
    check(model.value()->descriptor().expert_count == 3);
    check(model.value()->descriptor().layers.front().moe.top_k == 2);
}

void test_int8_expert_linear()
{
    Int8ExpertPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    check(static_cast<bool>(model.value()->descriptor().layers[0].moe.expert_weight_dtype == DType::Int8));

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));
    auto result = session.value()->decode(0);
    check(static_cast<bool>(result));

    const float normalized_input = 1.0f / std::sqrt(0.5f + 1e-5f);
    const float hidden = 1.0f + normalized_input;
    const float expected = hidden / std::sqrt(hidden * hidden / 2.0f + 1e-5f);
    check_near(result.value().logits[0], expected, 1e-5f);
    check_near(result.value().logits[1], 0.0f, 1e-5f);
    check(static_cast<bool>(session.value()->statistics().expert_assignments == 1));
    check(static_cast<bool>(session.value()->statistics().expert_batches == 1));
}

void test_invalid_int8_scale_is_rejected()
{
    Int8ExpertPackage package(true);
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(!model));
    check(static_cast<bool>(model.error().code == ErrorCode::InvalidModel));
}

void test_attention_kv_cache_and_reset()
{
    AttentionPackage package;
    TestRuntime runtime;
    auto model = runtime.load_model(package.path());
    check(static_cast<bool>(model));
    check(static_cast<bool>(model.value()->descriptor().layers[0].attention.kind == AttentionKind::Standard));

    auto session = runtime.create_session(model.value());
    check(static_cast<bool>(session));
    const std::vector<int32_t> prompt = {0};
    auto prefill = session.value()->prefill(prompt);
    check(static_cast<bool>(prefill));
    auto cached_decode = session.value()->decode(1);
    check(static_cast<bool>(cached_decode));
    check(static_cast<bool>(cached_decode.value().logits[0] > 0.1f));

    check(static_cast<bool>(session.value()->reset()));
    auto uncached_decode = session.value()->decode(1);
    check(static_cast<bool>(uncached_decode));
    check_near(uncached_decode.value().logits[0], 0.0f, 1e-6f);
    for (size_t rows : {1u, 4u, 2u})
    {
        check(static_cast<bool>(session.value()->reset()));
        check(session.value()->sequence_length() == 0);
        check(session.value()->statistics().kv_cache_logical_size == 0);
        auto fresh = runtime.create_session(model.value());
        check(static_cast<bool>(fresh));
        const std::vector<int32_t> input(rows, 0);
        auto reused_prefill = session.value()->prefill(input);
        auto fresh_prefill = fresh.value()->prefill(input);
        if (!reused_prefill)
            throw std::runtime_error("reused Attention prefill failed for " + std::to_string(rows) + " rows: " + reused_prefill.error().message);
        if (!fresh_prefill)
            throw std::runtime_error("fresh Attention prefill failed for " + std::to_string(rows) + " rows: " + fresh_prefill.error().message);
        check(static_cast<bool>(reused_prefill));
        check(static_cast<bool>(fresh_prefill));
        auto reused_decode = session.value()->decode(1);
        auto fresh_decode = fresh.value()->decode(1);
        check(static_cast<bool>(reused_decode));
        check(static_cast<bool>(fresh_decode));
        check(reused_decode.value().logits.size() == fresh_decode.value().logits.size());
        for (size_t i = 0; i < fresh_decode.value().logits.size(); ++i)
            check_near(reused_decode.value().logits[i], fresh_decode.value().logits[i], 1e-6f);
    }
}

void test_bfloat16_ring_kv_cache()
{
    AttentionPackage float32_package;
    AttentionPackage bfloat16_package(true);
    TestRuntime runtime;
    Option cpu_options;
    cpu_options.hybrid_mode = HybridMode::CpuOnly;
    auto float32_model = runtime.load_model(float32_package.path(), cpu_options);
    auto bfloat16_model = runtime.load_model(bfloat16_package.path(), cpu_options);
    check(static_cast<bool>(float32_model));
    check(static_cast<bool>(bfloat16_model));
    check(static_cast<bool>(bfloat16_model.value()->descriptor().kv_cache_dtype == DType::BFloat16));

    auto float32_session = runtime.create_session(float32_model.value());
    auto bfloat16_session = runtime.create_session(bfloat16_model.value());
    check(static_cast<bool>(float32_session));
    check(static_cast<bool>(bfloat16_session));

    const std::vector<int32_t> prompt = {0, 1, 0, 1, 0, 1, 0, 1};
    check(static_cast<bool>(float32_session.value()->prefill(prompt)));
    check(static_cast<bool>(bfloat16_session.value()->prefill(prompt)));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_logical_size == float32_session.value()->statistics().kv_cache_logical_size / 2));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_allocated_size == float32_session.value()->statistics().kv_cache_allocated_size / 2));

    for (uint32_t index = 0; index < 16; ++index)
        check(static_cast<bool>(bfloat16_session.value()->decode(static_cast<int32_t>(index % 2))));
    check(static_cast<bool>(bfloat16_session.value()->statistics().kv_cache_allocated_size <= bfloat16_session.value()->statistics().kv_cache_logical_size * 16));
}

void test_generic_mtp_int4_cache_key()
{
    AttentionPackage package(false, 0, false, false);
    ModelPackage model_package;
    model_package.root = package.path();
    model_package.manifest.model_type = "test_moe";
    std::ifstream manifest_stream(package.path() / "config.json", std::ios::binary);
    model_package.manifest.raw_json = {std::istreambuf_iterator<char>(manifest_stream),
                                       std::istreambuf_iterator<char>()};
    FixtureModelAdapter adapter;
    auto parsed = adapter.parse_model(model_package);
    check(static_cast<bool>(parsed));
    auto mapping = adapter.map_weights(model_package, parsed.value());
    check(static_cast<bool>(mapping));

    MoeModelDescriptor descriptor = std::move(parsed).value();
    descriptor.speculative_kind = SpeculativeModelKind::Mtp;
    descriptor.speculative_layer_count = 1;
    descriptor.speculative_block_size = 2;
    MoeDescriptor& moe = descriptor.layers.back().moe;
    moe.activation = ExpertActivation::Silu;
    moe.expert_weight_dtype = DType::Int4;

    const auto make_int4 = [](std::vector<uint32_t> shape) {
        TensorData tensor;
        tensor.dtype = DType::Int4;
        tensor.shape = std::move(shape);
        tensor.quantization_group_size = tensor.shape[1];
        tensor.quantization_scales.assign(tensor.shape[0], 1.0f);
        tensor.quantization_zero_points.assign(tensor.shape[0], 8);
        const size_t row_bytes = (tensor.shape[1] + 1) / 2;
        tensor.quantized_data.assign(static_cast<size_t>(tensor.shape[0]) * row_bytes, 0x88);
        return tensor;
    };
    const auto add_float = [&mapping](std::string name, std::vector<uint32_t> shape, float value = 0.0f) {
        TensorData tensor;
        tensor.dtype = DType::Float32;
        tensor.shape = std::move(shape);
        tensor.float32_data.assign(static_cast<size_t>(tensor.shape[0]) * tensor.shape[1], value);
        mapping.value().emplace(std::move(name), std::move(tensor));
    };
    const auto add_float_vector = [&mapping](std::string name, uint32_t size, float value) {
        TensorData tensor;
        tensor.dtype = DType::Float32;
        tensor.shape = {size};
        tensor.float32_data.assign(size, value);
        mapping.value().emplace(std::move(name), std::move(tensor));
    };

    mapping.value()["layers.0.experts.0.up.weight"] = make_int4({2, 2});
    mapping.value()["layers.0.experts.0.down.weight"] = make_int4({2, 2});
    add_float_vector("speculative.mtp.embedding_norm.weight", 2, 1.0f);
    add_float_vector("speculative.mtp.hidden_norm.weight", 2, 1.0f);
    add_float("speculative.mtp.input_projection.weight", {2, 4});
    add_float_vector("speculative.final_norm.weight", 2, 1.0f);
    add_float_vector("speculative.layers.0.pre_attention_norm.weight", 2, 1.0f);
    for (const char* projection : {"query", "key", "value", "output"})
        add_float(std::string("speculative.layers.0.attention.") + projection + ".weight", {2, 2});
    mapping.value().at("speculative.mtp.input_projection.weight").float32_data = {
        0.5f, 0.25f, -0.5f, 0.125f,
        -0.25f, 0.5f, 0.25f, -0.5f};
    mapping.value().at("speculative.layers.0.attention.query.weight").float32_data = {
        1.0f, 0.25f, -0.5f, 0.75f};
    mapping.value().at("speculative.layers.0.attention.key.weight").float32_data = {
        0.5f, -0.25f, 0.75f, 0.5f};
    mapping.value().at("speculative.layers.0.attention.value.weight").float32_data = {
        0.75f, 0.25f, -0.125f, 0.5f};
    mapping.value().at("speculative.layers.0.attention.output.weight").float32_data = {
        1.0f, -0.125f, 0.25f, 0.75f};
    add_float_vector("speculative.layers.0.pre_ffn_norm.weight", 2, 1.0f);
    add_float("speculative.layers.0.router.weight", {1, 2});
    for (const char* projection : {"gate", "up", "down"})
        add_float(std::string("speculative.layers.0.shared_expert.") + projection + ".weight", {2, 2});
    mapping.value()["speculative.layers.0.experts.0.gate_up.weight"] = make_int4({4, 2});
    mapping.value()["speculative.layers.0.experts.0.down.weight"] = make_int4({2, 2});

    MoeModelDescriptor hybrid_descriptor = descriptor;
    WeightMapping hybrid_mapping = mapping.value();
    CompiledModel compiled;
    auto status = compile_model(std::move(descriptor), std::move(mapping).value(), compiled,
                                HybridMode::CpuOnly);
    if (!status)
        throw std::runtime_error("generic INT4 MTP compilation failed: " + status.error().message);
    const ExpertPlan& expert = compiled.speculative.net.layer_plans.back().moe.experts.front();
    check(expert.cache_key == "quantized:speculative.layers.0.experts.0.");
    check(support_vulkan_expert(expert,
                                compiled.weights.at(expert.gate_up_weight),
                                compiled.weights.at(expert.down_weight),
                                compiled.opt.optimization_flags));
    check(support_vulkan_experts(compiled.weights,
                                 compiled.speculative.net.layer_plans.back().moe,
                                 compiled.opt.optimization_flags));

    TestRuntime runtime;
    if (has_usable_vulkan_device(runtime.info(), RuntimeVulkanAttention))
    {
        CompilerOption hybrid_options;
        hybrid_options.flags |= BackendVulkanDense | BackendVulkanAttention;
        hybrid_options.device_index = get_default_gpu_index();
        hybrid_options.vulkan_runtime = create_vulkan_runtime();
        hybrid_options.optimization_flags = compiled.opt.optimization_flags;
        CompiledModel hybrid;
        auto hybrid_status = compile_model(std::move(hybrid_descriptor),
                                           std::move(hybrid_mapping),
                                           hybrid,
                                           HybridMode::HybridExperts,
                                           hybrid_options);
        if (!hybrid_status)
            throw std::runtime_error("generic INT4 MTP Vulkan attention compilation failed: "
                                     + hybrid_status.error().message);

        const AttentionBlockPlan& cpu_attention = compiled.speculative.net.layer_plans.front().attention;
        const AttentionBlockPlan& gpu_attention = hybrid.speculative.net.layer_plans.front().attention;
        check(cpu_attention.kind == AttentionKind::Standard);
        check(gpu_attention.kind == AttentionKind::Standard);
        check(gpu_attention.vulkan_attention_operator != invalid_compiled_operator_handle);
        check(static_cast<bool>(hybrid.operators.at(gpu_attention.vulkan_attention_operator).attention));
        check(support_vulkan_attention(hybrid.operators, gpu_attention));
        check(hybrid.speculative.layer_nodes.size() == 1);
        const uint32_t attention_node_id = hybrid.speculative.layer_nodes.front().attention;
        check(attention_node_id < hybrid.speculative.net.layers.size());
        const Layer& attention_node = hybrid.speculative.net.layers[attention_node_id];
        check(attention_node.kind == LayerKind::Attention);
        check(attention_node.backend == ExecutionBackend::Vulkan);

        SessionState cpu_state;
        SessionState gpu_state;
        cpu_state.speculative_layers.resize(1);
        gpu_state.speculative_layers.resize(1);
        const uint32_t hidden_size = hybrid.descriptor.hidden_size;
        ActivationBuffer cold_context(1, hidden_size);
        cold_context.row(0)[0] = 0.75f;
        cold_context.row(0)[1] = -0.25f;
        ActivationBuffer prefix_context(2, hidden_size);
        prefix_context.row(0)[0] = -0.5f;
        prefix_context.row(0)[1] = 0.375f;
        prefix_context.row(1)[0] = 0.125f;
        prefix_context.row(1)[1] = 0.625f;

        LayerCache& gpu_cache = gpu_state.speculative_layers.front();
        constexpr uint64_t prefix_tokens = 3;
        const std::array<std::pair<const ActivationBuffer*, uint64_t>, 2> contexts = {{{&cold_context, 0}, {&prefix_context, 1}}};
        const auto append_context = [](const CompiledModel& model,
                                       ExecutionBackend backend,
                                       SessionState& state,
                                       const ActivationBuffer& context,
                                       uint64_t position) {
            return append_attention_context(model.weights,
                                            model.operators,
                                            model.speculative.net.layer_plans.front().attention,
                                            backend,
                                            position,
                                            state.speculative_layers.front(),
                                            state.attention_scratch,
                                            context,
                                            model.opt.optimization_flags);
        };
        for (const auto& [context, position] : contexts)
        {
            check(static_cast<bool>(append_context(compiled, ExecutionBackend::Cpu,
                                                   cpu_state, *context, position)));
            check(static_cast<bool>(append_context(hybrid, ExecutionBackend::Vulkan,
                                                   gpu_state, *context, position)));
            check(gpu_cache.vulkan_attention_cache != nullptr);
            check(gpu_cache.token_count == position + context->rows());
            check(gpu_cache.keys.empty() && gpu_cache.values.empty());
            check(gpu_cache.bfloat16_keys.empty() && gpu_cache.bfloat16_values.empty());
        }
        const auto retained_device_cache = gpu_cache.vulkan_attention_cache;
        check(static_cast<bool>(retained_device_cache));
        check(gpu_cache.token_count == prefix_tokens);
        check(gpu_cache.capacity_tokens >= prefix_tokens);
        check(gpu_cache.keys.empty() && gpu_cache.values.empty());
        check(gpu_cache.bfloat16_keys.empty() && gpu_cache.bfloat16_values.empty());
        check(!gpu_cache.vulkan_attention_state_unknown);
        check(gpu_state.attention_scratch.output.rows() == prefix_context.rows());
        check(gpu_state.attention_scratch.output.columns() == hidden_size);

        for (SessionState* state : {&cpu_state, &gpu_state})
        {
            state->mtp_pending_target_hidden.reset(1, hidden_size, false);
            state->mtp_pending_target_hidden.row(0)[0] = 0.375f;
            state->mtp_pending_target_hidden.row(0)[1] = -0.125f;
            state->mtp_pending_target_position = prefix_tokens;
        }
        check(static_cast<bool>(begin_state_cache_transaction(cpu_state.speculative_layers,
                                                              hybrid.speculative.block_size)));
        check(static_cast<bool>(begin_state_cache_transaction(gpu_state.speculative_layers,
                                                              hybrid.speculative.block_size)));
        SessionStatistics cpu_statistics;
        SessionStatistics gpu_statistics;
        SpeculativeProposal cpu_proposal;
        SpeculativeProposal gpu_proposal;
        const SpeculativeSampler first_token = [](std::span<const float>) -> Result<int32_t> {
            return 0;
        };
        auto cpu_proposed = propose_speculative(compiled,
                                                1,
                                                cpu_statistics,
                                                cpu_state,
                                                prefix_tokens + 1,
                                                cpu_proposal,
                                                first_token);
        auto gpu_proposed = propose_speculative(hybrid,
                                                1,
                                                gpu_statistics,
                                                gpu_state,
                                                prefix_tokens + 1,
                                                gpu_proposal,
                                                first_token);
        check(static_cast<bool>(cpu_proposed));
        check(static_cast<bool>(gpu_proposed));
        check(gpu_statistics.vulkan_attention_blocks > 0);
        check(gpu_statistics.vulkan_attention_cpu_fallbacks == 0);
        check(cpu_proposal.token_ids == gpu_proposal.token_ids);
        check(cpu_proposal.confidence_logits == gpu_proposal.confidence_logits);
        check(cpu_proposal.logits.size() == gpu_proposal.logits.size());
        check(cpu_state.speculative_layers.front().transaction.recorded_rows
              == hybrid.speculative.block_size);
        check(gpu_cache.transaction.recorded_rows == hybrid.speculative.block_size);
        for (size_t row = 0; row < cpu_proposal.logits.size(); ++row)
        {
            check(cpu_proposal.logits[row].values().size()
                  == gpu_proposal.logits[row].values().size());
            for (size_t column = 0; column < cpu_proposal.logits[row].values().size(); ++column)
                check_near(cpu_proposal.logits[row].values()[column],
                           gpu_proposal.logits[row].values()[column],
                           1e-4f);
        }

        check(static_cast<bool>(finish_state_cache_transaction(cpu_state.speculative_layers, 0)));
        check(static_cast<bool>(finish_state_cache_transaction(gpu_state.speculative_layers, 0)));
        check(cpu_state.speculative_layers.front().token_count == prefix_tokens);
        check(gpu_cache.token_count == prefix_tokens);
        check(gpu_cache.vulkan_attention_cache == retained_device_cache);
        check(gpu_cache.keys.empty() && gpu_cache.values.empty());
        check(gpu_cache.bfloat16_keys.empty() && gpu_cache.bfloat16_values.empty());
        check(!gpu_cache.vulkan_attention_state_unknown);
    }
}

} // namespace moe
} // namespace ncnn
