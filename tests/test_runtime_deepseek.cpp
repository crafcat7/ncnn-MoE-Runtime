#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

void test_expert_dispatcher_groups_routes()
{
    ExpertDispatchOptions options;
    options.expert_count = 6;
    options.top_k = 1;
    const std::vector<float> logits = {
        0.0f,
        5.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        5.0f,
        0.0f,
        4.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        4.0f,
    };
    ExpertDispatchPlan dispatch;
    auto dispatch_status = forward_router(logits, 4, options, dispatch);
    check(static_cast<bool>(dispatch_status));
    check(static_cast<bool>(dispatch.assignment_count == 4));
    check(static_cast<bool>(dispatch.batches.size() == 2));
    check(static_cast<bool>(dispatch.batches[0].expert_id == 1));
    check(static_cast<bool>(dispatch.batches[0].routes.size() == 2));
    check(static_cast<bool>(dispatch.batches[0].routes[0].token_index == 0));
    check(static_cast<bool>(dispatch.batches[0].routes[1].token_index == 2));
    check_near(dispatch.batches[0].routes[0].weight, 1.0f, 1e-6f);
    check(static_cast<bool>(dispatch.batches[1].expert_id == 5));
    check(static_cast<bool>(dispatch.batches[1].routes.size() == 2));
    check(static_cast<bool>(dispatch.batches[1].routes[0].token_index == 1));
    check(static_cast<bool>(dispatch.batches[1].routes[1].token_index == 3));

    options.expert_count = 3;
    options.top_k = 2;
    const std::vector<float> weighted_logits = {2.0f, 1.0f, 0.0f};
    ExpertDispatchPlan weighted;
    auto weighted_status = forward_router(weighted_logits, 1, options, weighted);
    check(static_cast<bool>(weighted_status));
    check(static_cast<bool>(weighted.assignment_count == 2));
    check(static_cast<bool>(weighted.batches.size() == 2));
    check(weighted.batches[0].routes[0].rank == 0);
    check(weighted.batches[1].routes[0].rank == 1);
    check_near(weighted.batches[0].routes[0].weight + weighted.batches[1].routes[0].weight, 1.0f, 1e-6f);
    const std::array<float, 6> repeated_weighted_logits = {
        2.0f,
        1.0f,
        0.0f,
        2.0f,
        1.0f,
        0.0f,
    };
    ExpertDispatchPlan weighted_general;
    auto weighted_general_status = forward_router(repeated_weighted_logits, 2, options, weighted_general);
    check(static_cast<bool>(weighted_general_status));
    check(weighted_general.batches.size() == weighted.batches.size());
    for (size_t batch_index = 0; batch_index < weighted.batches.size(); ++batch_index)
    {
        check(weighted_general.batches[batch_index].routes.size() == 2);
        check(weighted_general.batches[batch_index].routes[0].token_index == 0);
        check_near(weighted.batches[batch_index].routes[0].weight,
                   weighted_general.batches[batch_index].routes[0].weight,
                   1e-6f);
    }
    ExpertDispatchPlan reusable;
    auto dispatched_into = forward_router(weighted_logits, 1, options, reusable);
    check(static_cast<bool>(dispatched_into));
    check(static_cast<bool>(reusable.assignment_count == weighted.assignment_count));
    check(static_cast<bool>(reusable.batches.size() == weighted.batches.size()));
    for (size_t index = 0; index < reusable.batches.size(); ++index)
    {
        check(static_cast<bool>(reusable.batches[index].expert_id == weighted.batches[index].expert_id));
        check(reusable.batches[index].routes.front().rank == weighted.batches[index].routes.front().rank);
        check_near(reusable.batches[index].routes.front().weight, weighted.batches[index].routes.front().weight, 1e-6f);
    }
    const ExpertBatch* reused_batches = reusable.batches.data();
    const ExpertRoute* reused_first_route = reusable.batches.front().routes.data();
    const std::vector<float> next_logits = {0.0f, 1.0f, 2.0f};
    dispatched_into = forward_router(next_logits, 1, options, reusable);
    check(static_cast<bool>(dispatched_into));
    check(static_cast<bool>(reusable.batches.data() == reused_batches));
    check(static_cast<bool>(reusable.batches.front().routes.data() == reused_first_route));

    ExpertDispatchPlan handoff_plan;
    auto first_handoff = forward_router(weighted_logits, 1, options, handoff_plan);
    check(static_cast<bool>(first_handoff));
    check(handoff_plan.batches.size() == 2);
    ExpertBatch& first_handoff_batch = handoff_plan.batches.front();
    const uint32_t first_expert_id = first_handoff_batch.expert_id;
    const ExpertRoute first_route = first_handoff_batch.routes.front();
    const ExpertRoute* first_dispatch_data = first_handoff_batch.routes.data();

    ExpertState handoff_active;
    handoff_active.batch.routes.reserve(8);
    handoff_active.batch.routes.push_back({99, 99, 99.0f});
    const size_t previous_active_capacity = handoff_active.batch.routes.capacity();
    handoff_active.prepare(first_handoff_batch);
    check(handoff_active.batch.expert_id == first_expert_id);
    check(handoff_active.batch.routes.size() == 1);
    check(handoff_active.batch.routes.data() == first_dispatch_data);
    check(handoff_active.batch.routes.front().token_index == first_route.token_index);
    check(handoff_active.batch.routes.front().rank == first_route.rank);
    check_near(handoff_active.batch.routes.front().weight, first_route.weight, 1e-6f);
    check(first_handoff_batch.routes.empty());
    check(first_handoff_batch.routes.capacity() == previous_active_capacity);

    const size_t recycled_dispatch_capacity = first_handoff_batch.routes.capacity();
    auto second_handoff = forward_router(next_logits, 1, options, handoff_plan);
    check(static_cast<bool>(second_handoff));
    check(handoff_plan.batches.size() == 2);
    ExpertBatch& second_handoff_batch = handoff_plan.batches.front();
    check(second_handoff_batch.routes.capacity() == recycled_dispatch_capacity);
    const uint32_t second_expert_id = second_handoff_batch.expert_id;
    const ExpertRoute second_route = second_handoff_batch.routes.front();
    const ExpertRoute* second_dispatch_data = second_handoff_batch.routes.data();
    const size_t second_previous_active_capacity = handoff_active.batch.routes.capacity();
    handoff_active.cache_wait_us = 1;
    handoff_active.prepare(second_handoff_batch);
    check(handoff_active.batch.expert_id == second_expert_id);
    check(handoff_active.batch.routes.size() == 1);
    check(handoff_active.batch.routes.data() == second_dispatch_data);
    check(handoff_active.batch.routes.front().token_index == second_route.token_index);
    check(handoff_active.batch.routes.front().rank == second_route.rank);
    check_near(handoff_active.batch.routes.front().weight, second_route.weight, 1e-6f);
    check(second_handoff_batch.routes.empty());
    check(second_handoff_batch.routes.capacity() == second_previous_active_capacity);
    check(handoff_active.cache_wait_us == 0);

    const std::vector<float> invalid_logits = {1.0f, 2.0f};
    ExpertDispatchPlan invalid_plan;
    auto invalid = forward_router(invalid_logits, 1, options, invalid_plan);
    check(static_cast<bool>(!invalid));
    check(static_cast<bool>(invalid.error().code == ErrorCode::InvalidArgument));

    // A two-token call always uses the general path. Compare its first row
    // with the single-token call so fast-path coverage has an independent oracle.
    const auto check_dispatch_matches_general = [&](std::span<const float> test_logits, const ExpertDispatchOptions& test_options) {
        std::vector<float> repeated_logits(test_logits.begin(), test_logits.end());
        repeated_logits.insert(repeated_logits.end(), test_logits.begin(), test_logits.end());
        ExpertDispatchOptions general_options = test_options;
        std::vector<uint32_t> repeated_expert_ids;
        if (!test_options.explicit_expert_ids.empty())
        {
            repeated_expert_ids.assign(test_options.explicit_expert_ids.begin(), test_options.explicit_expert_ids.end());
            repeated_expert_ids.insert(repeated_expert_ids.end(), test_options.explicit_expert_ids.begin(), test_options.explicit_expert_ids.end());
            general_options.explicit_expert_ids = repeated_expert_ids;
        }
        ExpertDispatchPlan expected;
        auto expected_status = forward_router(repeated_logits, 2, general_options, expected);
        check(static_cast<bool>(expected_status));
        ExpertDispatchPlan actual;
        auto status = forward_router(test_logits, 1, test_options, actual);
        check(static_cast<bool>(status));
        check(actual.assignment_count * 2 == expected.assignment_count);
        check(actual.batches.size() == expected.batches.size());
        for (size_t batch_index = 0; batch_index < actual.batches.size(); ++batch_index)
        {
            const ExpertBatch& expected_batch = expected.batches[batch_index];
            const ExpertBatch& actual_batch = actual.batches[batch_index];
            std::vector<ExpertRoute> expected_routes;
            for (const ExpertRoute& route : expected_batch.routes)
            {
                if (route.token_index == 0)
                    expected_routes.push_back(route);
            }
            check(actual_batch.expert_id == expected_batch.expert_id);
            check(actual_batch.routes.size() == expected_routes.size());
            for (size_t route_index = 0; route_index < actual_batch.routes.size(); ++route_index)
            {
                check(actual_batch.routes[route_index].token_index == expected_routes[route_index].token_index);
                check(actual_batch.routes[route_index].rank == expected_routes[route_index].rank);
                check_near(actual_batch.routes[route_index].weight, expected_routes[route_index].weight, 1e-6f);
            }
        }
    };
    ExpertDispatchOptions sigmoid_options;
    sigmoid_options.expert_count = 4;
    sigmoid_options.top_k = 2;
    sigmoid_options.score_function = RouterScoreFunction::Sigmoid;
    sigmoid_options.normalization = RouterNormalization::None;
    sigmoid_options.routed_scaling_factor = 2.0f;
    const std::array<float, 4> tied_sigmoid_logits = {0.0f, 0.0f, -1.0f, -1.0f};
    check_dispatch_matches_general(tied_sigmoid_logits, sigmoid_options);

    // The selected scores must not already sum to one when checking normalization.
    const std::array<float, 4> sigmoid_logits = {2.0f, 1.0f, 0.0f, -1.0f};
    const float first_score = 1.0f / (1.0f + std::exp(-2.0f));
    const float second_score = 1.0f / (1.0f + std::exp(-1.0f));
    for (RouterNormalization normalization : {RouterNormalization::None, RouterNormalization::SelectedExperts})
    {
        sigmoid_options.normalization = normalization;
        ExpertDispatchPlan routed;
        auto routed_status = forward_router(sigmoid_logits, 1, sigmoid_options, routed);
        check(static_cast<bool>(routed_status));
        check(routed.batches.size() == 2);
        check(routed.batches[0].expert_id == 0);
        check(routed.batches[1].expert_id == 1);
        const float scale = sigmoid_options.routed_scaling_factor
                            / (normalization == RouterNormalization::SelectedExperts ? first_score + second_score : 1.0f);
        check_near(routed.batches[0].routes[0].weight, first_score * scale, 1e-4f);
        check_near(routed.batches[1].routes[0].weight, second_score * scale, 1e-4f);
        check_dispatch_matches_general(sigmoid_logits, sigmoid_options);
    }

    ExpertDispatchOptions wide_options;
    wide_options.expert_count = 20;
    std::array<float, 20> wide_logits;
    for (size_t index = 0; index < wide_logits.size(); ++index)
        wide_logits[index] = static_cast<float>(static_cast<int>(index % 7) - 3);
    wide_options.top_k = 16;
    check_dispatch_matches_general(wide_logits, wide_options);
    wide_options.top_k = 17;
    check_dispatch_matches_general(wide_logits, wide_options);

    // Distinct logits can map to the same approximate softmax score after
    // underflow; routing must still select them by their raw logit order.
    ExpertDispatchOptions softmax_options;
    softmax_options.expert_count = 3;
    softmax_options.top_k = 2;
    const std::array<float, 3> underflow_logits = {0.0f, -120.0f, -110.0f};
    check(float_approximate_exp(underflow_logits[1]) == float_approximate_exp(underflow_logits[2]));
    check(float_approximate_exp(underflow_logits[1]) == 0.0f);
    ExpertDispatchPlan underflow_fast;
    auto underflow_fast_status = forward_router(underflow_logits, 1, softmax_options, underflow_fast);
    check(static_cast<bool>(underflow_fast_status));
    check(underflow_fast.batches.size() == 2);
    check(underflow_fast.batches[0].expert_id == 0 && underflow_fast.batches[1].expert_id == 2);
    check(underflow_fast.batches[0].routes.front().rank == 0);
    check(underflow_fast.batches[1].routes.front().rank == 1);
    check(underflow_fast.batches[0].routes.front().weight == 1.0f);
    check(underflow_fast.batches[1].routes.front().weight == 0.0f);

    const std::array<float, 6> underflow_batch_logits = {0.0f, -120.0f, -110.0f,
                                                         0.0f, -111.0f, -109.0f};
    ExpertDispatchPlan underflow_general;
    underflow_general.scores = {13.0f, 17.0f, 19.0f};
    const float* preserved_score_data = underflow_general.scores.data();
    auto underflow_general_status = forward_router(underflow_batch_logits, 2, softmax_options, underflow_general);
    check(static_cast<bool>(underflow_general_status));
    check(underflow_general.batches.size() == 2);
    check(underflow_general.batches[0].expert_id == 0 && underflow_general.batches[1].expert_id == 2);
    check(underflow_general.scores.data() == preserved_score_data);
    check(underflow_general.scores.size() == 3);
    check(underflow_general.scores[0] == 13.0f && underflow_general.scores[1] == 17.0f
          && underflow_general.scores[2] == 19.0f);
    for (const ExpertBatch& batch : underflow_general.batches)
    {
        check(batch.routes.size() == 2);
        check(batch.routes[0].token_index == 0 && batch.routes[1].token_index == 1);
        check(batch.routes[0].rank == (batch.expert_id == 0 ? 0u : 1u));
        check(batch.routes[1].rank == (batch.expert_id == 0 ? 0u : 1u));
        check(batch.routes[0].weight == (batch.expert_id == 0 ? 1.0f : 0.0f));
        check(batch.routes[1].weight == (batch.expert_id == 0 ? 1.0f : 0.0f));
    }

    // Exact raw-logit ties keep the established lower-expert-id ordering.
    softmax_options.expert_count = 4;
    const std::array<float, 4> tied_softmax_logits = {0.0f, 2.0f, 2.0f, -1.0f};
    ExpertDispatchPlan tied_softmax;
    auto tied_softmax_status = forward_router(tied_softmax_logits, 1, softmax_options, tied_softmax);
    check(static_cast<bool>(tied_softmax_status));
    check(tied_softmax.batches.size() == 2);
    check(tied_softmax.batches[0].expert_id == 1 && tied_softmax.batches[1].expert_id == 2);
    check(tied_softmax.batches[0].routes.front().rank == 0);
    check(tied_softmax.batches[1].routes.front().rank == 1);
    check_near(tied_softmax.batches[0].routes.front().weight, 0.5f, 1e-6f);
    check_near(tied_softmax.batches[1].routes.front().weight, 0.5f, 1e-6f);

    // Top-K above the stack limit uses the multi-token general path while
    // retaining the same raw ordering and leaving the score scratch untouched.
    std::array<float, 40> wide_batch_logits{};
    for (size_t expert_id = 0; expert_id < 20; ++expert_id)
    {
        wide_batch_logits[expert_id] = static_cast<float>(expert_id);
        wide_batch_logits[20 + expert_id] = static_cast<float>(19 - expert_id);
    }
    ExpertDispatchPlan wide_batch;
    wide_batch.scores = {23.0f, 29.0f};
    const float* wide_score_data = wide_batch.scores.data();
    auto wide_batch_status = forward_router(wide_batch_logits, 2, wide_options, wide_batch);
    check(static_cast<bool>(wide_batch_status));
    check(wide_batch.assignment_count == 34);
    check(wide_batch.batches.size() == 20);
    check(wide_batch.scores.data() == wide_score_data);
    check(wide_batch.scores.size() == 2);
    check(wide_batch.scores[0] == 23.0f && wide_batch.scores[1] == 29.0f);
    float wide_denominator = 0.0f;
    for (int score = 3; score <= 19; ++score)
        wide_denominator += float_approximate_exp(static_cast<float>(score - 19));
    for (uint32_t expert_id = 0; expert_id < 20; ++expert_id)
    {
        const ExpertBatch& batch = wide_batch.batches[expert_id];
        check(batch.expert_id == expert_id);
        const bool selected_by_first = expert_id >= 3;
        const bool selected_by_second = expert_id <= 16;
        check(batch.routes.size() == static_cast<size_t>(selected_by_first + selected_by_second));
        size_t route_index = 0;
        if (selected_by_first)
        {
            const ExpertRoute& route = batch.routes[route_index++];
            check(route.token_index == 0 && route.rank == 19 - expert_id);
            check_near(route.weight,
                       float_approximate_exp(static_cast<float>(expert_id) - 19.0f) / wide_denominator,
                       1e-6f);
        }
        if (selected_by_second)
        {
            const ExpertRoute& route = batch.routes[route_index];
            check(route.token_index == 1 && route.rank == expert_id);
            check_near(route.weight,
                       float_approximate_exp(static_cast<float>(-static_cast<int>(expert_id))) / wide_denominator,
                       1e-6f);
        }
    }

    // Selection bias still ranks transformed scores. None publishes the
    // original probability; SelectedExperts renormalizes that selected score.
    ExpertDispatchOptions biased_softmax;
    biased_softmax.expert_count = 3;
    biased_softmax.top_k = 1;
    const std::array<float, 3> softmax_bias = {0.0f, 2.0f, 0.0f};
    biased_softmax.selection_bias = softmax_bias;
    const std::array<float, 3> biased_logits = {3.0f, 2.0f, 0.0f};
    const float bias_denominator = float_approximate_exp(0.0f) + float_approximate_exp(-1.0f)
                                   + float_approximate_exp(-3.0f);
    for (RouterNormalization normalization : {RouterNormalization::None, RouterNormalization::SelectedExperts})
    {
        biased_softmax.normalization = normalization;
        ExpertDispatchPlan biased_result;
        auto biased_status = forward_router(biased_logits, 1, biased_softmax, biased_result);
        check(static_cast<bool>(biased_status));
        check(biased_result.batches.size() == 1 && biased_result.batches[0].expert_id == 1);
        const float expected_weight = normalization == RouterNormalization::None
                                          ? float_approximate_exp(-1.0f) / bias_denominator
                                          : 1.0f;
        check_near(biased_result.batches[0].routes.front().weight, expected_weight, 1e-6f);
    }

    // Explicit Softmax IDs stay on the fallback path even when their raw score
    // is outside top-K and their selected weight underflows to zero.
    ExpertDispatchOptions explicit_underflow_options;
    explicit_underflow_options.expert_count = 2;
    explicit_underflow_options.top_k = 1;
    const std::array<uint32_t, 1> underflow_expert_ids = {1};
    explicit_underflow_options.explicit_expert_ids = underflow_expert_ids;
    const std::array<float, 2> explicit_underflow_logits = {0.0f, -120.0f};
    ExpertDispatchPlan explicit_underflow;
    auto explicit_underflow_status = forward_router(explicit_underflow_logits, 1, explicit_underflow_options,
                                                    explicit_underflow);
    check(!explicit_underflow_status);
    check(explicit_underflow_status.error().code == ErrorCode::InvalidModel);
    check(explicit_underflow_status.error().message == "selected router weights have a non-positive sum");

    const std::array<uint32_t, 2> batch_underflow_expert_ids = {1, 1};
    explicit_underflow_options.explicit_expert_ids = batch_underflow_expert_ids;
    const std::array<float, 4> batch_underflow_logits = {0.0f, -120.0f, 0.0f, -120.0f};
    ExpertDispatchPlan batch_underflow;
    auto batch_underflow_status = forward_router(batch_underflow_logits, 2, explicit_underflow_options,
                                                 batch_underflow);
    check(!batch_underflow_status);
    check(batch_underflow_status.error().code == ErrorCode::InvalidModel);
    check(batch_underflow_status.error().message == "selected router weights have a non-positive sum");

    ExpertDispatchOptions duplicate_options;
    duplicate_options.expert_count = 4;
    duplicate_options.top_k = 2;
    const std::array<uint32_t, 2> duplicate_experts = {1, 1};
    duplicate_options.explicit_expert_ids = duplicate_experts;
    const std::array<float, 4> duplicate_logits = {0.0f, 1.0f, 2.0f, 3.0f};
    check_dispatch_matches_general(duplicate_logits, duplicate_options);

    const auto check_invalid_normalization = [&](std::span<const float> test_logits, uint32_t token_count, ExpertDispatchOptions invalid_options) {
        invalid_options.normalization = static_cast<RouterNormalization>(-1);
        ExpertDispatchPlan rejected_plan;
        auto rejected = forward_router(test_logits, token_count, invalid_options, rejected_plan);
        check(!rejected);
        check(rejected.error().code == ErrorCode::InvalidArgument);
        check(rejected.error().message == "router normalization must be None or SelectedExperts");

        ExpertDispatchPlan preserved = weighted;
        const ExpertBatch* preserved_batches = preserved.batches.data();
        const ExpertRoute* preserved_routes = preserved.batches.front().routes.data();
        auto rejected_reuse = forward_router(test_logits, token_count, invalid_options, preserved);
        check(!rejected_reuse);
        check(rejected_reuse.error().code == ErrorCode::InvalidArgument);
        check(rejected_reuse.error().message == "router normalization must be None or SelectedExperts");
        check(preserved.batches.data() == preserved_batches);
        check(preserved.batches.front().routes.data() == preserved_routes);
        check(preserved.assignment_count == weighted.assignment_count);
        check(preserved.batches.size() == weighted.batches.size());
        for (size_t index = 0; index < preserved.batches.size(); ++index)
        {
            const ExpertBatch& expected = weighted.batches[index];
            check(preserved.batches[index].expert_id == expected.expert_id);
            check(preserved.batches[index].routes.size() == expected.routes.size());
            check(preserved.batches[index].routes.front().token_index == expected.routes.front().token_index);
            check(preserved.batches[index].routes.front().rank == expected.routes.front().rank);
            check(preserved.batches[index].routes.front().weight == expected.routes.front().weight);
        }

        invalid_options.routed_scaling_factor = 0.0f;
        rejected = forward_router(test_logits, token_count, invalid_options, rejected_plan);
        check(!rejected);
        check(rejected.error().message == "routed scaling factor must be finite and positive");
        rejected_reuse = forward_router(test_logits, token_count, invalid_options, preserved);
        check(!rejected_reuse);
        check(rejected_reuse.error().message == "routed scaling factor must be finite and positive");
    };
    check_invalid_normalization(weighted_logits, 1, options);
    ExpertDispatchOptions batch_options;
    batch_options.expert_count = 6;
    batch_options.top_k = 1;
    check_invalid_normalization(logits, 4, batch_options);
    check_invalid_normalization(wide_logits, 1, wide_options);
    check_invalid_normalization(duplicate_logits, 1, duplicate_options);

    // A late error in a reusable prefill plan must not publish partial routes.
    ExpertDispatchPlan preserved = dispatch;
    const ExpertBatch* preserved_batches = preserved.batches.data();
    std::array<const ExpertRoute*, 2> preserved_routes = {
        preserved.batches[0].routes.data(), preserved.batches[1].routes.data()};
    const auto check_preserved = [&] {
        check(preserved.batches.data() == preserved_batches);
        check(preserved.assignment_count == dispatch.assignment_count);
        check(preserved.batches.size() == 2);
        for (size_t index = 0; index < 2; ++index)
        {
            const ExpertBatch& expected = dispatch.batches[index];
            const ExpertBatch& actual = preserved.batches[index];
            check(actual.expert_id == expected.expert_id);
            check(actual.routes.data() == preserved_routes[index]);
            check(actual.routes.size() == expected.routes.size());
            for (size_t route = 0; route < actual.routes.size(); ++route)
            {
                check(actual.routes[route].token_index == expected.routes[route].token_index);
                check(actual.routes[route].rank == expected.routes[route].rank);
                check(actual.routes[route].weight == expected.routes[route].weight);
            }
        }
    };
    const std::array<uint32_t, 4> invalid_explicit = {1, 5, 1, 6};
    batch_options.explicit_expert_ids = invalid_explicit;
    auto rejected = forward_router(logits, 4, batch_options, preserved);
    check(!rejected);
    check(rejected.error().message == "explicit expert id is out of range");
    check_preserved();
    batch_options.explicit_expert_ids = {};
    batch_options.score_function = RouterScoreFunction::Sigmoid;
    std::vector<float> zero_sum_logits = logits;
    std::fill(zero_sum_logits.end() - batch_options.expert_count, zero_sum_logits.end(), -1000.0f);
    rejected = forward_router(zero_sum_logits, 4, batch_options, preserved);
    check(!rejected);
    check(rejected.error().message == "selected router weights have a non-positive sum");
    check_preserved();
    batch_options.score_function = RouterScoreFunction::Softmax;
    check(static_cast<bool>(forward_router(logits, 4, batch_options, preserved)));
    check(preserved.assignment_count == 4);
    check(preserved.batches.size() == 2);
    check(preserved.batches[0].expert_id == 1 && preserved.batches[1].expert_id == 5);
    check(preserved.batches[0].routes.size() == 2 && preserved.batches[1].routes.size() == 2);

    // Warm both publication buffers, then keep swapping the same route storage.
    check(static_cast<bool>(forward_router(logits, 4, batch_options, preserved)));
    const ExpertBatch* batch_storage = preserved.batches.data();
    const std::vector<ExpertRoute>* scratch_storage = preserved.route_scratch.data();
    std::array<const ExpertRoute*, 2> output_storage = {
        preserved.batches[0].routes.data(), preserved.batches[1].routes.data()};
    std::array<const ExpertRoute*, 2> pending_storage = {
        preserved.route_scratch[1].data(), preserved.route_scratch[5].data()};
    for (size_t pass = 0; pass < 4; ++pass)
    {
        check(static_cast<bool>(forward_router(logits, 4, batch_options, preserved)));
        check(preserved.batches.data() == batch_storage);
        check(preserved.route_scratch.data() == scratch_storage);
        for (size_t index = 0; index < 2; ++index)
        {
            check(preserved.batches[index].routes.data() == pending_storage[index]);
            const uint32_t expert_id = preserved.batches[index].expert_id;
            check(preserved.route_scratch[expert_id].data() == output_storage[index]);
            std::swap(output_storage[index], pending_storage[index]);
        }
    }
    // Decode may shrink the published plan; prefill still retains its route slots.
    check(static_cast<bool>(forward_router(std::span<const float>(logits.data(), 6), 1, batch_options, preserved)));
    check(preserved.batches.size() == 1);
    check(preserved.route_scratch.data() == scratch_storage);
    check(preserved.route_scratch[5].capacity() >= 2);
    check(static_cast<bool>(forward_router(logits, 4, batch_options, preserved)));
    check(preserved.batches.size() == 2);
    check(preserved.batches[0].routes.size() == 2 && preserved.batches[1].routes.size() == 2);
}

void test_deepseek_router_and_hyper_connection_kernels()
{
    ExpertDispatchOptions options;
    options.expert_count = 4;
    options.top_k = 2;
    options.score_function = RouterScoreFunction::SqrtSoftplus;
    options.routed_scaling_factor = 1.5f;
    const std::array<float, 4> selection_bias = {0.0f, 100.0f, 0.0f, 0.0f};
    options.selection_bias = selection_bias;
    const std::array<float, 4> logits = {4.0f, -4.0f, 3.0f, 2.0f};
    ExpertDispatchPlan routed;
    auto routed_status = forward_router(logits, 1, options, routed);
    check(static_cast<bool>(routed_status));
    check(routed.batches.size() == 2);
    check(routed.batches[0].expert_id == 0);
    check(routed.batches[1].expert_id == 1);
    check_near(routed.batches[0].routes[0].weight + routed.batches[1].routes[0].weight, 1.5f, 1e-5f);
    std::vector<float> repeated_logits(logits.begin(), logits.end());
    repeated_logits.insert(repeated_logits.end(), logits.begin(), logits.end());
    ExpertDispatchPlan general_routed;
    auto general_routed_status = forward_router(repeated_logits, 2, options, general_routed);
    check(static_cast<bool>(general_routed_status));
    check(general_routed.batches.size() == routed.batches.size());
    for (size_t batch_index = 0; batch_index < routed.batches.size(); ++batch_index)
    {
        check(routed.batches[batch_index].expert_id == general_routed.batches[batch_index].expert_id);
        check(general_routed.batches[batch_index].routes.size() == 2);
        check(general_routed.batches[batch_index].routes[0].token_index == 0);
        check_near(routed.batches[batch_index].routes[0].weight,
                   general_routed.batches[batch_index].routes[0].weight,
                   1e-6f);
    }
    ExpertDispatchPlan reusable;
    auto routed_reuse = forward_router(logits, 1, options, reusable);
    check(static_cast<bool>(routed_reuse));
    check(reusable.assignment_count == routed.assignment_count);
    check(reusable.batches.size() == routed.batches.size());
    for (size_t index = 0; index < reusable.batches.size(); ++index)
    {
        check(reusable.batches[index].expert_id == routed.batches[index].expert_id);
        check_near(reusable.batches[index].routes[0].weight, routed.batches[index].routes[0].weight, 1e-6f);
    }
    const ExpertBatch* reusable_batches = reusable.batches.data();
    const ExpertRoute* reusable_route = reusable.batches.front().routes.data();
    routed_reuse = forward_router(logits, 1, options, reusable);
    check(static_cast<bool>(routed_reuse));
    check(reusable.batches.data() == reusable_batches);
    check(reusable.batches.front().routes.data() == reusable_route);

    const std::array<uint32_t, 2> explicit_experts = {3, 2};
    options.selection_bias = {};
    options.explicit_expert_ids = explicit_experts;
    routed_status = forward_router(logits, 1, options, routed);
    check(static_cast<bool>(routed_status));
    check(routed.batches[0].expert_id == 2);
    check(routed.batches[1].expert_id == 3);
    std::vector<uint32_t> repeated_expert_ids(explicit_experts.begin(), explicit_experts.end());
    repeated_expert_ids.insert(repeated_expert_ids.end(), explicit_experts.begin(), explicit_experts.end());
    ExpertDispatchOptions general_explicit_options = options;
    general_explicit_options.explicit_expert_ids = repeated_expert_ids;
    ExpertDispatchPlan general_explicit;
    auto general_explicit_status = forward_router(repeated_logits, 2, general_explicit_options, general_explicit);
    check(static_cast<bool>(general_explicit_status));
    check(general_explicit.batches.size() == routed.batches.size());
    for (size_t batch_index = 0; batch_index < routed.batches.size(); ++batch_index)
    {
        check(routed.batches[batch_index].expert_id == general_explicit.batches[batch_index].expert_id);
        check(general_explicit.batches[batch_index].routes.size() == 2);
        check(general_explicit.batches[batch_index].routes[0].token_index == 0);
        check(routed.batches[batch_index].routes[0].rank == general_explicit.batches[batch_index].routes[0].rank);
        check_near(routed.batches[batch_index].routes[0].weight,
                   general_explicit.batches[batch_index].routes[0].weight,
                   1e-6f);
    }
    routed_reuse = forward_router(logits, 1, options, reusable);
    check(static_cast<bool>(routed_reuse));
    check(reusable.batches[0].expert_id == 2);
    check(reusable.batches[1].expert_id == 3);

    // Reuse the same plan across full-score and explicit-score paths.
    for (RouterScoreFunction score_function : {RouterScoreFunction::Softmax, RouterScoreFunction::Sigmoid, RouterScoreFunction::SqrtSoftplus})
    {
        for (RouterNormalization normalization : {RouterNormalization::None, RouterNormalization::SelectedExperts})
        {
            options.score_function = score_function;
            options.normalization = normalization;
            general_explicit_options.score_function = score_function;
            general_explicit_options.normalization = normalization;
            check(static_cast<bool>(forward_router(logits, 1, options, routed)));
            check(static_cast<bool>(forward_router(repeated_logits, 2, general_explicit_options, general_explicit)));
            check(general_explicit.assignment_count == 4);
            check(general_explicit.batches.size() == routed.batches.size());
            for (size_t batch_index = 0; batch_index < routed.batches.size(); ++batch_index)
            {
                const ExpertRoute& expected_route = routed.batches[batch_index].routes[0];
                check(general_explicit.batches[batch_index].expert_id == routed.batches[batch_index].expert_id);
                check(general_explicit.batches[batch_index].routes.size() == 2);
                for (uint32_t token_index = 0; token_index < 2; ++token_index)
                {
                    const ExpertRoute& actual_route = general_explicit.batches[batch_index].routes[token_index];
                    check(actual_route.token_index == token_index);
                    check(actual_route.rank == expected_route.rank);
                    check(actual_route.weight == expected_route.weight);
                }
            }
        }
    }

    ActivationBuffer hyper_input(1, 2);
    hyper_input.row(0)[0] = 2.0f;
    hyper_input.row(0)[1] = 4.0f;
    TensorData function;
    function.dtype = DType::Float32;
    function.shape = {8, 2};
    function.float32_data.resize(16, 0.0f);
    TensorData base;
    base.dtype = DType::Float32;
    base.shape = {8};
    base.float32_data.resize(8, 0.0f);
    TensorData scale;
    scale.dtype = DType::Float32;
    scale.shape = {3};
    scale.float32_data.resize(3, 0.0f);
    HyperConnectionMix mixed;
    HyperConnectionScratch hyper_scratch;
    auto mixed_result = hc_pre(hyper_input,
                               function,
                               scale,
                               base,
                               2,
                               2,
                               1e-6f,
                               1e-6f,
                               mixed,
                               hyper_scratch,
                               g_test_optimization_flags);
    check(static_cast<bool>(mixed_result));
    check_near(mixed.reduced.row(0)[0], 3.000006f, 1e-4f);

    // FP8 projections force input quantization through the retained auxiliary buffer.
    TensorData float8_function = function;
    float8_function.dtype = DType::Float8E4M3;
    float8_function.float32_data.clear();
    float8_function.quantized_data.assign(16, 0);
    float8_function.quantization_scales = {1.0f};
    HyperConnectionMix float8_mix;
    HyperConnectionScratch float8_scratch;
    auto float8_pre = hc_pre(hyper_input,
                             float8_function,
                             scale,
                             base,
                             2,
                             2,
                             1e-6f,
                             1e-6f,
                             float8_mix,
                             float8_scratch,
                             g_test_optimization_flags);
    check(static_cast<bool>(float8_pre));
    const std::byte* auxiliary_storage = float8_scratch.auxiliary.bytes().data();
    const uint64_t auxiliary_capacity = float8_scratch.auxiliary.allocated_bytes();
    check(auxiliary_capacity >= 2 * sizeof(float));
    float8_pre = hc_pre(hyper_input,
                        float8_function,
                        scale,
                        base,
                        2,
                        2,
                        1e-6f,
                        1e-6f,
                        float8_mix,
                        float8_scratch,
                        g_test_optimization_flags);
    check(static_cast<bool>(float8_pre));
    check(float8_scratch.auxiliary.bytes().data() == auxiliary_storage);
    check(float8_scratch.auxiliary.allocated_bytes() == auxiliary_capacity);
    for (size_t index = 0; index < mixed.reduced.values().size(); ++index)
        check_near(float8_mix.reduced.values()[index], mixed.reduced.values()[index], 1e-5f);

    TensorData int8_function;
    int8_function.dtype = DType::Int8;
    int8_function.shape = {8, 2};
    int8_function.int8_data.assign(16, 0);
    int8_function.quantization_scales.assign(8, 1.0f);
    HyperConnectionMix int8_mix;
    HyperConnectionScratch int8_scratch;
    auto int8_pre = hc_pre(hyper_input,
                           int8_function,
                           scale,
                           base,
                           2,
                           2,
                           1e-6f,
                           1e-6f,
                           int8_mix,
                           int8_scratch,
                           g_test_optimization_flags);
    check(static_cast<bool>(int8_pre));
    for (size_t index = 0; index < mixed.reduced.values().size(); ++index)
        check_near(int8_mix.reduced.values()[index], mixed.reduced.values()[index], 1e-5f);
    for (size_t index = 0; index < mixed.post.size(); ++index)
        check_near(int8_mix.post[index], mixed.post[index], 1e-5f);

    TensorData malformed_function = function;
    malformed_function.shape.clear();
    auto invalid_function = hc_pre(hyper_input, malformed_function, scale, base,
                                   2, 2, 1e-6f, 1e-6f, mixed, hyper_scratch, g_test_optimization_flags);
    check(!invalid_function);
    check(invalid_function.error().code == ErrorCode::InvalidModel);
    check(invalid_function.error().message == "invalid hyper-connection function tensor");

    malformed_function.shape = {4, 4};
    auto invalid_function_dimensions = hc_pre(hyper_input, malformed_function, scale, base,
                                              2, 2, 1e-6f, 1e-6f, mixed, hyper_scratch, g_test_optimization_flags);
    check(!invalid_function_dimensions);
    check(invalid_function_dimensions.error().code == ErrorCode::InvalidModel);
    check(invalid_function_dimensions.error().message == "invalid hyper-connection function tensor");

    TensorData malformed_base = base;
    malformed_base.shape = {8, 1};
    auto invalid_base = hc_pre(hyper_input, function, scale, malformed_base,
                               2, 2, 1e-6f, 1e-6f, mixed, hyper_scratch, g_test_optimization_flags);
    check(!invalid_base);
    check(invalid_base.error().code == ErrorCode::InvalidModel);
    check(invalid_base.error().message == "invalid hyper-connection base tensor");

    ActivationBuffer branch(1, 1);
    branch.row(0)[0] = 10.0f;
    ActivationBuffer connected_output;
    auto connected = hc_post(branch, hyper_input, mixed, 2, connected_output);
    check(static_cast<bool>(connected));
    check_near(connected_output.row(0)[0], 13.0f, 1e-3f);
    check_near(connected_output.row(0)[1], 13.0f, 1e-3f);
    HyperConnectionMix directed_mix;
    directed_mix.post = {0.0f, 0.0f};
    directed_mix.combine = {1.0f, 2.0f, 3.0f, 4.0f};
    branch.row(0)[0] = 0.0f;
    connected = hc_post(branch, hyper_input, directed_mix, 2, connected_output);
    check(static_cast<bool>(connected));
    check_near(connected_output.row(0)[0], 14.0f, 1e-5f);
    check_near(connected_output.row(0)[1], 20.0f, 1e-5f);

    ActivationBuffer four_way_branch(1, 2);
    four_way_branch.row(0)[0] = 10.0f;
    four_way_branch.row(0)[1] = 20.0f;
    ActivationBuffer four_way_residual(1, 8);
    const std::array<float, 8> four_way_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::copy_n(four_way_values.data(), four_way_values.size(), four_way_residual.row(0));
    HyperConnectionMix four_way_mix;
    four_way_mix.post = {1.0f, 2.0f, 3.0f, 4.0f};
    four_way_mix.combine = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f};
    connected = hc_post(four_way_branch, four_way_residual, four_way_mix, 4, connected_output);
    check(static_cast<bool>(connected));
    const std::array<float, 8> four_way_expected = {162.0f, 200.0f, 188.0f, 240.0f, 214.0f, 280.0f, 240.0f, 320.0f};
    for (size_t index = 0; index < four_way_expected.size(); ++index)
        check_near(connected_output.row(0)[index], four_way_expected[index], 1e-4f);

    auto aliased_branch = hc_post(branch, hyper_input, directed_mix, 2, branch);
    check(!aliased_branch);
    check(aliased_branch.error().code == ErrorCode::InvalidArgument);
    check_near(branch.row(0)[0], 0.0f, 1e-6f);
    auto aliased_residual = hc_post(branch, hyper_input, directed_mix, 2, hyper_input);
    check(!aliased_residual);
    check(aliased_residual.error().code == ErrorCode::InvalidArgument);
    check_near(hyper_input.row(0)[0], 2.0f, 1e-6f);
    check_near(hyper_input.row(0)[1], 4.0f, 1e-6f);

    ActivationBuffer larger_hyper_input(3, 4);
    for (size_t row = 0; row < larger_hyper_input.rows(); ++row)
        std::fill_n(larger_hyper_input.row(row), larger_hyper_input.columns(), 1.0f + static_cast<float>(row));
    TensorData larger_function = function;
    larger_function.shape = {8, 4};
    larger_function.float32_data.resize(32, 0.0f);
    auto enlarged = hc_pre(larger_hyper_input,
                           larger_function,
                           scale,
                           base,
                           2,
                           2,
                           1e-6f,
                           1e-6f,
                           mixed,
                           hyper_scratch,
                           g_test_optimization_flags);
    check(static_cast<bool>(enlarged));
    const uint64_t grown_reduced_capacity = mixed.reduced.allocated_bytes();
    const uint64_t grown_normalized_capacity = hyper_scratch.normalized.allocated_bytes();
    const std::byte* grown_reduced_data = mixed.reduced.bytes().data();
    const std::byte* grown_normalized_data = hyper_scratch.normalized.bytes().data();
    auto shrunk = hc_pre(hyper_input,
                         function,
                         scale,
                         base,
                         2,
                         2,
                         1e-6f,
                         1e-6f,
                         mixed,
                         hyper_scratch,
                         g_test_optimization_flags);
    check(static_cast<bool>(shrunk));
    check(mixed.reduced.allocated_bytes() == grown_reduced_capacity);
    check(hyper_scratch.normalized.allocated_bytes() == grown_normalized_capacity);
    check(mixed.reduced.bytes().data() == grown_reduced_data);
    check(hyper_scratch.normalized.bytes().data() == grown_normalized_data);
    check_near(mixed.reduced.row(0)[0], 3.000006f, 1e-4f);

    TensorData head_function;
    head_function.dtype = DType::Float32;
    head_function.shape = {2, 2};
    head_function.float32_data.resize(4, 0.0f);
    TensorData head_base;
    head_base.dtype = DType::Float32;
    head_base.shape = {2};
    head_base.float32_data.resize(2, 0.0f);
    TensorData head_scale;
    head_scale.dtype = DType::Float32;
    head_scale.shape = {1};
    head_scale.float32_data = {0.0f};
    ActivationBuffer head_output(1, 1);
    head_output.row(0)[0] = 99.0f;
    auto headed = hc_head(hyper_input,
                          head_function,
                          head_scale,
                          head_base,
                          2,
                          1e-6f,
                          1e-6f,
                          head_output,
                          hyper_scratch,
                          g_test_optimization_flags);
    check(static_cast<bool>(headed));
    check(head_output.rows() == 1);
    check(head_output.columns() == 1);
    check_near(head_output.row(0)[0], 3.000006f, 1e-4f);
    std::fill_n(head_output.row(0), head_output.columns(), 77.0f);
    headed = hc_head(hyper_input,
                     head_function,
                     head_scale,
                     head_base,
                     2,
                     1e-6f,
                     1e-6f,
                     head_output,
                     hyper_scratch,
                     g_test_optimization_flags);
    check(static_cast<bool>(headed));
    check_near(head_output.row(0)[0], 3.000006f, 1e-4f);

    TensorData float8_head_function;
    float8_head_function.dtype = DType::Float8E4M3;
    float8_head_function.shape = {2, 2};
    float8_head_function.quantized_data.assign(4, 0);
    float8_head_function.quantization_scales = {1.0f};
    ActivationBuffer float8_head_output;
    auto float8_head = hc_head(hyper_input,
                               float8_head_function,
                               head_scale,
                               head_base,
                               2,
                               1e-6f,
                               1e-6f,
                               float8_head_output,
                               float8_scratch,
                               g_test_optimization_flags);
    check(static_cast<bool>(float8_head));
    check_near(float8_head_output.row(0)[0], 3.000006f, 1e-4f);
    check(float8_scratch.auxiliary.bytes().data() == auxiliary_storage);
    check(float8_scratch.auxiliary.allocated_bytes() == auxiliary_capacity);

    TensorData int8_head_function;
    int8_head_function.dtype = DType::Int8;
    int8_head_function.shape = {2, 2};
    int8_head_function.int8_data.assign(4, 0);
    int8_head_function.quantization_scales.assign(2, 1.0f);
    ActivationBuffer int8_head_output;
    auto int8_head = hc_head(hyper_input,
                             int8_head_function,
                             head_scale,
                             head_base,
                             2,
                             1e-6f,
                             1e-6f,
                             int8_head_output,
                             int8_scratch,
                             g_test_optimization_flags);
    check(static_cast<bool>(int8_head));
    check_near(int8_head_output.row(0)[0], 3.000006f, 1e-4f);

    TensorData head4_function;
    head4_function.dtype = DType::Float32;
    head4_function.shape = {4, 8};
    head4_function.float32_data.resize(32, 0.0f);
    TensorData head4_base;
    head4_base.dtype = DType::Float32;
    head4_base.shape = {4};
    head4_base.float32_data.resize(4, 0.0f);
    ActivationBuffer head4_output;
    auto headed4 = hc_head(four_way_residual,
                           head4_function,
                           head_scale,
                           head4_base,
                           4,
                           1e-6f,
                           1e-6f,
                           head4_output,
                           hyper_scratch,
                           g_test_optimization_flags);
    check(static_cast<bool>(headed4));
    const float head4_scale = 0.500001f;
    check_near(head4_output.row(0)[0], 16.0f * head4_scale, 1e-4f);
    check_near(head4_output.row(0)[1], 20.0f * head4_scale, 1e-4f);
    auto aliased_head = hc_head(hyper_input,
                                head_function,
                                head_scale,
                                head_base,
                                2,
                                1e-6f,
                                1e-6f,
                                hyper_input,
                                hyper_scratch,
                                g_test_optimization_flags);
    check(!aliased_head);
    check(aliased_head.error().code == ErrorCode::InvalidArgument);
    check_near(hyper_input.row(0)[0], 2.0f, 1e-6f);
    check_near(hyper_input.row(0)[1], 4.0f, 1e-6f);

    for (float value : std::array<float, 7>{-448.0f, -1.5f, -0.001f, 0.0f, 0.5f, 6.0f, 448.0f})
    {
        const float round_trip = float8_e4m3_to_float(float_to_float8_e4m3(value));
        check(std::isfinite(round_trip));
        check(std::signbit(round_trip) == std::signbit(value) || value == 0.0f);
    }
    std::array<uint8_t, 256> float8_weights = {};
    std::array<float, 256> float8_input = {};
    for (size_t index = 0; index < float8_weights.size(); ++index)
    {
        uint8_t encoded = static_cast<uint8_t>(index);
        if ((encoded & UINT8_C(0x7f)) == UINT8_C(0x7f))
            encoded ^= UINT8_C(1);
        float8_weights[index] = encoded;
        float8_input[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.015625f;
    }
    const std::array<float, 2> float8_scales = {0.5f, 2.0f};
    float float8_reference = 0.0f;
    for (size_t block = 0; block < float8_scales.size(); ++block)
    {
        float partial = 0.0f;
        for (size_t index = block * 128; index < (block + 1) * 128; ++index)
            partial += float8_e4m3_to_float(float8_weights[index]) * float8_input[index];
        float8_reference += partial * float8_scales[block];
    }
    check_near(float8_e4m3_block_dot(float8_weights.data(), float8_scales.data(), float8_input.data(), 256, 128), float8_reference, 1e-3f);
    std::array<uint8_t, 4 * 256> float8_row_weights = {};
    std::array<float, 4> float8_row_reference = {};
    std::array<float, 4> float8_row_output = {};
    for (size_t row = 0; row < float8_row_reference.size(); ++row)
    {
        for (size_t index = 0; index < float8_weights.size(); ++index)
        {
            uint8_t encoded = static_cast<uint8_t>(index + row * 13);
            if ((encoded & UINT8_C(0x7f)) == UINT8_C(0x7f))
                encoded ^= UINT8_C(1);
            float8_row_weights[row * float8_weights.size() + index] = encoded;
            float8_row_reference[row] += float8_e4m3_to_float(encoded) * float8_input[index] * float8_scales[index / 128];
        }
    }
    float8_e4m3_block_dot_rows4(float8_row_weights.data(),
                                256,
                                float8_scales.data(),
                                float8_input.data(),
                                256,
                                128,
                                4,
                                float8_row_output.data());
    for (size_t row = 0; row < float8_row_output.size(); ++row)
        check_near(float8_row_output[row], float8_row_reference[row], 1e-3f);
    std::array<float, 256> quantized_float8_input = float8_input;
    std::array<float, 256> quantized_float8_expected = float8_input;
    for (size_t block_begin = 0; block_begin < quantized_float8_expected.size();
         block_begin += 128)
    {
        const size_t block_end = std::min(quantized_float8_expected.size(), block_begin + 128);
        float maximum = 1e-4f;
        for (size_t index = block_begin; index < block_end; ++index)
        {
            maximum = std::max(maximum, std::fabs(quantized_float8_expected[index]));
        }
        const float scale = std::exp2(std::ceil(std::log2(maximum / 448.0f)));
        for (size_t index = block_begin; index < block_end; ++index)
        {
            const float normalized = std::clamp(quantized_float8_expected[index] / scale,
                                                -448.0f,
                                                448.0f);
            quantized_float8_expected[index] = float8_e4m3_to_float(float_to_float8_e4m3(normalized))
                                               * scale;
        }
    }
    quantize_float8_e4m3_inplace(quantized_float8_input.data(),
                                 static_cast<uint32_t>(quantized_float8_input.size()),
                                 128,
                                 true,
                                 g_test_optimization_flags);
    for (size_t index = 0; index < quantized_float8_input.size(); ++index)
    {
        check_near(quantized_float8_input[index],
                   quantized_float8_expected[index],
                   0.0f);
    }
    std::array<float, 256> quantized_float8_copy = {};
    quantize_float8_e4m3(float8_input.data(), quantized_float8_copy.data(),
                         static_cast<uint32_t>(float8_input.size()),
                         128,
                         true,
                         g_test_optimization_flags);
    for (size_t index = 0; index < quantized_float8_copy.size(); ++index)
    {
        check_near(quantized_float8_copy[index], quantized_float8_input[index],
                   0.0f);
    }
    const float quantized_float8_reference = float8_e4m3_block_dot(float8_weights.data(),
                                                                   float8_scales.data(),
                                                                   quantized_float8_input.data(),
                                                                   static_cast<uint32_t>(quantized_float8_input.size()),
                                                                   128);
    check_near(float8_e4m3_quantized_input_dot(float8_weights.data(),
                                               float8_scales.data(),
                                               quantized_float8_input.data(),
                                               static_cast<uint32_t>(quantized_float8_input.size()),
                                               128,
                                               g_test_optimization_flags),
               quantized_float8_reference,
               1e-3f);
    std::array<float, 4> quantized_float8_row_reference = {};
    std::array<float, 4> quantized_float8_row_output = {};
    float8_e4m3_block_dot_rows4(float8_row_weights.data(),
                                256,
                                float8_scales.data(),
                                quantized_float8_input.data(),
                                256,
                                128,
                                4,
                                quantized_float8_row_reference.data());
    float8_e4m3_quantized_input_dot_rows(float8_row_weights.data(),
                                         256,
                                         float8_scales.data(),
                                         quantized_float8_input.data(),
                                         256,
                                         128,
                                         4,
                                         quantized_float8_row_output.data(),
                                         g_test_optimization_flags);
    for (size_t row = 0; row < quantized_float8_row_output.size(); ++row)
    {
        check_near(quantized_float8_row_output[row],
                   quantized_float8_row_reference[row], 1e-3f);
    }
    std::array<float, 4 * 256> quantized_float8_batch_input = {};
    std::array<float, 4 * 4> quantized_float8_batch_expected = {};
    std::array<float, 4 * 4> quantized_float8_batch_output = {};
    for (size_t token = 0; token < 4; ++token)
    {
        for (size_t index = 0; index < 256; ++index)
        {
            quantized_float8_batch_input[token * 256 + index] = quantized_float8_input[index]
                                                                * (1.0f + static_cast<float>(token) * 0.125f);
        }
        float8_e4m3_quantized_input_dot_rows(float8_row_weights.data(), 256, float8_scales.data(),
                                             quantized_float8_batch_input.data() + token * 256, 256, 128, 4,
                                             quantized_float8_batch_expected.data() + token * 4,
                                             g_test_optimization_flags);
    }
    float8_e4m3_quantized_input_dot_rows_batch(float8_row_weights.data(), 256, float8_scales.data(),
                                               quantized_float8_batch_input.data(), 256, 256, 128, 4, 4, 4,
                                               quantized_float8_batch_output.data(),
                                               g_test_optimization_flags);
    for (size_t index = 0; index < quantized_float8_batch_output.size();
         ++index)
    {
        check_near(quantized_float8_batch_output[index],
                   quantized_float8_batch_expected[index], 1e-2f);
    }
    std::array<uint8_t, 32> repeated_float8_weights = {};
    std::array<float, 32> unit_float8_input = {};
    unit_float8_input.fill(1.0f);
    const std::array<float, 1> unit_float8_scale = {1.0f};
    for (uint32_t code = 0; code < 256; ++code)
    {
        repeated_float8_weights.fill(static_cast<uint8_t>(code));
        const float fp32_value = float8_e4m3_block_dot(repeated_float8_weights.data(), unit_float8_scale.data(),
                                                       unit_float8_input.data(), 32, 32);
        const float bfloat16_value = float8_e4m3_quantized_input_dot(repeated_float8_weights.data(), unit_float8_scale.data(),
                                                                     unit_float8_input.data(),
                                                                     32,
                                                                     32,
                                                                     g_test_optimization_flags);
        if ((code & UINT32_C(0x7f)) == UINT32_C(0x7f))
        {
            check(std::isnan(fp32_value));
            check(std::isnan(bfloat16_value));
        }
        else
        {
            check_near(bfloat16_value, fp32_value, 1e-3f);
        }
    }
    for (uint32_t count : std::array<uint32_t, 6>{31, 32, 33, 127, 128,
                                                  129})
    {
        check_near(float8_e4m3_quantized_input_dot(float8_weights.data(), float8_scales.data(),
                                                   quantized_float8_input.data(),
                                                   count,
                                                   128,
                                                   g_test_optimization_flags),
                   float8_e4m3_block_dot(float8_weights.data(), float8_scales.data(),
                                         quantized_float8_input.data(), count, 128),
                   1e-3f);
    }
    std::array<uint8_t, 8 * 256> float8_rows8_weights = {};
    for (size_t row = 0; row < 8; ++row)
    {
        for (size_t index = 0; index < 256; ++index)
        {
            uint8_t encoded = static_cast<uint8_t>(index + row * 19);
            if ((encoded & UINT8_C(0x7f)) == UINT8_C(0x7f))
                encoded ^= UINT8_C(1);
            float8_rows8_weights[row * 256 + index] = encoded;
        }
    }
    for (uint32_t row_count = 1; row_count <= 8; ++row_count)
    {
        std::array<float, 8> rows8_output = {};
        float8_e4m3_quantized_input_dot_rows(float8_rows8_weights.data(), 256, float8_scales.data(),
                                             quantized_float8_input.data(), 256, 128, row_count,
                                             rows8_output.data(),
                                             g_test_optimization_flags);
        for (uint32_t row = 0; row < row_count; ++row)
        {
            check_near(rows8_output[row],
                       float8_e4m3_block_dot(float8_rows8_weights.data() + static_cast<size_t>(row) * 256,
                                             float8_scales.data(), quantized_float8_input.data(), 256,
                                             128),
                       1e-3f);
        }
    }
    check(static_cast<bool>(std::string(float8_kernel_name()).size() > 0));
    const std::string float8_kernel_name = float8_linear_kernel_name(g_test_optimization_flags);
    check(static_cast<bool>(!float8_kernel_name.empty()));

    constexpr uint32_t pair_bfloat16_columns = 35;
    auto make_bfloat16_pair_weight = [pair_bfloat16_columns](uint32_t output_columns, uint32_t seed) {
        TensorData matrix;
        matrix.dtype = DType::BFloat16;
        matrix.shape = {output_columns, pair_bfloat16_columns};
        matrix.bfloat16_data.resize(matrix.element_count());
        for (size_t index = 0; index < matrix.bfloat16_data.size(); ++index)
        {
            const float value = static_cast<float>(static_cast<int>((index * 13 + seed * 7) % 37) - 18)
                                * 0.0078125f;
            matrix.bfloat16_data[index] = float_to_bfloat16(value);
        }
        return matrix;
    };
    const TensorData bfloat16_pair_first = make_bfloat16_pair_weight(9, 1);
    const TensorData bfloat16_pair_second = make_bfloat16_pair_weight(5, 2);
    ActivationBuffer bfloat16_pair_input(1, pair_bfloat16_columns);
    for (uint32_t column = 0; column < pair_bfloat16_columns; ++column)
    {
        bfloat16_pair_input.row(0)[column] = static_cast<float>(static_cast<int>((column * 11) % 29) - 14)
                                             * 0.015625f;
    }
    ActivationBuffer bfloat16_pair_first_output;
    ActivationBuffer bfloat16_pair_second_output;
    const bool bfloat16_pair_used = forward_linear({&bfloat16_pair_first, &bfloat16_pair_second},
                                                   bfloat16_pair_input,
                                                   {&bfloat16_pair_first_output, &bfloat16_pair_second_output},
                                                   g_test_optimization_flags);
#if defined(__aarch64__) || defined(_M_ARM64)
    check(bfloat16_pair_used);
    for (const auto& projection : {std::pair<const TensorData*, const ActivationBuffer*>{
                                       &bfloat16_pair_first, &bfloat16_pair_first_output},
                                   std::pair<const TensorData*, const ActivationBuffer*>{
                                       &bfloat16_pair_second, &bfloat16_pair_second_output}})
    {
        for (uint32_t output_column = 0; output_column < projection.first->shape[0]; ++output_column)
        {
            float expected = 0.0f;
            for (uint32_t input_column = 0; input_column < pair_bfloat16_columns; ++input_column)
            {
                expected += bfloat16_to_float(projection.first->bfloat16_data[static_cast<size_t>(output_column) * pair_bfloat16_columns + input_column])
                            * bfloat16_pair_input.row(0)[input_column];
            }
            check_near(projection.second->row(0)[output_column], expected, 1e-6f);
        }
    }

    TensorData mismatched_bfloat16_pair = bfloat16_pair_second;
    mismatched_bfloat16_pair.shape[1] = pair_bfloat16_columns - 1;
    ActivationBuffer untouched_pair_output(1, 2);
    untouched_pair_output.row(0)[0] = 3.0f;
    untouched_pair_output.row(0)[1] = -4.0f;
    check(!forward_linear({&bfloat16_pair_first, &mismatched_bfloat16_pair},
                          bfloat16_pair_input,
                          {&untouched_pair_output, &bfloat16_pair_second_output},
                          g_test_optimization_flags));
    check(untouched_pair_output.columns() == 2);
    check(untouched_pair_output.row(0)[0] == 3.0f);
    check(untouched_pair_output.row(0)[1] == -4.0f);
    check(!forward_linear({&bfloat16_pair_first, &bfloat16_pair_second},
                          bfloat16_pair_input,
                          {&untouched_pair_output, &untouched_pair_output},
                          g_test_optimization_flags));
    check(untouched_pair_output.columns() == 2);
    check(untouched_pair_output.row(0)[0] == 3.0f);
    check(!forward_linear({&bfloat16_pair_first, &bfloat16_pair_second},
                          bfloat16_pair_input,
                          {&bfloat16_pair_input, &bfloat16_pair_second_output},
                          g_test_optimization_flags));
    check(bfloat16_pair_input.row(0)[0] == -0.21875f);
    TensorData missing_bfloat16_storage = bfloat16_pair_second;
    missing_bfloat16_storage.bfloat16_data.clear();
    check(!forward_linear({&bfloat16_pair_first, &missing_bfloat16_storage},
                          bfloat16_pair_input,
                          {&untouched_pair_output, &bfloat16_pair_second_output},
                          g_test_optimization_flags));
    check(untouched_pair_output.columns() == 2);
    check(untouched_pair_output.row(0)[0] == 3.0f);
    ActivationBuffer bfloat16_pair_non_float_output(1, 2, DType::BFloat16);
    check(!forward_linear({&bfloat16_pair_first, &bfloat16_pair_second},
                          bfloat16_pair_input,
                          {&bfloat16_pair_non_float_output, &bfloat16_pair_second_output},
                          g_test_optimization_flags));
    ActivationBuffer bfloat16_pair_non_float_input(1, pair_bfloat16_columns, DType::BFloat16);
    check(!forward_linear({&bfloat16_pair_first, &bfloat16_pair_second},
                          bfloat16_pair_non_float_input,
                          {&untouched_pair_output, &bfloat16_pair_second_output},
                          g_test_optimization_flags));
#if NCNN_MOE_WITH_NCNN && NCNN_MOE_WITH_VULKAN
    if (get_gpu_count() > 0)
    {
        TensorData pair_linear_weight;
        pair_linear_weight.dtype = DType::Float32;
        pair_linear_weight.shape = {2, pair_bfloat16_columns};
        pair_linear_weight.float32_data.resize(pair_linear_weight.element_count(), 0.125f);
        const VulkanRuntimePtr pair_runtime = create_vulkan_runtime();
        const auto pair_linear = Linear::create(pair_linear_weight,
                                                nullptr,
                                                get_default_gpu_index(),
                                                pair_runtime,
                                                g_test_optimization_flags);
        check(static_cast<bool>(pair_linear));
        CompiledOperator pair_operator;
        pair_operator.linear = pair_linear;
        check(!forward_linear({&bfloat16_pair_first, &bfloat16_pair_second},
                              bfloat16_pair_input,
                              {&untouched_pair_output, &bfloat16_pair_second_output},
                              g_test_optimization_flags,
                              {&pair_operator, nullptr}));
        check(untouched_pair_output.columns() == 2);
        check(untouched_pair_output.row(0)[0] == 3.0f);
    }
#endif

#else
    check(!bfloat16_pair_used);
#endif

    auto make_float8_projection = [](uint32_t seed,
                                     uint32_t output_columns) {
        TensorData matrix;
        matrix.dtype = DType::Float8E4M3;
        matrix.shape = {output_columns, 128};
        const size_t element_count = matrix.element_count();
        std::shared_ptr<uint8_t[]> storage(new uint8_t[element_count], std::default_delete<uint8_t[]>());
        for (size_t index = 0; index < element_count; ++index)
        {
            const float value = static_cast<float>(static_cast<int>((index * 7 + seed * 11) % 43) - 21)
                                * 0.03125f;
            storage[index] = float_to_float8_e4m3(value);
        }
        matrix.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
        matrix.mapped_size = element_count;
        matrix.quantization_scales.resize((output_columns + 127) / 128);
        for (size_t index = 0; index < matrix.quantization_scales.size();
             ++index)
        {
            matrix.quantization_scales[index] = index == 0 ? 0.5f : 2.0f;
        }
        return matrix;
    };
    const TensorData float8_gate = make_float8_projection(1, 7);
    const TensorData float8_up = make_float8_projection(2, 7);
    ActivationBuffer fused_input(2, 128);
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < fused_input.columns(); ++column)
        {
            fused_input.row(token_index)[column] = static_cast<float>(static_cast<int>((token_index * 17 + column * 5) % 37)
                                                                      - 18)
                                                   * 0.015625f;
        }
    }
    ActivationBuffer reference_up = forward_linear(float8_up,
                                                   fused_input,
                                                   g_test_optimization_flags);
    const ActivationBuffer reference_raw_up = reference_up;
    const ActivationBuffer reference_gate = forward_linear(float8_gate,
                                                           fused_input,
                                                           g_test_optimization_flags);
    for (size_t token_index = 0; token_index < reference_up.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < reference_up.columns(); ++column)
        {
            const float gate_value = reference_gate.row(token_index)[column];
            reference_up.row(token_index)[column] *= gate_value / (1.0f + std::exp(-gate_value));
        }
    }
    ActivationBuffer paired_gate;
    ActivationBuffer paired_up;
    check(forward_linear({&float8_gate, &float8_up},
                         fused_input,
                         {&paired_gate, &paired_up},
                         g_test_optimization_flags));
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < paired_gate.columns(); ++column)
        {
            check_near(paired_gate.row(token_index)[column],
                       reference_gate.row(token_index)[column], 1e-6f);
            check_near(paired_up.row(token_index)[column],
                       reference_raw_up.row(token_index)[column],
                       1e-6f);
        }
    }
    ActivationBuffer scratch_pair_gate;
    ActivationBuffer scratch_pair_up;
    ActivationBuffer quantized_input_scratch;
    check(forward_linear({&float8_gate, &float8_up},
                         fused_input,
                         {&scratch_pair_gate, &scratch_pair_up},
                         g_test_optimization_flags,
                         {},
                         &quantized_input_scratch));
    const size_t initial_quantized_capacity = quantized_input_scratch.allocated_bytes();
    const std::byte* initial_quantized_address = quantized_input_scratch.bytes().data();
    std::fill(quantized_input_scratch.mutable_bytes().begin(),
              quantized_input_scratch.mutable_bytes().end(),
              std::byte{0xa5});
    check(forward_linear({&float8_gate, &float8_up},
                         fused_input,
                         {&scratch_pair_gate, &scratch_pair_up},
                         g_test_optimization_flags,
                         {},
                         &quantized_input_scratch));
    check(quantized_input_scratch.allocated_bytes() == initial_quantized_capacity);
    check(quantized_input_scratch.bytes().data() == initial_quantized_address);
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (size_t column = 0; column < scratch_pair_gate.columns(); ++column)
        {
            check_near(scratch_pair_gate.row(token_index)[column],
                       paired_gate.row(token_index)[column], 1e-6f);
            check_near(scratch_pair_up.row(token_index)[column],
                       paired_up.row(token_index)[column], 1e-6f);
        }
    }
    ActivationBuffer larger_fused_input(4, fused_input.columns());
    for (size_t token_index = 0; token_index < larger_fused_input.rows();
         ++token_index)
    {
        std::copy_n(fused_input.row(token_index % fused_input.rows()),
                    fused_input.columns(),
                    larger_fused_input.row(token_index));
    }
    ActivationBuffer larger_default_gate;
    ActivationBuffer larger_default_up;
    check(forward_linear({&float8_gate, &float8_up},
                         larger_fused_input,
                         {&larger_default_gate, &larger_default_up},
                         g_test_optimization_flags));
    check(forward_linear({&float8_gate, &float8_up},
                         larger_fused_input,
                         {&scratch_pair_gate, &scratch_pair_up},
                         g_test_optimization_flags,
                         {},
                         &quantized_input_scratch));
    const size_t grown_quantized_capacity = quantized_input_scratch.allocated_bytes();
    const std::byte* grown_quantized_address = quantized_input_scratch.bytes().data();
    check(grown_quantized_capacity >= initial_quantized_capacity);
    for (size_t token_index = 0; token_index < larger_fused_input.rows();
         ++token_index)
    {
        for (size_t column = 0; column < larger_default_gate.columns(); ++column)
        {
            check_near(scratch_pair_gate.row(token_index)[column],
                       larger_default_gate.row(token_index)[column], 1e-6f);
            check_near(scratch_pair_up.row(token_index)[column],
                       larger_default_up.row(token_index)[column], 1e-6f);
        }
    }
    std::fill(quantized_input_scratch.mutable_bytes().begin(),
              quantized_input_scratch.mutable_bytes().end(),
              std::byte{0x5a});
    check(forward_linear({&float8_gate, &float8_up},
                         fused_input,
                         {&scratch_pair_gate, &scratch_pair_up},
                         g_test_optimization_flags,
                         {},
                         &quantized_input_scratch));
    check(quantized_input_scratch.allocated_bytes() == grown_quantized_capacity);
    check(quantized_input_scratch.bytes().data() == grown_quantized_address);
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (size_t column = 0; column < scratch_pair_gate.columns(); ++column)
        {
            check_near(scratch_pair_gate.row(token_index)[column],
                       paired_gate.row(token_index)[column], 1e-6f);
            check_near(scratch_pair_up.row(token_index)[column],
                       paired_up.row(token_index)[column], 1e-6f);
        }
    }

    const std::array<TensorData, 4> four_float8_matrices{
        make_float8_projection(3, 129),
        make_float8_projection(4, 257),
        make_float8_projection(5, 385),
        make_float8_projection(6, 513)};
    ActivationBuffer four_float8_input(8, 128);
    for (size_t token_index = 0; token_index < four_float8_input.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < four_float8_input.columns(); ++column)
        {
            four_float8_input.row(token_index)[column] = static_cast<float>(static_cast<int>((token_index * 17 + column * 5) % 37)
                                                                            - 18)
                                                         * 0.015625f;
        }
    }
    const ActivationBuffer four_float8_input_copy = four_float8_input;
    const uint64_t four_float8_workload = (129u + 257u + 385u + 513u)
                                          * 128u * four_float8_input.rows();
    std::array<ActivationBuffer, 4> four_float8_outputs;
    for (size_t i = 0; i < four_float8_outputs.size(); ++i)
    {
        four_float8_outputs[i].reset(1, 2, false);
        four_float8_outputs[i].row(0)[0] = static_cast<float>(81 + i);
        four_float8_outputs[i].row(0)[1] = static_cast<float>(-91 - i);
    }
    ActivationBuffer four_float8_quantized_input;
    {
        CpuOpenMpThreadLimitScope thread_limit;
        thread_limit.set(1);
        check(cpu_linear_team_size(four_float8_workload, DType::Float8E4M3) == 1);
        check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                               &four_float8_matrices[2], &four_float8_matrices[3]},
                              four_float8_input,
                              {&four_float8_outputs[0], &four_float8_outputs[1],
                               &four_float8_outputs[2], &four_float8_outputs[3]},
                              g_test_optimization_flags,
                              {},
                              &four_float8_quantized_input));
        for (size_t i = 0; i < four_float8_outputs.size(); ++i)
        {
            check(four_float8_outputs[i].columns() == 2);
            check(four_float8_outputs[i].row(0)[0] == static_cast<float>(81 + i));
            check(four_float8_outputs[i].row(0)[1] == static_cast<float>(-91 - i));
        }
        check(four_float8_quantized_input.rows() == 0);
    }

    {
        CpuOpenMpThreadLimitScope thread_limit;
        thread_limit.set(8);
        const int selected_team = cpu_linear_team_size(four_float8_workload,
                                                       DType::Float8E4M3);
        const bool four_used = forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                                               &four_float8_matrices[2], &four_float8_matrices[3]},
                                              four_float8_input,
                                              {&four_float8_outputs[0], &four_float8_outputs[1],
                                               &four_float8_outputs[2], &four_float8_outputs[3]},
                                              g_test_optimization_flags,
                                              {},
                                              &four_float8_quantized_input);
        if (selected_team == 1)
        {
            check(!four_used);
            for (size_t i = 0; i < four_float8_outputs.size(); ++i)
            {
                check(four_float8_outputs[i].columns() == 2);
                check(four_float8_outputs[i].row(0)[0] == static_cast<float>(81 + i));
            }
            check(four_float8_quantized_input.rows() == 0);
        }
        else
        {
            check(four_used);
            std::array<ActivationBuffer, 4> first_four_outputs;
            for (size_t i = 0; i < four_float8_outputs.size(); ++i)
            {
                check(four_float8_outputs[i].columns() == four_float8_matrices[i].shape[0]);
                first_four_outputs[i] = four_float8_outputs[i];
                const ActivationBuffer expected = forward_linear(four_float8_matrices[i],
                                                                 four_float8_input,
                                                                 g_test_optimization_flags);
                check(first_four_outputs[i].rows() == expected.rows());
                check(first_four_outputs[i].columns() == expected.columns());
                for (size_t row = 0; row < expected.rows(); ++row)
                {
                    for (uint32_t column = 0; column < expected.columns(); ++column)
                    {
                        check_near(first_four_outputs[i].row(row)[column],
                                   expected.row(row)[column], 1e-6f);
                    }
                }
            }
            const size_t four_quantized_capacity = four_float8_quantized_input.allocated_bytes();
            const std::byte* four_quantized_address = four_float8_quantized_input.bytes().data();
            std::fill(four_float8_quantized_input.mutable_bytes().begin(),
                      four_float8_quantized_input.mutable_bytes().end(),
                      std::byte{0xa5});
            check(forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                                  &four_float8_matrices[2], &four_float8_matrices[3]},
                                 four_float8_input,
                                 {&four_float8_outputs[0], &four_float8_outputs[1],
                                  &four_float8_outputs[2], &four_float8_outputs[3]},
                                 g_test_optimization_flags,
                                 {},
                                 &four_float8_quantized_input));
            check(four_float8_quantized_input.allocated_bytes() == four_quantized_capacity);
            check(four_float8_quantized_input.bytes().data() == four_quantized_address);
            for (size_t i = 0; i < four_float8_outputs.size(); ++i)
            {
                for (size_t row = 0; row < first_four_outputs[i].rows(); ++row)
                {
                    for (uint32_t column = 0; column < first_four_outputs[i].columns(); ++column)
                    {
                        check_near(four_float8_outputs[i].row(row)[column],
                                   first_four_outputs[i].row(row)[column], 1e-6f);
                    }
                }
            }
        }
    }
    check(std::memcmp(four_float8_input.bytes().data(),
                      four_float8_input_copy.bytes().data(),
                      four_float8_input.bytes().size())
          == 0);

    const ActivationBuffer four_float8_quantized_before_rejections = four_float8_quantized_input;
    ActivationBuffer four_reject_output0(1, 2);
    ActivationBuffer four_reject_output1(1, 2);
    ActivationBuffer four_reject_output2(1, 2);
    ActivationBuffer four_reject_output3(1, 2);
    four_reject_output0.row(0)[0] = 101.0f;
    four_reject_output1.row(0)[0] = 102.0f;
    four_reject_output2.row(0)[0] = 103.0f;
    four_reject_output3.row(0)[0] = 104.0f;
    TensorData four_wrong_dtype = four_float8_matrices[2];
    four_wrong_dtype.dtype = DType::BFloat16;
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_wrong_dtype, &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output1,
                           &four_reject_output2, &four_reject_output3},
                          g_test_optimization_flags));
    TensorData four_wrong_shape = four_float8_matrices[2];
    four_wrong_shape.shape[1] = 127;
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_wrong_shape, &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output1,
                           &four_reject_output2, &four_reject_output3},
                          g_test_optimization_flags));
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_float8_matrices[2]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output1,
                           &four_reject_output2},
                          g_test_optimization_flags));
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_float8_matrices[2], &four_float8_matrices[3],
                           &four_float8_matrices[0]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output1,
                           &four_reject_output2, &four_reject_output3,
                           &four_reject_output0},
                          g_test_optimization_flags));
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_float8_matrices[2], &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output1,
                           &four_reject_output2},
                          g_test_optimization_flags));
    check(!forward_linear({&four_float8_matrices[0], nullptr,
                           &four_float8_matrices[2], &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output1,
                           &four_reject_output2, &four_reject_output3},
                          g_test_optimization_flags));
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_float8_matrices[2], &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_reject_output0, nullptr,
                           &four_reject_output2, &four_reject_output3},
                          g_test_optimization_flags));
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_float8_matrices[2], &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output1,
                           &four_reject_output2, &four_reject_output3},
                          g_test_optimization_flags,
                          {nullptr, nullptr,
                           nullptr}));
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_float8_matrices[2], &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output1,
                           &four_reject_output2, &four_reject_output3},
                          g_test_optimization_flags,
                          {},
                          &four_float8_input));
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_float8_matrices[2], &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output1,
                           &four_reject_output2, &four_reject_output3},
                          g_test_optimization_flags,
                          {},
                          &four_reject_output0));
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_float8_matrices[2], &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_float8_input, &four_reject_output1,
                           &four_reject_output2, &four_reject_output3},
                          g_test_optimization_flags));
    check(!forward_linear({&four_float8_matrices[0], &four_float8_matrices[1],
                           &four_float8_matrices[2], &four_float8_matrices[3]},
                          four_float8_input,
                          {&four_reject_output0, &four_reject_output0,
                           &four_reject_output2, &four_reject_output3},
                          g_test_optimization_flags));
    check(four_float8_quantized_input.rows() == four_float8_quantized_before_rejections.rows());
    check(four_float8_quantized_input.columns() == four_float8_quantized_before_rejections.columns());
    check(four_float8_quantized_input.allocated_bytes()
          == four_float8_quantized_before_rejections.allocated_bytes());
    check(four_float8_quantized_before_rejections.bytes().empty()
          || std::memcmp(four_float8_quantized_input.bytes().data(),
                         four_float8_quantized_before_rejections.bytes().data(),
                         four_float8_quantized_before_rejections.bytes().size())
                 == 0);
    check(four_reject_output0.columns() == 2 && four_reject_output0.row(0)[0] == 101.0f);
    check(four_reject_output1.columns() == 2 && four_reject_output1.row(0)[0] == 102.0f);
    check(four_reject_output2.columns() == 2 && four_reject_output2.row(0)[0] == 103.0f);
    check(four_reject_output3.columns() == 2 && four_reject_output3.row(0)[0] == 104.0f);

    ActivationBuffer scratch_linear_output;
    check(forward_linear({&float8_gate, &float8_up},
                         fused_input,
                         {&scratch_pair_gate, &scratch_pair_up},
                         g_test_optimization_flags,
                         {},
                         &quantized_input_scratch));
    forward_linear(float8_gate,
                   fused_input,
                   scratch_linear_output,
                   g_test_optimization_flags,
                   nullptr,
                   ExecutionBackend::Cpu,
                   &quantized_input_scratch);
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (size_t column = 0; column < scratch_linear_output.columns(); ++column)
            check_near(scratch_linear_output.row(token_index)[column],
                       reference_gate.row(token_index)[column], 1e-6f);
    }
    TensorData rms_weight;
    rms_weight.dtype = DType::Float32;
    rms_weight.shape = {128};
    rms_weight.float32_data.assign(128, 1.0f);
    const ActivationBuffer normalized_fused_input = forward_rms_norm(fused_input,
                                                                     rms_weight,
                                                                     1e-6f,
                                                                     0.0f);
    const ActivationBuffer reference_rms_projection = forward_linear(float8_gate,
                                                                     normalized_fused_input,
                                                                     g_test_optimization_flags);
    ActivationBuffer fused_rms_projection;
    check(forward_linear_rms_norm_float8(float8_gate, fused_input, rms_weight, 1e-6f,
                                         fused_rms_projection,
                                         g_test_optimization_flags));
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t column = 0;
             column < fused_rms_projection.columns(); ++column)
        {
            check_near(fused_rms_projection.row(token_index)[column],
                       reference_rms_projection.row(token_index)[column],
                       1e-6f);
        }
    }
    ActivationBuffer reused_rms_projection;
    for (size_t token_index = 0; token_index < quantized_input_scratch.rows();
         ++token_index)
    {
        std::fill_n(quantized_input_scratch.row(token_index),
                    quantized_input_scratch.columns(),
                    std::numeric_limits<float>::quiet_NaN());
    }
    check(forward_linear_rms_norm_float8(float8_gate, fused_input, rms_weight, 1e-6f,
                                         reused_rms_projection,
                                         g_test_optimization_flags,
                                         nullptr,
                                         &quantized_input_scratch));
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t column = 0;
             column < reused_rms_projection.columns(); ++column)
        {
            check(reused_rms_projection.row(token_index)[column]
                  == fused_rms_projection.row(token_index)[column]);
        }
        for (uint32_t column = 0;
             column < quantized_input_scratch.columns(); ++column)
            check(std::isfinite(quantized_input_scratch.row(token_index)[column]));
    }

    TensorData bfloat16_rms_weight;
    bfloat16_rms_weight.dtype = DType::BFloat16;
    bfloat16_rms_weight.shape = {128};
    bfloat16_rms_weight.bfloat16_data.resize(128);
    for (uint32_t column = 0; column < 128; ++column)
    {
        const float value = 1.0f + static_cast<float>(column % 4) * 0.125f;
        bfloat16_rms_weight.bfloat16_data[column] = float_to_bfloat16(value);
    }
    const ActivationBuffer normalized_bfloat16_input = forward_rms_norm(fused_input,
                                                                        bfloat16_rms_weight, 1e-6f, 0.0f);
    const ActivationBuffer reference_bfloat16_rms_projection = forward_linear(float8_gate,
                                                                              normalized_bfloat16_input, g_test_optimization_flags);
    ActivationBuffer reused_bfloat16_rms_projection;
    for (size_t token_index = 0; token_index < quantized_input_scratch.rows();
         ++token_index)
    {
        std::fill_n(quantized_input_scratch.row(token_index),
                    quantized_input_scratch.columns(),
                    std::numeric_limits<float>::quiet_NaN());
    }
    check(forward_linear_rms_norm_float8(float8_gate,
                                         fused_input,
                                         bfloat16_rms_weight,
                                         1e-6f,
                                         reused_bfloat16_rms_projection,
                                         g_test_optimization_flags,
                                         nullptr,
                                         &quantized_input_scratch));
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t column = 0;
             column < reused_bfloat16_rms_projection.columns(); ++column)
        {
            check_near(reused_bfloat16_rms_projection.row(token_index)[column],
                       reference_bfloat16_rms_projection.row(token_index)[column],
                       1e-6f);
        }
        for (uint32_t column = 0;
             column < quantized_input_scratch.columns(); ++column)
            check(std::isfinite(quantized_input_scratch.row(token_index)[column]));
    }
    ActivationBuffer fused_output;
    check(forward_gate_up_float8(float8_gate, float8_up, fused_input, ExpertActivation::Silu, 0.0f,
                                 fused_output,
                                 g_test_optimization_flags));
    check(fused_output.rows() == reference_up.rows());
    check(fused_output.columns() == reference_up.columns());
    for (size_t token_index = 0; token_index < reference_up.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < reference_up.columns(); ++column)
        {
            check_near(fused_output.row(token_index)[column],
                       reference_up.row(token_index)[column], 1e-6f);
        }
    }
    ActivationBuffer fused_quantized_input_scratch;
    ActivationBuffer warm_fused_output;
    check(forward_gate_up_float8(float8_gate, float8_up, fused_input,
                                 ExpertActivation::Silu, 0.0f,
                                 warm_fused_output,
                                 g_test_optimization_flags,
                                 nullptr,
                                 nullptr,
                                 &fused_quantized_input_scratch));
    check(fused_quantized_input_scratch.allocated_bytes() > 0);
    for (size_t index = 0; index < fused_output.values().size(); ++index)
        check_near(warm_fused_output.values()[index], fused_output.values()[index], 1e-6f);
    const uint64_t alternate_float8_flags = g_test_optimization_flags
                                            ^ OptimizationCpuFloat8Bf16Dot
                                            ^ OptimizationCpuFloat8SimdQuantize;
    ActivationBuffer smaller_fused_input(1, fused_input.columns());
    std::copy_n(fused_input.row(0), fused_input.columns(), smaller_fused_input.row(0));
    const std::array<const ActivationBuffer*, 4> reused_fused_inputs = {
        &fused_input, &smaller_fused_input, &larger_fused_input, &fused_input};
    const std::array<uint64_t, 4> reused_fused_flags = {
        g_test_optimization_flags, alternate_float8_flags,
        g_test_optimization_flags, alternate_float8_flags};
    size_t maximum_reused_rows = fused_input.rows();
    for (size_t pass = 0; pass < reused_fused_inputs.size(); ++pass)
    {
        const ActivationBuffer& test_input = *reused_fused_inputs[pass];
        const uint64_t optimization_flags = reused_fused_flags[pass];
        ActivationBuffer reference;
        ActivationBuffer actual;
        check(forward_gate_up_float8(float8_gate, float8_up, test_input,
                                     ExpertActivation::Silu, 0.0f,
                                     reference, optimization_flags));
        const size_t previous_capacity = fused_quantized_input_scratch.allocated_bytes();
        const std::byte* previous_storage = fused_quantized_input_scratch.bytes().data();
        std::fill(fused_quantized_input_scratch.mutable_bytes().begin(),
                  fused_quantized_input_scratch.mutable_bytes().end(),
                  pass % 2 == 0 ? std::byte{0xa5} : std::byte{0x5a});
        check(forward_gate_up_float8(float8_gate, float8_up, test_input,
                                     ExpertActivation::Silu, 0.0f,
                                     actual, optimization_flags,
                                     nullptr, nullptr,
                                     &fused_quantized_input_scratch));
        check(fused_quantized_input_scratch.allocated_bytes() >= previous_capacity);
        if (test_input.rows() <= maximum_reused_rows)
        {
            check(fused_quantized_input_scratch.allocated_bytes() == previous_capacity);
            check(fused_quantized_input_scratch.bytes().data() == previous_storage);
        }
        else
        {
            check(fused_quantized_input_scratch.allocated_bytes()
                  >= test_input.values().size() * sizeof(float));
            maximum_reused_rows = test_input.rows();
        }
        check(actual.rows() == reference.rows());
        check(actual.columns() == reference.columns());
        for (size_t index = 0; index < reference.values().size(); ++index)
            check_near(actual.values()[index], reference.values()[index], 1e-6f);
    }

    TensorData invalid_fused_gate = float8_gate;
    invalid_fused_gate.dtype = DType::BFloat16;
    ActivationBuffer unchanged_fused_output(1, 2);
    std::fill_n(unchanged_fused_output.row(0), unchanged_fused_output.columns(), 17.0f);
    const std::vector<float> unchanged_output_values(unchanged_fused_output.values().begin(),
                                                     unchanged_fused_output.values().end());
    const size_t unchanged_output_capacity = unchanged_fused_output.allocated_bytes();
    const std::byte* unchanged_output_storage = unchanged_fused_output.bytes().data();
    const size_t unchanged_scratch_capacity = fused_quantized_input_scratch.allocated_bytes();
    const std::byte* unchanged_scratch_storage = fused_quantized_input_scratch.bytes().data();
    std::fill(fused_quantized_input_scratch.mutable_bytes().begin(),
              fused_quantized_input_scratch.mutable_bytes().end(),
              std::byte{0x69});
    const std::vector<std::byte> unchanged_quantized_values(fused_quantized_input_scratch.bytes().begin(),
                                                            fused_quantized_input_scratch.bytes().end());
    check(!forward_gate_up_float8(invalid_fused_gate, float8_up, fused_input,
                                  ExpertActivation::Silu, 0.0f,
                                  unchanged_fused_output,
                                  g_test_optimization_flags,
                                  nullptr,
                                  nullptr,
                                  &fused_quantized_input_scratch));
    check(std::equal(unchanged_fused_output.values().begin(),
                     unchanged_fused_output.values().end(),
                     unchanged_output_values.begin(),
                     unchanged_output_values.end()));
    check(unchanged_fused_output.allocated_bytes() == unchanged_output_capacity);
    check(unchanged_fused_output.bytes().data() == unchanged_output_storage);
    check(fused_quantized_input_scratch.allocated_bytes() == unchanged_scratch_capacity);
    check(fused_quantized_input_scratch.bytes().data() == unchanged_scratch_storage);
    check(std::equal(fused_quantized_input_scratch.bytes().begin(),
                     fused_quantized_input_scratch.bytes().end(),
                     unchanged_quantized_values.begin(),
                     unchanged_quantized_values.end()));

    constexpr float deepseek_limit = 0.25f;
    ActivationBuffer deepseek_reference_up = forward_linear(float8_up,
                                                            fused_input,
                                                            g_test_optimization_flags);
    const ActivationBuffer deepseek_reference_gate = forward_linear(float8_gate,
                                                                    fused_input,
                                                                    g_test_optimization_flags);
    for (size_t token_index = 0;
         token_index < deepseek_reference_up.rows(); ++token_index)
    {
        for (uint32_t column = 0;
             column < deepseek_reference_up.columns(); ++column)
        {
            const float gate_value = std::min(deepseek_reference_gate.row(token_index)[column],
                                              deepseek_limit);
            const float up_value = std::clamp(deepseek_reference_up.row(token_index)[column],
                                              -deepseek_limit,
                                              deepseek_limit);
            deepseek_reference_up.row(token_index)[column] = up_value * gate_value / (1.0f + std::exp(-gate_value));
        }
    }
    ActivationBuffer deepseek_fused_output;
    check(forward_gate_up_float8(float8_gate, float8_up, fused_input,
                                 ExpertActivation::DeepSeekSwiGlu, deepseek_limit,
                                 deepseek_fused_output,
                                 g_test_optimization_flags));
    for (size_t token_index = 0;
         token_index < deepseek_reference_up.rows(); ++token_index)
    {
        for (uint32_t column = 0;
             column < deepseek_reference_up.columns(); ++column)
        {
            check_near(deepseek_fused_output.row(token_index)[column],
                       deepseek_reference_up.row(token_index)[column], 1e-6f);
        }
    }

    const TensorData scale_boundary_matrix = make_float8_projection(3, 136);
    const ActivationBuffer scale_boundary_output = forward_linear(scale_boundary_matrix,
                                                                  fused_input,
                                                                  g_test_optimization_flags);
    ActivationBuffer quantized_boundary_input = fused_input;
    for (size_t token_index = 0;
         token_index < quantized_boundary_input.rows(); ++token_index)
    {
        quantize_float8_e4m3_inplace(quantized_boundary_input.row(token_index),
                                     128,
                                     128,
                                     true,
                                     g_test_optimization_flags);
    }
    for (size_t token_index = 0; token_index < fused_input.rows();
         ++token_index)
    {
        for (uint32_t row = 0; row < 136; ++row)
        {
            const float expected = float8_e4m3_block_dot(scale_boundary_matrix.float8_values().data()
                                                             + static_cast<size_t>(row) * 128,
                                                         scale_boundary_matrix.quantization_scales.data() + row / 128,
                                                         quantized_boundary_input.row(token_index), 128, 128);
            check_near(scale_boundary_output.row(token_index)[row], expected,
                       1e-3f);
        }
    }
}

static ModelPackage deepseek_v4_package(const std::string& dspark_fields)
{
    ModelPackage package;
    package.manifest.model_type = "deepseek_v4";
    package.manifest.raw_json = R"({
        "vocab_size": 129280,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 4,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "sliding_window": 128,
        "max_position_embeddings": 1048576,
        "original_max_position_embeddings": 65536,
        "num_hash_layers": 3,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "compress_ratios": [0, 0, 4, 128],
)" + dspark_fields + R"(
        "expert_dtype": "fp4",
        "scoring_func": "sqrtsoftplus",
        "quant_method": "fp8",
        "fmt": "e4m3",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
        "rms_norm_eps": 0.000001,
        "hc_eps": 0.000001,
        "swiglu_limit": 10.0,
        "routed_scaling_factor": 1.5
    })";
    return package;
}

void test_deepseek_v4_descriptors()
{
    DeepSeekV4ModelAdapter adapter;
    auto flash = adapter.parse_model(deepseek_v4_package(""));
    check(static_cast<bool>(flash));
    check(flash.value().speculative_layer_count == 0);
    check(flash.value().speculative_target_layer_ids.empty());
    check(flash.value().layers.front().attention.initial_context_length == 65536);

    ModelPackage nested_initial_context = deepseek_v4_package("");
    nested_initial_context.manifest.raw_json = std::regex_replace(nested_initial_context.manifest.raw_json,
                                                                  std::regex(R"(\s*"original_max_position_embeddings"\s*:\s*65536\s*,\s*)"),
                                                                  "\n        ");
    nested_initial_context.manifest.raw_json = std::regex_replace(nested_initial_context.manifest.raw_json,
                                                                  std::regex(R"("expert_dtype"\s*:\s*"fp4")"),
                                                                  R"("rope_scaling":{"original_max_position_embeddings":65536},"expert_dtype":"fp4")");
    auto nested_initial_context_result = adapter.parse_model(nested_initial_context);
    check(static_cast<bool>(nested_initial_context_result));
    check(nested_initial_context_result.value().layers.front().attention.initial_context_length == 65536);

    ModelPackage conflicting_initial_context = deepseek_v4_package("");
    conflicting_initial_context.manifest.raw_json = std::regex_replace(conflicting_initial_context.manifest.raw_json,
                                                                       std::regex(R"("expert_dtype"\s*:\s*"fp4")"),
                                                                       R"("rope_scaling":{"original_max_position_embeddings":32768},"expert_dtype":"fp4")");
    auto conflicting_initial_context_result = adapter.parse_model(conflicting_initial_context);
    check(static_cast<bool>(conflicting_initial_context_result));
    check(conflicting_initial_context_result.value().layers.front().attention.initial_context_length == 32768);

    ModelPackage malformed_initial_context = deepseek_v4_package("");
    malformed_initial_context.manifest.raw_json = std::regex_replace(malformed_initial_context.manifest.raw_json,
                                                                     std::regex(R"("expert_dtype"\s*:\s*"fp4")"),
                                                                     R"("rope_scaling":{"original_max_position_embeddings":"bad"},"expert_dtype":"fp4")");
    auto malformed_initial_context_result = adapter.parse_model(malformed_initial_context);
    check(!malformed_initial_context_result);
    check(malformed_initial_context_result.error().message
          == "invalid DeepSeek-V4 integer field: original_max_position_embeddings");

    ModelPackage nested_quantization = deepseek_v4_package("");
    nested_quantization.manifest.raw_json = std::regex_replace(nested_quantization.manifest.raw_json,
                                                               std::regex(R"("quant_method"\s*:\s*"fp8")"),
                                                               R"("quantization_config":{"quant_method":"fp8","fmt":"e4m3","scale_fmt":"ue8m0","weight_block_size":[128,128]},"quant_method":"invalid")");
    auto nested_quantization_result = adapter.parse_model(nested_quantization);
    check(static_cast<bool>(nested_quantization_result));

    ModelPackage invalid_quantization = deepseek_v4_package("");
    invalid_quantization.manifest.raw_json = std::regex_replace(invalid_quantization.manifest.raw_json,
                                                                std::regex(R"("expert_dtype"\s*:)"),
                                                                R"("quantization_config":[1],"expert_dtype":)");
    auto invalid_quantization_result = adapter.parse_model(invalid_quantization);
    check(!invalid_quantization_result);
    check(invalid_quantization_result.error().message
          == "invalid DeepSeek-V4 object field: quantization_config");

    ModelPackage nested_rope_scaling = deepseek_v4_package("");
    nested_rope_scaling.manifest.raw_json = std::regex_replace(nested_rope_scaling.manifest.raw_json,
                                                               std::regex(R"("expert_dtype"\s*:\s*"fp4")"),
                                                               R"("rope_scaling":{"factor":24.0,"beta_slow":0.5,"beta_fast":16.0},"factor":3.0,"expert_dtype":"fp4")");
    auto nested_rope_scaling_result = adapter.parse_model(nested_rope_scaling);
    check(static_cast<bool>(nested_rope_scaling_result));
    check_near(nested_rope_scaling_result.value().layers.front().attention.rope_scaling_factor,
               24.0f,
               1e-6f);

    auto parsed = adapter.parse_model(deepseek_v4_package(R"(
        "dspark_target_layer_ids": [1, 2, 3],
        "dspark_block_size": 5,
        "dspark_noise_token_id": 127,
        "dspark_markov_rank": 256,
)"));
    check(static_cast<bool>(parsed));
    const MoeModelDescriptor& descriptor = parsed.value();
    check(descriptor.model_type == "deepseek_v4");
    check(descriptor.hyper_connection_multiplier == 4);
    check(descriptor.hash_routing_layer_count == 3);
    check(descriptor.speculative_kind == SpeculativeModelKind::DSpark);
    check(descriptor.speculative_layer_count == 3);
    check(descriptor.speculative_block_size == 5);
    check(descriptor.speculative_noise_token_id == 127);
    check(descriptor.speculative_markov_rank == 256);
    check(descriptor.layers[2].attention.kind == AttentionKind::MultiHeadLatent);
    check(descriptor.layers[2].attention.compression_ratio == 4);
    check(descriptor.layers[3].attention.compression_ratio == 128);
    check(descriptor.layers[0].moe.score_function == RouterScoreFunction::SqrtSoftplus);
    check(descriptor.layers[0].moe.shared_expert_count == 1);
    Option options;
    auto memory = plan_model_memory(descriptor, options, UINT64_C(8) * 1024 * 1024 * 1024);
    check(static_cast<bool>(memory));
    check(memory.value().selected_mode == ExpertMemoryMode::OnDemand);
    check(memory.value().estimated_dense_size < UINT64_C(10) * 1024 * 1024 * 1024);

    Option release_options;
    release_options.hybrid_mode = HybridMode::HybridExperts;
    release_options.expert_memory_mode = ExpertMemoryMode::OnDemand;
    const uint64_t physical_memory = UINT64_C(32) * 1024 * 1024 * 1024;
    for (DType projection_dtype : {DType::Float8E4M3, DType::BFloat16})
    {
        MoeModelDescriptor resident = flash.value();
        uint64_t reclaimed = static_cast<uint64_t>(resident.vocabulary_size) * resident.hidden_size * 2;
        for (LayerDescriptor& layer : resident.layers)
        {
            layer.attention.projection_weight_dtype = projection_dtype;
            layer.moe.shared_expert_weight_dtype = projection_dtype;
            const AttentionDescriptor& attention = layer.attention;
            const uint64_t grouped_columns = static_cast<uint64_t>(attention.head_count) * attention.head_dimension / attention.output_group_count;
            const std::array<std::pair<uint64_t, uint64_t>, 5> projections = {{{attention.query_lora_rank, resident.hidden_size},
                                                                               {static_cast<uint64_t>(attention.head_count) * attention.head_dimension, attention.query_lora_rank},
                                                                               {attention.head_dimension, resident.hidden_size},
                                                                               {static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank, grouped_columns},
                                                                               {resident.hidden_size, static_cast<uint64_t>(attention.output_group_count) * attention.output_lora_rank}}};
            const auto add_projection = [&](uint64_t rows, uint64_t columns) {
                reclaimed += rows * columns * (projection_dtype == DType::BFloat16 ? 2 : 1);
                if (projection_dtype == DType::Float8E4M3)
                    reclaimed += ((rows + 127) / 128) * ((columns + 127) / 128) * sizeof(float);
            };
            for (const auto& projection : projections)
                add_projection(projection.first, projection.second);
            if (attention.compression_ratio == 4)
                add_projection(static_cast<uint64_t>(attention.index_head_count) * attention.index_head_dimension, attention.query_lora_rank);
            if (projection_dtype == DType::BFloat16 && layer.moe.shared_expert_count != 0)
                reclaimed += static_cast<uint64_t>(resident.hidden_size) * layer.moe.intermediate_size * 3 * 2;
        }
        Option cpu_options = release_options;
        cpu_options.hybrid_mode = HybridMode::CpuOnly;
        auto retained = plan_model_memory(resident, cpu_options, physical_memory);
        auto released = plan_model_memory(resident, release_options, physical_memory, true);
        check(static_cast<bool>(retained) && static_cast<bool>(released));
        check(retained.value().estimated_dense_size > reclaimed);
        check(released.value().estimated_dense_size == retained.value().estimated_dense_size - reclaimed);
        check(released.value().expert_cache_size == retained.value().expert_cache_size + reclaimed);
        check(released.value().expert_cache_size + released.value().estimated_dense_size
              == retained.value().expert_cache_size + retained.value().estimated_dense_size);
        ModelMemoryPlan fallback_plan = released.value();
        check(static_cast<bool>(finalize_model_memory_plan(fallback_plan, release_options, 0)));
        check(fallback_plan.estimated_dense_size == retained.value().estimated_dense_size);
        check(fallback_plan.expert_cache_size == retained.value().expert_cache_size);
        ModelMemoryPlan resident_plan = released.value();
        check(static_cast<bool>(finalize_model_memory_plan(resident_plan, release_options, reclaimed)));
        check(resident_plan.estimated_dense_size == released.value().estimated_dense_size);
        check(resident_plan.expert_cache_size == released.value().expert_cache_size);
        check(static_cast<bool>(finalize_model_memory_plan(resident_plan, release_options, reclaimed)));
        check(resident_plan.estimated_dense_size == released.value().estimated_dense_size);
        ModelMemoryPlan partial_plan = released.value();
        check(static_cast<bool>(finalize_model_memory_plan(partial_plan, release_options, reclaimed / 2)));
        check(partial_plan.estimated_dense_size == retained.value().estimated_dense_size - reclaimed / 2);
        check(partial_plan.expert_cache_size == retained.value().expert_cache_size + reclaimed / 2);
        Option explicit_cache = release_options;
        explicit_cache.expert_cache_size = retained.value().host_memory_budget - retained.value().estimated_dense_size + 1;
        ModelMemoryPlan explicit_plan = released.value();
        explicit_plan.expert_cache_size = explicit_cache.expert_cache_size;
        auto rejected_cache = finalize_model_memory_plan(explicit_plan, explicit_cache, 0);
        check(!rejected_cache && rejected_cache.error().code == ErrorCode::InvalidArgument);
        check(explicit_plan.expert_cache_size == explicit_cache.expert_cache_size);
    }

    auto partial_dspark = adapter.parse_model(deepseek_v4_package(R"(
        "dspark_block_size": 5,
)"));
    check(!partial_dspark);

    ModelPackage invalid = deepseek_v4_package("");
    invalid.manifest.raw_json = std::regex_replace(invalid.manifest.raw_json,
                                                   std::regex(R"("o_groups"\s*:\s*8)"),
                                                   R"("o_groups": 0)");
    auto zero_output_groups = adapter.parse_model(invalid);
    check(!zero_output_groups);
    check(zero_output_groups.error().code == ErrorCode::InvalidModel);
    check(zero_output_groups.error().message
          == "unsupported DeepSeek-V4 architectural dimensions");
}

} // namespace moe
} // namespace ncnn
