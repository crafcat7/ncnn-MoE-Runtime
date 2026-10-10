#include "backends/ncnn/linear.h"
#include "backends/ncnn/router_vulkan.h"
#include "kernels/ops.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static void router_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static void router_check_plan(const ExpertDispatchPlan& actual,
                              const ExpertDispatchPlan& expected,
                              const std::string& label)
{
    router_check(actual.assignment_count == expected.assignment_count && actual.batches.size() == expected.batches.size(),
                 label + ": batch shape");
    for (size_t batch = 0; batch < actual.batches.size(); ++batch)
    {
        const ExpertBatch& observed = actual.batches[batch];
        const ExpertBatch& reference = expected.batches[batch];
        router_check(observed.expert_id == reference.expert_id && observed.routes.size() == reference.routes.size(),
                     label + ": expert ids");
        for (size_t route = 0; route < observed.routes.size(); ++route)
        {
            const ExpertRoute& value = observed.routes[route];
            const ExpertRoute& expected_value = reference.routes[route];
            router_check(value.token_index == expected_value.token_index && value.rank == expected_value.rank,
                         label + ": route order");
            router_check(std::isfinite(value.weight) && std::abs(value.weight - expected_value.weight) < 0.00001f + 0.00003f * std::abs(expected_value.weight),
                         label + ": route weight actual=" + std::to_string(value.weight) + " expected=" + std::to_string(expected_value.weight));
        }
    }
}

static TensorData router_identity(uint32_t size, DType dtype = DType::Float32)
{
    TensorData weight;
    weight.dtype = dtype;
    weight.shape = {size, size};
    if (dtype == DType::Float32)
    {
        weight.float32_data.assign(static_cast<size_t>(size) * size, 0.0f);
        for (uint32_t expert = 0; expert < size; ++expert)
            weight.float32_data[static_cast<size_t>(expert) * size + expert] = 1.0f;
    }
    else
    {
        weight.bfloat16_data.assign(static_cast<size_t>(size) * size, 0);
        for (uint32_t expert = 0; expert < size; ++expert)
            weight.bfloat16_data[static_cast<size_t>(expert) * size + expert] = float_to_bfloat16(1.0f);
    }
    return weight;
}

#if NCNN_MOE_WITH_VULKAN
static ActivationBuffer router_run(const Router_vulkan& router,
                                   const ActivationBuffer& input,
                                   const ExpertDispatchOptions& options,
                                   bool expect_record = true)
{
    const auto& context = router.context();
    const auto before = get_vulkan_statistics(context->runtime());
    RouterWorkspace_vulkan workspace;
    ncnn::VkMat staging;
    ncnn::VkMat device_input;
    ncnn::VkMat download;
    ncnn::VkCompute cmd(context->device(), context->command_optimization_flags());
    ActivationBuffer selected(input.rows(),
                              Router_vulkan::selected_columns(options.top_k));
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    router_check(fill_staging_upload(input, staging, context->staging_allocator()),
                 "router upload staging");
    router_check(record_mapped_activation_upload(staging, device_input, cmd,
                                                 context->device(),
                                                 router.option(), input.dtype()),
                 "router input upload");
    const bool recorded = router.record(device_input, options, workspace, cmd);
    router_check(recorded == expect_record, "router record validity");
    if (!recorded)
        return {};
    router_check(prepare_staging_batch(download, input.rows(), selected.columns(),
                                       context->staging_allocator()),
                 "router selected staging");
    router_check(record_prepared_activation_staging_download(workspace.selected, input.rows(), selected.columns(),
                                                             download, cmd, context->device(), router.option()),
                 "router compact selected download");
    router_check(submit_compute_and_wait(cmd, context->device()) == 0,
                 "router submit");
    ++context->runtime_state().compute_submissions;
    ++context->runtime_state().batch_uploads;
    ++context->runtime_state().batch_downloads;
    router_check(copy_staging_to_cpu_batch(download, selected),
                 "router compact result");
    const auto after = get_vulkan_statistics(context->runtime());
    router_check(after.compute_submissions == before.compute_submissions + 1 && after.batch_downloads == before.batch_downloads + 1,
                 "router projection and selection use one submission and one "
                 "compact download");
    router_check(selected.bytes().size() == input.rows() * (options.top_k * 2 + 1) * sizeof(float),
                 "router compact download size");
    return selected;
}

static void router_test_scores()
{
    constexpr uint32_t count = 256;
    const auto runtime = create_vulkan_runtime();
    const TensorData weight = router_identity(count);
    TensorData selection_bias;
    selection_bias.dtype = DType::Float32;
    selection_bias.shape = {count};
    selection_bias.float32_data.resize(count);
    for (uint32_t expert = 0; expert < count; ++expert)
        selection_bias.float32_data[expert] = static_cast<float>((expert * 31) % 17) * 0.01f;
    auto router = Router_vulkan::create(weight, nullptr, &selection_bias, 0,
                                        runtime, OptimizationDefaultFlags);
    router_check(static_cast<bool>(router), "router create");
    ActivationBuffer input(3, count);
    for (size_t row = 0; row < input.rows(); ++row)
        for (uint32_t expert = 0; expert < count; ++expert)
            input.row(row)[expert] = static_cast<float>((expert * 19 + row * 3) % count) * 0.0625f - 8.0f;
    for (RouterScoreFunction score :
         {RouterScoreFunction::Softmax, RouterScoreFunction::Sigmoid,
          RouterScoreFunction::SqrtSoftplus})
    {
        for (RouterNormalization normalization :
             {RouterNormalization::SelectedExperts, RouterNormalization::None})
        {
            ExpertDispatchOptions options;
            options.expert_count = count;
            options.top_k = 8;
            options.score_function = score;
            options.normalization = normalization;
            options.routed_scaling_factor = 2.5f;
            for (bool with_bias : {false, true})
            {
                options.selection_bias = with_bias ? selection_bias.float32_values()
                                                   : std::span<const float>{};
                ExpertDispatchPlan expected;
                ExpertDispatchPlan actual;
                router_check(static_cast<bool>(forward_router(input.values(), static_cast<uint32_t>(input.rows()),
                                                              options, expected)),
                             "CPU router oracle");
                const ActivationBuffer selected = router_run(*router, input, options);
                router_check(static_cast<bool>(Router_vulkan::decode_selected(selected, options, actual)),
                             "GPU selected decode");
                router_check_plan(actual, expected, "score/bias/normalization");
            }
        }
    }
    ExpertDispatchOptions options;
    options.expert_count = count;
    options.top_k = 8;
    // Dynamic selection bias differs from the resident factory bias.
    selection_bias.float32_data[0] = 3.0f;
    options.selection_bias = selection_bias.float32_values();
    ExpertDispatchPlan expected;
    ExpertDispatchPlan actual;
    router_check(static_cast<bool>(forward_router(input.values(), 3, options, expected)),
                 "dynamic bias CPU oracle");
    router_check(static_cast<bool>(Router_vulkan::decode_selected(router_run(*router, input, options), options, actual)),
                 "dynamic bias GPU decode");
    router_check_plan(actual, expected, "dynamic bias");
    const std::vector<uint32_t> ids = {4, 4, 2, 255, 0, 1, 2, 3,
                                       9, 8, 7, 6, 5, 4, 3, 2,
                                       255, 2, 255, 3, 255, 4, 255, 5};
    options.explicit_expert_ids = ids;
    router_check(static_cast<bool>(forward_router(input.values(), 3, options, expected)),
                 "hash router CPU oracle");
    router_check(static_cast<bool>(Router_vulkan::decode_selected(router_run(*router, input, options), options, actual)),
                 "hash router GPU decode");
    router_check_plan(actual, expected,
                      "hash router including duplicate expert ids");
}

static void router_test_ties_and_extremes()
{
    const TensorData weight = router_identity(32, DType::BFloat16);
    TensorData bias;
    bias.dtype = DType::BFloat16;
    bias.shape = {32};
    bias.bfloat16_data.assign(32, float_to_bfloat16(0.5f));
    auto router = Router_vulkan::create(weight, &bias, nullptr, 0, create_vulkan_runtime(),
                                        OptimizationDefaultFlags);
    router_check(static_cast<bool>(router), "BF16 router create with bias");
    ActivationBuffer input(1, 32);
    ExpertDispatchOptions options;
    options.expert_count = 32;
    options.top_k = 16;
    for (RouterScoreFunction score :
         {RouterScoreFunction::Softmax, RouterScoreFunction::Sigmoid,
          RouterScoreFunction::SqrtSoftplus})
    {
        options.score_function = score;
        std::fill(input.mutable_bytes().begin(), input.mutable_bytes().end(),
                  std::byte{0});
        ActivationBuffer logits = input;
        for (float& value : std::span<float>(logits.row(0), 32))
            value += 0.5f;
        ExpertDispatchPlan expected;
        ExpertDispatchPlan actual;
        router_check(static_cast<bool>(forward_router(logits.values(), 1, options, expected)),
                     "tie CPU oracle");
        router_check(static_cast<bool>(Router_vulkan::decode_selected(router_run(*router, input, options), options, actual)),
                     "tie GPU decode");
        router_check_plan(actual, expected, "tie uses expert id ascending");
        for (size_t batch = 0; batch < actual.batches.size(); ++batch)
            router_check(actual.batches[batch].expert_id == batch,
                         "tie selected lowest expert ids");
        for (uint32_t expert = 0; expert < 32; ++expert)
            input.row(0)[expert] = -95.0f + static_cast<float>(expert) * 0.5f;
        logits = input;
        for (float& value : std::span<float>(logits.row(0), 32))
            value += 0.5f;
        router_check(static_cast<bool>(forward_router(logits.values(), 1, options, expected)),
                     "extreme CPU oracle");
        router_check(static_cast<bool>(Router_vulkan::decode_selected(router_run(*router, input, options), options, actual)),
                     "extreme GPU decode");
        router_check_plan(actual, expected, "negative extreme logits");
    }
    options.top_k = 32;
    options.score_function = RouterScoreFunction::SqrtSoftplus;
    options.normalization = RouterNormalization::None;
    const ActivationBuffer selected = router_run(*router, input, options);
    ExpertDispatchPlan plan;
    router_check(static_cast<bool>(Router_vulkan::decode_selected(selected, options, plan))
                     && plan.assignment_count == 32,
                 "top-k equals expert count");
}

static void router_test_invalid()
{
    const TensorData weight = router_identity(8);
    auto router = Router_vulkan::create(weight, nullptr, nullptr, 0,
                                        create_vulkan_runtime(), OptimizationDefaultFlags);
    router_check(static_cast<bool>(router), "invalid test router create");
    ActivationBuffer input(2, 8);
    ExpertDispatchOptions options;
    options.expert_count = 8;
    options.top_k = 3;
    ExpertDispatchPlan plan;
    router_check(static_cast<bool>(forward_router(input.values(), 2, options, plan)),
                 "seed valid plan");
    const ExpertDispatchPlan previous = plan;
    input.row(1)[0] = std::numeric_limits<float>::infinity();
    const ActivationBuffer selected = router_run(*router, input, options);
    router_check(!Router_vulkan::decode_selected(selected, options, plan),
                 "GPU rejects nonfinite logits");
    router_check_plan(plan, previous, "late row error preserves plan");
    input.row(1)[0] = 0.0f;
    std::vector<uint32_t> invalid_ids(6, 9);
    options.explicit_expert_ids = invalid_ids;
    (void)router_run(*router, input, options, false);
    options.explicit_expert_ids = {};
    options.routed_scaling_factor = 0.0f;
    (void)router_run(*router, input, options, false);
    options.routed_scaling_factor = 1.0f;
    options.score_function = RouterScoreFunction::Sigmoid;
    for (size_t row = 0; row < 2; ++row)
        std::fill_n(input.row(row), 8, -1000.0f);
    router_check(!Router_vulkan::decode_selected(router_run(*router, input, options), options, plan),
                 "GPU rejects zero selected normalization sum");
    router_check_plan(plan, previous, "normalization error preserves plan");
    options.normalization = RouterNormalization::None;
    router_check(static_cast<bool>(Router_vulkan::decode_selected(router_run(*router, input, options), options, plan)),
                 "zero weights valid without normalization");
}
#endif

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
        ncnn::moe::router_test_scores();
        ncnn::moe::router_test_ties_and_extremes();
        ncnn::moe::router_test_invalid();
        std::cout << "Vulkan router tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Vulkan router test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
