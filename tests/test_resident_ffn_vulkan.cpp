#include "backends/ncnn/moecombine_vulkan.h"
#include "backends/ncnn/residentffn_vulkan.h"
#include "backends/ncnn/vulkancontext.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <command.h>
#include <mat.h>
#include <option.h>
#endif

namespace ncnn {
namespace moe {

static void region_check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

#if NCNN_MOE_WITH_VULKAN
static ncnn::Option region_option(const VulkanContext& context)
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

static bool same_io_counters(const VulkanStatistics& first, const VulkanStatistics& second)
{
    return first.compute_submissions == second.compute_submissions
           && first.batch_uploads == second.batch_uploads
           && first.batch_downloads == second.batch_downloads;
}

static ncnn::VkMat upload_fixture(const std::shared_ptr<VulkanContext>& context, const ActivationBuffer& source)
{
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const auto option = region_option(*context);
    ncnn::VkCompute command(context->device());
    ncnn::VkMat staging;
    ncnn::VkMat result;
    region_check(fill_staging_upload(source, staging, option.staging_vkallocator)
                     && record_mapped_upload(staging, result, command, option)
                     && submit_compute_and_wait(command, context->device()) == 0,
                 "resident FFN fixture upload failed");
    ++context->runtime_state().compute_submissions;
    ++context->runtime_state().batch_uploads;
    return result;
}

static void complete_existing_wave(const std::shared_ptr<VulkanContext>& context,
                                   ResidentFfnRegion_vulkan& region, ncnn::VkCompute& command,
                                   bool uploaded, bool downloaded)
{
    region_check(submit_compute_and_wait(command, context->device()) == 0,
                 "the existing Expert wave command failed");
    const auto before = get_vulkan_statistics(context->runtime());
    region_check(region.complete_wave(), "resident FFN wave publication failed");
    region_check(same_io_counters(before, get_vulkan_statistics(context->runtime())),
                 "resident FFN completion submitted or counted an extra command");
    // The Expert wave owner publishes these counters, as in production. The
    // region only returns flags for its first input and final aggregate.
    ++context->runtime_state().compute_submissions;
    if (uploaded) ++context->runtime_state().batch_uploads;
    if (downloaded) ++context->runtime_state().batch_downloads;
}

static void check_unpublished(ResidentFfnRegion_vulkan& region, ActivationBuffer& output)
{
    const auto* storage = output.values().data();
    const std::vector<float> values(output.values().begin(), output.values().end());
    const size_t rows = output.rows();
    const uint32_t columns = output.columns();
    region_check(!region.complete() && !region.take_output(output),
                 "an incomplete resident FFN region published an output");
    region_check(output.rows() == rows && output.columns() == columns && output.values().data() == storage
                     && std::equal(values.begin(), values.end(), output.values().begin()),
                 "failed resident FFN take_output changed the caller's buffer");
}

static ActivationBuffer canonical_oracle(const std::array<ActivationBuffer, 2>& values,
                                         const std::array<std::vector<ExpertRoute>, 2>& routes,
                                         size_t rows, uint32_t columns)
{
    ActivationBuffer result(rows, columns);
    std::fill_n(result.row(0), result.values().size(), 0.0f);
    for (size_t expert = 0; expert < values.size(); ++expert)
        for (size_t route = 0; route < routes[expert].size(); ++route)
            for (uint32_t column = 0; column < columns; ++column)
            {
                const float product = routes[expert][route].weight * values[expert].row(route)[column];
                result.row(routes[expert][route].token_index)[column] += product;
            }
    return result;
}

static void check_output(const ActivationBuffer& actual, const ActivationBuffer& expected)
{
    region_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(),
                 "resident FFN aggregate has the wrong shape");
    for (size_t index = 0; index < expected.values().size(); ++index)
        region_check(std::isfinite(actual.values()[index])
                         && std::abs(actual.values()[index] - expected.values()[index]) <= 1e-6f,
                     "resident FFN aggregate differs from the canonical CPU oracle");
}

static void test_two_waves(const std::shared_ptr<VulkanContext>& context,
                           const std::shared_ptr<MoeCombine_vulkan>& combiner)
{
    constexpr uint32_t columns = 8;
    ActivationBuffer normalized(4, columns);
    for (size_t index = 0; index < normalized.values().size(); ++index)
        normalized.row(0)[index] = static_cast<float>(index % 13) * 0.125f;
    std::array<ActivationBuffer, 2> values = {ActivationBuffer(3, columns), ActivationBuffer(3, columns)};
    for (size_t expert = 0; expert < values.size(); ++expert)
        for (size_t row = 0; row < values[expert].rows(); ++row)
            for (uint32_t column = 0; column < columns; ++column)
                values[expert].row(row)[column] = static_cast<float>(expert * 8 + row * 2 + column) * 0.125f;
    values[0].row(0)[0] = 16777216.0f;
    values[0].row(1)[0] = 1.0f;
    values[1].row(0)[0] = -16777216.0f;
    const std::array<std::vector<ExpertRoute>, 2> routes = {{
        {{0, 0, 1.0f}, {0, 1, 1.0f}, {2, 0, 0.5f}},
        {{0, 2, 1.0f}, {1, 0, 0.25f}, {2, 1, -0.5f}},
    }};
    const ActivationBuffer expected = canonical_oracle(values, routes, normalized.rows(), columns);
    region_check(expected.row(0)[0] == 0.0f,
                 "canonical cancellation fixture must distinguish Expert order from arrival order");
    const std::array<ncnn::VkMat, 2> device_values = {upload_fixture(context, values[0]), upload_fixture(context, values[1])};
    auto region = ResidentFfnRegion_vulkan::create(normalized, 2, columns);
    region_check(region && region->bind_context(context, combiner), "resident FFN region binding failed");
    ActivationBuffer output(2, 3);
    std::fill_n(output.row(0), output.values().size(), -37.0f);
    check_unpublished(*region, output);
    const auto initial = get_vulkan_statistics(context->runtime());
    ncnn::VkMat retained_input;
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        const auto option = region_option(*context);
        {
            ncnn::VkCompute aborted(context->device());
            ncnn::VkMat input;
            bool uploaded = false;
            bool downloaded = true;
            region_check(region->record_input(input, aborted, option, uploaded) && uploaded,
                         "an aborted first wave must record its initial upload");
            const std::array<ResidentFfnRegion_vulkan::Output, 2> duplicate = {{
                {1, device_values[1], routes[1]},
                {1, device_values[1], routes[1]},
            }};
            region_check(!region->record_outputs(duplicate, aborted, option, downloaded) && !downloaded,
                         "duplicate pending Expert slots must be rejected");
            region_check(aborted.reset() == 0, "an unsubmitted failed wave command did not reset");
            region->abort_wave();
            region_check(same_io_counters(initial, get_vulkan_statistics(context->runtime())),
                         "aborted region recording submitted or published transfer counters");
            check_unpublished(*region, output);
        }
        ncnn::VkCompute first(context->device());
        bool uploaded = false;
        bool downloaded = true;
        region_check(region->record_input(retained_input, first, option, uploaded) && uploaded,
                     "first committed Expert wave must record one normalized upload");
        const std::array<ResidentFfnRegion_vulkan::Output, 1> first_outputs = {{{1, device_values[1], routes[1]}}};
        region_check(region->record_outputs(first_outputs, first, option, downloaded) && !downloaded,
                     "a partial Expert wave must retain its device output without downloading");
        region_check(same_io_counters(initial, get_vulkan_statistics(context->runtime())),
                     "recording the first wave submitted an independent region command");
        complete_existing_wave(context, *region, first, uploaded, downloaded);
    }
    check_unpublished(*region, output);
    const auto partial = get_vulkan_statistics(context->runtime());
    region_check(partial.compute_submissions == initial.compute_submissions + 1
                     && partial.batch_uploads == initial.batch_uploads + 1
                     && partial.batch_downloads == initial.batch_downloads,
                 "first partial wave did not keep the expected existing-command transfer counts");
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        const auto option = region_option(*context);
        {
            ncnn::VkCompute aborted(context->device());
            ncnn::VkMat input;
            bool uploaded = true;
            bool downloaded = true;
            region_check(region->record_input(input, aborted, option, uploaded) && !uploaded,
                         "a committed normalized input must survive an aborted later wave");
            const std::array<ResidentFfnRegion_vulkan::Output, 1> duplicate = {{{1, device_values[1], routes[1]}}};
            region_check(!region->record_outputs(duplicate, aborted, option, downloaded) && !downloaded,
                         "an already committed Expert slot must reject a duplicate");
            region_check(aborted.reset() == 0, "a duplicate committed-slot wave command did not reset");
            region->abort_wave();
            check_unpublished(*region, output);
        }
        ncnn::VkCompute final(context->device());
        ncnn::VkMat input;
        bool uploaded = true;
        bool downloaded = false;
        region_check(region->record_input(input, final, option, uploaded) && !uploaded
                         && input.buffer() == retained_input.buffer() && input.buffer_offset() == retained_input.buffer_offset(),
                     "later Expert waves must reuse the exact committed normalized device input");
        const std::array<ResidentFfnRegion_vulkan::Output, 1> final_outputs = {{{0, device_values[0], routes[0]}}};
        region_check(region->record_outputs(final_outputs, final, option, downloaded) && downloaded,
                     "the last Expert wave must record the final aggregate download");
        region_check(same_io_counters(partial, get_vulkan_statistics(context->runtime())),
                     "recording the final reduction submitted or counted an independent command");
        check_unpublished(*region, output);
        complete_existing_wave(context, *region, final, uploaded, downloaded);
    }
    const auto completed = get_vulkan_statistics(context->runtime());
    region_check(completed.compute_submissions == initial.compute_submissions + 2
                     && completed.batch_uploads == initial.batch_uploads + 1
                     && completed.batch_downloads == initial.batch_downloads + 1,
                 "two Expert waves must use two existing submissions, one input upload and one final download");
    region_check(region->complete() && region->take_output(output) && !region->complete(),
                 "resident FFN aggregate must publish exactly once after the last completed wave");
    check_output(output, expected);
    region_check(output.row(0)[0] == 0.0f,
                 "out-of-order arrival changed the canonical floating-point Expert reduction");
    check_unpublished(*region, output);
}

static void test_single_warm_wave(const std::shared_ptr<VulkanContext>& context,
                                  const std::shared_ptr<MoeCombine_vulkan>& combiner)
{
    ActivationBuffer normalized(3, 8);
    std::fill_n(normalized.row(0), normalized.values().size(), 0.25f);
    ActivationBuffer values(2, 8);
    for (size_t index = 0; index < values.values().size(); ++index)
        values.row(0)[index] = static_cast<float>(index) * 0.125f;
    const std::array<ExpertRoute, 2> routes = {{{0, 0, 0.5f}, {2, 0, -0.25f}}};
    const ncnn::VkMat device = upload_fixture(context, values);
    auto region = ResidentFfnRegion_vulkan::create(normalized, 1, values.columns());
    region_check(region && region->bind_context(context, combiner), "single-wave region binding failed");
    const auto initial = get_vulkan_statistics(context->runtime());
    ActivationBuffer output;
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device());
        ncnn::VkMat input;
        bool uploaded = false;
        bool downloaded = false;
        const auto option = region_option(*context);
        region_check(region->record_input(input, command, option, uploaded) && uploaded,
                     "single warm Expert wave must upload normalized input once");
        const std::array<ResidentFfnRegion_vulkan::Output, 1> outputs = {{{0, device, routes}}};
        region_check(region->record_outputs(outputs, command, option, downloaded) && downloaded,
                     "single warm Expert wave must record its aggregate download in the same command");
        region_check(same_io_counters(initial, get_vulkan_statistics(context->runtime())),
                     "single-wave region recording submitted an extra command");
        complete_existing_wave(context, *region, command, uploaded, downloaded);
    }
    region_check(region->take_output(output), "single warm Expert wave did not publish its final output");
    ActivationBuffer expected(3, 8);
    std::fill_n(expected.row(0), expected.values().size(), 0.0f);
    for (size_t route = 0; route < routes.size(); ++route)
        for (uint32_t column = 0; column < values.columns(); ++column)
            expected.row(routes[route].token_index)[column] = routes[route].weight * values.row(route)[column];
    check_output(output, expected);
    const auto final = get_vulkan_statistics(context->runtime());
    region_check(final.compute_submissions == initial.compute_submissions + 1
                     && final.batch_uploads == initial.batch_uploads + 1
                     && final.batch_downloads == initial.batch_downloads + 1,
                 "single warm wave must keep one existing submit, one normalized upload and one aggregate download");
}

static void test_sparse_prefill_routes(const std::shared_ptr<VulkanContext>& context,
                                       const std::shared_ptr<MoeCombine_vulkan>& combiner)
{
    constexpr size_t rows = 257;
    constexpr uint32_t columns = 8;
    ActivationBuffer normalized(rows, columns);
    std::array<std::vector<ExpertRoute>, 2> routes;
    routes[0].push_back({256, 0, 1.0f});
    // Token ids are deliberately unsorted, with many empty token ranges.
    for (uint32_t token = 255; token != 0; --token)
    {
        if (token % 5 == 0) routes[0].push_back({token, 0, 0.5f});
        if (token % 7 == 0) routes[1].push_back({token, 0, -0.25f});
    }
    routes[0].push_back({0, 0, 0.25f});
    const size_t middle = routes[0].size();
    routes[0].push_back({256, 1, 1.0f});
    routes[0].push_back({17, 0, 0.5f});
    routes[0].push_back({256, 2, 1.0f});
    routes[1].push_back({256, 3, 0.25f});
    std::array<ActivationBuffer, 2> values = {ActivationBuffer(routes[0].size(), columns),
                                              ActivationBuffer(routes[1].size(), columns)};
    for (size_t expert = 0; expert < values.size(); ++expert)
        for (size_t index = 0; index < values[expert].values().size(); ++index)
            values[expert].row(0)[index] = static_cast<float>(index % 19) * 0.125f;
    values[0].row(0)[0] = 16777216.0f;
    values[0].row(middle)[0] = 1.0f;
    values[0].row(routes[0].size() - 1)[0] = -16777216.0f;
    values[1].row(routes[1].size() - 1)[0] = 1.0f;
    const ActivationBuffer expected = canonical_oracle(values, routes, rows, columns);
    region_check(expected.row(256)[0] == 0.25f && expected.row(2)[0] == 0.0f,
                 "sparse CSR fixture must distinguish duplicate order and empty tokens");
    const std::array<ncnn::VkMat, 2> device_values = {upload_fixture(context, values[0]), upload_fixture(context, values[1])};
    auto region = ResidentFfnRegion_vulkan::create(normalized, 2, columns);
    region_check(region && region->bind_context(context, combiner), "prefill region binding failed");
    const auto before = get_vulkan_statistics(context->runtime());
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device());
        ncnn::VkMat input;
        bool uploaded = false, downloaded = false;
        const auto option = region_option(*context);
        region_check(region->record_input(input, command, option, uploaded) && uploaded,
                     "prefill region input recording failed");
        const std::array<ResidentFfnRegion_vulkan::Output, 2> outputs = {{{1, device_values[1], routes[1]},
                                                                          {0, device_values[0], routes[0]}}};
        region_check(region->record_outputs(outputs, command, option, downloaded) && downloaded,
                     "prefill sparse CSR reduction recording failed");
        complete_existing_wave(context, *region, command, uploaded, downloaded);
    }
    ActivationBuffer output;
    region_check(region->take_output(output), "prefill sparse CSR reduction did not publish");
    check_output(output, expected);
    region_check(output.row(256)[0] == 0.25f, "CSR indexing changed repeated-route summation order");
    const auto after = get_vulkan_statistics(context->runtime());
    region_check(after.compute_submissions == before.compute_submissions + 1
                     && after.batch_uploads == before.batch_uploads + 1
                     && after.batch_downloads == before.batch_downloads + 1,
                 "prefill CSR indexing added a submission or activation transfer");
}
#endif

} // namespace moe
} // namespace ncnn

int main()
{
#if NCNN_MOE_WITH_VULKAN
    try
    {
        if (ncnn::moe::get_gpu_count() == 0)
        {
            std::cout << "SKIP resident FFN region: no Vulkan device\n";
            return 77;
        }
        for (uint32_t device = 0; device < ncnn::moe::get_gpu_count(); ++device)
        {
            const auto runtime = ncnn::moe::create_vulkan_runtime();
            const auto context = ncnn::moe::VulkanContext::acquire(device, runtime, ncnn::moe::OptimizationDefaultFlags);
            ncnn::moe::region_check(static_cast<bool>(context), "resident FFN test context creation failed");
            const auto combiner = ncnn::moe::MoeCombine_vulkan::create(context);
            ncnn::moe::region_check(static_cast<bool>(combiner), "resident FFN test combiner creation failed");
            ncnn::moe::test_two_waves(context, combiner);
            ncnn::moe::test_single_warm_wave(context, combiner);
            ncnn::moe::test_sparse_prefill_routes(context, combiner);
        }
        std::cout << "Resident FFN Vulkan region tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP resident FFN region: NCNN/Vulkan is disabled\n";
    return 77;
#endif
}
