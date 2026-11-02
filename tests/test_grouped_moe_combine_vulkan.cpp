#include "vulkan_test_support.h"
#include "backends/ncnn/moecombine_vulkan.h"
#include "backends/ncnn/residentffn_vulkan.h"
#include "backends/ncnn/vulkancontext.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if NCNN_MOE_WITH_VULKAN
#include <command.h>
#include <mat.h>
#include <option.h>
#endif

namespace ncnn {
namespace moe {

static void grouped_check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

#if NCNN_MOE_WITH_VULKAN
static constexpr uint32_t grouped_columns = 37;

static bool unchanged_io(const VulkanStatistics& a, const VulkanStatistics& b)
{
    return a.compute_submissions == b.compute_submissions && a.batch_uploads == b.batch_uploads && a.batch_downloads == b.batch_downloads;
}

static ncnn::VkMat grouped_upload(const std::shared_ptr<VulkanContext>& context, const ActivationBuffer& source)
{
    ncnn::VkMat output;
    grouped_check(vulkan_test_upload_raw(context, source, output), "grouped fixture upload failed");
    return output;
}

struct GroupedFixture
{
    std::vector<ActivationBuffer> values;
    std::vector<std::vector<ExpertRoute>> routes;
    std::vector<ncnn::VkMat> device;
    std::vector<MoeCombineDeviceInput_vulkan> inputs;

    void refresh()
    {
        inputs.resize(device.size());
        for (size_t index = 0; index < device.size(); ++index) inputs[index] = {&device[index], routes[index]};
    }
};

static GroupedFixture grouped_fixture(const std::shared_ptr<VulkanContext>& context, size_t count, bool repeated = false)
{
    GroupedFixture fixture;
    fixture.values.reserve(count);
    fixture.routes.resize(count);
    fixture.device.reserve(count);
    for (size_t index = 0; index < count; ++index)
    {
        const size_t rows = repeated && index == 0 ? 2 : 1;
        auto& values = fixture.values.emplace_back(rows, grouped_columns);
        for (size_t row = 0; row < rows; ++row)
            for (uint32_t column = 0; column < grouped_columns; ++column)
                values.row(row)[column] = static_cast<float>(static_cast<int>((index * 7 + column * 3 + row * 11) % 31) - 15) * 0.03125f;
        const float weight = index == 1 ? 1.00000011920928955078125f : index % 4 == 3             ? -0.5f
                                                                   : index % 4 == 0 && index != 0 ? 0.25f
                                                                                                  : 1.0f;
        fixture.routes[index].push_back({0, static_cast<uint32_t>(index), weight});
        if (rows == 2) fixture.routes[index].push_back({0, 99, -0.25f});
        // Column 0 detects a reduction reordered by wave arrival. Column 1
        // detects replacing separately rounded multiply/add with an FMA.
        if (index == 0)
        {
            values.row(0)[0] = 16777216.0f;
            values.row(0)[1] = -1.0000002384185791015625f;
        }
        else if (index == 1)
        {
            values.row(0)[0] = 0.99999988079071044921875f;
            values.row(0)[1] = 1.00000011920928955078125f;
        }
        else
        {
            values.row(0)[0] = index == 2 ? -16777216.0f : index == 3 ? 0.5f
                                                       : index == 4   ? 1.0f
                                                                      : values.row(0)[0];
            values.row(0)[1] = 0.0f;
        }
        fixture.device.push_back(grouped_upload(context, values));
    }
    fixture.refresh();
    return fixture;
}

static ActivationBuffer grouped_record(const std::shared_ptr<VulkanContext>& context,
                                       const MoeCombine_vulkan& combiner,
                                       const GroupedFixture& fixture, size_t rows,
                                       const ncnn::VkMat* shared = nullptr)
{
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const auto option = vulkan_test_option(*context);
    ncnn::VkCompute command(context->device());
    ncnn::VkMat combined, staging;
    std::vector<ncnn::VkMat> storage;
    const auto before = get_vulkan_statistics(context->runtime());
    grouped_check(combiner.record_device_inputs(fixture.inputs, rows, grouped_columns, combined, command, storage, shared),
                  "grouped/CSR reduction recording failed");
    grouped_check(unchanged_io(before, get_vulkan_statistics(context->runtime())), "record_device_inputs submitted work independently");
    grouped_check(prepare_staging_batch(staging, rows, grouped_columns, option.staging_vkallocator)
                      && record_prepared_activation_staging_download(combined, rows, grouped_columns, staging, command, context->device(), option)
                      && submit_compute_and_wait(command, context->device()) == 0,
                  "grouped reduction completion failed");
    ActivationBuffer output(rows, grouped_columns);
    grouped_check(copy_staging_to_cpu_batch(staging, output), "grouped reduction download failed");
    ++context->runtime_state().compute_submissions;
    ++context->runtime_state().batch_downloads;
    return output;
}

static void grouped_bitwise(const ActivationBuffer& actual, const ActivationBuffer& expected, const std::string& label)
{
    grouped_check(actual.rows() >= 1 && expected.rows() >= 1 && actual.columns() == expected.columns(), label + " shape mismatch");
    for (uint32_t column = 0; column < actual.columns(); ++column)
        grouped_check(std::bit_cast<uint32_t>(actual.row(0)[column]) == std::bit_cast<uint32_t>(expected.row(0)[column]),
                      label + " differs from canonical CSR bits at column " + std::to_string(column));
}

static void test_group_sizes(const std::shared_ptr<VulkanContext>& context, const MoeCombine_vulkan& combiner)
{
    for (const size_t count : {size_t(1), size_t(5), size_t(32)})
    {
        const auto fixture = grouped_fixture(context, count);
        // Two output rows force the unchanged CSR branch with the identical
        // token-0 input values, route weights, and canonical Expert ordering.
        const auto expected = grouped_record(context, combiner, fixture, 2);
        const auto actual = grouped_record(context, combiner, fixture, 1);
        grouped_bitwise(actual, expected, "K=" + std::to_string(count));
        for (uint32_t column = 0; column < grouped_columns; ++column)
            grouped_check(expected.row(1)[column] == 0.0f, "CSR oracle wrote an unrouted token");
        if (count >= 4) grouped_check(expected.row(0)[1] == 0.0f, "CSR FMA fixture must cancel to zero");
        if (count == 5) grouped_check(expected.row(0)[0] == 0.0f, "canonical order fixture must cancel to zero");
    }
}

static void test_csr_boundaries(const std::shared_ptr<VulkanContext>& context, const MoeCombine_vulkan& combiner)
{
    const auto repeated = grouped_fixture(context, 5, true);
    const auto expected = grouped_record(context, combiner, repeated, 2);
    const auto actual = grouped_record(context, combiner, repeated, 1);
    grouped_bitwise(actual, expected, "repeated token-0 routes must stay CSR");

    const auto single = grouped_fixture(context, 5);
    ActivationBuffer shared_one(1, grouped_columns), shared_two(2, grouped_columns);
    for (uint32_t column = 0; column < grouped_columns; ++column)
        shared_one.row(0)[column] = shared_two.row(0)[column] = static_cast<float>(column) * -0.0625f;
    const auto device_one = grouped_upload(context, shared_one);
    const auto device_two = grouped_upload(context, shared_two);
    const auto expected_shared = grouped_record(context, combiner, single, 2, &device_two);
    const auto actual_shared = grouped_record(context, combiner, single, 1, &device_one);
    grouped_bitwise(actual_shared, expected_shared, "Shared Expert must follow routed Experts through CSR");

    const std::lock_guard<std::mutex> lock(context->command_mutex());
    ncnn::VkCompute command(context->device());
    ncnn::VkMat output;
    std::vector<ncnn::VkMat> storage;
    auto inputs = single.inputs;
    const std::array<ExpertRoute, 1> invalid_route = {{{1, 0, 1.0f}}};
    inputs[0].routes = invalid_route;
    uint64_t count = 1234567;
    const auto before = get_vulkan_statistics(context->runtime());
    grouped_check(!combiner.record_device_inputs(inputs, 1, grouped_columns, output, command, storage, nullptr, &count)
                      && count == 1234567,
                  "failed recording published a dispatch count");
    grouped_check(command.reset() == 0 && unchanged_io(before, get_vulkan_statistics(context->runtime())),
                  "invalid recording submitted work or failed to reset");
}

static void assert_unpublished(ResidentFfnRegion_vulkan& region, ActivationBuffer& output)
{
    const auto* pointer = output.values().data();
    const std::vector<float> values(output.values().begin(), output.values().end());
    grouped_check(!region.complete() && !region.take_output(output) && output.rows() == 2 && output.columns() == 3
                      && output.values().data() == pointer && std::equal(values.begin(), values.end(), output.values().begin()),
                  "unfinished region published or modified the caller's output");
}

static void complete_wave(const std::shared_ptr<VulkanContext>& context, ResidentFfnRegion_vulkan& region,
                          ncnn::VkCompute& command, bool uploaded, bool downloaded)
{
    grouped_check(submit_compute_and_wait(command, context->device()) == 0, "existing wave submission failed");
    const auto before = get_vulkan_statistics(context->runtime());
    grouped_check(region.complete_wave() && unchanged_io(before, get_vulkan_statistics(context->runtime())),
                  "region completion failed or submitted extra work");
    ++context->runtime_state().compute_submissions;
    context->runtime_state().batch_uploads += uploaded;
    context->runtime_state().batch_downloads += downloaded;
}

static void test_grouped_waves(const std::shared_ptr<VulkanContext>& context, const std::shared_ptr<MoeCombine_vulkan>& combiner)
{
    const auto fixture = grouped_fixture(context, 5);
    const auto expected = grouped_record(context, *combiner, fixture, 2);
    ActivationBuffer normalized(1, grouped_columns), output(2, 3);
    std::fill_n(normalized.row(0), normalized.values().size(), 0.125f);
    std::fill_n(output.row(0), output.values().size(), -37.0f);
    auto region = ResidentFfnRegion_vulkan::create(normalized, 5, grouped_columns);
    grouped_check(region && region->bind_context(context, combiner), "grouped region creation failed");
    const auto initial = get_vulkan_statistics(context->runtime());
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        const auto option = vulkan_test_option(*context);
        ncnn::VkCompute command(context->device());
        ncnn::VkMat input;
        bool uploaded = false, downloaded = true;
        grouped_check(region->record_input(input, command, option, uploaded) && uploaded, "first wave must upload once");
        const std::array<ResidentFfnRegion_vulkan::Output, 3> outputs = {{{4, fixture.device[4], fixture.routes[4]},
                                                                          {2, fixture.device[2], fixture.routes[2]},
                                                                          {1, fixture.device[1], fixture.routes[1]}}};
        grouped_check(region->record_outputs(outputs, command, option, downloaded) && !downloaded
                          && region->pending_combine_dispatches() == 0 && region->completed_combine_dispatches() == 0,
                      "nonfinal wave recorded a reduction or published its count");
        assert_unpublished(*region, output);
        complete_wave(context, *region, command, uploaded, downloaded);
        grouped_check(region->pending_combine_dispatches() == 0 && region->completed_combine_dispatches() == 0,
                      "nonfinal completion published a reduction count");
    }
    assert_unpublished(*region, output);
    const auto partial = get_vulkan_statistics(context->runtime());
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        const auto option = vulkan_test_option(*context);
        // A partially appended invalid output set must never be committed.
        ncnn::VkCompute invalid(context->device());
        ncnn::VkMat input;
        bool uploaded = true, downloaded = true;
        grouped_check(region->record_input(input, invalid, option, uploaded) && !uploaded, "later wave repeated its upload");
        const std::array<ResidentFfnRegion_vulkan::Output, 2> duplicate = {{{0, fixture.device[0], fixture.routes[0]},
                                                                            {0, fixture.device[0], fixture.routes[0]}}};
        grouped_check(!region->record_outputs(duplicate, invalid, option, downloaded) && !region->complete_wave()
                          && region->pending_combine_dispatches() == 0 && region->completed_combine_dispatches() == 0,
                      "a rejected output set was committed or published a count");
        grouped_check(invalid.reset() == 0, "invalid wave reset failed");
        region->abort_wave();
        assert_unpublished(*region, output);

        const std::array<ResidentFfnRegion_vulkan::Output, 2> final_outputs = {{{3, fixture.device[3], fixture.routes[3]},
                                                                                {0, fixture.device[0], fixture.routes[0]}}};
        // Abort a complete recording too: its grouped count stays pending and
        // must not leak into committed statistics or the caller's output.
        ncnn::VkCompute aborted(context->device());
        grouped_check(region->record_input(input, aborted, option, uploaded) && !uploaded
                          && region->record_outputs(final_outputs, aborted, option, downloaded) && downloaded
                          && region->pending_combine_dispatches() > 0 && region->completed_combine_dispatches() == 0,
                      "final recording did not keep its reduction pending until completion");
        assert_unpublished(*region, output);
        grouped_check(aborted.reset() == 0, "unsubmitted final wave reset failed");
        region->abort_wave();
        grouped_check(region->pending_combine_dispatches() == 0 && region->completed_combine_dispatches() == 0
                          && unchanged_io(partial, get_vulkan_statistics(context->runtime())),
                      "aborted final wave published counters or submitted work");
        assert_unpublished(*region, output);

        ncnn::VkCompute retried(context->device());
        grouped_check(region->record_input(input, retried, option, uploaded) && !uploaded
                          && region->record_outputs(final_outputs, retried, option, downloaded) && downloaded
                          && region->pending_combine_dispatches() > 0 && region->completed_combine_dispatches() == 0,
                      "aborted final recording could not be retried");
        assert_unpublished(*region, output);
        complete_wave(context, *region, retried, uploaded, downloaded);
        grouped_check(region->pending_combine_dispatches() == 0 && region->completed_combine_dispatches() > 0,
                      "successful wait did not publish the completed reduction");
    }
    const auto final = get_vulkan_statistics(context->runtime());
    grouped_check(final.compute_submissions == initial.compute_submissions + 2
                      && final.batch_uploads == initial.batch_uploads + 1 && final.batch_downloads == initial.batch_downloads + 1,
                  "two waves added a reduction submission, repeated input upload, or premature download");
    grouped_check(region->take_output(output) && region->pending_combine_dispatches() == 0 && region->completed_combine_dispatches() == 0,
                  "final take did not consume the completed aggregate and count once");
    grouped_bitwise(output, expected, "out-of-order grouped region");
    grouped_check(output.row(0)[0] == 0.0f && output.row(0)[1] == 0.0f, "cross-wave reduction lost canonical rounding order");
    const auto* pointer = output.values().data();
    grouped_check(!region->take_output(output) && output.values().data() == pointer, "aggregate was published twice");
}
#endif

static int run_grouped_tests()
{
#if NCNN_MOE_WITH_VULKAN
    if (get_gpu_count() == 0) return 77;
    try
    {
        const auto runtime = create_vulkan_runtime();
        const auto context = VulkanContext::acquire(get_default_gpu_index(), runtime, OptimizationDefaultFlags);
        grouped_check(static_cast<bool>(context), "grouped combine Vulkan context unavailable");
        const auto combiner = MoeCombine_vulkan::create(context);
        grouped_check(static_cast<bool>(combiner), "grouped combine preparation failed");
        if (!combiner->supports_grouped_single_row())
        {
            std::cout << "SKIP: grouped single-row combine requires five storage bindings\n";
            return 77;
        }
        test_group_sizes(context, *combiner);
        test_csr_boundaries(context, *combiner);
        test_grouped_waves(context, combiner);
        std::cout << "Grouped combine bitwise CSR/cancellation, fallback shapes, wave publication and retry tests passed\n";
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
    return ncnn::moe::run_grouped_tests();
}
