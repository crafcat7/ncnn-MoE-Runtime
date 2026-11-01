#include "gpu_segment_fixture.h"
#include "backends/ncnn/expertbank_vulkan.h"
#include "backends/ncnn/modelpipeline.h"

#include <bit>
#include <limits>
#include <utility>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static constexpr uint64_t bank_flags = segment_flags & ~OptimizationCpuMxfp4Q8;
static constexpr uint32_t bank_width = 128;

struct BankExpertFixture
{
    TensorData gate;
    TensorData up;
    TensorData down;
    TensorData gate_bias;
    TensorData up_bias;
    TensorData down_bias;
};

static TensorData bank_weight(DType dtype, uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData result;
    result.dtype = dtype;
    result.shape = {rows, columns};
    const size_t size = static_cast<size_t>(rows) * columns;
    if (dtype == DType::MxFp4)
    {
        result.mxfp4_blocks.resize(size / 2);
        result.mxfp4_scales.resize(size / 32);
        std::fill_n(result.mxfp4_scales.data(), result.mxfp4_scales.size(), uint8_t{121});
        for (size_t index = 0; index < result.mxfp4_blocks.size(); ++index)
            result.mxfp4_blocks[index] = static_cast<uint8_t>(((index * 7 + seed * 3 + index / 13) % 16)
                                                              | (((index * 11 + seed * 5 + index / 17) % 16) << 4));
    }
    else
        for (size_t index = 0; index < size; ++index)
        {
            const float value = static_cast<float>(static_cast<int>((index * 7 + index / columns + seed * 3) % 29) - 14) * 0.00390625f;
            if (dtype == DType::BFloat16)
                result.bfloat16_data.push_back(float_to_bfloat16(value));
            else
                result.float32_data.push_back(value);
        }
    return result;
}

static TensorData bank_bias(uint32_t rows, uint32_t seed)
{
    TensorData result;
    result.dtype = DType::Float32;
    result.shape = {rows};
    for (uint32_t row = 0; row < rows; ++row)
        result.float32_data.push_back(static_cast<float>(static_cast<int>((row * 3 + seed) % 13) - 6) * 0.00390625f);
    return result;
}

static ExpertBankWeights_vulkan bank_view(const BankExpertFixture& fixture, ExpertLayout layout, ExpertActivation activation)
{
    ExpertBankWeights_vulkan result;
    result.gate_up = &fixture.gate;
    result.up = layout == ExpertLayout::GateUpDown ? &fixture.up : nullptr;
    result.down = &fixture.down;
    result.gate_up_bias = &fixture.gate_bias;
    result.up_bias = layout == ExpertLayout::GateUpDown ? &fixture.up_bias : nullptr;
    result.down_bias = &fixture.down_bias;
    result.layout = layout;
    result.activation = activation;
    result.activation_limit = activation == ExpertActivation::GptOssSwiGlu ? 7.0f : 0.0f;
    return result;
}

static std::vector<BankExpertFixture> bank_fixture(DType dtype, ExpertLayout layout, uint32_t count,
                                                   uint32_t output_columns = bank_width, uint32_t intermediate_columns = bank_width)
{
    std::vector<BankExpertFixture> result;
    result.reserve(count);
    for (uint32_t expert = 0; expert < count; ++expert)
    {
        BankExpertFixture item;
        const uint32_t gate_rows = layout == ExpertLayout::GateUpDown ? intermediate_columns : intermediate_columns * 2;
        item.gate = bank_weight(dtype, gate_rows, bank_width, expert * 7 + 1);
        if (layout == ExpertLayout::GateUpDown)
        {
            item.up = bank_weight(dtype, intermediate_columns, bank_width, expert * 7 + 2);
            item.up_bias = bank_bias(intermediate_columns, expert * 7 + 5);
        }
        item.down = bank_weight(dtype, output_columns, intermediate_columns, expert * 7 + 3);
        item.gate_bias = bank_bias(gate_rows, expert * 7 + 4);
        item.down_bias = bank_bias(output_columns, expert * 7 + 6);
        result.push_back(std::move(item));
    }
    return result;
}

static double bank_value(const TensorData& tensor, size_t index)
{
    if (tensor.dtype == DType::Float32) return tensor.float32_data[index];
    if (tensor.dtype == DType::BFloat16) return bfloat16_to_float(tensor.bfloat16_data[index]);
    static constexpr double values[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};
    const uint8_t code = static_cast<uint8_t>((tensor.mxfp4_blocks[index / 2] >> ((index % 2) * 4)) & 15);
    return std::ldexp(values[code], static_cast<int>(tensor.mxfp4_scales[index / 32]) - 127);
}

static std::vector<double> bank_values(const TensorData& tensor)
{
    std::vector<double> result(tensor.element_count());
    for (size_t index = 0; index < result.size(); ++index) result[index] = bank_value(tensor, index);
    return result;
}

static std::vector<ActivationBuffer> bank_cpu_experts(const std::vector<BankExpertFixture>& fixtures,
                                                      ExpertLayout layout, ExpertActivation activation,
                                                      const ActivationBuffer& input)
{
    std::vector<ActivationBuffer> result;
    for (const auto& fixture : fixtures)
    {
        const auto gate = bank_values(fixture.gate);
        const auto up = layout == ExpertLayout::GateUpDown ? bank_values(fixture.up) : std::vector<double>{};
        const auto down = bank_values(fixture.down);
        const uint32_t output_columns = fixture.down.shape[0];
        const uint32_t intermediate_columns = fixture.down.shape[1];
        auto& output = result.emplace_back(input.rows(), output_columns);
        for (size_t row = 0; row < input.rows(); ++row)
        {
            std::vector<double> activated(intermediate_columns);
            for (uint32_t middle = 0; middle < intermediate_columns; ++middle)
            {
                const uint32_t gate_row = layout == ExpertLayout::InterleavedGateUpDown ? middle * 2 : middle;
                const uint32_t up_row = layout == ExpertLayout::InterleavedGateUpDown ? middle * 2 + 1
                                        : layout == ExpertLayout::PackedGateUpDown    ? middle + intermediate_columns
                                                                                      : middle;
                double g = bank_value(fixture.gate_bias, gate_row);
                double u = bank_value(layout == ExpertLayout::GateUpDown ? fixture.up_bias : fixture.gate_bias, up_row);
                for (uint32_t column = 0; column < bank_width; ++column)
                {
                    g += gate[static_cast<size_t>(gate_row) * bank_width + column] * input.row(row)[column];
                    u += (layout == ExpertLayout::GateUpDown ? up : gate)[static_cast<size_t>(up_row) * bank_width + column] * input.row(row)[column];
                }
                if (activation == ExpertActivation::GptOssSwiGlu)
                {
                    g = std::min(g, 7.0);
                    u = std::clamp(u, -7.0, 7.0);
                    activated[middle] = g / (1.0 + std::exp(-1.702 * g)) * (u + 1.0);
                }
                else
                    activated[middle] = g / (1.0 + std::exp(-g)) * u;
            }
            for (uint32_t column = 0; column < output_columns; ++column)
            {
                double value = bank_value(fixture.down_bias, column);
                for (uint32_t middle = 0; middle < intermediate_columns; ++middle)
                    value += down[static_cast<size_t>(column) * intermediate_columns + middle] * activated[middle];
                output.row(row)[column] = static_cast<float>(value);
            }
        }
    }
    return result;
}

static ActivationBuffer bank_selection(size_t rows, uint32_t top_k, uint32_t expert_count)
{
    ActivationBuffer selected(rows, top_k * 2 + 1);
    for (size_t row = 0; row < rows; ++row)
    {
        for (uint32_t rank = 0; rank < top_k; ++rank)
        {
            selected.row(row)[rank] = static_cast<float>(expert_count - 1 - (rank * 5 + row * 3) % expert_count);
            selected.row(row)[top_k + rank] = static_cast<float>(rank + 1) / static_cast<float>(top_k * (top_k + 1) / 2);
        }
        selected.row(row)[top_k * 2] = 0.0f;
    }
    return selected;
}

static ActivationBuffer bank_cpu_combine(const std::vector<ActivationBuffer>& experts, const ActivationBuffer& selected, uint32_t top_k)
{
    ActivationBuffer output(selected.rows(), experts.front().columns());
    for (size_t row = 0; row < output.rows(); ++row)
        for (size_t column = 0; column < output.columns(); ++column)
        {
            float sum = 0.0f;
            for (uint32_t expert = 0; expert < experts.size(); ++expert)
                for (uint32_t rank = 0; rank < top_k; ++rank)
                    if (selected.row(row)[rank] == static_cast<float>(expert))
                    {
                        // Preserve the separately rounded multiply/add contract of the old canonical Combine.
                        volatile float product = experts[expert].row(row)[column] * selected.row(row)[top_k + rank];
                        sum += product;
                    }
            output.row(row)[column] = sum;
        }
    return output;
}

struct BankResult
{
    ActivationBuffer output;
    uint32_t validation = 0;
};

static ncnn::Option bank_option(const VulkanContext& context)
{
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_packing_layout = false;
    option.use_fp16_packed = option.use_fp16_storage = option.use_fp16_arithmetic = false;
    option.use_bf16_packed = option.use_bf16_storage = false;
    option.blob_vkallocator = context.blob_allocator();
    option.workspace_vkallocator = context.blob_allocator();
    option.staging_vkallocator = context.staging_allocator();
    return option;
}

static BankResult bank_execute(const ExpertBank_vulkan& bank, const ActivationBuffer& input,
                               const ActivationBuffer& selected, uint32_t top_k, bool abort_first = false)
{
    const auto context = bank.context();
    const auto& description = bank.description();
    const std::string label = " dtype=" + std::to_string(static_cast<int>(description.dtype))
                              + " layout=" + std::to_string(static_cast<int>(description.layout))
                              + " K=" + std::to_string(top_k) + " rows=" + std::to_string(input.rows())
                              + " intermediate=" + std::to_string(description.intermediate_columns)
                              + " output=" + std::to_string(description.output_columns);
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const auto option = bank_option(*context);
    BankResult result;
    result.output.reset(1, 1, false);
    result.output.row(0)[0] = -71.0f;
    for (bool discard : abort_first ? std::vector<bool>{true, false} : std::vector<bool>{false})
    {
        ncnn::VkCompute command(context->device());
        ncnn::VkMat input_upload, selected_upload, flag_upload, device_input, device_selected, validation, output, output_staging, flag_staging;
        std::vector<ncnn::VkMat> workspace;
        const uint32_t zero = 0;
        segment_check(fill_staging_upload(input, input_upload, option.staging_vkallocator)
                          && record_mapped_upload(input_upload, device_input, command, option)
                          && fill_staging_upload(selected, selected_upload, option.staging_vkallocator)
                          && record_mapped_upload(selected_upload, device_selected, command, option)
                          && fill_staging_values(&zero, 1, sizeof(zero), flag_upload, option.staging_vkallocator)
                          && record_mapped_upload(flag_upload, validation, command, option),
                      "cannot stage the bank operand fixture");
        const auto before = get_vulkan_statistics(context->runtime());
        segment_check(bank.record(device_input, device_selected, top_k, output, validation, command, workspace), "selected expert bank record failed" + label);
        segment_check(get_vulkan_statistics(context->runtime()).compute_submissions == before.compute_submissions
                          && result.output.rows() == 1 && result.output.columns() == 1 && result.output.row(0)[0] == -71.0f,
                      "record-only bank submitted work or published host output before its fence");
        if (discard)
        {
            segment_check(command.reset() == 0, "cannot discard an unsubmitted bank recording");
            continue;
        }
        segment_check(prepare_staging_batch(output_staging, input.rows(), description.output_columns, option.staging_vkallocator)
                          && record_prepared_activation_staging_download(output, input.rows(), description.output_columns, output_staging, command, context->device(), option)
                          && prepare_staging_batch(flag_staging, 1, 1, option.staging_vkallocator)
                          && record_prepared_activation_staging_download(validation, 1, 1, flag_staging, command, context->device(), option)
                          && submit_compute_and_wait(command, context->device()) == 0,
                      "bank command or final validation download failed");
        ActivationBuffer flag(1, 1);
        result.output.reset(input.rows(), description.output_columns, false);
        segment_check(copy_staging_to_cpu_batch(output_staging, result.output) && copy_staging_to_cpu_batch(flag_staging, flag),
                      "cannot publish completed bank fixture output");
        result.validation = std::bit_cast<uint32_t>(flag.row(0)[0]);
        ++context->runtime_state().compute_submissions;
    }
    return result;
}

static std::shared_ptr<ExpertBank_vulkan> bank_create(const std::vector<BankExpertFixture>& fixtures,
                                                      ExpertLayout layout, ExpertActivation activation,
                                                      const std::shared_ptr<VulkanContext>& context)
{
    std::vector<ExpertBankWeights_vulkan> descriptions;
    for (const auto& item : fixtures) descriptions.push_back(bank_view(item, layout, activation));
    const auto planned = ExpertBank_vulkan::plan(descriptions, context);
    segment_check(static_cast<bool>(planned), planned ? "" : planned.error().message);
    size_t callbacks = 0;
    std::weak_ptr<BankExpertFixture> previous;
    const auto created = ExpertBank_vulkan::create(planned.value(), context, [&](uint32_t index) -> Result<ExpertBankWeights_vulkan> {
        segment_check(previous.expired(), "bank retained a host expert lease while loading its successor");
        auto owner = std::make_shared<BankExpertFixture>(fixtures.at(index));
        previous = owner;
        auto weights = bank_view(*owner, layout, activation);
        weights.owner = owner;
        ++callbacks;
        return weights;
    });
    segment_check(static_cast<bool>(created), created ? "" : created.error().message);
    segment_check(callbacks == fixtures.size() && previous.expired(), "bank did not release its final host expert lease");
    segment_check(created.value()->physical_size() == planned.value().physical_size
                      && created.value()->description().expert_count == fixtures.size()
                      && created.value()->workspace_size(3, 4) == ExpertBank_vulkan::workspace_size(planned.value(), 3, 4),
                  "bank allocation or workspace charge differs from its metadata-only plan");
    return created.value();
}

static void bank_direct(const std::shared_ptr<VulkanContext>& context)
{
    ActivationBuffer input(3, bank_width);
    for (size_t row = 0; row < input.rows(); ++row)
        for (uint32_t column = 0; column < bank_width; ++column)
            input.row(row)[column] = static_cast<float>(static_cast<int>((column * 7 + row * 11) % 23) - 11) * 0.015625f;
    for (const DType dtype : {DType::MxFp4, DType::BFloat16, DType::Float32})
        for (const ExpertLayout layout : {ExpertLayout::InterleavedGateUpDown, ExpertLayout::PackedGateUpDown, ExpertLayout::GateUpDown})
        {
            const auto fixtures = bank_fixture(dtype, layout, 16);
            const auto activation = layout == ExpertLayout::InterleavedGateUpDown ? ExpertActivation::GptOssSwiGlu : ExpertActivation::Silu;
            const auto expected = bank_cpu_experts(fixtures, layout, activation, input);
            const auto bank = bank_create(fixtures, layout, activation, context);
            // Cover single-selection decode, batched small K, and the maximum K.
            const std::array<std::pair<uint32_t, size_t>, 3> selections = {{{1, 1}, {4, 3}, {16, 1}}};
            for (const auto& [top_k, rows] : selections)
            {
                ActivationBuffer current(rows, bank_width);
                for (size_t row = 0; row < rows; ++row) std::copy_n(input.row(row), bank_width, current.row(row));
                const auto selected = bank_selection(rows, top_k, 16);
                const auto actual = bank_execute(*bank, current, selected, top_k, top_k == 1 && rows == 1);
                segment_check(actual.validation == 0, "valid device routes set the bank validation error flag");
                segment_compare(actual.output.values(), bank_cpu_combine(expected, selected, top_k).values(),
                                "bank dtype " + std::to_string(static_cast<int>(dtype)) + " layout " + std::to_string(static_cast<int>(layout))
                                    + " K" + std::to_string(top_k) + " rows" + std::to_string(rows));
            }
            {
                constexpr uint32_t output_columns = bank_width + 1;
                const auto tail_fixtures = bank_fixture(dtype, layout, 16, output_columns);
                const auto tail_bank = bank_create(tail_fixtures, layout, activation, context);
                const auto selected = bank_selection(input.rows(), 4, 16);
                const auto actual = bank_execute(*tail_bank, input, selected, 4);
                segment_check(actual.validation == 0 && actual.output.columns() == output_columns,
                              "valid single/odd-column bank output was rejected or truncated");
                segment_compare(actual.output.values(), bank_cpu_combine(bank_cpu_experts(tail_fixtures, layout, activation, input), selected, 4).values(),
                                "bank output tail dtype " + std::to_string(static_cast<int>(dtype)) + " layout " + std::to_string(static_cast<int>(layout))
                                    + " columns " + std::to_string(output_columns));
            }
            if (dtype != DType::MxFp4)
            {
                constexpr uint32_t intermediate_columns = bank_width + 1;
                const auto tail_fixtures = bank_fixture(dtype, layout, 16, bank_width, intermediate_columns);
                const auto tail_bank = bank_create(tail_fixtures, layout, activation, context);
                const auto selected = bank_selection(input.rows(), 4, 16);
                const auto actual = bank_execute(*tail_bank, input, selected, 4);
                segment_check(actual.validation == 0 && tail_bank->description().intermediate_columns == intermediate_columns,
                              "valid single/odd intermediate bank columns were rejected or truncated");
                segment_compare(actual.output.values(), bank_cpu_combine(bank_cpu_experts(tail_fixtures, layout, activation, input), selected, 4).values(),
                                "bank intermediate tail dtype " + std::to_string(static_cast<int>(dtype)) + " layout " + std::to_string(static_cast<int>(layout))
                                    + " columns " + std::to_string(intermediate_columns));
            }
        }
}

static void bank_canonical_and_invalid(const std::shared_ptr<VulkanContext>& context)
{
    ActivationBuffer input(1, bank_width);
    std::fill_n(input.row(0), bank_width, 0.0f);
    for (const DType dtype : {DType::MxFp4, DType::BFloat16, DType::Float32})
    {
        auto fixtures = bank_fixture(dtype, ExpertLayout::InterleavedGateUpDown, 4);
        const std::array<float, 4> constants = {67108864.0f, 4.0f, -67108864.0f, 2.0f};
        for (size_t expert = 0; expert < fixtures.size(); ++expert)
        {
            auto& item = fixtures[expert];
            for (TensorData* weight : {&item.gate, &item.down})
            {
                std::fill(weight->float32_data.begin(), weight->float32_data.end(), 0.0f);
                std::fill(weight->bfloat16_data.begin(), weight->bfloat16_data.end(), uint16_t{0});
                std::fill_n(weight->mxfp4_blocks.data(), weight->mxfp4_blocks.size(), uint8_t{0});
            }
            std::fill(item.gate_bias.float32_data.begin(), item.gate_bias.float32_data.end(), 0.0f);
            std::fill(item.down_bias.float32_data.begin(), item.down_bias.float32_data.end(), constants[expert]);
        }
        const auto bank = bank_create(fixtures, ExpertLayout::InterleavedGateUpDown, ExpertActivation::Silu, context);
        ActivationBuffer selected(1, 9);
        const std::array<float, 4> ids = {2, 0, 3, 1};
        std::copy(ids.begin(), ids.end(), selected.row(0));
        std::fill_n(selected.row(0) + 4, 4, 0.25f);
        selected.row(0)[8] = 0.0f;
        const auto actual = bank_execute(*bank, input, selected, 4);
        segment_check(actual.validation == 0, "canonical route fixture is invalid");
        for (const float value : actual.output.values()) segment_check(value == 0.5f, "bank accumulation changed canonical Expert order");
        // Route validation is shared by formats; cancellation above still covers each format.
        if (dtype != DType::Float32) continue;
        ActivationBuffer zero_weight = selected;
        zero_weight.row(0)[4] = 0.0f;
        segment_check(bank_execute(*bank, input, zero_weight, 4).validation == 0, "a valid zero route weight was rejected");
        std::fill_n(zero_weight.row(0) + 4, 4, 0.0f);
        const auto zero_output = bank_execute(*bank, input, zero_weight, 4);
        segment_check(zero_output.validation == 0, "valid all-zero route weights were rejected");
        for (const float value : zero_output.output.values()) segment_check(value == 0.0f, "all-zero route weights produced nonzero bank output");
        auto zeros = fixtures;
        for (auto& expert : zeros) std::fill(expert.down_bias.float32_data.begin(), expert.down_bias.float32_data.end(), 0.0f);
        const auto zero_bank = bank_create(zeros, ExpertLayout::InterleavedGateUpDown, ExpertActivation::Silu, context);
        std::fill_n(zero_weight.row(0) + 4, 4, std::numeric_limits<float>::max());
        const auto overflow_sum = bank_execute(*zero_bank, input, zero_weight, 4);
        segment_check(overflow_sum.validation == 0, "finite nonnegative route weights were rejected because their sum overflows FP32");
        for (const float value : overflow_sum.output.values()) segment_check(value == 0.0f, "zero expert outputs changed under finite large route weights");
        for (const size_t field : {size_t{0}, size_t{4}, size_t{8}})
            for (const float invalid : {-1.0f, 4.0f, 0.5f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
            {
                if (field == 4 && invalid >= 0 && std::isfinite(invalid)) continue;
                if (field == 8 && invalid == 0.0f) continue;
                ActivationBuffer malformed = selected;
                malformed.row(0)[field] = invalid;
                const auto rejected = bank_execute(*bank, input, malformed, 4);
                segment_check(rejected.validation != 0, "invalid GPU route/weight/status was accepted");
                for (const float value : rejected.output.values()) segment_check(std::isfinite(value), "invalid device routes caused an out-of-range bank read");
            }
        ActivationBuffer duplicate = selected;
        duplicate.row(0)[1] = duplicate.row(0)[0];
        segment_check(bank_execute(*bank, input, duplicate, 4).validation != 0, "duplicate selected Expert IDs were accepted");
    }
}

static void bank_preparation_failures(const std::shared_ptr<VulkanContext>& context)
{
    auto fixtures = bank_fixture(DType::MxFp4, ExpertLayout::InterleavedGateUpDown, 4);
    std::vector<ExpertBankWeights_vulkan> descriptions;
    for (const auto& fixture : fixtures) descriptions.push_back(bank_view(fixture, ExpertLayout::InterleavedGateUpDown, ExpertActivation::Silu));
    const auto plan = ExpertBank_vulkan::plan(descriptions, context);
    segment_check(static_cast<bool>(plan), plan ? "" : plan.error().message);
    segment_check(!ExpertBank_vulkan::plan({}, context) && !ExpertBank_vulkan::plan(descriptions, {}),
                  "empty bank metadata or a missing Vulkan context was accepted");
    auto mismatched = fixtures[1];
    mismatched.down.shape[0] -= 1;
    auto invalid = descriptions;
    invalid[1] = bank_view(mismatched, ExpertLayout::InterleavedGateUpDown, ExpertActivation::Silu);
    const auto unsupported = ExpertBank_vulkan::plan(invalid, context);
    segment_check(!unsupported && unsupported.error().code == ErrorCode::UnsupportedModel,
                  "heterogeneous bank dimensions were not rejected before allocation");
    auto interleaved = std::make_shared<MxFp4FileStorage>();
    interleaved->interleave_rows = true;
    auto contradictory = fixtures.front();
    contradictory.gate.mxfp4_file_storage = interleaved;
    auto file_metadata = bank_view(contradictory, ExpertLayout::PackedGateUpDown, ExpertActivation::Silu);
    const auto packed_file = ExpertBank_vulkan::plan({&file_metadata, 1}, context);
    segment_check(!packed_file && packed_file.error().code == ErrorCode::UnsupportedModel,
                  "Packed gate/up metadata accepted interleaved file rows");
    contradictory.gate.mxfp4_file_storage.reset();
    contradictory.down.mxfp4_file_storage = interleaved;
    file_metadata = bank_view(contradictory, ExpertLayout::InterleavedGateUpDown, ExpertActivation::Silu);
    const auto down_file = ExpertBank_vulkan::plan({&file_metadata, 1}, context);
    segment_check(!down_file && down_file.error().code == ErrorCode::UnsupportedModel,
                  "an ordinary down projection accepted interleaved file rows");
    auto separate = bank_fixture(DType::MxFp4, ExpertLayout::GateUpDown, 1);
    separate.front().up.mxfp4_file_storage = interleaved;
    file_metadata = bank_view(separate.front(), ExpertLayout::GateUpDown, ExpertActivation::Silu);
    const auto up_file = ExpertBank_vulkan::plan({&file_metadata, 1}, context);
    segment_check(!up_file && up_file.error().code == ErrorCode::UnsupportedModel,
                  "a separate up projection accepted interleaved file rows");
    size_t loaded = 0;
    const auto failed = ExpertBank_vulkan::create(plan.value(), context, [&](uint32_t expert) -> Result<ExpertBankWeights_vulkan> {
        ++loaded;
        if (expert == 2) return Error{ErrorCode::IoError, "injected bank payload read failure"};
        return descriptions[expert];
    });
    segment_check(!failed && loaded == 3 && failed.error().code == ErrorCode::IoError,
                  "a selected bank swallowed its loader failure or continued loading experts");
    fixtures[2].gate.mxfp4_scales.resize(1);
    const auto malformed = ExpertBank_vulkan::create(plan.value(), context, [&](uint32_t expert) -> Result<ExpertBankWeights_vulkan> {
        return bank_view(fixtures[expert], ExpertLayout::InterleavedGateUpDown, ExpertActivation::Silu);
    });
    segment_check(!malformed && malformed.error().code == ErrorCode::InvalidModel,
                  "a selected bank accepted a truncated native MXFP4 scale payload");
    auto changed = plan.value();
    ++changed.storage_size;
    segment_check(!ExpertBank_vulkan::create(changed, context, [&](uint32_t expert) -> Result<ExpertBankWeights_vulkan> { return descriptions[expert]; }),
                  "bank preparation silently changed its selected physical layout");
}

static CompiledModel bank_model(bool gpu, DType dtype, bool native_delta, bool resident,
                                ExpertLayout layout = ExpertLayout::InterleavedGateUpDown)
{
    auto descriptor = segment_descriptor(native_delta);
    descriptor.expert_count = descriptor.experts_per_token = 4;
    auto mapping = segment_mapping(native_delta);
    for (uint32_t layer = 0; layer < segment_layers; ++layer)
    {
        auto& moe = descriptor.layers[layer].moe;
        moe.expert_count = moe.top_k = 4;
        moe.expert_weight_dtype = dtype;
        moe.layout = layout;
        moe.activation = dtype == DType::MxFp4 ? ExpertActivation::GptOssSwiGlu : ExpertActivation::Silu;
        moe.activation_limit = dtype == DType::MxFp4 ? 7.0f : 0.0f;
        moe.flags |= MoeDescriptorProjectionBias;
        const auto prefix = "layers." + std::to_string(layer) + ".";
        mapping[prefix + "router.weight"] = bank_weight(DType::BFloat16, 4, bank_width, layer + 19);
        auto experts = bank_fixture(dtype, ExpertLayout::InterleavedGateUpDown, 4);
        for (uint32_t expert = 0; expert < experts.size(); ++expert)
        {
            auto& item = experts[expert];
            if (layout == ExpertLayout::PackedGateUpDown)
            {
                segment_check(dtype == DType::BFloat16, "paired Packed engine fixture requires BF16 weights");
                const auto original = item.gate.bfloat16_data;
                const auto bias = item.gate_bias.float32_data;
                for (uint32_t middle = 0; middle < bank_width; ++middle)
                {
                    std::copy_n(original.data() + static_cast<size_t>(middle * 2) * bank_width, bank_width,
                                item.gate.bfloat16_data.data() + static_cast<size_t>(middle) * bank_width);
                    std::copy_n(original.data() + static_cast<size_t>(middle * 2 + 1) * bank_width, bank_width,
                                item.gate.bfloat16_data.data() + static_cast<size_t>(middle + bank_width) * bank_width);
                    item.gate_bias.float32_data[middle] = bias[middle * 2];
                    item.gate_bias.float32_data[middle + bank_width] = bias[middle * 2 + 1];
                }
            }
            for (TensorData* bias : {&item.gate_bias, &item.down_bias})
            {
                bias->dtype = DType::BFloat16;
                for (float value : bias->float32_data) bias->bfloat16_data.push_back(float_to_bfloat16(value));
                bias->float32_data.clear();
            }
            const auto name = prefix + "experts." + std::to_string(expert) + ".";
            mapping[name + "gate_up.weight"] = std::move(item.gate);
            mapping[name + "down.weight"] = std::move(item.down);
            mapping[name + "gate_up.bias"] = std::move(item.gate_bias);
            mapping[name + "down.bias"] = std::move(item.down_bias);
        }
    }
    CompilerOption options;
    options.optimization_flags = bank_flags;
    options.num_concurrent_sessions = 2;
    if (gpu)
    {
        options.flags |= BackendVulkanDense | BackendVulkanAttention | BackendVulkanExperts;
        options.device_index = get_default_gpu_index();
        options.vulkan_runtime = create_vulkan_runtime();
    }
    CompiledModel model;
    const auto compiled = compile_model(descriptor, mapping, model, gpu ? HybridMode::HybridExperts : HybridMode::CpuOnly, options);
    segment_check(static_cast<bool>(compiled), compiled ? "" : compiled.error().message);
    if (!gpu) return model;
    uint64_t original_persistent = 0;
    for (const auto& device : model.memory_plan.gpu_devices) original_persistent += device.admitted.persistent_size;
    const auto prepared = prepare_model_expert_banks(model, 4, resident ? UINT64_C(16) * 1024 * 1024 : 1);
    segment_check(static_cast<bool>(prepared), prepared ? "" : prepared.error().message);
    if (resident)
    {
        segment_check(model.gpu_expert_banks.size() == segment_layers
                          && std::all_of(model.gpu_expert_banks.begin(), model.gpu_expert_banks.end(), [](const auto& bank) { return static_cast<bool>(bank); })
                          && model.gpu_expert_bank_size > 0 && !model.expert_backend,
                      "an admitted model did not select every bank before reserving its graph arena");
        uint64_t persistent = 0;
        for (const auto& device : model.memory_plan.gpu_devices)
        {
            persistent += device.admitted.persistent_size;
            segment_check(device.expert_cache_reserve == 0, "resident banks retained a second streaming cache reserve");
        }
        segment_check(persistent >= original_persistent && persistent - original_persistent == model.gpu_expert_bank_size
                          && model.opt.expert_gpu_cache_size == 0 && model.opt.expert_gpu_victim_cache_size == 0,
                      "resident bank physical bytes were missing, multiplied by Sessions, or charged twice");
    }
    else
    {
        segment_check(model.gpu_expert_banks.empty() && model.gpu_expert_bank_size == 0,
                      "insufficient expert budget did not preserve the statically selected streaming plan");
        model.expert_backend = create_vulkan_expert_backend(UINT64_C(16) * 1024 * 1024, options.device_index, {}, model.vulkan_runtime, bank_flags);
        segment_check(static_cast<bool>(model.expert_backend), "cannot create the original GPU streaming backend");
    }
    const auto sealed = prepare_model_graph_resources(model, 4);
    segment_check(static_cast<bool>(sealed), sealed ? "" : sealed.error().message);
    return model;
}

static void bank_engine(const CompiledModel& cpu, const CompiledModel& gpu, bool resident)
{
    const auto& first_moe = gpu.descriptor.layers.front().moe;
    const std::string label = std::string(resident ? "banked" : "streaming")
                              + " dtype=" + std::to_string(static_cast<int>(first_moe.expert_weight_dtype))
                              + " layout=" + std::to_string(static_cast<int>(first_moe.layout))
                              + " nativeGDN=" + std::to_string(gpu.descriptor.layers[1].attention.projection_weight_dtype == DType::Float32);
    std::cerr << "bank engine fixture " << label << '\n';
    for (const auto output : {LogitsOutput::All, LogitsOutput::Last, LogitsOutput::None})
    {
        constexpr size_t rows = 3;
        const std::array<int32_t, 3> tokens = {2, 5, 1};
        const std::span<const int32_t> ids(tokens.data(), rows);
        SessionState expected, actual;
        SessionStatistics expected_stats, actual_stats;
        const auto reference = forward_model(cpu, ids, expected_stats, expected, 0, output);
        segment_check(static_cast<bool>(reference), reference ? "" : reference.error().message);
        const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
        const auto result = forward_model(gpu, ids, actual_stats, actual, 0, output);
        segment_check(static_cast<bool>(result), result ? "" : result.error().message);
        segment_compare(actual.logits.values(), expected.logits.values(), "sealed engine " + label + " rows=" + std::to_string(rows) + " logits=" + std::to_string(static_cast<int>(output)));
        segment_gpu_routes(actual_stats);
        const auto after = get_vulkan_statistics(gpu.vulkan_runtime);
        if (resident)
        {
            segment_check(after.resident_graph_route_download_bytes == before.resident_graph_route_download_bytes
                              && after.resident_graph_activation_download_bytes == before.resident_graph_activation_download_bytes,
                          "full-bank execution materialized Router IDs or hidden activations");
        }
        else
            segment_check(after.resident_graph_route_download_bytes - before.resident_graph_route_download_bytes
                              == rows * segment_layers * Router_vulkan::selected_columns(4) * sizeof(float),
                          "the insufficient-budget model did not retain its original GPU streaming route boundary");
        for (const auto& state : actual.layers)
            segment_check(!state.vulkan_attention_state_unknown, "successful bank/streaming execution poisoned Session state");
    }
    std::array<SessionState, 2> expected, actual;
    std::array<SessionStatistics, 2> expected_stats, actual_stats;
    std::array<std::vector<float>, 2> logits;
    std::array<DecodeBatchEntry, 2> entries;
    for (size_t row = 0; row < entries.size(); ++row)
    {
        const std::array<int32_t, 1> token = {static_cast<int32_t>(row + 2)};
        const auto reference = forward_model(cpu, token, expected_stats[row], expected[row], 0, LogitsOutput::All);
        segment_check(static_cast<bool>(reference), reference ? "" : reference.error().message);
        entries[row] = {token[0], &actual_stats[row], &actual[row], 0, &logits[row]};
    }
    BatchWorkspace workspace;
    const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
    const auto result = forward_decode_batch(gpu, entries, workspace);
    segment_check(static_cast<bool>(result), result ? "" : result.error().message);
    const auto after = get_vulkan_statistics(gpu.vulkan_runtime);
    if (resident)
        segment_check(after.resident_graph_route_download_bytes == before.resident_graph_route_download_bytes
                          && after.resident_graph_activation_download_bytes == before.resident_graph_activation_download_bytes,
                      "staged resident banks downloaded routes or intermediate activations");
    for (size_t row = 0; row < entries.size(); ++row)
    {
        segment_compare(logits[row], expected[row].logits.values(), "staged independent Session " + label + " row=" + std::to_string(row));
        segment_gpu_routes(actual_stats[row]);
        segment_check(actual[row].layers[0].token_count == 1 && actual[row].layers[1].gated_delta_token_count == 1,
                      "staged bank execution committed another Session's state");
    }
}

static void bank_chain_abort(const CompiledModel& gpu)
{
    const std::array<int32_t, 1> ids = {3};
    ActivationBuffer input;
    forward_embedding(gpu.weights.at(gpu.token_embedding), ids, input);
    std::array<LayerCache, segment_layers> caches;
    std::array<LayerCache*, segment_layers> pointers;
    std::array<ExpertDispatchOptions, segment_layers> options;
    for (size_t layer = 0; layer < segment_layers; ++layer)
    {
        pointers[layer] = &caches[layer];
        options[layer].expert_count = options[layer].top_k = 4;
    }
    const std::array<uint64_t, 1> positions = {0};
    ActivationBuffer logits(1, 1);
    logits.row(0)[0] = -73.0f;
    std::string error;
    options[0].top_k = 0;
    const auto before = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(!GpuSegment_vulkan::forward_banks(gpu, input, positions, pointers, options, true, true, logits, error)
                      && logits.rows() == 1 && logits.columns() == 1 && logits.row(0)[0] == -73.0f
                      && get_vulkan_statistics(gpu.vulkan_runtime).compute_submissions == before.compute_submissions,
                  "invalid unsubmitted full-bank transaction published output or submitted work");
    for (const auto& cache : caches)
        segment_check(!cache.vulkan_attention_state_unknown && cache.token_count == 0 && cache.gated_delta_token_count == 0,
                      "unsubmitted bank rejection modified Session state");
    options[0].top_k = 4;
    segment_check(gpu.descriptor.layers[1].attention.kind == AttentionKind::GatedDeltaNet,
                  "unknown-state regression requires a fresh GDN cache");
    caches[1].vulkan_attention_state_unknown = true;
    error.clear();
    segment_check(!GpuSegment_vulkan::forward_banks(gpu, input, positions, pointers, options, true, true, logits, error)
                      && !error.empty() && logits.rows() == 1 && logits.columns() == 1 && logits.row(0)[0] == -73.0f
                      && get_vulkan_statistics(gpu.vulkan_runtime).compute_submissions == before.compute_submissions,
                  "fresh unknown GDN state was submitted or published by the resident bank chain");
    for (size_t layer = 0; layer < caches.size(); ++layer)
    {
        const auto& cache = caches[layer];
        segment_check(cache.vulkan_attention_state_unknown == (layer == 1) && cache.token_count == 0 && cache.gated_delta_token_count == 0
                          && !cache.vulkan_attention_cache && !cache.gated_delta_device_state && cache.device_allocated_size == 0
                          && cache.gated_delta_convolution.empty() && cache.gated_delta_recurrent.empty(),
                      "unknown GDN preflight rejection allocated, reset, or published Session state");
    }
    caches[1].vulkan_attention_state_unknown = false;
    {
        std::array<LayerCache, segment_layers> late_caches;
        std::array<LayerCache*, segment_layers> late_pointers;
        for (size_t layer = 0; layer < segment_layers; ++layer) late_pointers[layer] = &late_caches[layer];
        late_caches[1].gated_delta_token_count = std::numeric_limits<uint64_t>::max();
        segment_check(!GpuSegment_vulkan::forward_banks(gpu, input, positions, late_pointers, options, true, true, logits, error)
                          && logits.rows() == 1 && logits.columns() == 1 && logits.row(0)[0] == -73.0f,
                      "recording failure after an asynchronous submit published output");
        for (const auto& cache : late_caches)
            segment_check(cache.vulkan_attention_state_unknown && cache.token_count == 0,
                          "late recording failure did not drain and invalidate the partially submitted chain");
        segment_check(late_caches[1].gated_delta_token_count == std::numeric_limits<uint64_t>::max(),
                      "late recording failure published GDN state");
        segment_check(!GpuSegment_vulkan::forward_banks(gpu, input, positions, late_pointers, options, true, true, logits, error),
                      "partially submitted state was replayed without reset");
    }
    const std::array<uint32_t, 4> duplicate = {0, 0, 1, 2};
    options[0].explicit_expert_ids = duplicate;
    segment_check(!GpuSegment_vulkan::forward_banks(gpu, input, positions, pointers, options, true, true, logits, error)
                      && logits.row(0)[0] == -73.0f && error.find("invalid GPU routing") != std::string::npos,
                  "a submitted invalid GPU route was published or retried on another backend");
    for (const auto& cache : caches)
        segment_check(cache.vulkan_attention_state_unknown && cache.token_count == 0 && cache.gated_delta_token_count == 0,
                      "failed submitted bank transaction did not poison its unpublished in-place state");
    options[0].explicit_expert_ids = {};
    const auto failed = get_vulkan_statistics(gpu.vulkan_runtime);
    segment_check(!GpuSegment_vulkan::forward_banks(gpu, input, positions, pointers, options, true, true, logits, error)
                      && get_vulkan_statistics(gpu.vulkan_runtime).compute_submissions == failed.compute_submissions
                      && logits.row(0)[0] == -73.0f,
                  "a bank transaction reused submitted state after failure without reset");
}

static void bank_models()
{
    for (const DType dtype : {DType::MxFp4, DType::BFloat16, DType::Float32})
    {
        const auto layout = dtype == DType::BFloat16 ? ExpertLayout::PackedGateUpDown : ExpertLayout::InterleavedGateUpDown;
        const auto cpu = bank_model(false, dtype, false, true, layout);
        const auto streaming = bank_model(true, dtype, false, false, layout);
        bank_engine(cpu, streaming, false);
        const auto resident = bank_model(true, dtype, false, true, layout);
        bank_engine(cpu, resident, true);
        bank_chain_abort(resident);
    }
    const auto interleaved_cpu = bank_model(false, DType::BFloat16, false, true);
    const auto interleaved_bank = bank_model(true, DType::BFloat16, false, true);
    bank_engine(interleaved_cpu, interleaved_bank, true);
    const auto cpu = bank_model(false, DType::BFloat16, true, true, ExpertLayout::PackedGateUpDown);
    const auto native_delta = bank_model(true, DType::BFloat16, true, true, ExpertLayout::PackedGateUpDown);
    bank_engine(cpu, native_delta, true);
}

static void ordered_batch_cases(const std::shared_ptr<VulkanContext>& context)
{
    const std::lock_guard<std::mutex> lock(context->command_mutex());
    const auto option = bank_option(*context);
    ncnn::Mat source(4096), result;
    for (int i = 0; i < source.w; ++i) source[i] = static_cast<float>(i) / 16;
    // Repeat to exercise command/fence reuse. Leaving the last run without an
    // explicit wait exercises exception/early-return ownership retirement.
    for (int iteration = 0; iteration < 3; ++iteration)
    {
        std::array<ncnn::VkMat, 2> values;
        {
            VulkanComputeBatch batches(*context);
            for (size_t i = 0; i < values.size(); ++i)
            {
                auto* command = batches.next();
                segment_check(command != nullptr, "cannot acquire ordered compute command");
                segment_check(batches.next() == nullptr, "unsubmitted recording was silently abandoned");
                if (i == 0)
                    command->record_clone(source, values[i], option);
                else
                    command->record_clone(values[i - 1], values[i], option);
                segment_check(!values[i].empty(), "cannot allocate ordered dependency operand");
                if (i + 1 == values.size()) command->record_clone(values[i], result, option);
                segment_check(batches.submit() == 0 && batches.pending_count() == i + 1,
                              "ordered submissions waited before their retirement boundary");
            }
            if (iteration != 2)
                segment_check(batches.wait() == 0 && batches.pending_count() == 0 && batches.submission_count() == values.size(),
                              "ordered command fences did not retire exactly once");
        }
        segment_check(result.w == source.w, "ordered output was not published after drain");
        for (int i = 0; i < source.w; ++i)
            segment_check(result[i] == source[i], "cross-submission producer visibility failed");
    }
}

static void bank_device_cases(uint32_t device_index, bool check_capability)
{
    const auto context = VulkanContext::acquire(device_index, create_vulkan_runtime(), bank_flags);
    segment_check(static_cast<bool>(context), "cannot acquire expert bank test context on device " + std::to_string(device_index));
    const auto& info = context->device()->info;
    const std::string label = "bank device " + std::to_string(device_index) + " (" + info.device_name()
                              + ", reported subgroup " + std::to_string(info.subgroup_size()) + ")";
    std::cerr << label << " direct/canonical/validation/preparation fixtures\n";
    ordered_batch_cases(context);
    if (check_capability)
    {
        const auto fixtures = bank_fixture(DType::Float32, ExpertLayout::InterleavedGateUpDown, 16, bank_width + 1, bank_width + 1);
        std::vector<ExpertBankWeights_vulkan> descriptions;
        for (const auto& fixture : fixtures) descriptions.push_back(bank_view(fixture, ExpertLayout::InterleavedGateUpDown, ExpertActivation::Silu));
        const auto planned = ExpertBank_vulkan::plan(descriptions, context);
        if (!planned && planned.error().code == ErrorCode::UnsupportedModel)
        {
            std::cout << "SKIP " << label << ": " << planned.error().message << '\n';
            return;
        }
        segment_check(static_cast<bool>(planned), planned ? "" : label + ": " + planned.error().message);
    }
    try
    {
        bank_direct(context);
        bank_canonical_and_invalid(context);
        bank_preparation_failures(context);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(label + ": " + error.what());
    }
    std::cout << label << " passed\n";
}

static int bank_run()
{
    if (get_gpu_count() == 0) return 77;
    const uint32_t default_device = get_default_gpu_index();
    bank_device_cases(default_device, false);
    for (uint32_t device_index = 0; device_index < get_gpu_count(); ++device_index)
        if (device_index != default_device) bank_device_cases(device_index, true);
    bank_models();
    std::cout << "Resident Expert bank dtype/layout/K/row parity, canonical combine, abort, validation, and lease ownership passed\n";
    return 0;
}
#endif

} // namespace moe
} // namespace ncnn

int main()
{
#if NCNN_MOE_WITH_VULKAN
    try
    {
        return ncnn::moe::bank_run();
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
