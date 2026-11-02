#include "vulkan_test_support.h"
#include "backends/ncnn/expertbank_vulkan.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
enum class BankCodeBasis
{
    None,
    GateUp,
    Down
};

struct BankNumericCase
{
    uint32_t intermediate;
    uint32_t output;
    uint8_t scale;
    int input_exponent;
    bool random_scales;
    bool cancellation;
    uint32_t input_columns = 128;
    bool projected_input = false;
    bool subnormal_gate_scale = false;
    BankCodeBasis code_basis = BankCodeBasis::None;
};

struct BankNumericExpert
{
    TensorData gate;
    TensorData down;
    TensorData gate_bias;
    TensorData down_bias;
};

static TensorData numeric_mxfp4(uint32_t rows, uint32_t columns)
{
    TensorData tensor;
    tensor.dtype = DType::MxFp4;
    tensor.shape = {rows, columns};
    tensor.mxfp4_blocks.resize(static_cast<size_t>(rows) * columns / 2);
    tensor.mxfp4_scales.resize(static_cast<size_t>(rows) * columns / 32);
    std::fill_n(tensor.mxfp4_blocks.data(), tensor.mxfp4_blocks.size(), uint8_t{0});
    std::fill_n(tensor.mxfp4_scales.data(), tensor.mxfp4_scales.size(), uint8_t{127});
    return tensor;
}

static std::array<BankNumericExpert, 4> numeric_experts(const BankNumericCase& test)
{
    std::array<BankNumericExpert, 4> experts;
    for (uint32_t expert = 0; expert < experts.size(); ++expert)
    {
        auto& item = experts[expert];
        item.gate = numeric_mxfp4(test.intermediate * 2, test.input_columns);
        item.down = numeric_mxfp4(test.output, test.intermediate);
        item.gate_bias.dtype = item.down_bias.dtype = DType::Float32;
        item.gate_bias.shape = {test.intermediate * 2};
        item.down_bias.shape = {test.output};
        item.gate_bias.float32_data.resize(test.intermediate * 2);
        if (test.code_basis != BankCodeBasis::None)
        {
            // SiLU(128) rounds to exactly 128. One coefficient per output
            // exposes each FP4 code at different packed-word positions as
            // a dyadic result without cancellation.
            std::fill_n(item.gate_bias.float32_data.data(), test.intermediate, 128.0f);
            item.down_bias.float32_data.resize(test.output, 0.0f);
            if (test.code_basis == BankCodeBasis::GateUp)
            {
                for (uint32_t middle = 0; middle < test.intermediate; ++middle)
                {
                    const size_t index = static_cast<size_t>(test.intermediate + middle) * test.input_columns + middle % 8;
                    item.gate.mxfp4_blocks[index / 2] = static_cast<uint8_t>((middle % 16) << ((index % 2) * 4));
                }
                std::fill_n(item.gate.mxfp4_scales.data(), item.gate.mxfp4_scales.size(), test.scale);
                for (uint32_t column = 0; column < test.output; ++column)
                {
                    const size_t index = static_cast<size_t>(column) * test.intermediate + column;
                    item.down.mxfp4_blocks[index / 2] = static_cast<uint8_t>(2u << ((index % 2) * 4));
                }
            }
            else
            {
                std::fill_n(item.gate_bias.float32_data.data() + test.intermediate, 8, std::ldexp(1.0f, test.input_exponent - 7));
                for (uint32_t column = 0; column < test.output; ++column)
                {
                    const size_t index = static_cast<size_t>(column) * test.intermediate + column % 8;
                    item.down.mxfp4_blocks[index / 2] = static_cast<uint8_t>((column % 16) << ((index % 2) * 4));
                }
                std::fill_n(item.down.mxfp4_scales.data(), item.down.mxfp4_scales.size(), test.scale);
            }
            continue;
        }
        for (uint32_t middle = 0; middle < test.intermediate; ++middle)
        {
            item.gate_bias.float32_data[middle] = 1.0f;
            const float magnitude = std::ldexp(1.0f + expert * 0.125f, test.input_exponent);
            item.gate_bias.float32_data[test.intermediate + middle] = test.random_scales
                                                                          ? magnitude * (static_cast<int>(middle % 13) - 6) * 0.125f
                                                                          : magnitude;
        }
        for (uint32_t column = 0; column < test.output; ++column)
            item.down_bias.float32_data.push_back((static_cast<int>((column + expert) % 7) - 3) * 0.125f);
        for (size_t index = 0; index < static_cast<size_t>(test.output) * test.intermediate; ++index)
        {
            const uint8_t code = test.random_scales  ? static_cast<uint8_t>((index * 13 + expert * 3) % 16)
                                 : test.cancellation ? static_cast<uint8_t>((index % 8 < 4) ? 7 : 15)
                                                     : uint8_t{7};
            item.down.mxfp4_blocks[index / 2] |= static_cast<uint8_t>(code << ((index % 2) * 4));
        }
        for (size_t block = 0; block < item.down.mxfp4_scales.size(); ++block)
            item.down.mxfp4_scales[block] = test.random_scales ? static_cast<uint8_t>(72 + (block * 31 + expert * 7) % 111) : test.scale;
        if (test.projected_input)
        {
            for (size_t index = 0; index < item.gate.mxfp4_blocks.size(); ++index)
                item.gate.mxfp4_blocks[index] = static_cast<uint8_t>(((index * 7 + expert * 3) % 16) | (((index * 11 + index / 7 + expert * 5) % 16) << 4));
            for (size_t block = 0; block < item.gate.mxfp4_scales.size(); ++block)
                item.gate.mxfp4_scales[block] = test.subnormal_gate_scale ? uint8_t{0}
                                                                          : static_cast<uint8_t>(120 + (block * 7 + expert) % 5);
        }
    }
    return experts;
}

static ExpertBankWeights_vulkan numeric_view(const BankNumericExpert& expert)
{
    ExpertBankWeights_vulkan weights;
    weights.gate_up = &expert.gate;
    weights.down = &expert.down;
    weights.gate_up_bias = &expert.gate_bias;
    weights.down_bias = &expert.down_bias;
    weights.layout = ExpertLayout::PackedGateUpDown;
    weights.activation = ExpertActivation::Silu;
    return weights;
}

static double numeric_value(const TensorData& tensor, size_t index)
{
    static constexpr double fp4[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};
    const uint8_t code = static_cast<uint8_t>((tensor.mxfp4_blocks[index / 2] >> ((index % 2) * 4)) & 15);
    return std::ldexp(fp4[code], static_cast<int>(tensor.mxfp4_scales[index / 32]) - 127);
}

static ActivationBuffer numeric_reference(const std::array<BankNumericExpert, 4>& experts,
                                          const BankNumericCase& test, const ActivationBuffer& input,
                                          const ActivationBuffer& selected, uint32_t top_k)
{
    std::array<std::vector<float>, 4> projected;
    for (size_t expert = 0; expert < experts.size(); ++expert)
    {
        const auto& item = experts[expert];
        for (size_t row = 0; row < input.rows(); ++row)
        {
            std::vector<double> activated(test.intermediate);
            for (uint32_t middle = 0; middle < test.intermediate; ++middle)
            {
                double gate = item.gate_bias.float32_data[middle];
                double up = item.gate_bias.float32_data[test.intermediate + middle];
                if (test.projected_input)
                    for (uint32_t k = 0; k < test.input_columns; ++k)
                    {
                        gate += numeric_value(item.gate, static_cast<size_t>(middle) * test.input_columns + k) * input.row(row)[k];
                        up += numeric_value(item.gate, static_cast<size_t>(test.intermediate + middle) * test.input_columns + k) * input.row(row)[k];
                    }
                activated[middle] = gate / (1.0 + std::exp(-gate)) * up;
            }
            for (uint32_t column = 0; column < test.output; ++column)
            {
                double value = item.down_bias.float32_data[column];
                for (uint32_t middle = 0; middle < test.intermediate; ++middle)
                {
                    const size_t index = static_cast<size_t>(column) * test.intermediate + middle;
                    value += numeric_value(item.down, index) * activated[middle];
                }
                projected[expert].push_back(static_cast<float>(value));
            }
        }
    }
    ActivationBuffer output(selected.rows(), test.output);
    for (size_t row = 0; row < selected.rows(); ++row)
        for (uint32_t column = 0; column < test.output; ++column)
        {
            float value = 0.0f;
            for (uint32_t expert = 0; expert < experts.size(); ++expert)
                for (uint32_t rank = 0; rank < top_k; ++rank)
                    if (selected.row(row)[rank] == static_cast<float>(expert))
                    {
                        volatile float term = projected[expert][row * test.output + column] * selected.row(row)[top_k + rank];
                        value += term;
                    }
            output.row(row)[column] = value;
        }
    return output;
}

static void numeric_fusion_case(const std::shared_ptr<VulkanContext>& context, const ExpertBank_vulkan& bank,
                                const ncnn::Option& option,
                                const ActivationBuffer& input, const ActivationBuffer& selected,
                                const ActivationBuffer& expected)
{
    ActivationBuffer shared(input.rows(), expected.columns()), residual(input.rows(), expected.columns());
    for (size_t row = 0; row < input.rows(); ++row)
        for (uint32_t column = 0; column < expected.columns(); ++column)
        {
            shared.row(row)[column] = 33554432.0f;
            residual.row(row)[column] = -33554432.0f;
        }
    ActivationBuffer residual_only(input.rows(), expected.columns()), fused(input.rows(), expected.columns()), flag(1, 1);
    {
        const std::lock_guard<std::mutex> lock(context->command_mutex());
        ncnn::VkCompute command(context->device());
        ncnn::VkMat input_staging, selected_staging, shared_staging, residual_staging, zero_staging;
        ncnn::VkMat device_input, device_selected, device_shared, device_residual, validation;
        ncnn::VkMat residual_output, fused_output, residual_output_staging, fused_output_staging, flag_staging;
        std::vector<ncnn::VkMat> workspace;
        const uint32_t zero = 0;
        segment_check(fill_staging_upload(input, input_staging, option.staging_vkallocator)
                          && record_mapped_upload(input_staging, device_input, command, option)
                          && fill_staging_upload(selected, selected_staging, option.staging_vkallocator)
                          && record_mapped_upload(selected_staging, device_selected, command, option)
                          && fill_staging_upload(shared, shared_staging, option.staging_vkallocator)
                          && record_mapped_upload(shared_staging, device_shared, command, option)
                          && fill_staging_upload(residual, residual_staging, option.staging_vkallocator)
                          && record_mapped_upload(residual_staging, device_residual, command, option)
                          && fill_staging_values(&zero, 1, sizeof(zero), zero_staging, option.staging_vkallocator)
                          && record_mapped_upload(zero_staging, validation, command, option)
                          && bank.record(device_input, device_selected, 1, residual_output, validation, command, workspace,
                                         nullptr, &device_residual)
                          && bank.record(device_input, device_selected, 1, fused_output, validation, command, workspace,
                                         &device_shared, &device_residual)
                          && prepare_staging_batch(residual_output_staging, input.rows(), expected.columns(), option.staging_vkallocator)
                          && record_prepared_activation_staging_download(residual_output, input.rows(), expected.columns(),
                                                                         residual_output_staging, command, context->device(), option)
                          && prepare_staging_batch(fused_output_staging, input.rows(), expected.columns(), option.staging_vkallocator)
                          && record_prepared_activation_staging_download(fused_output, input.rows(), expected.columns(),
                                                                         fused_output_staging, command, context->device(), option)
                          && prepare_staging_batch(flag_staging, 1, 1, option.staging_vkallocator)
                          && record_prepared_activation_staging_download(validation, 1, 1, flag_staging,
                                                                         command, context->device(), option)
                          && submit_compute_and_wait(command, context->device()) == 0
                          && copy_staging_to_cpu_batch(residual_output_staging, residual_only)
                          && copy_staging_to_cpu_batch(fused_output_staging, fused)
                          && copy_staging_to_cpu_batch(flag_staging, flag),
                      "cannot execute fused Expert bank numerical fixture");
    }
    segment_check(std::bit_cast<uint32_t>(flag.row(0)[0]) == 0, "valid fused routes set validation failure");
    for (size_t row = 0; row < input.rows(); ++row)
        for (size_t column = 0; column < expected.columns(); ++column)
        {
            volatile float residual_expected = residual.row(row)[column] + expected.row(row)[column];
            volatile float after_shared = expected.row(row)[column] + shared.row(row)[column];
            volatile float fused_expected = residual.row(row)[column] + after_shared;
            segment_check(std::bit_cast<uint32_t>(residual_only.row(row)[column])
                              == std::bit_cast<uint32_t>(residual_expected),
                          "residual-only Expert bank combine changed FP32 order");
            segment_check(std::bit_cast<uint32_t>(fused.row(row)[column]) == std::bit_cast<uint32_t>(fused_expected),
                          "shared/residual Expert bank combine changed FP32 order");
        }
}

static void numeric_case(const std::shared_ptr<VulkanContext>& context, const BankNumericCase& test)
{
    const auto experts = numeric_experts(test);
    std::array<ExpertBankWeights_vulkan, 4> weights;
    for (size_t expert = 0; expert < experts.size(); ++expert) weights[expert] = numeric_view(experts[expert]);
    const auto planned = ExpertBank_vulkan::plan(weights, context);
    segment_check(static_cast<bool>(planned), planned ? "" : planned.error().message);
    const auto created = ExpertBank_vulkan::create(planned.value(), context, [&](uint32_t expert) -> Result<ExpertBankWeights_vulkan> { return weights[expert]; });
    segment_check(static_cast<bool>(created), created ? "" : created.error().message);
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    option.blob_vkallocator = context->blob_allocator();
    option.workspace_vkallocator = context->blob_allocator();
    option.staging_vkallocator = context->staging_allocator();
    // Keep every numerical stress case; pair decode with small K and prefill with K=1.
    const std::array<std::pair<uint32_t, size_t>, 2> selections = {{{4, 1}, {1, 3}}};
    for (const auto& [top_k, rows] : selections)
    {
        ActivationBuffer input(rows, test.input_columns), selected(rows, top_k * 2 + 1);
        // Distinct rows catch accidental reuse of the first row's projection.
        for (size_t row = 0; row < rows; ++row)
            for (uint32_t k = 0; k < test.input_columns; ++k)
                input.row(row)[k] = test.code_basis == BankCodeBasis::GateUp
                                        ? (k < 8 ? std::ldexp(1.0f, test.input_exponent - 7) : 0.0f)
                                    : test.projected_input
                                        ? std::ldexp(static_cast<float>(static_cast<int>((k * 7 + k / 13 + row * 5) % 19) - 9), test.subnormal_gate_scale ? 120 : -6)
                                        : 0.0f;
        for (size_t row = 0; row < rows; ++row)
        {
            for (uint32_t rank = 0; rank < top_k; ++rank)
            {
                selected.row(row)[rank] = static_cast<float>((3 + row - rank) % 4);
                selected.row(row)[top_k + rank] = 1.0f / top_k;
            }
            selected.row(row)[top_k * 2] = 0.0f;
        }
        const auto expected = numeric_reference(experts, test, input, selected, top_k);
        ActivationBuffer actual(rows, test.output), flag(1, 1);
        {
            const std::lock_guard<std::mutex> lock(context->command_mutex());
            ncnn::VkCompute command(context->device());
            ncnn::VkMat input_staging, selected_staging, zero_staging, device_input, device_selected, validation, output, output_staging, flag_staging;
            std::vector<ncnn::VkMat> workspace;
            const uint32_t zero = 0;
            segment_check(fill_staging_upload(input, input_staging, option.staging_vkallocator)
                              && record_mapped_upload(input_staging, device_input, command, option)
                              && fill_staging_upload(selected, selected_staging, option.staging_vkallocator)
                              && record_mapped_upload(selected_staging, device_selected, command, option)
                              && fill_staging_values(&zero, 1, sizeof(zero), zero_staging, option.staging_vkallocator)
                              && record_mapped_upload(zero_staging, validation, command, option)
                              && created.value()->record(device_input, device_selected, top_k, output, validation, command, workspace)
                              && prepare_staging_batch(output_staging, rows, test.output, option.staging_vkallocator)
                              && record_prepared_activation_staging_download(output, rows, test.output, output_staging, command, context->device(), option)
                              && prepare_staging_batch(flag_staging, 1, 1, option.staging_vkallocator)
                              && record_prepared_activation_staging_download(validation, 1, 1, flag_staging, command, context->device(), option)
                              && submit_compute_and_wait(command, context->device()) == 0
                              && copy_staging_to_cpu_batch(output_staging, actual) && copy_staging_to_cpu_batch(flag_staging, flag),
                          "cannot execute the indexed MXFP4 numerical fixture");
        }
        segment_check(std::bit_cast<uint32_t>(flag.row(0)[0]) == 0, "valid MXFP4 numeric routes set validation failure");
        const std::string label = "MXFP4 FP64 oracle H=" + std::to_string(test.input_columns)
                                  + " I=" + std::to_string(test.intermediate) + " O=" + std::to_string(test.output)
                                  + " scale=" + std::to_string(test.scale) + " exponent=" + std::to_string(test.input_exponent)
                                  + " random=" + std::to_string(test.random_scales) + " cancellation=" + std::to_string(test.cancellation)
                                  + " subnormal_gate_scale=" + std::to_string(test.subnormal_gate_scale)
                                  + " code_basis=" + std::to_string(static_cast<int>(test.code_basis))
                                  + " K=" + std::to_string(top_k) + " rows=" + std::to_string(rows);
        segment_compare(actual.values(), expected.values(), label);
        if (test.code_basis != BankCodeBasis::None)
        {
            static constexpr float fp4[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, 0, -0.5f, -1, -1.5f, -2, -3, -4, -6};
            for (size_t row = 0; row < rows; ++row)
                for (uint32_t column = 0; column < test.output; ++column)
                {
                    const float exact = std::ldexp(fp4[column % 16], static_cast<int>(test.scale) - 127 + test.input_exponent);
                    // +0 accumulation normalizes either zero code's sign.
                    segment_check(actual.row(row)[column] == exact && expected.row(row)[column] == exact,
                                  label + " exact FP4 code mismatch at " + std::to_string(column));
                }
        }
        if (top_k == 1 && test.code_basis == BankCodeBasis::Down && test.output == 5)
            numeric_fusion_case(context, *created.value(), option, input, selected, expected);
    }
}

static int numeric_run()
{
    if (get_gpu_count() == 0) return 77;
    const std::array<BankNumericCase, 18> cases = {{{32, 5, 7, 123, false, false, 32},
                                                    {32, 1, 1, 120, false, true},
                                                    {128, 3, 0, 120, false, false},
                                                    {128, 129, 254, -120, false, false},
                                                    {128, 4, 64, -31, false, false},
                                                    {128, 5, 190, 32, false, false},
                                                    {2880, 5, 127, 0, true, false},
                                                    {4096, 4, 127, 0, true, false},
                                                    {128, 5, 127, 0, true, false, 2880, true},
                                                    {128, 5, 127, 0, true, false, 4096, true},
                                                    {128, 5, 127, 0, false, false, 128, true, true},
                                                    {32, 16, 127, 0, false, false, 32, true, false, BankCodeBasis::GateUp},
                                                    {32, 16, 0, 123, false, false, 32, true, false, BankCodeBasis::GateUp},
                                                    {32, 16, 254, -118, false, false, 32, true, false, BankCodeBasis::GateUp},
                                                    {32, 16, 127, 0, false, false, 32, false, false, BankCodeBasis::Down},
                                                    {32, 16, 0, 123, false, false, 32, false, false, BankCodeBasis::Down},
                                                    {32, 16, 254, -118, false, false, 32, false, false, BankCodeBasis::Down},
                                                    {32, 5, 127, 0, false, false, 32, false, false, BankCodeBasis::Down}}};
    for (uint32_t device = 0; device < get_gpu_count(); ++device)
    {
        const auto context = VulkanContext::acquire(device, create_vulkan_runtime(), segment_flags);
        segment_check(static_cast<bool>(context), "cannot acquire numerical test Vulkan context");
        std::cout << "MXFP4 numerical fixtures device " << device << " " << context->device()->info.device_name()
                  << " subgroup " << context->device()->info.subgroup_size() << '\n';
        for (const auto& test : cases) numeric_case(context, test);
    }
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
        return ncnn::moe::numeric_run();
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
