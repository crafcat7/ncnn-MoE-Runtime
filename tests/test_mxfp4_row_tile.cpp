#include "backends/ncnn/linear.h"
#include "backends/ncnn/vulkan.h"
#include "kernels/ops.h"
#include "ncnn/moe/option.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static void row_tile_check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static TensorData row_tile_weight(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData weight;
    weight.dtype = DType::MxFp4;
    weight.shape = {rows, columns};
    weight.mxfp4_blocks.resize(static_cast<size_t>(rows) * columns / 2);
    weight.mxfp4_scales.resize(static_cast<size_t>(rows) * columns / 32);
    for (size_t index = 0; index < weight.mxfp4_blocks.size(); ++index)
    {
        const uint8_t low = static_cast<uint8_t>((index * 7 + seed * 3 + index / 11) % 16);
        const uint8_t high = static_cast<uint8_t>((index * 13 + seed * 5 + index / 17) % 16);
        weight.mxfp4_blocks[index] = static_cast<uint8_t>(low | high << 4);
    }
    for (size_t index = 0; index < weight.mxfp4_scales.size(); ++index)
        weight.mxfp4_scales[index] = index % 19 == 0 ? uint8_t(0) : static_cast<uint8_t>(123 + (index * 5 + seed) % 3);
    return weight;
}

static TensorData row_tile_bias(uint32_t columns, uint32_t seed)
{
    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {columns};
    bias.float32_data.resize(columns);
    for (uint32_t column = 0; column < columns; ++column)
        bias.float32_data[column] = static_cast<float>(static_cast<int>((column * 7 + seed) % 11) - 5) * 0.03125f;
    return bias;
}

static double row_tile_compare(const ActivationBuffer& actual, const ActivationBuffer& expected,
                               const std::string& label, float absolute, float relative)
{
    row_tile_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), label + ": shape");
    float maximum = 0.0f;
    double maximum_error = 0.0;
    for (size_t index = 0; index < expected.values().size(); ++index)
    {
        const float reference = expected.values()[index];
        const float observed = actual.values()[index];
        maximum = std::max(maximum, std::abs(reference));
        maximum_error = std::max(maximum_error, std::abs(static_cast<double>(observed) - reference));
        row_tile_check(std::isfinite(reference) && std::isfinite(observed)
                           && std::abs(observed - reference) <= absolute + relative * std::abs(reference),
                       label + ": element " + std::to_string(index) + " actual=" + std::to_string(observed)
                           + " expected=" + std::to_string(reference));
    }
    row_tile_check(maximum > 0.00001f, label + ": nonzero oracle");
    return maximum_error;
}

static void row_tile_test()
{
    const auto runtime = create_vulkan_runtime();
    const uint64_t oracle_flags = OptimizationDefaultFlags & ~OptimizationCpuMxfp4Q8 & ~OptimizationCpuFastSilu;
    const uint64_t flags_off = oracle_flags & ~OptimizationVulkanMxfp4RowTile;
    const uint64_t flags_on = oracle_flags | OptimizationVulkanMxfp4RowTile;
    struct Shape
    {
        uint32_t input;
        uint32_t intermediate;
        uint32_t output;
        bool bias;
    };
    const std::array<Shape, 3> shapes = {{{32, 32, 7, false}, {96, 64, 65, true}, {128, 96, 31, true}}};
    const std::array<ExpertActivation, 3> activations = {ExpertActivation::GptOssSwiGlu, ExpertActivation::DeepSeekSwiGlu, ExpertActivation::Silu};
    const std::array<size_t, 8> row_counts = {1, 3, 4, 5, 15, 16, 17, 32};
    double maximum_tile_error = 0.0;
    std::string maximum_tile_case;
    for (const auto& shape : shapes)
    {
        const auto gate_up = row_tile_weight(shape.intermediate * 2, shape.input, 1);
        const auto down = row_tile_weight(shape.output, shape.intermediate, 3);
        const auto gate_up_bias = row_tile_bias(shape.intermediate * 2, 2);
        const auto down_bias = row_tile_bias(shape.output, 4);
        const TensorData* gate_bias = shape.bias ? &gate_up_bias : nullptr;
        const TensorData* projection_bias = shape.bias ? &down_bias : nullptr;
        for (const auto activation : activations)
        {
            for (const float limit : {0.0f, 0.125f})
            {
                auto scalar = Mxfp4Expert_vulkan::create(gate_up, gate_bias, down, projection_bias, limit, 0,
                                                         activation, runtime, flags_off);
                auto tiled = Mxfp4Expert_vulkan::create(gate_up, gate_bias, down, projection_bias, limit, 0,
                                                        activation, runtime, flags_on);
                row_tile_check(scalar && tiled, "row tile expert creation");
                for (const size_t rows : row_counts)
                {
                    const std::string label = "rows=" + std::to_string(rows) + " input=" + std::to_string(shape.input)
                                              + " intermediate=" + std::to_string(shape.intermediate) + " output=" + std::to_string(shape.output)
                                              + " activation=" + std::to_string(static_cast<int>(activation)) + " limit=" + std::to_string(limit);
                    ActivationBuffer input(rows, shape.input);
                    for (size_t row = 0; row < rows; ++row)
                        for (uint32_t column = 0; column < shape.input; ++column)
                            input.row(row)[column] = static_cast<float>(static_cast<int>((row * 17 + column * 7 + column / 13) % 29) - 14) * 0.015625f;
                    ActivationBuffer activated;
                    forward_gate_up_mxfp4(gate_up, gate_bias, input, activation, limit, activated, oracle_flags);
                    const ActivationBuffer expected = projection_bias
                                                          ? forward_linear(down, *projection_bias, activated, oracle_flags)
                                                          : forward_linear(down, activated, oracle_flags);
                    ActivationBuffer scalar_output;
                    ActivationBuffer tiled_output;
                    const auto before_scalar = get_vulkan_statistics(runtime);
                    row_tile_check(scalar->forward(input, scalar_output), label + ": scalar forward");
                    const auto after_scalar = get_vulkan_statistics(runtime);
                    row_tile_check(after_scalar.compute_submissions == before_scalar.compute_submissions + 1,
                                   label + ": scalar actually submits GPU work");
                    row_tile_check(tiled->forward(input, tiled_output), label + ": tiled forward");
                    const auto after_tiled = get_vulkan_statistics(runtime);
                    row_tile_check(after_tiled.compute_submissions == after_scalar.compute_submissions + 1,
                                   label + ": tiled actually submits GPU work");
                    row_tile_compare(scalar_output, expected, label + ": scalar CPU parity", 0.0002f, 0.001f);
                    row_tile_compare(tiled_output, expected, label + ": tiled CPU parity", 0.0002f, 0.001f);
                    const double tile_error = row_tile_compare(tiled_output, scalar_output, label + ": tiled scalar parity", 0.00002f, 0.00002f);
                    if (rows >= 16 && (maximum_tile_case.empty() || tile_error > maximum_tile_error))
                    {
                        maximum_tile_error = tile_error;
                        maximum_tile_case = label;
                    }
                }
            }
        }
    }
    std::cout << "MXFP4 row tile on/off max_abs_error=" << std::scientific << std::setprecision(17) << maximum_tile_error
              << " case=" << maximum_tile_case << std::defaultfloat << std::setprecision(6) << '\n';
}
#endif // NCNN_MOE_WITH_VULKAN

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
        ncnn::moe::row_tile_test();
        std::cout << "MXFP4 row tile parity tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "MXFP4 row tile parity test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif // NCNN_MOE_WITH_VULKAN
}
