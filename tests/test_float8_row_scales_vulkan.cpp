#include "backends/ncnn/linear.h"
#include "backends/ncnn/vulkancontext.h"
#include "kernels/float8.h"
#include "kernels/ops.h"
#include "models/modeladapter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

#if NCNN_MOE_WITH_VULKAN
static void row_scale_check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static TensorData row_scale_matrix(uint32_t rows, uint32_t columns, uint32_t scale_rows, uint32_t seed)
{
    TensorData result;
    result.dtype = DType::Float8E4M3;
    result.shape = {rows, columns};
    result.float8_scale_row_size = scale_rows;
    const size_t count = static_cast<size_t>(rows) * columns;
    std::shared_ptr<uint8_t[]> storage(new uint8_t[count], std::default_delete<uint8_t[]>());
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
        {
            const int code = static_cast<int>((row * 7 + column * 11 + column / 17 + seed) % 17) - 8;
            storage[static_cast<size_t>(row) * columns + column] = float_to_float8_e4m3(code * 0.03125f);
        }
    result.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
    result.mapped_size = count;
    const uint32_t blocks = columns / 128;
    const uint32_t groups = rows / scale_rows + (rows % scale_rows != 0);
    result.quantization_scales.resize(static_cast<size_t>(groups) * blocks);
    for (uint32_t group = 0; group < groups; ++group)
        for (uint32_t block = 0; block < blocks; ++block)
        {
            const size_t index = static_cast<size_t>(group) * blocks + block;
            result.quantization_scales[index] = static_cast<float>((group * 3 + block + seed) % 7 + 1) * 0.125f;
        }
    return result;
}

static ActivationBuffer row_scale_input(uint32_t rows, uint32_t columns)
{
    ActivationBuffer result(rows, columns);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
            result.row(row)[column] = static_cast<float>(static_cast<int>((row * 5 + column * 3 + column / 7) % 13) - 6) * 0.03125f;
    return result;
}

// Decode one byte at a time and accumulate in FP64. Activation quantization is
// the existing FP8 execution policy; weight scales are indexed independently.
static ActivationBuffer row_scale_reference(const TensorData& matrix, const ActivationBuffer& input,
                                            uint32_t input_groups = 1, const TensorData* bias = nullptr)
{
    const uint32_t columns = matrix.shape[1];
    const uint32_t rows = matrix.shape[0];
    const uint32_t blocks = columns / 128;
    const uint32_t rows_per_group = rows / input_groups;
    ActivationBuffer quantized(input.rows(), input.columns());
    for (size_t token = 0; token < input.rows(); ++token)
        quantize_float8_e4m3(input.row(token), quantized.row(token), input.columns(), 128, true, 0);
    ActivationBuffer output(input.rows(), rows);
    const auto values = matrix.float8_values();
    for (size_t token = 0; token < input.rows(); ++token)
        for (uint32_t row = 0; row < rows; ++row)
        {
            double sum = bias ? bias->float32_values()[row] : 0.0;
            for (uint32_t column = 0; column < columns; ++column)
            {
                const size_t scale_index = static_cast<size_t>(row / matrix.float8_scale_row_size) * blocks + column / 128;
                const float weight = float8_e4m3_to_float(values[static_cast<size_t>(row) * columns + column]);
                sum += static_cast<double>(weight) * matrix.quantization_scales[scale_index]
                       * quantized.row(token)[row / rows_per_group * columns + column];
            }
            output.row(token)[row] = static_cast<float>(sum);
        }
    return output;
}

static void row_scale_compare(const ActivationBuffer& actual, const ActivationBuffer& expected, float tolerance = 0.0003f)
{
    row_scale_check(actual.rows() == expected.rows() && actual.columns() == expected.columns(), "FP8 row-scale output shape");
    for (size_t row = 0; row < actual.rows(); ++row)
        for (uint32_t column = 0; column < actual.columns(); ++column)
        {
            const float error = std::abs(actual.row(row)[column] - expected.row(row)[column]);
            if (!std::isfinite(actual.row(row)[column]) || error > tolerance * (1.f + std::abs(expected.row(row)[column])))
                throw std::runtime_error("FP8 row-scale mismatch at " + std::to_string(row) + "," + std::to_string(column)
                                         + ": actual=" + std::to_string(actual.row(row)[column])
                                         + " expected=" + std::to_string(expected.row(row)[column]));
        }
}

static TensorData row_scale_selected(uint32_t rows)
{
    const TensorData source = row_scale_matrix(512, 256, 128, 1);
    std::vector<uint32_t> indices(rows);
    for (uint32_t row = 0; row < rows; ++row)
        indices[row] = (row * 131 + 19) % 512;
    auto selected = select_tensor_rows(source, indices);
    row_scale_check(static_cast<bool>(selected), "lossless selected FP8 rows");
    row_scale_check(selected.value().dtype == DType::Float8E4M3 && selected.value().float8_scale_row_size == 1
                        && selected.value().float8_values().size() == static_cast<size_t>(rows) * 256
                        && selected.value().bfloat16_data.empty() && selected.value().float32_data.empty(),
                    "selected FP8 matrix must retain one-byte storage");
    for (uint32_t row = 0; row < rows; ++row)
    {
        for (uint32_t block = 0; block < 2; ++block)
            row_scale_check(selected.value().quantization_scales[row * 2 + block]
                                == source.quantization_scales[(indices[row] / 128) * 2 + block],
                            "selected FP8 row lost its original scale");
        row_scale_check(std::equal(selected.value().float8_values().begin() + row * 256,
                                   selected.value().float8_values().begin() + (row + 1) * 256,
                                   source.float8_values().begin() + indices[row] * 256),
                        "selected FP8 row bytes changed");
    }
    return std::move(selected).value();
}

static void row_scale_single(uint32_t scale_rows, uint32_t matrix_rows, uint32_t tokens, uint32_t input_groups)
{
    const TensorData matrix = scale_rows == 1 ? row_scale_selected(matrix_rows) : row_scale_matrix(matrix_rows, 256, scale_rows, 2);
    TensorData bias;
    bias.dtype = DType::Float32;
    bias.shape = {matrix_rows};
    bias.float32_data.resize(matrix_rows);
    for (uint32_t row = 0; row < matrix_rows; ++row)
        bias.float32_data[row] = static_cast<float>(static_cast<int>(row % 7) - 3) * 0.015625f;
    const auto input = row_scale_input(tokens, 256 * input_groups);
    const auto expected = row_scale_reference(matrix, input, input_groups, &bias);
    const auto runtime = create_vulkan_runtime();
    const auto op = Float8Linear_vulkan::create(matrix, &bias, input_groups, get_default_gpu_index(), runtime, OptimizationDefaultFlags);
    row_scale_check(static_cast<bool>(op), "create FP8 row-scale GPU projection");
    const auto before = get_vulkan_statistics(runtime);
    ActivationBuffer actual;
    row_scale_check(op->forward(input, actual), "execute FP8 row-scale GPU projection");
    const auto after = get_vulkan_statistics(runtime);
    row_scale_check(after.compute_submissions == before.compute_submissions + 1 && after.dispatches == before.dispatches + 1,
                    "FP8 row-scale projection must actually dispatch on GPU");
    row_scale_compare(actual, expected);
    const std::byte* allocation = actual.bytes().data();
    row_scale_check(op->forward(input, actual) && allocation == actual.bytes().data(), "warm FP8 projection reuses output storage");
    row_scale_compare(actual, expected);
}

static void row_scale_chains(uint32_t tokens)
{
    const TensorData first = row_scale_selected(256);
    const TensorData second = row_scale_matrix(128, 256, 1, 3);
    const TensorData up_weight = row_scale_matrix(256, 256, 1, 7);
    const auto input = row_scale_input(tokens, 256);
    const auto first_expected = row_scale_reference(first, input);
    const auto chain_expected = row_scale_reference(second, first_expected);
    const auto runtime = create_vulkan_runtime();
    const auto gate = Float8Linear_vulkan::create(first, nullptr, 1, get_default_gpu_index(), runtime, OptimizationDefaultFlags);
    const auto down = Float8Linear_vulkan::create(second, nullptr, 1, get_default_gpu_index(), runtime, OptimizationDefaultFlags);
    const auto up = Float8Linear_vulkan::create(up_weight, nullptr, 1, get_default_gpu_index(), runtime, OptimizationDefaultFlags);
    row_scale_check(gate && down && up, "create per-row-scale FP8 chain");
    ActivationBuffer actual;
    auto before = get_vulkan_statistics(runtime);
    row_scale_check(gate->forward_chain(input, *down, actual), "per-row-scale FP8 linear chain");
    row_scale_check(get_vulkan_statistics(runtime).compute_submissions == before.compute_submissions + 1, "FP8 chain shares one GPU submission");
    row_scale_compare(actual, chain_expected);

    TensorData norm;
    norm.dtype = DType::Float32;
    norm.shape = {256};
    norm.float32_data.resize(256);
    for (uint32_t column = 0; column < 256; ++column)
        norm.float32_data[column] = 0.75f + (column % 7) * 0.03125f;
    row_scale_check(gate->prepare_rms_norm(norm, 1e-6f), "prepare per-row-scale FP8 RMS chain");
    const auto normalized = forward_rms_norm(first_expected, norm, 1e-6f, 0.0f);
    const auto norm_expected = row_scale_reference(second, normalized);
    before = get_vulkan_statistics(runtime);
    row_scale_check(gate->forward_rms_norm_chain(input, *down, actual), "per-row-scale FP8 RMS chain");
    row_scale_check(get_vulkan_statistics(runtime).compute_submissions == before.compute_submissions + 1, "FP8 RMS chain stays on GPU");
    row_scale_compare(actual, norm_expected, 0.002f);
    ActivationBuffer parallel;
    row_scale_check(gate->forward_rms_norm_chain_parallel(input, *down, *up, actual, parallel), "per-row-scale FP8 parallel chain");
    row_scale_compare(actual, norm_expected, 0.002f);
    const auto up_expected = row_scale_reference(up_weight, input);
    row_scale_compare(parallel, up_expected);

    ActivationBuffer activated = first_expected;
    constexpr float limit = 0.125f;
    for (size_t row = 0; row < activated.rows(); ++row)
        for (uint32_t column = 0; column < activated.columns(); ++column)
        {
            const float gate_value = std::min(activated.row(row)[column], limit);
            const float up_value = std::clamp(up_expected.row(row)[column], -limit, limit);
            activated.row(row)[column] = gate_value / (1.f + std::exp(-gate_value)) * up_value;
        }
    const auto swiglu_expected = row_scale_reference(second, activated);
    before = get_vulkan_statistics(runtime);
    row_scale_check(gate->forward_swiglu_chain(input, *up, *down, ExpertActivation::DeepSeekSwiGlu, limit, actual),
                    "per-row-scale FP8 Shared Expert chain");
    row_scale_check(get_vulkan_statistics(runtime).compute_submissions == before.compute_submissions + 1, "FP8 Shared chain stays on GPU");
    row_scale_compare(actual, swiglu_expected, 0.002f);
}

static void row_scale_invalid()
{
    const auto runtime = create_vulkan_runtime();
    TensorData matrix = row_scale_matrix(129, 256, 1, 1);
    matrix.float8_scale_row_size = 0;
    row_scale_check(!Float8Linear_vulkan::create(matrix, nullptr, 1, get_default_gpu_index(), runtime, OptimizationDefaultFlags), "reject zero FP8 scale-row size");
    matrix.float8_scale_row_size = 1;
    matrix.quantization_scales.pop_back();
    row_scale_check(!Float8Linear_vulkan::create(matrix, nullptr, 1, get_default_gpu_index(), runtime, OptimizationDefaultFlags), "reject incomplete per-row FP8 scales");
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
        for (uint32_t tokens : {1u, 3u})
        {
            ncnn::moe::row_scale_single(128, 129, tokens, 1);
            ncnn::moe::row_scale_single(1, 129, tokens, 1);
            ncnn::moe::row_scale_single(1, 258, tokens, 3);
            ncnn::moe::row_scale_chains(tokens);
        }
        ncnn::moe::row_scale_invalid();
        std::cout << "FP8 per-row-scale Vulkan tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FP8 per-row-scale Vulkan test failed: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "SKIP: Vulkan backend disabled\n";
    return 77;
#endif
}
