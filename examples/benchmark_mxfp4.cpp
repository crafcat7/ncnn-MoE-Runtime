#include "backends/ncnn/ncnn_linear.h"
#include "kernels/cpu_ops.h"

#include "ncnn/moe/types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncnn {
namespace moe {

static uint32_t parse_dimension(const char* value, const char* name)
{
    const unsigned long parsed = std::stoul(value);
    if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(std::string(name) + " is out of range");
    }
    return static_cast<uint32_t>(parsed);
}

static uint32_t parse_index(const char* value, const char* name)
{
    const unsigned long parsed = std::stoul(value);
    if (parsed > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(std::string(name) + " is out of range");
    }
    return static_cast<uint32_t>(parsed);
}

static TensorData make_matrix(uint32_t output_columns, uint32_t input_columns)
{
    if (input_columns % 32 != 0)
        throw std::invalid_argument("input columns must be divisible by 32");
    const uint32_t block_count = input_columns / 32;
    TensorData matrix;
    matrix.dtype = DType::MxFp4;
    matrix.shape = {output_columns, input_columns};
    matrix.mxfp4_blocks.resize(static_cast<size_t>(output_columns) * block_count * 16);
    matrix.mxfp4_scales.resize(static_cast<size_t>(output_columns) * block_count);
    for (size_t index = 0; index < matrix.mxfp4_blocks.size(); ++index)
    {
        const uint8_t low = static_cast<uint8_t>((index * 5 + 1) & 15);
        const uint8_t high = static_cast<uint8_t>((index * 7 + 3) & 15);
        matrix.mxfp4_blocks[index] = static_cast<uint8_t>(low | high << 4);
    }
    for (size_t index = 0; index < matrix.mxfp4_scales.size(); ++index)
    {
        matrix.mxfp4_scales[index] = static_cast<uint8_t>(124 + index % 7);
    }
    return matrix;
}

static CpuBatch make_input(uint32_t token_count, uint32_t columns)
{
    CpuBatch input(token_count, columns);
    for (size_t token = 0; token < input.rows(); ++token)
    {
        for (uint32_t column = 0; column < columns; ++column)
        {
            input.row(token)[column] = static_cast<float>(static_cast<int>((column * 13 + token * 7) % 29) - 14) * 0.03125f;
        }
    }
    return input;
}

static double median_milliseconds(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 != 0)
        return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

static double elapsed_milliseconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static int benchmark_expert(uint32_t input_columns, uint32_t intermediate_columns, uint32_t token_count, uint32_t repeats, uint32_t device_index)
{
    if (intermediate_columns > std::numeric_limits<uint32_t>::max() / 2)
    {
        throw std::invalid_argument("intermediate columns are out of range");
    }
    TensorData gate_up = make_matrix(intermediate_columns * 2, input_columns);
    TensorData down = make_matrix(input_columns, intermediate_columns);
    const CpuBatch input = make_input(token_count, input_columns);
    CpuBatch cpu_output;
    std::vector<double> cpu_times;
    cpu_times.reserve(repeats);
    for (uint32_t repeat = 0; repeat <= repeats; ++repeat)
    {
        const auto started = std::chrono::steady_clock::now();
        const CpuBatch activated = fused_mxfp4_gate_up_batch(gate_up, nullptr, input, 7.0f);
        cpu_output = linear_batch(down, activated);
        if (repeat != 0)
        {
            cpu_times.push_back(elapsed_milliseconds(started));
        }
    }

    auto vulkan = NcnnVulkanMxfp4ExpertOperator::create(gate_up, nullptr, down, nullptr, 7.0f, device_index);
    if (!vulkan)
    {
        std::cout << "Vulkan MXFP4 expert unavailable\n";
        return 0;
    }
    CpuBatch vulkan_output;
    if (!vulkan->forward(input, vulkan_output))
    {
        std::cerr << "Vulkan MXFP4 expert warm-up failed\n";
        return 1;
    }
    std::vector<double> vulkan_times;
    vulkan_times.reserve(repeats);
    for (uint32_t repeat = 0; repeat < repeats; ++repeat)
    {
        const auto started = std::chrono::steady_clock::now();
        if (!vulkan->forward(input, vulkan_output))
        {
            std::cerr << "Vulkan MXFP4 expert failed\n";
            return 1;
        }
        vulkan_times.push_back(elapsed_milliseconds(started));
    }

    float maximum_error = 0.0f;
    float maximum_normalized_error = 0.0f;
    for (size_t row = 0; row < cpu_output.rows(); ++row)
    {
        for (uint32_t column = 0; column < cpu_output.columns(); ++column)
        {
            const float error = std::abs(cpu_output.row(row)[column] - vulkan_output.row(row)[column]);
            maximum_error = std::max(maximum_error, error);
            maximum_normalized_error = std::max(maximum_normalized_error, error / std::max(1.0f, std::abs(cpu_output.row(row)[column])));
        }
    }
    const uint64_t weight_bytes = gate_up.mxfp4_blocks.size() + gate_up.mxfp4_scales.size() + down.mxfp4_blocks.size() + down.mxfp4_scales.size();
    const double cpu_ms = median_milliseconds(cpu_times);
    const double vulkan_ms = median_milliseconds(vulkan_times);
    auto bandwidth = [weight_bytes, token_count](double milliseconds) { return static_cast<double>(weight_bytes) * token_count / (1024.0 * 1024.0 * 1024.0) / (milliseconds / 1000.0); };
    std::cout << "expert shape: " << token_count << " x " << input_columns << " -> " << intermediate_columns << " -> " << input_columns << '\n';
    std::cout << "Vulkan device: " << (device_index == automatic_vulkan_device_index ? -1 : static_cast<int64_t>(device_index)) << '\n';
    std::cout << "weight bytes: " << weight_bytes << '\n';
    std::cout << "CPU median: " << cpu_ms << " ms, " << bandwidth(cpu_ms) << " effective GiB/s\n";
    std::cout << "Vulkan median: " << vulkan_ms << " ms, " << bandwidth(vulkan_ms) << " effective GiB/s\n";
    std::cout << "speedup: " << cpu_ms / vulkan_ms << "x\n";
    std::cout << "maximum absolute error: " << maximum_error << '\n';
    std::cout << "maximum normalized error: " << maximum_normalized_error << '\n';
    return maximum_normalized_error <= 1e-4f ? 0 : 1;
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    try
    {
        const uint32_t input_columns = argc > 1 ? ncnn::moe::parse_dimension(argv[1], "input columns") : 2880;
        const uint32_t output_columns = argc > 2 ? ncnn::moe::parse_dimension(argv[2], "output columns") : 5760;
        const uint32_t token_count = argc > 3 ? ncnn::moe::parse_dimension(argv[3], "token count") : 1;
        const uint32_t repeats = argc > 4 ? ncnn::moe::parse_dimension(argv[4], "repeats") : 10;
        const uint32_t device_index = argc > 5 ? ncnn::moe::parse_index(argv[5], "Vulkan device index") : ncnn::moe::automatic_vulkan_device_index;
        const std::string mode = argc > 6 ? argv[6] : "projection";
        if (mode == "expert")
        {
            return ncnn::moe::benchmark_expert(input_columns, output_columns, token_count, repeats, device_index);
        }
        if (mode != "projection")
        {
            throw std::invalid_argument("mode must be projection or expert");
        }

        ncnn::moe::TensorData matrix = ncnn::moe::make_matrix(output_columns, input_columns);
        const ncnn::moe::CpuBatch input = ncnn::moe::make_input(token_count, input_columns);
        ncnn::moe::CpuBatch cpu_output = ncnn::moe::linear_batch(matrix, input);
        std::vector<double> cpu_times;
        cpu_times.reserve(repeats);
        for (uint32_t repeat = 0; repeat < repeats; ++repeat)
        {
            const auto started = std::chrono::steady_clock::now();
            cpu_output = ncnn::moe::linear_batch(matrix, input);
            cpu_times.push_back(ncnn::moe::elapsed_milliseconds(started));
        }

        auto vulkan = ncnn::moe::NcnnVulkanMxfp4Operator ::create(matrix, nullptr, device_index);
        if (!vulkan)
        {
            std::cout << "Vulkan MXFP4 projection unavailable\n";
            return 0;
        }
        ncnn::moe::CpuBatch vulkan_output;
        if (!vulkan->forward(input, vulkan_output))
        {
            std::cerr << "Vulkan MXFP4 warm-up failed\n";
            return 1;
        }
        std::vector<double> vulkan_times;
        vulkan_times.reserve(repeats);
        for (uint32_t repeat = 0; repeat < repeats; ++repeat)
        {
            const auto started = std::chrono::steady_clock::now();
            if (!vulkan->forward(input, vulkan_output))
            {
                std::cerr << "Vulkan MXFP4 projection failed\n";
                return 1;
            }
            vulkan_times.push_back(ncnn::moe::elapsed_milliseconds(started));
        }

        float maximum_error = 0.0f;
        std::vector<float> row_errors(cpu_output.rows(), 0.0f);
        for (size_t row = 0; row < cpu_output.rows(); ++row)
        {
            for (uint32_t column = 0; column < cpu_output.columns(); ++column)
            {
                maximum_error = std::max(maximum_error, std::abs(cpu_output.row(row)[column] - vulkan_output.row(row)[column]));
                row_errors[row] = std::max(row_errors[row], std::abs(cpu_output.row(row)[column] - vulkan_output.row(row)[column]));
            }
        }
        const uint64_t weight_bytes = matrix.mxfp4_blocks.size() + matrix.mxfp4_scales.size();
        const double cpu_ms = ncnn::moe::median_milliseconds(cpu_times);
        const double vulkan_ms = ncnn::moe::median_milliseconds(vulkan_times);
        const auto bandwidth = [weight_bytes, token_count](double milliseconds) { return static_cast<double>(weight_bytes) * token_count / (1024.0 * 1024.0 * 1024.0) / (milliseconds / 1000.0); };
        std::cout << "shape: " << token_count << " x " << input_columns << " -> " << output_columns << '\n';
        std::cout << "Vulkan device: " << (device_index == ncnn::moe::automatic_vulkan_device_index ? -1 : static_cast<int64_t>(device_index)) << '\n';
        std::cout << "weight bytes: " << weight_bytes << '\n';
        std::cout << "CPU median: " << cpu_ms << " ms, " << bandwidth(cpu_ms) << " effective GiB/s\n";
        std::cout << "Vulkan median: " << vulkan_ms << " ms, " << bandwidth(vulkan_ms) << " effective GiB/s\n";
        std::cout << "speedup: " << cpu_ms / vulkan_ms << "x\n";
        std::cout << "maximum absolute error: " << maximum_error << '\n';
        std::cout << "row maximum errors:";
        for (float error : row_errors)
            std::cout << ' ' << error;
        std::cout << '\n';
        return maximum_error <= 1e-3f ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
