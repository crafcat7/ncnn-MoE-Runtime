#include "backends/ncnn/ncnn_linear.h"
#include "kernels/cpu_bfloat16.h"
#include "kernels/cpu_float8.h"
#include "kernels/cpu_mxfp4.h"
#include "kernels/cpu_ops.h"

#include "ncnn/moe/types.h"
#include "ncnn/moe/runtime_config.h"

#include <algorithm>
#include <array>
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

static TensorData decode_bfloat16_matrix(const TensorData& matrix)
{
    if (matrix.dtype != DType::MxFp4 || matrix.shape.size() != 2 || matrix.shape[1] % 32 != 0)
        throw std::invalid_argument("MXFP4 matrix is invalid");

    static constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const uint32_t output_columns = matrix.shape[0];
    const uint32_t input_columns = matrix.shape[1];
    const uint32_t block_count = input_columns / 32;
    TensorData decoded;
    decoded.dtype = DType::BFloat16;
    decoded.shape = matrix.shape;
    decoded.bfloat16_data.resize(static_cast<size_t>(output_columns) * input_columns);
    for (uint32_t output = 0; output < output_columns; ++output)
    {
        const size_t packed_row = static_cast<size_t>(output) * block_count * 16;
        const size_t scale_row = static_cast<size_t>(output) * block_count;
        uint16_t* destination = decoded.bfloat16_data.data() + static_cast<size_t>(output) * input_columns;
        for (uint32_t block = 0; block < block_count; ++block)
        {
            const uint8_t encoded_scale = matrix.mxfp4_scales[scale_row + block];
            const float scale = std::ldexp(1.0f, encoded_scale == 0 ? -127 : static_cast<int>(encoded_scale) - 127);
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint8_t packed = matrix.mxfp4_blocks[packed_row + static_cast<size_t>(block) * 16 + lane / 2];
                const uint8_t value = (lane & 1) == 0 ? packed & 15 : packed >> 4;
                const float magnitude = magnitudes[value & 7];
                destination[block * 32 + lane] = float_to_bfloat16((value & 8) == 0 ? magnitude * scale : -magnitude * scale);
            }
        }
    }
    return decoded;
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

static TensorData make_bfloat16_matrix(
    uint32_t output_columns,
    uint32_t input_columns)
{
    if (input_columns % 4 != 0)
    {
        throw std::invalid_argument(
            "packed BF16 input columns must be divisible by 4");
    }
    TensorData matrix;
    matrix.dtype = DType::BFloat16;
    matrix.shape = {output_columns, input_columns};
    matrix.bfloat16_data.resize(
        static_cast<size_t>(output_columns) * input_columns);
    for (size_t index = 0; index < matrix.bfloat16_data.size(); ++index)
    {
        const float value = static_cast<float>(static_cast<int>((index * 17 + 5) % 67) - 33)
                            * 0.0009765625f;
        matrix.bfloat16_data[index] = float_to_bfloat16(value);
    }
    return matrix;
}

static TensorData make_float8_matrix(uint32_t output_columns,
                                     uint32_t input_columns,
                                     uint32_t seed)
{
    TensorData matrix;
    matrix.dtype = DType::Float8E4M3;
    matrix.shape = {output_columns, input_columns};
    const size_t element_count = static_cast<size_t>(output_columns) * input_columns;
    std::shared_ptr<uint8_t[]> storage(
        new uint8_t[element_count], std::default_delete<uint8_t[]>());
    for (size_t index = 0; index < element_count; ++index)
    {
        const float value = static_cast<float>(
                                static_cast<int>((index * 17 + seed * 13) % 61) - 30)
                            * 0.03125f;
        storage[index] = float_to_float8_e4m3(value);
    }
    matrix.mapped_data = std::shared_ptr<const uint8_t>(storage, storage.get());
    matrix.mapped_byte_count = element_count;
    const uint32_t output_blocks = (output_columns + 127) / 128;
    const uint32_t input_blocks = (input_columns + 127) / 128;
    matrix.quantization_scales.resize(
        static_cast<size_t>(output_blocks) * input_blocks);
    for (size_t index = 0; index < matrix.quantization_scales.size(); ++index)
    {
        matrix.quantization_scales[index] = std::ldexp(1.0f, static_cast<int>((index + seed) % 5) - 3);
    }
    return matrix;
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

static void print_command_statistics(
    const NcnnVulkanRuntimeCounters& before,
    const NcnnVulkanRuntimeCounters& after)
{
    std::cout << "command recording: "
              << after.command_dispatches - before.command_dispatches
              << " dispatch(es), "
              << after.command_pipeline_binds
                     - before.command_pipeline_binds
              << " pipeline bind(s), "
              << after.command_descriptor_bindings
                     - before.command_descriptor_bindings
              << " descriptor binding(s), "
              << after.command_push_constant_updates
                     - before.command_push_constant_updates
              << " push constant update(s), "
              << after.command_resource_barrier_calls
                     - before.command_resource_barrier_calls
              << " resource barrier call(s), "
              << after.command_buffer_resource_barriers
                     - before.command_buffer_resource_barriers
              << " buffer barrier(s), "
              << after.command_image_resource_barriers
                     - before.command_image_resource_barriers
              << " image barrier(s), "
              << after.command_redundant_pipeline_binds
                     - before.command_redundant_pipeline_binds
              << " redundant pipeline bind candidate(s)\n";
}

static int benchmark_expert(uint32_t input_columns, uint32_t intermediate_columns, uint32_t token_count, uint32_t repeats, uint32_t device_index)
{
    constexpr uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    if (intermediate_columns > std::numeric_limits<uint32_t>::max() / 2)
    {
        throw std::invalid_argument("intermediate columns are out of range");
    }
    TensorData gate_up = make_matrix(intermediate_columns * 2, input_columns);
    TensorData down = make_matrix(input_columns, intermediate_columns);
    const CpuBatch input = make_input(token_count, input_columns);
    CpuBatch cpu_output;
    Mxfp4Scratch cpu_scratch;
    Mxfp4Task cpu_task;
    cpu_task.gate_up = &gate_up;
    cpu_task.down = &down;
    cpu_task.input = &input;
    cpu_task.output = &cpu_output;
    cpu_task.activation = ExpertActivation::GptOssSwiGlu;
    cpu_task.activation_limit = 7.0f;
    const std::array<Mxfp4Task, 1> cpu_tasks = {cpu_task};
    std::vector<double> cpu_times;
    cpu_times.reserve(repeats);
    for (uint32_t repeat = 0; repeat <= repeats; ++repeat)
    {
        const auto started = std::chrono::steady_clock::now();
        if (!mxfp4_expert_batch(cpu_tasks, &cpu_scratch, optimization_flags))
        {
            std::cerr << "CPU MXFP4 expert failed\n";
            return 1;
        }
        if (repeat != 0)
        {
            cpu_times.push_back(elapsed_milliseconds(started));
        }
    }

    const uint64_t weight_bytes = gate_up.mxfp4_blocks.size() + gate_up.mxfp4_scales.size() + down.mxfp4_blocks.size() + down.mxfp4_scales.size();
    const double cpu_ms = median_milliseconds(cpu_times);
    auto bandwidth = [weight_bytes, token_count](double milliseconds) {
        return static_cast<double>(weight_bytes) * token_count / (1024.0 * 1024.0 * 1024.0) / (milliseconds / 1000.0);
    };
    std::cout << "expert shape: " << token_count << " x " << input_columns << " -> " << intermediate_columns << " -> " << input_columns << '\n';
    std::cout << "MXFP4 CPU kernel: " << mxfp4_kernel_name() << ", row group: " << mxfp4_decode_row_pair_group_size() << '\n';
    std::cout << "weight bytes: " << weight_bytes << '\n';
    std::cout << "CPU median: " << cpu_ms << " ms, " << bandwidth(cpu_ms) << " effective GiB/s\n";

    auto vulkan = NcnnVulkanMxfp4ExpertOperator::create(
        gate_up,
        nullptr,
        down,
        nullptr,
        7.0f,
        device_index,
        ExpertActivation::GptOssSwiGlu,
        context_instance,
        optimization_flags);
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
    const NcnnVulkanRuntimeCounters counters_before = NcnnLinearOperator::vulkan_execution_snapshot(context_instance).counters;
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
    const NcnnVulkanRuntimeCounters counters_after = NcnnLinearOperator::vulkan_execution_snapshot(context_instance).counters;

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
    const double vulkan_ms = median_milliseconds(vulkan_times);
    std::cout << "Vulkan device: " << (device_index == automatic_vulkan_device_index ? -1 : static_cast<int64_t>(device_index)) << '\n';
    std::cout << "Vulkan median: " << vulkan_ms << " ms, " << bandwidth(vulkan_ms) << " effective GiB/s\n";
    std::cout << "speedup: " << cpu_ms / vulkan_ms << "x\n";
    print_command_statistics(counters_before, counters_after);
    std::cout << "maximum absolute error: " << maximum_error << '\n';
    std::cout << "maximum normalized error: " << maximum_normalized_error << '\n';
    return maximum_normalized_error <= 1e-4f ? 0 : 1;
}

static int benchmark_cpu_mxfp4_q8_expert(
    uint32_t input_columns,
    uint32_t intermediate_columns,
    uint32_t token_count,
    uint32_t repeats)
{
    if (!mxfp4_q8_packed_kernel_available())
    {
        std::cout << "CPU MXFP4-Q8 packed kernel unavailable; exact kernel retained\n";
        return 0;
    }
    constexpr uint64_t base_optimization_flags = RuntimeOptimizationDefaultFlags;
    constexpr uint64_t reference_optimization_flags = base_optimization_flags & ~RuntimeOptimizationCpuMxfp4Q8;
    constexpr uint64_t q8_optimization_flags = base_optimization_flags | RuntimeOptimizationCpuMxfp4Q8;
    TensorData gate_up = make_matrix(intermediate_columns * 2, input_columns);
    TensorData down = make_matrix(input_columns, intermediate_columns);
    const CpuBatch input = make_input(token_count, input_columns);
    const std::array<Mxfp4Task, 1> task_template = {{Mxfp4Task{
        &gate_up,
        nullptr,
        &down,
        nullptr,
        &input,
        nullptr,
        ExpertActivation::GptOssSwiGlu,
        7.0f}}};

    CpuBatch reference;
    CpuBatch candidate;
    Mxfp4Scratch reference_scratch;
    Mxfp4Scratch candidate_scratch;
    auto run = [&](uint64_t flags,
                   CpuBatch& output,
                   Mxfp4Scratch& scratch) {
        Mxfp4Task task = task_template[0];
        task.output = &output;
        const std::array<Mxfp4Task, 1> tasks = {task};
        if (!mxfp4_expert_batch(tasks, &scratch, flags))
            throw std::runtime_error("CPU MXFP4 expert failed");
    };
    for (uint32_t warmup = 0; warmup < 3; ++warmup)
    {
        run(reference_optimization_flags, reference, reference_scratch);
        run(q8_optimization_flags, candidate, candidate_scratch);
    }

    std::vector<double> reference_times;
    std::vector<double> candidate_times;
    reference_times.reserve(repeats);
    candidate_times.reserve(repeats);
    for (uint32_t repeat = 0; repeat < repeats; ++repeat)
    {
        auto run_reference = [&]() {
            const auto started = std::chrono::steady_clock::now();
            run(reference_optimization_flags, reference, reference_scratch);
            reference_times.push_back(elapsed_milliseconds(started));
        };
        auto run_candidate = [&]() {
            const auto started = std::chrono::steady_clock::now();
            run(q8_optimization_flags, candidate, candidate_scratch);
            candidate_times.push_back(elapsed_milliseconds(started));
        };
        if ((repeat & 1) == 0)
        {
            run_reference();
            run_candidate();
        }
        else
        {
            run_candidate();
            run_reference();
        }
    }

    float maximum_error = 0.0f;
    float maximum_normalized_error = 0.0f;
    for (size_t row = 0; row < reference.rows(); ++row)
    {
        for (uint32_t column = 0; column < reference.columns(); ++column)
        {
            const float error = std::abs(candidate.row(row)[column] - reference.row(row)[column]);
            maximum_error = std::max(maximum_error, error);
            maximum_normalized_error = std::max(
                maximum_normalized_error,
                error / std::max(1.0f, std::abs(reference.row(row)[column])));
        }
    }
    const double reference_ms = median_milliseconds(reference_times);
    const double candidate_ms = median_milliseconds(candidate_times);
    const uint64_t weight_bytes = gate_up.mxfp4_blocks.size() + gate_up.mxfp4_scales.size()
                                  + down.mxfp4_blocks.size() + down.mxfp4_scales.size();
    auto bandwidth = [weight_bytes, token_count](double milliseconds) {
        return static_cast<double>(weight_bytes) * token_count
               / (1024.0 * 1024.0 * 1024.0)
               / (milliseconds / 1000.0);
    };
    std::cout << "CPU MXFP4-Q8 expert shape: " << token_count << " x "
              << input_columns << " -> " << intermediate_columns << " -> "
              << input_columns << '\n';
    std::cout << "MXFP4 CPU kernel: " << mxfp4_kernel_name()
              << ", Q8 activation: int8 per-32-element block\n";
    std::cout << "exact median: " << reference_ms << " ms, "
              << bandwidth(reference_ms) << " effective GiB/s\n";
    std::cout << "Q8 median: " << candidate_ms << " ms, "
              << bandwidth(candidate_ms) << " effective GiB/s\n";
    std::cout << "Q8 speedup: " << reference_ms / candidate_ms << "x\n";
    std::cout << "maximum absolute/normalized error: " << maximum_error
              << " / " << maximum_normalized_error << '\n';
    return 0;
}

static int benchmark_bfloat16_projection(
    uint32_t input_columns,
    uint32_t output_columns,
    uint32_t token_count,
    uint32_t repeats,
    uint32_t device_index)
{
    constexpr uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
    const NcnnVulkanContextInstancePtr context_instance = create_ncnn_vulkan_context_instance();
    TensorData matrix = make_bfloat16_matrix(
        output_columns,
        input_columns);
    CpuBatch input = make_input(token_count, input_columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            input.row(row)[column] += static_cast<float>(
                                          static_cast<int>((row * 7 + column * 3) % 11) - 5)
                                      * 1e-5f;
        }
    }
    const CpuBatch reference = linear_batch(matrix, input, optimization_flags);
    auto vulkan = NcnnVulkanBfloat16Operator::create(
        matrix,
        nullptr,
        device_index,
        context_instance,
        optimization_flags);
    if (!vulkan)
    {
        std::cout << "Vulkan packed-BF16 projection unavailable\n";
        return 0;
    }

    CpuBatch output;
    for (uint32_t warmup = 0; warmup < 3; ++warmup)
    {
        if (!vulkan->forward(input, output))
        {
            std::cerr << "Vulkan packed-BF16 warm-up failed\n";
            return 1;
        }
    }
    const NcnnVulkanRuntimeCounters counters_before = NcnnLinearOperator::vulkan_execution_snapshot(context_instance).counters;
    std::vector<double> times;
    times.reserve(repeats);
    for (uint32_t repeat = 0; repeat < repeats; ++repeat)
    {
        const auto started = std::chrono::steady_clock::now();
        if (!vulkan->forward(input, output))
        {
            std::cerr << "Vulkan packed-BF16 projection failed\n";
            return 1;
        }
        times.push_back(elapsed_milliseconds(started));
    }
    const NcnnVulkanRuntimeCounters counters_after = NcnnLinearOperator::vulkan_execution_snapshot(context_instance).counters;

    float maximum_error = 0.0f;
    float maximum_normalized_error = 0.0f;
    double squared_error = 0.0;
    for (size_t row = 0; row < reference.rows(); ++row)
    {
        for (uint32_t column = 0; column < reference.columns(); ++column)
        {
            const float expected = reference.row(row)[column];
            const float error = std::abs(output.row(row)[column] - expected);
            maximum_error = std::max(maximum_error, error);
            maximum_normalized_error = std::max(
                maximum_normalized_error,
                error / std::max(1.0f, std::abs(expected)));
            squared_error += static_cast<double>(error) * error;
        }
    }
    const double rms_error = std::sqrt(
        squared_error
        / static_cast<double>(reference.rows() * reference.columns()));
    const uint64_t weight_bytes = matrix.bfloat16_data.size() * sizeof(uint16_t);
    const double milliseconds = median_milliseconds(times);
    const double effective_bandwidth = static_cast<double>(weight_bytes) * token_count
                                       / (1024.0 * 1024.0 * 1024.0)
                                       / (milliseconds / 1000.0);
    const uint64_t submit_wait_microseconds = counters_after.submit_wait_time_microseconds
                                              - counters_before.submit_wait_time_microseconds;
    std::cout << "BF16 shape: " << token_count << " x "
              << input_columns << " -> " << output_columns << '\n';
    std::cout << "median: " << milliseconds << " ms, "
              << effective_bandwidth << " effective GiB/s\n";
    std::cout << "average submit/wait: "
              << static_cast<double>(submit_wait_microseconds)
                     / repeats / 1000.0
              << " ms\n";
    std::cout << "submissions/uploads/downloads: "
              << counters_after.compute_submissions
                     - counters_before.compute_submissions
              << " / "
              << counters_after.batch_uploads - counters_before.batch_uploads
              << " / "
              << counters_after.batch_downloads
                     - counters_before.batch_downloads
              << '\n';
    std::cout << "cooperative matrix dispatches: "
              << counters_after.bfloat16_cooperative_matrix_dispatches
                     - counters_before.bfloat16_cooperative_matrix_dispatches
              << '\n';
    print_command_statistics(counters_before, counters_after);
    std::cout << "maximum absolute/normalized error: "
              << maximum_error << " / " << maximum_normalized_error << '\n';
    std::cout << "RMS error: " << rms_error << '\n';
    return maximum_normalized_error <= 5e-3f ? 0 : 1;
}

static int benchmark_cpu_bfloat16_projection(
    uint32_t input_columns,
    uint32_t output_columns,
    uint32_t token_count,
    uint32_t repeats)
{
    constexpr uint64_t base_optimization_flags = RuntimeOptimizationDefaultFlags;
    constexpr uint64_t policy_flags = RuntimeOptimizationCpuBfloat16Batched | RuntimeOptimizationCpuBfloat16ForceSmall;
    const uint64_t reference_optimization_flags = base_optimization_flags & ~policy_flags;
    const uint64_t candidate_optimization_flags = base_optimization_flags | policy_flags;
    TensorData matrix = make_bfloat16_matrix(output_columns, input_columns);
    CpuBatch input = make_input(token_count, input_columns);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        for (uint32_t column = 0; column < input.columns(); ++column)
        {
            input.row(row)[column] += static_cast<float>(
                                          static_cast<int>((row * 7 + column * 3) % 11) - 5)
                                      * 1e-5f;
        }
    }

    CpuBatch reference;
    for (uint32_t warmup = 0; warmup < 3; ++warmup)
        reference = linear_batch(matrix, input, reference_optimization_flags);
    CpuBatch candidate;
    for (uint32_t warmup = 0; warmup < 3; ++warmup)
        candidate = linear_batch(matrix, input, candidate_optimization_flags);

    std::vector<double> reference_times;
    std::vector<double> candidate_times;
    reference_times.reserve(repeats);
    candidate_times.reserve(repeats);
    auto run_reference = [&]() {
        const auto started = std::chrono::steady_clock::now();
        reference = linear_batch(matrix, input, reference_optimization_flags);
        reference_times.push_back(elapsed_milliseconds(started));
    };
    auto run_candidate = [&]() {
        const auto started = std::chrono::steady_clock::now();
        candidate = linear_batch(matrix, input, candidate_optimization_flags);
        candidate_times.push_back(elapsed_milliseconds(started));
    };
    for (uint32_t repeat = 0; repeat < repeats; ++repeat)
    {
        if ((repeat & 1) == 0)
        {
            run_reference();
            run_candidate();
        }
        else
        {
            run_candidate();
            run_reference();
        }
    }

    float maximum_error = 0.0f;
    float maximum_normalized_error = 0.0f;
    double squared_error = 0.0;
    for (size_t row = 0; row < reference.rows(); ++row)
    {
        for (uint32_t column = 0; column < reference.columns(); ++column)
        {
            const float expected = reference.row(row)[column];
            const float error = std::abs(candidate.row(row)[column] - expected);
            maximum_error = std::max(maximum_error, error);
            maximum_normalized_error = std::max(
                maximum_normalized_error,
                error / std::max(1.0f, std::abs(expected)));
            squared_error += static_cast<double>(error) * error;
        }
    }
    const double rms_error = std::sqrt(
        squared_error
        / static_cast<double>(reference.rows() * reference.columns()));
    const double reference_ms = median_milliseconds(reference_times);
    const double candidate_ms = median_milliseconds(candidate_times);
    const uint64_t logical_weight_bytes = matrix.bfloat16_data.size() * sizeof(uint16_t) * token_count;
    auto bandwidth = [logical_weight_bytes](double milliseconds) {
        return static_cast<double>(logical_weight_bytes)
               / (1024.0 * 1024.0 * 1024.0)
               / (milliseconds / 1000.0);
    };
    std::cout << "CPU BF16 shape: " << token_count << " x "
              << input_columns << " -> " << output_columns << '\n';
    std::cout << "candidate kernel: "
              << bfloat16_batched_linear_kernel_name(candidate_optimization_flags) << '\n';
    std::cout << "reference median: " << reference_ms << " ms, "
              << bandwidth(reference_ms) << " logical GiB/s\n";
    std::cout << "candidate median: " << candidate_ms << " ms, "
              << bandwidth(candidate_ms) << " logical GiB/s\n";
    std::cout << "speedup: " << reference_ms / candidate_ms << "x\n";
    std::cout << "maximum absolute/normalized error: "
              << maximum_error << " / " << maximum_normalized_error << '\n';
    std::cout << "RMS error: " << rms_error << '\n';
    return maximum_normalized_error <= 5e-3f ? 0 : 1;
}

static int benchmark_cpu_float8_expert(uint32_t input_columns,
                                       uint32_t intermediate_columns,
                                       uint32_t token_count,
                                       uint32_t repeats)
{
    constexpr uint64_t optimization_flags = RuntimeOptimizationDefaultFlags;
    TensorData gate = make_float8_matrix(intermediate_columns,
                                         input_columns, 1);
    TensorData up = make_float8_matrix(intermediate_columns,
                                       input_columns, 2);
    TensorData down = make_float8_matrix(input_columns,
                                         intermediate_columns, 3);
    const CpuBatch input = make_input(token_count, input_columns);

    auto fp32_linear = [optimization_flags](const TensorData& matrix,
                                            const CpuBatch& source) {
        constexpr uint32_t block_size = 128;
        CpuBatch quantized = source;
        for (size_t token_index = 0; token_index < quantized.rows();
             ++token_index)
        {
            quantize_float8_e4m3_inplace(
                quantized.row(token_index), quantized.columns(), block_size,
                true, optimization_flags);
        }
        CpuBatch result(source.rows(), matrix.shape[0]);
        const uint32_t input_blocks = (matrix.shape[1] + block_size - 1) / block_size;
        for (uint32_t first_row = 0; first_row < matrix.shape[0];
             first_row += 4)
        {
            const uint32_t row_count = std::min<uint32_t>(4, matrix.shape[0] - first_row);
            const uint8_t* weights = matrix.float8_values().data()
                                     + static_cast<size_t>(first_row) * matrix.shape[1];
            const float* scales = matrix.quantization_scales.data()
                                  + static_cast<size_t>(first_row / block_size) * input_blocks;
            for (size_t token_index = 0; token_index < source.rows();
                 ++token_index)
            {
                float8_e4m3_block_dot_rows4(
                    weights, matrix.shape[1], scales,
                    quantized.row(token_index), matrix.shape[1], block_size,
                    row_count, result.row(token_index) + first_row);
            }
        }
        return result;
    };
    auto fp32_reference_expert = [&]() {
        CpuBatch up_output = fp32_linear(up, input);
        const CpuBatch gate_output = fp32_linear(gate, input);
        for (size_t token_index = 0; token_index < up_output.rows();
             ++token_index)
        {
            float* up_row = up_output.row(token_index);
            const float* gate_row = gate_output.row(token_index);
            for (uint32_t column = 0; column < up_output.columns(); ++column)
            {
                const float gate_value = gate_row[column];
                up_row[column] *= gate_value / (1.0f + std::exp(-gate_value));
            }
        }
        return fp32_linear(down, up_output);
    };

    auto baseline = [&]() {
        CpuBatch up_output = linear_batch(up, input, optimization_flags);
        const CpuBatch gate_output = linear_batch(gate, input, optimization_flags);
        for (size_t token_index = 0; token_index < up_output.rows();
             ++token_index)
        {
            float* up_row = up_output.row(token_index);
            const float* gate_row = gate_output.row(token_index);
            for (uint32_t column = 0; column < up_output.columns(); ++column)
            {
                const float gate_value = gate_row[column];
                up_row[column] *= gate_value / (1.0f + std::exp(-gate_value));
            }
        }
        return linear_batch(down, up_output, optimization_flags);
    };
    auto candidate = [&]() {
        CpuBatch activated;
        if (!fused_float8_gate_up_batch(
                gate, up, input, ExpertActivation::Silu, 0.0f,
                activated, optimization_flags))
        {
            throw std::runtime_error("fused CPU FP8 gate/up unavailable");
        }
        return linear_batch(down, activated, optimization_flags);
    };

    CpuBatch reference;
    CpuBatch fused;
    for (uint32_t warmup = 0; warmup < 5; ++warmup)
    {
        reference = baseline();
        fused = candidate();
    }
    std::vector<double> baseline_times;
    std::vector<double> candidate_times;
    baseline_times.reserve(repeats);
    candidate_times.reserve(repeats);
    for (uint32_t repeat = 0; repeat < repeats; ++repeat)
    {
        auto run_baseline = [&]() {
            const auto started = std::chrono::steady_clock::now();
            reference = baseline();
            baseline_times.push_back(elapsed_milliseconds(started));
        };
        auto run_candidate = [&]() {
            const auto started = std::chrono::steady_clock::now();
            fused = candidate();
            candidate_times.push_back(elapsed_milliseconds(started));
        };
        if ((repeat & 1) == 0)
        {
            run_baseline();
            run_candidate();
        }
        else
        {
            run_candidate();
            run_baseline();
        }
    }

    const CpuBatch fp32_reference = fp32_reference_expert();
    float maximum_error = 0.0f;
    float maximum_normalized_error = 0.0f;
    for (size_t token_index = 0; token_index < fp32_reference.rows();
         ++token_index)
    {
        for (uint32_t column = 0; column < fp32_reference.columns(); ++column)
        {
            const float expected = fp32_reference.row(token_index)[column];
            const float error = std::abs(fused.row(token_index)[column] - expected);
            maximum_error = std::max(maximum_error, error);
            maximum_normalized_error = std::max(
                maximum_normalized_error,
                error / std::max(1.0f, std::abs(expected)));
        }
    }
    const double baseline_ms = median_milliseconds(baseline_times);
    const double candidate_ms = median_milliseconds(candidate_times);
    std::cout << "CPU FP8 expert shape: " << token_count << " x "
              << input_columns << " -> " << intermediate_columns << " -> "
              << input_columns << '\n';
    std::cout << "FP8 linear kernel: " << float8_linear_kernel_name(optimization_flags)
              << ", row group: " << float8_linear_row_group_size(optimization_flags) << '\n';
    std::cout << "baseline median: " << baseline_ms << " ms\n";
    std::cout << "fused median: " << candidate_ms << " ms\n";
    std::cout << "speedup: " << baseline_ms / candidate_ms << "x\n";
    std::cout << "maximum absolute/normalized error: " << maximum_error
              << " / " << maximum_normalized_error << '\n';
    return maximum_normalized_error <= 1e-4f ? 0 : 1;
}

} // namespace moe
} // namespace ncnn

int main(int argc, char** argv)
{
    try
    {
        constexpr uint64_t optimization_flags = ncnn::moe::RuntimeOptimizationDefaultFlags;
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
        if (mode == "bfloat16")
        {
            return ncnn::moe::benchmark_bfloat16_projection(
                input_columns,
                output_columns,
                token_count,
                repeats,
                device_index);
        }
        if (mode == "cpu-bfloat16")
        {
            return ncnn::moe::benchmark_cpu_bfloat16_projection(
                input_columns,
                output_columns,
                token_count,
                repeats);
        }
        if (mode == "cpu-float8-expert")
        {
            return ncnn::moe::benchmark_cpu_float8_expert(
                input_columns,
                output_columns,
                token_count,
                repeats);
        }
        if (mode == "cpu-mxfp4-q8-expert")
        {
            return ncnn::moe::benchmark_cpu_mxfp4_q8_expert(
                input_columns,
                output_columns,
                token_count,
                repeats);
        }
        if (mode != "projection")
        {
            throw std::invalid_argument(
                "mode must be projection, bfloat16, cpu-bfloat16, cpu-float8-expert, cpu-mxfp4-q8-expert, or expert");
        }

        ncnn::moe::TensorData matrix = ncnn::moe::make_matrix(output_columns, input_columns);
        ncnn::moe::TensorData bfloat16_matrix = ncnn::moe::decode_bfloat16_matrix(matrix);
        const ncnn::moe::CpuBatch input = ncnn::moe::make_input(token_count, input_columns);
        ncnn::moe::CpuBatch cpu_output = ncnn::moe::linear_batch(matrix, input, optimization_flags);
        std::vector<double> cpu_times;
        cpu_times.reserve(repeats);
        for (uint32_t repeat = 0; repeat < repeats; ++repeat)
        {
            const auto started = std::chrono::steady_clock::now();
            cpu_output = ncnn::moe::linear_batch(matrix, input, optimization_flags);
            cpu_times.push_back(ncnn::moe::elapsed_milliseconds(started));
        }

        const ncnn::moe::NcnnVulkanContextInstancePtr context_instance = ncnn::moe::create_ncnn_vulkan_context_instance();
        auto vulkan = ncnn::moe::NcnnVulkanMxfp4Operator::create(
            matrix,
            nullptr,
            device_index,
            context_instance,
            optimization_flags);
        if (!vulkan)
        {
            std::cout << "Vulkan MXFP4 projection unavailable\n";
            return 0;
        }
        auto bfloat16_vulkan = ncnn::moe::NcnnLinearOperator::create(
            bfloat16_matrix,
            nullptr,
            ncnn::moe::NcnnLinearDevice::Vulkan,
            device_index,
            context_instance,
            optimization_flags);
        if (!bfloat16_vulkan)
        {
            std::cout << "Vulkan BF16-source projection unavailable\n";
            return 0;
        }
        auto packed_bfloat16_vulkan = ncnn::moe::NcnnVulkanBfloat16Operator::create(
            bfloat16_matrix,
            nullptr,
            device_index,
            context_instance,
            optimization_flags);
        if (!packed_bfloat16_vulkan)
        {
            std::cout << "Vulkan packed-BF16 projection unavailable\n";
            return 0;
        }
        ncnn::moe::CpuBatch vulkan_output;
        if (!vulkan->forward(input, vulkan_output))
        {
            std::cerr << "Vulkan MXFP4 warm-up failed\n";
            return 1;
        }
        ncnn::moe::CpuBatch bfloat16_vulkan_output;
        if (!bfloat16_vulkan->forward(input, bfloat16_vulkan_output))
        {
            std::cerr << "Vulkan BF16-source warm-up failed\n";
            return 1;
        }
        ncnn::moe::CpuBatch packed_bfloat16_vulkan_output;
        if (!packed_bfloat16_vulkan->forward(
                input,
                packed_bfloat16_vulkan_output))
        {
            std::cerr << "Vulkan packed-BF16 warm-up failed\n";
            return 1;
        }
        std::vector<double> vulkan_times;
        std::vector<double> bfloat16_vulkan_times;
        std::vector<double> packed_bfloat16_vulkan_times;
        vulkan_times.reserve(repeats);
        bfloat16_vulkan_times.reserve(repeats);
        packed_bfloat16_vulkan_times.reserve(repeats);
        for (uint32_t repeat = 0; repeat < repeats; ++repeat)
        {
            const auto started = std::chrono::steady_clock::now();
            if (!vulkan->forward(input, vulkan_output))
            {
                std::cerr << "Vulkan MXFP4 projection failed\n";
                return 1;
            }
            vulkan_times.push_back(ncnn::moe::elapsed_milliseconds(started));

            const auto bfloat16_started = std::chrono::steady_clock::now();
            if (!bfloat16_vulkan->forward(input, bfloat16_vulkan_output))
            {
                std::cerr << "Vulkan BF16-source projection failed\n";
                return 1;
            }
            bfloat16_vulkan_times.push_back(ncnn::moe::elapsed_milliseconds(bfloat16_started));

            const auto packed_bfloat16_started = std::chrono::steady_clock::now();
            if (!packed_bfloat16_vulkan->forward(
                    input,
                    packed_bfloat16_vulkan_output))
            {
                std::cerr << "Vulkan packed-BF16 projection failed\n";
                return 1;
            }
            packed_bfloat16_vulkan_times.push_back(
                ncnn::moe::elapsed_milliseconds(
                    packed_bfloat16_started));
        }

        float maximum_error = 0.0f;
        float maximum_bfloat16_error = 0.0f;
        float maximum_packed_bfloat16_error = 0.0f;
        std::vector<float> row_errors(cpu_output.rows(), 0.0f);
        for (size_t row = 0; row < cpu_output.rows(); ++row)
        {
            for (uint32_t column = 0; column < cpu_output.columns(); ++column)
            {
                maximum_error = std::max(maximum_error, std::abs(cpu_output.row(row)[column] - vulkan_output.row(row)[column]));
                maximum_bfloat16_error = std::max(maximum_bfloat16_error, std::abs(cpu_output.row(row)[column] - bfloat16_vulkan_output.row(row)[column]));
                maximum_packed_bfloat16_error = std::max(
                    maximum_packed_bfloat16_error,
                    std::abs(
                        cpu_output.row(row)[column]
                        - packed_bfloat16_vulkan_output.row(row)[column]));
                row_errors[row] = std::max(row_errors[row], std::abs(cpu_output.row(row)[column] - vulkan_output.row(row)[column]));
            }
        }
        const uint64_t weight_bytes = matrix.mxfp4_blocks.size() + matrix.mxfp4_scales.size();
        const uint64_t bfloat16_source_bytes = bfloat16_matrix.bfloat16_data.size() * sizeof(uint16_t);
        const uint64_t bfloat16_device_bytes = bfloat16_matrix.bfloat16_data.size() * sizeof(float);
        const double cpu_ms = ncnn::moe::median_milliseconds(cpu_times);
        const double vulkan_ms = ncnn::moe::median_milliseconds(vulkan_times);
        const double bfloat16_vulkan_ms = ncnn::moe::median_milliseconds(bfloat16_vulkan_times);
        const double packed_bfloat16_vulkan_ms = ncnn::moe::median_milliseconds(
            packed_bfloat16_vulkan_times);
        const auto bandwidth = [weight_bytes, token_count](double milliseconds) { return static_cast<double>(weight_bytes) * token_count / (1024.0 * 1024.0 * 1024.0) / (milliseconds / 1000.0); };
        const auto bfloat16_bandwidth = [bfloat16_device_bytes, token_count](double milliseconds) {
            return static_cast<double>(bfloat16_device_bytes) * token_count / (1024.0 * 1024.0 * 1024.0) / (milliseconds / 1000.0);
        };
        std::cout << "shape: " << token_count << " x " << input_columns << " -> " << output_columns << '\n';
        std::cout << "Vulkan device: " << (device_index == ncnn::moe::automatic_vulkan_device_index ? -1 : static_cast<int64_t>(device_index)) << '\n';
        std::cout << "MXFP4 weight bytes: " << weight_bytes << '\n';
        std::cout << "BF16 source/device weight bytes: " << bfloat16_source_bytes << " / " << bfloat16_device_bytes << '\n';
        std::cout << "CPU median: " << cpu_ms << " ms, " << bandwidth(cpu_ms) << " effective GiB/s\n";
        std::cout << "Vulkan MXFP4 median: " << vulkan_ms << " ms, " << bandwidth(vulkan_ms) << " effective GiB/s\n";
        std::cout << "Vulkan BF16-source median: " << bfloat16_vulkan_ms << " ms, " << bfloat16_bandwidth(bfloat16_vulkan_ms) << " effective GiB/s\n";
        std::cout << "Vulkan packed-BF16 median: "
                  << packed_bfloat16_vulkan_ms << " ms, "
                  << static_cast<double>(bfloat16_source_bytes) * token_count
                         / (1024.0 * 1024.0 * 1024.0)
                         / (packed_bfloat16_vulkan_ms / 1000.0)
                  << " effective GiB/s\n";
        std::cout << "MXFP4 speedup over CPU: " << cpu_ms / vulkan_ms << "x\n";
        std::cout << "MXFP4 speedup over Vulkan BF16-source: " << bfloat16_vulkan_ms / vulkan_ms << "x\n";
        std::cout << "packed-BF16 speedup over Vulkan BF16-source: "
                  << bfloat16_vulkan_ms / packed_bfloat16_vulkan_ms
                  << "x\n";
        std::cout << "maximum MXFP4 absolute error: " << maximum_error << '\n';
        std::cout << "maximum BF16-source absolute error: " << maximum_bfloat16_error << '\n';
        std::cout << "maximum packed-BF16 absolute error: "
                  << maximum_packed_bfloat16_error << '\n';
        std::cout << "row maximum errors:";
        for (float error : row_errors)
            std::cout << ' ' << error;
        std::cout << '\n';
        return maximum_error <= 1e-3f
                       && maximum_packed_bfloat16_error <= 1e-3f
                   ? 0
                   : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
